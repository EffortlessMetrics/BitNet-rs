# PR Gate Success

`PR Gate Success` is the single required check that branch
protection should gate merges on. It is the deliverable of PR 19 in
the strict policy / CI economics rollout.

## What it aggregates

The aggregator runs as a single GitHub Actions job
(`.github/workflows/pr-gate.yml`) on every `pull_request`. It polls
the GitHub Checks API for the conclusions of the upstream
default-PR lanes that this rollout treats as authoritative:

* **CI Core Success** — the `ci-core-success` job inside
  `.github/workflows/ci-core.yml`. Already aggregates Build & Test,
  Clippy, Documentation, and BDD Grid Check.
* **Policy** — the strict-policy lane added in PR 04
  (`.github/workflows/policy.yml`). Runs `xtask check-file-policy`
  with `--fail-on-error`, plus advisory runs of
  `ci-lane-whitelist`, `lint-inheritance`, `clippy-exceptions`,
  `no-panic-family`, and `policy-report`.
* **`pr-check (no-features)`**, **`pr-check (cpu)`**, **`pr-check (cpu+full-cli)`** —
  the 3-combo PR matrix in `.github/workflows/feature-matrix.yml`.
  The `pr-check` matrix has no aggregator job, so each matrix label
  is listed individually.

Lanes that are **not** required by `PR Gate Success`:

* `ripr static exposure` (advisory only)
* All macOS / GPU / Docker / Coverage / Crossval / Property /
  Model-validation lanes (label- or path-gated; opting in is the
  PR author's decision)
* Any deep / nightly / labelled lane

This list is intentional. Making the long-tail lanes required
defeats the rollout's LEM-economics goal: `> 95%` of PRs should
land for `< 35` LEM with the long-tail lanes opt-in.

## Posture in this PR

`PR Gate Success` is **not yet branch-protection-required**. The
workflow runs on every PR and reports a single check, but branch
protection still gates on the existing `ci-core-success` summary.
The migration to `PR Gate Success` happens in a separate change
once one full sprint of these conclusions has been observed and
the timing / flake characteristics are understood.

When the migration happens, the change is in
`Settings → Branches → main → Required status checks`:

* **Add:** `PR Gate Success`
* **Remove:** every individual leaf-job name currently required
  (e.g. `Build & Test (ubuntu-latest)`, `Clippy`, `Documentation`,
  `BDD Grid Check`, `Feature Matrix PR`)

After the migration, branch protection has exactly one required
check (`PR Gate Success`), and that check is the single source of
truth for "is this PR ready to merge?".

## Aggregation strategy

GitHub Actions does not allow `needs:` across workflows when both
fire on the same `pull_request` event. Two patterns are common:

| Pattern        | Trade-off                                           |
| -------------- | --------------------------------------------------- |
| `workflow_run` | Loses the "single check on the PR head" UX         |
| Checks-API poll | Single check; pays a polling job (~25 min budget)  |

This PR uses the Checks-API poll pattern because branch protection
wants a single required check that returns a single conclusion on
the PR head. The poll uses a 25-minute deadline (50 × 30 s); most
default PRs converge in 10–15 minutes.

## Failure modes

| Upstream lane status      | PR Gate Success verdict                |
| ------------------------- | -------------------------------------- |
| All `success`             | `success`                              |
| Any `failure`             | `failure`                              |
| Any `cancelled`/`timed_out`| `failure`                             |
| Required lane `skipped`   | `failure` (required lanes must run)    |
| Pending after 25 minutes  | `failure` (timeout)                    |

Optional lanes (none today) would be allowed to be `skipped`. The
right place to encode "lane X is conditionally required when paths
Y change" is the risk-pack routing layer added in PR 17, not the
PR Gate Success workflow itself.

## Operational notes

* The aggregator depends on the `gh` CLI being available on the
  runner. `ubuntu-22.04` ships with it.
* `permissions: { checks: read, actions: read }` is sufficient to
  read the upstream check conclusions; no write permissions are
  granted.
* The aggregator is idempotent: re-running it picks up the latest
  upstream conclusions and re-evaluates.
