# PR Gate Success

`PR Gate Success` is the single required check that branch
protection should gate merges on. It is the deliverable of PR 19 in
the strict policy / CI economics rollout.

## What it aggregates

The aggregator runs as a single GitHub Actions job
(`.github/workflows/pr-gate.yml`) on every `pull_request`. It computes
`ci-plan.json` with `xtask ci plan`, reads `selected_lanes[]`, and polls the
GitHub Checks API for the check runs mapped from lanes with `blocking = true`.

Common selected blocking lanes map to these upstream checks:

* **`ci-core-build-test`** -> **CI Core Success**.
* **`feature-matrix-pr`** -> **`pr-check (no-features)`** and
  **`pr-check (cpu)`**.
* **`feature-matrix-full-cli`** -> **`pr-check (cpu+full-cli)`** when
  CLI/server/validation/model-cache paths or the `full-cli` label select it.
* **`policy`** -> **Policy**.
* **`compatibility-msrv`** -> **Route MSRV Compatibility** and
  **Minimum Supported Rust Version**.
* **`gpu-native`** -> the five **`native-check (...)`** GPU-matrix compile
  checks when GPU paths are changed and a `gpu`, `gpu-ci`, or `full-ci` label
  makes `gpu-ci-matrix.yml` emit those PR jobs.
* **`always-on-guards`** -> **Guards** and **Check PR Size**. If the PR has
  `mechanical-change` or `ai-native`, **PR Size Guard** intentionally skips
  **Check PR Size**; in that acknowledged-large-PR case, **PR Gate Success**
  requires **Guards** but does not wait for **Check PR Size**.

Lanes that are **not** required by `PR Gate Success`:

* `ripr static exposure` (advisory static mutation-exposure analysis)
* All macOS, GPU Docker, Coverage, Crossval, Property, and Model-validation
  lanes (label- or path-gated; opting in is the PR author's decision)
* Any deep / nightly / labelled lane

This list is intentional. Making the long-tail lanes required
defeats the rollout's LEM-economics goal: `> 95%` of PRs should
land for `< 35` LEM with the long-tail lanes opt-in.

## Posture

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
| Checks-API poll | Single check; pays a polling job when upstream lanes are slow |

This PR uses the Checks-API poll pattern because branch protection
wants a single required check that returns a single conclusion on
the PR head. The gate recomputes the same `ci-plan.json` as PR Plan
instead of duplicating path-classification logic. The poll uses a
55-minute deadline (110 × 30 s); most default PRs converge in 10–15
minutes, while slower selected lanes still have enough room for their
final rollup job. The aggregator deadline must exceed the healthy cap of
any selected upstream lane plus rollup and status propagation cushion.

## Failure modes

| `ci-plan.json` lane state | Upstream check status | PR Gate Success verdict |
| ------------------------- | --------------------- | ----------------------- |
| Selected blocking         | `success`             | `success` when all selected blocking checks pass |
| Selected blocking         | `failure`, `cancelled`, or `timed_out` | `failure` |
| Selected blocking         | `skipped`             | `failure` |
| Selected blocking         | Pending after 55 minutes | `failure` |
| Selected advisory         | Any status            | Reported, not blocking |
| Unselected                | `skipped` or missing  | Allowed |

Exception: **Check PR Size** may be omitted from the required upstream set when
the PR carries `mechanical-change` or `ai-native`, because
`.github/workflows/pr-size-guard.yml` deliberately skips that job for
acknowledged large PRs.

If `ci-plan.json` selects a blocking lane that PR Gate does not know how to map
to check names, the gate fails closed. That makes route/schema drift visible
instead of silently treating a selected proof lane as advisory.

## Operational notes

* The aggregator depends on the `gh` CLI and `jq` being available on the
  runner. `ubuntu-22.04` ships with both.
* The workflow checks out the PR and runs `cargo run --locked -p xtask
  --no-default-features -- ci plan` before polling upstream checks.
* `permissions: { checks: read, actions: read }` is sufficient to
  read the upstream check conclusions; no write permissions are
  granted.
* The aggregator is idempotent: re-running it picks up the latest
  upstream conclusions and re-evaluates.
