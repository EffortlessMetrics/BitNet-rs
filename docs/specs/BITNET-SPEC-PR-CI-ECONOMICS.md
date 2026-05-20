# BITNET-SPEC-PR-CI-ECONOMICS: PR Write-Action CI Economics

Status: proposed
Owner: release/ci
Created: 2026-05-19
Linked proposal: n/a
Linked specs:
[BITNET-SPEC-PR-QUEUE-DISPOSITION](BITNET-SPEC-PR-QUEUE-DISPOSITION.md)
Linked ADRs:
[BITNET-ADR-0006](../adr/BITNET-ADR-0006-pr-closure-creates-backlog.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: `policy/pr-ci-actions.toml`

## Purpose

PR queue work can burn CI without changing code. Closing, reopening, rebasing,
pushing, retargeting, labeling, recreating, and rerunning workflows are write
actions. Many of them can trigger webhooks, status recomputation, workflow
selection, reviewer notifications, or branch-protection recalculation.

This spec defines when PR write actions are allowed during burn-down and when CI
spend is justified. The rule is simple: no CI for archaeology. Hosted CI should
be spent on approved merge candidates, approved clean ports, branch refreshes
needed to prove those candidates, and required proof.

## Source-Of-Truth Authorities

PR CI economics truth lives in:

- this spec;
- [CI Cost and Verification Policy](../ci/cost-and-verification-policy.md);
- `policy/pr-ci-actions.toml`;
- [PR queue disposition spec](BITNET-SPEC-PR-QUEUE-DISPOSITION.md);
- PR bodies and comments that explain why a write action was necessary.

Workflow files implement routing. They do not make bulk write actions safe by
default.

## Write Actions

These actions are writes and must be treated as potentially CI-relevant:

| Action | Risk | Required condition |
| --- | --- | --- |
| close PR | Hides backlog and may trigger state churn | Valid disposition under PR queue spec |
| reopen PR | Re-enters queue and may retrigger checks | Reason and intended next action |
| push branch | Triggers CI on the PR branch | Merge candidate, clean port, or required proof |
| rebase/merge main | Triggers CI on refreshed branch | Needed for current merge/proof decision |
| retarget PR | Changes review and merge context | Explicit base correction or lane decision |
| label PR | May select optional workflows | Label must match real risk/proof need |
| rerun workflow | Spends CI again | Failed required check, known flake, or fixed input |
| recreate PR | Loses identity and triggers CI | Allowed only under PR disposition successor rules |

Bulk versions of these actions require explicit approval unless the policy file
marks a narrower safe case. The default is no bulk close, no bulk reopen, no
bulk recreate, and no one-for-one replacement wave.

## Read-Only Queue Work

Read-only archaeology is allowed and should be preferred before writes:

```text
gh pr view
gh pr diff
gh api reads
existing reports
existing CI logs
local git show/log/diff
```

Read-only review can classify a PR, find a successor, identify a stale stack, or
decide that a branch is a merge candidate. It should not be followed by a write
unless one of the required conditions is true.

## Allowed CI Spend During Burn-Down

CI spend is allowed for:

- approved merge candidates;
- approved clean ports;
- branch refreshes needed to prove a merge candidate;
- focused proof required by the selected spec or plan;
- reruns of failed required checks when the failure is plausibly flaky or the
  input has changed;
- reviewer or branch-protection requirements for the current PR.

CI spend is not allowed for:

- archaeology without a merge or proof decision;
- bulk close or reopen waves;
- recreating valid PRs to reduce visible queue size;
- refreshing stale stacks only to inspect them;
- rerunning optional workflows without a proof need;
- label changes that select expensive lanes without matching risk.

## Rerun Rules

Rerunning a workflow must name one of these reasons:

- required check failed and the log points to an infrastructure or flake class;
- PR input changed after a fix or restack;
- branch protection requires a fresh check on the current head;
- the selected plan requires the proof and the previous receipt is stale.

Do not rerun a workflow simply because a historical PR has old red checks.

## Acceptance Examples

| Case | Required handling |
| --- | --- |
| PR is a current-main merge candidate | Refresh if needed, run required CI, merge when green |
| PR is stale but content may be durable | Review read-only first; restack only if it becomes a candidate |
| PR will close because a successor landed | Close with disposition; no workflow rerun needed |
| PR has future work and no successor | Create/link tracking before close; do not rerun CI |
| Optional hardware label would select expensive lanes | Apply only when the proof requires hardware |
| Historical PR has red checks from months ago | Do not rerun for archaeology |

## Proof Commands

Current contract validation:

```bash
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
git diff --check
```

Future enforcement may load `policy/pr-ci-actions.toml` in a PR action checker
and fail close/reopen/recreate records that lack the required disposition,
reason, approval, or proof context.

## Non-Goals

- Do not encode today's exact open PR order.
- Do not change workflow routing in this spec PR.
- Do not disable required CI for real merge candidates.
- Do not treat optional pending jobs as required gates unless branch protection
  or the selected plan says so.
- Do not use CI spend as a substitute for content audit.
