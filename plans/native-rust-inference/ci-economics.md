# CI Economics Rollout

The default PR lane should be Linux-only, model-free, hardware-free,
Docker-free, coverage-free, and crate/risk scoped. Expensive proof remains
available through labels, main, schedule, release, workflow dispatch, or
hardware campaigns.

## Work item: CI-1

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md`
Linked ADRs:
Campaign:
Blocks: CI-2
Blocked by: native inference plan

### Goal

Remove macOS from ordinary PRs.

### Production delta

macOS runs on main/manual/merge-group/path-risk/labels instead of every
ordinary PR.

### Non-goals

No weakening of release or risk-routed proof.

### Acceptance

Default PRs no longer run macOS format or clippy lanes unless selected by
policy.

### Proof commands

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### Rollback

Restore macOS PR triggers.

## Work item: CI-2

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs:
Campaign:
Blocks: CI-3
Blocked by: CI-1

### Goal

Move performance tracking off default PRs.

### Production delta

Performance runs on main, schedule, workflow dispatch, or performance labels.

### Non-goals

No removal of performance proof.

### Acceptance

Skipped PR performance lanes report policy skip instead of pass.

### Proof commands

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### Rollback

Restore default PR performance trigger.

## Work item: CI-3

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md`
Linked ADRs:
Campaign:
Blocks: CI-5
Blocked by: CI-1

### Goal

Preserve in-flight CI evidence for bounded Apple and hardware-adjacent lanes.

### Production delta

Selected Apple Silicon, Apple M4 SLM eval, and Apple M4 inference-ops tier-0
jobs use timeout caps as the budget control and no longer cancel started runs
on a newer push to the same PR/ref.

### Non-goals

No expansion of default PR coverage, no live model downloads, and no removal of
job-level `timeout-minutes` caps.

### Acceptance

Selected bounded Apple lanes set `cancel-in-progress: false` and document why
timeouts, not cancellation, are the cap for long-running evidence jobs.

### Proof commands

```bash
git diff --check
rg -n "cancel-in-progress: true" .github/workflows/apple-silicon.yml .github/workflows/apple-m4-slm-eval-tier0.yml .github/workflows/apple-m4-inference-ops-tier0.yml
```

### Rollback

Restore `cancel-in-progress: true` for the affected workflows.

## Work item: CI-5

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md`
Linked ADRs:
Campaign:
Blocks: CI-6, CI-7, CI-8, CI-9
Blocked by: CI-1, CI-2

### Goal

Emit stable `ci-plan.json` routing schema.

### Production delta

PR Gate can distinguish selected blocking lanes, advisory lanes, skipped lanes,
changed packages, canaries, and budget estimates.

### Non-goals

No broad CI expansion.

### Acceptance

Schema version, classification booleans, selected/skipped lanes, package set,
and LEM budget fields are stable.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- ci-plan --format json
git diff --check
```

### Rollback

Remove the generated plan and keep current PR Gate behavior.
