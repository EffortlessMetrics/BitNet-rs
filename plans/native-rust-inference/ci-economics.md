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
