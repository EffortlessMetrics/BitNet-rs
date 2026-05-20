# BitNet Source Sync

This repository is the high-volume same-repo PR workspace for BitNet-rs.
`EffortlessMetrics/BitNet-rs` remains the public source of truth until the
controlled cutover completes.

## 2026-05-20 source refresh

Source repo: `EffortlessMetrics/BitNet-rs`

Source `main` imported through:

```text
71f19f095785f1883e32ebdce34ee1da768d449d
```

Swarm base before import:

```text
7baa4fdd4a4e1ba58dc4928bbce6d78bcfd5857f
```

Preserved swarm-only route/proof files:

```text
.github/workflows/em-ci-routed-rust.yml
docs/ci/routed-verification-rollout.md
docs/development/runner-baseline.md
docs/migrations/lunar-lake-from-bitnet-rs.md
```

`release.yml` was kept from the swarm base for this sync so release, signing,
and publish behavior remains owned by the public source repo until a deliberate
follow-up move.
