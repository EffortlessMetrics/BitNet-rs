# Runtime Performance Work

Runtime performance work implements
[BITNET-SPEC-0014](../../docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md).

## Work item: PERF-CONTRACT-001

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: benchmark qualification reviews
Blocked by: native inference plan

### Goal

Add receipt schema support for the runtime performance fields.

### Production delta

Benchmark and receipt explanation paths can report phase timing, transfer,
launch, VRAM, power/thermal context, fallback, profile, backend, and route
identity.

### Non-goals

No speedup claim.

### Acceptance

Missing phase fields are explicit `not_applicable` or `missing`, not zero.

### Proof commands

```bash
cargo test --workspace --no-default-features --features cpu
git diff --check
```

### Rollback

Remove new schema fields and preserve older receipt decoding where possible.

## Work item: PERF-CONTRACT-002

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: exact-profile speed reviews
Blocked by: PERF-CONTRACT-001

### Goal

Teach `receipts explain` to separate reported metrics from accepted claims.

### Production delta

Users can see TTFT, throughput, speedup, residency, and server readiness status
without assuming raw timing equals a promoted claim.

### Non-goals

No model promotion.

### Acceptance

Explanation emits `qualified`, `reported_only`, `not_available`, `accepted`,
`rejected`, or `not_reviewed` states where appropriate.

### Proof commands

```bash
cargo test --workspace --no-default-features --features cpu
```

### Rollback

Revert explain output changes.
