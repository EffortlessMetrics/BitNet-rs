# Dense Qwen2.5 Optimization

Qwen2.5 0.5B Q8_0 is dense CUDA product CLI-ready and exact-profile
server-ready. Speedup, full residency, and BitNet proof remain false.

## Work item: CUDA-DENSE-QWEN25-OPS-001

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-DENSE-QWEN25-OPS-002
Blocked by: native inference plan

### Goal

Produce `docs/reports/CUDA_DENSE_QWEN25_RESIDENCY_BOTTLENECKS.md`.

### Production delta

Report ranks model load, H2D upload, D2H logits, launch count, KV movement,
workspace reuse, and per-token wall-time blockers.

### Non-goals

No optimization or claim promotion.

### Acceptance

Report cites one-token, short-decode, warm-session, benchmark review, H2D/D2H,
and server readiness receipts.

### Proof commands

```bash
git diff --check
```

### Rollback

Revert the report.

## Work item: CUDA-DENSE-QWEN25-OPS-002

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-DENSE-QWEN25-OPS-003
Blocked by: CUDA-DENSE-QWEN25-OPS-001

### Goal

Add persistent handles for dense Qwen2.5.

### Production delta

Avoid per-request model/context/weight setup where the route can safely reuse
state.

### Non-goals

No speedup or full-residency claim.

### Acceptance

Receipt shows `model_loaded_once=true`, `cuda_context_once=true`,
`weights_uploaded_once=true`, `per_request_model_load=false`,
`workspace_reused=true`, and `fallback_used=false`.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- chat --device cuda --model <qwen25>
```

### Rollback

Return to previous lifecycle and keep existing exact-profile server readiness.

## Work item: CUDA-DENSE-QWEN25-OPS-003

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-DENSE-QWEN25-PERF-007
Blocked by: CUDA-DENSE-QWEN25-OPS-002

### Goal

Reduce logits/top-k transfer when greedy or top-k proof is sufficient.

### Production delta

Reduce D2H bytes or explain why reduction is not possible while preserving
selected-token equality and top-k evidence.

### Non-goals

No quality regression and no speedup claim.

### Acceptance

Quality receipts remain unchanged and D2H byte change is recorded.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- ask --device cuda --model <qwen25> "..."
```

### Rollback

Restore full logits transfer.

## Work item: CUDA-DENSE-QWEN25-PERF-007

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: exact-profile status updates
Blocked by: CUDA-DENSE-QWEN25-OPS-002, CUDA-DENSE-QWEN25-OPS-003

### Goal

Review Qwen2.5 speed/residency requalification after optimization.

### Production delta

Accept or reject exact profiles; keep BitNet proof false.

### Non-goals

No broad speedup or full residency without proof.

### Acceptance

Model coverage and status docs agree with governed review decisions.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Demote any accepted claims whose receipts do not satisfy the spec.
