# Dense Qwen2.5 Optimization

Qwen2.5 0.5B Q8_0 is dense CUDA product CLI-ready and exact-profile
server-ready. Speedup, full residency, and BitNet proof remain false.

## Work item: CUDA-DENSE-QWEN25-OPS-001

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-DENSE-QWEN25-OPS-002
Blocked by: none

### Goal

Produce `docs/reports/CUDA_DENSE_QWEN25_RESIDENCY_BOTTLENECKS.md`.

### Production delta

Report ranks model load, H2D upload, D2H logits, launch count, KV movement,
workspace reuse, and per-token wall-time blockers. Landed in PR #5985.

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

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-DENSE-QWEN25-OPS-003
Blocked by: none

### Goal

Add persistent handles for dense Qwen2.5.

### Production delta

Dense Qwen warm-session receipts now expose and validate stable persistent-handle
aliases for one model load, one CUDA context, upload-once weights, no
per-request model load, workspace reuse, and fallback false. Landed in PR #5995.

### Non-goals

No speedup or full-residency claim.

### Acceptance

Receipt shows `model_loaded_once=true`, `cuda_context_once=true`,
`weights_uploaded_once=true`, `per_request_model_load=false`,
`workspace_reused=true`, and `fallback_used=false`.

### Proof commands

```bash
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation dense_gguf_qwen_warm_session
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain
```

### Rollback

Revert PR #5995 receipt aliases and validators while keeping existing
exact-profile server readiness.

## Work item: CUDA-DENSE-QWEN25-OPS-003

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-DENSE-QWEN25-PERF-007
Blocked by: none

### Goal

Reduce logits/top-k transfer when greedy or top-k proof is sufficient.

### Production delta

Dense Qwen short-decode and warm-session receipts now expose and validate
logits-transfer accounting. PR #6010 records that D2H bytes are not reduced
yet because the CPU sampler still requires full logits until a device top-k
sampler exists, while preserving selected-token equality, top-k evidence, and
quality evidence.

### Non-goals

No quality regression and no speedup claim.

### Acceptance

Quality receipts remain unchanged, D2H byte accounting is recorded, and any
future `device_to_host_bytes_reduced=true` claim must prove actual D2H bytes
fell below the full-logits envelope.

### Proof commands

```bash
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation dense_gguf_qwen_short_decode
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation dense_gguf_qwen_warm_session
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation
git diff --check
```

### Rollback

Revert PR #6010 receipt aliases and validators. Existing Qwen2.5 product CLI
and exact-profile server readiness evidence remains unchanged.

## Work item: CUDA-DENSE-QWEN25-PERF-007

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: exact-profile status updates
Blocked by: none

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
