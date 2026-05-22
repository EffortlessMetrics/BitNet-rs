# Dense Qwen2.5 Optimization

Qwen2.5 0.5B Q8_0 is dense CUDA product CLI-ready and exact-profile
server-ready. Speedup, full residency, and BitNet proof remain false.

## Work item: CUDA-DENSE-QWEN25-OPS-001

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `nvidia-5070ti`
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

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: exact-profile status updates
Blocked by: none

### Goal

Review Qwen2.5 speed/residency requalification after optimization.

### Production delta

This slice records the post-OPS-002/OPS-003 requalification review. No speed,
benchmark-qualified, full-residency, broad dense GGUF, or BitNet proof claim is
promoted because the reviewed CUDA profiles remain slower than same-artifact CPU
means, pure H2D timing is still unavailable, and logits-transfer accounting
still records full-logits D2H until a device top-k sampler exists.

### Non-goals

No broad speedup or full residency without proof.

### Acceptance

Model coverage and status docs agree with the governed review decision:
`benchmark_qualified=false`, `speedup_claim=false`,
`full_residency_claim=false`, exact-profile `server_ready=true`, and
`bitnet_packed_i2s_qk256_proof=false`.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Revert the review report and restore the previous next-proof text. No claim
booleans were promoted by this work item.

## Work item: CUDA-DENSE-QWEN25-OPS-004

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: refreshed exact-profile comparator after reduced-D2H proof receipts
Consumed by: PR #6098 device top-k proof-receipt path
Blocked by: CUDA-DENSE-QWEN25-PERF-007

### Goal

Harden the Qwen2.5 logits-transfer reduction contract before runtime work can
claim reduced D2H bytes.

### Production delta

PR #6076 added `docs/reports/CUDA_DENSE_QWEN25_OPS_004_LOGITS_TRANSFER_REDUCTION_CONTRACT.md`
and tightened dense Qwen short-decode and warm-session receipt validation so a
`device_to_host_bytes_reduced=true` claim is rejected unless the receipt also
proves a CUDA device-side top-k or greedy sampler path,
`sampling_location=cuda_device`, measured D2H bytes below the full-logits
envelope, selected-token equality, top-k evidence preservation, and unchanged
quality receipts.

Follow-on PR #6098 added the device top-k proof-receipt path that consumes this
contract. It did not promote Qwen2.5 speedup, benchmark-qualified speed, full
residency, server readiness, broad dense GGUF support, or BitNet QK256 proof.

### Non-goals

OPS-004 itself did not add a runtime sampler, CUDA kernel, benchmark
requalification, speedup promotion, full-residency promotion,
server-readiness change, or BitNet proof change.

### Acceptance

Reduced-transfer receipts with the CPU full-logits sampler fail validation.
Reduced-transfer receipts with `device_top_k_cuda_sampler` or
`device_greedy_cuda_sampler`, `sampling_location=cuda_device`, correct byte
math, selected-token equality, top-k evidence, and unchanged quality receipts
can validate.

### Proof commands

```bash
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features dense_gguf_qwen_short_decode
cargo run --locked -p xtask --no-default-features -- check-model-coverage
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Rollback

Revert PR #6076 receipt-validator tightening and the OPS-004 report. Existing
Qwen2.5 product CLI and exact-profile server readiness evidence remains
unchanged.
