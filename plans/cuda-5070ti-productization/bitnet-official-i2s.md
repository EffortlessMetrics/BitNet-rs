# Official BitNet 2B I2_S/QK256 CUDA Product Path

This file owns the official Microsoft BitNet 2B I2_S/QK256 path on the
9950X3D + RTX 5070 Ti product bench.

## Current Claim

The current campaign and model coverage matrix say the official I2_S/QK256 row
is `product_cli_ready`, `accelerator_answer_ready=true`, and
`speedup_claim=false`.

Allowed route:

```text
model_class = bitnet
route = bitnet_qk256_cuda
selected_backend = nvidia-rtx-5070-ti-cuda
cpu_reference = amd-9950x3d-cpu-avx512
bitnet_packed_i2s_qk256_proof = true
dense_regular_llm_cuda_proof = false
```

## Canonical Receipts

Existing proof should be reconciled against:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-proof.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-short-decode.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/strict-cuda-ask-math.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-002-repeated-strict-ask.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json
```

## Next PRs

### CUDA-PROD-008: Reconcile Proof State

Link:
`docs/tracking/campaigns/nvidia-5070ti/active.toml`

Acceptance:

- Current BitNet row lists last strict ask, corpus, parity, warm-session, and
  benchmark receipts.
- Next missing proof is benchmark qualification, not basic CUDA execution.
- `speedup_claim=false` remains visible.

Proof:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Rollback: revert the docs reconciliation.

### CUDA-PROD-009: Strict Ask/Chat Preflight

Acceptance:

- `bitnet cuda doctor` and strict ask/chat paths expose selected backend,
  tokenizer authority, prompt authority, QK256 route readiness, receipt path,
  fallback status, and speed claim status.
- Missing tokenizer, generic CUDA proof ambiguity, and CPU fallback fail closed.

Proof:

```bash
cargo test --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli cuda_doctor ask_strict
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
git diff --check
```

Rollback: revert the CLI preflight changes; keep historical receipts.

### CUDA-PROD-010: Benchmark Qualification

Profiles:

```text
one_token
short_decode_8
short_decode_32
warm_session_3_turns
warm_session_10_turns
```

Acceptance:

- Each profile records CPU and CUDA distributions, transfer timing sources,
  kernel timing, VRAM high-water mark, fallback status, decision, and reason.
- Speedup can be accepted or rejected per exact profile.
- There is no global CUDA speedup claim.

Proof:

```bash
cargo test --locked -p bitnet-bench-receipts --no-default-features
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --cuda-benchmark-receipt <receipt>
git diff --check
```

Rollback: remove the new qualification receipts and demote any status row that
depends on them.

## Claim Boundary

This lane may claim official BitNet I2_S/QK256 strict CUDA proof only for
committed receipts. It must not claim dense SLM proof, server readiness, full
CUDA residency, or speedup outside accepted profiles.
