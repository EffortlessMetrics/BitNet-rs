# CUDA-DENSE-PERF-001 Benchmark Baseline

## Scope

`CUDA-DENSE-PERF-001` normalizes existing dense Qwen RTX 5070 Ti evidence into
a benchmark baseline receipt. It consumes the committed one-token, short-decode,
and warm-session strict CUDA proof receipts for the SHA-verified
`qwen2.5-0.5b-instruct-q8_0.gguf` artifact.

This is a receipt and validation slice. It does not change tokenizer behavior,
loader behavior, transformer runtime behavior, CUDA kernel math, BitNet QK256
code, or server behavior.

## What It Records

The committed receipt records:

```text
artifact_kind: dense_gguf_qwen_cuda_benchmark_baseline
selected_route: dense_regular_llm_cuda
selected_backend: nvidia-rtx-5070-ti-cuda
fallback_used: false
profiles: one_token, short_decode_8, warm_session_3_turns
total_kernel_invocations: 11154
total_kernel_time_ms: 13182.4159
total_host_to_device_bytes: 2027132448
total_device_to_host_bytes: 20055552
speedup_claim: false
benchmark_qualified_speedup: false
full_cuda_residency_claimed: false
bitnet_packed_i2s_qk256_proof: false
```

Each source receipt is referenced by path and SHA256 so later repeated
CPU/CUDA comparator work can distinguish baseline evidence from accepted
performance claims.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json
```

## Non-Claims

This PR does not claim:

```text
dense Qwen speedup
benchmark_qualified_speedup
full CUDA residency
server readiness
BitNet packed I2_S/QK256 proof
runtime tokenizer, loader, transformer, or CUDA kernel math changes
```

The receipt explicitly keeps profile-specific speedup qualification blocked
until a later repeated same-artifact CPU/CUDA comparator and threshold review
exist.

## Validation

Expected validation:

```text
cargo run --locked -p bitnet-bench-receipts --bin dense_qwen_cuda_benchmark_baseline_receipt --no-default-features
cargo test --locked -p bitnet-bench-receipts --no-default-features dense_gguf_qwen_cuda_benchmark_baseline -- --nocapture
cargo fmt -p bitnet-bench-receipts -- --check
cargo run --release --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --release --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
