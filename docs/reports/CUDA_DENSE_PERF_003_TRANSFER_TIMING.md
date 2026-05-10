# CUDA-DENSE-PERF-003 Transfer Timing

## Scope

`CUDA-DENSE-PERF-003` adds measured device-to-host logits download timing to
the dense Qwen strict CUDA runtime receipts. It uses the verified
`qwen2.5-0.5b-instruct-q8_0.gguf` artifact on the RTX 5070 Ti and keeps the
existing dense regular-LLM CUDA claim boundary.

This is receipt instrumentation and validation. It does not change tokenizer
behavior, loader behavior, transformer runtime behavior, CUDA kernel math,
BitNet QK256 code, dense ask/chat UX, or server behavior.

## What It Records

The committed receipts record:

```text
selected_route: dense_regular_llm_cuda
selected_backend: nvidia-rtx-5070-ti-cuda
transfer_timing_status: device_to_host_measured_host_to_device_unmeasured
host_to_device_ms: null
host_to_device_ms_source: not_measured_by_dense_qwen_runtime
device_to_host_ms_source: wall_clock_extract_logits_2d_local
fallback_used: false
speedup_claim: false
benchmark_qualified_speedup: false
full_cuda_residency_claimed: false
server_ready_claimed: false
bitnet_packed_i2s_qk256_proof: false
```

Measured logits download timing from the generated receipts:

| Receipt | D2H logits download ms | H2D timing |
| --- | ---: | --- |
| `dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json` | 0.8534 | explicitly unmeasured |
| `dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json` | 6.3089 | explicitly unmeasured |
| `dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json` | 18.7415 | explicitly unmeasured |

## Receipts

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-003-transfer-timing/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-003-transfer-timing/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-003-transfer-timing/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json
```

Receipt SHA256 values:

| Receipt | SHA256 |
| --- | --- |
| one-token | `0b74e8d094a341a10710e3ec7c70a94062f67246bdff4f30fd3e9612c94c4ec4` |
| short-decode | `f72609cf52764f5e05aa0c981622c5333720794bcb87be59810722cf35beb502` |
| warm-session | `ecf07d473bfdad091b806887e0744cc075af331e306b7a2fb9ad6c55d2f5d6ff` |

## Non-Claims

This PR does not claim:

```text
dense Qwen speedup
benchmark_qualified_speedup
full CUDA residency
server readiness
BitNet packed I2_S/QK256 proof
dense ask/chat readiness
host-to-device transfer timing
runtime tokenizer, loader, transformer, or CUDA kernel math changes
```

## Validation

Expected validation:

```text
cargo fmt -p bitnet-cli -p bitnet-receipts-core -p bitnet-receipts -- --check
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features qwen_ -- --nocapture
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-one-token-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-003-transfer-timing\dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-short-decode-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --one-token-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-003-transfer-timing\dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-003-transfer-timing\dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-warm-session-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --one-token-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-003-transfer-timing\dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json --short-decode-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-003-transfer-timing\dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-003-transfer-timing\dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json
cargo run --release --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --release --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
