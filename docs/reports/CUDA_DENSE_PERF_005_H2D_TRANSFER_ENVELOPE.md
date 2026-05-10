# CUDA-DENSE-PERF-005 H2D Transfer Envelope

## Scope

`CUDA-DENSE-PERF-005` adds host-to-device timing evidence to the dense Qwen
strict CUDA runtime receipts without upgrading any speed or residency claim.

The measured value is intentionally labeled as:

```text
host_to_device_ms_source: wall_clock_model_load_with_cuda_weight_upload
host_to_device_ms_scope: model_load_wall_clock_envelope
host_to_device_ms_includes_non_transfer_overhead: true
transfer_timing_status: host_to_device_model_load_envelope_device_to_host_measured
```

This is a model-load wall-clock envelope around the CUDA load/upload phase, not
pure CUDA event copy timing.

## Receipts

Generated on the RTX 5070 Ti lane with the verified
`qwen2.5-0.5b-instruct-q8_0.gguf` artifact:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-005-h2d-transfer-envelope/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-005-h2d-transfer-envelope/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-005-h2d-transfer-envelope/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json
```

## Evidence

| Profile | H2D envelope ms | D2H logits ms | Transfer timing status |
| --- | ---: | ---: | --- |
| `one_token` | 3513.8495 | 0.8953 | `host_to_device_model_load_envelope_device_to_host_measured` |
| `short_decode_8` | 3419.3919 | 6.5654 | `host_to_device_model_load_envelope_device_to_host_measured` |
| `warm_session_3_turns` | 3526.1035 | 19.2179 | `host_to_device_model_load_envelope_device_to_host_measured` |

All three receipts preserve:

```text
selected_route: dense_regular_llm_cuda
fallback_used: false
speedup_claim: false
benchmark_qualified_speedup: false
bitnet_packed_i2s_qk256_proof: false
```

## Validator

The dense Qwen receipt validator now accepts both historical D2H-only receipts
and the new H2D-envelope receipts. For new H2D-envelope receipts it requires:

- non-negative `host_to_device_ms`;
- source `wall_clock_model_load_with_cuda_weight_upload`;
- scope `model_load_wall_clock_envelope`;
- `host_to_device_ms_includes_non_transfer_overhead=true`;
- matching `timing.host_to_device_ms` and
  `tensor_residency.transfer_accounting.host_to_device_ms`.

## Claim Boundary

May claim:

- dense Qwen strict CUDA runtime receipts record a measured H2D model-load
  wall-clock envelope;
- D2H logits download timing remains measured;
- the receipts remain same-artifact, RTX 5070 Ti, dense_regular_llm_cuda, and
  fallback-free.

Must not claim:

- pure CUDA event H2D copy timing;
- accepted dense Qwen CUDA speedup;
- `benchmark_qualified_speedup=true`;
- full CUDA residency;
- server readiness;
- BitNet packed I2_S/QK256 proof from dense CUDA evidence.

## Validation

Validation used CUDA Toolkit v12.9 on PATH for CUDA-enabled Cargo commands.

```text
cargo fmt -p bitnet-cli -p bitnet-receipts-core -p bitnet-receipts
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features qwen_ -- --nocapture
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-one-token-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-005-h2d-transfer-envelope\dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-short-decode-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --one-token-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-005-h2d-transfer-envelope\dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-005-h2d-transfer-envelope\dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-warm-session-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --one-token-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-005-h2d-transfer-envelope\dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json --short-decode-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-005-h2d-transfer-envelope\dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-10\dense-qwen-perf-005-h2d-transfer-envelope\dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json
```
