# CUDA-DENSE-PERF-006 H2D Envelope Qualification

## Scope

`CUDA-DENSE-PERF-006` refreshes the dense Qwen benchmark qualification review
to consume the `CUDA-DENSE-PERF-005` strict CUDA runtime receipts. Those receipts
record an H2D model-load wall-clock envelope:

```text
host_to_device_ms_source: wall_clock_model_load_with_cuda_weight_upload
host_to_device_ms_scope: model_load_wall_clock_envelope
host_to_device_ms_includes_non_transfer_overhead: true
```

The review keeps pure CUDA event H2D copy timing blocked.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-qwen-perf-006-h2d-envelope-qualification/dense-gguf-qwen-benchmark-qualification-h2d-envelope.json
```

## Evidence

| Profile | H2D envelope ms | D2H logits ms | CUDA mean total ms | CPU mean total ms |
| --- | ---: | ---: | ---: | ---: |
| `one_token` | 3513.8495 | 0.8953 | 3978.5710 | 2872.8428 |
| `short_decode_8` | 3419.3919 | 6.5654 | 4199.9896 | 3528.0687 |
| `warm_session_3_turns` | 3526.1035 | 19.2179 | 5034.9288 | 4596.1352 |

## Qualification Decision

No dense Qwen CUDA profile is speedup-qualified.

Blocked requirements:

- CUDA mean total time remains slower than the same-artifact CPU reference mean
  for every reviewed profile.
- Pure host-to-device CUDA event copy timing is still unmeasured; the current H2D
  value is a model-load envelope that includes non-transfer overhead.
- No profile-specific speedup threshold has been accepted.

## Claim Boundary

May claim:

- dense Qwen benchmark qualification consumes H2D model-load envelope receipts;
- profile reviews record H2D envelope timing, source, scope, and overhead flag;
- D2H logits timing remains measured.

Must not claim:

- pure CUDA event H2D copy timing;
- accepted dense Qwen CUDA speedup;
- `benchmark_qualified_speedup=true`;
- full CUDA residency;
- server readiness;
- BitNet packed I2_S/QK256 proof from dense CUDA evidence.

## Validation

```text
cargo fmt -p bitnet-bench-receipts -- --check
cargo test --locked -p bitnet-bench-receipts --no-default-features dense_gguf_qwen_benchmark_qualification -- --nocapture
cargo run --locked -p bitnet-bench-receipts --bin dense_qwen_cuda_benchmark_qualification_receipt --no-default-features
```
