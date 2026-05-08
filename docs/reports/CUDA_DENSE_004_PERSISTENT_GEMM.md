# CUDA-DENSE-004 Persistent Dense GEMM Fixture

Date: 2026-05-08

## Scope

`CUDA-DENSE-004` extends the dense regular-LLM CUDA lane from single-launch
tensor residency to a persistent deterministic fixture session. It records that
the FP16 GEMM fixture creates one CUDA context, loads one CUDA module, uploads
input tensors once, keeps the output tensor device-resident across repeated
launches, and performs no per-run host-to-device uploads.

This is still fixture-level evidence. It is not BitNet packed I2_S/QK256 proof,
dense GGUF inference, general chat quality, a speedup claim, a persistent dense
model session, server readiness, or full CUDA residency.

## Receipt

Primary receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-persistent.json
```

## Fixture

| Field | Value |
| --- | --- |
| Artifact kind | `dense_regular_llm_cuda` |
| Claim | `dense_regular_llm_cuda_persistent_fixture_residency_tested` |
| Fixture ID | `dense_f16_gemm_m2_n3_k4` |
| Kernel ID | `dense_f16_gemm_cuda` |
| Reference backend | `amd-9950x3d-cpu-avx512` |
| Target backend | `nvidia-rtx-5070-ti-cuda` |
| Runtime API | `cuda` |
| Repeated runs | `3` |
| Fallback | `false` |
| Speedup claim | `false` |

## Persistent Evidence

The receipt records:

```text
context creations: 1
module loads: 1
input uploads: 2
output allocations: 1
persistent handles: 3
kernel launches: 3
per-run host_to_device_bytes: 0
total host_to_device_bytes: 40
total device_to_host_bytes: 72
```

The repeated fixture launches pass parity against the CPU AVX-512 reference with
`max_abs_error=0.0`.

## Claim Boundary

May claim:

- dense regular-LLM CUDA persistent fixture-session residency is recorded for
  the deterministic FP16 GEMM fixture
- the fixture performs repeated fallback-free RTX 5070 Ti CUDA launches with one
  context/module and upload-once input device buffers
- dense regular-LLM CUDA receipts remain separate from BitNet packed I2_S/QK256
  proof

Must not claim:

- dense regular-LLM CUDA proves BitNet packed I2_S inference
- dense regular-LLM CUDA proves QK256 inference
- dense regular-LLM CUDA speedup exists
- general dense GGUF CUDA inference is complete
- persistent dense model/session inference exists
- full dense regular-LLM CUDA residency exists

## Validation

```text
cargo fmt -p bitnet-kernels -p bitnet-receipts-core -p bitnet-receipts -- --check
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda dense_f16 -- --nocapture
BITNET_RUN_RTX5070TI_DENSE_CUDA_GEMM_SESSION=1 BITNET_RTX5070TI_DENSE_CUDA_GEMM_SESSION_RECEIPT=ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-persistent.json BITNET_RTX5070TI_DENSE_CUDA_GEMM_SESSION_ARTIFACT_PATH=ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-persistent.json BITNET_RTX5070TI_DENSE_CUDA_GEMM_SESSION_RUNS=3 cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda live_rtx5070ti_dense_f16_cuda_gemm_persistent_session_when_enabled -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation -- --nocapture
cargo test --locked -p bitnet-receipts-core --lib -- --nocapture
```
