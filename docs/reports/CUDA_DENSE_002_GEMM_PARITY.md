# CUDA-DENSE-002 Dense FP16 GEMM Parity

Date: 2026-05-08

## Scope

`CUDA-DENSE-002` adds the first dense regular-LLM CUDA GEMM smoke/parity fixture
after the `dense_regular_llm_cuda` receipt boundary. It is a small FP16 GEMM
fixture for the RTX 5070 Ti CUDA lane.

This is not BitNet packed I2_S or QK256 proof. It is not dense GGUF inference,
general chat quality, a speedup claim, or full CUDA residency.

## Fixture

| Field | Value |
| --- | --- |
| Receipt | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-parity.json` |
| Artifact kind | `dense_regular_llm_cuda` |
| Fixture ID | `dense_f16_gemm_m2_n3_k4` |
| Kernel ID | `dense_f16_gemm_cuda` |
| Reference backend | `amd-9950x3d-cpu-avx512` |
| Target backend | `nvidia-rtx-5070-ti-cuda` |
| Runtime API | `cuda` |
| Fallback | `false` |
| Speedup claim | `false` |

## Result

The live RTX 5070 Ti fixture passed CPU/CUDA parity:

| Metric | Value |
| --- | ---: |
| Max absolute error | `0.0` |
| Mean absolute error | `0.0` |
| Tolerance | `0.002` |
| Host-to-device bytes | `40` |
| Device-to-host bytes | `24` |
| Kernel launches | `1` |

The kernel uses embedded PTX so the smoke fixture can run on a driver-only CUDA
path without requiring NVRTC in `PATH`.

## Claim Boundary

May claim:

- a dense regular-LLM CUDA FP16 GEMM smoke/parity fixture exists
- the committed receipt records fallback-free RTX 5070 Ti CUDA execution
- dense regular-LLM CUDA receipts remain separate from BitNet packed I2_S/QK256 proof

Must not claim:

- dense regular-LLM CUDA proves BitNet packed I2_S inference
- dense regular-LLM CUDA proves QK256 inference
- dense regular-LLM CUDA speedup exists
- general dense GGUF CUDA inference is complete
- full dense regular-LLM CUDA residency exists

## Validation

```text
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda dense_f16 -- --nocapture
BITNET_RUN_RTX5070TI_DENSE_CUDA_GEMM=1 BITNET_RTX5070TI_DENSE_CUDA_GEMM_RECEIPT=ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-parity.json cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda live_rtx5070ti_dense_f16_cuda_gemm_matches_cpu_reference_when_enabled -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation -- --nocapture
```
