# CUDA-DENSE-003 Dense Tensor Residency

Date: 2026-05-08

## Scope

`CUDA-DENSE-003` extends the dense regular-LLM CUDA lane with fixture-level
tensor residency evidence. It records that the deterministic dense FP16 GEMM
fixture places its input and output tensors in CUDA device buffers for the
kernel launch, accounts for host/device transfer bytes, and preserves the dense
regular-LLM receipt boundary.

This is not BitNet packed I2_S or QK256 proof. It is not dense GGUF inference,
general chat quality, a speedup claim, a persistent dense session, or full CUDA
residency.

## Receipt

Primary receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-residency.json
```

## Fixture

| Field | Value |
| --- | --- |
| Artifact kind | `dense_regular_llm_cuda` |
| Claim | `dense_regular_llm_cuda_tensor_residency_tested` |
| Fixture ID | `dense_f16_gemm_m2_n3_k4` |
| Kernel ID | `dense_f16_gemm_cuda` |
| Reference backend | `amd-9950x3d-cpu-avx512` |
| Target backend | `nvidia-rtx-5070-ti-cuda` |
| Runtime API | `cuda` |
| Fallback | `false` |
| Speedup claim | `false` |

## Residency Evidence

The receipt records:

```text
input tensors uploaded once: true
output tensor CUDA-resident during kernel: true
device buffers: 3
persistent handles claimed: false
host_to_device_bytes: 40
device_to_host_bytes: 24
```

Tensor residency is scoped to a single deterministic fixture launch:

| Tensor | Dtype | Shape | Residency | Transfer |
| --- | --- | --- | --- | --- |
| `a` | `f16` | `2 x 4` | `cuda_device_buffer` | uploaded once, 16 bytes |
| `b` | `f16` | `4 x 3` | `cuda_device_buffer` | uploaded once, 24 bytes |
| `c` | `f32` | `2 x 3` | `cuda_device_buffer` | downloaded for parity, 24 bytes |

The transfer accounting matches `kernel_stats`.

## Claim Boundary

May claim:

- dense regular-LLM CUDA tensor residency is recorded for the deterministic
  FP16 GEMM fixture
- the fixture records fallback-free RTX 5070 Ti CUDA execution
- dense regular-LLM CUDA receipts remain separate from BitNet packed I2_S/QK256
  proof

Must not claim:

- dense regular-LLM CUDA proves BitNet packed I2_S inference
- dense regular-LLM CUDA proves QK256 inference
- dense regular-LLM CUDA speedup exists
- general dense GGUF CUDA inference is complete
- persistent dense model/session residency exists
- full dense regular-LLM CUDA residency exists

## Validation

```text
cargo fmt -p bitnet-kernels -p bitnet-receipts-core -p bitnet-receipts -- --check
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda dense_f16 -- --nocapture
BITNET_RUN_RTX5070TI_DENSE_CUDA_GEMM=1 BITNET_RTX5070TI_DENSE_CUDA_GEMM_RECEIPT=ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-residency.json BITNET_RTX5070TI_DENSE_CUDA_GEMM_ARTIFACT_PATH=ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-residency.json cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda live_rtx5070ti_dense_f16_cuda_gemm_matches_cpu_reference_when_enabled -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation -- --nocapture
cargo test --locked -p bitnet-receipts-core --lib -- --nocapture
```
