# CUDA-DENSE-005 Model-Family Planner Bridge

`CUDA-DENSE-005` bridges the dense regular-LLM CUDA fixture lane to the
model-aware execution-plan receipt contract. It does not add dense GGUF
inference or new CUDA math.

## Receipt Change

The committed dense FP16 fixture receipts now include an `execution_plan`
section with:

| Field | Value |
| --- | --- |
| `planner_version` | `cuda-planner-004` |
| `model_family` | `qwen` |
| `quantization` | `dense_fp16` |
| `selected_route` | `dense_regular_llm_cuda` |
| `dense_regular_llm_cuda` | `true` |
| `bitnet_packed_qk256_cuda` | `false` |
| `cuda_bitnet_qk256_ops` | `0` |
| `cpu_fallback_ops` | `0` |
| `unsupported_ops` | `0` |
| `strict_cuda_ready` | `true` |
| `speedup_claim` | `false` |
| `full_cuda_residency_claimed` | `false` |

## Evidence

Updated receipts:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-parity.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-residency.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-f16-gemm-persistent.json
```

The dense receipt validator now requires the dense `execution_plan` route and
rejects missing, BitNet QK256, CPU fallback, unsupported, speedup, or
full-residency planner claims.

## Claim Boundary

May claim:

- Dense regular-LLM CUDA fixture receipts carry a model-aware planner route.
- Qwen-family dense FP16 metadata is represented as `dense_regular_llm_cuda`.
- Dense planner receipts remain separate from BitNet packed I2_S/QK256 proof.

Must not claim:

- Dense GGUF inference works.
- Dense CUDA speedup exists.
- Dense CUDA proves BitNet packed I2_S or QK256 inference.
- Any CUDA kernel, tokenizer, loader, transformer, or server behavior changed.
- Full dense CUDA residency is proven.
