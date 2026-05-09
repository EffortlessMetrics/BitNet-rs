# CUDA-DENSE-010 Live Dense GGUF Linear Parity

## Summary

`CUDA-DENSE-010` records a live RTX 5070 Ti dense GGUF single-linear CUDA parity
receipt using the existing `dense-gguf-linear-parity` harness.

The proof uses the verified Qwen2.5 0.5B Instruct Q8_0 GGUF artifact and
extracts the `attention_q` tensor from the real model file:

```text
Qwen2.5 Q8_0 GGUF
  -> descriptor-driven dense linear fixture extraction
  -> FP16 GEMM bridge layout
  -> RTX 5070 Ti dense_f16_gemm_cuda
  -> dense_gguf_linear_cuda_parity receipt
```

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-linear-cuda-parity-qwen25-q8.json
```

Key receipt fields:

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_linear_cuda_parity` |
| `claim` | `dense_gguf_linear_cuda_parity_tested` |
| `model.sha256` | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| `selected_backend` | `nvidia-rtx-5070-ti-cuda` |
| `runtime_api` | `cuda` |
| `fallback_used` | `false` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `execution_plan.cuda_dense_regular_llm_ops` | `1` |
| `execution_plan.cuda_bitnet_qk256_ops` | `0` |
| `parity.passed` | `true` |
| `parity.max_abs_error` | `0.0` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.full_cuda_residency_claimed` | `false` |

## Commands

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  model fetch qwen2.5-0.5b-instruct-q8_0

$model = Join-Path $env:LOCALAPPDATA `
  'bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf'

cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
  dense-gguf-linear-parity `
  --model $model `
  --role attention_q `
  --device-index 0 `
  --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-linear-cuda-parity-qwen25-q8.json
```

## Claim Boundary

May claim:

- a verified Qwen2.5 Q8_0 dense GGUF tensor can be extracted from the real
  artifact and routed through the existing dense FP16 CUDA bridge;
- the live receipt records fallback-free RTX 5070 Ti dense single-linear CUDA
  parity with dense `execution_plan` routing;
- dense GGUF linear CUDA parity remains separate from BitNet packed I2_S/QK256
  proof.

Must not claim:

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA proves BitNet packed inference;
- speedup;
- full CUDA residency;
- tokenizer, prompt-template, transformer, QK256, or server behavior changed.
