# CUDA-DENSE-011 Live Dense GGUF Linear Role Sweep

## Summary

`CUDA-DENSE-011` records a live RTX 5070 Ti dense GGUF linear-role sweep
using the existing `dense-gguf-linear-parity` harness.

The proof uses the verified Qwen2.5 0.5B Instruct Q8_0 GGUF artifact and
extracts the first-layer attention and MLP linear roles, plus the output
projection, from the real model file:

```text
Qwen2.5 Q8_0 GGUF
  -> descriptor-driven dense linear fixture extraction
  -> FP16 GEMM bridge layout
  -> RTX 5070 Ti dense_f16_gemm_cuda
  -> per-role dense_gguf_linear_cuda_parity receipts
```

This is still a linear-fixture proof. It does not claim Qwen one-token decode,
short decode, chat, dense full-model inference, BitNet packed I2_S/QK256 proof,
speedup, or full CUDA residency.

## Receipts

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-linear-role-sweep-qwen25-q8/
```

| Role | Tensor | Matrix | Parity | Max abs error | H2D bytes | D2H bytes |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `attention_q` | `blk.0.attn_q.weight` | `896 x 896` | pass | `0.0` | `1607424` | `3584` |
| `attention_k` | `blk.0.attn_k.weight` | `128 x 896` | pass | `0.0` | `231168` | `512` |
| `attention_v` | `blk.0.attn_v.weight` | `128 x 896` | pass | `0.0` | `231168` | `512` |
| `attention_output` | `blk.0.attn_output.weight` | `896 x 896` | pass | `0.0` | `1607424` | `3584` |
| `mlp_gate` | `blk.0.ffn_gate.weight` | `4864 x 896` | pass | `0.0` | `8718080` | `19456` |
| `mlp_up` | `blk.0.ffn_up.weight` | `4864 x 896` | pass | `0.0` | `8718080` | `19456` |
| `mlp_down` | `blk.0.ffn_down.weight` | `896 x 4864` | pass | `0.0` | `8726016` | `3584` |
| `output` | `output.weight` | `151936 x 896` | pass | `0.0` | `272271104` | `607744` |

Shared receipt invariants:

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
| `execution_plan.strict_cuda_ready` | `true` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.full_cuda_residency_claimed` | `false` |

## Commands

```powershell
$model = Join-Path $env:LOCALAPPDATA `
  'bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf'

$outDir = `
  'ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-linear-role-sweep-qwen25-q8'

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$roles = @(
  'attention_q',
  'attention_k',
  'attention_v',
  'attention_output',
  'mlp_gate',
  'mlp_up',
  'mlp_down',
  'output'
)

foreach ($role in $roles) {
  cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
    dense-gguf-linear-parity `
    --model $model `
    --role $role `
    --device-index 0 `
    --json-out (Join-Path $outDir "$role.json")
}
```

## Claim Boundary

May claim:

- the verified Qwen2.5 0.5B Q8_0 dense GGUF artifact provides real attention,
  MLP, and output linear tensors that can be extracted and routed through the
  existing dense FP16 CUDA bridge;
- each recorded role receipt passes CPU/CUDA single-linear parity on the RTX
  5070 Ti with `fallback_used=false`;
- dense linear CUDA evidence remains separated from BitNet packed I2_S/QK256
  proof.

Must not claim:

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA proves BitNet packed inference;
- speedup;
- full CUDA residency;
- tokenizer, prompt-template, transformer, QK256, or server behavior changed.
