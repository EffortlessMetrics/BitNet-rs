# CUDA-DENSE-012 Aggregate Dense GGUF Linear Role Sweep

## Summary

`CUDA-DENSE-012` adds an aggregate `dense-gguf-linear-role-sweep` receipt
surface for the RTX 5070 Ti dense regular-LLM CUDA lane.

The proof uses the verified Qwen2.5 0.5B Instruct Q8_0 GGUF artifact and
routes eight descriptor-extracted dense linear roles through the existing dense
FP16 CUDA GEMM bridge in one command:

```text
Qwen2.5 Q8_0 GGUF
  -> descriptor-driven dense linear fixture extraction
  -> FP16 GEMM bridge layout
  -> RTX 5070 Ti dense_f16_gemm_cuda
  -> aggregate dense_gguf_linear_role_sweep_cuda_parity receipt
```

This is still a linear-fixture proof. It does not claim Qwen one-token decode,
short decode, chat, dense full-model inference, BitNet packed I2_S/QK256 proof,
speedup, persistent-session residency, or full CUDA residency.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-linear-role-sweep-aggregate-qwen25-q8.json
```

Shared invariants:

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_linear_role_sweep_cuda_parity` |
| `claim` | `dense_gguf_linear_role_sweep_cuda_parity_tested` |
| `model.sha256` | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| `selected_backend` | `nvidia-rtx-5070-ti-cuda` |
| `runtime_api` | `cuda` |
| `fallback_used` | `false` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `execution_plan.cuda_dense_regular_llm_ops` | `8` |
| `execution_plan.cuda_bitnet_qk256_ops` | `0` |
| `execution_plan.strict_cuda_ready` | `true` |
| `linear_role_sweep.roles_total` | `8` |
| `linear_role_sweep.roles_passed` | `8` |
| `linear_role_sweep.max_abs_error` | `0.0` |
| `linear_role_sweep.host_to_device_bytes` | `302110464` |
| `linear_role_sweep.device_to_host_bytes` | `658432` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.full_cuda_residency_claimed` | `false` |

## Covered Roles

| Role | Tensor | H2D bytes | D2H bytes |
| --- | --- | ---: | ---: |
| `attention_q` | `blk.0.attn_q.weight` | `1607424` | `3584` |
| `attention_k` | `blk.0.attn_k.weight` | `231168` | `512` |
| `attention_v` | `blk.0.attn_v.weight` | `231168` | `512` |
| `attention_output` | `blk.0.attn_output.weight` | `1607424` | `3584` |
| `mlp_gate` | `blk.0.ffn_gate.weight` | `8718080` | `19456` |
| `mlp_up` | `blk.0.ffn_up.weight` | `8718080` | `19456` |
| `mlp_down` | `blk.0.ffn_down.weight` | `8726016` | `3584` |
| `output` | `output.weight` | `272271104` | `607744` |

## Commands

```powershell
$model = Join-Path $env:LOCALAPPDATA `
  'bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf'

cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
  dense-gguf-linear-role-sweep `
  --model $model `
  --device-index 0 `
  --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-linear-role-sweep-aggregate-qwen25-q8.json

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-linear-role-sweep-aggregate-qwen25-q8.json
```

## Claim Boundary

May claim:

- multiple real Qwen2.5 0.5B Q8_0 dense GGUF linear roles can be extracted and
  routed through the existing dense FP16 CUDA bridge by one aggregate command;
- the aggregate receipt records `dense_regular_llm_cuda` routing,
  `fallback_used=false`, transfer accounting, and parity pass for every covered
  role;
- dense linear CUDA evidence remains separated from BitNet packed I2_S/QK256
  proof.

Must not claim:

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA proves BitNet packed inference;
- speedup;
- persistent-session or full CUDA residency;
- tokenizer, prompt-template, transformer, QK256, or server behavior changed.
