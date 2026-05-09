# CUDA-DENSE-013 Dense GGUF One-Layer Planner Gap

## Summary

`CUDA-DENSE-013` adds a receipt-backed dense GGUF one-layer execution-plan
diagnostic for the RTX 5070 Ti dense regular-LLM CUDA lane.

The proof uses the verified Qwen2.5 0.5B Instruct Q8_0 GGUF artifact and records
the first dense transformer layer as a planner contract:

```text
Qwen2.5 Q8_0 GGUF descriptors
  -> model-aware dense regular-LLM CUDA planner
  -> dense linears routable to dense_regular_llm_cuda
  -> non-linear layer ops rejected under strict CUDA
  -> dense_gguf_one_layer_execution_plan receipt
```

This is a fail-closed planner receipt. It does not execute a full dense layer,
does not run Qwen one-token decode, and does not claim dense GGUF inference.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-one-layer-plan-qwen25-q8.json
```

Key fields:

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_one_layer_execution_plan` |
| `claim` | `dense_gguf_one_layer_execution_plan_gap_recorded` |
| `model.sha256` | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| `selected_backend` | `nvidia-rtx-5070-ti-cuda` |
| `runtime_api` | `cuda` |
| `fallback_used` | `false` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `execution_plan.cuda_dense_regular_llm_ops` | `7` |
| `execution_plan.cuda_bitnet_qk256_ops` | `0` |
| `execution_plan.unsupported_ops` | `7` |
| `execution_plan.strict_cuda_ready` | `false` |
| `one_layer_plan.total_ops` | `14` |
| `one_layer_plan.linear_cuda_ops_total` | `7` |
| `one_layer_plan.unsupported_strict_cuda_ops_total` | `7` |
| `claim_boundary.dense_gguf_one_layer_execution_plan_claimed` | `true` |
| `claim_boundary.dense_gguf_one_layer_inference_claimed` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.full_cuda_residency_claimed` | `false` |

## Layer-0 Planner Coverage

| Role | Route | Status |
| --- | --- | --- |
| `attention_norm` | `unsupported` | `unsupported_strict_cuda` |
| `attention_q` | `dense_regular_llm_cuda` | `cuda_routable` |
| `attention_k` | `dense_regular_llm_cuda` | `cuda_routable` |
| `attention_v` | `dense_regular_llm_cuda` | `cuda_routable` |
| `rope` | `unsupported` | `unsupported_strict_cuda` |
| `attention_scores` | `unsupported` | `unsupported_strict_cuda` |
| `attention_softmax` | `unsupported` | `unsupported_strict_cuda` |
| `attention_v_mix` | `unsupported` | `unsupported_strict_cuda` |
| `attention_output` | `dense_regular_llm_cuda` | `cuda_routable` |
| `ffn_norm` | `unsupported` | `unsupported_strict_cuda` |
| `mlp_gate` | `dense_regular_llm_cuda` | `cuda_routable` |
| `mlp_up` | `dense_regular_llm_cuda` | `cuda_routable` |
| `mlp_activation` | `unsupported` | `unsupported_strict_cuda` |
| `mlp_down` | `dense_regular_llm_cuda` | `cuda_routable` |

## Commands

```powershell
$model = Join-Path $env:LOCALAPPDATA `
  'bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf'

cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
  dense-gguf-one-layer-plan `
  --model $model `
  --device-index 0 `
  --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-plan-qwen25-q8.json

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-plan-qwen25-q8.json
```

## Claim Boundary

May claim:

- the dense planner can classify a Qwen2.5 Q8_0 first-layer contract into
  dense CUDA-routable linears and unsupported strict CUDA non-linear ops;
- the receipt records `fallback_used=false`, `cpu_fallback_ops=0`, and
  `strict_cuda_ready=false` for the one-layer gap;
- this dense planner receipt remains separated from BitNet packed I2_S/QK256
  proof.

Must not claim:

- dense GGUF inference works;
- a full Qwen layer, one-token decode, short decode, or chat works on CUDA;
- dense CUDA proves BitNet packed inference;
- speedup;
- persistent-session or full CUDA residency;
- tokenizer, prompt-template, transformer, loader, QK256, or server behavior
  changed.
