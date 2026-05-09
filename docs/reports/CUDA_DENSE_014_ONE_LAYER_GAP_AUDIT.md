# CUDA-DENSE-014 Dense GGUF One-Layer Gap Audit

## Summary

`CUDA-DENSE-014` extends the dense GGUF one-layer planner receipt with an
explicit gap audit for the verified Qwen2.5 0.5B Instruct Q8_0 artifact on the
RTX 5070 Ti lane.

The receipt still does not execute dense GGUF inference. It records which
layer-0 operations are CUDA-routable today, which strict CUDA operations are
blocked, and which dependencies each blocked operation needs before a full dense
layer can be claimed.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-one-layer-plan-qwen25-q8.json
```

Key fields:

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_one_layer_execution_plan` |
| `claim` | `dense_gguf_one_layer_execution_plan_gap_recorded` |
| `gap_audit.cuda_routable_linear_ops_total` | `7` |
| `gap_audit.unsupported_ops_total` | `7` |
| `gap_audit.cpu_fallback_ops_total` | `0` |
| `gap_audit.strict_cuda_ready` | `false` |
| `gap_audit.strict_cuda_rejects_cpu_fallback` | `true` |
| `gap_audit.unsupported_ops_have_dependency_notes` | `true` |
| `claim_boundary.dense_gguf_one_layer_inference_claimed` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.full_cuda_residency_claimed` | `false` |

## Gap Audit

| Role | Op type | Kernel status | Dependencies |
| --- | --- | --- | --- |
| `attention_norm` | `rmsnorm` | `missing_cuda_kernel` | `hidden_state` |
| `ffn_norm` | `rmsnorm` | `missing_cuda_kernel` | `attention_residual_state` |
| `rope` | `rope` | `missing_cuda_kernel` | `attention_q`, `attention_k`, `position_ids` |
| `attention_scores` | `attention` | `missing_cuda_kernel` | `rope_q`, `rope_k`, `causal_mask` |
| `attention_softmax` | `softmax` | `missing_cuda_kernel` | `attention_scores` |
| `attention_v_mix` | `attention` | `missing_cuda_kernel` | `attention_softmax`, `attention_v` |
| `mlp_activation` | `activation` | `missing_cuda_kernel` | `mlp_gate`, `mlp_up` |

Candidate order recorded in the receipt:

```text
attention_norm
ffn_norm
rope
attention_scores
attention_softmax
attention_v_mix
mlp_activation
```

This order is a governed gap-audit order for future implementation work. It is
not a claim that any of these kernels already exists.

## Commands

```powershell
$model = Join-Path $env:LOCALAPPDATA `
  'bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf'

cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
  dense-gguf-one-layer-plan `
  --model $model `
  --device-index 0 `
  --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-plan-qwen25-q8.json

cargo test --locked -p bitnet-receipts --test cuda_receipt_validation `
  --no-default-features dense_gguf_one_layer -- --nocapture
```

## Claim Boundary

May claim:

- the dense GGUF one-layer planner receipt now includes dependency notes for
  unsupported strict CUDA non-linear ops;
- strict CUDA still rejects CPU fallback for those ops;
- the gap audit records `cpu_fallback_ops_total=0`, `strict_cuda_ready=false`,
  and no dense inference claim.

Must not claim:

- dense GGUF inference works;
- a full Qwen layer, one-token decode, short decode, or chat works on CUDA;
- dense CUDA proves BitNet packed inference;
- speedup;
- full CUDA residency;
- tokenizer, prompt-template, transformer, loader, QK256, or server behavior
  changed.
