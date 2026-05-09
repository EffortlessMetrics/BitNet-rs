# CUDA-DENSE-022 One-Layer Attention-Score Route

`CUDA-DENSE-022` updates the dense GGUF one-layer planner state after
`CUDA-DENSE-021` proved strict RTX 5070 Ti CUDA attention-score parity for the
verified Qwen2.5 0.5B Q8_0 metadata-derived fixture.

The one-layer receipt now treats `attention_scores` as
`dense_regular_llm_cuda` routable. The receipt still fails closed for the
remaining unsupported strict CUDA layer gaps and does not claim dense GGUF
inference.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-one-layer-plan-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_one_layer_execution_plan` |
| `claim` | `dense_gguf_one_layer_execution_plan_gap_recorded` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `execution_plan.cuda_dense_regular_llm_ops` | `11` |
| `execution_plan.unsupported_ops` | `3` |
| `one_layer_plan.total_ops` | `14` |
| `one_layer_plan.linear_cuda_ops_total` | `7` |
| `one_layer_plan.norm_cuda_ops_total` | `2` |
| `one_layer_plan.rope_cuda_ops_total` | `1` |
| `one_layer_plan.attention_score_cuda_ops_total` | `1` |
| `gap_audit.attention_score_cuda_parity_available` | `true` |
| `gap_audit.next_candidate_gap` | `attention_softmax` |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## Remaining One-Layer Gaps

```text
attention_softmax
attention_v_mix
mlp_activation
```

## May Claim

- The dense GGUF one-layer planner now routes `attention_scores` to the dense
  regular-LLM CUDA route based on `CUDA-DENSE-021` parity evidence.
- The one-layer gap receipt records 11 CUDA-routable dense ops, including 7
  linears, 2 RMSNorm ops, 1 RoPE op, and 1 attention-score op.
- Attention softmax is now the next governed dense one-layer gap candidate.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, or dense CUDA kernel math
  changed.

## Validation

```powershell
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda model_aware_dense -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_one_layer -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features dense_gguf_one_layer -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-one-layer-plan --model <verified-qwen2.5-q8-gguf> --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-plan-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-plan-qwen25-q8.json
```

## Next Step

The next dense CUDA slice should target attention softmax and V-mix parity. It
should remain below dense GGUF token/decode/chat, speedup, server, and
full-residency claims.
