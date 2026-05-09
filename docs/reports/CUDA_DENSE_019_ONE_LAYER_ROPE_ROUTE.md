# CUDA-DENSE-019 One-Layer RoPE Route

`CUDA-DENSE-019` updates the dense GGUF one-layer planner state after
`CUDA-DENSE-018` proved strict RTX 5070 Ti CUDA RoPE parity for the verified
Qwen2.5 0.5B Q8_0 metadata-derived fixture.

The one-layer receipt now treats the layer RoPE op as
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
| `execution_plan.cuda_dense_regular_llm_ops` | `10` |
| `execution_plan.unsupported_ops` | `4` |
| `one_layer_plan.total_ops` | `14` |
| `one_layer_plan.linear_cuda_ops_total` | `7` |
| `one_layer_plan.norm_cuda_ops_total` | `2` |
| `one_layer_plan.rope_cuda_ops_total` | `1` |
| `gap_audit.rmsnorm_cuda_parity_available` | `true` |
| `gap_audit.rope_cuda_parity_available` | `true` |
| `gap_audit.next_candidate_gap` | `attention_scores` |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## Remaining One-Layer Gaps

```text
attention_scores
attention_softmax
attention_v_mix
mlp_activation
```

## May Claim

- Dense GGUF layer-0 linears, RMSNorm fixtures, and RoPE are planner-routable
  to the dense regular-LLM CUDA route.
- The one-layer gap receipt records 7 CUDA-routable dense linear ops, 2
  CUDA-routable RMSNorm ops, 1 CUDA-routable RoPE op, and 4 remaining
  unsupported strict CUDA gaps.
- Attention score computation is now the next governed dense one-layer gap
  candidate.

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

The next scoped dense CUDA proof should target the attention score, mask,
softmax, and V-mix gaps. It should remain below dense GGUF one-token, decode,
chat, server, speedup, and full-residency claims.
