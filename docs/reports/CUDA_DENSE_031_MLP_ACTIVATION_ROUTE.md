# CUDA-DENSE-031 MLP Activation Planner Route

`CUDA-DENSE-031` promotes the verified dense GGUF `mlp_activation`
fixture from `CUDA-DENSE-030` into the one-layer planner route. The refreshed
one-layer execution-plan receipt records all 14 governed layer-0 ops as
`dense_regular_llm_cuda`, with zero unsupported strict CUDA ops.

This remains planner-route evidence. It does not claim dense GGUF inference,
Qwen one-token/decode/chat, speedup, persistent-session residency, full CUDA
residency, server readiness, or BitNet packed I2_S/QK256 proof.

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
| `execution_plan.cuda_dense_regular_llm_ops` | `14` |
| `execution_plan.unsupported_ops` | `0` |
| `execution_plan.strict_cuda_ready` | `true` |
| `one_layer_plan.total_ops` | `14` |
| `one_layer_plan.mlp_activation_cuda_ops_total` | `1` |
| `gap_audit.mlp_activation_cuda_parity_available` | `true` |
| `gap_audit.next_candidate_gap` | `none` |
| `gap_audit.next_required_proof` | `one_layer_cpu_reference_harness` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- The dense GGUF one-layer planner now routes verified `mlp_activation` to
  `dense_regular_llm_cuda`.
- The one-layer execution-plan receipt records 14 dense CUDA-routable ops,
  zero unsupported strict CUDA ops, and `strict_cuda_ready=true`.
- The next governed dense CUDA proof is a full one-layer CPU reference harness.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, CUDA kernel math, or
  BitNet CUDA behavior changed.

## Validation

```powershell
cargo test --locked -p bitnet-kernels --lib --no-default-features model_aware_dense_fp16_routes_mlp_activation_to_dense_cuda -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_one_layer -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features one_layer -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-one-layer-plan --model <verified-qwen2.5-q8-gguf> --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-plan-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-plan-qwen25-q8.json
```

## Next Step

The next dense CUDA slice should build the full one-layer CPU reference harness
that composes the already verified routed ops, without claiming token
generation or dense GGUF inference.
