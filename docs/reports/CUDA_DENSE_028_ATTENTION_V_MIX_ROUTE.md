# CUDA-DENSE-028 Attention V-Mix Planner Route

`CUDA-DENSE-028` promotes the dense GGUF one-layer planner route for
`attention_v_mix` after `CUDA-DENSE-027` proved strict RTX 5070 Ti CUDA V-mix
fixture parity.

This is planner and receipt evidence only. It does not claim dense GGUF
inference, Qwen token/decode/chat readiness, speedup, full CUDA residency, or
BitNet packed I2_S/QK256 proof.

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
| `execution_plan.cuda_dense_regular_llm_ops` | `13` |
| `execution_plan.unsupported_ops` | `1` |
| `one_layer_plan.attention_v_mix_cuda_ops_total` | `1` |
| `gap_audit.attention_v_mix_cuda_parity_available` | `true` |
| `gap_audit.next_candidate_gap` | `mlp_activation` |
| `gap_audit.unsupported_ops_total` | `1` |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.qwen_one_token_cuda_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- The dense GGUF one-layer planner routes verified `attention_v_mix` to
  `dense_regular_llm_cuda`.
- The refreshed one-layer receipt records 13 CUDA-routable dense ops for the
  verified Qwen2.5 0.5B Q8_0 artifact.
- The only remaining explicit unsupported strict CUDA one-layer gap is
  `mlp_activation`.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- persistent-session or full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, BitNet CUDA behavior, or
  CUDA kernel math changed.

## Validation

```powershell
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda attention_v_mix -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_one_layer -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features one_layer -- --nocapture
$env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;' + $env:PATH
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-one-layer-plan --model <verified-qwen2.5-q8-gguf> --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-plan-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-plan-qwen25-q8.json
```

## Next Step

The next dense CUDA slice should create a bounded CPU-reference fixture for
`mlp_activation` without claiming Qwen one-token/decode/chat or dense GGUF
inference. That follow-up is recorded as
[`CUDA-DENSE-029`](CUDA_DENSE_029_MLP_ACTIVATION_FIXTURE.md).
