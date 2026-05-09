# CUDA-DENSE-027 Attention V-Mix CUDA Parity

`CUDA-DENSE-027` runs the `CUDA-DENSE-026` dense GGUF attention V-mix fixture
through a strict RTX 5070 Ti CUDA F32 V-mix kernel and compares the context
vectors against the CPU reference.

This is fixture-level CUDA parity only. It does not promote the one-layer
planner route, does not claim dense GGUF inference, and does not claim Qwen
token/decode/chat readiness.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-v-mix-cuda-parity-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_attention_v_mix_cuda_parity` |
| `claim` | `dense_gguf_attention_v_mix_cuda_parity_tested` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `execution_plan.cuda_dense_regular_llm_ops` | `1` |
| `attention_v_mix_fixture.q_heads` | `14` |
| `attention_v_mix_fixture.kv_heads` | `2` |
| `attention_v_mix_fixture.head_dim` | `64` |
| `attention_v_mix_fixture.seq_len` | `4` |
| `attention_v_mix_fixture.context_count` | `3584` |
| `attention_v_mix_fixture.cuda_kernel_status` | `parity_passed` |
| `kernel_stats[0].kernel_id` | `dense_attention_v_mix_f32_cuda` |
| `kernel_stats[0].invocations` | `1` |
| `kernel_stats[0].fallback_invocations` | `0` |
| `tensor_residency.inputs` | probabilities + values uploaded once |
| `tensor_residency.outputs` | context downloaded for parity check only |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.qwen_one_token_cuda_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- The dense GGUF attention V-mix fixture runs through the strict RTX 5070 Ti
  CUDA F32 V-mix kernel.
- The CUDA output matches the CPU-reference context vectors within the governed
  fixture tolerance.
- The receipt records kernel invocation, fallback, transfer-byte, and tensor
  residency fields for this single fixture.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- persistent-session or full dense CUDA residency is proven;
- the one-layer planner route has been promoted;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, or BitNet CUDA behavior
  changed.

## Validation

```powershell
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda attention_v_mix -- --nocapture
cargo test --locked -p bitnet-cli --bin bitnet --no-default-features --features cpu,cuda,full-cli attention_v_mix -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features attention_v_mix -- --nocapture
$env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;' + $env:PATH
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-attention-v-mix-cuda-parity --model <verified-qwen2.5-q8-gguf> --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-v-mix-cuda-parity-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-v-mix-cuda-parity-qwen25-q8.json
```

## Next Step

`CUDA-DENSE-028` is the planner-route follow-up for verified `attention_v_mix`.
After that route promotion, `mlp_activation` remains the next strict CUDA
one-layer gap. That follow-up should remain below dense GGUF token/decode/chat,
speedup, server, and full-residency claims.
