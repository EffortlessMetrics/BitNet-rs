# CUDA-DENSE-021 Attention-Score CUDA Parity

`CUDA-DENSE-021` adds the first strict CUDA parity proof for dense GGUF
attention-score computation. It uses the `CUDA-DENSE-020` metadata-derived
Qwen2.5 0.5B Q8_0 RoPE Q/K fixture, runs scaled causal QK scores through an
RTX 5070 Ti CUDA F32 kernel, and compares the output against the CPU reference.

This is still fixture-level dense regular-LLM CUDA evidence. It is not dense
GGUF inference, Qwen one-token generation, short decode, chat, speedup, or full
CUDA residency.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-score-cuda-parity-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_attention_score_cuda_parity` |
| `claim` | `dense_gguf_attention_score_cuda_parity_tested` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `execution_plan.cuda_dense_regular_llm_ops` | `1` |
| `selected_backend` | `nvidia-rtx-5070-ti-cuda` |
| `runtime_api` | `cuda` |
| `fallback_used` | `false` |
| `kernel_stats[0].kernel_id` | `dense_attention_scores_f32_cuda` |
| `kernel_stats[0].invocations` | `1` |
| `kernel_stats[0].fallback_invocations` | `0` |
| `kernel_stats[0].host_to_device_bytes` | `16384` |
| `kernel_stats[0].device_to_host_bytes` | `896` |
| `attention_score_fixture.fixture_id` | `dense_gguf_attention_scores_qwen_layer0_q14_kv2_d64_s4` |
| `attention_score_fixture.q_heads` | `14` |
| `attention_score_fixture.kv_heads` | `2` |
| `attention_score_fixture.heads_per_kv_group` | `7` |
| `attention_score_fixture.head_dim` | `64` |
| `attention_score_fixture.seq_len` | `4` |
| `attention_score_fixture.attention_scale` | `0.125` |
| `attention_score_fixture.score_count` | `224` |
| `attention_score_fixture.finite_scores` | `140` |
| `attention_score_fixture.causal_masked_scores` | `84` |
| `parity.compared_scores` | `224` |
| `parity.max_abs_error` | `7.450580596923828e-9` |
| `parity.mean_abs_error` | `6.188461965095371e-10` |
| `parity.tolerance` | `0.00025` |
| `parity.passed` | `true` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_attention_score_cuda_parity_claimed` | `true` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- A metadata-derived Qwen2.5 0.5B Q8_0 dense GGUF attention-score fixture runs
  on the RTX 5070 Ti through `dense_attention_scores_f32_cuda`.
- The CUDA output matches the CPU reference within the governed
  attention-score tolerance.
- The receipt records Q/K RoPE dependencies, GQA head mapping, causal mask
  counts, transfer bytes, selected CUDA backend identity, and no fallback.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- persistent-session or full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, prompt-template, transformer, server, or QK256 behavior
  changed;
- the dense one-layer planner now routes `attention_scores`.

## Validation

```powershell
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda attention_score -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli attention_score -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features attention_score -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-attention-score-cuda-parity --model <verified-qwen2.5-q8-gguf> --layer-index 0 --seq-len 4 --position-offset 1 --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-score-cuda-parity-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-score-cuda-parity-qwen25-q8.json
```

## Next Step

The next dense CUDA slice should either promote attention-score planner routing
in the one-layer receipt or start the next governed gap: attention softmax and
V-mix parity. It should remain below dense GGUF token/decode/chat and speedup
claims.
