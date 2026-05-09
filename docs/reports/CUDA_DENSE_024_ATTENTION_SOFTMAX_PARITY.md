# CUDA-DENSE-024 Attention-Softmax CUDA Parity

`CUDA-DENSE-024` adds a strict RTX 5070 Ti CUDA attention-softmax fixture
kernel for the verified Qwen2.5 0.5B Q8_0 dense GGUF artifact. It consumes the
metadata-derived attention-score fixture from `CUDA-DENSE-021`/`CUDA-DENSE-023`
and compares CUDA probabilities against the CPU-reference softmax.

This is still fixture-level evidence. It does not promote dense GGUF inference,
Qwen one-token/decode/chat, server readiness, speedup, full CUDA residency, or
BitNet packed I2_S/QK256 proof.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-softmax-cuda-parity-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_attention_softmax_cuda_parity` |
| `claim` | `dense_gguf_attention_softmax_cuda_parity_tested` |
| `attention_softmax_fixture.fixture_id` | `dense_gguf_attention_softmax_qwen_layer0_q14_kv2_s4` |
| `attention_softmax_fixture.source_attention_score_fixture_id` | `dense_gguf_attention_scores_qwen_layer0_q14_kv2_d64_s4` |
| `attention_softmax_fixture.source_attention_score_artifact_kind` | `dense_gguf_attention_score_cuda_parity` |
| `attention_softmax_fixture.q_heads` | `14` |
| `attention_softmax_fixture.kv_heads` | `2` |
| `attention_softmax_fixture.seq_len` | `4` |
| `attention_softmax_fixture.row_count` | `56` |
| `attention_softmax_fixture.probability_count` | `224` |
| `attention_softmax_fixture.causal_zero_probabilities` | `84` |
| `attention_softmax_fixture.cuda_kernel_status` | `parity_passed` |
| `kernel_stats[0].kernel_id` | `dense_attention_softmax_f32_cuda` |
| `kernel_stats[0].invocations` | `1` |
| `kernel_stats[0].fallback_invocations` | `0` |
| `kernel_stats[0].host_to_device_bytes` | `896` |
| `kernel_stats[0].device_to_host_bytes` | `896` |
| `parity.max_abs_error` | `5.960464477539063e-8` |
| `parity.mean_abs_error` | `5.654458679060781e-9` |
| `parity.tolerance` | `0.00025` |
| `parity.first_divergence` | `null` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `execution_plan.cuda_dense_regular_llm_ops` | `1` |
| `execution_plan.unsupported_ops` | `0` |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_attention_softmax_cuda_parity_claimed` | `true` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- A metadata-derived Qwen2.5 0.5B Q8_0 attention-softmax fixture runs through a
  strict RTX 5070 Ti CUDA F32 softmax kernel.
- CUDA probabilities match the CPU-reference probabilities for the committed
  fixture within the governed tolerance.
- The receipt records CUDA backend identity, kernel identity, transfer byte
  accounting, tensor residency for this fixture, and no CPU fallback.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, prompt-template, transformer, server, QK256 math, or
  planner route promotion changed.

## Validation

```powershell
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda attention_softmax -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli attention_softmax -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features attention_softmax -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-attention-softmax-cuda-parity --model <verified-qwen2.5-q8-gguf> --layer-index 0 --seq-len 4 --position-offset 1 --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-softmax-cuda-parity-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-softmax-cuda-parity-qwen25-q8.json
```

The live CUDA receipt generation required the CUDA Toolkit `v12.9\bin` directory
on `PATH` so `nvrtc64_120_0.dll` could be loaded by the process.

## Next Step

The next scoped dense CUDA proof should move to the attention-value mix fixture
or keep improving transfer/kernel timing for this path. It should remain below
dense GGUF one-token, decode, chat, server, speedup, full-residency, and BitNet
packed proof claims.
