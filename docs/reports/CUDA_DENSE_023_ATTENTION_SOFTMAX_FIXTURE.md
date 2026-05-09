# CUDA-DENSE-023 Attention-Softmax Fixture

`CUDA-DENSE-023` extracts a dense GGUF attention-softmax CPU-reference fixture
for the verified Qwen2.5 0.5B Q8_0 artifact after `CUDA-DENSE-021` proved
attention-score CUDA parity and `CUDA-DENSE-022` made attention scores
planner-routable.

This is a fixture and gap-audit slice. It does not add a CUDA
attention-softmax kernel and does not claim dense GGUF inference.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-softmax-fixture-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_attention_softmax_fixture_extraction` |
| `claim` | `dense_gguf_attention_softmax_fixture_extracted` |
| `attention_softmax_fixture.fixture_id` | `dense_gguf_attention_softmax_qwen_layer0_q14_kv2_s4` |
| `attention_softmax_fixture.source_attention_score_fixture_id` | `dense_gguf_attention_scores_qwen_layer0_q14_kv2_d64_s4` |
| `attention_softmax_fixture.source_attention_score_artifact_kind` | `dense_gguf_attention_score_cuda_parity` |
| `attention_softmax_fixture.q_heads` | `14` |
| `attention_softmax_fixture.kv_heads` | `2` |
| `attention_softmax_fixture.seq_len` | `4` |
| `attention_softmax_fixture.row_count` | `56` |
| `attention_softmax_fixture.probability_count` | `224` |
| `attention_softmax_fixture.causal_zero_probabilities` | `84` |
| `attention_softmax_fixture.max_row_sum_abs_error` | `5.960464477539063e-8` |
| `attention_softmax_fixture.cpu_reference_computed` | `true` |
| `attention_softmax_fixture.cuda_kernel_status` | `missing_cuda_kernel` |
| `attention_softmax_gap_audit.input_dependencies` | `attention_scores` |
| `attention_softmax_gap_audit.next_required_proof` | `cuda_attention_softmax_kernel_parity` |
| `execution_plan.selected_route` | `unsupported` |
| `execution_plan.selected_backend` | `unsupported_strict_cuda` |
| `execution_plan.runtime_api` | `none` |
| `execution_plan.cuda_dense_regular_llm_ops` | `0` |
| `execution_plan.unsupported_ops` | `1` |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_attention_softmax_fixture_extraction_claimed` | `true` |
| `claim_boundary.dense_regular_llm_cuda_claimed` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- A metadata-derived Qwen2.5 0.5B Q8_0 dense GGUF attention-softmax
  CPU-reference fixture exists after attention-score CUDA parity.
- The receipt records the attention-score dependency, probability count,
  causal zero-probability count, row-sum error, probability SHA256, and the
  missing CUDA attention-softmax kernel gap.
- The next governed dense CUDA proof is `cuda_attention_softmax_kernel_parity`.

## Must Not Claim

- dense regular LLM CUDA attention-softmax parity exists;
- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, planner routing, or
  dense CUDA kernel math changed.

## Validation

```powershell
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli attention_softmax -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features attention_softmax -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-attention-softmax-fixture --model <verified-qwen2.5-q8-gguf> --layer-index 0 --seq-len 4 --position-offset 1 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-softmax-fixture-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-softmax-fixture-qwen25-q8.json
```

## Next Step

The next scoped dense CUDA proof should implement and validate CUDA
attention-softmax kernel parity for this fixture. It should remain below dense
GGUF one-token, decode, chat, server, speedup, and full-residency claims.
