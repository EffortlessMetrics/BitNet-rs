# CUDA-DENSE-020 Attention-Score Fixture

`CUDA-DENSE-020` extracts a dense GGUF attention-score CPU-reference fixture
for the verified Qwen2.5 0.5B Q8_0 artifact after `CUDA-DENSE-018` proved
metadata-derived Q/K RoPE CUDA parity and `CUDA-DENSE-019` made RoPE
planner-routable.

This is a fixture and gap-audit slice. It does not add a CUDA attention-score
kernel and does not claim dense GGUF inference.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-score-fixture-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_attention_score_fixture_extraction` |
| `claim` | `dense_gguf_attention_score_fixture_extracted` |
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
| `attention_score_fixture.cpu_reference_computed` | `true` |
| `attention_score_fixture.cuda_kernel_status` | `missing_cuda_kernel` |
| `attention_score_gap_audit.input_dependencies` | `rope_q`, `rope_k`, `causal_mask` |
| `attention_score_gap_audit.next_required_proof` | `cuda_attention_score_kernel_parity` |
| `execution_plan.selected_route` | `unsupported` |
| `execution_plan.selected_backend` | `unsupported_strict_cuda` |
| `execution_plan.runtime_api` | `none` |
| `execution_plan.cuda_dense_regular_llm_ops` | `0` |
| `execution_plan.unsupported_ops` | `1` |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_attention_score_fixture_extraction_claimed` | `true` |
| `claim_boundary.dense_regular_llm_cuda_claimed` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- A metadata-derived Qwen2.5 0.5B Q8_0 dense GGUF attention-score
  CPU-reference fixture exists after verified RoPE Q/K fixture generation.
- The receipt records Q/K RoPE dependencies, GQA head mapping, causal mask
  counts, score count, score SHA256, and the missing CUDA attention-score
  kernel gap.
- The next governed dense CUDA proof is `cuda_attention_score_kernel_parity`.

## Must Not Claim

- dense regular LLM CUDA attention-score parity exists;
- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, or dense CUDA kernel math
  changed.

## Validation

```powershell
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli attention_score -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features attention_score -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-attention-score-fixture --model <verified-qwen2.5-q8-gguf> --layer-index 0 --seq-len 4 --position-offset 1 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-score-fixture-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-score-fixture-qwen25-q8.json
```

## Next Step

The next scoped dense CUDA proof should implement and validate CUDA
attention-score kernel parity for this fixture. It should remain below dense
GGUF one-token, decode, chat, server, speedup, and full-residency claims.
