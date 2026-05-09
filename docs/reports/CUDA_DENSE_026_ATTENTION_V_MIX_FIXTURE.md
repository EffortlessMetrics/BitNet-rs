# CUDA-DENSE-026 Attention V-Mix Fixture

`CUDA-DENSE-026` extracts a CPU-reference dense GGUF attention V-mix fixture
after `CUDA-DENSE-025` made `attention_softmax` CUDA-routable in the one-layer
planner.

The fixture records deterministic `softmax(scores) x V` context vectors for the
verified Qwen2.5 0.5B Q8_0 artifact. It deliberately records the CUDA V-mix
kernel as missing and does not claim dense GGUF inference.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-attention-v-mix-fixture-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_attention_v_mix_fixture_extraction` |
| `claim` | `dense_gguf_attention_v_mix_fixture_extracted` |
| `execution_plan.selected_route` | `unsupported` |
| `execution_plan.unsupported_ops` | `1` |
| `attention_v_mix_fixture.q_heads` | `14` |
| `attention_v_mix_fixture.kv_heads` | `2` |
| `attention_v_mix_fixture.head_dim` | `64` |
| `attention_v_mix_fixture.seq_len` | `4` |
| `attention_v_mix_fixture.context_count` | `3584` |
| `attention_v_mix_fixture.cuda_kernel_status` | `missing_cuda_kernel` |
| `attention_v_mix_gap_audit.next_required_proof` | `cuda_attention_v_mix_kernel_parity` |
| `attention_v_mix_gap_audit.source_attention_softmax_cuda_parity_available` | `true` |
| `attention_v_mix_gap_audit.source_attention_v_cuda_parity_available` | `true` |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- A dense GGUF attention V-mix CPU-reference fixture exists for the verified
  Qwen2.5 0.5B Q8_0 artifact.
- The fixture records deterministic context-vector hashes and source dependency
  authority for verified attention-softmax and attention-V inputs.
- The next governed dense CUDA proof is `cuda_attention_v_mix_kernel_parity`.

## Must Not Claim

- attention V-mix CUDA parity exists;
- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, or dense CUDA kernel math
  changed.

## Validation

```powershell
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli attention_v_mix -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features attention_v_mix -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-attention-v-mix-fixture --model <verified-qwen2.5-q8-gguf> --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-v-mix-fixture-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-attention-v-mix-fixture-qwen25-q8.json
```

## Next Step

The next dense CUDA slice should add strict RTX 5070 Ti CUDA parity for the
attention V-mix fixture. It should remain below dense GGUF token/decode/chat,
speedup, server, and full-residency claims.
