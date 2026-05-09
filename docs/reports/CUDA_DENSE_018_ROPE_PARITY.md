# CUDA-DENSE-018 RoPE Parity

`CUDA-DENSE-018` adds a strict RTX 5070 Ti CUDA RoPE parity fixture for the
verified Qwen2.5 0.5B Q8_0 dense GGUF artifact.

The fixture is derived from GGUF metadata: architecture, attention head counts,
KV head counts, head dimension, and RoPE frequency base. It runs deterministic
Q and K inputs through the `dense_rope_f32_cuda` kernel and compares the CUDA
outputs against the CPU reference implementation.

This is a fixture-level dense regular-LLM CUDA proof. It does not claim dense
GGUF inference, Qwen one-token generation, short decode, chat, speedup, full
residency, or BitNet packed I2_S/QK256 proof.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-rope-cuda-parity-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_rope_cuda_parity` |
| `claim` | `dense_gguf_rope_cuda_parity_tested` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `model.architecture` | `qwen2` |
| `model_family` | `qwen` |
| `rope_fixture.q_heads` | `14` |
| `rope_fixture.kv_heads` | `2` |
| `rope_fixture.head_dim` | `64` |
| `rope_fixture.seq_len` | `4` |
| `rope_fixture.rope_base` | `1000000.0` |
| `kernel_stats[0].kernel_id` | `dense_rope_f32_cuda` |
| `kernel_stats[0].invocations` | `2` |
| `kernel_stats[0].kernel_launches` | `2` |
| `parity.max_abs_error` | `5.960464477539063e-8` |
| `parity.mean_abs_error` | `8.384841265751675e-9` |
| `parity.passed` | `true` |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- Metadata-derived Q/K RoPE fixtures for the verified Qwen2.5 0.5B Q8_0 dense
  GGUF artifact pass strict RTX 5070 Ti CUDA parity against CPU references.
- The receipt records the dense regular-LLM CUDA route, `dense_rope_f32_cuda`
  kernel launches, tensor residency, transfer byte accounting, and
  `fallback_used=false`.
- A later governed one-layer planner update can mark RoPE CUDA-routable based
  on this parity evidence.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, or BitNet proof behavior
  changed.

## Validation

```powershell
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda rope -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_rope -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features dense_gguf_rope -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-rope-cuda-parity --model <verified-qwen2.5-q8-gguf> --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-rope-cuda-parity-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-rope-cuda-parity-qwen25-q8.json
```

## Next Step

The next scoped dense CUDA proof should update the one-layer planner to route
verified RoPE ops. That follow-up should reduce the remaining one-layer gaps
from RoPE, attention scores, attention softmax, attention V mix, and MLP
activation to the attention and MLP activation gaps only.
