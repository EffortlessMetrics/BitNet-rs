# CUDA-DENSE-033 One-Layer CPU Reference Harness

`CUDA-DENSE-033` implements the dense GGUF layer-0 CPU reference harness
defined by `CUDA-DENSE-032` after the route-complete one-layer planner from
`CUDA-DENSE-031`.

The harness is CPU-only. It composes the already governed dense layer phases
against deterministic input and emits per-phase hashes plus a final layer output
hash. This creates the comparison anchor for `CUDA-DENSE-034`; it does not run
the layer through CUDA and does not claim dense GGUF inference.

## Command

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  dense-gguf-one-layer-cpu-reference `
  --model <verified-qwen2.5-q8-gguf> `
  --layer-index 0 `
  --seq-len 4 `
  --position-offset 1 `
  --json-out <receipt.json>
```

## Receipt

The command emits:

```text
artifact_kind: dense_gguf_one_layer_cpu_reference
claim: dense_gguf_one_layer_cpu_reference_recorded
runtime_api: cpu
selected_backend: cpu_reference
fallback_used: false
speedup_claim: false
```

The receipt records these governed phases:

| Phase | Role |
| --- | --- |
| `deterministic_input` | deterministic input hidden states |
| `attention_norm` | attention RMSNorm |
| `attention_q` | Q projection |
| `attention_k` | K projection |
| `attention_v` | V projection |
| `rope` | Q/K RoPE |
| `attention_scores` | causal attention scores |
| `attention_softmax` | attention probabilities |
| `attention_v_mix` | probability times V context |
| `attention_output` | attention output projection |
| `first_residual` | post-attention residual |
| `ffn_norm` | FFN RMSNorm |
| `mlp_gate` | MLP gate projection |
| `mlp_up` | MLP up projection |
| `mlp_activation` | SiLU gate times up |
| `mlp_down` | MLP down projection |
| `second_residual` | final layer output |

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_one_layer_cpu_reference` |
| `claim` | `dense_gguf_one_layer_cpu_reference_recorded` |
| `reference_harness.layer_index` | `0` |
| `reference_harness.cpu_reference_only` | `true` |
| `reference_harness.phases_total` | `17` |
| `reference_harness.final_output_sha256` | recorded |
| `reference_harness.next_required_proof` | `one_layer_cuda_integrated_parity` |
| `claim_boundary.dense_gguf_one_layer_cpu_reference_claimed` | `true` |
| `claim_boundary.dense_regular_llm_cuda_claimed` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |
| `speedup_claim` | `false` |

## May Claim

- A governed dense GGUF layer-0 CPU reference harness exists.
- The harness emits deterministic input, per-phase hashes, and final layer
  output hash in a validated receipt.
- The next governed dense CUDA proof can compare an integrated CUDA layer pass
  against this CPU reference.

## Must Not Claim

- CUDA integrated one-layer execution works;
- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- persistent-session or full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, CUDA kernel math, or
  BitNet CUDA behavior changed.

## Validation

```powershell
cargo fmt -p bitnet-cli -p bitnet-receipts-core -p bitnet-receipts -- --check
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,full-cli one_layer_cpu_reference -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features one_layer_cpu_reference -- --nocapture
cargo check --locked -p bitnet-cli --no-default-features --features cpu,full-cli
```

No committed hardware receipt is added by this PR. The full Qwen GGUF artifact
is external and the claim is CPU reference harness implementation plus validator
coverage, not RTX 5070 Ti execution evidence.

## Next Step

`CUDA-DENSE-034` should run the same layer through the integrated dense CUDA
path and compare the full layer output against this CPU reference receipt.
