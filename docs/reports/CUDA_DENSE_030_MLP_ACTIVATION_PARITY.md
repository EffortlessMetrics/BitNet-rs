# CUDA-DENSE-030 MLP Activation CUDA Parity

`CUDA-DENSE-030` runs the `CUDA-DENSE-029` dense GGUF MLP activation
fixture through a strict RTX 5070 Ti CUDA F32 kernel and compares the output
against the CPU-reference `SiLU(mlp_gate) * mlp_up` activation vector.

This remains fixture-level dense regular-LLM CUDA evidence. It does not claim
dense GGUF inference, Qwen token/decode/chat, speedup, route promotion,
persistent-session residency, full CUDA residency, or BitNet packed I2_S/QK256
proof.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-mlp-activation-cuda-parity-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_mlp_activation_cuda_parity` |
| `claim` | `dense_gguf_mlp_activation_cuda_parity_tested` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `execution_plan.cuda_dense_regular_llm_ops` | `1` |
| `mlp_activation_fixture.activation_kind` | `silu_gate_times_up` |
| `mlp_activation_fixture.activation_count` | `4864` |
| `mlp_activation_fixture.cuda_kernel_status` | `parity_passed` |
| `kernel_stats[0].kernel_id` | `dense_mlp_activation_f32_cuda` |
| `kernel_stats[0].host_to_device_bytes` | `38912` |
| `kernel_stats[0].device_to_host_bytes` | `19456` |
| `parity.compared_activations` | `4864` |
| `parity.max_abs_error` | `7.450580596923828e-9` |
| `parity.mean_abs_error` | `6.282118575340334e-11` |
| `parity.tolerance` | `0.00025` |
| `mlp_activation_gap_audit.next_required_proof` | `one_layer_route_promotion` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- The metadata-derived dense GGUF MLP activation fixture passes strict RTX
  5070 Ti CUDA F32 parity against the CPU reference.
- The receipt records CUDA tensor residency and transfer-byte accounting for
  the gate, up, and activation vectors.
- The next governed dense CUDA proof is one-layer route promotion for verified
  `mlp_activation`.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, or BitNet CUDA behavior
  changed.

## Validation

```powershell
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda mlp_activation -- --nocapture
cargo test --locked -p bitnet-cli --bin bitnet --no-default-features --features cpu,cuda,full-cli mlp_activation -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features mlp_activation -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-mlp-activation-cuda-parity --model <verified-qwen2.5-q8-gguf> --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-mlp-activation-cuda-parity-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-mlp-activation-cuda-parity-qwen25-q8.json
```

## Next Step

The planner-route follow-up is
[`CUDA-DENSE-031`](CUDA_DENSE_031_MLP_ACTIVATION_ROUTE.md), which promotes
verified `mlp_activation` into the one-layer planner route while leaving dense
GGUF token/decode/chat, speedup, server, and full-residency claims false.
