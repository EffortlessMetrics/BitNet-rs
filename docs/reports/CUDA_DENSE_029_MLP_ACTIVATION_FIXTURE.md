# CUDA-DENSE-029 MLP Activation Fixture

`CUDA-DENSE-029` extracts a CPU-reference dense GGUF MLP activation fixture
after `CUDA-DENSE-028` made `attention_v_mix` CUDA-routable in the one-layer
planner.

The fixture records deterministic `SiLU(mlp_gate) * mlp_up` activation values
for the verified Qwen2.5 0.5B Q8_0 artifact. It deliberately records the CUDA
MLP activation kernel as missing and does not claim dense GGUF inference.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-mlp-activation-fixture-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_mlp_activation_fixture_extraction` |
| `claim` | `dense_gguf_mlp_activation_fixture_extracted` |
| `execution_plan.selected_route` | `unsupported` |
| `execution_plan.unsupported_ops` | `1` |
| `mlp_activation_fixture.activation_kind` | `silu_gate_times_up` |
| `mlp_activation_fixture.activation_count` | `4864` |
| `mlp_activation_fixture.source_mlp_gate_tensor` | `blk.0.ffn_gate.weight` |
| `mlp_activation_fixture.source_mlp_up_tensor` | `blk.0.ffn_up.weight` |
| `mlp_activation_fixture.cuda_kernel_status` | `missing_cuda_kernel` |
| `mlp_activation_gap_audit.next_required_proof` | `cuda_mlp_activation_kernel_parity` |
| `mlp_activation_gap_audit.source_mlp_gate_cuda_parity_available` | `true` |
| `mlp_activation_gap_audit.source_mlp_up_cuda_parity_available` | `true` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- A dense GGUF MLP activation CPU-reference fixture exists for the verified
  Qwen2.5 0.5B Q8_0 artifact.
- The fixture records deterministic activation hashes and source dependency
  authority for verified MLP gate and MLP up inputs.
- The next governed dense CUDA proof is `cuda_mlp_activation_kernel_parity`.

## Must Not Claim

- MLP activation CUDA parity exists;
- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, QK256 math, BitNet CUDA behavior, or
  dense CUDA kernel math changed.

## Validation

```powershell
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli mlp_activation -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features mlp_activation -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-mlp-activation-fixture --model <verified-qwen2.5-q8-gguf> --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-mlp-activation-fixture-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-mlp-activation-fixture-qwen25-q8.json
```

## Next Step

The next dense CUDA slice is
[`CUDA-DENSE-030`](CUDA_DENSE_030_MLP_ACTIVATION_PARITY.md), strict RTX 5070
Ti CUDA parity for the MLP activation fixture. It remains below dense GGUF
token/decode/chat, speedup, server, and full-residency claims.
