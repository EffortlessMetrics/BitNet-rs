# CUDA-DENSE-035 Integrated One-Layer CUDA Parity Harness

`CUDA-DENSE-035` implements the diagnostic harness defined by
`CUDA-DENSE-034`. The new command composes the existing governed dense CUDA
component launchers into a full layer-0 pass and compares the results against
the `CUDA-DENSE-033` CPU reference harness.

This is still a one-layer diagnostic. It does not claim dense GGUF inference,
Qwen one-token generation, short decode, chat, speedup, persistent residency,
full residency, server readiness, or BitNet packed I2_S/QK256 proof.

## Command

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
  dense-gguf-one-layer-cuda-parity `
  --model <verified-qwen2.5-q8-gguf> `
  --layer-index 0 `
  --seq-len 4 `
  --position-offset 1 `
  --device-index 0 `
  --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-one-layer-cuda-integrated-parity-qwen25-q8.json
```

The command loads the same Qwen-family dense GGUF tensor set used by the CPU
reference harness, builds the deterministic input fixture, executes the
CUDA-routable phases, records measured host glue for residual operations, and
emits a `dense_gguf_one_layer_cuda_integrated_parity` receipt.

## Routed Phases

| Phase | Route |
| --- | --- |
| `deterministic_input` | host deterministic input |
| `attention_norm` | dense RMSNorm CUDA |
| `attention_q` | dense linear CUDA |
| `attention_k` | dense linear CUDA |
| `attention_v` | dense linear CUDA |
| `rope` | dense RoPE CUDA |
| `attention_scores` | dense attention-score CUDA |
| `attention_softmax` | dense attention-softmax CUDA |
| `attention_v_mix` | dense attention V-mix CUDA |
| `attention_output` | dense linear CUDA |
| `first_residual` | measured host glue |
| `ffn_norm` | dense RMSNorm CUDA |
| `mlp_gate` | dense linear CUDA |
| `mlp_up` | dense linear CUDA |
| `mlp_activation` | dense MLP activation CUDA |
| `mlp_down` | dense linear CUDA |
| `second_residual` | measured host glue |

The governed CUDA op count remains 14. The deterministic input and two residual
adds are explicitly recorded as non-kernel phases, not hidden fallback.

## Receipt Contract

The receipt validator requires:

- `artifact_kind=dense_gguf_one_layer_cuda_integrated_parity`;
- `claim=dense_gguf_one_layer_cuda_integrated_parity_recorded`;
- `selected_backend=nvidia-rtx-5070-ti-cuda`;
- `runtime_api=cuda`;
- `fallback_used=false`;
- `speedup_claim=false`;
- a dense execution plan with 14 CUDA-routable governed ops and zero unsupported
  or CPU fallback ops;
- a CPU reference source of kind `dense_gguf_one_layer_cpu_reference`;
- per-phase hashes, output lengths, maximum absolute error, tolerance, and pass
  status;
- per-op CUDA kernel identities, launch counts, and transfer bytes;
- aggregate H2D/D2H transfer accounting;
- tensor residency scoped only to integrated one-layer diagnostics;
- claim-boundary booleans keeping dense inference, Qwen token/decode/chat,
  speedup, persistent/full residency, server readiness, and BitNet proof false.

## May Claim

- `CUDA-DENSE-035` implements the integrated dense GGUF one-layer CUDA parity
  diagnostic harness.
- The harness compares the full layer-0 CUDA-routable plan against the
  `CUDA-DENSE-033` CPU reference output.
- The receipt validator rejects claim leakage from one-layer parity into dense
  inference, speedup, full residency, server readiness, or BitNet packed proof.

## Must Not Claim

- Dense GGUF inference works.
- Qwen one-token, short decode, or chat works on CUDA.
- Dense CUDA speedup is accepted.
- Persistent-session or full dense CUDA residency is proven.
- Dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference.
- Tokenizer, loader, transformer, server, QK256 math, CUDA kernel math, or
  BitNet CUDA behavior changed.

## Validation

```powershell
cargo fmt -p bitnet-cli -p bitnet-receipts-core -p bitnet-receipts -- --check
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_one_layer -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features one_layer -- --nocapture
cargo test --locked -p bitnet-cli --test cli_smoke --no-default-features --features cpu,cuda,full-cli -- --nocapture
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
cargo run --release --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --release --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
