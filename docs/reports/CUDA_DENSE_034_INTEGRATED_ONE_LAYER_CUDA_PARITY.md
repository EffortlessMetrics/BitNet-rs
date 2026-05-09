# CUDA-DENSE-034 Integrated One-Layer CUDA Parity Contract

`CUDA-DENSE-034` is the next dense CUDA proof after `CUDA-DENSE-033`.
`CUDA-DENSE-033` created the CPU-only full layer-0 reference harness. This
item defines the contract for the next implementation: run the same governed
layer through the integrated CUDA-routable plan and compare the result against
that CPU reference.

This report is a tracker contract only. It does not add CUDA execution, hardware
receipts, dense GGUF inference, token generation, chat, speedup, or full
residency claims.

## Required Implementation Shape

The future implementation should add a strict CUDA command that accepts the same
verified Qwen-family dense GGUF artifact and layer settings used by the CPU
reference harness:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
  dense-gguf-one-layer-cuda-parity `
  --model <verified-qwen2.5-q8-gguf> `
  --layer-index 0 `
  --seq-len 4 `
  --position-offset 1 `
  --device-index 0 `
  --json-out <receipt.json>
```

The implementation should reuse the verified component kernels already governed
by the dense CUDA lane:

| Phase | Route Requirement |
| --- | --- |
| `attention_norm` | dense RMSNorm CUDA |
| `attention_q` | dense linear CUDA |
| `attention_k` | dense linear CUDA |
| `attention_v` | dense linear CUDA |
| `rope` | dense RoPE CUDA |
| `attention_scores` | dense attention-score CUDA |
| `attention_softmax` | dense attention-softmax CUDA |
| `attention_v_mix` | dense attention V-mix CUDA |
| `attention_output` | dense linear CUDA |
| `first_residual` | explicit CUDA or measured host operation |
| `ffn_norm` | dense RMSNorm CUDA |
| `mlp_gate` | dense linear CUDA |
| `mlp_up` | dense linear CUDA |
| `mlp_activation` | dense MLP activation CUDA |
| `mlp_down` | dense linear CUDA |
| `second_residual` | explicit CUDA or measured host operation |

Residual adds may remain explicit measured glue for this item, but they must be
visible in the receipt and must not hide CPU fallback.

## Required Receipt

The future implementation should emit:

```text
artifact_kind: dense_gguf_one_layer_cuda_integrated_parity
claim: dense_gguf_one_layer_cuda_integrated_parity_recorded
selected_backend: nvidia-rtx-5070-ti-cuda
runtime_api: cuda
fallback_used: false
speedup_claim: false
```

Required receipt evidence:

- model family, artifact path, and artifact SHA;
- execution plan route `dense_regular_llm_cuda`;
- CPU reference fixture identity and final output hash from the same inputs;
- CUDA final output hash;
- final output parity status and max absolute error;
- per-phase status for all layer phases;
- per-op kernel identities and launch counts where CUDA kernels run;
- aggregate host-to-device and device-to-host bytes;
- explicit residency for weights, inputs, intermediates, and outputs;
- claim boundary values keeping inference, token/decode/chat, speedup,
  persistent residency, full residency, server readiness, and BitNet packed
  proof false.

## Acceptance

- Same verified Qwen2.5 0.5B Q8_0 dense GGUF artifact as the CPU reference
  harness.
- Layer 0 only.
- Same `seq_len` and `position_offset` as the CPU reference receipt.
- All 14 governed one-layer ops route through `dense_regular_llm_cuda`.
- `fallback_used=false`.
- CPU/CUDA final layer output parity is recorded with a bounded tolerance.
- Per-op kernel stats and aggregate transfer bytes are recorded.
- Receipt validation rejects dense GGUF inference, Qwen token/decode/chat,
  speedup, persistent/full residency, and BitNet packed I2_S/QK256 proof claims.

## May Claim

- `CUDA-DENSE-034` defines the integrated one-layer CUDA parity contract.
- The future implementation has a precise CPU reference anchor from
  `CUDA-DENSE-033`.
- The future implementation may claim one-layer CUDA parity only after the
  integrated CUDA receipt validates.

## Must Not Claim

- Integrated one-layer CUDA execution exists from this tracker-only PR.
- Dense GGUF inference works.
- Qwen one-token, short decode, or chat works on CUDA.
- Dense CUDA speedup is accepted.
- Persistent-session or full dense CUDA residency is proven.
- Dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference.
- Tokenizer, loader, transformer, server, QK256 math, CUDA kernel math, or
  BitNet CUDA behavior changed.

## Validation For This Contract PR

```powershell
cargo run --release --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --release --locked -p xtask --no-default-features -- campaign generate --check
cargo run --release --locked -p xtask --no-default-features -- campaign doctor
git diff --check
```
