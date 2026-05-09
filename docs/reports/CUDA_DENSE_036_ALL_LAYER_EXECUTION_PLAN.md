# CUDA-DENSE-036 All-Layer Dense GGUF Execution Plan Contract

`CUDA-DENSE-036` is the next dense CUDA contract after `CUDA-DENSE-035`.
`CUDA-DENSE-035` proved the integrated CUDA-routable layer-0 plan against the
CPU reference harness. This item defines the next receipt boundary: inspect the
complete Qwen-family dense GGUF transformer stack and prove whether every layer
has the same CUDA-routable block plan before any token-generation claim is
allowed.

This report is a tracker contract only. It does not add all-layer execution,
hardware receipts, dense GGUF inference, Qwen one-token generation, short
decode, chat, speedup, persistent residency, full residency, server readiness,
or BitNet packed I2_S/QK256 proof.

## Required Implementation Shape

The future implementation should add a strict planning command that accepts the
same verified Qwen-family dense GGUF artifact used by the layer-0 fixtures:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
  dense-gguf-all-layer-plan `
  --model <verified-qwen2.5-q8-gguf> `
  --device-index 0 `
  --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-all-layer-plan-qwen25-q8.json
```

The command should inspect every transformer layer in the artifact, compare the
layer op graph against the `CUDA-DENSE-035` layer-0 governed plan, and emit a
planning receipt. It should not execute a full model pass or generate tokens.

## Required Receipt

The future implementation should emit:

```text
artifact_kind: dense_gguf_all_layer_execution_plan
claim: dense_gguf_all_layer_execution_plan_recorded
selected_backend: nvidia-rtx-5070-ti-cuda
runtime_api: cuda
fallback_used: false
speedup_claim: false
dense_gguf_inference_claimed: false
qwen_one_token_cuda_claimed: false
full_cuda_residency_claimed: false
bitnet_packed_i2s_qk256_proof: false
```

Required receipt evidence:

- model family, artifact path, and artifact SHA;
- execution route `dense_regular_llm_cuda`;
- transformer layer count from the GGUF metadata;
- `transformer_layers_total`;
- `layers_with_complete_cuda_block_plan`;
- `layer_plan_matches_layer0`;
- `layer_differences`, empty only when every inspected layer matches the
  governed layer-0 graph;
- per-layer counts for the 14 governed dense CUDA block ops;
- `unsupported_strict_cuda_ops` for any layer-specific gap;
- explicit model-boundary gaps for token embedding lookup, final norm, LM
  head/logits, KV cache policy, and sampling;
- strict CUDA readiness scoped to the transformer-block plan only;
- claim boundary values keeping dense inference, Qwen token/decode/chat,
  speedup, persistent/full residency, server readiness, and BitNet packed proof
  false.

## Governed Block Ops

Each transformer layer should be checked for the same governed block ops proven
through the one-layer lane:

| Phase | Required route |
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
| `ffn_norm` | dense RMSNorm CUDA |
| `mlp_gate` | dense linear CUDA |
| `mlp_up` | dense linear CUDA |
| `mlp_activation` | dense MLP activation CUDA |
| `mlp_down` | dense linear CUDA |

Residual adds may remain explicit measured host glue in later implementation
receipts, but they must stay visible and must not hide CPU fallback.

## Model-Boundary Gaps

The all-layer plan is not enough for dense inference. The receipt must keep the
following model-boundary gaps explicit until later proof items close them:

| Gap | Required disposition |
| --- | --- |
| `token_embedding` | listed as not yet governed by all-layer block planning |
| `final_norm` | listed as not yet governed by all-layer block planning |
| `lm_head_logits` | listed as not yet governed by all-layer block planning |
| `kv_cache_policy` | listed as pending explicit residency and transfer policy |
| `sampling` | listed as pending sampler integration and claim boundary |

These gaps are blockers for Qwen one-token, short decode, chat, and server
claims, even if every transformer block routes cleanly.

## Acceptance

- The contract is docs/tracker only.
- The future implementation must inspect every Qwen-family transformer layer.
- The future implementation must report per-layer routed op counts and any
  layer-specific graph differences.
- Dense regular-LLM CUDA remains separate from BitNet packed I2_S/QK256 proof.
- Model-boundary gaps are explicit before any one-token or decode claim.
- `fallback_used=false`, `speedup_claim=false`,
  `full_cuda_residency_claimed=false`, and
  `dense_gguf_inference_claimed=false` remain required.

## May Claim

- `CUDA-DENSE-036` defines the governed dense GGUF all-layer execution-plan
  receipt contract.
- The next implementation must prove whether every Qwen-family transformer
  layer has the same CUDA-routable block plan as layer 0 or list exact
  differences.
- Model-boundary gaps must be explicit before Qwen one-token, short decode,
  chat, benchmark, or server claims.

## Must Not Claim

- All-layer execution-plan receipt implementation exists from this tracker-only
  PR.
- Hardware receipt evidence exists.
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
git diff --check
```
