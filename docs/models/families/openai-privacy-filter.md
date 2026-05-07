# OpenAI privacy-filter

## Status
- Repo status: design_scaffold.
- First target: token_classification_cpu_or_onnx_smoke.
- Local test tier: tier_a_local.
- Implementation owner lane: token_classification, onnx_runtime_future, safetensors_loader, transformersjs_reference.
- Design-only? no for planning a CPU/ONNX smoke; yes for calibration and production masking until receipts exist.

## Source-backed facts
- Model variants: OpenAI privacy-filter token-classification model.
- Context: 128K.
- Modalities: text tokens for PII detection/masking.
- Tokenizer/prompt: not an autoregressive chat/generate template.
- Architecture features: bidirectional token-classification model, pre-norm transformer encoder-style stack, 8 transformer blocks, GQA with 14 query heads and 2 KV heads, sparse MoE FFN with 128 experts and top-4 routing, `d_model=640`, banded attention with band size 128 and effective window 257.
- Quantization/runtime notes: ONNX, safetensors, and Transformers.js surfaces exist in supplied card.
- License: Apache 2.0.
- Known tool support: token classification, not generation.
- Source links: Source links are tracked in `docs/tracking/model-family-foundation/source-index.yaml`; supplied notes must be verified against model cards before implementation claims advance.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| privacy-filter | 1.5B | 50M active | 128K | text token classification | CPU/ONNX smoke | tier_a_local |

## Implementation contract
### Loader
Start with ONNX or safetensors path planning; do not force through autoregressive decoder loaders.
### Tokenizer
Tokenizer must align input tokens to output labels.
### Prompt template
No generate/chat template; use token-classification input policy.
### Architecture module
Requires encoder/bidirectional attention, classification head, per-token logits, and MoE routing if native.
### Kernels/backend
ONNX/reference backend may be first; native support requires receipts.
### Receipts
Receipt must include output shape `[tokens, 33]`, BIOES taxonomy, and Viterbi decoder.
### Tests
First smoke labels a short deterministic input and checks shape/classes.

## Explicit non-claims
- OpenAI privacy-filter is not an autoregressive decoder.
- A token-classification smoke is not a generation claim.
- Calibration, masking quality, and operating points are future-gated.

## First proof target
```json
{
  "claim": "privacy_filter_token_classification_smoke",
  "model_family": "openai-privacy-filter",
  "task": "token_classification",
  "input_tokens": 12,
  "output_shape": [12, 33],
  "span_decoder": "bioes_viterbi",
  "labels": ["private_person", "private_email", "private_phone"],
  "generation_claim": false,
  "fallback_used": false,
  "speedup_claim": false
}
```

## Work items
- PRIVACY-DOC-001: Add privacy-filter family doc.
- PRIVACY-DOC-002: Add token-classification lane doc.
- PRIVACY-DOC-003: Add BIOES/Viterbi receipt schema.
- PRIVACY-DOC-004: Add local CPU/ONNX smoke proof plan.
- PRIVACY-DOC-005: Add calibration/operating-point future-gate.
