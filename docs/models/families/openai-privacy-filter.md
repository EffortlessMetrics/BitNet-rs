# OpenAI privacy-filter

## Status
- Repo status: design_scaffold
- First target: token_classification_cpu_or_onnx_smoke
- Local test tier: tier_a_local
- Implementation owner lane: token_classification, onnx_runtime_future, safetensors_loader, transformersjs_reference
- Design-only? no

## Source-backed facts
- Model type: bidirectional token-classification model for PII detection/masking.
- Context: 128K.
- Modalities: token labels, not autoregressive generation.
- Tokenizer/prompt: tokenizer policy TBD from model card; no chat-template generation contract.
- Architecture features: pre-norm encoder-style stack, 8 transformer blocks, GQA with 14 query heads and 2 KV heads, sparse MoE FFN with 128 experts and top-4 routing, d_model=640, banded attention size 128 and effective window 257 tokens.
- Quantization/runtime notes: ONNX, safetensors, and Transformers.js surfaces exist in supplied card.
- License: Apache 2.0.
- Known tool support: constrained Viterbi decoding produces coherent spans.
- Source links: supplied OpenAI privacy-filter card; source-index `openai-privacy-filter-card`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---|---|---|---|---|---|
| privacy-filter | 1.5B total | 50M active | 128K | token labels | CPU/ONNX token-classification smoke | tier_a_local |

## Implementation contract
### Loader
ONNX and safetensors surfaces are tracked separately; loader scaffold is not classification proof.

### Tokenizer
Record tokenizer source explicitly; unknown or sibling fallback must remain a non-claim until receipt-backed.

### Prompt template
No `generate()` prompt template. Record tokenization and label-head contract instead.

### Architecture module
Implement as token_classifier with bidirectional encoder attention, banded attention, sparse MoE, BIOES labels, and Viterbi span decoding.

### Kernels/backend
Do not route through BitNet QK256/W1.58 kernels and call the family supported.

### Receipts
Use token-classification proof with output shape and labels; generation_claim must be false.

### Tests
First test is CPU/ONNX per-token logits plus constrained span decoder smoke.

## Explicit non-claims
- OpenAI privacy-filter is token classification, not generation.
- It must not be forced into `generate()`.
- BIOES/Viterbi documentation does not imply calibrated PII quality.
- ONNX/safetensors surfaces do not imply native bitnet-rs loader support.

## First proof target

```json
{
  "claim": "privacy_filter_token_classification_smoke",
  "model_family": "openai-privacy-filter",
  "task": "token_classification",
  "input_tokens": 12,
  "output_shape": [
    12,
    33
  ],
  "span_decoder": "bioes_viterbi",
  "labels": [
    "private_person",
    "private_email",
    "private_phone"
  ],
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
