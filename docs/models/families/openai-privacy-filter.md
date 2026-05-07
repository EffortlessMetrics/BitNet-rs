# OpenAI privacy-filter

## Status
- Repo status: design_scaffold
- First target: token_classification_cpu_or_onnx_smoke
- Local test tier: tier_a_local
- Implementation owner lane: token_classification, onnx_runtime_future, safetensors_loader, transformersjs_reference
- Design-only? no

## Source-backed facts
- Model type: bidirectional token-classification model for PII detection/masking, not an autoregressive decoder.
- License: Apache 2.0.
- Parameters: 1.5B total and 50M active.
- Context: 128K.
- Architecture features: pre-norm transformer encoder-style stack with 8 blocks, GQA with 14 query heads and 2 KV heads, sparse MoE FFN with 128 experts and top-4 routing, d_model = 640.
- Attention: banded attention with band size 128 and effective window 257 tokens.
- Output taxonomy: 8 privacy categories expanded to BIOES labels plus O for 33 token-level classes.
- Inference: constrained Viterbi decoding produces coherent spans.
- Known tool support: ONNX, safetensors, and Transformers.js surfaces exist in supplied card notes.
- Source links: supplied planning notes; model card URL TBD.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| privacy-filter | 1.5B | 50M | 128K | token labels | CPU/ONNX token-classification smoke | tier_a_local |

## Implementation contract
### Loader
Record intended formats and external-reference commands only; no loader claim exists until receipt-backed.

### Tokenizer
Tokenizer source and fallback status must be explicit. Unknown tokenizer details remain `TBD`.

### Prompt template
Prompt template behavior is documented as a contract and must be receipt-backed before any support claim.

### Architecture module
Architecture notes are design rails until implemented, smoke-tested, parity-tested where applicable, and receipt-backed.

### Kernels/backend
No BitNet QK256, mixed-precision, long-context, multimodal, or speedup claim is allowed from documentation.

### Receipts
Use design-only or external-reference templates until local proof is feasible.

### Tests
No local runtime tests are implied by this document.

## Explicit non-claims
- OpenAI privacy-filter is token classification, not generation.
- A span-classification smoke does not imply autoregressive generation support.
- Calibration and operating-point claims are future-gated.

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

