# OpenAI Privacy Filter

## Status
- Repo status: design_scaffold
- First target: token_classification_cpu_or_onnx_smoke
- Local test tier: tier_a_local
- Implementation owner lane: token_classification, onnx_runtime_future, safetensors_loader, transformersjs_reference
- Design-only? no for token-classification smoke planning

## Source-backed facts
- Model variants: Bidirectional token-classification model for PII detection/masking.
- Context: 128K.
- Modalities: Text token classification.
- Tokenizer/prompt: Not an autoregressive decoder and must not be forced into `generate()`.
- Architecture features: pre-norm encoder-style stack; 8 blocks; GQA 14 query heads and 2 KV heads; sparse MoE FFN with 128 experts, top-4 routing; d_model=640; banded attention size 128/effective 257 tokens; BIOES labels; constrained Viterbi span decoder.
- Quantization/runtime notes: ONNX, safetensors, and Transformers.js surfaces exist in supplied card.
- License: Apache 2.0.
- Known tool support: Token-classification lane; ONNX/runtime and safetensors future gates.
- Source links: Supplied planning notes in this change; model-card URLs must be verified before advancing beyond `documented` or `design_scaffold`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---:|---:|---|---|---|---|
| privacy-filter | 1.5B | 50M active | 128K | text token labels | CPU/ONNX token smoke | tier_a_local |

## Implementation contract
### Loader
ONNX or safetensors design path; must expose encoder outputs and classification head, not generation logits.

### Tokenizer
Tokenizer source must align token labels to input tokens.

### Prompt template
No chat template required for generation; task input is classification text/token sequence.

### Architecture module
Token classifier with bidirectional encoder, banded attention, sparse MoE FFN, BIOES label head, Viterbi span decoder.

### Kernels/backend
Token classification path; generation kernels and QK256 proof do not apply.

### Receipts
Receipt records output shape `[tokens, 33]`, label taxonomy, span decoder, `generation_claim=false`, fallback, and speedup flags.

### Tests
Future local CPU/ONNX smoke: 12-token input, 33-class output shape, selected labels, coherent spans.

## Explicit non-claims
- Privacy-filter docs do not mean ONNX or safetensors loading works.
- Token classification is not generation and must not be exposed as `generate()` support.
- BIOES/Viterbi notes do not prove calibration, recall, precision, masking quality, or operating point.

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
