# Kimi K2.6

## Status
- Repo status: design_only
- First target: kimi-k2.6_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: design lanes and external_reference
- Design-only? yes

## Source-backed facts
- Model variants: Kimi K2.6.
- Context: 256K.
- Modalities: vision support in Unsloth GGUF notes.
- Tokenizer/prompt: Kimi-specific chat template uses `<|im_system|>`, `<|im_user|>`, `<|im_assistant|>`, and `<think>`.
- Architecture features: 1T-parameter hybrid-thinking MoE-scale model, long context, extreme model size, dynamic quant reference.
- Quantization/runtime notes: full precision disk footprint around 610 GB; dynamic 2-bit around 340-350 GB; Q4 around 584 GB; Q8 around 595 GB.
- License: source card verification required.
- Known tool support: GGUF dynamic quant references are external/design-only locally.
- Source links: supplied Kimi K2.6 notes; source-index `kimi26-unsloth-notes`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---|---|---|---|---|---|
| Kimi K2.6 | 1T | TBD | 256K | text/vision | design-only prompt/tokenizer/loader notes | tier_c_design_only |

## Implementation contract
### Loader
Document intended GGUF/safetensors/projector inputs before adding runtime code. Loader notes are not inference proof.

### Tokenizer
Record tokenizer source explicitly; unknown or sibling fallback must remain a non-claim until receipt-backed.

### Prompt template
Record family-specific chat template switches separately from tokenizer loading.

### Architecture module
Map features to `ARCHITECTURE_FEATURE_GLOSSARY.md`; mark unknowns TBD.

### Kernels/backend
Do not route through BitNet QK256/W1.58 kernels and call the family supported.

### Receipts
Use templates under `ci/model-receipts/_templates`; coverage booleans must be explicit.

### Tests
First tests are future one-token, shape, router, or token-classification smokes only.

## Explicit non-claims
- This family is design-only on current hardware.
- Docs do not imply local loader, inference, speed, quality, long-context, multimodal, or kernel support.
- Any future local/offload/reference artifact must be receipt-backed.

## First proof target

```json
{
  "claim": "design_only",
  "model_family": "kimi-k2.6",
  "local_execution_claim": false,
  "loader_claim": false,
  "inference_claim": false,
  "speedup_claim": false
}
```

## Work items
- KIMI26-DOC-001: Add Kimi K2.6 design-only family doc.
- KIMI26-DOC-002: Add chat template and thinking-mode notes.
- KIMI26-DOC-003: Add hardware infeasibility/local non-claim section.
- KIMI26-DOC-004: Add future MoE/huge-model loader design notes.
- KIMI26-DOC-005: Add external-reference receipt template.
