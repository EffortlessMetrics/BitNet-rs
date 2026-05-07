# Kimi K2.6

## Status
- Repo status: design_only
- First target: design_only_prompt_tokenizer_loader_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: prompt_template, external_reference, design lane
- Design-only? yes

## Source-backed facts
- Model variants: Kimi K2.6 is a 1T-parameter hybrid-thinking model.
- Context: 256K.
- Modalities: Vision support in Unsloth GGUF notes.
- Tokenizer/prompt: Kimi-specific chat template uses `<|im_system|>`, `<|im_user|>`, `<|im_assistant|>`, and `<think>`.
- Architecture features: hybrid_thinking, long_context, vision, extreme_model_size, dynamic_quant_reference.
- Quantization/runtime notes: Full precision ~610 GB; dynamic 2-bit ~340-350 GB; Q4 ~584 GB; Q8 ~595 GB.
- License: TBD from model card.
- Known tool support: External/offload/reference only until receipt-backed.
- Source links: Supplied planning notes in this change; model-card URLs must be verified before advancing beyond `documented` or `design_scaffold`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---:|---:|---|---|---|---|
| K2.6 | 1T | TBD | 256K | text/vision | design-only notes | tier_c_design_only |

## Implementation contract
### Loader
Design-only loader notes. No local loader, inference, or speed claim is allowed.

### Tokenizer
Record tokenizer source and template source before any support claim.

### Prompt template
Record family-specific prompt controls and disable unsupported fallbacks.

### Architecture module
Architecture notes only; implementation details remain TBD until source-backed and parity-tested.

### Kernels/backend
No local kernel claim. Non-BitNet QK256/BitNet evidence is invalid.

### Receipts
Use `generative-design-only.json` or `external-reference-proof.json` until narrow execution receipts exist.

### Tests
YAML/JSON parse only now; future external reference command receipts.

## Explicit non-claims
- Kimi K2.6 is design-only on current hardware.
- Any future local artifact is offload/reference unless receipt-backed.
- No speed, quality, long-context, or full-inference claim is allowed from docs.

## First proof target
Design-only receipt with local execution, loader, inference, and speedup claims all false.

## Work items
- KIMI26-DOC-001: Add Kimi K2.6 family doc.
- KIMI26-DOC-002: Add prompt/template and architecture notes.
- KIMI26-DOC-003: Add hardware infeasibility/local non-claim section.
- KIMI26-DOC-004: Add external-reference receipt plan.
