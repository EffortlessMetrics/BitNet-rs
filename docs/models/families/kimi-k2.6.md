# Kimi K2.6

## Status
- Repo status: design_only.
- First target: design_only_prompt_tokenizer_loader_notes.
- Local test tier: tier_c_design_only.
- Implementation owner lane: moe_decoder_design, prompt_template, gguf_dynamic_quant_design.
- Design-only? yes.

## Source-backed facts
- Model variants: Kimi K2.6 hybrid-thinking 1T-parameter family.
- Context: 256K.
- Modalities: vision support appears in supplied Unsloth GGUF notes.
- Tokenizer/prompt: Kimi chat template uses `<|im_system|>`, `<|im_user|>`, `<|im_assistant|>`, and `<think>` in supplied notes.
- Architecture features: hybrid_thinking, long_context, vision, extreme_model_size, dynamic_quant_reference.
- Quantization/runtime notes: full precision disk footprint around 610 GB; dynamic 2-bit around 340-350 GB; Q4 around 584 GB; Q8 around 595 GB.
- License: TBD from model card.
- Known tool support: external/offload/reference only for current local hardware.
- Source links: Source links are tracked in `docs/tracking/model-family-foundation/source-index.yaml`; supplied notes must be verified against model cards before implementation claims advance.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| K2.6 | 1T | TBD | 256K | text/vision in supplied notes | design-only prompt/loader notes | tier_c_design_only |

## Implementation contract
### Loader
Record huge-model and dynamic-quant loader notes only.
### Tokenizer
Record tokenizer source and chat-template source before any external receipt.
### Prompt template
Capture Kimi-specific role tokens and thinking marker.
### Architecture module
MoE and huge-model details remain design-only until source-backed enough to implement.
### Kernels/backend
No local kernel claim on 16 GB hardware.
### Receipts
Use design-only or external-reference receipts.
### Tests
No local inference test is planned in first pass.

## Explicit non-claims
- Kimi K2.6 is design-only on current hardware.
- Any future local artifact is offload/reference unless receipt-backed.
- No speed, quality, long-context, or full-inference claim is allowed from docs.

## First proof target
Use `generative-design-only.json` or `external-reference-proof.json` with `local_execution_claim=false`.

## Work items
- KIMI26-DOC-001: Add Kimi K2.6 design-only family doc.
- KIMI26-DOC-002: Add chat template and thinking-mode notes.
- KIMI26-DOC-003: Add hardware infeasibility/local non-claim section.
- KIMI26-DOC-004: Add future MoE/huge-model loader design notes.
- KIMI26-DOC-005: Add external-reference receipt template.
