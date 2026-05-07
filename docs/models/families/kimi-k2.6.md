# Kimi K2.6

## Status
- Repo status: design_only
- First target: design_only_prompt_tokenizer_loader_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: moe_decoder_design, prompt_template, gguf_dynamic_quant_design
- Design-only? yes

## Source-backed facts
- Model type: 1T-parameter hybrid-thinking model.
- Context: 256K.
- Modalities: vision support appears in supplied Unsloth GGUF notes.
- Tokenizer/prompt: Kimi-specific chat template uses `<|im_system|>`, `<|im_user|>`, `<|im_assistant|>`, and `<think>` in supplied notes.
- Architecture features: hybrid_thinking, long_context, vision, extreme_model_size, dynamic_quant_reference.
- Quantization/runtime notes: full precision disk footprint around 610 GB; dynamic 2-bit around 340–350 GB; Q4 around 584 GB; Q8 around 595 GB.
- License: TBD from model card before runtime work.
- Known tool support: GGUF dynamic quant reference only.
- Source links: supplied planning notes; model card URL TBD.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| K2.6 | 1T | TBD | 256K | text/vision | design-only prompt/tokenizer/loader notes | tier_c_design_only |

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
- Kimi K2.6 is design-only on current hardware.
- Any future local artifact is offload/reference unless receipt-backed.
- No speed, quality, long-context, or full-inference claim is allowed from docs.

## First proof target

```json
{
  "claim": "design_only",
  "model_family": "kimi-k2.6",
  "variant": "k2.6",
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

