# Qwen 3.5

## Status
- Repo status: design_scaffold
- First target: qwen3.5-0.8b-or-2b-text-only-gguf
- Local test tier: tier_a_local
- Implementation owner lane: dense_decoder, prompt_template, tokenizer, gguf_dense_quant, moe_decoder_future
- Design-only? no

## Source-backed facts
- Model variants: 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, and 397B-A17B.
- Context: 256K and 201 languages in supplied notes.
- Modalities: multimodal-family notes include causal LM with a vision encoder in fine-tuning context.
- Tokenizer/prompt: hybrid reasoning has thinking and non-thinking modes; small models have reasoning disabled by default and require chat template kwargs to enable.
- Architecture features: hybrid_thinking, thinking_disabled_by_default_for_small, tool_calling, long_context, multimodal_projector_external.
- Quantization/runtime notes: 0.8B/2B 4-bit around 3.5 GB, 4B around 5.5 GB, 9B around 6.5 GB; 27B around 17 GB and 35B-A3B around 22 GB.
- Known tool support: separate multimodal projector files may be required for GGUF compatibility.
- License: TBD from model card before runtime work.
- Source links: supplied planning notes; model card URL TBD.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8B | 0.8B | dense | 256K source capability | text first | first local text-only GGUF candidate | tier_a_local |
| 2B | 2B | dense | 256K source capability | text first | first local text-only GGUF candidate | tier_a_local |
| 4B | 4B | dense | 256K source capability | text first | follow-up local text-only | tier_a_local |
| 9B | 9B | dense | 256K source capability | text first | larger local text-only | tier_a_local |
| 27B | 27B | dense | 256K source capability | text/multimodal family | offload/reduced context | tier_b_partial |
| 35B-A3B | 35B total | A3B | 256K source capability | text/multimodal family | future MoE gate | tier_b_partial |
| 122B-A10B | 122B total | A10B | 256K source capability | text/multimodal family | design-gated | tier_c_design_only |
| 397B-A17B | 397B total | A17B | 256K source capability | text/multimodal family | design-gated | tier_c_design_only |

## Implementation contract
### Loader
Record intended artifact formats and tensor mappings only; no loader claim exists until receipt-backed.

### Tokenizer
Tokenizer source, vocabulary, and fallback status must be explicit in receipts.

### Prompt template
Prompt template behavior must be recorded before any smoke test.

### Architecture module
Architecture features are design notes until implemented and parity-tested.

### Kernels/backend
Backend selection and fallback must be recorded; no speedup claim is allowed from docs.

### Receipts
Use model receipt templates and set untested coverage fields to false.

### Tests
First proof is a strict deterministic smoke only, not quality or performance evidence.

## Explicit non-claims
- Qwen3.5 small-model docs do not imply a loader exists.
- A 0.8B or 2B text-only proof does not imply multimodal projector support.
- 27B, 35B-A3B, 122B-A10B, and 397B-A17B are larger or design-gated until receipt-backed.

## First proof target

```json
{
  "claim": "qwen3.5_small_text_only_one_token",
  "model_family": "qwen3.5",
  "variant": "0.8b_or_2b",
  "task": "text_generation",
  "context_requested": 2048,
  "full_context_claim": false,
  "multimodal_claim": false,
  "thinking_enabled": false,
  "fallback_used": false,
  "speedup_claim": false
}
```

## Work items
- QWEN35-DOC-001: Add Qwen3.5 family doc.
- QWEN35-DOC-002: Add small-model-first implementation plan for 0.8B/2B/4B/9B.
- QWEN35-DOC-003: Add thinking-mode prompt contract with small-model default disabled.
- QWEN35-DOC-004: Add 27B/35B-A3B future-gated plan.
- QWEN35-DOC-005: Add multimodal projector future gate.
- QWEN35-DOC-006: Add first local proof template for 0.8B or 2B.

