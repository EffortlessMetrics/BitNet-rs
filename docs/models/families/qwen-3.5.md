# Qwen 3.5

## Status
- Repo status: design_scaffold.
- First target: qwen3.5-0.8b-or-2b-text-only-gguf.
- Local test tier: tier_a_local for 0.8B/2B/4B/9B; tier_b_partial for 27B/35B-A3B; tier_c_design_only for 122B/397B.
- Implementation owner lane: dense_decoder, prompt_template, tokenizer, gguf_dense_quant, moe_decoder_future.
- Design-only? no for small text-only targets; yes for projector, larger MoE, and full context until receipts exist.

## Source-backed facts
- Model variants: 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, and 397B-A17B.
- Context: 256K context and 201 languages.
- Modalities: causal LM with vision encoder notes in fine-tuning context; projector compatibility is future-gated.
- Tokenizer/prompt: hybrid reasoning; small-model reasoning disabled by default, and enabling thinking requires chat-template kwargs.
- Architecture features: hybrid_thinking, thinking_disabled_by_default_for_small, tool_calling, long_context, multimodal_projector_external.
- Quantization/runtime notes: 0.8B/2B 4-bit around 3.5 GB, 4B around 5.5 GB, 9B around 6.5 GB, 27B around 17 GB, and 35B-A3B around 22 GB.
- License: TBD from model card.
- Known tool support: separate multimodal projector files may be required for GGUF compatibility.
- Source links: Source links are tracked in `docs/tracking/model-family-foundation/source-index.yaml`; supplied notes must be verified against model cards before implementation claims advance.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8B | 0.8B | dense | 256K source capability | text-first | first local proof | tier_a_local |
| 2B | 2B | dense | 256K source capability | text-first | first local proof | tier_a_local |
| 4B | 4B | dense | 256K source capability | text-first | follow-up local proof | tier_a_local |
| 9B | 9B | dense | 256K source capability | text-first | follow-up local proof | tier_a_local |
| 27B | 27B | dense | 256K source capability | text/projector TBD | offload future gate | tier_b_partial |
| 35B-A3B | 35B | 3B active | 256K source capability | text/projector TBD | MoE future gate | tier_b_partial |
| 122B-A10B | 122B | 10B active | 256K source capability | TBD | design gate | tier_c_design_only |
| 397B-A17B | 397B | 17B active | 256K source capability | TBD | design gate | tier_c_design_only |

## Implementation contract
### Loader
Prioritize small dense GGUF variants before any large/MoE path.
### Tokenizer
Tokenizer source and hash must be explicit before proof.
### Prompt template
Small-model receipts must record `thinking_enabled=false` unless kwargs explicitly enable it.
### Architecture module
Dense small models are first; MoE variants are future-gated.
### Kernels/backend
Non-BitNet dense path must not use QK256 proof.
### Receipts
Small proof uses one-token text generation with no speed, multimodal, or full-context claim.
### Tests
Use deterministic greedy generation at context 2048 first.

## Explicit non-claims
- Qwen3.5 small proof does not prove 27B, 35B-A3B, 122B, or 397B.
- 256K context is not a bitnet-rs runtime claim.
- Multimodal projector support is not implemented until receipt-backed.

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
