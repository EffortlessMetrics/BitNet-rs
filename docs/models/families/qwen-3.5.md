# Qwen 3.5

## Status
- Repo status: design_scaffold
- First target: qwen3.5-0.8b-or-2b-text-only-gguf
- Local test tier: tier_a_local
- Implementation owner lane: dense_decoder, prompt_template, tokenizer, gguf_dense_quant, moe_decoder_future
- Design-only? no for small text-only targets; yes for very large variants locally

## Source-backed facts
- Model variants: 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B.
- Context: 256K context and 201 languages in supplied notes.
- Modalities: Causal LM with vision encoder in fine-tuning context; GGUF multimodal projector files may be separate.
- Tokenizer/prompt: Hybrid reasoning; small models have reasoning disabled by default and require chat-template kwargs to enable thinking.
- Architecture features: hybrid_thinking, thinking_disabled_by_default_for_small, tool_calling, long_context, multimodal_projector_external.
- Quantization/runtime notes: 4-bit estimates: 0.8B/2B ~3.5 GB, 4B ~5.5 GB, 9B ~6.5 GB, 27B ~17 GB, 35B-A3B ~22 GB.
- License: TBD from model card.
- Known tool support: Best early non-BitNet local target because small models fit the local tier.
- Source links: Supplied planning notes in this change; model-card URLs must be verified before advancing beyond `documented` or `design_scaffold`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---:|---:|---|---|---|---|
| 0.8B | 0.8B | N/A | 256K source note | text; multimodal TBD | first local text-only | tier_a_local |
| 2B | 2B | N/A | 256K source note | text; multimodal TBD | first local text-only | tier_a_local |
| 4B | 4B | N/A | 256K source note | text; multimodal TBD | local text-only | tier_a_local |
| 9B | 9B | N/A | 256K source note | text; multimodal TBD | local text-only | tier_a_local |
| 27B | 27B | N/A | 256K | text/multimodal TBD | offload/future | tier_b_partial |
| 35B-A3B | 35B | 3B active TBD | 256K | text/multimodal TBD | MoE future gate | tier_b_partial |
| 122B-A10B | 122B | 10B active TBD | 256K | text/multimodal TBD | design gate | tier_c_design_only |
| 397B-A17B | 397B | 17B active TBD | 256K | text/multimodal TBD | design gate | tier_c_design_only |

## Implementation contract
### Loader
Start with GGUF dense quant for 0.8B or 2B text-only. Projector and MoE artifacts are future-gated.

### Tokenizer
Explicit Qwen3.5 tokenizer. No fallback for support claims.

### Prompt template
Small-model receipts default `thinking_enabled=false`; enabling thinking must record chat-template kwargs.

### Architecture module
Multimodal decoder family with dense small models and future MoE gates.

### Kernels/backend
Non-BitNet dense path; QK256 proof is not evidence.

### Receipts
Use one-token proof with 2048 context request and no full-context/multimodal/speedup claim.

### Tests
Future: tokenizer/template tests and deterministic one-token smoke for 0.8B or 2B.

## Explicit non-claims
- Qwen3.5 docs do not mean models load.
- Small-model one-token proof does not prove 256K context.
- Thinking mode remains disabled unless a receipt explicitly enables it.
- Multimodal projector and MoE support are future-gated.

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
