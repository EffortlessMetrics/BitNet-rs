# Qwen 3.5

## Status
- Repo status: design_scaffold
- First target: qwen3.5-0.8b-or-2b-text-only-gguf
- Local test tier: tier_a_local
- Implementation owner lane: dense_decoder, prompt_template, tokenizer, gguf_dense_quant, moe_decoder_future
- Design-only? no

## Source-backed facts
- Model variants: 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, and 397B-A17B.
- Context: 256K and 201 languages.
- Modalities: supplied notes describe a causal LM with a vision encoder in fine-tuning context.
- Tokenizer/prompt: hybrid reasoning with thinking and non-thinking modes; small models disable reasoning by default and require chat-template kwargs to enable thinking.
- Architecture features: hybrid thinking, tool calling, long context, multimodal projector external.
- Quantization/runtime notes: 0.8B/2B 4-bit around 3.5 GB, 4B around 5.5 GB, 9B around 6.5 GB; 27B around 17 GB and 35B-A3B around 22 GB.
- License: source card verification required.
- Known tool support: Separate multimodal projector files may be required for GGUF compatibility.
- Source links: supplied Qwen3.5 notes; source-index entries `qwen35-model-notes` and `qwen35-hardware-table`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---|---|---|---|---|---|
| 0.8B | 0.8B | 0.8B | 256K source capability | text | first local GGUF proof | tier_a_local |
| 2B | 2B | 2B | 256K source capability | text | first local GGUF proof | tier_a_local |
| 4B | 4B | 4B | 256K source capability | text | local GGUF proof after 0.8B/2B | tier_a_local |
| 9B | 9B | 9B | 256K source capability | text | local GGUF proof after smaller models | tier_a_local |
| 27B | 27B | 27B | 256K source capability | text/image TBD | offload/future gate | tier_b_partial |
| 35B-A3B | 35B | A3B | 256K source capability | text/image TBD | MoE future gate | tier_b_partial |
| 122B-A10B | 122B | A10B | 256K source capability | TBD | design-gated | tier_c_design_only |
| 397B-A17B | 397B | A17B | 256K source capability | TBD | design-gated | tier_c_design_only |

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
- Qwen3.5 small-model text-only proof does not imply 256K context works.
- Thinking remains disabled by default for small-model proof unless the receipt explicitly enables it.
- Multimodal projector support is future-gated.
- 27B/35B/122B/397B are larger or design-gated targets.

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
