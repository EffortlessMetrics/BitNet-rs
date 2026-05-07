# Gemma 4

## Status
- Repo status: design_scaffold
- First target: gemma4-e2b-it-q4-text-only
- Local test tier: tier_a_local
- Implementation owner lane: dense_decoder, moe_decoder_future, multimodal_future, gguf_dense_quant
- Design-only? no

## Source-backed facts
- Model variants: E2B, E4B, 31B dense, and 26B A4B MoE.
- Context: E2B/E4B use 128K; 31B/26B-A4B use 256K.
- Modalities: E2B/E4B text/image/audio; 31B/26B-A4B text/image.
- Tokenizer/prompt: vocabulary is 262K; thinking mode uses `<|think|>` and thought-channel delimiters.
- Architecture features: hybrid local/global attention; final layer global; global layers use unified K/V and p-RoPE; small models use PLE.
- MoE notes: 26B A4B has 8 active and 128 total experts plus a shared expert.
- Quantization/runtime notes: Q4 memory estimates are E2B ~3.2 GB, E4B ~5 GB, 31B ~17.4 GB, 26B-A4B ~15.6 GB, excluding KV/software overhead.
- Known tool support: MTP draft models are noted in Google overview and are future-gated here.
- License: TBD from model card before runtime work.
- Source links: supplied planning notes; model card URL TBD.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| E2B | ~2B effective | dense | 128K | text/image/audio | E2B IT Q4 text-only | tier_a_local |
| E4B | ~4B effective | dense | 128K | text/image/audio | E4B IT Q4 text-only after E2B | tier_a_local |
| 31B | 31B | dense | 256K | text/image | quantized/offload design | tier_b_partial |
| 26B-A4B | 26B total | A4B; 8/128 experts + shared | 256K | text/image | future MoE gate | tier_b_partial |

## Implementation contract
### Loader
Record intended artifact formats and tensor mappings only; no loader claim exists until receipt-backed.

### Tokenizer
Tokenizer source, vocabulary, and fallback status must be explicit in receipts.

### Prompt template
Prompt template behavior must be recorded before any smoke test.

### Architecture module
Track per_layer_embeddings, sliding_window_attention, global_attention, shared_kv_cache, p_rope, thinking_mode, and mtp_drafter_future. Generic `ModelArchitecture::Gemma` defaults are not a Gemma 4 implementation and must not overload the existing Gemma 2 catalog entry.

### Kernels/backend
Backend selection and fallback must be recorded; no speedup claim is allowed from docs.

### Receipts
Use model receipt templates and set untested coverage fields to false.

### Tests
First proof is a strict deterministic smoke only, not quality or performance evidence.

## Explicit non-claims
- Gemma 4 documented does not mean Gemma 4 loads.
- E2B/E4B text-only proof does not mean image/audio works.
- MTP draft support is future-gated.
- 26B-A4B MoE support is future-gated.

## First proof target

```json
{
  "claim": "gemma4_e2b_text_only_one_token",
  "model_family": "gemma4",
  "variant": "e2b",
  "task": "text_generation",
  "multimodal_claim": false,
  "moe_claim": false,
  "fallback_used": false,
  "speedup_claim": false
}
```

## Work items
- GEMMA4-DOC-001: Add Gemma 4 family doc and variant table.
- GEMMA4-DOC-002: Add Gemma 4 prompt/template contract.
- GEMMA4-DOC-003: Add Gemma 4 text-only E2B proof plan.
- GEMMA4-DOC-004: Add PLE/shared-KV/sliding-global attention implementation notes.
- GEMMA4-DOC-005: Add MoE 26B-A4B future gate.
- GEMMA4-DOC-006: Add multimodal future gate.

