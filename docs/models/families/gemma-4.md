# Gemma 4

## Status
- Repo status: design_scaffold
- First target: gemma4-e2b-it-q4-text-only
- Local test tier: tier_a_local
- Implementation owner lane: dense_decoder, moe_decoder_future, multimodal_future, gguf_dense_quant
- Design-only? no

## Source-backed facts
- Model variants: E2B, E4B, 31B dense, 26B A4B MoE.
- Context: E2B/E4B 128K; 31B and 26B-A4B 256K.
- Modalities: E2B/E4B text/image/audio; 31B and 26B-A4B text/image.
- Tokenizer/prompt: vocabulary 262K; thinking mode uses `<|think|>` and thought-channel delimiters.
- Architecture features: hybrid local/global attention, final layer global, unified K/V and p-RoPE in global layers, PLE in small models.
- Quantization/runtime notes: Q4 estimates E2B ~3.2 GB, E4B ~5 GB, 31B ~17.4 GB, 26B-A4B ~15.6 GB, excluding KV/software overhead.
- License: source card verification required before redistribution claims.
- Known tool support: MTP draft models are noted by Google overview and remain future-gated.
- Source links: supplied model-family notes; source-index entries `gemma4-google-overview` and `gemma4-memory-notes`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---|---|---|---|---|---|
| E2B | E2B | TBD | 128K | text/image/audio | text-only Q4 proof | tier_a_local |
| E4B | E4B | TBD | 128K | text/image/audio | text-only Q4 proof | tier_a_local |
| 31B dense | 31B | 31B | 256K | text/image | quantized/offload design | tier_b_partial |
| 26B-A4B | 26B total | A4B; 8 active / 128 total experts plus shared expert | 256K | text/image | MoE future gate | tier_b_partial |

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
- Gemma 4 documented does not mean Gemma 4 loads.
- Generic `ModelArchitecture::Gemma` support and Gemma 2 catalog entries are not Gemma 4 support.
- E2B/E4B text-only proof does not mean image/audio works.
- MTP draft support is future-gated.
- 26B-A4B MoE support is future-gated.

## First proof target

```json
{
  "claim": "gemma4_e2b_text_only_one_token_plan",
  "full_inference_claim": false,
  "multimodal_claim": false,
  "moe_claim": false,
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
