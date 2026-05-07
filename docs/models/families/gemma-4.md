# Gemma 4

## Status
- Repo status: design_scaffold
- First target: gemma4-e2b-it-q4-text-only
- Local test tier: tier_a_local
- Implementation owner lane: dense_decoder, gguf_dense_quant, moe_decoder_future, multimodal_future
- Design-only? no for E2B/E4B text-only planning; yes for unproven multimodal/MoE/MTP claims

## Source-backed facts
- Model variants: E2B, E4B, 31B dense, 26B A4B MoE.
- Context: E2B/E4B 128K; 31B/26B-A4B 256K.
- Modalities: E2B/E4B text/image/audio; 31B/26B-A4B text/image.
- Tokenizer/prompt: Vocabulary 262K; thinking mode uses `<|think|>` and thought-channel delimiters.
- Architecture features: PLE on small models; hybrid local/global attention; final layer global; global layers use unified K/V and p-RoPE; 26B A4B has 8 active / 128 total experts plus shared expert; all models have MTP draft models in supplied overview.
- Quantization/runtime notes: Q4 estimates: E2B ~3.2 GB, E4B ~5 GB, 31B ~17.4 GB, 26B-A4B ~15.6 GB, excluding KV/software overhead.
- License: TBD from model card before support claims.
- Known tool support: Generic `ModelArchitecture::Gemma` detector and Gemma-family defaults exist, but they are not Gemma 4 support; Gemma 2 catalog entries must remain separate.
- Source links: Supplied planning notes in this change; model-card URLs must be verified before advancing beyond `documented` or `design_scaffold`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---:|---:|---|---|---|---|
| E2B | 2B effective | N/A | 128K | text/image/audio | text-only Q4 smoke | tier_a_local |
| E4B | 4B effective | N/A | 128K | text/image/audio | text-only Q4 smoke | tier_a_local |
| 31B dense | 31B | N/A | 256K | text/image | quantized/offload design | tier_b_partial |
| 26B-A4B | 26B | 4B active / 8 experts active | 256K | text/image | MoE future gate | tier_b_partial |

## Implementation contract
### Loader
Start with text-only GGUF dense quant for E2B/E4B. MoE, multimodal, and MTP artifacts require separate loader contracts.

### Tokenizer
Use explicit Gemma 4 tokenizer source; no fallback tokenizer can support a claim.

### Prompt template
Record `<|think|>` and thought-channel delimiter behavior. Text-only smoke should set thinking according to receipt.

### Architecture module
Plan multimodal decoder with dense E2B/E4B/31B path and future 26B-A4B MoE gate.

### Kernels/backend
Non-BitNet dense path; no QK256 or BitNet W1.58 proof may count.

### Receipts
First receipt must set multimodal, MoE, long-context, MTP, speedup, and full-inference claims false.

### Tests
Future: tokenizer/template parse, one-token deterministic text smoke, then parity.

## Explicit non-claims
- Gemma 4 documented does not mean Gemma 4 loads.
- E2B/E4B text-only proof does not mean image/audio works.
- MTP draft support is future-gated.
- 26B-A4B MoE support is future-gated.

## First proof target
`generative-one-token-proof.json` with `model_family=gemma4`, `variant=gemma4-e2b-it-q4`, `task=text_generation`, `generated_tokens=1`, and feature claims false.

## Work items
- GEMMA4-DOC-001: Add Gemma 4 family doc and variant table.
- GEMMA4-DOC-002: Add Gemma 4 prompt/template contract.
- GEMMA4-DOC-003: Add Gemma 4 text-only E2B proof plan.
- GEMMA4-DOC-004: Add PLE/shared-KV/sliding-global attention implementation notes.
- GEMMA4-DOC-005: Add MoE 26B-A4B future gate.
- GEMMA4-DOC-006: Add multimodal future gate.
