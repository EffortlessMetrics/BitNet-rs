# Gemma 4

## Status
- Repo status: design_scaffold.
- First target: gemma4-e2b-it-q4-text-only.
- Local test tier: tier_a_local for E2B/E4B text-only; tier_b_partial for 31B and 26B-A4B quantized/offload.
- Implementation owner lane: dense_decoder, gguf_dense_quant, moe_decoder_future, multimodal_future.
- Design-only? no for text-only E2B/E4B planning; yes for multimodal, MTP, and MoE until receipts exist.

## Source-backed facts
- Model variants: E2B, E4B, 31B dense, and 26B A4B MoE.
- Context: E2B/E4B have 128K context; 31B and 26B-A4B have 256K context.
- Modalities: E2B/E4B support text/image/audio; 31B and 26B-A4B support text/image.
- Tokenizer/prompt: vocabulary is 262K; thinking mode uses `<|think|>` and thought-channel delimiters.
- Architecture features: hybrid local/global attention, final layer global attention, unified K/V and p-RoPE on global layers, PLE on small models, and MTP draft models.
- Quantization/runtime notes: Q4 estimates are E2B ~3.2 GB, E4B ~5 GB, 31B ~17.4 GB, and 26B-A4B ~15.6 GB, not including KV/software overhead.
- License: TBD from model card.
- Known tool support: TBD; do not overload existing Gemma 2 catalog entries.
- Source links: Source links are tracked in `docs/tracking/model-family-foundation/source-index.yaml`; supplied notes must be verified against model cards before implementation claims advance.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| E2B | ~2B effective | dense | 128K | text/image/audio | text-only Q4 smoke | tier_a_local |
| E4B | ~4B effective | dense | 128K | text/image/audio | text-only Q4 smoke | tier_a_local |
| 31B | 31B | dense | 256K | text/image | offload/design | tier_b_partial |
| 26B-A4B | 26B total | 4B active; 8 active / 128 total experts plus shared expert | 256K | text/image | MoE future gate | tier_b_partial |

## Implementation contract
### Loader
Create a separate Gemma 4 catalog/loader mapping; generic `ModelArchitecture::Gemma` and Gemma 2 defaults are not Gemma 4 support.
### Tokenizer
Record 262K vocabulary source and tokenizer hash before proof.
### Prompt template
Record thinking delimiters and whether thinking is enabled.
### Architecture module
Plan dense E2B/E4B/31B separately from 26B-A4B MoE; PLE/shared-KV/p-RoPE are future implementation notes.
### Kernels/backend
Do not route through BitNet QK256 kernels as evidence.
### Receipts
First receipt must be text-only, one-token, no multimodal, no MoE, no MTP, no speed claim.
### Tests
Start with strict deterministic one-token text generation at reduced context.

## Explicit non-claims
- Gemma 4 documented does not mean Gemma 4 loads.
- E2B/E4B text-only proof does not mean image/audio works.
- MTP draft support is future-gated.
- 26B-A4B MoE support is future-gated.

## First proof target
Use `ci/model-receipts/_templates/generative-one-token-proof.json` with `model_family=gemma4`, `variant=e2b`, `multimodal_claim=false`, `moe_claim=false`, and `speedup_claim=false`.

## Work items
- GEMMA4-DOC-001: Add Gemma 4 family doc and variant table.
- GEMMA4-DOC-002: Add Gemma 4 prompt/template contract.
- GEMMA4-DOC-003: Add Gemma 4 text-only E2B proof plan.
- GEMMA4-DOC-004: Add PLE/shared-KV/sliding-global attention implementation notes.
- GEMMA4-DOC-005: Add MoE 26B-A4B future gate.
- GEMMA4-DOC-006: Add multimodal future gate.
