# DeepSeek V4

## Status
- Repo status: design_only.
- First target: design_only_architecture_notes.
- Local test tier: tier_c_design_only.
- Implementation owner lane: moe_decoder_design, long_context_design, mixed_precision_design.
- Design-only? yes.

## Source-backed facts
- Model variants: DeepSeek V4 Flash and DeepSeek V4 Pro, with Base and instruct variants.
- Context: 1M context.
- Modalities: text focus in supplied notes; multimodal TBD.
- Tokenizer/prompt: TBD from model cards.
- Architecture features: MoE, compressed_sparse_attention, heavily_compressed_attention, manifold-constrained hyper-connections, million-token context, and fp4_fp8_mixed precision.
- Quantization/runtime notes: Flash has 284B total and 13B active; Pro has 1.6T total and 49B active. Base uses FP8 mixed; instruct uses FP4 + FP8 mixed, with MoE experts in FP4 and most other parameters in FP8 according to supplied notes.
- License: TBD from model card.
- Known tool support: TBD.
- Source links: Source links are tracked in `docs/tracking/model-family-foundation/source-index.yaml`; supplied notes must be verified against model cards before implementation claims advance.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| V4 Flash Base/Instruct | 284B | 13B | 1M | text TBD | design-only architecture notes | tier_c_design_only |
| V4 Pro Base/Instruct | 1.6T | 49B | 1M | text TBD | design-only architecture notes | tier_c_design_only |

## Implementation contract
### Loader
Design-only for huge MoE and mixed precision.
### Tokenizer
TBD from model cards.
### Prompt template
TBD from instruct cards.
### Architecture module
CSA/HCA/mHC are terminology notes until implemented and parity-tested.
### Kernels/backend
FP4/FP8 mixed kernels do not exist until separately implemented and receipt-backed.
### Receipts
Use design-only receipt only in first pass.
### Tests
No local inference test is planned in first pass.

## Explicit non-claims
- DeepSeek V4 Flash/Pro docs do not mean FP4/FP8 mixed kernels exist.
- 1M context is a source-backed model capability, not a bitnet-rs runtime claim.
- CSA/HCA/mHC are architecture notes until implemented and parity-tested.

## First proof target
Design-only receipt with architecture notes and no local execution claim.

## Work items
- DSV4-DOC-001: Add DeepSeek V4 family doc.
- DSV4-DOC-002: Add Flash vs Pro variant table.
- DSV4-DOC-003: Add CSA/HCA/mHC terminology and implementation unknowns.
- DSV4-DOC-004: Add FP4/FP8 mixed precision future-gate.
- DSV4-DOC-005: Add design-only local hardware warning.
