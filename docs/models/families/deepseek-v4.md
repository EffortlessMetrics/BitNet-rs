# DeepSeek V4

## Status
- Repo status: design_only
- First target: design_only_architecture_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: prompt_template, external_reference, design lane
- Design-only? yes

## Source-backed facts
- Model variants: Flash 284B total / 13B active; Pro 1.6T total / 49B active; Base and instruct variants.
- Context: 1M.
- Modalities: Text; other modalities TBD.
- Tokenizer/prompt: Prompt template TBD from model cards before implementation.
- Architecture features: compressed_sparse_attention, heavily_compressed_attention, manifold_constrained_hyper_connections, million_token_context, fp4_fp8_mixed, huge_moe.
- Quantization/runtime notes: Base uses FP8 mixed; instruct versions use FP4 + FP8 mixed, with MoE experts in FP4 and most other params in FP8 according to supplied notes.
- License: TBD from model card.
- Known tool support: External/offload/reference only until receipt-backed.
- Source links: Supplied planning notes in this change; model-card URLs must be verified before advancing beyond `documented` or `design_scaffold`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---:|---:|---|---|---|---|
| V4 Flash Base/Instruct | 284B | 13B | 1M | text | design-only notes | tier_c_design_only |
| V4 Pro Base/Instruct | 1.6T | 49B | 1M | text | design-only notes | tier_c_design_only |

## Implementation contract
### Loader
Design-only loader notes. No local loader, inference, or speed claim is allowed.

### Tokenizer
Record tokenizer source and template source before any support claim.

### Prompt template
Record family-specific prompt controls and disable unsupported fallbacks.

### Architecture module
Architecture notes only; implementation details remain TBD until source-backed and parity-tested.

### Kernels/backend
No local kernel claim. Non-BitNet QK256/BitNet evidence is invalid.

### Receipts
Use `generative-design-only.json` or `external-reference-proof.json` until narrow execution receipts exist.

### Tests
YAML/JSON parse only now; future external reference command receipts.

## Explicit non-claims
- DeepSeek V4 is design-only on current hardware.
- Any future local artifact is offload/reference unless receipt-backed.
- No speed, quality, long-context, or full-inference claim is allowed from docs.

## First proof target
Design-only receipt with local execution, loader, inference, and speedup claims all false.

## Work items
- DSV4-DOC-001: Add DeepSeek V4 family doc.
- DSV4-DOC-002: Add prompt/template and architecture notes.
- DSV4-DOC-003: Add hardware infeasibility/local non-claim section.
- DSV4-DOC-004: Add external-reference receipt plan.
