# DeepSeek V4 Flash and Pro

## Status
- Repo status: design_only
- First target: design_only_architecture_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: moe_decoder_design, long_context_design, mixed_precision_design
- Design-only? yes

## Source-backed facts
- Variants: DeepSeek V4 Flash and DeepSeek V4 Pro, each with Base and instruct variants.
- DeepSeek V4 Flash: 284B total, 13B active, 1M context.
- DeepSeek V4 Pro: 1.6T total, 49B active, 1M context.
- Architecture features: MoE, compressed sparse attention (CSA), heavily compressed attention (HCA), mHC residual/hyper-connection mechanism, and Muon optimizer during training.
- Precision: Base uses FP8 mixed; instruct versions use FP4 + FP8 mixed, with MoE experts in FP4 and most other parameters in FP8 according to supplied model card notes.
- License: TBD from model card before runtime work.
- Known tool support: mixed-precision and huge-MoE reference only.
- Source links: supplied planning notes; model card URL TBD.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| V4 Flash Base/Instruct | 284B | 13B | 1M | text | design-only architecture notes | tier_c_design_only |
| V4 Pro Base/Instruct | 1.6T | 49B | 1M | text | design-only architecture notes | tier_c_design_only |

## Implementation contract
### Loader
Record intended formats and external-reference commands only; no loader claim exists until receipt-backed.

### Tokenizer
Tokenizer source and fallback status must be explicit. Unknown tokenizer details remain `TBD`.

### Prompt template
Prompt template behavior is documented as a contract and must be receipt-backed before any support claim.

### Architecture module
Architecture notes are design rails until implemented, smoke-tested, parity-tested where applicable, and receipt-backed.

### Kernels/backend
No BitNet QK256, mixed-precision, long-context, multimodal, or speedup claim is allowed from documentation.

### Receipts
Use design-only or external-reference templates until local proof is feasible.

### Tests
No local runtime tests are implied by this document.

## Explicit non-claims
- DeepSeek V4 Flash/Pro docs do not mean FP4/FP8 mixed kernels exist.
- 1M context is a source-backed model capability, not a bitnet-rs runtime claim.
- CSA/HCA/mHC are architecture notes until implemented and parity-tested.

## First proof target

```json
{
  "claim": "design_only",
  "model_family": "deepseek-v4",
  "variant": "flash_or_pro",
  "local_execution_claim": false,
  "loader_claim": false,
  "inference_claim": false,
  "speedup_claim": false
}
```

## Work items
- DSV4-DOC-001: Add DeepSeek V4 family doc.
- DSV4-DOC-002: Add Flash vs Pro variant table.
- DSV4-DOC-003: Add CSA/HCA/mHC terminology and implementation unknowns.
- DSV4-DOC-004: Add FP4/FP8 mixed precision future-gate.
- DSV4-DOC-005: Add design-only local hardware warning.

