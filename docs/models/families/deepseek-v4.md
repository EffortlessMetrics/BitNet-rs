# DeepSeek V4

## Status
- Repo status: design_only
- First target: deepseek-v4_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: design lanes and external_reference
- Design-only? yes

## Source-backed facts
- Model variants: DeepSeek V4 Flash and DeepSeek V4 Pro, each with Base and instruct variants.
- Context: 1M source capability for Flash and Pro.
- Modalities: text source facts only in supplied notes; other modalities TBD.
- Tokenizer/prompt: instruct/base prompt specifics TBD pending model-card verification.
- Architecture features: MoE, CSA/HCA hybrid attention, mHC residual/hyper-connection, Muon optimizer during training.
- Quantization/runtime notes: Flash 284B total/13B active; Pro 1.6T total/49B active; Base uses FP8 mixed; instruct uses FP4 + FP8 mixed with MoE experts in FP4 and most other params in FP8.
- License: source card verification required.
- Known tool support: mixed precision and long-context support are future-gated.
- Source links: supplied DeepSeek V4 notes; source-index `deepseek-v4-model-card`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---|---|---|---|---|---|
| DeepSeek V4 Flash Base/Instruct | 284B | 13B | 1M | text TBD | design-only architecture notes | tier_c_design_only |
| DeepSeek V4 Pro Base/Instruct | 1.6T | 49B | 1M | text TBD | design-only architecture notes | tier_c_design_only |

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
- This family is design-only on current hardware.
- Docs do not imply local loader, inference, speed, quality, long-context, multimodal, or kernel support.
- Any future local/offload/reference artifact must be receipt-backed.

## First proof target

```json
{
  "claim": "design_only",
  "model_family": "deepseek-v4",
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
