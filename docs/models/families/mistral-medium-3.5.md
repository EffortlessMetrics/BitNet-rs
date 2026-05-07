# Mistral Medium 3.5

## Status
- Repo status: design_only
- First target: mistral-medium-3.5_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: design lanes and external_reference
- Design-only? yes

## Source-backed facts
- Model variants: dense 128B.
- Context: 256K.
- Modalities: text + image input, text output.
- Tokenizer/prompt: reasoning effort configurable per request as `none` or `high`; function calls, JSON, and agentic use.
- Architecture features: dense decoder, multimodal text-image, reasoning effort, function calling, long context, EAGLE drafter future.
- Quantization/runtime notes: vLLM recommended with tensor parallelism; local full path is beyond current hardware.
- License: Modified MIT, not plain MIT.
- Known tool support: EAGLE draft model exists for speculative decoding.
- Source links: supplied Mistral Medium 3.5 notes; source-index `mistral-medium-35-card`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---|---|---|---|---|---|
| Mistral Medium 3.5 | 128B | 128B | 256K | text/image in, text out | design-only prompt reasoning-effort contract | tier_c_design_only |

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
  "model_family": "mistral-medium-3.5",
  "local_execution_claim": false,
  "loader_claim": false,
  "inference_claim": false,
  "speedup_claim": false
}
```

## Work items
- MM35-DOC-001: Add Mistral Medium 3.5 design-only family doc.
- MM35-DOC-002: Add reasoning_effort prompt/API contract.
- MM35-DOC-003: Add EAGLE speculative decoding future-gate.
- MM35-DOC-004: Add license caveat.
- MM35-DOC-005: Add external-reference receipt plan.
