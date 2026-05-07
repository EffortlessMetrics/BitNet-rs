# Mistral Medium 3.5

## Status
- Repo status: design_only
- First target: design_only_prompt_reasoning_effort_contract
- Local test tier: tier_c_design_only
- Implementation owner lane: prompt_template, external_reference, design lane
- Design-only? yes

## Source-backed facts
- Model variants: Dense 128B.
- Context: 256K.
- Modalities: Multimodal text + image input, text output.
- Tokenizer/prompt: Reasoning effort configurable per request: `none` or `high`; function calls / JSON / agentic use.
- Architecture features: reasoning_effort, multimodal_text_image, function_calling, long_context, eagle_drafter_future.
- Quantization/runtime notes: vLLM recommended with tensor parallelism; local full path is beyond current hardware; EAGLE draft model exists.
- License: TBD from model card; Mistral Medium 3.5 notes say Modified MIT, not plain MIT.
- Known tool support: External/offload/reference only until receipt-backed.
- Source links: Supplied planning notes in this change; model-card URLs must be verified before advancing beyond `documented` or `design_scaffold`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---:|---:|---|---|---|---|
| Medium 3.5 | 128B | N/A | 256K | text+image input, text output | external/design-only | tier_c_design_only |

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
- Mistral Medium 3.5 is design-only on current hardware.
- Any future local artifact is offload/reference unless receipt-backed.
- No speed, quality, long-context, or full-inference claim is allowed from docs.

## First proof target
Design-only receipt with local execution, loader, inference, and speedup claims all false.

## Work items
- MM35-DOC-001: Add Mistral Medium 3.5 family doc.
- MM35-DOC-002: Add prompt/template and architecture notes.
- MM35-DOC-003: Add hardware infeasibility/local non-claim section.
- MM35-DOC-004: Add external-reference receipt plan.
