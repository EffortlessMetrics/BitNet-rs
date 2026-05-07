# GLM-5.1

## Status
- Repo status: design_only
- First target: design_only_prompt_template_and_moe_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: prompt_template, external_reference, design lane
- Design-only? yes

## Source-backed facts
- Model variants: Full model is 744B parameters with 40B active.
- Context: 200K.
- Modalities: Text/tools; other modalities TBD.
- Tokenizer/prompt: Thinking enabled by default; disable with chat-template kwargs; GLM-5.1 uses same architecture as GLM-5 with `chat_template.jinja` changes for tool exposure, reasoning-history reconstruction, and tool-message rendering.
- Architecture features: thinking_enabled_by_default, long_context, tool_calling, huge_moe.
- Quantization/runtime notes: Full disk ~1.65 TB; dynamic 2-bit ~220 GB; dynamic 1-bit ~200 GB; `UD-IQ2_M` ~236 GB.
- License: TBD from model card.
- Known tool support: External/offload/reference only until receipt-backed.
- Source links: Supplied planning notes in this change; model-card URLs must be verified before advancing beyond `documented` or `design_scaffold`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---:|---:|---|---|---|---|
| GLM-5.1 | 744B | 40B | 200K | text/tools | design-only notes | tier_c_design_only |

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
- GLM-5.1 is design-only on current hardware.
- Any future local artifact is offload/reference unless receipt-backed.
- No speed, quality, long-context, or full-inference claim is allowed from docs.

## First proof target
Design-only receipt with local execution, loader, inference, and speedup claims all false.

## Work items
- GLM51-DOC-001: Add GLM-5.1 family doc.
- GLM51-DOC-002: Add prompt/template and architecture notes.
- GLM51-DOC-003: Add hardware infeasibility/local non-claim section.
- GLM51-DOC-004: Add external-reference receipt plan.
