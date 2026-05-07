# Mistral Medium 3.5

## Status
- Repo status: design_only
- First target: design_only_prompt_reasoning_effort_contract
- Local test tier: tier_c_design_only
- Implementation owner lane: dense_decoder_design, prompt_template, tool_calling, speculative_decoding_future
- Design-only? yes

## Source-backed facts
- Model type: dense 128B.
- Context: 256K.
- Modalities: text + image input and text output.
- Tokenizer/prompt: reasoning effort configurable per request as `none` or `high`.
- Architecture features: reasoning_effort, multimodal_text_image, function_calling, long_context, eagle_drafter_future.
- Quantization/runtime notes: vLLM recommended with tensor parallelism; local full path is beyond current hardware.
- License: Modified MIT, not plain MIT.
- Known tool support: EAGLE draft model exists for speculative decoding.
- Source links: supplied planning notes; model card URL TBD.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| Medium 3.5 | 128B | dense | 256K | text/image input; text output | design-only prompt/API contract | tier_c_design_only |

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
- Mistral Medium 3.5 full local path is design-only on current hardware.
- Reasoning-effort docs do not imply an implemented API contract.
- EAGLE speculative decoding is future-gated until receipt-backed.

## First proof target

```json
{
  "claim": "design_only",
  "model_family": "mistral-medium-3.5",
  "variant": "128b",
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

