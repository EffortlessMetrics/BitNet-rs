# Mistral Medium 3.5

## Status
- Repo status: design_only.
- First target: design_only_prompt_reasoning_effort_contract.
- Local test tier: tier_c_design_only for full path; tier_b_partial only if external/offload reference exists.
- Implementation owner lane: dense_decoder_design, prompt_template, tool_calling, speculative_decoding_future.
- Design-only? yes for full local path.

## Source-backed facts
- Model variants: Mistral Medium 3.5 dense 128B.
- Context: 256K.
- Modalities: text + image input, text output.
- Tokenizer/prompt: reasoning effort configurable per request as `none` or `high`; function calls / JSON / agentic use.
- Architecture features: dense_decoder, reasoning_effort, multimodal_text_image, function_calling, long_context, eagle_drafter_future.
- Quantization/runtime notes: vLLM recommended with tensor parallelism; local full path is beyond current hardware.
- License: Modified MIT, not plain MIT.
- Known tool support: EAGLE draft model exists for speculative decoding.
- Source links: Source links are tracked in `docs/tracking/model-family-foundation/source-index.yaml`; supplied notes must be verified against model cards before implementation claims advance.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| Medium 3.5 | 128B dense | dense | 256K | text+image input, text output | design-only reasoning contract | tier_c_design_only |

## Implementation contract
### Loader
Design-only unless an external/offload reference is added.
### Tokenizer
TBD from model card.
### Prompt template
Record reasoning_effort values and tool/function-call structure.
### Architecture module
Dense decoder full path is beyond local 16 GB proof.
### Kernels/backend
No vLLM/tensor-parallel external result counts as native bitnet-rs proof.
### Receipts
Use external-reference receipt for external vLLM runs.
### Tests
No local full-path test is planned in first pass.

## Explicit non-claims
- Mistral Medium 3.5 full local inference is not claimed.
- Modified MIT is not plain MIT.
- EAGLE draft existence is not speculative decoding support.

## First proof target
Design-only prompt/API receipt with `local_execution_claim=false`.

## Work items
- MM35-DOC-001: Add Mistral Medium 3.5 design-only family doc.
- MM35-DOC-002: Add reasoning_effort prompt/API contract.
- MM35-DOC-003: Add EAGLE speculative decoding future-gate.
- MM35-DOC-004: Add license caveat.
- MM35-DOC-005: Add external-reference receipt plan.
