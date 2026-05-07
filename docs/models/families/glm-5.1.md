# GLM 5.1

## Status
- Repo status: design_only.
- First target: design_only_prompt_template_and_moe_notes.
- Local test tier: tier_c_design_only.
- Implementation owner lane: moe_decoder_design, prompt_template, tool_calling, external_reference.
- Design-only? yes.

## Source-backed facts
- Model variants: GLM-5.1 full model.
- Context: 200K.
- Modalities: text/tooling focus in supplied notes; multimodal TBD.
- Tokenizer/prompt: thinking enabled by default and disabled with chat-template kwargs; `chat_template.jinja` changes focus on tool exposure, reasoning-history reconstruction, and tool-message rendering.
- Architecture features: same architecture as GLM-5 in supplied notes, huge_moe, tool_calling, long_context.
- Quantization/runtime notes: 744B total parameters, 40B active; full disk size around 1.65 TB; dynamic 2-bit around 220 GB, dynamic 1-bit around 200 GB, and `UD-IQ2_M` around 236 GB.
- License: TBD from model card.
- Known tool support: future OpenVINO/llama.cpp reference receipt plan.
- Source links: Source links are tracked in `docs/tracking/model-family-foundation/source-index.yaml`; supplied notes must be verified against model cards before implementation claims advance.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| GLM-5.1 | 744B | 40B | 200K | text/tools TBD | design-only prompt/MoE notes | tier_c_design_only |

## Implementation contract
### Loader
External-reference or future graph loader notes only.
### Tokenizer
Tokenizer and template source must be recorded before any proof.
### Prompt template
Record thinking default and disable kwargs.
### Architecture module
Huge MoE details are design-only until source and routing proofs exist.
### Kernels/backend
No local full-model kernel claim.
### Receipts
Use design-only or external-reference receipt.
### Tests
No local inference test is planned in first pass.

## Explicit non-claims
- GLM-5.1 docs do not imply local loading or inference.
- Thinking/tool template notes do not imply prompt parity.
- Quantized disk sizes are not VRAM fit or performance claims.

## First proof target
Design-only receipt for prompt-template and MoE notes; future external reference may cite OpenVINO or llama.cpp.

## Work items
- GLM51-DOC-001: Add GLM-5.1 design-only family doc.
- GLM51-DOC-002: Add thinking/tool/chat-template behavior notes.
- GLM51-DOC-003: Add model-size/hardware non-claim policy.
- GLM51-DOC-004: Add future OpenVINO/llama.cpp reference receipt plan.
