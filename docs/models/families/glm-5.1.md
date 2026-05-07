# GLM-5.1

## Status
- Repo status: design_only
- First target: design_only_prompt_template_and_moe_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: moe_decoder_design, prompt_template, tool_calling, external_reference
- Design-only? yes

## Source-backed facts
- Full model: 744B parameters, 40B active.
- Context: 200K.
- Tokenizer/prompt: thinking enabled by default; disable with chat-template kwargs.
- Architecture features: same architecture as GLM-5 per supplied notes; chat_template.jinja changes focus on tool exposure, reasoning-history reconstruction, and tool-message rendering.
- Quantization/runtime notes: full disk size around 1.65 TB; dynamic 2-bit around 220 GB; dynamic 1-bit around 200 GB; `UD-IQ2_M` around 236 GB.
- License: TBD from model card before runtime work.
- Known tool support: future OpenVINO/llama.cpp reference receipt only.
- Source links: supplied planning notes; model card URL TBD.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| GLM-5.1 | 744B | 40B | 200K | text/tooling | design-only prompt/MoE notes | tier_c_design_only |

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
- GLM-5.1 is design-only on current hardware.
- Thinking/tool-template notes do not imply prompt-template implementation.
- No speed, quality, local execution, long-context, or full-inference claim is allowed from docs.

## First proof target

```json
{
  "claim": "design_only",
  "model_family": "glm-5.1",
  "variant": "full",
  "local_execution_claim": false,
  "loader_claim": false,
  "inference_claim": false,
  "speedup_claim": false
}
```

## Work items
- GLM51-DOC-001: Add GLM-5.1 design-only family doc.
- GLM51-DOC-002: Add thinking/tool/chat-template behavior notes.
- GLM51-DOC-003: Add model-size/hardware non-claim policy.
- GLM51-DOC-004: Add future OpenVINO/llama.cpp reference receipt plan.

