# GLM-5.1

## Status
- Repo status: design_only
- First target: glm-5.1_notes
- Local test tier: tier_c_design_only
- Implementation owner lane: design lanes and external_reference
- Design-only? yes

## Source-backed facts
- Model variants: GLM-5.1 full model.
- Context: 200K.
- Modalities: text/tooling per supplied notes; other modalities TBD.
- Tokenizer/prompt: thinking enabled by default; disable with chat-template kwargs; `chat_template.jinja` changes focus on tool exposure, reasoning-history reconstruction, and tool-message rendering.
- Architecture features: same architecture as GLM-5 per supplied notes, huge MoE, tool calling, long context.
- Quantization/runtime notes: 744B parameters, 40B active, full disk around 1.65 TB; dynamic 2-bit around 220 GB, dynamic 1-bit around 200 GB, `UD-IQ2_M` around 236 GB.
- License: source card verification required.
- Known tool support: future OpenVINO/llama.cpp reference receipt plan only.
- Source links: supplied GLM-5.1 notes; source-index `glm51-model-notes`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---|---|---|---|---|---|
| GLM-5.1 | 744B | 40B | 200K | text/tool TBD | design-only prompt/template and MoE notes | tier_c_design_only |

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
  "model_family": "glm-5.1",
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
