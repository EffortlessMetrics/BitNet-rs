# Qwen 3.6

## Status
- Repo status: design_scaffold
- First target: qwen3.6-27b-text-only-gguf-design
- Local test tier: tier_b_partial
- Implementation owner lane: dense_decoder, moe_decoder_for_35b_a3b, prompt_template, tool_calling
- Design-only? no

## Source-backed facts
- Model variants: 27B and 35B-A3B.
- Context: 256K maximum 262,144, extendable to 1M via YaRN in supplied notes.
- Modalities: multimodal hybrid-thinking.
- Tokenizer/prompt: developer role for agentic coding tools; thinking and non-thinking modes have different recommended sampling settings.
- Architecture features: tool calling, improved nested tool-call parsing, long context, multimodal projector external flows.
- Quantization/runtime notes: Unsloth notes 27B 4-bit around 18 GB total memory and 35B-A3B around 23 GB.
- License: source card verification required.
- Known tool support: Some GGUF flows require separate multimodal projector files; do not assume Ollama compatibility.
- Source links: supplied Qwen3.6 notes; source-index entries `qwen36-model-notes` and `qwen36-unsloth-memory`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---|---|---|---|---|---|
| 27B | 27B | 27B | 262,144 / YaRN to 1M design | text/image TBD | text-only reduced-context design | tier_b_partial |
| 35B-A3B | 35B | A3B | 262,144 / YaRN to 1M design | text/image TBD | MoE future gate | tier_b_partial |

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
- Qwen3.6 docs do not imply 256K context works in bitnet-rs.
- 27B offload/design does not imply full 5070Ti residency.
- 35B-A3B MoE is not implemented until expert routing is receipt-backed.

## First proof target

```json
{
  "claim": "qwen3.6_27b_text_only_design",
  "context_requested": 2048,
  "full_context_claim": false,
  "multimodal_claim": false,
  "moe_claim": false,
  "speedup_claim": false
}
```

## Work items
- QWEN36-DOC-001: Add Qwen3.6 family doc.
- QWEN36-DOC-002: Add Qwen3.6 prompt/template settings for thinking and non-thinking.
- QWEN36-DOC-003: Add 27B text-only reduced-context proof plan.
- QWEN36-DOC-004: Add 35B-A3B MoE future gate.
- QWEN36-DOC-005: Add multimodal projector future gate.
