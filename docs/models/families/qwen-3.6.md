# Qwen 3.6

## Status
- Repo status: design_scaffold
- First target: qwen3.6-27b-text-only-gguf-design
- Local test tier: tier_b_partial
- Implementation owner lane: dense_decoder, moe_decoder_for_35b_a3b, prompt_template, tool_calling
- Design-only? no

## Source-backed facts
- Model variants: 27B and 35B-A3B.
- Context: maximum context 262,144 with supplied notes saying extendable to 1M via YaRN.
- Modalities: multimodal hybrid-thinking family.
- Tokenizer/prompt: thinking and non-thinking modes have different recommended sampling settings.
- Architecture features: hybrid_thinking, tool_calling, developer_role, long_context, yarn_future, multimodal_projector_external.
- Quantization/runtime notes: supplied Unsloth notes list 27B 4-bit around 18 GB total memory and 35B-A3B 4-bit around 23 GB.
- License: TBD from model card before runtime work.
- Known tool support: some GGUF flows require separate multimodal projector files; do not assume Ollama compatibility.
- Source links: supplied planning notes; model card URL TBD.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| 27B | 27B | dense | 256K / YaRN notes to 1M | multimodal | text-only reduced-context design | tier_b_partial |
| 35B-A3B | 35B total | A3B | 256K / YaRN notes to 1M | multimodal | future MoE gate | tier_b_partial |

## Implementation contract
### Loader
Record intended artifact formats and tensor mappings only; no loader claim exists until receipt-backed.

### Tokenizer
Tokenizer source, vocabulary, and fallback status must be explicit in receipts.

### Prompt template
Prompt template behavior must be recorded before any smoke test.

### Architecture module
Architecture features are design notes until implemented and parity-tested.

### Kernels/backend
Backend selection and fallback must be recorded; no speedup claim is allowed from docs.

### Receipts
Use model receipt templates and set untested coverage fields to false.

### Tests
First proof is a strict deterministic smoke only, not quality or performance evidence.

## Explicit non-claims
- Qwen3.6 docs do not imply 256K context works in bitnet-rs.
- 27B offload/design does not imply full 5070Ti residency.
- 35B-A3B MoE is not implemented until expert routing is receipt-backed.

## First proof target

```json
{
  "claim": "qwen36_27b_text_only_reduced_context_design",
  "model_family": "qwen3.6",
  "variant": "27b",
  "task": "text_generation",
  "context_requested": 2048,
  "full_context_claim": false,
  "multimodal_claim": false,
  "fallback_used": "TBD",
  "speedup_claim": false
}
```

## Work items
- QWEN36-DOC-001: Add Qwen3.6 family doc.
- QWEN36-DOC-002: Add Qwen3.6 prompt/template settings for thinking and non-thinking.
- QWEN36-DOC-003: Add 27B text-only reduced-context proof plan.
- QWEN36-DOC-004: Add 35B-A3B MoE future gate.
- QWEN36-DOC-005: Add multimodal projector future gate.

