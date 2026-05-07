# Qwen 3.6

## Status
- Repo status: design_scaffold
- First target: qwen3.6-27b-text-only-gguf-design
- Local test tier: tier_b_partial
- Implementation owner lane: dense_decoder, moe_decoder_for_35b_a3b, prompt_template, tool_calling
- Design-only? no for reduced-context design; local proof remains partial/offload only

## Source-backed facts
- Model variants: 27B and 35B-A3B.
- Context: 256K context / 262,144 tokens, extendable to 1M via YaRN in supplied notes.
- Modalities: Multimodal hybrid-thinking.
- Tokenizer/prompt: Developer role for agentic coding tools; improved nested tool-calling parsing; thinking and non-thinking modes have different recommended sampling settings.
- Architecture features: hybrid_thinking, developer_role, tool_calling, long_context, yarn_future, multimodal_projector_external.
- Quantization/runtime notes: Unsloth notes 27B 4-bit around 18 GB total memory and 35B-A3B 4-bit around 23 GB.
- License: TBD from model card.
- Known tool support: Some GGUF flows require separate multimodal projector files; do not assume Ollama compatibility.
- Source links: Supplied planning notes in this change; model-card URLs must be verified before advancing beyond `documented` or `design_scaffold`.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
|---|---:|---:|---|---|---|---|
| 27B | 27B | N/A | 256K | multimodal | text-only reduced-context design | tier_b_partial |
| 35B-A3B | 35B | 3B active TBD | 256K | multimodal | MoE future gate | tier_b_partial |

## Implementation contract
### Loader
Text-only GGUF design first; projector and 35B-A3B expert routing are future-gated.

### Tokenizer
Tokenizer source must be explicit and variant-compatible.

### Prompt template
Separate thinking and non-thinking template settings; developer role and nested tools are opt-in receipt fields.

### Architecture module
Multimodal decoder with dense 27B plan and MoE gate for 35B-A3B.

### Kernels/backend
Non-BitNet dense/MoE lanes; QK256 proof is not evidence.

### Receipts
Reduced-context receipts must record context requested and `full_context_claim=false`.

### Tests
Future: prompt rendering tests, one-token text-only smoke with reduced context.

## Explicit non-claims
- Qwen3.6 docs do not imply 256K context works in bitnet-rs.
- 27B offload/design does not imply full 5070Ti residency.
- 35B-A3B MoE is not implemented until expert routing is receipt-backed.

## First proof target
Design receipt or future one-token receipt with reduced context, text-only, no speedup claim.

## Work items
- QWEN36-DOC-001: Add Qwen3.6 family doc.
- QWEN36-DOC-002: Add Qwen3.6 prompt/template settings for thinking and non-thinking.
- QWEN36-DOC-003: Add 27B text-only reduced-context proof plan.
- QWEN36-DOC-004: Add 35B-A3B MoE future gate.
- QWEN36-DOC-005: Add multimodal projector future gate.
