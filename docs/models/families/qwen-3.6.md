# Qwen 3.6

## Status
- Repo status: design_scaffold.
- First target: qwen3.6-27b-text-only-gguf-design.
- Local test tier: tier_b_partial.
- Implementation owner lane: dense_decoder, moe_decoder_for_35b_a3b, prompt_template, tool_calling.
- Design-only? no for planning; yes for full 256K/1M, multimodal projector, and 35B-A3B MoE until receipts exist.

## Source-backed facts
- Model variants: 27B and 35B-A3B.
- Context: 256K context; maximum context 262,144 and extendable to 1M via YaRN in supplied notes.
- Modalities: multimodal hybrid-thinking.
- Tokenizer/prompt: thinking and non-thinking modes have different recommended sampling; developer role is supported for agentic coding tools.
- Architecture features: hybrid_thinking, tool_calling, developer_role, long_context, yarn_future, multimodal_projector_external.
- Quantization/runtime notes: 27B 4-bit around 18 GB total memory; 35B-A3B 4-bit around 23 GB total memory; some GGUF flows require separate multimodal projector files.
- License: TBD from model card.
- Known tool support: do not assume Ollama compatibility when projector files are separate.
- Source links: Source links are tracked in `docs/tracking/model-family-foundation/source-index.yaml`; supplied notes must be verified against model cards before implementation claims advance.

## Variants
| Variant | Params | Active params | Context | Modalities | First repo target | Local test tier |
| --- | --- | --- | --- | --- | --- | --- |
| 27B | 27B | dense | 256K / YaRN to 1M in notes | multimodal text/image TBD | reduced-context text-only design | tier_b_partial |
| 35B-A3B | 35B | 3B active | 256K / YaRN to 1M in notes | multimodal text/image TBD | MoE future gate | tier_b_partial |

## Implementation contract
### Loader
Start with text-only GGUF design; projector files are separate future-gated artifacts.
### Tokenizer
Verify tokenizer and chat template from model card before scaffolding.
### Prompt template
Record thinking/non-thinking sampling settings and developer role behavior.
### Architecture module
Dense 27B and MoE 35B-A3B must be separate contracts.
### Kernels/backend
No full-residency or 256K claim on 16 GB hardware without receipt.
### Receipts
Reduced-context receipt must set `long_context_claim=false` and `multimodal_claim=false`.
### Tests
First proof is one-token text-only if memory/offload plan is available.

## Explicit non-claims
- Qwen3.6 docs do not imply 256K context works in bitnet-rs.
- 27B offload/design does not imply full 5070Ti residency.
- 35B-A3B MoE is not implemented until expert routing is receipt-backed.

## First proof target
Use generative one-token receipt with `context_requested` reduced, `full_context_claim=false`, and `fallback_used` recorded.

## Work items
- QWEN36-DOC-001: Add Qwen3.6 family doc.
- QWEN36-DOC-002: Add Qwen3.6 prompt/template settings for thinking and non-thinking.
- QWEN36-DOC-003: Add 27B text-only reduced-context proof plan.
- QWEN36-DOC-004: Add 35B-A3B MoE future gate.
- QWEN36-DOC-005: Add multimodal projector future gate.
