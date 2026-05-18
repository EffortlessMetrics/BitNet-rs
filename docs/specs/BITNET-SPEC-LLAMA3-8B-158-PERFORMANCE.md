# BITNET-SPEC-LLAMA3-8B-158-PERFORMANCE

Status: proposed
Owner: performance
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: no performance claim until exact-profile review
Policy impact: no policy exception

## Profiles

Required profiles are `cold_load`, `warm_load`, `artifact_convert_time`,
`one_token`, `first_token`, `prefill_128_decode_16`,
`prefill_512_decode_32`, `decode_32`, `decode_128`, `warm_session_3_turns`,
`warm_session_10_turns`, and `server_nonstream_exact_profile`.

## Metrics

Receipts must record `model_load_ms`, `tokenizer_load_ms`, `prompt_render_ms`,
`tokenize_ms`, `prefill_ms`, `first_token_ms`, `decode_total_ms`,
`steady_tok_per_s`, `kernel_time_ms`, launch count, memory high-water, VRAM
high-water, H2D/D2H bytes and timing, thread count, selected backend, selected
kernel, and `fallback_used`.

## Promotion rule

No performance claim is allowed without same artifact, same tokenizer, same
prompt profile, `fallback=false`, answer quality passed, CPU comparator, and
exact-profile review.
