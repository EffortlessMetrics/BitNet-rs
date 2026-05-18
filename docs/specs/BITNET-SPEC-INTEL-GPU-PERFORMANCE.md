# BITNET-SPEC-INTEL-GPU-PERFORMANCE

## Purpose

Define Intel GPU efficiency claims without overclaiming from a single timing or
quality-free benchmark.

## Profiles

- `cold_load`
- `warm_load`
- `one_token`
- `ask_short`
- `ask_normal`
- `prefill_128_decode_16`
- `prefill_512_decode_32`
- `decode_32`
- `decode_128`
- `warm_session_3_turns`
- `warm_session_10_turns`
- `resident_10x_ask_short`
- `server_nonstream_exact_profile`

## Required timing fields

Receipts should record `model_load_ms`, `tokenizer_load_ms`, `prompt_render_ms`,
`tokenize_ms`, `runtime_context_init_ms`, `kernel_or_graph_compile_ms`,
`weight_upload_ms`, `prefill_ms`, `first_token_ms`, `decode_total_ms`,
`steady_tok_per_s`, `kernel_or_graph_time_ms`, `launch_count`, H2D/D2H bytes and
timing, VRAM/shared-memory high-water, and power/thermal context.

## Promotion requirements

Performance promotion requires `quality_passed=true`, `fallback_used=false`,
`profile_timing_applicable=true`, same-model same-profile comparator evidence,
two same-device history receipts for a performance claim, and an accepted claim
review.
