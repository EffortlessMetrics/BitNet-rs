# BITNET-SPEC-INTEL-GPU-PERFORMANCE

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines benchmark qualification; no speed claim.
Policy impact: No exception.

## Profiles

Intel GPU performance evidence is profile-specific:

```text
cold_load
warm_load
one_token
ask_short
ask_normal
prefill_128_decode_16
prefill_512_decode_32
decode_32
decode_128
warm_session_3_turns
warm_session_10_turns
resident_10x_ask_short
server_nonstream_exact_profile
```

## Required timing fields

Receipts used for performance claims must record, or explicitly mark
not-applicable:

```text
model_load_ms
tokenizer_load_ms
prompt_render_ms
tokenize_ms
runtime_context_init_ms
kernel_or_graph_compile_ms
weight_upload_ms
prefill_ms
first_token_ms
decode_total_ms
steady_tok_per_s
kernel_or_graph_time_ms
launch_count
H2D/D2H bytes and timing
VRAM/shared-memory high-water
power/thermal context
```

## Promotion requirements

A performance claim requires:

```text
quality_passed=true
fallback_used=false
profile_timing_applicable=true
same-model same-profile comparator
two same-device history receipts for performance claim
claim reviewed and accepted
```

A single fast run is a benchmark candidate only. It cannot become a public
speedup claim without the full gate above.
