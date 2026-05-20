# BITNET-SPEC-FALCON3-FAMILY-PERFORMANCE: Falcon3 Performance Contract

Status: draft
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0012](../proposals/BITNET-PROP-0012-falcon3-family-supported-models.md)
Linked specs: n/a
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [Falcon3 family implementation plan](../../plans/falcon3-family/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines future gates only; no promotion
Policy impact: no policy exception

## Purpose

Define exact-profile performance gates for Falcon3. Performance is not claimable from detection, successful loading, answer quality, or generic backend execution.

## Profiles

```text
cold_load
warm_load
artifact_probe
one_token
first_token
prefill_128_decode_16
prefill_512_decode_32
decode_32
decode_128
warm_session_3_turns
warm_session_10_turns
server_nonstream_exact_profile
```

## Required Metrics

```text
model_load_ms
tokenizer_load_ms
prompt_render_ms
tokenize_ms
prefill_ms
first_token_ms
decode_total_ms
steady_tok_per_s
kernel_time_ms
launch_count
memory high-water
VRAM high-water
H2D/D2H bytes and timing
thread count
selected backend
selected kernel
fallback_used
```

## Promotion Rule

```text
No performance claim without same artifact, same tokenizer, same prompt profile,
fallback=false, answer quality passed, CPU comparator, and exact-profile review.
```

Speedup, full residency, product CLI readiness, and server readiness are separate claims. Server profiles require exact endpoint/profile receipts after product CLI readiness.
