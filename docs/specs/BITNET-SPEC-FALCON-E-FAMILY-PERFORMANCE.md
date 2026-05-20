# BITNET-SPEC-FALCON-E-FAMILY-PERFORMANCE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-CPU.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-CUDA.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-APPLE.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-A770-OPENCL.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: performance claims only after exact-profile review
Policy impact: no policy exception

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

## Required metrics

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

## Promotion rule

```text
No performance claim without same artifact, same tokenizer, same prompt profile,
fallback=false, answer quality passed, CPU comparator, and exact-profile review.
```

## Boundaries

Model-card memory-footprint and benchmark tables are external claims, not
BitNet-rs support proof. BitNet-rs benchmark receipts must be exact-profile,
artifact-specific, backend-specific, and quality-gated.
