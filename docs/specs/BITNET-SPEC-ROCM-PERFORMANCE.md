# BITNET-SPEC-ROCM-PERFORMANCE: ROCm Profile-Specific Performance

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm route contract](BITNET-SPEC-ROCM-ROUTE-CONTRACT.md), [ROCm quality](BITNET-SPEC-ROCM-QUALITY.md), [ROCm residency](BITNET-SPEC-ROCM-RESIDENCY.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines performance receipt rules; no speed promotion
Policy impact: no CI policy exception

## Purpose

Make ROCm speed claims exact-profile claims. Timings may be recorded before
speed is accepted, but no receipt, status row, or CLI summary may imply global
ROCm speedup.

## Profiles

```text
kernel_micro_qk256_gemv
kernel_micro_qk256_gemm
single_layer_decode
one_token
ask_short
ask_normal
prefill_128_decode_16
prefill_512_decode_32
decode_32
decode_128
warm_session_3_turns
warm_session_10_turns
server_nonstream_exact_profile
```

## Required Timing Fields

```text
model_load_ms
tokenizer_load_ms
prompt_render_ms
tokenize_ms
rocm_context_init_ms
hip_module_compile_ms
kernel_compile_ms
weight_upload_ms
prefill_ms
first_token_ms
decode_total_ms
steady_tok_per_s
kernel_time_ms
launch_count
H2D_bytes
H2D_ms
D2H_bytes
D2H_ms
VRAM_high_water
power_temperature_context
fallback_used
```

## Promotion Requirements

```text
quality_passed=true
fallback_used=false
profile_timing_applicable=true
same-model CPU comparator
same-model CUDA/A770 comparator where useful
two same-device history receipts
speedup accepted by review
```

A rejected profile may remain valuable benchmark evidence, but it must keep
`speedup_claim=false` and must not promote product support beyond the accepted
quality/residency level.
