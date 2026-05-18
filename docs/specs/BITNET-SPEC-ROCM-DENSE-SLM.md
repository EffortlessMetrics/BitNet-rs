# BITNET-SPEC-ROCM-DENSE-SLM: ROCm Dense SLM Route

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm route contract](BITNET-SPEC-ROCM-ROUTE-CONTRACT.md), [ROCm quality](BITNET-SPEC-ROCM-QUALITY.md), [ROCm performance](BITNET-SPEC-ROCM-PERFORMANCE.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines dense SLM ROCm requirements; no model promotion
Policy impact: no CI policy exception

## Purpose

Define dense model ROCm support separately from BitNet QK256. Dense SLM proof
must identify the exact artifact, tensor route, tokenizer, prompt authority,
answer quality, fallback state, performance profile, and residency scope for the
selected AMD ROCm backend.

## Initial Model Ladder

```text
Qwen2.5 0.5B Q8_0
Qwen2.5 1.5B Q4_K_M
Qwen3 0.6B Q8_0
SmolLM2 360M / 1.7B if CPU reference passes
Llama 3.2 1B / 3B candidate
Gemma/Phi small candidates
```

## Proof Ladder

```text
artifact contract
CPU answer sanity
ROCm all-layer plan
single-linear parity
one-token proof
short-decode proof
warm-session proof
benchmark review
product CLI promotion
server exact-profile review
```

## Not-Claims

```text
not_bitnet_qk256
not_cuda
not_full_residency
not_speedup_without_review
```

A dense SLM ROCm receipt must not set BitNet QK256 proof fields true, and a
BitNet QK256 receipt must not promote dense SLM model support.
