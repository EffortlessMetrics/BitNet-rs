# BITNET-SPEC-LLAMA3-8B-158-I2S

Status: proposed
Owner: cpu-proof
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: no I2_S support until layout and answer receipts pass
Policy impact: no policy exception

## Purpose

Define the `I2_S` route requirements for this exact Llama3-derived model. This
route is promising because upstream lists x86 and ARM `I2_S`, but it cannot
inherit Microsoft 2B QK256 proof.

## Required layout contract

The `I2_S` proof must define GGUF metadata, QK256 row/block layout, weight scale
semantics, activation quantization to `I8_S`, `act_scale`, `act_sum`, integer
dot behavior, tail-column behavior, row stride behavior, embedding policy,
output head policy, and compatibility with existing
`qk256-scalar-i8s-scaled-gemv` semantics.

## Required kernel IDs

- `llama3-8b-158-i2s-scalar-reference-gemv`
- `llama3-8b-158-i2s-avx2-gemv`
- `llama3-8b-158-i2s-avx512-gemv`
- `llama3-8b-158-i2s-cuda-gemv`
- `llama3-8b-158-i2s-apple-neon-gemv`

These IDs may alias existing QK256 kernels only after shape/layout proof says
they are compatible for this artifact.

## Hard rules

No `I2_S/QK256` route claim may be made until converted artifact layout matches
the scalar oracle. CUDA and Apple `I2_S` receipts must follow CPU answer-ready
for the exact artifact and prompt profile.
