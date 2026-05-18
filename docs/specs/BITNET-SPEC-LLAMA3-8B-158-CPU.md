# BITNET-SPEC-LLAMA3-8B-158-CPU

Status: proposed
Owner: cpu-proof
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: no CPU answer claim until strict receipts pass
Policy impact: no policy exception

## CPU paths

Candidate CPU paths are x86 `I2_S` scalar, x86 `I2_S` AVX2, x86 `I2_S` AVX512,
x86 `TL2` scalar, x86 `TL2` AVX2/AVX512, ARM `I2_S` scalar/NEON, and ARM `TL1`
scalar/NEON.

## Acceptance

CPU answer readiness requires a reference-good artifact, Rust loader route
recognition, scalar oracle fixture pass, strict CPU answer corpus pass,
`fallback=false`, prompt IDs, generated IDs, decoded text, and `speedup=false`
until benchmark review.

## Hard rules

CPU proof does not prove CUDA, Apple, TL route families, server readiness, or
speedup. Microsoft 2B CPU proof cannot be inherited by this Llama3-derived
artifact.
