# BITNET-SPEC-FALCON3-FAMILY-A770-OPENCL: Falcon3 A770/OpenCL Contract

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

Define the optional later A770/OpenCL path for Falcon3 without implying CUDA, Microsoft 2B, or full-residency proof.

## A770 Route

```text
Falcon3 1B I2_S CPU answer-ready
→ I2_S layout compatible with QK256 OpenCL
→ A770 fixture parity
→ one-token proof
→ behavior suite
→ exact-profile benchmark
```

## Receipt Requirements

A770 receipts must include hardware identity, driver/runtime identity, requested backend, selected backend, selected kernel, exact artifact hash, tokenizer/prompt receipt, route ID, fallback=false, fixture/corpus ID, generated IDs, decoded text, H2D/D2H transfer metadata when applicable, and explicit not-claims for CUDA, Apple, speedup, server readiness, and full residency.

## Hard Rules

```text
A770 Falcon3 proof is not CUDA proof.
A770 Falcon3 proof is not Microsoft 2B proof.
A770 QK256 proof does not imply full device residency.
A770 proof does not prove all Falcon3 sizes or TL routes.
```
