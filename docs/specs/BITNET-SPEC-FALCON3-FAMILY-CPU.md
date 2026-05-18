# BITNET-SPEC-FALCON3-FAMILY-CPU: Falcon3 CPU Contract

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

Define CPU proof requirements for Falcon3 exact artifacts and routes.

## CPU Paths

```text
x86 I2_S scalar
x86 I2_S AVX2
x86 I2_S AVX512
x86 TL2 scalar/AVX candidate
ARM I2_S scalar/NEON
ARM TL1 scalar/NEON candidate
```

## CPU Acceptance

```text
artifact inventory exists
reference-good receipt exists
Rust loader recognizes Falcon3 route
I2_S/TL scalar oracle passes fixtures
strict CPU answer corpus passes
fallback=false
prompt IDs recorded
generated IDs recorded
decoded text recorded
speedup=false
```

## Receipt Requirements

CPU receipts must include requested backend, selected backend, selected kernel, exact artifact ID/hash, tokenizer/prompt receipt, route ID, fixture or corpus ID, deterministic generation settings, prompt IDs, generated IDs, decoded text, fallback flag, and explicit not-claims for CUDA, Apple, A770, TL when not selected, speedup, product CLI readiness, and server readiness.
