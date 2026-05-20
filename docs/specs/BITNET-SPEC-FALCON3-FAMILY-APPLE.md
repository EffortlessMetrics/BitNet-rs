# BITNET-SPEC-FALCON3-FAMILY-APPLE: Falcon3 Apple Contract

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

Define Apple CPU/NEON and later Metal boundaries for Falcon3 without conflating MacBook, M4 Mac mini, CPU/NEON, and Metal evidence.

## Apple Route

```text
MacBook artifact inventory
M4 / MacBook reference runner
ARM I2_S CPU/NEON strict answer proof
ARM TL1 candidate after layout proof
Metal phase candidates only after CPU/NEON proof
```

## Required Apple Receipt Fields

Apple receipts must record machine identity, chip, memory, macOS, storage context, backend label, requested route, selected route, selected kernel, fallback=false, tokenizer/prompt authority, prompt IDs, generated IDs, decoded text, thermal/mobile context when available, and cleanup status.

## Hard Rules

```text
MacBook proof does not prove M4 Mac Mini.
M4 proof does not prove MacBook.
Apple CPU/NEON proof does not prove Metal.
Metal phase proof does not prove full Metal inference.
Falcon3 Apple proof is not Falcon-E or Microsoft 2B Apple proof.
```
