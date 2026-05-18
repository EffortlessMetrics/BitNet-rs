# BITNET-SPEC-FALCON3-FAMILY-TL1-TL2: Falcon3 TL1/TL2 Contract

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

Prepare listed Falcon3 TL routes without blocking the direct I2_S path. TL1/TL2 are separate proof families and are not QK256.

## Required Layout Topics

```text
TL1 tensor layout
TL2 tensor layout
lookup-table semantics
bit packing
group/block size
scale semantics
activation type
row stride
tail behavior
GGUF metadata
difference from I2_S/QK256
```

## Required Proof Order

1. Route/conversion authority.
2. Synthetic TL1/TL2 fixture corpus.
3. Scalar TL oracle.
4. Reference-runner proof for an exact artifact or converted output.
5. CPU route proof.
6. Accelerator proof only after scalar oracle and CPU proof.

## Hard Rules

```text
TL1/TL2 are not QK256.
No TL accelerator work before scalar TL oracle exists.
No x86 TL2 claim from x86 I2_S proof.
No ARM TL1 claim from ARM I2_S proof.
No TL proof from Microsoft 2B I2_S proof.
```
