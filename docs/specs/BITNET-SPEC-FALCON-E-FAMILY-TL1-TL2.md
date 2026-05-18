# BITNET-SPEC-FALCON-E-FAMILY-TL1-TL2

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-ROUTE-COMPATIBILITY.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: TL route registration only
Policy impact: no policy exception

## Purpose

Prepare the listed Falcon-E TL routes without blocking the direct `I2_S` lane.
TL1/TL2 work starts with layout documentation and scalar oracles, not
accelerator kernels.

## Required definitions

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

## Fixture requirements

TL fixtures must cover tiny synthetic matrices, known outputs, row stride, tail
columns, scale edge cases, lookup-table decoding, and unsupported-route
diagnostics. They must not call I2_S/QK256 functions.

## Hard rules

```text
TL1/TL2 are not QK256.
No TL accelerator work before scalar TL oracle exists.
No x86 TL2 claim from x86 I2_S proof.
No ARM TL1 claim from ARM I2_S proof.
```
