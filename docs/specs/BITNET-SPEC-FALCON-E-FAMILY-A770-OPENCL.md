# BITNET-SPEC-FALCON-E-FAMILY-A770-OPENCL

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-CPU.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-I2S.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: A770 support only after exact OpenCL receipts
Policy impact: no policy exception

## A770 route

```text
Falcon-E 1B I2_S CPU answer-ready
→ I2_S layout compatible with QK256 OpenCL
→ A770 fixture parity
→ one-token proof
→ behavior suite
→ exact-profile benchmark
```

## Required evidence

A770 receipts must record artifact SHA, CPU comparator, OpenCL device identity,
selected backend, selected kernel, I2_S/QK256 compatibility receipt, fixture
parity, fallback status, generated token IDs, decoded text, unsupported op
count, memory/VRAM high-water, and speed claim boundary.

## Hard rules

```text
A770 Falcon-E proof is not CUDA proof.
A770 Falcon-E proof is not Microsoft 2B proof.
A770 QK256 proof does not imply full device residency.
```
