# BITNET-SPEC-ROCM-STATUS-SURFACE: ROCm Status And Doctor UX

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm route contract](BITNET-SPEC-ROCM-ROUTE-CONTRACT.md), [ROCm device identity](BITNET-SPEC-ROCM-DEVICE-IDENTITY.md), [ROCm kernel compile](BITNET-SPEC-ROCM-KERNEL-COMPILE.md), [ROCm quality](BITNET-SPEC-ROCM-QUALITY.md), [ROCm performance](BITNET-SPEC-ROCM-PERFORMANCE.md), [ROCm residency](BITNET-SPEC-ROCM-RESIDENCY.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines future status UX; no CLI promotion
Policy impact: no CI policy exception

## Purpose

Make ROCm status visible without overclaiming. Status commands must summarize
the next missing proof and preserve not-claims instead of turning a toolkit path
or source-text test into product support.

## Commands

```bash
bitnet rocm doctor --format json
bitnet model status --device amd-rocm
bitnet receipts explain <receipt>
bitnet gpu doctor --vendor amd
```

## Status Output Fields

```text
ROCm installed
HIP available
selected AMD GPU
GFX target
supported/unsupported official status
kernel source compile status
tiny kernel smoke status
BitNet QK256 status
dense SLM status
quality status
performance status
residency status
server status
not-claims
next proof required
```

## Fail-Closed Behavior

`--device amd-rocm --strict` must fail before generation or benchmarking when a
required ROCm proof field is missing. Non-strict diagnostics may report missing
prerequisites, but they must set fallback and support fields truthfully and must
not promote an accelerator claim.
