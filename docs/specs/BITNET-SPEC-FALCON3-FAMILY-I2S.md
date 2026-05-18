# BITNET-SPEC-FALCON3-FAMILY-I2S: Falcon3 I2_S Contract

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

Define the highest-priority Falcon3 kernel/layout proof path. Direct Falcon3 1B and 7B GGUF candidates expose `ggml-model-i2_s.gguf`, but Falcon3 I2_S must be proven layout-compatible before using existing QK256 kernels as aliases.

## Required Layout Topics

```text
Falcon3 GGUF metadata required for I2_S
QK256 / grouped block layout verification
weight scale semantics
activation quantization to I8_S
act_scale
act_sum
integer dot correction
tail-column behavior
row stride behavior
embedding policy
LM head / tied-head policy
compatibility with existing qk256-scalar-i8s-scaled-gemv
```

## Candidate Kernel IDs

```text
falcon3-i2s-scalar-reference-gemv
falcon3-i2s-avx2-gemv
falcon3-i2s-avx512-gemv
falcon3-i2s-cuda-gemv
falcon3-i2s-apple-neon-gemv
falcon3-i2s-opencl-gemv
```

These IDs may alias existing QK256 kernels only after compatibility proof records exact metadata, block layout, scale semantics, tail behavior, row stride, and fixture parity.

## Hard Rule

```text
Existing Microsoft 2B QK256 kernel proof does not automatically prove Falcon3 I2_S layout compatibility.
```
