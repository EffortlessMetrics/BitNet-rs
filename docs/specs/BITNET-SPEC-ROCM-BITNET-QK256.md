# BITNET-SPEC-ROCM-BITNET-QK256: ROCm BitNet QK256 Semantics

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm route contract](BITNET-SPEC-ROCM-ROUTE-CONTRACT.md), [ROCm kernel compile](BITNET-SPEC-ROCM-KERNEL-COMPILE.md), [ROCm quality](BITNET-SPEC-ROCM-QUALITY.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines BitNet ROCm requirements; no model promotion
Policy impact: no CI policy exception

## Purpose

Define the ROCm BitNet route so packed I2_S/QK256 proof cannot be satisfied by
dense ROCm kernels, diagnostic F32 kernels, source-text tests, or other backend
proof.

## Required Semantics

The production BitNet ROCm path must match the existing CPU scalar oracle and
CUDA/A770 semantics:

```text
official Microsoft I2_S/QK256 GGUF
canonical QK256 packed layout
BitNet.cpp-aligned I2_S code mapping
activation quantization to I8_S
act_scale
act_sum
integer dot
(dot - act_sum) / act_scale * weight_scale
tail-column behavior
row-stride behavior
strict tokenizer/template authority
```

## Kernel IDs

```text
qk256-rocm-i8s-scaled-gemv
qk256-rocm-i8s-scaled-gemm
qk256-rocm-f32-diagnostic-gemv
```

## Fixture Scope

The scalar oracle fixture set must include rows `1`, `2`, `7`, and `32`; columns
`1`, `2`, `127`, `128`, `129`, `255`, `256`, `257`, `512`, and `1024`; all-zero,
all-one, all-two, all-three, cyclic, and pseudorandom packed patterns; zero,
constant, signed-ramp, and pseudorandom activations; and weight scales `0.125`,
`0.5`, and `1.0`.

## Hard Rules

```text
No-scale F32 diagnostic GEMV cannot satisfy production BitNet QK256 proof.
Dense SLM ROCm kernels cannot satisfy BitNet packed I2_S/QK256 proof.
HIP source text proof cannot satisfy QK256 model proof.
```

Model-level BitNet ROCm promotion requires fallback false, selected AMD GPU,
selected HIP/ROCm runtime, QK256 invocation counters, CPU/ROCm parity or first
divergence classification, strict tokenizer/template authority, and answer
quality evidence.
