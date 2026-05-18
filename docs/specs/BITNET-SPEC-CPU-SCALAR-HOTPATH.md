# BitNet CPU Scalar Hot-Path Contract

Status: draft
Owner: cpu-proof campaign
Created: 2026-05-18
Linked proposal: n/a
Linked specs: docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md; docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md; docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md
Linked ADRs: n/a
Linked plan: plans/cpu-scalar/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines scalar CPU steady-state proof requirements; does not claim performance by itself.
Policy impact: No policy exception.

## Purpose

This specification defines what is forbidden in scalar steady state and what
counters are required before scalar performance receipts can be interpreted. The
scalar lane should be usable, measurable, and auditable rather than a slow
accidental fallback hidden behind allocation and conversion noise.

## Steady-state scalar must not do

Scalar CPU steady-state proof must not rely on:

```text
whole-matrix dequantization
per-token packed-weight flattening
per-token qk256_tensor.to_vec2::<u8>()
per-layer Vec<Vec<f32>> input materialization
per-layer Vec<Vec<f32>> output materialization
hidden fallback to diagnostic dense path
ambiguous kernel IDs
```

Early correctness PRs may retain transitional allocations, but any receipt that
claims scalar hot-path performance must expose the counters below and state which
transitional costs remain.

## Required counters

Receipts for scalar CPU runs must be able to expose this shape:

```json
{
  "scalar_hot_path": {
    "qk256_f32_scalar_invocations": 0,
    "qk256_i8s_scaled_scalar_invocations": 0,
    "qk256_scalar_gemm_invocations": 0,
    "flat_weight_extract_count": 0,
    "input_vec2_materialization_count": 0,
    "output_vecvec_allocation_count": 0,
    "workspace_reuse_count": 0
  }
}
```

Counter names may be implemented in a receipt crate or dispatch wrapper, but the
meaning must remain stable:

| Counter | Meaning |
| --- | --- |
| `qk256_f32_scalar_invocations` | F32/no-scale scalar GEMV invocations. |
| `qk256_i8s_scaled_scalar_invocations` | BitNet scaled I2_S × I8_S scalar GEMV invocations. |
| `qk256_scalar_gemm_invocations` | Scalar packed GEMM invocations, split into precise IDs once scaled GEMM exists. |
| `flat_weight_extract_count` | Packed weight bytes copied out of tensor storage for dispatch. |
| `input_vec2_materialization_count` | Per-layer or per-call `Vec<Vec<f32>>` input materializations. |
| `output_vecvec_allocation_count` | Per-layer or per-call `Vec<Vec<f32>>` output allocations. |
| `workspace_reuse_count` | Reuse events for scalar scratch or output workspace. |

## Hot-path proof rule

A strict scalar answer-corpus receipt can prove correctness while transitional
allocation counters are non-zero. A scalar performance receipt must preserve the
same model, tokenizer, prompts, generated IDs or explicit divergence, selected
kernel, and fallback status while reporting these counters before making any
claim about improved scalar efficiency.

## Claim boundary

This hot-path contract supports only scalar CPU correctness and scalar CPU
baseline performance evidence. It does not prove AVX2, AVX-512, CUDA, OpenCL,
OpenVINO, NPU, server, or dense-SLM behavior.
