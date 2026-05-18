# BITNET-SPEC-CPU-SCALAR-HOTPATH: CPU Scalar Hot-Path Contract

Status: proposed

Linked plan:
[CPU scalar implementation plan](../../plans/cpu-scalar/implementation-plan.md)

Linked specs:
[CPU scalar kernel contract](BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md),
[CPU scalar performance contract](BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md)

## Purpose

This spec defines what is forbidden in scalar steady-state CPU BitNet inference
and which counters must expose remaining hot-path overhead. Scalar is allowed to
be slower than optimized SIMD lanes, but it must not be accidentally slow because
it repeatedly copies, dequantizes, or materializes data structures that should be
stable packed views or reusable workspaces.

## Forbidden In Steady-State Scalar Proof

Strict scalar steady-state receipts must not rely on:

```text
whole-matrix dequantization
per-token packed-weight flattening
per-token qk256_tensor.to_vec2::<u8>()
per-layer Vec<Vec<f32>> input materialization
per-layer Vec<Vec<f32>> output materialization
hidden fallback to diagnostic dense path
ambiguous kernel IDs
```

If one of these remains during a transitional PR, the receipt must report the
counter and the plan item must state the rollback and cleanup path. Transitional
evidence is not a final scalar hot-path proof.

## Required Counters

Receipts for scalar CPU work must expose scalar hot-path counters with stable
field names:

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

Implementation may add more detailed counters, but these names are the minimum
receipt contract.

## Counter Semantics

| Counter | Meaning |
| --- | --- |
| `qk256_f32_scalar_invocations` | Number of no-scale F32 scalar QK256 calls. |
| `qk256_i8s_scaled_scalar_invocations` | Number of BitNet scaled I2_S x I8_S scalar QK256 calls. |
| `qk256_scalar_gemm_invocations` | Number of scalar QK256 GEMM calls; precise kernel ID must distinguish F32 and scaled I8_S when available. |
| `flat_weight_extract_count` | Number of packed-weight flatten/copy operations at runtime. |
| `input_vec2_materialization_count` | Number of `Vec<Vec<f32>>`-style input row materializations. |
| `output_vecvec_allocation_count` | Number of `Vec<Vec<f32>>`-style output row allocations. |
| `workspace_reuse_count` | Number of times reusable scalar workspace storage was used instead of allocating fresh scratch. |

## Acceptance Requirements

A strict scalar hot-path proof must show:

- `selected_kernel` is a precise scalar ID;
- `fallback_used=false`;
- the scaled BitNet path records `qk256_i8s_scaled_scalar_invocations > 0`;
- no AVX2, AVX-512, CUDA, Metal, OpenCL, OpenVINO, or NPU kernel was selected;
- no whole-matrix dequantized/reference path substituted for scaled BitNet
  I8_S math;
- remaining transitional allocations are counted, not hidden.

## Non-Goals

This spec does not require scalar to beat accelerated lanes. It only requires
scalar to be honest, directly measured, and not wasteful in ways that obscure
phase timing or oracle comparisons.
