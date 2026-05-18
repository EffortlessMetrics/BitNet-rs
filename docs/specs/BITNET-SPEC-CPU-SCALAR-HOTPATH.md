# BitNet CPU Scalar Hot-Path Contract

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-scalar/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines scalar CPU hot-path evidence; does not claim speedup.
Policy impact: No policy exception.

## Purpose

The scalar lane must be accurate enough to be the oracle and efficient enough to
be a usable forced CPU fallback. This contract defines what scalar steady-state
inference should not do and what counters must make remaining overhead visible.

## Forbidden Steady-State Behavior

A scalar steady-state proof should not perform:

- whole-matrix dequantization;
- per-token packed-weight flattening;
- per-token `qk256_tensor.to_vec2::<u8>()` extraction;
- per-layer `Vec<Vec<f32>>` input materialization;
- per-layer `Vec<Vec<f32>>` output materialization;
- hidden fallback to a diagnostic dense path;
- ambiguous kernel IDs such as `reference` or unqualified `qk256-scalar-gemv` in
  new receipts.

Diagnostic tests may still use intentionally dequantized or materialized paths,
but those tests must be labeled diagnostic and must not support steady-state
performance claims.

## Required Hot-Path Counters

Receipts that exercise scalar QK256 should expose a `scalar_hot_path` object
with at least these fields:

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

Counters may be zero for profiles that do not exercise the corresponding path.
Strict scalar BitNet answer or benchmark receipts should have
`qk256_i8s_scaled_scalar_invocations > 0` and `fallback_used=false`.

## Hot-Path Acceptance Rails

- Removing a materialization must preserve generated IDs or record exact
  divergence.
- Allocation reductions must not introduce a new tolerance.
- Workspace reuse must not hide mutable cross-request state in receipts.
- No scalar performance receipt may set `speedup_claim=true` unless a separate
  reviewed performance claim authorizes it.
