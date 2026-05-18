# BitNet CPU Scalar Hot-Path Contract

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-scalar/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines scalar CPU efficiency and instrumentation requirements; does not claim throughput.
Policy impact: No policy exception.

## Purpose

This spec defines what the scalar CPU steady state must avoid and what counters
must be present before scalar performance receipts can be trusted. The scalar
lane is allowed to be slower than SIMD lanes, but it must not be accidentally
slow because every token re-materializes packed weights or row buffers.

## Forbidden Steady-State Behavior

Scalar steady-state inference should not do the following:

```text
whole-matrix dequantization
per-token packed-weight flattening
per-token qk256_tensor.to_vec2::<u8>()
per-layer Vec<Vec<f32>> input materialization
per-layer Vec<Vec<f32>> output materialization
hidden fallback to diagnostic dense path
ambiguous kernel IDs
```

Diagnostic commands may still use expensive conversions when they are explicitly
scoped as diagnostic and receipts or logs do not present them as the production
scalar hot path.

## Required Hot-Path Counters

Scalar receipts must grow a `scalar_hot_path` object with these fields or an
explicit schema successor that preserves the same meaning:

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

Counter semantics:

| Counter | Meaning |
| --- | --- |
| `qk256_f32_scalar_invocations` | Number of F32/no-scale scalar QK256 calls. |
| `qk256_i8s_scaled_scalar_invocations` | Number of production scaled I2_S x I8_S scalar QK256 calls. |
| `qk256_scalar_gemm_invocations` | Number of scalar QK256 GEMM/prefill calls, split further by schema version when scaled GEMM exists. |
| `flat_weight_extract_count` | Number of times packed weight bytes are copied/flattened from tensor storage for dispatch. |
| `input_vec2_materialization_count` | Number of per-layer or per-call `Vec<Vec<f32>>` input materializations. |
| `output_vecvec_allocation_count` | Number of per-layer or per-call `Vec<Vec<f32>>` output allocations. |
| `workspace_reuse_count` | Number of calls that reused scalar workspace rather than allocating new scratch. |

## Steady-State Target

A mature scalar decode receipt should trend toward:

```text
qk256_i8s_scaled_scalar_invocations > 0
flat_weight_extract_count == 0 after model/view warmup
input_vec2_materialization_count == 0
output_vecvec_allocation_count == 0
workspace_reuse_count > 0 after first use
fallback_used == false for strict scalar
```

Until the target is reached, receipts must keep the counters visible so follow-on
performance PRs can show before/after movement without making speedup claims.

## Implementation Rails

- Keep packed QK256/I2_S weights packed on the hot path.
- Expose borrowed packed views from loader/model state rather than copying bytes
  during each forward call.
- Use flat contiguous input and output buffers with row stride instead of nested
  row vectors.
- Reuse scalar workspace for activation quantization, output scratch, and row
  scratch.
- Preserve generated token IDs or record exact divergence when a hot-path change
  alters execution order or allocation behavior.

## Acceptance

Hot-path PRs must include before/after counter evidence, strict fallback status,
claim boundaries, and rollback instructions. Performance PRs must preserve model
SHA, tokenizer source, prompt IDs, generated IDs, decoded text, requested and
selected kernel, thread count, and `speedup_claim=false` unless a separate
reviewed speedup proof exists.
