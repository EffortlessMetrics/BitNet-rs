# BitNet CPU Scalar Kernel Contract

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-scalar/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines scalar CPU oracle requirements; does not upgrade public support alone.
Policy impact: No policy exception.

## Purpose

Scalar CPU BitNet is the trusted correctness oracle and the portable fallback
path for hosts without an accelerated kernel. It is not an accidental fallback,
a dense dequantized substitute, or a speedup claim. A strict scalar run must
prove real GGUF loading, strict tokenizer authority, canonical packed
QK256/I2_S layout, BitNet.cpp-style scaled I2_S × I8_S scalar math,
deterministic transformer support ops, `fallback_used=false`, and exact
requested/selected scalar kernel identity.

## Scalar Kernel Families

There are two scalar QK256 paths and receipts must never blur them:

| Path | Role | Production meaning |
| --- | --- | --- |
| F32/no-scale QK256 scalar | dequant-style diagnostic and reference path | Useful for pack/unpack and no-scale kernel checks. |
| scaled I2_S × I8_S scalar | BitNet.cpp-style real BitNet matmul semantics | Production scalar BitNet decode/prefill path. |

Real BitNet I2_S tensors with inline scale must use the scaled I8S path. They
must not silently route through F32/no-scale scalar compute.

## Required Stable Kernel IDs

New receipts and routing metadata must use precise IDs:

```rust
pub const QK256_SCALAR_F32_GEMV_KERNEL_ID: &str =
    "qk256-scalar-f32-gemv";

pub const QK256_SCALAR_F32_GEMM_KERNEL_ID: &str =
    "qk256-scalar-f32-gemm";

pub const QK256_SCALAR_I8S_SCALED_GEMV_KERNEL_ID: &str =
    "qk256-scalar-i8s-scaled-gemv";

pub const QK256_SCALAR_I8S_SCALED_GEMM_KERNEL_ID: &str =
    "qk256-scalar-i8s-scaled-gemm";
```

Compatibility aliases may remain for older callers only:

| Compatibility ID | Canonical meaning |
| --- | --- |
| `qk256-scalar-gemv` | alias for `qk256-scalar-f32-gemv` |
| `qk256-scalar-gemm` | alias for `qk256-scalar-f32-gemm` |

Receipts produced after this contract should prefer the precise IDs and should
include compatibility IDs only as explicit alias metadata when needed.

## Scaled I2_S × I8_S Semantics

The scaled scalar production path follows the BitNet.cpp-style formula:

1. quantize each activation row to I8_S;
2. record activation scale and activation sum;
3. compute the integer dot over packed I2_S codes using the canonical QK256
   block layout and ternary code map;
4. produce output with `(dot - act_sum) / act_scale * weight_scale`;
5. preserve documented tail-column behavior and repeatability.

Any future SIMD, GPU, NPU, or graph lane that claims native packed BitNet
correctness compares against this scalar path. Scalar does not need to compare
to an optimized lane to be considered correct.

## Required Proof Types

Scalar PRs must build toward these proof types:

- layout proof;
- pack/unpack proof;
- F32 no-scale GEMV proof;
- scaled I2_S × I8_S GEMV proof;
- scalar GEMM proof;
- tail-column proof;
- repeatability proof;
- answer-corpus proof;
- long-decode proof;
- phase benchmark proof.

## Strict Selection Rules

- Strict scalar requests must select the exact scalar kernel and must set
  `fallback_used=false`.
- Strict accelerated requests must fail rather than silently selecting scalar
  when the accelerated kernel is unavailable.
- Non-strict accelerated requests may fall back to scalar only when the receipt
  records `fallback_used=true`, the fallback reason, and the selected precise
  scalar kernel ID.
- Unknown kernel IDs are errors.

## Non-goals

This contract does not prove:

- SIMD correctness or speed;
- GPU, NPU, server, or graph execution;
- scalar speedup over optimized lanes;
- dense SLM quality from BitNet scalar work;
- broad chat quality from a tiny answer corpus.
