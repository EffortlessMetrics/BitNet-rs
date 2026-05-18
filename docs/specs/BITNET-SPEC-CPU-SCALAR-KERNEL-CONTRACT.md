# BitNet CPU Scalar Kernel Contract

Status: draft
Owner: cpu-proof campaign
Created: 2026-05-18
Linked proposal: n/a
Linked specs: docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md; docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md; docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md
Linked ADRs: n/a
Linked plan: plans/cpu-scalar/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines the CPU scalar oracle/fallback claim boundary; does not change support tier by itself.
Policy impact: No policy exception.

## Purpose

This specification defines what `scalar` means for the BitNet CPU lane. The
scalar lane is the trusted CPU oracle and a usable fallback path for machines
without SIMD or for diagnosis. It is not a speedup lane, not a GPU/NPU lane, and
not a dense-reference substitute for packed BitNet execution.

A strict scalar BitNet run must prove all of the following before it can be used
as scalar oracle evidence:

```text
real GGUF
strict tokenizer
canonical packed QK256/I2_S layout
BitNet.cpp-style scaled I2_S × I8_S scalar math
deterministic CPU transformer ops
fallback_used=false
requested_kernel == selected_kernel
answer corpus passes
long decode is stable
phase timings are measured
no hidden dequantized/reference substitution
```

## Scalar path taxonomy

There are two scalar paths, and receipts, dispatch metadata, tests, and docs must
not blur them.

| Path | Role | Production meaning |
| --- | --- | --- |
| F32/no-scale QK256 scalar | Dequant-style QK256 diagnostic/reference path | Useful for layout and packed-code diagnostics. |
| Scaled I2_S × I8_S scalar | BitNet.cpp-style real BitNet matmul semantics | Production scalar BitNet path and optimized-kernel oracle. |

The scaled path quantizes activation rows to I8_S, records activation scale and
sum, computes the integer dot over packed I2_S codes, and applies:

```text
(dot - act_sum) / act_scale * weight_scale
```

Real BitNet I2_S tensors with inline scale must use the scaled path. The
F32/no-scale scalar path must not be substituted for scaled BitNet I8_S math in
strict runs.

## Required scalar kernel IDs

New receipts and routing surfaces must use precise kernel IDs:

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
| `qk256-scalar-gemv` | Alias for `qk256-scalar-f32-gemv`. |
| `qk256-scalar-gemm` | Alias for `qk256-scalar-f32-gemm`. |

Compatibility aliases must not appear in newly emitted strict BitNet receipts
when the precise scalar path is known.

## Selection contract

Strict scalar selection is not fallback. A strict request for
`qk256-scalar-i8s-scaled-gemv` must select that same kernel ID and emit
`fallback_used=false`.

Strict accelerated requests must not silently select scalar. If AVX2, AVX-512,
CUDA, or another accelerated kernel is requested strictly and is unavailable,
the run must fail rather than choose scalar with a warning. Non-strict or auto
selection may use scalar only when receipts explicitly record the requested
kernel, selected kernel, `fallback_used`, and fallback reason.

## Required scalar proof types

The scalar lane requires these proof types before broad oracle claims:

```text
layout proof
pack/unpack proof
F32 no-scale GEMV proof
scaled I2_S×I8_S GEMV proof
scalar GEMM proof
tail-column proof
repeatability proof
answer-corpus proof
long-decode proof
phase benchmark proof
```

Proof receipts must preserve prompt IDs, generated IDs, decoded text, tokenizer
source, model SHA, backend, selected kernel, requested kernel, and fallback
status for answer and performance evidence.

## Non-goals

This specification does not authorize:

```text
No SIMD proof.
No GPU/NPU proof.
No speedup claim.
No dense SLM proof from BitNet scalar.
No broad chat-quality claim from tiny corpus.
```

## Acceptance

A PR that implements this contract must show that precise scalar kernel IDs
exist, the scaled I2_S × I8_S path can be selected explicitly, strict scalar is
not marked fallback, and strict accelerated requests cannot silently select
scalar.
