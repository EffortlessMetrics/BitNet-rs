# BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT: CPU Scalar Kernel Contract

Status: proposed

Linked plan:
[CPU scalar implementation plan](../../plans/cpu-scalar/implementation-plan.md)

Linked references:
[CPU path plan](../bitnet/BITNET_CPU_PATH_PLAN.md),
[Kernel matrix](../bitnet/BITNET_KERNEL_MATRIX.md),
[Receipt fields](../bitnet/BITNET_RECEIPT_FIELDS.md)

## Purpose

This spec defines what `scalar` means for BitNet CPU inference. The scalar lane
is the trusted CPU oracle and a usable fallback path for machines without SIMD or
for diagnosis. It is not a hidden replacement for requested accelerated kernels
and it is not a speedup claim.

A strict scalar BitNet run proves:

```text
real GGUF
strict tokenizer
canonical packed QK256/I2_S layout
BitNet.cpp-style scaled I2_S x I8_S scalar math
deterministic CPU transformer ops
fallback_used=false
requested_kernel == selected_kernel
answer corpus passes
long decode is stable
phase timings are measured
no hidden dequantized/reference substitution
```

## Scalar Path Taxonomy

There are two scalar paths and receipts must not blur them.

| Path | Role | Production meaning |
| --- | --- | --- |
| F32/no-scale QK256 scalar | Dequant-style QK256 diagnostic/oracle path | Useful diagnostic/reference path; not a substitute for scaled BitNet I8_S math. |
| Scaled I2_S x I8_S scalar | BitNet.cpp-style real BitNet matmul semantics | Production scalar BitNet decode and prefill path. |

The scaled path quantizes each activation row to I8_S, records activation scale
and activation sum, computes the integer dot over packed I2_S codes, then
applies:

```text
(dot - act_sum) / act_scale * weight_scale
```

## Required Scalar Kernel IDs

New receipts and selection metadata must use precise kernel IDs:

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

Compatibility aliases may remain only for existing callers and historical
receipts:

```text
qk256-scalar-gemv -> qk256-scalar-f32-gemv
qk256-scalar-gemm -> qk256-scalar-f32-gemm
```

New strict BitNet I2_S receipts must not report `qk256-scalar-gemv` when the
scaled I8_S path ran.

## Kernel Selection Contract

Strict mode must enforce requested/selected identity:

| Request | Expected selected kernel | Fallback |
| --- | --- | --- |
| `qk256-scalar-f32-gemv` | `qk256-scalar-f32-gemv` | `false` |
| `qk256-scalar-f32-gemm` | `qk256-scalar-f32-gemm` | `false` |
| `qk256-scalar-i8s-scaled-gemv` | `qk256-scalar-i8s-scaled-gemv` | `false` |
| `qk256-scalar-i8s-scaled-gemm` | `qk256-scalar-i8s-scaled-gemm` | `false` |
| strict accelerated kernel unavailable | error | not scalar substitution |
| non-strict accelerated kernel unavailable | scalar may be selected only with explicit `fallback_used=true` and reason | `true` |
| unknown kernel | error | not fallback |

Auto mode may select scalar when no accelerated path is available, but receipts
must still record the actual precise scalar ID and whether that was a fallback
from a more specific requested kernel.

## Required Scalar Proof Types

A scalar CPU lane is incomplete until these proof types exist or are explicitly
tracked as pending:

```text
layout proof
pack/unpack proof
F32 no-scale GEMV proof
scaled I2_S x I8_S GEMV proof
scalar GEMM proof
tail-column proof
repeatability proof
answer-corpus proof
long-decode proof
phase benchmark proof
```

## Acceptance Requirements

Runtime PRs that implement this spec must provide:

- scalar unit tests and scalar fixtures;
- strict answer-corpus or scoped receipt evidence;
- `fallback_used=false` evidence for strict scalar runs;
- receipt schema validation that records requested and selected kernel IDs;
- `git diff --check`;
- claim boundaries and rollback path.

Performance PRs must additionally provide before/after receipts for the same
prompt, model, tokenizer, prompt IDs, generated IDs, decoded text, backend, and
fallback status. `speedup_claim` must remain `false` unless separately reviewed.

## Non-Goals

```text
No SIMD proof.
No GPU/NPU proof.
No speedup claim.
No dense SLM proof from BitNet scalar.
No broad chat-quality claim from tiny corpus.
```

## Claim Boundary

This spec may support the claim that scalar CPU BitNet behavior is precisely
identified, selectable, and receipt-backed after the linked implementation work
lands. It must not be used to claim accelerated performance, GPU/NPU execution,
server readiness, or broad answer quality.
