# BitNet CPU Scalar Kernel Contract

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-scalar/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines the scalar CPU correctness/oracle contract; does not promote a support tier by itself.
Policy impact: No policy exception.

## Purpose

This spec defines what the BitNet-rs CPU lane means by **scalar**. Scalar is the
trusted CPU oracle and usable CPU fallback path for packed QK256/I2_S BitNet
inference. It is not a SIMD, GPU, NPU, server, or speedup claim.

A strict scalar BitNet run must prove all of the following without substituting a
diagnostic dense/dequantized route:

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

## Scalar Kernel Families

The scalar lane contains two different paths. Receipts, dispatch, and tests must
not blur them.

| Path | Meaning | Production role |
| --- | --- | --- |
| F32 no-scale QK256 scalar | Unpacks QK256 ternary codes and multiplies by F32 activations without BitNet.cpp I8_S activation scaling. | Diagnostic/reference path for layout and unpack behavior. |
| Scaled I2_S x I8_S scalar | Quantizes activations to I8_S, records activation scale and sum, performs integer dot over packed I2_S codes, then applies `(dot - act_sum) / act_scale * weight_scale`. | Production scalar BitNet matmul path. |

Real BitNet I2_S tensors with inline scale must use the scaled I2_S x I8_S
scalar path when scalar is selected. They must not be routed through the F32
no-scale scalar path unless the run is explicitly diagnostic and receipts say so.

## Required Scalar Kernel IDs

New receipts and dispatch decisions must use precise scalar IDs:

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

Compatibility aliases may remain for older callers and historic receipts:

| Legacy ID | Compatibility target | New receipt behavior |
| --- | --- | --- |
| `qk256-scalar-gemv` | `qk256-scalar-f32-gemv` | Do not emit for new F32/no-scale receipts except as an alias field if required. |
| `qk256-scalar-gemm` | `qk256-scalar-f32-gemm` | Do not emit for new F32/no-scale receipts except as an alias field if required. |

The compatibility aliases do not authorize ambiguous production receipts. A real
scaled BitNet run must name `qk256-scalar-i8s-scaled-gemv` or
`qk256-scalar-i8s-scaled-gemm` as applicable.

## Selection Contract

Strict scalar requests must select scalar and must not be marked as fallback:

```text
requested_kernel = qk256-scalar-i8s-scaled-gemv
selected_kernel = qk256-scalar-i8s-scaled-gemv
fallback_used = false
fallback_reason = null
```

Strict accelerated requests must fail rather than silently selecting scalar when
the requested accelerated kernel is unavailable. Non-strict accelerated requests
may fall back to scalar only when the receipt records `fallback_used=true`, the
fallback reason, and the actual scalar kernel ID.

## Required Scalar Proof Types

Each scalar implementation stage must preserve or add evidence for these proof
types:

| Proof type | Requirement |
| --- | --- |
| layout proof | Canonical QK256/I2_S block geometry, row stride, alignment, and tail handling are shared with loader/layout authority. |
| pack/unpack proof | Packed bytes decode to the verified ternary code map `0 -> -1`, `1 -> 0`, `2 -> +1`, `3 -> 0`. |
| F32 no-scale GEMV proof | Diagnostic scalar GEMV output is deterministic and identifies the F32/no-scale kernel. |
| scaled I2_S x I8_S GEMV proof | Production scalar decode path matches BitNet.cpp-style I8_S activation quantization and scaled output formula. |
| scalar GEMM proof | Prefill path is deterministic; scaled GEMM must equal repeated scaled GEMV once implemented. |
| tail-column proof | Non-multiple-of-256 columns have exact, repeatable tail behavior. |
| repeatability proof | Same inputs, model, tokenizer, prompt, and seed/greedy policy produce identical IDs and receipts. |
| answer-corpus proof | Strict scalar answer-corpus artifacts record prompt IDs, generated IDs, decoded text, quality gate, tokenizer source, model SHA, selected kernel, and fallback status. |
| long-decode proof | Longer scalar decode remains stable with no hidden fallback or dequantized substitution. |
| phase benchmark proof | Scalar profiles record timings and counters without speedup claims. |

## Claim Boundary

Scalar work may claim:

- scalar is the CPU correctness oracle for optimized packed CPU kernels after the
  relevant proof receipts pass;
- strict scalar selection can be used for diagnosis when receipts prove
  `fallback_used=false` and requested/selected kernel equality;
- scalar performance has been measured for exact profiles after scalar-only
  receipts exist.

Scalar work must not claim:

- SIMD, AVX2, AVX-512, NEON, CUDA, Metal, OpenVINO, NPU, or server readiness;
- speedup over another lane;
- dense SLM correctness from BitNet scalar evidence;
- broad chat quality from a tiny answer corpus;
- new numeric tolerances without updating the parity policy.

## Acceptance

A scalar-kernel implementation PR satisfies this contract only when it includes
scoped scalar tests or receipts, strict fallback evidence, exact kernel identity,
claim boundaries, and rollback notes. Runtime PRs must also validate receipt
schema and run `git diff --check`.
