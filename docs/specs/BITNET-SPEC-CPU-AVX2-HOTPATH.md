# BITNET-SPEC-CPU-AVX2-HOTPATH: CPU AVX2 BitNet hot-path proof

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: BITNET-SPEC-0013-model-onboarding-proof-ladder.md, BITNET-SPEC-0014-runtime-performance-contract.md
Linked ADRs: ../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md
Linked plan: ../../plans/cpu-avx2-bitnet/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CPU AVX2 BitNet I2_S/QK256 remains exact-profile only until receipts promote individual profiles.
Policy impact: none

## Purpose

This spec defines when the Rust CPU AVX2 lane may say that the official
Microsoft BitNet I2_S/QK256 model is using the production optimized BitNet hot
path. It narrows the existing CPU proof lane from "correctness exists" to "the
normal Rust CPU user path selected the real scaled BitNet AVX2 kernels, did not
fall back, preserved scalar parity, and emitted phase evidence before any
performance claim."

## Target end state

The AVX2 CPU path is fully working only when the official Microsoft BitNet
I2_S/QK256 artifact runs through the normal Rust CPU user path with all of the
following properties:

- real GGUF loader authority and strict tokenizer authority are used;
- generated answers are intelligible under the governed answer corpus;
- selected kernels are AVX2 BitNet kernels for the executed BitNet hot path;
- strict mode has no hidden scalar, dequantized, diagnostic, mock, or
  reference-only fallback;
- scalar and AVX2 generated token IDs remain equal, or the receipt records the
  first divergence with enough evidence to block promotion;
- prefill, first-token, and decode performance are measured per profile before
  a profile is promoted.

## Scope

This spec applies only to the CPU AVX2 BitNet I2_S/QK256 proof family. It does
not apply to CUDA, NPU, OpenVINO, A770, M4, dense SLMs, Qwen, TL1/TL2,
GPU-int2, server readiness, or broad chat-quality claims.

## Scalar oracle gate

The canonical scalar packed path remains the correctness oracle. Every AVX2
kernel added by this lane must compare against scalar fixtures and must preserve
strict scalar-versus-AVX2 generated-token parity before it can be used as an
optimization claim.

An AVX2 optimization PR must not merge as a promotion PR if it changes generated
IDs without a divergence receipt and explicit blocker classification.

## Strict fallback rules

Strict mode fails closed. If the request asks for AVX2 and strict mode is in
force, the run must fail when the selected path is scalar, dequantized,
diagnostic-only, mock/reference-only, or any other non-AVX2 substitute.
Warning-only fallback is not acceptable in proof runs.

Non-strict fallback may select scalar when AVX2/FMA or other required CPU
features are unavailable, but the receipt must set `fallback_used=true` and must
record a concrete `fallback_reason`.

## Required receipt fields

Every CPU AVX2 BitNet proof receipt must retain the existing strict CPU fields
and include the following logical values:

```json
{
  "requested_backend": "cpu",
  "selected_backend": "cpu-rust",
  "requested_kernel": "...",
  "selected_kernel": "...",
  "kernel_family": "i2_s|qk256",
  "runtime_api": "cpu",
  "fallback_used": false,
  "fallback_reason": null,
  "model": {
    "loader_mode": "real_gguf",
    "quant_format": "i2_s",
    "sha256": "..."
  },
  "tokenizer": {
    "source": "...",
    "strict": true
  }
}
```

Hot-path proof receipts must additionally distinguish the no-scale F32-style
QK256 path from the scaled BitNet I2_S x I8_S path:

```json
{
  "qk256_hot_path": {
    "scaled_i8s_scalar_invocations": 0,
    "scaled_i8s_avx2_invocations": 0,
    "f32_scalar_invocations": 0,
    "f32_avx2_invocations": 0,
    "flat_bytes_extracted_count": 0,
    "input_rows_materialized_count": 0,
    "output_rows_allocated_count": 0,
    "tensor_to_vec_count": 0
  }
}
```

The exact schema may nest these fields under the repository's receipt model, but
`bitnet receipts explain` and validators must be able to report the same facts.

## Scaled I2_S x I8_S hot-path requirement

The official BitNet I2_S/QK256 inline-scale inference path is not satisfied by
labeling the existing no-scale F32-style AVX2 GEMV as BitNet AVX2. For inline
scale tensors, the governed production path is:

```text
activation f32 -> per-token I8_S quantization
packed I2_S weights + I8_S activation integer dot
inline weight scale and sum correction
f32 output
```

The lane must prove whether strict real BitNet AVX2 inference actually executes
an optimized scaled I2_S x I8_S AVX2 implementation. If receipts show that
inline-scale inference only executed the no-scale F32 AVX2 path, scalar
substitution, tensor dequantization, or per-call materialized rows, the AVX2
hot-path claim remains blocked.

## Hot-path validator failures

Receipt validation must fail for CPU AVX2 hot-path proof when any of the
following are true:

- requested AVX2 selected scalar in strict mode;
- the selected kernel name says AVX2 but AVX2 invocation counters are zero;
- inline-scale BitNet proof records only no-scale F32 AVX2 invocations;
- `fallback_used=false` while counters show scalar substitution;
- audited hot-path materialization counters exceed the profile's accepted
  boundary.

## Performance promotion requirements

Performance claims require phase receipts. At minimum, a profile promotion must
record model load, tokenizer load, prompt render, prefill, first-token, decode,
tokens/sec where applicable, selected backend, selected kernel, fallback status,
CPU feature set, model identity, workload shape, and profile name.

Promotions are exact-profile only. A speedup for `micro_qk256_scaled_gemv` does
not imply speedup for first token, decode, prefill, warm sessions, server,
other models, or other quantization formats.

## Forbidden claims

This lane must not claim:

- CUDA, NPU, OpenVINO, A770, M4, Metal, GPU-int2, TL1/TL2, or dense SLM support;
- server readiness or deployment readiness;
- broad chat quality;
- generic BitNet model support beyond the exact artifact and profile proven;
- speedup or throughput without accepted exact-profile receipts;
- AVX2 execution when counters show scalar, no-scale, dequantized, diagnostic,
  mock, or reference-only execution.
