# BITNET-SPEC-CPU-AVX2-HOTPATH: CPU AVX2 BitNet Hot-Path Proof

Status: proposed
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-avx2-bitnet/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Governs exact-profile CPU AVX2 BitNet support claims.
Policy impact: none

## Purpose

This spec defines when the Rust CPU AVX2 path for the official Microsoft BitNet
I2_S/QK256 artifact may be treated as the real production hot path instead of a
correctness-only or diagnostic path. The lane is fully working only when strict
normal Rust CPU inference proves all of the following in receipts:

- authoritative real GGUF loading;
- strict tokenizer authority;
- canonical packed QK256/I2_S layout;
- scalar packed correctness oracle;
- explicit AVX2 kernel selection;
- no hidden scalar, dequantized, diagnostic, mock, or reference-only fallback;
- stable scalar-versus-AVX2 generated-token parity;
- phase timings good enough for exact-profile promotion.

## Scope

This spec applies only to CPU AVX2 BitNet I2_S/QK256 work for the official
Microsoft BitNet artifact routed through the normal Rust CPU user path. It does
not govern CUDA, NPU, OpenVINO, Metal, Vulkan, server readiness, dense SLMs,
Qwen, or broad chat-quality claims.

## Target end state

AVX2 CPU is fully working for this lane when the official BitNet I2_S/QK256
model runs through strict Rust CPU inference with:

1. `requested_backend = "cpu"` and `selected_backend = "cpu-rust"`;
2. real GGUF loader mode and model SHA-256 identity;
3. strict tokenizer source identity;
4. `kernel_family = "i2_s"` or `"qk256"` as appropriate to the proof;
5. selected scaled AVX2 BitNet kernel IDs when inline scale is present;
6. `fallback_used = false` and `fallback_reason = null`;
7. generated-token parity against the scalar packed path;
8. phase timing receipts for promoted profiles.

## Strict fallback rules

Strict mode must fail closed. If the user requests AVX2 and the selected path is
scalar, dequantized, diagnostic, mock, reference-only, or no-scale F32 GEMV for
an inline-scale BitNet tensor, the run must fail rather than emit a warning-only
fallback.

Non-strict auto mode may choose scalar when AVX2 is unavailable, but the receipt
must set `fallback_used = true` with a concrete `fallback_reason` whenever the
requested route differs from the selected route.

## Required receipt fields

Every hot-path proof receipt must include these fields or their schema-equivalent
locations:

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

Hot-path receipts must also expose QK256 execution counters once
CPU-AVX2-HOTPATH-001 lands:

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

## Scalar parity gate

Scalar packed QK256 remains the correctness oracle. AVX2 optimization PRs must
compare against the canonical scalar packed path and preserve generated-token
parity for strict proof prompts. If token IDs drift, the PR must include an
explicit divergence receipt with first-divergence evidence and must not merge as
a silent optimization.

## Scaled I2_S x I8_S hot-path requirement

Real inline-scale BitNet inference uses the scaled I2_S x I8_S activation flow:
quantize activation to I8_S, compute the integer dot over packed I2_S data, and
apply the scale/sum correction. Therefore:

- the no-scale F32-style QK256 AVX2 GEMV is not a substitute for the scaled
  BitNet path;
- inline-scale tensors must select a scaled scalar or scaled AVX2 kernel ID;
- strict requested scaled AVX2 must error if the scaled AVX2 kernel cannot run;
- receipts must distinguish scaled I2_S x I8_S invocations from no-scale F32
  GEMV invocations.

## Performance promotion requirements

No performance claim may be promoted from this lane without phase receipts that
identify model, tokenizer, requested and selected backend, requested and selected
kernel, fallback status, CPU feature set, workload shape, and timing phases.
The initial governed profiles are:

```text
micro_qk256_scaled_gemv
layer_0_decode
prefill_128
prefill_512
first_token
decode_32
decode_128
warm_session_3_turns
```

Promotion is profile-by-profile. A passing microkernel profile does not prove
first-token, decode, prefill, warm-session, server, or broad local-answer
performance.

## Forbidden claims

This lane must not claim:

- GPU, NPU, OpenVINO, Metal, Vulkan, CUDA, or server readiness;
- dense SLM or Qwen readiness;
- general BitNet model support beyond the evidenced artifact;
- broad chat quality;
- speedup without accepted phase receipt review;
- AVX2 execution when counters show scalar or no-scale F32 substitution;
- fallback-free execution without explicit fallback fields.

## Acceptance examples

A strict AVX2 inline-scale proof is valid only if it reports a scaled AVX2
selected kernel, `fallback_used = false`, and `scaled_i8s_avx2_invocations > 0`
with scalar and no-scale substitution counters at zero for the audited hot path.

A strict AVX2 inline-scale proof is invalid if it reports an AVX2 selected kernel
while `scaled_i8s_avx2_invocations = 0`, if it uses only no-scale F32 AVX2 GEMV,
or if it records scalar invocations while claiming `fallback_used = false`.
