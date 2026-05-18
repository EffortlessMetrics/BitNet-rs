# BitNet CPU Scalar Performance Contract

Status: draft
Owner: cpu-proof campaign
Created: 2026-05-18
Linked proposal: n/a
Linked specs: docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md; docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md; docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md
Linked ADRs: n/a
Linked plan: plans/cpu-scalar/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines receipt-backed scalar baseline measurement without speedup claims.
Policy impact: No policy exception.

## Purpose

This specification defines how scalar CPU efficiency is measured without
overclaiming. The scalar target is accurate enough to be the oracle, usable
enough on machines without SIMD or when forcing scalar for diagnosis, and
instrumented enough that faster lanes can compare against it. It is not intended
to beat AVX2, AVX-512, CUDA, or other accelerated paths.

## Required profiles

Scalar-only benchmark receipts must support these profiles:

```text
micro_f32_gemv
micro_i8s_scaled_gemv
micro_scalar_gemm
layer_0_decode
prefill_128
prefill_512
first_token
decode_32
decode_128
warm_session
```

Profile receipts must identify whether they measure diagnostic F32/no-scale
scalar, production scaled I2_S × I8_S scalar, scalar prefill GEMM, transformer
support ops, or end-to-end decode.

## Required fields

Scalar performance receipts must include fields equivalent to:

```json
{
  "wall_ms": "...",
  "median_ms": "...",
  "p95_ms": "...",
  "prompt_tps": "...",
  "decode_tps": "...",
  "selected_kernel": "qk256-scalar-i8s-scaled-gemv",
  "fallback_used": false,
  "allocations": "...",
  "flat_weight_extract_count": "...",
  "thread_count": "...",
  "model_sha256": "...",
  "tokenizer_source": "..."
}
```

The receipt must also preserve requested kernel, selected backend, CPU feature
set, prompt length, generation length, batch size, and whether generated IDs
match the comparison run or diverge explicitly.

## Performance claim boundary

Scalar receipts are baselines. They must set or imply `speedup_claim=false`
unless a separate reviewed performance claim explicitly scopes the comparison.
A scalar performance PR must include before and after receipts for the same
prompt, model, tokenizer, and generation settings, and must keep generated IDs
unchanged or record exact divergence.

## Thread-count policy

Scalar thread counts must be measured, not assumed. Decode may remain
low-thread if extra threads increase cache or KV traffic. Prefill may use
row/tile threading only when receipts show that it preserves output parity and
improves the selected profile.
