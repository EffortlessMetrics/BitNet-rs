# BitNet CPU Scalar Performance Contract

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`, `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-scalar/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines exact-profile scalar measurement requirements; does not claim speedup.
Policy impact: No policy exception.

## Purpose

This spec defines how scalar CPU performance is measured without overclaiming.
Scalar is the reference and fallback baseline. It is not expected to beat AVX2,
AVX-512, NEON, CUDA, or other accelerated paths.

Scalar performance evidence answers these questions:

- Is strict scalar usable enough for diagnosis or machines without SIMD?
- Which scalar phases dominate runtime?
- Did a hot-path cleanup reduce allocations or copies without changing outputs?
- Which exact scalar receipt should optimized lanes compare against?

## Required Profiles

Scalar-only receipts should cover these profiles as the lane matures:

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

Micro profiles may use synthetic shapes when they name shape, layout, and kernel.
Layer and decode profiles must preserve model and tokenizer identity when they
are used as inference evidence.

## Required Receipt Fields

Scalar performance receipts must include the runtime-performance contract fields
and these scalar-specific fields:

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

Receipts must also preserve:

- requested backend and selected backend;
- requested kernel and selected kernel;
- fallback reason;
- prompt token IDs and generated token IDs when generation is involved;
- decoded text for answer or decode profiles;
- phase timings for prefill, first token, decode, and warm session when present;
- scalar hot-path counters when the receipt schema supports them;
- `speedup_claim=false` unless a separate reviewed comparator proves otherwise.

## Performance Claim Boundary

Scalar receipts may claim only exact-profile scalar measurements, such as:

```text
strict scalar decode_32 on model SHA <sha> selected qk256-scalar-i8s-scaled-gemv with fallback=false measured median/p95 timing
```

Scalar receipts must not claim:

- speedup over SIMD or accelerator lanes;
- general CPU support for every model or artifact;
- broad answer quality;
- server readiness;
- GPU, NPU, or OpenVINO readiness;
- performance portability from one machine, thread count, prompt, or model to another.

## Before/After Rules

Every scalar performance PR must include comparable before/after evidence when
it changes the hot path:

```text
same model artifact
same model SHA
same tokenizer and prompt template
same prompt IDs
same generated IDs or explicit divergence
same selected scalar kernel unless kernel identity is the change
same strict fallback policy
speedup_claim=false unless separately reviewed
```

Allocation and hot-path counter improvements should be described as reductions in
copy/materialization/workspace behavior, not as speedup unless exact-profile
speedup proof exists.

## Acceptance

A scalar performance PR is acceptable when it records strict selected scalar
kernel identity, `fallback_used=false` for strict scalar, exact profile identity,
hot-path counters where available, generated-ID preservation or divergence, no
speedup claim, and a rollback path.
