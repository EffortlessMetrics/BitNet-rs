# BITNET-SPEC-CPU-SCALAR-PERFORMANCE: CPU Scalar Performance Contract

Status: proposed

Linked plan:
[CPU scalar implementation plan](../../plans/cpu-scalar/implementation-plan.md)

Linked specs:
[CPU scalar kernel contract](BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md),
[CPU scalar hot-path contract](BITNET-SPEC-CPU-SCALAR-HOTPATH.md),
[CPU scalar parity contract](BITNET-SPEC-CPU-SCALAR-PARITY.md)

## Purpose

This spec defines scalar performance evidence without overclaiming. Scalar is
the reference/performance baseline for machines without SIMD and for diagnostic
runs. Scalar receipts must be stable and phase-aware, but they do not carry a
speedup claim.

## Required Profiles

Scalar performance work should cover these profiles as the implementation
matures:

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

Profile names must be stable enough for before/after receipt comparison.

## Required Receipt Fields

Scalar performance receipts must include these fields or their schema-approved
equivalents:

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

The receipt must also preserve prompt IDs, generated IDs, decoded text,
requested backend/kernel, selected backend/kernel, tokenizer strictness, and
model artifact identity whenever it is tied to answer or decode evidence.

## Performance Rails

Scalar performance PRs must obey these rails:

1. No speedup claim from scalar receipts.
2. No AVX2, AVX-512, CUDA, Metal, OpenCL, OpenVINO, or NPU selection in
   scalar-only receipts.
3. Same prompt, model, tokenizer, prompt IDs, generated IDs, decoded text,
   backend, and fallback status for before/after comparisons, or an explicit
   divergence classification.
4. No whole-matrix dequantization in final steady-state scalar proof.
5. No broad answer-quality claim from tiny corpora.
6. No GPU/NPU/server claims from scalar work.

## Acceptance Requirements

A scalar performance receipt can be used as a baseline only when it records:

- phase/profile name;
- wall/median/p95 timing or an explicit not-run status;
- selected precise scalar kernel ID;
- `fallback_used=false` for strict scalar runs;
- allocation and hot-path counters from the scalar hot-path contract;
- thread count and CPU identity;
- model SHA-256 and tokenizer source for model-level runs;
- `speedup_claim=false` unless a separate reviewed speedup proof exists.

## Non-Goals

This spec does not define SIMD speedup thresholds, GPU/NPU performance, dense
SLM performance, or server readiness performance. Those belong to their own
proof families and receipts.
