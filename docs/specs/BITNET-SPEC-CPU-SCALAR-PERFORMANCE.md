# BitNet CPU Scalar Performance Contract

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-scalar/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines scalar baseline receipts; does not claim scalar speedup.
Policy impact: No policy exception.

## Purpose

Scalar performance evidence exists to make the oracle usable and to give faster
lanes a stable baseline. It must not overclaim. Scalar receipts are baseline and
fallback evidence, not proof that scalar beats AVX2, AVX-512, NEON, GPU, NPU, or
server paths.

## Required Profiles

Scalar-only benchmark evidence should cover:

- `micro_f32_gemv`;
- `micro_i8s_scaled_gemv`;
- `micro_scalar_gemm`;
- `layer_0_decode`;
- `prefill_128`;
- `prefill_512`;
- `first_token`;
- `decode_32`;
- `decode_128`;
- `warm_session`.

Profiles may land incrementally, but every receipt must name its phase and must
record whether it uses F32/no-scale scalar, scaled I8S scalar GEMV, or scaled
I8S scalar GEMM.

## Required Receipt Fields

Scalar performance receipts must include these fields or explicitly document why
a field is unavailable for the profile:

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

When comparing before/after performance PRs, receipts must preserve prompt IDs,
generated IDs, decoded text, tokenizer source, model SHA, backend, selected
kernel, and fallback status, or they must record exact divergence.

## Performance Rails

- `speedup_claim=false` is the default for scalar receipts.
- Auto-selected AVX2/AVX-512/NEON receipts are not scalar-only baselines.
- A scalar receipt must not select CUDA, NPU, server, or graph lanes.
- Thread-count experiments must record the chosen thread count and may choose a
  low decode thread count if evidence shows it is better.
- Allocation and hot-path counters are part of the performance claim boundary.
