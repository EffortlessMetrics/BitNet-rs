# CPU AVX2 BitNet Hot-Path Implementation Plan

Status: active
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md`, `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-avx2-bitnet/README.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Exact-profile CPU AVX2 BitNet support only after profile receipts pass review.
Policy impact: none

## Sequence

| Order | Work item | Title | Production delta |
| ---: | --- | --- | --- |
| 0 | CPU-AVX2-HOTPATH-000 | `docs(cpu): add AVX2 BitNet hot-path implementation plan` | Documentation, spec, status, and tracker rails only. |
| 1 | CPU-AVX2-HOTPATH-001 | `diag(cpu): record BitNet QK256 hot-path execution counters` | Runtime counters and receipt fields; no math changes. |
| 2 | CPU-AVX2-HOTPATH-002 | `receipts(cpu): validate AVX2 hot-path counters` | Receipt validation rejects hidden fallback and missing counters. |
| 3 | CPU-AVX2-HOTPATH-003 | `test(cpu): add scaled I2S-I8S AVX2 parity fixtures` | Scalar-oracle fixture suite for scaled I2_S x I8_S behavior. |
| 4 | CPU-AVX2-HOTPATH-004 | `feat(cpu): add AVX2 scaled I2S-I8S QK256 GEMV` | Direct AVX2 scaled GEMV API, gated by runtime CPU features. |
| 5 | CPU-AVX2-HOTPATH-005 | `feat(cpu): select scaled AVX2 QK256 kernel explicitly` | Kernel IDs and strict selection metadata for scaled AVX2. |
| 6 | CPU-AVX2-HOTPATH-006 | `feat(cpu): route inline-scale BitNet QK256 through scaled AVX2` | Transformer dispatch uses the scaled AVX2 selector when selected. |
| 7 | CPU-AVX2-HOTPATH-007 | `perf(cpu): cache QK256 packed views for CPU dispatch` | Remove avoidable per-call packed-byte and row materialization. |
| 8 | CPU-AVX2-HOTPATH-008 | `perf(cpu): add reusable BitNet CPU decode workspace` | Reuse activation, output, attention, logits, and optional code scratch. |
| 9 | CPU-AVX2-HOTPATH-009 | `bench(cpu): add strict AVX2 phase timing profiles` | Micro, layer, prefill, first-token, decode, and warm-session receipts. |
| 10 | CPU-AVX2-HOTPATH-010 | `docs(cpu): review AVX2 performance qualification` | Profile-by-profile promotion decisions without global speed claims. |
| 11 | CPU-AVX2-HOTPATH-011 | `test(cpu): add BitNet CPU answer corpus v2` | Broader classified answer corpus with scalar and AVX2 results. |
| 12 | CPU-AVX2-HOTPATH-012 | `test(cpu): add scalar-vs-AVX2 long decode parity` | Deterministic 16/32/128-token parity and first-divergence receipts. |
| 13 | CPU-AVX2-HOTPATH-013 | `perf(cpu): optimize BitNet QK256 prefill path` | Evidence-guided prefill work without hot-path dequantization. |
| 14 | CPU-AVX2-HOTPATH-014 | `diag(cpu): profile non-QK256 transformer CPU ops` | Rank support-op bottlenecks before non-QK256 optimization. |
| 15 | CPU-AVX2-HOTPATH-015 | `docs(cpu): publish AVX2 BitNet support status` | User-facing exact support status and claim boundary. |

## Work item: CPU-AVX2-HOTPATH-000

Status: active
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md`, `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/cpu-proof/active.toml`
Blocks: CPU-AVX2-HOTPATH-001
Blocked by: CPU-ANSWER-007

### Goal

Add repository-local docs, spec, status, and tracker rails for the CPU AVX2
BitNet hot-path campaign so later PRs do not rediscover scope or blur proof
families.

### Production delta

No runtime delta. This is documentation and tracker state only.

### Non-goals

- Do not change tokenizer or prompt policy.
- Do not change scalar semantics.
- Do not edit runtime dispatch, kernels, receipts, fixtures, or benchmark code.
- Do not claim speedup, server readiness, GPU, NPU, OpenVINO, dense SLM, Qwen,
  or broad chat quality.

### Acceptance

- `plans/cpu-avx2-bitnet/README.md` exists and names source links and operating
  rules.
- `plans/cpu-avx2-bitnet/implementation-plan.md` sequences PR-sized work from
  hot-path counters through status publication.
- `docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md` defines target end state, strict
  fallback rules, required receipt fields, scalar parity gates, scaled I2_S x
  I8_S hot-path requirements, performance promotion requirements, and forbidden
  claims.
- `docs/bitnet/BITNET_CPU_AVX2_STATUS.md` records current claim status without
  overclaiming.
- `docs/tracking/campaigns/cpu-proof/active.toml` includes the ready follow-on
  `CPU-AVX2-HOTPATH-001` counter work item.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
git diff --check
```

### Rollback

Revert the new plan/spec/status files and the CPU-AVX2-HOTPATH tracker entries.
No runtime state or generated receipts are affected.

## Work item: CPU-AVX2-HOTPATH-001

Status: ready after CPU-AVX2-HOTPATH-000 merges
Linked specs: `docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md`
Campaign: `docs/tracking/campaigns/cpu-proof/active.toml`
Blocks: CPU-AVX2-HOTPATH-002
Blocked by: CPU-AVX2-HOTPATH-000

### Goal

Record actual BitNet QK256 hot-path execution counters in strict scalar and
strict AVX2 answer-corpus receipts without changing math or making speed claims.

### Required counters

```text
qk256_f32_scalar_gemv_invocations
qk256_f32_avx2_gemv_invocations
qk256_i8s_scaled_scalar_invocations
qk256_i8s_scaled_avx2_invocations
qk256_flat_bytes_extracted_count
qk256_input_rows_materialized_count
qk256_output_rows_allocated_count
qk256_tensor_to_vec_count
```

### Receipt shape

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

### Acceptance

- Strict scalar proof emits scalar counters.
- Strict AVX2 proof emits the actual selected path counters.
- Receipts distinguish no-scale F32 GEMV from scaled I2_S x I8_S GEMV.
- Requested and selected kernel and fallback fields remain explicit.
- Answer corpus remains green and scalar-versus-AVX2 parity remains `failed=0`.
- No speed claim is made.
