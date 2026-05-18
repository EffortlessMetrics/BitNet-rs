# BitNet CPU AVX-512 Kernel Contract

Status: Draft
Owner: BitNet CPU proof campaign
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/amd-9950x3d-cpu-roadmap.md`, `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-avx512/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CPU AVX-512 remains explicit-proof only until receipts promote profile-specific use.
Policy impact: No policy exception.

## Purpose

This spec defines what counts as AVX-512 CPU kernel proof for BitNet-rs. It prevents AVX-512 detection, roadmap labels, or receipt labels from being treated as optimized AVX-512 execution evidence.

The AVX-512 lane is a wider x86 CPU lane inside the existing BitNet CPU proof family. It must inherit scalar packed correctness, compare against AVX2, reject hidden fallback in strict mode, and avoid speed claims until phase and sustained receipts prove them.

## Claim Boundary

The following statements are hard non-goals unless a later spec and receipt set explicitly upgrades them:

- AVX-512 detection does not prove AVX-512 kernel execution.
- AVX-512 dispatch selection does not prove optimized packed BitNet execution.
- AVX-512 execution does not prove speedup.
- AVX-512 microbenchmark speed does not prove decode, first-token, or prefill speed.
- AVX-512 short-burst speed does not prove sustained performance.
- AVX-512 CPU proof does not prove CUDA, OpenCL, OpenVINO, NPU, server, or production readiness.
- AVX2 proof does not prove AVX-512 proof.

## Proof Vocabulary

| Proof class | Required evidence | May claim | Must not claim |
| --- | --- | --- | --- |
| AVX-512 detection proof | Runtime CPUID/subfeature receipt records required features. | AVX-512-capable CPU features were detected. | An AVX-512 kernel ran. |
| AVX-512 dispatch proof | Request, strictness, selected kernel ID, and fallback status are recorded. | The selector chose an AVX-512 kernel ID. | The selected path produced correct or faster outputs. |
| AVX-512 kernel execution proof | Kernel-specific invocation counters prove the AVX-512 path executed with `fallback_used=false`. | The named AVX-512 kernel executed. | It is faster than scalar or AVX2. |
| AVX-512 parity proof | Scalar-vs-AVX512 fixtures and answer/logit/token parity artifacts pass or classify divergence. | AVX-512 preserves the scoped reference behavior. | Other profiles or devices are equivalent. |
| AVX-512 performance proof | Micro, layer, prefill, first-token, and decode phase receipts compare scalar, AVX2, and AVX-512. | AVX-512 is faster for the measured profile only. | Global auto-selection or sustained behavior is proven. |
| AVX-512 sustained-performance proof | Long-duration receipts include power, thermal, affinity, scheduler, and cache-domain context. | AVX-512 is sustained for the measured profile and hardware context. | Other CPUs or power modes behave the same. |

## Stable Kernel IDs

The AVX-512 CPU lane reserves these stable QK256 kernel IDs:

```rust
pub const QK256_AVX512_F32_GEMV_KERNEL_ID: &str =
    "qk256-avx512-f32-gemv";

pub const QK256_AVX512_I8S_SCALED_GEMV_KERNEL_ID: &str =
    "qk256-avx512-i8s-scaled-gemv";

pub const QK256_AVX512_I8S_SCALED_GEMM_KERNEL_ID: &str =
    "qk256-avx512-i8s-scaled-gemm";
```

The F32/no-scale GEMV kernel may land first because it mirrors the existing AVX2 smoke path. The scaled I2_S x I8_S GEMV kernel is the decode hot-path target. The scaled GEMM kernel is the prefill target.

If a VNNI implementation lands later, it must use a distinct kernel ID, for example `qk256-avx512vnni-i8s-scaled-gemv`, because VNNI changes the accumulation strategy and must not silently replace the baseline AVX-512BW path.

## Required Receipt Fields

Strict AVX-512 CPU receipts must distinguish requested, selected, detected, required, used, and executed state. A proof receipt must include fields equivalent to:

```json
{
  "requested_backend": "cpu",
  "selected_backend": "amd-9950x3d-cpu-avx512",
  "requested_kernel": "qk256-avx512-i8s-scaled-gemv",
  "selected_kernel": "qk256-avx512-i8s-scaled-gemv",
  "fallback_used": false,
  "fallback_reason": null,
  "cpu": {
    "arch": "x86_64",
    "features_detected": ["avx512f", "avx512bw", "avx512vl", "avx512vnni"],
    "features_required": ["avx512f", "avx512bw"],
    "features_used": ["avx512f", "avx512bw"],
    "threads": 16
  },
  "qk256_hot_path": {
    "f32_scalar_invocations": 0,
    "f32_avx2_invocations": 0,
    "f32_avx512_invocations": 0,
    "i8s_scaled_scalar_invocations": 0,
    "i8s_scaled_avx2_invocations": 0,
    "i8s_scaled_avx512_invocations": 420
  },
  "parity": {
    "reference_kernel": "qk256-scalar-i8s-scaled-gemv",
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0,
    "generated_token_agreement": true
  }
}
```

A label such as `i2_s-avx512-reference` is diagnostic unless it is backed by selected kernel IDs and invocation counters for the concrete AVX-512 QK256 hot path under test.

## Required Runtime Behavior

1. A strict request for an AVX-512 kernel must fail if the required AVX-512 features are unavailable or if scalar/AVX2 would be selected instead.
2. A non-strict request may fall back only when the receipt records `fallback_used=true`, the fallback reason, and the selected non-AVX512 kernel.
3. Auto-selection must not promote AVX-512 solely because CPUID reports AVX-512 support.
4. AVX-512 functions must be target-feature gated and runtime checked; workspace-wide `target-cpu=native` is not proof.
5. AVX-512 receipts must remain CPU-only and must not imply GPU, NPU, OpenVINO, server, or production support.

## Acceptance For The Lane

AVX-512 is considered first-class only when all of the following exist:

1. `avx512` or a concrete AVX-512 QK256 kernel can be explicitly requested.
2. Runtime feature detection validates the required subfeatures.
3. The selected AVX-512 kernel has a stable kernel ID.
4. Strict mode rejects scalar/AVX2 fallback.
5. Scalar-vs-AVX512 parity passes on synthetic and real QK256 fixtures.
6. AVX2-vs-AVX512 comparison receipts exist.
7. Answer-corpus receipts include AVX-512 invocation counters.
8. Phase benchmark receipts show micro, layer, prefill, first-token, and decode behavior.
9. Sustained 9950X3D receipts include power, thermal, affinity, scheduler, and cache-domain context.
10. User-facing docs distinguish detected, selected, executed, faster, and sustained AVX-512 states.
