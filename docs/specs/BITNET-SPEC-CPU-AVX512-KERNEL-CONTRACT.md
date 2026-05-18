# BitNet CPU AVX-512 Kernel Contract

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs:
- `docs/specs/amd-9950x3d-cpu-roadmap.md`
- `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`
Linked ADRs: n/a
Linked plan:
- `plans/cpu-avx512/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: AVX-512 remains explicit/campaign-only until parity, phase, and sustained receipts justify profile-scoped promotion.
Policy impact: No policy exception.

## Purpose

This spec defines what counts as AVX-512 CPU kernel proof for BitNet-rs.
It prevents detection probes, labels, or short diagnostic receipts from being
reported as optimized AVX-512 kernel execution or speed proof.

The AVX-512 lane is a wider x86 CPU lane for QK256/I2_S work. It is not a
separate accelerator family and it does not imply CUDA, OpenCL, OpenVINO, NPU,
server, or broad chat-quality readiness.

## Proof vocabulary

| Proof term | Required evidence | Explicit non-proof |
|---|---|---|
| AVX-512 detection proof | Runtime feature probe records the required AVX-512 subfeatures for the host. | Detection alone does not prove dispatch or execution. |
| AVX-512 dispatch proof | A strict or explicit request records requested kernel, selected kernel, fallback status, fallback reason, and required/detected CPU features. | A receipt label without dispatch metadata is not dispatch proof. |
| AVX-512 kernel execution proof | Invocation counters show the AVX-512 QK256 hot path ran with count greater than zero and scalar/AVX2 counters did not secretly satisfy the request. | Selecting a backend name is not execution proof. |
| AVX-512 parity proof | Scalar-vs-AVX512 parity passes for synthetic fixtures, real QK256 fixtures, and generated-token/logits evidence where required by the plan item. | Matching generated IDs alone do not prove logits parity. |
| AVX-512 performance proof | Phase-specific receipts compare scalar, AVX2, and AVX-512 for the claimed profile. | Microbench speed does not prove decode, prefill, or answer-corpus speed. |
| AVX-512 sustained-performance proof | Sustained receipts record duration, power/thermal context where available, affinity/cache-domain context where available, and no fallback. | Short boost behavior is not sustained performance. |

## Required kernel IDs

The AVX-512 QK256 lane uses stable kernel IDs. Implementations may land in
sequence, but receipts and docs must use these IDs exactly once implemented:

```rust
pub const QK256_AVX512_F32_GEMV_KERNEL_ID: &str =
    "qk256-avx512-f32-gemv";

pub const QK256_AVX512_I8S_SCALED_GEMV_KERNEL_ID: &str =
    "qk256-avx512-i8s-scaled-gemv";

pub const QK256_AVX512_I8S_SCALED_GEMM_KERNEL_ID: &str =
    "qk256-avx512-i8s-scaled-gemm";
```

`qk256-avx512-f32-gemv` is the early no-scale F32 parity and smoke target. The
scaled `i8s` GEMV ID is the production BitNet decode hot-path target. The
scaled `i8s` GEMM ID is the prefill target.

A VNNI implementation must use a distinct ID, for example
`qk256-avx512vnni-i8s-scaled-gemv`, because VNNI can change accumulation shape
and requires separate parity proof.

## Required receipt fields

AVX-512 proof receipts must distinguish requested backend, selected backend,
requested kernel, selected kernel, fallback status, feature detection, feature
requirements, hot-path counters, and parity summary. A minimal strict scaled
GEMV receipt has this shape:

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

Receipt validators must fail strict AVX-512 proof if the selected kernel is
scalar or AVX2, if fallback status is missing, or if AVX-512 invocation counters
are absent or zero for a receipt that claims AVX-512 execution.

## Correctness requirements

1. Scalar packed QK256 remains the correctness oracle.
2. AVX-512 F32/no-scale GEMV must match the scalar no-scale QK256 oracle.
3. AVX-512 scaled I2_S × I8_S GEMV must match the scalar BitNet.cpp-compatible
   scaled oracle, including activation quantization, activation scale and sum,
   integer dot behavior, and final scale application.
4. AVX-512 must compare against AVX2 before any speed claim.
5. Repeated AVX-512 runs must be deterministic for the same inputs and runtime
   feature set.

## Performance requirements

AVX-512 execution is not a speed claim. A speed claim is profile-scoped and
requires receipts for the profile being claimed. Required profile classes are:

- micro QK256 F32 GEMV;
- micro QK256 I8S scaled GEMV;
- layer decode;
- prefill;
- first token;
- decode loops;
- warm session;
- sustained decode.

A profile can be promoted only when parity passes, the answer-corpus evidence
for that profile is acceptable, the phase profile beats AVX2, sustained behavior
does not regress, fallback is false, and receipt validation accepts the
promotion.

## Hard non-goals

- AVX-512 detection does not prove AVX-512 execution.
- AVX-512 dispatch does not prove optimized kernel execution.
- AVX-512 execution does not prove speedup.
- AVX-512 speedup on a microbench does not prove decode speedup.
- AVX-512 short burst performance does not prove sustained performance.
- AVX-512 CPU proof does not prove CUDA, OpenCL, OpenVINO, NPU, server, or
  general answer-quality readiness.
