# BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT: CPU AVX-512 Kernel Contract

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/amd-9950x3d-cpu-roadmap.md`, `docs/bitnet/BITNET_KERNEL_MATRIX.md`, `docs/bitnet/BITNET_CPU_PATH_PLAN.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-avx512/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: AVX-512 remains explicit-proof CPU work until parity, receipt counters, phase benchmarks, and sustained-profile receipts exist.
Policy impact: n/a

## Purpose

This spec defines what BitNet-rs may count as AVX-512 CPU proof. It prevents
AVX-512 CPU feature detection, roadmap labels, or answer receipts with
AVX-512-like names from being treated as optimized AVX-512 QK256 execution.

The AVX-512 lane is a wider x86 CPU lane under the existing BitNet CPU proof
stack. It must inherit scalar packed correctness, compare against AVX2, and
emit receipts that prove which kernel actually ran before any speed or support
claim is made.

## Claim Vocabulary

### AVX-512 detection proof

AVX-512 detection proof means the process recorded runtime CPU feature probes
for the required AVX-512 subfeatures on the current machine.

Detection proof is not dispatch proof. It only says the CPU and current process
appear able to expose the named ISA features.

### AVX-512 dispatch proof

AVX-512 dispatch proof means a strict CPU selection path requested an AVX-512
kernel family and the selector returned an AVX-512 kernel ID without silently
falling back to scalar or AVX2.

Dispatch proof is not execution proof unless the hot-path receipt counters also
show AVX-512 invocations.

### AVX-512 kernel execution proof

AVX-512 kernel execution proof means the selected kernel has a distinct stable
AVX-512 kernel ID and receipts record non-zero AVX-512 invocation counters for
the relevant QK256 hot path.

An answer-corpus receipt whose `selected_kernel` only contains an AVX-512 label
is not sufficient unless it also records the stable kernel ID, detected and
required CPU features, fallback state, and invocation counters.

### AVX-512 parity proof

AVX-512 parity proof means scalar-vs-AVX512 comparisons pass for the applicable
kernel shape and data contract. For performance-sensitive comparisons, AVX2-vs-
AVX512 evidence must also exist so reviewers can separate AVX-512-specific
behavior from already-proven scalar and AVX2 behavior.

Parity proof can exist at several levels:

- synthetic QK256 fixture parity;
- real QK256 tensor parity;
- logits/top-k parity;
- generated-token agreement.

The receipt must state which levels passed and which levels were not run.

### AVX-512 performance proof

AVX-512 performance proof means phase-scoped benchmark receipts show AVX-512
beating the relevant scalar or AVX2 baseline for a named profile. A microbench
win does not prove decode, prefill, first-token, or full-answer speedup.

### AVX-512 sustained-performance proof

AVX-512 sustained-performance proof means a sustained profile, such as a
10-minute decode or warm-session loop, records duration, power/cooling context,
CPU topology, and selected kernel counters. Short boost behavior must not be
used as the sustained claim.

## Required Stable Kernel IDs

The AVX-512 lane reserves these QK256 kernel IDs:

```rust
pub const QK256_AVX512_F32_GEMV_KERNEL_ID: &str =
    "qk256-avx512-f32-gemv";

pub const QK256_AVX512_I8S_SCALED_GEMV_KERNEL_ID: &str =
    "qk256-avx512-i8s-scaled-gemv";

pub const QK256_AVX512_I8S_SCALED_GEMM_KERNEL_ID: &str =
    "qk256-avx512-i8s-scaled-gemm";
```

`qk256-avx512-f32-gemv` may land first as a no-scale F32 GEMV smoke path that
mirrors the current AVX2 verification shape. `qk256-avx512-i8s-scaled-gemv` is
the BitNet decode hot-path target because it must preserve inline-scale
I2_S-by-I8_S semantics. `qk256-avx512-i8s-scaled-gemm` is the prefill target.

A future VNNI implementation must use a separate stable ID, for example
`qk256-avx512vnni-i8s-scaled-gemv`, because VNNI can change the accumulation
shape and must not silently replace the baseline AVX-512BW path.

## Runtime Feature Contract

Baseline AVX-512 QK256 kernels must probe subfeatures explicitly. The minimum
baseline for the non-VNNI AVX-512 lane is:

```text
avx512f
avx512bw
```

Receipts should distinguish detected, required, and used features:

```json
{
  "features_detected": ["avx512f", "avx512bw", "avx512vl"],
  "features_required": ["avx512f", "avx512bw"],
  "features_used": ["avx512f", "avx512bw"]
}
```

Do not assume `avx512vnni` unless the runtime probe records it and the selected
kernel ID identifies a VNNI kernel.

## Strict Fallback Contract

If a user or proof run requests an AVX-512 kernel in strict mode and required
features or compiled feature gates are unavailable, the run must fail. It must
not emit scalar or AVX2 execution as if AVX-512 ran.

If non-strict fallback is allowed, the receipt must set `fallback_used=true`,
record the fallback reason, and record the selected scalar or AVX2 kernel ID.

## Required Receipt Fields

A strict AVX-512 receipt must contain enough data to prove request, selection,
fallback behavior, feature support, and actual hot-path execution:

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
    "features_detected": [
      "avx512f",
      "avx512bw",
      "avx512vl",
      "avx512vnni"
    ],
    "features_required": [
      "avx512f",
      "avx512bw"
    ],
    "features_used": [
      "avx512f",
      "avx512bw"
    ],
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

## Benchmark Scope

AVX-512 benchmark receipts must name the profile they measure. The minimum
profile vocabulary is:

```text
micro_qk256_f32_gemv
micro_qk256_i8s_scaled_gemv
layer_0_decode
prefill_128
prefill_512
first_token
decode_32
decode_128
warm_session_3_turns
sustained_decode_10min
```

Each profile must record the selected kernel, scalar/AVX2 comparison baseline,
thread count, CPU feature metadata, and fallback status. 9950X3D receipts should
also record core affinity, scheduler policy, CCD/cache-domain context, and power
or cooling context when available.

## Hard Non-Goals

- AVX-512 detection does not prove AVX-512 execution.
- AVX-512 dispatch does not prove AVX-512 hot-path invocation.
- AVX-512 execution does not prove speedup.
- AVX-512 speedup on a microbench does not prove decode speedup.
- AVX-512 short burst performance does not prove sustained performance.
- AVX-512 CPU proof does not prove CUDA, OpenCL, OpenVINO, NPU, Metal, server,
  or general chat-quality readiness.
- AVX-512 must not be implemented by compiling the whole workspace with a
  global `-C target-cpu=native` shortcut.
