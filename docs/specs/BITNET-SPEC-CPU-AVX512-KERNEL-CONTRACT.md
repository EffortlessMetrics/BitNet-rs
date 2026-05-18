# BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT: CPU AVX-512 Kernel Contract

Status: proposed
Linked roadmap: [AMD Ryzen 9 9950X3D CPU Roadmap](amd-9950x3d-cpu-roadmap.md)
Linked plan: [CPU AVX-512 implementation plan](../../plans/cpu-avx512/implementation-plan.md)
Applies to: CPU QK256/I2_S AVX-512 kernels, strict CPU kernel selection,
answer-corpus receipts, parity receipts, microbench receipts, phase benchmark
receipts, 9950X3D proof artifacts

## Purpose

AVX-512 support must not be inferred from CPUID detection, a receipt label, or
an AVX2 proof run. This spec defines the evidence needed before BitNet-rs may
claim that the CPU AVX-512 lane is detected, selectable, executed, correct,
faster for a specific profile, or sustained on the 9950X3D lane.

The contract is intentionally staged: first prove that AVX-512 can be requested
and executed as a distinct QK256 kernel, then prove scalar/AVX2 parity, then
record profile and sustained-performance receipts before any speed claim or
auto-selection promotion.

## Proof Terms

| Term | Required evidence | Allowed claim |
| --- | --- | --- |
| AVX-512 detection proof | Runtime CPUID/subfeature probe records the detected x86 features. | AVX-512 was detected on this machine. |
| AVX-512 dispatch proof | A strict or explicit request records requested kernel, selected kernel, fallback status, fallback reason, and required features. | AVX-512 was selected or strict selection failed honestly. |
| AVX-512 kernel execution proof | The selected AVX-512 stable kernel ID is distinct from scalar/AVX2 IDs and invocation counters for that AVX-512 hot path are greater than zero. | AVX-512 kernel code executed for this workload. |
| AVX-512 parity proof | Scalar-vs-AVX512 and AVX2-vs-AVX512 comparisons pass for the governed synthetic and real QK256 fixtures. | AVX-512 matches the governed reference for this fixture/profile. |
| AVX-512 performance proof | Micro, layer, prefill, first-token, and decode receipts compare scalar, AVX2, and AVX-512 for the same model/profile. | AVX-512 is faster only for the measured profile whose receipt accepts the comparison. |
| AVX-512 sustained-performance proof | Long-running 9950X3D receipts record duration, power/thermal context when available, core/CCD/cache-domain context, and AVX2 comparator behavior. | AVX-512 sustains the measured profile under the recorded platform conditions. |

A lower proof level never implies a higher proof level. For example, detection
proof does not imply dispatch proof, and execution proof does not imply speedup.

## Stable Kernel IDs

The AVX-512 lane must use stable kernel IDs that cannot be confused with scalar
or AVX2 execution:

```rust
pub const QK256_AVX512_F32_GEMV_KERNEL_ID: &str =
    "qk256-avx512-f32-gemv";

pub const QK256_AVX512_I8S_SCALED_GEMV_KERNEL_ID: &str =
    "qk256-avx512-i8s-scaled-gemv";

pub const QK256_AVX512_I8S_SCALED_GEMM_KERNEL_ID: &str =
    "qk256-avx512-i8s-scaled-gemm";
```

`qk256-avx512-f32-gemv` may land first because it mirrors the existing
no-scale F32-style AVX2 GEMV structure. The production decode target is
`qk256-avx512-i8s-scaled-gemv`, which must mirror the scalar BitNet I2_S × I8_S
inline-scale semantics before any speed-oriented VNNI variant lands. The prefill
target is `qk256-avx512-i8s-scaled-gemm`.

A future VNNI implementation must use its own stable ID, such as
`qk256-avx512vnni-i8s-scaled-gemv`, because a different accumulation strategy is
a separate proof surface.

## Required Feature Probes

Receipts and strict dispatch must distinguish detected, required, and used
subfeatures. The baseline scaled AVX-512 kernel may require only `avx512f` and
`avx512bw`; VNNI must not be assumed unless it is probed.

Required helper surface for follow-on implementation PRs:

```rust
pub fn avx512_f_available() -> bool;
pub fn avx512_bw_available() -> bool;
pub fn avx512_vl_available() -> bool;
pub fn avx512_vnni_available() -> bool;
pub fn avx512_f_bw_available() -> bool;
pub fn avx512_f_bw_vl_available() -> bool;
pub fn avx512_bitnet_i8s_available() -> bool;
```

All helpers must return `false` on unsupported architectures or builds without
panicking.

## Required Receipt Fields

Strict AVX-512 proof receipts must record requested and selected CPU state plus
exact hot-path counters. Field names may live in the repo's receipt schema, but
the following information is mandatory:

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

For answer-corpus receipts, an AVX-512 label alone is insufficient. At least one
AVX-512 invocation counter for the selected hot path must be greater than zero,
and scalar/AVX2 counters must distinguish fallback from intentional comparator
runs.

## Strict Fallback Requirement

A strict request for an AVX-512 kernel must fail if the required AVX-512
subfeatures or target-feature-gated kernel are unavailable. It must not silently
select scalar or AVX2 while recording `fallback_used=false`.

Non-strict explicit AVX-512 requests may fall back only if the receipt records:

- requested kernel;
- selected fallback kernel;
- `fallback_used=true`;
- a non-null fallback reason;
- detected, required, and missing CPU features.

## Implementation Rails

- Do not compile the whole workspace with `-C target-cpu=native` to create the
  AVX-512 lane. Use target-feature-gated functions and runtime checks.
- Implement the exact/scalar-equivalent path before optimized variants.
- Keep no-scale F32 GEMV and scaled BitNet I2_S × I8_S GEMV as separate proof
  surfaces with separate kernel IDs and counters.
- Treat AVX-512 VNNI as a later, separately identified kernel family member.
- Keep auto-selection disabled for AVX-512 until parity, answer-corpus, phase,
  and sustained receipts justify profile-specific promotion.

## Claim Boundary

- AVX-512 detection does not prove AVX-512 dispatch.
- AVX-512 dispatch does not prove AVX-512 kernel execution.
- AVX-512 execution does not prove speedup.
- AVX-512 microbench speedup does not prove decode speedup.
- AVX-512 short-burst performance does not prove sustained performance.
- AVX-512 CPU proof does not prove CUDA, OpenCL, OpenVINO, NPU, Metal, WGPU,
  server readiness, or general answer quality.

## Proof Commands

Documentation-only contract changes must run:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
```

Follow-on runtime PRs must add the scoped cargo test, clippy, benchmark, and
receipt validation commands listed by the implementation plan and active goal.

## Non-Goals

- No CUDA, OpenCL, OpenVINO, NPU, Metal, WGPU, or server support is added by this
  contract.
- No AVX-512 speedup is claimed by this contract.
- No auto-selection promotion is authorized before the required receipts exist.
- No model answer-quality claim is made without a strict answer-corpus receipt.

## Related Sources

- [CPU ISA selection spec](BITNET-SPEC-CPU-ISA-SELECTION.md)
- [AMD Ryzen 9 9950X3D CPU Roadmap](amd-9950x3d-cpu-roadmap.md)
- [BitNet Kernel Matrix](../bitnet/BITNET_KERNEL_MATRIX.md)
- [BitNet CPU Path Plan](../bitnet/BITNET_CPU_PATH_PLAN.md)
- [CPU AVX-512 implementation plan](../../plans/cpu-avx512/implementation-plan.md)
