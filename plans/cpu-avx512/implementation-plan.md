# CPU AVX-512 Implementation Plan

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs:
- `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`
- `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`
- `docs/specs/amd-9950x3d-cpu-roadmap.md`
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: AVX-512 remains unpromoted until proof receipts satisfy this plan.
Policy impact: No policy exception.

## Goal

Bring the AVX-512 CPU lane from detection/receipt-label state to first-class
BitNet-rs AVX-512 execution on the 9950X3D, with strict fallback rejection,
scalar/AVX2 parity, phase-scoped performance receipts, sustained receipts, and
no overclaims.

## Non-negotiable rails

- AVX-512 detection is not kernel proof.
- AVX2 proof is not AVX-512 proof.
- AVX-512 execution is not speedup.
- AVX-512 microbench speed is not decode speed.
- Short boost behavior is not sustained performance.
- CPU AVX-512 proof is not CUDA, OpenCL, OpenVINO, NPU, server, or general
  answer-quality proof.
- Strict requested AVX-512 must fail if AVX-512 cannot execute.
- Do not promote auto-selection until parity, phase, and sustained receipts
  justify a profile-specific promotion.
- Do not implement AVX-512 by compiling the whole workspace with
  `-C target-cpu=native`; use target-feature-gated functions and runtime checks.

## PR sequence

| PR | Title | Scope | Acceptance |
|---|---|---|---|
| 0 | `docs(cpu): add AVX-512 kernel contract` | Add AVX-512 specs and this plan; update kernel matrix, 9950X3D roadmap, and cpu-proof active tracker. | No runtime changes; claim boundaries and PR queue are encoded; tracker item exists; proof commands pass. |
| 1 | `feat(cpu): expose AVX-512 subfeature detection` | Add subfeature helpers in `bitnet-cpu-detect`; do not change dispatch. | Non-x86 returns false without panic; ordering helpers work; no model behavior changes. |
| 2 | `feat(quant): add AVX-512 feature gates` | Add `bitnet-quantization` `avx512` feature plumbing and gated module surface. | `cargo check` passes for cpu, cpu+avx2, and cpu+avx512 feature sets. |
| 3 | `feat(cpu): add AVX-512 QK256 F32 GEMV` | Add no-scale F32 AVX-512 GEMV and kernel ID. | Scalar parity, repeated-run equality, and strict unavailable failure pass; no answer-corpus changes. |
| 4 | `feat(cpu): add AVX-512 QK256 kernel selection` | Extend QK256 selection metadata for explicit AVX-512 requests. | Auto remains conservative; strict missing AVX-512 errors; non-strict fallback is recorded. |
| 5 | `diag(cpu): record AVX-512 QK256 invocation counters` | Add F32 and scaled scalar/AVX2/AVX512 hot-path counters to receipts. | Strict AVX-512 proof can show AVX-512 invocation count greater than zero; no speed claim. |
| 6 | `test(cpu): add scaled I2S-I8S AVX-512 fixtures` | Lock scalar scaled I2_S × I8_S behavior before SIMD implementation. | Fixtures cover scales, tails, code patterns, and activation ranges; tests are reusable by AVX-512. |
| 7 | `feat(cpu): add AVX-512 scaled I2S-I8S QK256 GEMV` | Add baseline AVX-512BW scaled GEMV that mirrors scalar first. | Scalar-vs-AVX512 parity, tail coverage, repeated-run equality, and strict detection pass. |
| 8 | `feat(cpu): route inline-scale QK256 through AVX-512` | Wire explicit scaled AVX-512 into transformer/QK256 dispatch. | Real BitNet run can record selected scaled AVX-512 kernel, invocation count greater than zero, and fallback false. |
| 9 | `test(cpu): refresh strict AVX-512 answer corpus` | Produce strict AVX-512 answer-corpus and scalar/AVX2/AVX512 parity receipts. | Pass/fail is exact; divergence reports are retained; no speed claim. |
| 10 | `bench(cpu): add QK256 AVX-512 microbench receipts` | Compare scalar, AVX2, AVX512-F32, and AVX512-I8S scaled shapes. | Micro receipt emitted with CPU features, selected kernel, timing distribution, and no model-level speed claim. |
| 11 | `bench(cpu): add 9950X3D AVX-512 phase receipts` | Record prefill, first-token, decode, and warm-session phase receipts. | Cache-domain/core-affinity context is recorded or explicitly unavailable; no sustained claim yet. |
| 12 | `bench(cpu): record sustained 9950X3D AVX-512 profile` | Run 10-minute sustained decode or warm-session comparison. | Sustained receipt exists; short boost no longer drives claims. |
| 13 | `feat(cpu): promote AVX-512 auto-selection by profile` | Add profile-specific promotion only where receipts justify it. | Auto does not blindly choose AVX-512; promotion ledger exists; scalar/AVX2/AVX512 remain forceable. |

## Default validation commands

```bash
cargo fmt --all -- --check
cargo test --locked -p bitnet-cpu-detect --no-default-features --features avx512
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx512 i2s_qk256
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx512 --test qk256_avx512_parity_tests
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx2,avx512 --test qk256_avx2_parity_tests
cargo check --locked -p bitnet-cli --no-default-features --features cpu,full-cli
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

PRs should run the subset listed by their active-goal item plus `git diff
--check`. Hardware-only commands may be recorded as unavailable when the PR does
not run on the 9950X3D host.
