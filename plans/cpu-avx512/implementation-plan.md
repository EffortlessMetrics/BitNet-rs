# CPU AVX-512 Implementation Plan

This plan sequences the CPU AVX-512 lane from documentation and strict dispatch
rails to real QK256 execution, parity receipts, profile benchmarks, and
profile-scoped auto-selection. It intentionally starts with documentation
because AVX-512 detection or a receipt label is not optimized AVX-512 kernel
proof.

## Source Links

- Spec: `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`
- Spec: `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`
- Roadmap: `docs/specs/amd-9950x3d-cpu-roadmap.md`
- Matrix: `docs/bitnet/BITNET_KERNEL_MATRIX.md`
- CPU path plan: `docs/bitnet/BITNET_CPU_PATH_PLAN.md`
- Campaign: `docs/tracking/campaigns/cpu-proof/active.toml`

## Work Item: CPU-AVX512-000

Status: ready
Campaign: `docs/tracking/campaigns/cpu-proof/active.toml`
Blocked by: CPU-ANSWER-007
Blocks: CPU-AVX512-001 through CPU-AVX512-013

### Goal

Add the AVX-512 kernel contract, strict CPU ISA selection rails, AVX-512 PR
queue, 9950X3D proof requirements, and kernel-matrix claim boundary before any
runtime AVX-512 implementation starts.

### Production Delta

No runtime delta. This item is documentation and planning only.

### Non-Goals

- Do not add AVX-512 target-feature code.
- Do not add an `avx512` crate feature.
- Do not change answer-corpus receipts.
- Do not change CUDA, NPU, OpenCL, OpenVINO, Metal, WGPU, or server claims.
- Do not claim AVX-512 speedup or auto-selection.

### Acceptance

- `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md` exists and defines
  detection, dispatch, execution, parity, performance, and sustained proof.
- `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md` exists and defines `auto`,
  `scalar`, `avx2`, `avx512`, and `avx512-vnni` strict/fallback behavior.
- `docs/bitnet/BITNET_KERNEL_MATRIX.md` lists the AVX-512 QK256 kernel IDs and
  states that AVX-512 must inherit scalar parity and compare against AVX2 before
  speed claims.
- `docs/specs/amd-9950x3d-cpu-roadmap.md` names required AVX-512 profiles,
  comparisons, hardware metadata, and proof boundaries.
- The CPU proof active goal names this documentation work item and the follow-on
  AVX-512 sequence.

### Proof Commands

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
```

### Rollback

Revert the AVX-512 spec files, this plan, and the related documentation/tracker
edits. No runtime migration is needed because this item has no runtime delta.

## Follow-On PR Queue

| Order | ID | Title | Scope | Acceptance summary |
| --- | --- | --- | --- | --- |
| 1 | CPU-AVX512-001 | `feat(cpu): expose AVX-512 subfeature detection` | `bitnet-cpu-detect` helpers only | Non-x86-safe helpers expose AVX-512F/BW/VL/VNNI and composite checks; no dispatch change. |
| 2 | CPU-AVX512-002 | `feat(quant): add AVX-512 feature gates` | `bitnet-quantization` feature plumbing | `cpu`, `cpu,avx2`, and `cpu,avx512` checks compile without selecting AVX-512 automatically. |
| 3 | CPU-AVX512-003 | `feat(cpu): add AVX-512 QK256 F32 GEMV` | No-scale F32-style QK256 GEMV | Scalar parity, repeated-run equality, and strict unavailable failure for `qk256-avx512-f32-gemv`. |
| 4 | CPU-AVX512-004 | `feat(cpu): add AVX-512 QK256 kernel selection` | Selection metadata | Explicit AVX-512 strict requests fail when unavailable; non-strict fallback records `fallback_used=true`; auto remains unpromoted. |
| 5 | CPU-AVX512-005 | `diag(cpu): record AVX-512 QK256 invocation counters` | Receipt counters | Answer-corpus and proof receipts distinguish scalar, AVX2, and AVX-512 hot-path invocations. |
| 6 | CPU-AVX512-006 | `test(cpu): add scaled I2S-I8S AVX-512 fixtures` | Scalar oracle fixtures | Scaled I2_S × I8_S scalar expected values are locked for AVX-512 reuse. |
| 7 | CPU-AVX512-007 | `feat(cpu): add AVX-512 scaled I2S-I8S QK256 GEMV` | Baseline scaled AVX-512 GEMV | `qk256-avx512-i8s-scaled-gemv` mirrors scalar semantics, covers tails, and passes parity. |
| 8 | CPU-AVX512-008 | `feat(cpu): route inline-scale QK256 through AVX-512` | Transformer/QK256 dispatch | Explicit strict AVX-512 real BitNet runs show selected scaled AVX-512 kernel, counters > 0, and fallback false. |
| 9 | CPU-AVX512-009 | `test(cpu): refresh strict AVX-512 answer corpus` | 9950X3D answer proof | Official Microsoft I2_S GGUF strict receipts record tokenizer, selected kernel, counters, and parity diagnostics; no speed claim. |
| 10 | CPU-AVX512-010 | `bench(cpu): add QK256 AVX-512 microbench receipts` | Microbench receipts | Scalar, AVX2, AVX-512 F32, and AVX-512 scaled shapes emit median/p95 and feature context. |
| 11 | CPU-AVX512-011 | `bench(cpu): add 9950X3D AVX-512 phase receipts` | Phase benchmark receipts | Prefill, first-token, decode, and warm-session receipts compare scalar, AVX2, AVX-512, and optional CUDA diagnostics. |
| 12 | CPU-AVX512-012 | `bench(cpu): record sustained 9950X3D AVX-512 profile` | Sustained/cache-domain proof | Ten-minute decode or warm-session receipts record thermal/power/core/CCD context and AVX2 comparator behavior. |
| 13 | CPU-AVX512-013 | `feat(cpu): promote AVX-512 auto-selection by profile` | Promotion ledger and selector | Auto selects AVX-512 only for profiles accepted by parity, answer, phase, sustained, fallback-free receipts. |

## Default Validation Set For Runtime AVX-512 PRs

Use the scoped subset that matches the changed files, and add receipt or
benchmark commands when the work item introduces receipts:

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

## 9950X3D Hardware Proof Commands

These commands are for the 9950X3D hardware lane after the scaled kernel and
receipt counters exist. They are not required for the documentation-only first
PR.

```bash
cargo build --locked -p bitnet-cli --no-default-features --features cpu,full-cli,avx512 --release

target/release/bitnet.exe answer-corpus \
  --device cpu \
  --cpu-kernel qk256-avx512-i8s-scaled-gemv \
  --strict-loader \
  --strict-tokenizer \
  --model <official-bitnet-i2s-gguf> \
  --tokenizer <tokenizer-json> \
  --json-out ci/hardware/windows-9950x3d-rtx5070ti/<date>/cpu-avx512-answer-corpus.json

target/release/bitnet.exe answer-parity \
  --left <scalar receipt> \
  --right <avx512 receipt> \
  --json-out ci/hardware/windows-9950x3d-rtx5070ti/<date>/cpu-scalar-vs-avx512-answer-parity.json
```

## Rollback Guidance For Runtime PRs

Runtime PRs must keep scalar and AVX2 behavior independently reversible. If an
AVX-512 implementation regresses parity, revert that specific AVX-512 file,
feature gate, selector branch, or promotion ledger without weakening strict
fallback enforcement or scalar/AVX2 proofs.
