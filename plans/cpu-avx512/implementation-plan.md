# CPU AVX-512 Implementation Plan

Status: Draft
Owner: BitNet CPU proof campaign
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`, `docs/specs/amd-9950x3d-cpu-roadmap.md`
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: No AVX-512 speed or auto-selection claim until the listed proof receipts exist.
Policy impact: No policy exception.

## Goal

Bring the AVX-512 CPU lane from detection and receipt-label evidence to first-class BitNet-rs AVX-512 execution on the 9950X3D, with strict fallback rejection, scalar/AVX2 parity, invocation counters, phase benchmarks, sustained receipts, and no overclaims.

## Claim Boundary

This plan does not authorize CUDA, OpenCL, OpenVINO, NPU, server, production, or global speed claims. It also does not authorize `auto` to choose AVX-512 from CPUID alone.

## PR Queue

### PR 0 - Docs/spec rails

Title: `docs(cpu): add AVX-512 kernel contract`

Scope:

- add `docs/specs/BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md`;
- add `docs/specs/BITNET-SPEC-CPU-ISA-SELECTION.md`;
- add this implementation plan;
- update `docs/bitnet/BITNET_KERNEL_MATRIX.md`;
- update `docs/specs/amd-9950x3d-cpu-roadmap.md`;
- update `docs/tracking/campaigns/cpu-proof/active.toml`.

Acceptance:

- no runtime changes;
- AVX-512 detection, dispatch, execution, parity, performance, and sustained-performance proofs are distinct;
- strict requested AVX-512 fallback is specified as fatal;
- the PR queue is encoded;
- the cpu-proof tracker includes the AVX-512 contract work item;
- `git diff --check` passes;
- `cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof` passes or records a non-code environment limitation;
- `cargo run --locked -p xtask --no-default-features -- campaign generate --check` passes or records a non-code environment limitation.

### PR 1 - AVX-512 subfeature detection

Title: `feat(cpu): expose AVX-512 subfeature detection`

Scope:

- add `avx512_f_available()`;
- add `avx512_bw_available()`;
- add `avx512_vl_available()`;
- add `avx512_vnni_available()`;
- add `avx512_f_bw_available()`;
- do not change dispatch.

Acceptance:

- non-x86 builds return false without panics;
- tests assert helper relationships and no panic behavior;
- receipts can serialize detected subfeatures;
- model behavior is unchanged.

### PR 2 - Quantization feature plumbing

Title: `feat(quant): add AVX-512 feature gates`

Scope:

- add `avx512 = ["dep:bitnet-cpu-detect", "bitnet-cpu-detect/avx512"]` to `bitnet-quantization`;
- add target/feature-gated module plumbing for `i2s_qk256_avx512`;
- do not add dispatch behavior yet.

Acceptance:

- `cargo check --locked -p bitnet-quantization --no-default-features --features cpu`;
- `cargo check --locked -p bitnet-quantization --no-default-features --features cpu,avx2`;
- `cargo check --locked -p bitnet-quantization --no-default-features --features cpu,avx512`.

### PR 3 - AVX-512 F32/no-scale GEMV smoke

Title: `feat(cpu): add AVX-512 QK256 F32 GEMV`

Scope:

- add `qk256-avx512-f32-gemv`;
- mirror scalar no-scale QK256 and the existing AVX2 F32-style path;
- cover rows `1, 2, 7, 32`, columns `256, 257, 300, 512, 513, 1024`, code patterns, and activation patterns.

Acceptance:

- scalar parity;
- repeated-run equality;
- strict failure if AVX-512 is unavailable;
- no answer-corpus changes;
- no speed claim.

### PR 4 - AVX-512 kernel selection metadata

Title: `feat(cpu): add AVX-512 QK256 kernel selection`

Scope:

- extend QK256 selection data structures for requested/selected AVX-512;
- keep `auto` conservative and not CPUID-only;
- implement strict and non-strict fallback metadata.

Acceptance:

- strict unavailable AVX-512 errors;
- non-strict unavailable AVX-512 records fallback;
- explicit available AVX-512 selects AVX-512;
- AVX2 behavior remains unchanged.

### PR 5 - Hot-path counters

Title: `diag(cpu): record AVX-512 QK256 invocation counters`

Scope:

- add F32 scalar, AVX2, and AVX-512 counters;
- add I8S scaled scalar, AVX2, and AVX-512 counters;
- expose counters in answer-corpus or kernel receipts.

Acceptance:

- strict AVX-512 F32 proof can show AVX-512 invocation count greater than zero;
- receipts distinguish labels from execution;
- no speed claim.

### PR 6 - Scaled I2_S x I8_S AVX-512 fixtures

Title: `test(cpu): add scaled I2S-I8S AVX-512 fixtures`

Scope:

- lock scalar oracle behavior for inline-scale I2_S x I8_S QK256;
- cover weight scales `0.125, 0.5, 1.0`, tail columns, code patterns, and activation ranges.

Acceptance:

- scalar oracle documented;
- expected values stable;
- tests are reusable by AVX-512 implementation.

### PR 7 - Scaled I2_S x I8_S AVX-512 baseline kernel

Title: `feat(cpu): add AVX-512 scaled I2S-I8S QK256 GEMV`

Scope:

- add `qk256-avx512-i8s-scaled-gemv`;
- mirror scalar semantics first;
- defer VNNI to a separately identified kernel.

Acceptance:

- scalar-vs-AVX512 parity;
- tail coverage;
- repeated-run equality;
- strict runtime detection;
- no model-level promotion.

### PR 8 - Wire scaled AVX-512 into transformer/QK256 dispatch

Title: `feat(cpu): route inline-scale QK256 through AVX-512`

Scope:

- route inline-scale QK256 through explicit AVX-512 selection when available;
- preserve strict fallback semantics.

Acceptance:

- real BitNet run can show selected kernel `qk256-avx512-i8s-scaled-gemv`;
- AVX-512 I8S scaled invocation count is greater than zero;
- `fallback_used=false`;
- generated IDs remain unchanged where parity requires;
- no speed claim.

### PR 9 - Answer-corpus AVX-512 proof refresh

Title: `test(cpu): refresh strict AVX-512 answer corpus`

Scope:

- run official Microsoft BitNet I2_S GGUF in strict CPU mode with requested AVX-512 scaled GEMV;
- emit scalar-vs-AVX512 and AVX2-vs-AVX512 parity evidence.

Acceptance:

- receipt records real GGUF, strict tokenizer, selected AVX-512 kernel, counters, fallback false, and pass/fail evidence;
- any divergence is classified;
- no speedup claim.

### PR 10 - AVX-512 microbench

Title: `bench(cpu): add QK256 AVX-512 microbench receipts`

Scope:

- compare scalar, AVX2, AVX512-F32, and AVX512-I8S-scaled paths across rows `1, 32, 128, 512, 2048` and columns `256, 512, 1024, 2048, 4096`.

Acceptance:

- micro receipt emitted with median, p95, bandwidth estimate, selected kernel, CPU features, threads, and affinity when known;
- no model-level speedup claim.

### PR 11 - 9950X3D phase benchmark

Title: `bench(cpu): add 9950X3D AVX-512 phase receipts`

Scope:

- compare scalar, AVX2, AVX-512, and CUDA diagnostic if available for prefill, first-token, decode, and warm-session profiles.

Acceptance:

- phase receipts emitted;
- cache-domain/core-affinity context recorded or marked unavailable;
- no sustained claim.

### PR 12 - Sustained-power and cache-domain proof

Title: `bench(cpu): record sustained 9950X3D AVX-512 profile`

Scope:

- run at least a ten-minute decode or warm-session loop comparing AVX2 and AVX-512;
- record temperature, power mode, cooling, frequency, affinity, and CCD/cache context when available.

Acceptance:

- sustained receipt exists;
- short boost no longer drives the claim;
- promotion can only happen where sustained profile is better or justified.

### PR 13 - Auto-selection promotion

Title: `feat(cpu): promote AVX-512 auto-selection by profile`

Scope:

- add profile-specific promotion only after parity, answer-corpus, phase, sustained, no-fallback, and receipt-validator evidence exists.

Acceptance:

- `auto` does not blindly choose AVX-512;
- profile-specific promotion ledger exists;
- users can still force scalar, AVX2, or AVX-512.

## Default Validation Set

Use the scoped commands from the active work item for each PR. Runtime AVX-512 PRs should additionally consider:

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
