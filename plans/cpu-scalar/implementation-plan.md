# CPU Scalar Implementation Plan

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Sequences scalar CPU oracle/fallback productization; does not promote runtime support by itself.
Policy impact: No policy exception.

## Goal

Make scalar CPU BitNet inference a first-class, accurate, efficient,
receipt-backed oracle and fallback path. The scalar lane should prove real GGUF,
strict tokenizer, canonical packed QK256/I2_S layout, BitNet.cpp-style scaled
I2_S x I8_S math, deterministic CPU transformer ops, strict fallback behavior,
answer-corpus quality gates, long-decode stability, phase timing, and no hidden
dequantized/reference substitution.

## Current State

- The CPU path plan names scalar packed kernels as the correctness floor for
  optimized CPU kernels.
- `crates/bitnet-quantization/src/i2s_qk256.rs` contains no-scale F32 scalar
  GEMV/GEMM foundations and a BitNet.cpp-style scaled I2_S x I8_S scalar GEMV
  path.
- Inline-scale QK256 dispatch currently needs first-class requested/selected
  kernel metadata for the scaled path.
- Current dispatch still has known allocation/copy hot spots: flat packed byte
  extraction, input `Vec<Vec<f32>>` materialization, output `Vec<Vec<f32>>`
  allocation, and output flattening.
- Existing performance evidence is useful but not yet a clean scalar-only
  baseline because auto profiles can select AVX2.

## Claim Boundary

This lane may claim scalar CPU oracle/fallback progress only for exact proof
items that have landed. It must not claim SIMD speedup, GPU/NPU/server readiness,
broad chat quality, dense SLM proof, or new tolerances.

## Work Items

| Order | ID | Title | Runtime delta | Acceptance summary |
| --- | --- | --- | --- | --- |
| 0 | CPU-SCALAR-000 | Add scalar specs and tracker rails | No | Specs and plan exist; CPU tracker names the lane; claim boundaries explicit. |
| 1 | CPU-SCALAR-001 | Split scalar QK256 kernel IDs | Yes | Precise F32/no-scale and I8_S-scaled scalar IDs exist; legacy aliases remain compatibility-only. |
| 2 | CPU-SCALAR-002 | Add scaled scalar selection metadata | Yes | Scaled GEMV selection reports requested/selected/fallback; strict accelerated fallback to scalar errors. |
| 3 | CPU-SCALAR-003 | Route inline-scale QK256 through selected scalar kernel | Yes | Inline-scale branch records precise scaled scalar selected kernel and `fallback_used=false` for strict scalar. |
| 4 | CPU-SCALAR-004 | Record scalar QK256 hot-path counters | Yes | Receipts expose scalar invocation, flat extraction, materialization, allocation, and workspace counters. |
| 5 | CPU-SCALAR-005 | Harden scalar I2_S/I8_S fixtures | Test | Edge columns, rows, scales, patterns, activation classes, tails, act sums, integer dots, and repeatability pass. |
| 6 | CPU-SCALAR-006 | Record strict scalar BitNet answer corpus | Evidence | Official Microsoft I2_S artifact records strict scalar selected kernel, tokenizer, quality result, and fallback=false. |
| 7 | CPU-SCALAR-007 | Compare scalar baseline against AVX2 receipts | Evidence | Same prompts, generated IDs, decoded text, first divergence classification, and fallback=false on both lanes. |
| 8 | CPU-SCALAR-008 | Cache QK256 flat packed bytes for scalar dispatch | Perf | Generated IDs unchanged; flat weight extraction counter reduced; no whole-matrix dequantization. |
| 9 | CPU-SCALAR-009 | Use flat buffers for scalar QK256 rows | Perf | Input/output nested row materialization counters reduced; tensor output and generated IDs unchanged. |
| 10 | CPU-SCALAR-010 | Add reusable scalar CPU workspace | Perf | Activation/output scratch reused; no per-layer activation allocation in strict scalar steady state; IDs unchanged. |
| 11 | CPU-SCALAR-011 | Optimize scalar I8_S activation quantization | Perf | Into-buffer helper is bit-exact with old helper; same act_sum, act_scale, dot, scaled output, and IDs. |
| 12 | CPU-SCALAR-012 | Add scaled scalar QK256 prefill GEMM | Yes | Batched scaled GEMM equals repeated scaled GEMV; prefill receipt can select scaled scalar GEMM. |
| 13 | CPU-SCALAR-013 | Add scalar-only BitNet phase receipts | Evidence | Scalar profiles record precise selected scalar kernel, fallback=false, no AVX2 selection, and no speedup claim. |
| 14 | CPU-SCALAR-014 | Audit scalar transformer support-op timing | Diagnostic | Support-op report ranks scalar hot spots without behavior change. |
| 15 | CPU-SCALAR-015 | Optimize highest-cost scalar support op | Perf | Before/after support-op receipt; generated IDs unchanged; no tolerance drift. |
| 16 | CPU-SCALAR-016 | Establish scalar thread-count envelope | Bench | Thread-count receipts pick default scalar policy by evidence for exact machines/profiles. |
| 17 | CPU-SCALAR-017 | Publish scalar BitNet status | Docs | Users and accelerated lanes can see scalar correctness, answer, long-decode, performance, and non-claim status. |

## First Work Item: CPU-SCALAR-000

Status: ready
Campaign: `docs/tracking/campaigns/cpu-proof/active.toml`
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`,
`docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`,
`docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`,
`docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
Blocks: CPU-SCALAR-001 through CPU-SCALAR-017
Blocked by: CPU-ANSWER-007

### Goal

Add scalar specs, this implementation plan, and campaign tracker rails so future
runtime PRs have precise kernel IDs, strict fallback rules, hot-path counters,
parity levels, performance profiles, and claim boundaries.

### Production Delta

No runtime delta. This is docs and tracking only.

### Non-Goals

- Do not change Rust runtime code.
- Do not modify tokenizer, prompt-template, loader, scalar math, SIMD, GPU, NPU,
  or server behavior.
- Do not generate or edit proof receipts.
- Do not claim scalar performance, answer quality, long-decode stability, SIMD
  parity, or speedup from this planning PR.

### Acceptance

- Four scalar specs exist under `docs/specs/`.
- This plan exists under `plans/cpu-scalar/`.
- `docs/bitnet/BITNET_CPU_PATH_PLAN.md` points to the scalar specs and plan.
- `docs/bitnet/BITNET_KERNEL_MATRIX.md` distinguishes F32/no-scale scalar IDs
  from scaled I2_S x I8_S scalar IDs.
- `docs/tracking/campaigns/cpu-proof/active.toml` contains `CPU-SCALAR-000` with
  docs-only allowed paths and explicit claim boundaries.

### Proof Commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Rollback

Revert the four scalar specs, this plan, and the scalar tracker/docs references.
No runtime state or generated receipt needs cleanup because this item is docs
only.

## Runtime Proof Baseline For Follow-On PRs

Follow-on runtime PRs should start from these commands and add narrower commands
from their work item:

```bash
cargo test --locked -p bitnet-quantization --no-default-features --features cpu i2s_qk256 --lib
cargo test --locked -p bitnet-qk256-dispatch --no-default-features --features cpu
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli answer_corpus
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
