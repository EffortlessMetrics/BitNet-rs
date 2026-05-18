# CPU Scalar Implementation Plan

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Sequences scalar CPU oracle and fallback productization.
Policy impact: No policy exception.

## Goal

Make scalar CPU BitNet inference a first-class, accurate, efficient,
receipt-backed oracle and fallback path. The production scalar path is the
BitNet.cpp-style scaled I2_S × I8_S QK256 path; the F32/no-scale scalar path
remains a useful diagnostic and reference path.

## Claim Boundary

This plan may claim that scalar CPU contracts and sequencing exist after the
plan PR merges. Runtime PRs may claim only the specific behavior proven by their
fixtures and receipts. The plan does not claim scalar speedup, GPU/NPU/server
support, broad chat quality, or dense SLM quality.

## Hard Rails

1. No scalar hidden fallback in strict accelerated runs.
2. No F32/no-scale scalar substitution for scaled BitNet I8S.
3. No whole-matrix dequantization in steady-state scalar proof.
4. No new tolerance without updating the scalar parity contract.
5. No speedup claim from scalar receipts unless separately reviewed.
6. No broad answer-quality claim from a tiny corpus.
7. No GPU/NPU/server claims from scalar work.
8. No ambiguous `reference` labels in receipts; name the actual scalar kernel.
9. Preserve generated IDs or record exact divergence in performance PRs.

## PR Sequence

| Item | Title | Scope | Exit proof |
| --- | --- | --- | --- |
| CPU-SCALAR-000 | `docs(cpu): add scalar kernel contract and hot-path plan` | Add scalar specs, this plan, CPU plan/matrix links, and tracker rails. | Docs only; claim boundaries explicit. |
| CPU-SCALAR-001 | `feat(cpu): split scalar QK256 kernel IDs` | Add precise F32/no-scale and I8S-scaled scalar GEMV/GEMM IDs with compatibility aliases. | `cargo test --locked -p bitnet-quantization --no-default-features --features cpu i2s_qk256 --lib`. |
| CPU-SCALAR-002 | `feat(cpu): add kernel selection for scaled scalar I8S GEMV` | Add scaled GEMV selection and `*_with_kernel_selection` metadata. | Strict scalar is not fallback; strict unavailable AVX2 errors. |
| CPU-SCALAR-003 | `feat(cpu): route inline-scale QK256 through selected scalar kernel` | Wire inline-scale QK256 dispatch through selected scaled scalar/SIMD path. | Receipts record `qk256-scalar-i8s-scaled-gemv` and `fallback_used=false`. |
| CPU-SCALAR-004 | `diag(cpu): record scalar QK256 hot-path counters` | Add scalar QK256 invocation and materialization counters. | Receipts expose counters and do not claim accelerated lanes. |
| CPU-SCALAR-005 | `test(cpu): harden scalar I2S/I8S fixtures` | Add column, row, scale, pattern, activation, tail, and repeatability fixtures. | `cargo test --locked -p bitnet-quantization --no-default-features --features cpu scalar_i8s`. |
| CPU-SCALAR-006 | `test(cpu): record strict scalar BitNet answer corpus` | Run official Microsoft I2_S artifact with strict scalar request and receipt. | Tiny corpus passes or blocker recorded; `speedup_claim=false`. |
| CPU-SCALAR-007 | `test(cpu): compare scalar baseline against AVX2 receipts` | Compare strict scalar and strict AVX2 receipts. | Same prompt IDs, generated IDs, decoded text, and fallback false or classified divergence. |
| CPU-SCALAR-008 | `perf(cpu): cache QK256 flat packed bytes for scalar dispatch` | Replace per-call flat packed byte extraction with borrowed packed views. | Generated IDs unchanged; extraction count reduced. |
| CPU-SCALAR-009 | `perf(cpu): use flat buffers for scalar QK256 rows` | Replace `Vec<Vec<f32>>` input/output materialization with flat buffers. | Tensor output and generated IDs unchanged; materialization counters reduced. |
| CPU-SCALAR-010 | `perf(cpu): add reusable scalar CPU workspace` | Add reusable scalar workspace for activations and outputs. | Workspace reuse count increases; answers unchanged. |
| CPU-SCALAR-011 | `perf(cpu): optimize scalar I8S activation quantization` | Add `quantize_row_i8_s_activation_into` preserving exact behavior. | Bit-exact quant vector, int dot, scaled output, and generated IDs. |
| CPU-SCALAR-012 | `feat(cpu): add scaled scalar QK256 prefill GEMM` | Add scaled I8S scalar GEMM for prefill. | Batched GEMM equals repeated scaled GEMV. |
| CPU-SCALAR-013 | `bench(cpu): add scalar-only BitNet phase receipts` | Add scalar-only micro, layer, prefill, first-token, decode, and warm-session receipts. | Precise scalar kernel IDs, fallback false, no AVX2 selected, no speedup claim. |
| CPU-SCALAR-014 | `diag(cpu): audit scalar transformer support-op timing` | Measure embedding, norm, RoPE, attention, KV, FFN, logits, sampling phases. | Ranked hot support-op report with no behavior change. |
| CPU-SCALAR-015 | `perf(cpu): optimize highest-cost scalar support op` | Optimize the highest-evidence scalar support-op target. | Before/after receipt, generated IDs unchanged, no tolerance drift. |
| CPU-SCALAR-016 | `bench(cpu): establish scalar thread-count envelope` | Benchmark thread counts for prefill, first token, and decode. | Default scalar thread policy is evidence-backed and recorded. |
| CPU-SCALAR-017 | `docs(cpu): publish scalar BitNet status` | Add public scalar CPU status surface. | Users can tell what scalar is for and what is not claimed. |

## Validation Baseline

Runtime PRs should run the scoped commands listed in the active work item plus
`git diff --check`. Common scalar validation commands are:

```bash
cargo test --locked -p bitnet-quantization --no-default-features --features cpu i2s_qk256 --lib
cargo test --locked -p bitnet-qk256-dispatch --no-default-features --features cpu
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli answer_corpus
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Docs-only PRs may use the campaign checker/generator and diff check without
runtime tests when they do not change Rust code or receipts.

## Rollback Path

Each runtime PR must preserve the previous scalar path until the new path is
fixture- and receipt-proven. If strict scalar answer receipts or generated IDs
regress without an accepted divergence classification, revert the runtime patch
and keep the docs/tracker claim boundary narrowed.
