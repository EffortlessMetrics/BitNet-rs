# CPU Scalar BitNet Implementation Plan

Status: draft
Owner: cpu-proof campaign
Created: 2026-05-18
Linked proposal: n/a
Linked specs: docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md; docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md; docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md; docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Sequences scalar CPU oracle/fallback productization; individual PRs provide proof before claim changes.
Policy impact: No policy exception.

## Goal

Make scalar CPU BitNet inference a first-class, accurate, efficient,
receipt-backed oracle and fallback path. The first-class scalar path is the
BitNet.cpp-style scaled I2_S × I8_S path; the F32/no-scale scalar path remains a
useful diagnostic/oracle path but must not be substituted for real scaled BitNet
math.

## Hard rules

- Do not use scalar as a hidden fallback in strict accelerated runs.
- Do not use no-scale F32 scalar as a substitute for scaled BitNet I8S scalar.
- Do not change tokenizer or prompt policy as part of scalar kernel work.
- Do not change scalar math without fixtures and answer receipts.
- Do not invent new tolerances.
- Do not touch GPU, NPU, or server lanes.
- Do not claim speedup from scalar receipts.
- Preserve generated IDs or record exact divergence.

## PR sequence

| PR | Title | Purpose | Acceptance |
| --- | --- | --- | --- |
| CPU-SCALAR-000 | `docs(cpu): add scalar kernel contract and hot-path plan` | Add scalar specs, implementation plan, CPU plan links, kernel matrix clarifications, and tracker rails. | Docs only; no runtime changes; claim boundaries explicit. |
| CPU-SCALAR-001 | `feat(cpu): split scalar QK256 kernel IDs` | Add precise F32/no-scale and I8S-scaled scalar GEMV/GEMM IDs while keeping compatibility aliases. | `cargo test --locked -p bitnet-quantization --no-default-features --features cpu i2s_qk256 --lib`. |
| CPU-SCALAR-002 | `feat(cpu): add kernel selection for scaled scalar I8S GEMV` | Add selection metadata and strict fallback behavior for scaled scalar GEMV. | Strict scalar is not fallback; strict requested AVX2 cannot silently select scalar; scaled kernel identity reaches selection metadata. |
| CPU-SCALAR-003 | `feat(cpu): route inline-scale QK256 through selected scalar kernel` | Wire `forward_qk256_cpu` inline-scale dispatch through selected scalar/SIMD kernel path. | Strict scalar answer-corpus receipt records `qk256-scalar-i8s-scaled-gemv`, `fallback_used=false`, and unchanged generated IDs. |
| CPU-SCALAR-004 | `diag(cpu): record scalar QK256 hot-path counters` | Add scalar invocation, extraction, materialization, allocation, and workspace counters. | Receipts expose counters; strict scalar run has scaled I8S invocations and no accelerated claim. |
| CPU-SCALAR-005 | `test(cpu): harden scalar I2S/I8S fixtures` | Expand fixtures across rows, cols, tails, scales, patterns, activations, integer dot, act sum, and repeatability. | `cargo test --locked -p bitnet-quantization --no-default-features --features cpu scalar_i8s`. |
| CPU-SCALAR-006 | `test(cpu): record strict scalar BitNet answer corpus` | Emit strict Microsoft I2_S scalar answer-corpus receipt with real GGUF, strict tokenizer, selected scaled scalar kernel, and fallback false. | Tiny corpus passes or exact blocker is recorded; no broad chat claim; `speedup_claim=false`. |
| CPU-SCALAR-007 | `test(cpu): compare scalar baseline against AVX2 receipts` | Compare scalar strict and AVX2 strict receipts for same prompts, IDs, decoded text, and fallback status. | Same prompt IDs, generated IDs, decoded text, first divergence null or classified, fallback false on both. |
| CPU-SCALAR-008 | `perf(cpu): cache QK256 flat packed bytes for scalar dispatch` | Remove per-call flat packed-byte extraction by exposing/borrowing packed views. | Generated IDs unchanged; flat weight extraction count reduced; no whole-matrix dequant. |
| CPU-SCALAR-009 | `perf(cpu): use flat buffers for scalar QK256 rows` | Replace `Vec<Vec<f32>>` input/output materialization with flat buffers and boundary conversion. | Same tensor output and generated IDs; materialization/allocation counters reduced. |
| CPU-SCALAR-010 | `perf(cpu): add reusable scalar CPU workspace` | Add workspace for activation quantization, outputs, scratch codes, and row output reuse. | No per-layer activation allocation; workspace reuse count greater than zero; answers unchanged. |
| CPU-SCALAR-011 | `perf(cpu): optimize scalar I8S activation quantization` | Add exact `quantize_row_i8_s_activation_into` style helper using reusable workspace. | Bit-exact q vector, act sum, scale, int dot, scaled output, and generated IDs. |
| CPU-SCALAR-012 | `feat(cpu): add scaled scalar QK256 prefill GEMM` | Add production scaled I8S scalar GEMM for prefill. | Batched GEMM equals repeated scaled GEMV; prefill receipt can select `qk256-scalar-i8s-scaled-gemm`. |
| CPU-SCALAR-013 | `bench(cpu): add scalar-only BitNet phase receipts` | Emit scalar-only micro, layer, prefill, first-token, decode, and warm-session receipts. | Precise scalar selected kernel, fallback false, no AVX2 selected, no speedup claim. |
| CPU-SCALAR-014 | `diag(cpu): audit scalar transformer support-op timing` | Measure embedding gather, RMSNorm, RoPE, attention, softmax, KV, FFN, output head, sampling, and stop costs. | Report ranks hot scalar support ops; no behavior change; next target selected from evidence. |
| CPU-SCALAR-015 | `perf(cpu): optimize highest-cost scalar support op` | Optimize the highest-cost scalar support op from CPU-SCALAR-014 evidence. | Before/after support-op receipt; generated IDs unchanged; no tolerance drift. |
| CPU-SCALAR-016 | `bench(cpu): establish scalar thread-count envelope` | Measure thread counts for prefill, first token, and decode profiles. | Default scalar thread count chosen by evidence and recorded in receipts. |
| CPU-SCALAR-017 | `docs(cpu): publish scalar BitNet status` | Publish user-facing scalar CPU status and non-claims. | Users can tell scalar correctness, answer, long-decode, performance, default role, and excluded claims. |

## Validation baseline

Run the proof commands listed by the active work item. Runtime PRs should expect
at least these commands unless the selected item narrows or expands them:

```bash
cargo test --locked -p bitnet-quantization --no-default-features --features cpu i2s_qk256 --lib
cargo test --locked -p bitnet-qk256-dispatch --no-default-features --features cpu
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli answer_corpus
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Docs-only PRs may substitute documentation checks plus `git diff --check`, but
must not claim runtime proof.

## Rollback path

Each runtime PR must be reversible to the previous scalar implementation by
restoring the prior kernel-selection path or compatibility alias while retaining
receipt evidence that identifies which kernel was selected. Performance PRs must
include before and after receipts so regressions can be rolled back without
losing scalar correctness proof.
