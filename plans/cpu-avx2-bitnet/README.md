# CPU AVX2 BitNet Hot-Path Plan

This plan scopes the CPU AVX2 BitNet campaign that follows the completed
loader, tokenizer, packed-layout, scalar-kernel, AVX2-dispatch, decode, receipt,
and answer-parity rails in the `cpu-proof` tracker.

## Goal

AVX2 CPU support is fully working only when the official Microsoft BitNet
I2_S/QK256 model runs through the normal Rust CPU user path with:

- strict GGUF loader authority;
- strict tokenizer authority;
- canonical packed QK256 layout;
- scalar packed kernels as the correctness oracle;
- selected AVX2 BitNet kernels for the real production hot path;
- no hidden scalar, dequantized, diagnostic, mock, or reference-only fallback;
- scalar-vs-AVX2 generated-token parity; and
- phase-timed prefill, first-token, and decode receipts good enough to promote
  exact profiles one by one.

## Scope boundary

This campaign is only for CPU AVX2 BitNet I2_S/QK256 execution. It does not
claim CUDA, NPU, OpenVINO, Intel Arc A770, Apple M4, dense SLM, Qwen, server
readiness, or broad chat quality.

## Source-of-truth links

- Spec: [`docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md`](../../docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md)
- Implementation sequence: [`implementation-plan.md`](implementation-plan.md)
- Status: [`docs/bitnet/BITNET_CPU_AVX2_STATUS.md`](../../docs/bitnet/BITNET_CPU_AVX2_STATUS.md)
- Campaign tracker: [`docs/tracking/campaigns/cpu-proof/active.toml`](../../docs/tracking/campaigns/cpu-proof/active.toml)

## Near-term board

| Item | Purpose | Claim boundary |
| --- | --- | --- |
| CPU-AVX2-HOTPATH-000 | Add the docs/spec/plan/tracker rails. | Planning only; no runtime change. |
| CPU-AVX2-HOTPATH-001 | Record QK256 hot-path execution counters. | Observability only; no speed claim. |
| CPU-AVX2-HOTPATH-002 | Validate receipts for hidden fallback. | Receipt truth only; no new kernel. |
| CPU-AVX2-HOTPATH-003 | Add scaled I2_S x I8_S parity fixtures. | Fixture behavior only. |
| CPU-AVX2-HOTPATH-004 | Implement scaled I2_S x I8_S AVX2 GEMV. | Microkernel parity only until wired. |
| CPU-AVX2-HOTPATH-005 | Select scaled AVX2 kernels explicitly. | Selection metadata only. |
| CPU-AVX2-HOTPATH-006 | Route inline-scale transformer dispatch through scaled AVX2. | Strict AVX2 path only if counters prove it. |
| CPU-AVX2-HOTPATH-007 | Cache packed views and reduce materialization. | Performance evidence only with receipts. |
| CPU-AVX2-HOTPATH-008 | Add reusable CPU decode workspace. | Allocation reduction only. |
| CPU-AVX2-HOTPATH-009 | Add strict phase timing profiles. | Raw timing evidence only. |
| CPU-AVX2-HOTPATH-010 | Review exact-profile performance qualification. | Exact-profile promotions only. |
| CPU-AVX2-HOTPATH-011 | Expand BitNet CPU answer corpus v2. | Corpus quality evidence only. |
| CPU-AVX2-HOTPATH-012 | Add long-decode deterministic parity. | Parity evidence only. |
| CPU-AVX2-HOTPATH-013 | Optimize prefill path. | Prefill evidence only. |
| CPU-AVX2-HOTPATH-014 | Profile non-QK256 CPU support ops. | Bottleneck ranking only. |
| CPU-AVX2-HOTPATH-015 | Publish user-facing AVX2 support status. | Status reflects proven receipts only. |
