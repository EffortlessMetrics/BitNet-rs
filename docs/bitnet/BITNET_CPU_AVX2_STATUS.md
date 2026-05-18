# BitNet CPU AVX2 status

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: ../specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md
Linked ADRs: ../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md
Linked plan: ../../plans/cpu-avx2-bitnet/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CPU AVX2 BitNet I2_S/QK256 remains exact-profile and receipt-gated.
Policy impact: none

## Current status

The CPU proof lane has merged strict loader, tokenizer, packed layout, scalar,
AVX2 dispatch, decode, receipt, benchmark, and answer-corpus rails. The next
support question is narrower: whether strict real-model AVX2 inference is
executing the scaled BitNet I2_S x I8_S AVX2 path required by inline-scale
QK256 tensors, rather than a scalar substitute or the no-scale F32-style AVX2
GEMV path.

## Claim boundary

| Area | Status | Claim boundary |
| --- | --- | --- |
| Official Microsoft BitNet I2_S scalar | Correctness oracle | Scalar remains the oracle for AVX2 parity and generated-token gates. |
| AVX2 no-scale QK256 F32-style GEMV | Existing support surface | Does not prove inline-scale BitNet I2_S x I8_S production hot-path execution. |
| AVX2 scaled I2_S x I8_S GEMV | Candidate / unproven | Must be counted, validated, implemented if missing, selected explicitly, and wired before promotion. |
| Strict fallback truth | Required | Requested AVX2 in strict mode must fail if the selected path is scalar, dequantized, diagnostic, mock, or reference-only. |
| Answer corpus | Existing strict proof input | Must remain green for scalar and AVX2, or record first divergence and block promotion. |
| Long decode | Future gate | Requires deterministic scalar-vs-AVX2 token parity or first-divergence evidence. |
| Speed | Not globally claimed | Exact profiles only after phase receipts and performance review. |
| Server | False for this lane | No server readiness claim is made by CPU AVX2 hot-path proof. |
| GPU/NPU/OpenVINO/A770/M4/dense SLM/Qwen | Out of scope | Proof families are separate and cannot inherit this lane's evidence. |

## Near-term board

1. CPU-AVX2-HOTPATH-001: docs/spec/plan/tracker rails.
2. CPU-AVX2-HOTPATH-002: hot-path counters.
3. CPU-AVX2-HOTPATH-003: receipt validator for hidden fallback.
4. CPU-AVX2-HOTPATH-004: scaled I2_S x I8_S fixtures.
5. CPU-AVX2-HOTPATH-005: scaled AVX2 microkernel.
6. CPU-AVX2-HOTPATH-006: explicit scaled kernel selection.
7. CPU-AVX2-HOTPATH-007: dispatch wiring.
8. CPU-AVX2-HOTPATH-008: packed-view/materialization cleanup.
9. CPU-AVX2-HOTPATH-009: reusable decode workspace.
10. CPU-AVX2-HOTPATH-010: phase benchmark receipts.
11. CPU-AVX2-HOTPATH-011: exact-profile performance review.
12. CPU-AVX2-HOTPATH-012: answer corpus v2.
13. CPU-AVX2-HOTPATH-013: long-decode parity.
14. CPU-AVX2-HOTPATH-014: prefill optimization.
15. CPU-AVX2-HOTPATH-015: non-QK256 op bottleneck audit.
16. CPU-AVX2-HOTPATH-016: user-facing support status.

## Next proof

The first runtime proof must add QK256 hot-path counters to strict scalar and
strict AVX2 receipts. It must distinguish no-scale F32 GEMV from scaled
I2_S x I8_S GEMV, keep requested/selected kernel and fallback fields explicit,
preserve scalar-vs-AVX2 answer parity, and make no speed claim.
