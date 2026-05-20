# BitNet CPU AVX2 Status

Status: active
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md`, `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-avx2-bitnet/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Documents exact CPU AVX2 BitNet claim boundaries.
Policy impact: none

## Current status

The CPU proof lane has strict loader, tokenizer, packed-layout, scalar truth,
AVX2 dispatch, decode, receipt, and answer-corpus rails. The next proof gap is
hot-path truth: strict real BitNet inference must prove whether inline-scale
I2_S/QK256 execution is using an optimized scaled I2_S x I8_S AVX2 path or a
scalar, diagnostic, dequantized, materialized, or no-scale F32-style substitute.

## Support table

| Artifact / scope | Scalar | AVX2 scaled I8S | Answer corpus | Long decode | Speed | Server |
| --- | --- | --- | --- | --- | --- | --- |
| Official Microsoft BitNet I2_S/QK256 through Rust CPU path | Correctness oracle | Candidate; hot-path counters required before promotion | Existing strict corpus rails; v2 expansion planned | Planned | Exact profiles only after phase review | Not claimed |

## Required next evidence

Before AVX2 can be promoted beyond candidate status, receipts must show:

- `selected_kernel = "qk256-avx2-i8s-scaled-gemv"` or the final accepted scaled
  AVX2 kernel ID for inline-scale tensors;
- `qk256_hot_path.scaled_i8s_avx2_invocations > 0`;
- `qk256_hot_path.scaled_i8s_scalar_invocations = 0` for strict AVX2 proofs;
- no hidden no-scale F32-only substitution for inline-scale BitNet tensors;
- `fallback_used = false` and `fallback_reason = null`;
- generated-token parity with the scalar packed oracle.

## Current receipt counter surface

`CPU-AVX2-HOTPATH-001` wires the existing QK256 dispatch counters into strict
CPU BitNet run receipts and answer-corpus case rows under `qk256_hot_path`.
Those counters distinguish:

- no-scale F32 scalar/AVX2 GEMV invocations;
- scaled I2_S x I8_S scalar/AVX2 GEMV invocations;
- QK256 flat-byte extraction;
- input-row materialization;
- output-row allocation.

This is a receipt-observability step only. It does not optimize or promote
scaled AVX2 execution and does not make a speedup claim.

## Claim boundary

This status page does not claim GPU, NPU, OpenVINO, CUDA, server readiness,
dense SLM readiness, Qwen readiness, broad chat quality, or global speedup. Any
future speed statement must cite an accepted exact-profile receipt and review.

## Planned promotion path

1. Add QK256 hot-path counters to receipts.
2. Validate receipts against hidden fallback and missing-counter cases.
3. Add scaled I2_S x I8_S scalar-oracle fixtures.
4. Implement and select a scaled AVX2 QK256 GEMV kernel.
5. Wire inline-scale transformer dispatch through the scaled AVX2 selector.
6. Remove avoidable QK256 materialization and add reusable CPU workspace.
7. Add phase timing profiles and review exact-profile promotion decisions.
8. Expand answer corpus and long-decode deterministic parity before broadening
   correctness confidence.
