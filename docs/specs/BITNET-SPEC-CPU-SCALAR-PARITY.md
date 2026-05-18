# BitNet CPU Scalar Parity Contract

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`, `docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-scalar/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Establishes scalar CPU as oracle for optimized lanes.
Policy impact: No policy exception.

## Purpose

Scalar CPU QK256/I2_S defines the correctness floor for accelerated kernels.
Optimized lanes compare to scalar; scalar does not compare to optimized lanes
for correctness.

## Parity Levels

| Level | Required proof |
| --- | --- |
| byte layout | Exact equality for canonical packed bytes, block geometry, row stride, and tensor alignment. |
| block unpack | Exact code extraction for all QK256 groups and tails. |
| integer dot | Exact integer accumulation behavior, including documented wrapping or widening semantics. |
| scaled I8S output | Exact output or documented scalar tolerance already recorded in this parity policy. |
| model logits | Bounded top-k/token evidence with model SHA, tokenizer source, prompt IDs, and selected scalar kernel. |
| generated IDs | Exact greedy equality when comparing scalar variants or scalar against an optimized lane intended to be semantically identical. |
| answer text | Quality-gated by the answer corpus without making broad chat-quality claims. |

## Tolerance Policy

No new tolerance may be introduced by implementation PRs unless this contract is
updated in the same PR with:

- the affected level;
- the numerical bound;
- the reason exact equality is not possible;
- fixtures or receipts demonstrating the bound;
- rollback guidance.

## Divergence Classification

When scalar output diverges from another lane, receipts should classify the
first divergence as one of:

- prompt/tokenizer/template mismatch;
- byte-layout or pack/unpack mismatch;
- activation quantization mismatch;
- integer-dot mismatch;
- scaling or tail-column mismatch;
- transformer support-op mismatch;
- sampler/stop-policy mismatch;
- optimized-lane backend defect.

Scalar receipts may also expose diagnostic no-scale F32 evidence, but that path
must not be treated as the production BitNet I8S scaled result.

## Claim Boundary

Passing scalar parity proves a bounded scalar CPU oracle for the named model,
prompt, tokenizer, layout, and kernel. It does not prove SIMD speedup, GPU/NPU
support, server readiness, or broad answer quality.
