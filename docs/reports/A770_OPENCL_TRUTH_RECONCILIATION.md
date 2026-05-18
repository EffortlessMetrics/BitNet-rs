# A770 OpenCL truth reconciliation

Status: diagnostic reconciliation
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs:
- `docs/specs/intel-arc-a770-gpu-roadmap.md`
- `docs/specs/a770-bitnet-claim-boundary.md`
Linked ADRs: n/a
Linked plan: `plans/a770-bitnet-claim-boundary-implementation.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion
Policy impact: none

## Decision

Keep the A770 OpenCL BitNet route at `diagnostic` until claim-grade receipts are
committed and validated. This reconciliation PR does not land kernels, dispatch
changes, benchmark data, or model-inference receipts.

The current repository contains route and capability declarations for the A770
OpenCL lane, but it does not contain committed proof artifacts that justify a
full-inference, trusted-partial, performance, or residency promotion.

## Evidence inventory

| Surface | Current committed state | Reconciliation result |
|---|---|---|
| Campaign active goal | A770 lane objective is OpenCL-first validation with selected-device receipts; A770 backend identity remains a prerequisite. | Add a dedicated truth-reconciliation item before follow-on A770 execution work. |
| Campaign narrative | A770-003 was the only ready item and A770-004 through A770-007 were future proposed work. | Record that proof inventory must agree with route and capability matrices before promotion. |
| Route matrix | A770 BitNet QK256, embedding, and tied LM-head routes are present but `claim_level = "diagnostic"` with empty `proof_receipts`. | Keep them diagnostic and explicitly tie the reason to missing committed claim-grade receipts. |
| Kernel capability matrix | QK256, embedding, and tied LM-head are `diagnostic`; dense, support-op, KV, attention, and full-residency rows remain missing. | Keep diagnostic/missing status and clarify that no full-inference route is proved. |
| Receipt paths | No A770 proof receipts were found under `ci/hardware/amd-5700x-intel-a770/`, `ci/hardware/intel-arc-a770/`, or `docs/reports/` beyond non-proof reconciliation notes; campaign events contain tracker state only. | Do not promote the uploaded/transcript proof state. |
| Runtime dispatch | No reconciliation change is made to dispatch. | Future PRs must add strict OpenCL dispatch and receipts before route promotion. |

## Claim boundary after reconciliation

Allowed claim for this PR:

```text
The A770 OpenCL proof state is reconciled as diagnostic-only in committed repo
state until claim-grade receipts are present.
```

Not claimed by this PR:

```text
A770 OpenCL execution works
BitNet inference works on A770 OpenCL
trusted partial A770 acceleration is claim-ready
full A770 inference is proved
QK256 linears, embedding, or LM-head are production-routed through A770 OpenCL
attention, KV, softmax, RMSNorm, RoPE, support-op, or full-device residency
dense SLM, Gemma, or small-LLM support on A770 OpenCL
performance or speedup
server readiness
```

## Follow-up gates

The next OpenCL PRs should prove, in order:

1. selected-device A770 OpenCL smoke with fallback false;
2. OpenCL kernel compile smoke with build logs;
3. official QK256 scalar fixtures for grouped layout and scaled I2_S × I8_S math;
4. A770 OpenCL QK256 parity;
5. strict dispatch routing with fail-closed fallback semantics;
6. one-token and answer-corpus quality receipts;
7. long-decode behavior receipts;
8. phase timing and same-device history receipts;
9. a separate promotion PR if and only if the preceding receipts pass.

## Rollback plan

Revert this documentation/tracker reconciliation if claim-grade A770 receipts are
landed in the same branch and the route matrix, capability matrix, campaign
tracker, and generated dashboards are updated to point to those receipts.
