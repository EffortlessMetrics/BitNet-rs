# OPENCL_A770_000 Truth Reconciliation

Status: diagnostic-current-state preserved
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/intel-arc-a770-gpu-roadmap.md`, `docs/specs/a770-bitnet-claim-boundary.md`
Linked ADRs: n/a
Linked plan: `plans/a770-bitnet-claim-boundary-implementation.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion
Policy impact: n/a

## Decision

The current repository does not contain claim-grade A770 OpenCL BitNet proof
artifacts matching the transcript-level state described in the handoff. The
committed truth therefore remains diagnostic for BitNet QK256, embedding, and
LM-head A770 OpenCL routes, and unsupported for selected attention, dense SLM,
Gemma-class, support-op residency, and full-device residency routes.

No full-inference, full-residency, dense-model, server-readiness, or speedup
claim is made by this reconciliation.

## Inspection Summary

- `ci/hardware/amd-5700x-intel-a770/` contains the A770 kernel capability
  matrix but no dated claim-grade receipt directory.
- `docs/tracking/campaigns/intel-a770/events/` contains no A770 proof event
  beyond the placeholder.
- `docs/reports/` contains no A770 OpenCL BitNet proof report with committed
  receipt paths.
- `ci/hardware/device-kernel-routing.toml` already classifies A770 BitNet
  QK256, embedding, and LM-head routes as `diagnostic` with empty
  `proof_receipts`.
- `ci/hardware/amd-5700x-intel-a770/a770-kernel-capability-matrix.json` already
  classifies A770 BitNet QK256, embedding, and LM-head kernels as `diagnostic`
  with empty `proof_receipts`.
- `ci/claims/claim-ledger.json` and generated `docs/claims.md` keep the
  trusted-partial experience claim at `diagnostic`.
- The model contract now records `support: diagnostic` and
  `target_support: trusted_partial` so target intent cannot be mistaken for a
  current public support claim.

## Reconciled Current State

| Surface | Current state | Reconciled action |
|---|---|---|
| Campaign active item | Truth reconciliation is first | Added `A770-000` as the ready item and made `A770-003` depend on it. |
| Campaign narrative | A770-003 was previously first ready item | Added an explicit diagnostic-current-state note. |
| Model contract | Target support was encoded as current support | Split current `support: diagnostic` from `target_support: trusted_partial`. |
| Route matrix | QK256 / embedding / LM-head diagnostic | Kept diagnostic with empty proof receipts. |
| Kernel matrix | QK256 / embedding / LM-head diagnostic | Kept diagnostic with empty proof receipts. |
| Claim ledger | Trusted partial experience diagnostic | Kept diagnostic. |
| Receipts | No claim-grade A770 BitNet receipts committed | No route promotion. |

## Claim Boundary

Allowed claim for this PR:

```text
The committed A770 OpenCL BitNet source-of-truth files agree that current route
state is diagnostic or unsupported until claim-grade receipts land.
```

Not claimed:

```text
A770 OpenCL execution works
BitNet inference works on A770
QK256 linears ran on A770
embedding ran on A770
LM-head logits ran on A770
fallback_used=false for a BitNet A770 run
full inference
full device residency
performance speedup
server readiness
dense SLM/Gemma/small LLM support
```

## Rollback Plan

Revert this report, the campaign manifest/dashboard updates, and the model
contract support split. Because no runtime code or kernels changed, rollback is
limited to source-of-truth documentation and tracking files.
