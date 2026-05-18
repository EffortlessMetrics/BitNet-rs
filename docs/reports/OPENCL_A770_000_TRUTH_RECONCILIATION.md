# OPENCL_A770_000 Truth Reconciliation

Status: complete
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs:

- docs/specs/intel-arc-a770-gpu-roadmap.md
- docs/specs/a770-bitnet-claim-boundary.md

Linked ADRs: n/a
Linked plan: plans/a770-bitnet-claim-boundary-implementation.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: keeps A770 OpenCL at diagnostic claim level until committed receipts prove promotion
Policy impact: n/a

## Scope

This report reconciles the committed repository state for the Intel Arc A770
OpenCL BitNet lane. It does not add kernels, dispatch behavior, benchmark
claims, or inference claims.

## Inspection Summary

The committed repository does not contain claim-grade A770 receipts matching the
later transcript state described outside the repository. In particular, the
committed A770 hardware directory contains only the capability matrix and no
per-run receipt bundle under a dated A770 run directory. The committed campaign
event directory contains only its placeholder. No local branch whose name
matches `a770` or `A770` exists in this checkout.

The committed source of truth is therefore the conservative state:

- `ci/hardware/device-kernel-routing.toml` keeps A770 BitNet QK256, embedding,
  and LM-head routes at `claim_level = "diagnostic"` with empty proof receipts.
- `ci/hardware/amd-5700x-intel-a770/a770-kernel-capability-matrix.json` keeps
  QK256, embedding, and LM-head kernels diagnostic or missing with empty proof
  receipts.
- `ci/claims/claim-ledger.json` keeps the trusted-partial A770 experience claim
  at `status = "diagnostic"`.
- The A770 model contract now records the current support state as diagnostic
  while preserving `target_support: trusted_partial` for the intended promotion.
- The campaign tracker now makes this reconciliation the next ready work item,
  ahead of backend-identity, probe, smoke, parity, and receipt work.

## Decision

Because no committed claim-grade A770 proof receipts were found, this PR does
not promote any A770 OpenCL route. The repo remains diagnostic for A770 BitNet
OpenCL until later PRs land selected-device smoke, compile/execution proof,
strict QK256 routing, quality gates, phase timing, and same-device history.

## Claim Boundary

This reconciliation may claim only:

```text
The committed A770 OpenCL source-of-truth files agree that trusted partial
BitNet A770 acceleration is not yet claim-grade.
```

It must not claim:

```text
A770 OpenCL execution works.
A770 QK256/embedding/LM-head operations ran with fallback_used=false.
Full BitNet inference works on A770.
Full support-op residency or full device residency is proven.
Dense SLM, Gemma, or small LLM OpenCL support exists.
OpenVINO GPU proof is native OpenCL proof.
Generic OpenCL or Arc 140V proof is A770 proof.
```

## Rollback Plan

Revert this documentation/tracker reconciliation if a branch with claim-grade
A770 receipts is intentionally merged first. Any such replacement must commit
receipts under the A770 hardware/report/tracking paths and update the route
matrix, kernel capability matrix, claim ledger, model contract, active goal,
and campaign tracker in one coherent promotion PR.
