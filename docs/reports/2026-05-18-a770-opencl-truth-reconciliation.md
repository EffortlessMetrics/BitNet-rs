# A770 OpenCL Truth Reconciliation

Status: diagnostic
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs:
- `docs/specs/intel-arc-a770-gpu-roadmap.md`
- `docs/specs/a770-bitnet-claim-boundary.md`
Linked ADRs:
- `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Linked plan:
- `plans/a770-bitnet-claim-boundary-implementation.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion
Policy impact: none

## Finding

No committed claim-grade A770 OpenCL BitNet inference receipt was found in the
repository for the transcript-level state that described full inference or all
QK256 linears running on A770. Repository evidence remains limited to explicit
A770 route declarations and diagnostic capability rows with empty proof receipt
lists.

## Reconciled Source of Truth

| Surface | Reconciled state |
|---|---|
| Active campaign | `A770-OPENCL-TRUTH-000` is the current ready item; backend-identity work follows after this reconciliation. |
| Route matrix | A770 BitNet QK256, embedding, and LM-head routes remain `diagnostic` with empty proof receipts. |
| Kernel capability matrix | A770 BitNet QK256, embedding, and LM-head capabilities remain `diagnostic`; support ops, dense SLM, Gemma, and full residency remain missing or unsupported. |
| Model contract | A770 is `diagnostic` with `target_support: trusted_partial`; the target is not claim-ready until receipts land. |
| Claim ledger | `a770.bitnet.trusted_partial_experience` remains `diagnostic` with no evidence. |

## Claim Boundary

This reconciliation may claim only that the repository truth has been narrowed to
diagnostic A770 OpenCL BitNet status pending committed proof. It must not claim:

- A770 OpenCL BitNet execution works.
- Full inference is proven on A770.
- CPU, CUDA, OpenVINO GPU, generic OpenCL, or Arc 140V evidence proves native
  A770 OpenCL execution.
- QK256-only or route declaration evidence proves full residency, attention/KV
  residency, dense SLM support, Gemma support, server readiness, or speedup.

## Next Required Evidence

A later runtime PR may promote A770 status only after it commits strict
selected-device OpenCL receipts under `ci/hardware/amd-5700x-intel-a770/<date>/`,
links those receipts from the route and capability matrices, and records quality,
fallback, route identity, model identity, and timing evidence required by the
A770 claim-boundary plan.
