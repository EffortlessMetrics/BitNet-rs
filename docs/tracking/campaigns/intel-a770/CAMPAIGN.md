# Intel Arc A770 Campaign

Campaign ID: `intel-a770`

Status: active

## Objective

Make the Intel Arc A770 a receipt-backed OpenCL-first BitNet acceleration lane. OpenVINO GPU is a reference lane and must not be used as native OpenCL proof.

## End State

- A770 backend identity is distinct from generic OpenCL, Intel NPU, Arc 140V, CUDA, and CPU fallback.
- Runtime probe records OpenCL, Level Zero, OpenVINO GPU, PCI, VRAM, ReBAR, and render-node facts.
- Tiny OpenCL smoke executes with fallback=false.
- CPU/OpenCL parity exists for one kernel or subgraph.
- Receipts preserve selected device identity before performance claims.

## Hard Constraints

- OpenCL-first for native A770 proof.
- OpenVINO GPU is reference only.
- CPU fallback cannot count as A770 execution.
- Performance claims require driver, PCIe, ReBAR, VRAM, power, and thermal context.

## Current Reconciliation

The current repository has no committed claim-grade A770 OpenCL BitNet inference
receipt. The route matrix, kernel capability matrix, claim ledger, and model
contract therefore remain at `diagnostic` for A770 BitNet until a later PR lands
strict selected-device OpenCL execution receipts. Local or uploaded transcript
claims are not product claims unless their receipt artifacts are committed under
`ci/hardware/amd-5700x-intel-a770/<date>/`, summarized in `docs/reports/`, and
linked from the matrices.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| A770-OPENCL-TRUTH-000 | ready | Reconcile tracker, route, capability, claim, and contract state before runtime work. |
| A770-003 | proposed | Preserve selected-device identity after reconciliation lands. |
| A770-004 | proposed | Add runtime probe. |
| A770-005 | proposed | Run OpenCL smoke. |
| A770-006 | proposed | Add CPU/OpenCL parity. |
| A770-007 | proposed | Record receipt identity. |

## Review Policy

A770 runtime PRs are non-stackable. Do not combine A770 with Intel NPU, Arc 140V, or CPU proof changes unless the campaign manifest explicitly allows it.
