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
- Tracker, route matrix, capability matrix, and committed receipt inventory agree before route promotion.

## Hard Constraints

- OpenCL-first for native A770 proof.
- OpenVINO GPU is reference only.
- CPU fallback cannot count as A770 execution.
- Performance claims require driver, PCIe, ReBAR, VRAM, power, and thermal context.
- Full inference, trusted partial acceleration, support-op residency, and dense-model claims require committed claim-grade receipts before promotion.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| A770-OPENCL-TRUTH-001 | in_progress | Reconcile tracker, route matrix, kernel matrix, and committed proof inventory before any A770 OpenCL claim promotion. |
| A770-003 | ready | Preserve selected-device identity after truth reconciliation. |
| A770-004 | proposed | Add runtime probe. |
| A770-005 | proposed | Run OpenCL smoke. |
| A770-006 | proposed | Add CPU/OpenCL parity. |
| A770-007 | proposed | Record receipt identity. |

## Review Policy

A770 runtime PRs are non-stackable. Do not combine A770 with Intel NPU, Arc 140V, or CPU proof changes unless the campaign manifest explicitly allows it.

## Current Evidence Reconciliation

As of 2026-05-18, the committed repository does not contain claim-grade A770
OpenCL proof receipts under `ci/hardware/amd-5700x-intel-a770/`,
`ci/hardware/intel-arc-a770/`, or `docs/reports/` beyond non-proof
reconciliation notes; the A770 campaign event log contains tracker state only. Therefore the campaign remains diagnostic-only for A770 BitNet QK256,
embedding, and tied LM-head routes. The uploaded or local transcript state that
mentions full inference, all QK256 linears on A770, and CPU/A770 greedy-token
parity is not promoted unless those artifacts are committed and pass the
claim-boundary gates.

The next runtime-facing steps remain selected-device OpenCL smoke, kernel
compile smoke, official QK256 scalar/OpenCL parity, strict dispatch routing,
quality-gated answer receipts, phase timing, and same-device history receipts.
