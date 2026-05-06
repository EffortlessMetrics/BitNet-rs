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

## Work Items

| Work item | Status | Notes |
|---|---|---|
| A770-003 | ready | Preserve selected-device identity. |
| A770-004 | proposed | Add runtime probe. |
| A770-005 | proposed | Run OpenCL smoke. |
| A770-006 | proposed | Add CPU/OpenCL parity. |
| A770-007 | proposed | Record receipt identity. |

## Review Policy

A770 runtime PRs are non-stackable. Do not combine A770 with Intel NPU, Arc 140V, or CPU proof changes unless the campaign manifest explicitly allows it.
