# Intel Arc A770 Campaign

Campaign ID: `intel-a770`

Status: active

## Objective

Make the Intel Arc A770 a receipt-backed OpenCL-first BitNet acceleration lane. OpenVINO GPU is a reference lane and must not be used as native OpenCL proof. Uncommitted transcript evidence cannot promote committed A770 claims.

## End State

- A770 backend identity is distinct from generic OpenCL, Intel NPU, Arc 140V, CUDA, and CPU fallback.
- Tracker, route matrix, kernel matrix, model contract, and claim ledger agree on the committed proof level.
- Runtime probe records OpenCL, Level Zero, OpenVINO GPU, PCI, VRAM, ReBAR, and render-node facts.
- Tiny OpenCL smoke executes with fallback=false.
- CPU/OpenCL parity exists for one kernel or subgraph.
- Receipts preserve selected device identity before performance claims.

## Hard Constraints

- OpenCL-first for native A770 proof.
- Uncommitted transcript evidence cannot promote A770 claims.
- OpenVINO GPU is reference only.
- Generic OpenCL and Arc 140V proof cannot count as A770 proof.
- CPU fallback cannot count as A770 execution.
- Performance claims require driver, PCIe, ReBAR, VRAM, power, and thermal context.

## Work Items

| Work item | Status | Notes |
| --- | --- | --- |
| A770-000 | merged | Reconcile committed tracker, matrices, claim ledger, and model contract before runtime work. |
| A770-003 | merged | Preserve selected-device identity after reconciliation. |
| A770-004 | merged | Add runtime probe. |
| A770-005 | merged | Tiny selected-device OpenCL smoke on real A770 DEV_56A0, without BitNet inference claims. |
| A770-006 | merged | Add selected-device OpenCL `matmul_i2s` CPU parity. |
| A770-007 | merged | Record receipt identity. |
| A770-006R | merged | Refresh the `matmul_i2s` parity fixture with explicit activation and packed-weight operand ordering before benchmark-baseline work. |
| A770-008 | merged | Record diagnostic benchmark-baseline timing for the selected-device `matmul_i2s` parity fixture without speedup or production BitNet claims. |
| A770-009 | merged | Add selected-device A770 OpenCL parity for grouped QK256 I2_S bytes with prequantized I8_S activation scale/sum correction, still fixture-only. |
| A770-010 | in progress | Make strict A770 OpenCL QK256 dispatch requests fail closed while the production OpenCL QK256 runtime is not wired, and record non-strict A770 requests as CPU fallback/not-routed evidence. |

## Current Claim Boundary

Committed A770 OpenCL proof remains diagnostic. The campaign does not currently claim full BitNet inference, trusted partial acceleration, performance speedup, support-op residency, full device residency, dense SLM support, Gemma support, or native OpenCL proof from OpenVINO GPU. The selected-device QK256 scaled fixture is not production transformer dispatch, and strict A770 QK256 requests must fail closed until the OpenCL runtime is wired.

## Review Policy

A770 runtime PRs are non-stackable. Do not combine A770 with Intel NPU, Arc 140V, or CPU proof changes unless the campaign manifest explicitly allows it.
