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
| A770-010 | merged | Make strict A770 OpenCL QK256 dispatch requests fail closed while the production OpenCL QK256 runtime is not wired, and record non-strict A770 requests as CPU fallback/not-routed evidence. |
| A770-011 | merged | Wire a first inline-scaled selected-device A770 OpenCL QK256 dispatch candidate while keeping CPU activation quantization, quality, residency, speed, and full-inference claims closed. |
| A770-012 | merged | Commit a strict selected-device A770 OpenCL QK256 dispatch receipt and wire it into diagnostic route/kernel matrices without promoting inference, quality, residency, speed, or trusted-partial claims. |
| A770-013 | merged | Wire the CLI OpenCL feature into the real BitNet model/QK256 dispatch stack and add strict A770 route receipt fields without promoting inference, quality, residency, speed, or trusted-partial claims. |
| A770-014 | merged | Align the official BitNet model contract route matrix with the committed A770 diagnostic QK256 route without promoting answer quality, residency, speed, or trusted-partial claims. |
| A770-015 | merged | Add an A770 OpenCL answer-corpus route contract for seeded prompt evidence while keeping live execution, answer quality, residency, speed, and trusted-partial claims closed. |
| A770-016 | pr_open | Bind A770 OpenCL answer-corpus child runs to the committed BitNet model contract and diagnostic QK256 route metadata without promoting live answer quality, residency, speed, or full-inference claims. |

## Current Claim Boundary

Committed A770 OpenCL proof remains diagnostic. The campaign does not currently claim full BitNet inference, trusted partial acceleration, performance speedup, support-op residency, full device residency, dense SLM support, Gemma support, or native OpenCL proof from OpenVINO GPU. The selected-device QK256 scaled fixture, A770-011 dispatch candidate, A770-012 strict dispatch receipt, A770-013 CLI route receipt scaffold, A770-014 model-contract route alignment, A770-015 answer-corpus route contract, and A770-016 answer-corpus proof-route binding do not prove GPU-resident activation quantization, answer quality, selected attention residency, resident KV, live answer-corpus execution, or full BitNet inference.

## Review Policy

A770 runtime PRs are non-stackable. Do not combine A770 with Intel NPU, Arc 140V, or CPU proof changes unless the campaign manifest explicitly allows it.
