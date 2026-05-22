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
| A770-016 | merged | Bind A770 OpenCL answer-corpus child runs to the committed BitNet model contract and diagnostic QK256 route metadata without promoting live answer quality, residency, speed, or full-inference claims. |
| A770-017 | merged | Align startup OpenCL runtime detection with the in-process dynamic OpenCL probe so A770 answer-corpus child runs can reach the committed diagnostic route without depending on `clinfo`. |
| A770-018 | merged | Record the first committed live five-case A770 OpenCL answer-corpus diagnostic receipt for the official BitNet 2B I2_S model, keeping reference parity, broad answer quality, residency, speed, and trusted-partial claims closed. |
| A770-019 | merged | Restore the OpenCL runtime availability build by matching the selected-device runtime probe wrapper type, keeping A770 answer quality, parity, residency, speed, and completion claims closed. |
| A770-020 | merged | Record same-box AMD 5700X AVX2 CPU and Intel A770 OpenCL answer-corpus receipts with one-step top-k logits, compare them with the answer-parity tool, and preserve the first CPU/A770 logits divergence without claiming exact parity. |
| A770-021 | merged | Add a compact logits top-k frontier to the CPU/A770 answer-parity receipt, separating same-output top-k drift from generated-output drift without changing runtime math or promoting parity. |
| A770-022 | merged | Record focused multi-step CPU/A770 logits for the generated-output divergent `yes_no_water` case and classify whether the first generated token mismatch has logit context. |
| A770-023 | merged | Add compact first-mismatch cross-chosen logit-margin evidence for the generated-output divergent `yes_no_water` case. |
| A770-024 | merged | Add a seeded A770 BitNet answer-readiness corpus contract before the next live CPU/A770 quality and parity run. |
| A770-025 | merged | Record the live seeded CPU AVX2 versus Intel A770 OpenCL answer-readiness run, including quality failures and parity divergence without promoting readiness or parity. |
| A770-026 | in_progress | Classify the A770-025 answer-readiness quality failures and CPU/A770 parity frontier before any scorer or runtime change. |

## Current Claim Boundary

Committed A770 OpenCL proof remains diagnostic. The campaign does not currently claim full BitNet inference, trusted partial acceleration, performance speedup, support-op residency, full device residency, dense SLM support, Gemma support, or native OpenCL proof from OpenVINO GPU. The selected-device QK256 scaled fixture, A770-011 dispatch candidate, A770-012 strict dispatch receipt, A770-013 CLI route receipt scaffold, A770-014 model-contract route alignment, A770-015 answer-corpus route contract, A770-016 answer-corpus proof-route binding, A770-017 OpenCL runtime detection alignment, A770-018 live five-case answer-corpus diagnostic receipt, A770-019 OpenCL runtime-probe build fix, A770-020 CPU/A770 answer-parity diagnostic, A770-021 logits top-k frontier, A770-022 multi-step generated-output frontier, A770-023 first-mismatch logit-margin frontier, A770-024 seeded answer-readiness corpus contract, A770-025 live seeded answer-readiness diagnostic, and A770-026 answer-readiness failure-frontier classification do not prove GPU-resident activation quantization, broad answer quality, selected attention residency, resident KV, reference parity, strict A770 answer readiness, CPU/A770 answer parity, or full BitNet inference.

## Review Policy

A770 runtime PRs are non-stackable. Do not combine A770 with Intel NPU, Arc 140V, or CPU proof changes unless the campaign manifest explicitly allows it.
