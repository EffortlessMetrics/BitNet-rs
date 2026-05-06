# Intel NPU Campaign

Campaign ID: `intel-npu`

Status: active

## Objective

Validate Intel Lunar Lake NPU through OpenVINO static-shape detection, smoke, parity, and receipts without colliding with CPU or GPU lanes.

## End State

- Intel NPU backend identity is distinct from Intel GPU, OpenCL, CPU, and generic accelerator labels.
- Probe records device-node, driver, OpenVINO runtime, available devices, NPU visibility, and environment facts.
- Tiny OpenVINO NPU graph smoke executes with fallback=false.
- Static-shape BitNet subgraph parity is receipt-backed before inference claims.

## Hard Constraints

- Device-node detection is not inference.
- OpenVINO NPU smoke is not full BitNet inference.
- CPU fallback cannot count as NPU execution.
- Do not assume WSL can see the NPU unless OpenVINO reports `NPU` inside WSL.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| NPU-002 | merged | Preserve Intel NPU backend identity. |
| NPU-003 | merged | Add runtime detection. |
| NPU-004 | proposed | Add smoke probe command. |
| NPU-005 | proposed | Run tiny OpenVINO NPU graph smoke. |
| NPU-006 | proposed | Add receipt fields. |

## Review Policy

NPU runtime PRs are non-stackable. Do not mix NPU work with A770 OpenCL, Arc 140V, or CPU kernel implementation unless the item explicitly names that dependency.
