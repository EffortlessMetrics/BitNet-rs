<!-- GENERATED: do not edit by hand. Run cargo run --no-default-features -p xtask --no-default-features -- campaign generate. -->
# Intel NPU validation Campaign Status

- Campaign: `intel-npu`
- State: `active`
- Objective: Validate Intel Lunar Lake NPU through OpenVINO static-shape detection, smoke, parity, and receipts without conflating NPU, GPU, or CPU work.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| NPU-002 | merged | #3722 | `codex/intel-npu/NPU-002-lite-backend-identity` | Preserve Intel NPU requested and selected backend identity without mapping it to Metal, CUDA, generic GPU, or CPU fallback. |
| NPU-003 | merged | #3739 | `codex/intel-npu/NPU-003-openvino-runtime-probe` | Add Intel NPU runtime detection fields that keep OS accelerator evidence separate from OpenVINO NPU visibility and record OpenVINO NPU full name, driver/compiler/memory properties, runtime device, proof_stage=runtime_detected, and fallback_used=false without graph execution claims. |

## Hard Constraints

- Device-node detection is not inference.
- OpenVINO NPU smoke is not full BitNet inference.
- CPU fallback cannot count as NPU execution.
- Do not assume WSL can see the NPU unless OpenVINO reports NPU inside WSL.
