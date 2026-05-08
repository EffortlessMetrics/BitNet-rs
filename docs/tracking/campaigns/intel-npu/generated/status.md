<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Intel NPU validation Campaign Status

- Campaign: `intel-npu`
- State: `active`
- Objective: Validate Intel Lunar Lake NPU through OpenVINO static-shape detection, smoke, parity, and receipts without conflating NPU, GPU, or CPU work.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| NPU-002 | merged | #3722 | `codex/intel-npu/NPU-002-lite-backend-identity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Preserve Intel NPU requested and selected backend identity without mapping it to Metal, CUDA, generic GPU, or CPU fallback. |
| NPU-003 | merged | #3739 | `codex/intel-npu/NPU-003-openvino-runtime-probe` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add Intel NPU runtime detection fields that keep OS accelerator evidence separate from OpenVINO NPU visibility and record OpenVINO NPU full name, driver/compiler/memory properties, runtime device, proof_stage=runtime_detected, and fallback_used=false without graph execution claims. |
| NPU-004 | merged | #3830 | `codex/intel-npu/NPU-004-smoke-probe-command` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add an Intel NPU smoke probe command that writes a machine-readable OpenVINO NPU runtime visibility receipt with requested/selected backend identity, runtime API/device, strict mode, proof_stage=runtime_detected, fallback_used=false, and no graph, kernel, or BitNet inference claims. |
| NPU-005 | merged | #3846 | `codex/intel-npu/NPU-005-openvino-graph-smoke` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a tiny static OpenVINO NPU graph smoke command that compiles and runs a fixed F16 matmul-add graph on runtime device NPU when available, writes a machine-readable receipt with shape_mode=static, graph identity, timing, requested/selected backend identity, fallback_used=false, cpu_fallback_allowed=false, and never claims BitNet inference or NPU acceleration. |
| NPU-007 | merged | #3873 | `codex/intel-npu/NPU-007-bitnet-subgraph-parity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add selected static BitNet RMSNorm subgraph parity through OpenVINO NPU with CPU reference comparison, selected backend/runtime identity, timing, fallback_used=false, and no full BitNet inference or QK256 decode claims. |
| NPU-008 | pr_open | #3963 | `codex/intel-npu/NPU-008-linear-subgraph-parity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add selected static BitNet linear-projection subgraph parity through OpenVINO NPU with CPU reference comparison, selected backend/runtime identity, timing, fallback_used=false, and no full BitNet inference, acceleration, or QK256 decode claims. |
| NPU-006 | merged | #3860 | `codex/intel-npu/NPU-006-receipt-fields` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Harden Intel NPU probe and tiny graph smoke receipts with structured backend_runtime, shape_contract, fallback_policy, graph identity, timing, and kernels_or_graphs fields while preserving fallback_used=false and no BitNet inference claims. |

## Hard Constraints

- Device-node detection is not inference.
- OpenVINO NPU smoke is not full BitNet inference.
- CPU fallback cannot count as NPU execution.
- Do not assume WSL can see the NPU unless OpenVINO reports NPU inside WSL.
