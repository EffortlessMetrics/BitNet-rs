<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# NVIDIA RTX 5070 Ti validation Campaign Status

- Campaign: `nvidia-5070ti`
- State: `active`
- Objective: Validate RTX 5070 Ti as a CUDA-first BitNet acceleration lane with selected-device receipts and no CPU, OpenCL, WGPU, or generic GPU conflation.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| RTX5070TI-003 | pr_open | #3679 | `codex/rtx5070ti-003-backend-identity` | Preserve RTX 5070 Ti requested and selected CUDA backend identity without adding kernels or inference claims. |
| RTX5070TI-004 | proposed | TBD | `codex/nvidia-5070ti/RTX5070TI-004-cuda-nvml-probe` | Add RTX 5070 Ti CUDA and NVML runtime probe without claiming kernel execution or BitNet inference. |

## Hard Constraints

- CUDA visibility is not kernel execution.
- WGPU smoke is not CUDA proof.
- CPU fallback cannot count as CUDA execution.
- Performance claims require driver, CUDA, VRAM, power, and thermal context.
