<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# NVIDIA RTX 5070 Ti validation Campaign Status

- Campaign: `nvidia-5070ti`
- State: `active`
- Objective: Validate RTX 5070 Ti as a CUDA-first BitNet acceleration lane with selected-device receipts and no CPU, OpenCL, WGPU, or generic GPU conflation.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| RTX5070TI-003 | merged | #3679 | `codex/rtx5070ti-003-backend-identity` | Preserve RTX 5070 Ti requested and selected CUDA backend identity without adding kernels or inference claims. |
| RTX5070TI-004 | merged | #3691 | `codex/nvidia-5070ti/RTX5070TI-004-cuda-nvml-probe` | Add RTX 5070 Ti CUDA and NVML runtime probe without claiming kernel execution or BitNet inference. |
| RTX5070TI-005 | merged | #3723 | `codex/nvidia-5070ti/RTX5070TI-005-smoke-receipt` | Compile and run a tiny CUDA kernel on the selected RTX 5070 Ti with a fallback-free smoke receipt and no BitNet inference or speedup claim. |
| RTX5070TI-006 | ready | TBD | `codex/nvidia-5070ti/RTX5070TI-006-cuda-cpu-parity` | Run one deterministic fixture through 9950X3D CPU reference and RTX 5070 Ti CUDA target, record error metrics and fallback-free selected-device identity, and emit mismatch debug artifacts without full inference claims. |

## Hard Constraints

- CUDA visibility is not kernel execution.
- WGPU smoke is not CUDA proof.
- CPU fallback cannot count as CUDA execution.
- Performance claims require driver, CUDA, VRAM, power, and thermal context.
