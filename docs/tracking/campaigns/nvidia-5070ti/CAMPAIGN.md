# NVIDIA 5070 Ti Campaign

Campaign ID: `nvidia-5070ti`

Status: active

## Objective

Validate RTX 5070 Ti as a CUDA-first BitNet acceleration lane with selected-device receipts and no CPU, OpenCL, WGPU, or generic GPU conflation.

## End State

- RTX 5070 Ti CUDA backend identity is distinct from generic CUDA and WGPU.
- CUDA and NVML probe facts are recorded before kernel execution claims.
- CUDA smoke, parity, receipts, and benchmarks are sequenced after identity.

## Hard Constraints

- CUDA visibility is not kernel execution.
- WGPU smoke is not CUDA proof.
- CPU fallback cannot count as CUDA execution.
- Performance claims require driver, CUDA, VRAM, power, and thermal context.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| RTX5070TI-003 | merged | Preserved selected-device CUDA identity in #3679. |
| RTX5070TI-004 | merged | Added CUDA and NVML runtime probe in #3691. |
| RTX5070TI-005 | ready | Run tiny CUDA kernel smoke after #3691. |

## Review Policy

CUDA PRs are non-stackable when they touch backend identity, kernels, receipts, or benchmark interpretation.
