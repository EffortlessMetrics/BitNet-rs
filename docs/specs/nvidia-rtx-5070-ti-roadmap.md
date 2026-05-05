# NVIDIA RTX 5070 Ti Roadmap

## Purpose

This document defines the NVIDIA RTX 5070 Ti validation lane for BitNet-rs. The 5070 Ti lane is CUDA-first, with wgpu/Vulkan/D3D12 as a cross-platform reference lane.

Primary labels:

```text
nvidia-rtx-5070-ti-cuda
nvidia-rtx-5070-ti-wgpu
```

The first useful milestone is CUDA kernel smoke with a receipt proving `selected_backend=nvidia-rtx-5070-ti-cuda` and `fallback_used=false`.

## Hardware Baseline

Expected RTX 5070 Ti facts:

| Property | Expected value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Architecture | Blackwell |
| CUDA cores | 8,960 |
| Tensor cores | 5th-generation |
| AI performance | 1,406 AI TOPS |
| RT cores | 4th-generation |
| Memory | 16GB GDDR7 |
| Memory bus | 256-bit |
| CUDA compute capability | 12.0 |
| Graphics power | 300W |
| Recommended system power | 750W |

Board partner cards may vary in clocks, cooling, and power limits. Receipts must record the actual board and runtime-reported values.

## Claim Boundary

- PCI detection is not CUDA runtime proof.
- CUDA runtime visibility is not CUDA kernel execution.
- CUDA kernel smoke is not CPU/CUDA parity.
- CPU/CUDA parity is not full inference.
- CUDA proof is not wgpu/Vulkan proof.
- wgpu/Vulkan smoke is not CUDA kernel proof.
- CPU fallback cannot count as CUDA execution.

## Runtime Paths

### Native CUDA Path

Milestones:

1. NVIDIA driver and CUDA runtime visibility.
2. NVML visibility when available.
3. Strict selected-device identity.
4. Tiny CUDA kernel smoke.
5. CPU/CUDA parity.
6. Receipt-backed CUDA kernel/subgraph proof.
7. Benchmark baseline with driver, CUDA version, compute capability, VRAM, power, and temperature.

### wgpu/Vulkan Reference Path

The wgpu lane is for shader portability and non-CUDA comparison.

Milestones:

1. Adapter probe.
2. Tiny shader smoke.
3. CPU/wgpu parity.
4. Receipt records backend API, adapter name, driver, and fallback status.

Do not use wgpu smoke as CUDA kernel proof.

## Receipt Fields

Minimum CUDA receipt:

```json
{
  "requested_backend": "nvidia-rtx-5070-ti",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "compute_capability": "12.0",
  "vram_bytes": 17179869184,
  "driver_version": "...",
  "cuda_version": "...",
  "fallback_backend": null,
  "fallback_used": false
}
```

Minimum wgpu receipt:

```json
{
  "requested_backend": "nvidia-rtx-5070-ti-wgpu",
  "selected_backend": "nvidia-rtx-5070-ti-wgpu",
  "runtime_api": "wgpu",
  "backend_api": "vulkan|d3d12",
  "adapter_name": "NVIDIA GeForce RTX 5070 Ti",
  "fallback_used": false
}
```

## Validation Bundle

The machine bundle lives in:

```text
docs/hardware/nvidia-rtx-5070-ti-validation.md
```

It must collect:

- OS and kernel/build.
- PCI device identity.
- NVIDIA driver version.
- CUDA version.
- Compute capability.
- VRAM.
- Power limit and current draw when available.
- Temperature when available.
- Optional Vulkan/wgpu visibility.

## Work Plan

### RTX5070TI-001 - Add Backend Lane

Docs/tracking only. Add CUDA-first and wgpu reference lanes.

### RTX5070TI-002 - Machine Profile

Collect OS, PCI, driver, CUDA, compute capability, VRAM, power, and optional Vulkan data.

### RTX5070TI-003 - Backend Identity

Preserve requested and selected backend identity for RTX 5070 Ti CUDA.

### RTX5070TI-004 - CUDA/NVML Probe

Report CUDA runtime visibility, NVML data, driver, CUDA version, compute capability, VRAM, power, and temperature.

### RTX5070TI-005 - CUDA Kernel Smoke

Compile and run a tiny CUDA kernel on RTX 5070 Ti.

### RTX5070TI-006 - CPU/CUDA Parity

Compare one CUDA kernel/subgraph output against CPU.

### RTX5070TI-007 - Receipts

Record CUDA runtime, driver, compute capability, VRAM, power, fallback status, and kernel IDs.

### RTX5070TI-008 - Benchmark Baseline

Compare CPU against RTX 5070 Ti CUDA for the validated kernel/subgraph.

### RTX5070TI-009 - wgpu/Vulkan Smoke

Run a tiny wgpu/Vulkan/D3D12 shader smoke as a cross-platform reference path.

## Do Not

- Do not make CUDA a generic GPU claim.
- Do not count CPU fallback as CUDA execution.
- Do not count wgpu/Vulkan smoke as CUDA kernel proof.
- Do not omit compute capability from receipts.
- Do not make benchmark claims without driver, CUDA, VRAM, power, and temperature context.

## Related Contract Docs

- `docs/hardware/HARDWARE_MATRIX.md`
- `docs/hardware/PROOF_STAGES.md`
- `docs/hardware/LANE_OWNERSHIP.md`
- `docs/hardware/BENCHMARK_PROTOCOL.md`
