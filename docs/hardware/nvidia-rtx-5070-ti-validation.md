# NVIDIA RTX 5070 Ti Validation Profile

## Purpose

This file defines the hardware data bundle for the NVIDIA RTX 5070 Ti validation lane. It is a CUDA-first profile, with wgpu/Vulkan/D3D12 as a cross-platform reference lane.

Roadmap:

```text
docs/specs/nvidia-rtx-5070-ti-roadmap.md
```

## Hardware Baseline

- GPU: NVIDIA GeForce RTX 5070 Ti.
- Architecture: Blackwell.
- CUDA cores: 8,960.
- Tensor cores: 5th-generation.
- AI performance: 1,406 AI TOPS.
- RT cores: 4th-generation.
- Memory: 16GB GDDR7.
- Memory bus: 256-bit.
- CUDA compute capability: 12.0.
- Graphics power: 300W.
- Recommended system power: 750W.

Record actual board, driver, CUDA runtime, power, and thermal values in receipts.

## Claim Boundary

- PCI detection is not CUDA runtime proof.
- CUDA runtime visibility is not CUDA kernel execution.
- CUDA smoke is not CPU/CUDA parity.
- CPU fallback cannot count as CUDA execution.
- wgpu/Vulkan/D3D12 smoke cannot count as CUDA kernel proof.
- Performance claims require driver, CUDA, VRAM, power, and thermal context.

## Linux Bundle

```bash
set -eux

echo "=== OS ==="
uname -a
cat /etc/os-release || true

echo "=== PCI / NVIDIA ==="
lspci -nn | grep -Ei 'vga|3d|display|nvidia|5070' || true
lspci -vv | grep -A50 -Ei 'VGA.*NVIDIA|3D.*NVIDIA|Display.*NVIDIA' || true

echo "=== NVIDIA SMI ==="
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,power.limit,power.draw,temperature.gpu,compute_cap --format=csv || true

echo "=== CUDA compiler/runtime ==="
which nvcc || true
nvcc --version || true

echo "=== Vulkan / WGPU reference ==="
vulkaninfo --summary || true

echo "=== Rust / toolchain ==="
rustc --version
cargo --version
```

## Windows PowerShell Bundle

```powershell
$ErrorActionPreference = "Continue"

Write-Host "=== GPU devices ==="
Get-PnpDevice | Where-Object {
  $_.FriendlyName -match "NVIDIA|5070|GeForce"
} | Format-List *

Write-Host "=== NVIDIA SMI ==="
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,power.limit,power.draw,temperature.gpu,compute_cap --format=csv

Write-Host "=== CUDA ==="
where nvcc
nvcc --version
```

## First CUDA Receipt

The first useful receipt is a CUDA smoke proof:

```json
{
  "hardware": "nvidia-rtx-5070-ti",
  "requested_backend": "nvidia-rtx-5070-ti",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "compute_capability": "12.0",
  "vram_bytes": 17179869184,
  "driver_version": "...",
  "cuda_version": "...",
  "fallback_used": false,
  "status": "kernel_smoke_tested"
}
```

## Optional wgpu Receipt

```json
{
  "hardware": "nvidia-rtx-5070-ti",
  "requested_backend": "nvidia-rtx-5070-ti-wgpu",
  "selected_backend": "nvidia-rtx-5070-ti-wgpu",
  "runtime_api": "wgpu",
  "backend_api": "vulkan",
  "adapter_name": "NVIDIA GeForce RTX 5070 Ti",
  "fallback_used": false,
  "status": "kernel_smoke_tested"
}
```

This is cross-platform shader proof only, not CUDA proof.

## Benchmark Notes

Benchmarks must record:

- OS and kernel/build.
- NVIDIA driver version.
- CUDA version.
- Compute capability.
- VRAM.
- Power limit and draw when available.
- Temperature when available.
- Selected backend.
- Fallback status.
- Cold and warm timing.

Do not make benchmark claims without the machine context above.
