# Windows 9950X3D RTX 5070 Ti Validation Profile

## Purpose

This file defines the Windows proof bench that combines the AMD Ryzen 9
9950X3D CPU reference path with the NVIDIA GeForce RTX 5070 Ti CUDA target. It
is a machine profile and validation plan, not proof that CUDA inference works.

Machine ID:

```text
windows-9950x3d-rtx5070ti
```

Related lane docs:

```text
docs/hardware/amd-9950x3d-validation.md
docs/hardware/nvidia-rtx-5070-ti-validation.md
docs/specs/nvidia-rtx-5070-ti-roadmap.md
```

## Role

| Component | Role |
|---|---|
| AMD Ryzen 9 9950X3D | CPU reference path for scalar, AVX2, AVX-512, and strict model proof. |
| NVIDIA GeForce RTX 5070 Ti 16GB | CUDA target for probe, smoke, parity, BitNet kernels, and benchmarks. |
| Windows | Primary validation OS for this proof bench. |
| WGPU / D3D12 / Vulkan | Optional comparison lane only; never CUDA proof. |

## Backend Labels

Use narrow labels in requests, selected backend summaries, and receipts:

```text
amd-9950x3d-cpu-scalar
amd-9950x3d-cpu-avx2
amd-9950x3d-cpu-avx512
nvidia-rtx-5070-ti-cuda
nvidia-rtx-5070-ti-wgpu
```

Do not collapse this machine into `gpu`, `cuda`, `nvidia`, `accelerated`, or
`blackwell` labels. Generic `cuda` may remain available, but it is not the
same request as `nvidia-rtx-5070-ti-cuda`.

## Required Machine Facts

| Field | Why it matters |
|---|---|
| Windows version / build | CUDA, driver, and toolchain behavior differs by Windows version. |
| CPU model | Confirms the 9950X3D CPU reference path. |
| CPU features | Confirms AVX2 / AVX-512 proof and fallback references. |
| RAM amount / speed | Affects load, prefill, CPU baseline, and parity context. |
| GPU name | Must resolve to NVIDIA GeForce RTX 5070 Ti. |
| GPU VRAM | Expected 16GB, but receipts must record the runtime value. |
| NVIDIA driver version | Required for CUDA reproducibility. |
| CUDA runtime version | Required for CUDA proof. |
| CUDA toolkit / `nvcc` version | Required for NVRTC and build debugging. |
| Compute capability | Expected 12.0 for RTX 5070 Ti; receipts must record actual. |
| Power limit / thermals | Required before performance claims. |
| WGPU/D3D12/Vulkan visibility | Optional comparison lane; not CUDA proof. |
| Rust toolchain | Needed for reproducible Codex builds. |

## Windows Probe Bundle

Capture this output locally when opening a machine-profile or probe work item.
Commit only small normalized JSON receipts when the work item explicitly calls
for them.

```powershell
$ErrorActionPreference = "Continue"

Write-Host "=== Windows ==="
Get-ComputerInfo | Select-Object OsName, OsVersion, WindowsVersion, OsBuildNumber, CsSystemType

Write-Host "=== CPU ==="
Get-CimInstance Win32_Processor |
  Format-List Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed

Write-Host "=== Memory ==="
Get-CimInstance Win32_PhysicalMemory |
  Format-Table Manufacturer,Capacity,Speed,ConfiguredClockSpeed,PartNumber

Write-Host "=== GPU devices ==="
Get-PnpDevice | Where-Object {
  $_.FriendlyName -match "NVIDIA|5070|GeForce"
} | Format-List *

Write-Host "=== NVIDIA SMI ==="
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,power.limit,power.draw,temperature.gpu,compute_cap --format=csv

Write-Host "=== CUDA toolkit ==="
where nvcc
nvcc --version

Write-Host "=== Rust ==="
rustc --version
cargo --version

Write-Host "=== Optional WGPU / Vulkan / D3D12 ==="
where vulkaninfo
vulkaninfo --summary
```

## Artifact Paths

Use ISO dates under the combined machine ID:

```text
ci/hardware/windows-9950x3d-rtx5070ti/<date>/machine-profile.json
ci/hardware/windows-9950x3d-rtx5070ti/<date>/cuda-probe.json
ci/hardware/windows-9950x3d-rtx5070ti/<date>/nvml-probe.json
ci/hardware/windows-9950x3d-rtx5070ti/<date>/cuda-smoke.json
ci/hardware/windows-9950x3d-rtx5070ti/<date>/cuda-parity.json
ci/hardware/windows-9950x3d-rtx5070ti/<date>/cuda-benchmark.json
ci/hardware/windows-9950x3d-rtx5070ti/<date>/strict-bitnet-cuda-proof.json
```

Do not commit bulky local logs, raw command transcripts, GGUFs, model files, or
binary traces in normal docs or implementation PRs.

Current normalized artifacts:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-smoke.json
```

This RTX5070TI-005 artifact is a tiny CUDA kernel smoke receipt only. It records
`claim=kernel_smoke_tested` and does not claim BitNet inference, parity, or
speedup.

## Claim Boundary

- `nvidia-smi` visibility is not CUDA kernel execution.
- Creating a CUDA context is not CPU/CUDA parity.
- WGPU/D3D12/Vulkan smoke is not CUDA proof.
- Tiny CUDA smoke is not full BitNet inference.
- I2S matmul parity is not QK256 packed-kernel inference.
- Dense FP16/BF16 regular-LLM CUDA kernels are not BitNet packed-kernel proof.
- CPU fallback cannot count as CUDA execution under strict CUDA requests.
- Full BitNet CUDA inference cannot be claimed while QK256 CUDA remains scaffold-only.

## Practical Order

1. Run environment sanity: `nvidia-smi`, `nvcc --version`, `rustc --version`,
   and `cargo --version`.
2. Implement `RTX5070TI-003` to preserve selected-device identity before probe
   or kernel work.
3. Implement `RTX5070TI-004` to emit a normalized CUDA/NVML probe.
4. Implement `RTX5070TI-005` to prove a tiny CUDA kernel runs fallback-free.
5. Implement `RTX5070TI-006` to compare an existing CUDA kernel against the
   9950X3D CPU reference path.
6. Implement receipt counters and benchmark baselines only after parity.
7. Start the `CUDA-BITNET-*` wave only after CUDA receipts are hard to fake.

If CUDA compilation fails during sanity checks, capture driver version, CUDA
toolkit version, cudarc/NVRTC errors, `PATH`, `CUDA_PATH`, and the Rust target
triple before changing CUDA kernels.

## First CUDA Probe Receipt

```json
{
  "machine_id": "windows-9950x3d-rtx5070ti",
  "requested_backend": "nvidia-rtx-5070-ti-cuda",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "reference_backend": "amd-9950x3d-cpu-avx512",
  "fallback_used": false,
  "cuda": {
    "device_name": "NVIDIA GeForce RTX 5070 Ti",
    "compute_capability": "12.0",
    "driver_version": "TBD",
    "cuda_runtime_version": "TBD",
    "cuda_toolkit_version": "TBD",
    "vram_bytes": 17179869184
  },
  "claim": "cuda_runtime_probe_recorded"
}
```

This receipt is runtime identity only. It must not claim kernel execution.

## First Kernel Smoke Receipt

```json
{
  "machine_id": "windows-9950x3d-rtx5070ti",
  "requested_backend": "nvidia-rtx-5070-ti-cuda",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "kernel_id": "cuda_tiny_vector_add",
  "fallback_used": false,
  "result": "pass",
  "claim": "kernel_smoke_tested"
}
```

This receipt proves a tiny CUDA kernel only. It must not claim BitNet inference
or speedup.

## Strict BitNet CUDA Proof Target

Later BitNet CUDA receipts must add model, tokenizer, quantization, kernel
family, execution phase, fallback, and coverage fields:

```json
{
  "claim": "strict_bitnet_cuda_inference",
  "model": "microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf",
  "tokenizer_source": "gguf|sibling-tokenizer-json|explicit",
  "quantization": "W1.58A8",
  "kernel_family": "i2_s|qk256",
  "execution_phase": "end_to_end_inference",
  "requested_backend": "nvidia-rtx-5070-ti-cuda",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "reference_backend": "amd-9950x3d-cpu-avx512",
  "fallback_used": false,
  "cuda_kernel_invocations": 1,
  "cpu_fallback_ops": []
}
```

The first strict BitNet CUDA proof should set `speedup_claim=false`.
