# NVIDIA RTX 5070 Ti Roadmap

## Purpose

This document defines the NVIDIA RTX 5070 Ti validation lane for BitNet-rs. The 5070 Ti lane is CUDA-first, with wgpu/Vulkan/D3D12 as a cross-platform reference lane.

Primary labels:

```text
nvidia-rtx-5070-ti-cuda
nvidia-rtx-5070-ti-wgpu
```

The first useful milestone is CUDA kernel smoke with a receipt proving `selected_backend=nvidia-rtx-5070-ti-cuda` and `fallback_used=false`.

## Current Implementation Boundary

The CUDA lane is scaffolded infrastructure, not a working CUDA inference path.
The existing CUDA provider can create a cudarc context, compile CUDA source
with NVRTC, and load kernel-provider-level functions such as I2S matmul and
quantization helpers. That is not proof that the transformer forward path
routes BitNet inference through CUDA.

QK256 CUDA is explicitly scaffold-only until the packed kernel path is
implemented and wired. Any `launch_qk256_gemv` path that returns an explicit
"scaffold only" or "not yet compiled" error must block full 1-bit BitNet CUDA
claims.

The first CUDA lane can progress through device probe, tiny kernel smoke, and
I2S/matmul parity before full BitNet inference. Full inference requires the
later `CUDA-BITNET-*` work items.

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

## Windows Proof Bench

The primary Windows CUDA proof bench combines:

| Component | Role |
|---|---|
| AMD Ryzen 9 9950X3D | CPU reference path for scalar, AVX2, AVX-512, and strict model proof. |
| NVIDIA GeForce RTX 5070 Ti 16GB | CUDA target for probe, smoke, parity, BitNet kernels, and benchmark receipts. |
| Windows | Primary validation OS for this box. |
| WGPU / D3D12 / Vulkan | Optional comparison lane only; never CUDA proof. |

Machine profile:

```text
docs/hardware/windows-9950x3d-rtx5070ti-validation.md
```

Receipt machine ID:

```text
windows-9950x3d-rtx5070ti
```

## Claim Boundary

- PCI detection is not CUDA runtime proof.
- CUDA runtime visibility is not CUDA kernel execution.
- CUDA kernel smoke is not CPU/CUDA parity.
- CPU/CUDA parity is not full inference.
- CUDA proof is not wgpu/Vulkan proof.
- wgpu/Vulkan smoke is not CUDA kernel proof.
- CPU fallback cannot count as CUDA execution.
- Dense FP16/BF16 regular-LLM CUDA kernels are not BitNet packed-kernel proof.
- I2S CUDA parity is not QK256 CUDA inference.
- Full BitNet CUDA inference cannot be claimed while QK256 CUDA is scaffold-only.

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
  "machine_id": "windows-9950x3d-rtx5070ti",
  "requested_backend": "nvidia-rtx-5070-ti-cuda",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "reference_backend": "amd-9950x3d-cpu-avx512",
  "fallback_backend": null,
  "fallback_used": false,
  "cuda": {
    "device_index": 0,
    "device_name": "NVIDIA GeForce RTX 5070 Ti",
    "compute_capability": "12.0",
    "driver_version": "...",
    "cuda_runtime_version": "...",
    "cuda_toolkit_version": "...",
    "nvrtc_version": "...",
    "vram_bytes": 17179869184
  }
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

Minimum CUDA kernel stats:

```json
{
  "kernel_stats": [
    {
      "kernel_id": "cuda_tiny_vector_add",
      "invocations": 1,
      "fallback_invocations": 0,
      "host_to_device_bytes": 4096,
      "device_to_host_bytes": 4096,
      "kernel_launches": 1,
      "kernel_time_ms": null
    }
  ]
}
```

Strict BitNet CUDA proof receipts must also preserve:

```json
{
  "model": {
    "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
    "file": "ggml-model-i2_s.gguf",
    "loader_mode": "strict",
    "fallback_loader_used": false
  },
  "bitnet": {
    "quantization": "W1.58A8",
    "kernel_family": "i2_s|qk256",
    "layout": "packed",
    "weights_uploaded_once": true,
    "per_token_weight_upload": false
  },
  "execution_coverage": {
    "linear_layers_total": 0,
    "linear_layers_on_cuda": 0,
    "linear_layers_cpu_fallback": 0,
    "unsupported_ops": []
  },
  "speedup_claim": false
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

## BitNet CUDA Implementation Wave

Start these only after `RTX5070TI-003` through `RTX5070TI-008` make CUDA
identity, probe, smoke, parity, receipts, and benchmark baselines durable.

### CUDA-BITNET-001 - Persistent Context and Weight Handles

Create a CUDA BitNet context with persistent device, stream, weight handles,
activation workspace, and stats. Weights must be uploadable once, and receipts
must record `per_token_weight_upload=false`.

### CUDA-BITNET-002 - Reusable CUDA I2S Linear Primitive

Turn existing I2S CUDA matmul into a reusable backend primitive for real layer
shapes. It must handle tails and padding, match CPU reference, and record
kernel IDs and invocation stats.

### CUDA-BITNET-003 - CUDA QK256 Fused Dequant GEMV

Replace the scaffold-only QK256 launch path with a real fused packed-weight
dequant plus GEMV kernel. It must support official BitNet GGUF shapes and pass
CPU QK256 scalar parity before any full inference claim.

### CUDA-BITNET-004 - Prepack and Upload BitNet Weights Once

At strict GGUF load time, validate BitNet layout, pack or normalize weights for
CUDA, upload them once, and store per-layer CUDA weight handles. Decode must not
repack or upload weights per token.

### CUDA-BITNET-005 - Route BitNetLinear Through CUDA

Wire the actual transformer forward path so `BitNetLinear` dispatches through
the selected CUDA backend. Strict CUDA mode must fail on unsupported CPU
fallback and coverage counters must record total, CUDA-routed, and fallback
linear layers.

### CUDA-BITNET-006 - One-Token Strict BitNet CUDA Proof

Run the official GGUF in strict mode for one greedy token. The proof must record
CUDA kernel invocation count greater than zero, CPU fallback count zero,
CPU/CUDA greedy or top-1 parity, and `speedup_claim=false`.

### CUDA-BITNET-007 - Short Decode BitNet CUDA Proof

Extend the one-token proof to a short greedy decode and record prefill,
first-token, steady-state decode timing, CUDA memory high-water mark, kernel
invocations, and CPU fallback operations.

### CUDA-BITNET-008 - BitNet CUDA Benchmark Baseline

Benchmark only after correctness. Compare 9950X3D CPU scalar, AVX2, AVX-512,
and RTX 5070 Ti CUDA on the same model, tokenizer, prompt profile, strict
loader mode, and fallback-free receipt.

## Dense CUDA Reference Lane

Regular LLM CUDA support is useful, but it is separate from BitNet packed-kernel
proof. A future dense lane may share device selection, probes, context lifetime,
allocator, workspace, stats, parity harness, and benchmark protocol, but FP16,
BF16, or INT8 dense kernels must be labeled as `dense_regular_llm`, not BitNet
packed inference.

## GitHub Issue Tracking

After this roadmap lands, create or link GitHub issues with titles that begin
with the work item ID, for example
`RTX5070TI-003: Preserve RTX 5070 Ti CUDA backend identity`. Each issue should link back to
`docs/tracking/bitnet-alignment/workstream-ledger.yaml`, copy the acceptance
criteria for that item, and record the PR that advances it. Do not combine
distinct proof stages into one implementation PR.

## Do Not

- Do not make CUDA a generic GPU claim.
- Do not count CPU fallback as CUDA execution.
- Do not count wgpu/Vulkan smoke as CUDA kernel proof.
- Do not omit compute capability from receipts.
- Do not leave QK256 CUDA scaffold-only while claiming full 1-bit inference.
- Do not upload weights every token and call it real inference.
- Do not use dense regular-LLM CUDA kernels as BitNet packed-kernel proof.
- Do not make benchmark claims without driver, CUDA, VRAM, power, and temperature context.

## Related Contract Docs

- `docs/hardware/HARDWARE_MATRIX.md`
- `docs/hardware/PROOF_STAGES.md`
- `docs/hardware/LANE_OWNERSHIP.md`
- `docs/hardware/BENCHMARK_PROTOCOL.md`
- `docs/hardware/windows-9950x3d-rtx5070ti-validation.md`
- `ci/hardware/_templates/cuda-probe-receipt.json`
- `ci/hardware/_templates/cuda-smoke-receipt.json`
- `ci/hardware/_templates/cuda-parity-receipt.json`
- `ci/hardware/_templates/strict-bitnet-cuda-proof.json`
