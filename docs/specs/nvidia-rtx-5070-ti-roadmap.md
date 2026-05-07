# NVIDIA RTX 5070 Ti Roadmap

## Purpose

This document defines the NVIDIA RTX 5070 Ti validation lane for BitNet-rs. The 5070 Ti lane is CUDA-first, with wgpu/Vulkan/D3D12 as a cross-platform reference lane.

Primary labels:

```text
nvidia-rtx-5070-ti-cuda
nvidia-rtx-5070-ti-wgpu
```

The first useful milestone was CUDA kernel smoke with a receipt proving
`selected_backend=nvidia-rtx-5070-ti-cuda` and `fallback_used=false`. The
BitNet proof lane has since progressed through strict selected-device identity,
CUDA/NVML probe, smoke, CPU/CUDA parity, counters, benchmarks, persistent CUDA
BitNet context, upload-once weights, QK256 CUDA GEMV, transformer-path routing,
one-token proof, short decode proof, and a same-model benchmark baseline.

## Current Proof State

The NVIDIA campaign now records `RTX5070TI-003` through `RTX5070TI-008` and
`CUDA-BITNET-001` through `CUDA-BITNET-009` as merged. The strict BitNet CUDA
proof receipts record the selected RTX 5070 Ti CUDA backend, official GGUF,
explicit tokenizer, W1.58A8 packed layout, QK256 CUDA kernel invocations,
upload-once weight residency, zero BitNet linear CPU fallback, and measured
timing.

QK256 CUDA is no longer a roadmap blocker for the completed proof lane. The
merged QK256 CUDA path is proof-backed by parity tests and strict routed
receipts. Any future regression that reintroduces a non-compiled QK256 launch
path, CPU fallback, per-token weight upload, or missing kernel counters must
downgrade the claim until the receipts are refreshed.

`CUDA-DENSE-001` remains proposed as an optional dense regular-LLM reference
lane. It is separate from the completed BitNet packed I2S/QK256 proof and must
not be used to satisfy BitNet packed-kernel acceptance.

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
- I2S CUDA parity alone is not QK256 CUDA inference.
- Full BitNet CUDA inference claims require strict receipts with selected RTX
  5070 Ti CUDA backend identity, QK256 CUDA invocation counts, zero CPU
  fallback, and upload-once weight residency.
- Speedup claims require same-model, same-tokenizer, fallback-free benchmark
  receipts. The current strict benchmark baseline keeps `speedup_claim=false`.

## Proof Ledger

RTX 5070 Ti CUDA BitNet proof state:

| Field | Proof value |
|---|---|
| Strict selected backend | `nvidia-rtx-5070-ti-cuda` |
| Runtime API | `cuda` |
| Machine ID | `windows-9950x3d-rtx5070ti` |
| Model | `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` |
| Tokenizer | explicit strict `llama3` |
| Quantization | `W1.58A8` |
| Packed layout | `gguf_packed_i2_s` |
| CUDA kernel family | `qk256` |
| Kernel ID | `qk256_gemv_cuda` |
| Weights uploaded once | `true` |
| Per-token weight upload | `false` |
| One-token CUDA invocations | `210` |
| Short-decode CUDA invocations | `1680` |
| BitNet linear CPU fallback | `0` |
| Fallback used | `false` |
| Speedup claim | `false` |

Committed proof receipts:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-smoke.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-parity.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-benchmark.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-proof.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-short-decode.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json
```

Allowed claims:

- The RTX 5070 Ti CUDA selected backend has receipt-backed probe, smoke,
  parity, counters, and benchmark evidence.
- Strict BitNet CUDA one-token and short-decode proofs route BitNet linear work
  through QK256 CUDA kernels with zero BitNet linear CPU fallback.
- The current benchmark is a fallback-free baseline and does not make a speedup
  claim.

Not allowed:

- Do not claim dense regular-LLM CUDA as BitNet packed inference.
- Do not claim WGPU, Vulkan, D3D12, or generic `cuda` as RTX 5070 Ti CUDA proof.
- Do not claim speedup unless a later same-model fallback-free benchmark receipt
  explicitly upgrades `speedup_claim`.

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

Merged in #3679. Preserved requested and selected backend identity for RTX 5070
Ti CUDA.

### RTX5070TI-004 - CUDA/NVML Probe

Merged in #3691. Reported CUDA runtime visibility, NVML data, driver, CUDA
version, compute capability, VRAM, power, and temperature.

### RTX5070TI-005 - CUDA Kernel Smoke

Merged in #3723. Compiled and ran a tiny CUDA kernel on RTX 5070 Ti.

### RTX5070TI-006 - CPU/CUDA Parity

Merged in #3749. Compared one CUDA kernel/subgraph output against CPU.

### RTX5070TI-007 - Receipts

Merged in #3756. Recorded CUDA runtime, driver, compute capability, VRAM, power,
fallback status, and kernel IDs.

### RTX5070TI-008 - Benchmark Baseline

Merged in #3770. Compared CPU against RTX 5070 Ti CUDA for the validated
kernel/subgraph.

### RTX5070TI-009 - wgpu/Vulkan Smoke

Run a tiny wgpu/Vulkan/D3D12 shader smoke as a cross-platform reference path.

## BitNet CUDA Implementation Wave

Start these only after `RTX5070TI-003` through `RTX5070TI-008` make CUDA
identity, probe, smoke, parity, receipts, and benchmark baselines durable.

### CUDA-BITNET-001 - Persistent Context and Weight Handles

Merged in #3776. Created a CUDA BitNet context with persistent device, stream,
weight handles, activation workspace, and stats.

### CUDA-BITNET-002 - Reusable CUDA I2S Linear Primitive

Merged in #3782. Turned I2S CUDA matmul into a reusable backend primitive for
real layer shapes.

### CUDA-BITNET-003 - CUDA QK256 Fused Dequant GEMV

Merged in #3786. Replaced the former QK256 placeholder launch path with a real
fused packed-weight dequant plus GEMV CUDA kernel.

### CUDA-BITNET-004 - Prepack and Upload BitNet Weights Once

Merged in #3790. Added strict GGUF CUDA weight handling so BitNet weights can be
packed or normalized for CUDA and uploaded once.

### CUDA-BITNET-005 - Route BitNetLinear Through CUDA

Merged in #3792. Wired the actual transformer forward path so `BitNetLinear`
dispatches through the selected CUDA backend with coverage counters.

### CUDA-BITNET-006 - One-Token Strict BitNet CUDA Proof

Merged in #3801. Added the official-GGUF one-token strict BitNet CUDA proof with
CUDA kernel invocations, zero CPU fallback, CPU/CUDA agreement, and
`speedup_claim=false`.

### CUDA-BITNET-007 - Short Decode BitNet CUDA Proof

Merged in #3806. Extended the one-token proof to a short greedy decode with
timing, CUDA memory high-water mark, kernel invocations, and CPU fallback
operations.

### CUDA-BITNET-008 - BitNet CUDA Benchmark Baseline

Merged in #3823. Added a same-model strict BitNet CUDA benchmark baseline with
CPU reference comparison and `speedup_claim=false`.

### CUDA-BITNET-009 - Routed Upload-Once Strict Proof

Merged in #3837. Refreshed the strict one-token, short-decode, and benchmark
receipts so routed QK256 CUDA inference records `weights_uploaded_once=true`,
`per_token_weight_upload=false`, `qk256_gemv_cuda` invocations greater than zero,
and zero BitNet linear CPU fallback.

## Dense CUDA Reference Lane

Regular LLM CUDA support is useful, but it is separate from BitNet packed-kernel
proof. `CUDA-DENSE-001` remains optional proposed work. It may share device
selection, probes, context lifetime, allocator, workspace, stats, parity
harness, and benchmark protocol, but FP16, BF16, or INT8 dense kernels must be
labeled as `dense_regular_llm`, not BitNet packed inference.

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
- Do not regress QK256 CUDA back to a non-compiled or fallback path while
  preserving full BitNet CUDA claims.
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
