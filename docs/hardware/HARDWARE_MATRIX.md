# Hardware Validation Matrix

## Purpose

Hardware validation in BitNet-rs is lane-based. Each lane must preserve:

```text
hardware identity
runtime identity
selected backend identity
proof stage
receipt/artifact
claim allowed
```

Do not merge CPU, Metal, MPSGraph, CUDA, WGPU, Intel Arc GPU, OpenVINO GPU, OpenVINO NPU, or AMD ROCm claims just because the hardware, vendor name, or accelerator class overlaps.

## Lane Matrix

| Machine | Proof label | Primary role | Must not claim |
|---|---|---|---|
| i5-8250U | `intel-i5-8250u-cpu-avx2` | SLM CPU lead plus legacy/low-power BitNet comparison baseline | GPU/NPU acceleration or blocking BitNet CPU leadership |
| Core Ultra 7 258V CPU | `intel-258v-cpu-avx2` | BitNet CPU lead for strict real-GGUF validation, scalar/AVX2 answer parity, phase receipts, and same-machine CPU reference artifacts | GPU/NPU acceleration |
| UHD 620 | `intel-uhd-620-openvino-gpu` | Optional/deferred OpenVINO GPU smoke on the 8250U box | CPU proof or primary GPU performance |
| Ryzen 7 5700X | `amd-5700x-cpu-avx2` | Mainstream AM4 / DDR4 desktop support validator; primary support around A770 work | AVX-512, accelerator proof, or BitNet CPU ownership |
| Ryzen 9 9950X3D | `amd-9950x3d-cpu-avx512` | Modern AM5 / DDR5 / AVX-512 / cache-sensitive support validator; primary support around RTX 5070 Ti work | GPU/NPU acceleration or BitNet CPU ownership |
| Arc A770 16GB | `intel-arc-a770-opencl` | Discrete GPU OpenCL kernel/perf lane | NPU support, generic GPU proof |
| Arc A770 OpenVINO | `intel-arc-a770-openvino-gpu` | OpenVINO GPU reference lane | Native BitNet kernel proof |
| Arc 140V | `intel-arc-140v-opencl` | Lunar Lake shared-memory iGPU comparison lane | A770-equivalent performance |
| Arc 140V OpenVINO | `intel-arc-140v-openvino-gpu` | OpenVINO `GPU.0` reference lane | Native OpenCL kernel proof |
| 258V NPU | `intel-npu-openvino` / `intel_258v_npu_openvino` | OpenVINO static-shape NPU lane | Full decode/QK256 acceleration |
| M4 Mac mini | `apple-m4-metal` | Apple Silicon Metal GPU lane | CPU fallback or MPSGraph proof |
| M4 Mac mini MPSGraph | `apple-m4-mpsgraph` | Apple graph/reference lane, possible Neural Engine routing | Native Metal kernel proof |
| M4 Mac mini CPU | `apple-m4-cpu-neon` | ARM64 CPU/NEON fallback and parity | AVX2/AVX-512 behavior |
| RTX 5070 Ti | `nvidia-rtx-5070-ti-cuda` | Modern NVIDIA CUDA kernel/perf lane | Generic GPU or wgpu proof |
| RTX 5070 Ti WGPU/Vulkan | `nvidia-rtx-5070-ti-wgpu` | Cross-platform GPU reference lane | CUDA kernel proof |
| Selected AMD Radeon / Radeon PRO / Instinct | `amd-rocm` (registered only; selected backend must become concrete, for example `amd-radeon-rx-7900-xtx-rocm`) | AMD GPU HIP/ROCm lane scaffold for selected devices | Generic AMD GPU support, CUDA/OpenCL/WGPU/OpenVINO/CPU proof, source-text-as-compile proof, speed, full residency, or server readiness |

## Machine Summary

| Machine | Primary lane | Secondary lane | Notes |
|---|---|---|---|
| i5-8250U | `intel-i5-8250u-cpu-avx2` | `intel-uhd-620-openvino-gpu` | SLM CPU lead; legacy/low-power BitNet comparison only |
| Ryzen 7 5700X | `amd-5700x-cpu-avx2` | `amd-5700x-cpu-scalar` | Support validator; primary desktop support around A770 work |
| Ryzen 9 9950X3D | `amd-9950x3d-cpu-avx512` | `amd-9950x3d-cpu-avx2`, `amd-9950x3d-cpu-scalar` | Support validator; primary desktop support around RTX 5070 Ti work |
| A770 16GB | `intel-arc-a770-opencl` | `intel-arc-a770-openvino-gpu` | Intel discrete GPU kernel lane |
| 258V | `intel-258v-cpu-avx2`, `intel-arc-140v-opencl`, `intel-npu-openvino` | `intel-arc-140v-openvino-gpu` | Lunar Lake tri-device box; BitNet CPU lead and same-machine CPU/NPU/Arc comparison plate |
| M4 Mac mini | `apple-m4-metal` | `apple-m4-mpsgraph`, `apple-m4-cpu-neon` | Apple Silicon Metal lane |
| RTX 5070 Ti | `nvidia-rtx-5070-ti-cuda` | `nvidia-rtx-5070-ti-wgpu` | Modern NVIDIA CUDA lane |
| AMD ROCm candidate host | `amd-rocm` | selected concrete AMD ROCm route TBD | Registered scaffold only; first real product target depends on available supported AMD GPU and recorded GFX/HIP/ROCm identity |

## Required Identity Fields

Every receipt or hardware artifact should preserve separate identity fields.
The example path below is a required shape, not a claim that the artifact already
exists; A770 OpenCL claims require a committed receipt linked from the route and
capability matrices.

```json
{
  "requested_backend": "intel-arc-a770",
  "selected_backend": "intel-arc-a770-opencl",
  "runtime_api": "opencl",
  "resolved_device": {
    "name": "Intel(R) Arc(TM) A770 Graphics",
    "pci_device_id": "0x56A0"
  },
  "fallback_used": false,
  "artifact_path": "ci/hardware/amd-5700x-intel-a770/<date>/opencl-smoke.json"
}
```

For OpenVINO GPU:

```json
{
  "requested_backend": "intel-arc-a770-openvino-gpu",
  "selected_backend": "openvino-gpu",
  "runtime_api": "openvino",
  "openvino_device": "GPU.1",
  "full_device_name": "Intel(R) Arc(TM) A770 Graphics",
  "fallback_used": false
}
```

For OpenVINO NPU:

```json
{
  "requested_backend": "intel-npu",
  "selected_backend": "intel-npu-openvino",
  "runtime_api": "openvino",
  "runtime_device": "NPU",
  "shape_mode": "static",
  "fallback_used": false
}
```

For Apple Metal:

```json
{
  "requested_backend": "apple-m4",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "resolved_device": {
    "chip": "Apple M4",
    "gpu_cores": 10
  },
  "fallback_used": false
}
```

For NVIDIA CUDA:

```json
{
  "requested_backend": "nvidia-rtx-5070-ti",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "resolved_device": {
    "name": "NVIDIA GeForce RTX 5070 Ti",
    "compute_capability": "12.0"
  },
  "fallback_used": false
}
```

For AMD desktop CPU:

```json
{
  "requested_backend": "cpu",
  "selected_backend": "amd-9950x3d-cpu-avx512",
  "runtime_api": "cpu",
  "resolved_device": {
    "vendor": "AMD",
    "model": "Ryzen 9 9950X3D",
    "avx512_detected": true
  },
  "fallback_used": false
}
```

For AMD ROCm:

```json
{
  "requested_backend": "amd-rocm",
  "selected_backend": "amd-radeon-rx-7900-xtx-rocm",
  "runtime_api": "hip",
  "runtime_stack": "rocm",
  "rocm_version": "7.2.3",
  "hip_version": "7.2.53211",
  "gfx_target": "gfx1100",
  "resolved_device": {
    "name": "AMD Radeon RX 7900 XTX",
    "arch": "RDNA3",
    "pci_bus_id": "..."
  },
  "fallback_used": false
}
```

The example above is a required receipt shape, not a current support claim.
`amd-rocm` may be requested by users, but proof must resolve to a concrete
selected backend and must preserve ROCm/HIP version, GFX target, device
identity, proof stage, fallback state, and applicable not-claims.

Never use these as proof labels:

```text
intel
gpu
npu
oneapi
openvino
metal
mpsgraph
cuda
wgpu
amd
gpu
rocm
hip
radeon
```

## First PR Queue

| Lane | First item | Purpose |
|---|---|---|
| 8250U CPU | `KBL8250U-001` | Docs and machine profile only |
| 5700X CPU | `AMD5700X-001` | Docs and machine profile only |
| 9950X3D CPU | `AMD9950X3D-001` | Docs and machine profile only |
| A770 | `A770-001` | Docs and backend status only |
| Arc 140V | `ARC140V-001` | Integrated GPU docs and status only |
| 258V NPU | `NPU-001` | OpenVINO NPU docs and status only |
| 258V platform | `LNL258V-001` | Tri-device platform profile only |
| M4 Mac mini | `M4-001` | Apple Silicon docs and status only |
| RTX 5070 Ti | `RTX5070TI-001` | NVIDIA CUDA docs and status only |
| AMD ROCm | `ROCM-DOCS-000` | Docs and backend status only; no runtime, compile, execution, model, speed, residency, or server claim |

Implementation sequence for any lane:

```text
identity
probe
smoke
parity
receipts
benchmark
inference contribution
```

Do not jump from docs or detection directly to benchmark claims.

## Related Contract Docs

- `docs/hardware/PROOF_STAGES.md`
- `docs/hardware/LANE_OWNERSHIP.md`
- `docs/hardware/BENCHMARK_PROTOCOL.md`
- `docs/hardware/machine-profile.schema.yaml`
- `ci/hardware/README.md`

BitNet-specific proof also requires:

- `docs/specs/a770-bitnet-claim-boundary.md` for A770 trusted BitNet product claims.
- `plans/a770-bitnet-claim-boundary-implementation.md` for the PR-by-PR A770 implementation sequence.
- `docs/bitnet/BITNET_MODEL_CONTRACT.md`
- `docs/bitnet/BITNET_QUANTIZATION_CONTRACT.md`
- `docs/bitnet/BITNET_KERNEL_MATRIX.md`
- `docs/bitnet/BITNET_RUNTIME_PHASES.md`
- `docs/bitnet/BITNET_REFERENCE_RUNS.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
- `docs/bitnet/BITNET_BENCHMARK_PROTOCOL.md`
- `docs/bitnet/BITNET_PARITY_TOLERANCES.md`

Hardware docs answer:

```text
Which machine/runtime/device ran?
```

BitNet docs answer:

```text
Which model, tokenizer, quantization format, kernel family, execution phase, and reference path did it run?
```
