# Intel Arc A770 GPU Roadmap

## Purpose

This document defines the Intel Arc A770 validation lane for BitNet-rs. The A770 is a discrete Intel GPU lane, not a Lunar Lake NPU lane.

Primary target:

```text
intel-arc-a770-opencl
```

Secondary reference target:

```text
intel-arc-a770-openvino-gpu
```

The first useful milestone is native OpenCL kernel smoke with CPU parity and a receipt proving the selected A770 backend with `fallback_used=false`.

BitNet product claims for real question-answer usage are governed by the
stricter claim boundary in:

```text
docs/specs/a770-bitnet-claim-boundary.md
```

The PR-by-PR implementation plan is:

```text
plans/a770-bitnet-claim-boundary-implementation.md
```

## Hardware Profile

The expected target is Intel Arc A770 16GB:

| Property | Expected value |
|---|---|
| Architecture | Alchemist / Xe HPG |
| Xe-cores | 32 |
| XMX engines | 512 |
| Vector engines | 512 |
| INT8 peak | 262 TOPS |
| PCI device ID | 0x56A0 |
| Memory | 16GB GDDR6 |
| Memory bus | 256-bit |
| Bandwidth | 560 GB/s |
| TBP | 225W |
| Interface | PCIe 4.0 x16 |
| Runtime support | oneAPI, Level Zero, OpenCL 3.0, OpenVINO GPU, Vulkan |

There is also an 8GB A770 variant. Validation artifacts must record the exact board, VRAM, and PCI device identity instead of assuming 16GB.

Resizable BAR matters for performance. A770 can function without ReBAR, but performance claims should require the machine profile to record UEFI boot, CSM/Legacy status, Above 4G Decoding, ReBAR status, and PCIe link width/generation.

## Claim Boundary

Do not claim A770 execution from detection alone.

| Evidence | Allowed claim |
|---|---|
| `clinfo` sees Arc A770 | Runtime detected |
| OpenCL program compiles for A770 | Compile smoke |
| Tiny OpenCL kernel executes on A770 | Kernel smoke tested |
| CPU/OpenCL parity passes | Parity tested |
| Receipt records selected A770 backend and no fallback | Receipt backed |
| Benchmark beats CPU baseline with artifact | Diagnostic performance evidence |

OpenVINO GPU graph smoke is reference-runtime evidence. It is not native BitNet OpenCL kernel proof.

CPU fallback cannot count as A770 execution.

For BitNet b1.58, performance claims also require the stricter claim-boundary
gates for model contract, prompt quality, route identity, fallback status,
resources, and same-device history. The first product claim is trusted partial
A770 acceleration, not full A770 residency. Selected attention, resident KV,
attention scores, softmax, attention value mix, full support-op residency, and
full device residency require separate promotion receipts.

## Backend Labels

Use explicit labels:

```text
requested_backend = "intel-arc-a770"
selected_backend = "intel-arc-a770-opencl"
runtime_api = "opencl"
pci_device_id = "0x56A0"
memory_kind = "dedicated-vram"
```

For OpenVINO GPU reference runs:

```text
requested_backend = "intel-arc-a770-openvino-gpu"
selected_backend = "openvino-gpu"
openvino_device = "GPU.X"
```

Do not use ambiguous labels such as `intel`, `gpu`, `oneapi`, or plain `GPU`.

## Runtime Paths

### Native OpenCL Path

This is the main BitNet path for A770.

Milestones:

1. OpenCL and Level Zero detection.
2. Strict A770 selected-device identity.
3. Tiny OpenCL kernel smoke.
4. `matmul_i2s` or equivalent minimal compute smoke with CPU parity.
5. QK256-adjacent OpenCL kernel/subgraph parity.
6. Receipt-backed benchmark baseline.

This path is the likely Intel performance lane for packed BitNet/QK256-style kernels.

### OpenVINO GPU Reference Path

OpenVINO GPU is a graph/runtime comparison lane.

Milestones:

1. Probe OpenVINO `available_devices`.
2. Resolve the exact `GPU.X` device for A770.
3. Compile a fixed-shape tiny graph to that `GPU.X`.
4. Compare output against CPU expected output.
5. Record OpenVINO version, full device name, selected GPU index, and fallback status.

If an iGPU is present, the iGPU is typically `GPU.0`, so A770 may be `GPU.1`. Receipts must record the resolved device.

## Probe Shape

Suggested probe result:

```rust
pub struct IntelArcA770Probe {
    pub available: bool,
    pub pci_device_id: Option<String>,
    pub opencl_available: bool,
    pub opencl_platform_name: Option<String>,
    pub opencl_device_name: Option<String>,
    pub opencl_driver_version: Option<String>,
    pub level_zero_available: bool,
    pub xpu_smi_available: bool,
    pub render_node_accessible: Option<bool>,
    pub openvino_gpu_visible: bool,
    pub openvino_gpu_device: Option<String>,
    pub vram_bytes: Option<u64>,
    pub rebar_enabled: Option<bool>,
    pub pcie_link: Option<String>,
    pub failure_reason: Option<String>,
}
```

## Receipt Fields

Minimum native OpenCL receipt:

```json
{
  "requested_backend": "intel-arc-a770",
  "selected_backend": "intel-arc-a770-opencl",
  "fallback_backend": null,
  "fallback_used": false,
  "runtime": {
    "api": "opencl",
    "platform": "Intel",
    "device_name": "Intel(R) Arc(TM) A770 Graphics",
    "driver_version": "...",
    "pci_device_id": "0x56A0",
    "vram_bytes": 17179869184,
    "rebar_enabled": true,
    "pcie_link": "PCIe 4.0 x16"
  },
  "kernels_or_graphs": [
    "matmul_i2s_opencl_smoke"
  ]
}
```

Minimum OpenVINO GPU reference receipt:

```json
{
  "requested_backend": "intel-arc-a770-openvino-gpu",
  "selected_backend": "openvino-gpu",
  "openvino_device": "GPU.1",
  "full_device_name": "Intel(R) Arc(TM) A770 Graphics",
  "fallback_used": false,
  "graph": {
    "name": "tiny_matmul_f16",
    "shape_mode": "static"
  }
}
```

## Validation Bundle

The hardware bundle lives in `docs/hardware/intel-arc-a770-validation.md`.

It must collect:

- OS and kernel/build.
- Motherboard/CPU/chipset.
- A770 exact board and VRAM.
- PCI ID and PCIe link.
- ReBAR status.
- OpenCL platform/device/driver.
- Level Zero visibility.
- OpenVINO `GPU.X` identity.
- Linux render-node permissions.
- Power/utilization tooling where available.

## Work Plan

### A770-001 - Add Backend Lane

Docs/tracking only. Add backend status, workstream items, roadmap, and validation profile.

### A770-002 - Machine Profile

Record A770 hardware, driver, OpenCL, Level Zero, OpenVINO GPU, ReBAR, PCIe, and permissions facts.

### A770-003 - Backend Identity

Preserve requested and selected backend identity.

Suggested identity:

```rust
RequestedBackend::IntelArcA770 { opencl_index: Option<usize> }
SelectedBackend::OpenClIntelArc {
    opencl_index: usize,
    device_name: String,
    pci_device_id: Option<String>,
}
```

### A770-004 - Runtime Probe

Detect OpenCL, Level Zero, render-node access, OpenVINO GPU visibility, VRAM, ReBAR, and PCI identity.

### A770-005 - OpenCL Kernel Smoke

Compile and run a tiny OpenCL kernel on A770, then compare output against CPU.

### A770-006 - BitNet OpenCL Parity

Run `matmul_i2s` or another minimal BitNet kernel/subgraph on A770 through OpenCL and compare against CPU.

### A770-007 - Receipts

Record runtime identity, selected device, fallback status, driver, PCI ID, VRAM, ReBAR, and kernel IDs.

### A770-008 - Benchmark Baseline

Compare CPU scalar/AVX2 against A770 OpenCL for the validated kernel/subgraph.

### A770-009 - OpenVINO GPU Smoke

Run a tiny fixed-shape OpenVINO graph on the resolved A770 `GPU.X` device.

### A770-010 - OpenVINO llama.cpp GGUF Reference

Evaluate one OpenVINO-validated GGUF through llama.cpp/OpenVINO on A770. This is a reference lane, not proof of native bitnet-rs GPU inference.

## Practical Direction

A770 is the Intel lane most likely to pay off for custom BitNet kernels. Prioritize:

```text
OpenCL detection
OpenCL kernel smoke
matmul_i2s CPU parity
QK256-adjacent OpenCL subgraph parity
benchmark receipts
```

Keep OpenVINO GPU as a comparison and reference path.

## Related Roadmaps

- `docs/specs/intel-lunar-lake-258v-platform-roadmap.md`
- `docs/specs/intel-lunar-lake-gpu-roadmap.md`
- `docs/specs/intel-lunar-lake-npu-roadmap.md`

The A770 lane is the discrete GPU performance lane. It should not own Arc 140V shared-memory laptop comparisons or OpenVINO NPU validation.
