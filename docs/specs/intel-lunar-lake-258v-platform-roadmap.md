# Intel Lunar Lake 258V Platform Roadmap

## Purpose

This document defines the Core Ultra 7 258V Lunar Lake laptop as a tri-device validation platform for BitNet-rs. It does not merge CPU, GPU, and NPU claims.

Proof lanes:

| Subdevice | Primary proof label | Owner |
|---|---|---|
| CPU | `intel-258v-cpu-avx2` / `cpu-avx2` | CPU runtime proof |
| Integrated GPU | `intel-arc-140v-opencl` | Intel Arc GPU validation |
| Integrated GPU reference | `intel-arc-140v-openvino-gpu` | Intel Arc GPU validation |
| NPU | `intel-npu-openvino` / `intel_258v_npu_openvino` | Intel NPU validation |

The platform lane exists to collect machine facts and compare proof artifacts from those lanes on the same laptop. It does not replace the i5-8250U active AVX2 implementation lane.

## Platform Baseline

Expected Core Ultra 7 258V facts:

| Component | Expected value |
|---|---|
| Platform | Lunar Lake |
| CPU | 8 cores / 8 threads |
| CPU topology | 4 P-cores + 4 low-power E-cores |
| CPU power | 17W base / 37W max turbo |
| CPU ISA | AVX2 |
| Memory | 32GB LPDDR5X-8533 max, shared |
| Overall INT8 peak | 115 TOPS |

Integrated GPU baseline:

| Component | Expected value |
|---|---|
| GPU | Intel Arc 140V |
| Architecture | Xe2-LPG |
| Xe-cores | 8 |
| Vector engines | 8 per Xe-core |
| INT8 peak | 64 TOPS |
| PCI device ID | 0x64A0 |
| Runtime support | OpenCL 3.0, Level Zero, OpenVINO GPU, WindowsML, DirectML, ONNX RT, WebGPU, WebNN |

NPU baseline:

| Component | Expected value |
|---|---|
| NPU | Intel AI Boost |
| INT8 peak | 47 TOPS |
| Primary runtime | OpenVINO NPU |
| Initial shape mode | Static-shape graph smoke |
| Runtime support | OpenVINO, WindowsML, DirectML, ONNX RT, WebNN |

## Claim Boundary

- CPU AVX2 correctness is not Arc 140V or NPU execution.
- Arc 140V OpenCL execution is not NPU execution.
- OpenVINO `GPU.0` execution is not native OpenCL kernel proof.
- OpenVINO `NPU` smoke is not full BitNet inference.
- CPU fallback cannot count as GPU or NPU execution.
- GPU fallback cannot count as NPU execution.
- Shared-memory laptop results must not be compared directly to A770 without memory, power, and thermal context.
- WSL does not count as NPU-capable unless OpenVINO sees `NPU` inside WSL.
- 258V CPU validation must not reshape shared CPU implementation unless the ledger item explicitly scopes that work.

## Device Routing

Expected OpenVINO routing:

```text
258V laptop:
  Arc 140V -> likely GPU.0
  Intel AI Boost NPU -> NPU
```

Receipts must record resolved devices:

```json
{
  "cpu_backend": "cpu-avx2",
  "gpu_backend": "intel-arc-140v-opencl",
  "openvino_gpu_device": "GPU.0",
  "npu_backend": "intel-npu-openvino",
  "openvino_npu_device": "NPU"
}
```

## Required Machine Facts

Collect these before moving 258V platform status beyond scaffold:

- Native Windows, native Linux, or WSL context.
- OS version and kernel/build.
- CPU model, core count, thread count, and AVX2 support.
- Memory capacity, speed, and shared-memory pressure context.
- Power mode and thermal profile.
- Arc 140V OpenCL platform/device/driver.
- Level Zero visibility through `sycl-ls` or `ze_info`.
- OpenVINO version and `available_devices`.
- OpenVINO `GPU.0` full device name.
- Intel NPU driver version or OS device evidence.
- OpenVINO `NPU` visibility.
- Static-shape NPU compile and smoke results when available.

## Validation Bundles

Machine commands live in:

```text
docs/hardware/intel-258v-validation.md
```

The first platform receipt is detection-only:

```json
{
  "platform": "core-ultra-7-258v",
  "cpu_backend": "intel-258v-cpu-avx2",
  "gpu_backend": "intel-arc-140v-opencl",
  "npu_backend": "intel-npu-openvino",
  "openvino_available_devices": ["CPU", "GPU", "NPU"],
  "opencl_arc_140v_visible": true,
  "level_zero_visible": true,
  "npu_visible": true,
  "status": "runtime_detected"
}
```

This receipt does not prove BitNet inference, GPU kernel parity, or NPU subgraph parity.

## Buildout Contract

The implementation contract for the immediate Lunar Lake 258V buildout lives in:

```text
docs/specs/intel-lunar-lake-258v-buildout-plan.md
```

Use that document for PR boundaries, probe structures, receipt fields, acceptance gates, and CI commands. This roadmap remains the platform-level claim boundary and lane overview.

## Work Plan

### LNL258V-001 - Add Platform Profile

Docs/tracking only. Add the 258V platform profile, claim boundaries, data bundles, and cross-links to CPU, Arc 140V, and NPU lanes.

### LNL258V-002 - Platform Probe Bundle

Docs/tracking only. Define the same-machine platform probe bundle that ties
Lunar Lake CPU, Arc 140V, and Intel NPU visibility artifacts together without
turning visibility into execution proof.

The bundle documents these future artifact paths:

```text
ci/hardware/intel-258v/YYYY-MM-DD/platform-probe.json
ci/hardware/intel-258v/YYYY-MM-DD/arc-140v-runtime-probe.json
ci/hardware/intel-258v/YYYY-MM-DD/npu-openvino-runtime-probe.json
ci/hardware/intel-258v/YYYY-MM-DD/platform-comparison-index.json
```

Required bundle fields:

- `machine_id`, OS/build, native/WSL context, memory, power, and thermal context.
- CPU model, topology, AVX2 visibility, and AVX-512 non-claim.
- Arc 140V OpenCL, Level Zero, OpenVINO `GPU.0`, and exact identity evidence.
- Intel NPU OS device evidence, OpenVINO `NPU`, driver/compiler/memory properties.
- Requested backend, selected backend, runtime API, proof stage, fallback status, and artifact path for each lane.

Claim boundaries:

- The bundle records detection/runtime visibility only.
- `platform-comparison-index.json` is an index, not an execution receipt.
- Arc 140V visibility is not OpenCL kernel execution.
- OpenVINO `NPU` visibility is not graph execution.
- No BitNet inference, parity, benchmark, or accelerator contribution claim is allowed.

### CPU258V-001 - Add CPU AVX2 Validation Lane

Document 258V CPU as a parallel Lunar Lake validation lane for the same CPU path. The 8250U remains the active AVX2 implementation/proof lane; 258V CPU validates behavior on the Lunar Lake platform and supports same-machine comparison against Arc 140V and NPU artifacts.

### ARC140V-001 - Add Integrated GPU Lane

Document Arc 140V as a separate integrated GPU lane from Intel NPU.

### NPU-003 - Add Intel NPU Runtime Detection

Probe OpenVINO `NPU`, driver hints, and runtime visibility for the 258V NPU path.

### ARC140V-002 - Add Arc 140V Runtime Probe

Probe OpenCL, Level Zero, and OpenVINO `GPU.0` for Arc 140V.

### Future Comparison Work

Only after lane-specific smoke and parity receipts exist, compare:

```text
CPU AVX2 baseline
Arc 140V OpenCL parity and benchmark
OpenVINO GPU.0 reference smoke
OpenVINO NPU static-shape smoke
NPU subgraph parity, if proven
```

## Related Roadmaps

- `docs/specs/intel-lunar-lake-gpu-roadmap.md`
- `docs/specs/intel-lunar-lake-npu-roadmap.md`
- `docs/hardware/intel-258v-validation.md`

Keep implementation work item ownership separate even when the same laptop produces the receipts.
