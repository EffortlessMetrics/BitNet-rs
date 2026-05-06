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


## Runtime Enablement Build-Out Plan

The 258V laptop is the platform validation lane, not the owner of the shared CPU BitNet implementation. The build-out sequence must preserve that separation:

| PR / lane | Purpose | May touch | Must not touch |
|---|---|---|---|
| `LNL258V-RUN-001` | Add visibility-only 258V platform probe and receipt schema. | `bitnet-device-probe`, CLI or `xtask` probe entry point, receipt types, this hardware/spec documentation. | QK256 CPU kernels, transformer hot path, model loader behavior. |
| `NPU-002-lite` | Preserve Intel NPU identity before OpenVINO execution work. | Backend request parsing, device config mapping, strict NPU failure behavior, receipt identity fields. | OpenVINO graph execution, CPU kernels. |
| `ARC140V-002` | Prove Arc 140V runtime identity separately from CPU and NPU. | Arc probe code, OpenCL/Level Zero/OpenVINO GPU visibility receipts, Arc docs. | CPU kernels, NPU runtime. |
| `CPU258V-001` | Add 258V CPU-only validation harness after strict CPU loader/tokenizer work lands. | Receipt emission, validation commands, hardware artifacts, campaign docs. | Shared QK256 dispatch, quantized transformer internals unless stacked on CPU-proof work. |

The dependency order is identity first, strict CPU proof second, 258V CPU validation third:

```text
NPU-002-lite -> ARC140V-002 -> LNL258V-RUN-001
CPU-BITNET-001/002 remain owned by cpu-proof/i5-8250U
CPU258V-001 validates the merged strict CPU path on Lunar Lake
```

### LNL258V-RUN-001 acceptance contract

`LNL258V-RUN-001` should produce one platform-level detection artifact. It records what the laptop can see; it does not run inference and must not claim BitNet correctness or speed.

Required probe shape:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lnl258vPlatformProbe {
    pub platform: String,
    pub os: String,
    pub arch: String,
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub cpu_has_avx2: bool,
    pub cpu_has_avx512: bool,
    pub opencl_arc_140v_visible: bool,
    pub opencl_platform_name: Option<String>,
    pub opencl_device_name: Option<String>,
    pub opencl_driver_version: Option<String>,
    pub pci_device_id: Option<String>,
    pub level_zero_visible: bool,
    pub level_zero_devices: Vec<String>,
    pub openvino_version: Option<String>,
    pub openvino_available_devices: Vec<String>,
    pub openvino_gpu_device: Option<String>,
    pub openvino_gpu_full_name: Option<String>,
    pub openvino_npu_visible: bool,
    pub openvino_npu_full_name: Option<String>,
    pub accel_device_present: bool,
    pub accel_devices: Vec<String>,
    pub intel_vpu_driver_seen: bool,
    pub npu_driver_version: Option<String>,
    pub power_mode: Option<String>,
    pub thermal_profile: Option<String>,
    pub shared_memory_bytes: Option<u64>,
    pub status: String,
    pub failure_reason: Option<String>,
}

pub fn probe_lnl258v_platform() -> Lnl258vPlatformProbe;
```

Minimum JSON fields:

```json
{
  "platform": "core-ultra-7-258v",
  "os": "linux|windows|wsl",
  "arch": "x86_64",
  "cpu_brand": "Intel Core Ultra 7 258V",
  "cpu_cores": 8,
  "cpu_threads": 8,
  "cpu_has_avx2": true,
  "cpu_has_avx512": false,
  "opencl_arc_140v_visible": true,
  "opencl_platform_name": "Intel(R) OpenCL Graphics",
  "opencl_device_name": "Intel(R) Arc(TM) 140V Graphics",
  "opencl_driver_version": "...",
  "pci_device_id": "0x64A0",
  "level_zero_visible": true,
  "level_zero_devices": ["Intel(R) Arc(TM) 140V Graphics"],
  "openvino_version": "...",
  "openvino_available_devices": ["CPU", "GPU.0", "NPU"],
  "openvino_gpu_device": "GPU.0",
  "openvino_gpu_full_name": "Intel(R) Arc(TM) 140V Graphics",
  "openvino_npu_visible": true,
  "openvino_npu_full_name": "...",
  "accel_device_present": true,
  "accel_devices": ["/dev/accel/accel0"],
  "intel_vpu_driver_seen": true,
  "npu_driver_version": "...",
  "status": "runtime_detected"
}
```

### CPU258V-001 acceptance contract

`CPU258V-001` starts only after strict loader/tokenizer authority is available from the CPU-proof lane. It records strict CPU behavior on the 258V and must not take over shared CPU implementation work.

Required receipt shape:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuBitnetValidationReceipt {
    pub machine: String,
    pub requested_backend: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub loader_mode: String,
    pub minimal_loader_fallback_used: bool,
    pub tokenizer_source: String,
    pub mock_tensors_used: bool,
    pub kernel_family: String,
    pub requested_kernel: String,
    pub selected_kernel: String,
    pub fallback_used: bool,
    pub fallback_reason: Option<String>,
    pub cpu_features: Vec<String>,
    pub threads: usize,
    pub phase: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub tokens_per_second: Option<f64>,
    pub first_token_latency_ms: Option<f64>,
}
```

Strict 258V CPU validation must assert:

- `loader_mode = "real_gguf"`.
- `minimal_loader_fallback_used = false`.
- `mock_tensors_used = false`.
- `requested_backend` and `selected_backend` both identify the CPU AVX2 lane.
- `requested_kernel` equals `selected_kernel` unless the receipt explicitly records a non-strict fallback failure.
- `fallback_used = false` for strict proof receipts.
- Decode and prefill phases are reported separately.

## Work Plan

### LNL258V-001 - Add Platform Profile

Docs/tracking only. Add the 258V platform profile, claim boundaries, data bundles, and cross-links to CPU, Arc 140V, and NPU lanes.

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
