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


## Build-Out Contracts

The following contracts define the implementation surfaces needed to build the 258V validation platform without moving CPU kernel ownership away from the CPU proof lane.

### LNL258V-RUN-001 - Platform Runtime Probe

Goal: add a visibility-only platform probe for the Lunar Lake laptop. This probe collects CPU, Arc 140V, Level Zero, OpenVINO GPU, OpenVINO NPU, memory, power, and thermal facts. It must not claim BitNet inference, GPU parity, NPU execution, or CPU performance.

Allowed implementation surface:

| Area | Expected files | Notes |
|---|---|---|
| Device probe | `crates/bitnet-device-probe/src/lib.rs`, `crates/bitnet-device-probe/src/intel_lnl258v.rs` | Add the probe type and OS/runtime collectors. |
| CLI or xtask | `crates/bitnet-cli/src/commands/probe_platform.rs` or `xtask/src/platform_probe.rs` | Emit JSON artifacts only. |
| Receipts | `crates/bitnet-receipts-core/src/lib.rs` | Add platform runtime metadata only if needed by the probe artifact. |
| Docs/artifacts | `docs/hardware/intel-258v-validation.md`, this roadmap | Keep claim boundaries visible. |

Forbidden implementation surface:

- QK256 scalar or AVX2 kernels.
- Transformer hot paths.
- GPU or NPU graph execution.
- Any CPU, GPU, or NPU fallback that is counted as target-device proof.

Required Rust shape:

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

Acceptance checks:

- JSON artifact contains `platform="core-ultra-7-258v"`, OS, architecture, CPU AVX2 status, and AVX-512 status.
- Arc 140V evidence records PCI ID, OpenCL platform/device/driver, Level Zero visibility, and OpenVINO `GPU.0` full name when visible.
- NPU evidence records `/dev/accel` or OS device hints, Intel VPU/NPU driver hints, OpenVINO `NPU` visibility, and driver/compiler properties when available.
- `status` is visibility-oriented, for example `runtime_detected`, `partial`, or `unavailable`; it is never an inference result.

### NPU-002-lite - Backend Identity Before Runtime

Goal: stop `npu` from aliasing to Metal, CUDA, generic GPU, OpenCL, or CPU fallback. This is an identity PR, not an OpenVINO graph-execution PR.

Required behavior:

- Add explicit backend request labels such as `IntelNpu` and `OpenVinoNpu`.
- Add explicit device config labels such as `IntelNpu(usize)` and `OpenVinoNpu`.
- Map `npu`, `intel-npu`, and `openvino-npu` to an NPU identity, not `Device::Metal` and not `DeviceConfig::Gpu(0)`.
- In strict mode, fail before inference when Intel NPU is requested but unavailable.
- Receipts must be able to represent `requested_backend=intel-npu` and `selected_backend=intel-npu-openvino` without implying a successful smoke run.

Acceptance checks:

- `npu` never selects Metal on non-Apple hardware.
- `npu` never counts a CPU fallback as NPU execution.
- NPU availability reports identity/probe results and a failure reason when runtime support is missing.

### ARC140V-002 - Exact Integrated GPU Probe

Goal: prove that the integrated GPU visible on the laptop is Arc 140V through OpenCL, Level Zero, and OpenVINO `GPU.0` evidence. This PR must not alter CPU kernels or NPU execution.

Required Rust shape:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelArc140vProbe {
    pub available: bool,
    pub pci_device_id: Option<String>,
    pub opencl_available: bool,
    pub opencl_platform_name: Option<String>,
    pub opencl_device_name: Option<String>,
    pub opencl_driver_version: Option<String>,
    pub level_zero_available: bool,
    pub level_zero_devices: Vec<String>,
    pub openvino_gpu_visible: bool,
    pub openvino_gpu_device: Option<String>,
    pub openvino_gpu_full_name: Option<String>,
    pub shared_memory_bytes: Option<u64>,
    pub power_mode: Option<String>,
    pub failure_reason: Option<String>,
}

pub fn probe_intel_arc_140v() -> IntelArc140vProbe;
```

Detection rules:

- Prefer exact PCI ID `0x64A0` when available.
- Match OpenCL and OpenVINO full names against `Intel(R) Arc(TM) 140V Graphics`.
- Record OpenCL, Level Zero, and OpenVINO independently.
- CPU fallback cannot count as Arc 140V proof.

### CPU258V-001 - CPU Validation Harness

Goal: add a 258V CPU-only validation harness that records strict loader, tokenizer, kernel, backend, and benchmark evidence on Lunar Lake without taking ownership of shared CPU implementation work.

Allowed implementation surface:

| Area | Expected files | Notes |
|---|---|---|
| CLI validation | `crates/bitnet-cli/src/commands/validate_cpu_bitnet.rs` or an `eval` subcommand extension | Run strict CPU validation and emit receipts. |
| Receipts | `crates/bitnet-receipts-core/src/lib.rs` | Add validation receipt fields when missing. |
| Tracking/docs | CPU proof tracker and this platform doc | Cross-link to 8250U-owned CPU work. |

Forbidden by default:

- QK256 dispatch rewrites.
- Scalar/AVX2 kernel rewrites.
- Transformer hot-path rewrites.

Required Rust shape:

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

pub fn run_cpu_bitnet_validation(args: &CpuBitnetValidationArgs) -> anyhow::Result<CpuBitnetValidationReceipt>;
```

Acceptance checks:

- Strict receipt records `loader_mode=real_gguf`, `minimal_loader_fallback_used=false`, and `mock_tensors_used=false`.
- Tokenizer source is explicit: override, GGUF embedded, sibling `tokenizer.json`, sibling `tokenizer.model`, or strict failure.
- `requested_kernel` and `selected_kernel` match in strict success receipts.
- `fallback_used=false` for strict CPU success receipts.
- Decode and prefill phases are reported separately when benchmark data exists.

## Downstream CPU Dependencies

The 258V CPU lane should validate merged CPU work in this order:

| Dependency | Owning lane | Required result before 258V claims |
|---|---|---|
| CPU-BITNET-001 loader authority | i5-8250U CPU proof | Strict GGUF loading cannot use minimal/mock fallback. |
| CPU-BITNET-002 tokenizer authority | i5-8250U CPU proof | Tokenizer source is strict and receipt-backed. |
| CPU-BITNET-003 QK256 layout/scalar truth | CPU proof | Packed layout and scalar output are canonical. |
| CPU-BITNET-004 AVX2 decode/prefill | CPU proof | AVX2 is parity-checked against scalar and selected explicitly. |
| CPU-BITNET-005 runtime cleanup | CPU proof | No hot-path thread-pool rebuilds or dequantizing fallback in strict mode. |
| CPU-BITNET-006 receipts/benchmarks | CPU proof | Receipts include requested/selected kernel, phase, fallback, and timing fields. |

258V validation may run these artifacts and report same-machine results. It should not reassign ownership of these implementation tasks.

## Required Receipt Additions

258V platform and CPU validation receipts need these optional sections when implemented:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitnetExecutionMetadata {
    pub kernel_family: String,
    pub kernel_format: String,
    pub layout: String,
    pub layout_source: String,
    pub requested_kernel: String,
    pub selected_kernel: String,
    pub dequantizes_before_compute: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderMetadata {
    pub mode: String,
    pub minimal_loader_fallback_used: bool,
    pub tokenizer_source: String,
    pub mock_tensors_used: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformRuntimeMetadata {
    pub machine: String,
    pub cpu_features: Vec<String>,
    pub opencl_device_name: Option<String>,
    pub openvino_available_devices: Vec<String>,
    pub runtime_device: Option<String>,
    pub power_mode: Option<String>,
    pub thermal_profile: Option<String>,
}
```

Strict 258V CPU receipts are invalid for performance claims if any of these are missing: model hash, tokenizer source, loader fallback status, requested kernel, selected kernel, fallback status, benchmark phase, thread count, or machine/platform linkage.

## Build and Validation Commands

Initial PR checks:

```bash
cargo fmt --all -- --check
cargo test --locked -p bitnet-device-probe
cargo test --locked -p bitnet-receipts-core
git diff --check
```

CPU proof checks after the 8250U-owned implementation pieces land:

```bash
cargo test --locked -p bitnet-models --no-default-features --features cpu
cargo test --locked -p bitnet-tokenizers --no-default-features --features cpu
cargo test --locked -p bitnet-qk256-dispatch
cargo test --locked -p bitnet-quantization
cargo test --locked -p bitnet-inference
cargo test --release -p bitnet-quantization bench_avx2_speedup -- --ignored --nocapture
```

Manual 258V probe commands are defined in `docs/hardware/intel-258v-validation.md` and should be archived alongside any generated JSON receipt.

## Related Roadmaps

- `docs/specs/intel-lunar-lake-gpu-roadmap.md`
- `docs/specs/intel-lunar-lake-npu-roadmap.md`
- `docs/hardware/intel-258v-validation.md`

Keep implementation work item ownership separate even when the same laptop produces the receipts.
