# Intel Lunar Lake 258V BitNet Buildout Plan

## Purpose

This document converts the Lunar Lake validation report into buildable work for BitNet-rs. The Core Ultra 7 258V laptop is a same-machine validation platform, not the owner of the shared CPU implementation sequence.

The highest-priority rule is:

```text
Use the Core Ultra 7 258V to prove platform identity, backend identity, runtime visibility, strict receipts, and CPU validation artifacts. Keep core CPU loader/tokenizer/QK256 implementation ownership with the active CPU proof lane unless a ledger item explicitly transfers or stacks that work.
```

## Lane Ownership

| Lane | Proof label | Role | Editing boundary |
|---|---|---|---|
| i5-8250U CPU | `intel-i5-8250u-cpu-avx2` | Active low-power AVX2 CPU implementation and strict proof lane | Owns shared CPU implementation sequence unless a work item says otherwise. |
| 258V CPU | `intel-258v-cpu-avx2` | Lunar Lake CPU validation and same-machine comparison lane | Records validation artifacts; must not reshape shared CPU hot paths by default. |
| Arc 140V | `intel-arc-140v-opencl`, `intel-arc-140v-openvino-gpu` | Integrated GPU identity, smoke, parity, and benchmark lane | Must not count CPU fallback as GPU proof. |
| Intel NPU | `intel-npu-openvino` | OpenVINO NPU identity, static-shape smoke, and subgraph lane | Must not be routed through Metal or generic GPU identities. |
| 258V platform | `core-ultra-7-258v` | Machine fact collector tying CPU/GPU/NPU artifacts together | Does not merge CPU, GPU, and NPU claims. |

## Immediate Buildout Sequence

Do these identity and validation surfaces before claiming real BitNet performance on the 258V laptop:

1. **NPU-002-lite**: preserve Intel NPU backend identity and stop mapping `npu` to Metal or generic GPU paths.
2. **ARC140V-002**: prove Arc 140V runtime identity through PCI ID, OpenCL, Level Zero, and OpenVINO `GPU.0` evidence.
3. **LNL258V-RUN-001**: add a visibility-only 258V platform probe receipt that records CPU, Arc 140V, OpenVINO GPU, and OpenVINO NPU facts without claiming inference.
4. **CPU-BITNET-001 / CPU-BITNET-002**: land strict GGUF loader and tokenizer authority in the active 8250U-owned CPU proof lane.
5. **CPU258V-001**: run the merged CPU path on the 258V laptop and emit strict validation receipts for scalar-vs-AVX2 parity, loader/tokenizer authority, kernel identity, and decode/prefill metrics.
6. **Comparison work**: only after the lane receipts exist, compare CPU AVX2, Arc 140V, and OpenVINO NPU artifacts on the same shared-memory platform.

## PR Contracts

| PR | Purpose | May touch | Must not touch | Acceptance |
|---|---|---|---|---|
| `NPU-002-lite` | Identity-before-runtime cleanup for Intel NPU. | Backend request parsing, device config, NPU backend selection, probe metadata, receipt identity fields. | OpenVINO graph execution, CPU QK256 kernels, transformer hot path. | `npu`, `intel-npu`, and `openvino-npu` preserve NPU identity; strict NPU requests fail cleanly when unavailable; receipts can represent requested and selected NPU backends. |
| `ARC140V-002` | Exact Arc 140V runtime identity probe. | `bitnet-device-probe`, Arc 140V probe module, hardware docs, optional CLI/xtask probe command. | CPU kernels, NPU runtime, transformer hot path. | Probe records PCI ID when available, OpenCL device name/driver, Level Zero visibility, OpenVINO `GPU.0` full name, and never emits CPU fallback as Arc proof. |
| `LNL258V-RUN-001` | Whole-laptop visibility receipt. | `bitnet-device-probe`, receipt schema, CLI/xtask platform probe command, 258V docs. | QK256 CPU kernels, shared transformer hot path. | JSON artifact records CPU AVX2 facts, Arc 140V visibility, Level Zero visibility, OpenVINO devices, NPU visibility, power/thermal context, and `status=runtime_detected` or a failure reason. |
| `CPU258V-001` | 258V CPU validation harness. | CLI validation command, receipts, hardware/tracking docs, machine profile artifacts. | Shared CPU implementation logic unless stacked on the CPU proof lane. | Strict CPU receipt records loader mode, tokenizer source, mock/fallback status, requested/selected kernel, phase metrics, and same-machine platform link. |

## Required Platform Probe Surface

Add a dedicated Lunar Lake platform probe surface instead of overloading generic device detection:

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

Initial files:

- `crates/bitnet-device-probe/src/intel_lnl258v.rs`
- `crates/bitnet-device-probe/src/lib.rs`
- CLI or xtask probe command for writing the JSON artifact
- `crates/bitnet-receipts-core/src/lib.rs`
- `docs/hardware/intel-258v-validation.md`
- `docs/specs/intel-lunar-lake-258v-platform-roadmap.md`

The platform probe is visibility-only. It must not compile or run a model, and it must not mark CPU fallback as GPU or NPU success.

## Required Arc 140V Probe Surface

Add a probe that can distinguish Arc 140V from a generic Intel GPU:

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

- Prefer exact PCI device ID `0x64A0` when available.
- Match OpenCL/OpenVINO full device names against `Intel(R) Arc(TM) 140V Graphics`.
- Record OpenCL, Level Zero, and OpenVINO `GPU.0` as separate facts.
- Do not allow CPU fallback to satisfy any Arc 140V proof field.

## Required NPU Identity Cleanup

The NPU identity work is deliberately smaller than real OpenVINO graph execution. It must make backend selection honest first.

Add explicit request/config variants such as:

```rust
pub enum BackendRequest {
    IntelNpu,
    OpenVinoNpu,
    IntelArc140v,
    IntelArc140vOpenVinoGpu,
}

pub enum DeviceConfig {
    IntelNpu(usize),
    OpenVinoNpu,
    IntelArc140v(usize),
    OpenVinoGpu(usize),
}
```

Backend token mapping should preserve NPU identity:

```rust
pub fn map_device_token(token: &str) -> Option<Device> {
    match token {
        "cpu" => Some(Device::Cpu),
        "cuda" | "gpu" => Some(Device::Cuda(0)),
        "opencl" | "intel-gpu" => Some(Device::OpenCL(0)),
        "npu" | "intel-npu" | "openvino-npu" => Some(Device::Npu),
        "metal" => Some(Device::Metal),
        _ => None,
    }
}
```

Strict mode must fail before inference if Intel NPU is requested and no Intel NPU runtime/probe result is available. It must not silently select Metal, generic GPU, or CPU.

## Required 258V CPU Validation Receipt

The 258V CPU harness should measure the merged CPU path without taking over the implementation surfaces. It should emit a receipt shaped like:

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

pub fn run_cpu_bitnet_validation(
    args: &CpuBitnetValidationArgs,
) -> anyhow::Result<CpuBitnetValidationReceipt>;
```

Strict 258V CPU validation is invalid if any of these are true:

- `minimal_loader_fallback_used == true`
- `mock_tensors_used == true`
- `fallback_used == true`
- `requested_kernel != selected_kernel`
- tokenizer source is missing or says a compatibility fallback was used
- the receipt cannot link to the 258V platform probe artifact

## Downstream CPU Implementation Dependencies

The 258V CPU lane depends on the CPU proof lane delivering these implementation surfaces:

| Concern | Canonical authority | Required outcome |
|---|---|---|
| GGUF loader | `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs` | Strict mode rejects minimal/mock fallback and emits layout/quantization metadata. |
| Tokenizer | `crates/bitnet-tokenizers/src/auto.rs`, `crates/bitnet-tokenizers/src/gguf_loader.rs`, `crates/bitnet-cli/src/tokenizer_discovery.rs` | Deterministic source precedence and receipt-backed source reporting. |
| QK256 layout | `crates/bitnet-qk256-layout-core/src/lib.rs` | One canonical packed matrix semantics and dimension/orientation validation. |
| QK256 dispatch | `crates/bitnet-qk256-dispatch/src/lib.rs` | Requested/selected scalar or AVX2 kernel is visible to strict mode and receipts. |
| Scalar/AVX2 kernels | `crates/bitnet-quantization/src/i2s_qk256.rs`, `crates/bitnet-quantization/src/i2s_qk256_avx2.rs` | Scalar truth, AVX2 parity, decode path, and prefill-oriented batched path. |
| Quantized linear | `crates/bitnet-inference/src/layers/quantized_linear.rs` | Avoid repeated hot-path extraction, reshape, dequantization, and allocation. |
| CPU backend | `crates/bitnet-inference/src/backends.rs` | Avoid per-call global thread-pool construction; record backend identity and fallback status. |
| Receipts | `crates/bitnet-receipts-core/src/lib.rs` | Store loader, tokenizer, platform, requested/selected kernel, and phase metrics. |

## Required Receipt Schema Extensions

Add BitNet-specific metadata to inference receipts so claims can be audited:

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

These fields should be attached to the existing inference receipt as optional structured metadata until all callers are migrated.

## Probe Commands To Preserve

The manual 258V platform bundle remains the human-readable source of truth. The Rust probe should mirror these facts in JSON:

```bash
uname -a
cat /etc/os-release
lscpu
free -h
lspci -nn | grep -Ei 'vga|display|intel|arc|140v|64a0|npu|vpu|neural|accel'
ls -l /dev/dri/renderD*
ls -l /dev/accel
dmesg | grep -Ei 'intel_vpu|ivpu|vpu|npu|accel' | tail -200
clinfo
sycl-ls
ze_info
python3 - <<'PY'
import json
import openvino as ov
core = ov.Core()
out = {
  "openvino_version": ov.__version__,
  "available_devices": list(core.available_devices),
  "devices": {}
}
for dev in core.available_devices:
    props = {}
    for prop in [
        "FULL_DEVICE_NAME",
        "SUPPORTED_PROPERTIES",
        "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        "DEVICE_ARCHITECTURE",
        "DEVICE_UUID",
    ]:
        try:
            props[prop] = str(core.get_property(dev, prop))
        except Exception as e:
            props[prop] = "ERR: " + repr(e)
    out["devices"][dev] = props
print(json.dumps(out, indent=2))
PY
```

When OpenVINO reports `NPU`, record extended NPU properties when available: `NPU_DRIVER_VERSION`, `NPU_COMPILER_VERSION`, `NPU_DEVICE_TOTAL_MEM_SIZE`, `NPU_DEVICE_ALLOC_MEM_SIZE`, and `NPU_MAX_TILES`.

## Test And CI Gates

| Work item | Required commands | Required assertions |
|---|---|---|
| NPU identity | `cargo fmt --all -- --check`; `cargo test --locked -p bitnet-device-config-core`; `cargo test --locked -p bitnet-inference`; `cargo test --locked -p bitnet-device-probe` | `npu` is not Metal or generic GPU; strict unavailable NPU fails before inference. |
| Arc 140V probe | `cargo fmt --all -- --check`; `cargo test --locked -p bitnet-device-probe` | Arc probe records exact identity fields and does not emit CPU fallback proof. |
| 258V platform probe | `cargo fmt --all -- --check`; `cargo test --locked -p bitnet-device-probe`; `cargo test --locked -p bitnet-receipts-core` | Visibility-only JSON includes CPU AVX2, Arc 140V, OpenVINO devices, NPU visibility, and failure reason when incomplete. |
| Strict loader/tokenizer | `cargo test --locked -p bitnet-models --no-default-features --features cpu`; `cargo test --locked -p bitnet-tokenizers --no-default-features --features cpu`; `cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli` | Minimal loader fallback and hidden tokenizer fallback are impossible in strict mode. |
| QK256 CPU proof | `cargo test --locked -p bitnet-quantization`; `cargo test --locked -p bitnet-qk256-dispatch`; `cargo test --locked -p bitnet-inference`; `cargo test --release -p bitnet-quantization bench_avx2_speedup -- --ignored --nocapture` | Scalar truth, AVX2 parity, requested/selected kernel reporting, and zero strict fallback. |
| 258V CPU validation | `cargo fmt --all -- --check`; `cargo test --locked -p bitnet-receipts-core`; `cargo test --locked -p bitnet-cli`; manual 258V platform bundle | Receipt records loader/tokenizer/kernel/platform/phase metrics and links to the same-machine platform artifact. |

## First Acceptance Targets

These are validation gates, not performance guarantees:

| Benchmark | First-pass target |
|---|---|
| QK256 AVX2 decode GEMV vs scalar | At least `2.0x`, with `3.0x` as the near-term target. |
| Prefill AVX2 path vs scalar | At least `1.5x` for `T >= 128`. |
| End-to-end strict decode | At least `1.5x` over scalar-only with the same model, prompt, and thread count. |
| Loader/tokenizer overhead | Less than `5%` of steady-state total beyond first-run load. |
| Strict fallback rate | Exactly `0`. |

If the kernel microbench passes but end-to-end decode does not, investigate allocation, tensor materialization, dequantization fallback, thread-pool construction, and receipt-blind dispatch before changing the AVX2 inner loop.

## Done Criteria

The Lunar Lake 258V buildout is ready to move from planning to proof only when all of these exist:

- A visibility-only platform receipt for the exact 258V laptop.
- An Arc 140V probe receipt with exact GPU identity evidence.
- An Intel NPU identity receipt that cannot be confused with Metal, CPU, or generic GPU.
- Strict CPU proof from the CPU implementation lane with no loader/tokenizer/mock fallback.
- 258V CPU validation receipts proving the merged CPU path on Lunar Lake with requested/selected kernel fields.
- Benchmark artifacts split by `prefill`, `first_token`, and `decode_steady_state`.
- Cross-device comparison docs that cite separate CPU, Arc 140V, and NPU artifacts instead of merging claims.
