# Intel Core Ultra 7 258V CPU Validation and Runtime Enablement Buildout

## Purpose

This document turns the Lunar Lake 258V validation report into repo-facing buildout requirements. It keeps three facts separate:

1. the i5-8250U lane owns the active CPU BitNet implementation sequence;
2. the Core Ultra 7 258V CPU lane validates that CPU path on modern Lunar Lake AVX2 silicon;
3. the same laptop also provides Arc 140V and Intel NPU comparison lanes, but those lanes must not be conflated with CPU proof.

The 258V laptop is therefore a same-machine validation platform, not the owner of shared CPU QK256 dispatch, loader, tokenizer, or transformer hot-path implementation unless a tracker item explicitly says otherwise.

## Highest-confidence conclusion

Do not use the Core Ultra 7 258V laptop to take over CPU BitNet implementation. Use it to produce visibility receipts first, then strict CPU validation receipts after the i5-8250U CPU proof lane lands the authoritative loader and tokenizer work.

The expected sequence is:

1. preserve backend identity and runtime probes for Lunar Lake devices;
2. let `CPU-BITNET-001` and `CPU-BITNET-002` settle loader and tokenizer authority under the CPU proof lane;
3. validate scalar versus AVX2 CPU behavior on the 258V with strict receipts;
4. only then compare CPU AVX2, Arc 140V, and OpenVINO NPU artifacts on the same thermally constrained shared-memory laptop.

## Claim boundaries

| Area | Allowed 258V claim before strict receipts | Forbidden claim |
|---|---|---|
| Platform probe | CPU/GPU/NPU runtimes are visible or not visible. | BitNet inference works on any device. |
| 258V CPU lane | The merged CPU path validates or fails on Lunar Lake AVX2. | The 258V lane owns shared CPU implementation by default. |
| Arc 140V lane | OpenCL, Level Zero, and OpenVINO GPU identity are visible. | CPU fallback counts as Arc 140V proof. |
| Intel NPU lane | OpenVINO can enumerate or compile to `NPU` when proven. | NPU maps to Metal, generic GPU, OpenCL, or CPU fallback. |
| Performance | A benchmark receipt records requested/selected kernels and fallback status. | Reasonable-speed claims without strict real-model receipts. |

## Blocking gaps to document and build out

### Loader authority

Strict CPU proof requires one authoritative real GGUF loader path. Compatibility loaders may exist, but they must be opt-in and invalid for strict proof if they use minimal parsing, synthesized tensors, or mock transformer weights.

Required buildout:

- strict mode rejects minimal-loader fallback;
- receipts record `loader.mode`, `loader.minimal_loader_fallback_used`, and `loader.mock_tensors_used`;
- QK256 matrix orientation is normalized to canonical `[out_dim, in_dim]` metadata;
- compatibility fallback remains explicit and cannot silently support proof claims.

### Tokenizer authority

Tokenizer precedence must be deterministic and receipt-backed:

1. explicit override;
2. GGUF-embedded tokenizer;
3. sibling `tokenizer.json`;
4. sibling `tokenizer.model`;
5. strict failure.

Required buildout:

- tokenizer loading returns both the tokenizer and a typed tokenizer-source report;
- receipts record the tokenizer source used by the actual CLI/inference path;
- no hidden basic-tokenizer fallback is allowed in strict proof mode.

### QK256 CPU kernels and dispatch

The scalar and AVX2 QK256 kernels are valid 258V CPU targets, but the validation lane must distinguish raw kernel work from end-to-end runtime speed.

Required buildout:

- scalar QK256 remains the correctness floor;
- AVX2 QK256 is parity-checked against scalar output;
- dispatch records requested kernel, selected kernel, and fallback status;
- decode and prefill interfaces are both represented;
- hot paths avoid repeated tensor materialization, flattening, and allocation where possible.

### Runtime identity

The 258V platform cannot support trustworthy receipts while `npu` can alias to Metal or generic GPU surfaces.

Required buildout:

- explicit backend/device variants exist for Intel NPU, OpenVINO NPU, Arc 140V OpenCL, and OpenVINO GPU;
- strict Intel NPU requests fail before inference when unavailable;
- Arc 140V probes prove exact identity through PCI ID, OpenCL device name, Level Zero visibility, or OpenVINO `GPU.0` full name;
- CPU fallback never satisfies GPU or NPU proof.

## PR buildout sequence

| PR | Purpose | May touch | Must not touch |
|---|---|---|---|
| `NPU-002-lite` / `NPU-002` | Preserve Intel NPU backend identity before runtime work. | device config, backend selection, NPU mapping, strict failure behavior, receipt identity | OpenVINO graph execution, CPU kernels |
| `ARC140V-002` | Prove Arc 140V runtime identity through OpenCL, Level Zero, and OpenVINO GPU evidence. | device probe, GPU roadmap docs, receipts | CPU kernels, NPU runtime |
| `LNL258V-RUN-001` | Add 258V visibility-only platform probe and platform receipt. | `bitnet-device-probe`, CLI/xtask probe command, receipts, hardware docs | QK256 CPU kernels, transformer hot path |
| `CPU-BITNET-001` | Strict GGUF loader authority. | model loader, CLI strict-mode wiring, receipts | GPU/NPU runtime claims |
| `CPU-BITNET-002` | Strict tokenizer authority. | tokenizer crates, CLI tokenizer discovery, receipts | CPU kernel performance claims |
| `CPU-BITNET-003` | QK256 layout and scalar dispatch truth. | layout core, dispatch, scalar tests | 258V-specific platform probe ownership |
| `CPU-BITNET-004` | AVX2 decode/prefill path and parity. | quantization kernels, dispatch tests | GPU/NPU claims |
| `CPU-BITNET-005` | Transformer/runtime cleanup for real BitNet execution. | inference layers, CPU backend, generation path | backend identity shortcuts |
| `CPU-BITNET-006` | Receipt and benchmark fields. | receipts, benchmark docs, CLI emission | unsupported performance claims |
| `CPU258V-001` | 258V CPU validation harness and same-machine receipts. | validation command, receipt generation, hardware/tracking docs | shared CPU implementation unless stacked on CPU proof work |

## Required 258V platform probe API

The visibility-only platform probe should produce a single machine-readable artifact without claiming inference.

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

The initial status values should distinguish at least:

- `not_lunar_lake_258v`;
- `runtime_not_detected`;
- `partial_runtime_detected`;
- `runtime_detected`.

## Required Arc 140V probe API

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

- prefer exact PCI ID `0x64A0`;
- match OpenCL/OpenVINO full device name to `Intel(R) Arc(TM) 140V Graphics` when available;
- record Level Zero separately from OpenCL and OpenVINO;
- never treat CPU fallback as Arc 140V proof.

## Required 258V CPU validation receipt API

The 258V CPU validation harness should measure the merged CPU path without taking ownership of that implementation.

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

Strict CPU validation receipts are invalid if:

- `minimal_loader_fallback_used` is `true`;
- `mock_tensors_used` is `true`;
- tokenizer source is missing or ambiguous;
- `requested_kernel != selected_kernel` without a failed strict result;
- `fallback_used` is `true` for a strict proof pass.

## Receipt schema additions

`InferenceReceipt` should be able to carry these optional BitNet-specific surfaces:

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

## Manual probe bundle fields

The 258V validation bundle must continue to collect the human/manual commands in `docs/hardware/intel-258v-validation.md`. Machine-readable probe output should normalize those facts into fields equivalent to:

```json
{
  "platform": "core-ultra-7-258v",
  "os": "linux|windows",
  "arch": "x86_64",
  "cpu_model": "Intel Core Ultra 7 258V",
  "cpu_cores": 8,
  "cpu_threads": 8,
  "cpu_flags": ["avx2", "fma", "sse4_2"],
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

## Test and check plan

### Platform and identity checks

```bash
cargo fmt --all -- --check
cargo test --locked -p bitnet-device-config-core
cargo test --locked -p bitnet-device-probe
cargo test --locked -p bitnet-receipts-core
```

Acceptance:

- `npu` does not alias to Metal, CUDA, OpenCL, generic GPU, or CPU;
- strict Intel NPU request fails cleanly when unavailable;
- Arc 140V probe cannot report CPU fallback as GPU proof;
- 258V platform probe emits visibility facts only.

### Loader and tokenizer checks

```bash
cargo test --locked -p bitnet-models --no-default-features --features cpu
cargo test --locked -p bitnet-tokenizers --no-default-features --features cpu
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli
```

Acceptance:

- strict mode rejects minimal GGUF fallback;
- strict receipts record `minimal_loader_fallback_used=false`;
- explicit tokenizer override wins;
- GGUF tokenizer wins over siblings when no override exists;
- sibling `tokenizer.json` wins over `tokenizer.model` when both exist;
- missing tokenizer fails in strict mode.

### QK256 CPU checks

```bash
cargo test --locked -p bitnet-quantization qk256_lut_basic -- --exact
cargo test --locked -p bitnet-quantization qk256_block_decode_golden -- --exact
cargo test --locked -p bitnet-quantization qk256_tiny_gemv_e2e -- --exact
cargo test --locked -p bitnet-quantization test_qk256_size_tolerance -- --exact
cargo test --locked -p bitnet-quantization test_gemv_qk256_avx2_smoke -- --exact
cargo test --release -p bitnet-quantization bench_avx2_speedup -- --ignored --nocapture
```

Acceptance:

- lookup-table mapping remains `0 -> -2`, `1 -> -1`, `2 -> +1`, `3 -> +2`;
- tail handling matches scalar reference within the documented tolerance;
- size tolerance accepts up to `+128` bytes and rejects `+129` bytes;
- AVX2 output matches scalar output within mixed absolute/relative tolerance;
- benchmark output records requested kernel, selected kernel, and fallback status.

## First-pass 258V performance gates

These are acceptance gates, not guarantees:

| Benchmark | Initial gate | Reason |
|---|---:|---|
| QK256 AVX2 GEMV versus scalar | `>= 2.0x`, target `>= 3.0x` | AVX2 should be worth selecting. |
| Prefill AVX2 path versus scalar | `>= 1.5x` for `T >= 128` | Prefill is heavier and memory-sensitive. |
| End-to-end strict decode | `>= 1.5x` over scalar-only build | Kernel wins must survive runtime overhead. |
| Loader/tokenizer overhead after first run | `< 5%` of total time | Scaffolding must not hide kernel speed. |
| Strict fallback rate | `0` | Correctness requirement. |

If these gates fail, prioritize runtime plumbing before rewriting only the AVX2 inner kernel: tensor extraction, allocation, dispatch, prefill batching, thread-pool setup, and dequantizing fallback paths are all likely culprits.

## Documentation update requirements for implementation PRs

Every PR in this sequence must update documentation before claiming support:

- update `docs/hardware/intel-258v-validation.md` when probe fields or manual commands change;
- update `docs/specs/intel-lunar-lake-258v-platform-roadmap.md` when the platform claim boundary changes;
- update `docs/specs/intel-lunar-lake-gpu-roadmap.md` when Arc 140V identity requirements change;
- update `docs/specs/intel-lunar-lake-npu-roadmap.md` when NPU identity, runtime smoke, or OpenVINO properties change;
- update `docs/bitnet/BITNET_RECEIPT_FIELDS.md` when receipt fields change;
- update `docs/tracking/campaigns/*/active.toml` when lane ownership, allowed paths, or acceptance criteria change.
