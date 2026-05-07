# Intel Lunar Lake NPU Roadmap

## Purpose

This document defines the first Intel NPU validation lane for BitNet-rs. The target hardware is Intel Lunar Lake NPU, and the initial runtime path is OpenVINO NPU.

The first milestone is not full BitNet inference. The first useful milestone is strict Intel NPU detection, runtime smoke, no CPU fallback, and machine-readable evidence.

Minimum useful path:

```text
Intel NPU driver installed
OpenVINO Runtime enumerates NPU
bitnet-rs compiles a tiny static-shape OpenVINO model to NPU
strict smoke receipt proves selected_backend=intel-npu-openvino and fallback_used=false
BitNet subgraph parity begins only after smoke evidence exists
```

## Current Claim Boundary

The repository has NPU-shaped scaffolding, but it must not be treated as a working Intel NPU execution path yet.

- `/dev/accel/accel*` detection proves only that a kernel device node exists.
- OpenVINO NPU enumeration proves only that a runtime can see an NPU device.
- `ov::Core::compile_model(model, "NPU")` proves that OpenVINO accepted the NPU compile path for that model.
- A tiny OpenVINO graph proves only that the NPU runtime can execute a test graph.
- Intel GPU/OpenCL execution is a separate backend lane and should be labeled separately.
- Static-shape smoke does not prove dynamic decode-loop support.
- FP16, INT8, U8, INT4, or NF4 OpenVINO paths do not prove packed BitNet QK256 acceleration.
- QK256 CPU execution does not count as Intel NPU execution.
- CPU fallback does not count as Intel NPU execution.
- BitNet subgraph parity does not count as full Intel NPU inference.
- Full inference claims require receipts that record requested backend, selected backend, runtime identity, and fallback status.

## Current Repo Boundaries

Current NPU-shaped code is not enough to claim Lunar Lake NPU support.

- `crates/bitnet-inference/src/npu.rs` currently treats `npu` as Metal-shaped backend routing.
- Device configuration currently risks routing `npu` through generic GPU or CUDA-shaped paths.
- The existing kernel NPU module is Qualcomm QNN/SNPE-oriented and returns explicit not-wired errors for core operations.
- `bitnet-device-probe` currently treats `/dev/accel/accel*` as the main NPU signal.
- The `npu` feature is currently tied to the repo's oneAPI/OpenCL path, which is not the same as Intel NPU through OpenVINO.

Implementation work should use explicit proof labels:

```text
intel-npu-openvino
intel-gpu-opencl
cpu-avx2
cpu-scalar
```

Avoid ambiguous labels such as:

```text
intel
gpu
npu
oneapi
```

## OpenVINO and Driver Constraints

Encode these as implementation constraints until the Lunar Lake laptop provides receipts:

- Intel's Linux NPU driver uses VPU naming in kernel-facing surfaces, so probes should search both NPU and VPU terms, especially `intel_vpu`.
- OpenVINO NPU execution should use `ov::Core::compile_model(model, "NPU")`.
- OpenVINO NPU requires an installed Intel NPU driver.
- Windows-native validation should be supported early; Intel's current Windows NPU driver page lists Windows 11 23H2/24H2/25H2, OpenVINO 2026.1 support, and Lunar Lake under supported platforms. Do not assume WSL can expose the NPU.
- Linux validation is kernel and driver-version sensitive. Collect `/dev/accel`, `intel_vpu` logs, and OpenVINO device enumeration before claiming runtime availability.
- OpenVINO NPU currently supports only static model shapes for the initial smoke/subgraph lane.
- OpenVINO NPU internal inference precision supports F32, F16, and U8 for quantized models; hardware computation precision is FP16.
- OpenVINO NPU model caching and compile latency matter. Receipts should distinguish first-ever compile/inference, cached compile, first inference, and steady-state inference.
- OpenVINO GenAI LLM-on-NPU uses shape/context configuration such as `MAX_PROMPT_LEN` and `MIN_RESPONSE_LEN`; these settings are reference data, not proof that BitNet decode works.
- Remote tensor or cross-device memory-sharing work must reuse the same OpenVINO Core object for a shared Level Zero context when that path is attempted later.

## Validation Levels

### Level 0 - Detected

The device probe reports Intel NPU runtime or device visibility.

Allowed claim:

```text
Intel NPU detected.
```

Required evidence:

- Requested backend is Intel NPU.
- Probe reports whether `/dev/accel/accel*` is present.
- Probe reports whether the kernel driver appears present.
- Probe reports whether the Intel-supported runtime is available.
- Probe reports whether OpenVINO can enumerate an `NPU` device.
- Probe reports runtime version, device name, and driver hint when available.

### Level 1 - Runtime Smoke Tested

A tiny graph executes on Intel NPU with no CPU fallback.

Allowed claim:

```text
Intel NPU runtime executes test graphs.
```

Required evidence:

- Requested backend is Intel NPU.
- Selected backend is Intel NPU through OpenVINO or equivalent runtime.
- Fallback is disabled or rejected.
- Result artifact records pass or fail.

### Level 2 - BitNet Subgraph Parity

One isolated BitNet subgraph runs on Intel NPU and matches CPU reference output within an explicit tolerance.

Allowed claim:

```text
Intel NPU can execute selected BitNet subgraphs.
```

Required evidence:

- CPU reference output.
- NPU output.
- `max_abs_error`.
- `mean_abs_error`.
- Latency.
- Fallback status.

### Level 3 - Inference Path

A strict model run uses Intel NPU for meaningful work and emits a receipt.

Allowed claim:

```text
Intel NPU inference path works for this model and configuration.
```

Required evidence:

- Requested backend.
- Selected backend.
- Runtime name and device identity.
- Kernel or graph identifiers.
- Fallback status.
- Model and configuration identifiers.

### Level 4 - Useful Performance

A receipt-backed benchmark beats the CPU baseline for a defined phase.

Allowed claim:

```text
Intel NPU accelerates BitNet for this workload.
```

Required evidence:

- CPU baseline.
- Intel NPU timing.
- Workload phase, such as prefill or decode.
- Model and prompt configuration.
- Fallback status.

## Backend Identity Requirements

Intel NPU should be represented as a distinct backend identity. It must not be routed through Metal, CUDA, or generic GPU configuration.

Suggested command parsing:

```text
--device cpu           -> CPU
--device cuda          -> CUDA device 0
--device opencl        -> OpenCL device 0
--device metal         -> Metal
--device npu           -> Intel NPU device 0, or an architecture-specific alias only when documented
--device intel-npu     -> Intel NPU device 0
--device intel-npu:1   -> Intel NPU device 1
--device openvino-npu  -> Intel NPU through OpenVINO device 0
```

Receipt labels:

```text
requested_backend = "intel-npu"
selected_backend = "intel-npu-openvino"
runtime = "openvino"
runtime_device = "NPU"
kernel_driver_hint = "intel_vpu"
```

Strict mode rules:

- If Intel NPU is requested and unavailable, fail before inference.
- If Intel NPU is requested and CPU fallback is used, fail validation.
- Fake detection controls are ignored in strict mode.

Auto mode rules:

- CPU fallback may be allowed.
- Receipts and smoke artifacts must record requested backend, selected backend, fallback backend, and fallback reason.

## Detection Layers

Detection should be layered and conservative.

| Layer | Evidence | Working claim |
|---|---|---|
| `/dev/accel/accel*` exists | Kernel driver or device node is present | No |
| Kernel module or driver appears active | Driver hint from OS commands or probe APIs | No |
| Runtime loads | Intel-supported runtime can be loaded | No |
| OpenVINO sees `NPU` | NPU plugin or device is visible | Runtime detected |
| OpenVINO compiles tiny model to `NPU` | `compile_model(..., "NPU")` succeeds | Compile path tested |
| Tiny graph runs | Real NPU graph execution happened | Runtime smoke tested |
| BitNet subgraph parity passes | Useful BitNet work ran on NPU | Subgraph parity tested |
| Full inference receipt validates | End-to-end path used Intel NPU | Receipt backed |

Suggested probe result:

```rust
pub struct IntelNpuProbe {
    pub proof_stage: String,
    pub requested_backend: String,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub runtime_device: Option<String>,

    pub available: bool,

    pub os: String,
    pub arch: String,

    pub accel_device_present: bool,
    pub accel_devices: Vec<String>,

    pub intel_vpu_driver_seen: bool,
    pub driver_version: Option<String>,

    pub openvino_runtime_available: bool,
    pub openvino_version: Option<String>,

    pub openvino_npu_visible: bool,
    pub openvino_available_devices: Vec<String>,
    pub openvino_npu_full_name: Option<String>,

    pub compiler_version: Option<String>,
    pub device_total_mem_size: Option<u64>,
    pub device_alloc_mem_size: Option<u64>,
    pub max_tiles: Option<u32>,

    pub fallback_used: bool,
    pub failure_reason: Option<String>,
}
```

## Lunar Lake Data Bundle

Collect one of these bundles on the Lunar Lake laptop before moving Intel NPU status beyond `scaffold`. Windows-native validation is expected to be the fastest first path if the laptop is running Windows 11 with the Intel NPU driver and OpenVINO installed.

Do not assume WSL can see the NPU.

### Windows PowerShell

```powershell
$ErrorActionPreference = "Continue"

Write-Host "=== Windows ==="
Get-ComputerInfo | Select-Object OsName, OsVersion, WindowsVersion, CsSystemType

Write-Host "=== NPU PnP devices ==="
Get-PnpDevice | Where-Object {
  $_.FriendlyName -match "NPU|Neural|AI Boost|VPU|Multimedia Video"
} | Format-List *

Write-Host "=== OpenVINO Python ==="
python - <<'PY'
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
    ]:
        try:
            props[prop] = str(core.get_property(dev, prop))
        except Exception as e:
            props[prop] = "ERR: " + repr(e)
    out["devices"][dev] = props

print(json.dumps(out, indent=2))
PY
```

### Linux

```bash
set -eux

echo "=== OS ==="
uname -a
cat /etc/os-release || true

echo "=== NPU/VPU PCI ==="
lspci -nn | grep -Ei 'npu|vpu|neural|accel|intel' || true

echo "=== accel devices ==="
ls -l /dev/accel || true

echo "=== driver logs ==="
dmesg | grep -Ei 'intel_vpu|ivpu|vpu|npu|accel' | tail -200 || true

echo "=== OpenVINO ==="
python3 - <<'PY'
import json
import openvino as ov

core = ov.Core()
out = {
    "openvino_version": ov.__version__,
    "available_devices": list(core.available_devices),
}
print(json.dumps(out, indent=2))
PY
```

The bundle should answer:

- Whether the target is native Windows, native Linux, or WSL.
- Whether `/dev/accel/accel*` exists.
- Which kernel driver, Windows PnP device, or module appears active.
- Whether the Intel iGPU is visible separately.
- Whether OpenVINO is installed.
- Whether OpenVINO enumerates `NPU`.

Optional Intel GPU/OpenCL comparison on Linux:

```bash
which clinfo || true
clinfo | grep -Ei 'Platform Name|Device Name|Device Vendor|Device Type' || true
```

## OpenVINO Visibility Check

The first runtime milestone is OpenVINO visibility, not inference.

```bash
python - <<'PY'
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
    ]:
        try:
            props[prop] = str(core.get_property(dev, prop))
        except Exception as e:
            props[prop] = "ERR: " + repr(e)
    out["devices"][dev] = props

print(json.dumps(out, indent=2))
PY
```

Expected receipt or smoke output fields:

```json
{
  "openvino_version": "...",
  "available_devices": ["CPU", "GPU", "NPU"],
  "npu_full_device_name": "...",
  "npu_supported_properties": [],
  "driver_version": "...",
  "compiler_version": "...",
  "device_total_mem_size": 0,
  "device_alloc_mem_size": 0,
  "max_tiles": 1
}
```

## Operation Constraint Checklist

The key implementation question is whether Lunar Lake NPU through OpenVINO can execute useful BitNet-shaped work, not just any neural network graph.

Start with fixed-shape OpenVINO models. Do not start by routing the normal autoregressive decode loop to NPU.

Test these before attempting full inference:

| Test | Purpose |
|---|---|
| FP16 matmul tiny graph | Basic NPU graph execution |
| INT8 matmul tiny graph | Quantized graph execution |
| Dynamic batch or sequence shape | Decode-time shape practicality |
| Constant-weight matmul | Whether weights can be compiled into the graph |
| RMSNorm or LayerNorm | Transformer block support |
| SiLU or GELU | FFN activation support |
| RoPE-equivalent ops | Position encoding feasibility |
| Gather or token embedding | Embedding offload feasibility |
| KV-cache-shaped input/output | Decode-loop copy overhead feasibility |

Receipts for smoke and subgraph tests should include shape contract fields:

```json
{
  "shape_mode": "static",
  "max_prompt_len": 1024,
  "min_response_len": 128,
  "input_shape": [1, 16],
  "output_shape": [1, 16]
}
```

If packed I2_S or QK256 compute is not supported directly, initial useful paths may be:

- CPU or Intel GPU/OpenCL for packed QK256 kernels.
- Intel NPU/OpenVINO for selected FP16 or INT8 subgraphs only.
- Intel NPU smoke only while CPU AVX2 or Intel GPU/OpenCL remains the performance lane.

## PR Plan

### NPU-001 - Add Intel NPU Backend Lane

Documentation and tracking only.

Scope:

- Add `npu` workstream.
- Add `intel_npu` backend status as `scaffold`.
- Record claim boundaries.
- Add follow-up work items.

No runtime code, CPU kernels, QK256 dispatch, or server inference changes.

### NPU-002 - Preserve Intel NPU Backend Identity

Fix backend selection before implementation work.

Acceptance:

- `npu` no longer maps to Metal.
- `npu` no longer maps to CUDA or generic GPU.
- `intel-npu` preserves Intel NPU identity and device index.
- Strict Intel NPU mode fails before inference when unavailable.

### NPU-003 - Add Intel NPU Runtime Detection

Add Intel-specific probe fields.

Expected capability shape:

```rust
pub struct IntelNpuProbe {
    pub proof_stage: String,
    pub requested_backend: String,
    pub selected_backend: Option<String>,
    pub runtime_api: Option<String>,
    pub runtime_device: Option<String>,

    pub available: bool,

    pub os: String,
    pub arch: String,

    pub accel_device_present: bool,
    pub accel_devices: Vec<String>,

    pub intel_vpu_driver_seen: bool,
    pub driver_version: Option<String>,

    pub openvino_runtime_available: bool,
    pub openvino_version: Option<String>,

    pub openvino_npu_visible: bool,
    pub openvino_available_devices: Vec<String>,
    pub openvino_npu_full_name: Option<String>,

    pub compiler_version: Option<String>,
    pub device_total_mem_size: Option<u64>,
    pub device_alloc_mem_size: Option<u64>,
    pub max_tiles: Option<u32>,

    pub fallback_used: bool,
    pub failure_reason: Option<String>,
}
```

The `npu` feature must compile without OpenVINO installed.
`BITNET_NPU_FAKE=intel` is allowed only outside strict mode.
Native Windows and native Linux artifacts should both be valid probe inputs. WSL artifacts are useful only if they prove OpenVINO can see `NPU`.

### NPU-004 - Add Intel NPU Smoke Probe Command

Add a CLI or xtask command that writes machine-readable output without running model inference.

CLI shape:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- intel-npu-probe \
  --json-out ci/hardware/intel-258v/<date>/npu-openvino-runtime-probe.json
```

Use `--strict` when the caller requires OpenVINO to report `NPU`; strict mode
writes the receipt and then fails if OpenVINO NPU visibility is absent.

Example artifact shape:

```json
{
  "requested_backend": "intel-npu",
  "selected_backend": "intel-npu-openvino",
  "runtime_api": "openvino",
  "runtime_device": "NPU",
  "proof_stage": "runtime_detected",
  "accel_device_present": true,
  "intel_vpu_driver_seen": true,
  "openvino_runtime_available": true,
  "openvino_version": "2026.1",
  "openvino_available_devices": ["CPU", "GPU", "NPU"],
  "openvino_npu_visible": true,
  "openvino_npu_full_name": "Intel(R) AI Boost",
  "driver_hint": "intel_vpu",
  "driver_version": "...",
  "compiler_version": "...",
  "device_total_mem_size": 0,
  "device_alloc_mem_size": 0,
  "max_tiles": 1,
  "strict_mode": true,
  "fallback_used": false
}
```

### NPU-005 - Run Tiny OpenVINO NPU Graph Smoke

Run a static F16 graph on Intel NPU through OpenVINO, such as a matmul plus add, and write a receipt-like artifact.

Keep OpenVINO optional. CPU builds must not require OpenVINO libraries or plugins to be installed.

CLI shape:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- intel-npu-smoke \
  --json-out ci/hardware/intel-258v/<date>/npu-tiny-graph-smoke.json
```

Use `--strict` when the caller requires the tiny graph to compile and execute on
OpenVINO `NPU`; strict mode writes the receipt and then fails if graph execution
does not pass. A failed or unavailable smoke receipt must keep
`fallback_used=false` and `cpu_fallback_allowed=false`.

Recommended Cargo shape when implementation starts:

```toml
[features]
npu = ["openvino-npu"]
openvino-npu = ["dep:openvino"]

[dependencies]
openvino = { version = "0.10", optional = true }
```

Execution boundary:

```text
read_model("tiny_matmul.xml")
compile_model(model, "NPU")
create_infer_request()
set input tensor
infer()
read output tensor
compare against CPU expected output
```

Example artifact shape:

```json
{
  "requested_backend": "intel-npu",
  "selected_backend": "intel-npu-openvino",
  "test": "tiny_matmul_f16_1x16",
  "runtime": "openvino",
  "runtime_device": "NPU",
  "shape_mode": "static",
  "input_shape": [1, 16],
  "output_shape": [1, 16],
  "precision": "F16",
  "fallback_used": false,
  "cpu_fallback_allowed": false,
  "result": "pass"
}
```

This moves the backend beyond `scaffold` only after a real Lunar Lake artifact exists.

### NPU-006 - Record Backend Identity and Fallback in Receipts

Receipts should include:

```json
{
  "requested_backend": "intel-npu",
  "selected_backend": "intel-npu-openvino",
  "fallback_backend": null,
  "fallback_used": false,
  "backend_runtime": {
    "name": "openvino",
    "version": "2026.1",
    "device": "NPU",
    "device_name": "...",
    "driver_version": "...",
    "compiler_version": "...",
    "max_tiles": 1
  },
  "shape_contract": {
    "shape_mode": "static",
    "input_shape": [1, 16],
    "output_shape": [1, 16]
  },
  "graph": {
    "name": "tiny_matmul_f16",
    "precision": "F16",
    "cache_dir": "target/bitnet/openvino-cache"
  },
  "timing": {
    "first_ever_compile_and_infer_ms": null,
    "cached_compile_ms": null,
    "first_infer_ms": null,
    "steady_state_infer_ms": null
  },
  "kernels_or_graphs": [
    "tiny_matmul_openvino_npu"
  ]
}
```

Strict Intel NPU validation fails when `fallback_used` is true.

### NPU-007 - Prototype BitNet Subgraph Parity

Start with one isolated subgraph before attempting full autoregressive inference.

Candidate order:

1. RMSNorm.
2. Dense or linear projection.
3. FFN block.
4. Attention prefill block.
5. Output head.

Each prototype must compare CPU reference output with NPU output and record error metrics, latency, selected backend, and fallback status.

### NPU-008 - Evaluate OpenVINO llama.cpp GGUF Backend

Use OpenVINO 2026.1's preview llama.cpp GGUF backend as an external Intel NPU reference lane, not as bitnet-rs production architecture.

Acceptance:

- Run one OpenVINO-validated GGUF on Lunar Lake NPU through llama.cpp/OpenVINO.
- Record model, OpenVINO version, driver version, selected device, fallback status, shape or context settings, and timing.
- Decide whether this informs bitnet-rs native graph lowering.
- Do not claim BitNet QK256 GGUF NPU support from this reference lane alone.

## Collision Avoidance

The Intel NPU lane should avoid:

```text
crates/bitnet-quantization/src/qk256/**
crates/bitnet-qk256-dispatch/**
CPU AVX2/FMA work
decode-loop CPU hot path changes
KV-cache CPU optimization
```

The Intel NPU lane may focus on:

```text
docs/tracking/bitnet-alignment/**
docs/specs/intel-lunar-lake-npu-roadmap.md
crates/bitnet-device-probe/**
crates/bitnet-common/src/types.rs
crates/bitnet-device-config-core/**
crates/bitnet-cli-config-core/**
crates/bitnet-inference/src/npu.rs
crates/bitnet-kernels/src/npu/**
crates/bitnet-cli device or probe commands
receipt backend identity fields
manual Lunar Lake smoke artifacts
```

## Practical Direction

Intel NPU may be useful for graph-level INT8 or FP16 workloads before it is useful for packed BitNet QK256 kernels. If the Intel runtime cannot execute the packed low-bit compute shape directly, the project should evaluate:

1. OpenVINO graph path: represent selected subgraphs as FP16, INT8, INT4, or NF4-style OpenVINO graphs.
2. Hybrid path: keep packed QK256 on CPU or Intel GPU/OpenCL and use NPU for selected static graph components.
3. Research path: explore whether BitNet-specific packed ops can be lowered into an OpenVINO custom or accepted graph form.
4. Fallback path: keep NPU smoke or OpenVINO GenAI only and prioritize CPU AVX2 or Intel GPU/OpenCL for custom kernels.

The next meaningful milestone is a strict Lunar Lake smoke artifact proving that Intel NPU was requested, selected, used for a tiny graph, and did not fall back to CPU.

The Lunar Lake data bundle should decide the next implementation target:

```text
OpenVINO NPU now
Intel GPU/OpenCL first
CPU AVX2 first while NPU remains smoke-only
```

## Related Intel Hardware Lanes

Do not merge these proof lanes:

| Hardware | Roadmap | Primary proof path |
|---|---|---|
| Intel Arc A770 | `docs/specs/intel-arc-a770-gpu-roadmap.md` | Native OpenCL kernels first, OpenVINO GPU second |
| Intel Arc 140V | `docs/specs/intel-lunar-lake-gpu-roadmap.md` | Shared-memory OpenCL and OpenVINO GPU.0 |
| Core Ultra 7 258V platform | `docs/specs/intel-lunar-lake-258v-platform-roadmap.md` | CPU AVX2, Arc 140V GPU, and Intel NPU comparison bundle |

The NPU lane owns OpenVINO `NPU` smoke and static-shape NPU subgraph experiments. The Intel Arc GPU lane owns OpenCL and OpenVINO `GPU.X` validation.

## External References

- Intel Linux NPU driver: https://github.com/intel/linux-npu-driver
- OpenVINO NPU device documentation: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- Intel NPU Driver for Windows: https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
- OpenVINO GenAI on NPU: https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html
- OpenVINO release notes: https://docs.openvino.ai/releasenotes
- OpenVINO Rust bindings: https://docs.rs/openvino
- OpenVINO memory sharing for NPU on Lunar Lake: https://www.intel.com/content/www/us/en/support/articles/000100965.html
