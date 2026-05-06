# Intel Core Ultra 7 258V Validation Profile

## Purpose

This file defines the validation bundle for the Core Ultra 7 258V Lunar Lake laptop. The machine is a tri-device validation platform:

| Device | Proof lane |
|---|---|
| CPU | `intel-258v-cpu-avx2` / `cpu-avx2` Lunar Lake validation and fallback |
| Integrated GPU | `intel-arc-140v-opencl` and `intel-arc-140v-openvino-gpu` |
| NPU | `intel-npu-openvino` / `intel_258v_npu_openvino` |

The 258V laptop should not be treated as a single generic Intel accelerator.

The 258V CPU lane validates the same CPU path on Lunar Lake and provides same-machine comparison against Arc 140V and NPU results. It does not replace the i5-8250U active AVX2 implementation/proof lane.

Platform roadmap:

```text
docs/specs/intel-lunar-lake-258v-platform-roadmap.md
```

## Expected Platform Facts

Expected Core Ultra 7 258V profile:

| Component | Expected value |
|---|---|
| Platform | Lunar Lake |
| CPU | 8 cores / 8 threads |
| CPU topology | 4 P-cores + 4 low-power E-cores |
| CPU backend | CPU AVX2 |
| Memory | Up to 32GB LPDDR5X-8533 shared |
| Integrated GPU | Intel Arc 140V |
| GPU peak | 64 INT8 TOPS |
| GPU PCI device ID | 0x64A0 |
| NPU | Intel AI Boost NPU |
| NPU peak | 47 INT8 TOPS |
| Overall platform peak | 115 INT8 TOPS |

The CPU supports AVX2, but this profile should not assume AVX-512.

## Required Machine Facts

Record these before moving any 258V hardware lane beyond `scaffold`:

| Fact | Why it matters |
|---|---|
| Native Windows, native Linux, or WSL | Do not assume WSL can see the NPU. |
| OpenVINO version | NPU and GPU plugin support is version-sensitive. |
| Intel NPU driver version | Required for NPU receipts. |
| OpenVINO `available_devices` | Should show CPU/GPU/NPU when fully visible. |
| Arc 140V OpenCL visibility | Determines iGPU kernel lane viability. |
| Level Zero visibility | Future lower-level/SYCL path. |
| OpenVINO `GPU.0` full name | Confirms Arc 140V reference target. |
| NPU `compile_model(..., "NPU")` success | Compile path proof. |
| Static-shape tiny graph result | Runtime smoke proof. |
| Shared memory pressure | 32GB LPDDR5X is shared by CPU/GPU/NPU. |
| Power mode / thermal profile | Laptop results depend heavily on power policy. |

## Claim Boundary

- CPU AVX2 correctness does not count as Arc 140V or NPU execution.
- Arc 140V OpenCL execution does not count as NPU execution.
- OpenVINO NPU execution does not count as native OpenCL GPU execution.
- OpenVINO `GPU.0` smoke does not prove BitNet OpenCL kernel acceleration.
- OpenVINO `NPU` smoke does not prove full BitNet inference.
- CPU or GPU fallback cannot count as NPU execution.
- 258V CPU validation must record artifacts without reshaping shared CPU implementation unless explicitly scoped by a ledger item.


## Build-Out Gates

Use this profile as a gate checklist before making any 258V runtime claim.

| Gate | Required evidence | Claim allowed |
|---|---|---|
| Platform visibility | `Lnl258vPlatformProbe` JSON with CPU, Arc 140V, OpenVINO GPU, OpenVINO NPU, memory, power, and OS fields. | The 258V platform is detected. |
| Intel NPU identity | `npu`, `intel-npu`, and `openvino-npu` preserve NPU identity and fail strictly when unavailable. | NPU request routing is identity-safe. |
| Arc 140V identity | PCI ID `0x64A0` or exact Arc 140V full device name through OpenCL/Level Zero/OpenVINO. | Arc 140V runtime is visible. |
| Strict CPU validation | Real GGUF loader, strict tokenizer source, selected CPU AVX2 kernel, no mock tensors, no fallback. | 258V CPU validates the CPU BitNet path. |
| Same-machine comparison | CPU, GPU, and NPU receipts link to the same platform probe artifact. | The lanes can be compared on one laptop. |

The first three gates are visibility and identity work. They do not prove BitNet inference. Strict CPU validation starts only after the CPU-proof lane lands loader and tokenizer authority.

## Manual Probe Additions

The platform bundle above captures the broad machine state. Add these OpenVINO property probes when OpenVINO is installed so the eventual machine-readable probe has exact source fields to mirror.

### OpenVINO device inventory

```python
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
```

### OpenVINO NPU property inventory

```python
import json
import openvino as ov

core = ov.Core()
out = {"available_devices": list(core.available_devices), "npu": {}}
if any(d == "NPU" or d.startswith("NPU.") for d in core.available_devices):
    for prop in [
        "FULL_DEVICE_NAME",
        "SUPPORTED_PROPERTIES",
        "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        "NPU_DRIVER_VERSION",
        "NPU_COMPILER_VERSION",
        "NPU_DEVICE_TOTAL_MEM_SIZE",
        "NPU_DEVICE_ALLOC_MEM_SIZE",
        "NPU_MAX_TILES",
    ]:
        try:
            out["npu"][prop] = str(core.get_property("NPU", prop))
        except Exception as e:
            out["npu"][prop] = "ERR: " + repr(e)
print(json.dumps(out, indent=2))
```

## Machine-Readable Artifact Targets

Store raw manual logs and normalized JSON separately:

```text
ci/receipts/258v/platform/raw/<date>-<os>-probe.txt
ci/receipts/258v/platform/lnl258v-platform.json
ci/receipts/258v/cpu/cpu-bitnet-validation.json
ci/receipts/258v/gpu/arc140v-runtime-probe.json
ci/receipts/258v/npu/intel-npu-runtime-probe.json
```

The normalized platform artifact must include at least these exact identity fields:

```json
{
  "platform": "core-ultra-7-258v",
  "os": "linux|windows|wsl",
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
  "power_mode": "...",
  "thermal_profile": "...",
  "shared_memory_bytes": 0,
  "status": "runtime_detected"
}
```

For strict 258V CPU validation, the CPU artifact must include loader, tokenizer, kernel, backend, and phase fields in addition to benchmark metrics:

```json
{
  "machine": "core-ultra-7-258v",
  "requested_backend": "intel-258v-cpu-avx2",
  "selected_backend": "intel-258v-cpu-avx2",
  "runtime_api": "cpu",
  "loader_mode": "real_gguf",
  "minimal_loader_fallback_used": false,
  "tokenizer_source": "gguf|override|sibling-tokenizer-json|sibling-tokenizer-model",
  "mock_tensors_used": false,
  "kernel_family": "qk256|i2_s|tl2",
  "requested_kernel": "qk256-avx2-gemv",
  "selected_kernel": "qk256-avx2-gemv",
  "fallback_used": false,
  "fallback_reason": null,
  "phase": "prefill|decode_steady_state",
  "prompt_tokens": 0,
  "generated_tokens": 0,
  "tokens_per_second": null,
  "first_token_latency_ms": null
}
```

## Windows PowerShell Bundle

```powershell
$ErrorActionPreference = "Continue"

Write-Host "=== Windows ==="
Get-ComputerInfo | Select-Object OsName, OsVersion, WindowsVersion, CsSystemType

Write-Host "=== CPU ==="
Get-CimInstance Win32_Processor | Format-List Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed

Write-Host "=== Memory ==="
Get-CimInstance Win32_PhysicalMemory | Format-Table Capacity, Speed, Manufacturer, PartNumber

Write-Host "=== Intel GPU / NPU PnP ==="
Get-PnpDevice | Where-Object {
  $_.FriendlyName -match "Arc|140V|NPU|Neural|AI Boost|VPU|Intel.*Graphics"
} | Format-List *

Write-Host "=== OpenCL ==="
where clinfo
clinfo | Select-String -Pattern "Platform Name|Device Name|Device Vendor|Driver Version|OpenCL C"

Write-Host "=== Level Zero / oneAPI ==="
where sycl-ls
sycl-ls
where ze_info
ze_info

Write-Host "=== OpenVINO ==="
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

## Linux Bundle

```bash
set -eux

echo "=== OS ==="
uname -a
cat /etc/os-release || true

echo "=== CPU ==="
lscpu || true

echo "=== Memory ==="
free -h || true

echo "=== GPU / NPU PCI ==="
lspci -nn | grep -Ei 'vga|display|intel|arc|140v|64a0|npu|vpu|neural|accel' || true

echo "=== DRM render nodes ==="
ls -l /dev/dri/renderD* || true
stat -c "%G %n" /dev/dri/renderD* || true
groups "$USER"

echo "=== accel devices ==="
ls -l /dev/accel || true

echo "=== NPU driver logs ==="
dmesg | grep -Ei 'intel_vpu|ivpu|vpu|npu|accel' | tail -200 || true

echo "=== OpenCL ==="
which clinfo || true
clinfo | grep -Ei 'Platform Name|Device Name|Device Vendor|Device Version|Driver Version|OpenCL C|Max compute units|Global memory size' || true

echo "=== Level Zero / oneAPI ==="
which sycl-ls || true
sycl-ls || true
which ze_info || true
ze_info || true

echo "=== OpenVINO ==="
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
    ]:
        try:
            props[prop] = str(core.get_property(dev, prop))
        except Exception as e:
            props[prop] = "ERR: " + repr(e)
    out["devices"][dev] = props
print(json.dumps(out, indent=2))
PY
```

## First Platform Receipt

The first 258V platform receipt should establish visibility only:

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
  "fallback_used": false,
  "status": "runtime_detected"
}
```

This is not an inference claim. Smoke, parity, and benchmark receipts come later.

## Ownership

Proof lanes:

- CPU AVX2 remains under CPU runtime proof.
- Arc 140V OpenCL and OpenVINO GPU are owned by the Intel Arc GPU workstream.
- Intel AI Boost NPU and OpenVINO NPU are owned by the Intel NPU workstream.

The platform profile ties the lanes together for comparison, but it does not merge their claims.
