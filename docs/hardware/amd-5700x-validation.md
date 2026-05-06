# AMD Ryzen 7 5700X Validation Profile

## Purpose

This file defines the hardware data bundle for the AMD Ryzen 7 5700X CPU validation lane. It is a CPU-first profile, not an accelerator profile.

Roadmap:

```text
docs/specs/amd-5700x-cpu-roadmap.md
```

## Hardware Baseline

- CPU: AMD Ryzen 7 5700X.
- Architecture: Zen 3 / Vermeer.
- Socket: AM4.
- Cores / threads: 8 / 16.
- Base / boost: 3.4 GHz / up to 4.6 GHz.
- L3 cache: 32 MB.
- TDP: 65 W.
- Memory: DDR4-3200.
- PCIe: PCIe 4.0.
- Extensions: AVX2, AVX, FMA3.
- Integrated graphics: none.

## Claim Boundary

- CPU model detection is not kernel execution.
- AVX2 detection is not AVX2 kernel proof.
- 5700X has no AVX-512 lane.
- CPU proof is not GPU/NPU proof.
- DDR4/AM4 results are not directly comparable to DDR5/AM5 without memory context.
- Strict CPU proof must not use GPU, NPU, mock, or fallback execution paths.

## Required Machine Facts

| Fact | Why it matters |
|---|---|
| OS and kernel / Windows build | CPU scheduling and power behavior differ. |
| CPU model and flags | Confirms AVX2 and absence of AVX-512. |
| Core/thread topology | Confirms 8C/16T target. |
| Memory type/capacity | DDR4 context is required for comparisons. |
| Governor/power mode | Separates performance from powersave behavior. |
| Frequency during run | Distinguishes boost from sustained behavior. |
| Thermal state | Sustained desktop performance still depends on cooling. |
| Optional OpenVINO CPU visibility | Optional runtime reference for CPU plugin. |

## Linux Bundle

```bash
set -eux

echo "=== OS ==="
uname -a
cat /etc/os-release || true

echo "=== CPU ==="
lscpu
grep -m1 "model name" /proc/cpuinfo || true
grep -m1 "flags" /proc/cpuinfo || true

echo "=== Topology ==="
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ || true
numactl --hardware || true

echo "=== Frequency / governor ==="
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u || true
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | head || true

echo "=== Thermal ==="
find /sys/class/thermal -maxdepth 2 -type f -name temp -print -exec cat {} \; 2>/dev/null || true

echo "=== Memory ==="
free -h
sudo dmidecode -t memory 2>/dev/null || true

echo "=== Rust ==="
rustc --version
cargo --version

echo "=== OpenVINO CPU optional ==="
python3 - <<'PY' || true
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

## Windows PowerShell Bundle

```powershell
$ErrorActionPreference = "Continue"

Write-Host "=== Windows ==="
Get-ComputerInfo | Select-Object OsName, OsVersion, WindowsVersion, CsSystemType

Write-Host "=== CPU ==="
Get-CimInstance Win32_Processor | Format-List Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed

Write-Host "=== Memory ==="
Get-CimInstance Win32_PhysicalMemory | Format-Table Capacity, Speed, Manufacturer, PartNumber

Write-Host "=== Power plan ==="
powercfg /GETACTIVESCHEME

Write-Host "=== Rust ==="
rustc --version
cargo --version

Write-Host "=== OpenVINO CPU optional ==="
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

## First CPU Receipt

```json
{
  "machine_id": "amd-5700x",
  "requested_backend": "cpu",
  "selected_backend": "amd-5700x-cpu-avx2",
  "fallback_used": false,
  "cpu": {
    "vendor": "AMD",
    "model": "Ryzen 7 5700X",
    "architecture": "Zen 3",
    "cores": 8,
    "threads": 16,
    "l3_cache_bytes": 33554432,
    "avx2_detected": true,
    "avx512_detected": false,
    "tdp_watts": 65
  },
  "gpu_or_npu_used": false,
  "status": "cpu_feature_profile_recorded"
}
```

This is not a performance claim. DDR4/AM4 sustained benchmark receipts come later.
