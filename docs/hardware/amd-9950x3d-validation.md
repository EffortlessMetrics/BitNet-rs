# AMD Ryzen 9 9950X3D Validation Profile

## Purpose

This file defines the hardware data bundle for the AMD Ryzen 9 9950X3D CPU validation lane. It is a CPU-first profile, not an accelerator profile.

Roadmap:

```text
docs/specs/amd-9950x3d-cpu-roadmap.md
```

## Hardware Baseline

- CPU: AMD Ryzen 9 9950X3D.
- Architecture: Zen 5 / Granite Ridge.
- Socket: AM5.
- Cores / threads: 16 / 32.
- Base / boost: 4.3 GHz / up to 5.7 GHz.
- L3 cache: 128 MB.
- TDP: 170 W.
- Memory: DDR5.
- PCIe: PCIe 5.0.
- Extensions: AVX-512, AVX2, AVX, FMA3.
- Cooling: liquid cooler recommended for optimal performance.

## Claim Boundary

- CPU model detection is not kernel execution.
- AVX-512 detection is not AVX-512 kernel proof.
- AVX2 proof is not AVX-512 proof.
- Short boost behavior is not sustained performance.
- X3D/cache-sensitive wins require benchmark receipts.
- Strict CPU proof must not use GPU, NPU, mock, or fallback execution paths.

## Required Machine Facts

| Fact | Why it matters |
|---|---|
| OS and kernel / Windows build | CPU scheduling and power behavior differ. |
| CPU model and flags | Confirms AVX2 and AVX-512. |
| Core/thread topology | Dual-CCD and scheduling behavior matter. |
| Cache-domain context | X3D behavior can be cache-placement sensitive. |
| Memory type/capacity | DDR5 context is required for comparisons. |
| Governor/power mode | Separates performance from powersave behavior. |
| Frequency during run | Distinguishes boost from sustained behavior. |
| Thermal/cooling state | 170 W desktop results depend on cooling. |
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
  "machine_id": "amd-9950x3d",
  "requested_backend": "cpu",
  "selected_backend": "amd-9950x3d-cpu-avx512",
  "fallback_used": false,
  "cpu": {
    "vendor": "AMD",
    "model": "Ryzen 9 9950X3D",
    "architecture": "Zen 5",
    "cores": 16,
    "threads": 32,
    "l3_cache_bytes": 134217728,
    "avx2_detected": true,
    "avx512_detected": true,
    "tdp_watts": 170
  },
  "gpu_or_npu_used": false,
  "status": "cpu_feature_profile_recorded"
}
```

This is not a performance claim. Cache-sensitive and sustained-power receipts come later.
