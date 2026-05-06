# Intel Core i5-8250U Validation Profile

## Purpose

This file defines the hardware data bundle for the Intel Core i5-8250U CPU validation lane. It is a CPU-first profile, not a GPU/NPU profile.

Roadmap:

```text
docs/specs/intel-kaby-lake-8250u-cpu-roadmap.md
```

## Hardware Baseline

- Processor: Intel Core i5-8250U.
- Former codename: Kaby Lake R.
- Cores / threads: 4 / 8.
- Base / turbo: 1.60 GHz / 3.40 GHz.
- Cache: 6 MB Smart Cache.
- TDP: 15 W, with cTDP-up 25 W and cTDP-down 10 W.
- Memory: DDR4-2400 or LPDDR3-2133, dual channel.
- Max memory bandwidth: 37.5 GB/s.
- CPU path: AVX2, no AVX-512.
- Optional iGPU: Intel UHD Graphics 620, device ID 0x5917.

## Claim Boundary

- CPU model detection is not kernel execution.
- AVX2 detection is not AVX2 kernel proof.
- CPU proof is not GPU proof.
- UHD 620 OpenVINO visibility is optional and does not count as CPU-path progress.
- Short turbo performance is not sustained performance.
- Strict CPU proof must not use GPU, NPU, mock, or fallback execution paths.

## Required Machine Facts

| Fact | Why it matters |
|---|---|
| OS and kernel / Windows build | CPU scheduling and power behavior differ. |
| CPU model and flags | Confirms AVX2 and absence of AVX-512. |
| Core/thread count | Confirms 4C/8T target. |
| Memory type/capacity | CPU inference may become memory-bandwidth-sensitive. |
| Governor/power mode | Separates performance from powersave behavior. |
| Frequency during run | Distinguishes turbo from sustained behavior. |
| Thermal state | Sustained mobile performance can throttle. |
| OpenVINO CPU visibility | Optional runtime reference for CPU plugin. |
| UHD 620 visibility | Optional/deferred iGPU smoke only. |

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

echo "=== Frequency / governor ==="
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u || true
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | head || true

echo "=== Thermal ==="
find /sys/class/thermal -maxdepth 2 -type f -name temp -print -exec cat {} \; 2>/dev/null || true

echo "=== Memory ==="
free -h
sudo dmidecode -t memory 2>/dev/null || true

echo "=== OpenVINO CPU ==="
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

echo "=== Optional UHD 620 GPU visibility ==="
which clinfo || true
clinfo | grep -Ei 'Platform Name|Device Name|Device Vendor|Device Version|Driver Version|OpenCL C' || true
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

Write-Host "=== Intel GPU optional ==="
Get-PnpDevice | Where-Object {
  $_.FriendlyName -match "UHD|620|Intel.*Graphics"
} | Format-List *

Write-Host "=== OpenVINO CPU ==="
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

The first useful receipt is a CPU feature and dispatch proof:

```json
{
  "hardware": "intel-core-i5-8250u",
  "requested_backend": "cpu",
  "selected_backend": "intel-i5-8250u-cpu-avx2",
  "fallback_used": false,
  "cpu": {
    "cores": 4,
    "threads": 8,
    "avx2": true,
    "avx512": false,
    "selected_kernel_path": "avx2"
  },
  "gpu_or_npu_used": false,
  "status": "cpu_feature_profile_recorded"
}
```

This is not a performance claim. Sustained-load receipts come later.

## Sustained-Load Reporting

Separate at least three phases when benchmarking:

- Cold turbo.
- Warm sustained.
- Throttled, if observed.

Receipts should record duration, frequency, temperature if available, governor or power plan, and whether the machine was on battery or AC power.
