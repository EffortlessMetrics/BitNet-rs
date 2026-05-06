# Intel Arc A770 Validation Profile

## Purpose

This file defines the hardware data bundle for the Intel Arc A770 validation lane. It is a machine profile and probe checklist, not a claim that A770 execution works.

Roadmap:

```text
docs/specs/intel-arc-a770-gpu-roadmap.md
```

## Required Machine Facts

Record these before moving A770 beyond `scaffold`:

| Fact | Why it matters |
|---|---|
| OS and kernel/build | Driver and OpenCL stack differ sharply. |
| Motherboard, CPU, chipset | ReBAR support and PCIe behavior. |
| ReBAR enabled | Required for optimal Arc A-Series performance claims. |
| PCIe generation/link width | A770 should be close to PCIe 4.0 x16. |
| Exact A770 board and VRAM | Confirms 16GB path, not 8GB. |
| PCI device ID | Expected A770 ID is 0x56A0. |
| Driver version | Required in receipts. |
| OpenCL platform/device name | Confirms native kernel target. |
| Level Zero visibility | Future lower-level/SYCL path. |
| OpenVINO GPU.X index | Avoids accidentally targeting an iGPU. |
| Linux render-node permissions | Required for non-root GPU compute. |
| Power/thermal behavior | A770 is a 225W-class card; benchmarks need stable clocks. |

## Claim Boundary

- `clinfo` visibility is runtime detection only.
- OpenCL program compilation is compile smoke only.
- Tiny OpenCL kernel execution is kernel smoke only.
- CPU/OpenCL parity is parity only.
- CPU fallback cannot count as A770 execution.
- Performance claims require benchmark artifacts and receipts.

## Linux Bundle

```bash
set -eux

echo "=== OS ==="
uname -a
cat /etc/os-release || true

echo "=== PCI / A770 ==="
lspci -nn | grep -Ei 'vga|display|intel|arc|a770|56a0' || true
lspci -vv | grep -A40 -Ei 'VGA.*Intel|Display.*Intel' || true

echo "=== ReBAR / PCI resources ==="
lspci -vv | grep -Ei 'Resizable BAR|Region|Memory at' || true

echo "=== DRM render nodes ==="
ls -l /dev/dri/renderD* || true
stat -c "%G %n" /dev/dri/renderD* || true
groups "$USER"

echo "=== OpenCL ==="
which clinfo || true
clinfo | grep -Ei 'Platform Name|Device Name|Device Vendor|Device Version|Driver Version|OpenCL C|Max compute units|Global memory size' || true

echo "=== Level Zero / oneAPI ==="
which sycl-ls || true
sycl-ls || true
which ze_info || true
ze_info || true

echo "=== XPU-SMI ==="
which xpu-smi || true
xpu-smi discovery || true
xpu-smi dump -d 0 -m 0,1,2,3,4,5 || true

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

## Windows PowerShell Bundle

```powershell
$ErrorActionPreference = "Continue"

Write-Host "=== GPU devices ==="
Get-PnpDevice | Where-Object {
  $_.FriendlyName -match "Arc|A770|Intel.*Graphics"
} | Format-List *

Write-Host "=== OpenCL tools ==="
where clinfo
clinfo | Select-String -Pattern "Platform Name|Device Name|Device Vendor|Driver Version|OpenCL C|Global memory size"

Write-Host "=== oneAPI / Level Zero tools ==="
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

## Receipt Checklist

Every A770 smoke, parity, or benchmark artifact should record:

- Requested backend.
- Selected backend.
- Runtime API, such as OpenCL or OpenVINO.
- Device name.
- PCI device ID.
- VRAM bytes.
- Driver version.
- OpenCL platform and device index.
- OpenVINO `GPU.X` index for reference runs.
- ReBAR status when available.
- PCIe link width/generation when available.
- Kernel or graph ID.
- Fallback status and fallback reason.

## First Hardware Receipt

The first useful hardware receipt is not an inference run. It is a machine profile showing:

```json
{
  "hardware": "intel-arc-a770",
  "pci_device_id": "0x56A0",
  "vram_bytes": 17179869184,
  "opencl_available": true,
  "level_zero_available": true,
  "openvino_gpu_visible": true,
  "openvino_gpu_device": "GPU.1",
  "rebar_enabled": true,
  "fallback_used": false,
  "status": "runtime_detected"
}
```

This proves detection only. Kernel smoke and parity are later receipts.
