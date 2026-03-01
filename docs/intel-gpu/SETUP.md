# Intel GPU / A770 — Setup Guide

> Prerequisites, driver installation, and verification for running the
> BitNet-rs OpenCL backend on Intel Arc GPUs.

---

## 1. Prerequisites

| Requirement            | Minimum                  | Recommended              |
|------------------------|--------------------------|--------------------------|
| GPU                    | Intel Arc A750           | Intel Arc A770 (16 GB)   |
| Linux kernel           | 6.2                      | 6.5+ (better i915/xe)   |
| Distribution           | Ubuntu 22.04 LTS         | Ubuntu 24.04 LTS        |
| Intel Compute Runtime  | 23.22+                   | Latest stable            |
| OpenCL ICD Loader      | `ocl-icd-libopencl1`    | Same                     |
| Rust toolchain         | 1.92.0 (MSRV)           | Nightly (for benchmarks) |

> **Note:** Windows support is experimental. The `bitnet-device-probe` crate
> dynamically loads `OpenCL.dll`, but kernel testing targets Linux exclusively.

---

## 2. Ubuntu Setup

### 2.1 Add the Intel GPU repository

```bash
# Import Intel GPG key
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg

# Add the repository (Ubuntu 22.04 / 24.04)
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu.list

sudo apt update
```

### 2.2 Install compute-runtime packages

```bash
sudo apt install -y \
  intel-opencl-icd \
  intel-level-zero-gpu \
  level-zero \
  libze-intel-gpu1 \
  ocl-icd-libopencl1 \
  clinfo
```

### 2.3 Install firmware (if needed)

Kernel 6.2+ ships Arc firmware, but you may need the latest from
`linux-firmware`:

```bash
sudo apt install -y linux-firmware
# Reboot to pick up new firmware blobs
sudo reboot
```

### 2.4 Add your user to the `render` group

```bash
sudo usermod -aG render $USER
sudo usermod -aG video $USER
# Log out and back in for group changes to take effect
```

---

## 3. Verification

### 3.1 `clinfo` — OpenCL device enumeration

```bash
clinfo | grep -E "Device Name|Device Version|Driver Version|Max compute units"
```

Expected output (A770):

```
  Device Name                                     Intel(R) Arc(TM) A770 Graphics
  Device Version                                  OpenCL 3.0 NEO
  Driver Version                                  23.35.27191.42
  Max compute units                               512
```

### 3.2 `sycl-ls` — Level Zero / SYCL enumeration (optional)

If you have the Intel oneAPI base toolkit installed:

```bash
sycl-ls
```

Expected:

```
[opencl:gpu:0] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics
```

### 3.3 `vulkaninfo` — Vulkan driver check (optional)

```bash
vulkaninfo --summary 2>/dev/null | grep -A2 "GPU id"
```

### 3.4 BitNet-rs device probe

```bash
cargo run -p bitnet-device-probe --no-default-features --features cpu 2>&1 | \
  grep -i opencl
```

---

## 4. Troubleshooting

### Missing firmware

**Symptom:** `dmesg` shows `i915: firmware failed to load` or device not
appearing in `clinfo`.

**Fix:**

```bash
sudo apt install linux-firmware
sudo update-initramfs -u
sudo reboot
```

Alternatively, download the latest blobs from
<https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git>
and copy them to `/lib/firmware/i915/`.

### Permission errors (`CL_DEVICE_NOT_FOUND`)

**Symptom:** `clinfo` shows 0 devices, but `sudo clinfo` works.

**Fix:** Ensure your user is in the `render` and `video` groups:

```bash
groups $USER          # should include 'render' and 'video'
ls -la /dev/dri/      # renderD128 should be group 'render'
```

If `/dev/dri/renderD128` has the wrong group, create a udev rule:

```bash
echo 'SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="render", MODE="0660"' | \
  sudo tee /etc/udev/rules.d/70-intel-gpu.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Wrong OpenCL ICD loaded

**Symptom:** `clinfo` reports a CPU-only platform (e.g., Portable Computing
Language) instead of the Intel GPU.

**Fix:** Check which ICD files are present:

```bash
ls /etc/OpenCL/vendors/
# Should include intel.icd
cat /etc/OpenCL/vendors/intel.icd
# Should contain: /usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so
```

If the Intel ICD is missing, reinstall `intel-opencl-icd`.

### Kernel version too old

**Symptom:** Device appears but returns `CL_OUT_OF_RESOURCES` or hangs on
kernel execution.

**Fix:** Upgrade to kernel 6.2+ (6.5+ recommended). Check with:

```bash
uname -r
```

On Ubuntu, install the HWE kernel:

```bash
sudo apt install linux-generic-hwe-22.04
sudo reboot
```

### Building BitNet-rs with OpenCL support

```bash
# CPU-only build (always works, uses CPU reference kernels)
cargo build --no-default-features --features cpu

# OpenCL-enabled build (requires compute-runtime installed)
cargo build --no-default-features --features cpu,oneapi

# Run CPU-reference tests (no GPU required)
cargo nextest run --workspace --no-default-features --features cpu
```

---

## See Also

* [ARCHITECTURE.md](ARCHITECTURE.md) — Backend architecture reference
* [ROADMAP.md](ROADMAP.md) — Development roadmap
* [`docs/INTEL_GPU_SETUP.md`](../INTEL_GPU_SETUP.md) — Legacy setup guide
