# Intel Arc GPU Setup Guide

## Supported Hardware
- Intel Arc A770 (16GB) — DG2/Alchemist
- Intel Arc A750, A580
- Intel Arc A-series mobile GPUs

## Prerequisites
- Linux kernel 6.2+ (for i915 Arc support)
- Ubuntu 22.04+ or Fedora 37+

## Driver Installation (Ubuntu)

### 1. Add Intel GPU Repository
```bash
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg

echo "deb [signed-by=/usr/share/keyrings/intel-graphics.gpg arch=amd64] \
  https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu.list

sudo apt-get update
```

### 2. Install Compute Runtime
```bash
sudo apt-get install -y \
  intel-opencl-icd \
  libze-intel-gpu1 \
  libze1 \
  clinfo \
  intel-gpu-tools
```

### 3. Verify Installation
```bash
# Check OpenCL devices
clinfo | grep "Device Name"
# Should show: Intel(R) Arc(TM) A770 Graphics

# Check GPU utilization tool
intel_gpu_top  # Press Ctrl+C to exit

# Check kernel driver
dmesg | grep i915 | tail -5
```

## Building BitNet-rs for Intel GPU

```bash
# CPU + OpenCL build
cargo build --release --no-default-features --features cpu

# Run inference
RUST_LOG=info cargo run -p bitnet-cli --no-default-features --features cpu -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Hello" --max-tokens 16
```

## Troubleshooting

### No GPU detected
- Check `dmesg | grep i915` for driver errors
- Ensure firmware is installed: `sudo apt install linux-firmware`
- Verify user is in `render` group: `sudo usermod -aG render $USER`

### OpenCL not available
- Install ICD: `sudo apt install intel-opencl-icd`
- Check ICD config: `ls /etc/OpenCL/vendors/`
- Run `clinfo` to verify

### Performance issues
- Use `intel_gpu_top` to monitor GPU utilization
- Ensure `intel-media-va-driver-non-free` is installed for hardware decode
- Check power management: `cat /sys/class/drm/card0/gt_boost_freq_mhz`
