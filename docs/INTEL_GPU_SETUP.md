# Intel Arc GPU Setup Guide

## Overview

BitNet-rs supports Intel Arc GPUs (A770, A750, etc.) via OpenCL through Intel's Compute Runtime.
This guide covers environment setup, driver installation, and configuration for Intel GPU inference.

## Requirements

### Hardware
- Intel Arc A-series GPU (A770, A750, A580, etc.)
- PCIe 4.0 x16 slot (recommended)
- 8GB+ VRAM for 2B parameter models

### Software
- Linux kernel ≥ 6.2 (i915 driver with Arc support)
- Ubuntu 22.04/24.04, Fedora 37+, or equivalent
- Intel Compute Runtime (OpenCL 3.0 + Level Zero 1.3)

## Installation

### Step 1: Verify Kernel Support

```bash
uname -r  # Should be ≥ 6.2
dmesg | grep i915  # Should show GuC/HuC loaded
lspci | grep -i vga  # Should show Intel Arc device
```

### Step 2: Install Intel GPU Drivers

#### Ubuntu 22.04/24.04

```bash
# Add Intel GPU repository
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg

echo "deb [signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu jammy unified" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

sudo apt-get update

# Install compute runtime packages
sudo apt-get install -y \
  libze-intel-gpu1 \
  libze1 \
  intel-opencl-icd \
  clinfo \
  intel-gpu-tools
```

#### Fedora 37+

```bash
sudo dnf install -y \
  intel-opencl \
  level-zero \
  clinfo \
  intel-gpu-tools
```

### Step 3: Verify Installation

```bash
# Check OpenCL devices
clinfo | grep -i intel

# Check GPU utilization tool
intel_gpu_top  # Should show your Arc device

# Optional: Check Vulkan
vulkaninfo | grep -i intel
```

### Step 4: User Permissions

Add your user to the `render` and `video` groups:

```bash
sudo usermod -aG render,video $USER
# Log out and back in for changes to take effect
```

## Building BitNet-rs with Intel GPU Support

```bash
# Build with OneAPI/OpenCL support
cargo build --release --no-default-features --features oneapi

# Build with both CPU and Intel GPU
cargo build --release --no-default-features --features cpu,oneapi

# Build optimized for native CPU + Intel GPU
RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
  cargo build --release --no-default-features --features cpu,oneapi
```

## Running Inference

```bash
# Auto-detect device (prefers GPU if available)
cargo run -p bitnet-cli --no-default-features --features oneapi,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Hello world" \
  --max-tokens 32

# Force OpenCL device
cargo run -p bitnet-cli --no-default-features --features oneapi,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --device opencl \
  --prompt "Hello world" \
  --max-tokens 32
```

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `BITNET_GPU_FAKE` | `oneapi`, `none` | Override GPU detection for testing |
| `BITNET_STRICT_MODE` | `1` | Force real hardware detection |
| `RUST_LOG` | `info`, `debug` | Control log verbosity |

## Profiling

### Environment Variable

Set `BITNET_OPENCL_PROFILE=1` to enable per-kernel event timing.  Timing
uses OpenCL's `cl_event` profiling counters (QUEUED → SUBMIT → START → END)
so there is near-zero overhead when disabled.

```bash
BITNET_OPENCL_PROFILE=1 RUST_LOG=info \
  cargo run -p bitnet-cli --no-default-features --features oneapi,full-cli -- run \
    --model models/model.gguf --tokenizer models/tokenizer.json \
    --prompt "Hello" --max-tokens 8
```

A summary is printed at the end of inference:

```
=== OpenCL Profiling Report ===
Kernel                       Queue        Submit          Exec         Total
--------------------------------------------------------------------------------
matmul_i2s               0.012ms       0.003ms       1.234ms       1.249ms
softmax                  0.008ms       0.002ms       0.456ms       0.466ms
--------------------------------------------------------------------------------
Kernels launched: 2  Total exec: 1.690ms  Total wall: 1.715ms
```

The report includes GFLOPS and memory bandwidth when the kernel dimensions
are known (e.g. matmul M×N×K).

### intel_gpu_top

```bash
# Real-time GPU utilisation while inference is running
sudo intel_gpu_top
```

### OpenCL Event Timing

BitNet-rs uses OpenCL event profiling internally when `BITNET_OPENCL_PROFILE=1`
is set.  Lower-level per-event logs are also emitted at `RUST_LOG=debug`.

## Troubleshooting

### "No OpenCL devices found"
1. Check driver installation: `clinfo`
2. Check permissions: user must be in `render` and `video` groups
3. Check kernel module: `lsmod | grep i915`
4. Check firmware: `dmesg | grep -i guc`

### "OpenCL kernel compilation failed"
1. Ensure `intel-opencl-icd` is installed
2. Try updating Intel Compute Runtime to latest version
3. Check `RUST_LOG=debug` output for OpenCL compiler errors

### Performance Issues
1. Use `intel_gpu_top` to check GPU utilization
2. Ensure PCIe link is x16 (check `lspci -vv`)
3. Try adjusting work-group sizes (advanced)

## Architecture

```
┌─────────────────────────────────────┐
│         BitNet Inference Engine      │
├─────────┬───────────┬───────────────┤
│ CPU     │ CUDA      │ OpenCL        │
│ (SIMD)  │ (NVIDIA)  │ (Intel Arc)   │
├─────────┴───────────┴───────────────┤
│        KernelProvider Trait          │
├─────────┬───────────┬───────────────┤
│ AVX2/   │ cudarc    │ opencl3       │
│ AVX-512 │ (PTX)     │ (.cl kernels) │
└─────────┴───────────┴───────────────┘
```

## Known Limitations (MVP)

- OpenCL backend is initial implementation; performance tuning ongoing
- Level Zero backend planned for future optimization
- Vulkan compute path planned as alternative
- Some operations may fall back to CPU during early development
