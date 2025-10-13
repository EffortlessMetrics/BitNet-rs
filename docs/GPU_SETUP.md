# GPU Setup Guide

This guide covers GPU setup for BitNet-rs across different platforms, with special focus on WSL2 CUDA support.

## Table of Contents
- [Quick Start](#quick-start)
- [WSL2 CUDA Setup](#wsl2-cuda-setup)
- [Native Linux Setup](#native-linux-setup)
- [macOS Metal Setup](#macos-metal-setup)
- [Verification](#verification)

## Quick Start

### One-Click Commands

```bash
# Check GPU availability
cargo xtask gpu-preflight

# Run GPU smoke tests
cargo xtask gpu-smoke

# Run all tests with GPU
cargo gpu-tests

# Run demos
cargo xtask demo --which all
```

### Cargo Aliases

The following aliases are available for quick testing:

- `cargo tw` - Test workspace with CPU features
- `cargo gpu-tests` - Test workspace with GPU features
- `cargo cpu-tests` - Test workspace with CPU features (explicit)
- `cargo gpu-smoke` - Run GPU smoke test
- `cargo gpu-parity` - Run GPU parity tests
- `cargo gpu-build` - Build with GPU features
- `cargo demo-sys` - Run system reporting demo
- `cargo demo-all` - Run all demos

## GPU Backend Support

BitNet-rs supports multiple GPU backends:

- **CUDA** - NVIDIA GPUs (requires CUDA toolkit)
- **Metal** - Apple Silicon (built-in on macOS)
- **ROCm** - AMD GPUs (requires ROCm installation)
- **WebGPU** - Platform-dependent fallback (not guaranteed)

## WSL2 CUDA Setup

### Prerequisites

1. **Windows 11** or **Windows 10** (version 21H2 or later)
2. **WSL2** (not WSL1)
3. **NVIDIA GPU** with recent Windows driver

### Step 1: Verify WSL2

```powershell
# In Windows PowerShell (as admin)
wsl -l -v
# Should show VERSION 2 for your distro
```

If you see VERSION 1, upgrade:
```powershell
wsl --set-version <distro-name> 2
```

### Step 2: Install NVIDIA Windows Driver

1. Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Choose **Game Ready Driver** or **Studio Driver** (both support WSL)
3. Install and reboot

⚠️ **Important**: Do NOT install Linux NVIDIA drivers inside WSL. The Windows driver provides GPU support automatically.

### Step 3: Verify GPU Access in WSL

```bash
# Inside your WSL2 distro
nvidia-smi
```

You should see your GPU listed. If not:
- Ensure you have a recent Windows NVIDIA driver (version 470+ for WSL support)
- Check that `/usr/lib/wsl/lib` exists and contains `libcuda.so.1`

### Step 4: (Optional) Install CUDA Toolkit

The CUDA runtime is provided by the Windows driver, but if you need to compile CUDA code locally:

```bash
# Ubuntu/Debian in WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

# Or for a minimal install (just nvcc compiler)
sudo apt-get -y install nvidia-cuda-toolkit
```

### Step 5: Fix Library Path (if needed)

If you get `libcuda.so.1 not found` errors:

```bash
# Add WSL library path
echo "/usr/lib/wsl/lib" | sudo tee /etc/ld.so.conf.d/10-wsl.conf
sudo ldconfig
```

### Docker in WSL2

For GPU-enabled Docker containers:

1. Install Docker Desktop with WSL2 backend
2. Enable WSL integration for your distro in Docker Desktop settings
3. GPU support works automatically:

```bash
export DOCKER_BUILDKIT=1
docker compose --profile gpu up --build bitnet-gpu
```

## Native Linux Setup

### NVIDIA GPUs (CUDA)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit

# Fedora/RHEL
sudo dnf install -y nvidia-driver cuda

# Arch
sudo pacman -S nvidia cuda

# Set environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Verify
nvidia-smi
nvcc --version
cargo xtask gpu-preflight
```

### AMD GPUs (ROCm)

```bash
# Ubuntu 22.04
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt-get install -y ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=rocm

# Set environment
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH

# Verify
rocm-smi
cargo xtask gpu-preflight
```

## macOS Metal Setup

Metal support is built-in on macOS with Apple Silicon or AMD GPUs:

```bash
# No installation needed, just verify
cargo xtask gpu-preflight
# Should show "Metal" as available
```

## Verification

### Quick Test

```bash
# From the BitNet-rs repository root
# GPU detection
cargo xtask gpu-preflight

# Should output something like:
# 🔍 GPU Preflight Check
# ═════════════════════
#
# Available GPU backends: CUDA 12.3
```

### Build with GPU Support

```bash
# Build with GPU features
cargo build --locked --workspace --no-default-features --features gpu

# Run GPU smoke tests
cargo xtask gpu-smoke

# Or use Make shortcuts
make gpu        # Preflight check
make gpu-smoke  # Smoke tests
```

### Docker GPU Test

```bash
# Ensure Docker Desktop has WSL integration enabled (for WSL users)
export DOCKER_BUILDKIT=1

# Build and run GPU-enabled container
docker compose --profile gpu up --build bitnet-gpu

# Check logs for GPU initialization
docker logs bitnet-gpu | grep -i gpu
```

## Testing

### GPU Smoke Test

Tests basic GPU functionality with CPU parity checking:

```bash
# Run with default settings
cargo xtask gpu-smoke

# Run with custom size and tolerance
GPU_TEST_SIZE=medium GPU_TEST_TOLERANCE=0.95 cargo xtask gpu-smoke
```

Test sizes:
- `tiny` - 16x16 matrices (default, fastest)
- `small` - 64x64 matrices
- `medium` - 256x256 matrices

### Parity Testing

Compares GPU and CPU results to ensure correctness:

```bash
cargo test --no-default-features --package bitnet-kernels --test gpu_smoke --features gpu
```

## CI/CD Integration

The GPU workflow (`.github/workflows/gpu.yml`) provides:

1. **GPU Preflight** - Checks for GPU availability
2. **GPU Build** - Builds with GPU features
3. **GPU Smoke Tests** - Runs basic functionality tests
4. **GPU Parity Tests** - Verifies CPU-GPU consistency

### Self-Hosted Runners

GPU tests require self-hosted runners with GPU access. The workflow gracefully skips GPU tests when no GPU is available, allowing CPU-only testing to proceed.

To set up a self-hosted runner:

1. Go to Settings → Actions → Runners in your GitHub repository
2. Add a new self-hosted runner with the `gpu` label
3. Ensure CUDA/ROCm is installed on the runner machine

## Feature Flags

```toml
# Enable GPU support
[features]
gpu = ["bitnet-kernels/cuda", "bitnet-inference/gpu"]

# Or use specific backend
cuda = ["gpu"]  # Alias for backward compatibility
```

## Troubleshooting

### WSL2 Issues

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found | Update Windows NVIDIA driver (needs WSL support) |
| `libcuda.so.1` not found | Add `/usr/lib/wsl/lib` to ldconfig (see WSL2 Setup Step 5) |
| Docker can't access GPU | Enable WSL integration in Docker Desktop settings |
| CUDA version mismatch | Use the CUDA toolkit version matching your driver |
| Slow file I/O | Store code in Linux filesystem (`/home/...`), not Windows mounts (`/mnt/c/...`) |

### Native Linux Issues

| Problem | Solution |
|---------|----------|
| `nvidia-smi` shows error | Check if nouveau is blacklisted, reboot after driver install |
| CUDA not detected | Set `export CUDA_HOME=/usr/local/cuda` |
| ROCm not detected | Set `export ROCM_PATH=/opt/rocm` |
| Permission denied | Add user to `video` and `render` groups: `sudo usermod -aG video,render $USER` |

### General Issues

| Problem | Solution |
|---------|----------|
| Build fails with GPU features | Ensure CUDA/ROCm toolkit is installed |
| GPU tests timeout | Check GPU memory usage, close other GPU apps |
| Inconsistent results | Set `CUDA_VISIBLE_DEVICES=0` to use specific GPU |
| Out of memory | Reduce batch size or use smaller test size |

## Performance Notes

- GPU acceleration is most beneficial for large models and batch processing
- Small operations may be faster on CPU due to transfer overhead
- Use the smoke test to verify speedup for your specific use case

## Development Workflow

1. **Check GPU availability**: `cargo xtask gpu-preflight`
2. **Build with GPU**: `cargo gpu-build`
3. **Run tests**: `cargo gpu-tests`
4. **Verify parity**: `cargo gpu-parity`
5. **Run benchmarks**: `cargo bench --features gpu`

## Environment Variables

- `CUDA_HOME` - Path to CUDA installation
- `ROCM_PATH` - Path to ROCm installation
- `GPU_TEST_SIZE` - Test matrix size (tiny/small/medium)
- `GPU_TEST_TOLERANCE` - Parity test tolerance (0.0-1.0)
- `BITNET_DETERMINISTIC` - Enable deterministic GPU operations
- `BITNET_SEED` - Set seed for reproducible GPU runs
