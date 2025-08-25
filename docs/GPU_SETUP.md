# GPU Setup Guide

This guide explains how to use the GPU infrastructure in BitNet-rs.

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
- **WebGPU** - Universal fallback (always available)

## Setup Instructions

### NVIDIA GPUs (CUDA)

1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
2. Set environment variable: `export CUDA_HOME=/usr/local/cuda`
3. Verify: `cargo xtask gpu-preflight`

### AMD GPUs (ROCm)

1. Install ROCm: https://rocm.docs.amd.com
2. Set environment variable: `export ROCM_PATH=/opt/rocm`
3. Verify: `cargo xtask gpu-preflight`

### Apple Silicon (Metal)

Metal support is built-in on macOS. No additional setup required.

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
cargo test --package bitnet-kernels --test gpu_smoke --features cuda
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

### No GPU Detected

If `cargo xtask gpu-preflight` doesn't detect your GPU:

1. Check driver installation: `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)
2. Verify environment variables: `CUDA_HOME` or `ROCM_PATH`
3. Ensure the GPU toolkit is in your PATH

### Build Errors

If you get linker errors when building with GPU features:

1. Ensure CUDA/ROCm toolkit is properly installed
2. Check that `nvcc` (CUDA) or `hipcc` (ROCm) is in PATH
3. Try building with CPU features only: `cargo build --features cpu`

### Test Failures

If GPU tests fail:

1. Check the error message for specific issues
2. Try running with higher tolerance: `GPU_TEST_TOLERANCE=0.90`
3. Verify GPU memory is available: `nvidia-smi` or `rocm-smi`

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