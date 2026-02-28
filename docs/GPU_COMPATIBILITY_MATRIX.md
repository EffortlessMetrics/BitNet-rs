# GPU Hardware Compatibility Matrix

This document provides a comprehensive view of GPU hardware support,
feature availability per backend, and known issues. For setup instructions
see [GPU_SETUP.md](GPU_SETUP.md), [INTEL_GPU_SETUP.md](INTEL_GPU_SETUP.md),
and [cuda-configuration-guide.md](cuda-configuration-guide.md).

> **Current release**: v0.1.0-qna-mvp — CUDA is the primary GPU backend.
> OpenCL (Intel Arc) is in early integration. Other backends are planned.

## Hardware Compatibility

### NVIDIA GPUs (CUDA)

| GPU | Architecture | Compute Cap. | Status | Min Driver | VRAM | Notes |
|-----|-------------|-------------|--------|------------|------|-------|
| H100 | Hopper | 9.0 | ✅ Supported | CUDA 12.x | 80 GB | Tensor Core / BF16 / FP8 |
| A100 | Ampere | 8.0 | ✅ Supported | CUDA 11.x | 40/80 GB | BF16 Tensor Cores |
| RTX 4090 | Ada Lovelace | 8.9 | ✅ Supported | CUDA 12.x | 24 GB | Primary dev target |
| RTX 4080 | Ada Lovelace | 8.9 | ✅ Supported | CUDA 12.x | 16 GB | |
| RTX 4070 Ti | Ada Lovelace | 8.9 | ✅ Supported | CUDA 12.x | 12 GB | |
| RTX 4060 | Ada Lovelace | 8.9 | ✅ Supported | CUDA 12.x | 8 GB | Min VRAM for 2B models |
| RTX 3090 | Ampere | 8.6 | ✅ Supported | CUDA 11.x | 24 GB | BF16 support |
| RTX 3080 | Ampere | 8.6 | ✅ Supported | CUDA 11.x | 10/12 GB | |
| RTX 3070 | Ampere | 8.6 | ✅ Supported | CUDA 11.x | 8 GB | |
| RTX 3060 | Ampere | 8.6 | ✅ Supported | CUDA 11.x | 12 GB | |
| RTX 2080 Ti | Turing | 7.5 | ✅ Supported | CUDA 11.x | 11 GB | FP16 Tensor Cores |
| RTX 2070 | Turing | 7.5 | ✅ Supported | CUDA 11.x | 8 GB | |
| GTX 1080 Ti | Pascal | 6.1 | ✅ Supported | CUDA 11.x | 11 GB | FP16 only (no Tensor Cores) |
| GTX 1060 | Pascal | 6.1 | ⚠️ Minimum | CUDA 11.x | 6 GB | Limited by VRAM |

**Minimum requirement**: Compute Capability 6.0+ (Pascal architecture, 2016+).

### Intel GPUs (OpenCL)

| GPU | Architecture | EU Count | Status | Min Driver | VRAM | Notes |
|-----|-------------|----------|--------|------------|------|-------|
| Arc A770 | Alchemist (Xe-HPG) | 512 | ✅ Supported | Compute Runtime 23.x | 16 GB | Primary Intel target |
| Arc A750 | Alchemist (Xe-HPG) | 448 | ✅ Supported | Compute Runtime 23.x | 8 GB | |
| Arc A580 | Alchemist (Xe-HPG) | 384 | 🔄 Expected | Compute Runtime 23.x | 8 GB | Lower EU count |
| Arc A380 | Alchemist (Xe-HPG) | 128 | 🔄 Expected | Compute Runtime 23.x | 6 GB | Entry-level; fewer CUs |
| Arc B580 | Battlemage (Xe2-HPG) | 320 | 🔄 Expected | Compute Runtime 24.x | 12 GB | Xe2 architecture |

**Requires**: Linux kernel ≥ 6.2 with i915 driver, Intel Compute Runtime
(OpenCL 3.0 + Level Zero 1.3). See [INTEL_GPU_SETUP.md](INTEL_GPU_SETUP.md).

### AMD GPUs (ROCm) — Planned

| GPU | Architecture | Status | Target Release | Notes |
|-----|-------------|--------|---------------|-------|
| RX 7900 XTX | RDNA 3 | 🔮 Planned | v0.3.0 | ROCm 5.7+ required |
| RX 7900 XT | RDNA 3 | 🔮 Planned | v0.3.0 | |
| RX 7800 XT | RDNA 3 | 🔮 Planned | v0.3.0 | |
| MI250X | CDNA 2 | 🔮 Planned | v0.3.0 | Data-center GPU |
| MI300X | CDNA 3 | 🔮 Planned | v0.3.0 | |

ROCm/HIP kernel stubs exist in `crates/bitnet-kernels/src/rocm/` mirroring
the CUDA kernel structure. Runtime integration is not yet implemented.

### Apple Silicon (Metal) — Planned

| Chip | Status | Target Release | Notes |
|------|--------|---------------|-------|
| M1 / M1 Pro / M1 Max | 🔮 Planned | v0.3.0 | macOS 13+ (Ventura) |
| M2 / M2 Pro / M2 Max | 🔮 Planned | v0.3.0 | |
| M3 / M3 Pro / M3 Max | 🔮 Planned | v0.3.0 | |
| M4 / M4 Pro / M4 Max | 🔮 Planned | v0.3.0 | |

Metal GPU detection is already present via `system_profiler` in
`crates/bitnet-kernels/src/gpu_utils.rs`. Compute kernel implementation
is tracked in the roadmap at `docs/reference/macos-26-apple-silicon-roadmap.md`.

## Feature Support per Backend

| Feature | CUDA | OpenCL (Intel) | Vulkan | ROCm | Metal | WebGPU |
|---------|:----:|:--------------:|:------:|:----:|:-----:|:------:|
| MatMul (QK256 GEMV) | ✅ | ✅ | 🔮 | 🔮 | 🔮 | 🔮 |
| Attention (FlashAttn-style) | ✅ | ✅ | 🔮 | 🔮 | 🔮 | 🔮 |
| RMSNorm / LayerNorm | ✅ | ✅ | 🔮 | 🔮 | 🔮 | 🔮 |
| KV Cache management | ✅ | 🔄 | 🔮 | 🔮 | 🔮 | 🔮 |
| Mixed precision (FP16) | ✅ | 🔄 | 🔮 | 🔮 | 🔮 | 🔮 |
| Mixed precision (BF16) | ✅¹ | ❌ | 🔮 | 🔮 | 🔮 | 🔮 |
| Tensor Cores (WMMA) | ✅² | N/A | N/A | 🔮 | N/A | N/A |
| Batch processing | ✅ | 🔄 | 🔮 | 🔮 | 🔮 | 🔮 |
| Multi-GPU | 🔄 | 🔮 | 🔮 | 🔮 | 🔮 | 🔮 |
| Memory pool | ✅ | 🔄 | 🔮 | 🔮 | 🔮 | 🔮 |
| Device-aware launch params | ✅ | 🔄 | 🔮 | 🔮 | 🔮 | 🔮 |

**Legend**: ✅ Implemented — 🔄 In progress / partial — 🔮 Planned — ❌ Not applicable

¹ BF16 requires Compute Capability 8.0+ (Ampere and newer).
² Tensor Core WMMA requires Compute Capability 7.0+ (Volta/Turing and newer).

## Precision Mode Support by GPU Generation

| GPU Generation | FP32 | FP16 | BF16 | Tensor Cores | Auto Mode Selection |
|---------------|:----:|:----:|:----:|:------------:|:-------------------:|
| Pascal (CC 6.x) | ✅ | ✅ | ❌ | ❌ | FP16 |
| Volta/Turing (CC 7.x) | ✅ | ✅ | ❌ | ✅ (FP16) | FP16 + WMMA |
| Ampere (CC 8.x) | ✅ | ✅ | ✅ | ✅ (FP16/BF16) | BF16 |
| Ada Lovelace (CC 8.9) | ✅ | ✅ | ✅ | ✅ (FP16/BF16/FP8) | BF16 |
| Hopper (CC 9.0) | ✅ | ✅ | ✅ | ✅ (FP16/BF16/FP8) | BF16 |

BitNet-rs automatically selects the optimal precision mode for the
detected hardware. See `MixedPrecisionKernel::detect_best_precision()`
in `crates/bitnet-kernels/src/gpu/mixed_precision.rs`.

## Build Feature Flags

| Feature | What it enables |
|---------|----------------|
| `gpu` | Umbrella: enables `cuda` + `vulkan` + device-probe |
| `cuda` | CUDA backend via `cudarc 0.17.8` (CUDA 12.0 bindings) |
| `oneapi` | Intel OpenCL backend for Arc GPUs |
| `rocm` / `hip` | ROCm/HIP backend stubs |
| `vulkan` | Vulkan compute backend (planned) |
| `cpu` | CPU SIMD kernels (AVX2/AVX-512/NEON) |

```bash
# CUDA only
cargo build --no-default-features --features gpu

# Intel Arc only
cargo build --no-default-features --features oneapi

# CPU + CUDA
cargo build --no-default-features --features cpu,gpu

# CPU + Intel Arc
cargo build --no-default-features --features cpu,oneapi
```

Always use the unified cfg predicate for GPU-gated code:

```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
```

## Driver Installation Links

### NVIDIA

| Resource | URL |
|----------|-----|
| CUDA Toolkit 12.x | <https://developer.nvidia.com/cuda-downloads> |
| NVIDIA Drivers | <https://www.nvidia.com/Download/index.aspx> |
| cuDNN | <https://developer.nvidia.com/cudnn> |
| Container Toolkit | <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/> |

### Intel

| Resource | URL |
|----------|-----|
| Intel Compute Runtime | <https://github.com/intel/compute-runtime/releases> |
| Intel GPU Drivers (Linux) | <https://dgpu-docs.intel.com/driver/installation.html> |
| oneAPI Base Toolkit | <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html> |
| Level Zero | <https://github.com/oneapi-src/level-zero/releases> |

### AMD (future)

| Resource | URL |
|----------|-----|
| ROCm | <https://rocm.docs.amd.com/en/latest/deploy/linux/install.html> |
| ROCm Docker | <https://hub.docker.com/r/rocm/dev-ubuntu-22.04> |

## Known Issues and Workarounds

### CUDA

| Issue | Affected | Workaround |
|-------|----------|------------|
| Missing `-lcuda` at link time | All CUDA builds | Export `LD_LIBRARY_PATH=$CUDA_HOME/lib64` |
| No CUDA device at runtime | Containers | Pass `--gpus all` or use NVIDIA Container Toolkit |
| Flaky GPU tests in CI | CI environments | Serialize with `--test-threads=1`; split compile-only and runtime jobs |
| PTX compilation timeout | Older GPUs | Increase `CUDA_LAUNCH_TIMEOUT`; pre-compile PTX for target arch |

### Intel OpenCL

| Issue | Affected | Workaround |
|-------|----------|------------|
| "No OpenCL devices found" | Missing driver | Install `intel-opencl-icd`; add user to `render` + `video` groups |
| Kernel compilation failure | Driver mismatch | Update Intel Compute Runtime to latest release |
| Low performance | All Intel GPUs | Check PCIe link width with `lspci -vv`; ensure x16 connection |
| Linux kernel < 6.2 | Arc GPUs | Upgrade kernel; i915 Arc support requires ≥ 6.2 |

### General

| Issue | Affected | Workaround |
|-------|----------|------------|
| `BITNET_GPU_FAKE` ignored | Strict mode | Expected: `BITNET_STRICT_MODE=1` disables fake GPU detection |
| Silent CPU fallback | Misconfigured builds | Check `BackendStartupSummary` log: `requested=X detected=[…] selected=Y` |

## Runtime Device Detection

BitNet-rs probes GPU availability at startup with a 5-second timeout per
probe command. Detection results are cached via `OnceLock`. The
`BackendStartupSummary` emitted at `info` level shows:

```
Backend: requested=gpu detected=[cuda] selected=cuda
```

Use `BITNET_GPU_FAKE=cuda` (or `oneapi`) to override detection in tests.
This is blocked when `BITNET_STRICT_MODE=1`.

## Related Documentation

- [GPU_SETUP.md](GPU_SETUP.md) — CUDA 12.x setup guide
- [INTEL_GPU_SETUP.md](INTEL_GPU_SETUP.md) — Intel Arc OpenCL setup
- [cuda-configuration-guide.md](cuda-configuration-guide.md) — CUDA memory and tuning
- [gpu-kernel-architecture.md](gpu-kernel-architecture.md) — Kernel design decisions
- [GPU_PERFORMANCE_EXPECTATIONS.md](GPU_PERFORMANCE_EXPECTATIONS.md) — Throughput targets
- [performance-guide.md](performance-guide.md) — General performance optimization
