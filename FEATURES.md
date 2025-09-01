# BitNet.rs Feature Flags Documentation

This document provides comprehensive documentation for all feature flags available in the BitNet.rs project.

## Overview

BitNet.rs uses Cargo feature flags to control compilation of optional functionality. **Default features are intentionally empty** to prevent unwanted dependencies and allow precise control over the build.

## Quick Reference

```bash
# CPU-only build (most common)
cargo build --no-default-features --features cpu

# GPU-enabled build
cargo build --no-default-features --features cuda

# Full feature set
cargo build --features full
```

## Core Features

### `cpu`
**Purpose:** Enable CPU inference with optimized kernels  
**Dependencies:** None (pure Rust)  
**When to use:** For CPU-only deployments without GPU requirements

```bash
cargo build --no-default-features --features cpu
```

Enables:
- Fallback scalar kernels
- Platform-specific SIMD optimizations (if available)
- Multi-threaded CPU inference
- Automatic CPU feature detection

### `gpu`
**Purpose:** Enable advanced GPU acceleration with device-aware quantization  
**Dependencies:** CUDA toolkit 11.0+, cudarc crate  
**When to use:** For GPU-accelerated inference with automatic CPU fallback

```bash
cargo build --no-default-features --features gpu
```

Enables:
- Device-aware quantization with automatic GPU detection
- Comprehensive GPU quantization kernels (I2S, TL1, TL2)
- Intelligent fallback to optimized CPU kernels
- CUDA kernel compilation with bit-packing optimizations
- GPU memory management with automatic cleanup
- Multi-GPU device selection and management
- Thread-safe concurrent GPU operations
- Performance monitoring and device statistics

Requirements:
- NVIDIA GPU with compute capability 6.0+
- CUDA toolkit installed and `nvcc` in PATH
- Set `LD_LIBRARY_PATH` to include CUDA libraries

### `cuda`
**Purpose:** Backward-compatible alias for `gpu` feature  
**Dependencies:** Same as `gpu` feature  
**When to use:** For backward compatibility with existing build scripts

```bash
cargo build --no-default-features --features cuda
```

Note: This is an alias for the `gpu` feature. New projects should use `gpu` for clarity.

### `ffi`
**Purpose:** Enable C++ FFI bridge for cross-validation  
**Dependencies:** cc, bindgen, C++ compiler  
**When to use:** For comparing against original C++ implementation

```bash
cargo build --features ffi
```

Enables:
- FFI bindings to BitNet C++ implementation
- Cross-validation testing capabilities
- Performance comparison tools

Note: Requires building or linking the C++ BitNet library.

## SIMD Features

### `avx2`
**Purpose:** Enable x86_64 AVX2 SIMD optimizations  
**Dependencies:** x86_64 CPU with AVX2 support  
**When to use:** Automatically detected on compatible CPUs

### `avx512`
**Purpose:** Enable x86_64 AVX-512 SIMD optimizations  
**Dependencies:** x86_64 CPU with AVX-512F support  
**When to use:** For Intel server/workstation CPUs

### `neon`
**Purpose:** Enable ARM NEON SIMD optimizations  
**Dependencies:** ARM64/AArch64 CPU  
**When to use:** For ARM-based systems (Apple Silicon, ARM servers)

## Development Features

### `crossval`
**Purpose:** Enable cross-validation against C++ implementation  
**Dependencies:** `ffi` feature, C++ BitNet implementation  
**When to use:** For validating Rust implementation correctness

```bash
cargo test --features "cpu,ffi,crossval"
```

### `full`
**Purpose:** Enable all features for maximum functionality  
**Dependencies:** All of the above  
**When to use:** For development and testing

```bash
cargo build --features full
```

## Feature Combinations

### Common Configurations

**Production CPU deployment:**
```bash
cargo build --release --no-default-features --features cpu
```

**Production GPU deployment with device-aware quantization:**
```bash
cargo build --release --no-default-features --features "cpu,gpu"
```

**Development with GPU validation:**
```bash
cargo build --features "cpu,gpu,ffi,crossval"
```

**Maximum performance with GPU acceleration:**
```bash
cargo build --release --features "cpu,gpu,avx512"
```

## Workspace Crate Features

### bitnet-kernels
- `cpu`: CPU kernel implementations with SIMD optimizations
- `gpu`: Advanced GPU kernels with device-aware quantization
- `cuda`: Backward-compatible alias for `gpu`
- `ffi`: FFI bridge to C++ kernels
- `ffi-bridge`: Build-time FFI compilation

### bitnet-inference
- `cpu`: CPU inference engine with optimized kernels
- `gpu`: GPU inference engine with device-aware quantization
- `cuda`: Backward-compatible alias for `gpu`
- `async`: Async/await support (always enabled)

### bitnet-cli
- Inherits features from dependencies
- No crate-specific features

### bitnet-server
- `cpu`: CPU inference endpoints
- `gpu`: GPU inference endpoints with device-aware quantization
- `cuda`: Backward-compatible alias for `gpu`

## Feature Resolution

The workspace uses Cargo's resolver version 2 to prevent feature unification across dependencies:

```toml
[workspace]
resolver = "2"
```

This ensures:
- Dev-dependencies don't affect production builds
- Features are resolved per-package
- No unexpected feature activation

## Troubleshooting

### Issue: Build fails with undefined references
**Cause:** FFI enabled without C++ library  
**Solution:** Either disable FFI or build the C++ library:
```bash
# Option 1: Disable FFI
cargo build --no-default-features --features cpu

# Option 2: Build C++ library
cargo xtask fetch-cpp
cargo build --features "cpu,ffi"
```

### Issue: CUDA features not working
**Cause:** CUDA toolkit not installed or not in PATH  
**Solution:** Install CUDA toolkit and set environment:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: Features seem to be ignored
**Cause:** Default features interfering  
**Solution:** Always use `--no-default-features`:
```bash
cargo build --no-default-features --features cpu
```

### Issue: Unexpected dependencies pulled in
**Cause:** Feature unification in workspace  
**Solution:** Ensure `resolver = "2"` in workspace Cargo.toml

## Platform Support Matrix

| Platform | CPU | GPU/CUDA | AVX2 | AVX512 | NEON | FFI | Device-Aware |
|----------|-----|----------|------|--------|------|-----|---------------|
| Linux x86_64 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Linux ARM64 | ✅ | ✅* | ❌ | ❌ | ✅ | ✅ | ✅ |
| macOS x86_64 | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅** |
| macOS ARM64 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅** |
| Windows x86_64 | ✅ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ✅ |

✅ Supported  
❌ Not supported  
⚠️ Experimental  
\* CUDA on ARM64 requires NVIDIA Jetson or similar  
\*\* Device-aware on macOS falls back to optimized CPU kernels

## Device-Aware Quantization Features

The `gpu` feature enables advanced device-aware quantization with the following capabilities:

### Supported Quantization Algorithms
- **I2S (2-bit Signed)**: Native GPU kernels with optimized bit-packing
- **TL1 (Table Lookup 1)**: GPU-accelerated table lookup quantization
- **TL2 (Table Lookup 2)**: Advanced table lookup with GPU optimization

### Device Selection and Fallback
- **Automatic Device Detection**: Scans for available CUDA devices at runtime
- **Graceful Fallback**: Seamlessly falls back to optimized CPU kernels on GPU failure
- **Device Affinity**: Maintains consistent device usage throughout quantization pipeline
- **Error Recovery**: Comprehensive error handling with detailed failure reporting

### Performance Optimizations
- **CUDA Kernel Integration**: Custom CUDA kernels with atomic operations
- **Memory Management**: Automatic GPU memory allocation and cleanup
- **Concurrent Operations**: Thread-safe GPU operations with proper synchronization
- **Performance Monitoring**: Built-in statistics and device usage tracking

### Testing and Validation
```bash
# Test device-aware quantization
cargo test -p bitnet-kernels --features gpu --test gpu_quantization

# Test GPU vs CPU accuracy
cargo test -p bitnet-kernels --features gpu test_gpu_vs_cpu_quantization_accuracy --ignored

# Test automatic fallback
cargo test -p bitnet-kernels --features gpu test_gpu_quantization_fallback --ignored
```

## Performance Impact

Enabling features affects performance and binary size:

| Feature | Performance Impact | Binary Size Impact | Notes |
|---------|-------------------|--------------------|-------|
| `cpu` | Baseline | +2 MB | Optimized SIMD kernels |
| `gpu` | 2-15x faster | +12 MB | Device-aware quantization |
| `cuda` | Same as `gpu` | Same as `gpu` | Backward-compatible alias |
| `avx2` | 1.5-2x faster | +0.5 MB | x86_64 SIMD optimization |
| `avx512` | 2-3x faster | +1 MB | High-end Intel CPUs |
| `neon` | 1.5-2x faster | +0.5 MB | ARM64 SIMD optimization |
| `ffi` | No impact | +4 MB | Cross-validation only |

## Best Practices

1. **Production builds:** Use minimal features needed
2. **Development:** Use `full` for comprehensive testing
3. **CI/CD:** Test with multiple feature combinations
4. **Benchmarking:** Compare with and without SIMD features
5. **Deployment:** Document required features in README

## Future Features

Planned features for future releases:

- `rocm`: AMD GPU support via ROCm
- `metal`: Apple Metal GPU support
- `vulkan`: Cross-platform GPU via Vulkan
- `wgpu`: WebGPU support for browsers
- `onnx`: ONNX model format support
- `tflite`: TensorFlow Lite support

## Contributing

When adding new features:

1. Update this document
2. Add feature to relevant Cargo.toml files
3. Gate code with `#[cfg(feature = "name")]`
4. Add tests for feature-gated code
5. Update CI to test new feature