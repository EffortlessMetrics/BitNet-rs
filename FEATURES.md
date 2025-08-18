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

### `cuda`
**Purpose:** Enable NVIDIA GPU acceleration  
**Dependencies:** CUDA toolkit 11.0+, cudarc crate  
**When to use:** For GPU-accelerated inference on NVIDIA hardware

```bash
cargo build --no-default-features --features cuda
```

Enables:
- CUDA kernel compilation and execution
- GPU memory management
- Mixed-precision computation (FP16/BF16)
- Multi-GPU support
- Async GPU operations

Requirements:
- NVIDIA GPU with compute capability 6.0+
- CUDA toolkit installed
- Set `LD_LIBRARY_PATH` to include CUDA libraries

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

**Production GPU deployment:**
```bash
cargo build --release --no-default-features --features "cpu,cuda"
```

**Development with validation:**
```bash
cargo build --features "cpu,cuda,ffi,crossval"
```

**Maximum performance:**
```bash
cargo build --release --features "cpu,cuda,avx512"
```

## Workspace Crate Features

### bitnet-kernels
- `cpu`: CPU kernel implementations
- `cuda`: CUDA GPU kernels
- `ffi`: FFI bridge to C++ kernels
- `ffi-bridge`: Build-time FFI compilation

### bitnet-inference
- `cpu`: CPU inference engine
- `cuda`: GPU inference engine
- `async`: Async/await support (always enabled)

### bitnet-cli
- Inherits features from dependencies
- No crate-specific features

### bitnet-server
- `cpu`: CPU inference endpoints
- `cuda`: GPU inference endpoints

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

| Platform | CPU | CUDA | AVX2 | AVX512 | NEON | FFI |
|----------|-----|------|------|--------|------|-----|
| Linux x86_64 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Linux ARM64 | ✅ | ✅* | ❌ | ❌ | ✅ | ✅ |
| macOS x86_64 | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| macOS ARM64 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Windows x86_64 | ✅ | ✅ | ✅ | ✅ | ❌ | ⚠️ |

✅ Supported  
❌ Not supported  
⚠️ Experimental  
\* CUDA on ARM64 requires NVIDIA Jetson or similar

## Performance Impact

Enabling features affects performance and binary size:

| Feature | Performance Impact | Binary Size Impact |
|---------|-------------------|-------------------|
| `cpu` | Baseline | +2 MB |
| `cuda` | 2-10x faster | +8 MB |
| `avx2` | 1.5-2x faster | +0.5 MB |
| `avx512` | 2-3x faster | +1 MB |
| `neon` | 1.5-2x faster | +0.5 MB |
| `ffi` | No impact | +4 MB |

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