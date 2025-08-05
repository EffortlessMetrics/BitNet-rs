# BitNet.rs Feature Flags

This document describes the feature flags available in BitNet.rs and their usage.

## Overview

BitNet.rs uses Cargo feature flags to enable optional functionality and optimize builds for specific use cases. This allows users to include only the components they need, reducing compilation time and binary size.

## Core Features

### `default = ["cpu"]`

The default feature set includes CPU inference with optimized kernels. This provides a good balance of functionality and performance for most use cases.

### `cpu`

Enables CPU-based inference with optimized SIMD kernels.

**Includes:**
- `kernels`: High-performance compute kernels
- `inference`: CPU inference engine
- `tokenizers`: Tokenization support
- `bitnet-kernels/cpu-optimized`: Optimized CPU kernels

**Example:**
```toml
[dependencies]
bitnet = { version = "0.1", features = ["cpu"] }
```

### `gpu`

Enables GPU acceleration via CUDA.

**Includes:**
- `kernels`: High-performance compute kernels
- `inference`: GPU inference engine
- `tokenizers`: Tokenization support
- `bitnet-kernels/cuda`: CUDA GPU kernels
- `bitnet-inference/gpu`: GPU inference support

**Requirements:**
- CUDA Toolkit 11.0 or later
- Compatible NVIDIA GPU

**Example:**
```toml
[dependencies]
bitnet = { version = "0.1", features = ["gpu"] }
```

## Component Features

### `kernels`

Enables the high-performance compute kernel system. This is automatically included with `cpu` and `gpu` features.

**Provides:**
- Kernel abstraction layer
- Runtime kernel selection
- Fallback implementations

### `inference`

Enables the inference engine system. Requires `kernels`.

**Provides:**
- CPU and GPU inference engines
- Streaming generation
- Batch processing
- Sampling strategies

### `tokenizers`

Enables tokenization support for text processing.

**Provides:**
- GPT-2 tokenizer
- SentencePiece tokenizer
- HuggingFace tokenizer integration

## SIMD Optimization Features

### `avx2`

Enables AVX2 SIMD optimizations for x86_64 processors.

**Requirements:**
- x86_64 processor with AVX2 support
- Automatically detected at runtime

### `avx512`

Enables AVX-512 SIMD optimizations for x86_64 processors.

**Requirements:**
- x86_64 processor with AVX-512 support
- Automatically detected at runtime

### `neon`

Enables ARM NEON SIMD optimizations for ARM64 processors.

**Requirements:**
- ARM64 processor with NEON support
- Automatically detected at runtime

## Language Binding Features

These features are informational only - the actual bindings are provided by separate crates.

### `python`

Indicates Python binding support. Use the `bitnet-py` crate for actual Python bindings.

### `wasm`

Indicates WebAssembly support. Use the `bitnet-wasm` crate for WebAssembly bindings.

### `ffi`

Indicates C API support. Use the `bitnet-ffi` crate for C API bindings.

## Application Features

These features are informational only - the actual applications are provided by separate crates.

### `server`

Indicates HTTP server support. Use the `bitnet-server` crate for the HTTP server.

### `cli`

Indicates CLI support. Use the `bitnet-cli` crate for the command-line interface.

## Convenience Features

### `full`

Enables all available features for maximum functionality.

**Includes:**
- `cpu`: CPU inference
- `gpu`: GPU inference
- `avx2`: AVX2 optimizations
- `avx512`: AVX-512 optimizations
- `neon`: ARM NEON optimizations

**Example:**
```toml
[dependencies]
bitnet = { version = "0.1", features = ["full"] }
```

### `minimal`

Provides only the core functionality without inference capabilities.

**Includes:**
- Core types and utilities
- Model loading and definitions
- Quantization algorithms

**Use case:** When you only need model loading or quantization without inference.

**Example:**
```toml
[dependencies]
bitnet = { version = "0.1", features = ["minimal"] }
```

## Feature Combinations

### Common Combinations

#### CPU-only with SIMD optimizations
```toml
[dependencies]
bitnet = { version = "0.1", features = ["cpu", "avx2", "neon"] }
```

#### GPU with all optimizations
```toml
[dependencies]
bitnet = { version = "0.1", features = ["gpu", "avx2", "avx512", "neon"] }
```

#### Model loading only
```toml
[dependencies]
bitnet = { version = "0.1", features = ["minimal"] }
```

### Platform-Specific Recommendations

#### x86_64 Linux/Windows
```toml
[dependencies]
bitnet = { version = "0.1", features = ["cpu", "avx2"] }
# Add "gpu" if CUDA is available
# Add "avx512" for newer processors
```

#### ARM64 (Apple Silicon, ARM servers)
```toml
[dependencies]
bitnet = { version = "0.1", features = ["cpu", "neon"] }
```

#### WebAssembly
Use the `bitnet-wasm` crate instead:
```toml
[dependencies]
bitnet-wasm = { version = "0.1" }
```

## Build Optimization

### Release Builds

For optimal performance in release builds, consider:

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
strip = true
```

### Target-Specific Builds

For maximum performance on known hardware:

```bash
# x86_64 with AVX2
RUSTFLAGS="-C target-cpu=haswell" cargo build --release --features="cpu,avx2"

# x86_64 with AVX-512
RUSTFLAGS="-C target-cpu=skylake-avx512" cargo build --release --features="cpu,avx512"

# ARM64 with NEON
RUSTFLAGS="-C target-cpu=native" cargo build --release --features="cpu,neon"
```

## Troubleshooting

### CUDA Not Found

If you get CUDA-related errors with the `gpu` feature:

1. Ensure CUDA Toolkit is installed
2. Check that `nvcc` is in your PATH
3. Verify your GPU is CUDA-compatible
4. Consider using CPU-only features as fallback

### SIMD Instructions Not Available

If SIMD optimizations don't work:

1. Check your processor capabilities: `cat /proc/cpuinfo` (Linux) or `sysctl -n machdep.cpu.features` (macOS)
2. The library will automatically fall back to non-SIMD implementations
3. Use `minimal` feature for basic functionality

### Compilation Errors

If you encounter compilation errors:

1. Ensure you're using Rust 1.70.0 or later
2. Try building with fewer features enabled
3. Check that all required system dependencies are installed
4. Use `cargo clean` to clear build cache

## Performance Impact

| Feature | Binary Size Impact | Compilation Time | Runtime Performance |
|---------|-------------------|------------------|-------------------|
| `minimal` | Smallest | Fastest | Basic |
| `cpu` | Medium | Medium | Good |
| `cpu,avx2` | Medium+ | Medium+ | Better |
| `cpu,avx512` | Medium+ | Medium+ | Best (x86_64) |
| `cpu,neon` | Medium+ | Medium+ | Best (ARM64) |
| `gpu` | Large | Slow | Excellent |
| `full` | Largest | Slowest | Maximum |

Choose features based on your specific requirements for binary size, compilation time, and runtime performance.