# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Core Development Commands
```bash
# Build the project with CPU support (no default features)
cargo build --release --no-default-features --features cpu

# Build with GPU/CUDA support
cargo build --release --no-default-features --features cuda

# Build with both CPU and GPU
cargo build --release --no-default-features --features "cpu,cuda"

# Build with all optimizations
cargo build --release --features full

# Run all tests (fast, Rust-only)
cargo test --workspace --no-default-features --features cpu

# Run GPU tests
cargo test -p bitnet-kernels --no-default-features --features cuda

# Run tests for specific crate
cargo test -p bitnet-kernels --no-default-features --features cpu

# Run benchmarks
cargo bench --workspace --no-default-features --features cpu

# Cross-validation testing (requires C++ dependencies and FFI)
cargo test --workspace --features "cpu,ffi,crossval"

# Test CUDA functionality
cargo run -p bitnet-kernels --example cuda_smoke --no-default-features --features cuda
```

### Code Quality Commands
```bash
# Format all code
cargo fmt --all

# Run clippy with pedantic lints
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit

# Check for outdated dependencies
cargo outdated

# Find unused dependencies
cargo machete

# Generate code coverage
cargo llvm-cov --workspace --features cpu --html
```

### Development Tools (xtask)

#### Download Model - Production-Ready with Advanced Features
```bash
# Download BitNet GGUF model from Hugging Face
cargo xtask download-model  # Downloads default model

# Advanced options
cargo xtask download-model \
  --id microsoft/bitnet-b1.58-2B-4T-gguf \
  --file ggml-model-i2_s.gguf \
  --sha256 <hex> \               # Verify with SHA256
  --rev main \                   # Pin to branch/tag/commit
  --no-progress \                # Disable progress bar (CI-friendly)
  --verbose \                    # Debug output
  --base-url https://mirror.com # Use alternative repository

# Features:
# ✅ Resumable downloads with smart Content-Range validation
# ✅ 429 rate limiting with Retry-After header support
# ✅ 304 Not Modified optimization via ETag/Last-Modified
# ✅ Concurrent download protection (file locking)
# ✅ Disk space validation with helpful errors
# ✅ Path traversal protection
# ✅ Atomic writes with fsync guarantees
# ✅ Ctrl-C graceful handling with resume support
# ✅ Automatic proxy support (respects HTTP[S]_PROXY)

# Private repos: HF_TOKEN=xxx cargo xtask download-model

```

#### Other xtask Commands
```bash
# Fetch and build Microsoft BitNet C++ (for cross-validation)
# Now uses official Microsoft BitNet repository!
cargo xtask fetch-cpp  # Fetches from github.com/microsoft/BitNet.git (main branch)
cargo xtask fetch-cpp --tag main --force --clean
cargo xtask fetch-cpp --tag <commit-hash>  # Pin to specific commit for reproducible builds
# ✅ Validates build artifacts and adapts to repository structure changes

# Run deterministic cross-validation tests
cargo xtask crossval  # Auto-discovers model in models/ directory
cargo xtask crossval --model path/to/specific.gguf  # Use specific model
cargo xtask crossval --dry-run  # Print env and command without running

# Full cross-validation workflow (download + fetch + test) - ONE COMMAND!
cargo xtask full-crossval  # Runs all three steps sequentially
cargo xtask full-crossval --force  # Force redownload/rebuild
# ✅ Fully integrated with official Microsoft BitNet repository

# Generate test fixtures
cargo xtask gen-fixtures --size tiny --output test-fixtures/
cargo xtask gen-fixtures --size small --output test-fixtures/
cargo xtask gen-fixtures --size medium --output test-fixtures/
# ✅ Now generates realistic GGUF-like metadata and weight files

# Setup cross-validation environment
cargo xtask setup-crossval

# Clean all caches (interactive with size reporting)
cargo xtask clean-cache
# ✅ Shows directory sizes and asks for confirmation
# ✅ Cleans: target/, ~/.cache/bitnet_cpp/, crossval/fixtures/, models/

# Check feature flag consistency
cargo xtask check-features

# Run performance benchmarks
cargo xtask benchmark --platform current
```

### Useful Cargo Aliases (defined in .cargo/config.toml)
```bash
cargo check-all    # Check all code
cargo test-all     # Run all tests
cargo quality      # Run quality checks
cargo coverage     # Generate coverage report
cargo crossval     # Run cross-validation tests
```

## High-Level Architecture

### Workspace Structure
BitNet.rs is organized as a Rust workspace with 12 specialized crates, each serving a specific purpose:

#### Core Library Architecture
- **`bitnet`** (root): Main library crate providing the unified public API. Re-exports functionality from other crates based on feature flags.
- **`bitnet-common`**: Foundation layer with shared types (`BitNetConfig`, `BitNetError`, `Device`), traits, and utilities used across all crates.
- **`bitnet-models`**: Model loading, format handling (GGUF, SafeTensors, HuggingFace), and model definitions.
- **`bitnet-quantization`**: 1-bit quantization algorithms (I2_S, TL1, TL2) and dequantization logic.
- **`bitnet-kernels`**: High-performance compute kernels with SIMD optimizations (AVX2/AVX-512/NEON) and CUDA support.
- **`bitnet-inference`**: High-level inference engine that orchestrates model execution, generation, and streaming.
- **`bitnet-tokenizers`**: Text tokenization and processing, supporting various tokenizer formats.

#### Application Layer
- **`bitnet-cli`**: Command-line interface for inference, conversion, and benchmarking.
- **`bitnet-server`**: HTTP inference server with REST API endpoints.

#### Language Bindings
- **`bitnet-ffi`**: C API for cross-language interoperability.
- **`bitnet-py`**: Python bindings via PyO3.
- **`bitnet-wasm`**: WebAssembly bindings for browser deployment.

#### Cross-Validation
- **`bitnet-sys`**: FFI bindings to original C++ implementation for comparison.
- **`crossval`**: Cross-validation framework for testing against C++ implementation.

### Key Design Patterns

1. **Feature-Gated Architecture**: The root `bitnet` crate uses feature flags to conditionally include functionality, allowing users to minimize binary size and dependencies.

2. **Zero-Copy Operations**: Models are memory-mapped when possible, and tensor operations avoid unnecessary allocations through careful lifetime management.

3. **SIMD Abstraction**: The kernels crate provides a unified interface over platform-specific SIMD instructions, with runtime CPU feature detection.

4. **Async/Await**: The inference engine uses Tokio for async operations, enabling efficient batch processing and streaming generation.

5. **Cross-Validation**: The `crossval` crate allows systematic comparison with the original C++ implementation to ensure correctness and measure performance improvements.

### Performance Optimizations

- **Compile-time optimizations**: Heavy use of const generics and inline hints
- **Memory layout**: Cache-friendly data structures with aligned allocations
- **Vectorization**: Hand-tuned SIMD kernels for critical operations
- **Parallelization**: Rayon for data parallelism, Tokio for task parallelism

### Testing Strategy

- **Unit tests**: Each crate has comprehensive unit tests
- **Integration tests**: The `tests/` directory contains cross-crate integration tests  
- **Benchmarks**: Criterion benchmarks for performance-critical code
- **Cross-validation**: Automated testing against C++ implementation
- **Property-based testing**: Using proptest for edge cases

## Important Considerations

- **MSRV**: Minimum Supported Rust Version is 1.70.0
- **Feature Flags**: Default features are **empty** to prevent unwanted dependencies. You must explicitly enable features:
  - `cpu`: CPU inference with SIMD optimizations
  - `cuda`: NVIDIA GPU support
  - `ffi`: C++ FFI bridge (required for cross-validation)
  - `crossval`: Cross-validation against C++ implementation (requires `ffi`)
- **Cross-Validation**: The `crossval` feature downloads and builds the C++ implementation, significantly increasing build time. It's disabled by default.
- **Binary Distribution**: Release builds use LTO, single codegen unit, and strip symbols for minimal binary size.
- **CUDA Requirements**: CUDA toolkit 11.0+ and appropriate GPU drivers. Set `LD_LIBRARY_PATH` to include CUDA libraries.

## Troubleshooting Guide

### Common Build Issues

1. **FFI Linker Errors**: If you see `undefined reference to bitnet_cpp_*`, either disable FFI (`--no-default-features --features cpu`) or build the C++ library (`cargo xtask fetch-cpp`).

2. **CUDA Compilation Errors**: Ensure CUDA toolkit is installed and `nvcc` is in PATH. The kernel code uses `signed char`/`unsigned char` instead of `int8_t`/`uint8_t` for compatibility.

3. **Empty Default Features**: This is intentional to prevent accidental dependencies. Always specify features explicitly.

4. **Feature Name Changes**: Use `cuda` instead of `gpu` for GPU support.