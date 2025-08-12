# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Core Development Commands
```bash
# Build the project with default features (CPU only)
cargo build --release

# Build with GPU support
cargo build --release --features gpu

# Build with all optimizations
cargo build --release --features full

# Run all tests (fast, Rust-only)
cargo test --workspace --features cpu

# Run tests for specific crate
cargo test -p bitnet-kernels --features cpu

# Run benchmarks
cargo bench --workspace --features cpu

# Cross-validation testing (requires C++ dependencies)
cargo test --workspace --features crossval
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
```bash
# Download BitNet GGUF model from Hugging Face (supports resumable downloads)
cargo xtask download-model  # Downloads default model
cargo xtask download-model --id microsoft/bitnet-b1.58-2B-4T-gguf --file ggml-model-i2_s.gguf
cargo xtask download-model --sha256 <hex>  # Verify download with SHA256
# Supports HF_TOKEN for private repos: HF_TOKEN=xxx cargo xtask download-model

# Fetch and build Microsoft BitNet C++ (for cross-validation)
cargo xtask fetch-cpp  # Uses default tag b1-65-ggml
cargo xtask fetch-cpp --tag b1-65-ggml --force --clean

# Run deterministic cross-validation tests
cargo xtask crossval  # Auto-discovers model, or downloads if missing
cargo xtask crossval --model path/to/specific.gguf  # Use specific model
cargo xtask crossval --dry-run  # Print env and command without running

# Full cross-validation workflow (download + fetch + test) - ONE COMMAND!
cargo xtask full-crossval  # Runs all three steps
cargo xtask full-crossval --force  # Force redownload/rebuild

# Generate test fixtures
cargo xtask gen-fixtures --size small --output crossval/fixtures/

# Setup cross-validation environment
cargo xtask setup-crossval

# Clean all caches
cargo xtask clean-cache

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
- **Feature Flags**: Default features include only CPU support. GPU and cross-validation features must be explicitly enabled.
- **Cross-Validation**: The `crossval` feature downloads and builds the C++ implementation, significantly increasing build time. It's disabled by default.
- **Binary Distribution**: Release builds use LTO, single codegen unit, and strip symbols for minimal binary size.