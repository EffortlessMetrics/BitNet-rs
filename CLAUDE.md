# CLAUDE.md

This file provides guidance to Claude (claude.ai) when working with the BitNet.rs codebase.

## Build and Development Commands

### Core Development Commands
```bash
# Build with CPU support (no default features)
cargo build --release --no-default-features --features cpu

# Build with GPU support and device-aware quantization
cargo build --release --no-default-features --features gpu

# Build with IQ2_S quantization support (requires GGML FFI)
cargo build --release --no-default-features --features "cpu,iq2s-ffi"

# Run tests (fast, Rust-only)
cargo test --workspace --no-default-features --features cpu

# Run GPU tests with device-aware quantization (requires CUDA)
cargo test --workspace --no-default-features --features gpu

# Run convolution kernel tests
cargo test -p bitnet-kernels --no-default-features --features cpu convolution

# Run PyTorch reference convolution tests (requires Python and PyTorch)
cargo test -p bitnet-kernels conv2d_reference_cases -- --ignored

# Run verification script
./scripts/verify-tests.sh

# Run benchmarks
cargo bench --workspace --no-default-features --features cpu

# Cross-validation testing (requires C++ dependencies)
cargo test --workspace --features "cpu,ffi,crossval"
```

### Code Quality Commands
```bash
# Format all code
cargo fmt --all

# Run clippy with pedantic lints
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit

# Run tests with concurrency caps (prevents resource storms)
scripts/preflight.sh && cargo t2                     # 2-thread CPU tests
scripts/preflight.sh && cargo crossval-capped        # Cross-validation with caps
scripts/e2e-gate.sh cargo test --features crossval   # Gate heavy E2E tests
```

### Development Tools (xtask)
```bash
# Download BitNet model from Hugging Face
cargo run -p xtask -- download-model

# Fetch and build Microsoft BitNet C++ for cross-validation
cargo run -p xtask -- fetch-cpp

# Full workflow (download + fetch + test)
cargo run -p xtask -- full-crossval

# Check feature flag consistency
cargo run -p xtask -- check-features
```

## High-Level Architecture

### Workspace Structure
BitNet.rs is organized as a Rust workspace with specialized crates:

#### Core Library
- **`bitnet`** (root): Main library with unified public API
- **`bitnet-common`**: Shared types, traits, and utilities
- **`bitnet-models`**: Model loading and format handling (GGUF, SafeTensors)
- **`bitnet-quantization`**: 1-bit quantization algorithms
- **`bitnet-kernels`**: High-performance SIMD/CUDA kernels with 2D convolution support
- **`bitnet-inference`**: Inference engine with streaming support
- **`bitnet-tokenizers`**: Universal tokenizer support
- **`bitnet-server`**: HTTP server for BitNet inference with health monitoring

#### Compatibility Layer
- **`bitnet-compat`**: GGUF compatibility fixes and diagnostics
- **`bitnet-ffi`**: C API for llama.cpp drop-in replacement
- **`bitnet-py`**: Python bindings compatible with llama-cpp-python

#### Cross-Validation
- **`crossval`**: Framework for testing against C++ implementation
- Tests use `BITNET_GGUF` or `CROSSVAL_GGUF` environment variable for model path

### Key Design Patterns

1. **Feature-Gated Architecture**: Default features are **empty** - always specify features explicitly
2. **Zero-Copy Operations**: Memory-mapped models, careful lifetime management
3. **SIMD Abstraction**: Unified interface over platform-specific instructions
4. **Cross-Validation**: Systematic comparison with C++ for correctness

## Important Considerations

### MSRV
Minimum Supported Rust Version: **1.89.0** (uses Rust 2024 edition)

### Feature Flags
Default features are **empty** to prevent unwanted dependencies:
- `cpu`: CPU inference with SIMD optimizations, includes native I2_S support
- `gpu`: NVIDIA GPU support with advanced device-aware quantization and automatic fallback
- `cuda`: Backward-compatible alias for `gpu` feature
- `iq2s-ffi`: IQ2_S quantization via GGML FFI (requires vendored GGML files)
- `ffi`: C++ FFI bridge (required for cross-validation) with enhanced safety documentation
- `crossval`: Cross-validation against C++ (increases build time)

### Quantization Support
BitNet-rs supports multiple quantization formats with device-aware acceleration:
- **I2_S**: Native Rust implementation with intelligent GPU/CPU selection and automatic fallback
- **TL1**: Table lookup quantization with GPU acceleration and CPU fallback
- **TL2**: Advanced table lookup quantization with optimized GPU kernels and device-aware execution
- **IQ2_S**: GGML-compatible quantization with 82-byte block layout and 4-level [-2,-1,1,2] mapping
- **Standard formats**: Q4_0, Q5_0, Q8_0, etc. (planned)

All quantizers support device-aware operations with automatic GPU acceleration and transparent CPU fallback.

### Compatibility Guarantees
We maintain strict compatibility with llama.cpp:
- C API functions have exact signature matches
- Python API is drop-in compatible
- We handle models that llama.cpp fails on (e.g., GPT-2 without pre-tokenizer)
- See COMPATIBILITY.md for detailed guarantees

## Troubleshooting

### Common Build Issues

1. **FFI Linker Errors**: Either disable FFI (`--no-default-features --features cpu`) or build C++ (`cargo xtask fetch-cpp`)

2. **CUDA Compilation**: Ensure CUDA toolkit is installed and `nvcc` is in PATH

3. **CUDA Device Query Failures**: See [GPU Development Guide](docs/gpu-development.md#advanced-gpucuda-troubleshooting) for comprehensive troubleshooting

4. **Cross-Validation Path**: Set `BITNET_GGUF` environment variable to model path

5. **Git Metadata in Builds**: The `bitnet-server` crate uses `vergen-gix` v1.x to capture Git metadata. Ensure `.git` is available during builds or set `VERGEN_GIT_SHA` and `VERGEN_GIT_BRANCH` environment variables

## Key Files

- `COMPATIBILITY.md`: API stability guarantees and truth tables
- `MIGRATION.md`: Step-by-step migration guide from llama.cpp
- `.github/workflows/compatibility.yml`: CI compatibility tests
- `crossval/`: Cross-validation test suite

## Specialized Documentation

For detailed information on specific topics, see:

- **[GPU Development Guide](docs/gpu-development.md)**: CUDA device querying, GPU testing strategies, and troubleshooting
- **[Test Suite Guide](docs/test-suite.md)**: Comprehensive testing framework, configuration, and specialized testing strategies  
- **[GGUF Inspection Guide](docs/gguf-inspection.md)**: Metadata inspection, categorization, and JSON serialization
- **[Streaming API Guide](docs/streaming-api.md)**: Real-time token streaming with Server-Sent Events
- **[Concurrency Caps Guide](docs/concurrency-caps.md)**: Resource management and concurrency control

## Environment Variables

### Runtime Variables
- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `HF_TOKEN`: Hugging Face token for private repos
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `RAYON_NUM_THREADS`: Control CPU parallelism

### Build-time Variables (for Git metadata)
- `VERGEN_GIT_SHA`: Override Git SHA (useful in CI/Docker without .git)
- `VERGEN_GIT_BRANCH`: Override Git branch
- `VERGEN_GIT_DESCRIBE`: Override Git describe output
- `VERGEN_IDEMPOTENT`: Set to "1" for reproducible builds

## Development Workflow

### Standard Development Process
1. **Making Changes**: Always run tests for affected crates
2. **Before Committing**: Run `cargo fmt` and `cargo clippy`
3. **Cross-Validation**: Run `cargo xtask crossval` for inference changes
4. **Compatibility**: Check COMPATIBILITY.md before changing public APIs

### GPU/CUDA Development
For GPU development best practices, PR management, and hardware-dependent testing strategies, see the [GPU Development Guide](docs/gpu-development.md).

## Repository Contracts (for Claude)

### Safe Operations
- **Default features are empty** → always pass `--no-default-features --features cpu|cuda`
- **Never mutate large binaries or GGUF in place** → use `bitnet-compat export-fixed` for new files
- **Prefer `xtask` over ad-hoc scripts** for downloads/crossval/build steps
- **Print commands before long operations** → use `--dry-run` where available
- **No destructive cleanup** without confirmation → avoid `rm -rf target/` or `~/.cache/bitnet_cpp/`

### Determinism & Environment
- `BITNET_DETERMINISTIC=1` + `BITNET_SEED=42` → force stable runs
- `RAYON_NUM_THREADS=1` → single-threaded CPU determinism
- `RUSTFLAGS="-C target-cpu=native"` → local perf builds (not CI)
- macOS FFI: set `DYLD_LIBRARY_PATH=target/release`
- Linux FFI: set `LD_LIBRARY_PATH=target/release`

## Fast Recipes

```bash
# Quick compile & test (CPU, MSRV-accurate)
rustup run 1.89.0 cargo test --workspace --no-default-features --features cpu

# Quick compile & test with concurrency caps
scripts/preflight.sh && cargo t2

# Validate GGUF file
cargo run -p bitnet-cli -- compat-check model.gguf
cargo run -p bitnet-cli -- compat-check model.gguf --json  # JSON output

# Inspect model metadata (human-readable with categorization)
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- model.gguf

# Export model metadata as JSON
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json model.gguf

# Teacher-forcing evaluation with perplexity calculation
cargo run -p bitnet-cli -- score --model model.gguf --file test.txt
cargo run -p bitnet-cli -- score --model model.gguf --file validation.txt --device cuda --batch-size 8 --json-out results.json

# Model evaluation with external tokenizer and token limits
cargo run -p bitnet-cli -- score --model model.gguf --file large-dataset.txt --tokenizer tokenizer.json --max-tokens 1000

# Full cross-validation (deterministic)
export BITNET_GGUF="$PWD/models/bitnet/ggml-model-i2_s.gguf"
export BITNET_DETERMINISTIC=1 BITNET_SEED=42
cargo run -p xtask -- full-crossval

# Export fixed GGUF safely (non-destructive)
cargo run -p bitnet-cli -- compat-fix "$BITNET_GGUF" fixed.gguf
cat fixed.gguf.compat.json   # audit stamp

# FFI smoke test (build + run)
cargo build -p bitnet-ffi --release --no-default-features --features cpu
export LD_LIBRARY_PATH=target/release  # or DYLD_LIBRARY_PATH on macOS
./scripts/ffi_smoke.sh

# Quick lint check
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```