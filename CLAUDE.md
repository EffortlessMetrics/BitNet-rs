# CLAUDE.md

This file provides guidance to Claude (claude.ai) when working with the BitNet.rs codebase.

## Build and Development Commands

### Core Development Commands
```bash
# Build with CPU support (no default features)
cargo build --release --no-default-features --features cpu

# Build with GPU/CUDA support
cargo build --release --no-default-features --features cuda

# Run tests (fast, Rust-only)
cargo test --workspace --no-default-features --features cpu

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

# Generate code coverage
cargo llvm-cov --workspace --features cpu --html
```

### Development Tools (xtask)
```bash
# Download BitNet model from Hugging Face
cargo run -p xtask -- download-model

# Fetch and build Microsoft BitNet C++ for cross-validation
cargo run -p xtask -- fetch-cpp

# Run cross-validation tests
cargo run -p xtask -- crossval

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
- **`bitnet-kernels`**: High-performance SIMD/CUDA kernels
- **`bitnet-inference`**: Inference engine with streaming support
- **`bitnet-tokenizers`**: Universal tokenizer support

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
- `cpu`: CPU inference with SIMD optimizations
- `cuda`: NVIDIA GPU support
- `ffi`: C++ FFI bridge (required for cross-validation)
- `crossval`: Cross-validation against C++ (increases build time)

### Testing Strategy
- **Unit tests**: Each crate has comprehensive tests
- **Integration tests**: Cross-crate tests in `tests/`
- **Cross-validation**: Automated testing against C++ implementation
- **CI gates**: Compatibility tests block on every PR

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

3. **Cross-Validation Path**: Set `BITNET_GGUF` environment variable to model path

## Development Workflow

1. **Making Changes**: Always run tests for affected crates
2. **Before Committing**: Run `cargo fmt` and `cargo clippy`
3. **Cross-Validation**: Run `cargo xtask crossval` for inference changes
4. **Compatibility**: Check COMPATIBILITY.md before changing public APIs

## Key Files

- `COMPATIBILITY.md`: API stability guarantees and truth tables
- `MIGRATION.md`: Step-by-step migration guide from llama.cpp
- `.github/workflows/compatibility.yml`: CI compatibility tests
- `crossval/`: Cross-validation test suite

## Environment Variables

- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `HF_TOKEN`: Hugging Face token for private repos
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `RAYON_NUM_THREADS`: Control CPU parallelism

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

# Full cross-validation (deterministic)
export BITNET_GGUF="$PWD/models/bitnet/ggml-model-i2_s.gguf"
export BITNET_DETERMINISTIC=1 BITNET_SEED=42
cargo run -p xtask -- full-crossval

# Check model compatibility (read-only)
cargo run -p bitnet-cli -- compat-check "$BITNET_GGUF"

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