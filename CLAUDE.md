# CLAUDE.md

This file provides essential guidance to Claude (claude.ai) when working with the BitNet.rs codebase.

## Quick Reference

### Essential Commands
```bash
# Build with CPU support (default features are EMPTY)
cargo build --release --no-default-features --features cpu

# Build with GPU support
cargo build --release --no-default-features --features gpu

# Run tests (always specify features explicitly)
cargo test --workspace --no-default-features --features cpu
cargo test --workspace --no-default-features --features gpu

# Format and lint
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Development workflow (xtask)
cargo run -p xtask -- download-model
cargo run -p xtask -- verify --model models/bitnet/model.gguf --tokenizer models/bitnet/tokenizer.json
cargo run -p xtask -- infer --model models/bitnet/model.gguf --prompt "Test" --deterministic
```

## Architecture & Key Concepts

BitNet.rs is a Rust workspace with specialized crates for neural network inference.

### Core Design Principles
1. **Feature-Gated Architecture**: Default features are **EMPTY** - always specify features explicitly
2. **Zero-Copy Operations**: Memory-mapped models, careful lifetime management
3. **Device-Aware Computing**: Automatic GPU/CPU selection with graceful fallback
4. **Cross-Validation**: Systematic comparison with C++ reference implementation

### Key Workspace Crates
- **`bitnet`** (root): Main library with unified public API
- **`bitnet-inference`**: Inference engine with streaming support
- **`bitnet-quantization`**: 1-bit quantization (I2_S, TL1, TL2, IQ2_S)
- **`bitnet-kernels`**: High-performance SIMD/CUDA kernels
- **`bitnet-tokenizers`**: Universal tokenizer with GGUF integration
- **`bitnet-models`**: Model loading (GGUF, SafeTensors)
- **`crossval`**: Framework for testing against C++ implementation

## Important Considerations

### MSRV
Minimum Supported Rust Version: **1.90.0** (Rust 2024 edition)

### Feature Flags
**Default features are EMPTY** - always specify explicitly:
- `cpu`: CPU inference with SIMD optimizations
- `gpu`: NVIDIA GPU support with mixed precision (FP16/BF16)
- `ffi`: C++ FFI bridge for gradual migration
- `spm`: SentencePiece tokenizer support
- `crossval`: Cross-validation against C++ implementation
- WebAssembly: `browser`, `nodejs`, `debug`, `inference`

### Supported Quantization
- **I2_S**: Native 2-bit signed quantization with GPU/CPU auto-selection
- **TL1/TL2**: Table lookup quantization with vectorized operations
- **IQ2_S**: GGML-compatible with 82-byte blocks
- Device-aware operations with automatic GPU acceleration and CPU fallback

### Universal Tokenizer
- Auto-detecting tokenizer with GGUF integration
- Supports BPE, SentencePiece, LLaMA variants, TikToken
- Graceful fallback for testing and compatibility validation
- O(1) byte lookup performance with optimized UTF-8 handling

## Common Issues

1. **FFI Linker Errors**: Disable FFI (`--no-default-features --features cpu`) or build C++: `cargo xtask fetch-cpp`
2. **CUDA Issues**: Ensure CUDA toolkit installed and `nvcc` in PATH
3. **Cross-Validation**: Set `BITNET_GGUF` environment variable to model path
4. **GGUF Errors**: Use `cargo run -p bitnet-cli -- compat-check model.gguf` for validation

## Development Workflow

1. Always run tests for affected crates
2. Format and lint before committing: `cargo fmt --all && cargo clippy`
3. Cross-validate inference changes: `cargo xtask crossval`
4. Check COMPATIBILITY.md before API changes

## 4-Step Developer Workflow

```bash
# 1. Download model and tokenizer
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf --file ggml-model-i2_s.gguf

# 2. Verify compatibility
cargo run -p xtask -- verify --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json

# 3. Test inference
cargo run -p xtask -- infer --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json --prompt "Test" --deterministic

# 4. Benchmark (local only)
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release -p xtask
cargo run -p xtask -- benchmark --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json --tokens 128
```

## Key Files & Documentation

**Essential Files:**
- `COMPATIBILITY.md`: API stability guarantees
- `MIGRATION.md`: Migration guide from llama.cpp
- `crossval/`: Cross-validation test suite

**Detailed Documentation:**
- [Build Commands](docs/build-commands.md): Comprehensive build and test commands
- [Architecture Overview](docs/architecture-overview.md): System design and components
- [Quantization Support](docs/quantization-support.md): Quantization formats and GPU acceleration
- [Environment Variables](docs/environment-variables.md): Configuration variables
- [Validation Framework](docs/validation-framework.md): Testing and evaluation
- [GPU Development Guide](docs/gpu-development.md): CUDA development
- [Test Suite Guide](docs/test-suite.md): Testing framework
- [Performance Benchmarking Guide](docs/performance-benchmarking.md): Performance testing

## Repository Contracts

### Safe Operations for Claude
- **Default features are EMPTY** → always use `--no-default-features --features cpu|gpu`
- **Never modify GGUF files in-place** → use `bitnet-compat export-fixed` for new files
- **Prefer xtask over scripts** → use `cargo run -p xtask --` for model operations
- **No destructive cleanup** → avoid `rm -rf target/` without confirmation

### Environment Setup
- `BITNET_DETERMINISTIC=1 BITNET_SEED=42` → reproducible runs
- `RAYON_NUM_THREADS=1` → single-threaded determinism
- FFI: `LD_LIBRARY_PATH=target/release` (Linux), `DYLD_LIBRARY_PATH=target/release` (macOS)

## Fast Recipes

```bash
# Quick test
cargo test --workspace --no-default-features --features cpu

# WASM build
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features browser

# Validate model
cargo run -p bitnet-cli -- compat-check model.gguf

# Cross-validation
export BITNET_GGUF="models/bitnet/model.gguf" BITNET_DETERMINISTIC=1 BITNET_SEED=42
cargo run -p xtask -- full-crossval

# Strict testing (no mocks)
BITNET_STRICT_TOKENIZERS=1 BITNET_STRICT_NO_FAKE_GPU=1 scripts/verify-tests.sh
```

For comprehensive command references, see [Build Commands](docs/build-commands.md).

**Important:** See detailed documentation in `docs/` directory for comprehensive information on all topics.