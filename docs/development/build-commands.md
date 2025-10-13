# Build Commands Reference

This document provides comprehensive build and development commands for BitNet.rs.

## Core Development Commands

### Basic Builds
```bash
# Build with CPU support (no default features)
cargo build --no-default-features --release --features cpu

# Build with GPU support and device-aware quantization
cargo build --no-default-features --release --features gpu

# Build with IQ2_S quantization support (requires GGML FFI)
cargo build --no-default-features --release --features "cpu,iq2s-ffi"

# Build with FFI quantization bridge support (requires C++ library)
cargo build --no-default-features --release --features "cpu,ffi"

# Build CLI with full validation features (includes inspect command)
cargo build --no-default-features --release -p bitnet-cli --features cpu,full-cli
```

### WebAssembly Builds
```bash
# Add WASM target
rustup target add wasm32-unknown-unknown

# Basic WASM builds
cargo build --no-default-features --features cpu --release --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features
cargo build --no-default-features --release --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features browser
cargo build --no-default-features --release --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features nodejs

# Build WASM with enhanced debugging and error reporting
cargo build --no-default-features --release --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features "browser,debug"
```

### Testing Commands
```bash
# Run tests (fast, Rust-only)
cargo test --no-default-features --workspace --no-default-features --features cpu

# Run GPU tests with device-aware quantization (requires CUDA)
cargo test --no-default-features --workspace --no-default-features --features gpu

# Run GGUF validation tests (includes tensor alignment validation)
cargo test --no-default-features -p bitnet-inference --test gguf_header --no-default-features --features cpu
cargo test --no-default-features -p bitnet-inference --test gguf_fuzz --no-default-features --features cpu
cargo test --no-default-features -p bitnet-inference --test engine_inspect --no-default-features --features cpu

# Test enhanced GGUF tensor alignment validation
cargo test --no-default-features -p bitnet-models --test gguf_min --no-default-features --features cpu -- test_tensor_alignment
cargo test --no-default-features -p bitnet-models --no-default-features --features cpu -- gguf_min::tests::loads_two_tensors

# Run verification script
./scripts/verify-tests.sh

# Test enhanced prefill functionality and batch inference
cargo test --no-default-features -p bitnet-cli --test cli_smoke --no-default-features --features cpu
cargo test --no-default-features -p bitnet-inference --test batch_prefill --no-default-features --features cpu
```

### Benchmarking Commands
```bash
# Run benchmarks
cargo bench --no-default-features --workspace --no-default-features --features cpu

# SIMD optimization benchmarks
cargo bench --no-default-features -p bitnet-quantization --bench simd_comparison --no-default-features --features cpu
cargo bench --no-default-features -p bitnet-quantization simd_vs_scalar --no-default-features --features cpu

# Mixed precision benchmarks and performance analysis
cargo bench --no-default-features -p bitnet-kernels --bench mixed_precision_bench --no-default-features --features gpu
```

### Cross-Validation Commands
```bash
# Cross-validation testing (requires C++ dependencies)
cargo test --no-default-features --workspace --no-default-features --features "cpu,ffi,crossval"

# FFI quantization bridge tests (compares FFI vs Rust implementations)
cargo test --no-default-features -p bitnet-kernels --no-default-features --features "cpu,ffi" test_ffi_quantize_matches_rust

# SIMD kernel compatibility and performance tests
cargo test --no-default-features -p bitnet-quantization --test simd_compatibility --no-default-features --features cpu
cargo test --no-default-features -p bitnet-quantization test_i2s_simd_scalar_parity --no-default-features --features cpu
cargo test --no-default-features -p bitnet-quantization test_simd_performance_baseline --no-default-features --features cpu
```

### GPU-Specific Commands
```bash
# Mixed precision GPU kernel tests (requires CUDA)
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_kernel_creation
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_matmul_accuracy
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_precision_mode_validation
```

### Tokenizer Testing
```bash
# SentencePiece (SPM) tokenizer tests (requires spm feature)
cargo test --no-default-features -p bitnet-tokenizers --no-default-features --features "cpu,spm,integration-tests" --test tokenizer_contracts test_sentencepiece_tokenizer_contract
cargo test --no-default-features -p bitnet-tokenizers --no-default-features --features "cpu,spm,integration-tests" -- --ignored  # SPM model tests (requires actual .model file)
cargo test --no-default-features -p bitnet-tokenizers --no-default-features --features "cpu,spm,integration-tests" -- --quiet

# Strict tokenizer mode testing (prevents fallback to mock tokenizers)
BITNET_STRICT_TOKENIZERS=1 cargo test --no-default-features -p bitnet-tokenizers --no-default-features --features "cpu,spm"
BITNET_STRICT_TOKENIZERS=1 cargo test --no-default-features -p bitnet-tokenizers --no-default-features --features cpu -- --quiet
```

### WebAssembly Testing
```bash
# WebAssembly tests (requires wasm32 target)
rustup target add wasm32-unknown-unknown
cargo test --no-default-features --features cpu -p bitnet-wasm --target wasm32-unknown-unknown --no-default-features
wasm-pack test --node crates/bitnet-wasm/  # Requires wasm-pack for browser tests
wasm-pack test --chrome --headless crates/bitnet-wasm/  # Browser testing with headless Chrome
```

## Code Quality Commands

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
scripts/e2e-gate.sh cargo test --no-default-features --features crossval   # Gate heavy E2E tests

# Generate code coverage
cargo llvm-cov --workspace --features cpu --html
```

## Development Tools (xtask)

### Model Management
```bash
# Download BitNet model from Hugging Face
cargo run -p xtask -- download-model

# Vendor GGML quantization files for IQ2_S support
cargo run -p xtask -- vendor-ggml --commit <llama.cpp-commit>

# Fetch and build Microsoft BitNet C++ for cross-validation
cargo run -p xtask -- fetch-cpp
```

### Cross-Validation Workflow
```bash
# Run cross-validation tests
cargo run -p xtask -- crossval
# Full workflow (download + fetch + test)
cargo run -p xtask -- full-crossval

# Check feature flag consistency
cargo run -p xtask -- check-features
```

### Model Verification
```bash
# Verify model configuration and tokenizer compatibility
cargo run -p xtask -- verify --model models/bitnet/model.gguf
cargo run -p xtask -- verify --model models/bitnet/model.gguf --tokenizer models/bitnet/tokenizer.json
cargo run -p xtask -- verify --model models/bitnet/model.gguf --tokenizer models/bitnet/tokenizer.model  # SPM tokenizer
cargo run -p xtask -- verify --model models/bitnet/model.gguf --format json
```

### Inference Testing
```bash
# Run simple inference for smoke testing (requires --features inference for real inference)
cargo run -p xtask --features inference -- infer --model models/bitnet/model.gguf --prompt "The capital of France is" --tokenizer models/bitnet/tokenizer.json
cargo run -p xtask --features inference -- infer --model models/bitnet/model.gguf --prompt "The capital of France is" --tokenizer models/bitnet/tokenizer.model  # SPM tokenizer
cargo run -p xtask -- infer --model models/bitnet/model.gguf --prompt "Hello world" --allow-mock --format json
cargo run -p xtask --features inference -- infer --model models/bitnet/model.gguf --prompt "Test prompt" --max-new-tokens 64 --temperature 0.7 --gpu
```

### Model Validation
```bash
# Build CLI with validation features
cargo build --no-default-features -p bitnet-cli --features cpu,full-cli

# Inspect LayerNorm and projection weight statistics (architecture-aware validation)
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto models/model.gguf

# Validate with custom policy
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate policy \
  --policy examples/policies/custom.yml \
  --policy-key my-model:f16 \
  models/model.gguf

# JSON output for CI integration
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto --json models/model.gguf > validation.json

# Strict mode (fail on validation warnings)
BITNET_STRICT_MODE=1 \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto models/model.gguf

# Full 3-stage validation (LayerNorm, projection, linguistic sanity)
./scripts/validate_gguf.sh models/model.gguf models/tokenizer.json

# Export clean GGUF from SafeTensors with LayerNorm preservation
./scripts/export_clean_gguf.sh \
  models/safetensors-checkpoint \
  models/tokenizer.json \
  models/clean
```

## Fast Recipes

```bash
# Quick compile & test (CPU, MSRV-accurate)
rustup run 1.90.0 cargo test --no-default-features --workspace --no-default-features --features cpu

# Quick compile & test with concurrency caps
scripts/preflight.sh && cargo t2

# Quick WASM build and test (browser-compatible)
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features browser
cargo test --no-default-features --features cpu -p bitnet-wasm --target wasm32-unknown-unknown --no-default-features

# Quick WASM build with enhanced debugging
cargo build --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features "browser,debug"

# Quick lint check
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

For more detailed information on specific testing strategies, see:
- [Test Suite Guide](test-suite.md)
- [GPU Development Guide](gpu-development.md)
- [Performance Benchmarking Guide](performance-benchmarking.md)
