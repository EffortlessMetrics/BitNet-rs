# CLAUDE.md

This file provides guidance to Claude (claude.ai) when working with the BitNet.rs codebase.

## Command Reference

### Core Development Commands
```bash
# Build with CPU support (no default features)
cargo build --release --no-default-features --features cpu

# Build with GPU/CUDA support
cargo build --release --no-default-features --features gpu

# Build with IQ2_S quantization support (requires GGML FFI)
cargo build --release --no-default-features --features "cpu,iq2s-ffi"

# Build with native I2_S support (no external dependencies)
cargo build --release --no-default-features --features cpu

# Run tests (fast, Rust-only)
cargo test --workspace --no-default-features --features cpu

# Run GPU tests (requires CUDA)
cargo test --workspace --no-default-features --features gpu

# Run GPU validation and benchmarks
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_integration
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_smoke

# Run GPU integration tests (comprehensive validation)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_integration --ignored

# Run GPU examples
cargo run -p bitnet-kernels --example gpu_validation --no-default-features --features gpu
cargo run -p bitnet-kernels --example simple_gpu_test --no-default-features --features gpu

# Run GGUF validation tests
cargo test -p bitnet-inference --test gguf_header
cargo test -p bitnet-inference --test gguf_fuzz
cargo test -p bitnet-inference --test engine_inspect

# Run async smoke test with synthetic GGUF
printf "GGUF\x02\x00\x00\x00" > /tmp/t.gguf && \
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf && \
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf && \
BITNET_GGUF=/tmp/t.gguf cargo test -p bitnet-inference --features rt-tokio --test smoke

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

# Generate code coverage
cargo llvm-cov --workspace --features cpu --html

# Run tests with concurrency caps (prevents resource storms)
scripts/preflight.sh && cargo t2                     # 2-thread CPU tests
scripts/preflight.sh && cargo crossval-capped        # Cross-validation with caps
scripts/e2e-gate.sh cargo test --features crossval   # Gate heavy E2E tests
```

### Development Tools (xtask)
```bash
# Download BitNet model from Hugging Face
cargo run -p xtask -- download-model

# Vendor GGML quantization files for IQ2_S support
cargo run -p xtask -- vendor-ggml --commit <llama.cpp-commit>

# Fetch and build Microsoft BitNet C++ for cross-validation
cargo run -p xtask -- fetch-cpp

# Run cross-validation tests
cargo run -p xtask -- crossval

# Full workflow (download + fetch + test)
cargo run -p xtask -- full-crossval

# Check feature flag consistency
cargo run -p xtask -- check-features
```

## Architecture Overview (Explanation)

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
- `gpu`: NVIDIA GPU support with device-aware quantization and automatic fallback (replaces `cuda`)
- `cuda`: Backward-compatible alias for `gpu` feature
- `iq2s-ffi`: IQ2_S quantization via GGML FFI (requires vendored GGML files)
- `ffi`: C++ FFI bridge (required for cross-validation) with Rust 2024 safety compliance and enhanced documentation
- `crossval`: Cross-validation against C++ (increases build time)

### Quantization Support
BitNet-rs supports multiple quantization formats:
- **I2_S**: Native Rust implementation, always available with `cpu` feature
- **IQ2_S**: Dual implementation:
  - Native Rust: Optimized implementation with SIMD support
  - GGML FFI: Via `iq2s-ffi` feature for llama.cpp compatibility
  - Backend parity testing ensures both implementations produce identical results
- **Standard formats**: Q4_0, Q5_0, Q8_0, etc. (planned)

To test IQ2_S implementations:
```bash
# Run IQ2_S backend validation
./scripts/test-iq2s-backend.sh

# Run unit tests
cargo test --package bitnet-models --no-default-features --features "cpu,iq2s-ffi"
```

### Testing Strategy
- **Unit tests**: Each crate has comprehensive tests
- **Integration tests**: Cross-crate tests in `tests/`
- **Property-based testing**: Fuzz testing for GGUF parser robustness
- **Cross-validation**: Automated testing against C++ implementation
- **CI gates**: Compatibility tests block on every PR

### Compatibility Guarantees
We maintain strict compatibility with llama.cpp:
- C API functions have exact signature matches
- Python API is drop-in compatible
- We handle models that llama.cpp fails on (e.g., GPT-2 without pre-tokenizer)
- See COMPATIBILITY.md for detailed guarantees

## Troubleshooting (How-To Guide)

### Common Build Issues

1. **FFI Linker Errors**: Either disable FFI (`--no-default-features --features cpu`) or build C++ (`cargo xtask fetch-cpp`)

2. **FFI Safety (Rust 2024)**: All FFI functions are marked as `unsafe fn` for proper memory safety. C clients are unaffected, but Rust callers must use `unsafe` blocks

3. **CUDA Compilation**: Ensure CUDA toolkit is installed and `nvcc` is in PATH. For device-aware quantization, CUDA 11.0+ is recommended

4. **Cross-Validation Path**: Set `BITNET_GGUF` environment variable to model path

5. **Git Metadata in Builds**: The `bitnet-server` crate uses `vergen-gix` v1.x to capture Git metadata. Ensure `.git` is available during builds or set `VERGEN_GIT_SHA` and `VERGEN_GIT_BRANCH` environment variables

6. **sccache Build Failures**: If experiencing "No such file or directory" errors during compilation, disable sccache:
   ```bash
   RUSTC_WRAPPER="" cargo test --workspace --no-default-features --features cpu
   ```

7. **Test Hangs/Thread Pool Conflicts**: Use thread caps and deterministic execution:
   ```bash
   RUST_TEST_THREADS=1 RAYON_NUM_THREADS=2 cargo test -p <crate> --no-default-features --features cpu -- --test-threads=1
   ```

8. **File Lock Contention**: Kill stray cargo processes and clean build artifacts:
   ```bash
   pkill -f 'cargo test' || true
   cargo clean
   ```

9. **GPU Memory Issues**: For CUDA out-of-memory errors during quantization:
   ```bash
   # Test GPU memory health
   cargo test -p bitnet-kernels --no-default-features --features gpu gpu_memory_health
   
   # Use single-threaded GPU tests to reduce memory pressure
   CUDA_VISIBLE_DEVICES=0 cargo test -p bitnet-kernels --no-default-features --features gpu -- --test-threads=1
   ```

10. **Device-aware Quantization Fallback**: If GPU quantization fails, verify automatic CPU fallback:
    ```bash
    # Enable debug logging to see fallback behavior
    RUST_LOG=bitnet_kernels=debug cargo test -p bitnet-kernels --no-default-features --features gpu
    ```

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

### Runtime Variables
- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `HF_TOKEN`: Hugging Face token for private repos
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `RAYON_NUM_THREADS`: Control CPU parallelism
- `CUDA_VISIBLE_DEVICES`: Control which GPU devices are visible (e.g., `0,1` or `0`)
- `RUST_LOG`: Enable debug logging for GPU operations (e.g., `bitnet_kernels=debug`)

### Build-time Variables (for Git metadata)
- `VERGEN_GIT_SHA`: Override Git SHA (useful in CI/Docker without .git)
- `VERGEN_GIT_BRANCH`: Override Git branch
- `VERGEN_GIT_DESCRIBE`: Override Git describe output
- `VERGEN_IDEMPOTENT`: Set to "1" for reproducible builds

## Concurrency Caps Implementation

BitNet-rs implements a comprehensive concurrency capping strategy to prevent resource exhaustion during parallel test execution:

### Core Components

1. **Preflight Script** (`scripts/preflight.sh`):
   - Monitors system resource usage (PIDs, file descriptors, load)
   - Sets conservative concurrency defaults
   - Auto-degrades to single-threaded mode under high system load
   - Configures BLAS thread limits for Python validation scripts

2. **E2E Gate** (`scripts/e2e-gate.sh`):
   - Limits concurrent heavy test suites (cross-validation, integration tests)
   - Uses file locking to queue test runs when system is busy
   - Integrates with preflight caps for resource management

3. **Cargo Aliases** (`.cargo/config.toml`):
   - `t2`: Run tests with 2-thread cap
   - `t1`: Run tests with 1-thread (deterministic mode)
   - `crossval-capped`: Cross-validation with thread caps
   - `gpu-capped`: GPU tests with concurrency caps and GPU features
   - `gpu-integration`: GPU integration tests with GPU features
   - `gpu-smoke`: Basic GPU smoke tests
   - `gpu-build`: Build workspace with GPU features

4. **Test Infrastructure** (`tests/common/concurrency_caps.rs`):
   - Rayon thread pool initialization with caps
   - Async task concurrency limits
   - Deterministic execution helpers

5. **Container Limits** (`docker-compose.test.yml`):
   - Hard resource limits (CPU, memory, PIDs, file descriptors)
   - Isolated test environments for different workloads
   - Volume caching for faster subsequent builds

### Environment Variables

```bash
# Thread control
RUST_TEST_THREADS=2      # Rust test parallelism
RAYON_NUM_THREADS=2      # Rayon thread pool size
CROSSVAL_WORKERS=2       # Cross-validation test workers

# BLAS thread limits (Python scripts)
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1

# BitNet deterministic execution
BITNET_DETERMINISTIC=1
BITNET_SEED=42

# GPU control
CUDA_VISIBLE_DEVICES=0   # Single GPU for testing
RUST_LOG=bitnet_kernels=debug  # GPU debug logging
```

### Usage Examples

```bash
# Standard capped test execution
scripts/preflight.sh && cargo t2

# Gate heavy E2E tests (max 2 concurrent)
scripts/e2e-gate.sh ./scripts/crossval.sh

# Container-isolated testing
docker-compose -f docker-compose.test.yml up rust-cpu-tests

# CI/deterministic mode
RUST_TEST_THREADS=1 RAYON_NUM_THREADS=1 cargo t1

# GPU tests with memory constraints
CUDA_VISIBLE_DEVICES=0 cargo test -p bitnet-kernels --no-default-features --features gpu -- --test-threads=1

# GPU integration tests with debug logging
RUST_LOG=bitnet_kernels=debug cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_integration --ignored

# Use GPU-specific aliases for convenience
cargo gpu-smoke         # Quick GPU smoke tests
cargo gpu-integration   # Full GPU integration tests  
cargo gpu-capped        # GPU tests with concurrency limits
cargo gpu-build         # Build with GPU support
```

## Repository Contracts (for Claude)

### Safe Operations
- **Default features are empty** → always pass `--no-default-features --features cpu|gpu`
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

## Quick Start Recipes (Tutorial)

```bash
# Quick compile & test (CPU, MSRV-accurate)
rustup run 1.89.0 cargo test --workspace --no-default-features --features cpu

# Quick compile & test with concurrency caps
scripts/preflight.sh && cargo t2

# Validate GGUF file
cargo run -p bitnet-cli -- compat-check model.gguf
cargo run -p bitnet-cli -- compat-check model.gguf --json  # JSON output

# Inspect model metadata
cargo run -p bitnet-cli -- inspect --model model.gguf

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

## Test Suite

### Running Tests

```bash
# Run all tests with CPU features
cargo test --workspace --no-default-features --features cpu

# Run tests with thread caps for deterministic execution
RUST_TEST_THREADS=1 RAYON_NUM_THREADS=2 cargo test --workspace --no-default-features --features cpu -- --nocapture --test-threads=1

# Deterministic single-threaded test execution
RUST_TEST_THREADS=1 RAYON_NUM_THREADS=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 cargo test --workspace --no-default-features --features cpu -- --nocapture --test-threads=1

# Run tests without sccache if experiencing build issues
RUSTC_WRAPPER="" cargo test --workspace --no-default-features --features cpu

# Run specific test suites
cargo test --package bitnet-tests --no-default-features --features fixtures
cargo test --package bitnet-tests --features reporting,trend

# Run integration tests
cargo test --package bitnet-tests --features integration-tests,fixtures

# Run examples
cargo run --example reporting_example --features reporting
cargo run --example ci_reporting_example --features reporting,trend
cargo run --example debugging_example --features fixtures

# Run GGUF format validation tests
cargo test -p bitnet-inference --test gguf_header                 # Pure parser test
cargo test -p bitnet-inference --no-default-features --features rt-tokio --test smoke -- --nocapture  # Async smoke test

# Run verification script for all tests
./scripts/verify-tests.sh
```

### Test Configuration

The test suite uses a feature-gated configuration system:
- `fixtures`: Enables fixture management and test data generation
- `reporting`: Enables test reporting (JSON, HTML, Markdown, JUnit)
- `trend`: Enables trend analysis and performance tracking
- `integration-tests`: Enables full integration test suite

### Test Features

- **Parallel Test Execution**: Configurable parallelism with resource limits
- **Fixture Management**: Automatic test data generation and caching
- **CI Integration**: JUnit output, exit codes, and CI-specific optimizations
- **Error Reporting**: Detailed error messages with recovery suggestions
- **Performance Tracking**: Benchmark results and regression detection

## Validation Framework

### Evaluation Commands

```bash
# Evaluate perplexity on a corpus (token-weighted NLL)
target/release/bitnet eval \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --text-file crossval/data/ppl_smoke.txt

# Teacher-forcing with explicit token IDs + logit dump
target/release/bitnet eval \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --teacher-force-ids 1,2,3,4,5,6 \
  --dump-logit-steps 6 --logits-topk 10 \
  --json-out /tmp/tf_eval.json

# Deterministic greedy generation with logit tapping
target/release/bitnet run \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --prompt "Define entropy." \
  --max-new-tokens 32 --greedy \
  --deterministic --threads 1 \
  --dump-logit-steps 8 --logits-topk 10 \
  --json-out /tmp/run.json
```

### Validation Tests

```bash
# Tokenizer parity check
BITNET_BIN=target/release/bitnet \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
scripts/test-tokenizer-parity.py --smoke

# Logit parity with tau-b correlation
PROP_EXAMPLES=10 TAU_STEPS=24 LOGIT_TOPK=10 TAU_MIN=0.60 \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
scripts/logit-parity.sh

# NLL parity (token-weighted)
DELTA_NLL_MAX=1e-2 \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
PPL_FILE=crossval/data/ppl_smoke.txt \
scripts/nll-parity.sh
```

### Key Validation Features

1. **Token-Weighted NLL**: Proper corpus perplexity using `Σ(token_nlls) / Σ(predicted_tokens)`
2. **Teacher-Forcing**: Exact decode path with causal masking and position encoding
3. **Deterministic Top-K**: Stable sorting with tie-breaking by token ID, NaN demotion
4. **Logit Dumping**: Capture top-k logits at each generation step for analysis
5. **Tau-b Correlation**: Score-aware rank correlation for quantization robustness

### Validation Infrastructure Improvements

Recent enhancements to the validation framework include:
- **FFI Safety Validation**: Enhanced memory safety checks for C API layer
- **MSRV Compliance**: Automated testing against minimum supported Rust version (1.89.0)
- **Resource-Aware Testing**: Concurrency caps and system resource monitoring
- **Comprehensive Test Coverage**: GGUF parser validation, async smoke tests, and synthetic model generation
- **Build Matrix Validation**: Cross-platform testing with feature flag combinations