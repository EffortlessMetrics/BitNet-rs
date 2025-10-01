# Environment Variables Reference

This document describes all environment variables used throughout BitNet.rs for configuration, testing, and development.

## Runtime Variables

### Model and Testing Configuration
- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `HF_TOKEN`: Hugging Face token for private repos
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `BITNET_STRICT_MODE`: Prevent mock inference fallbacks ("1" enables strict mode for production)

### Performance and Parallelism
- `RAYON_NUM_THREADS`: Control CPU parallelism
- `BITNET_GPU_FAKE`: Mock GPU backend detection for testing (e.g., "cuda", "metal", "cuda,rocm")

## Strict Testing Mode Variables

These variables prevent "Potemkin passes" (false positives) in performance and integration tests:

- `BITNET_STRICT_MODE=1`: **Primary strict mode** - Prevents ALL mock inference fallbacks, essential for production deployment and accurate performance measurement
- `BITNET_STRICT_TOKENIZERS=1`: Forbid mock tokenizer fallbacks in perf/integration tests (includes SPM tokenizer fallbacks)
- `BITNET_STRICT_NO_FAKE_GPU=1`: Forbid fake GPU backends in perf/integration tests

## Build-time Variables

For Git metadata capture (used by `bitnet-server` crate with `vergen-gix`):

- `VERGEN_GIT_SHA`: Override Git SHA (useful in CI/Docker without .git)
- `VERGEN_GIT_BRANCH`: Override Git branch
- `VERGEN_GIT_DESCRIBE`: Override Git describe output
- `VERGEN_IDEMPOTENT`: Set to "1" for reproducible builds

## FFI Configuration

### Compiler Selection
```bash
# GCC (default)
export CC=gcc CXX=g++

# Clang
export CC=clang CXX=clang++
```

### Library Path Configuration
```bash
# Linux FFI
export LD_LIBRARY_PATH=target/release

# macOS FFI
export DYLD_LIBRARY_PATH=target/release
```

## GPU Development Variables

For GPU development, testing, and mock scenarios:

```bash
# Test GPU backend detection
cargo test --no-default-features --features cpu -p bitnet-kernels --no-default-features test_gpu_info_summary

# Mock GPU scenarios for testing
BITNET_GPU_FAKE="cuda" cargo test --no-default-features --features cpu -p bitnet-kernels test_gpu_info_mocked_scenarios
BITNET_GPU_FAKE="metal" cargo run -p xtask -- download-model --dry-run
BITNET_GPU_FAKE="cuda,rocm" cargo test --no-default-features -p bitnet-kernels --features gpu
```

## Determinism Configuration

For reproducible builds and testing:

```bash
# Force stable runs with strict mode (no mock fallbacks)
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

# Single-threaded CPU determinism for testing
export RAYON_NUM_THREADS=1

# Production deterministic inference with real quantization
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p xtask -- infer --model model.gguf --prompt "Test"

# Local performance builds (not CI)
export RUSTFLAGS="-C target-cpu=native"
```

## Strict Testing Examples

```bash
# Primary strict mode - prevents ALL mock inference fallbacks
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-inference --no-default-features --features cpu
BITNET_STRICT_MODE=1 cargo run -p xtask -- infer --model model.gguf --prompt "Test"

# CPU baseline (no mocks involved)
cargo bench --no-default-features --features cpu -p bitnet-quantization --bench simd_comparison

# GPU perf (strict, real hardware only)
BITNET_STRICT_NO_FAKE_GPU=1 \
BITNET_STRICT_MODE=1 \
cargo bench --no-default-features -p bitnet-kernels --bench mixed_precision_bench --no-default-features --features gpu

# Strict integration/tokenizer tests (no mock fallbacks)
BITNET_STRICT_TOKENIZERS=1 \
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu -p bitnet-tokenizers -- --quiet

BITNET_STRICT_NO_FAKE_GPU=1 \
BITNET_STRICT_MODE=1 \
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu -- --quiet

# Combined strict testing for production validation
BITNET_STRICT_MODE=1 \
BITNET_STRICT_TOKENIZERS=1 \
BITNET_STRICT_NO_FAKE_GPU=1 \
scripts/verify-tests.sh

# Cross-validation with strict mode (no mock fallbacks)
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo run -p xtask -- crossval
```

## System Metrics Variables

For server monitoring and system metrics collection:

```bash
# Test system metrics collection in server
cargo test --no-default-features -p bitnet-server --features prometheus test_system_metrics_collection

# Run server with system metrics enabled
cargo run -p bitnet-server --features prometheus --bin server &
curl http://localhost:8080/metrics | grep "system_"

# Test memory tracking integration with system metrics
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_memory_tracking_comprehensive

# Validate system metrics in monitoring stack
cd monitoring && docker-compose up -d
curl http://localhost:9090/api/v1/query?query=system_cpu_usage_percent
```

For more information on specific topics, see:
- [GPU Development Guide](development/gpu-development.md) - GPU-specific environment variables and testing
- [Test Suite Guide](development/test-suite.md) - Testing configuration and variables
- [Performance Benchmarking Guide](performance-benchmarking.md) - Performance testing variables