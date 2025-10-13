# Environment Variables Reference

This document describes all environment variables used throughout BitNet.rs for configuration, testing, and development.

## Runtime Variables

### Model and Testing Configuration
- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `HF_TOKEN`: Hugging Face token for private repos
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `BITNET_STRICT_MODE`: Prevent mock inference fallbacks and validate LayerNorm gamma statistics ("1" enables strict mode for production)
  - Prevents all mock inference paths
  - Validates LayerNorm gamma weights have mean ≈ 1.0
  - Fails immediately on suspicious LayerNorm statistics (mean outside [0.5, 2.0])
  - In non-strict mode (default), issues warnings but continues

### Model Validation and Correction Policy
- `BITNET_CORRECTION_POLICY`: Path to YAML policy file defining model-specific corrections
  - **Value**: Absolute or relative path to policy YAML file (e.g., `/path/to/policy.yml`)
  - **Purpose**: Enable runtime corrections for known-bad models with fingerprinted, auditable fixes
  - **Format**: YAML file specifying model fingerprints and correction parameters
  - **Usage**:
    ```bash
    # Enable policy-driven corrections
    export BITNET_CORRECTION_POLICY=/path/to/correction-policy.yml
    export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
    cargo run -p bitnet-cli -- run --model model.gguf
    ```
  - **Important**: Both `BITNET_CORRECTION_POLICY` and `BITNET_ALLOW_RUNTIME_CORRECTIONS` must be set

- `BITNET_ALLOW_RUNTIME_CORRECTIONS`: Enable runtime corrections (must be used with BITNET_CORRECTION_POLICY)
  - **Value**: "1" to enable (disabled by default)
  - **Purpose**: Safety gate preventing accidental application of corrections
  - **Warning**: CI blocks correction flags - runtime corrections are for known-bad models only
  - **Proper fix**: Always prefer regenerating GGUF with LayerNorm weights in FP16/FP32 (not quantized)
  - **Usage**:
    ```bash
    # Inspect model statistics first
    cargo run -p bitnet-cli -- inspect --ln-stats model.gguf

    # Apply corrections if needed (temporary workaround)
    export BITNET_CORRECTION_POLICY=./model-corrections.yml
    export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
    cargo run -p bitnet-cli -- run --model model.gguf
    ```

### Performance and Parallelism
- `RAYON_NUM_THREADS`: Control CPU parallelism

### GPU Feature Detection (Issue #439)
- `BITNET_GPU_FAKE`: Override GPU detection for deterministic testing and device-aware fallback validation
  - **Values**:
    - `none`: Disable GPU detection (test CPU fallback paths)
    - `cuda` or `gpu`: Enable fake GPU detection (test GPU code paths without hardware)
    - `metal`, `rocm`: Simulate specific GPU backends
    - Multiple backends: `cuda,rocm` (comma-separated)
  - **Usage with Preflight**:
    ```bash
    # Test CPU fallback behavior
    BITNET_GPU_FAKE=none cargo run -p xtask -- preflight
    # Expected: "✗ GPU: Not available at runtime"

    # Test GPU path without hardware
    BITNET_GPU_FAKE=cuda cargo run -p xtask -- preflight
    # Expected: "✓ GPU: Available"
    ```
  - **Device-Aware Testing**:
    ```bash
    # Test quantization device selection with fake GPU
    BITNET_GPU_FAKE=cuda cargo test --no-default-features --features gpu -p bitnet-quantization

    # Test CPU fallback in GPU-compiled binary
    BITNET_GPU_FAKE=none cargo test --no-default-features --features gpu -p bitnet-inference
    ```

## Strict Testing Mode Variables

These variables prevent "Potemkin passes" (false positives) in performance and integration tests by eliminating mock inference paths:

### Primary Strict Mode (Issue #261 Implementation)
- `BITNET_STRICT_MODE=1`: **Primary strict mode** - Prevents ALL mock inference fallbacks, essential for production deployment and accurate performance measurement
  - Enables `fail_on_mock`, `require_quantization`, and `validate_performance` checks
  - Fails fast when mock computation paths are detected
  - Validates performance metrics to reject suspicious values (>150 tok/s flagged as potentially mock)
  - Required for production deployments to ensure real quantized inference

### Detailed Strict Mode Controls (Issue #261 - Granular Configuration)
- `BITNET_STRICT_FAIL_ON_MOCK=1`: Fail immediately when mock computation is detected in inference pipeline
  - Activated automatically when `BITNET_STRICT_MODE=1`
  - Can be enabled independently for targeted testing
  - Validates all tensor operations and kernel calls for mock usage

- `BITNET_STRICT_REQUIRE_QUANTIZATION=1`: Require real quantization kernels (I2S/TL1/TL2) to be available and used
  - Activated automatically when `BITNET_STRICT_MODE=1`
  - Prevents fallback to FP32 computation when quantization expected
  - Validates device-aware quantization kernel selection

- `BITNET_STRICT_VALIDATE_PERFORMANCE=1`: Validate performance metrics for realistic values
  - Activated automatically when `BITNET_STRICT_MODE=1`
  - Rejects performance metrics from mock computation paths
  - Flags unrealistic throughput (>150 tok/s) as suspicious

- `BITNET_CI_ENHANCED_STRICT=1`: Enhanced strict mode for CI environments (Issue #261 - AC6)
  - Activates when both `CI` environment variable and this flag are set
  - Enables `ci_enhanced_mode`, `log_all_validations`, and `fail_fast_on_any_mock`
  - Provides comprehensive logging for CI pipeline debugging
  - Ensures production-grade validation in automated testing

### Legacy Strict Mode Variables
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

### Basic Strict Mode Usage (Issue #261)
```bash
# Primary strict mode - prevents ALL mock inference fallbacks
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-inference --features cpu
BITNET_STRICT_MODE=1 cargo run -p xtask -- infer --model model.gguf --prompt "Test"

# Production inference with strict mode (realistic performance: 10-20 tok/s CPU, 50-100 tok/s GPU)
BITNET_STRICT_MODE=1 cargo run -p xtask -- infer \
  --model models/bitnet-model.gguf \
  --prompt "Explain quantum computing" \
  --deterministic
```

### Granular Strict Mode Controls (Issue #261)
```bash
# Fail immediately on mock detection
BITNET_STRICT_FAIL_ON_MOCK=1 \
cargo test -p bitnet-inference --features cpu test_inference_real_computation

# Require real quantization kernels (I2S/TL1/TL2)
BITNET_STRICT_REQUIRE_QUANTIZATION=1 \
cargo test -p bitnet-quantization --features cpu test_quantization_kernel_integration

# Validate performance metrics for realistic values
BITNET_STRICT_VALIDATE_PERFORMANCE=1 \
cargo run -p xtask -- benchmark --model model.gguf --tokens 128

# CI enhanced strict mode (comprehensive validation)
CI=1 BITNET_CI_ENHANCED_STRICT=1 BITNET_STRICT_MODE=1 \
cargo test --workspace --features cpu
```

### Performance Testing with Strict Mode
```bash
# CPU baseline with real quantization (no mocks)
BITNET_STRICT_MODE=1 \
cargo bench --no-default-features --features cpu -p bitnet-quantization --bench simd_comparison

# GPU performance with strict hardware validation
BITNET_STRICT_NO_FAKE_GPU=1 \
BITNET_STRICT_MODE=1 \
cargo bench -p bitnet-kernels --bench mixed_precision_bench --features gpu

# Realistic CPU performance baselines (Issue #261 - AC7)
# Expected: I2S 10-20 tok/s, TL1 12-18 tok/s, TL2 10-15 tok/s
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo run -p xtask -- benchmark --features cpu --quantization i2s

# Realistic GPU performance baselines (Issue #261 - AC8)
# Expected: Mixed precision 50-100 tok/s, GPU utilization >80%
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
cargo run -p xtask -- benchmark --features gpu --quantization i2s
```

### Strict Integration Testing
```bash
# Strict tokenizer tests (no mock fallbacks)
BITNET_STRICT_TOKENIZERS=1 \
BITNET_STRICT_MODE=1 \
cargo test --features cpu -p bitnet-tokenizers -- --quiet

# Strict GPU kernel tests (real hardware only)
BITNET_STRICT_NO_FAKE_GPU=1 \
BITNET_STRICT_MODE=1 \
cargo test --no-default-features -p bitnet-kernels --features gpu -- --quiet

# Combined strict testing for production validation
BITNET_STRICT_MODE=1 \
BITNET_STRICT_TOKENIZERS=1 \
BITNET_STRICT_NO_FAKE_GPU=1 \
scripts/verify-tests.sh

# Cross-validation with strict mode (Issue #261 - AC9)
# Validates quantization accuracy: I2S ≥99.8%, TL1/TL2 ≥99.6% vs FP32
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