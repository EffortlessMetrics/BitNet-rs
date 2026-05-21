# Performance Benchmarking and Regression Detection

> Claim boundary: benchmark infrastructure and historical example numbers do
> not prove current speedup, CUDA/GPU readiness, server readiness, residency, or
> product readiness. Current performance claims require exact-profile receipts,
> model coverage, status docs, specs, and claim gates.

BitNet-rs includes a comprehensive performance benchmarking infrastructure designed to detect performance regressions, track improvements, and ensure consistent performance across platforms.

## 🚀 Quick Start

### Running Basic Benchmarks

```bash
# Setup environment and run CPU benchmarks with strict mode (no mock fallbacks)
./scripts/setup-perf-env.sh
BITNET_STRICT_MODE=1 ./scripts/run-performance-benchmarks.sh

# Run GPU benchmarks with strict mode (requires CUDA, GPU-accelerated alpha)
BITNET_STRICT_MODE=1 ./scripts/run-performance-benchmarks.sh --features gpu

# Run with cross-validation against C++ implementation and strict mode
BITNET_STRICT_MODE=1 ./scripts/run-performance-benchmarks.sh --include-cpp
```

### Detecting Performance Regressions

```bash
# Run regression analysis on latest benchmark results
python3 scripts/detect-performance-regression.py benchmark-results/performance-report.json

# Fail build on critical regressions (useful for CI)
python3 scripts/detect-performance-regression.py \
  benchmark-results/performance-report.json \
  --fail-on-regression
```

## 📊 Architecture Overview

The benchmarking infrastructure consists of several integrated components:

### Core Components

1. **Setup Script** (`scripts/setup-perf-env.sh`)
   - Configures deterministic benchmark environment
   - Generates test fixtures
   - Sets up cross-compilation toolchains
   - Builds optimized binaries

2. **Benchmark Runner** (`scripts/run-performance-benchmarks.sh`)
   - Executes Criterion benchmarks
   - Runs comparison tests against C++ implementation
   - Generates comprehensive performance reports
   - Supports cross-platform testing

3. **Regression Detector** (`scripts/detect-performance-regression.py`)
   - Compares current results against established baselines
   - Detects critical regressions, warnings, and improvements
   - Provides detailed analysis and alerting

4. **Baseline Generator** (`scripts/generate-performance-baselines.sh`)
   - Creates performance baselines from actual benchmark runs
   - Supports multiple platforms and configurations
   - Generates reproducible, hardware-specific baselines

5. **Benchmark Comparison** (`benchmark_comparison.py`)
   - Direct performance comparison between Rust and C++ implementations
   - Measures end-to-end inference performance
   - Validates correctness of generated outputs

### Integration Points

- **GitHub Actions**: Automated performance tracking via `.github/workflows/performance-tracking.yml`
- **Criterion Integration**: Structured benchmark data collection and analysis
- **Cross-validation**: Systematic comparison with C++ reference implementation
- **Baseline Management**: Version-controlled performance expectations in `crossval/baselines.json`

## 🔧 Configuration and Setup

### Environment Setup

The setup script configures a deterministic benchmarking environment:

```bash
# Basic setup with CPU features
./scripts/setup-perf-env.sh

# Setup with GPU features
./scripts/setup-perf-env.sh --features gpu

# Cross-compilation setup
./scripts/setup-perf-env.sh --cross --target aarch64-unknown-linux-gnu --use-cross

# Skip C++ cross-validation
./scripts/setup-perf-env.sh --skip-cpp
```

**Key Environment Variables (Issue #261 - Strict Mode):**
- `BITNET_STRICT_MODE=1`: **PRIMARY** - Prevent all mock inference fallbacks (essential for accurate performance)
- `BITNET_DETERMINISTIC=1`: Enable deterministic mode for reproducible results
- `BITNET_SEED=42`: Set random seed for reproducibility
- `RAYON_NUM_THREADS=1`: Single-threaded CPU execution for determinism
- `RUSTFLAGS="-C target-cpu=native -C opt-level=3"`: Maximum optimization
- `BITNET_CI_ENHANCED_STRICT=1`: Enhanced strict mode for CI environments (with `CI=1`)

### Test Fixtures

Test fixtures are automatically generated using the xtask system:

```bash
# Generate small test fixtures
cargo run --no-default-features -p xtask -- gen-fixtures --size small --output crossval/fixtures/

# Generate fixtures of different sizes
cargo run --no-default-features -p xtask -- gen-fixtures --size tiny --output crossval/fixtures/
cargo run --no-default-features -p xtask -- gen-fixtures --size medium --output crossval/fixtures/
```

## 📈 Running Benchmarks

### Comprehensive Benchmark Suite

The benchmark runner provides multiple execution modes:

```bash
# Standard CPU benchmarks
./scripts/run-performance-benchmarks.sh

# GPU benchmarks with extended timeout
./scripts/run-performance-benchmarks.sh --features gpu --timeout 600

# Cross-compilation benchmarks
./scripts/run-performance-benchmarks.sh \
  --target aarch64-unknown-linux-gnu \
  --use-cross \
  --features cpu

# High-precision benchmarks
./scripts/run-performance-benchmarks.sh \
  --iterations 10 \
  --tokens 64 \
  --timeout 900
```

### Benchmark Output

The benchmark runner generates structured output files:

- `benchmark-results/performance-report.json`: Machine-readable performance summary
- `benchmark-results/performance-report.md`: Human-readable performance report
- `benchmark-results/rust-results.json`: Criterion benchmark data
- `benchmark-results/comparison-results.json`: Rust vs C++ comparison results
- `benchmark-results/system-info.json`: System and environment information

### Performance Metrics

**Throughput Metrics:**
- Tokens per second for inference operations
- End-to-end generation throughput
- Comparison throughput (Rust vs C++)

**Latency Metrics:**
- P50, P95, P99 latency percentiles
- First token latency
- Model loading time
- Average inference time

**Resource Metrics:**
- Memory usage (MB)
- CPU utilization (%)
- GPU memory usage (if applicable)

**Quality Metrics:**
- Accuracy scores
- Correctness validation
- Output comparison results

## 🔍 Regression Detection

### Baseline Management

Performance baselines are stored in `crossval/baselines.json` and contain:

```json
{
  "baselines": {
    "linux-x86_64": {
      "rust_implementation": {
        "throughput_tokens_per_second": 125.3,
        "latency_p50_ms": 89.2,
        "memory_usage_mb": 1024.5,
        "accuracy_score": 0.9987
      }
    }
  },
  "thresholds": {
    "critical": {
      "throughput_decrease_percent": 15.0,
      "latency_increase_percent": 25.0
    },
    "warning": {
      "throughput_decrease_percent": 8.0,
      "latency_increase_percent": 15.0
    }
  }
}
```

### Generating New Baselines

Generate baselines from actual benchmark runs:

```bash
# Generate baseline for current platform
./scripts/generate-performance-baselines.sh

# Generate baselines for multiple platforms
./scripts/generate-performance-baselines.sh \
  --platforms linux-x86_64,linux-aarch64,macos-aarch64 \
  --iterations 10

# Quick baseline generation
./scripts/generate-performance-baselines.sh \
  --iterations 5 \
  --timeout 300
```

### Regression Analysis

The regression detector provides detailed analysis:

```bash
# Basic regression analysis
python3 scripts/detect-performance-regression.py \
  benchmark-results/performance-report.json

# Platform-specific analysis
python3 scripts/detect-performance-regression.py \
  benchmark-results/performance-report.json \
  --platform linux-aarch64

# JSON output for CI integration
python3 scripts/detect-performance-regression.py \
  benchmark-results/performance-report.json \
  --format json \
  --output regression-analysis.json

# Fail on critical regressions
python3 scripts/detect-performance-regression.py \
  benchmark-results/performance-report.json \
  --fail-on-regression
```

**Alert Levels:**
- 🚨 **CRITICAL**: Performance degradation exceeding critical thresholds
- ⚠️ **WARNING**: Performance degradation exceeding warning thresholds
- ✅ **IMPROVEMENT**: Performance improvements detected
- ℹ️ **STABLE**: Performance within acceptable ranges

## 🔄 CI/CD Integration

### Automated Performance Tracking

The performance tracking workflow (`.github/workflows/performance-tracking.yml`) provides:

- **Scheduled Runs**: Daily performance tracking at 4 AM UTC
- **Push Triggers**: Performance analysis on critical path changes
- **Manual Triggers**: On-demand baseline updates and analysis
- **Multi-Platform**: Linux x86_64/ARM64, macOS x86_64/ARM64 support
- **Artifact Retention**: 90-day retention for benchmark results

### Workflow Features

**Environment Setup:**
- Automated Rust toolchain configuration
- Cross-compilation tool installation
- Test fixture generation
- Optimized binary compilation

**Benchmark Execution:**
- Deterministic environment configuration
- Timeout protection against stalled benchmarks
- Graceful fallback for failed benchmarks
- Comprehensive result collection

**Analysis and Reporting:**
- Automated regression detection
- Performance comparison with baselines
- GitHub issue creation for critical regressions
- Artifact upload for detailed analysis

### GitHub Actions Configuration

```yaml
# Trigger performance tracking
name: Performance Baseline Tracking
on:
  schedule:
    - cron: '0 4 * * *'  # Daily at 4 AM UTC
  workflow_dispatch:
    inputs:
      update_baselines:
        description: 'Update baseline performance numbers'
        type: boolean
      platform_filter:
        description: 'Platform to test (all, linux, macos)'
        type: choice
        options: [all, linux, macos]
```

## 🛠️ Advanced Usage

### Custom Benchmark Development

Create custom benchmarks using the Criterion framework:

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use bitnet_inference::Engine;

fn benchmark_inference(c: &mut Criterion) {
    let engine = Engine::new("path/to/model.gguf").unwrap();

    c.bench_function("inference_benchmark", |b| {
        b.iter(|| {
            engine.generate("Test prompt", 32)
        })
    });
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
```

### Cross-Platform Testing

Test performance across different architectures:

```bash
# Linux ARM64 (requires cross)
./scripts/run-performance-benchmarks.sh \
  --target aarch64-unknown-linux-gnu \
  --use-cross

# macOS ARM64 (M1/M2)
./scripts/run-performance-benchmarks.sh \
  --target aarch64-apple-darwin

# WebAssembly (requires wasm32 target)
rustup target add wasm32-unknown-unknown
cargo bench --no-default-features --features cpu -p bitnet-wasm --target wasm32-unknown-unknown
```

### Performance Profiling

Integrate with profiling tools for detailed analysis:

```bash
# CPU profiling with perf
RUSTFLAGS="-C force-frame-pointers=yes" \
perf record --call-graph=dwarf \
./scripts/run-performance-benchmarks.sh

# Memory profiling with Valgrind
valgrind --tool=massif \
./scripts/run-performance-benchmarks.sh

# GPU profiling with NVIDIA Nsight
nsys profile ./scripts/run-performance-benchmarks.sh --features gpu
```

## 📋 Troubleshooting

### Common Issues

**Environment Setup Failures:**
```bash
# Check Rust toolchain
rustc --version
cargo --version

# Verify test fixtures
ls -la crossval/fixtures/

# Check binary compilation
cargo build --release --no-default-features --features cpu
```

**Benchmark Failures:**
```bash
# Run with verbose output
./scripts/run-performance-benchmarks.sh --timeout 60 2>&1 | tee benchmark.log

# Check system resources
free -h
df -h
```

**Regression Detection Issues:**
```bash
# Verify baselines file
python3 -c "import json; print(json.load(open('crossval/baselines.json'))['version'])"

# Manual regression check
python3 scripts/detect-performance-regression.py \
  benchmark-results/performance-report.json \
  --format human
```

### Performance Debugging

**Slow Benchmarks:**
- Reduce iteration count: `--iterations 3`
- Decrease token count: `--tokens 16`
- Increase timeout: `--timeout 600`
- Skip C++ comparison: `--skip-cpp`

**Inconsistent Results:**
- Verify deterministic mode: `BITNET_DETERMINISTIC=1`
- Check CPU frequency scaling
- Ensure single-threaded execution: `RAYON_NUM_THREADS=1`
- Run multiple iterations and average results

**Cross-Compilation Issues:**
- Install cross-compilation tools: `cargo install cross`
- Check target availability: `rustup target list --installed`
- Verify target binary: `file target/aarch64-unknown-linux-gnu/release/bitnet-cli`

## 📚 Best Practices

### Baseline Generation

1. **Clean Environment**: Generate baselines on clean commits with stable performance
2. **Representative Hardware**: Use hardware representative of production environments
3. **Multiple Runs**: Run baseline generation multiple times and average results
4. **Documentation**: Document hardware specifications and environmental conditions
5. **Version Control**: Commit baseline updates with detailed commit messages

### Continuous Monitoring

1. **Regular Updates**: Update baselines monthly or after significant performance changes
2. **Threshold Tuning**: Adjust regression thresholds based on historical variance
3. **Platform Coverage**: Maintain baselines for all supported platforms
4. **Alert Management**: Configure appropriate alerts for critical regressions
5. **Historical Analysis**: Track performance trends over time

### Performance Optimization

1. **Profile First**: Use profiling tools to identify performance bottlenecks
2. **Measure Impact**: Quantify performance impact of optimizations
3. **Regression Testing**: Verify optimizations don't cause regressions elsewhere
4. **Documentation**: Document optimization techniques and their impact
5. **Benchmarking**: Add benchmarks for critical performance paths

## 🔗 Related Documentation

- [GPU Development Guide](development/gpu-development.md): GPU-specific performance considerations
- [Test Suite Guide](development/test-suite.md): Comprehensive testing framework
- [CLAUDE.md](../CLAUDE.md): Complete development commands and workflows
- [Concurrency Caps Guide](concurrency-caps.md): Resource management for performance testing

## 📊 Performance Targets

### Current Performance Baselines (Linux x86_64)

**IMPORTANT: Receipts Over Claims**

All performance metrics below are backed by receipt artifacts in `ci/inference.json`. No performance claims are made without verifiable evidence.

**Validated CPU Baselines (Issue #254 - Real Neural Network Inference)**

Receipt: [ci/inference.json](../ci/inference.json)

**I2S BitNet32-F16 (Production Recommended):**
- **Throughput**: SIMD-optimised (hardware-dependent, not yet benchmarked)
- **First Token Latency**: 250ms
- **Average Token Latency**: 50-100ms
- **Memory Usage**: 1024MB for 2B parameter model
- **Compute Path**: Real quantized GEMV (no FP32 staging)
- **Kernels**: `i2s_gemv`, `rope_apply`, `attention_real`
- **Accuracy**: MSE ≤ 8.5e-6 vs FP32 (tolerance: 1e-5)
- **Deterministic**: Yes (BITNET_DETERMINISTIC=1, seed=42)
- **Environment**: RAYON_NUM_THREADS=1 for reproducibility
- **Status**: ✅ Working (SIMD-optimised)

**I2S QK256 (GGML) - MVP Scalar Kernels:**
- **Throughput**: ~0.1 tokens/sec (scalar implementation)
- **First Token Latency**: ~10 seconds
- **Average Token Latency**: ~10 seconds per token
- **Memory Usage**: 1024MB for 2B parameter model
- **Compute Path**: Real quantized GEMV (scalar, no SIMD)
- **Kernels**: `qk256_scalar_dequant`, `qk256_scalar_matmul`
- **Accuracy**: Bit-exact with GGML reference
- **Status**: ⚠️ MVP validation only (scalar kernels)
- **Roadmap**: v0.2.0 targets ≥3× with AVX2 (nibble-LUT + FMA tiling)
- **Recommendation**: Limit to `--max-tokens 4-16` for quick validation

**TL1 Quantization (Table Lookup - NEON optimized):**
- **Throughput**: 18.2 Melem/s (matrix elements per second)
- **Accuracy**: MSE ≤ 7.2e-5 vs FP32 (tolerance: 1e-4)
- **Compute Path**: Real table lookup matmul
- **Device**: CPU (NEON vectorization on ARM, scalar fallback on x86)
- **Status**: 🚧 Experimental

**TL2 Quantization (Table Lookup - AVX optimized):**
- **Throughput**: 0.58 Melem/s (baseline, optimization ongoing)
- **Accuracy**: MSE ≤ 9.8e-5 vs FP32 (tolerance: 1e-4)
- **Compute Path**: Real table lookup matmul
- **Device**: CPU (AVX2/AVX-512 on x86)
- **Status**: 🚧 Experimental

### Reproducing Performance Benchmarks

All benchmarks can be reproduced with deterministic configuration:

```bash
# CPU I2S BitNet32-F16 benchmark (production quantization)
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run --no-default-features -p xtask -- benchmark --features cpu --quantization i2s

# QK256 scalar benchmark (validation only - SLOW)
# WARNING: This will take ~20 minutes for 128 tokens
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "Test" \
  --max-tokens 8  # Keep small for QK256

# TL1 benchmark (ARM NEON optimized)
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo test --no-default-features --features cpu -p bitnet-kernels test_tl1_kernel_accuracy_envelope

# TL2 benchmark (x86 AVX optimized)
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo test --no-default-features --features cpu -p bitnet-kernels test_tl2_kernel_accuracy_envelope

# Verify receipt artifact
cat ci/inference.json | jq '.compute_path' # Must be "real"
cat ci/inference.json | jq '.performance_baseline'
```

### QK256 Performance Roadmap

**Current State (v0.1.0):**
- Scalar implementation: ~0.1 tok/s
- Correctness-first approach for MVP validation
- AVX2 foundation established with ~1.2× uplift

**Optimization Plan (v0.2.0):**

| Optimization | Expected Impact | Status |
|--------------|-----------------|--------|
| Nibble-LUT unpack (`pshufb`) | 1.5-2× | Planned |
| FMA tiling (8-16 rows) | 1.5-2× | Planned |
| Load combine (reduce crossings) | 1.2-1.3× | Planned |
| Prefetch (code + input) | 1.1-1.2× | Planned |
| **Combined Target** | **≥3×** (0.3+ tok/s) | v0.2.0 |

**Long-term Target (v0.3.0+):**
- AVX-512 support: Additional 1.5-2× over AVX2
- Multi-threading: Linear scaling with cores
- GPU implementation: 50-100× over scalar CPU

**Benchmarking QK256 Optimizations:**
```bash
# Benchmark scalar baseline
cargo bench --no-default-features --features cpu --bench kernel_benchmarks -- qk256_scalar

# Benchmark AVX2 implementation (when available)
cargo bench --no-default-features --features cpu,avx2 --bench kernel_benchmarks -- qk256_avx2

# Compare scalar vs AVX2
cargo bench --no-default-features --features cpu,avx2 --bench kernel_benchmarks -- qk256
```

### Receipt Artifact Schema

Performance baselines are validated against the receipt schema defined in Issue #254:

```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-03T00:00:00Z",
  "compute_path": "real",
  "backend": "cpu",
  "kernels": ["i2s_gemv", "rope_apply", "attention_real"],
  "deterministic": true,
  "performance_baseline": {
    "tokens_generated": 100,
    "total_time_ms": 5000,
    "tokens_per_second": 20.0,
    "first_token_latency_ms": 250,
    "average_token_latency_ms": 50,
    "memory_usage_mb": 1024
  }
}
```

### Performance Validation Gates

CI enforces strict validation:

1. **compute_path** MUST be "real" (not "mock")
2. **kernels** MUST NOT contain "mock" entries
3. **accuracy_tests** MUST pass tolerance checks (I2S: 1e-5, TL1/TL2: 1e-4)
4. **determinism_tests** MUST show identical sequences across runs

See `.github/workflows/performance-tracking.yml` for CI gate implementation.

### Cross-Validation Status

**C++ Reference Parity:**
- **I2S Tolerance**: 1e-5 MSE
- **Validation Command**: `cargo run --no-default-features -p xtask -- crossval`
- **Status**: Available when BITNET_GGUF environment variable set

These targets are automatically validated through the regression detection system and receipt artifact generation. All claims must be backed by verifiable receipts.

## 🔒 Strict Mode Performance Testing (Issue #261)

### Overview

BitNet-rs Issue #261 implemented comprehensive strict mode controls to eliminate mock inference paths and ensure accurate performance reporting. All performance benchmarks MUST use strict mode to prevent false positives from mock computation.

### Strict Mode Environment Variables

| Variable | Purpose | Usage |
|----------|---------|-------|
| `BITNET_STRICT_MODE=1` | Primary strict mode - prevents ALL mock fallbacks | **Required for all production benchmarks** |
| `BITNET_STRICT_FAIL_ON_MOCK=1` | Fail immediately on mock detection | Activated by `BITNET_STRICT_MODE=1` |
| `BITNET_STRICT_REQUIRE_QUANTIZATION=1` | Require real I2S/TL1/TL2 kernels | Activated by `BITNET_STRICT_MODE=1` |
| `BITNET_STRICT_VALIDATE_PERFORMANCE=1` | Validate realistic performance metrics | Activated by `BITNET_STRICT_MODE=1` |
| `BITNET_CI_ENHANCED_STRICT=1` | Enhanced CI validation with comprehensive logging | Use in CI with `CI=1` |

### Strict Mode Benchmark Examples

```bash
# CPU baseline with strict mode (I2S quantization, SIMD-optimised)
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo run --no-default-features -p xtask -- benchmark --features cpu --quantization i2s

# GPU baseline with strict mode (mixed precision, GPU-accelerated alpha)
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
cargo run --no-default-features -p xtask -- benchmark --features gpu --quantization i2s

# Cross-validation with strict mode (validates ≥99.8% I2S accuracy)
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo run --no-default-features -p xtask -- crossval

# CI enhanced strict mode (comprehensive validation)
CI=1 \
BITNET_CI_ENHANCED_STRICT=1 \
BITNET_STRICT_MODE=1 \
cargo test --workspace --no-default-features --features cpu
```

### Performance Validation Thresholds

Strict mode enforces realistic performance expectations:

| Quantization | CPU Performance | GPU Performance | Accuracy Target |
|--------------|----------------|-----------------|-----------------|
| I2S (2-bit) | SIMD-optimised (AVX-512 > AVX2 > NEON) | GPU-accelerated (alpha, FP16/BF16) | ≥99.8% vs FP32 |
| TL1 (table lookup) | SIMD-optimised (ARM NEON) | N/A | ≥99.6% vs FP32 |
| TL2 (table lookup) | SIMD-optimised (x86 AVX) | N/A | ≥99.6% vs FP32 |

**Validation Rules:**
- Performance >150 tok/s flagged as potentially mock computation
- Computation type MUST be `Real` (not `Mock`)
- Accuracy MUST meet thresholds in cross-validation
- GPU utilization MUST be >80% for GPU benchmarks

### Strict Mode Test Coverage

```bash
# Unit tests for strict mode enforcement (AC2)
cargo test --no-default-features --features cpu -p bitnet-common test_strict_mode_from_env_detailed
cargo test --no-default-features --features cpu -p bitnet-common test_strict_mode_ci_enhanced

# Integration tests for real quantization (AC3)
BITNET_STRICT_MODE=1 \
cargo test -p bitnet-quantization --no-default-features --features cpu test_i2s_simd_scalar_parity

# Performance validation tests (AC7, AC8)
BITNET_STRICT_MODE=1 \
cargo test -p bitnet-kernels --no-default-features --features cpu test_cpu_performance_baselines
BITNET_STRICT_MODE=1 \
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_performance_baselines

# CI mock rejection tests (AC6)
CI=1 BITNET_CI_ENHANCED_STRICT=1 BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu test_ci_enhanced_strict_mode_comprehensive
```

### Receipts and Evidence

All performance claims MUST be backed by verifiable receipts:

1. **Receipt Artifacts**: `ci/inference.json` contains validated performance baselines
2. **Computation Path**: MUST be `"real"` (not `"mock"`)
3. **Kernel Usage**: Real quantization kernels (`i2s_gemv`, `rope_apply`, `attention_real`)
4. **Determinism**: Identical sequences across runs with same seed

**Example Receipt Validation:**
```bash
# Verify receipt shows real computation
cat ci/inference.json | jq '.compute_path'
# Expected: "real"

# Verify realistic performance
cat ci/inference.json | jq '.performance_baseline.tokens_per_second'
# Expected: 10-20 for CPU, 50-100 for GPU

# Verify no mock kernels
cat ci/inference.json | jq '.kernels | map(select(contains("mock")))'
# Expected: []
```

### CI Integration

```yaml
# .github/workflows/performance-tracking.yml
- name: Run strict mode benchmarks
  env:
    BITNET_STRICT_MODE: "1"
    BITNET_CI_ENHANCED_STRICT: "1"
    BITNET_DETERMINISTIC: "1"
    BITNET_SEED: "42"
    CI: "1"
  run: |
    # Setup and run benchmarks
    ./scripts/setup-perf-env.sh
    ./scripts/run-performance-benchmarks.sh --features cpu

    # Validate receipts
    python3 scripts/validate-performance-receipts.py ci/inference.json

    # Detect regressions
    python3 scripts/detect-performance-regression.py \
      benchmark-results/performance-report.json \
      --fail-on-regression
```

### Troubleshooting Strict Mode Failures

**Error: "Strict mode: Mock computation detected"**
- Cause: Inference path using mock fallback instead of real quantization
- Solution: Ensure quantization kernels are properly integrated and feature flags are correct

**Error: "Strict mode: Suspicious performance detected: X tok/s"**
- Cause: Performance >150 tok/s suggests mock computation
- Solution: Verify real quantization kernels are being used, check computation path in receipts

**Error: "Strict mode: Required quantization kernel not available"**
- Cause: I2S/TL1/TL2 kernel not available for current device
- Solution: Build with correct feature flags (`--features cpu` or `--features gpu`)

For more information, see:
- [Environment Variables](environment-variables.md) - Complete strict mode variable documentation
- [Quantization Support](reference/quantization-support.md) - Quantization accuracy and performance details
- [Issue #261 Specification](explanation/specs/issue-261-mock-performance-reporting-elimination-spec.md) - Technical implementation details
