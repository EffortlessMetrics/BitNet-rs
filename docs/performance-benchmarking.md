# Performance Benchmarking and Regression Detection

BitNet.rs includes a comprehensive performance benchmarking infrastructure designed to detect performance regressions, track improvements, and ensure consistent performance across platforms.

## 🚀 Quick Start

### Running Basic Benchmarks

```bash
# Setup environment and run CPU benchmarks
./scripts/setup-perf-env.sh
./scripts/run-performance-benchmarks.sh

# Run GPU benchmarks (requires CUDA)
./scripts/run-performance-benchmarks.sh --features gpu

# Run with cross-validation against C++ implementation
./scripts/run-performance-benchmarks.sh --include-cpp
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

**Key Environment Variables:**
- `BITNET_DETERMINISTIC=1`: Enable deterministic mode
- `BITNET_SEED=42`: Set random seed for reproducibility
- `RAYON_NUM_THREADS=1`: Single-threaded CPU execution
- `RUSTFLAGS="-C target-cpu=native -C opt-level=3"`: Maximum optimization

### Test Fixtures

Test fixtures are automatically generated using the xtask system:

```bash
# Generate small test fixtures
cargo run -p xtask -- gen-fixtures --size small --output crossval/fixtures/

# Generate fixtures of different sizes
cargo run -p xtask -- gen-fixtures --size tiny --output crossval/fixtures/
cargo run -p xtask -- gen-fixtures --size medium --output crossval/fixtures/
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
cargo bench -p bitnet-wasm --target wasm32-unknown-unknown
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

**Rust Implementation:**
- Throughput: 125+ tokens/second
- Latency P50: <90ms
- Memory Usage: <1.1GB
- Accuracy: >99.8%

**Performance Ratios (Rust vs C++):**
- Throughput: 1.15x faster
- Memory Efficiency: 11% less memory
- Load Time: 34% faster startup

These targets are automatically validated through the regression detection system and updated as performance improvements are made.