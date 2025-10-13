# BitNet-rs Performance Baseline & CI Gating System

This directory contains the performance baseline infrastructure for BitNet-rs, designed to prevent performance regressions and ensure consistent quality across releases.

## Overview

The baseline system provides:

- **Performance Baselines**: JSON files containing expected performance metrics
- **Regression Detection**: Automated comparison with configurable thresholds
- **CI Integration**: Commands that fail on performance regressions
- **Multiple Output Formats**: Human-readable, JSON, JUnit XML, and Markdown reports

## Directory Structure

```
benchmarks/
├── baseline/               # Performance baseline JSON files
│   ├── cpu/               # CPU-specific baselines
│   │   ├── quantization/  # Quantization benchmarks (I2S, TL1, TL2)
│   │   ├── inference/     # Inference engine benchmarks
│   │   └── kernels/       # Kernel-specific benchmarks
│   └── gpu/               # GPU-specific baselines
│       ├── quantization/  # GPU quantization with CUDA acceleration
│       ├── inference/     # GPU inference benchmarks
│       └── kernels/       # Mixed precision and GPU kernel benchmarks
├── thresholds/            # Threshold configuration files
│   └── default.toml       # Default regression thresholds
├── reports/               # Generated comparison reports
└── CI/                    # CI-specific configurations and scripts
```

## Quick Start

### 1. Compare Current Results Against Baseline

```bash
# Compare CPU quantization benchmarks (auto-detects baseline)
cargo run -p xtask -- bench-compare \
  --current target/criterion/quantization_sizes/report.json \
  --category quantization \
  --device cpu

# Compare with specific baseline and custom thresholds
cargo run -p xtask -- bench-compare \
  --current my_benchmark_results.json \
  --baseline benchmarks/baseline/cpu/quantization/i2s_baseline.json \
  --thresholds benchmarks/thresholds/strict.toml \
  --format markdown \
  --output report.md
```

### 2. CI Integration

```bash
# CI mode applies 1.5x multiplier to thresholds for environment variance
cargo run -p xtask -- bench-compare \
  --current $BENCHMARK_RESULTS \
  --category quantization \
  --ci \
  --format junit \
  --output junit-results.xml \
  --fail-on-regression
```

### 3. Generate New Baselines

```bash
# Run benchmarks and create new baseline
cargo bench -p bitnet-quantization --no-default-features --features cpu > /tmp/bench.log
cargo run -p xtask -- bench-compare \
  --current target/criterion/quantization_sizes/report.json \
  --baseline /dev/null \
  --format json \
  --output benchmarks/baseline/cpu/quantization/new_baseline.json
```

## Baseline Format

Baseline JSON files follow this structure:

```json
{
  "name": "I2S Quantization Baseline (CPU)",
  "version": "1.0.0",
  "created": "2025-01-01T00:00:00Z",
  "git": {
    "sha": "baseline",
    "branch": "main",
    "tag": "v0.1.0"
  },
  "environment": {
    "device": "cpu",
    "arch": "x86_64",
    "simd": "avx2",
    "rust_version": "1.90.0"
  },
  "benchmarks": {
    "quantization_sizes": {
      "I2S_quantize/1024": {
        "mean_ns": 15420,
        "std_ns": 1250,
        "throughput_elements_per_sec": 66389038,
        "regression_threshold_percent": 15.0
      }
    }
  },
  "metadata": {
    "description": "Baseline performance metrics for I2S quantization on CPU",
    "test_iterations": 100,
    "warmup_iterations": 10
  }
}
```

## Threshold Configuration

Thresholds are configured in TOML files with hierarchical precedence:

```toml
# benchmarks/thresholds/default.toml

[quantization]
i2s_cpu = 15.0      # 15% regression threshold for I2S CPU
tl1_cpu = 15.0      # 15% for TL1 CPU
tl2_cpu = 15.0      # 15% for TL2 CPU
i2s_gpu = 20.0      # 20% for I2S GPU (higher due to GPU variance)

[inference]
prefill_latency = 20.0     # Prefill operations
decode_latency = 15.0      # Token decode operations
end_to_end = 25.0          # Full inference pipeline

[ci]
multiplier = 1.5    # Apply 1.5x to all thresholds in CI environments

[overrides]
# Specific test overrides
"I2S_block_size" = 10.0    # Block size optimization is very stable
```

## Command Reference

### `cargo xtask bench-compare`

Compare benchmark results against baseline with regression detection.

#### Arguments

- `--current <PATH>`: Path to current benchmark results (required)
- `--baseline <PATH>`: Path to baseline JSON (auto-detected if not provided)
- `--device <DEVICE>`: Device type for baseline selection (cpu, gpu, auto) [default: auto]
- `--category <CATEGORY>`: Benchmark category (quantization, inference, kernels, all) [default: all]
- `--thresholds <PATH>`: Path to threshold configuration file
- `--format <FORMAT>`: Output format (human, json, junit, markdown) [default: human]
- `--output <PATH>`: Output file path (defaults to stdout)
- `--ci`: CI mode - apply CI threshold multipliers
- `--fail-on-regression`: Exit with error code on regression [default: true]
- `--verbose`: Verbose output with detailed comparison

#### Exit Codes

- `0`: Success - no regressions detected
- `17`: Benchmark failed - performance regressions detected
- Other codes: System errors (file not found, parse errors, etc.)

#### Examples

```bash
# Basic comparison with auto-detection
cargo xtask bench-compare --current results.json

# Specific device and category
cargo xtask bench-compare \
  --current cpu_results.json \
  --device cpu \
  --category quantization

# CI integration with JUnit output
cargo xtask bench-compare \
  --current $CI_BENCHMARK_RESULTS \
  --ci \
  --format junit \
  --output test-results.xml

# Custom thresholds and verbose output
cargo xtask bench-compare \
  --current results.json \
  --thresholds benchmarks/thresholds/strict.toml \
  --verbose \
  --format markdown \
  --output benchmark-report.md
```

## CI Integration

### GitHub Actions Example

```yaml
name: Performance Regression Check

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: 1.90.0
          default: true

      - name: Run Benchmarks
        run: |
          cargo bench -p bitnet-quantization \
            --no-default-features \
            --features cpu \
            -- --output-format json > benchmark_results.json

      - name: Check for Regressions
        run: |
          cargo run -p xtask -- bench-compare \
            --current benchmark_results.json \
            --category quantization \
            --device cpu \
            --ci \
            --format junit \
            --output junit-results.xml \
            --fail-on-regression

      - name: Upload Results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: benchmark-results
          path: |
            benchmark_results.json
            junit-results.xml
```

### Makefile Integration

```makefile
# Performance regression check
.PHONY: check-performance
check-performance:
    cargo bench --workspace --no-default-features --features cpu
    cargo run -p xtask -- bench-compare \
        --current target/criterion/report.json \
        --ci \
        --fail-on-regression

# Update baselines (run manually when performance changes are expected)
.PHONY: update-baselines
update-baselines:
    cargo bench --workspace --no-default-features --features cpu
    cargo run -p xtask -- bench-compare \
        --current target/criterion/quantization_sizes/report.json \
        --format json \
        --output benchmarks/baseline/cpu/quantization/i2s_baseline.json
```

## Best Practices

### 1. Baseline Management

- **Update baselines** only when intentional performance changes are made
- **Version baselines** with git tags to track changes over time
- **Document baseline updates** in commit messages with justification
- **Review baseline changes** carefully in pull requests

### 2. Threshold Selection

- **Start conservative** with 15-20% thresholds and tighten over time
- **Account for environment variance** with CI multipliers
- **Use different thresholds** for different benchmark types:
  - Compute-bound operations: 10-15%
  - Memory-bound operations: 15-20%
  - End-to-end workflows: 20-25%
  - GPU operations: 20-30% (higher variance)

### 3. CI Integration

- **Run benchmarks in consistent environments** (same hardware, OS)
- **Use stable runner configurations** to minimize noise
- **Fail CI on regressions** to prevent performance degradation
- **Generate reports** for manual review of borderline cases

### 4. Debugging Regressions

When regressions are detected:

1. **Review the changes** in the current commit/PR
2. **Check system load** during benchmark execution
3. **Validate reproducibility** by running benchmarks multiple times
4. **Compare detailed metrics** using verbose output
5. **Consider threshold adjustments** if variance is consistently higher

## Output Formats

### Human-Readable (Default)

```
📊 Benchmark Comparison Report
==============================

Baseline: I2S Quantization Baseline (CPU)
Current:  Current Benchmark Results

📈 Summary:
  Total tests: 25
  Regressions: 2
  Improvements: 3
  Stable: 20

🚨 Performance Regressions:
  ❌ quantization_sizes.I2S_quantize/16384.mean_ns: 18.5% regression (235600 → 279284) [threshold: 15.0%]
  ❌ quantization_sizes.I2S_quantize/65536.throughput_elements_per_sec: 22.1% regression (69518000 → 54123000) [threshold: 15.0%]

❌ Result: FAILED - Performance regressions detected
```

### JSON Format

```json
{
  "baseline_name": "I2S Quantization Baseline (CPU)",
  "current_name": "Current Benchmark Results",
  "has_regressions": true,
  "summary": {
    "total_tests": 25,
    "regressions_count": 2,
    "improvements_count": 3,
    "stable_count": 20
  },
  "regressions": [
    {
      "test_name": "quantization_sizes_I2S_quantize/16384",
      "metric_type": "mean_ns",
      "baseline_value": 235600.0,
      "current_value": 279284.0,
      "regression_percent": 18.5,
      "threshold_percent": 15.0
    }
  ]
}
```

### JUnit XML Format

```xml
<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="benchmark_comparison" tests="25" failures="2" errors="0">
  <testcase name="quantization_sizes_I2S_quantize/16384.mean_ns" classname="benchmark">
    <failure message="Performance regression: 18.50% (threshold: 15.0%)">
      Baseline: 235600.00, Current: 279284.00
    </failure>
  </testcase>
</testsuite>
```

### Markdown Format

Perfect for GitHub PR comments and documentation.

## Troubleshooting

### Common Issues

1. **Baseline not found**
   - Verify baseline files exist in `benchmarks/baseline/`
   - Check device and category arguments match directory structure
   - Use `--baseline` to specify path explicitly

2. **High variance in CI**
   - Increase CI multiplier in threshold configuration
   - Use dedicated CI runners with consistent hardware
   - Consider disabling CPU frequency scaling

3. **False positives**
   - Review threshold configuration
   - Check for system load during benchmarks
   - Use multiple runs and statistical analysis

4. **Missing dependencies**
   - Ensure `toml` crate is available for threshold parsing
   - Check feature flags for GPU/inference benchmarks

### Debug Commands

```bash
# Validate threshold parsing
cargo run -p xtask -- bench-compare \
  --current /dev/null \
  --baseline /dev/null \
  --thresholds benchmarks/thresholds/default.toml \
  --verbose

# Check baseline auto-detection
cargo run -p xtask -- bench-compare \
  --current /dev/null \
  --device cpu \
  --category quantization \
  --verbose

# Test output formats
cargo run -p xtask -- bench-compare \
  --current examples/sample_results.json \
  --baseline examples/sample_baseline.json \
  --format json
```

## Contributing

When adding new benchmarks or modifying existing ones:

1. **Update baselines** for affected categories
2. **Add appropriate thresholds** to configuration files
3. **Test CI integration** with your changes
4. **Document new benchmark categories** in this README
5. **Validate cross-platform compatibility** (CPU vs GPU, different architectures)

For questions or issues with the baseline system, please see the main BitNet-rs documentation or open an issue on GitHub.
