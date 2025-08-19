# Cross-Validation Setup and Usage Guide

## Introduction

Cross-validation in BitNet.rs ensures that the Rust implementation produces results consistent with the reference C++ implementation. This guide covers setup, configuration, and usage of the cross-validation framework.

## Overview

The cross-validation system compares:
- **Accuracy**: Token-level output comparison between implementations
- **Performance**: Speed and memory usage comparison
- **Behavior**: Error handling and edge case behavior

## Prerequisites

### System Requirements

- Rust toolchain (1.89+)
- C++ compiler (GCC 9+ or Clang 10+)
- CMake 3.15+
- Git (for submodule management)
- At least 8GB RAM for larger models
- 10GB free disk space for test fixtures

### C++ Implementation Setup

1. **Initialize the C++ submodule:**
```bash
git submodule update --init --recursive
```

2. **Build the C++ implementation:**
```bash
cd legacy/bitnet.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

3. **Verify C++ build:**
```bash
./bitnet_cpp --version
```

### Environment Configuration

Create a test configuration file:

```bash
# Create test configuration
cp tests/config.example.toml tests/config.toml
```

Edit `tests/config.toml`:

```toml
[crossval]
enabled = true
cpp_binary_path = "legacy/bitnet.cpp/build/bitnet_cpp"
tolerance = { min_token_accuracy = 0.95, max_probability_divergence = 0.1 }
timeout = "300s"

[fixtures]
auto_download = true
cache_dir = "tests/cache"
max_cache_size = "10GB"

[reporting]
output_dir = "target/crossval-reports"
formats = ["html", "json"]
include_artifacts = true
```

## Basic Usage

### Running Cross-Validation Tests

```bash
# Run all cross-validation tests
cargo test --test crossval_tests

# Run specific cross-validation test
cargo test --test crossval_tests test_accuracy_comparison

# Run with verbose output
cargo test --test crossval_tests -- --nocapture

# Run with specific model
CROSSVAL_MODEL=small_test_model cargo test --test crossval_tests
```

### Environment Variables

Key environment variables for cross-validation:

```bash
# C++ binary location (overrides config)
export BITNET_CPP_BINARY="/path/to/bitnet_cpp"

# Test model selection
export CROSSVAL_MODEL="specific_model_name"

# Cache directory
export BITNET_TEST_CACHE="/path/to/cache"

# Tolerance settings
export CROSSVAL_MIN_ACCURACY="0.99"
export CROSSVAL_MAX_DIVERGENCE="0.05"

# Enable debug logging
export RUST_LOG="crossval=debug"
```

## Configuration

### Tolerance Settings

Configure comparison tolerances based on your requirements:

```toml
[crossval.tolerance]
# Minimum token-level accuracy (0.0 to 1.0)
min_token_accuracy = 0.95

# Maximum probability distribution divergence
max_probability_divergence = 0.1

# Maximum acceptable performance regression (ratio)
max_performance_regression = 2.0

# Floating-point comparison epsilon
float_epsilon = 1e-6

# Maximum allowed first mismatch position
max_mismatch_position = 1000
```

### Test Case Configuration

Define custom test cases:

```toml
[[crossval.test_cases]]
name = "simple_generation"
input = "Hello, how are you today?"
max_tokens = 50
temperature = 0.7

[[crossval.test_cases]]
name = "long_context"
input_file = "fixtures/long_context.txt"
max_tokens = 100
temperature = 0.1

[[crossval.test_cases]]
name = "edge_case_tokens"
input = "Special tokens: <|endoftext|> [MASK] <unk>"
max_tokens = 20
temperature = 0.0
```

### Model Configuration

Configure test models:

```toml
[[crossval.models]]
name = "small_test_model"
url = "https://example.com/models/small_model.gguf"
checksum = "sha256:abc123..."
size = "100MB"
description = "Small model for basic testing"

[[crossval.models]]
name = "large_test_model"
url = "https://example.com/models/large_model.gguf"
checksum = "sha256:def456..."
size = "2GB"
description = "Large model for comprehensive testing"
```

## Advanced Usage

### Custom Cross-Validation Tests

Create custom cross-validation tests:

```rust
use crate::crossval::{CrossValidationSuite, ComparisonTestCase, ComparisonTolerance};

#[tokio::test]
async fn test_custom_scenario() {
    let tolerance = ComparisonTolerance {
        min_token_accuracy: 0.99,
        max_probability_divergence: 0.05,
        max_performance_regression: 1.5,
    };
    
    let mut suite = CrossValidationSuite::new(tolerance);
    
    // Add custom test case
    let test_case = ComparisonTestCase {
        name: "custom_scenario",
        input: "Your custom input text here",
        config: InferenceConfig {
            max_tokens: 100,
            temperature: 0.8,
            top_p: 0.9,
            repetition_penalty: 1.1,
            ..Default::default()
        },
    };
    
    suite.add_test_case(test_case);
    
    // Load model and run comparison
    let fixtures = FixtureManager::new();
    let model_path = fixtures.get_model_fixture("your_model").await?;
    
    let result = suite.run_comparison(&model_path).await?;
    
    // Custom validation
    assert!(result.summary.overall_accuracy >= 0.99);
    assert!(result.summary.performance_ratio <= 1.5);
    
    // Detailed analysis
    for test_result in &result.test_results {
        if let Some(mismatch) = &test_result.accuracy_result.first_mismatch {
            println!("First mismatch at position {}: Rust={}, C++={}", 
                    mismatch.position, mismatch.rust_token, mismatch.cpp_token);
        }
    }
}
```

### Batch Comparison

Run comparisons across multiple models:

```rust
#[tokio::test]
async fn test_multiple_models() {
    let models = vec![
        "small_test_model",
        "medium_test_model", 
        "large_test_model",
    ];
    
    let test_cases = vec![
        ComparisonTestCase {
            name: "standard_generation",
            input: "The quick brown fox",
            config: InferenceConfig::default(),
        },
        ComparisonTestCase {
            name: "creative_generation",
            input: "Once upon a time",
            config: InferenceConfig {
                temperature: 0.9,
                top_p: 0.8,
                ..Default::default()
            },
        },
    ];
    
    let fixtures = FixtureManager::new();
    let mut all_results = Vec::new();
    
    for model_name in models {
        let model_path = fixtures.get_model_fixture(model_name).await?;
        
        let mut suite = CrossValidationSuite::new(ComparisonTolerance::default());
        suite.add_test_cases(test_cases.clone());
        
        let result = suite.run_comparison(&model_path).await?;
        all_results.push((model_name, result));
    }
    
    // Analyze results across models
    for (model_name, result) in &all_results {
        println!("Model {}: Accuracy={:.3}, Performance Ratio={:.2}", 
                model_name, 
                result.summary.overall_accuracy,
                result.summary.performance_ratio);
    }
}
```

### Performance Profiling

Enable detailed performance profiling:

```rust
#[tokio::test]
async fn test_performance_profiling() {
    let mut suite = CrossValidationSuite::new(ComparisonTolerance::default());
    suite.enable_profiling(true);
    
    let test_case = ComparisonTestCase {
        name: "performance_test",
        input: "Generate a detailed explanation of machine learning",
        config: InferenceConfig {
            max_tokens: 200,
            ..Default::default()
        },
    };
    
    suite.add_test_case(test_case);
    
    let fixtures = FixtureManager::new();
    let model_path = fixtures.get_model_fixture("performance_model").await?;
    
    let result = suite.run_comparison(&model_path).await?;
    
    // Analyze performance metrics
    let rust_metrics = &result.rust_metrics;
    let cpp_metrics = &result.cpp_metrics;
    
    println!("Rust Performance:");
    println!("  Model Load Time: {:?}", rust_metrics.model_load_time);
    println!("  Inference Time: {:?}", rust_metrics.inference_time);
    println!("  Peak Memory: {} MB", rust_metrics.peak_memory / 1024 / 1024);
    println!("  Tokens/sec: {:.2}", rust_metrics.tokens_per_second);
    
    println!("C++ Performance:");
    println!("  Model Load Time: {:?}", cpp_metrics.model_load_time);
    println!("  Inference Time: {:?}", cpp_metrics.inference_time);
    println!("  Peak Memory: {} MB", cpp_metrics.peak_memory / 1024 / 1024);
    println!("  Tokens/sec: {:.2}", cpp_metrics.tokens_per_second);
    
    // Performance assertions
    assert!(rust_metrics.tokens_per_second > 0.0);
    assert!(cpp_metrics.tokens_per_second > 0.0);
    
    let performance_ratio = rust_metrics.inference_time.as_secs_f64() / 
                           cpp_metrics.inference_time.as_secs_f64();
    assert!(performance_ratio <= 2.0, "Rust should not be more than 2x slower");
}
```

## Troubleshooting

### Common Issues

#### 1. C++ Binary Not Found

**Error:** `CrossValidationError: C++ binary not found at path`

**Solutions:**
- Verify C++ implementation is built: `ls legacy/bitnet.cpp/build/bitnet_cpp`
- Check configuration: `cpp_binary_path` in `tests/config.toml`
- Set environment variable: `export BITNET_CPP_BINARY="/correct/path"`

#### 2. Model Download Failures

**Error:** `FixtureError: Download failed`

**Solutions:**
- Check internet connection
- Verify model URLs in configuration
- Clear cache: `rm -rf tests/cache`
- Use local models: Set `BITNET_TEST_CACHE` to local directory

#### 3. Accuracy Mismatches

**Error:** `Accuracy below threshold: 0.85 < 0.95`

**Investigation steps:**
1. Check first mismatch location in test output
2. Verify model compatibility between implementations
3. Check for floating-point precision issues
4. Review tokenization differences

**Solutions:**
- Adjust tolerance settings if differences are acceptable
- Update model to compatible version
- Fix implementation bugs causing differences

#### 4. Performance Regressions

**Error:** `Performance regression: 3.2x slower than baseline`

**Investigation steps:**
1. Profile both implementations
2. Check for debug builds (should use release)
3. Monitor memory usage patterns
4. Analyze algorithmic differences

**Solutions:**
- Optimize critical paths in Rust implementation
- Enable compiler optimizations
- Adjust performance expectations if necessary

#### 5. Timeout Issues

**Error:** `Test timed out after 300s`

**Solutions:**
- Increase timeout in configuration
- Use smaller models for testing
- Optimize test cases
- Check for infinite loops or deadlocks

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Enable debug logging
export RUST_LOG="crossval=debug,bitnet=debug"

# Run with debug output
cargo test --test crossval_tests -- --nocapture

# Save debug output to file
cargo test --test crossval_tests -- --nocapture 2>&1 | tee debug.log
```

### Collecting Diagnostics

When reporting issues, collect:

```bash
# System information
uname -a
rustc --version
gcc --version

# Build information
cargo --version
cmake --version

# Test configuration
cat tests/config.toml

# Recent test output
cargo test --test crossval_tests -- --nocapture 2>&1 | tail -100

# Cache status
ls -la tests/cache/
du -sh tests/cache/
```

## Continuous Integration

### GitHub Actions Integration

Add cross-validation to CI pipeline:

```yaml
name: Cross-Validation Tests

on: [push, pull_request]

jobs:
  crossval:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        submodules: recursive
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
    
    - name: Build C++ implementation
      run: |
        cd legacy/bitnet.cpp
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)
    
    - name: Cache test fixtures
      uses: actions/cache@v3
      with:
        path: tests/cache
        key: crossval-fixtures-${{ hashFiles('tests/fixtures.toml') }}
    
    - name: Run cross-validation tests
      run: cargo test --test crossval_tests
      env:
        BITNET_CPP_BINARY: legacy/bitnet.cpp/build/bitnet_cpp
        RUST_LOG: info
    
    - name: Upload test reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: crossval-reports
        path: target/crossval-reports/
```

### Scheduled Regression Testing

Set up nightly regression tests:

```yaml
name: Nightly Cross-Validation

on:
  schedule:
    - cron: '0 2 * * *'  # Run at 2 AM daily

jobs:
  comprehensive-crossval:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        model: [small, medium, large]
        
    steps:
    # ... setup steps ...
    
    - name: Run comprehensive cross-validation
      run: |
        cargo test --test crossval_tests test_comprehensive_${{ matrix.model }}
      env:
        CROSSVAL_MODEL: ${{ matrix.model }}_test_model
        CROSSVAL_MIN_ACCURACY: "0.99"
        RUST_LOG: debug
    
    - name: Report regressions
      if: failure()
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: 'Cross-validation regression detected',
            body: 'Nightly cross-validation failed for ${{ matrix.model }} model. Please investigate.',
            labels: ['bug', 'regression', 'cross-validation']
          });
```

## Best Practices

### 1. Model Selection
- Use representative models for your use case
- Include both small (fast) and large (comprehensive) models
- Test with different model formats (GGUF, SafeTensors)
- Regularly update test models

### 2. Test Case Design
- Cover common use cases and edge cases
- Include various input lengths and types
- Test different inference parameters
- Use deterministic settings for reproducibility

### 3. Tolerance Tuning
- Start with strict tolerances and relax as needed
- Document reasons for tolerance adjustments
- Monitor tolerance trends over time
- Use different tolerances for different scenarios

### 4. Performance Monitoring
- Track performance trends over time
- Set up alerts for significant regressions
- Profile both implementations regularly
- Consider hardware differences in CI

### 5. Maintenance
- Regularly update C++ implementation
- Keep test fixtures current
- Monitor and clean cache usage
- Update documentation with new findings

This guide provides comprehensive coverage of cross-validation setup and usage. Regular cross-validation ensures that the Rust implementation maintains compatibility and performance parity with the reference C++ implementation.