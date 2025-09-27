# Cross-Validation Tools

This directory contains tools and documentation for cross-validating BitNet.rs against the legacy C++ implementation. These tools ensure numerical accuracy and performance parity during development and testing.

> **Note**: Cross-validation is primarily used for development and testing. For production use, BitNet.rs is the recommended implementation due to its superior performance and safety guarantees.

## Overview

Cross-validation serves several purposes:

- **Numerical Accuracy**: Verify that BitNet.rs produces identical outputs to the legacy implementation
- **Performance Benchmarking**: Compare speed and memory usage between implementations
- **Regression Testing**: Detect when changes affect compatibility
- **Migration Validation**: Ensure smooth migration from legacy implementations

## Tools and Scripts

### Core Cross-Validation Framework

Located in [`/crossval/`](../../crossval/):

- **`crossval/src/`** - Cross-validation library implementation
- **`crossval/tests/`** - Token equivalence and accuracy tests
- **`crossval/benches/`** - Performance benchmarking suite
- **`crossval/fixtures/`** - Test models and datasets

### CI/CD Scripts

Located in [`/ci/`](../../ci/):

- **`fetch_bitnet_cpp.sh/.ps1`** - Download and build legacy C++ implementation
- **`apply_patches.sh/.ps1`** - Apply minimal patches if needed
- **`bump_bitnet_tag.sh/.ps1`** - Version management for C++ dependency

### FFI Bindings

Located in [`/crates/bitnet-sys/`](../../crates/bitnet-sys/):

- **Low-level FFI bindings** to C++ implementation
- **Safe wrappers** around unsafe C++ calls
- **Feature-gated compilation** (only when `crossval` feature is enabled)

## Comparison Methodology

### 1. Numerical Accuracy Testing

Cross-validation ensures bit-exact compatibility between implementations:

```rust
// Example accuracy test
#[test]
fn test_token_equivalence() {
    let rust_tokens = bitnet_rs::generate(&model, prompt);
    let cpp_tokens = bitnet_cpp::generate(&model, prompt);
    
    // Verify exact token match
    assert_eq!(rust_tokens, cpp_tokens);
}
```

**Validation Criteria:**
- **Token-level matching**: Outputs must be identical
- **Floating-point tolerance**: Within 1e-6 for numerical operations
- **Model format compatibility**: Same model produces same results
- **Deterministic behavior**: Identical inputs produce identical outputs

### 2. Performance Benchmarking

Comprehensive performance comparison across multiple dimensions:

```rust
// Example performance benchmark
fn benchmark_inference(c: &mut Criterion) {
    let model = load_test_model();
    
    c.bench_function("rust_inference", |b| {
        b.iter(|| bitnet_rs::generate(&model, "test prompt"))
    });
    
    c.bench_function("cpp_inference", |b| {
        b.iter(|| bitnet_cpp::generate(&model, "test prompt"))
    });
}
```

**Benchmark Metrics:**
- **Inference Speed**: Tokens per second
- **Memory Usage**: Peak and average memory consumption
- **Cold Start Time**: Time to first token
- **Batch Processing**: Throughput with multiple requests
- **Model Loading**: Time to load and initialize models

### 3. API Compatibility Testing

Ensures migration compatibility:

```rust
// Example API compatibility test
#[test]
fn test_api_compatibility() {
    // Test that Rust API produces same results as C++ API
    let rust_result = bitnet_rs::Model::load("test.gguf")?.generate("test");
    let cpp_result = bitnet_cpp_api::load_model("test.gguf").generate("test");
    
    assert_eq!(rust_result, cpp_result);
}
```

## Usage Instructions

### Prerequisites

1. **Enable cross-validation features**:
   ```bash
   # Required for cross-validation
   cargo build --features crossval
   ```

2. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt install clang cmake build-essential
   
   # macOS
   xcode-select --install
   brew install cmake
   
   # Windows
   # Install Visual Studio with C++ tools and CMake
   ```

3. **Download C++ implementation**:
   ```bash
   # Automatic setup
   ./ci/fetch_bitnet_cpp.sh
   
   # Or manual setup
   export BITNET_CPP_PATH=/path/to/bitnet.cpp
   ```

### Running Cross-Validation Tests

#### Basic Accuracy Tests

```bash
# Run all cross-validation tests
cargo test --features crossval

# Run specific test categories
cargo test --features crossval token_equivalence
cargo test --features crossval numerical_accuracy
cargo test --features crossval api_compatibility
```

#### Performance Benchmarks

```bash
# Run all benchmarks
cargo bench --features crossval

# Run specific benchmark categories
cargo bench --features crossval inference_speed
cargo bench --features crossval memory_usage
cargo bench --features crossval model_loading
```

#### Comprehensive Validation

```bash
# Full validation suite (takes longer)
cargo test --features crossval --release
cargo bench --features crossval --release

# Generate detailed reports
cargo bench --features crossval -- --output-format html
```

### Custom Test Fixtures

Create custom test cases for your specific models:

1. **Add test model** to `crossval/fixtures/`:
   ```json
   {
     "name": "my_custom_model",
     "model_path": "my_model.gguf",
     "test_prompts": [
       "Custom test prompt 1",
       "Custom test prompt 2"
     ]
   }
   ```

2. **Run tests with custom fixture**:
   ```bash
   cargo test --features crossval -- --test-fixture my_custom_model
   ```

## Interpreting Results

### Accuracy Test Results

**✅ PASS - Token Equivalence**
```
[PASS] token_equivalence_test: Rust=150 tokens, C++=150 tokens
✓ All tokens match exactly
```

**❌ FAIL - Token Mismatch**
```
[FAIL] token_equivalence_test: Token mismatch at position 42
  Rust token: 1234
  C++ token:  1235
  Difference: 1
```

**Action**: Investigate numerical precision or algorithm differences

### Performance Benchmark Results

**Example Output**:
```
Benchmark Results:
┌─────────────────┬─────────────┬─────────────┬─────────────┐
│ Test            │ BitNet.rs   │ BitNet C++  │ Improvement │
├─────────────────┼─────────────┼─────────────┼─────────────┤
│ Inference Speed │ 1,250 tok/s │ 520 tok/s   │ 2.4x faster │
│ Memory Usage    │ 2.1 GB      │ 3.2 GB      │ 34% less    │
│ Cold Start      │ 0.8s        │ 2.1s        │ 2.6x faster │
│ Model Loading   │ 1.2s        │ 4.5s        │ 3.8x faster │
└─────────────────┴─────────────┴─────────────┴─────────────┘
```

**Interpretation**:
- **Green numbers**: BitNet.rs is faster/more efficient
- **Red numbers**: Potential regression (investigate)
- **Yellow numbers**: Within expected variance

### Memory Usage Analysis

```
Memory Usage Comparison:
┌─────────────────┬─────────────┬─────────────┐
│ Component       │ BitNet.rs   │ BitNet C++  │
├─────────────────┼─────────────┼─────────────┤
│ Model Weights   │ 1.2 GB      │ 1.2 GB      │
│ KV Cache        │ 512 MB      │ 768 MB      │
│ Working Memory  │ 256 MB      │ 1.1 GB      │
│ Overhead        │ 64 MB       │ 312 MB      │
├─────────────────┼─────────────┼─────────────┤
│ Total           │ 2.0 GB      │ 3.4 GB      │
└─────────────────┴─────────────┴─────────────┘
```

## Continuous Integration

### Automated Cross-Validation

Cross-validation runs automatically in CI:

```yaml
# .github/workflows/crossval.yml
name: Cross-Validation
on: [push, pull_request]

jobs:
  crossval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup C++ implementation
        run: ./ci/fetch_bitnet_cpp.sh
      - name: Run cross-validation
        run: cargo test --features crossval
      - name: Run benchmarks
        run: cargo bench --features crossval
```

### Nightly Validation

Extended validation runs nightly:

```yaml
# .github/workflows/nightly-crossval.yml
name: Nightly Cross-Validation
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  extended-crossval:
    runs-on: [ubuntu-latest, windows-latest, macos-latest]
    steps:
      - name: Extended model testing
        run: cargo test --features crossval --release -- --include-slow-tests
      - name: Performance regression detection
        run: cargo bench --features crossval --baseline
```

## Troubleshooting

### Common Issues

#### 1. C++ Implementation Not Found

**Error**:
```
ERROR: BitNet C++ implementation not found at ~/.cache/bitnet_cpp
```

**Solution**:
```bash
# Download and build C++ implementation
./ci/fetch_bitnet_cpp.sh

# Or set custom path
export BITNET_CPP_PATH=/path/to/bitnet.cpp
```

#### 2. Clang Not Available

**Error**:
```
ERROR: clang not found - cannot generate C++ bindings
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install clang libclang-dev

# macOS
xcode-select --install

# Windows
# Install LLVM from https://llvm.org/
```

#### 3. Token Mismatch Errors

**Error**:
```
Token mismatch at position 15: Rust=1234, C++=1235
```

**Investigation Steps**:
1. Check model file integrity
2. Verify identical model loading parameters
3. Check for floating-point precision differences
4. Review recent algorithm changes

#### 4. Performance Regression

**Error**:
```
Performance regression detected: 15% slower than baseline
```

**Investigation Steps**:
1. Profile the specific slow operation
2. Check for debug builds (use `--release`)
3. Verify system resources and load
4. Compare with historical benchmarks

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Enable debug logging
RUST_LOG=debug cargo test --features crossval

# Enable trace logging for specific modules
RUST_LOG=bitnet_crossval=trace cargo test --features crossval
```

### Performance Profiling

Profile cross-validation performance:

```bash
# CPU profiling
cargo install flamegraph
cargo flamegraph --features crossval --bench performance

# Memory profiling
cargo install heaptrack
heaptrack cargo bench --features crossval
```

## Best Practices

### 1. Regular Validation

- **Run cross-validation** before major releases
- **Include in CI/CD** for automatic validation
- **Monitor performance trends** over time
- **Validate with real models** from your use case

### 2. Test Coverage

- **Cover edge cases** with custom fixtures
- **Test different model sizes** and architectures
- **Include stress tests** with large inputs
- **Validate error conditions** and recovery

### 3. Performance Monitoring

- **Establish baselines** for performance metrics
- **Track regressions** with automated alerts
- **Profile regularly** to identify bottlenecks
- **Document performance characteristics** for different hardware

### 4. Documentation

- **Document test procedures** for your team
- **Maintain fixture descriptions** and expected results
- **Record performance baselines** and acceptable variance
- **Update troubleshooting guides** with new issues

## Contributing

### Adding New Tests

1. **Create test fixture** in `crossval/fixtures/`
2. **Implement test logic** in `crossval/tests/`
3. **Add benchmark** in `crossval/benches/`
4. **Update documentation** with new test description

### Improving Performance

1. **Profile existing code** to identify bottlenecks
2. **Implement optimizations** with feature flags
3. **Validate improvements** with benchmarks
4. **Document performance characteristics**

### Reporting Issues

When reporting cross-validation issues:

1. **Include system information** (OS, hardware, versions)
2. **Provide reproduction steps** with minimal example
3. **Attach benchmark results** if performance-related
4. **Include logs** with debug information

## Future Improvements

### Planned Features

- **Distributed benchmarking** across multiple machines
- **Historical performance tracking** with trend analysis
- **Automated regression detection** with bisection
- **Model-specific optimization** recommendations

### Research Areas

- **Numerical precision analysis** for different quantization methods
- **Performance modeling** for hardware prediction
- **Automated test generation** from model characteristics
- **Cross-platform performance** optimization

---

**Need help with cross-validation?** Visit our [GitHub Discussions](https://github.com/microsoft/BitNet/discussions) or check the [troubleshooting guide](../../docs/troubleshooting.md).