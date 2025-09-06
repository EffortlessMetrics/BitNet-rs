# Test Suite Guide

This document covers the comprehensive test suite for BitNet.rs, including running tests, configuration, and specialized testing strategies.

## Running Tests

### Basic Test Execution

```bash
# Run all tests with CPU features
cargo test --workspace --no-default-features --features cpu

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

### Convolution Tests

```bash
# Run convolution unit tests
cargo test -p bitnet-kernels --no-default-features --features cpu convolution

# Run PyTorch reference convolution tests (requires Python and PyTorch)
cargo test -p bitnet-kernels conv2d_reference_cases -- --ignored

# Test specific convolution functionality
cargo test -p bitnet-kernels --no-default-features --features cpu test_conv2d_basic_functionality
cargo test -p bitnet-kernels --no-default-features --features cpu test_conv2d_with_bias
cargo test -p bitnet-kernels --no-default-features --features cpu test_conv2d_stride
cargo test -p bitnet-kernels --no-default-features --features cpu test_conv2d_padding
cargo test -p bitnet-kernels --no-default-features --features cpu test_conv2d_dilation

# Test quantized convolution
cargo test -p bitnet-kernels --no-default-features --features cpu test_conv2d_quantized_i2s
cargo test -p bitnet-kernels --no-default-features --features cpu test_conv2d_quantized_tl1
cargo test -p bitnet-kernels --no-default-features --features cpu test_conv2d_quantized_with_bias
```

### GPU-Specific Tests

```bash
# GPU smoke tests (basic availability, run on CI with GPU)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_smoke

# GPU integration tests (comprehensive, manual execution)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_quantization --ignored

# GPU performance tests (benchmarking, development only)
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_performance --ignored

# GPU vs CPU quantization accuracy
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy --ignored

# GPU fallback mechanism testing
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored
```

### Cross-Validation Tests

```bash
# Cross-validation testing (requires C++ dependencies)
cargo test --workspace --features "cpu,ffi,crossval"

# Full cross-validation workflow
cargo run -p xtask -- full-crossval

# Cross-validation with concurrency caps
scripts/preflight.sh && cargo crossval-capped
```

## Test Configuration

The test suite uses a feature-gated configuration system:

- **`fixtures`**: Enables fixture management and test data generation
- **`reporting`**: Enables test reporting (JSON, HTML, Markdown, JUnit)
- **`trend`**: Enables trend analysis and performance tracking  
- **`integration-tests`**: Enables full integration test suite

## Test Features

- **Parallel Test Execution**: Configurable parallelism with resource limits
- **Fixture Management**: Automatic test data generation and caching
- **CI Integration**: JUnit output, exit codes, and CI-specific optimizations
- **Error Reporting**: Detailed error messages with recovery suggestions
- **Performance Tracking**: Benchmark results and regression detection

## Testing Strategy

### Core Testing Framework
- **Unit tests**: Each crate has comprehensive tests
- **Integration tests**: Cross-crate tests in `tests/`
- **Property-based testing**: Fuzz testing for GGUF parser robustness
- **Cross-validation**: Automated testing against C++ implementation
- **CI gates**: Compatibility tests block on every PR

### GPU Testing Strategy

GPU testing requires special consideration due to hardware dependencies and resource management. See [GPU Development Guide](gpu-development.md#gpu-testing-strategy) for comprehensive coverage of GPU testing categories, hardware-specific test configuration, and CI/CD considerations.

### Concurrency-Capped Testing

Use concurrency caps to prevent resource exhaustion:

```bash
# Run tests with concurrency caps (prevents resource storms)
scripts/preflight.sh && cargo t2                     # 2-thread CPU tests
scripts/preflight.sh && cargo crossval-capped        # Cross-validation with caps
scripts/e2e-gate.sh cargo test --features crossval   # Gate heavy E2E tests
```

See [Concurrency Caps Guide](concurrency-caps.md) for detailed information on preflight scripts, e2e gates, and resource management strategies.

### Performance Tracking Tests

The performance tracking infrastructure includes comprehensive test coverage for metrics collection, validation, and environment configuration:

```bash
# Run all performance tracking tests
cargo test -p bitnet-inference --features integration-tests --test performance_tracking_tests

# Run specific performance test categories
cargo test --test performance_tracking_tests performance_metrics_tests
cargo test --test performance_tracking_tests performance_tracker_tests  
cargo test --test performance_tracking_tests environment_variable_tests

# Test InferenceEngine performance integration
cargo test -p bitnet-inference --features integration-tests test_engine_performance_tracking_integration

# Test platform-specific memory and performance tracking
cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking
cargo test -p bitnet-kernels --no-default-features --features cpu test_performance_tracking

# GPU performance validation with comprehensive metrics
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_validation_comprehensive
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_memory_management
```

#### Performance Test Categories

1. **Performance Metrics Tests**: Validate metric computation, validation, and accuracy
2. **Performance Tracker Tests**: Test state management and metrics aggregation  
3. **Environment Variable Tests**: Validate configuration through environment variables
4. **Integration Tests**: End-to-end performance tracking with InferenceEngine
5. **Platform-Specific Tests**: Memory tracking and CPU kernel selection monitoring
6. **GPU Performance Tests**: GPU memory management and performance benchmarking

See [Performance Tracking Guide](performance-tracking.md) for detailed usage examples and configuration options.

## Specialized Test Commands

### GGUF Validation Tests

```bash
# Run GGUF validation tests
cargo test -p bitnet-inference --test gguf_header
cargo test -p bitnet-inference --test gguf_fuzz
cargo test -p bitnet-inference --test engine_inspect

# Run async smoke test with synthetic GGUF
printf "GGUF\x02\x00\x00\x00" > /tmp/t.gguf && \
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf && \
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf && \
BITNET_GGUF=/tmp/t.gguf cargo test -p bitnet-inference --features rt-tokio --test smoke
```

### Convolution Testing Framework

The convolution testing framework includes comprehensive validation against PyTorch reference implementations and extensive unit testing for various parameter combinations.

#### PyTorch Reference Testing

The convolution implementation includes optional PyTorch reference tests that validate correctness by comparing outputs with PyTorch's `F.conv2d` implementation:

```bash
# Prerequisites: Install Python and PyTorch
pip install torch

# Run PyTorch reference tests (ignored by default)
cargo test -p bitnet-kernels conv2d_reference_cases -- --ignored

# Verbose output to see test details
cargo test -p bitnet-kernels conv2d_reference_cases -- --ignored --nocapture
```

The reference tests cover:
- **Basic convolution**: Simple 2D convolution operations
- **Stride operations**: Various stride configurations (1x1, 2x2)
- **Padding operations**: Zero padding with different configurations
- **Dilation operations**: Dilated convolutions for expanded receptive fields
- **Parameter combinations**: Mixed stride, padding, and dilation

#### Quantization Testing

Comprehensive testing of quantized convolution operations:

```bash
# Test I2S quantization (2-bit signed)
cargo test -p bitnet-kernels test_conv2d_quantized_i2s

# Test TL1 quantization (table lookup)
cargo test -p bitnet-kernels test_conv2d_quantized_tl1

# Test TL2 quantization (advanced table lookup)  
cargo test -p bitnet-kernels test_conv2d_quantized_tl2

# Test quantization with bias
cargo test -p bitnet-kernels test_conv2d_quantized_with_bias

# Test scale factor application
cargo test -p bitnet-kernels test_conv2d_quantized_scale_factor
```

#### Error Handling and Validation

The convolution tests include comprehensive error handling validation:

```bash
# Test dimension mismatch errors
cargo test -p bitnet-kernels test_conv2d_dimension_mismatch

# Test invalid input size errors
cargo test -p bitnet-kernels test_conv2d_invalid_input_size

# Test invalid bias size errors
cargo test -p bitnet-kernels test_conv2d_invalid_bias_size

# Test quantized weight size validation
cargo test -p bitnet-kernels test_conv2d_quantized_invalid_weight_size

# Test scale size validation
cargo test -p bitnet-kernels test_conv2d_quantized_invalid_scale_size
```

### IQ2_S Backend Tests

```bash
# Build with IQ2_S quantization support (requires GGML FFI)
cargo build --release --no-default-features --features "cpu,iq2s-ffi"

# Run IQ2_S backend validation
./scripts/test-iq2s-backend.sh

# Run unit tests
cargo test --package bitnet-models --no-default-features --features "cpu,iq2s-ffi"
```

### Streaming Tests

```bash
# Test streaming generation
cargo run --example streaming_generation --no-default-features --features cpu

# Test server streaming
cargo test -p bitnet-server --no-default-features --features cpu streaming

# Test token ID accuracy
cargo test -p bitnet-inference --no-default-features --features cpu test_token_id_streaming
```

For more streaming functionality and Server-Sent Events testing, see the [Streaming API Guide](streaming-api.md).

## Environment Variables for Testing

### Runtime Variables
- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `RAYON_NUM_THREADS`: Control CPU parallelism

### Test-Specific Variables
- `RUST_TEST_THREADS`: Rust test parallelism
- `CROSSVAL_WORKERS`: Cross-validation test workers

For complete list of environment variables, see the main project documentation.