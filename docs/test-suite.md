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