# BitNet Cross-Validation Framework

This crate provides cross-validation functionality to compare the BitNet Rust implementation against the original C++ implementation for numerical accuracy and performance.

## Features

- **Token-level equivalence testing**: Ensures identical outputs between implementations
- **Performance benchmarking**: Compares throughput and latency
- **Automated test fixtures**: Small test models for consistent validation
- **Feature-gated**: Zero overhead when cross-validation is disabled

## Usage

### Basic Cross-Validation

```bash
# Enable cross-validation features
cargo test --features crossval
cargo bench --features crossval
```

### Running Specific Tests

```bash
# Test token equivalence
cargo test --features crossval token_equivalence

# Run performance benchmarks
cargo bench --features crossval performance
```

### Configuration

Cross-validation behavior can be configured:

```rust
use bitnet_crossval::{CrossvalConfig, comparison::CrossValidator};

let config = CrossvalConfig {
    tolerance: 1e-6,        // Floating-point comparison tolerance
    max_tokens: 1000,       // Maximum tokens to compare
    benchmark: true,        // Enable performance measurement
};

let validator = CrossValidator::new(config);
```

## Setup Requirements

### 1. C++ Implementation

The cross-validation framework requires access to the original BitNet C++ implementation:

```bash
# Download and build C++ implementation
cargo run -p xtask -- fetch-cpp
```

This downloads the official Microsoft BitNet.cpp to `~/.cache/bitnet_cpp/` and builds it.

### 2. System Dependencies

- **Rust**: Standard Rust toolchain
- **Clang**: Required for generating C++ bindings
  - Ubuntu/Debian: `apt install clang`
  - macOS: `xcode-select --install`
  - Windows: Install LLVM from https://llvm.org/

### 3. Test Fixtures

Create or generate test fixtures:

```bash
# Generate standard test fixtures
cargo xtask gen-fixtures

# Or create custom fixtures in crossval/fixtures/
```

## Architecture

### Feature Gates

Cross-validation is behind the `crossval` feature to ensure zero overhead when disabled:

```toml
[dependencies]
bitnet-crossval = { version = "0.1", features = ["crossval"] }
```

### Components

- **`cpp_bindings`**: Safe FFI wrappers around C++ implementation
- **`comparison`**: High-level comparison and validation logic
- **`fixtures`**: Test model and data management
- **`utils`**: Numerical comparison and performance utilities

### Test Structure

```
crossval/
├── src/           # Core cross-validation logic
├── tests/         # Token equivalence tests
├── benches/       # Performance benchmarks
├── fixtures/      # Test models and data
└── build.rs       # C++ binding generation
```

## Test Fixtures

Test fixtures are small models and datasets used for validation:

```json
{
  "name": "minimal_test",
  "model_path": "minimal_model.gguf",
  "test_prompts": [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog."
  ],
  "expected_tokens": null
}
```

See `fixtures/README.md` for detailed fixture documentation.

## Performance Benchmarking

The framework includes Criterion-based benchmarks:

```bash
# Run all benchmarks
cargo bench --features crossval

# Run specific benchmark group
cargo bench --features crossval rust_inference
cargo bench --features crossval cpp_inference
cargo bench --features crossval comparison
```

Benchmark results are saved to `target/criterion/` with HTML reports.

## Integration with CI

Cross-validation can be integrated into CI pipelines:

```yaml
# .github/workflows/crossval.yml
- name: Setup C++ implementation
  run: ./ci/fetch_bitnet_cpp.sh

- name: Run cross-validation tests
  run: cargo test --features crossval

- name: Run performance benchmarks
  run: cargo bench --features crossval
```

## Troubleshooting

### Common Issues

1. **"BitNet C++ implementation not found"**
   - Run `./ci/fetch_bitnet_cpp.sh` to download C++ code
   - Check that `~/.cache/bitnet_cpp/` exists

2. **"clang not found"**
   - Install clang development tools
   - Ensure clang is in your PATH

3. **"Unable to generate bindings"**
   - Verify C++ headers are present
   - Check that bindgen dependencies are installed

4. **Test fixtures not found**
   - Run `cargo xtask gen-fixtures` to create test models
   - Or create custom fixtures in `fixtures/` directory

### Debug Mode

Enable debug logging for detailed cross-validation information:

```bash
RUST_LOG=debug cargo test --features crossval
```

## Contributing

When adding new cross-validation functionality:

1. Keep the `crossval` feature gate for all C++ dependencies
2. Add appropriate error handling for missing C++ implementation
3. Include both positive and negative test cases
4. Update documentation and examples
5. Ensure tests work with minimal fixtures

## License

This crate follows the same license as the main BitNet project.