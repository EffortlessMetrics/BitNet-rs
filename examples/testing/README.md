# bitnet-rs Testing Framework Examples

This directory contains comprehensive examples demonstrating how to use the bitnet-rs testing framework across all testing categories. These examples serve as both documentation and practical templates for implementing tests in your own projects.

## 📁 Directory Structure

```
examples/testing/
├── README.md                           # This file
├── unit_tests/                         # Unit testing examples
│   ├── example_bitnet_common_tests.rs  # Common crate unit tests
│   └── example_bitnet_models_tests.rs  # Models crate unit tests
├── integration_tests/                  # Integration testing examples
│   ├── example_workflow_integration.rs # End-to-end workflow tests
│   └── example_component_interaction.rs # Component interaction tests
├── cross_validation/                   # Cross-implementation comparison
│   └── example_rust_cpp_comparison.rs  # Rust vs C++ comparison tests
├── performance/                        # Performance benchmarking
│   └── example_benchmarks.rs          # Comprehensive benchmarks
└── ci_cd/                             # CI/CD integration examples
    ├── github_actions_example.yml     # GitHub Actions workflow
    ├── test_configuration_examples.toml # Test configurations
    ├── docker_testing_example.dockerfile # Docker testing setup
    └── test_scripts_example.sh        # Automation scripts
```

## 🧪 Unit Testing Examples

### `unit_tests/example_bitnet_common_tests.rs`

Demonstrates comprehensive unit testing patterns including:

- **Basic functionality tests** with setup and teardown
- **Error handling validation** for various error conditions
- **Property-based testing** using proptest for invariant validation
- **Performance testing** with benchmarking and assertions
- **Edge case testing** for boundary conditions
- **Async operation testing** for asynchronous functions
- **Memory usage testing** to validate resource consumption
- **Concurrent access testing** for thread safety

**Key Features:**
```rust
// Property-based testing example
proptest! {
    #[test]
    fn test_quantization_config_invariants(
        bits in 1u8..=8u8,
        group_size in 32u32..=1024u32,
        symmetric in any::<bool>()
    ) {
        let config = QuantizationConfig::new(bits, group_size, symmetric);
        prop_assert!(config.bits() >= 1 && config.bits() <= 8);
        prop_assert!(config.group_size().is_power_of_two());
    }
}

// Performance testing example
#[tokio::test]
async fn test_model_config_serialization_performance() {
    let start = Instant::now();
    for _ in 0..1000 {
        let _serialized = serde_json::to_string(&config).unwrap();
    }
    let avg_duration = start.elapsed() / 1000;
    assert!(avg_duration.as_millis() < 1);
}
```

### `unit_tests/example_bitnet_models_tests.rs`

Shows model-specific testing patterns:

- **Model loading and validation** with mock data
- **Format detection and conversion** testing
- **Metadata validation** and consistency checks
- **Error handling** for invalid model files
- **Large model handling** with memory efficiency tests
- **Concurrent model access** validation
- **Parameterized tests** for different model formats

## 🔗 Integration Testing Examples

### `integration_tests/example_workflow_integration.rs`

Demonstrates end-to-end workflow testing:

- **Complete inference workflows** from input to output
- **Streaming inference** with chunk validation
- **Batch processing** with efficiency testing
- **Model quantization workflows** with compression validation
- **Error recovery** and resilience testing
- **Configuration validation** across components

**Key Features:**
```rust
#[tokio::test]
async fn test_complete_inference_workflow() {
    // Setup: Create test environment
    let model = BitNetModel::from_file(&model_path).await.unwrap();
    let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();
    let mut engine = InferenceEngine::new(model, tokenizer, config).await.unwrap();

    // Execute: Run inference
    let result = engine.generate("The future of AI is").await.unwrap();

    // Verify: Check results
    assert!(!result.text.is_empty());
    assert!(result.text.starts_with("The future of AI is"));
}
```

### `integration_tests/example_component_interaction.rs`

Shows component interaction testing:

- **Model-tokenizer compatibility** validation
- **Cross-crate error propagation** testing
- **Resource sharing** between components
- **Configuration propagation** validation
- **Data flow validation** through the pipeline
- **Component lifecycle management** testing
- **Memory management** across components

## ⚖️ Cross-Validation Examples

### `cross_validation/example_rust_cpp_comparison.rs`

Demonstrates accuracy and performance comparison:

- **Basic accuracy comparison** with configurable tolerance
- **Edge case comparisons** for robustness testing
- **Performance regression detection** with strict thresholds
- **Probability distribution comparison** for model consistency
- **Model format compatibility** testing
- **Regression test cases** for known issues

**Key Features:**
```rust
#[tokio::test]
async fn test_basic_accuracy_comparison() {
    let tolerance = ComparisonTolerance {
        min_token_accuracy: 0.95,
        max_probability_divergence: 0.1,
        max_performance_regression: 2.0,
    };

    let mut suite = CrossValidationSuite::new(tolerance);
    let result = suite.run_comparison(&model_path).await.unwrap();

    assert!(result.summary.overall_success);
    for test_result in &result.test_results {
        assert!(test_result.accuracy_result.passes_tolerance);
    }
}
```

## 🚀 Performance Benchmarking Examples

### `performance/example_benchmarks.rs`

Comprehensive performance testing using Criterion:

- **Inference throughput benchmarks** for different input lengths
- **Latency benchmarks** for various scenarios
- **Memory usage benchmarks** across model sizes
- **Batch processing benchmarks** with scaling analysis
- **Streaming performance** comparison
- **Concurrent access benchmarks** for scalability
- **Model loading benchmarks** for different formats

**Key Features:**
```rust
pub fn benchmark_inference_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_throughput");

    for length in [10, 50, 100, 200, 500] {
        let input = "word ".repeat(length);
        group.throughput(Throughput::Elements(length as u64));
        group.bench_with_input(
            BenchmarkId::new("tokens_per_second", length),
            &input,
            |b, input| {
                b.to_async(&rt).iter(|| async {
                    engine.generate(input).await.unwrap()
                });
            },
        );
    }
}
```

## 🔄 CI/CD Integration Examples

### `ci_cd/github_actions_example.yml`

Complete GitHub Actions workflow demonstrating:

- **Multi-platform testing** (Ubuntu, Windows, macOS)
- **Matrix builds** with different Rust versions
- **Parallel job execution** with dependencies
- **Test result aggregation** and reporting
- **Coverage collection** and upload
- **Artifact management** for test results
- **Performance regression detection**
- **Release validation** pipeline

**Key Jobs:**
- `unit-tests`: Runs unit tests with coverage across platforms
- `integration-tests`: Executes integration test suites
- `cross-validation`: Compares Rust vs C++ implementations
- `performance-benchmarks`: Runs performance tests and reports
- `test-report`: Generates comprehensive HTML reports
- `release-validation`: Validates releases with strict criteria

### `ci_cd/test_configuration_examples.toml`

Comprehensive configuration examples for different scenarios:

- **Default configuration** for general development
- **CI-optimized configuration** for automated testing
- **Development configuration** for fast iteration
- **Performance testing configuration** with strict requirements
- **Platform-specific configurations** (Windows, macOS, Docker)
- **Memory-constrained configuration** for limited environments
- **Security testing configuration** with detailed logging

### `ci_cd/docker_testing_example.dockerfile`

Multi-stage Docker setup for containerized testing:

- **Builder stage** with all development dependencies
- **Tester stage** with complete testing environment
- **Production stage** optimized for runtime testing
- **C++ BitNet integration** for cross-validation
- **Python dependencies** for test reporting
- **Comprehensive examples** of usage commands

### `ci_cd/test_scripts_example.sh`

Automation scripts providing:

- **Environment setup** and cleanup
- **Dependency checking** and installation
- **Test execution** for all categories
- **Report generation** with multiple formats
- **Watch mode** for continuous testing
- **Docker integration** for containerized runs
- **Memory profiling** and security auditing

## 🚀 Getting Started

### 1. Basic Unit Testing

```bash
# Copy unit test examples to your project
cp examples/testing/unit_tests/example_bitnet_common_tests.rs tests/

# Run unit tests
cargo test --lib
```

### 2. Integration Testing

```bash
# Copy integration test examples
cp examples/testing/integration_tests/* tests/

# Run integration tests
cargo test --test integration_tests
```

### 3. Cross-Validation Testing

```bash
# Setup C++ BitNet (if available)
git clone https://github.com/microsoft/BitNet.git bitnet-cpp
cd bitnet-cpp && make

# Copy cross-validation examples
cp examples/testing/cross_validation/* tests/

# Run cross-validation tests
BITNET_CPP_BINARY=./bitnet-cpp/bitnet cargo test --test cross_validation_tests
```

### 4. Performance Benchmarking

```bash
# Install criterion
cargo install cargo-criterion

# Copy benchmark examples
cp examples/testing/performance/* benches/

# Run benchmarks
cargo criterion
```

### 5. CI/CD Integration

```bash
# Copy GitHub Actions workflow
cp examples/testing/ci_cd/github_actions_example.yml .github/workflows/

# Copy test configurations
cp examples/testing/ci_cd/test_configuration_examples.toml test-config.toml

# Make test script executable
cp examples/testing/ci_cd/test_scripts_example.sh scripts/test.sh
chmod +x scripts/test.sh

# Run comprehensive tests
./scripts/test.sh comprehensive
```

### 6. Docker Testing

```bash
# Build test image
docker build -f examples/testing/ci_cd/docker_testing_example.dockerfile -t bitnet-rs-test .

# Run tests in container
docker run --rm -v $(pwd)/test-results:/app/test-results bitnet-rs-test
```

## 📊 Test Reporting

The examples include comprehensive test reporting capabilities:

- **HTML reports** with interactive visualizations
- **JSON reports** for machine processing
- **JUnit XML** for CI integration
- **Markdown reports** for documentation
- **Coverage reports** with line-by-line analysis
- **Performance charts** and trend analysis

## 🔧 Customization

All examples are designed to be easily customizable:

1. **Modify test configurations** in the TOML files
2. **Adjust tolerance levels** for cross-validation
3. **Add custom test cases** following the patterns
4. **Extend CI workflows** with additional jobs
5. **Customize reporting formats** and outputs

## 📚 Best Practices Demonstrated

- **Test isolation** and cleanup
- **Deterministic testing** with controlled randomness
- **Resource management** and memory monitoring
- **Error handling** and recovery testing
- **Performance regression detection**
- **Cross-platform compatibility**
- **Comprehensive documentation** and examples

## 🤝 Contributing

When adding new test examples:

1. Follow the established patterns and naming conventions
2. Include comprehensive documentation and comments
3. Add both positive and negative test cases
4. Include performance and memory considerations
5. Update this README with new examples

## 📖 Additional Resources

- [Testing Framework Design Document](../../.kiro/specs/testing-framework-implementation/design.md)
- [Testing Framework Requirements](../../.kiro/specs/testing-framework-implementation/requirements.md)
- [Testing Framework Tasks](../../.kiro/specs/testing-framework-implementation/tasks.md)
- [bitnet-rs Documentation](../../docs/)

These examples provide a complete foundation for implementing comprehensive testing in the bitnet-rs project and serve as templates for similar Rust projects requiring extensive testing frameworks.
