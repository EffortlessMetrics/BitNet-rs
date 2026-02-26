# Test Authoring Guidelines and Best Practices

## Introduction

This guide provides comprehensive guidelines for writing effective tests in the BitNet-rs testing framework. Following these practices ensures consistent, maintainable, and reliable tests.

## Test Structure and Organization

### Directory Structure

```
tests/
├── common/                 # Shared test utilities
│   ├── harness.rs         # Test execution framework
│   ├── fixtures.rs        # Test data management
│   ├── config.rs          # Configuration management
│   └── utils.rs           # Common test utilities
├── unit/                  # Unit tests
│   ├── bitnet_common/     # Tests for bitnet-common crate
│   ├── bitnet_models/     # Tests for bitnet-models crate
│   └── ...
├── integration/           # Integration tests
│   ├── workflows/         # End-to-end workflow tests
│   ├── components/        # Component interaction tests
│   └── configurations/    # Configuration tests
├── crossval/              # Cross-validation tests
│   ├── accuracy/          # Accuracy comparison tests
│   ├── performance/       # Performance comparison tests
│   └── regression/        # Regression tests
└── fixtures/              # Test data and models
    ├── models/            # Test model files
    ├── datasets/          # Test datasets
    └── configs/           # Test configurations
```

### Naming Conventions

#### Test Files
- Unit tests: `test_<module_name>.rs`
- Integration tests: `<feature>_integration_test.rs`
- Cross-validation tests: `<scenario>_crossval_test.rs`

#### Test Functions
- Use descriptive names: `test_model_loading_with_invalid_path`
- Follow pattern: `test_<action>_<condition>_<expected_result>`
- Use snake_case consistently

#### Test Modules
```rust
#[cfg(test)]
mod tests {
    use super::*;

    mod model_loading {
        use super::*;

        #[tokio::test]
        async fn test_load_valid_model_succeeds() {
            // Test implementation
        }

        #[tokio::test]
        async fn test_load_invalid_model_fails() {
            // Test implementation
        }
    }

    mod inference {
        use super::*;
        // More tests...
    }
}
```

## Writing Unit Tests

### Basic Unit Test Structure

```rust
use crate::common::{TestHarness, TestConfig, FixtureManager};
use bitnet_models::BitNetModel;

#[tokio::test]
async fn test_model_loading_success() {
    // Arrange
    let config = TestConfig::default();
    let harness = TestHarness::new(config);
    let fixtures = FixtureManager::new();

    let model_path = fixtures.get_model_fixture("small_test_model").await
        .expect("Failed to get test model");

    // Act
    let result = BitNetModel::from_file(&model_path).await;

    // Assert
    assert!(result.is_ok(), "Model loading should succeed");
    let model = result.unwrap();
    assert_eq!(model.config().model_type, "bitnet");
    assert!(model.config().vocab_size > 0);
}

#[tokio::test]
async fn test_model_loading_invalid_path() {
    // Arrange
    let invalid_path = Path::new("/nonexistent/model.gguf");

    // Act
    let result = BitNetModel::from_file(invalid_path).await;

    // Assert
    assert!(result.is_err(), "Loading nonexistent model should fail");
    match result.unwrap_err() {
        ModelError::FileNotFound(path) => {
            assert_eq!(path, invalid_path);
        }
        _ => panic!("Expected FileNotFound error"),
    }
}
```

### Testing Async Functions

```rust
#[tokio::test]
async fn test_async_inference() {
    let model = setup_test_model().await;
    let tokens = vec![1, 2, 3, 4, 5];

    let result = model.generate_from_tokens(&tokens, &InferenceConfig::default()).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(!output.tokens.is_empty());
    assert_eq!(output.tokens.len(), tokens.len() + 1); // Should generate one more token
}
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_tokenization_roundtrip(text in "\\PC*") {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tokenizer = setup_test_tokenizer().await;

            let tokens = tokenizer.encode(&text).await.unwrap();
            let decoded = tokenizer.decode(&tokens).await.unwrap();

            // Roundtrip should preserve semantic content
            prop_assert!(!tokens.is_empty() || text.is_empty());
            prop_assert!(decoded.len() <= text.len() * 2); // Reasonable bound
        });
    }
}
```

### Testing Error Conditions

```rust
#[tokio::test]
async fn test_inference_with_empty_tokens() {
    let model = setup_test_model().await;
    let empty_tokens = vec![];

    let result = model.generate_from_tokens(&empty_tokens, &InferenceConfig::default()).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        InferenceError::EmptyInput => {
            // Expected error
        }
        other => panic!("Expected EmptyInput error, got {:?}", other),
    }
}

#[tokio::test]
async fn test_model_loading_corrupted_file() {
    let fixtures = FixtureManager::new();
    let corrupted_path = fixtures.create_corrupted_model_fixture().await
        .expect("Failed to create corrupted fixture");

    let result = BitNetModel::from_file(&corrupted_path).await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ModelError::CorruptedFile(_)));
}
```

## Writing Integration Tests

### Workflow Integration Tests

```rust
use crate::common::{TestHarness, WorkflowTestCase};

#[tokio::test]
async fn test_complete_inference_workflow() {
    let harness = TestHarness::new(TestConfig::default());

    let test_case = WorkflowTestCase {
        name: "complete_inference_workflow",
        model_name: "small_test_model",
        input_text: "Hello, world!",
        expected_min_tokens: 5,
        expected_max_tokens: 20,
    };

    let result = harness.run_workflow_test(test_case).await;

    assert!(result.is_ok());
    let workflow_result = result.unwrap();

    // Validate each step
    assert!(workflow_result.model_loaded);
    assert!(workflow_result.tokenization_successful);
    assert!(workflow_result.inference_successful);
    assert!(workflow_result.output_tokens.len() >= 5);
    assert!(workflow_result.output_tokens.len() <= 20);

    // Validate performance
    assert!(workflow_result.total_duration < Duration::from_secs(10));
    assert!(workflow_result.peak_memory < 1024 * 1024 * 1024); // 1GB limit
}
```

### Component Interaction Tests

```rust
#[tokio::test]
async fn test_model_tokenizer_integration() {
    let model = setup_test_model().await;
    let tokenizer = model.tokenizer();

    let test_text = "The quick brown fox jumps over the lazy dog.";

    // Test tokenization
    let tokens = tokenizer.encode(test_text).await
        .expect("Tokenization should succeed");

    // Test that model can process these tokens
    let inference_result = model.generate_from_tokens(&tokens, &InferenceConfig::default()).await
        .expect("Inference should succeed");

    // Validate integration
    assert!(!tokens.is_empty());
    assert!(!inference_result.tokens.is_empty());
    assert!(inference_result.tokens[0..tokens.len()] == tokens[..]);
}
```

## Writing Cross-Validation Tests

### Accuracy Comparison Tests

```rust
use crate::crossval::{CrossValidationSuite, ComparisonTolerance};

#[tokio::test]
async fn test_inference_accuracy_comparison() {
    let tolerance = ComparisonTolerance {
        min_token_accuracy: 0.95,
        max_probability_divergence: 0.1,
        max_performance_regression: 2.0,
    };

    let mut suite = CrossValidationSuite::new(tolerance);

    let test_cases = vec![
        ComparisonTestCase {
            name: "simple_generation",
            input: "Hello, how are you?",
            config: InferenceConfig::default(),
        },
        ComparisonTestCase {
            name: "long_context",
            input: "A".repeat(1000),
            config: InferenceConfig::default(),
        },
    ];

    suite.add_test_cases(test_cases);

    let model_path = fixtures.get_model_fixture("comparison_model").await
        .expect("Failed to get comparison model");

    let result = suite.run_comparison(&model_path).await
        .expect("Comparison should complete");

    // Validate results
    assert!(result.summary.overall_accuracy >= 0.95);
    assert!(result.summary.performance_ratio <= 2.0);

    for test_result in &result.test_results {
        assert!(test_result.accuracy_result.passes_tolerance);
    }
}
```

### Performance Comparison Tests

```rust
#[tokio::test]
async fn test_performance_comparison() {
    let mut suite = CrossValidationSuite::new(ComparisonTolerance::default());

    let performance_test = ComparisonTestCase {
        name: "performance_benchmark",
        input: "Generate a story about artificial intelligence.",
        config: InferenceConfig {
            max_tokens: 100,
            temperature: 0.7,
            ..Default::default()
        },
    };

    suite.add_test_case(performance_test);

    let model_path = fixtures.get_model_fixture("performance_model").await
        .expect("Failed to get performance model");

    let result = suite.run_comparison(&model_path).await
        .expect("Performance comparison should complete");

    // Validate performance metrics
    let perf_comparison = &result.test_results[0].performance_comparison;

    // Rust should be competitive with C++
    assert!(perf_comparison.throughput_ratio <= 2.0,
           "Rust should not be more than 2x slower than C++");

    // Memory usage should be reasonable
    assert!(perf_comparison.memory_ratio <= 1.5,
           "Rust should not use more than 1.5x memory of C++");
}
```

## Best Practices

### 1. Test Independence

```rust
// Good: Each test is independent
#[tokio::test]
async fn test_model_loading() {
    let model = setup_fresh_model().await; // Create new instance
    // Test logic...
}

#[tokio::test]
async fn test_model_inference() {
    let model = setup_fresh_model().await; // Create new instance
    // Test logic...
}

// Bad: Tests depend on shared state
static mut SHARED_MODEL: Option<BitNetModel> = None;

#[tokio::test]
async fn test_model_loading() {
    unsafe {
        SHARED_MODEL = Some(setup_model().await);
    }
}

#[tokio::test]
async fn test_model_inference() {
    unsafe {
        let model = SHARED_MODEL.as_ref().unwrap(); // Depends on previous test
        // Test logic...
    }
}
```

### 2. Clear Assertions

```rust
// Good: Specific, descriptive assertions
#[tokio::test]
async fn test_tokenization_output() {
    let tokenizer = setup_tokenizer().await;
    let tokens = tokenizer.encode("Hello world").await.unwrap();

    assert_eq!(tokens.len(), 2, "Should tokenize 'Hello world' into 2 tokens");
    assert!(tokens[0] > 0, "First token should be positive");
    assert!(tokens[1] > 0, "Second token should be positive");
    assert_ne!(tokens[0], tokens[1], "Tokens should be different");
}

// Bad: Vague assertions
#[tokio::test]
async fn test_tokenization_output() {
    let tokenizer = setup_tokenizer().await;
    let tokens = tokenizer.encode("Hello world").await.unwrap();

    assert!(tokens.len() > 0); // Too vague
    assert!(tokens.iter().all(|&t| t > 0)); // Not descriptive
}
```

### 3. Proper Error Testing

```rust
// Good: Test specific error conditions
#[tokio::test]
async fn test_invalid_model_format() {
    let invalid_path = create_invalid_model_file().await;

    let result = BitNetModel::from_file(&invalid_path).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        ModelError::InvalidFormat { format, expected } => {
            assert_eq!(format, "unknown");
            assert!(expected.contains("gguf") || expected.contains("safetensors"));
        }
        other => panic!("Expected InvalidFormat error, got {:?}", other),
    }
}

// Bad: Generic error testing
#[tokio::test]
async fn test_invalid_model_format() {
    let invalid_path = create_invalid_model_file().await;
    let result = BitNetModel::from_file(&invalid_path).await;
    assert!(result.is_err()); // Too generic
}
```

### 4. Resource Management

```rust
// Good: Proper cleanup
#[tokio::test]
async fn test_with_temporary_files() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("test_model.gguf");

    // Create test file
    create_test_model_file(&model_path).await;

    // Run test
    let result = BitNetModel::from_file(&model_path).await;
    assert!(result.is_ok());

    // Cleanup happens automatically when temp_dir is dropped
}

// Use RAII pattern for resource management
struct TestResource {
    _temp_dir: tempfile::TempDir,
    model_path: PathBuf,
}

impl TestResource {
    async fn new() -> Self {
        let temp_dir = tempfile::tempdir().unwrap();
        let model_path = temp_dir.path().join("test_model.gguf");
        create_test_model_file(&model_path).await;

        Self {
            _temp_dir: temp_dir,
            model_path,
        }
    }
}
```

### 5. Performance Testing

```rust
#[tokio::test]
async fn test_inference_performance() {
    let model = setup_performance_model().await;
    let tokens = vec![1, 2, 3, 4, 5];

    let start = Instant::now();
    let result = model.generate_from_tokens(&tokens, &InferenceConfig::default()).await;
    let duration = start.elapsed();

    assert!(result.is_ok());
    assert!(duration < Duration::from_millis(100),
           "Inference should complete within 100ms, took {:?}", duration);

    // Test memory usage
    let memory_before = get_memory_usage();
    let _result = model.generate_from_tokens(&tokens, &InferenceConfig::default()).await;
    let memory_after = get_memory_usage();
    let memory_used = memory_after - memory_before;

    assert!(memory_used < 100 * 1024 * 1024,
           "Should use less than 100MB, used {}MB", memory_used / 1024 / 1024);
}
```

## Common Patterns

### Test Fixtures and Setup

```rust
// Common setup function
async fn setup_test_model() -> BitNetModel {
    let fixtures = FixtureManager::new();
    let model_path = fixtures.get_model_fixture("default_test_model").await
        .expect("Failed to get test model");

    BitNetModel::from_file(&model_path).await
        .expect("Failed to load test model")
}

// Parameterized tests
async fn test_model_with_different_configs(config: InferenceConfig) {
    let model = setup_test_model().await;
    let tokens = vec![1, 2, 3];

    let result = model.generate_from_tokens(&tokens, &config).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_various_inference_configs() {
    let configs = vec![
        InferenceConfig { temperature: 0.1, ..Default::default() },
        InferenceConfig { temperature: 0.7, ..Default::default() },
        InferenceConfig { temperature: 1.0, ..Default::default() },
        InferenceConfig { max_tokens: 10, ..Default::default() },
        InferenceConfig { max_tokens: 100, ..Default::default() },
    ];

    for config in configs {
        test_model_with_different_configs(config).await;
    }
}
```

### Mocking and Test Doubles

```rust
// Mock implementation for testing
#[derive(Default)]
struct MockTokenizer {
    encode_calls: Arc<Mutex<Vec<String>>>,
    decode_calls: Arc<Mutex<Vec<Vec<u32>>>>,
}

#[async_trait]
impl Tokenizer for MockTokenizer {
    async fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_calls.lock().unwrap().push(text.to_string());
        Ok(text.chars().map(|c| c as u32).collect())
    }

    async fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        self.decode_calls.lock().unwrap().push(tokens.to_vec());
        Ok(tokens.iter().map(|&t| t as u8 as char).collect())
    }
}

#[tokio::test]
async fn test_with_mock_tokenizer() {
    let mock_tokenizer = MockTokenizer::default();
    let model = BitNetModel::with_tokenizer(Box::new(mock_tokenizer.clone()));

    let result = model.process_text("test").await;
    assert!(result.is_ok());

    // Verify mock was called correctly
    let encode_calls = mock_tokenizer.encode_calls.lock().unwrap();
    assert_eq!(encode_calls.len(), 1);
    assert_eq!(encode_calls[0], "test");
}
```

## Testing Checklist

Before submitting tests, ensure:

- [ ] Tests are independent and can run in any order
- [ ] All resources are properly cleaned up
- [ ] Error conditions are tested with specific assertions
- [ ] Performance-critical paths have performance tests
- [ ] Tests have descriptive names and clear assertions
- [ ] Async functions are properly tested with `#[tokio::test]`
- [ ] Property-based tests are used for complex invariants
- [ ] Integration tests cover realistic workflows
- [ ] Cross-validation tests compare against C++ implementation
- [ ] Tests run quickly (unit tests <1s, integration tests <10s)
- [ ] Tests are deterministic and reproducible
- [ ] Documentation is updated for new test patterns

Following these guidelines ensures that tests are reliable, maintainable, and provide valuable feedback during development.
