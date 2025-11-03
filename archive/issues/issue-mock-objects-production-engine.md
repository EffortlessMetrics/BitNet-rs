# [Mock Objects] Extensive mock implementations in ProductionInferenceEngine test module should be consolidated

## Problem Description

The test module in `crates/bitnet-inference/src/production_engine.rs` contains extensive `MockModel` and `MockTokenizer` implementations that are duplicated across multiple test files. These mock objects are comprehensive but could be simplified and moved to a dedicated test utilities module to reduce code duplication and improve maintainability.

## Environment

- **File**: `crates/bitnet-inference/src/production_engine.rs`
- **Module**: `tests` (with `#[cfg(test)]`)
- **Structs**: `MockModel`, `MockTokenizer`
- **Crate**: `bitnet-inference`

## Current Implementation Analysis

The test module contains extensive mock implementations:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct MockModel {
        config: BitNetConfig,
        // ... extensive implementation
    }

    impl Model for MockModel {
        // ... comprehensive trait implementation
    }

    struct MockTokenizer {
        // ... extensive implementation
    }

    impl Tokenizer for MockTokenizer {
        // ... comprehensive trait implementation
    }

    // Multiple test functions using these mocks
}
```

## Root Cause Analysis

1. **Code Duplication**: Mock objects are likely duplicated across multiple test modules
2. **Excessive Complexity**: Mock implementations are more comprehensive than needed for specific tests
3. **Poor Organization**: Test utilities mixed with production code in same file
4. **Maintenance Overhead**: Changes to mock interfaces require updates in multiple locations
5. **Testing Inefficiency**: Overly complex mocks slow down test execution and maintenance

## Impact Assessment

**Severity**: Medium - Code Maintenance & Testing Efficiency
**Affected Components**:
- Test maintainability and readability
- Code duplication across test modules
- Test execution performance
- Developer productivity for writing new tests

**Technical Debt**:
- Increased maintenance burden for mock objects
- Potential inconsistency between mock implementations
- Harder to understand what each test actually validates
- Slower test development cycle

## Proposed Solution

### Option 1: Consolidate into Test Utilities Module (Recommended)

Create a dedicated test utilities crate with reusable mock implementations:

```rust
// In crates/bitnet-test-utils/src/mocks/mod.rs

pub mod model;
pub mod tokenizer;

pub use model::MockModel;
pub use tokenizer::MockTokenizer;

// In crates/bitnet-test-utils/src/mocks/model.rs
pub struct MockModel {
    config: BitNetConfig,
    responses: HashMap<String, Tensor>,
}

impl MockModel {
    pub fn new(config: BitNetConfig) -> Self {
        Self {
            config,
            responses: HashMap::new(),
        }
    }

    pub fn with_response(mut self, input: impl Into<String>, output: Tensor) -> Self {
        self.responses.insert(input.into(), output);
        self
    }

    pub fn simple() -> Self {
        // Create a simple mock with minimal configuration
        Self::new(BitNetConfig::default())
    }
}

impl Model for MockModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified implementation that returns pre-configured responses
        // or generates simple synthetic responses
        if let Some(response) = self.responses.get(&format!("{:?}", input)) {
            Ok(response.clone())
        } else {
            // Generate simple synthetic response based on input shape
            self.generate_synthetic_response(input)
        }
    }

    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    // Only implement methods actually needed by tests
}
```

### Option 2: Simplified In-Place Mocks

Simplify the existing mocks to only implement what's needed:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleMockModel {
        vocab_size: usize,
    }

    impl SimpleMockModel {
        fn new(vocab_size: usize) -> Self {
            Self { vocab_size }
        }
    }

    impl Model for SimpleMockModel {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            // Minimal implementation for testing
            let batch_size = input.shape()[0];
            let seq_len = input.shape()[1];
            Ok(Tensor::zeros((batch_size, seq_len, self.vocab_size), DType::F32, &Device::Cpu)?)
        }

        fn config(&self) -> &BitNetConfig {
            &BitNetConfig::default()
        }
    }

    struct SimpleMockTokenizer;

    impl Tokenizer for SimpleMockTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            // Simple tokenization for testing
            Ok(text.chars().map(|c| c as u32).collect())
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            Ok(tokens.iter().map(|&t| char::from(t as u8)).collect())
        }

        fn vocab_size(&self) -> usize {
            65536 // Simple vocab size
        }
    }
}
```

### Option 3: Builder Pattern for Configurable Mocks

Create flexible mocks using builder pattern:

```rust
// In crates/bitnet-test-utils/src/builders/mod.rs

pub struct MockModelBuilder {
    config: BitNetConfig,
    responses: HashMap<String, Tensor>,
    behavior: MockBehavior,
}

#[derive(Default)]
pub enum MockBehavior {
    #[default]
    Synthetic,      // Generate synthetic responses
    Echo,           // Echo input as output
    Constant(Tensor), // Always return constant tensor
    Error(String),  // Always return error
}

impl MockModelBuilder {
    pub fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            responses: HashMap::new(),
            behavior: MockBehavior::Synthetic,
        }
    }

    pub fn with_config(mut self, config: BitNetConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_behavior(mut self, behavior: MockBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    pub fn with_response(mut self, input: String, output: Tensor) -> Self {
        self.responses.insert(input, output);
        self
    }

    pub fn build(self) -> MockModel {
        MockModel {
            config: self.config,
            responses: self.responses,
            behavior: self.behavior,
        }
    }
}

// Usage in tests:
#[test]
fn test_inference_with_custom_model() {
    let mock_model = MockModelBuilder::new()
        .with_behavior(MockBehavior::Constant(expected_output))
        .build();

    let engine = ProductionInferenceEngine::new(mock_model, mock_tokenizer);
    // ... test logic
}
```

## Implementation Plan

### Phase 1: Audit and Analysis
- [ ] Identify all mock implementations across the codebase
- [ ] Analyze which mock methods are actually used in tests
- [ ] Document common patterns and requirements for mock objects
- [ ] Assess test coverage and performance impact

### Phase 2: Test Utilities Crate Setup
- [ ] Create `crates/bitnet-test-utils` crate with appropriate dependencies
- [ ] Design flexible mock interfaces based on actual usage patterns
- [ ] Implement core mock objects with builder patterns
- [ ] Add documentation and usage examples

### Phase 3: Migration and Consolidation
- [ ] Migrate existing tests to use centralized mock objects
- [ ] Remove duplicated mock implementations
- [ ] Update test dependencies to use test utilities crate
- [ ] Simplify test code by removing unnecessary mock complexity

### Phase 4: Testing and Validation
- [ ] Ensure all tests still pass with new mock implementations
- [ ] Validate test performance impact (should improve)
- [ ] Add integration tests for mock utilities themselves
- [ ] Update documentation for test development workflows

## Testing Strategy

### Mock Validation Testing
```rust
#[test]
fn test_mock_model_behavior() {
    let mock = MockModelBuilder::new()
        .with_behavior(MockBehavior::Echo)
        .build();

    let input = Tensor::ones((1, 10, 512), DType::F32, &Device::Cpu).unwrap();
    let output = mock.forward(&input).unwrap();

    // Verify mock behaves as expected
    assert_eq!(input.shape(), output.shape());
}

#[test]
fn test_mock_builder_pattern() {
    let expected_output = Tensor::zeros((1, 10, 1000), DType::F32, &Device::Cpu).unwrap();
    let mock = MockModelBuilder::new()
        .with_response("test_input".to_string(), expected_output.clone())
        .build();

    // Verify builder pattern works correctly
    assert!(mock.responses.contains_key("test_input"));
}
```

### Performance Testing
```rust
#[test]
fn test_mock_performance() {
    let mock = MockModelBuilder::new()
        .with_behavior(MockBehavior::Synthetic)
        .build();

    let start = Instant::now();
    for _ in 0..100 {
        let input = create_test_tensor();
        let _output = mock.forward(&input).unwrap();
    }
    let duration = start.elapsed();

    // Mock should be fast for testing
    assert!(duration < Duration::from_millis(100));
}
```

## Related Issues/PRs

- Test utilities and testing framework improvements
- Performance optimization of test suites
- Developer experience improvements for writing tests
- Code organization and modularization

## Acceptance Criteria

### For Consolidation (Option 1)
- [ ] Dedicated `bitnet-test-utils` crate created with mock objects
- [ ] All duplicate mock implementations removed from individual test files
- [ ] Builder pattern implemented for configurable mock behavior
- [ ] Test execution time improved by at least 20%
- [ ] All existing tests continue to pass
- [ ] Documentation provided for using test utilities
- [ ] Mock objects support common testing patterns (synthetic data, error injection, etc.)

### For Simplification (Option 2)
- [ ] Mock implementations simplified to minimal required functionality
- [ ] Code duplication eliminated within existing test modules
- [ ] Test readability and maintainability improved
- [ ] Mock complexity reduced by at least 50%
- [ ] All tests continue to validate the intended behavior

## Notes

The goal is to balance test utility and maintainability. Mock objects should be simple enough to understand quickly but flexible enough to support various testing scenarios. Consider the testing pyramid principle - unit tests should be fast and focused, so mocks should be lightweight.

If mock objects are reused across multiple crates, a dedicated test utilities crate is the best approach. If they're only used within a single crate, simplified in-place mocks may be sufficient.
