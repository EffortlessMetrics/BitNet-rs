# [Test Architecture] Refactor extensive mock objects in `engine.rs` test module

## Problem Description

The test module in `crates/bitnet-inference/src/engine.rs` contains extensive `MockModel` and `MockTokenizer` implementations that are well-contained within `#[cfg(test)]` blocks but could be optimized for maintainability and reusability. While the current implementation correctly provides comprehensive test coverage, the mock objects are quite extensive and may benefit from simplification and potential extraction to dedicated test utilities.

## Environment

- **File**: `crates/bitnet-inference/src/engine.rs`
- **Components**: Test module with `MockModel` and `MockTokenizer` structs
- **Testing Framework**: Tokio async tests with comprehensive coverage
- **Current Architecture**: Inline mock implementations within the main engine module

## Root Cause Analysis

### Current Implementation Analysis

The mock objects provide comprehensive trait implementations:

```rust
struct MockModel {
    config: BitNetConfig,
}

impl Model for MockModel {
    fn config(&self) -> &BitNetConfig { &self.config }
    fn forward(&self, _input: &ConcreteTensor, _cache: &mut dyn std::any::Any) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 50257]))
    }
    fn logits(&self, _hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 10, 50257]))
    }
}

struct MockTokenizer;
impl Tokenizer for MockTokenizer {
    fn encode(&self, _text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }
    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        Ok("Mock decoded text".to_string())
    }
    fn vocab_size(&self) -> usize { 50257 }
}
```

### Issues Identified

1. **Mock Complexity**: The mock implementations are comprehensive but could be simplified for specific test scenarios
2. **Potential Reusability**: These mocks may be useful in other test modules across the inference crate
3. **Hardcoded Values**: Mock responses use hardcoded values that may not reflect realistic model behavior
4. **Limited Configurability**: Mocks cannot be easily configured for different test scenarios

## Impact Assessment

- **Severity**: Low - Does not affect production functionality
- **Maintainability**: Medium - Extensive inline mocks increase test module size
- **Test Quality**: Medium - Current approach provides good isolation but limited flexibility
- **Code Duplication Risk**: Medium - Similar mocks may be needed in other test modules

## Proposed Solution

### Primary Approach: Extract to Dedicated Test Utilities

Create a shared test utilities module that provides configurable mock implementations:

```rust
// crates/bitnet-inference/tests/utils/mocks.rs

pub struct ConfigurableMockModel {
    config: BitNetConfig,
    forward_response: ConcreteTensor,
    logits_response: ConcreteTensor,
}

impl ConfigurableMockModel {
    pub fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            forward_response: ConcreteTensor::mock(vec![1, 50257]),
            logits_response: ConcreteTensor::mock(vec![1, 10, 50257]),
        }
    }

    pub fn with_config(mut self, config: BitNetConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_forward_response(mut self, response: ConcreteTensor) -> Self {
        self.forward_response = response;
        self
    }
}

impl Model for ConfigurableMockModel {
    fn config(&self) -> &BitNetConfig { &self.config }
    fn forward(&self, _input: &ConcreteTensor, _cache: &mut dyn std::any::Any) -> Result<ConcreteTensor> {
        Ok(self.forward_response.clone())
    }
    fn logits(&self, _hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        Ok(self.logits_response.clone())
    }
}

pub struct ConfigurableMockTokenizer {
    vocab_size: usize,
    encode_response: Vec<u32>,
    decode_response: String,
}

impl ConfigurableMockTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257,
            encode_response: vec![1, 2, 3],
            decode_response: "Mock decoded text".to_string(),
        }
    }

    pub fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    pub fn with_encode_response(mut self, tokens: Vec<u32>) -> Self {
        self.encode_response = tokens;
        self
    }
}

impl Tokenizer for ConfigurableMockTokenizer {
    fn encode(&self, _text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        Ok(self.encode_response.clone())
    }
    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        Ok(self.decode_response.clone())
    }
    fn vocab_size(&self) -> usize { self.vocab_size }
}
```

### Alternative Approach: Simplify Inline Mocks

If extraction is not desired, simplify the current inline implementations:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Simplified mock focused only on required functionality
    struct SimpleMockModel;
    impl Model for SimpleMockModel {
        fn config(&self) -> &BitNetConfig {
            static CONFIG: BitNetConfig = BitNetConfig::default();
            &CONFIG
        }
        fn forward(&self, _: &ConcreteTensor, _: &mut dyn std::any::Any) -> Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 50257]))
        }
        fn logits(&self, _: &ConcreteTensor) -> Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 10, 50257]))
        }
    }

    struct SimpleMockTokenizer;
    impl Tokenizer for SimpleMockTokenizer {
        fn encode(&self, _: &str, _: bool, _: bool) -> Result<Vec<u32>> { Ok(vec![1, 2, 3]) }
        fn decode(&self, _: &[u32]) -> Result<String> { Ok("test".to_string()) }
        fn vocab_size(&self) -> usize { 50257 }
    }
}
```

## Implementation Plan

### Phase 1: Assessment and Planning
- [ ] Analyze current mock usage across all test methods
- [ ] Identify common patterns and requirements
- [ ] Determine if other test modules could benefit from shared mocks
- [ ] Choose between extraction vs. simplification approach

### Phase 2: Implementation
- [ ] Create test utilities module structure if extraction approach is chosen
- [ ] Implement configurable mock objects with builder pattern
- [ ] Add comprehensive documentation and usage examples
- [ ] Ensure mock behavior remains consistent with existing tests

### Phase 3: Migration and Integration
- [ ] Update existing tests to use new mock implementations
- [ ] Verify all tests continue to pass with identical behavior
- [ ] Add new test cases that leverage mock configurability
- [ ] Update module-level documentation

### Phase 4: Validation and Documentation
- [ ] Run full test suite to ensure no regressions
- [ ] Add examples of mock usage in different scenarios
- [ ] Document mock architecture in test documentation
- [ ] Consider adding property-based testing with configurable mocks

## Testing Strategy

### Regression Testing
```bash
# Ensure all existing tests continue to pass
cargo test --package bitnet-inference engine --no-default-features --features cpu

# Run with different feature combinations
cargo test --package bitnet-inference engine --no-default-features --features gpu
```

### Mock Validation Testing
```rust
#[cfg(test)]
mod mock_validation_tests {
    use super::*;

    #[test]
    fn test_mock_model_consistency() {
        let mock = ConfigurableMockModel::new();
        let config = mock.config();
        assert_eq!(config.vocab_size, 50257);

        // Verify forward pass behavior
        let input = ConcreteTensor::mock(vec![1, 10]);
        let cache = &mut ();
        let output = mock.forward(&input, cache).unwrap();
        assert_eq!(output.shape(), vec![1, 50257]);
    }

    #[test]
    fn test_mock_tokenizer_consistency() {
        let mock = ConfigurableMockTokenizer::new();
        assert_eq!(mock.vocab_size(), 50257);

        let tokens = mock.encode("test", true, true).unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);

        let text = mock.decode(&tokens).unwrap();
        assert!(!text.is_empty());
    }
}
```

### Performance Impact Testing
```rust
#[test]
fn test_mock_performance_overhead() {
    let start = std::time::Instant::now();
    let mock = ConfigurableMockModel::new();
    let creation_time = start.elapsed();

    assert!(creation_time.as_micros() < 100, "Mock creation should be fast");
}
```

## Related Issues/PRs

- Links to any existing test architecture discussions
- Related to overall test infrastructure improvements
- May relate to cross-validation testing framework

## Acceptance Criteria

- [ ] Mock objects are simplified or extracted to reusable utilities
- [ ] All existing tests continue to pass without modification (if simplification) or with minimal changes (if extraction)
- [ ] Mock implementations remain focused on test requirements without unnecessary complexity
- [ ] Documentation clearly explains mock architecture and usage patterns
- [ ] Performance overhead of mock objects is minimal
- [ ] Code duplication is eliminated if mocks are reused across modules

## Priority: Low

This is a code quality and maintainability improvement that enhances the test architecture without affecting production functionality. While beneficial for long-term maintenance, it can be addressed during regular refactoring cycles.