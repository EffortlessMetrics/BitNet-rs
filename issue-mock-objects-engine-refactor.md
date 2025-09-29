# [Testing] Refactor extensive mock objects in `engine.rs` test module for improved maintainability

## Problem Description

The `tests` module in `crates/bitnet-inference/src/engine.rs` contains extensive mock implementations (`MockModel` and `MockTokenizer`) that are well-designed but could benefit from refactoring to improve maintainability and reusability. While these mocks are correctly placed within `#[cfg(test)]` blocks, they implement comprehensive trait methods that may not all be necessary for the current test scenarios.

## Environment

- **File:** `crates/bitnet-inference/src/engine.rs` (lines 1759-1822)
- **Affected Components:** Test infrastructure, `InferenceEngine` testing
- **Traits Implemented:** `Model`, `Tokenizer`
- **Current Test Coverage:** 8+ test functions using these mocks

## Root Cause Analysis

1. **Mock Complexity**: The `MockModel` and `MockTokenizer` structs implement all trait methods, even those that return static values or aren't critical for current test scenarios
2. **Code Duplication Risk**: These well-designed mocks could be reused across other test modules but are currently isolated to `engine.rs`
3. **Maintenance Overhead**: Changes to the `Model` or `Tokenizer` traits require updates to these extensive mock implementations
4. **Test Clarity**: Some mock methods return hardcoded values that may not reflect realistic test scenarios

## Current Implementation Analysis

### MockModel Implementation
```rust
struct MockModel {
    config: BitNetConfig,
}

impl Model for MockModel {
    fn config(&self) -> &BitNetConfig { &self.config }
    fn forward(&self, _input: &ConcreteTensor, _cache: &mut dyn std::any::Any) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 50257]))  // Hardcoded dimensions
    }
    fn embed(&self, _tokens: &[u32]) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 10, 768]))  // Hardcoded shape
    }
    fn logits(&self, _hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 10, 50257]))  // Hardcoded vocab size
    }
}
```

### MockTokenizer Implementation
```rust
struct MockTokenizer;

impl Tokenizer for MockTokenizer {
    fn encode(&self, _text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        Ok(vec![1, 2, 3])  // Static token sequence
    }
    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        Ok("mock generated text".to_string())  // Static output
    }
    fn vocab_size(&self) -> usize { 50257 }  // GPT-2 vocab size hardcoded
    // ... other methods with static returns
}
```

## Impact Assessment

**Severity:** Low-Medium
**Component:** Testing Infrastructure
**Affected Areas:**
- Test maintainability and readability
- Code reusability across test modules
- Mock reliability for different test scenarios
- Future trait evolution compatibility

## Proposed Solution

### 1. Create Dedicated Test Utilities Module

Create `crates/bitnet-inference/tests/common/mod.rs` for shared test utilities:

```rust
// crates/bitnet-inference/tests/common/mod.rs
use bitnet_common::{BitNetConfig, ConcreteTensor, Result};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;

pub struct ConfigurableMockModel {
    config: BitNetConfig,
    forward_shape: Vec<usize>,
    embed_shape: Vec<usize>,
    logits_shape: Vec<usize>,
}

impl ConfigurableMockModel {
    pub fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            forward_shape: vec![1, 50257],
            embed_shape: vec![1, 10, 768],
            logits_shape: vec![1, 10, 50257],
        }
    }

    pub fn with_shapes(mut self, forward: Vec<usize>, embed: Vec<usize>, logits: Vec<usize>) -> Self {
        self.forward_shape = forward;
        self.embed_shape = embed;
        self.logits_shape = logits;
        self
    }

    pub fn with_config(mut self, config: BitNetConfig) -> Self {
        self.config = config;
        self
    }
}

impl Model for ConfigurableMockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(&self, _input: &ConcreteTensor, _cache: &mut dyn std::any::Any) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(self.forward_shape.clone()))
    }

    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor> {
        let mut shape = self.embed_shape.clone();
        if shape.len() >= 2 {
            shape[1] = tokens.len();  // Dynamic sequence length
        }
        Ok(ConcreteTensor::mock(shape))
    }

    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        let mut shape = self.logits_shape.clone();
        if let Some(seq_len) = hidden.shape().get(1) {
            if shape.len() >= 2 {
                shape[1] = *seq_len;  // Match input sequence length
            }
        }
        Ok(ConcreteTensor::mock(shape))
    }
}

pub struct ConfigurableMockTokenizer {
    vocab_size: usize,
    encode_output: Vec<u32>,
    decode_output: String,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl ConfigurableMockTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257,
            encode_output: vec![1, 2, 3],
            decode_output: "mock generated text".to_string(),
            eos_token_id: Some(50256),
            pad_token_id: None,
        }
    }

    pub fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    pub fn with_encode_output(mut self, tokens: Vec<u32>) -> Self {
        self.encode_output = tokens;
        self
    }

    pub fn with_decode_output(mut self, text: String) -> Self {
        self.decode_output = text;
        self
    }
}

impl Tokenizer for ConfigurableMockTokenizer {
    fn encode(&self, text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        // Simple length-based encoding for more realistic behavior
        let mut tokens = self.encode_output.clone();
        tokens.extend(std::iter::repeat(1).take(text.len() / 4));  // Rough text-to-token ratio
        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        if tokens.is_empty() {
            Ok(String::new())
        } else {
            Ok(format!("{} (from {} tokens)", self.decode_output, tokens.len()))
        }
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        Some(format!("piece_{}", token))
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }
}

// Convenience constructors for common test scenarios
pub fn simple_mock_model() -> ConfigurableMockModel {
    ConfigurableMockModel::new()
}

pub fn simple_mock_tokenizer() -> ConfigurableMockTokenizer {
    ConfigurableMockTokenizer::new()
}
```

### 2. Update engine.rs Tests

Replace current mock implementations with shared utilities:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Import from common test utilities
    use crate::tests::common::{simple_mock_model, simple_mock_tokenizer, ConfigurableMockModel, ConfigurableMockTokenizer};

    #[tokio::test]
    async fn test_inference_engine_creation() {
        let model = Arc::new(simple_mock_model());
        let tokenizer = Arc::new(simple_mock_tokenizer());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_prefill_with_custom_shapes() {
        let model = Arc::new(
            ConfigurableMockModel::new()
                .with_shapes(
                    vec![1, 32000],      // Custom vocab size
                    vec![1, 20, 512],    // Custom embedding dims
                    vec![1, 20, 32000]   // Matching logits
                )
        );
        let tokenizer = Arc::new(
            ConfigurableMockTokenizer::new()
                .with_vocab_size(32000)
                .with_encode_output(vec![1, 15, 23, 42, 100])
        );
        let device = Device::Cpu;

        let mut engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let tokens = vec![1, 15, 23, 42, 100];

        let result = engine.prefill(&tokens).await;
        assert!(result.is_ok(), "Prefill should handle custom configurations");
    }

    // ... other tests using configurable mocks
}
```

### 3. Add Common Test Utilities Crate Structure

```
crates/bitnet-inference/tests/
├── common/
│   ├── mod.rs          # Main test utilities
│   ├── mock_model.rs   # Detailed MockModel implementations
│   ├── mock_tokenizer.rs # Detailed MockTokenizer implementations
│   └── fixtures.rs     # Test data fixtures
└── integration_tests.rs
```

## Implementation Plan

### Phase 1: Create Test Utilities Infrastructure
- [ ] Create `tests/common/mod.rs` with configurable mock implementations
- [ ] Implement `ConfigurableMockModel` with builder pattern
- [ ] Implement `ConfigurableMockTokenizer` with builder pattern
- [ ] Add convenience constructors for common test scenarios

### Phase 2: Migrate Existing Tests
- [ ] Update `engine.rs` tests to use new shared utilities
- [ ] Remove old `MockModel` and `MockTokenizer` from `engine.rs`
- [ ] Ensure all existing tests pass with new implementations
- [ ] Add regression tests for mock configurability

### Phase 3: Enhance Test Coverage
- [ ] Add tests with different model configurations
- [ ] Add tests with various tokenizer settings
- [ ] Create performance comparison tests
- [ ] Add documentation for test utility usage

### Phase 4: Validation and Documentation
- [ ] Run full test suite to ensure no regressions
- [ ] Update test documentation with new patterns
- [ ] Create examples for other test modules
- [ ] Performance validation of mock implementations

## Testing Strategy

### Unit Tests
```bash
# Test the new mock utilities
cargo test --package bitnet-inference tests::common

# Test engine.rs with new mocks
cargo test --package bitnet-inference engine::tests

# Full inference package tests
cargo test --package bitnet-inference --no-default-features --features cpu
```

### Integration Tests
```bash
# Cross-module mock usage tests
cargo test --package bitnet-inference --test integration_tests

# Performance regression tests
cargo test --package bitnet-inference --release -- --nocapture performance
```

### Validation Criteria
- [ ] All existing tests pass with new mock implementations
- [ ] Mock setup time is < 1ms for simple configurations
- [ ] Memory usage for mocks is < 1MB
- [ ] New mocks support all existing test scenarios
- [ ] Test code readability improved by configurable mocks

## Alternative Approaches

### 1. Minimal Refactor (Lower Impact)
- Keep existing mocks in place
- Add builder methods to existing structs
- Create type aliases for common configurations

### 2. Trait-Based Mock System (Higher Complexity)
- Create mock traits that can be customized per test
- Use procedural macros for mock generation
- Support runtime behavior modification

### 3. External Mock Library Integration
- Integrate with `mockall` or similar crate
- Generate mocks automatically from traits
- Trade simplicity for advanced mocking features

## Risk Assessment

**Low Risk:**
- Changes are isolated to test modules
- No impact on production code paths
- Backward compatibility maintained during transition

**Potential Issues:**
- Test migration complexity if mocks have subtle behavioral differences
- Need to ensure new mocks maintain test validity
- Documentation updates required for test patterns

## Success Criteria

1. **Code Quality**: Reduced duplication, improved maintainability
2. **Test Flexibility**: Easy configuration for different test scenarios
3. **Reusability**: Mocks available across multiple test modules
4. **Performance**: No degradation in test execution time
5. **Documentation**: Clear examples for using new test utilities

## Related Issues

- Mock improvements should consider future `Model` and `Tokenizer` trait evolution
- Test utilities could be expanded for other BitNet.rs components
- Consider integration with existing `ConcreteTensor::mock()` patterns

---

**Labels:** `testing`, `refactoring`, `maintenance`, `low-priority`
**Assignee:** Test Infrastructure Team
**Epic:** Testing Infrastructure Improvements