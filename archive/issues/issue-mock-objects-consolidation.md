# [Testing] Consolidate and optimize extensive mock objects in production_engine.rs

## Problem Description

The `tests` module in `crates/bitnet-inference/src/production_engine.rs` contains extensive mock implementations (`MockModel` and `MockTokenizer`) that are well-placed within `#[cfg(test)]` blocks but could benefit from consolidation, optimization, and potential reuse across the testing infrastructure. These mocks are currently duplicated across different test files, making maintenance difficult and testing inconsistent.

## Environment
- **File**: `crates/bitnet-inference/src/production_engine.rs`
- **Test Module**: `#[cfg(test)] mod tests`
- **Mock Objects**: `MockModel`, `MockTokenizer`
- **MSRV**: Rust 1.90.0
- **Testing Framework**: Standard Rust testing + potential proptest integration

## Reproduction Steps

1. Examine the current mock implementations:
   ```bash
   cd /home/steven/code/Rust/BitNet-rs
   rg -A 10 -B 2 "MockModel\|MockTokenizer" --type rust
   ```

2. Check for duplicate mock patterns across test files:
   ```bash
   find . -name "*.rs" -exec grep -l "Mock.*Model\|Mock.*Tokenizer" {} \;
   ```

3. Run tests to see mock usage:
   ```bash
   cargo test -p bitnet-inference production_engine
   ```

**Current State**:
- Mock objects are extensive but potentially over-engineered
- Duplicated mock logic across different test modules
- Inconsistent mock behavior between tests

**Desired State**:
- Centralized, reusable mock implementations
- Consistent behavior across all tests
- Simplified mock objects focused on testing needs
- Improved test performance and maintainability

## Root Cause Analysis

### Current Mock Implementation Issues

1. **Code Duplication**: Similar mock objects exist in multiple test files
2. **Over-Engineering**: Mocks implement more functionality than tests require
3. **Inconsistent Behavior**: Different mock implementations may behave differently
4. **Maintenance Burden**: Changes require updates across multiple files
5. **Testing Gaps**: Complex mocks may hide real implementation issues

### Mock Object Complexity

The current mock objects are feature-complete but may be unnecessarily complex:

```rust
#[cfg(test)]
mod tests {
    // Extensive MockModel implementation
    struct MockModel {
        config: BitNetConfig,
        // ... many fields for comprehensive mocking
    }

    impl Model for MockModel {
        // Full trait implementation with complex logic
        // May be more than what tests actually need
    }

    // Extensive MockTokenizer implementation
    struct MockTokenizer {
        vocab: HashMap<String, u32>,
        // ... comprehensive tokenizer state
    }

    impl Tokenizer for MockTokenizer {
        // Full trait implementation
        // Potentially duplicates logic from other test files
    }
}
```

## Impact Assessment

- **Severity**: Low-Medium (code quality and maintainability)
- **Development Impact**:
  - Increased test maintenance overhead
  - Potential for inconsistent test behavior
  - Slower test execution due to complex mocks
  - Difficulty adding new tests due to mock complexity

- **Code Quality Impact**:
  - Reduced maintainability of test suite
  - Potential for mock drift from real implementations
  - Increased cognitive load for developers
  - Harder to ensure comprehensive test coverage

## Proposed Solution

Create a centralized, optimized mock object framework that provides consistent, lightweight mocks for testing while eliminating duplication and improving maintainability.

### Technical Implementation

#### 1. Centralized Mock Framework

```rust
// crates/bitnet-inference/tests/common/mocks.rs
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use bitnet_inference::{Model, Tokenizer, BitNetConfig, BitNetTensor};

/// Lightweight mock model for testing
#[derive(Debug, Clone)]
pub struct MockModel {
    config: BitNetConfig,
    behavior: MockModelBehavior,
    call_log: Arc<Mutex<Vec<MockCall>>>,
}

#[derive(Debug, Clone)]
pub struct MockModelBehavior {
    /// Predefined outputs for forward calls
    pub forward_outputs: Vec<BitNetTensor>,
    /// Whether to simulate errors
    pub should_error: bool,
    /// Simulated processing delay
    pub processing_delay: Option<std::time::Duration>,
    /// Custom output generator
    pub output_generator: Option<Arc<dyn Fn(&BitNetTensor) -> Result<BitNetTensor> + Send + Sync>>,
}

#[derive(Debug, Clone)]
pub enum MockCall {
    Forward { input_shape: Vec<usize>, timestamp: std::time::Instant },
    LoadWeights { timestamp: std::time::Instant },
    GetConfig { timestamp: std::time::Instant },
}

impl MockModel {
    /// Create a simple mock model with default behavior
    pub fn new_simple(vocab_size: usize, hidden_size: usize) -> Self {
        let config = BitNetConfig {
            vocab_size,
            hidden_size,
            num_layers: 12,
            num_heads: 12,
            max_position_embeddings: 2048,
            quantization_type: QuantizationType::I2S,
            use_bias: false,
        };

        Self {
            config,
            behavior: MockModelBehavior::default(),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a mock model with custom behavior
    pub fn with_behavior(config: BitNetConfig, behavior: MockModelBehavior) -> Self {
        Self {
            config,
            behavior,
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a mock that returns specific outputs
    pub fn with_outputs(config: BitNetConfig, outputs: Vec<BitNetTensor>) -> Self {
        Self::with_behavior(config, MockModelBehavior {
            forward_outputs: outputs,
            ..Default::default()
        })
    }

    /// Create a mock that simulates errors
    pub fn with_errors(config: BitNetConfig) -> Self {
        Self::with_behavior(config, MockModelBehavior {
            should_error: true,
            ..Default::default()
        })
    }

    /// Get the call log for verification
    pub fn call_log(&self) -> Vec<MockCall> {
        self.call_log.lock().unwrap().clone()
    }

    /// Reset the call log
    pub fn reset_call_log(&self) {
        self.call_log.lock().unwrap().clear();
    }

    fn log_call(&self, call: MockCall) {
        self.call_log.lock().unwrap().push(call);
    }
}

impl Model for MockModel {
    fn forward(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        self.log_call(MockCall::Forward {
            input_shape: input.shape().to_vec(),
            timestamp: std::time::Instant::now(),
        });

        if self.behavior.should_error {
            return Err(BitNetError::ModelError("Mock error".to_string()));
        }

        if let Some(delay) = self.behavior.processing_delay {
            std::thread::sleep(delay);
        }

        // Use custom generator if available
        if let Some(ref generator) = self.behavior.output_generator {
            return generator(input);
        }

        // Use predefined outputs if available
        if !self.behavior.forward_outputs.is_empty() {
            let output_index = (self.call_log().len() - 1) % self.behavior.forward_outputs.len();
            return Ok(self.behavior.forward_outputs[output_index].clone());
        }

        // Default: create a simple output tensor
        let batch_size = input.shape()[0];
        let vocab_size = self.config.vocab_size;

        // Create logits with shape [batch_size, vocab_size]
        let logits = BitNetTensor::zeros(
            &[batch_size, vocab_size],
            input.dtype(),
            input.device()
        )?;

        Ok(logits)
    }

    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn load_weights(&mut self, _weights: &[u8]) -> Result<()> {
        self.log_call(MockCall::LoadWeights {
            timestamp: std::time::Instant::now(),
        });

        if self.behavior.should_error {
            return Err(BitNetError::ModelError("Failed to load weights".to_string()));
        }

        Ok(())
    }

    fn memory_usage(&self) -> usize {
        // Simple estimation for testing
        self.config.vocab_size * self.config.hidden_size * 2 // 2 bytes per parameter
    }
}

impl Default for MockModelBehavior {
    fn default() -> Self {
        Self {
            forward_outputs: Vec::new(),
            should_error: false,
            processing_delay: None,
            output_generator: None,
        }
    }
}

/// Lightweight mock tokenizer for testing
#[derive(Debug, Clone)]
pub struct MockTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    behavior: MockTokenizerBehavior,
    call_log: Arc<Mutex<Vec<MockTokenizerCall>>>,
}

#[derive(Debug, Clone)]
pub struct MockTokenizerBehavior {
    pub should_error: bool,
    pub encoding_delay: Option<std::time::Duration>,
    pub custom_vocab: Option<HashMap<String, u32>>,
}

#[derive(Debug, Clone)]
pub enum MockTokenizerCall {
    Encode { text: String, timestamp: std::time::Instant },
    Decode { tokens: Vec<u32>, timestamp: std::time::Instant },
    TokenToPiece { token: u32, timestamp: std::time::Instant },
}

impl MockTokenizer {
    /// Create a simple mock tokenizer with basic vocabulary
    pub fn new_simple() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Basic vocabulary for testing
        let basic_words = vec![
            "<pad>", "<unk>", "<bos>", "<eos>",
            "the", "a", "an", "and", "or", "but",
            "hello", "world", "test", "example", "mock",
            "model", "tokenizer", "bitnet", "inference",
        ];

        for (id, word) in basic_words.iter().enumerate() {
            vocab.insert(word.to_string(), id as u32);
            reverse_vocab.insert(id as u32, word.to_string());
        }

        Self {
            vocab,
            reverse_vocab,
            behavior: MockTokenizerBehavior::default(),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a mock tokenizer with custom vocabulary
    pub fn with_vocab(vocab: HashMap<String, u32>) -> Self {
        let reverse_vocab: HashMap<u32, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        Self {
            vocab,
            reverse_vocab,
            behavior: MockTokenizerBehavior::default(),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a mock tokenizer that simulates errors
    pub fn with_errors() -> Self {
        let mut tokenizer = Self::new_simple();
        tokenizer.behavior.should_error = true;
        tokenizer
    }

    pub fn call_log(&self) -> Vec<MockTokenizerCall> {
        self.call_log.lock().unwrap().clone()
    }

    pub fn reset_call_log(&self) {
        self.call_log.lock().unwrap().clear();
    }

    fn log_call(&self, call: MockTokenizerCall) {
        self.call_log.lock().unwrap().push(call);
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.log_call(MockTokenizerCall::Encode {
            text: text.to_string(),
            timestamp: std::time::Instant::now(),
        });

        if self.behavior.should_error {
            return Err(BitNetError::TokenizationError("Mock encoding error".to_string()));
        }

        if let Some(delay) = self.behavior.encoding_delay {
            std::thread::sleep(delay);
        }

        // Simple word-based tokenization for testing
        let tokens: Vec<u32> = text
            .split_whitespace()
            .map(|word| {
                self.vocab.get(word).copied().unwrap_or(1) // 1 = <unk>
            })
            .collect();

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.log_call(MockTokenizerCall::Decode {
            tokens: tokens.to_vec(),
            timestamp: std::time::Instant::now(),
        });

        if self.behavior.should_error {
            return Err(BitNetError::TokenizationError("Mock decoding error".to_string()));
        }

        let words: Vec<String> = tokens
            .iter()
            .map(|&token| {
                self.reverse_vocab.get(&token)
                    .unwrap_or(&"<unk>".to_string())
                    .clone()
            })
            .collect();

        Ok(words.join(" "))
    }

    fn token_to_piece(&self, token: u32) -> Result<String> {
        self.log_call(MockTokenizerCall::TokenToPiece {
            token,
            timestamp: std::time::Instant::now(),
        });

        if self.behavior.should_error {
            return Err(BitNetError::TokenizationError("Mock token_to_piece error".to_string()));
        }

        Ok(self.reverse_vocab.get(&token)
           .unwrap_or(&"<unk>".to_string())
           .clone())
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn eos_token_id(&self) -> u32 {
        3 // "<eos>" in our basic vocabulary
    }

    fn bos_token_id(&self) -> u32 {
        2 // "<bos>" in our basic vocabulary
    }

    fn pad_token_id(&self) -> u32 {
        0 // "<pad>" in our basic vocabulary
    }
}

impl Default for MockTokenizerBehavior {
    fn default() -> Self {
        Self {
            should_error: false,
            encoding_delay: None,
            custom_vocab: None,
        }
    }
}
```

#### 2. Test Utilities and Builders

```rust
// crates/bitnet-inference/tests/common/test_utils.rs
use super::mocks::{MockModel, MockTokenizer, MockModelBehavior, MockTokenizerBehavior};
use bitnet_inference::{BitNetConfig, BitNetTensor, QuantizationType};

/// Builder for creating test configurations
pub struct TestConfigBuilder {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    quantization_type: QuantizationType,
}

impl TestConfigBuilder {
    pub fn new() -> Self {
        Self {
            vocab_size: 1000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            quantization_type: QuantizationType::I2S,
        }
    }

    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    pub fn hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }

    pub fn small_model(mut self) -> Self {
        self.vocab_size = 100;
        self.hidden_size = 64;
        self.num_layers = 2;
        self.num_heads = 2;
        self
    }

    pub fn large_model(mut self) -> Self {
        self.vocab_size = 50000;
        self.hidden_size = 4096;
        self.num_layers = 48;
        self.num_heads = 32;
        self
    }

    pub fn build(self) -> BitNetConfig {
        BitNetConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            max_position_embeddings: 2048,
            quantization_type: self.quantization_type,
            use_bias: false,
        }
    }
}

/// Builder for creating mock models with specific behaviors
pub struct MockModelBuilder {
    config: BitNetConfig,
    behavior: MockModelBehavior,
}

impl MockModelBuilder {
    pub fn new(config: BitNetConfig) -> Self {
        Self {
            config,
            behavior: MockModelBehavior::default(),
        }
    }

    pub fn with_error(mut self) -> Self {
        self.behavior.should_error = true;
        self
    }

    pub fn with_delay(mut self, delay: std::time::Duration) -> Self {
        self.behavior.processing_delay = Some(delay);
        self
    }

    pub fn with_outputs(mut self, outputs: Vec<BitNetTensor>) -> Self {
        self.behavior.forward_outputs = outputs;
        self
    }

    pub fn with_custom_generator<F>(mut self, generator: F) -> Self
    where
        F: Fn(&BitNetTensor) -> Result<BitNetTensor> + Send + Sync + 'static,
    {
        self.behavior.output_generator = Some(Arc::new(generator));
        self
    }

    pub fn build(self) -> MockModel {
        MockModel::with_behavior(self.config, self.behavior)
    }
}

/// Test data generators
pub struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate a test input tensor
    pub fn input_tensor(batch_size: usize, sequence_length: usize) -> Result<BitNetTensor> {
        let shape = &[batch_size, sequence_length];
        let data: Vec<u32> = (0..batch_size * sequence_length)
            .map(|i| (i % 1000) as u32)  // Token IDs 0-999
            .collect();

        BitNetTensor::new(data, shape, candle_core::DType::U32, &candle_core::Device::Cpu)
    }

    /// Generate expected output logits
    pub fn output_logits(batch_size: usize, vocab_size: usize) -> Result<BitNetTensor> {
        let shape = &[batch_size, vocab_size];
        let data: Vec<f32> = (0..batch_size * vocab_size)
            .map(|i| (i as f32 * 0.01) % 1.0)  // Simple pattern for testing
            .collect();

        BitNetTensor::new(data, shape, candle_core::DType::F32, &candle_core::Device::Cpu)
    }

    /// Generate test text samples
    pub fn text_samples(count: usize) -> Vec<String> {
        let base_texts = vec![
            "The quick brown fox jumps over the lazy dog",
            "Hello world, this is a test",
            "Machine learning models are fascinating",
            "BitNet inference with quantization",
            "Testing mock objects in Rust",
        ];

        (0..count)
            .map(|i| {
                let base = &base_texts[i % base_texts.len()];
                if i < base_texts.len() {
                    base.to_string()
                } else {
                    format!("{} {}", base, i)
                }
            })
            .collect()
    }
}

/// Common test assertions
pub trait TestAssertions {
    fn assert_shape_eq(&self, expected: &[usize]);
    fn assert_device_eq(&self, expected: &candle_core::Device);
    fn assert_dtype_eq(&self, expected: candle_core::DType);
}

impl TestAssertions for BitNetTensor {
    fn assert_shape_eq(&self, expected: &[usize]) {
        assert_eq!(
            self.shape(),
            expected,
            "Tensor shape mismatch: expected {:?}, got {:?}",
            expected,
            self.shape()
        );
    }

    fn assert_device_eq(&self, expected: &candle_core::Device) {
        assert_eq!(
            self.device(),
            expected,
            "Tensor device mismatch: expected {:?}, got {:?}",
            expected,
            self.device()
        );
    }

    fn assert_dtype_eq(&self, expected: candle_core::DType) {
        assert_eq!(
            self.dtype(),
            expected,
            "Tensor dtype mismatch: expected {:?}, got {:?}",
            expected,
            self.dtype()
        );
    }
}
```

#### 3. Updated Production Engine Tests

```rust
// crates/bitnet-inference/src/production_engine.rs
#[cfg(test)]
mod tests {
    use super::*;

    // Import centralized mocks instead of defining locally
    use crate::tests::common::{
        mocks::{MockModel, MockTokenizer},
        test_utils::{TestConfigBuilder, MockModelBuilder, TestDataGenerator},
    };

    #[test]
    fn test_production_engine_basic_functionality() {
        let config = TestConfigBuilder::new().small_model().build();
        let model = MockModelBuilder::new(config.clone()).build();
        let tokenizer = MockTokenizer::new_simple();

        let engine = ProductionInferenceEngine::new(
            Arc::new(model),
            Arc::new(tokenizer),
            config,
        ).unwrap();

        let input_text = "test input";
        let result = engine.generate(input_text, 10).unwrap();

        assert!(!result.is_empty());
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_production_engine_error_handling() {
        let config = TestConfigBuilder::new().small_model().build();
        let model = MockModelBuilder::new(config.clone()).with_error().build();
        let tokenizer = MockTokenizer::with_errors();

        let engine = ProductionInferenceEngine::new(
            Arc::new(model),
            Arc::new(tokenizer),
            config,
        ).unwrap();

        let result = engine.generate("test", 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_production_engine_performance_characteristics() {
        let config = TestConfigBuilder::new().build();
        let delay = std::time::Duration::from_millis(10);
        let model = MockModelBuilder::new(config.clone())
            .with_delay(delay)
            .build();
        let tokenizer = MockTokenizer::new_simple();

        let engine = ProductionInferenceEngine::new(
            Arc::new(model),
            Arc::new(tokenizer),
            config,
        ).unwrap();

        let start = std::time::Instant::now();
        let _result = engine.generate("test input", 5).unwrap();
        let elapsed = start.elapsed();

        // Should take at least the delay time for forward passes
        assert!(elapsed >= delay);
    }

    #[test]
    fn test_production_engine_call_logging() {
        let config = TestConfigBuilder::new().small_model().build();
        let model = MockModelBuilder::new(config.clone()).build();
        let tokenizer = MockTokenizer::new_simple();

        // Reset call logs
        model.reset_call_log();
        tokenizer.reset_call_log();

        let engine = ProductionInferenceEngine::new(
            Arc::new(model.clone()),
            Arc::new(tokenizer.clone()),
            config,
        ).unwrap();

        engine.generate("hello world", 3).unwrap();

        // Verify expected calls were made
        let model_calls = model.call_log();
        let tokenizer_calls = tokenizer.call_log();

        assert!(!model_calls.is_empty(), "Model should have been called");
        assert!(!tokenizer_calls.is_empty(), "Tokenizer should have been called");

        // Should have both encode and decode calls
        assert!(tokenizer_calls.iter().any(|call| matches!(call, MockTokenizerCall::Encode { .. })));
        assert!(tokenizer_calls.iter().any(|call| matches!(call, MockTokenizerCall::Decode { .. })));
    }
}
```

#### 4. Property-Based Testing Integration

```rust
// crates/bitnet-inference/tests/property_tests.rs
#[cfg(test)]
mod property_tests {
    use super::common::{
        mocks::{MockModel, MockTokenizer},
        test_utils::{TestConfigBuilder, MockModelBuilder, TestDataGenerator},
    };
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_tokenizer_encode_decode_roundtrip(text in r"[a-zA-Z0-9 ]{1,100}") {
            let tokenizer = MockTokenizer::new_simple();

            let tokens = tokenizer.encode(&text).unwrap();
            let decoded = tokenizer.decode(&tokens).unwrap();

            // For our simple mock, this should be exact for known words
            // For unknown words, we accept approximation
            prop_assume!(!text.trim().is_empty());
        }

        #[test]
        fn test_model_forward_output_shape(
            batch_size in 1usize..5,
            vocab_size in 100usize..1000,
            hidden_size in 64usize..512
        ) {
            let config = TestConfigBuilder::new()
                .vocab_size(vocab_size)
                .hidden_size(hidden_size)
                .build();

            let model = MockModelBuilder::new(config).build();
            let input = TestDataGenerator::input_tensor(batch_size, 10).unwrap();

            let output = model.forward(&input).unwrap();

            // Output should have shape [batch_size, vocab_size]
            prop_assert_eq!(output.shape(), &[batch_size, vocab_size]);
        }
    }
}
```

## Implementation Plan

### Phase 1: Mock Framework Design (Week 1)
- [ ] Design centralized mock architecture
- [ ] Create base mock traits and structures
- [ ] Implement MockModel with configurable behavior
- [ ] Implement MockTokenizer with configurable behavior

### Phase 2: Test Utilities (Week 2)
- [ ] Create test builders and generators
- [ ] Add property-based testing support
- [ ] Implement call logging and verification
- [ ] Add performance testing utilities

### Phase 3: Migration and Consolidation (Week 3)
- [ ] Migrate existing tests to use centralized mocks
- [ ] Remove duplicate mock implementations
- [ ] Update all test files to use common framework
- [ ] Add comprehensive test coverage

### Phase 4: Documentation and Optimization (Week 4)
- [ ] Document mock framework usage
- [ ] Optimize mock performance
- [ ] Add examples and best practices
- [ ] Validate all tests pass with new framework

## Testing Strategy

### Mock Framework Tests
```rust
#[cfg(test)]
mod mock_framework_tests {
    use super::*;

    #[test]
    fn test_mock_model_basic_functionality() {
        let config = TestConfigBuilder::new().small_model().build();
        let model = MockModel::new_simple(config.vocab_size, config.hidden_size);

        let input = TestDataGenerator::input_tensor(1, 10).unwrap();
        let output = model.forward(&input).unwrap();

        output.assert_shape_eq(&[1, config.vocab_size]);
    }

    #[test]
    fn test_mock_model_call_logging() {
        let config = TestConfigBuilder::new().small_model().build();
        let model = MockModel::new_simple(config.vocab_size, config.hidden_size);

        assert_eq!(model.call_log().len(), 0);

        let input = TestDataGenerator::input_tensor(2, 5).unwrap();
        model.forward(&input).unwrap();

        let calls = model.call_log();
        assert_eq!(calls.len(), 1);

        if let MockCall::Forward { input_shape, .. } = &calls[0] {
            assert_eq!(input_shape, &[2, 5]);
        } else {
            panic!("Expected Forward call");
        }
    }

    #[test]
    fn test_mock_tokenizer_vocabulary() {
        let tokenizer = MockTokenizer::new_simple();

        // Test known words
        let tokens = tokenizer.encode("hello world").unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_mock_error_simulation() {
        let config = TestConfigBuilder::new().small_model().build();
        let model = MockModelBuilder::new(config).with_error().build();

        let input = TestDataGenerator::input_tensor(1, 10).unwrap();
        let result = model.forward(&input);

        assert!(result.is_err());
    }
}
```

### Performance Tests
```rust
#[cfg(test)]
mod performance_tests {
    #[test]
    fn test_mock_overhead() {
        let config = TestConfigBuilder::new().build();
        let model = MockModel::new_simple(config.vocab_size, config.hidden_size);
        let input = TestDataGenerator::input_tensor(1, 100).unwrap();

        let iterations = 1000;
        let start = std::time::Instant::now();

        for _ in 0..iterations {
            let _ = model.forward(&input).unwrap();
        }

        let elapsed = start.elapsed();
        let per_call = elapsed / iterations;

        // Mock should be very fast
        assert!(per_call < std::time::Duration::from_micros(100));
    }
}
```

## Performance Impact

### Improvements
- **Reduced Test Compilation Time**: Shared mock objects compile once
- **Faster Test Execution**: Lightweight mocks reduce overhead
- **Lower Memory Usage**: Elimination of duplicate mock implementations

### Metrics
- **Mock Creation Time**: <1ms for standard configurations
- **Forward Pass Time**: <10μs for typical test inputs
- **Memory Overhead**: <1KB per mock instance

## Acceptance Criteria

- [ ] Centralized mock framework provides all necessary testing primitives
- [ ] All existing tests migrate to use centralized mocks without behavioral changes
- [ ] Mock objects support configurable behavior for diverse testing scenarios
- [ ] Call logging and verification support comprehensive test assertions
- [ ] Property-based testing integration works seamlessly
- [ ] Performance overhead is minimal (<10% of original mock cost)
- [ ] Documentation clearly explains mock usage patterns
- [ ] No duplicate mock implementations remain in codebase
- [ ] Test coverage maintains or improves with new framework
- [ ] Mock framework is extensible for future testing needs

## Dependencies

- Standard Rust testing framework
- `proptest` for property-based testing (optional)
- `serde` for configuration serialization (optional)
- Existing BitNet-rs inference types and traits

## Related Issues

- Test infrastructure improvements
- Code quality and maintainability
- Development workflow optimization
- Testing best practices standardization

## Labels
- `testing`
- `refactoring`
- `code-quality`
- `technical-debt`
- `developer-experience`
- `priority-low`
- `good-first-issue`
