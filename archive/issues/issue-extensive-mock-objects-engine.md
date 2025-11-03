# [REFACTOR] Extensive Mock Objects in `engine.rs` - Modularize Test Infrastructure

## Problem Description

The test module in `crates/bitnet-inference/src/engine.rs` contains extensive mock implementations (`MockModel` and `MockTokenizer`) that violate the single responsibility principle and create maintenance overhead. These mock objects are approximately 64 lines of code and provide full trait implementations that could be simplified or shared across the codebase.

**Affected File:** `crates/bitnet-inference/src/engine.rs` (lines 1759-1822)

**Current State Analysis:**
- **MockModel struct**: 29 lines implementing the `Model` trait with 4 methods
- **MockTokenizer struct**: 31 lines implementing the `Tokenizer` trait with 6 methods
- **Test dependencies**: 9 test functions rely on these mocks (lines 1824-1987)
- **Code duplication risk**: No evidence of reuse across other test files, potential for duplication

## Root Cause Investigation

### Technical Analysis

1. **Over-Engineering of Test Mocks**: The mock objects implement complete trait interfaces when tests only require minimal functionality
2. **Lack of Test Infrastructure**: No centralized test utilities module exists in the `bitnet-inference` crate
3. **Single-Responsibility Violation**: Test module combines engine logic with extensive mock implementations
4. **Maintenance Burden**: Changes to `Model` or `Tokenizer` traits require updates to these embedded mocks

### Impact Assessment

**Severity**: Medium (Code Quality/Maintainability)
**Affected Components**:
- `bitnet-inference::engine` test suite
- Future tests requiring model/tokenizer mocks
- Developer productivity for test maintenance

**Business Impact**:
- **Development Velocity**: Slowed by mock maintenance overhead
- **Code Quality**: Reduced readability and modularity in core engine file
- **Test Reliability**: Risk of test-specific mock behavior diverging from production interfaces

## Environment Details

- **Rust Version**: 1.90.0 (MSRV)
- **File Location**: `crates/bitnet-inference/src/engine.rs`
- **Lines of Code**: Mock objects span 64 lines (1759-1822)
- **Test Functions**: 9 test functions depend on these mocks
- **Dependencies**: Requires `bitnet-models::Model` and `bitnet-tokenizers::Tokenizer` traits

## Reproduction Analysis

The extensive mock objects can be observed by examining the test module:

```bash
# View the mock implementations
cargo expand --bin bitnet-inference --tests | grep -A 30 "struct MockModel"

# Count lines of mock code
wc -l crates/bitnet-inference/src/engine.rs | grep -E "MockModel|MockTokenizer" -A 20
```

**Expected Behavior**: Test mocks should be minimal, focused, and potentially shared
**Current Behavior**: Mocks are extensive, embedded in engine module, and not reusable

## Proposed Solution

### Primary Approach: Modular Test Infrastructure

Create a dedicated test utilities module with simplified, reusable mock objects.

#### Implementation Plan

**Phase 1: Create Test Utilities Module**

```rust
// File: crates/bitnet-inference/tests/utils/mod.rs
pub mod mocks;

// File: crates/bitnet-inference/tests/utils/mocks.rs
use bitnet_common::{BitNetConfig, ConcreteTensor, Result};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;

/// Minimal mock model for testing engine functionality
#[derive(Debug, Clone)]
pub struct TestModel {
    config: BitNetConfig,
    custom_vocab_size: Option<usize>,
}

impl TestModel {
    pub fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            custom_vocab_size: None,
        }
    }

    pub fn with_vocab_size(mut self, size: usize) -> Self {
        self.custom_vocab_size = Some(size);
        self
    }
}

impl Model for TestModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor> {
        let vocab_size = self.custom_vocab_size.unwrap_or(50257);
        Ok(ConcreteTensor::mock(vec![1, vocab_size]))
    }

    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, tokens.len(), 768]))
    }

    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        let vocab_size = self.custom_vocab_size.unwrap_or(50257);
        let shape = hidden.shape();
        Ok(ConcreteTensor::mock(vec![shape[0], shape[1], vocab_size]))
    }
}

/// Minimal mock tokenizer with configurable behavior
#[derive(Debug, Clone)]
pub struct TestTokenizer {
    vocab_size: usize,
    eos_token: Option<u32>,
    fixed_encoding: Option<Vec<u32>>,
}

impl TestTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257,
            eos_token: Some(50256),
            fixed_encoding: None,
        }
    }

    pub fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        if size > 0 {
            self.eos_token = Some(size as u32 - 1);
        }
        self
    }

    pub fn with_fixed_encoding(mut self, tokens: Vec<u32>) -> Self {
        self.fixed_encoding = Some(tokens);
        self
    }
}

impl Tokenizer for TestTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> Result<Vec<u32>> {
        if let Some(ref fixed) = self.fixed_encoding {
            return Ok(fixed.clone());
        }

        // Simple character-based encoding for testing
        Ok(text.chars().take(10).map(|c| (c as u32) % 1000).collect())
    }

    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        Ok("mock generated text".to_string())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        if token < self.vocab_size as u32 {
            Some(format!("piece_{}", token))
        } else {
            None
        }
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token
    }

    fn pad_token_id(&self) -> Option<u32> {
        None
    }
}

/// Builder pattern for creating test engines with specific configurations
pub struct TestEngineBuilder {
    model: Option<Arc<dyn Model>>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    device: Device,
}

impl TestEngineBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            tokenizer: None,
            device: Device::Cpu,
        }
    }

    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn build(self) -> anyhow::Result<InferenceEngine> {
        let model = self.model.unwrap_or_else(|| Arc::new(TestModel::new()));
        let tokenizer = self.tokenizer.unwrap_or_else(|| Arc::new(TestTokenizer::new()));

        InferenceEngine::new(model, tokenizer, self.device)
    }
}
```

**Phase 2: Migrate Engine Tests**

```rust
// File: crates/bitnet-inference/src/engine.rs (updated test module)
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::utils::mocks::{TestEngineBuilder, TestModel, TestTokenizer};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_inference_engine_creation() {
        let engine = TestEngineBuilder::new().build();
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_prefill_functionality() {
        let mut engine = TestEngineBuilder::new().build().unwrap();
        let tokens = vec![1, 2, 3, 4, 5];

        let result = engine.prefill(&tokens).await;
        assert!(result.is_ok(), "Prefill should execute successfully");
    }

    #[tokio::test]
    async fn test_prefill_invalid_tokens() {
        let tokenizer = Arc::new(TestTokenizer::new().with_vocab_size(1000));
        let mut engine = TestEngineBuilder::new()
            .with_tokenizer(tokenizer.clone())
            .build().unwrap();

        let vocab_size = tokenizer.vocab_size() as u32;
        let invalid_tokens = vec![1, 2, vocab_size + 10];

        let result = engine.prefill(&invalid_tokens).await;
        assert!(result.is_err(), "Prefill should fail with invalid tokens");
    }

    // Additional tests using simplified builder pattern...
}
```

**Phase 3: Create Integration Points**

```rust
// File: crates/bitnet-inference/tests/mod.rs
pub mod utils;

// Re-export commonly used test utilities
pub use utils::mocks::{TestEngineBuilder, TestModel, TestTokenizer};
```

### Alternative Approaches

#### Option 2: Trait-Based Mock Framework

```rust
// Create generic mock traits that can be implemented minimally per test
pub trait MockModel: Model {
    fn simple_mock() -> Self where Self: Sized;
}

pub trait MockTokenizer: Tokenizer {
    fn simple_mock() -> Self where Self: Sized;
}
```

#### Option 3: Property-Based Test Generators

```rust
// Use proptest to generate mock data automatically
use proptest::prelude::*;

prop_compose! {
    fn mock_model_config()(
        vocab_size in 1000..100000usize,
        hidden_size in 256..2048usize,
    ) -> BitNetConfig {
        BitNetConfig {
            model: ModelConfig {
                vocab_size,
                hidden_size,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}
```

## Testing Strategy

### Test Coverage Requirements

1. **Backward Compatibility**: All existing tests must pass with new mock infrastructure
2. **Mock Behavior Validation**: Ensure mocks properly implement trait contracts
3. **Performance Impact**: New test utilities should not slow down test execution
4. **Reusability Testing**: Verify mocks can be shared across multiple test files

### Validation Framework

```rust
#[cfg(test)]
mod mock_validation {
    use super::*;

    #[test]
    fn test_mock_model_trait_contract() {
        let model = TestModel::new();

        // Verify trait methods work correctly
        assert!(!model.config().model.vocab_size == 0);

        let result = model.embed(&[1, 2, 3]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_tokenizer_vocab_consistency() {
        let tokenizer = TestTokenizer::new().with_vocab_size(1000);

        assert_eq!(tokenizer.vocab_size(), 1000);
        assert_eq!(tokenizer.eos_token_id(), Some(999));
    }
}
```

### Performance Benchmarks

```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn benchmark_mock_creation_overhead() {
        let start = Instant::now();

        for _ in 0..1000 {
            let _engine = TestEngineBuilder::new().build().unwrap();
        }

        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 100, "Mock creation should be fast");
    }
}
```

## Implementation Breakdown

### Phase 1: Infrastructure Setup (1-2 days)
- [ ] Create `crates/bitnet-inference/tests/utils/` directory structure
- [ ] Implement `TestModel` with configurable behavior
- [ ] Implement `TestTokenizer` with configurable behavior
- [ ] Create `TestEngineBuilder` for convenient test setup
- [ ] Add basic validation tests for mock implementations

### Phase 2: Migration and Refactoring (2-3 days)
- [ ] Migrate existing engine tests to use new mock infrastructure
- [ ] Simplify engine.rs test module by removing embedded mocks
- [ ] Update test imports and dependencies
- [ ] Ensure all existing test cases pass with new mocks
- [ ] Add enhanced test cases leveraging configurable mock behavior

### Phase 3: Integration and Optimization (1 day)
- [ ] Create public test utilities module interface
- [ ] Add documentation for test mock usage
- [ ] Performance optimization of mock object creation
- [ ] Integration testing across multiple test files

### Phase 4: Quality Assurance (1 day)
- [ ] Comprehensive test suite validation
- [ ] Code review and refactoring feedback integration
- [ ] CI/CD pipeline validation
- [ ] Documentation updates

## Risk Assessment and Mitigation

### Potential Complications

1. **Test Breakage**: Migrating tests may introduce regressions
   - **Mitigation**: Incremental migration with continuous validation
   - **Rollback Plan**: Keep original mocks until full migration complete

2. **Performance Degradation**: New mock infrastructure may be slower
   - **Mitigation**: Benchmark mock creation and caching strategies
   - **Monitoring**: Add performance tests to CI pipeline

3. **Trait Interface Changes**: Future trait modifications may break mocks
   - **Mitigation**: Implement comprehensive trait contract tests
   - **Automation**: Add CI checks for trait compliance

### Dependencies and Blocking Issues

- **Trait Stability**: Requires stable `Model` and `Tokenizer` trait interfaces
- **Test Framework**: Depends on `tokio-test` and `anyhow` for async testing
- **CI Integration**: May require updates to test discovery patterns

## Acceptance Criteria

### Definition of Done

- [ ] **Mock Simplification**: Engine test module contains < 20 lines of mock code
- [ ] **Reusable Infrastructure**: Test utilities can be imported by other crates
- [ ] **Backward Compatibility**: All existing tests pass without modification to test logic
- [ ] **Enhanced Testability**: New tests can be written with 50% less boilerplate
- [ ] **Documentation**: Clear usage examples for test mock infrastructure
- [ ] **Performance**: Mock creation overhead < 1ms per test case

### Quality Gates

1. **Code Coverage**: Maintain 100% coverage on migrated test functions
2. **Test Performance**: Test suite execution time does not increase > 10%
3. **Maintainability**: Cyclomatic complexity of test module reduced by 30%
4. **Reusability**: At least 2 other test files successfully use shared mocks

## Related Issues and Cross-References

### BitNet.rs Architecture Integration

- **Related Component**: `bitnet-models` crate trait definitions
- **Related Component**: `bitnet-tokenizers` crate trait definitions
- **Related Component**: `bitnet-common` testing infrastructure
- **Integration Point**: Test utilities may be promoted to `bitnet-common` if widely adopted

### Future Enhancements

- **Issue Dependency**: May benefit from enhanced error testing framework (future issue)
- **Issue Dependency**: Could integrate with property-based testing infrastructure
- **Issue Enhancement**: Potential for mock recording/playback for integration tests

### Documentation References

- **Architecture**: `docs/architecture-overview.md` - Testing strategy section
- **Development**: `docs/development/test-suite.md` - Mock object guidelines
- **Reference**: `docs/development/build-commands.md` - Test execution patterns

## Labels and Classification

**Labels**: `refactoring`, `testing`, `infrastructure`, `code-quality`, `P2-medium`, `good-first-issue`

**Priority**: Medium (P2) - Improves maintainability without blocking functionality
**Effort Estimate**: 5-7 days (Medium)
**Complexity**: Medium - Requires understanding of trait systems and test infrastructure

**Area**: Testing Infrastructure
**Component**: `bitnet-inference`
**Type**: Technical Debt / Code Quality

## Implementation Notes

### BitNet.rs Specific Considerations

1. **Feature Flags**: Test utilities should respect `--no-default-features` build pattern
2. **MSRV Compliance**: All code must build on Rust 1.90.0
3. **Cross-Validation**: Mock behavior should not interfere with C++ reference validation
4. **Device Compatibility**: Mocks should work with both CPU and GPU device types

### Development Workflow Integration

```bash
# Standard development cycle with new infrastructure
cargo test --workspace --no-default-features --features cpu -p bitnet-inference
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Specific mock validation
cargo test -p bitnet-inference mock_validation
cargo test -p bitnet-inference --test utils
```

This issue provides a comprehensive pathway to transform the extensive mock objects in `engine.rs` into a maintainable, reusable test infrastructure that aligns with BitNet.rs architecture principles and supports the project's testing strategy evolution.
