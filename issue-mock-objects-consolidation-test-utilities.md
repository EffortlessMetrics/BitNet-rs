# [Test Infrastructure] Consolidate Mock Objects Into Centralized Test Utilities Module

## Problem Description

The BitNet.rs inference crate has extensive code duplication in test mock objects, with `MockModel` and `MockTokenizer` structs independently defined across **17+ test files**. This creates a significant maintenance burden, inconsistent testing behavior, and violates the DRY (Don't Repeat Yourself) principle. The current scattered approach makes it difficult to maintain consistent mock behavior and update test utilities across the codebase.

## Environment

- **Crate**: `crates/bitnet-inference/`
- **MSRV**: Rust 1.90.0
- **Test Framework**: Standard Rust + `tokio::test`
- **Affected Files**: 17+ test files and source modules with embedded test blocks

## Current State Analysis

### Affected Files with Mock Duplication

Analysis reveals `MockModel` and `MockTokenizer` definitions in:

**Source modules with embedded tests:**
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/engine.rs:1759`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/production_engine.rs:548`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/streaming.rs:534`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/backends.rs:295`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/fallback.rs:9`

**Dedicated test files:**
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/performance_tests.rs:23`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/ac10_error_handling_robustness.rs:152`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/unit_tests.rs:19`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/performance_tracking_tests.rs:22`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/integration_tests.rs:23`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/ac8_mock_implementation_replacement.rs:58`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/batch_prefill.rs:10`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/batch_tests.rs:117`
- And more...

### Code Duplication Evidence

Each file independently implements similar but slightly different mock structures:

```rust
// Common pattern repeated across 17+ files
struct MockModel {
    config: BitNetConfig,
}

struct MockTokenizer;

impl Model for MockModel { /* similar implementations */ }
impl Tokenizer for MockTokenizer { /* similar implementations */ }
```

## Root Cause Analysis

1. **Organic Growth**: Tests were created independently without establishing shared utilities
2. **Lack of Test Infrastructure**: No centralized location for reusable test components
3. **Copy-Paste Development**: Developers duplicated existing mock implementations
4. **Missing Guidelines**: No established patterns for test utilities in the project

## Impact Assessment

### Severity: High
### Affected Components: All inference tests

**Maintenance Impact:**
- Changes to trait interfaces require updates in 17+ locations
- Inconsistent mock behavior leads to unreliable tests
- New developers must choose between creating new mocks or finding existing ones

**Development Velocity Impact:**
- Slows down refactoring efforts
- Increases cognitive load when writing tests
- Higher risk of test implementation bugs due to inconsistency

**Code Quality Impact:**
- ~500+ lines of duplicated mock code
- Violation of DRY principles
- Reduced maintainability score

## Proposed Solution

### Primary Approach: Centralized Test Utilities Module

Create a comprehensive test utilities infrastructure with reusable, configurable mock objects.

#### Implementation Plan

**1. Create Test Utilities Module Structure**

```rust
// crates/bitnet-inference/tests/common/mod.rs
pub mod mocks;
pub mod fixtures;
pub mod assertions;

// crates/bitnet-inference/tests/common/mocks.rs
use bitnet_common::{BitNetConfig, ConcreteTensor};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ConfigurableMockModel {
    config: BitNetConfig,
    forward_behavior: ForwardBehavior,
    latency_simulation: Option<std::time::Duration>,
}

#[derive(Debug, Clone)]
pub enum ForwardBehavior {
    Success(ConcreteTensor),
    Error(String),
    Custom(Arc<dyn Fn(&ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> + Send + Sync>),
}

impl ConfigurableMockModel {
    pub fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            forward_behavior: ForwardBehavior::Success(ConcreteTensor::mock(vec![1, 50257])),
            latency_simulation: None,
        }
    }

    pub fn with_config(mut self, config: BitNetConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_forward_error(mut self, error: impl Into<String>) -> Self {
        self.forward_behavior = ForwardBehavior::Error(error.into());
        self
    }

    pub fn with_latency(mut self, duration: std::time::Duration) -> Self {
        self.latency_simulation = Some(duration);
        self
    }
}

#[derive(Debug, Clone)]
pub struct ConfigurableMockTokenizer {
    vocab_size: usize,
    encode_behavior: EncodeBehavior,
    eos_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum EncodeBehavior {
    Success(Vec<u32>),
    Error(String),
    Custom(Arc<dyn Fn(&str) -> bitnet_common::Result<Vec<u32>> + Send + Sync>),
}

impl ConfigurableMockTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257,
            encode_behavior: EncodeBehavior::Success(vec![1, 2, 3]),
            eos_token_id: Some(50256),
        }
    }

    pub fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    pub fn with_encode_error(mut self, error: impl Into<String>) -> Self {
        self.encode_behavior = EncodeBehavior::Error(error.into());
        self
    }
}
```

**2. Implement Trait Implementations with Error Injection**

```rust
impl Model for ConfigurableMockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<ConcreteTensor> {
        // Simulate latency if configured
        if let Some(duration) = self.latency_simulation {
            std::thread::sleep(duration);
        }

        match &self.forward_behavior {
            ForwardBehavior::Success(tensor) => Ok(tensor.clone()),
            ForwardBehavior::Error(msg) => Err(bitnet_common::Error::InferenceError(msg.clone())),
            ForwardBehavior::Custom(func) => func(input),
        }
    }

    fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 10, 768]))
    }

    fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 10, 50257]))
    }
}

impl Tokenizer for ConfigurableMockTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> bitnet_common::Result<Vec<u32>> {
        match &self.encode_behavior {
            EncodeBehavior::Success(tokens) => Ok(tokens.clone()),
            EncodeBehavior::Error(msg) => Err(bitnet_common::Error::TokenizationError(msg.clone())),
            EncodeBehavior::Custom(func) => func(text),
        }
    }

    fn decode(&self, _tokens: &[u32]) -> bitnet_common::Result<String> {
        Ok("mock generated text".to_string())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, _token: u32) -> Option<String> {
        Some("piece".to_string())
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        None
    }
}
```

**3. Create Helper Functions and Fixtures**

```rust
// crates/bitnet-inference/tests/common/fixtures.rs
pub fn default_mock_model() -> Arc<ConfigurableMockModel> {
    Arc::new(ConfigurableMockModel::new())
}

pub fn default_mock_tokenizer() -> Arc<ConfigurableMockTokenizer> {
    Arc::new(ConfigurableMockTokenizer::new())
}

pub fn error_prone_model() -> Arc<ConfigurableMockModel> {
    Arc::new(ConfigurableMockModel::new()
        .with_forward_error("Simulated model error"))
}

pub fn high_latency_model() -> Arc<ConfigurableMockModel> {
    Arc::new(ConfigurableMockModel::new()
        .with_latency(std::time::Duration::from_millis(100)))
}

pub fn small_vocab_tokenizer() -> Arc<ConfigurableMockTokenizer> {
    Arc::new(ConfigurableMockTokenizer::new()
        .with_vocab_size(1000))
}
```

**4. Migration Strategy**

Create backward-compatible type aliases:

```rust
// crates/bitnet-inference/tests/common/mocks.rs
pub type MockModel = ConfigurableMockModel;
pub type MockTokenizer = ConfigurableMockTokenizer;

// Helper functions for easy migration
pub fn mock_model() -> Arc<MockModel> {
    default_mock_model()
}

pub fn mock_tokenizer() -> Arc<MockTokenizer> {
    default_mock_tokenizer()
}
```

### Alternative Solutions Considered

1. **Per-module test utilities**: Would reduce duplication but limit reusability
2. **Trait-based mock framework**: Higher complexity, may be overkill for current needs
3. **External mock library**: Adds dependency, may not fit BitNet-specific requirements

## Implementation Breakdown

### Phase 1: Infrastructure Setup
- [ ] Create `tests/common/` module structure
- [ ] Implement `ConfigurableMockModel` and `ConfigurableMockTokenizer`
- [ ] Add comprehensive documentation and examples
- [ ] Create migration guide

### Phase 2: Systematic Migration
- [ ] Update `engine.rs` tests to use centralized mocks
- [ ] Update `production_engine.rs` tests
- [ ] Update `streaming.rs` tests
- [ ] Update `backends.rs` tests
- [ ] Migrate integration test files (8 files)
- [ ] Migrate specialized test files (batch, performance, etc.)

### Phase 3: Enhanced Features
- [ ] Add performance simulation capabilities
- [ ] Implement error injection patterns
- [ ] Create test data generators
- [ ] Add assertion helpers

### Phase 4: Cleanup and Validation
- [ ] Remove duplicated mock implementations
- [ ] Run comprehensive test suite
- [ ] Update documentation
- [ ] Performance regression testing

## Testing Strategy

### Verification Criteria
1. **All existing tests pass** after migration
2. **No performance degradation** in test execution
3. **Consistent mock behavior** across all test files
4. **Improved maintainability** - single location for updates

### Test Plan
```bash
# Run full test suite to ensure no regressions
cargo test --workspace --no-default-features --features cpu

# Verify new test utilities work correctly
cargo test -p bitnet-inference --test common

# Performance benchmarking
cargo test -p bitnet-inference --release -- --ignored perf
```

### Rollback Plan
- Keep original mock implementations commented out during migration
- Maintain git branches for each migration phase
- Automated testing before each merge

## Acceptance Criteria

- [ ] All 17+ duplicate mock implementations consolidated into single utilities module
- [ ] Zero test regressions after migration
- [ ] Configurable mock behavior supports error injection and performance testing
- [ ] Documentation includes migration guide and best practices
- [ ] CI/CD pipeline validates test utilities module
- [ ] Development velocity improved (measured by time to add new tests)

## Related Issues

- **Code Quality**: Addresses technical debt in test infrastructure
- **Developer Experience**: Improves ease of writing new tests
- **Maintainability**: Reduces maintenance overhead for trait changes

## Implementation Notes

### BitNet-Specific Considerations
- Mock objects must support both CPU and GPU backends
- Consider quantization-aware mock behavior for `I2_S`, `TL1`, `TL2` methods
- Ensure compatibility with streaming inference patterns
- Support for GGUF-specific model configurations

### Performance Considerations
- Configurable latency simulation for performance testing
- Memory-efficient mock implementations
- Zero-cost abstractions where possible

This consolidation represents a significant improvement to the test infrastructure, reducing maintenance burden and improving developer productivity across the BitNet.rs inference codebase.