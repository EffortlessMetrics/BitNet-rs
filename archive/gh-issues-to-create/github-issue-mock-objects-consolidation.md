# [Test Infrastructure] Consolidate duplicate mock objects into shared test utilities

## Problem Description

Multiple test modules across the `bitnet-inference` crate define extensive, duplicated mock objects (`MockModel`, `MockTokenizer`, `MockBackend`). This creates maintenance overhead and code duplication across test files.

## Environment
- **Affected Files**:
  - `crates/bitnet-inference/src/backends.rs` (MockModel)
  - `crates/bitnet-inference/src/engine.rs` (MockModel, MockTokenizer)
  - `crates/bitnet-inference/src/production_engine.rs` (MockModel, MockTokenizer)
  - `crates/bitnet-inference/src/streaming.rs` (MockModel, MockTokenizer, MockBackend)
- **Crate**: `bitnet-inference`
- **Context**: Test modules with `#[cfg(test)]` blocks

## Root Cause Analysis

1. **Code Duplication**: Similar mock implementations are repeated across multiple test modules
2. **Maintenance Overhead**: Changes to mock behavior require updates in multiple files
3. **Inconsistency Risk**: Mock implementations may drift apart over time
4. **Test Complexity**: Each test module reinvents common testing infrastructure

## Impact Assessment
- **Severity**: Medium
- **Impact**: Developer productivity, test maintenance
- **Affected Components**: All `bitnet-inference` test suites
- **Technical Debt**: Increasing complexity as more tests are added

## Current Mock Objects Analysis

### Duplicated Mocks Found:
1. **MockModel** - Found in 4 files with similar implementations
2. **MockTokenizer** - Found in 3 files with similar implementations
3. **MockBackend** - Found in 1 file but likely needed elsewhere

### Current Mock Complexity:
- Extensive trait implementations
- Hardcoded test data
- Repetitive boilerplate code
- Inconsistent mock behavior across files

## Proposed Solution

Create a centralized test utilities module to house all shared mock objects and testing infrastructure.

### Implementation Plan

1. **Create Test Utilities Module**:
```
crates/bitnet-inference/tests/
├── mod.rs
└── utils/
    ├── mod.rs
    ├── mocks.rs
    └── fixtures.rs
```

2. **Centralized Mock Implementations** (`tests/utils/mocks.rs`):
```rust
use bitnet_common::{BitNetConfig, Model, Tokenizer, Backend};

pub struct MockModel {
    config: BitNetConfig,
    forward_calls: std::sync::atomic::AtomicUsize,
}

impl MockModel {
    pub fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            forward_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn with_config(config: BitNetConfig) -> Self {
        Self {
            config,
            forward_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn forward_call_count(&self) -> usize {
        self.forward_calls.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Model for MockModel {
    // Simplified, focused implementation
    fn config(&self) -> &BitNetConfig { &self.config }

    fn forward(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        self.forward_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Minimal mock behavior
        Ok(input.clone())
    }
}

pub struct MockTokenizer {
    vocab_size: usize,
    eos_token_id: u32,
}

impl MockTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocab size
            eos_token_id: 50256,
        }
    }

    pub fn with_vocab_size(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            eos_token_id: (vocab_size - 1) as u32,
        }
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Simple mock: return text length as tokens
        Ok((0..text.len().min(10)).map(|i| i as u32).collect())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(format!("decoded_{}_tokens", tokens.len()))
    }

    fn vocab_size(&self) -> usize { self.vocab_size }
    fn eos_token_id(&self) -> u32 { self.eos_token_id }
}

pub struct MockBackend {
    device_type: DeviceType,
}

impl MockBackend {
    pub fn cpu() -> Self {
        Self { device_type: DeviceType::Cpu }
    }

    pub fn gpu() -> Self {
        Self { device_type: DeviceType::Gpu }
    }
}

impl Backend for MockBackend {
    fn device_type(&self) -> DeviceType { self.device_type }

    fn forward(&self, model: &dyn Model, input: &BitNetTensor) -> Result<BitNetTensor> {
        model.forward(input)
    }
}
```

3. **Test Fixtures** (`tests/utils/fixtures.rs`):
```rust
pub fn sample_config() -> BitNetConfig {
    BitNetConfig {
        vocab_size: 50257,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12,
        // ... reasonable test defaults
    }
}

pub fn sample_generation_config() -> GenerationConfig {
    GenerationConfig {
        max_length: 50,
        temperature: 1.0,
        top_k: None,
        top_p: None,
        // ... test defaults
    }
}
```

4. **Update Existing Tests**:
   - Remove duplicated mock definitions
   - Import from centralized utilities
   - Simplify test setup code

### Example Test Migration:
```rust
// Before (in each test file):
#[cfg(test)]
mod tests {
    struct MockModel { /* extensive implementation */ }

    #[test]
    fn test_something() {
        let mock_model = MockModel::new();
        // test code
    }
}

// After:
#[cfg(test)]
mod tests {
    use crate::tests::utils::mocks::{MockModel, MockTokenizer};

    #[test]
    fn test_something() {
        let mock_model = MockModel::new();
        // test code
    }
}
```

## Testing Strategy
- **Refactoring Tests**: Ensure all existing tests continue to pass
- **Mock Behavior Tests**: Verify centralized mocks work correctly
- **Integration Tests**: Test mock objects work across different test files
- **Performance Tests**: Ensure no performance regression in test execution

## Implementation Tasks
- [ ] Create `tests/utils/` directory structure
- [ ] Implement centralized `MockModel` with essential functionality
- [ ] Implement centralized `MockTokenizer` with essential functionality
- [ ] Implement centralized `MockBackend` with essential functionality
- [ ] Create common test fixtures and utilities
- [ ] Migrate `backends.rs` tests to use centralized mocks
- [ ] Migrate `engine.rs` tests to use centralized mocks
- [ ] Migrate `production_engine.rs` tests to use centralized mocks
- [ ] Migrate `streaming.rs` tests to use centralized mocks
- [ ] Remove old mock implementations
- [ ] Update test documentation and examples

## Benefits After Implementation
- **Reduced Code Duplication**: Single source of truth for mock objects
- **Easier Maintenance**: Changes to mock behavior in one place
- **Consistent Test Behavior**: All tests use identical mock implementations
- **Improved Test Development**: Faster test writing with pre-built utilities
- **Better Mock Functionality**: Centralized mocks can have richer features

## Acceptance Criteria
- [ ] All existing tests continue to pass
- [ ] No duplicated mock object definitions in test modules
- [ ] Centralized mock objects support all required test scenarios
- [ ] Test utilities are well-documented with examples
- [ ] Mock objects are feature-complete but minimal
- [ ] Test execution time doesn't increase significantly
- [ ] Mock objects support test-specific customization

## Backward Compatibility
- Existing test code will need updates to import from new location
- Mock behavior should remain functionally identical
- Consider deprecation warnings for transition period

## Labels
- `test-infrastructure`
- `tech-debt`
- `refactoring`
- `priority-medium`
- `good-first-issue`

## Related Issues
- Test infrastructure improvements
- Developer experience enhancements
- Code quality initiatives
