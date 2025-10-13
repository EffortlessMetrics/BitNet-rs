# [REFACTOR] Consolidate and Simplify Extensive Mock Objects in Production Engine Test Module

## Problem Description

The `tests` module in `crates/bitnet-inference/src/production_engine.rs` contains extensive `MockModel` and `MockTokenizer` implementations that are duplicated across multiple files throughout the codebase. This leads to code duplication, maintenance overhead, and inconsistent test behavior. Analysis shows that similar mock objects exist in **at least 13 files** for `MockModel` and **11 files** for `MockTokenizer`, indicating a systemic issue with test infrastructure.

## Environment

- **Files Affected**:
  - Primary: `crates/bitnet-inference/src/production_engine.rs` (lines 548-611)
  - Secondary: 13+ files with `MockModel` duplications
  - Secondary: 11+ files with `MockTokenizer` duplications
- **Rust Version**: MSRV 1.90.0 (Rust 2024 edition)
- **Feature Flags**: Affects both `cpu` and `inference` feature builds
- **Build Configuration**: All test configurations (`cargo test --workspace`)

## Root Cause Analysis

### Technical Investigation

1. **Mock Object Complexity**: The current `MockModel` implementation spans ~78 lines (548-626) with extensive trait implementations that often provide more functionality than required for specific tests.

2. **Code Duplication Pattern**:
   ```rust
   // Pattern repeated across 13+ files:
   struct MockModel {
       config: BitNetConfig,
       // Additional fields vary by file
   }
   ```

3. **Inconsistent Implementation**: Each file implements mock objects differently, leading to:
   - Inconsistent behavior across test suites
   - Different error handling patterns
   - Varying levels of feature completeness
   - Maintenance burden when trait interfaces change

4. **Violation of DRY Principle**: Core mock functionality is reimplemented rather than shared, making global changes to mock behavior require updates across multiple files.

### Impact Assessment

- **Maintenance Overhead**: High - Changes to `Model` or `Tokenizer` traits require updates in 20+ locations
- **Test Reliability**: Medium - Inconsistent mock implementations can lead to different test behaviors
- **Developer Experience**: Medium - New contributors must understand multiple mock variants
- **Code Quality**: Medium - Duplicated code reduces overall codebase quality metrics

## Reproduction Steps

1. **Code Analysis**:
   ```bash
   # Count MockModel implementations
   rg "struct MockModel" crates/bitnet-inference/ --files-with-matches | wc -l
   # Result: 13 files

   # Count MockTokenizer implementations
   rg "struct MockTokenizer" crates/bitnet-inference/ --files-with-matches | wc -l
   # Result: 11 files
   ```

2. **Verify Duplication**:
   ```bash
   # Compare implementations
   rg "impl Model for MockModel" crates/bitnet-inference/ -A 20
   # Shows varying implementations across files
   ```

3. **Build Impact**:
   ```bash
   cargo test --workspace --no-default-features --features cpu
   # All tests pass but with duplicated mock code
   ```

**Expected Result**: Centralized, reusable mock implementations
**Actual Result**: Extensive duplication across 20+ files

## Proposed Solution

### Primary Approach: Centralized Test Utilities Module

Create a dedicated test utilities crate within the inference module to provide standardized, feature-rich mock implementations.

#### Implementation Plan

**Phase 1: Create Test Utilities Infrastructure**

```rust
// File: crates/bitnet-inference/src/test_utils.rs
#[cfg(test)]
pub mod test_utils {
    use bitnet_common::{BitNetConfig, ConcreteTensor, Device};
    use bitnet_models::Model;
    use bitnet_tokenizers::Tokenizer;
    use std::sync::Arc;

    /// Configurable mock model with comprehensive test capabilities
    pub struct ConfigurableMockModel {
        pub config: BitNetConfig,
        pub should_fail_forward: bool,
        pub should_fail_embed: bool,
        pub should_fail_logits: bool,
        pub forward_delay_ms: Option<u64>,
        pub custom_vocab_size: Option<usize>,
    }

    impl ConfigurableMockModel {
        pub fn new() -> Self {
            Self {
                config: BitNetConfig::default(),
                should_fail_forward: false,
                should_fail_embed: false,
                should_fail_logits: false,
                forward_delay_ms: None,
                custom_vocab_size: None,
            }
        }

        pub fn with_failure_mode(mut self, mode: MockFailureMode) -> Self {
            match mode {
                MockFailureMode::Forward => self.should_fail_forward = true,
                MockFailureMode::Embed => self.should_fail_embed = true,
                MockFailureMode::Logits => self.should_fail_logits = true,
                MockFailureMode::All => {
                    self.should_fail_forward = true;
                    self.should_fail_embed = true;
                    self.should_fail_logits = true;
                }
            }
            self
        }

        pub fn with_latency(mut self, delay_ms: u64) -> Self {
            self.forward_delay_ms = Some(delay_ms);
            self
        }

        pub fn with_vocab_size(mut self, size: usize) -> Self {
            self.custom_vocab_size = Some(size);
            self
        }
    }

    #[derive(Debug, Clone)]
    pub enum MockFailureMode {
        Forward,
        Embed,
        Logits,
        All,
    }

    impl Model for ConfigurableMockModel {
        fn config(&self) -> &BitNetConfig {
            &self.config
        }

        fn forward(
            &self,
            _input: &ConcreteTensor,
            _cache: &mut dyn std::any::Any,
        ) -> bitnet_common::Result<ConcreteTensor> {
            if self.should_fail_forward {
                return Err(bitnet_common::BitNetError::Inference(
                    bitnet_common::InferenceError::GenerationFailed {
                        reason: "Mock forward failure".to_string(),
                    }
                ));
            }

            if let Some(delay) = self.forward_delay_ms {
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }

            let vocab_size = self.custom_vocab_size.unwrap_or(1000);
            Ok(ConcreteTensor::mock(vec![1, 10, vocab_size]))
        }

        fn embed(&self, tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
            if self.should_fail_embed {
                return Err(bitnet_common::BitNetError::Inference(
                    bitnet_common::InferenceError::GenerationFailed {
                        reason: "Mock embed failure".to_string(),
                    }
                ));
            }
            Ok(ConcreteTensor::mock(vec![1, tokens.len(), 768]))
        }

        fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
            if self.should_fail_logits {
                return Err(bitnet_common::BitNetError::Inference(
                    bitnet_common::InferenceError::GenerationFailed {
                        reason: "Mock logits failure".to_string(),
                    }
                ));
            }
            let vocab_size = self.custom_vocab_size.unwrap_or(1000);
            Ok(ConcreteTensor::mock(vec![1, 10, vocab_size]))
        }
    }

    /// Configurable mock tokenizer for comprehensive testing
    pub struct ConfigurableMockTokenizer {
        pub should_fail_encode: bool,
        pub should_fail_decode: bool,
        pub custom_vocab_size: usize,
        pub encode_delay_ms: Option<u64>,
        pub decode_delay_ms: Option<u64>,
        pub custom_tokens: Option<Vec<u32>>,
        pub custom_text: Option<String>,
    }

    impl ConfigurableMockTokenizer {
        pub fn new() -> Self {
            Self {
                should_fail_encode: false,
                should_fail_decode: false,
                custom_vocab_size: 1000,
                encode_delay_ms: None,
                decode_delay_ms: None,
                custom_tokens: None,
                custom_text: None,
            }
        }

        pub fn with_encode_failure(mut self) -> Self {
            self.should_fail_encode = true;
            self
        }

        pub fn with_decode_failure(mut self) -> Self {
            self.should_fail_decode = true;
            self
        }

        pub fn with_custom_response(mut self, tokens: Vec<u32>, text: String) -> Self {
            self.custom_tokens = Some(tokens);
            self.custom_text = Some(text);
            self
        }

        pub fn with_latency(mut self, encode_ms: u64, decode_ms: u64) -> Self {
            self.encode_delay_ms = Some(encode_ms);
            self.decode_delay_ms = Some(decode_ms);
            self
        }
    }

    impl Tokenizer for ConfigurableMockTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_bos: bool,
            _add_special: bool,
        ) -> bitnet_common::Result<Vec<u32>> {
            if self.should_fail_encode {
                return Err(bitnet_common::BitNetError::Inference(
                    bitnet_common::InferenceError::TokenizationFailed {
                        reason: "Mock encode failure".to_string(),
                    }
                ));
            }

            if let Some(delay) = self.encode_delay_ms {
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }

            Ok(self.custom_tokens.clone().unwrap_or_else(|| vec![1, 2, 3]))
        }

        fn decode(&self, _tokens: &[u32]) -> bitnet_common::Result<String> {
            if self.should_fail_decode {
                return Err(bitnet_common::BitNetError::Inference(
                    bitnet_common::InferenceError::TokenizationFailed {
                        reason: "Mock decode failure".to_string(),
                    }
                ));
            }

            if let Some(delay) = self.decode_delay_ms {
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }

            Ok(self.custom_text.clone().unwrap_or_else(|| "mock generated text".to_string()))
        }

        fn vocab_size(&self) -> usize {
            self.custom_vocab_size
        }

        fn token_to_piece(&self, _token: u32) -> Option<String> {
            Some("mock".to_string())
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(999)
        }

        fn pad_token_id(&self) -> Option<u32> {
            None
        }
    }

    /// Factory functions for common test scenarios
    pub fn create_simple_mock_model() -> ConfigurableMockModel {
        ConfigurableMockModel::new()
    }

    pub fn create_failing_mock_model(mode: MockFailureMode) -> ConfigurableMockModel {
        ConfigurableMockModel::new().with_failure_mode(mode)
    }

    pub fn create_slow_mock_model(delay_ms: u64) -> ConfigurableMockModel {
        ConfigurableMockModel::new().with_latency(delay_ms)
    }

    pub fn create_simple_mock_tokenizer() -> ConfigurableMockTokenizer {
        ConfigurableMockTokenizer::new()
    }

    pub fn create_failing_mock_tokenizer() -> ConfigurableMockTokenizer {
        ConfigurableMockTokenizer::new().with_encode_failure().with_decode_failure()
    }

    /// Test helper for creating production engine with mocks
    pub fn create_test_production_engine() -> Result<crate::ProductionInferenceEngine, bitnet_common::BitNetError> {
        let model = Arc::new(create_simple_mock_model());
        let tokenizer = Arc::new(create_simple_mock_tokenizer());
        let device = Device::Cpu;

        crate::ProductionInferenceEngine::new(model, tokenizer, device)
    }
}
```

**Phase 2: Update Production Engine Tests**

```rust
// File: crates/bitnet-inference/src/production_engine.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use std::sync::Arc;

    #[test]
    fn test_performance_metrics_collector() {
        let mut collector = PerformanceMetricsCollector::new();
        collector.set_device_type(&Device::Cpu);

        let duration = Duration::from_millis(100);
        collector.record_prefill_metrics(10, duration);
        collector.record_decode_metrics(5, duration);
        collector.finalize_metrics(15, duration * 2);

        assert_eq!(collector.timing_metrics.prefill_ms, Some(100));
        assert_eq!(collector.timing_metrics.decode_ms, Some(100));
        assert_eq!(collector.throughput_metrics.total_tokens, 15);
    }

    #[test]
    fn test_device_manager() {
        let manager = DeviceManager::new(Device::Cpu);
        assert_eq!(manager.primary_device, Device::Cpu);
        assert_eq!(manager.fallback_device, Device::Cpu);

        let optimal = manager.get_optimal_device();
        assert_eq!(optimal, Device::Cpu);
    }

    #[test]
    fn test_generation_result() {
        let timing = TimingMetrics::default();
        let throughput = ThroughputMetrics::default();
        let performance = PerformanceMetrics::default();

        let mut result =
            GenerationResult::new("test".to_string(), 5, performance, timing, throughput);

        result.calculate_quality_score();
        assert!(result.quality_score.is_some());
    }

    #[tokio::test]
    #[cfg(feature = "inference")]
    async fn test_production_engine_creation() {
        let result = create_test_production_engine();
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[cfg(feature = "inference")]
    async fn test_production_engine_with_failing_model() {
        let model = Arc::new(create_failing_mock_model(MockFailureMode::Forward));
        let tokenizer = Arc::new(create_simple_mock_tokenizer());
        let device = Device::Cpu;

        if let Ok(mut engine) = ProductionInferenceEngine::new(model, tokenizer, device) {
            let config = GenerationConfig::default();
            let result = engine.generate_text("test", config).await;
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        if let Ok(engine) = create_test_production_engine() {
            let metrics = engine.collect_metrics().await;
            assert_eq!(metrics.backend_type, "CPU");
        }
    }

    #[tokio::test]
    #[cfg(feature = "inference")]
    async fn test_performance_tracking_with_latency() {
        let model = Arc::new(create_slow_mock_model(50)); // 50ms delay
        let tokenizer = Arc::new(create_simple_mock_tokenizer());
        let device = Device::Cpu;

        if let Ok(mut engine) = ProductionInferenceEngine::new(model, tokenizer, device) {
            let config = GenerationConfig { max_new_tokens: 5, ..Default::default() };
            let result = engine.generate_text("test prompt", config).await;

            if let Ok(gen_result) = result {
                assert!(gen_result.timing_metrics.total_ms >= 50);
                assert!(gen_result.performance_metrics.total_latency_ms >= 50);
            }
        }
    }
}
```

**Phase 3: Migration Strategy**

1. **Create test_utils module**: Add the centralized test utilities
2. **Update lib.rs**: Export test utilities for crate-wide access
   ```rust
   #[cfg(test)]
   pub mod test_utils;
   ```
3. **Gradual migration**: Update files one by one, replacing local mocks with centralized ones
4. **Validation**: Ensure all tests pass with new mock implementations

### Alternative Approaches

#### Option 2: Trait-Based Mock Framework
Create a trait-based system similar to mockall for dynamic mock behavior:

```rust
pub trait MockModelBehavior {
    fn should_fail_forward(&self) -> bool { false }
    fn should_fail_embed(&self) -> bool { false }
    fn forward_latency(&self) -> Duration { Duration::ZERO }
}
```

#### Option 3: Macro-Generated Mocks
Use procedural macros to generate consistent mock implementations:

```rust
#[derive_mock]
trait Model {
    // Automatically generates MockModel with configurable behavior
}
```

**Trade-offs Analysis**:
- **Centralized Module**: ✅ Simple, maintainable, immediate benefit
- **Trait-Based**: ✅ Flexible, ❌ More complex, higher learning curve
- **Macro-Generated**: ✅ Consistent, ❌ Complex implementation, compilation overhead

## Implementation Plan

### Phase 1: Infrastructure Setup (Week 1)
- [ ] Create `crates/bitnet-inference/src/test_utils.rs` with centralized mock implementations
- [ ] Update `crates/bitnet-inference/src/lib.rs` to export test utilities
- [ ] Create comprehensive documentation for mock usage patterns
- [ ] Add feature flags to ensure test utilities are only compiled during testing

### Phase 2: Production Engine Migration (Week 1)
- [ ] Replace mock implementations in `production_engine.rs` with centralized versions
- [ ] Update all test functions to use new mock factory functions
- [ ] Add enhanced test cases leveraging new mock capabilities (failure modes, latency testing)
- [ ] Verify all production engine tests pass with new implementation

### Phase 3: Codebase-Wide Migration (Week 2-3)
- [ ] **High Priority Files** (5 files):
  - `crates/bitnet-inference/src/engine.rs`
  - `crates/bitnet-inference/src/streaming.rs`
  - `crates/bitnet-inference/tests/integration_tests.rs`
  - `crates/bitnet-inference/tests/performance_tests.rs`
  - `crates/bitnet-inference/tests/batch_tests.rs`
- [ ] **Medium Priority Files** (8 files):
  - All `ac*_*.rs` test files
  - `unit_tests.rs`, `performance_tracking_tests.rs`
- [ ] **Low Priority Files** (7 files):
  - Remaining test files and specialized modules

### Phase 4: Quality Assurance (Week 3)
- [ ] Run comprehensive test suite: `cargo test --workspace --all-features`
- [ ] Performance regression testing to ensure mock changes don't impact test speed
- [ ] Documentation review and update
- [ ] Code review and approval process

### Phase 5: Cleanup and Documentation (Week 4)
- [ ] Remove all remaining duplicate mock implementations
- [ ] Update developer documentation with new mock usage guidelines
- [ ] Create migration guide for future contributors
- [ ] Add linting rules to prevent future mock duplication

## Testing Strategy

### Validation Approach

1. **Backward Compatibility Testing**:
   ```bash
   # Ensure all existing tests pass
   cargo test --workspace --no-default-features --features cpu
   cargo test --workspace --no-default-features --features gpu
   cargo test --workspace --all-features
   ```

2. **Performance Impact Assessment**:
   ```bash
   # Measure test execution time before/after
   time cargo test --package bitnet-inference --lib production_engine
   ```

3. **Mock Functionality Validation**:
   ```bash
   # Test all failure modes and edge cases
   cargo test --package bitnet-inference test_mock_failure_modes
   cargo test --package bitnet-inference test_mock_latency_simulation
   ```

4. **Integration Testing**:
   ```bash
   # Verify cross-module compatibility
   cargo test --workspace --features integration-tests
   ```

### Acceptance Criteria

#### Functional Requirements
- [ ] **Mock Consolidation**: All 24+ duplicate mock implementations replaced with centralized versions
- [ ] **Feature Parity**: New mocks provide same functionality as existing implementations
- [ ] **Enhanced Capabilities**: Support for failure injection, latency simulation, and custom responses
- [ ] **Test Coverage**: All existing tests pass without modification to test logic
- [ ] **Performance**: No regression in test execution time (< 5% increase acceptable)

#### Non-Functional Requirements
- [ ] **Maintainability**: Single source of truth for mock implementations
- [ ] **Documentation**: Comprehensive usage guidelines and examples
- [ ] **Type Safety**: Compile-time verification of mock configurations
- [ ] **Extensibility**: Easy addition of new mock behaviors without breaking changes

#### Quality Gates
- [ ] **Code Coverage**: Maintain >90% test coverage for production engine module
- [ ] **Clippy Clean**: Zero clippy warnings in test utilities module
- [ ] **Documentation**: All public mock APIs documented with examples
- [ ] **Performance**: Test suite completion time within 10% of baseline

## Risk Assessment and Mitigation

### Technical Risks

1. **Test Breakage During Migration** (High Impact, Medium Probability)
   - **Mitigation**: Gradual migration with continuous testing
   - **Rollback Plan**: Git feature branches for easy reversion

2. **Performance Regression** (Medium Impact, Low Probability)
   - **Mitigation**: Benchmark testing before/after migration
   - **Monitoring**: Automated performance regression detection

3. **Mock Behavior Inconsistency** (High Impact, Low Probability)
   - **Mitigation**: Comprehensive test suite for mock behaviors
   - **Validation**: Cross-reference with existing test expectations

### Project Risks

1. **Development Timeline** (Medium Impact, Medium Probability)
   - **Mitigation**: Phased approach allows for early feedback and course correction
   - **Contingency**: Prioritize high-impact files first

2. **Team Coordination** (Low Impact, Medium Probability)
   - **Mitigation**: Clear documentation and communication plan
   - **Process**: Regular sync meetings during migration

## Related Issues and Cross-References

### Direct Dependencies
- Issue #251: Production-Ready Inference Server (uses production engine extensively)
- Issue #260: Mock Elimination in Inference Tests (directly related refactoring effort)
- PR #XXX: Enhanced batch processing and SIMD support (may conflict with mock changes)

### Architectural Components
- **bitnet-inference**: Primary affected crate
- **bitnet-models**: Model trait definitions (interface stability critical)
- **bitnet-tokenizers**: Tokenizer trait definitions (interface stability critical)
- **bitnet-common**: Error types and result handling (used in mock implementations)

### Documentation Updates Required
- `/docs/development/test-suite.md`: Update mock usage guidelines
- `/docs/development/build-commands.md`: Add test utility build instructions
- `CONTRIBUTING.md`: Add mock implementation guidelines for contributors

### Future Enhancements
- Integration with property-based testing frameworks (QuickCheck)
- Automatic mock generation from trait definitions
- Performance profiling integration for mock overhead analysis

## Labels and Priority Classification

**Labels**:
- `refactoring` - Code structure improvement
- `testing` - Test infrastructure enhancement
- `P2-medium` - Important but not blocking critical features
- `effort/M` - Medium effort, 3-4 weeks estimated
- `area/inference` - Affects inference engine functionality
- `good-first-issue` - Phase 1 suitable for new contributors

**Priority**: **Medium** - Improves maintainability and reduces technical debt without blocking critical features

**Effort Estimate**: **3-4 weeks** for complete migration across all affected files

**Dependencies**: None blocking, can proceed immediately

---

## Implementation Notes

### Code Examples for Common Patterns

**Before (Duplicated)**:
```rust
// In multiple files:
struct MockModel { config: BitNetConfig }
impl Model for MockModel { /* 20+ lines of implementation */ }
```

**After (Centralized)**:
```rust
// Single location:
use crate::test_utils::{create_simple_mock_model, MockFailureMode};

let model = Arc::new(create_simple_mock_model());
let failing_model = Arc::new(create_failing_mock_model(MockFailureMode::Forward));
```

### Migration Checklist Template

For each file migration:
- [ ] Identify existing mock implementations
- [ ] Map mock behaviors to centralized equivalents
- [ ] Replace mock instantiation with factory functions
- [ ] Remove local mock struct definitions
- [ ] Update imports to use test_utils module
- [ ] Run file-specific tests to verify compatibility
- [ ] Update any file-specific mock behaviors to use configuration options

This comprehensive refactoring will significantly improve the maintainability and consistency of the BitNet.rs test infrastructure while providing enhanced testing capabilities for the production inference engine.
