# [Testing] Test Infrastructure Mock Object Consolidation and Framework Modernization

## Problem Description

BitNet.rs test suite suffers from extensive code duplication and fragmented mock object implementations across multiple modules. Five critical test infrastructure issues require systematic resolution:

1. **Duplicate Mock Objects**: `MockModel` and `MockTokenizer` implementations duplicated across `engine.rs`, `production_engine.rs`, and `backends.rs`
2. **Inconsistent Mock Implementations**: Mock objects have different capabilities and interfaces across modules
3. **Placeholder Test Frameworks**: `create_mock_discovery()` panics instead of providing proper mock functionality
4. **Quantization Test Bypass**: `MockTensor::quantize()` bypasses `DeviceAwareQuantizer` testing critical selection logic
5. **Fragmented Test Utilities**: No centralized test infrastructure causing maintenance overhead

## Environment

- **Affected Crates**: `bitnet-inference`, `bitnet-tokenizers`, `bitnet-quantization`
- **Primary Files**:
  - `crates/bitnet-inference/src/engine.rs` (tests module)
  - `crates/bitnet-inference/src/production_engine.rs` (tests module)
  - `crates/bitnet-inference/src/backends.rs` (tests module)
  - `crates/bitnet-tokenizers/src/fallback.rs` (tests module)
  - `crates/bitnet-quantization/src/property_tests.rs`
- **Test Architecture**: `cargo test --workspace --no-default-features --features cpu`
- **Mock Framework**: Currently ad-hoc, requires standardization

## Root Cause Analysis

### Code Duplication Problems

1. **Redundant Mock Implementations**: Three separate `MockModel` implementations with different features
   ```rust
   // engine.rs - Basic mock
   struct MockModel { config: BitNetConfig }

   // production_engine.rs - Production mock with additional methods
   struct MockModel { config: BitNetConfig, performance_metrics: HashMap<String, f64> }

   // backends.rs - Backend-specific mock
   struct MockModel { config: BitNetConfig, device_type: DeviceType }
   ```

2. **Inconsistent Mock Behavior**: Different mock implementations return different default values and have varying method implementations

3. **Test Framework Gaps**: Missing proper mock framework leading to panic-based test stubs

### Testing Architecture Issues

1. **Mock Discovery Placeholder**: Test framework relies on filesystem access instead of proper mocking
   ```rust
   fn create_mock_discovery() -> TokenizerDiscovery {
       match TokenizerDiscovery::from_gguf(&test_path) {
           Err(_) => panic!("requires valid GGUF file or mock framework")
       }
   }
   ```

2. **Quantization Logic Bypass**: Tests skip `DeviceAwareQuantizer` preventing validation of critical selection paths
   ```rust
   match qtype {
       QuantizationType::I2S => crate::i2s::quantize_i2s(self), // Direct call bypasses selection logic
   }
   ```

## Impact Assessment

- **Severity**: Medium-High - Affects test quality and maintenance efficiency
- **Code Quality Impact**: High technical debt from duplicated mock implementations
- **Test Coverage Impact**: Critical paths untested due to mock bypasses
- **Maintenance Burden**: Multiple mock implementations require synchronized updates
- **Developer Experience**: Confusing test infrastructure slows development

## Proposed Solution

### 1. Centralized Test Utilities Module

**Unified Mock Framework Architecture**:
```rust
// crates/bitnet-test-utils/src/lib.rs
pub mod mocks {
    pub mod model;
    pub mod tokenizer;
    pub mod tensor;
    pub mod discovery;
}

pub mod builders {
    pub mod config_builder;
    pub mod tensor_builder;
    pub mod model_builder;
}

pub mod assertions {
    pub mod tensor_assertions;
    pub mod accuracy_assertions;
    pub mod performance_assertions;
}

pub mod fixtures {
    pub mod test_data;
    pub mod sample_models;
    pub mod reference_outputs;
}
```

### 2. Comprehensive Mock Model System

**Unified MockModel with Builder Pattern**:
```rust
// crates/bitnet-test-utils/src/mocks/model.rs
pub struct MockModel {
    config: BitNetConfig,
    behavior: ModelBehavior,
    performance: PerformanceProfile,
    device_compatibility: DeviceCompatibility,
}

pub struct ModelBehavior {
    forward_latency: Duration,
    memory_usage: usize,
    accuracy_profile: AccuracyProfile,
    error_injection: Option<ErrorInjectionConfig>,
}

impl MockModel {
    pub fn builder() -> MockModelBuilder {
        MockModelBuilder::default()
    }

    pub fn simple() -> Self {
        Self::builder().build()
    }

    pub fn production_like() -> Self {
        Self::builder()
            .with_realistic_performance()
            .with_production_config()
            .build()
    }

    pub fn gpu_compatible() -> Self {
        Self::builder()
            .with_device_compatibility(DeviceCompatibility::GpuPreferred)
            .build()
    }
}

pub struct MockModelBuilder {
    config: Option<BitNetConfig>,
    behavior: ModelBehavior,
    performance: PerformanceProfile,
    device_compatibility: DeviceCompatibility,
}

impl MockModelBuilder {
    pub fn with_config(mut self, config: BitNetConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.behavior.forward_latency = latency;
        self
    }

    pub fn with_error_injection(mut self, config: ErrorInjectionConfig) -> Self {
        self.behavior.error_injection = Some(config);
        self
    }

    pub fn build(self) -> MockModel {
        MockModel {
            config: self.config.unwrap_or_default(),
            behavior: self.behavior,
            performance: self.performance,
            device_compatibility: self.device_compatibility,
        }
    }
}

impl Model for MockModel {
    fn forward(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        // Simulate configurable latency
        if self.behavior.forward_latency > Duration::ZERO {
            std::thread::sleep(self.behavior.forward_latency);
        }

        // Error injection for testing error paths
        if let Some(error_config) = &self.behavior.error_injection {
            if error_config.should_inject_error() {
                return Err(error_config.generate_error());
            }
        }

        // Generate realistic output based on accuracy profile
        let output = self.generate_output_tensor(input)?;
        Ok(output)
    }

    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn memory_usage(&self) -> usize {
        self.performance.memory_usage
    }

    fn device_requirements(&self) -> DeviceRequirements {
        self.device_compatibility.requirements()
    }
}
```

### 3. Advanced Mock Tokenizer System

**Configurable MockTokenizer**:
```rust
// crates/bitnet-test-utils/src/mocks/tokenizer.rs
pub struct MockTokenizer {
    vocab_size: usize,
    special_tokens: SpecialTokens,
    encoding_behavior: EncodingBehavior,
    performance_profile: TokenizerPerformance,
}

pub enum EncodingBehavior {
    Deterministic { seed: u64 },
    Realistic { distribution: TokenDistribution },
    WorstCase { max_tokens: usize },
    ErrorProne { error_rate: f64 },
}

impl MockTokenizer {
    pub fn gpt2_like() -> Self {
        Self {
            vocab_size: 50257,
            special_tokens: SpecialTokens::gpt2_defaults(),
            encoding_behavior: EncodingBehavior::Realistic {
                distribution: TokenDistribution::gpt2_distribution()
            },
            performance_profile: TokenizerPerformance::fast(),
        }
    }

    pub fn llama_like() -> Self {
        Self {
            vocab_size: 32000,
            special_tokens: SpecialTokens::llama_defaults(),
            encoding_behavior: EncodingBehavior::Realistic {
                distribution: TokenDistribution::llama_distribution()
            },
            performance_profile: TokenizerPerformance::medium(),
        }
    }

    pub fn with_error_injection(mut self, error_rate: f64) -> Self {
        self.encoding_behavior = EncodingBehavior::ErrorProne { error_rate };
        self
    }
}

impl Tokenizer for MockTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        match &self.encoding_behavior {
            EncodingBehavior::Deterministic { seed } => {
                self.deterministic_tokenize(text, *seed)
            }
            EncodingBehavior::Realistic { distribution } => {
                self.realistic_tokenize(text, distribution)
            }
            EncodingBehavior::WorstCase { max_tokens } => {
                self.worst_case_tokenize(text, *max_tokens)
            }
            EncodingBehavior::ErrorProne { error_rate } => {
                if fastrand::f64() < *error_rate {
                    Err(anyhow::anyhow!("Injected tokenization error"))
                } else {
                    self.deterministic_tokenize(text, 42)
                }
            }
        }
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // Implement consistent detokenization logic
        self.reverse_tokenization(tokens)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}
```

### 4. Proper Mock Discovery Framework

**Robust TokenizerDiscovery Mock**:
```rust
// crates/bitnet-test-utils/src/mocks/discovery.rs
pub struct MockTokenizerDiscovery {
    embedded_tokenizer: Option<Arc<dyn Tokenizer + Send + Sync>>,
    colocated_paths: Vec<PathBuf>,
    cache_locations: Vec<PathBuf>,
    download_sources: Vec<DownloadInfo>,
    behavior: DiscoveryBehavior,
}

pub enum DiscoveryBehavior {
    AlwaysSucceed,
    AlwaysFail,
    PartialSuccess { success_rate: f64 },
    Sequence { responses: Vec<DiscoveryResponse> },
}

impl MockTokenizerDiscovery {
    pub fn with_embedded_tokenizer(tokenizer: Arc<dyn Tokenizer + Send + Sync>) -> Self {
        Self {
            embedded_tokenizer: Some(tokenizer),
            colocated_paths: vec![],
            cache_locations: vec![],
            download_sources: vec![],
            behavior: DiscoveryBehavior::AlwaysSucceed,
        }
    }

    pub fn with_colocated_files(paths: Vec<PathBuf>) -> Self {
        Self {
            embedded_tokenizer: None,
            colocated_paths: paths,
            cache_locations: vec![],
            download_sources: vec![],
            behavior: DiscoveryBehavior::AlwaysSucceed,
        }
    }

    pub fn with_behavior(mut self, behavior: DiscoveryBehavior) -> Self {
        self.behavior = behavior;
        self
    }
}

impl TokenizerDiscovery for MockTokenizerDiscovery {
    fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer + Send + Sync>>> {
        match &self.behavior {
            DiscoveryBehavior::AlwaysFail => {
                Err(anyhow::anyhow!("Mock discovery configured to fail"))
            }
            DiscoveryBehavior::PartialSuccess { success_rate } => {
                if fastrand::f64() < *success_rate {
                    Ok(self.embedded_tokenizer.clone())
                } else {
                    Err(anyhow::anyhow!("Mock partial failure"))
                }
            }
            _ => Ok(self.embedded_tokenizer.clone()),
        }
    }

    fn check_colocated_tokenizers(&self) -> Result<Option<PathBuf>> {
        if self.colocated_paths.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.colocated_paths[0].clone()))
        }
    }

    fn infer_download_source(&self) -> Result<Option<DownloadInfo>> {
        if self.download_sources.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.download_sources[0].clone()))
        }
    }
}

// Convenient test helper functions
pub fn create_mock_discovery() -> MockTokenizerDiscovery {
    MockTokenizerDiscovery::with_embedded_tokenizer(
        Arc::new(MockTokenizer::gpt2_like())
    )
}

pub fn create_failing_mock_discovery() -> MockTokenizerDiscovery {
    MockTokenizerDiscovery::with_behavior(DiscoveryBehavior::AlwaysFail)
}
```

### 5. Device-Aware Quantization Testing

**Proper DeviceAwareQuantizer Integration**:
```rust
// crates/bitnet-test-utils/src/mocks/tensor.rs
pub struct MockTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    device_type: DeviceType,
    quantization_strategy: QuantizationStrategy,
}

impl MockTensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: (0..size).map(|i| (i as f32) * 0.1).collect(),
            shape,
            device_type: DeviceType::Cpu,
            quantization_strategy: QuantizationStrategy::Automatic,
        }
    }

    pub fn with_device(mut self, device_type: DeviceType) -> Self {
        self.device_type = device_type;
        self
    }

    pub fn with_quantization_strategy(mut self, strategy: QuantizationStrategy) -> Self {
        self.quantization_strategy = strategy;
        self
    }
}

#[cfg(test)]
impl Quantize for MockTensor {
    fn quantize(&self, qtype: QuantizationType) -> Result<QuantizedTensor, crate::BitNetError> {
        // Use proper DeviceAwareQuantizer to test selection logic
        let quantizer = DeviceAwareQuantizer::new()
            .with_device_preference(self.device_type)
            .with_strategy(self.quantization_strategy);

        // Test the actual quantizer selection path
        quantizer.quantize_with_validation(
            self.data.as_slice(),
            qtype,
            &self.shape
        )
    }
}

// Alternative implementation for testing quantizer bypass
impl MockTensor {
    pub fn quantize_direct(&self, qtype: QuantizationType) -> Result<QuantizedTensor, crate::BitNetError> {
        // Direct quantization for testing individual quantizer implementations
        match qtype {
            QuantizationType::I2S => crate::i2s::quantize_i2s(self),
            QuantizationType::TL1 => crate::tl1::quantize_tl1(self),
            QuantizationType::TL2 => crate::tl2::quantize_tl2(self),
        }
    }

    pub fn quantize_with_device_aware(&self, qtype: QuantizationType) -> Result<QuantizedTensor, crate::BitNetError> {
        // Full device-aware quantization path for integration testing
        let quantizer = DeviceAwareQuantizer::new();
        quantizer.quantize_with_validation(self.data.as_slice(), qtype, &self.shape)
    }
}
```

## Implementation Plan

### Phase 1: Test Utilities Infrastructure (Week 1-2)
- [ ] Create `bitnet-test-utils` crate with proper module structure
- [ ] Implement `MockModelBuilder` with comprehensive configuration options
- [ ] Create `MockTokenizer` with realistic behavior simulation
- [ ] Add test data fixtures and sample model configurations

### Phase 2: Mock Discovery Framework (Week 2-3)
- [ ] Implement `MockTokenizerDiscovery` with configurable behavior
- [ ] Replace panic-based test stubs with proper mock framework
- [ ] Add error injection and partial failure simulation
- [ ] Create convenient test helper functions

### Phase 3: Quantization Test Integration (Week 3-4)
- [ ] Update `MockTensor::quantize()` to use `DeviceAwareQuantizer`
- [ ] Add separate direct quantization methods for unit testing
- [ ] Implement device-aware quantization path testing
- [ ] Create comprehensive quantization test scenarios

### Phase 4: Mock Consolidation (Week 4-5)
- [ ] Replace all duplicate mock implementations with centralized versions
- [ ] Update existing tests to use new mock framework
- [ ] Remove obsolete mock code from individual modules
- [ ] Add migration guide for test updates

### Phase 5: Enhanced Test Capabilities (Week 5-6)
- [ ] Add performance profiling mocks for benchmarking tests
- [ ] Implement error injection frameworks for robustness testing
- [ ] Create realistic test scenarios matching production workloads
- [ ] Add cross-validation test helpers

## Testing Strategy

### Mock Framework Validation
```rust
#[test]
fn test_mock_model_builder_pattern() {
    let model = MockModel::builder()
        .with_latency(Duration::from_millis(10))
        .with_error_injection(ErrorInjectionConfig::new(0.1))
        .build();

    let input = MockTensor::new(vec![1, 64, 768]);

    // Test multiple runs for error injection
    let mut successes = 0;
    let mut errors = 0;

    for _ in 0..100 {
        match model.forward(&input.to_bitnet_tensor()) {
            Ok(_) => successes += 1,
            Err(_) => errors += 1,
        }
    }

    // Should have roughly 10% error rate
    assert!(errors > 5 && errors < 20);
    assert!(successes > 80);
}

#[test]
fn test_mock_tokenizer_consistency() {
    let tokenizer = MockTokenizer::gpt2_like();

    let text = "Hello, world!";
    let tokens = tokenizer.tokenize(text).unwrap();
    let reconstructed = tokenizer.detokenize(&tokens).unwrap();

    // Mock should be internally consistent
    assert_eq!(text, reconstructed);
}
```

### Device-Aware Quantization Testing
```rust
#[test]
fn test_mock_tensor_device_aware_quantization() {
    let tensor = MockTensor::new(vec![32, 64])
        .with_device(DeviceType::Gpu)
        .with_quantization_strategy(QuantizationStrategy::Aggressive);

    // Test that DeviceAwareQuantizer is properly used
    let quantized = tensor.quantize(QuantizationType::I2S).unwrap();

    // Verify quantization metadata
    assert_eq!(quantized.device_type, DeviceType::Gpu);
    assert!(quantized.kernel_used.contains("gpu") || quantized.kernel_used.contains("cuda"));
}

#[test]
fn test_quantization_path_comparison() {
    let tensor = MockTensor::new(vec![32, 64]);

    // Compare direct vs device-aware quantization
    let direct_result = tensor.quantize_direct(QuantizationType::I2S).unwrap();
    let device_aware_result = tensor.quantize_with_device_aware(QuantizationType::I2S).unwrap();

    // Results should be numerically equivalent
    assert_tensors_close(&direct_result.data, &device_aware_result.data, 1e-6);

    // But metadata should differ
    assert_ne!(direct_result.quantization_path, device_aware_result.quantization_path);
}
```

### Mock Discovery Validation
```rust
#[test]
fn test_mock_discovery_error_scenarios() {
    let discovery = MockTokenizerDiscovery::with_behavior(
        DiscoveryBehavior::PartialSuccess { success_rate: 0.5 }
    );

    let mut successes = 0;
    let mut failures = 0;

    for _ in 0..100 {
        match discovery.try_extract_embedded_tokenizer() {
            Ok(_) => successes += 1,
            Err(_) => failures += 1,
        }
    }

    // Should have roughly 50% success rate
    assert!(successes > 30 && successes < 70);
    assert!(failures > 30 && failures < 70);
}
```

## Risk Assessment

### Implementation Risks
1. **Breaking Changes**: Replacing existing mocks may break current tests
   - *Mitigation*: Gradual migration with compatibility wrappers, comprehensive test validation
2. **Performance Overhead**: More sophisticated mocks may slow test execution
   - *Mitigation*: Performance profiling, lightweight default configurations
3. **Complexity Increase**: More features may make test setup more complex
   - *Mitigation*: Simple defaults, comprehensive documentation, builder patterns

### Quality Risks
1. **Test Coverage Gaps**: Migration may miss edge cases from existing tests
   - *Mitigation*: Systematic test inventory, parallel validation during migration
2. **Mock Behavior Drift**: Sophisticated mocks may not match real implementation behavior
   - *Mitigation*: Regular cross-validation with real implementations, behavior consistency tests

## Acceptance Criteria

### Consolidation Requirements
- [ ] All duplicate mock implementations removed from individual modules
- [ ] Single centralized mock framework serving all test modules
- [ ] No panic-based test stubs remaining in codebase
- [ ] Device-aware quantization properly tested through mocks

### Quality Requirements
- [ ] All existing tests continue to pass with new mock framework
- [ ] Mock implementations provide realistic behavior simulation
- [ ] Comprehensive error injection capabilities for robustness testing
- [ ] Test execution time increased by <10% despite enhanced capabilities

### Usability Requirements
- [ ] Simple default configurations for common test scenarios
- [ ] Builder patterns for complex test setup requirements
- [ ] Clear documentation and migration guide for developers
- [ ] Consistent mock behavior across all test modules

## Related Issues

- BitNet.rs #260: Mock elimination project (complementary work)
- BitNet.rs #251: Production-ready inference server (benefits from better testing)
- BitNet.rs #218: Device-aware quantization system (requires proper testing)

## Implementation Notes

### BitNet.rs Integration
- Maintain compatibility with existing feature flag architecture (`--features cpu,gpu`)
- Integrate with `crossval` framework for reference implementation testing
- Use existing error handling patterns with `anyhow`
- Follow project testing conventions and naming patterns

### Migration Strategy
1. **Parallel Implementation**: Implement new mock framework alongside existing mocks
2. **Gradual Migration**: Update tests module by module to use new framework
3. **Validation Period**: Run both old and new mock tests in parallel during transition
4. **Cleanup Phase**: Remove obsolete mock implementations after full migration