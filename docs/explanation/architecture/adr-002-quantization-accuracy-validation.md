# ADR-002: Quantization Accuracy Validation Strategy for Real BitNet Models

## Status

**PROPOSED** - Quantization validation framework for Issue #218 real model integration

## Context

BitNet-rs implements 1-bit quantization algorithms (I2S, TL1, TL2) that are critical for neural network inference accuracy. With real BitNet model integration, we need comprehensive validation that these quantization operations maintain numerical accuracy compared to reference implementations while providing device-aware optimization.

### Current Quantization State

1. **Synthetic Testing**: Quantization tested with generated test data
2. **No Reference Validation**: No comparison against C++ implementation
3. **Device Inconsistency**: CPU and GPU paths may diverge
4. **Performance Unknown**: Real-world quantization performance unmeasured
5. **Tolerance Undefined**: No established accuracy thresholds

### Technical Requirements

- **Numerical Accuracy**: Maintain quantization precision with real model weights
- **Cross-Platform Consistency**: Identical results across CPU/GPU and platforms
- **Performance Validation**: Device-aware optimization without accuracy loss
- **Reference Compliance**: Match C++ implementation within tolerance
- **Error Detection**: Identify and prevent quantization accuracy degradation

## Decision

We will implement a **Comprehensive Quantization Validation Framework** with device-aware testing, configurable tolerances, and automated reference validation for all supported quantization formats.

### Core Quantization Validation Strategy

#### 1. Quantization Format Support Matrix

**Decision**: Support all BitNet quantization formats with device-specific optimization

```rust
/// Supported quantization formats with validation requirements
#[derive(Debug, Clone, Copy)]
pub enum QuantizationFormat {
    /// 2-bit signed quantization (BitNet primary format)
    I2S {
        /// Enable GPU acceleration if available
        gpu_accelerated: bool,
        /// Validation tolerance (default: ±1e-5)
        tolerance: f32,
    },

    /// Table lookup quantization Level 1
    TL1 {
        /// Table size for optimization
        table_size: usize,
        /// Validation tolerance (default: ±1e-4)
        tolerance: f32,
    },

    /// Table lookup quantization Level 2
    TL2 {
        /// Enhanced table size
        table_size: usize,
        /// Validation tolerance (default: ±1e-4)
        tolerance: f32,
    },

    /// GGML-compatible IQ2_S format
    IQ2S {
        /// Use FFI bridge to GGML implementation
        use_ffi: bool,
        /// Validation tolerance (default: ±1e-5)
        tolerance: f32,
    },
}
```

**Validation Requirements per Format**:
- **I2S**: Primary focus with strictest tolerance (±1e-5)
- **TL1/TL2**: Balanced accuracy/performance (±1e-4)
- **IQ2_S**: GGML compatibility validation (±1e-5)

**Rationale**:
- **Format Coverage**: Support all BitNet quantization methods
- **Tolerance Specification**: Clear accuracy requirements per format
- **Device Optimization**: Enable GPU acceleration where beneficial
- **Reference Compliance**: Match existing implementation expectations

#### 2. Device-Aware Validation Framework

**Decision**: Implement comprehensive GPU/CPU validation with parity testing

```rust
/// Device-aware quantization validation
pub struct QuantizationValidator {
    /// Supported devices for validation
    devices: Vec<Device>,
    /// Tolerance configuration per format
    tolerances: ToleranceConfig,
    /// Cross-validation configuration
    cross_validation: CrossValidationConfig,
    /// Performance benchmarking configuration
    performance_config: PerformanceConfig,
}

impl QuantizationValidator {
    /// Validate quantization accuracy across all devices
    pub fn validate_device_parity(
        &self,
        tensors: &[Tensor],
        format: QuantizationFormat
    ) -> DeviceParityResult;

    /// Validate against C++ reference implementation
    pub fn validate_cpp_reference(
        &self,
        tensors: &[Tensor],
        format: QuantizationFormat
    ) -> ReferenceValidationResult;

    /// Benchmark performance across devices
    pub fn benchmark_performance(
        &self,
        tensors: &[Tensor],
        format: QuantizationFormat
    ) -> PerformanceBenchmarkResult;
}
```

**Device Validation Requirements**:
1. **GPU/CPU Parity**: Identical outputs within tolerance
2. **Memory Consistency**: Consistent memory usage patterns
3. **Performance Validation**: GPU acceleration without accuracy loss
4. **Fallback Testing**: CPU fallback maintains accuracy

**Rationale**:
- **Device Consistency**: Ensure identical results across execution paths
- **Performance Assurance**: Validate that optimization preserves accuracy
- **Regression Detection**: Catch device-specific accuracy issues
- **Production Confidence**: Guarantee consistent production behavior

#### 3. Numerical Tolerance Configuration

**Decision**: Implement configurable tolerance system with format-specific defaults

```rust
/// Comprehensive tolerance configuration
#[derive(Debug, Clone)]
pub struct ToleranceConfig {
    /// Per-format tolerance specifications
    pub format_tolerances: HashMap<QuantizationFormat, f32>,
    /// Cross-platform tolerance adjustments
    pub platform_adjustments: HashMap<Platform, f32>,
    /// Statistical validation configuration
    pub statistical_config: StatisticalConfig,
    /// Outlier detection thresholds
    pub outlier_thresholds: OutlierConfig,
}

impl ToleranceConfig {
    /// Production-grade tolerance configuration
    pub fn production() -> Self {
        Self {
            format_tolerances: [
                (QuantizationFormat::I2S { .. }, 1e-5),
                (QuantizationFormat::TL1 { .. }, 1e-4),
                (QuantizationFormat::TL2 { .. }, 1e-4),
                (QuantizationFormat::IQ2S { .. }, 1e-5),
            ].iter().cloned().collect(),
            platform_adjustments: Self::default_platform_adjustments(),
            statistical_config: StatisticalConfig::production(),
            outlier_thresholds: OutlierConfig::production(),
        }
    }

    /// Development-friendly tolerance (slightly relaxed)
    pub fn development() -> Self;

    /// CI-optimized tolerance (platform-aware)
    pub fn ci_optimized() -> Self;
}
```

**Tolerance Rationale**:
- **I2S Precision**: Strictest tolerance (±1e-5) as primary quantization format
- **TL1/TL2 Balance**: Moderate tolerance (±1e-4) balancing accuracy and performance
- **Platform Adaptation**: Account for floating-point platform differences
- **Statistical Methods**: Use correlation metrics for robustness

#### 4. Cross-Validation Against C++ Reference

**Decision**: Implement automated cross-validation with configurable comparison methods

```rust
/// C++ reference validation framework
pub struct CppReferenceValidator {
    /// Path to C++ implementation
    cpp_implementation: CppImplementation,
    /// Validation methodology
    validation_method: ValidationMethod,
    /// Tolerance configuration
    tolerance_config: ToleranceConfig,
    /// Performance comparison enabled
    performance_comparison: bool,
}

#[derive(Debug, Clone)]
pub enum ValidationMethod {
    /// Exact numerical comparison within tolerance
    NumericalComparison { tolerance: f32 },
    /// Statistical correlation analysis
    StatisticalCorrelation { min_correlation: f64 },
    /// Hybrid approach (numerical + statistical)
    Hybrid { numerical_tolerance: f32, min_correlation: f64 },
}

impl CppReferenceValidator {
    /// Validate quantization operation against C++ reference
    pub fn validate_quantization(
        &self,
        original_tensors: &[Tensor],
        rust_quantized: &[f32],
        format: QuantizationFormat
    ) -> CrossValidationResult;

    /// Run comprehensive validation suite
    pub fn run_validation_suite(
        &self,
        test_suite: &QuantizationTestSuite
    ) -> ComprehensiveValidationResult;
}
```

**Cross-Validation Requirements**:
1. **Numerical Accuracy**: Direct comparison within tolerance
2. **Statistical Robustness**: Correlation analysis for platform variations
3. **Performance Parity**: Comparable throughput characteristics
4. **Comprehensive Coverage**: All quantization formats and device combinations

**Rationale**:
- **Accuracy Assurance**: Guarantee correctness against reference
- **Platform Robustness**: Handle floating-point platform differences
- **Regression Prevention**: Automated detection of accuracy degradation
- **Performance Validation**: Ensure optimization doesn't compromise accuracy

### Implementation Architecture

#### 1. Quantization Engine Integration

```rust
/// Enhanced quantization engine with validation
pub struct ValidatedQuantizationEngine {
    /// Core quantization implementation
    quantization_core: QuantizationCore,
    /// Validation framework
    validator: QuantizationValidator,
    /// Performance monitor
    performance_monitor: PerformanceMonitor,
    /// Device manager
    device_manager: DeviceManager,
}

impl ValidatedQuantizationEngine {
    /// Quantize tensors with automatic validation
    pub fn quantize_with_validation(
        &self,
        tensors: &[Tensor],
        format: QuantizationFormat,
        validation_config: ValidationConfig
    ) -> Result<ValidatedQuantizationResult, QuantizationError>;

    /// Run comprehensive accuracy validation
    pub fn validate_accuracy(
        &self,
        test_data: &TestData,
        validation_suite: ValidationSuite
    ) -> AccuracyValidationResult;
}

/// Comprehensive quantization result with validation metadata
#[derive(Debug)]
pub struct ValidatedQuantizationResult {
    /// Quantized tensor data
    pub quantized_tensors: QuantizedTensors,
    /// Validation results
    pub validation_results: ValidationResults,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Device information
    pub device_info: DeviceInfo,
}
```

#### 2. Real Model Tensor Processing

```rust
/// Real model tensor quantization with accuracy preservation
pub trait RealTensorQuantizer {
    /// Process real model weights with validation
    fn quantize_model_weights(
        &self,
        model: &BitNetModel,
        format: QuantizationFormat
    ) -> Result<QuantizedModelWeights, QuantizationError>;

    /// Validate quantized weights maintain model accuracy
    fn validate_model_accuracy(
        &self,
        original_model: &BitNetModel,
        quantized_weights: &QuantizedModelWeights
    ) -> ModelAccuracyResult;

    /// Benchmark quantization performance on real model
    fn benchmark_model_quantization(
        &self,
        model: &BitNetModel,
        formats: &[QuantizationFormat]
    ) -> QuantizationBenchmarkResult;
}
```

#### 3. Test Data Management for Real Models

```rust
/// Test data provider for quantization validation
pub struct QuantizationTestDataProvider {
    /// Real model data source
    model_source: ModelSource,
    /// Synthetic data generator for comprehensive testing
    synthetic_generator: SyntheticDataGenerator,
    /// Test data cache
    data_cache: TestDataCache,
}

impl QuantizationTestDataProvider {
    /// Get test tensors from real BitNet models
    pub fn get_real_model_tensors(&self, model_id: &str) -> Result<Vec<Tensor>, TestDataError>;

    /// Generate synthetic test data for edge cases
    pub fn generate_edge_case_data(&self, format: QuantizationFormat) -> Vec<Tensor>;

    /// Get comprehensive test suite for format
    pub fn get_test_suite(&self, format: QuantizationFormat) -> QuantizationTestSuite;
}
```

## Validation Testing Strategy

### 1. Unit Test Framework

```rust
/// Quantization accuracy unit tests
#[cfg(test)]
mod quantization_accuracy_tests {
    use super::*;

    /// Test I2S quantization accuracy with real model weights
    // AC:QV1
    #[test]
    fn test_i2s_quantization_accuracy_real_weights() {
        let real_tensors = load_real_model_tensors("bitnet-2b");
        let quantizer = I2SQuantizer::new(Device::CPU);

        let result = quantizer.quantize_with_validation(&real_tensors);
        assert!(result.accuracy_metrics.relative_error < 1e-5);
        assert!(result.validation_results.passed);
    }

    /// Test GPU/CPU parity for TL1 quantization
    // AC:QV2
    #[test]
    fn test_tl1_gpu_cpu_parity() {
        let test_tensors = generate_test_tensors();

        let gpu_result = TL1Quantizer::new(Device::GPU).quantize(&test_tensors);
        let cpu_result = TL1Quantizer::new(Device::CPU).quantize(&test_tensors);

        assert_arrays_equal(&gpu_result, &cpu_result, 1e-4);
    }

    /// Test cross-validation against C++ reference
    // AC:QV3
    #[test]
    fn test_cpp_cross_validation() {
        let validator = CppReferenceValidator::new();
        let test_suite = QuantizationTestSuite::comprehensive();

        let results = validator.run_validation_suite(&test_suite);
        assert!(results.overall_pass_rate > 0.95);
    }
}
```

### 2. Integration Test Framework

```rust
/// End-to-end quantization integration tests
#[cfg(test)]
mod quantization_integration_tests {
    use super::*;

    /// Test complete model quantization pipeline
    // AC:QV4
    #[test]
    fn test_end_to_end_model_quantization() {
        let model = load_real_bitnet_model("bitnet-2b.gguf");
        let engine = ValidatedQuantizationEngine::new();

        let result = engine.quantize_model_with_validation(&model, QuantizationFormat::I2S);

        assert!(result.validation_results.accuracy_preserved);
        assert!(result.performance_metrics.meets_targets());
    }

    /// Test performance regression detection
    // AC:QV5
    #[test]
    fn test_performance_regression_detection() {
        let benchmark_config = BenchmarkConfig::regression_detection();
        let results = run_quantization_benchmarks(benchmark_config);

        assert!(results.performance_regression_detected == false);
        assert!(results.accuracy_regression_detected == false);
    }
}
```

### 3. Performance Benchmark Framework

```rust
/// Quantization performance benchmarks
#[cfg(test)]
mod quantization_benchmarks {
    use super::*;

    /// Benchmark I2S quantization performance
    #[bench]
    fn bench_i2s_quantization_real_model(b: &mut Bencher) {
        let model_tensors = load_real_model_tensors("bitnet-2b");
        let quantizer = I2SQuantizer::new(Device::GPU);

        b.iter(|| {
            quantizer.quantize(&model_tensors)
        });
    }

    /// Benchmark device comparison
    #[bench]
    fn bench_device_performance_comparison(b: &mut Bencher) {
        let test_data = generate_large_tensor_set();

        b.iter(|| {
            benchmark_all_devices(&test_data, QuantizationFormat::TL1)
        });
    }
}
```

## Error Handling and Diagnostics

### 1. Quantization Error Classification

```rust
/// Comprehensive quantization error types
#[derive(Debug, thiserror::Error)]
pub enum QuantizationValidationError {
    /// Accuracy tolerance exceeded
    #[error("Accuracy tolerance exceeded: {actual_error} > {tolerance} for {format:?}")]
    AccuracyToleranceExceeded {
        format: QuantizationFormat,
        actual_error: f32,
        tolerance: f32,
        device: Device,
    },

    /// Device parity validation failed
    #[error("Device parity failed: GPU/CPU outputs differ by {difference} (tolerance: {tolerance})")]
    DeviceParityFailed {
        difference: f32,
        tolerance: f32,
        format: QuantizationFormat,
    },

    /// C++ reference validation failed
    #[error("C++ reference validation failed: correlation {correlation} < {min_correlation}")]
    CppValidationFailed {
        correlation: f64,
        min_correlation: f64,
        format: QuantizationFormat,
    },

    /// Performance regression detected
    #[error("Performance regression: {current_throughput} < {baseline_throughput} * {min_factor}")]
    PerformanceRegression {
        current_throughput: f64,
        baseline_throughput: f64,
        min_factor: f64,
    },
}
```

### 2. Diagnostic and Recovery Framework

```rust
/// Quantization diagnostics and recovery
pub struct QuantizationDiagnostics;

impl QuantizationDiagnostics {
    /// Diagnose quantization accuracy issues
    pub fn diagnose_accuracy_issue(
        error: &QuantizationValidationError
    ) -> DiagnosticReport;

    /// Suggest recovery actions for accuracy issues
    pub fn suggest_recovery_actions(
        error: &QuantizationValidationError
    ) -> Vec<RecoveryAction>;

    /// Generate detailed validation report
    pub fn generate_validation_report(
        results: &ValidationResults
    ) -> ValidationReport;
}
```

## Performance Targets and Success Metrics

### 1. Accuracy Targets

- **I2S Quantization**: ≤1e-5 relative error vs C++ reference
- **TL1/TL2 Quantization**: ≤1e-4 relative error vs C++ reference
- **Device Parity**: ≤1e-6 difference between GPU/CPU outputs
- **Cross-Platform**: ≤1e-5 variation across platforms
- **Model Preservation**: ≤0.1% perplexity degradation after quantization

### 2. Performance Targets

- **I2S GPU**: ≥90% of FP32 throughput
- **TL1/TL2 CPU**: ≥70% of unquantized throughput
- **Memory Efficiency**: ≤25% overhead for validation operations
- **Validation Speed**: ≤20% overhead for accuracy checking
- **Benchmark Stability**: ≤5% variance between runs

### 3. Quality Metrics

- **Test Coverage**: 100% of quantization operations
- **Validation Pass Rate**: ≥95% for all formats and devices
- **Regression Detection**: 100% catch rate for accuracy degradation
- **Platform Consistency**: ≤2% failure rate due to platform differences
- **Documentation Coverage**: Complete guides for all validation procedures

## Monitoring and Alerting

### 1. Continuous Monitoring

```rust
/// Quantization quality monitoring
pub struct QuantizationMonitor {
    /// Accuracy trend tracking
    accuracy_tracker: AccuracyTrendTracker,
    /// Performance regression detection
    performance_monitor: PerformanceRegressionMonitor,
    /// Alert configuration
    alert_config: AlertConfig,
}

impl QuantizationMonitor {
    /// Monitor accuracy trends over time
    pub fn track_accuracy_trends(&self) -> AccuracyTrendReport;

    /// Detect performance regressions
    pub fn detect_performance_regressions(&self) -> RegressionDetectionResult;

    /// Generate quality dashboard metrics
    pub fn generate_quality_metrics(&self) -> QualityMetrics;
}
```

### 2. Automated Alerting

```yaml
# CI alerting configuration
accuracy_alerts:
  i2s_accuracy_degradation:
    threshold: 1e-5
    action: fail_build

  device_parity_failure:
    threshold: 1e-6
    action: notify_team

  performance_regression:
    threshold: 10%  # 10% throughput decrease
    action: create_issue

quality_gates:
  validation_pass_rate:
    minimum: 95%
    measurement_window: 7_days

  cross_platform_consistency:
    maximum_failure_rate: 2%
    platforms: [linux, macos, windows]
```

## Conclusion

This quantization accuracy validation strategy ensures that BitNet-rs maintains numerical precision with real model weights while providing comprehensive device-aware optimization. The framework balances strict accuracy requirements with practical development needs through configurable tolerances, automated validation, and comprehensive error diagnostics.

The implementation provides confidence in production deployment by validating against C++ reference implementations while enabling efficient development through intelligent testing strategies and automated quality monitoring.
