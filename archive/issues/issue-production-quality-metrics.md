# [Quantization] Implement comprehensive production quality metrics for AccuracyReport

## Problem Description

The current `AccuracyReport` in `crates/bitnet-quantization/src/device_aware_quantizer.rs` provides basic accuracy metrics but lacks comprehensive production-quality validation measures. Based on the identified need for a `meets_production_quality` method, the quantization system needs enhanced metrics and validation criteria that ensure models meet production standards for deployment.

## Environment

- **File:** `crates/bitnet-quantization/src/device_aware_quantizer.rs` (lines 112-190)
- **Struct:** `AccuracyReport`
- **Current Metrics:** `max_absolute_error`, `mean_absolute_error`, `relative_error`, `passed`, `tolerance`
- **Missing Features:** Production quality thresholds, comprehensive statistical metrics

## Current Implementation Analysis

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyReport {
    pub quantization_type: QuantizationType,
    pub device: Device,
    pub max_absolute_error: f64,
    pub mean_absolute_error: f64,
    pub relative_error: f64,
    pub passed: bool,
    pub tolerance: f64,
    pub metrics: HashMap<String, f64>,
}
```

**Limitations of Current Implementation:**
1. **Basic Metrics Only**: Limited to error measurements without statistical rigor
2. **No Production Thresholds**: Missing standardized criteria for production readiness
3. **No Signal Quality Metrics**: Lacks SNR, correlation, and similarity measures
4. **Binary Pass/Fail**: No graduated quality assessment
5. **Limited Insight**: Doesn't provide actionable feedback for model optimization

## Root Cause Analysis

1. **Incremental Development**: Basic metrics were implemented first, advanced metrics deferred
2. **Lack of Industry Standards**: No established benchmarks for BitNet quantization quality
3. **Performance Focus**: Initial emphasis on speed over comprehensive validation
4. **Testing Gaps**: Production deployment requirements not fully specified

## Impact Assessment

**Severity:** Medium-High
**Component:** Quantization Quality Assurance
**Affected Areas:**
- Production model deployment without quality guarantees
- Inconsistent quantization validation across different models
- Lack of confidence in quantized model performance
- Debugging difficulty when models underperform

**Business Impact:**
- Risk of deploying suboptimal quantized models
- Inconsistent user experience due to quality variations
- Increased support burden from model quality issues
- Reduced adoption due to quality concerns

## Proposed Solution

### 1. Enhanced AccuracyReport Structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyReport {
    /// Basic identification
    pub quantization_type: QuantizationType,
    pub device: Device,
    pub tolerance: f64,

    /// Basic error metrics (existing)
    pub max_absolute_error: f64,
    pub mean_absolute_error: f64,
    pub relative_error: f64,
    pub passed: bool,

    /// Statistical quality metrics
    pub snr_db: f64,                    // Signal-to-noise ratio in dB
    pub pearson_correlation: f64,       // Linear correlation coefficient
    pub cosine_similarity: f64,         // Vector similarity measure
    pub mse: f64,                       // Mean squared error
    pub rmse: f64,                      // Root mean squared error
    pub mape: f64,                      // Mean absolute percentage error

    /// Distribution metrics
    pub std_dev_error: f64,             // Standard deviation of errors
    pub percentile_95_error: f64,       // 95th percentile error
    pub percentile_99_error: f64,       // 99th percentile error
    pub outlier_ratio: f64,             // Fraction of significant outliers

    /// Production quality assessment
    pub quality_score: f64,             // Overall quality score (0.0 to 1.0)
    pub production_ready: bool,         // Meets production criteria
    pub quality_grade: QualityGrade,    // Letter grade assessment

    /// Existing flexible metrics
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum QualityGrade {
    A,  // Excellent (>99% accuracy)
    B,  // Good (>95% accuracy)
    C,  // Acceptable (>90% accuracy)
    D,  // Poor (>80% accuracy)
    F,  // Failing (<80% accuracy)
}
```

### 2. Production Quality Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionQualityThresholds {
    /// Minimum signal-to-noise ratio (dB)
    pub min_snr_db: f64,
    /// Minimum Pearson correlation coefficient
    pub min_pearson_correlation: f64,
    /// Minimum cosine similarity
    pub min_cosine_similarity: f64,
    /// Maximum mean absolute error
    pub max_mae: f64,
    /// Maximum relative error
    pub max_relative_error: f64,
    /// Maximum outlier ratio
    pub max_outlier_ratio: f64,
    /// Minimum quality score for production
    pub min_quality_score: f64,
}

impl Default for ProductionQualityThresholds {
    fn default() -> Self {
        Self {
            min_snr_db: 40.0,                  // High signal quality
            min_pearson_correlation: 0.95,     // Strong linear correlation
            min_cosine_similarity: 0.95,       // High vector similarity
            max_mae: 0.05,                     // Low mean absolute error
            max_relative_error: 0.01,          // 1% relative error max
            max_outlier_ratio: 0.05,           // 5% outliers max
            min_quality_score: 0.9,            // 90% overall quality
        }
    }
}

impl ProductionQualityThresholds {
    /// Conservative thresholds for critical applications
    pub fn conservative() -> Self {
        Self {
            min_snr_db: 50.0,
            min_pearson_correlation: 0.98,
            min_cosine_similarity: 0.98,
            max_mae: 0.02,
            max_relative_error: 0.005,
            max_outlier_ratio: 0.02,
            min_quality_score: 0.95,
        }
    }

    /// Relaxed thresholds for development/testing
    pub fn relaxed() -> Self {
        Self {
            min_snr_db: 30.0,
            min_pearson_correlation: 0.90,
            min_cosine_similarity: 0.90,
            max_mae: 0.1,
            max_relative_error: 0.02,
            max_outlier_ratio: 0.1,
            min_quality_score: 0.8,
        }
    }
}
```

### 3. Enhanced Accuracy Calculation Methods

```rust
impl AccuracyReport {
    pub fn new_comprehensive(
        qtype: QuantizationType,
        device: Device,
        tolerance: f64,
    ) -> Self {
        Self {
            quantization_type: qtype,
            device,
            tolerance,
            max_absolute_error: 0.0,
            mean_absolute_error: 0.0,
            relative_error: 0.0,
            passed: false,
            snr_db: 0.0,
            pearson_correlation: 0.0,
            cosine_similarity: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mape: 0.0,
            std_dev_error: 0.0,
            percentile_95_error: 0.0,
            percentile_99_error: 0.0,
            outlier_ratio: 0.0,
            quality_score: 0.0,
            production_ready: false,
            quality_grade: QualityGrade::F,
            metrics: HashMap::new(),
        }
    }

    /// Update all metrics with comprehensive statistical analysis
    pub fn update_comprehensive_metrics(&mut self, original: &[f32], quantized: &[f32]) {
        // Basic error metrics (existing logic)
        self.update_errors(original, quantized);

        // Statistical quality metrics
        self.snr_db = self.calculate_snr_db(original, quantized);
        self.pearson_correlation = self.calculate_pearson_correlation(original, quantized);
        self.cosine_similarity = self.calculate_cosine_similarity(original, quantized);

        // Additional error metrics
        self.mse = self.calculate_mse(original, quantized);
        self.rmse = self.mse.sqrt();
        self.mape = self.calculate_mape(original, quantized);

        // Distribution analysis
        let errors: Vec<f64> = original.iter()
            .zip(quantized.iter())
            .map(|(o, q)| (*o - *q).abs() as f64)
            .collect();

        self.std_dev_error = self.calculate_std_dev(&errors);
        self.percentile_95_error = self.calculate_percentile(&errors, 0.95);
        self.percentile_99_error = self.calculate_percentile(&errors, 0.99);
        self.outlier_ratio = self.calculate_outlier_ratio(&errors);

        // Overall quality assessment
        self.quality_score = self.calculate_quality_score();
        self.quality_grade = self.calculate_quality_grade();
    }

    /// Check if metrics meet production quality thresholds
    pub fn meets_production_quality(&self, thresholds: &ProductionQualityThresholds) -> bool {
        self.snr_db >= thresholds.min_snr_db &&
        self.pearson_correlation >= thresholds.min_pearson_correlation &&
        self.cosine_similarity >= thresholds.min_cosine_similarity &&
        self.mean_absolute_error <= thresholds.max_mae &&
        self.relative_error <= thresholds.max_relative_error &&
        self.outlier_ratio <= thresholds.max_outlier_ratio &&
        self.quality_score >= thresholds.min_quality_score
    }

    /// Calculate signal-to-noise ratio in dB
    fn calculate_snr_db(&self, original: &[f32], quantized: &[f32]) -> f64 {
        let signal_power: f64 = original.iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>() / original.len() as f64;

        let noise_power: f64 = original.iter()
            .zip(quantized.iter())
            .map(|(o, q)| (*o as f64 - *q as f64).powi(2))
            .sum::<f64>() / original.len() as f64;

        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        }
    }

    /// Calculate Pearson correlation coefficient
    fn calculate_pearson_correlation(&self, original: &[f32], quantized: &[f32]) -> f64 {
        let n = original.len() as f64;
        let orig_mean = original.iter().sum::<f32>() as f64 / n;
        let quant_mean = quantized.iter().sum::<f32>() as f64 / n;

        let mut numerator = 0.0;
        let mut orig_var = 0.0;
        let mut quant_var = 0.0;

        for (o, q) in original.iter().zip(quantized.iter()) {
            let orig_dev = *o as f64 - orig_mean;
            let quant_dev = *q as f64 - quant_mean;

            numerator += orig_dev * quant_dev;
            orig_var += orig_dev.powi(2);
            quant_var += quant_dev.powi(2);
        }

        let denominator = (orig_var * quant_var).sqrt();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Calculate cosine similarity
    fn calculate_cosine_similarity(&self, original: &[f32], quantized: &[f32]) -> f64 {
        let dot_product: f64 = original.iter()
            .zip(quantized.iter())
            .map(|(o, q)| *o as f64 * *q as f64)
            .sum();

        let orig_magnitude: f64 = original.iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        let quant_magnitude: f64 = quantized.iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        if orig_magnitude > 0.0 && quant_magnitude > 0.0 {
            dot_product / (orig_magnitude * quant_magnitude)
        } else {
            0.0
        }
    }

    /// Calculate mean squared error
    fn calculate_mse(&self, original: &[f32], quantized: &[f32]) -> f64 {
        original.iter()
            .zip(quantized.iter())
            .map(|(o, q)| (*o as f64 - *q as f64).powi(2))
            .sum::<f64>() / original.len() as f64
    }

    /// Calculate mean absolute percentage error
    fn calculate_mape(&self, original: &[f32], quantized: &[f32]) -> f64 {
        let mut total_percentage_error = 0.0;
        let mut valid_count = 0;

        for (o, q) in original.iter().zip(quantized.iter()) {
            if o.abs() > 1e-10 {  // Avoid division by zero
                total_percentage_error += (((*o as f64 - *q as f64) / *o as f64).abs() * 100.0);
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            total_percentage_error / valid_count as f64
        } else {
            0.0
        }
    }

    /// Calculate overall quality score (0.0 to 1.0)
    fn calculate_quality_score(&self) -> f64 {
        // Weighted combination of various metrics
        let snr_score = (self.snr_db / 60.0).min(1.0).max(0.0);  // Normalize to 60dB max
        let correlation_score = self.pearson_correlation.max(0.0);
        let similarity_score = self.cosine_similarity.max(0.0);
        let error_score = (1.0 - (self.relative_error / 0.1).min(1.0)).max(0.0);  // 10% error = 0 score

        // Weighted average
        (snr_score * 0.25 + correlation_score * 0.25 + similarity_score * 0.25 + error_score * 0.25)
    }

    /// Assign letter grade based on quality score
    fn calculate_quality_grade(&self) -> QualityGrade {
        match self.quality_score {
            score if score >= 0.95 => QualityGrade::A,
            score if score >= 0.90 => QualityGrade::B,
            score if score >= 0.80 => QualityGrade::C,
            score if score >= 0.70 => QualityGrade::D,
            _ => QualityGrade::F,
        }
    }

    /// Generate detailed quality report
    pub fn generate_quality_report(&self, thresholds: &ProductionQualityThresholds) -> String {
        format!(
            "Quantization Quality Report\n\
            ===========================\n\
            Type: {:?} on {:?}\n\
            Overall Grade: {:?} (Score: {:.3})\n\
            Production Ready: {}\n\
            \n\
            Signal Quality:\n\
            - SNR: {:.2} dB (min: {:.2})\n\
            - Correlation: {:.4} (min: {:.4})\n\
            - Similarity: {:.4} (min: {:.4})\n\
            \n\
            Error Analysis:\n\
            - MAE: {:.6} (max: {:.6})\n\
            - Relative Error: {:.6} (max: {:.6})\n\
            - RMSE: {:.6}\n\
            - MAPE: {:.2}%\n\
            \n\
            Distribution:\n\
            - Std Dev: {:.6}\n\
            - 95th Percentile: {:.6}\n\
            - 99th Percentile: {:.6}\n\
            - Outlier Ratio: {:.2}% (max: {:.2}%)",
            self.quantization_type, self.device,
            self.quality_grade, self.quality_score,
            if self.meets_production_quality(thresholds) { "YES" } else { "NO" },
            self.snr_db, thresholds.min_snr_db,
            self.pearson_correlation, thresholds.min_pearson_correlation,
            self.cosine_similarity, thresholds.min_cosine_similarity,
            self.mean_absolute_error, thresholds.max_mae,
            self.relative_error, thresholds.max_relative_error,
            self.rmse,
            self.mape,
            self.std_dev_error,
            self.percentile_95_error,
            self.percentile_99_error,
            self.outlier_ratio * 100.0, thresholds.max_outlier_ratio * 100.0
        )
    }
}
```

### 4. Integration with Validation Pipeline

```rust
impl AccuracyValidator {
    pub fn validate_i2s_accuracy_comprehensive(
        &self,
        original: &[f32],
        quantized: &QuantizedTensor,
        thresholds: &ProductionQualityThresholds,
    ) -> Result<AccuracyReport> {
        let cpu_quantizer = CPUQuantizer::new(self.tolerance_config.clone());
        let dequantized = cpu_quantizer.dequantize_i2s(quantized)?;

        let mut report = AccuracyReport::new_comprehensive(
            QuantizationType::I2S,
            Device::Cpu,
            self.tolerance_config.i2s_tolerance,
        );

        // Comprehensive metric calculation
        report.update_comprehensive_metrics(original, &dequantized);

        // Production quality validation
        report.production_ready = report.meets_production_quality(thresholds);

        // Strict validation mode
        if self.tolerance_config.strict_validation && !report.production_ready {
            return Err(bitnet_common::BitNetError::Quantization(
                QuantizationError::QuantizationFailed {
                    reason: format!(
                        "Production quality validation failed:\n{}",
                        report.generate_quality_report(thresholds)
                    ),
                },
            ));
        }

        info!(
            "I2S comprehensive validation: grade={:?}, score={:.3}, production_ready={}",
            report.quality_grade, report.quality_score, report.production_ready
        );

        Ok(report)
    }
}
```

## Implementation Plan

### Phase 1: Core Metrics Infrastructure
- [ ] Extend `AccuracyReport` with comprehensive metrics fields
- [ ] Implement statistical calculation methods (SNR, correlation, similarity)
- [ ] Add `ProductionQualityThresholds` configuration system
- [ ] Create `QualityGrade` enumeration and scoring logic

### Phase 2: Advanced Analysis Features
- [ ] Implement distribution analysis (percentiles, outliers)
- [ ] Add mean absolute percentage error (MAPE) calculation
- [ ] Create comprehensive quality scoring algorithm
- [ ] Implement detailed quality reporting

### Phase 3: Integration and Validation
- [ ] Update `AccuracyValidator` to use comprehensive metrics
- [ ] Add production quality validation to quantization pipeline
- [ ] Implement threshold configuration for different use cases
- [ ] Add comprehensive test coverage for all metrics

### Phase 4: Documentation and Tooling
- [ ] Create quality assessment documentation
- [ ] Add CLI tools for quality analysis
- [ ] Implement quality visualization features
- [ ] Add benchmark comparisons with industry standards

## Testing Strategy

### Unit Tests
```bash
# Test individual metric calculations
cargo test --package bitnet-quantization accuracy_report::comprehensive_metrics

# Test production quality thresholds
cargo test --package bitnet-quantization production_quality_validation

# Test quality scoring and grading
cargo test --package bitnet-quantization quality_assessment
```

### Integration Tests
```bash
# Test with various quantization types
cargo test --package bitnet-quantization --test comprehensive_validation

# Test threshold configurations
cargo test --package bitnet-quantization --test quality_thresholds
```

### Benchmark Tests
```bash
# Performance impact of comprehensive metrics
cargo bench --package bitnet-quantization comprehensive_metrics
```

## Success Criteria

1. **Comprehensive Coverage**: All production-relevant quality metrics implemented
2. **Configurable Thresholds**: Flexible threshold system for different deployment scenarios
3. **Performance Impact**: < 10% overhead for comprehensive metric calculation
4. **Clear Reporting**: Actionable quality reports with specific improvement recommendations
5. **Integration**: Seamless integration with existing validation pipeline

## Related Issues

- Quantization algorithm optimization based on quality feedback
- Performance benchmarking with quality metrics
- Model deployment pipeline integration
- Cross-validation with reference implementations

---

**Labels:** `quantization`, `quality-assurance`, `production`, `enhancement`
**Assignee:** Quantization Team
**Epic:** Production Quality Validation
