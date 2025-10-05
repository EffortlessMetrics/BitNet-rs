//! Issue #261 Integration Test Helpers
//!
//! Shared utilities for test scaffolding across all Issue #261 acceptance criteria.
//! Provides quantization accuracy validation, performance measurement, feature gate utilities,
//! and environment configuration helpers.

#![allow(dead_code)]

use std::time::{Duration, Instant};

// ============================================================================
// Quantization Accuracy Validation Helpers
// ============================================================================

/// Calculate correlation coefficient between two vectors
pub fn calculate_correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let n = a.len() as f32;
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (ai, bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a == 0.0 || var_b == 0.0 {
        return 0.0;
    }

    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Calculate Mean Squared Error (MSE)
pub fn calculate_mse(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }

    let sum_squared_diff: f32 = a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum();

    sum_squared_diff / a.len() as f32
}

/// Calculate maximum absolute error
pub fn calculate_max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }

    a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).abs()).fold(0.0, f32::max)
}

/// Validate quantization accuracy against target
pub fn validate_quantization_accuracy(
    reference: &[f32],
    quantized_dequantized: &[f32],
    target_correlation: f32,
    max_mse: f32,
) -> Result<AccuracyReport, String> {
    let correlation = calculate_correlation(reference, quantized_dequantized);
    let mse = calculate_mse(reference, quantized_dequantized);
    let max_error = calculate_max_abs_error(reference, quantized_dequantized);

    let mut errors = Vec::new();

    if correlation < target_correlation {
        errors
            .push(format!("Correlation {:.6} below target {:.6}", correlation, target_correlation));
    }

    if mse > max_mse {
        errors.push(format!("MSE {:.2e} exceeds maximum {:.2e}", mse, max_mse));
    }

    if errors.is_empty() {
        Ok(AccuracyReport { correlation, mse, max_error, passed: true })
    } else {
        Err(errors.join("; "))
    }
}

/// Accuracy validation report
#[derive(Debug, Clone)]
pub struct AccuracyReport {
    pub correlation: f32,
    pub mse: f32,
    pub max_error: f32,
    pub passed: bool,
}

// ============================================================================
// Performance Measurement Helpers
// ============================================================================

/// Performance measurement helper
pub struct PerformanceMeasurement {
    start_time: Instant,
    warmup_iterations: usize,
    measurement_iterations: usize,
    measurements: Vec<Duration>,
}

impl PerformanceMeasurement {
    /// Create new performance measurement
    pub fn new(warmup_iterations: usize, measurement_iterations: usize) -> Self {
        Self {
            start_time: Instant::now(),
            warmup_iterations,
            measurement_iterations,
            measurements: Vec::with_capacity(measurement_iterations),
        }
    }

    /// Record warmup iteration (not included in measurements)
    pub fn record_warmup(&mut self) -> Duration {
        let now = Instant::now();
        let elapsed = now.duration_since(self.start_time);
        self.start_time = now;
        elapsed
    }

    /// Record measurement iteration
    pub fn record_measurement(&mut self) -> Duration {
        let now = Instant::now();
        let elapsed = now.duration_since(self.start_time);
        self.measurements.push(elapsed);
        self.start_time = now;
        elapsed
    }

    /// Get performance statistics
    pub fn statistics(&self) -> PerformanceStatistics {
        PerformanceStatistics::from_measurements(&self.measurements)
    }

    /// Check if measurements are complete
    pub fn is_complete(&self) -> bool {
        self.measurements.len() >= self.measurement_iterations
    }
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    pub mean_ms: f32,
    pub std_dev_ms: f32,
    pub min_ms: f32,
    pub max_ms: f32,
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    pub coefficient_of_variation: f32,
    pub sample_count: usize,
}

impl PerformanceStatistics {
    /// Create statistics from duration measurements
    pub fn from_measurements(measurements: &[Duration]) -> Self {
        let sample_count = measurements.len();
        if sample_count == 0 {
            return Self::default();
        }

        let ms_values: Vec<f32> = measurements.iter().map(|d| d.as_secs_f32() * 1000.0).collect();

        let sum: f32 = ms_values.iter().sum();
        let mean_ms = sum / sample_count as f32;

        let variance: f32 =
            ms_values.iter().map(|x| (x - mean_ms).powi(2)).sum::<f32>() / sample_count as f32;
        let std_dev_ms = variance.sqrt();

        let mut sorted = ms_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50_ms = sorted[sample_count / 2];
        let p95_ms = sorted[(sample_count as f32 * 0.95) as usize];
        let p99_ms = sorted[(sample_count as f32 * 0.99) as usize];

        Self {
            mean_ms,
            std_dev_ms,
            min_ms: sorted[0],
            max_ms: sorted[sample_count - 1],
            p50_ms,
            p95_ms,
            p99_ms,
            coefficient_of_variation: if mean_ms > 0.0 { std_dev_ms / mean_ms } else { 0.0 },
            sample_count,
        }
    }

    /// Convert to tokens per second (assuming 1 token per iteration)
    pub fn tokens_per_sec(&self) -> f32 {
        if self.mean_ms > 0.0 { 1000.0 / self.mean_ms } else { 0.0 }
    }
}

impl Default for PerformanceStatistics {
    fn default() -> Self {
        Self {
            mean_ms: 0.0,
            std_dev_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            coefficient_of_variation: 0.0,
            sample_count: 0,
        }
    }
}

// ============================================================================
// Feature Gate Utilities
// ============================================================================

/// Check if CPU feature is enabled
#[cfg(feature = "cpu")]
pub fn is_cpu_feature_enabled() -> bool {
    true
}

#[cfg(not(feature = "cpu"))]
pub fn is_cpu_feature_enabled() -> bool {
    false
}

/// Check if GPU feature is enabled
#[cfg(feature = "gpu")]
pub fn is_gpu_feature_enabled() -> bool {
    true
}

#[cfg(not(feature = "gpu"))]
pub fn is_gpu_feature_enabled() -> bool {
    false
}

/// Check if crossval feature is enabled
#[cfg(feature = "crossval")]
pub fn is_crossval_feature_enabled() -> bool {
    true
}

#[cfg(not(feature = "crossval"))]
pub fn is_crossval_feature_enabled() -> bool {
    false
}

/// Check if FFI feature is enabled
#[cfg(feature = "ffi")]
pub fn is_ffi_feature_enabled() -> bool {
    true
}

#[cfg(not(feature = "ffi"))]
pub fn is_ffi_feature_enabled() -> bool {
    false
}

/// Get current architecture
pub fn current_architecture() -> Architecture {
    #[cfg(target_arch = "x86_64")]
    {
        Architecture::X86_64
    }
    #[cfg(target_arch = "aarch64")]
    {
        Architecture::Aarch64
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Architecture::Other
    }
}

/// Architecture enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Other,
}

// ============================================================================
// Environment Configuration Helpers
// ============================================================================

/// Environment configuration for deterministic testing
pub struct DeterministicConfig {
    original_env: Vec<(String, Option<String>)>,
}

impl DeterministicConfig {
    /// Setup deterministic environment
    pub fn setup() -> Self {
        let vars = vec!["BITNET_DETERMINISTIC", "BITNET_SEED", "RAYON_NUM_THREADS"];

        let original_env: Vec<(String, Option<String>)> =
            vars.iter().map(|&var| (var.to_string(), std::env::var(var).ok())).collect();

        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("BITNET_SEED", "42");
            std::env::set_var("RAYON_NUM_THREADS", "1");
        }

        Self { original_env }
    }

    /// Restore original environment
    pub fn restore(self) {
        for (var, value) in self.original_env {
            match value {
                Some(v) => unsafe { std::env::set_var(&var, v) },
                None => unsafe { std::env::remove_var(&var) },
            }
        }
    }
}

/// Strict mode configuration for testing
pub struct StrictModeConfig {
    original_env: Vec<(String, Option<String>)>,
}

impl StrictModeConfig {
    /// Setup strict mode environment
    pub fn setup() -> Self {
        let vars = vec![
            "BITNET_STRICT_MODE",
            "BITNET_STRICT_FAIL_ON_MOCK",
            "BITNET_STRICT_REQUIRE_QUANTIZATION",
            "BITNET_STRICT_VALIDATE_PERFORMANCE",
        ];

        let original_env: Vec<(String, Option<String>)> =
            vars.iter().map(|&var| (var.to_string(), std::env::var(var).ok())).collect();

        unsafe {
            std::env::set_var("BITNET_STRICT_MODE", "1");
            std::env::set_var("BITNET_STRICT_FAIL_ON_MOCK", "1");
            std::env::set_var("BITNET_STRICT_REQUIRE_QUANTIZATION", "1");
            std::env::set_var("BITNET_STRICT_VALIDATE_PERFORMANCE", "1");
        }

        Self { original_env }
    }

    /// Restore original environment
    pub fn restore(self) {
        for (var, value) in self.original_env {
            match value {
                Some(v) => unsafe { std::env::set_var(&var, v) },
                None => unsafe { std::env::remove_var(&var) },
            }
        }
    }
}

// ============================================================================
// Mock Detection Helpers
// ============================================================================

/// Detect if performance metrics indicate mock computation
pub fn detect_mock_performance(tokens_per_sec: f32, device: &str) -> MockDetectionResult {
    if tokens_per_sec > 200.0 {
        MockDetectionResult::DefinitelyMock
    } else if tokens_per_sec > 150.0 {
        MockDetectionResult::Suspicious
    } else if device == "cpu" && tokens_per_sec > 30.0 {
        MockDetectionResult::Suspicious
    } else {
        MockDetectionResult::Legitimate
    }
}

/// Mock detection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockDetectionResult {
    Legitimate,
    Suspicious,
    DefinitelyMock,
}

// ============================================================================
// Test Assertion Helpers
// ============================================================================

/// Assert quantization accuracy meets target
pub fn assert_quantization_accuracy(
    reference: &[f32],
    quantized_dequantized: &[f32],
    target_correlation: f32,
    max_mse: f32,
) {
    let result = validate_quantization_accuracy(
        reference,
        quantized_dequantized,
        target_correlation,
        max_mse,
    );

    match result {
        Ok(report) => {
            println!("✓ Quantization accuracy validation passed:");
            println!("  Correlation: {:.6}", report.correlation);
            println!("  MSE: {:.2e}", report.mse);
            println!("  Max error: {:.2e}", report.max_error);
        }
        Err(e) => panic!("Quantization accuracy validation failed: {}", e),
    }
}

/// Assert performance within expected range
pub fn assert_performance_in_range(
    measured_tokens_per_sec: f32,
    min_expected: f32,
    max_expected: f32,
) {
    assert!(
        measured_tokens_per_sec >= min_expected && measured_tokens_per_sec <= max_expected,
        "Performance {:.2} tok/s outside expected range [{:.2}, {:.2}]",
        measured_tokens_per_sec,
        min_expected,
        max_expected
    );
}

/// Skip test if feature not enabled
#[macro_export]
macro_rules! skip_if_feature_disabled {
    ($feature:expr) => {
        if !$feature {
            eprintln!("Skipping test: required feature not enabled");
            return Ok(());
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_calculation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = calculate_correlation(&a, &b);
        assert!((corr - 1.0).abs() < 1e-6, "Perfect correlation should be 1.0");
    }

    #[test]
    fn test_mse_calculation() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let mse = calculate_mse(&a, &b);
        assert!(mse < 1e-10, "Identical vectors should have near-zero MSE");
    }

    #[test]
    fn test_performance_statistics() {
        let measurements = vec![
            Duration::from_millis(50),
            Duration::from_millis(51),
            Duration::from_millis(52),
            Duration::from_millis(53),
            Duration::from_millis(54),
        ];
        let stats = PerformanceStatistics::from_measurements(&measurements);
        assert!(stats.mean_ms > 50.0 && stats.mean_ms < 54.0);
        assert!(stats.tokens_per_sec() > 18.0 && stats.tokens_per_sec() < 21.0);
    }

    #[test]
    fn test_mock_detection() {
        assert_eq!(detect_mock_performance(17.5, "cpu"), MockDetectionResult::Legitimate);
        assert_eq!(detect_mock_performance(160.0, "gpu"), MockDetectionResult::Suspicious);
        assert_eq!(detect_mock_performance(250.0, "gpu"), MockDetectionResult::DefinitelyMock);
    }

    #[test]
    fn test_architecture_detection() {
        let arch = current_architecture();
        assert!(
            arch == Architecture::X86_64
                || arch == Architecture::Aarch64
                || arch == Architecture::Other
        );
    }
}
