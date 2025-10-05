//! Issue #261 Test Fixtures Module
//!
//! Comprehensive test fixtures for Mock Inference Performance Reporting Elimination.
//! Organized by acceptance criteria:
//!
//! - AC2: Strict mode configuration fixtures
//! - AC3/AC4: Quantization test data (I2S, TL1, TL2)
//! - AC5: GGUF model fixtures (tensor alignment, metadata)
//! - AC6: CI mock rejection fixtures
//! - AC7/AC8: Performance measurement fixtures (CPU/GPU baselines)
//! - AC9: Cross-validation reference data (C++ parity)

pub mod issue_261_crossval_reference_data;
pub mod issue_261_gguf_test_models;
pub mod issue_261_performance_test_data;
pub mod issue_261_quantization_test_data;
pub mod issue_261_strict_mode_fixtures;

// Re-export commonly used fixtures
pub use issue_261_crossval_reference_data::{
    CrossValFixture, CrossValReport, QuantizationAccuracyFixture, ValidationTargets,
    calculate_correlation, calculate_max_abs_error, calculate_mse, validate_crossval_results,
};

pub use issue_261_gguf_test_models::{
    GgufMetadata, GgufModelFixture, GgufQuantizationType, GgufTensorInfo, TensorAlignmentFixture,
    ValidationFlags, get_gguf_fixture_by_id, load_corrupted_model_fixtures,
    load_tensor_alignment_fixtures, load_valid_i2s_model_fixtures, load_valid_tl_model_fixtures,
    validate_gguf_fixture,
};

pub use issue_261_performance_test_data::{
    ComputationType, ComputeCapability, CpuArchitecture, GpuPerformanceFixture, LatencyPercentiles,
    MockDetectionResult, MockPerformanceFixture, PerformanceBaselineFixture, PerformanceRange,
    PerformanceStatistics, PrecisionMode, detect_mock_performance, load_mock_detection_fixtures,
    validate_performance_against_baseline,
};

pub use issue_261_quantization_test_data::{
    DeviceType, EdgeCaseFixture, ExpectedBehavior, QuantizationTestFixture, get_fixture_by_id,
    load_edge_case_fixtures, validate_fixture_integrity,
};

pub use issue_261_strict_mode_fixtures::{
    CiValidationFixture, DetectionConfidence, IndicatorType, KernelAvailability,
    MockDetectionPatternFixture, MockIndicator, MockScenario, StrictModeAction, StrictModeBehavior,
    StrictModeConfigFixture, ValidationCheckType, ValidationRule, determine_strict_mode_action,
    get_mock_pattern_by_id, get_strict_mode_fixture_by_id, load_ci_validation_fixtures,
    load_mock_detection_pattern_fixtures, load_strict_mode_config_fixtures,
    validate_environment_config,
};

// Platform-specific fixture loaders
#[cfg(feature = "cpu")]
pub use issue_261_quantization_test_data::load_i2s_cpu_fixtures;

#[cfg(feature = "gpu")]
pub use issue_261_quantization_test_data::load_i2s_gpu_fixtures;

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
pub use issue_261_quantization_test_data::{QuantizationType, load_tl2_cpu_fixtures};

#[cfg(all(feature = "cpu", target_arch = "aarch64"))]
pub use issue_261_quantization_test_data::{QuantizationType, load_tl1_cpu_fixtures};

#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
pub use issue_261_performance_test_data::{load_cpu_i2s_baselines, load_cpu_tl2_baselines};

#[cfg(all(feature = "cpu", target_arch = "aarch64"))]
pub use issue_261_performance_test_data::load_cpu_tl1_baselines;

#[cfg(feature = "gpu")]
pub use issue_261_performance_test_data::load_gpu_i2s_baselines;

#[cfg(feature = "crossval")]
pub use issue_261_crossval_reference_data::{
    load_i2s_crossval_fixtures, load_quantization_accuracy_fixtures, load_tl_crossval_fixtures,
};

// Convenience aliases for crossval
#[cfg(feature = "crossval")]
pub use load_i2s_crossval_fixtures as crossval_i2s;
#[cfg(feature = "crossval")]
pub use load_tl_crossval_fixtures as crossval_tl;

/// Fixture coverage summary
pub fn fixture_coverage_summary() -> String {
    format!(
        r#"Issue #261 Fixture Coverage:
- AC2: Strict mode configurations (4 fixtures)
- AC3: I2S quantization test data (5 CPU, 3 GPU fixtures)
- AC4: TL1/TL2 quantization test data (3 TL1, 3 TL2 fixtures)
- AC5: GGUF model fixtures (2 I2S, 2 TL, 3 corrupted, 5 alignment)
- AC6: CI mock rejection (5 CI validation, 5 mock detection patterns)
- AC7: CPU performance baselines (2 I2S, 1 TL1, 1 TL2)
- AC8: GPU performance baselines (3 fixtures)
- AC9: Cross-validation data (3 I2S, 2 TL, 3 accuracy fixtures)
Total: 45+ test fixtures covering all acceptance criteria
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_modules_compile() {
        // Ensure all fixture modules compile successfully
        println!("{}", fixture_coverage_summary());
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_quantization_fixtures_load() {
        let fixtures = load_i2s_cpu_fixtures();
        assert!(!fixtures.is_empty(), "Should load CPU I2S fixtures");
    }

    #[test]
    fn test_gguf_fixtures_load() {
        let fixtures = load_valid_i2s_model_fixtures();
        assert!(!fixtures.is_empty(), "Should load GGUF I2S model fixtures");
    }

    #[test]
    fn test_strict_mode_fixtures_load() {
        let fixtures = load_strict_mode_config_fixtures();
        assert!(!fixtures.is_empty(), "Should load strict mode config fixtures");
    }
}
