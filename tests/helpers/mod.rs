//! Issue #261 Test Helpers Module
//!
//! Shared utilities for integration testing across all acceptance criteria.

pub mod issue_261_test_helpers;

// Re-export commonly used helpers
pub use issue_261_test_helpers::{
    AccuracyReport, Architecture, DeterministicConfig, MockDetectionResult, PerformanceMeasurement,
    PerformanceStatistics, StrictModeConfig, assert_performance_in_range,
    assert_quantization_accuracy, calculate_correlation, calculate_max_abs_error, calculate_mse,
    current_architecture, detect_mock_performance, is_cpu_feature_enabled,
    is_crossval_feature_enabled, is_ffi_feature_enabled, is_gpu_feature_enabled,
    validate_quantization_accuracy,
};
