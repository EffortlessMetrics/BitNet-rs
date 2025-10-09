//! Issue #261 AC2: Strict Mode Enforcement Tests
//!
//! Tests for strict mode environment variable enforcement that prevents mock fallbacks.
//!
//! Specification: docs/explanation/specs/issue-261-mock-performance-reporting-elimination-spec.md
//! API Contract: docs/explanation/specs/issue-261-api-contracts.md
//! AC Reference: AC2 (lines 134-173)

use anyhow::Result;
use bitnet_common::{
    ComputationType, Device, MissingKernelScenario, MockInferencePath, PerformanceMetrics,
    QuantizationType, StrictModeConfig, StrictModeEnforcer,
};

/// AC:AC2
/// Test strict mode environment variable detection
#[test]
fn test_strict_mode_environment_variable_detection() -> Result<()> {
    unsafe {
        std::env::set_var("BITNET_STRICT_MODE", "1");
    }

    let config = StrictModeConfig::from_env();
    assert!(config.enabled, "BITNET_STRICT_MODE=1 should enable strict mode");
    assert!(config.fail_on_mock, "Strict mode should enable fail_on_mock");
    assert!(config.require_quantization, "Strict mode should enable require_quantization");
    assert!(config.validate_performance, "Strict mode should enable validate_performance");

    Ok(())
}

/// AC:AC2
/// Test strict mode prevents mock inference fallback
#[test]
#[cfg(feature = "cpu")]
fn test_strict_mode_prevents_mock_inference() -> Result<()> {
    unsafe {
        std::env::set_var("BITNET_STRICT_MODE", "1");
        std::env::set_var("BITNET_STRICT_FAIL_ON_MOCK", "1");
    }

    let enforcer = StrictModeEnforcer::new_fresh();
    let mock_path = MockInferencePath {
        description: "Test mock path".to_string(),
        uses_mock_computation: true,
        fallback_reason: "Kernel unavailable".to_string(),
    };

    let result = enforcer.validate_inference_path(&mock_path);
    assert!(result.is_err(), "Strict mode should prevent mock fallback");

    let error_message = format!("{:?}", result.unwrap_err());
    assert!(
        error_message.contains("Mock computation detected"),
        "Error should mention mock computation"
    );

    Ok(())
}

/// AC:AC2
/// Test strict mode requires quantization kernels
#[test]
#[cfg(feature = "cpu")]
fn test_strict_mode_requires_quantization_kernels() -> Result<()> {
    unsafe {
        std::env::set_var("BITNET_STRICT_MODE", "1");
        std::env::set_var("BITNET_STRICT_REQUIRE_QUANTIZATION", "1");
    }

    let enforcer = StrictModeEnforcer::new_fresh();
    let scenario = MissingKernelScenario {
        quantization_type: QuantizationType::I2S,
        device: Device::Cpu,
        fallback_available: true, // Fallback available but strict mode should reject it
    };

    let result = enforcer.validate_kernel_availability(&scenario);
    assert!(result.is_err(), "Strict mode should require quantization kernels");

    let error_message = format!("{:?}", result.unwrap_err());
    assert!(
        error_message.contains("Required quantization kernel not available"),
        "Error should mention missing kernel"
    );

    Ok(())
}

/// AC:AC2
/// Test strict mode validates performance metrics
#[test]
#[cfg(feature = "cpu")]
fn test_strict_mode_validates_performance_metrics() -> Result<()> {
    unsafe {
        std::env::set_var("BITNET_STRICT_MODE", "1");
        std::env::set_var("BITNET_STRICT_VALIDATE_PERFORMANCE", "1");
    }

    let enforcer = StrictModeEnforcer::new_fresh();
    let metrics = PerformanceMetrics {
        tokens_per_second: 200.0, // Suspicious mock performance
        computation_type: ComputationType::Mock,
        ..Default::default()
    };

    let result = enforcer.validate_performance_metrics(&metrics);
    assert!(result.is_err(), "Strict mode should reject mock performance metrics");

    let error_message = format!("{:?}", result.unwrap_err());
    assert!(
        error_message.contains("Mock computation detected"),
        "Error should mention mock computation"
    );

    Ok(())
}

/// AC:AC2
/// Test strict mode CI enhanced mode
#[test]
fn test_strict_mode_ci_enhanced_validation() -> Result<()> {
    unsafe {
        std::env::set_var("BITNET_STRICT_MODE", "1");
        std::env::set_var("CI", "true");
        std::env::set_var("BITNET_CI_ENHANCED_STRICT", "1");
    }

    let config = StrictModeConfig::from_env_with_ci_enhancements();
    assert!(config.ci_enhanced_mode, "CI enhanced mode should be enabled");
    assert!(config.fail_fast_on_any_mock, "CI should fail fast on any mock detection");
    assert!(config.log_all_validations, "CI should log all validations");

    Ok(())
}

/// AC:AC2
/// Test strict mode descriptive error messages
#[test]
#[cfg(feature = "cpu")]
fn test_strict_mode_descriptive_errors() -> Result<()> {
    unsafe {
        std::env::set_var("BITNET_STRICT_MODE", "1");
    }

    let enforcer = StrictModeEnforcer::new_fresh();
    let mock_path = MockInferencePath {
        description: "QLinear layer fallback".to_string(),
        uses_mock_computation: true,
        fallback_reason: "I2S kernel compilation failed".to_string(),
    };

    let result = enforcer.validate_inference_path(&mock_path);
    assert!(result.is_err());

    let error_message = format!("{:?}", result.unwrap_err());
    assert!(
        error_message.contains("QLinear layer fallback"),
        "Error should contain path description"
    );

    Ok(())
}

/// AC:AC2
/// Test strict mode integration with inference pipeline
#[test]
#[cfg(feature = "cpu")]
fn test_strict_mode_inference_pipeline_integration() -> Result<()> {
    unsafe {
        std::env::set_var("BITNET_STRICT_MODE", "1");
    }

    // Verify strict mode enforcer can be created and is enabled
    let enforcer = StrictModeEnforcer::new_fresh();
    assert!(enforcer.is_enabled(), "Strict mode should be enabled");

    // Verify that real computation passes strict mode validation
    let real_metrics = PerformanceMetrics {
        tokens_per_second: 15.0, // Realistic CPU performance
        computation_type: ComputationType::Real,
        ..Default::default()
    };

    let result = enforcer.validate_performance_metrics(&real_metrics);
    assert!(result.is_ok(), "Real computation should pass strict mode validation");

    Ok(())
}
