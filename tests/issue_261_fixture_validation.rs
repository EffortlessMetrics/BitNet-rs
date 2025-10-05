//! Issue #261 Fixture Validation Tests
//!
//! Comprehensive validation that all fixtures are accessible and properly structured.
//! Run with: cargo test --no-default-features --features cpu issue_261_fixture_validation

#![allow(unused_imports)]

use anyhow::Result;

// Import all fixture modules
mod fixtures;
mod helpers;

use fixtures::*;
use helpers::{
    is_cpu_feature_enabled, is_gpu_feature_enabled, is_crossval_feature_enabled,
    is_ffi_feature_enabled, current_architecture, Architecture,
};

/// Validate all fixture modules compile and load successfully
#[test]
fn test_all_fixture_modules_compile() -> Result<()> {
    println!("✓ All fixture modules compiled successfully");
    Ok(())
}

/// Validate quantization fixtures load correctly
#[test]
#[cfg(feature = "cpu")]
fn test_quantization_fixtures_load() -> Result<()> {
    let i2s_fixtures = load_i2s_cpu_fixtures();
    assert!(!i2s_fixtures.is_empty(), "Should load I2S CPU fixtures");
    println!("✓ Loaded {} I2S CPU fixtures", i2s_fixtures.len());

    #[cfg(target_arch = "x86_64")]
    {
        let tl2_fixtures = load_tl2_cpu_fixtures();
        assert!(!tl2_fixtures.is_empty(), "Should load TL2 CPU fixtures");
        println!("✓ Loaded {} TL2 CPU fixtures", tl2_fixtures.len());
    }

    #[cfg(target_arch = "aarch64")]
    {
        let tl1_fixtures = load_tl1_cpu_fixtures();
        assert!(!tl1_fixtures.is_empty(), "Should load TL1 CPU fixtures");
        println!("✓ Loaded {} TL1 CPU fixtures", tl1_fixtures.len());
    }

    let edge_cases = load_edge_case_fixtures();
    assert!(!edge_cases.is_empty(), "Should load edge case fixtures");
    println!("✓ Loaded {} edge case fixtures", edge_cases.len());

    Ok(())
}

/// Validate GGUF model fixtures load correctly
#[test]
fn test_gguf_model_fixtures_load() -> Result<()> {
    let i2s_models = load_valid_i2s_model_fixtures();
    assert!(!i2s_models.is_empty(), "Should load I2S model fixtures");
    println!("✓ Loaded {} valid I2S models", i2s_models.len());

    let tl_models = load_valid_tl_model_fixtures();
    assert!(!tl_models.is_empty(), "Should load TL model fixtures");
    println!("✓ Loaded {} valid TL models", tl_models.len());

    let corrupted_models = load_corrupted_model_fixtures();
    assert!(!corrupted_models.is_empty(), "Should load corrupted model fixtures");
    println!("✓ Loaded {} corrupted models", corrupted_models.len());

    let alignment_fixtures = load_tensor_alignment_fixtures();
    assert!(!alignment_fixtures.is_empty(), "Should load alignment fixtures");
    println!("✓ Loaded {} tensor alignment fixtures", alignment_fixtures.len());

    Ok(())
}

/// Validate performance fixtures load correctly
#[test]
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
fn test_performance_fixtures_load() -> Result<()> {
    let cpu_i2s = load_cpu_i2s_baselines();
    assert!(!cpu_i2s.is_empty(), "Should load CPU I2S baselines");
    println!("✓ Loaded {} CPU I2S baselines", cpu_i2s.len());

    let cpu_tl2 = load_cpu_tl2_baselines();
    assert!(!cpu_tl2.is_empty(), "Should load CPU TL2 baselines");
    println!("✓ Loaded {} CPU TL2 baselines", cpu_tl2.len());

    let mock_detection = load_mock_detection_pattern_fixtures();
    assert!(!mock_detection.is_empty(), "Should load mock detection fixtures");
    println!("✓ Loaded {} mock detection patterns", mock_detection.len());

    Ok(())
}

/// Validate cross-validation fixtures load correctly
#[test]
#[cfg(feature = "crossval")]
fn test_crossval_fixtures_load() -> Result<()> {
    let i2s_crossval = crossval_i2s();
    assert!(!i2s_crossval.is_empty(), "Should load I2S cross-validation fixtures");
    println!("✓ Loaded {} I2S cross-validation fixtures", i2s_crossval.len());

    let tl_crossval = crossval_tl();
    assert!(!tl_crossval.is_empty(), "Should load TL cross-validation fixtures");
    println!("✓ Loaded {} TL cross-validation fixtures", tl_crossval.len());

    let accuracy_fixtures = load_quantization_accuracy_fixtures();
    assert!(!accuracy_fixtures.is_empty(), "Should load quantization accuracy fixtures");
    println!("✓ Loaded {} quantization accuracy fixtures", accuracy_fixtures.len());

    Ok(())
}

/// Validate strict mode fixtures load correctly
#[test]
fn test_strict_mode_fixtures_load() -> Result<()> {
    let strict_configs = load_strict_mode_config_fixtures();
    assert!(!strict_configs.is_empty(), "Should load strict mode configs");
    println!("✓ Loaded {} strict mode configurations", strict_configs.len());

    let mock_patterns = load_mock_detection_pattern_fixtures();
    assert!(!mock_patterns.is_empty(), "Should load mock detection patterns");
    println!("✓ Loaded {} mock detection patterns", mock_patterns.len());

    let ci_validation = load_ci_validation_fixtures();
    assert!(!ci_validation.is_empty(), "Should load CI validation fixtures");
    println!("✓ Loaded {} CI validation scenarios", ci_validation.len());

    Ok(())
}

/// Validate test helpers compile and work correctly
#[test]
fn test_helpers_compile() -> Result<()> {
    // Test correlation calculation
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let corr = calculate_correlation(&a, &b);
    assert!((corr - 1.0).abs() < 1e-6, "Perfect correlation should be 1.0");
    println!("✓ Correlation calculation works");

    // Test MSE calculation
    let mse = calculate_mse(&a, &b);
    assert!(mse < 1e-10, "Identical vectors should have near-zero MSE");
    println!("✓ MSE calculation works");

    // Test architecture detection
    let arch = current_architecture();
    println!("✓ Detected architecture: {:?}", arch);

    Ok(())
}

/// Validate feature gate detection
#[test]
fn test_feature_gate_detection() -> Result<()> {
    println!("Feature gate status:");
    println!("  CPU: {}", is_cpu_feature_enabled());
    println!("  GPU: {}", is_gpu_feature_enabled());
    println!("  Crossval: {}", is_crossval_feature_enabled());
    println!("  FFI: {}", is_ffi_feature_enabled());

    #[cfg(feature = "cpu")]
    assert!(is_cpu_feature_enabled(), "CPU feature should be enabled");

    #[cfg(not(feature = "cpu"))]
    assert!(!is_cpu_feature_enabled(), "CPU feature should be disabled");

    println!("✓ Feature gate detection works");
    Ok(())
}

/// Validate fixture integrity
#[test]
#[cfg(feature = "cpu")]
fn test_fixture_integrity_validation() -> Result<()> {
    let fixtures = load_i2s_cpu_fixtures();
    for fixture in fixtures {
        validate_fixture_integrity(&fixture).expect("Fixture should be valid");
    }
    println!("✓ All I2S CPU fixtures passed integrity validation");
    Ok(())
}

/// Validate GGUF model fixture integrity
#[test]
fn test_gguf_fixture_integrity_validation() -> Result<()> {
    let models = load_valid_i2s_model_fixtures();
    for model in models {
        if model.validation_flags.check_tensor_alignment {
            validate_gguf_fixture(&model).expect("GGUF model should be valid");
        }
    }
    println!("✓ All valid GGUF models passed integrity validation");
    Ok(())
}

/// Print comprehensive fixture coverage summary
#[test]
fn test_fixture_coverage_summary() -> Result<()> {
    println!("\n{}", fixture_coverage_summary());
    Ok(())
}
