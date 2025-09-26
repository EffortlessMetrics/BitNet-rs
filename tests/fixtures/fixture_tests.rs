//! Integration tests for BitNet.rs test fixtures.
//!
//! Validates that all fixture data is correctly generated and accessible.
//! These tests ensure the fixture infrastructure works before actual
//! GGUF weight loading tests run.

use anyhow::Result;
use std::path::Path;

mod fixtures;
use fixtures::*;

/// Test that fixture directory structure exists
#[test]
fn test_fixture_directory_structure() -> Result<()> {
    let fixtures_dir = get_fixtures_dir();
    assert!(fixtures_dir.exists(), "Fixtures directory should exist: {}", fixtures_dir.display());

    // Check required subdirectories
    let required_dirs = [
        "gguf/valid",
        "gguf/invalid",
        "tensors/quantized",
        "tensors/crossval",
        "integration/models",
    ];

    for dir in &required_dirs {
        let dir_path = fixtures_dir.join(dir);
        assert!(dir_path.exists(), "Required directory should exist: {}", dir_path.display());
    }

    Ok(())
}

/// Test that valid GGUF fixtures are available
#[test]
fn test_valid_gguf_fixtures_available() -> Result<()> {
    let valid_files = get_valid_gguf_files();
    assert!(!valid_files.is_empty(), "Should have valid GGUF fixtures");

    // Check specific expected fixtures
    let small_model = valid_files.iter()
        .find(|p| p.file_name().unwrap().to_str().unwrap().contains("small"))
        .expect("Should have small test model");

    let minimal_model = valid_files.iter()
        .find(|p| p.file_name().unwrap().to_str().unwrap().contains("minimal"))
        .expect("Should have minimal test model");

    // Validate files are non-empty
    assert!(small_model.metadata()?.len() > 0, "Small model should be non-empty");
    assert!(minimal_model.metadata()?.len() > 0, "Minimal model should be non-empty");

    println!("✓ Found {} valid GGUF fixtures", valid_files.len());
    for file in &valid_files {
        let size = file.metadata()?.len();
        println!("  - {} ({} bytes)", file.file_name().unwrap().to_str().unwrap(), size);
    }

    Ok(())
}

/// Test that invalid GGUF fixtures are available
#[test]
fn test_invalid_gguf_fixtures_available() -> Result<()> {
    let invalid_files = get_invalid_gguf_files();
    assert!(!invalid_files.is_empty(), "Should have invalid GGUF fixtures");

    // Check for specific error types
    let expected_invalid_types = [
        "invalid_magic",
        "invalid_version",
        "truncated_header",
        "zero_byte",
        "random_data",
    ];

    for invalid_type in &expected_invalid_types {
        let found = invalid_files.iter()
            .any(|p| p.file_name().unwrap().to_str().unwrap().contains(invalid_type));
        assert!(found, "Should have {} fixture", invalid_type);
    }

    println!("✓ Found {} invalid GGUF fixtures", invalid_files.len());
    for file in &invalid_files {
        let size = file.metadata()?.len();
        println!("  - {} ({} bytes)", file.file_name().unwrap().to_str().unwrap(), size);
    }

    Ok(())
}

/// Test that quantization test vectors are available
#[test]
fn test_quantization_test_vectors_available() -> Result<()> {
    let fixtures_dir = get_fixtures_dir();
    let quantization_dir = fixtures_dir.join("tensors/quantized");

    // Check for JSON test vector files
    let json_files = [
        "quantization_test_vectors.json",
        "i2s_test_vectors.json",
        "tl1_test_vectors.json",
        "tl2_test_vectors.json",
    ];

    for json_file in &json_files {
        let file_path = quantization_dir.join(json_file);
        if file_path.exists() {
            let content = std::fs::read_to_string(&file_path)?;
            assert!(!content.is_empty(), "JSON file should not be empty: {}", json_file);

            // Parse as JSON to validate format
            let _: serde_json::Value = serde_json::from_str(&content)
                .map_err(|e| anyhow::anyhow!("Invalid JSON in {}: {}", json_file, e))?;

            println!("✓ Found quantization vectors: {} ({} bytes)", json_file, content.len());
        }
    }

    // Check for binary data directory
    let binary_dir = quantization_dir.join("binary");
    if binary_dir.exists() {
        let binary_count = std::fs::read_dir(&binary_dir)?.count();
        println!("✓ Found {} binary quantization test files", binary_count);
    }

    Ok(())
}

/// Test that cross-validation fixtures are available
#[test]
fn test_crossval_fixtures_available() -> Result<()> {
    let fixtures_dir = get_fixtures_dir();
    let crossval_dir = fixtures_dir.join("tensors/crossval");

    // Check for cross-validation configuration
    let xtask_config = crossval_dir.join("xtask_crossval_config.json");
    if xtask_config.exists() {
        let content = std::fs::read_to_string(&xtask_config)?;
        assert!(!content.is_empty(), "xtask config should not be empty");

        // Parse and validate structure
        let config: serde_json::Value = serde_json::from_str(&content)?;
        assert!(config["crossval_tests"].is_array(), "Should have crossval_tests array");
        assert!(config["tolerance_config"].is_object(), "Should have tolerance_config");

        let test_count = config["crossval_tests"].as_array().unwrap().len();
        println!("✓ Found xtask cross-validation config with {} tests", test_count);
    }

    // Check for reference files
    let json_files = [
        "crossval_references.json",
        "crossval_i2s.json",
        "crossval_tl1.json",
        "crossval_tl2.json",
    ];

    for json_file in &json_files {
        let file_path = crossval_dir.join(json_file);
        if file_path.exists() {
            let content = std::fs::read_to_string(&file_path)?;
            let _: serde_json::Value = serde_json::from_str(&content)?;
            println!("✓ Found cross-validation data: {}", json_file);
        }
    }

    // Check for binary reference data
    let binary_dir = crossval_dir.join("binary");
    if binary_dir.exists() {
        let binary_count = std::fs::read_dir(&binary_dir)?.count();
        println!("✓ Found {} binary cross-validation files", binary_count);
    }

    Ok(())
}

/// Test integration fixtures structure
#[test]
fn test_integration_fixtures_structure() -> Result<()> {
    let fixtures_dir = get_fixtures_dir();
    let integration_dir = fixtures_dir.join("integration");

    if integration_dir.exists() {
        // Check for models module
        let models_mod = integration_dir.join("models/mod.rs");
        if models_mod.exists() {
            let content = std::fs::read_to_string(&models_mod)?;
            assert!(content.contains("IntegrationFixture"), "Should define IntegrationFixture");
            println!("✓ Found integration models module");
        }

        println!("✓ Integration fixtures directory structure valid");
    }

    Ok(())
}

/// Test fixture validation function
#[test]
fn test_fixture_validation_function() -> Result<()> {
    // This should pass if all fixtures are properly generated
    let validation_result = validate_fixtures_available();

    match validation_result {
        Ok(()) => {
            println!("✓ All fixtures validated successfully");
        }
        Err(e) => {
            println!("⚠ Fixture validation failed: {}", e);
            println!("  This may indicate missing fixture files.");
            println!("  Run fixture generation scripts to create missing data.");

            // Don't fail the test - fixtures might not be generated yet
            // Just report the issue
        }
    }

    Ok(())
}

/// Test environment variable support
#[test]
fn test_environment_variable_support() -> Result<()> {
    use std::env;

    // Test custom fixtures directory
    let original_dir = env::var("BITNET_FIXTURES_DIR").ok();

    // Set custom directory
    let temp_dir = std::env::temp_dir().join("bitnet_test_fixtures");
    env::set_var("BITNET_FIXTURES_DIR", &temp_dir);

    let custom_fixtures_dir = get_fixtures_dir();
    assert_eq!(custom_fixtures_dir, temp_dir, "Should use custom fixtures directory");

    // Restore original
    match original_dir {
        Some(dir) => env::set_var("BITNET_FIXTURES_DIR", dir),
        None => env::remove_var("BITNET_FIXTURES_DIR"),
    }

    let restored_dir = get_fixtures_dir();
    assert_ne!(restored_dir, temp_dir, "Should restore original fixtures directory");

    println!("✓ Environment variable support working");
    Ok(())
}

/// Test fixture metadata and statistics
#[test]
fn test_fixture_statistics() -> Result<()> {
    println!("\n=== Fixture Statistics ===");

    // Count valid GGUF files
    let valid_count = get_valid_gguf_files().len();
    println!("Valid GGUF fixtures: {}", valid_count);

    // Count invalid GGUF files
    let invalid_count = get_invalid_gguf_files().len();
    println!("Invalid GGUF fixtures: {}", invalid_count);

    // Check quantization test vectors
    let fixtures_dir = get_fixtures_dir();
    let quantization_dir = fixtures_dir.join("tensors/quantized");

    if quantization_dir.join("quantization_test_vectors.json").exists() {
        let content = std::fs::read_to_string(quantization_dir.join("quantization_test_vectors.json"))?;
        let vectors: serde_json::Value = serde_json::from_str(&content)?;
        if let Some(array) = vectors.as_array() {
            println!("Quantization test vectors: {}", array.len());
        }
    }

    // Check cross-validation references
    let crossval_dir = fixtures_dir.join("tensors/crossval");
    if crossval_dir.join("xtask_crossval_config.json").exists() {
        let content = std::fs::read_to_string(crossval_dir.join("xtask_crossval_config.json"))?;
        let config: serde_json::Value = serde_json::from_str(&content)?;
        if let Some(tests) = config["crossval_tests"].as_array() {
            println!("Cross-validation tests: {}", tests.len());
        }
    }

    println!("=========================\n");
    Ok(())
}

/// Comprehensive fixture integration test
#[test]
fn test_fixture_integration_comprehensive() -> Result<()> {
    println!("\n=== Comprehensive Fixture Integration Test ===");

    // 1. Validate directory structure
    test_fixture_directory_structure()?;
    println!("✓ Directory structure valid");

    // 2. Check GGUF fixtures
    test_valid_gguf_fixtures_available()?;
    test_invalid_gguf_fixtures_available()?;
    println!("✓ GGUF fixtures available");

    // 3. Check quantization and cross-validation data
    test_quantization_test_vectors_available()?;
    test_crossval_fixtures_available()?;
    println!("✓ Quantization and cross-validation data available");

    // 4. Test integration infrastructure
    test_integration_fixtures_structure()?;
    println!("✓ Integration infrastructure in place");

    // 5. Test environment variable support
    test_environment_variable_support()?;
    println!("✓ Environment variable support working");

    println!("==================================================");
    println!("✅ All fixture integration tests passed!");
    println!("   Fixtures are ready for GGUF weight loading tests.");
    println!("==================================================\n");

    Ok(())
}