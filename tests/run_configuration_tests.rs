// Standalone test runner for configuration testing
// This allows us to test the configuration functionality independently

use bitnet_tests::{
    config::{
        ci_config, dev_config, load_config_from_env, load_config_from_file, load_test_config,
        merge_configs, minimal_config, save_config_to_file, validate_config, ComparisonTolerance,
        CrossValidationConfig, CustomFixture, FixtureConfig, ReportFormat, ReportingConfig,
        TestConfig,
    },
    config_validator::{ConfigValidator, ValidationResult},
    errors::{TestError, TestResult},
    FixtureManager,
};
use serde_json;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::{tempdir, TempDir};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Configuration Tests...");

    // Test 1: Default Configuration
    println!("\n1. Testing default configuration...");
    test_default_configuration().await?;
    println!("✓ Default configuration test passed");

    // Test 2: Predefined Configurations
    println!("\n2. Testing predefined configurations...");
    test_predefined_configurations().await?;
    println!("✓ Predefined configurations test passed");

    // Test 3: Configuration Validation
    println!("\n3. Testing configuration validation...");
    test_configuration_validation().await?;
    println!("✓ Configuration validation test passed");

    // Test 4: Configuration Merging
    println!("\n4. Testing configuration merging...");
    test_configuration_merging().await?;
    println!("✓ Configuration merging test passed");

    // Test 5: Configuration Serialization
    println!("\n5. Testing configuration serialization...");
    test_configuration_serialization().await?;
    println!("✓ Configuration serialization test passed");

    // Test 6: Environment Override
    println!("\n6. Testing environment override...");
    test_environment_override().await?;
    println!("✓ Environment override test passed");

    // Test 7: Feature Flag Combinations
    println!("\n7. Testing feature flag combinations...");
    test_feature_flag_combinations().await?;
    println!("✓ Feature flag combinations test passed");

    // Test 8: Platform-Specific Configuration
    println!("\n8. Testing platform-specific configuration...");
    test_platform_specific_configuration().await?;
    println!("✓ Platform-specific configuration test passed");

    // Test 9: Configuration Error Handling
    println!("\n9. Testing configuration error handling...");
    test_configuration_error_handling().await?;
    println!("✓ Configuration error handling test passed");

    // Test 10: Configuration Validator
    println!("\n10. Testing configuration validator...");
    test_configuration_validator().await?;
    println!("✓ Configuration validator test passed");

    println!("\n🎉 All configuration tests passed!");
    Ok(())
}

async fn test_default_configuration() -> TestResult<()> {
    let config = TestConfig::default();

    // Validate default values
    assert!(
        config.max_parallel_tests > 0,
        "Default parallel tests should be > 0"
    );
    assert!(
        config.test_timeout.as_secs() > 0,
        "Default timeout should be > 0"
    );
    assert!(
        !config.cache_dir.as_os_str().is_empty(),
        "Default cache dir should not be empty"
    );
    assert!(
        !config.log_level.is_empty(),
        "Default log level should not be empty"
    );
    assert!(
        config.coverage_threshold >= 0.0 && config.coverage_threshold <= 1.0,
        "Default coverage threshold should be between 0.0 and 1.0"
    );

    // Test configuration validation
    validate_config(&config)
        .map_err(|e| TestError::assertion(format!("Default config validation failed: {}", e)))?;

    Ok(())
}

async fn test_predefined_configurations() -> TestResult<()> {
    // Test CI configuration
    let ci_cfg = ci_config();
    assert!(
        ci_cfg.reporting.generate_coverage,
        "CI config should generate coverage"
    );
    assert!(
        ci_cfg.reporting.formats.contains(&ReportFormat::Junit),
        "CI config should include JUnit format"
    );
    assert_eq!(
        ci_cfg.log_level, "debug",
        "CI config should use debug logging"
    );
    validate_config(&ci_cfg)
        .map_err(|e| TestError::assertion(format!("CI config validation failed: {}", e)))?;

    // Test dev configuration
    let dev_cfg = dev_config();
    assert!(
        !dev_cfg.reporting.generate_coverage,
        "Dev config should skip coverage for speed"
    );
    assert_eq!(
        dev_cfg.reporting.formats,
        vec![ReportFormat::Html],
        "Dev config should only use HTML format"
    );
    assert_eq!(
        dev_cfg.log_level, "info",
        "Dev config should use info logging"
    );
    validate_config(&dev_cfg)
        .map_err(|e| TestError::assertion(format!("Dev config validation failed: {}", e)))?;

    // Test minimal configuration
    let minimal_cfg = minimal_config();
    assert_eq!(
        minimal_cfg.max_parallel_tests, 1,
        "Minimal config should use 1 parallel test"
    );
    assert!(
        !minimal_cfg.reporting.generate_coverage,
        "Minimal config should skip coverage"
    );
    assert!(
        !minimal_cfg.crossval.enabled,
        "Minimal config should disable cross-validation"
    );
    assert!(
        !minimal_cfg.fixtures.auto_download,
        "Minimal config should disable auto-download"
    );
    assert_eq!(
        minimal_cfg.log_level, "warn",
        "Minimal config should use warn logging"
    );
    validate_config(&minimal_cfg)
        .map_err(|e| TestError::assertion(format!("Minimal config validation failed: {}", e)))?;

    Ok(())
}

async fn test_configuration_validation() -> TestResult<()> {
    // Test valid configuration
    let valid_config = TestConfig::default();
    assert!(
        validate_config(&valid_config).is_ok(),
        "Valid config should pass validation"
    );

    // Test invalid parallel tests (zero)
    let mut invalid_config = TestConfig::default();
    invalid_config.max_parallel_tests = 0;
    assert!(
        validate_config(&invalid_config).is_err(),
        "Zero parallel tests should fail validation"
    );

    // Test invalid coverage threshold
    let mut invalid_config = TestConfig::default();
    invalid_config.coverage_threshold = 1.5;
    assert!(
        validate_config(&invalid_config).is_err(),
        "Coverage threshold > 1.0 should fail validation"
    );

    // Test invalid log level
    let mut invalid_config = TestConfig::default();
    invalid_config.log_level = "invalid_level".to_string();
    assert!(
        validate_config(&invalid_config).is_err(),
        "Invalid log level should fail validation"
    );

    Ok(())
}

async fn test_configuration_merging() -> TestResult<()> {
    let base_config = TestConfig::default();
    let mut override_config = TestConfig::default();

    // Modify override config
    override_config.max_parallel_tests = 16;
    override_config.log_level = "trace".to_string();
    override_config.coverage_threshold = 0.95;

    // Merge configurations
    let merged = merge_configs(base_config.clone(), override_config.clone());

    // Verify override values took precedence
    assert_eq!(
        merged.max_parallel_tests, 16,
        "Parallel tests should be overridden"
    );
    assert_eq!(merged.log_level, "trace", "Log level should be overridden");
    assert_eq!(
        merged.coverage_threshold, 0.95,
        "Coverage threshold should be overridden"
    );

    // Verify base values are preserved where not overridden
    assert_eq!(
        merged.cache_dir, base_config.cache_dir,
        "Cache dir should be preserved from base"
    );
    assert_eq!(
        merged.test_timeout, base_config.test_timeout,
        "Timeout should be preserved from base"
    );

    Ok(())
}

async fn test_configuration_serialization() -> TestResult<()> {
    let temp_dir =
        tempdir().map_err(|e| TestError::setup(format!("Failed to create temp dir: {}", e)))?;
    let config_path = temp_dir.path().join("test_config.toml");

    // Test saving configuration to file
    let original_config = TestConfig::default();
    save_config_to_file(&original_config, &config_path)
        .map_err(|e| TestError::execution(format!("Failed to save config: {}", e)))?;

    // Verify file was created
    assert!(config_path.exists(), "Config file should be created");

    // Test loading configuration from file
    let loaded_config = load_config_from_file(&config_path)
        .map_err(|e| TestError::execution(format!("Failed to load config: {}", e)))?;

    // Verify loaded config matches original
    assert_eq!(
        loaded_config.max_parallel_tests, original_config.max_parallel_tests,
        "Parallel tests should match"
    );
    assert_eq!(
        loaded_config.test_timeout, original_config.test_timeout,
        "Timeout should match"
    );
    assert_eq!(
        loaded_config.log_level, original_config.log_level,
        "Log level should match"
    );

    // Test JSON serialization
    let json_str = serde_json::to_string_pretty(&original_config)
        .map_err(|e| TestError::execution(format!("Failed to serialize to JSON: {}", e)))?;

    let deserialized_config: TestConfig = serde_json::from_str(&json_str)
        .map_err(|e| TestError::execution(format!("Failed to deserialize from JSON: {}", e)))?;

    assert_eq!(
        deserialized_config.max_parallel_tests, original_config.max_parallel_tests,
        "JSON roundtrip should preserve parallel tests"
    );

    Ok(())
}

async fn test_environment_override() -> TestResult<()> {
    // Save original environment
    let original_env: HashMap<String, Option<String>> = [
        "BITNET_TEST_PARALLEL",
        "BITNET_TEST_TIMEOUT",
        "BITNET_TEST_LOG_LEVEL",
    ]
    .iter()
    .map(|&key| (key.to_string(), env::var(key).ok()))
    .collect();

    // Test parallel tests override
    env::set_var("BITNET_TEST_PARALLEL", "8");
    let mut config = TestConfig::default();
    load_config_from_env(&mut config)
        .map_err(|e| TestError::execution(format!("Failed to load env config: {}", e)))?;
    assert_eq!(
        config.max_parallel_tests, 8,
        "Environment override for parallel tests failed"
    );

    // Test timeout override
    env::set_var("BITNET_TEST_TIMEOUT", "120");
    let mut config = TestConfig::default();
    load_config_from_env(&mut config)
        .map_err(|e| TestError::execution(format!("Failed to load env config: {}", e)))?;
    assert_eq!(
        config.test_timeout,
        Duration::from_secs(120),
        "Environment override for timeout failed"
    );

    // Test log level override
    env::set_var("BITNET_TEST_LOG_LEVEL", "trace");
    let mut config = TestConfig::default();
    load_config_from_env(&mut config)
        .map_err(|e| TestError::execution(format!("Failed to load env config: {}", e)))?;
    assert_eq!(
        config.log_level, "trace",
        "Environment override for log level failed"
    );

    // Restore original environment
    for (key, value) in original_env {
        match value {
            Some(val) => env::set_var(&key, val),
            None => env::remove_var(&key),
        }
    }

    Ok(())
}

async fn test_feature_flag_combinations() -> TestResult<()> {
    // Test all coverage features enabled
    let mut coverage_config = TestConfig::default();
    coverage_config.reporting.generate_coverage = true;
    coverage_config.reporting.include_artifacts = true;
    coverage_config.coverage_threshold = 0.9;
    validate_config(&coverage_config)
        .map_err(|e| TestError::assertion(format!("Coverage config validation failed: {}", e)))?;

    // Test all performance features enabled
    let mut performance_config = TestConfig::default();
    performance_config.reporting.generate_performance = true;
    performance_config.max_parallel_tests = num_cpus::get();
    performance_config.test_timeout = Duration::from_secs(60);
    validate_config(&performance_config).map_err(|e| {
        TestError::assertion(format!("Performance config validation failed: {}", e))
    })?;

    // Test minimal features (all disabled)
    let mut minimal_features_config = TestConfig::default();
    minimal_features_config.reporting.generate_coverage = false;
    minimal_features_config.reporting.generate_performance = false;
    minimal_features_config.fixtures.auto_download = false;
    minimal_features_config.crossval.enabled = false;
    validate_config(&minimal_features_config).map_err(|e| {
        TestError::assertion(format!("Minimal features config validation failed: {}", e))
    })?;

    Ok(())
}

async fn test_platform_specific_configuration() -> TestResult<()> {
    let base_config = TestConfig::default();

    // Test macOS-specific configuration
    #[cfg(target_os = "macos")]
    {
        let mut macos_config = base_config.clone();
        macos_config.max_parallel_tests = std::cmp::min(num_cpus::get(), 8); // Limit on macOS
        validate_config(&macos_config)
            .map_err(|e| TestError::assertion(format!("macOS config validation failed: {}", e)))?;
    }

    // Test Linux-specific configuration
    #[cfg(target_os = "linux")]
    {
        let mut linux_config = base_config.clone();
        linux_config.max_parallel_tests = num_cpus::get() * 2; // Can handle more on Linux
        validate_config(&linux_config)
            .map_err(|e| TestError::assertion(format!("Linux config validation failed: {}", e)))?;
    }

    // Test architecture-specific configuration
    #[cfg(target_arch = "x86_64")]
    {
        let mut x86_64_config = base_config.clone();
        x86_64_config.fixtures.max_cache_size = 10 * 1024 * 1024 * 1024; // 10GB on x86_64
        validate_config(&x86_64_config)
            .map_err(|e| TestError::assertion(format!("x86_64 config validation failed: {}", e)))?;
    }

    Ok(())
}

async fn test_configuration_error_handling() -> TestResult<()> {
    let temp_dir =
        tempdir().map_err(|e| TestError::setup(format!("Failed to create temp dir: {}", e)))?;

    // Test malformed TOML configuration
    let malformed_config_path = temp_dir.path().join("malformed.toml");
    std::fs::write(&malformed_config_path, "invalid toml content [[[")
        .map_err(|e| TestError::setup(format!("Failed to write malformed config: {}", e)))?;

    let load_result = load_config_from_file(&malformed_config_path);
    assert!(load_result.is_err(), "Loading malformed TOML should fail");

    // Test missing configuration file
    let missing_config_path = temp_dir.path().join("missing.toml");
    let missing_result = load_config_from_file(&missing_config_path);
    assert!(
        missing_result.is_err(),
        "Loading missing config file should fail"
    );

    Ok(())
}

async fn test_configuration_validator() -> TestResult<()> {
    let temp_dir =
        tempdir().map_err(|e| TestError::setup(format!("Failed to create temp dir: {}", e)))?;
    let config_path = temp_dir.path().join("test_config.toml");

    let test_config = TestConfig::default();
    save_config_to_file(&test_config, &config_path)
        .map_err(|e| TestError::setup(format!("Failed to save test config: {}", e)))?;

    let validator = ConfigValidator::from_file(&config_path)
        .map_err(|e| TestError::execution(format!("Failed to create validator: {}", e)))?;

    let validation_result = validator.validate();
    assert!(
        validation_result.is_valid(),
        "Default config should be valid"
    );
    assert_eq!(
        validation_result.errors.len(),
        0,
        "Default config should have no errors"
    );

    Ok(())
}
