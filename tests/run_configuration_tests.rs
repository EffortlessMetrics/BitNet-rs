// Standalone test runner for configuration testing
// This allows us to test the configuration functionality independently

#[cfg(feature = "integration-tests")]
mod integration_tests {
    use bitnet_tests::units::BYTES_PER_MB;
    use bitnet_tests::{
        config::{
            ReportFormat, TestConfig, ci_config, dev_config, load_config_from_env,
            load_config_from_file, merge_configs, minimal_config, save_config_to_file,
            validate_config,
        },
        config_validator::ConfigValidator,
        errors::{TestError, TestOpResult},
    };
    use serde_json;
    use std::collections::HashMap;
    use std::env;
    use std::time::Duration;
    use tempfile::tempdir;

    pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
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
        test_validation().await?;
        println!("✓ Configuration validation test passed");

        // Test 4: Configuration Merging
        println!("\n4. Testing configuration merging...");
        test_config_merge().await?;
        println!("✓ Configuration merging test passed");

        // Test 5: Configuration Serialization
        println!("\n5. Testing configuration serialization...");
        test_serialization().await?;
        println!("✓ Configuration serialization test passed");

        // Test 6: Environment-based Configuration
        println!("\n6. Testing environment-based configuration...");
        test_environment_configuration().await?;
        println!("✓ Environment configuration test passed");

        // Test 7: File-based Configuration
        println!("\n7. Testing file-based configuration...");
        test_file_configuration().await?;
        println!("✓ File configuration test passed");

        // Test 8: Custom Configuration
        println!("\n8. Testing custom configuration...");
        test_custom_configuration().await?;
        println!("✓ Custom configuration test passed");

        println!("\n✅ All configuration tests passed!");
        Ok(())
    }

    async fn test_default_configuration() -> TestOpResult<()> {
        let config = TestConfig::default();

        // Verify default values
        assert_eq!(config.max_parallel_tests, 4);
        assert_eq!(config.timeout_seconds, 300);
        assert!(config.fail_fast);
        assert!(config.capture_stdout);
        assert!(config.capture_stderr);
        assert_eq!(config.retry_attempts, 0);
        assert_eq!(config.memory_limit_mb, None);
        assert_eq!(config.report_format, ReportFormat::Json);
        assert_eq!(config.output_dir, std::path::PathBuf::from("test-results"));

        println!("  Default configuration validated successfully");
        Ok(())
    }

    async fn test_predefined_configurations() -> TestOpResult<()> {
        // Test minimal config
        let minimal = minimal_config();
        assert_eq!(minimal.max_parallel_tests, 1);
        assert_eq!(minimal.timeout_seconds, 30);
        assert!(!minimal.fail_fast);
        println!("  Minimal configuration: {:?}", minimal);

        // Test dev config
        let dev = dev_config();
        assert_eq!(dev.max_parallel_tests, num_cpus::get());
        assert_eq!(dev.timeout_seconds, 600);
        assert!(dev.fail_fast);
        println!("  Dev configuration: {:?}", dev);

        // Test CI config
        let ci = ci_config();
        assert_eq!(ci.max_parallel_tests, 2);
        assert_eq!(ci.timeout_seconds, 1800);
        assert!(!ci.fail_fast);
        assert_eq!(ci.retry_attempts, 2);
        println!("  CI configuration: {:?}", ci);

        Ok(())
    }

    async fn test_environment_configuration() -> TestOpResult<()> {
        // Set environment variables
        env::set_var("BITNET_TEST_PARALLEL", "8");
        env::set_var("BITNET_TEST_TIMEOUT", "120");
        env::set_var("BITNET_TEST_FAIL_FAST", "false");
        env::set_var("BITNET_TEST_CAPTURE_STDOUT", "false");
        env::set_var("BITNET_TEST_RETRY", "3");
        env::set_var("BITNET_TEST_MEMORY_LIMIT_MB", "2048");

        // Load config from environment
        let config = load_config_from_env()?;

        // Verify values were loaded correctly
        assert_eq!(config.max_parallel_tests, 8);
        assert_eq!(config.timeout_seconds, 120);
        assert!(!config.fail_fast);
        assert!(!config.capture_stdout);
        assert_eq!(config.retry_attempts, 3);
        assert_eq!(config.memory_limit_mb, Some(2048));

        // Clean up environment variables
        env::remove_var("BITNET_TEST_PARALLEL");
        env::remove_var("BITNET_TEST_TIMEOUT");
        env::remove_var("BITNET_TEST_FAIL_FAST");
        env::remove_var("BITNET_TEST_CAPTURE_STDOUT");
        env::remove_var("BITNET_TEST_RETRY");
        env::remove_var("BITNET_TEST_MEMORY_LIMIT_MB");

        println!("  Environment configuration loaded successfully");
        Ok(())
    }

    async fn test_file_configuration() -> TestOpResult<()> {
        let temp_dir =
            tempdir().map_err(|e| TestError::io(format!("Failed to create temp dir: {}", e)))?;

        // Create a test configuration
        let config = TestConfig {
            max_parallel_tests: 16,
            timeout_seconds: 240,
            fail_fast: false,
            capture_stdout: false,
            capture_stderr: false,
            retry_attempts: 5,
            memory_limit_mb: Some(4096),
            report_format: ReportFormat::Html,
            output_dir: temp_dir.path().to_path_buf(),
            verbose: true,
            quiet: false,
            log_level: "debug".to_string(),
            test_threads: Some(8),
            test_filter: Some("integration".to_string()),
            exclude_filter: Some("slow".to_string()),
            shuffle_tests: true,
            seed: Some(42),
            coverage: true,
            profile: false,
            benchmark: false,
            environment: HashMap::from([("TEST_ENV".to_string(), "production".to_string())]),
            feature_flags: vec!["feature1".to_string(), "feature2".to_string()],
            custom_args: vec!["--custom-arg".to_string()],
        };

        // Save to file
        let config_path = temp_dir.path().join("test-config.toml");
        save_config_to_file(&config, &config_path)?;

        // Load from file
        let loaded_config = load_config_from_file(&config_path)?;

        // Verify loaded config matches original
        assert_eq!(loaded_config.max_parallel_tests, config.max_parallel_tests);
        assert_eq!(loaded_config.timeout_seconds, config.timeout_seconds);
        assert_eq!(loaded_config.fail_fast, config.fail_fast);
        assert_eq!(loaded_config.retry_attempts, config.retry_attempts);
        assert_eq!(loaded_config.memory_limit_mb, config.memory_limit_mb);
        assert_eq!(loaded_config.test_filter, config.test_filter);
        assert_eq!(loaded_config.seed, config.seed);

        println!("  File configuration saved and loaded successfully");
        Ok(())
    }

    async fn test_config_merge() -> TestOpResult<()> {
        let base = TestConfig {
            max_parallel_tests: 4,
            timeout_seconds: 300,
            fail_fast: true,
            test_filter: Some("unit".to_string()),
            ..Default::default()
        };

        let override_config = TestConfig {
            max_parallel_tests: 8,
            verbose: true,
            test_filter: Some("integration".to_string()),
            exclude_filter: Some("slow".to_string()),
            ..Default::default()
        };

        let merged = merge_configs(&base, &override_config);

        // Verify merge behavior - override values should take precedence
        assert_eq!(merged.max_parallel_tests, 8); // overridden
        assert_eq!(merged.timeout_seconds, 300); // from base
        assert!(merged.fail_fast); // from base
        assert!(merged.verbose); // overridden
        assert_eq!(merged.test_filter, Some("integration".to_string())); // overridden
        assert_eq!(merged.exclude_filter, Some("slow".to_string())); // from override

        println!("  Configuration merge completed successfully");
        Ok(())
    }

    async fn test_custom_configuration() -> TestOpResult<()> {
        // Create a custom configuration with specific settings
        let custom_config = TestConfig {
            max_parallel_tests: 32,
            timeout_seconds: 60,
            fail_fast: false,
            capture_stdout: true,
            capture_stderr: true,
            retry_attempts: 1,
            memory_limit_mb: Some(8192),
            report_format: ReportFormat::Junit,
            output_dir: std::path::PathBuf::from("/tmp/custom-test-results"),
            verbose: false,
            quiet: true,
            log_level: "error".to_string(),
            test_threads: Some(16),
            test_filter: Some("critical".to_string()),
            exclude_filter: Some("experimental".to_string()),
            shuffle_tests: false,
            seed: Some(12345),
            coverage: false,
            profile: true,
            benchmark: true,
            environment: HashMap::from([
                ("CUSTOM_VAR".to_string(), "custom_value".to_string()),
                ("TEST_MODE".to_string(), "performance".to_string()),
            ]),
            feature_flags: vec!["perf".to_string(), "advanced".to_string()],
            custom_args: vec!["--measure-memory".to_string(), "--track-allocations".to_string()],
        };

        // Validate custom configuration
        assert_eq!(custom_config.max_parallel_tests, 32);
        assert_eq!(custom_config.memory_limit_mb, Some(8192));
        assert_eq!(custom_config.report_format, ReportFormat::Junit);
        assert!(custom_config.quiet);
        assert!(custom_config.profile);
        assert!(custom_config.benchmark);
        assert_eq!(custom_config.environment.get("TEST_MODE"), Some(&"performance".to_string()));
        assert_eq!(custom_config.feature_flags.len(), 2);
        assert_eq!(custom_config.custom_args.len(), 2);

        println!("  Custom configuration validated successfully");
        Ok(())
    }

    async fn test_serialization() -> TestOpResult<()> {
        let config = TestConfig {
            max_parallel_tests: 12,
            timeout_seconds: 180,
            fail_fast: true,
            memory_limit_mb: Some(2048),
            report_format: ReportFormat::Json,
            test_filter: Some("smoke".to_string()),
            seed: Some(99),
            environment: HashMap::from([("KEY1".to_string(), "value1".to_string())]),
            feature_flags: vec!["flag1".to_string()],
            ..Default::default()
        };

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&config)
            .map_err(|e| TestError::serialization(format!("Failed to serialize config: {}", e)))?;

        // Deserialize from JSON
        let deserialized: TestConfig = serde_json::from_str(&json).map_err(|e| {
            TestError::serialization(format!("Failed to deserialize config: {}", e))
        })?;

        // Verify round-trip
        assert_eq!(deserialized.max_parallel_tests, config.max_parallel_tests);
        assert_eq!(deserialized.timeout_seconds, config.timeout_seconds);
        assert_eq!(deserialized.fail_fast, config.fail_fast);
        assert_eq!(deserialized.memory_limit_mb, config.memory_limit_mb);
        assert_eq!(deserialized.test_filter, config.test_filter);
        assert_eq!(deserialized.seed, config.seed);

        println!("  Configuration serialization round-trip successful");
        Ok(())
    }

    async fn test_validation() -> TestOpResult<()> {
        // Test valid configuration
        let valid_config = TestConfig {
            max_parallel_tests: 4,
            timeout_seconds: 300,
            memory_limit_mb: Some(1024 * BYTES_PER_MB as u64),
            ..Default::default()
        };

        assert!(validate_config(&valid_config), "Valid config should pass validation");

        // Test invalid configuration (timeout too short)
        let invalid_config = TestConfig { timeout_seconds: 0, ..Default::default() };

        assert!(!validate_config(&invalid_config), "Invalid config should fail validation");

        // Test with ConfigValidator
        let validator = ConfigValidator::new(&valid_config)
            .map_err(|e| TestError::execution(format!("Failed to create validator: {}", e)))?;

        let validation_result = validator.validate();
        assert!(validation_result.is_valid(), "Default config should be valid");
        assert_eq!(validation_result.errors.len(), 0, "Default config should have no errors");

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "integration-tests")]
    {
        integration_tests::run().await?
    }

    #[cfg(not(feature = "integration-tests"))]
    {
        println!("Configuration tests require the 'integration-tests' feature.");
        println!(
            "Run with: cargo run --bin run_configuration_tests --features integration-tests,fixtures"
        );
    }

    Ok(())
}
