// Configuration testing module for BitNet.rs testing framework
//
// This module implements comprehensive tests for various configuration combinations,
// feature flag testing, platform-specific configuration tests, validation tests,
// and configuration migration/compatibility tests.
//
// Requirements: 5.3 - Integration testing framework configuration testing

use bitnet_tests::{
    config::{
        ci_config, dev_config, load_config_from_env, load_config_from_file, load_test_config,
        merge_configs, minimal_config, save_config_to_file, validate_config, ComparisonTolerance,
        CrossValidationConfig, CustomFixture, FixtureConfig, ReportFormat, ReportingConfig,
        TestConfig,
    },
    config_validator::{ConfigValidator, ValidationResult},
    errors::{TestError, TestResult},
    harness::{TestCase, TestHarness, TestSuite},
    init_logging,
    results::{TestMetrics, TestResult as TestCaseResult, TestStatus},
    FixtureManager,
};
use serde_json;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::{tempdir, TempDir};

/// Test suite for configuration testing
pub struct ConfigurationTestSuite {
    temp_dirs: Vec<TempDir>,
    original_env: HashMap<String, Option<String>>,
}

impl ConfigurationTestSuite {
    pub fn new() -> Self {
        Self { temp_dirs: Vec::new(), original_env: HashMap::new() }
    }

    /// Save current environment variables for restoration
    fn save_env_var(&mut self, key: &str) {
        let current_value = env::var(key).ok();
        self.original_env.insert(key.to_string(), current_value);
    }

    /// Restore environment variables
    fn restore_env_vars(&self) {
        for (key, value) in &self.original_env {
            match value {
                Some(val) => env::set_var(key, val),
                None => env::remove_var(key),
            }
        }
    }

    /// Create a temporary directory for testing
    fn create_temp_dir(&mut self) -> TestResult<PathBuf> {
        let temp_dir = tempdir()
            .map_err(|e| TestError::setup(format!("Failed to create temp directory: {}", e)))?;
        let path = temp_dir.path().to_path_buf();
        self.temp_dirs.push(temp_dir);
        Ok(path)
    }
}

impl TestSuite for ConfigurationTestSuite {
    fn name(&self) -> &str {
        "Configuration Testing Suite"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(DefaultConfigurationTest),
            Box::new(PredefinedConfigurationsTest),
            Box::new(ConfigurationCombinationsTest),
            Box::new(EnvironmentOverrideTest),
            Box::new(ConfigurationValidationTest),
            Box::new(ConfigurationMergingTest),
            Box::new(ConfigurationSerializationTest),
            Box::new(FeatureFlagCombinationTest),
            Box::new(PlatformSpecificConfigurationTest),
            Box::new(ConfigurationMigrationTest),
            Box::new(ConfigurationCompatibilityTest),
            Box::new(ConfigurationErrorHandlingTest),
            Box::new(ConfigurationPerformanceTest),
            Box::new(ConfigurationValidatorTest),
        ]
    }
}

impl Drop for ConfigurationTestSuite {
    fn drop(&mut self) {
        self.restore_env_vars();
    }
}

/// Test default configuration values and behavior
struct DefaultConfigurationTest;

#[async_trait::async_trait]
impl TestCase for DefaultConfigurationTest {
    fn name(&self) -> &str {
        "Default Configuration Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test default configuration creation
        let config = TestConfig::default();

        // Validate default values
        assert!(config.max_parallel_tests > 0, "Default parallel tests should be > 0");
        assert!(config.test_timeout.as_secs() > 0, "Default timeout should be > 0");
        assert!(!config.cache_dir.as_os_str().is_empty(), "Default cache dir should not be empty");
        assert!(!config.log_level.is_empty(), "Default log level should not be empty");
        assert!(
            (0.0..=1.0).contains(&config.coverage_threshold),
            "Default coverage threshold should be between 0.0 and 1.0"
        );

        // Validate fixture configuration defaults
        assert!(config.fixtures.auto_download, "Auto download should be enabled by default");
        assert!(config.fixtures.max_cache_size > 0, "Max cache size should be > 0");
        assert!(config.fixtures.cleanup_interval.as_secs() > 0, "Cleanup interval should be > 0");
        assert!(config.fixtures.download_timeout.as_secs() > 0, "Download timeout should be > 0");

        // Validate cross-validation configuration defaults
        assert!(!config.crossval.enabled, "Cross-validation should be disabled by default");
        assert!(
            config.crossval.tolerance.min_token_accuracy > 0.0,
            "Min token accuracy should be > 0"
        );
        assert!(
            config.crossval.tolerance.max_probability_divergence >= 0.0,
            "Max probability divergence should be >= 0"
        );

        // Validate reporting configuration defaults
        assert!(!config.reporting.formats.is_empty(), "Should have at least one report format");
        assert!(
            config.reporting.generate_coverage,
            "Coverage generation should be enabled by default"
        );

        // Test configuration validation
        validate_config(&config).map_err(|e| {
            TestError::assertion(format!("Default config validation failed: {}", e))
        })?;

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test predefined configuration presets (CI, dev, minimal)
struct PredefinedConfigurationsTest;

#[async_trait::async_trait]
impl TestCase for PredefinedConfigurationsTest {
    fn name(&self) -> &str {
        "Predefined Configurations Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test CI configuration
        let ci_cfg = ci_config();
        assert!(ci_cfg.reporting.generate_coverage, "CI config should generate coverage");
        assert!(
            ci_cfg.reporting.formats.contains(&ReportFormat::Junit),
            "CI config should include JUnit format"
        );
        assert_eq!(ci_cfg.log_level, "debug", "CI config should use debug logging");
        validate_config(&ci_cfg)
            .map_err(|e| TestError::assertion(format!("CI config validation failed: {}", e)))?;

        // Test dev configuration
        let dev_cfg = dev_config();
        assert!(!dev_cfg.reporting.generate_coverage, "Dev config should skip coverage for speed");
        assert_eq!(
            dev_cfg.reporting.formats,
            vec![ReportFormat::Html],
            "Dev config should only use HTML format"
        );
        assert_eq!(dev_cfg.log_level, "info", "Dev config should use info logging");
        validate_config(&dev_cfg)
            .map_err(|e| TestError::assertion(format!("Dev config validation failed: {}", e)))?;

        // Test minimal configuration
        let minimal_cfg = minimal_config();
        assert_eq!(minimal_cfg.max_parallel_tests, 1, "Minimal config should use 1 parallel test");
        assert!(!minimal_cfg.reporting.generate_coverage, "Minimal config should skip coverage");
        assert!(!minimal_cfg.crossval.enabled, "Minimal config should disable cross-validation");
        assert!(!minimal_cfg.fixtures.auto_download, "Minimal config should disable auto-download");
        assert_eq!(minimal_cfg.log_level, "warn", "Minimal config should use warn logging");
        validate_config(&minimal_cfg).map_err(|e| {
            TestError::assertion(format!("Minimal config validation failed: {}", e))
        })?;

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test various configuration combinations
struct ConfigurationCombinationsTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationCombinationsTest {
    fn name(&self) -> &str {
        "Configuration Combinations Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test high-performance configuration
        let mut high_perf_config = TestConfig::default();
        high_perf_config.max_parallel_tests = num_cpus::get() * 2;
        high_perf_config.test_timeout = Duration::from_secs(30);
        high_perf_config.reporting.generate_coverage = false;
        high_perf_config.reporting.generate_performance = true;
        validate_config(&high_perf_config).map_err(|e| {
            TestError::assertion(format!("High-perf config validation failed: {}", e))
        })?;

        // Test memory-constrained configuration
        let mut memory_constrained_config = TestConfig::default();
        memory_constrained_config.max_parallel_tests = 1;
        memory_constrained_config.fixtures.max_cache_size = 100 * BYTES_PER_MB; // 100MB
        memory_constrained_config.reporting.include_artifacts = false;
        validate_config(&memory_constrained_config).map_err(|e| {
            TestError::assertion(format!("Memory-constrained config validation failed: {}", e))
        })?;

        // Test comprehensive testing configuration
        let mut comprehensive_config = TestConfig::default();
        comprehensive_config.coverage_threshold = 0.95;
        comprehensive_config.crossval.enabled = true;
        comprehensive_config.crossval.performance_comparison = true;
        comprehensive_config.crossval.accuracy_comparison = true;
        comprehensive_config.reporting.formats = vec![
            ReportFormat::Html,
            ReportFormat::Json,
            ReportFormat::Junit,
            ReportFormat::Markdown,
        ];
        // Note: We don't validate this one since it requires C++ binary path

        // Test network-disabled configuration
        let mut network_disabled_config = TestConfig::default();
        network_disabled_config.fixtures.auto_download = false;
        network_disabled_config.fixtures.base_url = None;
        network_disabled_config.reporting.upload_reports = false;
        validate_config(&network_disabled_config).map_err(|e| {
            TestError::assertion(format!("Network-disabled config validation failed: {}", e))
        })?;

        // Test strict tolerance configuration
        let mut strict_config = TestConfig::default();
        strict_config.crossval.tolerance.min_token_accuracy = 0.9999999; // Very strict
        strict_config.crossval.tolerance.max_probability_divergence = 1e-8;
        strict_config.crossval.tolerance.numerical_tolerance = 1e-8;
        // Note: We don't validate this one since it requires C++ binary path

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test environment variable overrides
struct EnvironmentOverrideTest;

#[async_trait::async_trait]
impl TestCase for EnvironmentOverrideTest {
    fn name(&self) -> &str {
        "Environment Override Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Save original environment
        let original_env: HashMap<String, Option<String>> = [
            "BITNET_TEST_PARALLEL",
            "BITNET_TEST_TIMEOUT",
            "BITNET_TEST_CACHE_DIR",
            "BITNET_TEST_LOG_LEVEL",
            "BITNET_TEST_COVERAGE_THRESHOLD",
            "BITNET_TEST_CROSSVAL_ENABLED",
            "BITNET_TEST_AUTO_DOWNLOAD",
            "BITNET_TEST_REPORT_FORMATS",
        ]
        .iter()
        .map(|&key| (key.to_string(), env::var(key).ok()))
        .collect();

        // Test parallel tests override
        env::set_var("BITNET_TEST_PARALLEL", "8");
        let mut config = TestConfig::default();
        load_config_from_env(&mut config)
            .map_err(|e| TestError::execution(format!("Failed to load env config: {}", e)))?;
        assert_eq!(config.max_parallel_tests, 8, "Environment override for parallel tests failed");

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
        assert_eq!(config.log_level, "trace", "Environment override for log level failed");

        // Test coverage threshold override
        env::set_var("BITNET_TEST_COVERAGE_THRESHOLD", "0.85");
        let mut config = TestConfig::default();
        load_config_from_env(&mut config)
            .map_err(|e| TestError::execution(format!("Failed to load env config: {}", e)))?;
        assert_eq!(
            config.coverage_threshold, 0.85,
            "Environment override for coverage threshold failed"
        );

        // Test boolean overrides
        env::set_var("BITNET_TEST_CROSSVAL_ENABLED", "true");
        env::set_var("BITNET_TEST_AUTO_DOWNLOAD", "false");
        let mut config = TestConfig::default();
        load_config_from_env(&mut config)
            .map_err(|e| TestError::execution(format!("Failed to load env config: {}", e)))?;
        assert!(config.crossval.enabled, "Environment override for crossval enabled failed");
        assert!(!config.fixtures.auto_download, "Environment override for auto download failed");

        // Test report formats override
        env::set_var("BITNET_TEST_REPORT_FORMATS", "json,junit");
        let mut config = TestConfig::default();
        load_config_from_env(&mut config)
            .map_err(|e| TestError::execution(format!("Failed to load env config: {}", e)))?;
        assert_eq!(
            config.reporting.formats,
            vec![ReportFormat::Json, ReportFormat::Junit],
            "Environment override for report formats failed"
        );

        // Restore original environment
        for (key, value) in original_env {
            match value {
                Some(val) => env::set_var(&key, val),
                None => env::remove_var(&key),
            }
        }

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration validation
struct ConfigurationValidationTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationValidationTest {
    fn name(&self) -> &str {
        "Configuration Validation Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test valid configuration
        let valid_config = TestConfig::default();
        assert!(validate_config(&valid_config).is_ok(), "Valid config should pass validation");

        // Test invalid parallel tests (zero)
        let mut invalid_config = TestConfig::default();
        invalid_config.max_parallel_tests = 0;
        assert!(
            validate_config(&invalid_config).is_err(),
            "Zero parallel tests should fail validation"
        );

        // Test invalid parallel tests (too high)
        let mut invalid_config = TestConfig::default();
        invalid_config.max_parallel_tests = 200;
        assert!(
            validate_config(&invalid_config).is_err(),
            "Too many parallel tests should fail validation"
        );

        // Test invalid timeout (zero)
        let mut invalid_config = TestConfig::default();
        invalid_config.test_timeout = Duration::from_secs(0);
        assert!(validate_config(&invalid_config).is_err(), "Zero timeout should fail validation");

        // Test invalid timeout (too high)
        let mut invalid_config = TestConfig::default();
        invalid_config.test_timeout = Duration::from_secs(7200); // 2 hours
        assert!(
            validate_config(&invalid_config).is_err(),
            "Excessive timeout should fail validation"
        );

        // Test invalid coverage threshold (negative)
        let mut invalid_config = TestConfig::default();
        invalid_config.coverage_threshold = -0.1;
        assert!(
            validate_config(&invalid_config).is_err(),
            "Negative coverage threshold should fail validation"
        );

        // Test invalid coverage threshold (> 1.0)
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

        // Test invalid custom fixture (empty name)
        let mut invalid_config = TestConfig::default();
        invalid_config.fixtures.custom_fixtures.push(CustomFixture {
            name: "".to_string(),
            url: "https://example.com/model.bin".to_string(),
            checksum: "abc123".to_string(),
            description: None,
        });
        assert!(
            validate_config(&invalid_config).is_err(),
            "Empty fixture name should fail validation"
        );

        // Test invalid custom fixture (invalid URL)
        let mut invalid_config = TestConfig::default();
        invalid_config.fixtures.custom_fixtures.push(CustomFixture {
            name: "test".to_string(),
            url: "invalid-url".to_string(),
            checksum: "abc123".to_string(),
            description: None,
        });
        assert!(
            validate_config(&invalid_config).is_err(),
            "Invalid fixture URL should fail validation"
        );

        // Test invalid custom fixture (invalid checksum)
        let mut invalid_config = TestConfig::default();
        invalid_config.fixtures.custom_fixtures.push(CustomFixture {
            name: "test".to_string(),
            url: "https://example.com/model.bin".to_string(),
            checksum: "invalid-checksum!".to_string(),
            description: None,
        });
        assert!(
            validate_config(&invalid_config).is_err(),
            "Invalid fixture checksum should fail validation"
        );

        // Test empty report formats
        let mut invalid_config = TestConfig::default();
        invalid_config.reporting.formats.clear();
        assert!(
            validate_config(&invalid_config).is_err(),
            "Empty report formats should fail validation"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration merging
struct ConfigurationMergingTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationMergingTest {
    fn name(&self) -> &str {
        "Configuration Merging Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let base_config = TestConfig::default();
        let mut override_config = TestConfig::default();

        // Modify override config
        override_config.max_parallel_tests = 16;
        override_config.log_level = "trace".to_string();
        override_config.coverage_threshold = 0.95;
        override_config.fixtures.auto_download = false;
        override_config.crossval.enabled = true;
        override_config.reporting.generate_coverage = false;

        // Merge configurations
        let merged = merge_configs(base_config.clone(), override_config.clone());

        // Verify override values took precedence
        assert_eq!(merged.max_parallel_tests, 16, "Parallel tests should be overridden");
        assert_eq!(merged.log_level, "trace", "Log level should be overridden");
        assert_eq!(merged.coverage_threshold, 0.95, "Coverage threshold should be overridden");
        assert!(!merged.fixtures.auto_download, "Auto download should be overridden");
        assert!(merged.crossval.enabled, "Cross-validation should be overridden");
        assert!(!merged.reporting.generate_coverage, "Coverage generation should be overridden");

        // Verify base values are preserved where not overridden
        assert_eq!(
            merged.cache_dir, base_config.cache_dir,
            "Cache dir should be preserved from base"
        );
        assert_eq!(
            merged.test_timeout, base_config.test_timeout,
            "Timeout should be preserved from base"
        );

        // Test merging with empty collections
        let mut base_with_fixtures = TestConfig::default();
        base_with_fixtures.fixtures.custom_fixtures.push(CustomFixture {
            name: "base-fixture".to_string(),
            url: "https://example.com/base.bin".to_string(),
            checksum: "abc123".to_string(),
            description: None,
        });

        let mut override_with_fixtures = TestConfig::default();
        override_with_fixtures.fixtures.custom_fixtures.push(CustomFixture {
            name: "override-fixture".to_string(),
            url: "https://example.com/override.bin".to_string(),
            checksum: "def456".to_string(),
            description: None,
        });

        let merged_fixtures = merge_configs(base_with_fixtures, override_with_fixtures);
        assert_eq!(
            merged_fixtures.fixtures.custom_fixtures.len(),
            1,
            "Override fixtures should replace base fixtures"
        );
        assert_eq!(
            merged_fixtures.fixtures.custom_fixtures[0].name, "override-fixture",
            "Override fixture should be present"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration serialization and deserialization
struct ConfigurationSerializationTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationSerializationTest {
    fn name(&self) -> &str {
        "Configuration Serialization Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Create a temporary directory for test files
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
        assert_eq!(loaded_config.cache_dir, original_config.cache_dir, "Cache dir should match");
        assert_eq!(loaded_config.log_level, original_config.log_level, "Log level should match");
        assert_eq!(
            loaded_config.coverage_threshold, original_config.coverage_threshold,
            "Coverage threshold should match"
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
        assert_eq!(
            deserialized_config.log_level, original_config.log_level,
            "JSON roundtrip should preserve log level"
        );

        // Test configuration with custom fixtures
        let mut config_with_fixtures = TestConfig::default();
        config_with_fixtures.fixtures.custom_fixtures.push(CustomFixture {
            name: "test-model".to_string(),
            url: "https://example.com/test.bin".to_string(),
            checksum: "abcdef123456".to_string(),
            description: Some("Test model for serialization".to_string()),
        });

        let fixtures_path = temp_dir.path().join("fixtures_config.toml");
        save_config_to_file(&config_with_fixtures, &fixtures_path)
            .map_err(|e| TestError::execution(format!("Failed to save fixtures config: {}", e)))?;

        let loaded_fixtures_config = load_config_from_file(&fixtures_path)
            .map_err(|e| TestError::execution(format!("Failed to load fixtures config: {}", e)))?;

        assert_eq!(
            loaded_fixtures_config.fixtures.custom_fixtures.len(),
            1,
            "Custom fixtures should be preserved"
        );
        assert_eq!(
            loaded_fixtures_config.fixtures.custom_fixtures[0].name, "test-model",
            "Fixture name should be preserved"
        );
        assert_eq!(
            loaded_fixtures_config.fixtures.custom_fixtures[0].url, "https://example.com/test.bin",
            "Fixture URL should be preserved"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}
/// Test feature flag combinations
struct FeatureFlagCombinationTest;

#[async_trait::async_trait]
impl TestCase for FeatureFlagCombinationTest {
    fn name(&self) -> &str {
        "Feature Flag Combination Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test all coverage features enabled
        let mut coverage_config = TestConfig::default();
        coverage_config.reporting.generate_coverage = true;
        coverage_config.reporting.include_artifacts = true;
        coverage_config.coverage_threshold = 0.9;
        validate_config(&coverage_config).map_err(|e| {
            TestError::assertion(format!("Coverage config validation failed: {}", e))
        })?;

        // Test all performance features enabled
        let mut performance_config = TestConfig::default();
        performance_config.reporting.generate_performance = true;
        performance_config.max_parallel_tests = num_cpus::get();
        performance_config.test_timeout = Duration::from_secs(60);
        validate_config(&performance_config).map_err(|e| {
            TestError::assertion(format!("Performance config validation failed: {}", e))
        })?;

        // Test all cross-validation features enabled
        let mut crossval_config = TestConfig::default();
        crossval_config.crossval.enabled = true;
        crossval_config.crossval.performance_comparison = true;
        crossval_config.crossval.accuracy_comparison = true;
        crossval_config.crossval.test_cases = vec![
            "basic_inference".to_string(),
            "tokenization".to_string(),
            "model_loading".to_string(),
            "streaming".to_string(),
        ];
        // Note: We don't validate this since it requires C++ binary path

        // Test all fixture features enabled
        let mut fixture_config = TestConfig::default();
        fixture_config.fixtures.auto_download = true;
        fixture_config.fixtures.max_cache_size = 5 * BYTES_PER_MB * 1024; // 5GB
        fixture_config.fixtures.cleanup_interval = Duration::from_secs(12 * 60 * 60); // 12 hours
        fixture_config.fixtures.download_timeout = Duration::from_secs(600); // 10 minutes
        fixture_config.fixtures.custom_fixtures.push(CustomFixture {
            name: "large-model".to_string(),
            url: "https://example.com/large.bin".to_string(),
            checksum: "fedcba987654321".to_string(),
            description: Some("Large test model".to_string()),
        });
        validate_config(&fixture_config).map_err(|e| {
            TestError::assertion(format!("Fixture config validation failed: {}", e))
        })?;

        // Test all reporting features enabled
        let mut reporting_config = TestConfig::default();
        reporting_config.reporting.formats = vec![
            ReportFormat::Html,
            ReportFormat::Json,
            ReportFormat::Junit,
            ReportFormat::Markdown,
            ReportFormat::Csv,
        ];
        reporting_config.reporting.include_artifacts = true;
        reporting_config.reporting.generate_coverage = true;
        reporting_config.reporting.generate_performance = true;
        reporting_config.reporting.upload_reports = false; // Keep false to avoid network calls
        validate_config(&reporting_config).map_err(|e| {
            TestError::assertion(format!("Reporting config validation failed: {}", e))
        })?;

        // Test minimal features (all disabled)
        let mut minimal_features_config = TestConfig::default();
        minimal_features_config.reporting.generate_coverage = false;
        minimal_features_config.reporting.generate_performance = false;
        minimal_features_config.reporting.include_artifacts = false;
        minimal_features_config.reporting.upload_reports = false;
        minimal_features_config.fixtures.auto_download = false;
        minimal_features_config.crossval.enabled = false;
        minimal_features_config.crossval.performance_comparison = false;
        minimal_features_config.crossval.accuracy_comparison = false;
        validate_config(&minimal_features_config).map_err(|e| {
            TestError::assertion(format!("Minimal features config validation failed: {}", e))
        })?;

        // Test conflicting feature combinations
        let mut conflicting_config = TestConfig::default();
        conflicting_config.max_parallel_tests = 1; // Sequential
        conflicting_config.reporting.generate_performance = true; // But want performance metrics
        conflicting_config.test_timeout = Duration::from_secs(10); // Very short timeout
        conflicting_config.coverage_threshold = 0.99; // Very high coverage requirement
        validate_config(&conflicting_config).map_err(|e| {
            TestError::assertion(format!("Conflicting config validation failed: {}", e))
        })?;

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test platform-specific configuration
struct PlatformSpecificConfigurationTest;

#[async_trait::async_trait]
impl TestCase for PlatformSpecificConfigurationTest {
    fn name(&self) -> &str {
        "Platform-Specific Configuration Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let base_config = TestConfig::default();

        // Test Windows-specific configuration
        #[cfg(target_os = "windows")]
        {
            let mut windows_config = base_config.clone();
            windows_config.cache_dir = PathBuf::from("C:\\temp\\bitnet-test-cache");
            windows_config.reporting.output_dir = PathBuf::from("C:\\temp\\bitnet-reports");
            // Don't validate paths that might not exist on the test system
        }

        // Test Unix-specific configuration
        #[cfg(unix)]
        {
            let mut unix_config = base_config.clone();
            unix_config.cache_dir = PathBuf::from("/tmp/bitnet-test-cache");
            unix_config.reporting.output_dir = PathBuf::from("/tmp/bitnet-reports");
            // Don't validate paths that might not exist on the test system
        }

        // Test macOS-specific configuration
        #[cfg(target_os = "macos")]
        {
            let mut macos_config = base_config.clone();
            macos_config.cache_dir = PathBuf::from("/Users/Shared/bitnet-test-cache");
            macos_config.max_parallel_tests = std::cmp::min(num_cpus::get(), 8); // Limit on macOS
            validate_config(&macos_config).map_err(|e| {
                TestError::assertion(format!("macOS config validation failed: {}", e))
            })?;
        }

        // Test Linux-specific configuration
        #[cfg(target_os = "linux")]
        {
            let mut linux_config = base_config.clone();
            linux_config.cache_dir = PathBuf::from("/var/tmp/bitnet-test-cache");
            linux_config.max_parallel_tests = num_cpus::get() * 2; // Can handle more on Linux
            validate_config(&linux_config).map_err(|e| {
                TestError::assertion(format!("Linux config validation failed: {}", e))
            })?;
        }

        // Test architecture-specific configuration
        #[cfg(target_arch = "x86_64")]
        {
            let mut x86_64_config = base_config.clone();
            x86_64_config.fixtures.max_cache_size = 10 * BYTES_PER_MB * 1024; // 10GB on x86_64
            validate_config(&x86_64_config).map_err(|e| {
                TestError::assertion(format!("x86_64 config validation failed: {}", e))
            })?;
        }

        #[cfg(target_arch = "aarch64")]
        {
            let mut aarch64_config = base_config.clone();
            aarch64_config.fixtures.max_cache_size = 5 * BYTES_PER_MB * 1024; // 5GB on ARM64
            aarch64_config.max_parallel_tests = std::cmp::min(num_cpus::get(), 4); // Conservative on ARM
            validate_config(&aarch64_config).map_err(|e| {
                TestError::assertion(format!("aarch64 config validation failed: {}", e))
            })?;
        }

        // Test CI platform detection
        let is_ci = env::var("CI").is_ok()
            || env::var("GITHUB_ACTIONS").is_ok()
            || env::var("TRAVIS").is_ok()
            || env::var("JENKINS_URL").is_ok();

        if is_ci {
            let mut ci_optimized_config = base_config.clone();
            ci_optimized_config.max_parallel_tests = std::cmp::min(num_cpus::get(), 4); // Conservative in CI
            ci_optimized_config.test_timeout = Duration::from_secs(300); // Longer timeout in CI
            ci_optimized_config.reporting.generate_coverage = true;
            ci_optimized_config.reporting.formats.push(ReportFormat::Junit);
            validate_config(&ci_optimized_config).map_err(|e| {
                TestError::assertion(format!("CI-optimized config validation failed: {}", e))
            })?;
        }

        // Test container environment detection
        let is_container =
            std::path::Path::new("/.dockerenv").exists() || env::var("container").is_ok();

        if is_container {
            let mut container_config = base_config.clone();
            container_config.max_parallel_tests = std::cmp::min(num_cpus::get(), 2); // Very conservative in containers
            container_config.fixtures.max_cache_size = BYTES_PER_MB * 1024; // 1GB in containers
            container_config.test_timeout = Duration::from_secs(600); // Longer timeout in containers
            validate_config(&container_config).map_err(|e| {
                TestError::assertion(format!("Container config validation failed: {}", e))
            })?;
        }

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration migration and compatibility
struct ConfigurationMigrationTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationMigrationTest {
    fn name(&self) -> &str {
        "Configuration Migration Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test migration from older configuration format
        // Simulate an older config with missing fields
        let legacy_config_json = r#"{
            "max_parallel_tests": 4,
            "test_timeout": {"secs": 300, "nanos": 0},
            "cache_dir": "tests/cache",
            "log_level": "info",
            "coverage_threshold": 0.9
        }"#;

        // Parse as generic JSON first
        let legacy_value: serde_json::Value = serde_json::from_str(legacy_config_json)
            .map_err(|e| TestError::execution(format!("Failed to parse legacy JSON: {}", e)))?;

        // Create a new config with defaults and override with legacy values
        let mut migrated_config = TestConfig::default();
        if let Some(parallel) = legacy_value.get("max_parallel_tests").and_then(|v| v.as_u64()) {
            migrated_config.max_parallel_tests = parallel as usize;
        }
        if let Some(timeout_obj) = legacy_value.get("test_timeout") {
            if let Some(secs) = timeout_obj.get("secs").and_then(|v| v.as_u64()) {
                migrated_config.test_timeout = Duration::from_secs(secs);
            }
        }
        if let Some(cache_dir) = legacy_value.get("cache_dir").and_then(|v| v.as_str()) {
            migrated_config.cache_dir = PathBuf::from(cache_dir);
        }
        if let Some(log_level) = legacy_value.get("log_level").and_then(|v| v.as_str()) {
            migrated_config.log_level = log_level.to_string();
        }
        if let Some(threshold) = legacy_value.get("coverage_threshold").and_then(|v| v.as_f64()) {
            migrated_config.coverage_threshold = threshold;
        }

        // Verify migration worked
        assert_eq!(
            migrated_config.max_parallel_tests, 4,
            "Legacy parallel tests should be migrated"
        );
        assert_eq!(
            migrated_config.test_timeout,
            Duration::from_secs(300),
            "Legacy timeout should be migrated"
        );
        assert_eq!(
            migrated_config.cache_dir,
            PathBuf::from("tests/cache"),
            "Legacy cache dir should be migrated"
        );
        assert_eq!(migrated_config.log_level, "info", "Legacy log level should be migrated");
        assert_eq!(
            migrated_config.coverage_threshold, 0.9,
            "Legacy coverage threshold should be migrated"
        );

        // Verify new fields have defaults
        assert!(migrated_config.fixtures.auto_download, "New fixture fields should have defaults");
        assert!(!migrated_config.crossval.enabled, "New crossval fields should have defaults");
        assert!(
            !migrated_config.reporting.formats.is_empty(),
            "New reporting fields should have defaults"
        );

        validate_config(&migrated_config).map_err(|e| {
            TestError::assertion(format!("Migrated config validation failed: {}", e))
        })?;

        // Test forward compatibility - newer config with unknown fields
        let future_config_json = r#"{
            "max_parallel_tests": 8,
            "test_timeout": {"secs": 600, "nanos": 0},
            "cache_dir": "tests/cache",
            "log_level": "debug",
            "coverage_threshold": 0.95,
            "fixtures": {
                "auto_download": true,
                "max_cache_size": 5368709120,
                "cleanup_interval": {"secs": 86400, "nanos": 0},
                "download_timeout": {"secs": 300, "nanos": 0},
                "base_url": null,
                "custom_fixtures": []
            },
            "crossval": {
                "enabled": false,
                "tolerance": {
                    "min_token_accuracy": 0.999999,
                    "max_probability_divergence": 0.000001,
                    "max_performance_regression": 0.1,
                    "numerical_tolerance": 0.000001
                },
                "cpp_binary_path": null,
                "test_cases": ["basic_inference", "tokenization", "model_loading"],
                "performance_comparison": true,
                "accuracy_comparison": true
            },
            "reporting": {
                "output_dir": "test-reports",
                "formats": ["Html", "Json"],
                "include_artifacts": true,
                "generate_coverage": true,
                "generate_performance": true,
                "upload_reports": false
            },
            "future_field": "this should be ignored",
            "another_future_field": {
                "nested": "value"
            }
        }"#;

        // Parse as TestConfig - unknown fields should be ignored
        let future_config: TestConfig = serde_json::from_str(future_config_json)
            .map_err(|e| TestError::execution(format!("Failed to parse future config: {}", e)))?;

        // Verify known fields are parsed correctly
        assert_eq!(
            future_config.max_parallel_tests, 8,
            "Future config parallel tests should be parsed"
        );
        assert_eq!(
            future_config.test_timeout,
            Duration::from_secs(600),
            "Future config timeout should be parsed"
        );
        assert_eq!(future_config.log_level, "debug", "Future config log level should be parsed");

        validate_config(&future_config)
            .map_err(|e| TestError::assertion(format!("Future config validation failed: {}", e)))?;

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration compatibility across versions
struct ConfigurationCompatibilityTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationCompatibilityTest {
    fn name(&self) -> &str {
        "Configuration Compatibility Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test v0.1.0 configuration format
        let v0_1_0_config = TestConfig {
            max_parallel_tests: 4,
            test_timeout: Duration::from_secs(300),
            cache_dir: PathBuf::from("tests/cache"),
            log_level: "info".to_string(),
            coverage_threshold: 0.9,
            fixtures: FixtureConfig {
                auto_download: true,
                max_cache_size: 10 * BYTES_PER_MB * 1024,
                cleanup_interval: Duration::from_secs(24 * 60 * 60),
                download_timeout: Duration::from_secs(300),
                base_url: None,
                custom_fixtures: vec![],
            },
            crossval: CrossValidationConfig {
                enabled: false,
                tolerance: ComparisonTolerance {
                    min_token_accuracy: 0.999999,
                    max_probability_divergence: 1e-6,
                    max_performance_regression: 0.1,
                    numerical_tolerance: 1e-6,
                },
                cpp_binary_path: None,
                test_cases: vec!["basic_inference".to_string()],
                performance_comparison: true,
                accuracy_comparison: true,
            },
            reporting: ReportingConfig {
                output_dir: PathBuf::from("test-reports"),
                formats: vec![ReportFormat::Html, ReportFormat::Json],
                include_artifacts: true,
                generate_coverage: true,
                generate_performance: true,
                upload_reports: false,
            },
        };

        validate_config(&v0_1_0_config)
            .map_err(|e| TestError::assertion(format!("v0.1.0 config validation failed: {}", e)))?;

        // Test serialization/deserialization compatibility
        let serialized = serde_json::to_string(&v0_1_0_config).map_err(|e| {
            TestError::execution(format!("Failed to serialize v0.1.0 config: {}", e))
        })?;

        let deserialized: TestConfig = serde_json::from_str(&serialized).map_err(|e| {
            TestError::execution(format!("Failed to deserialize v0.1.0 config: {}", e))
        })?;

        assert_eq!(
            deserialized.max_parallel_tests, v0_1_0_config.max_parallel_tests,
            "Serialization should preserve parallel tests"
        );
        assert_eq!(
            deserialized.log_level, v0_1_0_config.log_level,
            "Serialization should preserve log level"
        );
        assert_eq!(
            deserialized.fixtures.auto_download, v0_1_0_config.fixtures.auto_download,
            "Serialization should preserve fixture settings"
        );

        // Test configuration with different enum variants
        let mut enum_test_config = TestConfig::default();
        enum_test_config.reporting.formats = vec![
            ReportFormat::Html,
            ReportFormat::Json,
            ReportFormat::Junit,
            ReportFormat::Markdown,
            ReportFormat::Csv,
        ];

        let enum_serialized = serde_json::to_string(&enum_test_config)
            .map_err(|e| TestError::execution(format!("Failed to serialize enum config: {}", e)))?;

        let enum_deserialized: TestConfig =
            serde_json::from_str(&enum_serialized).map_err(|e| {
                TestError::execution(format!("Failed to deserialize enum config: {}", e))
            })?;

        assert_eq!(
            enum_deserialized.reporting.formats.len(),
            5,
            "All report formats should be preserved"
        );
        assert!(
            enum_deserialized.reporting.formats.contains(&ReportFormat::Html),
            "HTML format should be preserved"
        );
        assert!(
            enum_deserialized.reporting.formats.contains(&ReportFormat::Csv),
            "CSV format should be preserved"
        );

        // Test configuration with optional fields
        let mut optional_test_config = TestConfig::default();
        optional_test_config.fixtures.base_url = Some("https://example.com/fixtures".to_string());
        optional_test_config.crossval.cpp_binary_path =
            Some(PathBuf::from("/usr/local/bin/bitnet-cpp"));

        let optional_serialized = serde_json::to_string(&optional_test_config).map_err(|e| {
            TestError::execution(format!("Failed to serialize optional config: {}", e))
        })?;

        let optional_deserialized: TestConfig = serde_json::from_str(&optional_serialized)
            .map_err(|e| {
                TestError::execution(format!("Failed to deserialize optional config: {}", e))
            })?;

        assert_eq!(
            optional_deserialized.fixtures.base_url,
            Some("https://example.com/fixtures".to_string()),
            "Optional base URL should be preserved"
        );
        assert_eq!(
            optional_deserialized.crossval.cpp_binary_path,
            Some(PathBuf::from("/usr/local/bin/bitnet-cpp")),
            "Optional C++ path should be preserved"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}
/// Test configuration error handling
struct ConfigurationErrorHandlingTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationErrorHandlingTest {
    fn name(&self) -> &str {
        "Configuration Error Handling Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test malformed TOML configuration
        let temp_dir =
            tempdir().map_err(|e| TestError::setup(format!("Failed to create temp dir: {}", e)))?;
        let malformed_config_path = temp_dir.path().join("malformed.toml");

        std::fs::write(&malformed_config_path, "invalid toml content [[[")
            .map_err(|e| TestError::setup(format!("Failed to write malformed config: {}", e)))?;

        let load_result = load_config_from_file(&malformed_config_path);
        assert!(load_result.is_err(), "Loading malformed TOML should fail");

        // Test missing configuration file
        let missing_config_path = temp_dir.path().join("missing.toml");
        let missing_result = load_config_from_file(&missing_config_path);
        assert!(missing_result.is_err(), "Loading missing config file should fail");

        // Test invalid environment variable values
        let original_env: HashMap<String, Option<String>> =
            ["BITNET_TEST_PARALLEL", "BITNET_TEST_TIMEOUT", "BITNET_TEST_COVERAGE_THRESHOLD"]
                .iter()
                .map(|&key| (key.to_string(), env::var(key).ok()))
                .collect();

        // Test invalid parallel tests value
        env::set_var("BITNET_TEST_PARALLEL", "invalid");
        let mut config = TestConfig::default();
        let env_result = load_config_from_env(&mut config);
        assert!(env_result.is_err(), "Invalid parallel tests env var should fail");

        // Test invalid timeout value
        env::set_var("BITNET_TEST_TIMEOUT", "not_a_number");
        let mut config = TestConfig::default();
        let env_result = load_config_from_env(&mut config);
        assert!(env_result.is_err(), "Invalid timeout env var should fail");

        // Test invalid coverage threshold value
        env::set_var("BITNET_TEST_COVERAGE_THRESHOLD", "invalid_float");
        let mut config = TestConfig::default();
        let env_result = load_config_from_env(&mut config);
        assert!(env_result.is_err(), "Invalid coverage threshold env var should fail");

        // Test invalid report formats
        env::set_var("BITNET_TEST_REPORT_FORMATS", "invalid_format,html");
        let mut config = TestConfig::default();
        let env_result = load_config_from_env(&mut config);
        assert!(env_result.is_err(), "Invalid report format should fail");

        // Restore original environment
        for (key, value) in original_env {
            match value {
                Some(val) => env::set_var(&key, val),
                None => env::remove_var(&key),
            }
        }

        // Test configuration with invalid paths
        let mut invalid_path_config = TestConfig::default();
        invalid_path_config.cache_dir =
            PathBuf::from("/invalid/path/that/does/not/exist/and/cannot/be/created");
        // Note: We don't validate this since it would require filesystem operations

        // Test configuration serialization errors
        // This is harder to test since serde_json is quite robust, but we can test with extreme values
        let mut extreme_config = TestConfig::default();
        extreme_config.max_parallel_tests = usize::MAX;
        extreme_config.test_timeout = Duration::from_secs(u64::MAX);

        // This should still serialize, but might cause issues in practice
        let serialize_result = serde_json::to_string(&extreme_config);
        assert!(serialize_result.is_ok(), "Even extreme values should serialize");

        // Test configuration with circular references in paths
        // This is more of a logical error than a serialization error
        let mut circular_config = TestConfig::default();
        circular_config.cache_dir = PathBuf::from("./cache");
        circular_config.reporting.output_dir = PathBuf::from("./cache/reports"); // Inside cache dir
        validate_config(&circular_config).map_err(|e| {
            TestError::assertion(format!("Circular path config validation failed: {}", e))
        })?;

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration performance
struct ConfigurationPerformanceTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationPerformanceTest {
    fn name(&self) -> &str {
        "Configuration Performance Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test configuration loading performance
        let load_start = std::time::Instant::now();
        for _ in 0..1000 {
            let _config = TestConfig::default();
        }
        let load_duration = load_start.elapsed();
        assert!(
            load_duration.as_millis() < 100,
            "Creating 1000 default configs should take < 100ms"
        );

        // Test configuration validation performance
        let config = TestConfig::default();
        let validate_start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = validate_config(&config);
        }
        let validate_duration = validate_start.elapsed();
        assert!(validate_duration.as_millis() < 50, "Validating 1000 configs should take < 50ms");

        // Test configuration serialization performance
        let serialize_start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = serde_json::to_string(&config);
        }
        let serialize_duration = serialize_start.elapsed();
        assert!(
            serialize_duration.as_millis() < 100,
            "Serializing 100 configs should take < 100ms"
        );

        // Test configuration deserialization performance
        let json_str = serde_json::to_string(&config).map_err(|e| {
            TestError::execution(format!("Failed to serialize config for perf test: {}", e))
        })?;

        let deserialize_start = std::time::Instant::now();
        for _ in 0..100 {
            let _: TestConfig = serde_json::from_str(&json_str).map_err(|e| {
                TestError::execution(format!("Failed to deserialize config in perf test: {}", e))
            })?;
        }
        let deserialize_duration = deserialize_start.elapsed();
        assert!(
            deserialize_duration.as_millis() < 100,
            "Deserializing 100 configs should take < 100ms"
        );

        // Test configuration merging performance
        let base_config = TestConfig::default();
        let override_config = ci_config();

        let merge_start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = merge_configs(base_config.clone(), override_config.clone());
        }
        let merge_duration = merge_start.elapsed();
        assert!(merge_duration.as_millis() < 50, "Merging 1000 configs should take < 50ms");

        // Test large configuration handling
        let mut large_config = TestConfig::default();
        for i in 0..100 {
            large_config.fixtures.custom_fixtures.push(CustomFixture {
                name: format!("fixture_{}", i),
                url: format!("https://example.com/fixture_{}.bin", i),
                checksum: format!("{:064x}", i),
                description: Some(format!("Test fixture number {}", i)),
            });
        }

        let large_validate_start = std::time::Instant::now();
        validate_config(&large_config)
            .map_err(|e| TestError::assertion(format!("Large config validation failed: {}", e)))?;
        let large_validate_duration = large_validate_start.elapsed();
        assert!(
            large_validate_duration.as_millis() < 100,
            "Validating large config should take < 100ms"
        );

        let large_serialize_start = std::time::Instant::now();
        let large_json = serde_json::to_string(&large_config).map_err(|e| {
            TestError::execution(format!("Failed to serialize large config: {}", e))
        })?;
        let large_serialize_duration = large_serialize_start.elapsed();
        assert!(
            large_serialize_duration.as_millis() < 50,
            "Serializing large config should take < 50ms"
        );

        let large_deserialize_start = std::time::Instant::now();
        let _: TestConfig = serde_json::from_str(&large_json).map_err(|e| {
            TestError::execution(format!("Failed to deserialize large config: {}", e))
        })?;
        let large_deserialize_duration = large_deserialize_start.elapsed();
        assert!(
            large_deserialize_duration.as_millis() < 50,
            "Deserializing large config should take < 50ms"
        );

        // Add performance metrics
        metrics.add_metric("config_load_time_ms", load_duration.as_millis() as f64);
        metrics.add_metric("config_validate_time_ms", validate_duration.as_millis() as f64);
        metrics.add_metric("config_serialize_time_ms", serialize_duration.as_millis() as f64);
        metrics.add_metric("config_deserialize_time_ms", deserialize_duration.as_millis() as f64);
        metrics.add_metric("config_merge_time_ms", merge_duration.as_millis() as f64);
        metrics.add_metric(
            "large_config_validate_time_ms",
            large_validate_duration.as_millis() as f64,
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration validator
struct ConfigurationValidatorTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationValidatorTest {
    fn name(&self) -> &str {
        "Configuration Validator Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test validator with default configuration
        let temp_dir =
            tempdir().map_err(|e| TestError::setup(format!("Failed to create temp dir: {}", e)))?;
        let config_path = temp_dir.path().join("test_config.toml");

        let test_config = TestConfig::default();
        save_config_to_file(&test_config, &config_path)
            .map_err(|e| TestError::setup(format!("Failed to save test config: {}", e)))?;

        let validator = ConfigValidator::from_file(&config_path)
            .map_err(|e| TestError::execution(format!("Failed to create validator: {}", e)))?;

        let validation_result = validator.validate();
        assert!(validation_result.is_valid(), "Default config should be valid");
        assert_eq!(validation_result.errors.len(), 0, "Default config should have no errors");

        // Test validator with problematic configuration
        let mut problematic_config = TestConfig::default();
        problematic_config.max_parallel_tests = num_cpus::get() * 4; // Too many
        problematic_config.test_timeout = Duration::from_secs(10); // Too short
        problematic_config.fixtures.custom_fixtures.push(CustomFixture {
            name: "test".to_string(),
            url: "http://example.com/insecure.bin".to_string(), // HTTP instead of HTTPS
            checksum: "abc".to_string(),                        // Too short
            description: None,
        });

        let problematic_path = temp_dir.path().join("problematic_config.toml");
        save_config_to_file(&problematic_config, &problematic_path)
            .map_err(|e| TestError::setup(format!("Failed to save problematic config: {}", e)))?;

        let problematic_validator = ConfigValidator::from_file(&problematic_path).map_err(|e| {
            TestError::execution(format!("Failed to create problematic validator: {}", e))
        })?;

        let problematic_result = problematic_validator.validate();
        assert!(
            problematic_result.is_valid(),
            "Problematic config should still be valid (warnings only)"
        );
        assert!(problematic_result.has_warnings(), "Problematic config should have warnings");
        assert!(problematic_result.warnings.len() > 0, "Should have at least one warning");

        // Check for specific warnings
        let warning_messages: Vec<String> =
            problematic_result.warnings.iter().map(|w| w.message.clone()).collect();

        let has_parallel_warning =
            warning_messages.iter().any(|msg| msg.contains("much higher than CPU cores"));
        assert!(has_parallel_warning, "Should warn about too many parallel tests");

        let has_timeout_warning =
            warning_messages.iter().any(|msg| msg.contains("Very short timeout"));
        assert!(has_timeout_warning, "Should warn about short timeout");

        let has_http_warning = warning_messages.iter().any(|msg| msg.contains("insecure HTTP URL"));
        assert!(has_http_warning, "Should warn about HTTP URL");

        let has_checksum_warning =
            warning_messages.iter().any(|msg| msg.contains("too short for security"));
        assert!(has_checksum_warning, "Should warn about short checksum");

        // Test validator summary
        let summary = problematic_result.summary();
        assert!(summary.contains("warnings"), "Summary should mention warnings");
        assert!(summary.contains("0 errors"), "Summary should show 0 errors");

        // Test validator with invalid configuration
        let mut invalid_config = TestConfig::default();
        invalid_config.max_parallel_tests = 0; // Invalid
        invalid_config.coverage_threshold = 1.5; // Invalid

        let invalid_path = temp_dir.path().join("invalid_config.toml");
        save_config_to_file(&invalid_config, &invalid_path)
            .map_err(|e| TestError::setup(format!("Failed to save invalid config: {}", e)))?;

        let invalid_validator = ConfigValidator::from_file(&invalid_path).map_err(|e| {
            TestError::execution(format!("Failed to create invalid validator: {}", e))
        })?;

        let invalid_result = invalid_validator.validate();
        assert!(!invalid_result.is_valid(), "Invalid config should not be valid");
        assert!(invalid_result.errors.len() > 0, "Invalid config should have errors");

        // Check for specific errors
        let error_messages: Vec<String> =
            invalid_result.errors.iter().map(|e| e.message.clone()).collect();

        let has_parallel_error = error_messages
            .iter()
            .any(|msg| msg.contains("max_parallel_tests must be greater than 0"));
        assert!(has_parallel_error, "Should error on zero parallel tests");

        let has_threshold_error = error_messages
            .iter()
            .any(|msg| msg.contains("coverage_threshold must be between 0.0 and 1.0"));
        assert!(has_threshold_error, "Should error on invalid coverage threshold");

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_tests::{init_logging, TestHarness};

    #[tokio::test]
    async fn test_configuration_test_suite() {
        let config = TestConfig::default();
        init_logging(&config).unwrap();

        let mut harness = TestHarness::new(config).await.unwrap();
        let suite = ConfigurationTestSuite::new();

        let result = harness.run_test_suite(suite).await.unwrap();

        // All tests should pass
        assert!(result.summary.failed == 0, "All configuration tests should pass");
        assert!(result.summary.passed > 0, "Should have passing tests");
    }

    #[tokio::test]
    async fn test_individual_configuration_tests() {
        let config = TestConfig::default();
        let fixtures = FixtureManager::new(&config.fixtures).await.unwrap();

        // Test default configuration test
        let default_test = DefaultConfigurationTest;
        assert!(default_test.setup(&fixtures).await.is_ok());
        assert!(default_test.execute().await.is_ok());
        assert!(default_test.cleanup().await.is_ok());

        // Test predefined configurations test
        let predefined_test = PredefinedConfigurationsTest;
        assert!(predefined_test.setup(&fixtures).await.is_ok());
        assert!(predefined_test.execute().await.is_ok());
        assert!(predefined_test.cleanup().await.is_ok());

        // Test configuration validation test
        let validation_test = ConfigurationValidationTest;
        assert!(validation_test.setup(&fixtures).await.is_ok());
        assert!(validation_test.execute().await.is_ok());
        assert!(validation_test.cleanup().await.is_ok());
    }
}
