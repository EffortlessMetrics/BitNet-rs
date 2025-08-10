// Configuration testing module for BitNet.rs testing framework
//
// This module implements comprehensive tests for various configuration combinations,
// feature flag testing, platform-specific configuration tests, validation tests,
// and configuration migration/compatibility tests.
//
// Requirements: 5.3 - Integration testing framework configuration testing

use bitnet_tests::common::{
    config::{
        ci_config, dev_config, load_test_config, merge_configs, minimal_config, save_config_to_file,
        validate_config, ComparisonTolerance, CrossValidationConfig, CustomFixture, FixtureConfig,
        ReportFormat, ReportingConfig, TestConfig,
    },
    config_validator::{ConfigValidator, ValidationResult},
    errors::{TestError, TestResult},
    harness::{TestCase, TestHarness, TestSuite},
    results::{TestMetrics, TestResult as TestCaseResult, TestStatus},
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
        Self {
            temp_dirs: Vec::new(),
            original_env: HashMap::new(),
        }
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

    async fn setup(&self, _fixtures: &bitnet_tests::FixtureManager) -> TestResult<()> {
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
            config.coverage_threshold >= 0.0 && config.coverage_threshold <= 1.0,
            "Defaul