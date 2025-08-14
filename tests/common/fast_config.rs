use super::config::{ReportFormat, TestConfig};
use std::time::Duration;

/// Fast configuration presets for different testing scenarios
/// This module provides optimized configurations for speed-critical testing

/// Create a configuration optimized for fast execution
pub fn fast_config() -> TestConfig {
    let mut config = TestConfig::default();

    // Reduce parallel tests to avoid resource contention
    config.max_parallel_tests = 2;

    // Shorter timeout for faster feedback
    config.test_timeout = Duration::from_secs(30);

    // Minimal logging for speed
    config.log_level = "error".to_string();

    // Disable coverage for speed
    config.reporting.generate_coverage = false;
    config.reporting.generate_performance = false;

    // Only JSON reports for minimal overhead
    config.reporting.formats = vec![ReportFormat::Json];

    // Disable cross-validation for speed
    config.crossval.enabled = false;

    // Disable auto-download to avoid network delays
    config.fixtures.auto_download = false;

    // Smaller cache size
    config.fixtures.max_cache_size = 100 * 1024 * 1024; // 100 MB

    config
}

/// Create a configuration for smoke tests (minimal validation)
pub fn smoke_test_config() -> TestConfig {
    let mut config = fast_config();

    // Even more aggressive settings for smoke tests
    config.max_parallel_tests = 1;
    config.test_timeout = Duration::from_secs(10);
    config.log_level = "warn".to_string();

    // No reporting for smoke tests
    config.reporting.include_artifacts = false;

    config
}

/// Create a configuration for unit tests only
pub fn unit_test_config() -> TestConfig {
    let mut config = fast_config();

    // Optimize for unit tests
    config.max_parallel_tests = 4;
    config.test_timeout = Duration::from_secs(60);

    // Enable basic reporting
    config.reporting.formats = vec![ReportFormat::Json, ReportFormat::Html];

    config
}

/// Create a configuration for quick integration tests
pub fn quick_integration_config() -> TestConfig {
    let mut config = TestConfig::default();

    // Balanced settings for integration tests
    config.max_parallel_tests = 2;
    config.test_timeout = Duration::from_secs(120);
    config.log_level = "info".to_string();

    // Enable basic coverage
    config.reporting.generate_coverage = true;
    config.reporting.generate_performance = false;

    // Standard reporting
    config.reporting.formats = vec![ReportFormat::Html, ReportFormat::Json];

    // Disable cross-validation for speed
    config.crossval.enabled = false;

    config
}

/// Get configuration based on environment variable or default to fast
pub fn get_fast_config_variant() -> TestConfig {
    match std::env::var("BITNET_FAST_CONFIG").as_deref() {
        Ok("smoke") => smoke_test_config(),
        Ok("unit") => unit_test_config(),
        Ok("integration") => quick_integration_config(),
        _ => fast_config(),
    }
}

/// Apply fast configuration overrides to an existing config
pub fn apply_fast_overrides(mut config: TestConfig) -> TestConfig {
    // Apply speed optimizations
    config.max_parallel_tests = config.max_parallel_tests.min(2);
    config.test_timeout = Duration::from_secs(config.test_timeout.as_secs().min(60));

    // Disable expensive operations
    config.reporting.generate_coverage = false;
    config.reporting.generate_performance = false;
    config.crossval.enabled = false;
    config.fixtures.auto_download = false;

    // Minimal reporting
    config.reporting.formats = vec![ReportFormat::Json];

    config
}

/// Configuration profiles for different speed requirements
#[derive(Debug, Clone)]
pub enum SpeedProfile {
    /// Fastest possible execution, minimal validation
    Lightning,
    /// Fast execution with basic validation
    Fast,
    /// Balanced speed and validation
    Balanced,
    /// Thorough validation, slower execution
    Thorough,
}

impl SpeedProfile {
    /// Get configuration for the specified speed profile
    pub fn to_config(&self) -> TestConfig {
        match self {
            SpeedProfile::Lightning => {
                let mut config = smoke_test_config();
                config.test_timeout = Duration::from_secs(5);
                config.max_parallel_tests = 1;
                config
            }
            SpeedProfile::Fast => fast_config(),
            SpeedProfile::Balanced => quick_integration_config(),
            SpeedProfile::Thorough => TestConfig::default(),
        }
    }
}

/// Fast configuration builder for method chaining
pub struct FastConfigBuilder {
    config: TestConfig,
}

impl FastConfigBuilder {
    /// Start with fast configuration
    pub fn new() -> Self {
        Self { config: fast_config() }
    }

    /// Start with a specific speed profile
    pub fn with_profile(profile: SpeedProfile) -> Self {
        Self { config: profile.to_config() }
    }

    /// Set maximum parallel tests
    pub fn max_parallel(mut self, count: usize) -> Self {
        self.config.max_parallel_tests = count;
        self
    }

    /// Set test timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.test_timeout = timeout;
        self
    }

    /// Set log level
    pub fn log_level<S: Into<String>>(mut self, level: S) -> Self {
        self.config.log_level = level.into();
        self
    }

    /// Enable or disable coverage
    pub fn coverage(mut self, enabled: bool) -> Self {
        self.config.reporting.generate_coverage = enabled;
        self
    }

    /// Enable or disable performance reporting
    pub fn performance(mut self, enabled: bool) -> Self {
        self.config.reporting.generate_performance = enabled;
        self
    }

    /// Set report formats
    pub fn formats(mut self, formats: Vec<ReportFormat>) -> Self {
        self.config.reporting.formats = formats;
        self
    }

    /// Enable or disable cross-validation
    pub fn crossval(mut self, enabled: bool) -> Self {
        self.config.crossval.enabled = enabled;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> TestConfig {
        self.config
    }
}

impl Default for FastConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_config() {
        let config = fast_config();
        assert_eq!(config.max_parallel_tests, 2);
        assert_eq!(config.test_timeout, Duration::from_secs(30));
        assert_eq!(config.log_level, "error");
        assert!(!config.reporting.generate_coverage);
        assert!(!config.crossval.enabled);
    }

    #[test]
    fn test_smoke_test_config() {
        let config = smoke_test_config();
        assert_eq!(config.max_parallel_tests, 1);
        assert_eq!(config.test_timeout, Duration::from_secs(10));
        assert!(!config.reporting.include_artifacts);
    }

    #[test]
    fn test_speed_profiles() {
        let lightning = SpeedProfile::Lightning.to_config();
        let fast = SpeedProfile::Fast.to_config();
        let balanced = SpeedProfile::Balanced.to_config();
        let thorough = SpeedProfile::Thorough.to_config();

        assert!(lightning.test_timeout < fast.test_timeout);
        assert!(fast.test_timeout <= balanced.test_timeout);
        assert!(balanced.test_timeout <= thorough.test_timeout);
    }

    #[test]
    fn test_fast_config_builder() {
        let config = FastConfigBuilder::new()
            .max_parallel(4)
            .timeout(Duration::from_secs(45))
            .log_level("debug")
            .coverage(true)
            .formats(vec![ReportFormat::Html, ReportFormat::Json])
            .build();

        assert_eq!(config.max_parallel_tests, 4);
        assert_eq!(config.test_timeout, Duration::from_secs(45));
        assert_eq!(config.log_level, "debug");
        assert!(config.reporting.generate_coverage);
        assert_eq!(config.reporting.formats.len(), 2);
    }

    #[test]
    fn test_apply_fast_overrides() {
        let mut original = TestConfig::default();
        original.max_parallel_tests = 8;
        original.test_timeout = Duration::from_secs(300);
        original.reporting.generate_coverage = true;

        let fast = apply_fast_overrides(original);
        assert!(fast.max_parallel_tests <= 2);
        assert!(fast.test_timeout.as_secs() <= 60);
        assert!(!fast.reporting.generate_coverage);
    }
}
