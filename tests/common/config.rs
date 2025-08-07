use crate::errors::{TestError, TestResult};
use crate::utils::get_optimal_parallel_tests;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Main configuration for the testing framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    /// Maximum number of tests to run in parallel
    pub max_parallel_tests: usize,
    /// Timeout for individual tests
    pub test_timeout: Duration,
    /// Directory for test cache and temporary files
    pub cache_dir: PathBuf,
    /// Logging level for test execution
    pub log_level: String,
    /// Minimum code coverage threshold (0.0 to 1.0)
    pub coverage_threshold: f64,
    /// Configuration for test fixtures
    pub fixtures: FixtureConfig,
    /// Configuration for cross-validation testing
    pub crossval: CrossValidationConfig,
    /// Configuration for test reporting
    pub reporting: ReportingConfig,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            max_parallel_tests: get_optimal_parallel_tests(),
            test_timeout: Duration::from_secs(crate::DEFAULT_TEST_TIMEOUT_SECS),
            cache_dir: PathBuf::from("tests/cache"),
            log_level: "info".to_string(),
            coverage_threshold: 0.9,
            fixtures: FixtureConfig::default(),
            crossval: CrossValidationConfig::default(),
            reporting: ReportingConfig::default(),
        }
    }
}

/// Configuration for test fixture management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureConfig {
    /// Automatically download missing fixtures
    pub auto_download: bool,
    /// Maximum cache size in bytes (0 = unlimited)
    pub max_cache_size: u64,
    /// How often to clean up old fixtures
    pub cleanup_interval: Duration,
    /// Timeout for fixture downloads
    pub download_timeout: Duration,
    /// Base URL for fixture downloads
    pub base_url: Option<String>,
    /// Custom fixture definitions
    pub custom_fixtures: Vec<CustomFixture>,
}

impl Default for FixtureConfig {
    fn default() -> Self {
        Self {
            auto_download: true,
            max_cache_size: 10 * 1024 * 1024 * 1024, // 10 GB
            cleanup_interval: Duration::from_secs(24 * 60 * 60), // 24 hours
            download_timeout: Duration::from_secs(300), // 5 minutes
            base_url: None,
            custom_fixtures: Vec::new(),
        }
    }
}

/// Custom fixture definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomFixture {
    pub name: String,
    pub url: String,
    pub checksum: String,
    pub description: Option<String>,
}

/// Configuration for cross-validation testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Enable cross-validation testing
    pub enabled: bool,
    /// Tolerance settings for comparisons
    pub tolerance: ComparisonTolerance,
    /// Path to C++ BitNet binary (if available)
    pub cpp_binary_path: Option<PathBuf>,
    /// Test cases to run for cross-validation
    pub test_cases: Vec<String>,
    /// Enable performance comparison
    pub performance_comparison: bool,
    /// Enable accuracy comparison
    pub accuracy_comparison: bool,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default since it requires C++ setup
            tolerance: ComparisonTolerance::default(),
            cpp_binary_path: None,
            test_cases: vec![
                "basic_inference".to_string(),
                "tokenization".to_string(),
                "model_loading".to_string(),
            ],
            performance_comparison: true,
            accuracy_comparison: true,
        }
    }
}

/// Tolerance settings for cross-implementation comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonTolerance {
    /// Minimum token accuracy (0.0 to 1.0)
    pub min_token_accuracy: f64,
    /// Maximum probability divergence
    pub max_probability_divergence: f64,
    /// Maximum acceptable performance regression (ratio)
    pub max_performance_regression: f64,
    /// Numerical tolerance for floating-point comparisons
    pub numerical_tolerance: f64,
}

impl Default for ComparisonTolerance {
    fn default() -> Self {
        Self {
            min_token_accuracy: 0.999999, // 1e-6 tolerance
            max_probability_divergence: 1e-6,
            max_performance_regression: 0.1, // 10% regression allowed
            numerical_tolerance: 1e-6,
        }
    }
}

/// Configuration for test reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Output directory for test reports
    pub output_dir: PathBuf,
    /// Report formats to generate
    pub formats: Vec<ReportFormat>,
    /// Include test artifacts in reports
    pub include_artifacts: bool,
    /// Generate code coverage reports
    pub generate_coverage: bool,
    /// Generate performance reports
    pub generate_performance: bool,
    /// Upload reports to external services
    pub upload_reports: bool,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("test-reports"),
            formats: vec![ReportFormat::Html, ReportFormat::Json],
            include_artifacts: true,
            generate_coverage: true,
            generate_performance: true,
            upload_reports: false,
        }
    }
}

/// Available report formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportFormat {
    Html,
    Json,
    Junit,
    Markdown,
    Csv,
}

/// Load test configuration from various sources
pub fn load_test_config() -> TestResult<TestConfig> {
    // Try to load from environment-specific config file first
    if let Ok(config_path) = std::env::var("BITNET_TEST_CONFIG") {
        return load_config_from_file(&PathBuf::from(config_path));
    }

    // Try standard config file locations
    let config_paths = [
        "bitnet-test.toml",
        "tests/config.toml",
        ".bitnet/test-config.toml",
    ];

    for path in &config_paths {
        let path_buf = PathBuf::from(path);
        if path_buf.exists() {
            return load_config_from_file(&path_buf);
        }
    }

    // Load from environment variables
    let mut config = TestConfig::default();
    load_config_from_env(&mut config)?;

    Ok(config)
}

/// Load configuration from a TOML file
fn load_config_from_file(path: &PathBuf) -> TestResult<TestConfig> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| TestError::config(format!("Failed to read config file {:?}: {}", path, e)))?;

    let mut config: TestConfig = toml::from_str(&contents)
        .map_err(|e| TestError::config(format!("Failed to parse config file {:?}: {}", path, e)))?;

    // Override with environment variables
    load_config_from_env(&mut config)?;

    Ok(config)
}

/// Load configuration from environment variables
fn load_config_from_env(config: &mut TestConfig) -> TestResult<()> {
    if let Ok(val) = std::env::var("BITNET_TEST_PARALLEL") {
        config.max_parallel_tests = val
            .parse()
            .map_err(|e| TestError::config(format!("Invalid BITNET_TEST_PARALLEL: {}", e)))?;
    }

    if let Ok(val) = std::env::var("BITNET_TEST_TIMEOUT") {
        let timeout_secs: u64 = val
            .parse()
            .map_err(|e| TestError::config(format!("Invalid BITNET_TEST_TIMEOUT: {}", e)))?;
        config.test_timeout = Duration::from_secs(timeout_secs);
    }

    if let Ok(val) = std::env::var("BITNET_TEST_CACHE_DIR") {
        config.cache_dir = PathBuf::from(val);
    }

    if let Ok(val) = std::env::var("BITNET_TEST_LOG_LEVEL") {
        config.log_level = val;
    }

    if let Ok(val) = std::env::var("BITNET_TEST_COVERAGE_THRESHOLD") {
        config.coverage_threshold = val.parse().map_err(|e| {
            TestError::config(format!("Invalid BITNET_TEST_COVERAGE_THRESHOLD: {}", e))
        })?;
    }

    if let Ok(val) = std::env::var("BITNET_TEST_CROSSVAL_ENABLED") {
        config.crossval.enabled = val.parse().map_err(|e| {
            TestError::config(format!("Invalid BITNET_TEST_CROSSVAL_ENABLED: {}", e))
        })?;
    }

    if let Ok(val) = std::env::var("BITNET_TEST_CPP_BINARY") {
        config.crossval.cpp_binary_path = Some(PathBuf::from(val));
    }

    Ok(())
}

/// Validate configuration settings
pub fn validate_config(config: &TestConfig) -> TestResult<()> {
    // Validate parallel test count
    if config.max_parallel_tests == 0 {
        return Err(TestError::config(
            "max_parallel_tests must be greater than 0",
        ));
    }

    if config.max_parallel_tests > 100 {
        return Err(TestError::config(
            "max_parallel_tests should not exceed 100",
        ));
    }

    // Validate timeout
    if config.test_timeout.as_secs() == 0 {
        return Err(TestError::config("test_timeout must be greater than 0"));
    }

    if config.test_timeout.as_secs() > 3600 {
        return Err(TestError::config("test_timeout should not exceed 1 hour"));
    }

    // Validate coverage threshold
    if config.coverage_threshold < 0.0 || config.coverage_threshold > 1.0 {
        return Err(TestError::config(
            "coverage_threshold must be between 0.0 and 1.0",
        ));
    }

    // Validate log level
    let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
    if !valid_log_levels.contains(&config.log_level.as_str()) {
        return Err(TestError::config(format!(
            "Invalid log_level '{}'. Must be one of: {}",
            config.log_level,
            valid_log_levels.join(", ")
        )));
    }

    // Validate cache directory
    if let Some(parent) = config.cache_dir.parent() {
        if !parent.exists() {
            return Err(TestError::config(format!(
                "Cache directory parent {:?} does not exist",
                parent
            )));
        }
    }

    // Validate cross-validation config
    if config.crossval.enabled {
        if config.crossval.tolerance.min_token_accuracy < 0.0
            || config.crossval.tolerance.min_token_accuracy > 1.0
        {
            return Err(TestError::config(
                "min_token_accuracy must be between 0.0 and 1.0",
            ));
        }

        if config.crossval.tolerance.max_probability_divergence < 0.0 {
            return Err(TestError::config(
                "max_probability_divergence must be non-negative",
            ));
        }

        if config.crossval.tolerance.numerical_tolerance <= 0.0 {
            return Err(TestError::config("numerical_tolerance must be positive"));
        }
    }

    // Validate reporting config
    if config.reporting.formats.is_empty() {
        return Err(TestError::config(
            "At least one report format must be specified",
        ));
    }

    Ok(())
}

/// Create a test configuration optimized for CI environments
pub fn ci_config() -> TestConfig {
    let mut config = TestConfig::default();

    // Use fewer parallel tests in CI to avoid resource contention
    config.max_parallel_tests = (config.max_parallel_tests / 2).max(1);

    // Shorter timeout in CI for faster feedback
    config.test_timeout = Duration::from_secs(120);

    // More verbose logging in CI
    config.log_level = "debug".to_string();

    // Always generate coverage in CI
    config.reporting.generate_coverage = true;

    // Include JUnit format for CI integration
    if !config.reporting.formats.contains(&ReportFormat::Junit) {
        config.reporting.formats.push(ReportFormat::Junit);
    }

    config
}

/// Create a test configuration optimized for development
pub fn dev_config() -> TestConfig {
    let mut config = TestConfig::default();

    // Use more parallel tests for faster local development
    config.max_parallel_tests = get_optimal_parallel_tests();

    // Longer timeout for debugging
    config.test_timeout = Duration::from_secs(600);

    // Less verbose logging for cleaner output
    config.log_level = "info".to_string();

    // Skip coverage by default in dev mode for speed
    config.reporting.generate_coverage = false;

    // Only generate HTML reports for easy viewing
    config.reporting.formats = vec![ReportFormat::Html];

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TestConfig::default();
        assert!(config.max_parallel_tests > 0);
        assert!(config.test_timeout.as_secs() > 0);
        assert!(!config.cache_dir.as_os_str().is_empty());
        assert!(!config.log_level.is_empty());
        assert!(config.coverage_threshold >= 0.0 && config.coverage_threshold <= 1.0);
    }

    #[test]
    fn test_validate_config() {
        let config = TestConfig::default();
        assert!(validate_config(&config).is_ok());

        let mut invalid_config = config.clone();
        invalid_config.max_parallel_tests = 0;
        assert!(validate_config(&invalid_config).is_err());

        let mut invalid_config = config.clone();
        invalid_config.coverage_threshold = 1.5;
        assert!(validate_config(&invalid_config).is_err());

        let mut invalid_config = config.clone();
        invalid_config.log_level = "invalid".to_string();
        assert!(validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_ci_config() {
        let config = ci_config();
        assert!(config.reporting.generate_coverage);
        assert!(config.reporting.formats.contains(&ReportFormat::Junit));
        assert_eq!(config.log_level, "debug");
    }

    #[test]
    fn test_dev_config() {
        let config = dev_config();
        assert!(!config.reporting.generate_coverage);
        assert_eq!(config.reporting.formats, vec![ReportFormat::Html]);
        assert_eq!(config.log_level, "info");
    }
}
