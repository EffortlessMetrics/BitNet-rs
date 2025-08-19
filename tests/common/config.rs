use super::errors::{TestError, TestOpResult as TestResultCompat};
use super::utils::get_optimal_parallel_tests;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Helper to pick non-empty PathBuf or fallback to default
pub fn pick_dir(env: &PathBuf, scenario: &PathBuf) -> PathBuf {
    if !env.as_os_str().is_empty() { env.clone() } else { scenario.clone() }
}

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
    #[cfg(feature = "fixtures")]
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
            test_timeout: Duration::from_secs(super::DEFAULT_TEST_TIMEOUT_SECS),
            cache_dir: PathBuf::from("tests/cache"),
            log_level: "info".to_string(),
            coverage_threshold: 0.9,
            #[cfg(feature = "fixtures")]
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
            max_cache_size: 10 * crate::BYTES_PER_GB, // 10 GB
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
pub fn load_test_config() -> TestResultCompat<TestConfig> {
    // Try to load from environment-specific config file first
    if let Ok(config_path) = std::env::var("BITNET_TEST_CONFIG") {
        return load_config_from_file(&PathBuf::from(config_path));
    }

    // Try standard config file locations
    let config_paths = ["bitnet-test.toml", "tests/config.toml", ".bitnet/test-config.toml"];

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
pub fn load_config_from_file(path: &Path) -> TestResultCompat<TestConfig> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| TestError::config(format!("Failed to read config file {:?}: {}", path, e)))?;

    let mut config: TestConfig = toml::from_str(&contents)
        .map_err(|e| TestError::config(format!("Failed to parse config file {:?}: {}", path, e)))?;

    // Override with environment variables
    load_config_from_env(&mut config)?;

    Ok(config)
}

/// Load configuration from environment variables
pub fn load_config_from_env(config: &mut TestConfig) -> TestResultCompat<()> {
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

    // Fixture configuration from environment
    #[cfg(feature = "fixtures")]
    {
        if let Ok(val) = std::env::var("BITNET_TEST_AUTO_DOWNLOAD") {
            config.fixtures.auto_download = val.parse().map_err(|e| {
                TestError::config(format!("Invalid BITNET_TEST_AUTO_DOWNLOAD: {}", e))
            })?;
        }

        if let Ok(val) = std::env::var("BITNET_TEST_MAX_CACHE_SIZE") {
            config.fixtures.max_cache_size = val.parse().map_err(|e| {
                TestError::config(format!("Invalid BITNET_TEST_MAX_CACHE_SIZE: {}", e))
            })?;
        }

        if let Ok(val) = std::env::var("BITNET_TEST_FIXTURE_BASE_URL") {
            config.fixtures.base_url = Some(val);
        }
    }

    // Reporting configuration from environment
    if let Ok(val) = std::env::var("BITNET_TEST_REPORT_DIR") {
        config.reporting.output_dir = PathBuf::from(val);
    }

    if let Ok(val) = std::env::var("BITNET_TEST_REPORT_FORMATS") {
        let formats: Result<Vec<ReportFormat>, _> = val
            .split(',')
            .map(|s| match s.trim().to_lowercase().as_str() {
                "html" => Ok(ReportFormat::Html),
                "json" => Ok(ReportFormat::Json),
                "junit" => Ok(ReportFormat::Junit),
                "markdown" => Ok(ReportFormat::Markdown),
                "csv" => Ok(ReportFormat::Csv),
                _ => Err(TestError::config(format!("Invalid report format: {}", s))),
            })
            .collect();
        config.reporting.formats = formats?;
    }

    if let Ok(val) = std::env::var("BITNET_TEST_GENERATE_COVERAGE") {
        config.reporting.generate_coverage = val.parse().map_err(|e| {
            TestError::config(format!("Invalid BITNET_TEST_GENERATE_COVERAGE: {}", e))
        })?;
    }

    Ok(())
}

/// Validate configuration settings
pub fn validate_config(config: &TestConfig) -> TestResultCompat<()> {
    // Validate parallel test count
    if config.max_parallel_tests == 0 {
        return Err(TestError::config("max_parallel_tests must be greater than 0"));
    }

    if config.max_parallel_tests > 100 {
        return Err(TestError::config("max_parallel_tests should not exceed 100"));
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
        return Err(TestError::config("coverage_threshold must be between 0.0 and 1.0"));
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

    // Validate cache directory (only check if it's an absolute path with non-existent parent)
    if config.cache_dir.is_absolute() {
        if let Some(parent) = config.cache_dir.parent() {
            if !parent.exists() {
                return Err(TestError::config(format!(
                    "Cache directory parent {:?} does not exist",
                    parent
                )));
            }
        }
    }

    // Validate cross-validation config
    if config.crossval.enabled {
        if config.crossval.tolerance.min_token_accuracy < 0.0
            || config.crossval.tolerance.min_token_accuracy > 1.0
        {
            return Err(TestError::config("min_token_accuracy must be between 0.0 and 1.0"));
        }

        if config.crossval.tolerance.max_probability_divergence < 0.0 {
            return Err(TestError::config("max_probability_divergence must be non-negative"));
        }

        if config.crossval.tolerance.numerical_tolerance <= 0.0 {
            return Err(TestError::config("numerical_tolerance must be positive"));
        }
    }

    // Validate fixture config
    #[cfg(feature = "fixtures")]
    {
        if config.fixtures.download_timeout.as_secs() == 0 {
            return Err(TestError::config("download_timeout must be greater than 0"));
        }
    }

    #[cfg(feature = "fixtures")]
    if config.fixtures.download_timeout.as_secs() > 3600 {
        return Err(TestError::config("download_timeout should not exceed 1 hour"));
    }

    #[cfg(feature = "fixtures")]
    if config.fixtures.cleanup_interval.as_secs() == 0 {
        return Err(TestError::config("cleanup_interval must be greater than 0"));
    }

    // Validate custom fixtures
    #[cfg(feature = "fixtures")]
    {
        for fixture in &config.fixtures.custom_fixtures {
            if fixture.name.is_empty() {
                return Err(TestError::config("Custom fixture name cannot be empty"));
            }

            if fixture.url.is_empty() {
                return Err(TestError::config("Custom fixture URL cannot be empty"));
            }

            if fixture.checksum.is_empty() {
                return Err(TestError::config("Custom fixture checksum cannot be empty"));
            }

            // Basic URL validation
            if !fixture.url.starts_with("http://") && !fixture.url.starts_with("https://") {
                return Err(TestError::config(format!(
                    "Invalid URL for fixture '{}': must start with http:// or https://",
                    fixture.name
                )));
            }

            // Basic checksum validation (should be hex string)
            if !fixture.checksum.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(TestError::config(format!(
                    "Invalid checksum for fixture '{}': must be a hex string",
                    fixture.name
                )));
            }
        }
    }

    // Validate reporting config
    if config.reporting.formats.is_empty() {
        return Err(TestError::config("At least one report format must be specified"));
    }

    // Validate output directory parent exists (only for absolute paths)
    if config.reporting.output_dir.is_absolute() {
        if let Some(parent) = config.reporting.output_dir.parent() {
            if !parent.exists() {
                return Err(TestError::config(format!(
                    "Report output directory parent {:?} does not exist",
                    parent
                )));
            }
        }
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

/// Create a test configuration optimized for minimal resource usage
pub fn minimal_config() -> TestConfig {
    let mut config = TestConfig::default();

    // Use minimal parallel tests
    config.max_parallel_tests = 1;

    // Shorter timeout for quick feedback
    config.test_timeout = Duration::from_secs(60);

    // Minimal logging
    config.log_level = "warn".to_string();

    // Disable coverage and performance reporting
    config.reporting.generate_coverage = false;
    config.reporting.generate_performance = false;

    // Only JSON reports for minimal overhead
    config.reporting.formats = vec![ReportFormat::Json];

    // Disable cross-validation
    config.crossval.enabled = false;

    // Disable auto-download to avoid network calls
    #[cfg(feature = "fixtures")]
    {
        config.fixtures.auto_download = false;
    }

    config
}

/// Merge two configurations, with the second taking precedence
pub fn merge_configs(base: TestConfig, override_config: TestConfig) -> TestConfig {
    TestConfig {
        max_parallel_tests: override_config.max_parallel_tests,
        test_timeout: override_config.test_timeout,
        cache_dir: override_config.cache_dir,
        log_level: override_config.log_level,
        coverage_threshold: override_config.coverage_threshold,
        #[cfg(feature = "fixtures")]
        fixtures: FixtureConfig {
            auto_download: override_config.fixtures.auto_download,
            max_cache_size: override_config.fixtures.max_cache_size,
            cleanup_interval: override_config.fixtures.cleanup_interval,
            download_timeout: override_config.fixtures.download_timeout,
            base_url: override_config.fixtures.base_url.or(base.fixtures.base_url),
            custom_fixtures: if override_config.fixtures.custom_fixtures.is_empty() {
                base.fixtures.custom_fixtures
            } else {
                override_config.fixtures.custom_fixtures
            },
        },
        crossval: CrossValidationConfig {
            enabled: override_config.crossval.enabled,
            tolerance: override_config.crossval.tolerance,
            cpp_binary_path: override_config
                .crossval
                .cpp_binary_path
                .or(base.crossval.cpp_binary_path),
            test_cases: if override_config.crossval.test_cases.is_empty() {
                base.crossval.test_cases
            } else {
                override_config.crossval.test_cases
            },
            performance_comparison: override_config.crossval.performance_comparison,
            accuracy_comparison: override_config.crossval.accuracy_comparison,
        },
        reporting: ReportingConfig {
            output_dir: override_config.reporting.output_dir,
            formats: if override_config.reporting.formats.is_empty() {
                base.reporting.formats
            } else {
                override_config.reporting.formats
            },
            include_artifacts: override_config.reporting.include_artifacts,
            generate_coverage: override_config.reporting.generate_coverage,
            generate_performance: override_config.reporting.generate_performance,
            upload_reports: override_config.reporting.upload_reports,
        },
    }
}

/// Save configuration to a TOML file
pub fn save_config_to_file(config: &TestConfig, path: &Path) -> TestResultCompat<()> {
    let toml_string = toml::to_string_pretty(config)
        .map_err(|e| TestError::config(format!("Failed to serialize config: {}", e)))?;

    std::fs::write(path, toml_string)
        .map_err(|e| TestError::config(format!("Failed to write config file {:?}: {}", path, e)))?;

    Ok(())
}

/// Get configuration value by key path (e.g., "fixtures.auto_download")
pub fn get_config_value(config: &TestConfig, key_path: &str) -> Option<String> {
    let parts: Vec<&str> = key_path.split('.').collect();

    match parts.as_slice() {
        ["max_parallel_tests"] => Some(config.max_parallel_tests.to_string()),
        ["test_timeout"] => Some(config.test_timeout.as_secs().to_string()),
        ["cache_dir"] => Some(config.cache_dir.to_string_lossy().to_string()),
        ["log_level"] => Some(config.log_level.clone()),
        ["coverage_threshold"] => Some(config.coverage_threshold.to_string()),

        #[cfg(feature = "fixtures")]
        ["fixtures", "auto_download"] => Some(config.fixtures.auto_download.to_string()),
        #[cfg(feature = "fixtures")]
        ["fixtures", "max_cache_size"] => Some(config.fixtures.max_cache_size.to_string()),
        #[cfg(feature = "fixtures")]
        ["fixtures", "download_timeout"] => {
            Some(config.fixtures.download_timeout.as_secs().to_string())
        }
        #[cfg(feature = "fixtures")]
        ["fixtures", "base_url"] => config.fixtures.base_url.clone(),

        ["crossval", "enabled"] => Some(config.crossval.enabled.to_string()),
        ["crossval", "performance_comparison"] => {
            Some(config.crossval.performance_comparison.to_string())
        }
        ["crossval", "accuracy_comparison"] => {
            Some(config.crossval.accuracy_comparison.to_string())
        }

        ["reporting", "output_dir"] => {
            Some(config.reporting.output_dir.to_string_lossy().to_string())
        }
        ["reporting", "include_artifacts"] => Some(config.reporting.include_artifacts.to_string()),
        ["reporting", "generate_coverage"] => Some(config.reporting.generate_coverage.to_string()),
        ["reporting", "generate_performance"] => {
            Some(config.reporting.generate_performance.to_string())
        }

        _ => None,
    }
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
        assert!((0.0..=1.0).contains(&config.coverage_threshold));
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

    #[test]
    fn test_minimal_config() {
        let config = minimal_config();
        assert_eq!(config.max_parallel_tests, 1);
        assert_eq!(config.test_timeout, Duration::from_secs(60));
        assert_eq!(config.log_level, "warn");
        assert!(!config.reporting.generate_coverage);
        assert!(!config.crossval.enabled);
        #[cfg(feature = "fixtures")]
        assert!(!config.fixtures.auto_download);
    }

    #[test]
    fn test_merge_configs() {
        let base = TestConfig::default();
        let mut override_config = TestConfig::default();
        override_config.max_parallel_tests = 8;
        override_config.log_level = "debug".to_string();

        let merged = merge_configs(base, override_config);
        assert_eq!(merged.max_parallel_tests, 8);
        assert_eq!(merged.log_level, "debug");
    }

    #[test]
    fn test_get_config_value() {
        let config = TestConfig::default();

        assert_eq!(
            get_config_value(&config, "max_parallel_tests"),
            Some(config.max_parallel_tests.to_string())
        );

        #[cfg(feature = "fixtures")]
        assert_eq!(
            get_config_value(&config, "fixtures.auto_download"),
            Some(config.fixtures.auto_download.to_string())
        );

        assert_eq!(get_config_value(&config, "invalid.path"), None);
    }

    #[test]
    #[cfg(feature = "fixtures")]
    fn test_validate_custom_fixtures() {
        let mut config = TestConfig::default();

        // Valid custom fixture
        config.fixtures.custom_fixtures.push(CustomFixture {
            name: "test-model".to_string(),
            url: "https://example.com/model.bin".to_string(),
            checksum: "abc123def456".to_string(),
            description: Some("Test model".to_string()),
        });

        assert!(validate_config(&config).is_ok());

        // Invalid URL
        config.fixtures.custom_fixtures[0].url = "invalid-url".to_string();
        assert!(validate_config(&config).is_err());

        // Invalid checksum
        config.fixtures.custom_fixtures[0].url = "https://example.com/model.bin".to_string();
        config.fixtures.custom_fixtures[0].checksum = "invalid-checksum!".to_string();
        assert!(validate_config(&config).is_err());
    }
}
