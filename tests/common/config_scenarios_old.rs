use super::config::{
    ComparisonTolerance, CrossValidationConfig, FixtureConfig, ReportFormat, ReportingConfig,
    TestConfig,
};
#[cfg(feature = "fixtures")]
use super::fast_config::{FastConfigBuilder, SpeedProfile};

#[cfg(not(feature = "fixtures"))]
use super::config_scenarios_simple::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration scenarios for different testing contexts
/// This module provides comprehensive configuration management for various testing scenarios

/// Enumeration of all supported testing scenarios
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TestingScenario {
    /// Unit testing scenario - fast, isolated tests
    Unit,
    /// Integration testing scenario - component interaction tests
    Integration,
    /// End-to-end testing scenario - full workflow tests
    EndToEnd,
    /// Performance testing scenario - benchmarking and performance validation
    Performance,
    /// Cross-validation scenario - comparing implementations
    CrossValidation,
    /// Regression testing scenario - detecting regressions
    Regression,
    /// Smoke testing scenario - basic functionality validation
    Smoke,
    /// Stress testing scenario - high load and resource exhaustion
    Stress,
    /// Security testing scenario - security vulnerability testing
    Security,
    /// Compatibility testing scenario - platform and version compatibility
    Compatibility,
    /// Development scenario - local development testing
    Development,
    /// CI/CD scenario - continuous integration testing
    ContinuousIntegration,
    /// Release validation scenario - pre-release testing
    ReleaseValidation,
    /// Debug scenario - debugging and troubleshooting
    Debug,
    /// Minimal scenario - minimal resource usage
    Minimal,
}

/// Configuration context for scenario-specific settings
#[derive(Debug, Clone)]
pub struct ConfigurationContext {
    /// The testing scenario
    pub scenario: TestingScenario,
    /// Environment type (development, ci, production)
    pub environment: EnvironmentType,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Time constraints
    pub time_constraints: TimeConstraints,
    /// Quality requirements
    pub quality_requirements: QualityRequirements,
    /// Platform-specific settings
    pub platform_settings: PlatformSettings,
}

/// Environment type for configuration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EnvironmentType {
    Development,
    ContinuousIntegration,
    Staging,
    Production,
    Testing,
}

/// Resource constraints for testing
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum memory usage in MB (0 = unlimited)
    pub max_memory_mb: u64,
    /// Maximum CPU usage percentage (0.0 to 1.0, 0.0 = unlimited)
    pub max_cpu_usage: f64,
    /// Maximum disk space for cache in MB
    pub max_disk_cache_mb: u64,
    /// Maximum parallel tests
    pub max_parallel_tests: Option<usize>,
    /// Network access allowed
    pub network_access: bool,
}

/// Time constraints for testing
#[derive(Debug, Clone)]
pub struct TimeConstraints {
    /// Maximum total execution time
    pub max_total_duration: Duration,
    /// Maximum individual test timeout
    pub max_test_timeout: Duration,
    /// Target feedback time for fast scenarios
    pub target_feedback_time: Option<Duration>,
    /// Enable fail-fast behavior
    pub fail_fast: bool,
}

/// Quality requirements for testing
#[derive(Debug, Clone)]
pub struct QualityRequirements {
    /// Minimum code coverage threshold
    pub min_coverage: f64,
    /// Enable comprehensive reporting
    pub comprehensive_reporting: bool,
    /// Enable performance monitoring
    pub performance_monitoring: bool,
    /// Enable cross-validation
    pub cross_validation: bool,
    /// Accuracy tolerance for comparisons
    pub accuracy_tolerance: f64,
}

/// Platform-specific settings
#[derive(Debug, Clone)]
pub struct PlatformSettings {
    /// Target platform
    pub platform: Platform,
    /// Platform-specific optimizations
    pub optimizations: Vec<String>,
    /// Platform-specific environment variables
    pub environment_variables: HashMap<String, String>,
}

/// Supported platforms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Platform {
    Windows,
    Linux,
    MacOS,
    Generic,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_mb: 0,         // unlimited
            max_cpu_usage: 0.0,       // unlimited
            max_disk_cache_mb: 1024,  // 1GB
            max_parallel_tests: None, // auto-detect
            network_access: true,
        }
    }
}

impl Default for TimeConstraints {
    fn default() -> Self {
        Self {
            max_total_duration: Duration::from_secs(3600), // 1 hour
            max_test_timeout: Duration::from_secs(300),    // 5 minutes
            target_feedback_time: None,
            fail_fast: false,
        }
    }
}

impl Default for QualityRequirements {
    fn default() -> Self {
        Self {
            min_coverage: 0.9,
            comprehensive_reporting: true,
            performance_monitoring: true,
            cross_validation: false,
            accuracy_tolerance: 1e-6,
        }
    }
}

impl Default for PlatformSettings {
    fn default() -> Self {
        Self {
            platform: Platform::Generic,
            optimizations: Vec::new(),
            environment_variables: HashMap::new(),
        }
    }
}

impl Default for ConfigurationContext {
    fn default() -> Self {
        Self {
            scenario: TestingScenario::Unit,
            environment: EnvironmentType::Development,
            resource_constraints: ResourceConstraints::default(),
            time_constraints: TimeConstraints::default(),
            quality_requirements: QualityRequirements::default(),
            platform_settings: PlatformSettings::default(),
        }
    }
}

/// Configuration manager for different testing scenarios
pub struct ScenarioConfigManager {
    base_config: TestConfig,
    scenario_overrides: HashMap<TestingScenario, TestConfig>,
    environment_overrides: HashMap<EnvironmentType, TestConfig>,
}

impl ScenarioConfigManager {
    /// Create a new scenario configuration manager
    pub fn new() -> Self {
        let mut manager = Self {
            base_config: TestConfig::default(),
            scenario_overrides: HashMap::new(),
            environment_overrides: HashMap::new(),
        };

        manager.initialize_scenario_overrides();
        manager.initialize_environment_overrides();
        manager
    }

    /// Initialize scenario-specific configuration overrides
    fn initialize_scenario_overrides(&mut self) {
        // Unit testing scenario
        let unit_config = FastConfigBuilder::with_profile(SpeedProfile::Fast)
            .max_parallel(8)
            .timeout(Duration::from_secs(30))
            .log_level("warn")
            .coverage(true)
            .performance(false)
            .crossval(false)
            .formats(vec![ReportFormat::Json, ReportFormat::Html])
            .build();
        self.scenario_overrides.insert(TestingScenario::Unit, unit_config);

        // Integration testing scenario
        let integration_config = FastConfigBuilder::with_profile(SpeedProfile::Balanced)
            .max_parallel(4)
            .timeout(Duration::from_secs(120))
            .log_level("info")
            .coverage(true)
            .performance(true)
            .crossval(false)
            .formats(vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Junit])
            .build();
        self.scenario_overrides.insert(TestingScenario::Integration, integration_config);

        // End-to-end testing scenario
        let mut e2e_config = FastConfigBuilder::with_profile(SpeedProfile::Thorough)
            .max_parallel(2)
            .timeout(Duration::from_secs(300))
            .log_level("info")
            .coverage(true)
            .performance(true)
            .crossval(false)
            .formats(vec![
                ReportFormat::Html,
                ReportFormat::Json,
                ReportFormat::Junit,
                ReportFormat::Markdown,
            ])
            .build();
        e2e_config.fixtures.auto_download = true;
        e2e_config.reporting.include_artifacts = true;
        self.scenario_overrides.insert(TestingScenario::EndToEnd, e2e_config);

        // Performance testing scenario
        let mut performance_config = TestConfig::default();
        performance_config.max_parallel_tests = 1; // Sequential for accurate measurements
        performance_config.test_timeout = Duration::from_secs(600);
        performance_config.log_level = "debug".to_string();
        performance_config.reporting.generate_performance = true;
        performance_config.reporting.generate_coverage = false; // Skip for performance
        performance_config.reporting.formats = vec![ReportFormat::Json, ReportFormat::Csv];
        self.scenario_overrides.insert(TestingScenario::Performance, performance_config);

        // Cross-validation scenario
        let mut crossval_config = TestConfig::default();
        crossval_config.max_parallel_tests = 1; // Sequential for deterministic comparison
        crossval_config.test_timeout = Duration::from_secs(900);
        crossval_config.log_level = "debug".to_string();
        crossval_config.crossval.enabled = true;
        crossval_config.crossval.performance_comparison = true;
        crossval_config.crossval.accuracy_comparison = true;
        crossval_config.crossval.tolerance = ComparisonTolerance {
            min_token_accuracy: 0.999999,
            max_probability_divergence: 1e-6,
            max_performance_regression: 0.1,
            numerical_tolerance: 1e-6,
        };
        crossval_config.reporting.formats =
            vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Markdown];
        self.scenario_overrides.insert(TestingScenario::CrossValidation, crossval_config);

        // Smoke testing scenario
        let smoke_config = FastConfigBuilder::with_profile(SpeedProfile::Lightning)
            .max_parallel(1)
            .timeout(Duration::from_secs(10))
            .log_level("error")
            .coverage(false)
            .performance(false)
            .crossval(false)
            .formats(vec![ReportFormat::Json])
            .build();
        self.scenario_overrides.insert(TestingScenario::Smoke, smoke_config);

        // Stress testing scenario
        let mut stress_config = TestConfig::default();
        stress_config.max_parallel_tests = num_cpus::get() * 2; // Oversubscribe for stress
        stress_config.test_timeout = Duration::from_secs(1800); // 30 minutes
        stress_config.log_level = "debug".to_string();
        stress_config.fixtures.max_cache_size = 100 * BYTES_PER_MB; // Limit cache for stress
        stress_config.reporting.generate_performance = true;
        stress_config.reporting.formats = vec![ReportFormat::Json, ReportFormat::Html];
        self.scenario_overrides.insert(TestingScenario::Stress, stress_config);

        // Security testing scenario
        let mut security_config = TestConfig::default();
        security_config.max_parallel_tests = 1; // Sequential for security tests
        security_config.test_timeout = Duration::from_secs(300);
        security_config.log_level = "debug".to_string();
        security_config.fixtures.auto_download = false; // No network for security
        security_config.reporting.include_artifacts = true;
        security_config.reporting.formats =
            vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Junit];
        self.scenario_overrides.insert(TestingScenario::Security, security_config);

        // Development scenario
        let dev_config = FastConfigBuilder::with_profile(SpeedProfile::Fast)
            .max_parallel(num_cpus::get())
            .timeout(Duration::from_secs(60))
            .log_level("info")
            .coverage(false) // Skip for speed in development
            .performance(false)
            .crossval(false)
            .formats(vec![ReportFormat::Html])
            .build();
        self.scenario_overrides.insert(TestingScenario::Development, dev_config);

        // Debug scenario
        let mut debug_config = TestConfig::default();
        debug_config.max_parallel_tests = 1; // Sequential for debugging
        debug_config.test_timeout = Duration::from_secs(3600); // Long timeout for debugging
        debug_config.log_level = "trace".to_string();
        debug_config.reporting.include_artifacts = true;
        debug_config.reporting.formats = vec![ReportFormat::Html, ReportFormat::Json];
        self.scenario_overrides.insert(TestingScenario::Debug, debug_config);

        // Minimal scenario
        let minimal_config = FastConfigBuilder::with_profile(SpeedProfile::Lightning)
            .max_parallel(1)
            .timeout(Duration::from_secs(30))
            .log_level("error")
            .coverage(false)
            .performance(false)
            .crossval(false)
            .formats(vec![ReportFormat::Json])
            .build();
        self.scenario_overrides.insert(TestingScenario::Minimal, minimal_config);
    }

    /// Initialize environment-specific configuration overrides
    fn initialize_environment_overrides(&mut self) {
        // Development environment
        let mut dev_env_config = TestConfig::default();
        dev_env_config.log_level = "info".to_string();
        dev_env_config.reporting.generate_coverage = false; // Skip for speed
        dev_env_config.reporting.formats = vec![ReportFormat::Html];
        self.environment_overrides.insert(EnvironmentType::Development, dev_env_config);

        // CI environment
        let mut ci_env_config = TestConfig::default();
        ci_env_config.max_parallel_tests = 4; // Conservative for CI stability
        ci_env_config.log_level = "debug".to_string();
        ci_env_config.reporting.generate_coverage = true;
        ci_env_config.reporting.formats =
            vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Junit];
        ci_env_config.reporting.upload_reports = true;
        self.environment_overrides.insert(EnvironmentType::ContinuousIntegration, ci_env_config);

        // Production environment
        let mut prod_env_config = TestConfig::default();
        prod_env_config.max_parallel_tests = 2; // Conservative for production
        prod_env_config.test_timeout = Duration::from_secs(600);
        prod_env_config.log_level = "warn".to_string();
        prod_env_config.reporting.generate_coverage = true;
        prod_env_config.reporting.generate_performance = true;
        prod_env_config.reporting.formats = vec![
            ReportFormat::Html,
            ReportFormat::Json,
            ReportFormat::Junit,
            ReportFormat::Markdown,
        ];
        self.environment_overrides.insert(EnvironmentType::Production, prod_env_config);
    }

    /// Get configuration for a specific scenario
    pub fn get_scenario_config(&self, scenario: &TestingScenario) -> TestConfig {
        if let Some(override_config) = self.scenario_overrides.get(scenario) {
            override_config.clone()
        } else {
            self.base_config.clone()
        }
    }

    /// Get configuration for a specific environment
    pub fn get_environment_config(&self, environment: &EnvironmentType) -> TestConfig {
        if let Some(override_config) = self.environment_overrides.get(environment) {
            override_config.clone()
        } else {
            self.base_config.clone()
        }
    }

    /// Get configuration for a specific context (scenario + environment + constraints)
    pub fn get_context_config(&self, context: &ConfigurationContext) -> TestConfig {
        let mut config = self.get_scenario_config(&context.scenario);
        let env_config = self.get_environment_config(&context.environment);

        // Merge environment overrides
        config = super::config::merge_configs(config, env_config);

        // Apply resource constraints
        self.apply_resource_constraints(&mut config, &context.resource_constraints);

        // Apply time constraints
        self.apply_time_constraints(&mut config, &context.time_constraints);

        // Apply quality requirements
        self.apply_quality_requirements(&mut config, &context.quality_requirements);

        // Apply platform settings
        self.apply_platform_settings(&mut config, &context.platform_settings);

        config
    }

    /// Apply resource constraints to configuration
    fn apply_resource_constraints(
        &self,
        config: &mut TestConfig,
        constraints: &ResourceConstraints,
    ) {
        if let Some(max_parallel) = constraints.max_parallel_tests {
            config.max_parallel_tests = config.max_parallel_tests.min(max_parallel);
        }

        if constraints.max_disk_cache_mb > 0 {
            config.fixtures.max_cache_size = constraints.max_disk_cache_mb * BYTES_PER_MB;
        }

        if !constraints.network_access {
            config.fixtures.auto_download = false;
            config.fixtures.base_url = None;
            config.reporting.upload_reports = false;
        }
    }

    /// Apply time constraints to configuration
    fn apply_time_constraints(&self, config: &mut TestConfig, constraints: &TimeConstraints) {
        // Apply test timeout constraint
        if constraints.max_test_timeout < config.test_timeout {
            config.test_timeout = constraints.max_test_timeout;
        }

        // Apply target feedback time for fast scenarios
        if let Some(target_time) = constraints.target_feedback_time {
            if target_time < Duration::from_secs(300) {
                // 5 minutes
                // Apply aggressive optimizations for fast feedback
                config.max_parallel_tests = config.max_parallel_tests.min(4);
                config.reporting.generate_coverage = false;
                config.reporting.generate_performance = false;
                config.reporting.formats = vec![ReportFormat::Json];
                config.crossval.enabled = false;
            }
        }
    }

    /// Apply quality requirements to configuration
    fn apply_quality_requirements(
        &self,
        config: &mut TestConfig,
        requirements: &QualityRequirements,
    ) {
        config.coverage_threshold = requirements.min_coverage;
        config.reporting.generate_coverage = requirements.comprehensive_reporting;
        config.reporting.generate_performance = requirements.performance_monitoring;
        config.crossval.enabled = requirements.cross_validation;

        if requirements.cross_validation {
            config.crossval.tolerance.min_token_accuracy = requirements.accuracy_tolerance;
            config.crossval.tolerance.numerical_tolerance = requirements.accuracy_tolerance;
        }

        if requirements.comprehensive_reporting {
            config.reporting.include_artifacts = true;
            if !config.reporting.formats.contains(&ReportFormat::Html) {
                config.reporting.formats.push(ReportFormat::Html);
            }
            if !config.reporting.formats.contains(&ReportFormat::Markdown) {
                config.reporting.formats.push(ReportFormat::Markdown);
            }
        }
    }

    /// Apply platform-specific settings to configuration
    fn apply_platform_settings(&self, config: &mut TestConfig, settings: &PlatformSettings) {
        match settings.platform {
            Platform::Windows => {
                // Windows-specific optimizations
                if config.max_parallel_tests > 8 {
                    config.max_parallel_tests = 8; // Conservative on Windows
                }
            }
            Platform::Linux => {
                // Linux-specific optimizations
                // Can handle more parallel tests typically
            }
            Platform::MacOS => {
                // macOS-specific optimizations
                if config.max_parallel_tests > 6 {
                    config.max_parallel_tests = 6; // Conservative on macOS
                }
            }
            Platform::Generic => {
                // No platform-specific optimizations
            }
        }
    }

    /// Create a configuration context from environment variables
    pub fn context_from_environment() -> ConfigurationContext {
        let mut context = ConfigurationContext::default();

        // Detect scenario from environment
        if let Ok(scenario_str) = std::env::var("BITNET_TEST_SCENARIO") {
            context.scenario = match scenario_str.to_lowercase().as_str() {
                "unit" => TestingScenario::Unit,
                "integration" => TestingScenario::Integration,
                "e2e" | "end-to-end" => TestingScenario::EndToEnd,
                "performance" => TestingScenario::Performance,
                "crossval" | "cross-validation" => TestingScenario::CrossValidation,
                "regression" => TestingScenario::Regression,
                "smoke" => TestingScenario::Smoke,
                "stress" => TestingScenario::Stress,
                "security" => TestingScenario::Security,
                "compatibility" => TestingScenario::Compatibility,
                "development" | "dev" => TestingScenario::Development,
                "ci" | "continuous-integration" => TestingScenario::ContinuousIntegration,
                "release" | "release-validation" => TestingScenario::ReleaseValidation,
                "debug" => TestingScenario::Debug,
                "minimal" => TestingScenario::Minimal,
                _ => TestingScenario::Unit,
            };
        }

        // Detect environment type
        if std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok() {
            context.environment = EnvironmentType::ContinuousIntegration;
        } else if std::env::var("BITNET_ENV").as_deref() == Ok("production") {
            context.environment = EnvironmentType::Production;
        } else if std::env::var("BITNET_ENV").as_deref() == Ok("staging") {
            context.environment = EnvironmentType::Staging;
        } else {
            context.environment = EnvironmentType::Development;
        }

        // Apply resource constraints from environment
        if let Ok(max_memory) = std::env::var("BITNET_MAX_MEMORY_MB") {
            if let Ok(memory_mb) = max_memory.parse::<u64>() {
                context.resource_constraints.max_memory_mb = memory_mb;
            }
        }

        if let Ok(max_parallel) = std::env::var("BITNET_MAX_PARALLEL") {
            if let Ok(parallel) = max_parallel.parse::<usize>() {
                context.resource_constraints.max_parallel_tests = Some(parallel);
            }
        }

        if std::env::var("BITNET_NO_NETWORK").is_ok() {
            context.resource_constraints.network_access = false;
        }

        // Apply time constraints from environment
        if let Ok(max_duration) = std::env::var("BITNET_MAX_DURATION_SECS") {
            if let Ok(duration_secs) = max_duration.parse::<u64>() {
                context.time_constraints.max_total_duration = Duration::from_secs(duration_secs);
            }
        }

        if let Ok(feedback_time) = std::env::var("BITNET_TARGET_FEEDBACK_SECS") {
            if let Ok(feedback_secs) = feedback_time.parse::<u64>() {
                context.time_constraints.target_feedback_time =
                    Some(Duration::from_secs(feedback_secs));
            }
        }

        if std::env::var("BITNET_FAIL_FAST").is_ok() {
            context.time_constraints.fail_fast = true;
        }

        // Apply quality requirements from environment
        if let Ok(min_coverage) = std::env::var("BITNET_MIN_COVERAGE") {
            if let Ok(coverage) = min_coverage.parse::<f64>() {
                context.quality_requirements.min_coverage = coverage;
            }
        }

        if std::env::var("BITNET_COMPREHENSIVE_REPORTING").is_ok() {
            context.quality_requirements.comprehensive_reporting = true;
        }

        if std::env::var("BITNET_ENABLE_CROSSVAL").is_ok() {
            context.quality_requirements.cross_validation = true;
        }

        // Detect platform
        context.platform_settings.platform = if cfg!(target_os = "windows") {
            Platform::Windows
        } else if cfg!(target_os = "linux") {
            Platform::Linux
        } else if cfg!(target_os = "macos") {
            Platform::MacOS
        } else {
            Platform::Generic
        };

        context
    }

    /// Get all available scenarios
    pub fn available_scenarios() -> Vec<TestingScenario> {
        vec![
            TestingScenario::Unit,
            TestingScenario::Integration,
            TestingScenario::EndToEnd,
            TestingScenario::Performance,
            TestingScenario::CrossValidation,
            TestingScenario::Regression,
            TestingScenario::Smoke,
            TestingScenario::Stress,
            TestingScenario::Security,
            TestingScenario::Compatibility,
            TestingScenario::Development,
            TestingScenario::ContinuousIntegration,
            TestingScenario::ReleaseValidation,
            TestingScenario::Debug,
            TestingScenario::Minimal,
        ]
    }

    /// Get scenario description
    pub fn scenario_description(scenario: &TestingScenario) -> &'static str {
        match scenario {
            TestingScenario::Unit => "Fast, isolated unit tests with high parallelism",
            TestingScenario::Integration => "Component interaction tests with balanced performance",
            TestingScenario::EndToEnd => {
                "Complete workflow validation with comprehensive reporting"
            }
            TestingScenario::Performance => {
                "Sequential performance benchmarking with detailed metrics"
            }
            TestingScenario::CrossValidation => {
                "Implementation comparison with strict accuracy requirements"
            }
            TestingScenario::Regression => "Regression detection with baseline comparison",
            TestingScenario::Smoke => "Basic functionality validation with minimal overhead",
            TestingScenario::Stress => "High-load testing with resource exhaustion scenarios",
            TestingScenario::Security => "Security vulnerability testing with isolated execution",
            TestingScenario::Compatibility => "Platform and version compatibility validation",
            TestingScenario::Development => "Local development testing optimized for fast feedback",
            TestingScenario::ContinuousIntegration => {
                "CI-optimized testing with comprehensive reporting"
            }
            TestingScenario::ReleaseValidation => "Pre-release validation with thorough testing",
            TestingScenario::Debug => {
                "Debugging-focused testing with verbose logging and artifacts"
            }
            TestingScenario::Minimal => {
                "Minimal resource usage testing for constrained environments"
            }
        }
    }
}

impl Default for ScenarioConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common configuration scenarios
pub mod scenarios {
    use super::*;

    /// Get configuration for unit testing
    pub fn unit_testing() -> TestConfig {
        ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Unit)
    }

    /// Get configuration for integration testing
    pub fn integration_testing() -> TestConfig {
        ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Integration)
    }

    /// Get configuration for performance testing
    pub fn performance_testing() -> TestConfig {
        ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Performance)
    }

    /// Get configuration for cross-validation testing
    pub fn cross_validation_testing() -> TestConfig {
        ScenarioConfigManager::new().get_scenario_config(&TestingScenario::CrossValidation)
    }

    /// Get configuration for smoke testing
    pub fn smoke_testing() -> TestConfig {
        ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Smoke)
    }

    /// Get configuration for development
    pub fn development() -> TestConfig {
        ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Development)
    }

    /// Get configuration for CI/CD
    pub fn continuous_integration() -> TestConfig {
        ScenarioConfigManager::new().get_scenario_config(&TestingScenario::ContinuousIntegration)
    }

    /// Get configuration based on current environment
    pub fn from_environment() -> TestConfig {
        let manager = ScenarioConfigManager::new();
        let context = ScenarioConfigManager::context_from_environment();
        manager.get_context_config(&context)
    }

    /// Get configuration for a specific context
    pub fn from_context(context: &ConfigurationContext) -> TestConfig {
        ScenarioConfigManager::new().get_context_config(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_config_manager() {
        let manager = ScenarioConfigManager::new();

        // Test unit testing scenario
        let unit_config = manager.get_scenario_config(&TestingScenario::Unit);
        assert_eq!(unit_config.log_level, "warn");
        assert!(unit_config.reporting.generate_coverage);
        assert!(!unit_config.crossval.enabled);

        // Test performance testing scenario
        let perf_config = manager.get_scenario_config(&TestingScenario::Performance);
        assert_eq!(perf_config.max_parallel_tests, 1);
        assert!(perf_config.reporting.generate_performance);
        assert!(!perf_config.reporting.generate_coverage);

        // Test cross-validation scenario
        let crossval_config = manager.get_scenario_config(&TestingScenario::CrossValidation);
        assert!(crossval_config.crossval.enabled);
        assert_eq!(crossval_config.max_parallel_tests, 1);
        assert_eq!(crossval_config.crossval.tolerance.min_token_accuracy, 0.999999);
    }

    #[test]
    fn test_configuration_context() {
        let mut context = ConfigurationContext::default();
        context.scenario = TestingScenario::Performance;
        context.environment = EnvironmentType::ContinuousIntegration;
        context.resource_constraints.max_parallel_tests = Some(2);
        context.time_constraints.max_test_timeout = Duration::from_secs(60);
        context.quality_requirements.min_coverage = 0.95;

        let manager = ScenarioConfigManager::new();
        let config = manager.get_context_config(&context);

        assert_eq!(config.max_parallel_tests, 2); // Resource constraint applied
        assert_eq!(config.test_timeout, Duration::from_secs(60)); // Time constraint applied
        assert_eq!(config.coverage_threshold, 0.95); // Quality requirement applied
    }

    #[test]
    fn test_resource_constraints() {
        let manager = ScenarioConfigManager::new();
        let mut context = ConfigurationContext::default();
        context.resource_constraints.max_parallel_tests = Some(1);
        context.resource_constraints.network_access = false;

        let config = manager.get_context_config(&context);
        assert_eq!(config.max_parallel_tests, 1);
        assert!(!config.fixtures.auto_download);
        assert!(!config.reporting.upload_reports);
    }

    #[test]
    fn test_time_constraints() {
        let manager = ScenarioConfigManager::new();
        let mut context = ConfigurationContext::default();
        context.time_constraints.target_feedback_time = Some(Duration::from_secs(120));

        let config = manager.get_context_config(&context);
        assert!(!config.reporting.generate_coverage); // Disabled for fast feedback
        assert_eq!(config.reporting.formats, vec![ReportFormat::Json]);
    }

    #[test]
    fn test_scenario_descriptions() {
        let description = ScenarioConfigManager::scenario_description(&TestingScenario::Unit);
        assert!(description.contains("Fast, isolated"));

        let description =
            ScenarioConfigManager::scenario_description(&TestingScenario::Performance);
        assert!(description.contains("Sequential performance"));
    }

    #[test]
    fn test_convenience_functions() {
        let unit_config = scenarios::unit_testing();
        assert_eq!(unit_config.log_level, "warn");

        let perf_config = scenarios::performance_testing();
        assert_eq!(perf_config.max_parallel_tests, 1);

        let smoke_config = scenarios::smoke_testing();
        assert_eq!(smoke_config.test_timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_available_scenarios() {
        let scenarios = ScenarioConfigManager::available_scenarios();
        assert!(scenarios.contains(&TestingScenario::Unit));
        assert!(scenarios.contains(&TestingScenario::Performance));
        assert!(scenarios.contains(&TestingScenario::CrossValidation));
        assert_eq!(scenarios.len(), 15);
    }
}
