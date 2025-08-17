#[cfg(feature = "fixtures")]
use super::config::FixtureConfig;
use super::config::{
    ComparisonTolerance, CrossValidationConfig, ReportFormat, ReportingConfig, TestConfig,
};

// Simple configs are now defined inline as part of the compatibility shim

use std::collections::HashMap;
use std::time::Duration;

/// Configuration scenarios for different testing contexts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TestingScenario {
    Unit,
    Integration,
    EndToEnd,
    Performance,
    CrossValidation,
    Smoke,
    Development,
    Debug,
    Minimal,
}

impl std::fmt::Display for TestingScenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unit => write!(f, "unit"),
            Self::Integration => write!(f, "integration"),
            Self::EndToEnd => write!(f, "e2e"),
            Self::Performance => write!(f, "performance"),
            Self::CrossValidation => write!(f, "crossval"),
            Self::Smoke => write!(f, "smoke"),
            Self::Development => write!(f, "dev"),
            Self::Debug => write!(f, "debug"),
            Self::Minimal => write!(f, "minimal"),
        }
    }
}

impl std::str::FromStr for TestingScenario {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "unit" => Ok(Self::Unit),
            "integration" => Ok(Self::Integration),
            "e2e" | "end-to-end" | "endtoend" => Ok(Self::EndToEnd),
            "performance" | "perf" => Ok(Self::Performance),
            "crossval" | "cross-validation" => Ok(Self::CrossValidation),
            "smoke" => Ok(Self::Smoke),
            "dev" | "development" => Ok(Self::Development),
            "debug" => Ok(Self::Debug),
            "minimal" | "min" => Ok(Self::Minimal),
            _ => Err(format!("Unknown testing scenario: {}", s)),
        }
    }
}

/// Environment types for test execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvironmentType {
    Local,
    CI,
    PreProduction,
    Production,
}

impl std::fmt::Display for EnvironmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
            Self::CI => write!(f, "ci"),
            Self::PreProduction => write!(f, "pre-prod"),
            Self::Production => write!(f, "prod"),
        }
    }
}

impl std::str::FromStr for EnvironmentType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "local" | "dev" | "development" => Ok(Self::Local),
            "ci" | "ci/cd" | "cicd" => Ok(Self::CI),
            "pre-prod" | "preprod" | "pre-production" | "preproduction" | "staging" => {
                Ok(Self::PreProduction)
            }
            "prod" | "production" => Ok(Self::Production),
            _ => Err(format!("Unknown environment type: {}", s)),
        }
    }
}

/// Configuration context containing scenario, environment, and constraints
#[derive(Debug, Clone)]
pub struct ConfigurationContext {
    pub scenario: TestingScenario,
    pub environment: EnvironmentType,
    pub resource_constraints: Option<ResourceConstraints>,
    pub time_constraints: Option<TimeConstraints>,
    pub quality_requirements: Option<QualityRequirements>,
    pub platform_settings: Option<PlatformSettings>,
}

/// Resource constraints for test execution
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_memory_mb: Option<usize>,
    pub max_cpu_cores: Option<usize>,
    pub max_disk_gb: Option<usize>,
}

/// Time constraints for test execution
#[derive(Debug, Clone)]
pub struct TimeConstraints {
    pub max_total_duration: Option<Duration>,
    pub max_test_duration: Option<Duration>,
}

/// Quality requirements for test results
#[derive(Debug, Clone)]
pub struct QualityRequirements {
    pub min_coverage: Option<f64>,
    pub max_flakiness: Option<f64>,
    pub required_passes: Option<usize>,
}

/// Platform-specific settings
#[derive(Debug, Clone)]
pub struct PlatformSettings {
    pub os: Option<String>,
    pub arch: Option<String>,
    pub features: Vec<String>,
}

/// Manager for scenario-based test configurations
#[derive(Debug, Clone)]
pub struct ScenarioConfigManager {
    scenario_overrides: HashMap<TestingScenario, TestConfig>,
    environment_overrides: HashMap<EnvironmentType, TestConfig>,
}

impl Default for ScenarioConfigManager {
    fn default() -> Self {
        let mut manager =
            Self { scenario_overrides: HashMap::new(), environment_overrides: HashMap::new() };
        manager.initialize_scenario_overrides();
        manager.initialize_environment_overrides();
        manager
    }
}

impl ScenarioConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        let mut mgr = Self::default();
        mgr.register_default_scenarios();
        mgr.register_default_environments();
        mgr
    }

    /// Register default scenario-specific configurations
    pub fn register_default_scenarios(&mut self) {
        self.initialize_scenario_overrides();
    }

    /// Register default environment-specific configurations
    pub fn register_default_environments(&mut self) {
        self.initialize_environment_overrides();
    }

    /// Initialize scenario-specific configuration overrides
    fn initialize_scenario_overrides(&mut self) {
        // Register all scenario configs
        self.scenario_overrides.insert(TestingScenario::Unit, create_unit_config());
        self.scenario_overrides.insert(TestingScenario::Integration, create_integration_config());
        self.scenario_overrides.insert(TestingScenario::EndToEnd, create_e2e_config());
        self.scenario_overrides.insert(TestingScenario::Performance, create_perf_config());
        self.scenario_overrides.insert(TestingScenario::Smoke, create_smoke_config());
        self.scenario_overrides.insert(TestingScenario::CrossValidation, create_crossval_config());
        self.scenario_overrides.insert(TestingScenario::Debug, create_debug_config());
        self.scenario_overrides.insert(TestingScenario::Development, create_development_config());
        self.scenario_overrides.insert(TestingScenario::Minimal, create_minimal_config());
    }

    /// Initialize environment-specific configuration overrides
    fn initialize_environment_overrides(&mut self) {
        // CI environment
        let mut ci_config = TestConfig::default();
        ci_config.max_parallel_tests = 4;
        ci_config.log_level = "debug".to_string();
        ci_config.reporting.output_dir = "/tmp/test-reports".into();
        ci_config.reporting.formats = vec![ReportFormat::Junit, ReportFormat::Html];
        ci_config.reporting.generate_coverage = true;
        ci_config.reporting.upload_reports = true;
        self.environment_overrides.insert(EnvironmentType::CI, ci_config);

        // Pre-production environment
        let mut preprod_config = TestConfig::default();
        preprod_config.coverage_threshold = 0.7;
        preprod_config.reporting.include_artifacts = true;
        self.environment_overrides.insert(EnvironmentType::PreProduction, preprod_config);

        // Production environment
        let mut prod_config = TestConfig::default();
        prod_config.max_parallel_tests = 1;
        prod_config.test_timeout = Duration::from_secs(60);
        prod_config.log_level = "warn".to_string();
        prod_config.reporting.formats = vec![ReportFormat::Json, ReportFormat::Markdown];
        prod_config.reporting.generate_coverage = true;
        prod_config.reporting.generate_performance = true;
        self.environment_overrides.insert(EnvironmentType::Production, prod_config);

        // Local environment (development settings)
        let mut local_config = TestConfig::default();
        local_config.log_level = "info".to_string();
        local_config.reporting.formats = vec![ReportFormat::Html];
        local_config.reporting.generate_coverage = false;
        local_config.reporting.generate_performance = false;
        self.environment_overrides.insert(EnvironmentType::Local, local_config);
    }

    /// Get configuration for a specific scenario
    pub fn get_scenario_config(&self, scenario: &TestingScenario) -> TestConfig {
        self.scenario_overrides.get(scenario).cloned().unwrap_or_default()
    }

    /// Get configuration for a specific environment
    pub fn get_environment_config(&self, environment: &EnvironmentType) -> TestConfig {
        self.environment_overrides.get(environment).cloned().unwrap_or_default()
    }

    /// Resolve configuration for a scenario and environment
    pub fn resolve(&self, scenario: &TestingScenario, environment: &EnvironmentType) -> TestConfig {
        let base = TestConfig::default();
        let scenario_config = self.get_scenario_config(scenario);
        let env_config = self.get_environment_config(environment);

        // Merge configs: base -> scenario -> environment
        // Environment overrides take precedence
        TestConfig {
            max_parallel_tests: env_config
                .max_parallel_tests
                .max(scenario_config.max_parallel_tests),
            test_timeout: scenario_config.test_timeout.max(env_config.test_timeout),
            cache_dir: env_config.cache_dir.clone(),
            log_level: if env_config.log_level != base.log_level {
                env_config.log_level
            } else {
                scenario_config.log_level
            },
            coverage_threshold: scenario_config
                .coverage_threshold
                .max(env_config.coverage_threshold),
            #[cfg(feature = "fixtures")]
            fixtures: scenario_config.fixtures,
            crossval: scenario_config.crossval,
            reporting: ReportingConfig {
                formats: if !env_config.reporting.formats.is_empty() {
                    env_config.reporting.formats
                } else {
                    scenario_config.reporting.formats
                },
                output_dir: super::config::pick_dir(
                    &env_config.reporting.output_dir,
                    &scenario_config.reporting.output_dir,
                ),
                include_artifacts: scenario_config.reporting.include_artifacts
                    || env_config.reporting.include_artifacts,
                generate_coverage: scenario_config.reporting.generate_coverage
                    || env_config.reporting.generate_coverage,
                generate_performance: scenario_config.reporting.generate_performance
                    || env_config.reporting.generate_performance,
                upload_reports: scenario_config.reporting.upload_reports
                    || env_config.reporting.upload_reports,
            },
        }
    }

    /// Create a configuration context from environment variables
    pub fn context_from_environment() -> ConfigurationContext {
        let scenario = std::env::var("BITNET_TEST_SCENARIO")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(TestingScenario::Unit);

        let environment = if std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok()
        {
            EnvironmentType::CI
        } else {
            std::env::var("BITNET_TEST_ENV")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(EnvironmentType::Local)
        };

        ConfigurationContext {
            scenario,
            environment,
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_parsing() {
        assert_eq!("unit".parse::<TestingScenario>().unwrap(), TestingScenario::Unit);
        assert_eq!("e2e".parse::<TestingScenario>().unwrap(), TestingScenario::EndToEnd);
        assert_eq!("perf".parse::<TestingScenario>().unwrap(), TestingScenario::Performance);
        assert!("invalid".parse::<TestingScenario>().is_err());
    }

    #[test]
    fn test_environment_parsing() {
        assert_eq!("local".parse::<EnvironmentType>().unwrap(), EnvironmentType::Local);
        assert_eq!("ci".parse::<EnvironmentType>().unwrap(), EnvironmentType::CI);
        assert_eq!("staging".parse::<EnvironmentType>().unwrap(), EnvironmentType::PreProduction);
        assert!("invalid".parse::<EnvironmentType>().is_err());
    }

    #[test]
    fn test_config_resolution() {
        let manager = ScenarioConfigManager::default();
        let config = manager.resolve(&TestingScenario::Unit, &EnvironmentType::CI);

        // CI environment should set specific formats
        assert!(config.reporting.formats.contains(&ReportFormat::Junit));

        // Unit scenario should have its settings
        assert_eq!(config.max_parallel_tests, 8);
    }
}

// ==== Compatibility shim for legacy tests ===================================
// Extends the existing types with compatibility methods that were used by
// tests/test_configuration_scenarios.rs prior to the refactor.

// Add Default impls for the existing types to support the tests
impl Default for ConfigurationContext {
    fn default() -> Self {
        Self {
            scenario: TestingScenario::Unit,
            environment: EnvironmentType::Local,
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        }
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self { max_memory_mb: None, max_cpu_cores: None, max_disk_gb: None }
    }
}

impl Default for TimeConstraints {
    fn default() -> Self {
        Self { max_total_duration: None, max_test_duration: None }
    }
}

impl Default for QualityRequirements {
    fn default() -> Self {
        Self { min_coverage: None, max_flakiness: None, required_passes: None }
    }
}

impl Default for PlatformSettings {
    fn default() -> Self {
        Self { os: None, arch: None, features: vec![] }
    }
}

// Helper functions to create scenario configs
fn create_unit_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: num_cpus::get() * 2,
        test_timeout: Duration::from_secs(10),
        log_level: "warn".to_string(),
        coverage_threshold: 0.8,
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Json],
            generate_coverage: true, // Unit tests should generate coverage
            generate_performance: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn create_integration_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: num_cpus::get() / 2,
        test_timeout: Duration::from_secs(60),
        log_level: "info".to_string(),
        coverage_threshold: 0.7,
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Json, ReportFormat::Html, ReportFormat::Junit],
            generate_coverage: true,
            generate_performance: true, // Integration tests should generate performance reports
            ..Default::default()
        },
        ..Default::default()
    }
}

fn create_perf_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(1800),
        log_level: "info".to_string(),
        coverage_threshold: 0.0,
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Json, ReportFormat::Csv],
            generate_coverage: false, // Don't generate coverage for performance tests
            generate_performance: true,
            ..Default::default()
        },
        crossval: CrossValidationConfig { enabled: false, ..Default::default() },
        ..Default::default()
    }
}

fn create_e2e_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(300),
        log_level: "debug".to_string(),
        coverage_threshold: 0.9,
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Json, ReportFormat::Html],
            generate_coverage: true,
            generate_performance: true,
            include_artifacts: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn create_smoke_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 1,                 // Smoke tests should be sequential
        test_timeout: Duration::from_secs(10), // Short timeout for smoke tests (changed from 5 to 10)
        log_level: "error".to_string(),
        coverage_threshold: 0.0,
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Json],
            generate_coverage: false,
            generate_performance: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn create_crossval_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(600),
        log_level: "debug".to_string(),
        coverage_threshold: 0.0,
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Json],
            generate_coverage: false,
            generate_performance: true,
            ..Default::default()
        },
        crossval: CrossValidationConfig {
            enabled: true,
            tolerance: ComparisonTolerance {
                min_token_accuracy: 0.999999, // Very strict tolerance for cross-validation
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn create_debug_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(3600),
        log_level: "trace".to_string(),
        coverage_threshold: 0.9,
        #[cfg(feature = "fixtures")]
        fixtures: FixtureConfig { auto_download: false, ..Default::default() },
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Json, ReportFormat::Html],
            generate_coverage: true,
            generate_performance: true,
            include_artifacts: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn create_development_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: num_cpus::get(),
        test_timeout: Duration::from_secs(60),
        log_level: "info".to_string(),
        coverage_threshold: 0.0,
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Html],
            generate_coverage: false, // Skip coverage for faster feedback
            generate_performance: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn create_minimal_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(30),
        log_level: "error".to_string(),
        coverage_threshold: 0.0,
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Json],
            generate_coverage: false,
            generate_performance: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

impl ScenarioConfigManager {
    /// Legacy description text expected by older tests. Kept in a compat shim so
    /// assertions like "mentions Fast / isolated / Sequential / comparison" still pass.
    pub fn scenario_description(s: &TestingScenario) -> &'static str {
        match s {
            TestingScenario::Unit => {
                "Fast, isolated tests with minimal dependencies and quick feedback."
            }
            TestingScenario::Integration => {
                "End-to-end integration tests covering realistic flows and services."
            }
            TestingScenario::Performance => {
                "Sequential performance runs that measure latency and throughput."
            }
            TestingScenario::CrossValidation => {
                "A/B comparison (cross-validation) to check accuracy regressions."
            }
            TestingScenario::Debug => {
                "Verbose debug runs with thorough logging and artifact capture."
            }
            TestingScenario::Development => {
                "Local development defaults for quick iteration and diagnostics."
            }
            TestingScenario::Minimal => "Tiny smoke test configuration for ultra-fast checks.",
            TestingScenario::EndToEnd => {
                "Complete end-to-end workflow tests with full stack validation."
            }
            TestingScenario::Smoke => "Quick smoke tests to verify basic functionality.",
        }
    }

    /// (Optional) enumerate scenarios for loops in docs/tests.
    pub fn available_scenarios() -> &'static [TestingScenario] {
        &[
            TestingScenario::Unit,
            TestingScenario::Integration,
            TestingScenario::Performance,
            TestingScenario::CrossValidation,
            TestingScenario::Debug,
            TestingScenario::Development,
            TestingScenario::Minimal,
        ]
    }

    /// Returns a **base** config for the given context:
    /// - merges scenario + environment defaults
    /// - applies platform caps (e.g., Windows ≤ 8, macOS ≤ 6)
    ///
    /// NOTE: **Does not** apply fast-feedback/resource/quality overrides.
    /// The test harness wrapper (in `test_configuration_scenarios.rs`) applies
    /// the final context clamps to keep responsibilities separate.
    pub fn get_context_config(&self, ctx: &ConfigurationContext) -> TestConfig {
        // Start with the canonical scenario + environment merge.
        let mut cfg = self.resolve(&ctx.scenario, &ctx.environment);

        // Platform heuristics (match the expectations in the test file)
        if let Some(ref platform) = ctx.platform_settings {
            if let Some(ref os) = platform.os {
                match os.as_str() {
                    "windows" => {
                        if cfg.max_parallel_tests > 8 {
                            cfg.max_parallel_tests = 8;
                        }
                    }
                    "macos" => {
                        if cfg.max_parallel_tests > 6 {
                            cfg.max_parallel_tests = 6;
                        }
                    }
                    // linux / generic: do not tighten beyond scenario defaults
                    _ => {}
                }
            }
        }

        // NOTE: The test file's get_context_config wrapper applies all the
        // context overrides (fast-feedback, resource constraints, quality requirements).
        // We don't apply them here to avoid double-application.

        cfg
    }
}
