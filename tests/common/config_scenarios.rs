use super::config::{
    ComparisonTolerance, CrossValidationConfig, ReportFormat, ReportingConfig,
    TestConfig,
};

// Always use simple configs for now since FastConfigBuilder isn't working
use super::config_scenarios_simple::*;

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
        let mut manager = Self {
            scenario_overrides: HashMap::new(),
            environment_overrides: HashMap::new(),
        };
        manager.initialize_scenario_overrides();
        manager.initialize_environment_overrides();
        manager
    }
}

impl ScenarioConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self::default()
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
        // For non-fixtures builds, use simple configs
        #[cfg(not(feature = "fixtures"))]
        {
            self.scenario_overrides.insert(TestingScenario::Unit, create_unit_config());
            self.scenario_overrides.insert(TestingScenario::Integration, create_integration_config());
            self.scenario_overrides.insert(TestingScenario::EndToEnd, create_e2e_config());
            self.scenario_overrides.insert(TestingScenario::Performance, create_perf_config());
            self.scenario_overrides.insert(TestingScenario::Smoke, create_smoke_config());
            
            // Development scenario
            self.scenario_overrides.insert(TestingScenario::Development, TestConfig {
                max_parallel_tests: num_cpus::get(),
                test_timeout: Duration::from_secs(60),
                log_level: "info".to_string(),
                coverage_threshold: 0.0,
                #[cfg(feature = "fixtures")]
                #[cfg(feature = "fixtures")]
                fixtures: FixtureConfig::default(),
                crossval: CrossValidationConfig::default(),
                reporting: ReportingConfig {
                    formats: vec![ReportFormat::Html],
                    ..Default::default()
                },
                ..Default::default()
            });
            
            // Minimal scenario
            self.scenario_overrides.insert(TestingScenario::Minimal, TestConfig {
                max_parallel_tests: 1,
                test_timeout: Duration::from_secs(30),
                log_level: "error".to_string(),
                coverage_threshold: 0.0,
                #[cfg(feature = "fixtures")]
                #[cfg(feature = "fixtures")]
                fixtures: FixtureConfig::default(),
                crossval: CrossValidationConfig::default(),
                reporting: ReportingConfig {
                    formats: vec![ReportFormat::Json],
                    ..Default::default()
                },
                ..Default::default()
            });
        }
        
        // With fixtures feature, FastConfigBuilder would be used
        #[cfg(feature = "fixtures")]
        {
            // This would use FastConfigBuilder when the feature is enabled
            // For now, we just use the same simple configs
            self.scenario_overrides.insert(TestingScenario::Unit, create_unit_config());
            self.scenario_overrides.insert(TestingScenario::Integration, create_integration_config());
            // ... etc
        }
        
        // CrossValidation scenario (same for both)
        let mut crossval_config = TestConfig::default();
        crossval_config.max_parallel_tests = 1;
        crossval_config.crossval = CrossValidationConfig {
            enabled: true,
            tolerance: ComparisonTolerance::default(),
            cpp_binary_path: None,
            test_cases: vec![],
            performance_comparison: true,
            accuracy_comparison: true,
        };
        crossval_config.reporting.formats = vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Markdown];
        self.scenario_overrides.insert(TestingScenario::CrossValidation, crossval_config);
        
        // Debug scenario (same for both)
        let mut debug_config = TestConfig::default();
        debug_config.max_parallel_tests = 1;
        debug_config.test_timeout = Duration::from_secs(3600);
        debug_config.log_level = "trace".to_string();
        debug_config.reporting.include_artifacts = true;
        debug_config.reporting.formats = vec![ReportFormat::Html, ReportFormat::Json];
        self.scenario_overrides.insert(TestingScenario::Debug, debug_config);
    }

    /// Initialize environment-specific configuration overrides
    fn initialize_environment_overrides(&mut self) {
        // CI environment
        let mut ci_config = TestConfig::default();
        ci_config.max_parallel_tests = 4;
        ci_config.reporting.output_dir = "/tmp/test-reports".into();
        ci_config.reporting.formats = vec![ReportFormat::Junit, ReportFormat::Html];
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
        prod_config.log_level = "error".to_string();
        self.environment_overrides.insert(EnvironmentType::Production, prod_config);
    }

    /// Get configuration for a specific scenario
    pub fn get_scenario_config(&self, scenario: &TestingScenario) -> TestConfig {
        self.scenario_overrides
            .get(scenario)
            .cloned()
            .unwrap_or_default()
    }

    /// Get configuration for a specific environment
    pub fn get_environment_config(&self, environment: &EnvironmentType) -> TestConfig {
        self.environment_overrides
            .get(environment)
            .cloned()
            .unwrap_or_default()
    }

    /// Resolve configuration for a scenario and environment
    pub fn resolve(&self, scenario: &TestingScenario, environment: &EnvironmentType) -> TestConfig {
        let base = TestConfig::default();
        let scenario_config = self.get_scenario_config(scenario);
        let env_config = self.get_environment_config(environment);
        
        // Merge configs: base -> scenario -> environment
        // Environment overrides take precedence
        TestConfig {
            max_parallel_tests: env_config.max_parallel_tests.max(scenario_config.max_parallel_tests),
            test_timeout: scenario_config.test_timeout.max(env_config.test_timeout),
            cache_dir: env_config.cache_dir.clone(),
            log_level: if env_config.log_level != base.log_level {
                env_config.log_level
            } else {
                scenario_config.log_level
            },
            coverage_threshold: scenario_config.coverage_threshold.max(env_config.coverage_threshold),
            #[cfg(feature = "fixtures")]
            fixtures: scenario_config.fixtures,
            crossval: scenario_config.crossval,
            reporting: ReportingConfig {
                formats: if !env_config.reporting.formats.is_empty() {
                    env_config.reporting.formats
                } else {
                    scenario_config.reporting.formats
                },
                output_dir: if !env_config.reporting.output_dir.as_os_str().is_empty() {
                    env_config.reporting.output_dir
                } else {
                    scenario_config.reporting.output_dir
                },
                include_artifacts: scenario_config.reporting.include_artifacts || env_config.reporting.include_artifacts,
                generate_coverage: scenario_config.reporting.generate_coverage || env_config.reporting.generate_coverage,
                generate_performance: scenario_config.reporting.generate_performance || env_config.reporting.generate_performance,
                upload_reports: scenario_config.reporting.upload_reports || env_config.reporting.upload_reports,
            },
        }
    }

    /// Create a configuration context from environment variables
    pub fn context_from_environment() -> ConfigurationContext {
        let scenario = std::env::var("BITNET_TEST_SCENARIO")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(TestingScenario::Unit);

        let environment = if std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok() {
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