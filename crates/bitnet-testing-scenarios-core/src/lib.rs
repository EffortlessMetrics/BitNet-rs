//! Scenario-driven configuration policies for test runtime orchestration.
//!
//! This crate keeps scenario and environment profile construction separate from the
//! full `bitnet_tests` crate so other frontends can compose the same behavior.
//!
//! The public types are intentionally small and stable. The `bitnet_tests`
//! compatibility layer converts these policy objects into the local `TestConfig`
//! model.

pub use bitnet_testing_profile::{
    ActiveContext, ExecutionEnvironment as EnvironmentType, TestingScenario,
};

/// Canonical alias maintained for readability in test-facing code.
pub type ScenarioType = TestingScenario;

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

/// Canonical scenario context coming from active environment detection.
#[derive(Debug, Clone)]
pub struct ConfigurationContext {
    pub scenario: TestingScenario,
    pub environment: EnvironmentType,
    pub resource_constraints: Option<ResourceConstraints>,
    pub time_constraints: Option<TimeConstraints>,
    pub quality_requirements: Option<QualityRequirements>,
    pub platform_settings: Option<PlatformSettings>,
}

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

/// Resource constraints used by legacy scenario merges.
#[derive(Debug, Clone, Default)]
pub struct ResourceConstraints {
    pub max_memory_mb: Option<usize>,
    pub max_cpu_cores: Option<usize>,
    pub max_disk_gb: Option<usize>,
}

/// Time constraints used by legacy scenario merges.
#[derive(Debug, Clone, Default)]
pub struct TimeConstraints {
    pub max_total_duration: Option<Duration>,
    pub max_test_duration: Option<Duration>,
}

/// Quality requirements used by legacy scenario merges.
#[derive(Debug, Clone, Default)]
pub struct QualityRequirements {
    pub min_coverage: Option<f64>,
    pub max_flakiness: Option<f64>,
    pub required_passes: Option<usize>,
}

/// Platform constraints used by legacy scenario merges.
#[derive(Debug, Clone, Default)]
pub struct PlatformSettings {
    pub os: Option<String>,
    pub arch: Option<String>,
    pub features: Vec<String>,
}

/// Lightweight representation of reporting configuration.
#[derive(Debug, Clone)]
pub struct ReportingProfile {
    pub output_dir: PathBuf,
    pub formats: Vec<ReportFormat>,
    pub include_artifacts: bool,
    pub generate_coverage: bool,
    pub generate_performance: bool,
    pub upload_reports: bool,
}

/// Canonical report format model used by policies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportFormat {
    Html,
    Json,
    Junit,
    Markdown,
    Csv,
}

#[derive(Debug, Clone)]
pub struct FixtureProfile {
    pub auto_download: bool,
    pub max_cache_size: u64,
    pub cleanup_interval: Duration,
    pub download_timeout: Duration,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ComparisonToleranceProfile {
    pub min_token_accuracy: f64,
    pub max_probability_divergence: f64,
    pub max_performance_regression: f64,
    pub numerical_tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct CrossValidationProfile {
    pub enabled: bool,
    pub tolerance: ComparisonToleranceProfile,
    pub cpp_binary_path: Option<PathBuf>,
    pub test_cases: Vec<String>,
    pub performance_comparison: bool,
    pub accuracy_comparison: bool,
}

/// Policy-level configuration that can be converted into test config objects.
#[derive(Debug, Clone)]
pub struct TestConfigProfile {
    pub max_parallel_tests: usize,
    pub test_timeout: Duration,
    pub cache_dir: PathBuf,
    pub log_level: String,
    pub coverage_threshold: f64,
    pub fixtures: FixtureProfile,
    pub crossval: CrossValidationProfile,
    pub reporting: ReportingProfile,
}

/// Manager for scenario-based policy defaults.
#[derive(Debug, Clone)]
pub struct ScenarioConfigManager {
    scenario_overrides: HashMap<TestingScenario, TestConfigProfile>,
    environment_overrides: HashMap<EnvironmentType, TestConfigProfile>,
}

impl Default for ReportingProfile {
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

impl Default for FixtureProfile {
    fn default() -> Self {
        Self {
            auto_download: true,
            max_cache_size: 10 * 1024 * 1024 * 1024,
            cleanup_interval: Duration::from_secs(24 * 60 * 60),
            download_timeout: Duration::from_secs(300),
            base_url: None,
        }
    }
}

impl Default for ComparisonToleranceProfile {
    fn default() -> Self {
        Self {
            min_token_accuracy: 0.999999,
            max_probability_divergence: 1e-6,
            max_performance_regression: 0.1,
            numerical_tolerance: 1e-6,
        }
    }
}

impl Default for CrossValidationProfile {
    fn default() -> Self {
        Self {
            enabled: false,
            tolerance: ComparisonToleranceProfile::default(),
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

impl Default for TestConfigProfile {
    fn default() -> Self {
        Self {
            max_parallel_tests: get_optimal_parallel_tests(),
            test_timeout: Duration::from_secs(300),
            cache_dir: PathBuf::from("tests/cache"),
            log_level: "info".to_string(),
            coverage_threshold: 0.9,
            fixtures: FixtureProfile::default(),
            crossval: CrossValidationProfile::default(),
            reporting: ReportingProfile::default(),
        }
    }
}

fn get_optimal_parallel_tests() -> usize {
    let cores = num_cpus::get();
    if is_ci_context() { (cores / 2).max(1) } else { cores.max(1) }
}

fn pick_dir(env: &Path, scenario: &Path) -> PathBuf {
    if !env.as_os_str().is_empty() { env.to_path_buf() } else { scenario.to_path_buf() }
}

fn is_ci_context() -> bool {
    std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
        || std::env::var("GITLAB_CI").is_ok()
        || std::env::var("JENKINS_URL").is_ok()
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
    pub fn new() -> Self {
        let mut mgr = Self::default();
        mgr.register_default_scenarios();
        mgr.register_default_environments();
        mgr
    }

    pub fn register_default_scenarios(&mut self) {
        self.initialize_scenario_overrides();
    }

    pub fn register_default_environments(&mut self) {
        self.initialize_environment_overrides();
    }

    fn initialize_scenario_overrides(&mut self) {
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

    fn initialize_environment_overrides(&mut self) {
        let ci_config = TestConfigProfile {
            max_parallel_tests: 8,
            log_level: "debug".to_string(),
            reporting: ReportingProfile {
                output_dir: PathBuf::from("/tmp/test-reports"),
                formats: vec![ReportFormat::Junit, ReportFormat::Html],
                generate_coverage: true,
                upload_reports: true,
                ..Default::default()
            },
            ..Default::default()
        };
        self.environment_overrides.insert(EnvironmentType::Ci, ci_config);

        let preprod_config = TestConfigProfile {
            coverage_threshold: 0.7,
            reporting: ReportingProfile { include_artifacts: true, ..Default::default() },
            ..Default::default()
        };
        self.environment_overrides.insert(EnvironmentType::PreProduction, preprod_config);

        let prod_config = TestConfigProfile {
            max_parallel_tests: 1,
            test_timeout: Duration::from_secs(60),
            log_level: "warn".to_string(),
            reporting: ReportingProfile {
                formats: vec![ReportFormat::Json, ReportFormat::Markdown],
                generate_coverage: true,
                generate_performance: true,
                ..Default::default()
            },
            ..Default::default()
        };
        self.environment_overrides.insert(EnvironmentType::Production, prod_config);

        let local_config = TestConfigProfile {
            log_level: "info".to_string(),
            reporting: ReportingProfile {
                formats: vec![ReportFormat::Html],
                generate_coverage: false,
                generate_performance: false,
                ..Default::default()
            },
            ..Default::default()
        };
        self.environment_overrides.insert(EnvironmentType::Local, local_config);
    }

    pub fn get_scenario_config(&self, scenario: &TestingScenario) -> TestConfigProfile {
        self.scenario_overrides.get(scenario).cloned().unwrap_or_default()
    }

    pub fn get_environment_config(&self, environment: &EnvironmentType) -> TestConfigProfile {
        self.environment_overrides.get(environment).cloned().unwrap_or_default()
    }

    pub fn resolve(
        &self,
        scenario: &TestingScenario,
        environment: &EnvironmentType,
    ) -> TestConfigProfile {
        let base = TestConfigProfile::default();
        let scenario_config = self.get_scenario_config(scenario);
        let env_config = self.get_environment_config(environment);

        TestConfigProfile {
            max_parallel_tests: if env_config.max_parallel_tests != base.max_parallel_tests {
                env_config.max_parallel_tests
            } else {
                scenario_config.max_parallel_tests
            },
            test_timeout: scenario_config.test_timeout.max(env_config.test_timeout),
            cache_dir: pick_dir(&env_config.cache_dir, &scenario_config.cache_dir),
            log_level: if env_config.log_level != base.log_level {
                env_config.log_level
            } else {
                scenario_config.log_level
            },
            coverage_threshold: scenario_config
                .coverage_threshold
                .max(env_config.coverage_threshold),
            fixtures: scenario_config.fixtures,
            crossval: scenario_config.crossval,
            reporting: ReportingProfile {
                formats: if !env_config.reporting.formats.is_empty() {
                    env_config.reporting.formats
                } else {
                    scenario_config.reporting.formats
                },
                output_dir: pick_dir(
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

    pub fn context_from_environment() -> ConfigurationContext {
        let active_context = ActiveContext::from_env();

        ConfigurationContext {
            scenario: active_context.scenario,
            environment: active_context.environment,
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        }
    }

    pub fn get_context_config(&self, ctx: &ConfigurationContext) -> TestConfigProfile {
        let mut cfg = self.resolve(&ctx.scenario, &ctx.environment);

        if let Some(ref platform) = ctx.platform_settings
            && let Some(ref os) = platform.os
        {
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
                _ => {}
            }
        }

        cfg
    }

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
}

fn create_unit_config() -> TestConfigProfile {
    let mut cfg = TestConfigProfile {
        max_parallel_tests: num_cpus::get() * 2,
        test_timeout: Duration::from_secs(10),
        log_level: "warn".to_string(),
        coverage_threshold: 0.8,
        ..TestConfigProfile::default()
    };
    cfg.reporting.formats = vec![ReportFormat::Json];
    cfg.reporting.generate_coverage = true;
    cfg.reporting.generate_performance = false;
    cfg
}

fn create_integration_config() -> TestConfigProfile {
    let mut cfg = TestConfigProfile {
        max_parallel_tests: num_cpus::get() / 2,
        test_timeout: Duration::from_secs(60),
        log_level: "info".to_string(),
        coverage_threshold: 0.7,
        ..TestConfigProfile::default()
    };
    cfg.reporting.formats = vec![ReportFormat::Json, ReportFormat::Html, ReportFormat::Junit];
    cfg.reporting.generate_coverage = true;
    cfg.reporting.generate_performance = true;
    cfg
}

fn create_perf_config() -> TestConfigProfile {
    let mut cfg = TestConfigProfile {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(1800),
        log_level: "info".to_string(),
        coverage_threshold: 0.0,
        ..TestConfigProfile::default()
    };
    cfg.reporting.formats = vec![ReportFormat::Json, ReportFormat::Csv];
    cfg.reporting.generate_coverage = false;
    cfg.reporting.generate_performance = true;
    cfg.crossval.enabled = false;
    cfg
}

fn create_e2e_config() -> TestConfigProfile {
    let mut cfg = TestConfigProfile {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(300),
        log_level: "debug".to_string(),
        coverage_threshold: 0.9,
        ..TestConfigProfile::default()
    };
    cfg.reporting.formats = vec![ReportFormat::Json, ReportFormat::Html];
    cfg.reporting.generate_coverage = true;
    cfg.reporting.generate_performance = true;
    cfg.reporting.include_artifacts = true;
    cfg
}

fn create_smoke_config() -> TestConfigProfile {
    let mut cfg = TestConfigProfile {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(10),
        log_level: "error".to_string(),
        coverage_threshold: 0.0,
        ..TestConfigProfile::default()
    };
    cfg.reporting.formats = vec![ReportFormat::Json];
    cfg.reporting.generate_coverage = false;
    cfg.reporting.generate_performance = false;
    cfg
}

fn create_crossval_config() -> TestConfigProfile {
    let mut cfg = TestConfigProfile {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(600),
        log_level: "debug".to_string(),
        coverage_threshold: 0.0,
        ..TestConfigProfile::default()
    };
    cfg.reporting.formats = vec![ReportFormat::Json];
    cfg.reporting.generate_coverage = false;
    cfg.reporting.generate_performance = true;
    cfg.crossval.enabled = true;
    cfg
}

fn create_debug_config() -> TestConfigProfile {
    let debug_timeout =
        std::env::var("BITNET_DEBUG_TIMEOUT_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(120);

    let mut cfg = TestConfigProfile {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(debug_timeout),
        log_level: "trace".to_string(),
        coverage_threshold: 0.9,
        ..TestConfigProfile::default()
    };

    cfg.fixtures.auto_download = false;
    cfg.reporting.formats = vec![ReportFormat::Json, ReportFormat::Html];
    cfg.reporting.generate_coverage = true;
    cfg.reporting.generate_performance = true;
    cfg.reporting.include_artifacts = true;
    cfg
}

fn create_development_config() -> TestConfigProfile {
    let mut cfg = TestConfigProfile {
        max_parallel_tests: num_cpus::get(),
        test_timeout: Duration::from_secs(60),
        log_level: "info".to_string(),
        coverage_threshold: 0.0,
        ..TestConfigProfile::default()
    };
    cfg.reporting.formats = vec![ReportFormat::Html];
    cfg.reporting.generate_coverage = false;
    cfg.reporting.generate_performance = false;
    cfg
}

fn create_minimal_config() -> TestConfigProfile {
    let mut cfg = TestConfigProfile {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(30),
        log_level: "error".to_string(),
        coverage_threshold: 0.0,
        ..TestConfigProfile::default()
    };
    cfg.reporting.formats = vec![ReportFormat::Json];
    cfg.reporting.generate_coverage = false;
    cfg.reporting.generate_performance = false;
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsing_and_descriptions() {
        assert_eq!("unit".parse::<TestingScenario>().unwrap(), TestingScenario::Unit);
        assert_eq!("e2e".parse::<TestingScenario>().unwrap(), TestingScenario::EndToEnd);
        assert_eq!("perf".parse::<TestingScenario>().unwrap(), TestingScenario::Performance);
        assert!("invalid".parse::<TestingScenario>().is_err());

        assert!(matches!("local".parse::<EnvironmentType>().unwrap(), EnvironmentType::Local));
    }

    #[test]
    fn test_config_resolution() {
        let manager = ScenarioConfigManager::default();
        let config = manager.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);

        assert!(config.reporting.formats.contains(&ReportFormat::Junit));
        assert_eq!(config.max_parallel_tests, 8);
    }
}
