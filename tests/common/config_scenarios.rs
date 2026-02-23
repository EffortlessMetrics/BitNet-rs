#[cfg(feature = "fixtures")]
use crate::config::FixtureConfig;
use crate::config::{
    ComparisonTolerance, CrossValidationConfig, ReportFormat, ReportingConfig, TestConfig,
};

use bitnet_testing_policy_tests as policy;

pub use policy::{
    ConfigurationContext, EnvironmentType, PlatformSettings, QualityRequirements,
    ResourceConstraints, TestConfigProfile, TestingScenario, TimeConstraints,
};

/// Manager façade that keeps the legacy `tests`-crate API while delegating all policy
/// data to `bitnet_testing_policy_runtime`.
#[derive(Debug, Clone, Default)]
pub struct ScenarioConfigManager {
    inner: policy::ScenarioConfigManager,
}

impl ScenarioConfigManager {
    /// Create a new configuration manager.
    pub fn new() -> Self {
        let mut mgr = Self::default();
        mgr.register_default_scenarios();
        mgr.register_default_environments();
        mgr
    }

    /// Register default scenario-specific configurations.
    pub fn register_default_scenarios(&mut self) {
        self.inner.register_default_scenarios();
    }

    /// Register default environment-specific configurations.
    pub fn register_default_environments(&mut self) {
        self.inner.register_default_environments();
    }

    /// Get configuration for a specific scenario.
    pub fn get_scenario_config(&self, scenario: &TestingScenario) -> TestConfig {
        convert_test_config(self.inner.get_scenario_config(scenario))
    }

    /// Get configuration for a specific environment.
    pub fn get_environment_config(&self, environment: &EnvironmentType) -> TestConfig {
        convert_test_config(self.inner.get_environment_config(environment))
    }

    /// Resolve configuration for a scenario and environment.
    pub fn resolve(&self, scenario: &TestingScenario, environment: &EnvironmentType) -> TestConfig {
        convert_test_config(self.inner.resolve(scenario, environment))
    }

    /// Create a configuration context from environment variables.
    pub fn context_from_environment() -> ConfigurationContext {
        policy::ScenarioConfigManager::context_from_environment()
    }

    /// Return configuration for merged scenario/environment/context.
    pub fn get_context_config(&self, ctx: &ConfigurationContext) -> TestConfig {
        convert_test_config(self.inner.get_context_config(ctx))
    }

    /// Human-readable scenario description.
    pub fn scenario_description(s: &TestingScenario) -> &'static str {
        policy::ScenarioConfigManager::scenario_description(s)
    }

    /// List of active scenario identifiers.
    pub fn available_scenarios() -> &'static [TestingScenario] {
        policy::ScenarioConfigManager::available_scenarios()
    }
}

fn convert_report_format(value: policy::ReportFormat) -> ReportFormat {
    match value {
        policy::ReportFormat::Html => ReportFormat::Html,
        policy::ReportFormat::Json => ReportFormat::Json,
        policy::ReportFormat::Junit => ReportFormat::Junit,
        policy::ReportFormat::Markdown => ReportFormat::Markdown,
        policy::ReportFormat::Csv => ReportFormat::Csv,
    }
}

fn convert_reporting(profile: policy::ReportingProfile) -> ReportingConfig {
    ReportingConfig {
        output_dir: profile.output_dir,
        formats: profile.formats.into_iter().map(convert_report_format).collect(),
        include_artifacts: profile.include_artifacts,
        generate_coverage: profile.generate_coverage,
        generate_performance: profile.generate_performance,
        upload_reports: profile.upload_reports,
    }
}

#[cfg(feature = "fixtures")]
fn convert_fixture(profile: policy::FixtureProfile) -> FixtureConfig {
    FixtureConfig {
        auto_download: profile.auto_download,
        max_cache_size: profile.max_cache_size,
        cleanup_interval: profile.cleanup_interval,
        download_timeout: profile.download_timeout,
        base_url: profile.base_url,
        custom_fixtures: Default::default(),
    }
}

fn convert_tolerance(profile: policy::ComparisonToleranceProfile) -> ComparisonTolerance {
    ComparisonTolerance {
        min_token_accuracy: profile.min_token_accuracy,
        max_probability_divergence: profile.max_probability_divergence,
        max_performance_regression: profile.max_performance_regression,
        numerical_tolerance: profile.numerical_tolerance,
    }
}

fn convert_cross_validation(profile: policy::CrossValidationProfile) -> CrossValidationConfig {
    CrossValidationConfig {
        enabled: profile.enabled,
        tolerance: convert_tolerance(profile.tolerance),
        cpp_binary_path: profile.cpp_binary_path,
        test_cases: profile.test_cases,
        performance_comparison: profile.performance_comparison,
        accuracy_comparison: profile.accuracy_comparison,
    }
}

fn convert_test_config(profile: TestConfigProfile) -> TestConfig {
    let coverage_threshold = profile.coverage_threshold;
    let reporting = convert_reporting(profile.reporting);
    let crossval = convert_cross_validation(profile.crossval);
    TestConfig {
        max_parallel_tests: profile.max_parallel_tests,
        test_timeout: profile.test_timeout,
        cache_dir: profile.cache_dir,
        log_level: profile.log_level,
        coverage_threshold,
        #[cfg(feature = "fixtures")]
        fixtures: convert_fixture(profile.fixtures),
        crossval,
        reporting,
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
        assert_eq!("ci".parse::<EnvironmentType>().unwrap(), EnvironmentType::Ci);
        assert_eq!("staging".parse::<EnvironmentType>().unwrap(), EnvironmentType::PreProduction);
        assert!("invalid".parse::<EnvironmentType>().is_err());
    }

    #[test]
    fn test_config_resolution() {
        let manager = ScenarioConfigManager::default();
        let config = manager.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);

        assert!(config.reporting.formats.contains(&ReportFormat::Junit));
        assert_eq!(config.max_parallel_tests, 8);
        assert!(config.reporting.generate_coverage);
        assert!(config.reporting.generate_performance);
    }

    #[test]
    fn test_context_default() {
        let context = ConfigurationContext::default();
        assert_eq!(context.scenario, TestingScenario::Unit);
        assert_eq!(context.environment, EnvironmentType::Local);
    }
}
