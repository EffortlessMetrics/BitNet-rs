use super::config::{
    ComparisonTolerance, CrossValidationConfig, FixtureConfig, ReportFormat, ReportingConfig,
    TestConfig,
};

#[cfg(feature = "fixtures")]
use super::fast_config::{FastConfigBuilder, SpeedProfile};

#[cfg(not(feature = "fixtures"))]
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

/// Environment types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvironmentType {
    Local,
    CI,
    PreProduction,
    Production,
}

/// Manager for scenario-based test configurations
#[derive(Debug, Clone)]
pub struct ScenarioConfigManager {
    scenario_overrides: HashMap<TestingScenario, TestConfig>,
    environment_overrides: HashMap<EnvironmentType, TestConfig>,
}

impl ScenarioConfigManager {
    /// Register default scenario-specific configurations
    pub fn register_default_scenarios(&mut self) {
        // Configure all scenarios with proper feature gating
        self.register_unit_scenario();
        self.register_integration_scenario();
        self.register_e2e_scenario();
        self.register_performance_scenario();
        self.register_crossval_scenario();
        self.register_smoke_scenario();
        self.register_dev_scenario();
        self.register_debug_scenario();
        self.register_minimal_scenario();
    }

    fn register_unit_scenario(&mut self) {
        #[cfg(feature = "fixtures")]
        let config = FastConfigBuilder::with_profile(SpeedProfile::Fast)
            .max_parallel(8)
            .timeout(Duration::from_secs(30))
            .log_level("warn")
            .coverage(true)
            .performance(false)
            .crossval(false)
            .formats(vec![ReportFormat::Json, ReportFormat::Html])
            .build();

        #[cfg(not(feature = "fixtures"))]
        let config = create_unit_config();

        self.scenario_overrides.insert(TestingScenario::Unit, config);
    }

    fn register_integration_scenario(&mut self) {
        #[cfg(feature = "fixtures")]
        let config = FastConfigBuilder::with_profile(SpeedProfile::Balanced)
            .max_parallel(4)
            .timeout(Duration::from_secs(120))
            .log_level("info")
            .coverage(true)
            .performance(true)
            .crossval(false)
            .formats(vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Junit])
            .build();

        #[cfg(not(feature = "fixtures"))]
        let config = create_integration_config();

        self.scenario_overrides.insert(TestingScenario::Integration, config);
    }

    fn register_e2e_scenario(&mut self) {
        #[cfg(feature = "fixtures")]
        let mut config = FastConfigBuilder::with_profile(SpeedProfile::Thorough)
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

        #[cfg(not(feature = "fixtures"))]
        let mut config = create_e2e_config();

        // Override to include additional reporting
        config.reporting.include_logs = true;
        config.reporting.include_artifacts = true;

        self.scenario_overrides.insert(TestingScenario::EndToEnd, config);
    }

    fn register_performance_scenario(&mut self) {
        #[cfg(feature = "fixtures")]
        let config = FastConfigBuilder::with_profile(SpeedProfile::Thorough)
            .max_parallel(1)
            .timeout(Duration::from_secs(600))
            .log_level("info")
            .coverage(false)
            .performance(true)
            .crossval(false)
            .formats(vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Markdown])
            .build();

        #[cfg(not(feature = "fixtures"))]
        let config = create_perf_config();

        self.scenario_overrides.insert(TestingScenario::Performance, config);
    }

    fn register_crossval_scenario(&mut self) {
        let mut config = TestConfig::default();
        config.max_parallel_tests = 1;
        config.enable_cross_validation = true;
        config.cross_validation = CrossValidationConfig {
            enabled: true,
            reference_impl_path: None,
            model_directory: Some("models".into()),
            tolerance: ComparisonTolerance::Relative(1e-6),
            comparison_mode: super::config::ComparisonMode::Relative,
            save_failures: true,
            parallel_validation: false,
            batch_size: 1,
            numerical_tolerance: 1e-6,
        };
        config.reporting.formats = vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Markdown];

        self.scenario_overrides.insert(TestingScenario::CrossValidation, config);
    }

    fn register_smoke_scenario(&mut self) {
        #[cfg(feature = "fixtures")]
        let config = FastConfigBuilder::with_profile(SpeedProfile::Lightning)
            .max_parallel(1)
            .timeout(Duration::from_secs(10))
            .log_level("error")
            .coverage(false)
            .performance(false)
            .crossval(false)
            .formats(vec![ReportFormat::Json])
            .build();

        #[cfg(not(feature = "fixtures"))]
        let config = create_smoke_config();

        self.scenario_overrides.insert(TestingScenario::Smoke, config);
    }

    fn register_dev_scenario(&mut self) {
        #[cfg(feature = "fixtures")]
        let config = FastConfigBuilder::with_profile(SpeedProfile::Fast)
            .max_parallel(num_cpus::get())
            .timeout(Duration::from_secs(60))
            .log_level("info")
            .coverage(false)
            .performance(false)
            .crossval(false)
            .formats(vec![ReportFormat::Html])
            .build();

        #[cfg(not(feature = "fixtures"))]
        let config = TestConfig {
            max_parallel_tests: num_cpus::get(),
            test_timeout: Duration::from_secs(60),
            log_level: "info".to_string(),
            enable_coverage: false,
            enable_performance_tracking: false,
            enable_cross_validation: false,
            reporting: ReportingConfig {
                formats: vec![ReportFormat::Html],
                ..Default::default()
            },
            ..Default::default()
        };

        self.scenario_overrides.insert(TestingScenario::Development, config);
    }

    fn register_debug_scenario(&mut self) {
        let mut config = TestConfig::default();
        config.max_parallel_tests = 1;
        config.test_timeout = Duration::from_secs(3600);
        config.log_level = "trace".to_string();
        config.reporting.include_artifacts = true;
        config.reporting.formats = vec![ReportFormat::Html, ReportFormat::Json];

        self.scenario_overrides.insert(TestingScenario::Debug, config);
    }

    fn register_minimal_scenario(&mut self) {
        #[cfg(feature = "fixtures")]
        let config = FastConfigBuilder::with_profile(SpeedProfile::Lightning)
            .max_parallel(1)
            .timeout(Duration::from_secs(30))
            .log_level("error")
            .coverage(false)
            .performance(false)
            .crossval(false)
            .formats(vec![ReportFormat::Json])
            .build();

        #[cfg(not(feature = "fixtures"))]
        let config = TestConfig {
            max_parallel_tests: 1,
            test_timeout: Duration::from_secs(30),
            log_level: "error".to_string(),
            enable_coverage: false,
            enable_performance_tracking: false,
            enable_cross_validation: false,
            reporting: ReportingConfig {
                formats: vec![ReportFormat::Json],
                ..Default::default()
            },
            ..Default::default()
        };

        self.scenario_overrides.insert(TestingScenario::Minimal, config);
    }

    // Copy rest of implementation from original file...
}
