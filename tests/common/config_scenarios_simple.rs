#[cfg(feature = "fixtures")]
use super::config::FixtureConfig;
/// Simplified configuration scenarios that work without heavy dependencies
use super::config::{CrossValidationConfig, ReportFormat, ReportingConfig, TestConfig};
use std::time::Duration;

/// Create a simple test config without FastConfigBuilder
pub fn create_unit_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 8,
        test_timeout: Duration::from_secs(30),
        log_level: "warn".to_string(),
        coverage_threshold: 0.8,
        #[cfg(feature = "fixtures")]
        fixtures: FixtureConfig::default(),
        crossval: CrossValidationConfig::default(),
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Json, ReportFormat::Html],
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn create_integration_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 4,
        test_timeout: Duration::from_secs(120),
        log_level: "info".to_string(),
        coverage_threshold: 0.8,
        #[cfg(feature = "fixtures")]
        fixtures: FixtureConfig::default(),
        crossval: CrossValidationConfig::default(),
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Junit],
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn create_e2e_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 2,
        test_timeout: Duration::from_secs(300),
        log_level: "info".to_string(),
        coverage_threshold: 0.8,
        #[cfg(feature = "fixtures")]
        fixtures: FixtureConfig::default(),
        crossval: CrossValidationConfig::default(),
        reporting: ReportingConfig {
            formats: vec![
                ReportFormat::Html,
                ReportFormat::Json,
                ReportFormat::Junit,
                ReportFormat::Markdown,
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn create_smoke_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(10),
        log_level: "error".to_string(),
        coverage_threshold: 0.0,
        #[cfg(feature = "fixtures")]
        fixtures: FixtureConfig::default(),
        crossval: CrossValidationConfig::default(),
        reporting: ReportingConfig { formats: vec![ReportFormat::Json], ..Default::default() },
        ..Default::default()
    }
}

pub fn create_perf_config() -> TestConfig {
    TestConfig {
        max_parallel_tests: 1,
        test_timeout: Duration::from_secs(600),
        log_level: "info".to_string(),
        coverage_threshold: 0.0,
        #[cfg(feature = "fixtures")]
        fixtures: FixtureConfig::default(),
        crossval: CrossValidationConfig::default(),
        reporting: ReportingConfig {
            formats: vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Markdown],
            ..Default::default()
        },
        ..Default::default()
    }
}
