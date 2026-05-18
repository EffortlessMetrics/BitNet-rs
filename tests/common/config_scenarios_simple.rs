#[cfg(feature = "fixtures")]
use super::config::FixtureConfig;
/// Simplified configuration scenarios that work without heavy dependencies
use super::config::{CrossValidationConfig, ReportFormat, ReportingConfig, TestConfig};
use std::time::Duration;

fn scenario_config(
    max_parallel_tests: usize,
    timeout_secs: u64,
    log_level: &str,
    coverage_threshold: f64,
    formats: Vec<ReportFormat>,
) -> TestConfig {
    TestConfig {
        max_parallel_tests,
        test_timeout: Duration::from_secs(timeout_secs),
        log_level: log_level.to_string(),
        coverage_threshold,
        #[cfg(feature = "fixtures")]
        fixtures: FixtureConfig::default(),
        crossval: CrossValidationConfig::default(),
        reporting: ReportingConfig { formats, ..Default::default() },
        ..Default::default()
    }
}

/// Create a simple test config without FastConfigBuilder
pub fn create_unit_config() -> TestConfig {
    scenario_config(8, 30, "warn", 0.8, vec![ReportFormat::Json, ReportFormat::Html])
}

pub fn create_integration_config() -> TestConfig {
    scenario_config(
        4,
        120,
        "info",
        0.8,
        vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Junit],
    )
}

pub fn create_e2e_config() -> TestConfig {
    scenario_config(
        2,
        300,
        "info",
        0.8,
        vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Junit, ReportFormat::Markdown],
    )
}

pub fn create_smoke_config() -> TestConfig {
    scenario_config(1, 10, "error", 0.0, vec![ReportFormat::Json])
}

pub fn create_perf_config() -> TestConfig {
    scenario_config(
        1,
        600,
        "info",
        0.0,
        vec![ReportFormat::Html, ReportFormat::Json, ReportFormat::Markdown],
    )
}
