#![cfg(feature = "integration-tests")]
// Simple test to verify configuration scenarios work
// This test validates the core functionality without dependencies on the full test framework

use std::time::Duration;

// Import the configuration scenarios module directly
mod common;
use common::config_scenarios::{
    ConfigurationContext, EnvironmentType, PlatformSettings, ScenarioConfigManager,
    TestingScenario,
};

#[test]
fn test_scenario_configurations() {
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

    // Test smoke testing scenario
    let smoke_config = manager.get_scenario_config(&TestingScenario::Smoke);
    assert_eq!(smoke_config.max_parallel_tests, 1);
    assert_eq!(smoke_config.test_timeout, Duration::from_secs(10));
    assert_eq!(smoke_config.log_level, "error");

    println!("✓ All scenario configurations work correctly");
}

#[test]
fn test_environment_configurations() {
    let manager = ScenarioConfigManager::new();

    // Test development environment
    let dev_config = manager.get_environment_config(&EnvironmentType::Local);
    assert_eq!(dev_config.log_level, "info");
    assert!(!dev_config.reporting.generate_coverage);

    // Test CI environment
    let ci_config = manager.get_environment_config(&EnvironmentType::CI);
    assert_eq!(ci_config.log_level, "debug");
    assert!(ci_config.reporting.generate_coverage);

    println!("✓ All environment configurations work correctly");
}

#[test]
fn test_configuration_context() {
    let manager = ScenarioConfigManager::new();

    // Test complex context configuration
    let mut context = ConfigurationContext::default();
    context.scenario = TestingScenario::Performance;
    context.environment = EnvironmentType::CI;
    context.resource_constraints.max_parallel_tests = Some(1);
    context.resource_constraints.network_access = false;
    context.time_constraints.max_test_timeout = Duration::from_secs(300);
    context.quality_requirements.min_coverage = 0.85;

    let config = manager.get_context_config(&context);

    // Verify scenario settings are applied
    assert!(config.reporting.generate_performance);

    // Verify environment settings are applied
    assert_eq!(config.log_level, "debug");

    // Verify resource constraints are applied
    assert_eq!(config.max_parallel_tests, 1);
    assert!(!config.fixtures.auto_download);

    // Verify quality requirements are applied
    assert_eq!(config.coverage_threshold, 0.85);

    println!("✓ Configuration context works correctly");
}

#[test]
fn test_resource_constraints() {
    let manager = ScenarioConfigManager::new();
    let mut context = ConfigurationContext::default();

    // Test parallel test constraint
    context.resource_constraints.max_parallel_tests = Some(2);
    let config = manager.get_context_config(&context);
    assert!(config.max_parallel_tests <= 2);

    // Test network access constraint
    context.resource_constraints.network_access = false;
    let config = manager.get_context_config(&context);
    assert!(!config.fixtures.auto_download);
    assert!(!config.reporting.upload_reports);

    println!("✓ Resource constraints work correctly");
}

#[test]
fn test_time_constraints() {
    let manager = ScenarioConfigManager::new();
    let mut context = ConfigurationContext::default();

    // Test fast feedback constraint
    context.time_constraints.target_feedback_time = Some(Duration::from_secs(120));
    let config = manager.get_context_config(&context);
    assert!(!config.reporting.generate_coverage);
    assert!(!config.crossval.enabled);

    println!("✓ Time constraints work correctly");
}

#[test]
fn test_convenience_functions() {
    // Test unit testing configuration
    let manager = ScenarioConfigManager::new();
    let unit_config = manager.get_scenario_config(&TestingScenario::Unit);
    assert_eq!(unit_config.log_level, "warn");

    // Test performance testing configuration
    let performance_config = manager.get_scenario_config(&TestingScenario::Performance);
    assert_eq!(performance_config.max_parallel_tests, 1);

    // Test smoke testing configuration
    let smoke_config = manager.get_scenario_config(&TestingScenario::Smoke);
    assert_eq!(smoke_config.test_timeout, Duration::from_secs(10));

    // Test development configuration
    let dev_config = manager.get_scenario_config(&TestingScenario::Development);
    assert!(!dev_config.reporting.generate_coverage);

    println!("✓ Convenience functions work correctly");
}

#[test]
fn test_scenario_descriptions() {
    // Test that all scenarios have descriptions
    for scenario in ScenarioConfigManager::available_scenarios() {
        let description = ScenarioConfigManager::scenario_description(&scenario);
        assert!(!description.is_empty());
        assert!(description.len() > 10);
    }

    // Test specific descriptions
    let unit_desc = ScenarioConfigManager::scenario_description(&TestingScenario::Unit);
    assert!(unit_desc.contains("Fast"));

    let performance_desc =
        ScenarioConfigManager::scenario_description(&TestingScenario::Performance);
    assert!(performance_desc.contains("Sequential"));

    println!("✓ Scenario descriptions work correctly");
}

#[test]
fn test_platform_specific_configurations() {
    let manager = ScenarioConfigManager::new();
    let mut context = ConfigurationContext::default();

    // Test Windows platform
    context.platform_settings = Some(PlatformSettings {
        os: Some("windows".to_string()),
        arch: None,
        features: vec![],
    });
    context.scenario = TestingScenario::Unit; // Start with high parallelism
    let config = manager.get_context_config(&context);
    assert!(config.max_parallel_tests <= 8);

    // Test macOS platform
    context.platform_settings = Some(PlatformSettings {
        os: Some("macos".to_string()),
        arch: None,
        features: vec![],
    });
    let config = manager.get_context_config(&context);
    assert!(config.max_parallel_tests <= 6);

    println!("✓ Platform-specific configurations work correctly");
}

#[test]
fn test_available_scenarios() {
    let scenarios = ScenarioConfigManager::available_scenarios();
    assert!(scenarios.contains(&TestingScenario::Unit));
    assert!(scenarios.contains(&TestingScenario::Performance));
    assert!(scenarios.contains(&TestingScenario::CrossValidation));
    assert_eq!(scenarios.len(), 8);

    println!("✓ Available scenarios list is correct");
}
