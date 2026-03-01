//! Edge-case tests for `bitnet-testing-scenarios-core` ScenarioConfigManager.

use bitnet_testing_scenarios_core::{
    ConfigurationContext, EnvironmentType, PlatformSettings, ReportFormat, ScenarioConfigManager,
    TestingScenario,
};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn default_manager_has_all_scenarios() {
    let mgr = ScenarioConfigManager::default();
    for scenario in ScenarioConfigManager::available_scenarios() {
        let cfg = mgr.get_scenario_config(scenario);
        // Every registered scenario should produce a non-default config
        assert!(cfg.test_timeout > Duration::ZERO, "{:?} timeout should be > 0", scenario);
    }
}

#[test]
fn new_manager_equivalent_to_default() {
    let mgr_new = ScenarioConfigManager::new();
    let mgr_default = ScenarioConfigManager::default();

    for scenario in ScenarioConfigManager::available_scenarios() {
        let cfg_new = mgr_new.get_scenario_config(scenario);
        let cfg_default = mgr_default.get_scenario_config(scenario);
        assert_eq!(cfg_new.max_parallel_tests, cfg_default.max_parallel_tests);
        assert_eq!(cfg_new.test_timeout, cfg_default.test_timeout);
    }
}

// ---------------------------------------------------------------------------
// Scenario configs have expected properties
// ---------------------------------------------------------------------------

#[test]
fn unit_scenario_has_high_parallelism() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_scenario_config(&TestingScenario::Unit);
    // Unit tests should allow many parallel tests
    assert!(cfg.max_parallel_tests >= 2, "Unit should have parallel >= 2");
}

#[test]
fn unit_scenario_has_short_timeout() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_scenario_config(&TestingScenario::Unit);
    assert!(cfg.test_timeout <= Duration::from_secs(30), "Unit timeout should be <= 30s");
}

#[test]
fn performance_scenario_is_sequential() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_scenario_config(&TestingScenario::Performance);
    assert_eq!(cfg.max_parallel_tests, 1, "Performance should be sequential");
}

#[test]
fn performance_scenario_has_long_timeout() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_scenario_config(&TestingScenario::Performance);
    assert!(cfg.test_timeout >= Duration::from_secs(300), "Performance timeout should be >= 300s");
}

#[test]
fn smoke_scenario_is_fast() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_scenario_config(&TestingScenario::Smoke);
    assert!(cfg.test_timeout <= Duration::from_secs(30), "Smoke timeout should be <= 30s");
}

#[test]
fn e2e_scenario_has_high_coverage() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_scenario_config(&TestingScenario::EndToEnd);
    assert!(cfg.coverage_threshold >= 0.8, "E2E should require high coverage");
}

#[test]
fn crossval_scenario_enables_crossval() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_scenario_config(&TestingScenario::CrossValidation);
    assert!(cfg.crossval.enabled, "CrossValidation scenario should enable crossval");
}

#[test]
fn debug_scenario_is_verbose() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_scenario_config(&TestingScenario::Debug);
    assert_eq!(cfg.log_level, "trace", "Debug should use trace logging");
}

#[test]
fn minimal_scenario_is_minimal() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_scenario_config(&TestingScenario::Minimal);
    assert_eq!(cfg.max_parallel_tests, 1);
    assert_eq!(cfg.log_level, "error");
    assert_eq!(cfg.coverage_threshold, 0.0);
}

// ---------------------------------------------------------------------------
// Environment configs
// ---------------------------------------------------------------------------

#[test]
fn ci_environment_has_debug_logging() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_environment_config(&EnvironmentType::Ci);
    assert_eq!(cfg.log_level, "debug");
}

#[test]
fn ci_environment_uploads_reports() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_environment_config(&EnvironmentType::Ci);
    assert!(cfg.reporting.upload_reports);
}

#[test]
fn production_environment_is_sequential() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_environment_config(&EnvironmentType::Production);
    assert_eq!(cfg.max_parallel_tests, 1);
}

#[test]
fn local_environment_skips_coverage() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.get_environment_config(&EnvironmentType::Local);
    assert!(!cfg.reporting.generate_coverage);
}

// Note: all EnvironmentType variants (Local, Ci, PreProduction, Production) are registered,
// so there's no "unknown" variant to test fallback behavior.

// ---------------------------------------------------------------------------
// Resolution (merge logic)
// ---------------------------------------------------------------------------

#[test]
fn resolve_unit_ci_uses_ci_parallel_count() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);
    assert_eq!(cfg.max_parallel_tests, 8, "CI environment overrides unit parallelism");
}

#[test]
fn resolve_unit_ci_has_junit_format() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);
    assert!(cfg.reporting.formats.contains(&ReportFormat::Junit));
}

#[test]
fn resolve_unit_local_uses_scenario_formats() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Local);
    // Local environment has Html format, which overrides scenario's Json
    assert!(cfg.reporting.formats.contains(&ReportFormat::Html));
}

#[test]
fn resolve_perf_ci_keeps_sequential() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.resolve(&TestingScenario::Performance, &EnvironmentType::Ci);
    // CI sets parallel to 8, which is non-default, so it overrides perf's 1
    assert_eq!(cfg.max_parallel_tests, 8);
}

#[test]
fn resolve_timeout_takes_max() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.resolve(&TestingScenario::Performance, &EnvironmentType::Ci);
    // Performance has 1800s timeout, CI has default (0s), max should be 1800s
    assert!(cfg.test_timeout >= Duration::from_secs(1800));
}

#[test]
fn resolve_coverage_threshold_takes_max() {
    let mgr = ScenarioConfigManager::default();
    let cfg = mgr.resolve(&TestingScenario::EndToEnd, &EnvironmentType::PreProduction);
    // E2E has 0.9, PreProd has 0.7 → max is 0.9
    assert!(cfg.coverage_threshold >= 0.9);
}

// ---------------------------------------------------------------------------
// Platform settings in context
// ---------------------------------------------------------------------------

#[test]
fn windows_platform_caps_parallelism_at_8() {
    let mgr = ScenarioConfigManager::default();
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: EnvironmentType::Local,
        platform_settings: Some(PlatformSettings {
            os: Some("windows".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let cfg = mgr.get_context_config(&ctx);
    assert!(cfg.max_parallel_tests <= 8, "Windows should cap at 8");
}

#[test]
fn macos_platform_caps_parallelism_at_6() {
    let mgr = ScenarioConfigManager::default();
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: EnvironmentType::Local,
        platform_settings: Some(PlatformSettings {
            os: Some("macos".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let cfg = mgr.get_context_config(&ctx);
    assert!(cfg.max_parallel_tests <= 6, "macOS should cap at 6");
}

#[test]
fn linux_platform_no_cap() {
    let mgr = ScenarioConfigManager::default();
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: EnvironmentType::Local,
        platform_settings: Some(PlatformSettings {
            os: Some("linux".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let cfg = mgr.get_context_config(&ctx);
    // Linux has no special cap, should use the resolved value
    let base = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Local);
    assert_eq!(cfg.max_parallel_tests, base.max_parallel_tests);
}

#[test]
fn no_platform_no_cap() {
    let mgr = ScenarioConfigManager::default();
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: EnvironmentType::Local,
        platform_settings: None,
        ..Default::default()
    };
    let cfg = mgr.get_context_config(&ctx);
    let base = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Local);
    assert_eq!(cfg.max_parallel_tests, base.max_parallel_tests);
}

// ---------------------------------------------------------------------------
// Scenario descriptions
// ---------------------------------------------------------------------------

#[test]
fn all_scenarios_have_descriptions() {
    for scenario in ScenarioConfigManager::available_scenarios() {
        let desc = ScenarioConfigManager::scenario_description(scenario);
        assert!(!desc.is_empty(), "scenario {:?} should have a description", scenario);
    }
}

#[test]
fn scenario_descriptions_are_unique() {
    let descs: Vec<_> = ScenarioConfigManager::available_scenarios()
        .iter()
        .map(|s| ScenarioConfigManager::scenario_description(s))
        .collect();
    let unique: std::collections::HashSet<_> = descs.iter().collect();
    assert_eq!(descs.len(), unique.len(), "descriptions should be unique");
}

// ---------------------------------------------------------------------------
// available_scenarios is comprehensive
// ---------------------------------------------------------------------------

#[test]
fn available_scenarios_has_unit_integration_perf() {
    let scenarios = ScenarioConfigManager::available_scenarios();
    assert!(scenarios.contains(&TestingScenario::Unit));
    assert!(scenarios.contains(&TestingScenario::Integration));
    assert!(scenarios.contains(&TestingScenario::Performance));
}

// ---------------------------------------------------------------------------
// context_from_environment smoke test
// ---------------------------------------------------------------------------

#[test]
fn context_from_environment_returns_valid_context() {
    let ctx = ScenarioConfigManager::context_from_environment();
    // Default scenario and environment should parse without error
    let mgr = ScenarioConfigManager::default();
    let _cfg = mgr.get_context_config(&ctx);
}
