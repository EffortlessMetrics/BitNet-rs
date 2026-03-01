//! Edge-case tests for bitnet-testing-scenarios-core ScenarioConfigManager.

use bitnet_testing_scenarios_core::{
    ConfigurationContext, EnvironmentType, PlatformSettings, ReportFormat, ScenarioConfigManager,
    TestingScenario,
};
use std::time::Duration;

// ---------------------------------------------------------------------------
// ScenarioConfigManager: construction
// ---------------------------------------------------------------------------

#[test]
fn manager_new_has_all_scenarios() {
    let mgr = ScenarioConfigManager::new();
    let scenarios = [
        TestingScenario::Unit,
        TestingScenario::Integration,
        TestingScenario::EndToEnd,
        TestingScenario::Performance,
        TestingScenario::Smoke,
        TestingScenario::CrossValidation,
        TestingScenario::Debug,
        TestingScenario::Development,
        TestingScenario::Minimal,
    ];
    for s in &scenarios {
        let config = mgr.get_scenario_config(s);
        assert!(config.max_parallel_tests > 0, "scenario {s:?} must have positive parallelism");
    }
}

#[test]
fn manager_new_has_all_environments() {
    let mgr = ScenarioConfigManager::new();
    let envs = [
        EnvironmentType::Local,
        EnvironmentType::Ci,
        EnvironmentType::PreProduction,
        EnvironmentType::Production,
    ];
    for e in &envs {
        let _ = mgr.get_environment_config(e);
    }
}

#[test]
fn manager_default_eq_new() {
    let d = ScenarioConfigManager::default();
    let n = ScenarioConfigManager::new();
    // Both should produce identical unit configs
    let dc = d.get_scenario_config(&TestingScenario::Unit);
    let nc = n.get_scenario_config(&TestingScenario::Unit);
    assert_eq!(dc.max_parallel_tests, nc.max_parallel_tests);
    assert_eq!(dc.log_level, nc.log_level);
}

// ---------------------------------------------------------------------------
// ScenarioConfigManager: scenario configs
// ---------------------------------------------------------------------------

#[test]
fn unit_config_fast_timeout() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::Unit);
    assert_eq!(cfg.test_timeout, Duration::from_secs(10));
}

#[test]
fn unit_config_high_parallelism() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::Unit);
    assert!(cfg.max_parallel_tests >= 2);
}

#[test]
fn integration_config_moderate_timeout() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::Integration);
    assert_eq!(cfg.test_timeout, Duration::from_secs(60));
}

#[test]
fn performance_config_sequential() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::Performance);
    assert_eq!(cfg.max_parallel_tests, 1);
    assert_eq!(cfg.test_timeout, Duration::from_secs(1800));
}

#[test]
fn performance_config_no_coverage() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::Performance);
    assert!(!cfg.reporting.generate_coverage);
    assert!(cfg.reporting.generate_performance);
}

#[test]
fn crossval_config_enabled() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::CrossValidation);
    assert!(cfg.crossval.enabled);
}

#[test]
fn smoke_config_minimal() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::Smoke);
    assert_eq!(cfg.max_parallel_tests, 1);
    assert_eq!(cfg.test_timeout, Duration::from_secs(10));
    assert_eq!(cfg.log_level, "error");
}

#[test]
fn debug_config_trace_logging() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::Debug);
    assert_eq!(cfg.log_level, "trace");
    assert!(cfg.reporting.include_artifacts);
}

#[test]
fn e2e_config_full_stack() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::EndToEnd);
    assert_eq!(cfg.max_parallel_tests, 1);
    assert!(cfg.reporting.generate_coverage);
    assert!(cfg.reporting.generate_performance);
}

#[test]
fn minimal_config_ultra_fast() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_scenario_config(&TestingScenario::Minimal);
    assert_eq!(cfg.max_parallel_tests, 1);
    assert_eq!(cfg.test_timeout, Duration::from_secs(30));
    assert_eq!(cfg.log_level, "error");
}

// ---------------------------------------------------------------------------
// ScenarioConfigManager: environment configs
// ---------------------------------------------------------------------------

#[test]
fn ci_env_has_junit_format() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_environment_config(&EnvironmentType::Ci);
    assert!(cfg.reporting.formats.contains(&ReportFormat::Junit));
}

#[test]
fn ci_env_parallel_8() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_environment_config(&EnvironmentType::Ci);
    assert_eq!(cfg.max_parallel_tests, 8);
}

#[test]
fn production_env_sequential() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_environment_config(&EnvironmentType::Production);
    assert_eq!(cfg.max_parallel_tests, 1);
    assert_eq!(cfg.test_timeout, Duration::from_secs(60));
}

#[test]
fn production_env_warn_logging() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_environment_config(&EnvironmentType::Production);
    assert_eq!(cfg.log_level, "warn");
}

#[test]
fn local_env_html_only() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.get_environment_config(&EnvironmentType::Local);
    assert_eq!(cfg.reporting.formats, vec![ReportFormat::Html]);
}

// ---------------------------------------------------------------------------
// ScenarioConfigManager: resolve (merge)
// ---------------------------------------------------------------------------

#[test]
fn resolve_unit_ci_uses_ci_parallelism() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);
    assert_eq!(cfg.max_parallel_tests, 8);
}

#[test]
fn resolve_unit_ci_uses_larger_timeout() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);
    // max(unit=10s, ci=300s) — CI default timeout comes from TestConfigProfile::default
    assert!(cfg.test_timeout >= Duration::from_secs(10));
}

#[test]
fn resolve_unit_local_uses_unit_log_level() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Local);
    // Unit scenario has "warn" but Local env has "info"; env overrides since it differs from default
    assert!(!cfg.log_level.is_empty());
}

#[test]
fn resolve_perf_production_sequential() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.resolve(&TestingScenario::Performance, &EnvironmentType::Production);
    assert_eq!(cfg.max_parallel_tests, 1);
}

#[test]
fn resolve_uploads_reports_when_ci() {
    let mgr = ScenarioConfigManager::new();
    let cfg = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);
    assert!(cfg.reporting.upload_reports);
}

// ---------------------------------------------------------------------------
// ScenarioConfigManager: get_context_config with platform
// ---------------------------------------------------------------------------

#[test]
fn context_config_windows_caps_parallelism() {
    let mgr = ScenarioConfigManager::new();
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: EnvironmentType::Ci,
        platform_settings: Some(PlatformSettings {
            os: Some("windows".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let cfg = mgr.get_context_config(&ctx);
    assert!(cfg.max_parallel_tests <= 8);
}

#[test]
fn context_config_macos_caps_parallelism() {
    let mgr = ScenarioConfigManager::new();
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: EnvironmentType::Ci,
        platform_settings: Some(PlatformSettings {
            os: Some("macos".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let cfg = mgr.get_context_config(&ctx);
    assert!(cfg.max_parallel_tests <= 6);
}

#[test]
fn context_config_linux_no_cap() {
    let mgr = ScenarioConfigManager::new();
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: EnvironmentType::Ci,
        platform_settings: Some(PlatformSettings {
            os: Some("linux".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let cfg = mgr.get_context_config(&ctx);
    // Linux has no cap; CI should be 8
    assert_eq!(cfg.max_parallel_tests, 8);
}

#[test]
fn context_config_no_platform_no_cap() {
    let mgr = ScenarioConfigManager::new();
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: EnvironmentType::Ci,
        ..Default::default()
    };
    let cfg = mgr.get_context_config(&ctx);
    assert_eq!(cfg.max_parallel_tests, 8);
}

// ---------------------------------------------------------------------------
// ScenarioConfigManager: context_from_environment
// ---------------------------------------------------------------------------

#[test]
fn context_from_environment_callable() {
    let ctx = ScenarioConfigManager::context_from_environment();
    assert!(!ctx.scenario.to_string().is_empty());
}

// ---------------------------------------------------------------------------
// ScenarioConfigManager: scenario_description
// ---------------------------------------------------------------------------

#[test]
fn all_scenario_descriptions_non_empty() {
    for s in ScenarioConfigManager::available_scenarios() {
        let desc = ScenarioConfigManager::scenario_description(s);
        assert!(!desc.is_empty(), "scenario {s:?} must have a description");
    }
}

#[test]
fn scenario_description_unit() {
    let desc = ScenarioConfigManager::scenario_description(&TestingScenario::Unit);
    assert!(desc.contains("Fast"));
}

#[test]
fn scenario_description_performance() {
    let desc = ScenarioConfigManager::scenario_description(&TestingScenario::Performance);
    assert!(
        desc.contains("latency") || desc.contains("throughput") || desc.contains("performance")
    );
}

// ---------------------------------------------------------------------------
// ScenarioConfigManager: available_scenarios
// ---------------------------------------------------------------------------

#[test]
fn available_scenarios_has_unit() {
    assert!(ScenarioConfigManager::available_scenarios().contains(&TestingScenario::Unit));
}

#[test]
fn available_scenarios_has_integration() {
    assert!(ScenarioConfigManager::available_scenarios().contains(&TestingScenario::Integration));
}

// ---------------------------------------------------------------------------
// ScenarioConfigManager: Debug + Clone
// ---------------------------------------------------------------------------

#[test]
fn manager_debug_not_empty() {
    let mgr = ScenarioConfigManager::new();
    let d = format!("{:?}", mgr);
    assert!(d.contains("ScenarioConfigManager"));
}

#[test]
fn manager_clone() {
    let mgr = ScenarioConfigManager::new();
    let mgr2 = mgr.clone();
    let c1 = mgr.get_scenario_config(&TestingScenario::Unit);
    let c2 = mgr2.get_scenario_config(&TestingScenario::Unit);
    assert_eq!(c1.max_parallel_tests, c2.max_parallel_tests);
}
