//! Property and integration tests for `bitnet-testing-scenarios`.
//!
//! The crate is a façade over `bitnet-testing-scenarios-core`. Tests verify the
//! `ScenarioConfigManager`, `TestConfigProfile`, and scenario/environment
//! resolution APIs maintain expected invariants.

use bitnet_testing_scenarios::{
    ConfigurationContext, EnvironmentType, ReportFormat, ScenarioConfigManager, TestingScenario,
};
use proptest::prelude::*;

const SCENARIOS: [TestingScenario; 9] = [
    TestingScenario::Unit,
    TestingScenario::Integration,
    TestingScenario::EndToEnd,
    TestingScenario::Performance,
    TestingScenario::CrossValidation,
    TestingScenario::Smoke,
    TestingScenario::Development,
    TestingScenario::Debug,
    TestingScenario::Minimal,
];

const ENVS: [EnvironmentType; 4] = [
    EnvironmentType::Local,
    EnvironmentType::Ci,
    EnvironmentType::PreProduction,
    EnvironmentType::Production,
];

// ── smoke tests ─────────────────────────────────────────────────────────────

#[test]
fn scenario_config_manager_new_does_not_panic() {
    let _mgr = ScenarioConfigManager::new();
}

#[test]
fn get_scenario_config_returns_valid_profile_for_all_scenarios() {
    let mgr = ScenarioConfigManager::new();
    for scenario in &SCENARIOS {
        let profile = mgr.get_scenario_config(scenario);
        assert!(profile.max_parallel_tests > 0, "max_parallel_tests > 0 for {scenario}");
        assert!(!profile.reporting.formats.is_empty(), "formats non-empty for {scenario}");
    }
}

#[test]
fn get_environment_config_returns_profiles_with_formats() {
    let mgr = ScenarioConfigManager::new();
    for env in &ENVS {
        let profile = mgr.get_environment_config(env);
        assert!(!profile.reporting.formats.is_empty(), "formats non-empty for {env:?}");
    }
}

#[test]
fn resolve_ci_environment_has_junit_format() {
    let mgr = ScenarioConfigManager::new();
    let profile = mgr.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);
    assert!(
        profile.reporting.formats.contains(&ReportFormat::Junit),
        "CI must include JUnit format"
    );
}

#[test]
fn crossval_scenario_has_crossval_enabled() {
    let mgr = ScenarioConfigManager::new();
    let profile = mgr.get_scenario_config(&TestingScenario::CrossValidation);
    assert!(profile.crossval.enabled, "CrossValidation scenario must set crossval.enabled = true");
}

#[test]
fn context_from_environment_does_not_panic() {
    let _ctx = ScenarioConfigManager::context_from_environment();
}

// ── proptest invariants ──────────────────────────────────────────────────────

proptest! {
    /// `resolve()` must always produce a profile with positive `max_parallel_tests`.
    #[test]
    fn resolved_profile_has_positive_parallelism(
        scenario_idx in 0usize..9,
        env_idx in 0usize..4,
    ) {
        let mgr = ScenarioConfigManager::new();
        let profile = mgr.resolve(&SCENARIOS[scenario_idx], &ENVS[env_idx]);
        prop_assert!(profile.max_parallel_tests > 0);
    }

    /// `resolve()` must always produce at least one reporting format.
    #[test]
    fn resolved_profile_has_report_formats(
        scenario_idx in 0usize..9,
        env_idx in 0usize..4,
    ) {
        let mgr = ScenarioConfigManager::new();
        let profile = mgr.resolve(&SCENARIOS[scenario_idx], &ENVS[env_idx]);
        prop_assert!(!profile.reporting.formats.is_empty());
    }

    /// `TestingScenario` must display to a string that parses back correctly.
    #[test]
    fn testing_scenario_display_roundtrip(idx in 0usize..9) {
        let s = SCENARIOS[idx];
        let text = s.to_string();
        prop_assert!(!text.is_empty());
        let parsed: TestingScenario = text.parse().expect("display must round-trip");
        prop_assert_eq!(s, parsed);
    }

    /// `EnvironmentType` must display to a string that parses back correctly.
    #[test]
    fn environment_type_display_roundtrip(idx in 0usize..4) {
        let e = ENVS[idx];
        let text = e.to_string();
        prop_assert!(!text.is_empty());
        let parsed: EnvironmentType = text.parse().expect("display must round-trip");
        prop_assert_eq!(e, parsed);
    }
}
