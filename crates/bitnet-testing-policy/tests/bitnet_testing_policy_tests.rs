//! Property and integration tests for `bitnet-testing-policy`.
//!
//! The crate is a façade over `bitnet-testing-policy-core`. Tests verify the
//! `PolicySnapshot` and configuration resolution APIs maintain expected invariants.

use bitnet_testing_policy::{
    ConfigurationContext, EnvironmentType, PolicySnapshot, ScenarioConfigManager, TestingScenario,
    resolve_context_profile, snapshot_from_env, validate_explicit_profile,
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
fn snapshot_from_env_does_not_panic() {
    let snapshot = snapshot_from_env();
    assert!(!snapshot.context.scenario.to_string().is_empty());
    assert!(!snapshot.context.environment.to_string().is_empty());
}

#[test]
fn policy_snapshot_default_context_resolves() {
    let ctx = ConfigurationContext::default();
    let snapshot = PolicySnapshot::from_configuration_context(ctx);
    assert!(!snapshot.summary().is_empty());
}

#[test]
fn scenario_config_manager_new_does_not_panic() {
    let _mgr = ScenarioConfigManager::new();
}

#[test]
fn resolve_context_profile_default_has_positive_parallelism() {
    let ctx = ConfigurationContext::default();
    let profile = resolve_context_profile(&ctx);
    assert!(profile.max_parallel_tests > 0, "max_parallel_tests must be at least 1");
}

#[test]
fn resolve_context_profile_reporting_formats_non_empty() {
    let ctx = ConfigurationContext::default();
    let profile = resolve_context_profile(&ctx);
    assert!(!profile.reporting.formats.is_empty(), "reporting formats must be non-empty");
}

#[test]
fn validate_explicit_profile_unit_local_returns_without_panic() {
    let _result = validate_explicit_profile(TestingScenario::Unit, EnvironmentType::Local);
}

#[test]
fn policy_snapshot_is_compatible_reflects_violations() {
    let snapshot = snapshot_from_env();
    if snapshot.is_compatible() {
        if let Some((missing, forbidden)) = snapshot.violations() {
            assert!(missing.is_empty() && forbidden.is_empty());
        }
    }
}

// ── proptest invariants ──────────────────────────────────────────────────────

proptest! {
    /// `resolve_context_profile` must always produce a profile with positive
    /// `max_parallel_tests` regardless of the scenario/environment combination.
    #[test]
    fn resolved_profile_has_positive_parallelism(
        scenario_idx in 0usize..9,
        env_idx in 0usize..4,
    ) {
        let ctx = ConfigurationContext {
            scenario: SCENARIOS[scenario_idx],
            environment: ENVS[env_idx],
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        };
        let profile = resolve_context_profile(&ctx);
        prop_assert!(profile.max_parallel_tests > 0);
    }

    /// `resolve_context_profile` must always return at least one reporting format.
    #[test]
    fn resolved_profile_has_at_least_one_report_format(
        scenario_idx in 0usize..9,
        env_idx in 0usize..4,
    ) {
        let ctx = ConfigurationContext {
            scenario: SCENARIOS[scenario_idx],
            environment: ENVS[env_idx],
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        };
        let profile = resolve_context_profile(&ctx);
        prop_assert!(!profile.reporting.formats.is_empty());
    }

    /// `PolicySnapshot::from_configuration_context` must produce a non-empty
    /// summary for every valid scenario/environment combination.
    #[test]
    fn policy_snapshot_summary_non_empty(
        scenario_idx in 0usize..9,
        env_idx in 0usize..4,
    ) {
        let ctx = ConfigurationContext {
            scenario: SCENARIOS[scenario_idx],
            environment: ENVS[env_idx],
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        };
        let snapshot = PolicySnapshot::from_configuration_context(ctx);
        prop_assert!(!snapshot.summary().is_empty());
    }
}
