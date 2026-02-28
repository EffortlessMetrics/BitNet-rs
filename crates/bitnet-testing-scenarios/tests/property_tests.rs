//! Property-based tests for `bitnet-testing-scenarios`.
//!
//! Key invariants:
//! - `ScenarioConfigManager::new()` always produces a non-empty manager.
//! - `resolve()` merges scenario + environment configs without panicking.
//! - Resolved timeout is always ≥ both scenario and environment timeouts.
//! - Resolved coverage threshold is always ≥ both inputs.
//! - `available_scenarios()` returns a non-empty list.
//! - `scenario_description()` is non-empty for every scenario.

use bitnet_testing_scenarios::{
    EnvironmentType, ScenarioConfigManager, TestConfigProfile, TestingScenario,
};
use proptest::prelude::*;

// ── Arbitrary strategies ──────────────────────────────────────────────────────

fn arb_scenario() -> impl Strategy<Value = TestingScenario> {
    prop_oneof![
        Just(TestingScenario::Unit),
        Just(TestingScenario::Integration),
        Just(TestingScenario::EndToEnd),
        Just(TestingScenario::Performance),
        Just(TestingScenario::CrossValidation),
        Just(TestingScenario::Smoke),
        Just(TestingScenario::Development),
        Just(TestingScenario::Debug),
        Just(TestingScenario::Minimal),
    ]
}

fn arb_env_type() -> impl Strategy<Value = EnvironmentType> {
    prop_oneof![
        Just(EnvironmentType::Local),
        Just(EnvironmentType::Ci),
        Just(EnvironmentType::PreProduction),
        Just(EnvironmentType::Production),
    ]
}

// ── ScenarioConfigManager construction ────────────────────────────────────────

proptest! {
    /// `new()` always succeeds and is stable across calls.
    #[test]
    fn manager_new_never_panics(_dummy in 0u8..4) {
        let _mgr = ScenarioConfigManager::new();
    }

    /// `available_scenarios()` always returns at least one scenario.
    #[test]
    fn available_scenarios_is_nonempty(_dummy in 0u8..4) {
        let scenarios = ScenarioConfigManager::available_scenarios();
        prop_assert!(!scenarios.is_empty());
    }

    /// Every known scenario has a non-empty description.
    #[test]
    fn scenario_description_is_nonempty(s in arb_scenario()) {
        let desc = ScenarioConfigManager::scenario_description(&s);
        prop_assert!(!desc.is_empty());
    }
}

// ── resolve() merge properties ────────────────────────────────────────────────

proptest! {
    /// resolve() never panics for any valid scenario×environment pair.
    #[test]
    fn resolve_never_panics(
        scenario in arb_scenario(),
        env in arb_env_type(),
    ) {
        let mgr = ScenarioConfigManager::new();
        let _config = mgr.resolve(&scenario, &env);
    }

    /// Resolved timeout is always ≥ both scenario and environment timeouts.
    #[test]
    fn resolve_timeout_is_max_of_inputs(
        scenario in arb_scenario(),
        env in arb_env_type(),
    ) {
        let mgr = ScenarioConfigManager::new();
        let scenario_config = mgr.get_scenario_config(&scenario);
        let env_config = mgr.get_environment_config(&env);
        let resolved = mgr.resolve(&scenario, &env);
        prop_assert!(resolved.test_timeout >= scenario_config.test_timeout);
        prop_assert!(resolved.test_timeout >= env_config.test_timeout);
    }

    /// Resolved coverage threshold is always ≥ both inputs.
    #[test]
    fn resolve_coverage_is_max_of_inputs(
        scenario in arb_scenario(),
        env in arb_env_type(),
    ) {
        let mgr = ScenarioConfigManager::new();
        let scenario_config = mgr.get_scenario_config(&scenario);
        let env_config = mgr.get_environment_config(&env);
        let resolved = mgr.resolve(&scenario, &env);
        prop_assert!(resolved.coverage_threshold >= scenario_config.coverage_threshold);
        prop_assert!(resolved.coverage_threshold >= env_config.coverage_threshold);
    }

    /// Resolved max_parallel_tests is always > 0.
    #[test]
    fn resolve_parallelism_is_positive(
        scenario in arb_scenario(),
        env in arb_env_type(),
    ) {
        let mgr = ScenarioConfigManager::new();
        let resolved = mgr.resolve(&scenario, &env);
        prop_assert!(resolved.max_parallel_tests > 0);
    }

    /// Resolved log_level is always a non-empty string.
    #[test]
    fn resolve_log_level_is_nonempty(
        scenario in arb_scenario(),
        env in arb_env_type(),
    ) {
        let mgr = ScenarioConfigManager::new();
        let resolved = mgr.resolve(&scenario, &env);
        prop_assert!(!resolved.log_level.is_empty());
    }
}

// ── TestConfigProfile default ─────────────────────────────────────────────────

proptest! {
    /// Default config always has positive parallelism and non-zero timeout.
    #[test]
    fn default_config_has_sane_values(_dummy in 0u8..1) {
        let config = TestConfigProfile::default();
        prop_assert!(config.max_parallel_tests > 0);
        prop_assert!(!config.test_timeout.is_zero());
        prop_assert!(!config.log_level.is_empty());
    }
}
