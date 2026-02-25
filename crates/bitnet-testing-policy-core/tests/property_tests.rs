use bitnet_testing_policy_core::{
    ConfigurationContext, EnvironmentType, PolicySnapshot, TestingScenario,
};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────────

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

// ── Property tests ───────────────────────────────────────────────────────────

proptest! {
    /// `PolicySnapshot::is_compatible()` never panics for any scenario/env pair.
    #[test]
    fn policy_snapshot_is_compatible_never_panics(
        scenario in arb_scenario(),
        env in arb_env_type(),
    ) {
        let context = ConfigurationContext {
            scenario,
            environment: env,
            ..ConfigurationContext::default()
        };
        let snapshot = PolicySnapshot::from_configuration_context(context);
        let _ = snapshot.is_compatible();
    }

    /// `summary()` always returns a non-empty string.
    #[test]
    fn policy_snapshot_summary_always_non_empty(
        scenario in arb_scenario(),
    ) {
        let context = ConfigurationContext {
            scenario,
            ..ConfigurationContext::default()
        };
        let snapshot = PolicySnapshot::from_configuration_context(context);
        prop_assert!(!snapshot.summary().is_empty());
    }

    /// `resolved_config.max_parallel_tests` is always ≥ 1.
    #[test]
    fn resolved_config_parallelism_positive(
        scenario in arb_scenario(),
    ) {
        let context = ConfigurationContext {
            scenario,
            ..ConfigurationContext::default()
        };
        let snapshot = PolicySnapshot::from_configuration_context(context);
        prop_assert!(snapshot.resolved_config.max_parallel_tests >= 1);
    }

    /// `resolved_config.reporting.formats` is never empty.
    #[test]
    fn resolved_config_has_at_least_one_report_format(
        scenario in arb_scenario(),
    ) {
        let context = ConfigurationContext {
            scenario,
            ..ConfigurationContext::default()
        };
        let snapshot = PolicySnapshot::from_configuration_context(context);
        prop_assert!(!snapshot.resolved_config.reporting.formats.is_empty());
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[test]
fn detect_produces_a_usable_snapshot() {
    let snap = PolicySnapshot::detect();
    assert!(!snap.summary().is_empty());
    assert!(snap.resolved_config.max_parallel_tests >= 1);
}

#[test]
fn context_scenario_preserved_in_snapshot() {
    let context = ConfigurationContext {
        scenario: TestingScenario::CrossValidation,
        ..ConfigurationContext::default()
    };
    let snap = PolicySnapshot::from_configuration_context(context);
    assert_eq!(snap.context.scenario, TestingScenario::CrossValidation);
}

#[test]
fn unit_scenario_has_reasonable_timeout() {
    let context =
        ConfigurationContext { scenario: TestingScenario::Unit, ..ConfigurationContext::default() };
    let snap = PolicySnapshot::from_configuration_context(context);
    // Timeout should be a sensible duration: > 1 second, < 24 hours.
    let secs = snap.resolved_config.test_timeout.as_secs();
    assert!(secs >= 1, "timeout unexpectedly short: {secs}s");
    assert!(secs <= 86_400, "timeout unexpectedly long: {secs}s");
}
