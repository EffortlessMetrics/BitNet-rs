use bitnet_runtime_profile_contract_core::{
    ActiveContext, ExecutionEnvironment, TestingScenario, active_profile_for,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn scenario_strategy() -> impl Strategy<Value = TestingScenario> {
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

fn environment_strategy() -> impl Strategy<Value = ExecutionEnvironment> {
    prop_oneof![
        Just(ExecutionEnvironment::Local),
        Just(ExecutionEnvironment::Ci),
        Just(ExecutionEnvironment::PreProduction),
        Just(ExecutionEnvironment::Production),
    ]
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    /// `is_supported()` must be consistent with `violations()`:
    /// if both missing and forbidden are empty → supported; otherwise → not supported.
    #[test]
    fn violations_consistent_with_is_supported(
        scenario in scenario_strategy(),
        environment in environment_strategy(),
    ) {
        let profile = active_profile_for(scenario, environment);
        let (missing, forbidden) = profile.violations();
        let both_empty = missing.labels().is_empty() && forbidden.labels().is_empty();

        if profile.cell.is_some() {
            // When a cell exists: is_supported ↔ no violations
            prop_assert_eq!(profile.is_supported(), both_empty);
        } else {
            // When no cell exists: never "supported" (no contract → not satisfied)
            prop_assert!(!profile.is_supported());
        }
    }

    /// `missing()` is exactly the labels of `violations().0`.
    #[test]
    fn missing_matches_violations_missing(
        scenario in scenario_strategy(),
        environment in environment_strategy(),
    ) {
        let profile = active_profile_for(scenario, environment);
        let via_missing = profile.missing();
        let via_violations = profile.violations().0.labels();
        prop_assert_eq!(via_missing, via_violations);
    }

    /// `forbidden()` is exactly the labels of `violations().1`.
    #[test]
    fn forbidden_matches_violations_forbidden(
        scenario in scenario_strategy(),
        environment in environment_strategy(),
    ) {
        let profile = active_profile_for(scenario, environment);
        let via_forbidden = profile.forbidden();
        let via_violations = profile.violations().1.labels();
        prop_assert_eq!(via_forbidden, via_violations);
    }

    /// Building a profile from an explicit context preserves the scenario and environment.
    #[test]
    fn from_context_preserves_scenario_and_environment(
        scenario in scenario_strategy(),
        environment in environment_strategy(),
    ) {
        let context = ActiveContext { scenario, environment };
        let profile = bitnet_runtime_profile_contract_core::ActiveProfile::from_context(context);
        prop_assert_eq!(profile.scenario, scenario);
        prop_assert_eq!(profile.environment, environment);
    }

    /// `active_profile_for` is deterministic: two calls with the same inputs produce
    /// equivalent profiles (same scenario/environment and same cell presence).
    #[test]
    fn active_profile_for_is_deterministic(
        scenario in scenario_strategy(),
        environment in environment_strategy(),
    ) {
        let p1 = active_profile_for(scenario, environment);
        let p2 = active_profile_for(scenario, environment);
        prop_assert_eq!(p1.scenario, p2.scenario);
        prop_assert_eq!(p1.environment, p2.environment);
        prop_assert_eq!(p1.cell.is_some(), p2.cell.is_some());
        prop_assert_eq!(p1.is_supported(), p2.is_supported());
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[test]
fn unit_local_profile_does_not_panic() {
    let profile = active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Local);
    // Just confirm the struct is constructable and accessors don't panic.
    let _ = profile.missing();
    let _ = profile.forbidden();
    let _ = profile.is_supported();
}

#[test]
fn e2e_ci_profile_does_not_panic() {
    let profile = active_profile_for(TestingScenario::EndToEnd, ExecutionEnvironment::Ci);
    let _ = profile.missing();
    let _ = profile.is_supported();
}
