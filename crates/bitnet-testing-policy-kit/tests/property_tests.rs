use bitnet_testing_policy_kit::{
    ExecutionEnvironment, TestingScenario, active_feature_labels, active_features,
    active_profile_for, profile_snapshot, resolve_active_profile,
    validate_active_profile_from_environment,
};
use proptest::prelude::*;

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

fn arb_env() -> impl Strategy<Value = ExecutionEnvironment> {
    prop_oneof![
        Just(ExecutionEnvironment::Local),
        Just(ExecutionEnvironment::Ci),
        Just(ExecutionEnvironment::PreProduction),
        Just(ExecutionEnvironment::Production),
    ]
}

proptest! {
    /// `active_feature_labels()` is always consistent with `active_features().labels()`.
    #[test]
    fn active_feature_labels_matches_active_features(_: ()) {
        let from_labels_fn = active_feature_labels();
        let from_set = active_features().labels();
        prop_assert_eq!(from_labels_fn, from_set);
    }

    /// `profile_snapshot()` returns a profile whose scenario and environment
    /// match the resolved context.
    #[test]
    fn profile_snapshot_scenario_env_match_context(_: ()) {
        let (profile, context, _violations) = profile_snapshot();
        prop_assert_eq!(
            format!("{:?}", profile.scenario),
            format!("{:?}", context.scenario)
        );
        prop_assert_eq!(
            format!("{:?}", profile.environment),
            format!("{:?}", context.environment)
        );
    }

    /// `active_profile_for(scenario, env)` returns a profile with matching fields.
    #[test]
    fn active_profile_for_preserves_scenario_and_env(
        scenario in arb_scenario(),
        env in arb_env(),
    ) {
        let profile = active_profile_for(scenario, env);
        prop_assert_eq!(
            format!("{:?}", profile.scenario),
            format!("{:?}", scenario)
        );
        prop_assert_eq!(
            format!("{:?}", profile.environment),
            format!("{:?}", env)
        );
    }

    /// `validate_active_profile_from_environment()` is consistent with
    /// `profile_snapshot()` violations.
    #[test]
    fn validate_env_consistent_with_snapshot(_: ()) {
        let via_validate = validate_active_profile_from_environment();
        let (_profile, _context, via_snapshot) = profile_snapshot();
        prop_assert_eq!(
            via_validate.is_some(),
            via_snapshot.is_some(),
            "validate_active_profile_from_environment and profile_snapshot disagreed on violation presence"
        );
    }

    /// `resolve_active_profile()` returns a non-default config (never panics).
    #[test]
    fn resolve_active_profile_does_not_panic(_: ()) {
        let _config = resolve_active_profile();
    }
}
