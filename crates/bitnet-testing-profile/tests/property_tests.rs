use bitnet_testing_profile::{
    ActiveContext, ExecutionEnvironment, TestingScenario, from_grid_environment,
    from_grid_scenario, to_grid_environment, to_grid_scenario, validate_explicit_profile,
    validate_profile_for_context,
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
    /// `to_grid_scenario` and `from_grid_scenario` are both identity functions.
    #[test]
    fn scenario_conversion_helpers_are_identity(scenario in arb_scenario()) {
        prop_assert_eq!(
            format!("{:?}", to_grid_scenario(scenario)),
            format!("{:?}", scenario)
        );
        prop_assert_eq!(
            format!("{:?}", from_grid_scenario(scenario)),
            format!("{:?}", scenario)
        );
    }

    /// `to_grid_environment` and `from_grid_environment` are both identity functions.
    #[test]
    fn environment_conversion_helpers_are_identity(env in arb_env()) {
        prop_assert_eq!(
            format!("{:?}", to_grid_environment(env)),
            format!("{:?}", env)
        );
        prop_assert_eq!(
            format!("{:?}", from_grid_environment(env)),
            format!("{:?}", env)
        );
    }

    /// `validate_explicit_profile` and `validate_profile_for_context` agree for any pair.
    #[test]
    fn explicit_and_context_validators_agree(
        scenario in arb_scenario(),
        env in arb_env(),
    ) {
        let via_explicit = validate_explicit_profile(scenario, env);
        let context = ActiveContext { scenario, environment: env };
        let via_context = validate_profile_for_context(context);
        prop_assert_eq!(
            via_explicit.is_some(),
            via_context.is_some(),
            "explicit and context validators disagree for {:?}/{:?}", scenario, env
        );
    }

    /// Composing `to_grid_scenario` then `from_grid_scenario` is still identity.
    #[test]
    fn scenario_round_trip_compose(scenario in arb_scenario()) {
        let round_tripped = from_grid_scenario(to_grid_scenario(scenario));
        prop_assert_eq!(
            format!("{:?}", round_tripped),
            format!("{:?}", scenario)
        );
    }
}
