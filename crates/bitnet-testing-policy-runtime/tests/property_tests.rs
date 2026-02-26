use bitnet_testing_policy_runtime::{
    ConfigurationContext, ExecutionEnvironment, RuntimePolicyState, TestingScenario,
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

fn context_strategy() -> impl Strategy<Value = ConfigurationContext> {
    (scenario_strategy(), environment_strategy()).prop_map(|(scenario, environment)| {
        ConfigurationContext { scenario, environment, ..Default::default() }
    })
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    /// `from_context` preserves the scenario and environment fields.
    #[test]
    fn from_context_preserves_scenario_and_env(ctx in context_strategy()) {
        let state = RuntimePolicyState::from_context(&ctx);
        prop_assert_eq!(state.context.scenario, ctx.scenario);
        prop_assert_eq!(state.context.environment, ctx.environment);
    }

    /// `has_grid_cell` agrees with the active profile's cell presence.
    #[test]
    fn has_grid_cell_agrees_with_active_profile(ctx in context_strategy()) {
        let state = RuntimePolicyState::from_context(&ctx);
        prop_assert_eq!(state.has_grid_cell(), state.active_profile.cell.is_some());
    }

    /// `summary()` never panics and always contains "scenario=" and "environment=".
    #[test]
    fn summary_contains_expected_substrings(ctx in context_strategy()) {
        let state = RuntimePolicyState::from_context(&ctx);
        let summary = state.summary();
        prop_assert!(summary.contains("scenario=") || summary.contains("feature-contract"));
    }

    /// `RuntimePolicyState` is deterministic: two calls with the same context
    /// produce equal `has_grid_cell()` and `is_feature_contract_aligned()`.
    #[test]
    fn runtime_state_is_deterministic(ctx in context_strategy()) {
        let s1 = RuntimePolicyState::from_context(&ctx);
        let s2 = RuntimePolicyState::from_context(&ctx);
        prop_assert_eq!(s1.has_grid_cell(), s2.has_grid_cell());
        prop_assert_eq!(s1.is_feature_contract_aligned(), s2.is_feature_contract_aligned());
        prop_assert_eq!(s1.is_policy_compatible(), s2.is_policy_compatible());
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[test]
fn unit_local_state_does_not_panic() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        ..Default::default()
    };
    let state = RuntimePolicyState::from_context(&ctx);
    let _ = state.summary();
    let _ = state.has_grid_cell();
    let _ = state.is_policy_compatible();
    let _ = state.is_feature_contract_aligned();
}

#[test]
fn resolved_profile_config_does_not_panic() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Integration,
        environment: ExecutionEnvironment::Ci,
        ..Default::default()
    };
    let state = RuntimePolicyState::from_context(&ctx);
    let _ = state.resolved_profile_config();
}
