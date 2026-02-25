//! Snapshot tests for bitnet-testing-policy-runtime
//!
//! Covers:
//! - `RuntimePolicyState` structural invariants
//! - `has_grid_cell()` for Unit/Local context
//! - `summary()` format stability
//! - `active_profile_for()` round-trip
use bitnet_testing_policy_runtime::{
    ConfigurationContext, ExecutionEnvironment, RuntimePolicyState, TestingScenario,
    active_profile_for,
};

#[test]
fn runtime_policy_state_unit_local_has_grid_cell() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        ..Default::default()
    };
    let state = RuntimePolicyState::from_context(&ctx);
    insta::assert_snapshot!("unit_local_has_grid_cell", state.has_grid_cell().to_string());
}

#[test]
fn runtime_policy_state_summary_contains_bracket() {
    let ctx = ConfigurationContext::default();
    let state = RuntimePolicyState::from_context(&ctx);
    let summary = state.summary();
    // Summary always ends with a feature-contract bracket label
    insta::assert_snapshot!("summary_has_bracket", summary.contains('[').to_string());
}

#[test]
fn active_profile_for_unit_local() {
    let profile = active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Local);
    insta::assert_snapshot!("active_profile_scenario", profile.scenario.to_string());
    insta::assert_snapshot!("active_profile_environment", profile.environment.to_string());
}
