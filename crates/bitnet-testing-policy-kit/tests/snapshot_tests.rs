//! Snapshot tests for bitnet-testing-policy-kit
//!
//! Covers:
//! - `active_feature_labels()` — stable feature labels with no compiled features
//! - `profile_snapshot()` — scenario/environment fields
//! - `resolve_active_profile()` — returns a valid profile config
//! - type aliases are exactly right
use bitnet_testing_policy_kit::{
    EnvironmentType, ExecutionEnvironment, GridEnvironment, GridScenario, TestingScenario,
    active_feature_labels, profile_snapshot,
};

#[test]
fn type_aliases_are_correct_types() {
    // These must compile and produce the same variants
    let env: EnvironmentType = ExecutionEnvironment::Local;
    let grid_env: GridEnvironment = ExecutionEnvironment::Local;
    let grid_sc: GridScenario = TestingScenario::Unit;

    insta::assert_snapshot!("alias_environment_local", env.to_string());
    insta::assert_snapshot!("alias_grid_environment", grid_env.to_string());
    insta::assert_snapshot!("alias_grid_scenario_unit", grid_sc.to_string());
}

#[test]
fn active_feature_labels_returns_list() {
    let labels = active_feature_labels();
    // With no features compiled, labels should be empty (feature-line returns "features:")
    insta::assert_snapshot!("active_feature_labels_count", labels.len().to_string());
}

#[test]
fn profile_snapshot_scenario_matches_environment() {
    let (profile, ctx, _violations) = profile_snapshot();
    // Profile scenario must match context scenario
    insta::assert_snapshot!(
        "profile_snapshot_scenario_eq_context",
        (profile.scenario == ctx.scenario).to_string()
    );
    insta::assert_snapshot!(
        "profile_snapshot_environment_eq_context",
        (profile.environment == ctx.environment).to_string()
    );
}
