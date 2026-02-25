//! Snapshot tests for bitnet-testing-policy-interop.
//!
//! Pins: Environment alias is ExecutionEnvironment, canonical grid row count,
//! and validate helpers delegate correctly.

use bitnet_testing_policy_interop::{
    Environment, ExecutionEnvironment, TestingScenario, canonical_grid, validate_active_profile_for,
};

#[test]
fn environment_alias_works_as_execution_environment() {
    // Environment is a type alias for ExecutionEnvironment; test that assignment compiles
    let env: Environment = ExecutionEnvironment::Local;
    insta::assert_debug_snapshot!(env);
}

#[test]
fn canonical_grid_row_count_stable() {
    insta::assert_snapshot!(canonical_grid().rows().len().to_string());
}

#[test]
fn validate_unit_local_returns_some() {
    let result = validate_active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Local);
    insta::assert_snapshot!(result.is_some().to_string());
}
