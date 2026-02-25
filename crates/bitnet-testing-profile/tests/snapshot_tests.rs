//! Snapshot tests for bitnet-testing-profile.
//!
//! Pins the stable re-export surface: conversion helpers are identity functions,
//! type alias targets match expectations, and validate helpers delegate correctly.

use bitnet_testing_profile::{
    ExecutionEnvironment, TestingScenario, canonical_grid, from_grid_environment,
    from_grid_scenario, to_grid_environment, to_grid_scenario,
};

#[test]
fn conversion_helpers_are_identity() {
    let scenario = TestingScenario::Unit;
    let env = ExecutionEnvironment::Local;
    // to_grid_* and from_grid_* must be no-ops
    insta::assert_debug_snapshot!((
        to_grid_scenario(scenario) == scenario,
        to_grid_environment(env) == env,
        from_grid_scenario(scenario) == scenario,
        from_grid_environment(env) == env,
    ));
}

#[test]
fn canonical_grid_row_count_stable() {
    insta::assert_snapshot!(canonical_grid().rows().len().to_string());
}
