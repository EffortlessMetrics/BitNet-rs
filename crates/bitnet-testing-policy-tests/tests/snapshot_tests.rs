//! Snapshot tests for bitnet-testing-policy-tests.
//!
//! Pins: PolicyDiagnostics::current() is constructible, GridScenario/GridEnvironment
//! aliases, and context round-trips via from_context().

use bitnet_testing_policy_tests::{
    ConfigurationContext, ExecutionEnvironment, GridEnvironment, GridScenario, PolicyDiagnostics,
    TestingScenario,
};

#[test]
fn grid_scenario_alias_is_testing_scenario() {
    let s: GridScenario = TestingScenario::Unit;
    insta::assert_debug_snapshot!(s);
}

#[test]
fn grid_environment_alias_is_execution_environment() {
    let e: GridEnvironment = ExecutionEnvironment::Local;
    insta::assert_debug_snapshot!(e);
}

#[test]
fn policy_diagnostics_from_context_has_cell() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        ..Default::default()
    };
    let diag = PolicyDiagnostics::from_context(&ctx);
    insta::assert_snapshot!(diag.profile().cell.is_some().to_string());
}
