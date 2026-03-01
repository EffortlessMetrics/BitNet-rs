//! Edge-case tests for bitnet-testing-policy-tests PolicyDiagnostics and helpers.

use bitnet_testing_policy_tests::{
    ConfigurationContext, ExecutionEnvironment, PolicyDiagnostics, TestingScenario, diagnostics,
    diagnostics_for_context, validate_active_profile_from_context,
};

// ---------------------------------------------------------------------------
// PolicyDiagnostics: current
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_current_has_context() {
    let diag = PolicyDiagnostics::current();
    assert!(!diag.context().scenario.to_string().is_empty());
    assert!(!diag.context().environment.to_string().is_empty());
}

#[test]
fn diagnostics_current_has_profile() {
    let diag = PolicyDiagnostics::current();
    assert!(!diag.profile().features.labels().is_empty());
}

#[test]
fn diagnostics_current_summary_not_empty() {
    let diag = PolicyDiagnostics::current();
    assert!(!diag.summary().is_empty());
}

// ---------------------------------------------------------------------------
// PolicyDiagnostics: from_context
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_from_context_unit_local() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let diag = PolicyDiagnostics::from_context(&ctx);
    assert_eq!(diag.context().scenario, TestingScenario::Unit);
    assert_eq!(diag.context().environment, ExecutionEnvironment::Local);
}

#[test]
fn diagnostics_from_context_integration_ci() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Integration,
        environment: ExecutionEnvironment::Ci,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let diag = PolicyDiagnostics::from_context(&ctx);
    assert_eq!(diag.context().scenario, TestingScenario::Integration);
}

// ---------------------------------------------------------------------------
// PolicyDiagnostics: violations
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_violations_callable() {
    let diag = PolicyDiagnostics::current();
    // May be Some or None
    let _ = diag.violations();
}

#[test]
fn diagnostics_unit_local_has_violations() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let diag = PolicyDiagnostics::from_context(&ctx);
    // Unit/Local is in the canonical grid
    assert!(diag.violations().is_some());
}

// ---------------------------------------------------------------------------
// PolicyDiagnostics: profile_config
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_profile_config_has_parallelism() {
    let diag = PolicyDiagnostics::current();
    let config = diag.profile_config();
    assert!(config.max_parallel_tests > 0);
}

#[test]
fn diagnostics_profile_config_has_reporting() {
    let ctx = ConfigurationContext::default();
    let diag = PolicyDiagnostics::from_context(&ctx);
    let config = diag.profile_config();
    assert!(!config.reporting.formats.is_empty());
}

// ---------------------------------------------------------------------------
// PolicyDiagnostics: is_grid_compatible
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_is_grid_compatible_callable() {
    let diag = PolicyDiagnostics::current();
    let _ = diag.is_grid_compatible();
}

// ---------------------------------------------------------------------------
// PolicyDiagnostics: is_feature_contract_consistent
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_is_feature_contract_consistent_callable() {
    let diag = PolicyDiagnostics::current();
    let _ = diag.is_feature_contract_consistent();
}

// ---------------------------------------------------------------------------
// PolicyDiagnostics: summary
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_summary_contains_feature_contract() {
    let diag = PolicyDiagnostics::current();
    let summary = diag.summary();
    assert!(
        summary.contains("feature-contract"),
        "summary should mention feature-contract: {summary}"
    );
}

// ---------------------------------------------------------------------------
// PolicyDiagnostics: Debug + Clone
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_debug() {
    let diag = PolicyDiagnostics::current();
    let d = format!("{:?}", diag);
    assert!(d.contains("PolicyDiagnostics"));
}

#[test]
fn diagnostics_clone() {
    let diag = PolicyDiagnostics::current();
    let diag2 = diag.clone();
    assert_eq!(diag.summary(), diag2.summary());
}

// ---------------------------------------------------------------------------
// convenience function: diagnostics()
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_convenience_not_empty() {
    let diag = diagnostics();
    assert!(!diag.summary().is_empty());
}

// ---------------------------------------------------------------------------
// convenience function: diagnostics_for_context()
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_for_context_callable() {
    let ctx = ConfigurationContext::default();
    let diag = diagnostics_for_context(&ctx);
    assert!(!diag.summary().is_empty());
}

// ---------------------------------------------------------------------------
// validate_active_profile_from_context
// ---------------------------------------------------------------------------

#[test]
fn validate_active_profile_from_context_unit_local() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let result = validate_active_profile_from_context(&ctx);
    assert!(result.is_some(), "Unit/Local should have a grid cell");
}

#[test]
fn validate_active_profile_from_context_default() {
    let ctx = ConfigurationContext::default();
    let result = validate_active_profile_from_context(&ctx);
    assert!(result.is_some());
}

// ---------------------------------------------------------------------------
// All scenarios produce diagnostics
// ---------------------------------------------------------------------------

#[test]
fn all_scenarios_produce_diagnostics() {
    let scenarios = [
        TestingScenario::Unit,
        TestingScenario::Integration,
        TestingScenario::EndToEnd,
        TestingScenario::Performance,
        TestingScenario::CrossValidation,
        TestingScenario::Smoke,
        TestingScenario::Development,
        TestingScenario::Debug,
        TestingScenario::Minimal,
    ];
    for scenario in &scenarios {
        let ctx = ConfigurationContext {
            scenario: *scenario,
            environment: ExecutionEnvironment::Local,
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        };
        let diag = PolicyDiagnostics::from_context(&ctx);
        let _ = diag.summary();
        let _ = diag.is_grid_compatible();
        let _ = diag.is_feature_contract_consistent();
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

#[test]
fn grid_scenario_alias() {
    use bitnet_testing_policy_tests::GridScenario;
    let _: GridScenario = TestingScenario::Unit;
}

#[test]
fn grid_environment_alias() {
    use bitnet_testing_policy_tests::GridEnvironment;
    let _: GridEnvironment = ExecutionEnvironment::Local;
}
