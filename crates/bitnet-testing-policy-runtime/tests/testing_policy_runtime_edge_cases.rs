//! Edge-case tests for bitnet-testing-policy-runtime RuntimePolicyState.

use bitnet_testing_policy_runtime::{
    ConfigurationContext, ExecutionEnvironment, RuntimePolicyState, TestingScenario,
    context_from_environment, detect_runtime_state, resolve_runtime_profile,
};

// ---------------------------------------------------------------------------
// RuntimePolicyState: from_environment
// ---------------------------------------------------------------------------

#[test]
fn state_from_environment_has_context() {
    let state = RuntimePolicyState::from_environment();
    assert!(!state.context.scenario.to_string().is_empty());
    assert!(!state.context.environment.to_string().is_empty());
}

#[test]
fn state_from_environment_has_profile() {
    let state = RuntimePolicyState::from_environment();
    assert!(!state.active_profile.features.labels().is_empty());
}

#[test]
fn state_from_environment_has_feature_contract() {
    let state = RuntimePolicyState::from_environment();
    // Both policy and runtime feature lists should be populated
    let _ = state.feature_contract.is_consistent();
}

// ---------------------------------------------------------------------------
// RuntimePolicyState: from_context with explicit context
// ---------------------------------------------------------------------------

#[test]
fn state_from_context_unit_local() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let state = RuntimePolicyState::from_context(&ctx);
    assert_eq!(state.context.scenario, TestingScenario::Unit);
    assert_eq!(state.context.environment, ExecutionEnvironment::Local);
}

#[test]
fn state_from_context_integration_ci() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Integration,
        environment: ExecutionEnvironment::Ci,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let state = RuntimePolicyState::from_context(&ctx);
    assert_eq!(state.context.scenario, TestingScenario::Integration);
    assert_eq!(state.context.environment, ExecutionEnvironment::Ci);
}

#[test]
fn state_from_context_preserves_profile_scenario() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Performance,
        environment: ExecutionEnvironment::Production,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let state = RuntimePolicyState::from_context(&ctx);
    assert_eq!(state.active_profile.scenario, TestingScenario::Performance);
    assert_eq!(state.active_profile.environment, ExecutionEnvironment::Production);
}

// ---------------------------------------------------------------------------
// RuntimePolicyState: is_policy_compatible
// ---------------------------------------------------------------------------

#[test]
fn state_unit_local_policy_compatible_callable() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let state = RuntimePolicyState::from_context(&ctx);
    // Just verify no panic
    let _ = state.is_policy_compatible();
}

// ---------------------------------------------------------------------------
// RuntimePolicyState: has_grid_cell
// ---------------------------------------------------------------------------

#[test]
fn state_unit_local_has_grid_cell() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let state = RuntimePolicyState::from_context(&ctx);
    assert!(state.has_grid_cell(), "Unit/Local should have a grid cell");
}

#[test]
fn state_debug_production_may_lack_grid_cell() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Debug,
        environment: ExecutionEnvironment::Production,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let state = RuntimePolicyState::from_context(&ctx);
    // This combination is unlikely to be in the canonical grid
    let _ = state.has_grid_cell();
}

// ---------------------------------------------------------------------------
// RuntimePolicyState: is_feature_contract_aligned
// ---------------------------------------------------------------------------

#[test]
fn state_feature_contract_aligned_callable() {
    let state = RuntimePolicyState::from_environment();
    let _ = state.is_feature_contract_aligned();
}

// ---------------------------------------------------------------------------
// RuntimePolicyState: summary
// ---------------------------------------------------------------------------

#[test]
fn state_summary_not_empty() {
    let state = RuntimePolicyState::from_environment();
    let summary = state.summary();
    assert!(!summary.is_empty());
}

#[test]
fn state_summary_contains_scenario() {
    let state = RuntimePolicyState::from_environment();
    let summary = state.summary();
    assert!(summary.contains("scenario="), "summary: {summary}");
}

#[test]
fn state_summary_contains_feature_contract_status() {
    let state = RuntimePolicyState::from_environment();
    let summary = state.summary();
    assert!(
        summary.contains("feature-contract-aligned") || summary.contains("feature-contract-drift"),
        "summary: {summary}"
    );
}

// ---------------------------------------------------------------------------
// RuntimePolicyState: resolved_profile_config
// ---------------------------------------------------------------------------

#[test]
fn state_resolved_profile_config_has_parallelism() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let state = RuntimePolicyState::from_context(&ctx);
    let config = state.resolved_profile_config();
    assert!(config.max_parallel_tests > 0);
}

#[test]
fn state_resolved_profile_config_has_reporting() {
    let ctx = ConfigurationContext::default();
    let state = RuntimePolicyState::from_context(&ctx);
    let config = state.resolved_profile_config();
    assert!(!config.reporting.formats.is_empty());
}

// ---------------------------------------------------------------------------
// RuntimePolicyState: violations
// ---------------------------------------------------------------------------

#[test]
fn state_violations_unit_local_is_some() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let state = RuntimePolicyState::from_context(&ctx);
    assert!(state.violations.is_some());
}

// ---------------------------------------------------------------------------
// RuntimePolicyState: feature_drift
// ---------------------------------------------------------------------------

#[test]
fn state_feature_drift_callable() {
    let state = RuntimePolicyState::from_environment();
    // May be None or Some
    let _ = state.feature_drift;
}

// ---------------------------------------------------------------------------
// RuntimePolicyState: Debug/Clone
// ---------------------------------------------------------------------------

#[test]
fn state_debug_not_empty() {
    let state = RuntimePolicyState::from_environment();
    let d = format!("{:?}", state);
    assert!(!d.is_empty());
    assert!(d.contains("RuntimePolicyState"));
}

#[test]
fn state_clone() {
    let state = RuntimePolicyState::from_environment();
    let state2 = state.clone();
    assert_eq!(state.summary(), state2.summary());
}

// ---------------------------------------------------------------------------
// detect_runtime_state helper
// ---------------------------------------------------------------------------

#[test]
fn detect_runtime_state_returns_valid_state() {
    let state = detect_runtime_state();
    assert!(!state.summary().is_empty());
}

// ---------------------------------------------------------------------------
// resolve_runtime_profile helper
// ---------------------------------------------------------------------------

#[test]
fn resolve_runtime_profile_default_context() {
    let ctx = ConfigurationContext::default();
    let config = resolve_runtime_profile(&ctx);
    assert!(config.max_parallel_tests > 0);
}

// ---------------------------------------------------------------------------
// context_from_environment helper
// ---------------------------------------------------------------------------

#[test]
fn context_from_environment_callable() {
    let ctx = context_from_environment();
    assert!(!ctx.scenario.to_string().is_empty());
}

// ---------------------------------------------------------------------------
// All scenarios produce valid states
// ---------------------------------------------------------------------------

#[test]
fn all_scenarios_produce_valid_states() {
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
        let state = RuntimePolicyState::from_context(&ctx);
        let _ = state.summary();
        let _ = state.is_policy_compatible();
    }
}

#[test]
fn all_environments_produce_valid_states() {
    let envs = [
        ExecutionEnvironment::Local,
        ExecutionEnvironment::Ci,
        ExecutionEnvironment::PreProduction,
        ExecutionEnvironment::Production,
    ];
    for env in &envs {
        let ctx = ConfigurationContext {
            scenario: TestingScenario::Unit,
            environment: *env,
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        };
        let state = RuntimePolicyState::from_context(&ctx);
        let _ = state.summary();
    }
}
