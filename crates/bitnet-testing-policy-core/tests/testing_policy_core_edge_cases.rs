//! Edge-case tests for bitnet-testing-policy-core PolicySnapshot and helpers.

use bitnet_testing_policy_core::{
    ActiveContext, ConfigurationContext, ExecutionEnvironment, PolicySnapshot, TestingScenario,
    active_context, resolve_context_profile, snapshot_from_env, validate_context,
    validate_explicit_profile,
};

// ---------------------------------------------------------------------------
// PolicySnapshot: detect (from env, non-mutating)
// ---------------------------------------------------------------------------

#[test]
fn snapshot_detect_produces_valid_context() {
    let snap = PolicySnapshot::detect();
    assert!(!snap.context.scenario.to_string().is_empty());
    assert!(!snap.context.environment.to_string().is_empty());
}

#[test]
fn snapshot_detect_resolved_config_has_positive_parallelism() {
    let snap = PolicySnapshot::detect();
    assert!(snap.resolved_config.max_parallel_tests > 0);
}

#[test]
fn snapshot_detect_scenario_config_has_positive_parallelism() {
    let snap = PolicySnapshot::detect();
    assert!(snap.scenario_config.max_parallel_tests > 0);
}

#[test]
fn snapshot_detect_environment_config_has_positive_parallelism() {
    let snap = PolicySnapshot::detect();
    assert!(snap.environment_config.max_parallel_tests > 0);
}

// ---------------------------------------------------------------------------
// PolicySnapshot: from_active_context with explicit context
// ---------------------------------------------------------------------------

#[test]
fn snapshot_from_active_context_unit_local() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let snap = PolicySnapshot::from_active_context(ctx);
    assert_eq!(snap.context.scenario, TestingScenario::Unit);
    assert_eq!(snap.context.environment, ExecutionEnvironment::Local);
}

#[test]
fn snapshot_from_active_context_integration_ci() {
    let ctx = ActiveContext {
        scenario: TestingScenario::Integration,
        environment: ExecutionEnvironment::Ci,
    };
    let snap = PolicySnapshot::from_active_context(ctx);
    assert_eq!(snap.context.scenario, TestingScenario::Integration);
    assert_eq!(snap.context.environment, ExecutionEnvironment::Ci);
}

#[test]
fn snapshot_from_active_context_e2e_production() {
    let ctx = ActiveContext {
        scenario: TestingScenario::EndToEnd,
        environment: ExecutionEnvironment::Production,
    };
    let snap = PolicySnapshot::from_active_context(ctx);
    assert_eq!(snap.context.scenario, TestingScenario::EndToEnd);
    assert_eq!(snap.context.environment, ExecutionEnvironment::Production);
}

// ---------------------------------------------------------------------------
// PolicySnapshot: from_configuration_context
// ---------------------------------------------------------------------------

#[test]
fn snapshot_from_configuration_context_default() {
    let ctx = ConfigurationContext::default();
    let snap = PolicySnapshot::from_configuration_context(ctx);
    assert!(snap.resolved_config.max_parallel_tests > 0);
}

#[test]
fn snapshot_from_configuration_context_smoke_local() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Smoke,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let snap = PolicySnapshot::from_configuration_context(ctx);
    assert_eq!(snap.context.scenario, TestingScenario::Smoke);
    assert_eq!(snap.context.environment, ExecutionEnvironment::Local);
}

// ---------------------------------------------------------------------------
// PolicySnapshot: is_compatible
// ---------------------------------------------------------------------------

#[test]
fn snapshot_unit_local_compatibility() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let snap = PolicySnapshot::from_active_context(ctx);
    // Should have a grid cell (Unit/Local is in the canonical grid)
    let _ = snap.is_compatible(); // Just verify no panic
}

#[test]
fn snapshot_performance_production_may_lack_grid_cell() {
    let ctx = ActiveContext {
        scenario: TestingScenario::Performance,
        environment: ExecutionEnvironment::Production,
    };
    let snap = PolicySnapshot::from_active_context(ctx);
    // This combination may not have a grid cell
    let _compat = snap.is_compatible();
}

// ---------------------------------------------------------------------------
// PolicySnapshot: violations
// ---------------------------------------------------------------------------

#[test]
fn snapshot_violations_unit_local() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let snap = PolicySnapshot::from_active_context(ctx);
    let violations = snap.violations();
    // Unit/Local should have a grid cell
    assert!(violations.is_some());
}

#[test]
fn snapshot_violations_returns_feature_sets() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let snap = PolicySnapshot::from_active_context(ctx);
    if let Some((missing, forbidden)) = snap.violations() {
        // Both should be FeatureSets (may or may not be empty)
        let _ = missing.labels();
        let _ = forbidden.labels();
    }
}

// ---------------------------------------------------------------------------
// PolicySnapshot: summary
// ---------------------------------------------------------------------------

#[test]
fn snapshot_summary_unit_local_contains_scenario() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let snap = PolicySnapshot::from_active_context(ctx);
    let summary = snap.summary();
    assert!(summary.contains("scenario=unit"), "summary: {summary}");
}

#[test]
fn snapshot_summary_unit_local_contains_environment() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let snap = PolicySnapshot::from_active_context(ctx);
    let summary = snap.summary();
    assert!(summary.contains("environment=local"), "summary: {summary}");
}

#[test]
fn snapshot_summary_no_grid_cell_contains_marker() {
    let ctx = ActiveContext {
        scenario: TestingScenario::Debug,
        environment: ExecutionEnvironment::Production,
    };
    let snap = PolicySnapshot::from_active_context(ctx);
    let summary = snap.summary();
    // Should contain "no matching grid cell" or "required="
    assert!(
        summary.contains("no matching grid cell") || summary.contains("required="),
        "summary: {summary}"
    );
}

// ---------------------------------------------------------------------------
// PolicySnapshot: Debug/Clone
// ---------------------------------------------------------------------------

#[test]
fn snapshot_debug_not_empty() {
    let snap = PolicySnapshot::detect();
    let d = format!("{:?}", snap);
    assert!(!d.is_empty());
    assert!(d.contains("PolicySnapshot"));
}

#[test]
fn snapshot_clone() {
    let snap = PolicySnapshot::detect();
    let snap2 = snap.clone();
    assert_eq!(snap.summary(), snap2.summary());
}

// ---------------------------------------------------------------------------
// snapshot_from_env helper
// ---------------------------------------------------------------------------

#[test]
fn snapshot_from_env_returns_valid_snapshot() {
    let snap = snapshot_from_env();
    assert!(!snap.summary().is_empty());
}

// ---------------------------------------------------------------------------
// resolve_context_profile
// ---------------------------------------------------------------------------

#[test]
fn resolve_context_profile_default_has_reporting() {
    let ctx = ConfigurationContext::default();
    let resolved = resolve_context_profile(&ctx);
    assert!(!resolved.reporting.formats.is_empty());
}

#[test]
fn resolve_context_profile_smoke_local() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Smoke,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let resolved = resolve_context_profile(&ctx);
    assert!(resolved.max_parallel_tests > 0);
}

// ---------------------------------------------------------------------------
// active_context helper
// ---------------------------------------------------------------------------

#[test]
fn active_context_preserves_scenario_environment() {
    let config = ConfigurationContext {
        scenario: TestingScenario::Performance,
        environment: ExecutionEnvironment::Ci,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    let ctx = active_context(&config);
    assert_eq!(ctx.scenario, TestingScenario::Performance);
    assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
}

// ---------------------------------------------------------------------------
// validate_context
// ---------------------------------------------------------------------------

#[test]
fn validate_context_unit_local_returns_some() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        resource_constraints: None,
        time_constraints: None,
        quality_requirements: None,
        platform_settings: None,
    };
    assert!(validate_context(&ctx).is_some());
}

// ---------------------------------------------------------------------------
// validate_explicit_profile
// ---------------------------------------------------------------------------

#[test]
fn validate_explicit_unit_local() {
    assert!(
        validate_explicit_profile(TestingScenario::Unit, ExecutionEnvironment::Local).is_some()
    );
}

#[test]
fn validate_explicit_integration_ci() {
    // Integration/Ci may or may not have a grid cell
    let _ = validate_explicit_profile(TestingScenario::Integration, ExecutionEnvironment::Ci);
}

// ---------------------------------------------------------------------------
// All scenario/environment combos don't panic
// ---------------------------------------------------------------------------

#[test]
fn all_scenarios_dont_panic_with_local() {
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
        let ctx = ActiveContext { scenario: *scenario, environment: ExecutionEnvironment::Local };
        let snap = PolicySnapshot::from_active_context(ctx);
        let _ = snap.summary();
        let _ = snap.is_compatible();
        let _ = snap.violations();
    }
}

#[test]
fn all_environments_dont_panic_with_unit() {
    let envs = [
        ExecutionEnvironment::Local,
        ExecutionEnvironment::Ci,
        ExecutionEnvironment::PreProduction,
        ExecutionEnvironment::Production,
    ];
    for env in &envs {
        let ctx = ActiveContext { scenario: TestingScenario::Unit, environment: *env };
        let snap = PolicySnapshot::from_active_context(ctx);
        let _ = snap.summary();
        let _ = snap.is_compatible();
    }
}
