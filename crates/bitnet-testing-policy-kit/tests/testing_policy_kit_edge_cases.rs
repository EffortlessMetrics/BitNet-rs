//! Edge-case tests for bitnet-testing-policy-kit helpers.

use bitnet_testing_policy_kit::{
    ExecutionEnvironment, TestingScenario, active_feature_labels, active_profile_from_environment,
    profile_snapshot, resolve_active_profile, validate_active_profile_from_environment,
    validate_profile,
};

// ---------------------------------------------------------------------------
// active_feature_labels
// ---------------------------------------------------------------------------

#[test]
fn active_feature_labels_not_empty() {
    let labels = active_feature_labels();
    // Some features are always present in test builds
    assert!(!labels.is_empty());
}

#[test]
fn active_feature_labels_are_strings() {
    let labels = active_feature_labels();
    for label in &labels {
        assert!(!label.is_empty());
    }
}

// ---------------------------------------------------------------------------
// active_profile_from_environment
// ---------------------------------------------------------------------------

#[test]
fn active_profile_from_environment_has_scenario() {
    let profile = active_profile_from_environment();
    assert!(!profile.scenario.to_string().is_empty());
}

#[test]
fn active_profile_from_environment_has_features() {
    let profile = active_profile_from_environment();
    assert!(!profile.features.labels().is_empty());
}

// ---------------------------------------------------------------------------
// validate_active_profile_from_environment
// ---------------------------------------------------------------------------

#[test]
fn validate_active_profile_from_environment_callable() {
    // May be Some or None
    let _ = validate_active_profile_from_environment();
}

// ---------------------------------------------------------------------------
// profile_snapshot
// ---------------------------------------------------------------------------

#[test]
fn profile_snapshot_returns_consistent_tuple() {
    let (profile, context, violations) = profile_snapshot();
    assert_eq!(profile.scenario, context.scenario);
    assert_eq!(profile.environment, context.environment);
    // violations may be Some or None
    let _ = violations;
}

#[test]
fn profile_snapshot_profile_has_features() {
    let (profile, _, _) = profile_snapshot();
    assert!(!profile.features.labels().is_empty());
}

// ---------------------------------------------------------------------------
// resolve_active_profile
// ---------------------------------------------------------------------------

#[test]
fn resolve_active_profile_has_positive_parallelism() {
    let config = resolve_active_profile();
    assert!(config.max_parallel_tests > 0);
}

#[test]
fn resolve_active_profile_has_non_empty_log_level() {
    let config = resolve_active_profile();
    assert!(!config.log_level.is_empty());
}

#[test]
fn resolve_active_profile_has_reporting_formats() {
    let config = resolve_active_profile();
    assert!(!config.reporting.formats.is_empty());
}

// ---------------------------------------------------------------------------
// validate_profile (backward compat)
// ---------------------------------------------------------------------------

#[test]
fn validate_profile_callable() {
    let _ = validate_profile();
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

#[test]
fn environment_type_alias_works() {
    use bitnet_testing_policy_kit::EnvironmentType;
    let _: EnvironmentType = ExecutionEnvironment::Local;
}

#[test]
fn grid_scenario_alias_works() {
    use bitnet_testing_policy_kit::GridScenario;
    let _: GridScenario = TestingScenario::Unit;
}

#[test]
fn grid_environment_alias_works() {
    use bitnet_testing_policy_kit::GridEnvironment;
    let _: GridEnvironment = ExecutionEnvironment::Ci;
}

// ---------------------------------------------------------------------------
// Re-exported helpers from underlying crates
// ---------------------------------------------------------------------------

#[test]
fn canonical_grid_accessible() {
    use bitnet_testing_policy_kit::canonical_grid;
    let grid = canonical_grid();
    assert!(!grid.rows().is_empty());
}

#[test]
fn feature_line_not_empty() {
    use bitnet_testing_policy_kit::feature_line;
    let line = feature_line();
    assert!(!line.is_empty());
}

#[test]
fn active_profile_summary_not_empty() {
    use bitnet_testing_policy_kit::active_profile_summary;
    let summary = active_profile_summary();
    assert!(!summary.is_empty());
}

#[test]
fn to_grid_scenario_identity() {
    use bitnet_testing_policy_kit::to_grid_scenario;
    assert_eq!(to_grid_scenario(TestingScenario::Unit), TestingScenario::Unit);
    assert_eq!(to_grid_scenario(TestingScenario::Integration), TestingScenario::Integration);
}

#[test]
fn from_grid_scenario_identity() {
    use bitnet_testing_policy_kit::from_grid_scenario;
    assert_eq!(from_grid_scenario(TestingScenario::Performance), TestingScenario::Performance);
}

#[test]
fn validate_explicit_profile_unit_local() {
    use bitnet_testing_policy_kit::validate_explicit_profile;
    let result = validate_explicit_profile(TestingScenario::Unit, ExecutionEnvironment::Local);
    // Unit/Local should have a grid cell
    assert!(result.is_some());
}
