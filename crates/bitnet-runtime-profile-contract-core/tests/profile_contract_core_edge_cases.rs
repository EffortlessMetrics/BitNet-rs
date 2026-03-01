//! Edge-case tests for bitnet-runtime-profile-contract-core ActiveProfile and helpers.

use bitnet_runtime_profile_contract_core::{
    ActiveContext, ActiveProfile, ExecutionEnvironment, TestingScenario, active_profile_for,
    active_profile_summary, active_profile_violation_labels, canonical_grid,
    validate_active_profile_for,
};

// ---------------------------------------------------------------------------
// ActiveProfile: from_context with explicit context
// ---------------------------------------------------------------------------

#[test]
fn active_profile_unit_local() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let profile = ActiveProfile::from_context(ctx);
    assert_eq!(profile.scenario, TestingScenario::Unit);
    assert_eq!(profile.environment, ExecutionEnvironment::Local);
}

#[test]
fn active_profile_unit_local_has_cell() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let profile = ActiveProfile::from_context(ctx);
    // Unit/Local should be in the canonical grid
    assert!(profile.cell.is_some(), "Unit/Local should have a grid cell");
}

#[test]
fn active_profile_features_not_empty() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let profile = ActiveProfile::from_context(ctx);
    // At minimum, cpu feature should be active in test builds
    let labels = profile.features.labels();
    assert!(!labels.is_empty(), "active features should not be empty");
}

#[test]
fn active_profile_integration_ci() {
    let ctx = ActiveContext {
        scenario: TestingScenario::Integration,
        environment: ExecutionEnvironment::Ci,
    };
    let profile = ActiveProfile::from_context(ctx);
    assert_eq!(profile.scenario, TestingScenario::Integration);
    assert_eq!(profile.environment, ExecutionEnvironment::Ci);
}

// ---------------------------------------------------------------------------
// ActiveProfile: is_supported
// ---------------------------------------------------------------------------

#[test]
fn active_profile_is_supported_unit_local() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let profile = ActiveProfile::from_context(ctx);
    // Just verify no panic; result depends on active features
    let _ = profile.is_supported();
}

#[test]
fn active_profile_is_supported_false_when_no_cell() {
    let ctx = ActiveContext {
        scenario: TestingScenario::Debug,
        environment: ExecutionEnvironment::Production,
    };
    let profile = ActiveProfile::from_context(ctx);
    if profile.cell.is_none() {
        assert!(!profile.is_supported());
    }
}

// ---------------------------------------------------------------------------
// ActiveProfile: violations
// ---------------------------------------------------------------------------

#[test]
fn active_profile_violations_unit_local() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let profile = ActiveProfile::from_context(ctx);
    let (missing, forbidden) = profile.violations();
    let _ = missing.labels();
    let _ = forbidden.labels();
}

#[test]
fn active_profile_violations_no_cell_returns_empty() {
    let ctx = ActiveContext {
        scenario: TestingScenario::Debug,
        environment: ExecutionEnvironment::Production,
    };
    let profile = ActiveProfile::from_context(ctx);
    if profile.cell.is_none() {
        let (missing, forbidden) = profile.violations();
        assert!(missing.is_empty());
        assert!(forbidden.is_empty());
    }
}

// ---------------------------------------------------------------------------
// ActiveProfile: missing and forbidden helpers
// ---------------------------------------------------------------------------

#[test]
fn active_profile_missing_returns_labels() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let profile = ActiveProfile::from_context(ctx);
    let missing = profile.missing();
    for label in &missing {
        assert!(!label.is_empty());
    }
}

#[test]
fn active_profile_forbidden_returns_labels() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let profile = ActiveProfile::from_context(ctx);
    let forbidden = profile.forbidden();
    for label in &forbidden {
        assert!(!label.is_empty());
    }
}

// ---------------------------------------------------------------------------
// ActiveProfile: Default impl
// ---------------------------------------------------------------------------

#[test]
fn active_profile_default_has_features() {
    let profile = ActiveProfile::default();
    assert!(!profile.features.labels().is_empty());
}

// ---------------------------------------------------------------------------
// ActiveProfile: Debug/Clone
// ---------------------------------------------------------------------------

#[test]
fn active_profile_debug_not_empty() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let profile = ActiveProfile::from_context(ctx);
    let d = format!("{:?}", profile);
    assert!(!d.is_empty());
    assert!(d.contains("ActiveProfile"));
}

#[test]
fn active_profile_clone() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let profile = ActiveProfile::from_context(ctx);
    let profile2 = profile.clone();
    assert_eq!(profile.scenario, profile2.scenario);
    assert_eq!(profile.environment, profile2.environment);
}

// ---------------------------------------------------------------------------
// canonical_grid
// ---------------------------------------------------------------------------

#[test]
fn canonical_grid_has_rows() {
    let grid = canonical_grid();
    assert!(!grid.rows().is_empty());
}

#[test]
fn canonical_grid_has_unit_local() {
    let grid = canonical_grid();
    let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local);
    assert!(cell.is_some(), "canonical grid should have Unit/Local");
}

#[test]
fn canonical_grid_unit_local_intent_not_empty() {
    let grid = canonical_grid();
    let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local).unwrap();
    assert!(!cell.intent.is_empty());
}

// ---------------------------------------------------------------------------
// active_profile_for helper
// ---------------------------------------------------------------------------

#[test]
fn active_profile_for_unit_local() {
    let profile = active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Local);
    assert_eq!(profile.scenario, TestingScenario::Unit);
    assert_eq!(profile.environment, ExecutionEnvironment::Local);
}

#[test]
fn active_profile_for_all_scenarios() {
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
        let profile = active_profile_for(*scenario, ExecutionEnvironment::Local);
        assert_eq!(profile.scenario, *scenario);
    }
}

// ---------------------------------------------------------------------------
// validate_active_profile_for
// ---------------------------------------------------------------------------

#[test]
fn validate_unit_local_returns_some() {
    assert!(
        validate_active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Local).is_some()
    );
}

#[test]
fn validate_returns_none_for_missing_grid_cell() {
    // Try a combo unlikely to be in the grid
    let result =
        validate_active_profile_for(TestingScenario::Debug, ExecutionEnvironment::Production);
    // If no grid cell, should return None
    if canonical_grid().find(TestingScenario::Debug, ExecutionEnvironment::Production).is_none() {
        assert!(result.is_none());
    }
}

// ---------------------------------------------------------------------------
// active_profile_summary
// ---------------------------------------------------------------------------

#[test]
fn active_profile_summary_not_empty() {
    let summary = active_profile_summary();
    assert!(!summary.is_empty());
}

#[test]
fn active_profile_summary_contains_scenario() {
    let summary = active_profile_summary();
    assert!(summary.contains("scenario="), "summary: {summary}");
}

// ---------------------------------------------------------------------------
// active_profile_violation_labels
// ---------------------------------------------------------------------------

#[test]
fn active_profile_violation_labels_is_callable() {
    // May return Some or None depending on whether there's a grid cell
    let _labels = active_profile_violation_labels();
}
