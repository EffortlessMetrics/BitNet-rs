//! BDD Grid Enforcement Tests
//!
//! These tests verify that the curated BDD grid is internally consistent and
//! that the currently-active feature flags satisfy the constraints for each
//! applicable grid cell.
//!
//! **This is executable policy** — it will fail CI if:
//! - Grid cells list duplicate (scenario, environment) pairs
//! - A cell's required features overlap its forbidden features
//! - The active feature set violates forbidden constraints for applicable cells
//!
//! Non-applicable cells (wrong environment, missing required features) are
//! skipped with a note rather than failed, so the suite stays green in partial
//! feature builds.

use bitnet_bdd_grid::{BddGrid, ExecutionEnvironment, FeatureSet, TestingScenario, curated};
use bitnet_runtime_feature_flags::active_features;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Structural invariants (always checked, no external deps)
// ---------------------------------------------------------------------------

#[test]
fn grid_has_no_duplicate_scenario_environment_pairs() {
    let grid = curated();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for cell in grid.rows() {
        let key = (format!("{:?}", cell.scenario), format!("{:?}", cell.environment));
        assert!(
            seen.insert(key.clone()),
            "Duplicate BDD cell: scenario={} environment={}",
            key.0,
            key.1
        );
    }
}

#[test]
fn no_cell_has_required_feature_in_forbidden() {
    let grid = curated();
    for cell in grid.rows() {
        let overlap = cell.required_features.forbidden_overlap(&cell.forbidden_features);
        assert!(
            overlap.is_empty(),
            "BDD cell {:?}/{:?} has features in both required and forbidden: {:?}",
            cell.scenario,
            cell.environment,
            overlap.labels()
        );
    }
}

#[test]
fn all_cells_have_non_empty_intent() {
    let grid = curated();
    for cell in grid.rows() {
        assert!(
            !cell.intent.is_empty(),
            "BDD cell {:?}/{:?} has empty intent string",
            cell.scenario,
            cell.environment
        );
    }
}

#[test]
fn grid_covers_all_required_scenarios() {
    let grid = curated();
    let required_scenarios = [
        TestingScenario::Unit,
        TestingScenario::Integration,
        TestingScenario::Smoke,
        TestingScenario::Performance,
    ];
    for scenario in &required_scenarios {
        let found = grid.rows_for_scenario(*scenario);
        assert!(
            !found.is_empty(),
            "BDD grid missing required scenario: {:?}",
            scenario
        );
    }
}

// ---------------------------------------------------------------------------
// Active-feature constraint checks
// (these validate the current build against applicable cells)
// ---------------------------------------------------------------------------

#[test]
fn active_features_do_not_violate_forbidden_for_ci_cells() {
    let grid = curated();
    let active = active_features();

    for cell in grid.rows() {
        if cell.environment != ExecutionEnvironment::Ci {
            continue;
        }
        let (_, forbidden_violations) = cell.violations(&active);
        assert!(
            forbidden_violations.is_empty(),
            "Active feature set violates forbidden constraints for CI cell \
             {:?}/{:?}. Forbidden features present: {:?}. \
             This means the build has mutually-exclusive features enabled.",
            cell.scenario,
            cell.environment,
            forbidden_violations.labels()
        );
    }
}

#[test]
fn grid_lookup_returns_correct_cell() {
    let grid = curated();

    // Unit/Local is the most commonly-exercised cell; verify we can find it
    let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local);
    assert!(
        cell.is_some(),
        "Curated BDD grid must contain a Unit/Local cell"
    );
    let cell = cell.unwrap();
    assert_eq!(cell.scenario, TestingScenario::Unit);
    assert_eq!(cell.environment, ExecutionEnvironment::Local);
}

#[test]
fn smoke_scenario_has_minimal_required_features() {
    let grid = curated();
    let cells = grid.rows_for_scenario(TestingScenario::Smoke);
    assert!(!cells.is_empty(), "smoke scenario must have at least one cell");

    for cell in cells {
        // Smoke tests should be runnable without GPU or FFI
        assert!(
            !cell.required_features.contains(bitnet_bdd_grid::BitnetFeature::Gpu),
            "Smoke scenario cell {:?} requires GPU — smoke tests should be GPU-optional",
            cell.environment
        );
        assert!(
            !cell.required_features.contains(bitnet_bdd_grid::BitnetFeature::Ffi),
            "Smoke scenario cell {:?} requires FFI — smoke tests should be FFI-optional",
            cell.environment
        );
    }
}

#[test]
fn active_feature_set_is_internally_consistent() {
    let active = active_features();

    // gpu + cuda: having neither is fine; having cuda without gpu is a warning
    // (not enforced here — just validated that the feature line is non-empty if
    //  features are active)
    let labels = active.labels();
    for label in &labels {
        assert!(!label.is_empty(), "Feature label must not be empty");
    }

    // Verify no duplicate labels
    let unique: HashSet<_> = labels.iter().collect();
    assert_eq!(unique.len(), labels.len(), "Active feature labels must be unique");
}
