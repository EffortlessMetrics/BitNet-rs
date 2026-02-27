//! Property-based tests for bitnet-bdd-grid.
//!
//! Key invariants tested:
//! - curated() grid is non-empty and has consistent row count
//! - Every cell's required/forbidden sets are disjoint
//! - find() is consistent with rows()
//! - supports() and violations() are inverses
//! - Cell intent strings are non-empty
//! - All (scenario, environment) pairs in curated() are unique
//! - Every feature label string is non-empty
//! - feature_set_from_names with empty input produces empty set
//! - supports() returns false when required features are absent

use bitnet_bdd_grid::{
    ExecutionEnvironment, FeatureSet, TestingScenario, curated, feature_set_from_names,
};
use proptest::prelude::*;

proptest! {
    /// curated() always returns the same row count (deterministic policy).
    #[test]
    fn curated_row_count_is_stable(_seed in 0u64..1000) {
        let count = curated().rows().len();
        prop_assert!(count > 0, "curated grid must be non-empty");
        // Run twice — lazy-static must be stable
        let count2 = curated().rows().len();
        prop_assert_eq!(count, count2);
    }

    /// Every cell's required and forbidden sets are disjoint.
    #[test]
    fn required_and_forbidden_always_disjoint(_seed in 0u64..1000) {
        for cell in curated().rows() {
            let overlap = cell.required_features.forbidden_overlap(&cell.forbidden_features);
            prop_assert!(
                overlap.is_empty(),
                "Cell {:?}/{:?}: required∩forbidden = {:?}",
                cell.scenario,
                cell.environment,
                overlap.labels()
            );
        }
    }

    /// find() returns the same cell as rows() iteration.
    #[test]
    fn find_matches_rows_iteration(_seed in 0u64..1000) {
        let grid = curated();
        for cell in grid.rows() {
            let found = grid.find(cell.scenario, cell.environment);
            prop_assert!(found.is_some(), "find() should always find a cell from rows()");
            let found_cell = found.unwrap();
            prop_assert_eq!(
                found_cell.intent,
                cell.intent,
                "find() returned different cell from rows()"
            );
        }
    }

    /// A feature set satisfying required+forbidden constraints always passes supports().
    #[test]
    fn supports_is_consistent_with_violations(_seed in 0u64..1000) {
        let grid = curated();
        // Build a feature set that satisfies the first cell
        if let Some(cell) = grid.rows().first() {
            let active = cell.required_features.clone();
            let supports = cell.supports(&active);
            let (missing, violated) = cell.violations(&active);
            if missing.is_empty() && violated.is_empty() {
                prop_assert!(supports, "supports() should return true when no violations");
            } else {
                prop_assert!(!supports || missing.is_empty(), "inconsistent supports/violations");
            }
        }
    }

    /// All cell intent strings are non-empty.
    #[test]
    fn all_intents_non_empty(_seed in 0u64..1000) {
        for cell in curated().rows() {
            prop_assert!(!cell.intent.is_empty(), "Cell {:?}/{:?} has empty intent", cell.scenario, cell.environment);
        }
    }

    /// Every (scenario, environment) pair in curated() is unique — no two cells share a key.
    #[test]
    fn all_scenario_env_pairs_are_unique(_seed in 0u64..1000) {
        let rows = curated();
        let cells = rows.rows();
        for (i, a) in cells.iter().enumerate() {
            for b in cells.iter().skip(i + 1) {
                prop_assert!(
                    !(a.scenario == b.scenario && a.environment == b.environment),
                    "Duplicate cell for {:?}/{:?}",
                    a.scenario,
                    a.environment,
                );
            }
        }
    }

    /// Every required-feature label returned by labels() is a non-empty string.
    #[test]
    fn every_required_feature_label_is_non_empty(_seed in 0u64..1000) {
        for cell in curated().rows() {
            for label in cell.required_features.labels() {
                prop_assert!(
                    !label.is_empty(),
                    "Empty feature label in required_features for {:?}/{:?}",
                    cell.scenario,
                    cell.environment
                );
            }
        }
    }

    /// `feature_set_from_names` with an empty slice always produces an empty FeatureSet.
    #[test]
    fn feature_set_from_names_empty_input_is_empty(_seed in 0u64..1000) {
        let set = feature_set_from_names(&[]);
        prop_assert!(set.is_empty(), "feature_set_from_names(&[]) must return an empty set");
    }

    /// `supports()` returns false when the active feature set is empty but required_features
    /// is non-empty — absent required features always block support.
    #[test]
    fn supports_false_when_required_features_absent(_seed in 0u64..1000) {
        let empty = FeatureSet::new();
        for cell in curated().rows() {
            if !cell.required_features.is_empty() {
                prop_assert!(
                    !cell.supports(&empty),
                    "Cell {:?}/{:?}: supports() must be false when active set is empty \
                     but required_features is non-empty",
                    cell.scenario,
                    cell.environment,
                );
            }
        }
    }
}

#[test]
fn curated_grid_has_expected_cell_count() {
    // Structural invariant: curated grid has exactly 22 cells
    assert_eq!(curated().rows().len(), 22, "curated grid should have 22 cells");
}

#[test]
fn unit_local_cell_exists() {
    let cell = curated().find(TestingScenario::Unit, ExecutionEnvironment::Local);
    assert!(cell.is_some(), "Unit/Local cell must exist in curated grid");
}

#[test]
fn smoke_preprod_cell_exists() {
    let cell = curated().find(TestingScenario::Smoke, ExecutionEnvironment::PreProduction);
    assert!(cell.is_some(), "Smoke/PreProduction cell must exist in curated grid");
}
