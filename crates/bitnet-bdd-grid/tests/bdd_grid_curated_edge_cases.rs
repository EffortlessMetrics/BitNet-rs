//! Edge-case tests for bitnet-bdd-grid curated grid API.

use bitnet_bdd_grid::{
    BddGrid, ExecutionEnvironment, TestingScenario, curated, feature_set_from_names,
};

// ---------------------------------------------------------------------------
// curated() grid construction
// ---------------------------------------------------------------------------

#[test]
fn curated_grid_is_non_empty() {
    let grid = curated();
    assert!(grid.rows().len() > 0, "curated grid should have rows");
}

#[test]
fn curated_grid_has_unit_local() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local);
    assert!(cell.is_some(), "Unit/Local should exist in curated grid");
}

#[test]
fn curated_grid_has_integration_local() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Integration, ExecutionEnvironment::Local);
    assert!(cell.is_some(), "Integration/Local should exist");
}

#[test]
fn curated_grid_has_integration_ci() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Integration, ExecutionEnvironment::Ci);
    assert!(cell.is_some(), "Integration/CI should exist");
}

#[test]
fn curated_grid_has_smoke_ci() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Smoke, ExecutionEnvironment::Ci);
    assert!(cell.is_some(), "Smoke/CI should exist");
}

#[test]
fn curated_grid_has_performance_ci() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Performance, ExecutionEnvironment::Ci);
    assert!(cell.is_some(), "Performance/CI should exist");
}

// ---------------------------------------------------------------------------
// curated() cell requirements
// ---------------------------------------------------------------------------

#[test]
fn unit_local_requires_core_features() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local).unwrap();
    let features = feature_set_from_names(&["inference", "kernels", "tokenizers"]);
    assert!(cell.supports(&features), "Unit/Local requires inference+kernels+tokenizers");
}

#[test]
fn unit_local_forbids_cpp_ffi() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local).unwrap();
    let features = feature_set_from_names(&["inference", "kernels", "tokenizers", "cpp-ffi"]);
    // cpp-ffi is forbidden in Unit/Local
    let (_, forbidden) = cell.violations(&features);
    assert!(!forbidden.is_empty(), "cpp-ffi should be forbidden in Unit/Local");
}

#[test]
fn unit_local_missing_features_detected() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local).unwrap();
    let empty = feature_set_from_names(&[]);
    let (missing, _) = cell.violations(&empty);
    assert!(!missing.is_empty(), "empty feature set should have missing required features");
}

// ---------------------------------------------------------------------------
// curated() is deterministic
// ---------------------------------------------------------------------------

#[test]
fn curated_grid_is_deterministic() {
    let grid1 = curated();
    let grid2 = curated();
    assert_eq!(grid1.rows().len(), grid2.rows().len());
    for (r1, r2) in grid1.rows().iter().zip(grid2.rows().iter()) {
        assert_eq!(r1.scenario, r2.scenario);
        assert_eq!(r1.environment, r2.environment);
    }
}

// ---------------------------------------------------------------------------
// All curated rows have non-empty intent
// ---------------------------------------------------------------------------

#[test]
fn all_curated_rows_have_intent() {
    let grid = curated();
    for row in grid.rows() {
        assert!(
            !row.intent.is_empty(),
            "row {:?}/{:?} should have intent",
            row.scenario,
            row.environment
        );
    }
}

// ---------------------------------------------------------------------------
// All curated rows have required features
// ---------------------------------------------------------------------------

#[test]
fn all_curated_rows_have_required_features() {
    let grid = curated();
    for row in grid.rows() {
        assert!(
            !row.required_features.is_empty(),
            "row {:?}/{:?} should have required features",
            row.scenario,
            row.environment
        );
    }
}

// ---------------------------------------------------------------------------
// Unique scenario/environment pairs
// ---------------------------------------------------------------------------

#[test]
fn curated_rows_have_unique_scenario_environment() {
    let grid = curated();
    let mut seen = std::collections::HashSet::new();
    for row in grid.rows() {
        let key = (row.scenario, row.environment);
        assert!(seen.insert(key), "duplicate row for {:?}/{:?}", row.scenario, row.environment);
    }
}

// ---------------------------------------------------------------------------
// feature_set_from_names
// ---------------------------------------------------------------------------

#[test]
fn feature_set_from_empty_names() {
    let fs = feature_set_from_names(&[]);
    assert!(fs.is_empty());
}

#[test]
fn feature_set_from_single_name() {
    let fs = feature_set_from_names(&["cpu"]);
    assert!(!fs.is_empty());
    assert!(fs.labels().contains(&"cpu".to_string()));
}

#[test]
fn feature_set_from_multiple_names() {
    let fs = feature_set_from_names(&["cpu", "gpu", "inference"]);
    let labels = fs.labels();
    assert!(labels.contains(&"cpu".to_string()));
    assert!(labels.contains(&"gpu".to_string()));
    assert!(labels.contains(&"inference".to_string()));
}

#[test]
fn feature_set_from_unknown_name_skipped() {
    // Unknown feature names should be silently skipped
    let fs = feature_set_from_names(&["nonexistent_feature_xyz"]);
    assert!(fs.is_empty());
}

// ---------------------------------------------------------------------------
// BddGrid re-export works
// ---------------------------------------------------------------------------

#[test]
fn bdd_grid_type_usable() {
    let grid = BddGrid::from_rows(&[]);
    assert_eq!(grid.rows().len(), 0);
}
