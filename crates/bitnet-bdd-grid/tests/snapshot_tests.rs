//! Snapshot tests for bitnet-bdd-grid.
//!
//! Pins the structure of the curated BDD grid: cell count, scenario/environment
//! pairs, required features, and intent strings. Any deliberate grid change
//! will require snapshot review.

use bitnet_bdd_grid::{ExecutionEnvironment, TestingScenario, curated};

/// Snapshot the entire grid as a sorted list of (scenario, environment, intent) triples.
/// This detects any accidental cell additions, removals, or intent string changes.
#[test]
fn snapshot_grid_summary() {
    let grid = curated();
    let mut summary: Vec<String> = grid
        .rows()
        .iter()
        .map(|cell| {
            format!(
                "{:?}/{:?}: required=[{}] intent={}",
                cell.scenario,
                cell.environment,
                cell.required_features.labels().join(","),
                cell.intent
            )
        })
        .collect();
    summary.sort(); // deterministic order
    insta::assert_snapshot!("grid_summary", summary.join("\n"));
}

/// Snapshot the Unit/Local cell specifically (highest-frequency cell in tests).
#[test]
fn snapshot_unit_local_cell() {
    let cell = curated()
        .find(TestingScenario::Unit, ExecutionEnvironment::Local)
        .expect("Unit/Local cell must exist");
    let s = format!(
        "required=[{}] optional=[{}] forbidden=[{}] intent={}",
        cell.required_features.labels().join(","),
        cell.optional_features.labels().join(","),
        cell.forbidden_features.labels().join(","),
        cell.intent
    );
    insta::assert_snapshot!("unit_local_cell", s);
}

/// Snapshot the EndToEnd/CI cell (the most complex cell).
#[test]
fn snapshot_e2e_ci_cell() {
    let cell = curated()
        .find(TestingScenario::EndToEnd, ExecutionEnvironment::Ci)
        .expect("EndToEnd/CI cell must exist");
    let s = format!(
        "required=[{}] optional=[{}] forbidden=[{}] intent={}",
        cell.required_features.labels().join(","),
        cell.optional_features.labels().join(","),
        cell.forbidden_features.labels().join(","),
        cell.intent
    );
    insta::assert_snapshot!("e2e_ci_cell", s);
}

/// Snapshot the count of cells to detect accidental additions/removals.
#[test]
fn snapshot_cell_count() {
    let count = curated().rows().len();
    insta::assert_snapshot!("cell_count", count.to_string());
}
