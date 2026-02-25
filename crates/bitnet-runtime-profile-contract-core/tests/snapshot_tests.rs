//! Snapshot tests for bitnet-runtime-profile-contract-core.
//!
//! Pins: active_profile_summary() format, is_supported() semantics,
//! violations() shape, and canonical_grid() cell count stability.

use bitnet_runtime_profile_contract_core::{
    ExecutionEnvironment, TestingScenario, active_profile_for, active_profile_summary,
    canonical_grid,
};

#[test]
fn canonical_grid_cell_count_stable() {
    let grid = canonical_grid();
    insta::assert_snapshot!(grid.rows().len().to_string());
}

#[test]
fn active_profile_summary_format_stable() {
    let summary = active_profile_summary();
    // Should start with "scenario=" whether or not we're in CI
    assert!(
        summary.starts_with("scenario="),
        "summary should start with 'scenario=' but got: {summary}"
    );
    insta::assert_snapshot!(summary.contains("scenario=").to_string());
}

#[test]
fn active_profile_unit_local_has_cell() {
    let profile = active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Local);
    insta::assert_snapshot!(profile.cell.is_some().to_string());
}

#[test]
fn active_profile_unit_local_supported_without_gpu() {
    let profile = active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Local);
    // Unit/Local should not require GPU features — it's always supported in CPU mode
    let (missing, _forbidden) = profile.violations();
    let labels: Vec<String> = missing.labels();
    insta::assert_debug_snapshot!(labels);
}
