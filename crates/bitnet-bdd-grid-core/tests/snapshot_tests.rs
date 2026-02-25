//! Snapshot tests for `bitnet-bdd-grid-core`.
//!
//! These tests pin the string representation of core BDD types
//! (`TestingScenario`, `ExecutionEnvironment`, `BitnetFeature`)
//! and the `BddCell` / `FeatureSet` debug output.

use bitnet_bdd_grid_core::{
    BddCell, BddGrid, BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario,
};
use insta::assert_snapshot;

// -- TestingScenario Display -------------------------------------------------

#[test]
fn scenario_display_strings() {
    let displays: Vec<String> = [
        TestingScenario::Unit,
        TestingScenario::Integration,
        TestingScenario::EndToEnd,
        TestingScenario::Performance,
        TestingScenario::CrossValidation,
        TestingScenario::Smoke,
        TestingScenario::Development,
        TestingScenario::Debug,
        TestingScenario::Minimal,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_snapshot!("scenario_display_strings", format!("{:#?}", displays));
}

// -- ExecutionEnvironment Display --------------------------------------------

#[test]
fn environment_display_strings() {
    let displays: Vec<String> = [
        ExecutionEnvironment::Local,
        ExecutionEnvironment::Ci,
        ExecutionEnvironment::PreProduction,
        ExecutionEnvironment::Production,
    ]
    .iter()
    .map(|e| e.to_string())
    .collect();
    assert_snapshot!("environment_display_strings", format!("{:#?}", displays));
}

// -- BitnetFeature Display ---------------------------------------------------

#[test]
fn feature_display_strings() {
    let features: &[BitnetFeature] = &[
        BitnetFeature::Cpu,
        BitnetFeature::Gpu,
        BitnetFeature::Cuda,
        BitnetFeature::Inference,
        BitnetFeature::Kernels,
        BitnetFeature::Tokenizers,
        BitnetFeature::Quantization,
        BitnetFeature::Cli,
        BitnetFeature::Server,
        BitnetFeature::Ffi,
        BitnetFeature::CrossValidation,
        BitnetFeature::Fixtures,
    ];
    let displays: Vec<String> = features.iter().map(|f: &BitnetFeature| f.to_string()).collect();
    assert_snapshot!("feature_display_strings", format!("{:#?}", displays));
}

// -- FeatureSet labels -------------------------------------------------------

#[test]
fn feature_set_labels_cpu_inference() {
    let mut fs = FeatureSet::new();
    fs.insert(BitnetFeature::Cpu);
    fs.insert(BitnetFeature::Inference);
    fs.insert(BitnetFeature::Kernels);
    let mut labels = fs.labels();
    labels.sort(); // stable order for snapshot
    assert_snapshot!("feature_set_labels_cpu_inference", format!("{:#?}", labels));
}

// -- BddCell debug -----------------------------------------------------------

#[test]
fn bdd_cell_unit_local_debug() {
    let mut required = FeatureSet::new();
    required.insert(BitnetFeature::Cpu);
    let cell = BddCell {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        required_features: required,
        optional_features: FeatureSet::new(),
        forbidden_features: FeatureSet::new(),
        intent: "Unit tests run without model or GPU",
    };
    assert_snapshot!("bdd_cell_unit_local_debug", format!("{:?}", cell));
}

// -- BddGrid validate --------------------------------------------------------

#[test]
fn bdd_grid_validate_missing_cell() {
    // validate() returns None if the cell is not in the grid
    static CELLS: &[BddCell] = &[];
    let grid = BddGrid::from_rows(CELLS);
    let features = FeatureSet::new();
    let result = grid.validate(TestingScenario::Unit, ExecutionEnvironment::Local, &features);
    assert_snapshot!("bdd_grid_validate_missing_cell", format!("{:?}", result));
}
