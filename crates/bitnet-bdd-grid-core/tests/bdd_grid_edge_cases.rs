//! Edge-case tests for bitnet-bdd-grid-core: enums (TestingScenario,
//! ExecutionEnvironment, BitnetFeature), FeatureSet, BddCell, BddGrid.

use bitnet_bdd_grid_core::{
    BddCell, BddGrid, BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario,
    feature_set_from_names,
};
use std::str::FromStr;

// ---------------------------------------------------------------------------
// TestingScenario — Display + FromStr
// ---------------------------------------------------------------------------

#[test]
fn scenario_display_all() {
    assert_eq!(TestingScenario::Unit.to_string(), "unit");
    assert_eq!(TestingScenario::Integration.to_string(), "integration");
    assert_eq!(TestingScenario::EndToEnd.to_string(), "e2e");
    assert_eq!(TestingScenario::Performance.to_string(), "performance");
    assert_eq!(TestingScenario::CrossValidation.to_string(), "crossval");
    assert_eq!(TestingScenario::Smoke.to_string(), "smoke");
    assert_eq!(TestingScenario::Development.to_string(), "development");
    assert_eq!(TestingScenario::Debug.to_string(), "debug");
    assert_eq!(TestingScenario::Minimal.to_string(), "minimal");
}

#[test]
fn scenario_from_str_canonical() {
    assert_eq!(TestingScenario::from_str("unit").unwrap(), TestingScenario::Unit);
    assert_eq!(TestingScenario::from_str("integration").unwrap(), TestingScenario::Integration);
    assert_eq!(TestingScenario::from_str("smoke").unwrap(), TestingScenario::Smoke);
}

#[test]
fn scenario_from_str_aliases() {
    assert_eq!(TestingScenario::from_str("e2e").unwrap(), TestingScenario::EndToEnd);
    assert_eq!(TestingScenario::from_str("end-to-end").unwrap(), TestingScenario::EndToEnd);
    assert_eq!(TestingScenario::from_str("endtoend").unwrap(), TestingScenario::EndToEnd);
    assert_eq!(TestingScenario::from_str("perf").unwrap(), TestingScenario::Performance);
    assert_eq!(
        TestingScenario::from_str("cross-validation").unwrap(),
        TestingScenario::CrossValidation
    );
    assert_eq!(TestingScenario::from_str("dev").unwrap(), TestingScenario::Development);
    assert_eq!(TestingScenario::from_str("min").unwrap(), TestingScenario::Minimal);
}

#[test]
fn scenario_from_str_case_insensitive() {
    assert_eq!(TestingScenario::from_str("UNIT").unwrap(), TestingScenario::Unit);
    assert_eq!(TestingScenario::from_str("Smoke").unwrap(), TestingScenario::Smoke);
}

#[test]
fn scenario_from_str_unknown() {
    assert!(TestingScenario::from_str("unknown").is_err());
    assert!(TestingScenario::from_str("").is_err());
}

#[test]
fn scenario_ordering() {
    assert!(TestingScenario::Unit < TestingScenario::Integration);
    assert!(TestingScenario::Smoke < TestingScenario::Minimal);
}

// ---------------------------------------------------------------------------
// ExecutionEnvironment — Display + FromStr
// ---------------------------------------------------------------------------

#[test]
fn environment_display_all() {
    assert_eq!(ExecutionEnvironment::Local.to_string(), "local");
    assert_eq!(ExecutionEnvironment::Ci.to_string(), "ci");
    assert_eq!(ExecutionEnvironment::PreProduction.to_string(), "pre-prod");
    assert_eq!(ExecutionEnvironment::Production.to_string(), "production");
}

#[test]
fn environment_from_str_canonical() {
    assert_eq!(ExecutionEnvironment::from_str("local").unwrap(), ExecutionEnvironment::Local);
    assert_eq!(ExecutionEnvironment::from_str("ci").unwrap(), ExecutionEnvironment::Ci);
    assert_eq!(
        ExecutionEnvironment::from_str("production").unwrap(),
        ExecutionEnvironment::Production
    );
}

#[test]
fn environment_from_str_aliases() {
    assert_eq!(ExecutionEnvironment::from_str("dev").unwrap(), ExecutionEnvironment::Local);
    assert_eq!(ExecutionEnvironment::from_str("development").unwrap(), ExecutionEnvironment::Local);
    assert_eq!(ExecutionEnvironment::from_str("ci/cd").unwrap(), ExecutionEnvironment::Ci);
    assert_eq!(ExecutionEnvironment::from_str("cicd").unwrap(), ExecutionEnvironment::Ci);
    assert_eq!(
        ExecutionEnvironment::from_str("pre-prod").unwrap(),
        ExecutionEnvironment::PreProduction
    );
    assert_eq!(
        ExecutionEnvironment::from_str("preprod").unwrap(),
        ExecutionEnvironment::PreProduction
    );
    assert_eq!(
        ExecutionEnvironment::from_str("staging").unwrap(),
        ExecutionEnvironment::PreProduction
    );
    assert_eq!(ExecutionEnvironment::from_str("prod").unwrap(), ExecutionEnvironment::Production);
}

#[test]
fn environment_from_str_case_insensitive() {
    assert_eq!(ExecutionEnvironment::from_str("CI").unwrap(), ExecutionEnvironment::Ci);
    assert_eq!(
        ExecutionEnvironment::from_str("PRODUCTION").unwrap(),
        ExecutionEnvironment::Production
    );
}

#[test]
fn environment_from_str_unknown() {
    assert!(ExecutionEnvironment::from_str("").is_err());
    assert!(ExecutionEnvironment::from_str("test").is_err());
}

// ---------------------------------------------------------------------------
// BitnetFeature — Display + FromStr
// ---------------------------------------------------------------------------

#[test]
fn feature_display_sample() {
    assert_eq!(BitnetFeature::Cpu.to_string(), "cpu");
    assert_eq!(BitnetFeature::Gpu.to_string(), "gpu");
    assert_eq!(BitnetFeature::Cuda.to_string(), "cuda");
    assert_eq!(BitnetFeature::Oneapi.to_string(), "oneapi");
    assert_eq!(BitnetFeature::CrossValidation.to_string(), "crossval");
    assert_eq!(BitnetFeature::Iq2sFfi.to_string(), "iq2s-ffi");
    assert_eq!(BitnetFeature::IntegrationTests.to_string(), "integration-tests");
}

#[test]
fn feature_from_str_sample() {
    assert_eq!(BitnetFeature::from_str("cpu").unwrap(), BitnetFeature::Cpu);
    assert_eq!(BitnetFeature::from_str("gpu").unwrap(), BitnetFeature::Gpu);
    assert_eq!(BitnetFeature::from_str("crossval").unwrap(), BitnetFeature::CrossValidation);
    assert_eq!(
        BitnetFeature::from_str("cross-validation").unwrap(),
        BitnetFeature::CrossValidation
    );
    assert_eq!(BitnetFeature::from_str("iq2s-ffi").unwrap(), BitnetFeature::Iq2sFfi);
}

#[test]
fn feature_from_str_unknown() {
    assert!(BitnetFeature::from_str("bogus").is_err());
    assert!(BitnetFeature::from_str("").is_err());
}

// ---------------------------------------------------------------------------
// FeatureSet
// ---------------------------------------------------------------------------

#[test]
fn feature_set_new_is_empty() {
    let set = FeatureSet::new();
    assert!(set.is_empty());
}

#[test]
fn feature_set_default_is_empty() {
    let set = FeatureSet::default();
    assert!(set.is_empty());
}

#[test]
fn feature_set_insert_and_contains() {
    let mut set = FeatureSet::new();
    assert!(!set.contains(BitnetFeature::Cpu));
    set.insert(BitnetFeature::Cpu);
    assert!(set.contains(BitnetFeature::Cpu));
}

#[test]
fn feature_set_insert_duplicate() {
    let mut set = FeatureSet::new();
    assert!(set.insert(BitnetFeature::Gpu)); // first insert → true
    assert!(!set.insert(BitnetFeature::Gpu)); // duplicate → false
}

#[test]
fn feature_set_extend() {
    let mut set = FeatureSet::new();
    set.extend([BitnetFeature::Cpu, BitnetFeature::Gpu, BitnetFeature::Inference]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Gpu));
    assert!(set.contains(BitnetFeature::Inference));
}

#[test]
fn feature_set_from_names_valid() {
    let set = FeatureSet::from_names(["cpu", "gpu", "inference"]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Gpu));
    assert!(set.contains(BitnetFeature::Inference));
}

#[test]
fn feature_set_from_names_invalid_ignored() {
    let set = FeatureSet::from_names(["cpu", "bogus", "gpu"]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Gpu));
    // "bogus" silently ignored
    assert_eq!(set.labels().len(), 2);
}

#[test]
fn feature_set_from_names_empty() {
    let set = FeatureSet::from_names(std::iter::empty::<&str>());
    assert!(set.is_empty());
}

#[test]
fn feature_set_labels_sorted() {
    let mut set = FeatureSet::new();
    set.extend([BitnetFeature::Gpu, BitnetFeature::Cpu]);
    let labels = set.labels();
    // BTreeSet → sorted order: cpu < gpu
    assert_eq!(labels, vec!["cpu", "gpu"]);
}

#[test]
fn feature_set_missing_required() {
    let active = FeatureSet::from_names(["cpu"]);
    let required = FeatureSet::from_names(["cpu", "inference"]);
    let missing = active.missing_required(&required);
    assert!(missing.contains(BitnetFeature::Inference));
    assert!(!missing.contains(BitnetFeature::Cpu));
}

#[test]
fn feature_set_forbidden_overlap() {
    let active = FeatureSet::from_names(["cpu", "gpu"]);
    let forbidden = FeatureSet::from_names(["gpu", "cuda"]);
    let overlap = active.forbidden_overlap(&forbidden);
    assert!(overlap.contains(BitnetFeature::Gpu));
    assert!(!overlap.contains(BitnetFeature::Cuda));
}

#[test]
fn feature_set_satisfies_ok() {
    let active = FeatureSet::from_names(["cpu", "inference"]);
    let required = FeatureSet::from_names(["cpu"]);
    let forbidden = FeatureSet::from_names(["wasm"]);
    assert!(active.satisfies(&required, &forbidden));
}

#[test]
fn feature_set_satisfies_missing() {
    let active = FeatureSet::from_names(["cpu"]);
    let required = FeatureSet::from_names(["cpu", "inference"]);
    let forbidden = FeatureSet::new();
    assert!(!active.satisfies(&required, &forbidden));
}

#[test]
fn feature_set_satisfies_forbidden() {
    let active = FeatureSet::from_names(["cpu", "wasm"]);
    let required = FeatureSet::from_names(["cpu"]);
    let forbidden = FeatureSet::from_names(["wasm"]);
    assert!(!active.satisfies(&required, &forbidden));
}

#[test]
fn feature_set_from_slice_of_features() {
    let set = FeatureSet::from(&[BitnetFeature::Cpu, BitnetFeature::Gpu][..]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Gpu));
}

#[test]
fn feature_set_from_slice_of_strs() {
    let set = FeatureSet::from(&["cpu", "gpu"][..]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Gpu));
}

#[test]
fn feature_set_eq() {
    let a = FeatureSet::from_names(["cpu", "gpu"]);
    let b = FeatureSet::from_names(["gpu", "cpu"]); // order doesn't matter
    assert_eq!(a, b);
}

#[test]
fn feature_set_from_names_helper() {
    let set = feature_set_from_names(&["cpu", "inference"]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Inference));
}

// ---------------------------------------------------------------------------
// BddCell — supports + violations
// ---------------------------------------------------------------------------

fn test_cell() -> BddCell {
    BddCell {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Ci,
        required_features: feature_set_from_names(&["cpu", "inference"]),
        optional_features: feature_set_from_names(&["trace"]),
        forbidden_features: feature_set_from_names(&["wasm"]),
        intent: "Core unit tests in CI",
    }
}

#[test]
fn cell_supports_valid() {
    let cell = test_cell();
    let active = FeatureSet::from_names(["cpu", "inference", "kernels"]);
    assert!(cell.supports(&active));
}

#[test]
fn cell_supports_missing_required() {
    let cell = test_cell();
    let active = FeatureSet::from_names(["cpu"]); // missing inference
    assert!(!cell.supports(&active));
}

#[test]
fn cell_supports_has_forbidden() {
    let cell = test_cell();
    let active = FeatureSet::from_names(["cpu", "inference", "wasm"]);
    assert!(!cell.supports(&active));
}

#[test]
fn cell_violations_clean() {
    let cell = test_cell();
    let active = FeatureSet::from_names(["cpu", "inference"]);
    let (missing, forbidden) = cell.violations(&active);
    assert!(missing.is_empty());
    assert!(forbidden.is_empty());
}

#[test]
fn cell_violations_missing() {
    let cell = test_cell();
    let active = FeatureSet::from_names(["cpu"]);
    let (missing, forbidden) = cell.violations(&active);
    assert!(missing.contains(BitnetFeature::Inference));
    assert!(forbidden.is_empty());
}

#[test]
fn cell_violations_forbidden() {
    let cell = test_cell();
    let active = FeatureSet::from_names(["cpu", "inference", "wasm"]);
    let (missing, forbidden) = cell.violations(&active);
    assert!(missing.is_empty());
    assert!(forbidden.contains(BitnetFeature::Wasm));
}

// ---------------------------------------------------------------------------
// BddGrid — from_rows, find, rows_for_scenario, validate
// ---------------------------------------------------------------------------

// Helper to leak cells into static for grid construction
fn make_grid() -> BddGrid {
    let cells: &'static [BddCell] = Box::leak(Box::new([
        BddCell {
            scenario: TestingScenario::Unit,
            environment: ExecutionEnvironment::Local,
            required_features: feature_set_from_names(&["cpu"]),
            optional_features: FeatureSet::new(),
            forbidden_features: FeatureSet::new(),
            intent: "Local unit",
        },
        BddCell {
            scenario: TestingScenario::Unit,
            environment: ExecutionEnvironment::Ci,
            required_features: feature_set_from_names(&["cpu", "inference"]),
            optional_features: FeatureSet::new(),
            forbidden_features: feature_set_from_names(&["wasm"]),
            intent: "CI unit",
        },
        BddCell {
            scenario: TestingScenario::Integration,
            environment: ExecutionEnvironment::Ci,
            required_features: feature_set_from_names(&["cpu", "inference", "kernels"]),
            optional_features: FeatureSet::new(),
            forbidden_features: FeatureSet::new(),
            intent: "CI integration",
        },
    ]));
    BddGrid::from_rows(cells)
}

#[test]
fn grid_rows_count() {
    let grid = make_grid();
    assert_eq!(grid.rows().len(), 3);
}

#[test]
fn grid_find_existing() {
    let grid = make_grid();
    let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local);
    assert!(cell.is_some());
    assert_eq!(cell.unwrap().intent, "Local unit");
}

#[test]
fn grid_find_missing() {
    let grid = make_grid();
    let cell = grid.find(TestingScenario::Performance, ExecutionEnvironment::Production);
    assert!(cell.is_none());
}

#[test]
fn grid_rows_for_scenario() {
    let grid = make_grid();
    let unit_rows = grid.rows_for_scenario(TestingScenario::Unit);
    assert_eq!(unit_rows.len(), 2);
    let integration_rows = grid.rows_for_scenario(TestingScenario::Integration);
    assert_eq!(integration_rows.len(), 1);
    let perf_rows = grid.rows_for_scenario(TestingScenario::Performance);
    assert!(perf_rows.is_empty());
}

#[test]
fn grid_validate_pass() {
    let grid = make_grid();
    let features = FeatureSet::from_names(["cpu", "inference"]);
    let result = grid.validate(TestingScenario::Unit, ExecutionEnvironment::Ci, &features);
    assert!(result.is_some());
    let (missing, forbidden) = result.unwrap();
    assert!(missing.is_empty());
    assert!(forbidden.is_empty());
}

#[test]
fn grid_validate_fail() {
    let grid = make_grid();
    let features = FeatureSet::from_names(["cpu"]);
    let result = grid.validate(TestingScenario::Unit, ExecutionEnvironment::Ci, &features);
    let (missing, _) = result.unwrap();
    assert!(missing.contains(BitnetFeature::Inference));
}

#[test]
fn grid_validate_missing_cell() {
    let grid = make_grid();
    let features = FeatureSet::from_names(["cpu"]);
    let result = grid.validate(TestingScenario::Smoke, ExecutionEnvironment::Local, &features);
    assert!(result.is_none());
}
