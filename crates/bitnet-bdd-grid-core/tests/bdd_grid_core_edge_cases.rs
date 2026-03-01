//! Edge-case tests for bitnet-bdd-grid-core types and grid logic.

use bitnet_bdd_grid_core::{
    BddCell, BddGrid, BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario,
    feature_set_from_names,
};
use std::str::FromStr;

// ---------------------------------------------------------------------------
// TestingScenario: FromStr parsing
// ---------------------------------------------------------------------------

#[test]
fn scenario_unit_parses() {
    assert_eq!(TestingScenario::from_str("unit"), Ok(TestingScenario::Unit));
}

#[test]
fn scenario_integration_parses() {
    assert_eq!(TestingScenario::from_str("integration"), Ok(TestingScenario::Integration));
}

#[test]
fn scenario_e2e_alias_end_to_end() {
    assert_eq!(TestingScenario::from_str("end-to-end"), Ok(TestingScenario::EndToEnd));
}

#[test]
fn scenario_e2e_alias_endtoend() {
    assert_eq!(TestingScenario::from_str("endtoend"), Ok(TestingScenario::EndToEnd));
}

#[test]
fn scenario_perf_alias() {
    assert_eq!(TestingScenario::from_str("perf"), Ok(TestingScenario::Performance));
}

#[test]
fn scenario_crossval_alias() {
    assert_eq!(TestingScenario::from_str("cross-validation"), Ok(TestingScenario::CrossValidation));
}

#[test]
fn scenario_dev_alias() {
    assert_eq!(TestingScenario::from_str("dev"), Ok(TestingScenario::Development));
}

#[test]
fn scenario_min_alias() {
    assert_eq!(TestingScenario::from_str("min"), Ok(TestingScenario::Minimal));
}

#[test]
fn scenario_unknown_errors() {
    assert!(TestingScenario::from_str("foobar").is_err());
}

#[test]
fn scenario_case_insensitive() {
    assert_eq!(TestingScenario::from_str("UNIT"), Ok(TestingScenario::Unit));
    assert_eq!(TestingScenario::from_str("Performance"), Ok(TestingScenario::Performance));
}

#[test]
fn scenario_display_roundtrip() {
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
    for s in &scenarios {
        let displayed = s.to_string();
        let parsed = TestingScenario::from_str(&displayed).unwrap();
        assert_eq!(*s, parsed, "roundtrip failed for {displayed}");
    }
}

#[test]
fn scenario_debug_is_not_empty() {
    let d = format!("{:?}", TestingScenario::Debug);
    assert!(!d.is_empty());
}

#[test]
fn scenario_clone_eq() {
    let s = TestingScenario::Smoke;
    let c = s;
    assert_eq!(s, c);
}

#[test]
fn scenario_ord_consistent() {
    // Unit < Integration < EndToEnd < ... (enum declaration order)
    assert!(TestingScenario::Unit < TestingScenario::Integration);
    assert!(TestingScenario::Integration < TestingScenario::EndToEnd);
}

// ---------------------------------------------------------------------------
// ExecutionEnvironment: FromStr parsing
// ---------------------------------------------------------------------------

#[test]
fn env_local_parses() {
    assert_eq!(ExecutionEnvironment::from_str("local"), Ok(ExecutionEnvironment::Local));
}

#[test]
fn env_dev_alias_maps_to_local() {
    assert_eq!(ExecutionEnvironment::from_str("dev"), Ok(ExecutionEnvironment::Local));
}

#[test]
fn env_development_alias_maps_to_local() {
    assert_eq!(ExecutionEnvironment::from_str("development"), Ok(ExecutionEnvironment::Local));
}

#[test]
fn env_ci_parses() {
    assert_eq!(ExecutionEnvironment::from_str("ci"), Ok(ExecutionEnvironment::Ci));
}

#[test]
fn env_cicd_alias() {
    assert_eq!(ExecutionEnvironment::from_str("ci/cd"), Ok(ExecutionEnvironment::Ci));
}

#[test]
fn env_cicd_no_slash_alias() {
    assert_eq!(ExecutionEnvironment::from_str("cicd"), Ok(ExecutionEnvironment::Ci));
}

#[test]
fn env_preprod_aliases() {
    for alias in &["pre-prod", "preprod", "pre-production", "preproduction", "staging"] {
        assert_eq!(
            ExecutionEnvironment::from_str(alias),
            Ok(ExecutionEnvironment::PreProduction),
            "alias {alias} should map to PreProduction"
        );
    }
}

#[test]
fn env_prod_aliases() {
    assert_eq!(ExecutionEnvironment::from_str("prod"), Ok(ExecutionEnvironment::Production));
    assert_eq!(ExecutionEnvironment::from_str("production"), Ok(ExecutionEnvironment::Production));
}

#[test]
fn env_unknown_errors() {
    assert!(ExecutionEnvironment::from_str("mars").is_err());
}

#[test]
fn env_case_insensitive() {
    assert_eq!(ExecutionEnvironment::from_str("CI"), Ok(ExecutionEnvironment::Ci));
    assert_eq!(ExecutionEnvironment::from_str("LOCAL"), Ok(ExecutionEnvironment::Local));
}

#[test]
fn env_display_roundtrip() {
    let envs = [
        ExecutionEnvironment::Local,
        ExecutionEnvironment::Ci,
        ExecutionEnvironment::PreProduction,
        ExecutionEnvironment::Production,
    ];
    for e in &envs {
        let displayed = e.to_string();
        let parsed = ExecutionEnvironment::from_str(&displayed).unwrap();
        assert_eq!(*e, parsed, "roundtrip failed for {displayed}");
    }
}

#[test]
fn env_ord_consistent() {
    assert!(ExecutionEnvironment::Local < ExecutionEnvironment::Ci);
    assert!(ExecutionEnvironment::Ci < ExecutionEnvironment::PreProduction);
    assert!(ExecutionEnvironment::PreProduction < ExecutionEnvironment::Production);
}

// ---------------------------------------------------------------------------
// BitnetFeature: FromStr parsing
// ---------------------------------------------------------------------------

#[test]
fn feature_cpu_parses() {
    assert_eq!(BitnetFeature::from_str("cpu"), Ok(BitnetFeature::Cpu));
}

#[test]
fn feature_gpu_parses() {
    assert_eq!(BitnetFeature::from_str("gpu"), Ok(BitnetFeature::Gpu));
}

#[test]
fn feature_all_known_parse() {
    let known = [
        ("cpu", BitnetFeature::Cpu),
        ("gpu", BitnetFeature::Gpu),
        ("cuda", BitnetFeature::Cuda),
        ("metal", BitnetFeature::Metal),
        ("vulkan", BitnetFeature::Vulkan),
        ("oneapi", BitnetFeature::Oneapi),
        ("inference", BitnetFeature::Inference),
        ("kernels", BitnetFeature::Kernels),
        ("tokenizers", BitnetFeature::Tokenizers),
        ("quantization", BitnetFeature::Quantization),
        ("cli", BitnetFeature::Cli),
        ("server", BitnetFeature::Server),
        ("ffi", BitnetFeature::Ffi),
        ("python", BitnetFeature::Python),
        ("wasm", BitnetFeature::Wasm),
        ("crossval", BitnetFeature::CrossValidation),
        ("trace", BitnetFeature::Trace),
        ("iq2s-ffi", BitnetFeature::Iq2sFfi),
        ("cpp-ffi", BitnetFeature::CppFfi),
        ("fixtures", BitnetFeature::Fixtures),
        ("reporting", BitnetFeature::Reporting),
        ("trend", BitnetFeature::Trend),
        ("integration-tests", BitnetFeature::IntegrationTests),
    ];
    for (name, expected) in &known {
        assert_eq!(BitnetFeature::from_str(name), Ok(*expected), "failed to parse feature: {name}");
    }
}

#[test]
fn feature_crossval_alias() {
    assert_eq!(BitnetFeature::from_str("cross-validation"), Ok(BitnetFeature::CrossValidation));
}

#[test]
fn feature_unknown_errors() {
    assert!(BitnetFeature::from_str("opengl").is_err());
}

#[test]
fn feature_case_insensitive() {
    assert_eq!(BitnetFeature::from_str("CPU"), Ok(BitnetFeature::Cpu));
    assert_eq!(BitnetFeature::from_str("Gpu"), Ok(BitnetFeature::Gpu));
}

#[test]
fn feature_display_roundtrip() {
    let features = [
        BitnetFeature::Cpu,
        BitnetFeature::Gpu,
        BitnetFeature::Cuda,
        BitnetFeature::Wasm,
        BitnetFeature::CrossValidation,
        BitnetFeature::IntegrationTests,
    ];
    for f in &features {
        let displayed = f.to_string();
        let parsed = BitnetFeature::from_str(&displayed).unwrap();
        assert_eq!(*f, parsed, "roundtrip failed for {displayed}");
    }
}

// ---------------------------------------------------------------------------
// FeatureSet: construction, membership, labels
// ---------------------------------------------------------------------------

#[test]
fn feature_set_new_is_empty() {
    let set = FeatureSet::new();
    assert!(set.is_empty());
}

#[test]
fn feature_set_insert_returns_true_for_new() {
    let mut set = FeatureSet::new();
    assert!(set.insert(BitnetFeature::Cpu));
}

#[test]
fn feature_set_insert_returns_false_for_duplicate() {
    let mut set = FeatureSet::new();
    set.insert(BitnetFeature::Cpu);
    assert!(!set.insert(BitnetFeature::Cpu));
}

#[test]
fn feature_set_contains_after_insert() {
    let mut set = FeatureSet::new();
    set.insert(BitnetFeature::Gpu);
    assert!(set.contains(BitnetFeature::Gpu));
    assert!(!set.contains(BitnetFeature::Cpu));
}

#[test]
fn feature_set_extend_adds_all() {
    let mut set = FeatureSet::new();
    set.extend([BitnetFeature::Cpu, BitnetFeature::Gpu, BitnetFeature::Inference]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Gpu));
    assert!(set.contains(BitnetFeature::Inference));
}

#[test]
fn feature_set_from_names_skips_unknown() {
    let set = FeatureSet::from_names(["cpu", "unknown-thing", "gpu"]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Gpu));
    assert!(!set.contains(BitnetFeature::Cuda));
}

#[test]
fn feature_set_from_names_empty_input() {
    let set = FeatureSet::from_names(Vec::<&str>::new());
    assert!(set.is_empty());
}

#[test]
fn feature_set_labels_sorted() {
    let mut set = FeatureSet::new();
    set.insert(BitnetFeature::Wasm);
    set.insert(BitnetFeature::Cpu);
    set.insert(BitnetFeature::Gpu);
    let labels = set.labels();
    // BTreeSet ordering = enum declaration order (Cpu < Gpu < Wasm)
    assert_eq!(labels, vec!["cpu", "gpu", "wasm"]);
}

#[test]
fn feature_set_missing_required() {
    let active = FeatureSet::from_names(["cpu", "inference"]);
    let required = FeatureSet::from_names(["cpu", "inference", "tokenizers"]);
    let missing = active.missing_required(&required);
    assert!(missing.contains(BitnetFeature::Tokenizers));
    assert!(!missing.contains(BitnetFeature::Cpu));
}

#[test]
fn feature_set_missing_required_when_satisfied() {
    let active = FeatureSet::from_names(["cpu", "inference", "tokenizers"]);
    let required = FeatureSet::from_names(["cpu", "inference"]);
    let missing = active.missing_required(&required);
    assert!(missing.is_empty());
}

#[test]
fn feature_set_forbidden_overlap() {
    let active = FeatureSet::from_names(["cpu", "cuda", "inference"]);
    let forbidden = FeatureSet::from_names(["cuda", "metal"]);
    let overlap = active.forbidden_overlap(&forbidden);
    assert!(overlap.contains(BitnetFeature::Cuda));
    assert!(!overlap.contains(BitnetFeature::Metal));
}

#[test]
fn feature_set_forbidden_overlap_empty() {
    let active = FeatureSet::from_names(["cpu", "inference"]);
    let forbidden = FeatureSet::from_names(["cuda", "metal"]);
    let overlap = active.forbidden_overlap(&forbidden);
    assert!(overlap.is_empty());
}

#[test]
fn feature_set_satisfies_both_constraints() {
    let active = FeatureSet::from_names(["cpu", "inference", "tokenizers"]);
    let required = FeatureSet::from_names(["cpu", "inference"]);
    let forbidden = FeatureSet::from_names(["cuda"]);
    assert!(active.satisfies(&required, &forbidden));
}

#[test]
fn feature_set_satisfies_fails_on_missing() {
    let active = FeatureSet::from_names(["cpu"]);
    let required = FeatureSet::from_names(["cpu", "inference"]);
    let forbidden = FeatureSet::new();
    assert!(!active.satisfies(&required, &forbidden));
}

#[test]
fn feature_set_satisfies_fails_on_forbidden() {
    let active = FeatureSet::from_names(["cpu", "cuda"]);
    let required = FeatureSet::from_names(["cpu"]);
    let forbidden = FeatureSet::from_names(["cuda"]);
    assert!(!active.satisfies(&required, &forbidden));
}

#[test]
fn feature_set_iter_count() {
    let mut set = FeatureSet::new();
    set.insert(BitnetFeature::Cpu);
    set.insert(BitnetFeature::Gpu);
    assert_eq!(set.iter().count(), 2);
}

#[test]
fn feature_set_from_slice_of_features() {
    let set = FeatureSet::from(&[BitnetFeature::Cpu, BitnetFeature::Gpu][..]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Gpu));
}

#[test]
fn feature_set_from_slice_of_strs() {
    let set = FeatureSet::from(&["cpu", "inference"][..]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Inference));
}

#[test]
fn feature_set_default_is_empty() {
    let set = FeatureSet::default();
    assert!(set.is_empty());
}

#[test]
fn feature_set_eq() {
    let a = FeatureSet::from_names(["cpu", "gpu"]);
    let b = FeatureSet::from_names(["gpu", "cpu"]);
    assert_eq!(a, b);
}

#[test]
fn feature_set_ne() {
    let a = FeatureSet::from_names(["cpu"]);
    let b = FeatureSet::from_names(["gpu"]);
    assert_ne!(a, b);
}

// ---------------------------------------------------------------------------
// feature_set_from_names helper function
// ---------------------------------------------------------------------------

#[test]
fn helper_feature_set_from_names_works() {
    let set = feature_set_from_names(&["cpu", "inference"]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert!(set.contains(BitnetFeature::Inference));
}

#[test]
fn helper_feature_set_from_names_skips_unknown() {
    let set = feature_set_from_names(&["cpu", "not-a-feature"]);
    assert!(set.contains(BitnetFeature::Cpu));
    assert_eq!(set.labels().len(), 1);
}

#[test]
fn helper_feature_set_from_names_empty() {
    let set = feature_set_from_names(&[]);
    assert!(set.is_empty());
}

// ---------------------------------------------------------------------------
// BddCell: supports and violations
// ---------------------------------------------------------------------------

fn make_cell(
    scenario: TestingScenario,
    env: ExecutionEnvironment,
    required: &[&str],
    forbidden: &[&str],
    intent: &'static str,
) -> BddCell {
    BddCell {
        scenario,
        environment: env,
        required_features: feature_set_from_names(required),
        optional_features: FeatureSet::new(),
        forbidden_features: feature_set_from_names(forbidden),
        intent,
    }
}

#[test]
fn bdd_cell_supports_when_all_required_present() {
    let cell = make_cell(
        TestingScenario::Unit,
        ExecutionEnvironment::Local,
        &["cpu", "inference"],
        &[],
        "unit local",
    );
    let active = feature_set_from_names(&["cpu", "inference", "tokenizers"]);
    assert!(cell.supports(&active));
}

#[test]
fn bdd_cell_rejects_when_missing_required() {
    let cell = make_cell(
        TestingScenario::Unit,
        ExecutionEnvironment::Local,
        &["cpu", "inference"],
        &[],
        "unit local",
    );
    let active = feature_set_from_names(&["cpu"]);
    assert!(!cell.supports(&active));
}

#[test]
fn bdd_cell_rejects_when_forbidden_present() {
    let cell =
        make_cell(TestingScenario::Unit, ExecutionEnvironment::Ci, &["cpu"], &["cuda"], "unit ci");
    let active = feature_set_from_names(&["cpu", "cuda"]);
    assert!(!cell.supports(&active));
}

#[test]
fn bdd_cell_violations_returns_missing_and_forbidden() {
    let cell = make_cell(
        TestingScenario::Integration,
        ExecutionEnvironment::Local,
        &["cpu", "inference", "tokenizers"],
        &["cuda"],
        "int local",
    );
    let active = feature_set_from_names(&["cpu", "cuda"]);
    let (missing, forbidden) = cell.violations(&active);
    assert!(missing.contains(BitnetFeature::Inference));
    assert!(missing.contains(BitnetFeature::Tokenizers));
    assert!(forbidden.contains(BitnetFeature::Cuda));
}

#[test]
fn bdd_cell_violations_empty_when_satisfied() {
    let cell =
        make_cell(TestingScenario::Smoke, ExecutionEnvironment::Ci, &["cpu"], &[], "smoke ci");
    let active = feature_set_from_names(&["cpu"]);
    let (missing, forbidden) = cell.violations(&active);
    assert!(missing.is_empty());
    assert!(forbidden.is_empty());
}

// ---------------------------------------------------------------------------
// BddGrid: find, rows_for_scenario, validate
// ---------------------------------------------------------------------------

fn leak_cells(cells: Vec<BddCell>) -> &'static [BddCell] {
    Box::leak(cells.into_boxed_slice())
}

#[test]
fn bdd_grid_find_existing_cell() {
    let cells = leak_cells(vec![
        make_cell(TestingScenario::Unit, ExecutionEnvironment::Local, &["cpu"], &[], "a"),
        make_cell(TestingScenario::Unit, ExecutionEnvironment::Ci, &["cpu"], &[], "b"),
    ]);
    let grid = BddGrid::from_rows(cells);
    let found = grid.find(TestingScenario::Unit, ExecutionEnvironment::Ci);
    assert!(found.is_some());
    assert_eq!(found.unwrap().intent, "b");
}

#[test]
fn bdd_grid_find_missing_cell() {
    let cells = leak_cells(vec![make_cell(
        TestingScenario::Unit,
        ExecutionEnvironment::Local,
        &["cpu"],
        &[],
        "a",
    )]);
    let grid = BddGrid::from_rows(cells);
    assert!(grid.find(TestingScenario::Integration, ExecutionEnvironment::Local).is_none());
}

#[test]
fn bdd_grid_rows_for_scenario() {
    let cells = leak_cells(vec![
        make_cell(TestingScenario::Unit, ExecutionEnvironment::Local, &["cpu"], &[], "a"),
        make_cell(TestingScenario::Integration, ExecutionEnvironment::Ci, &["cpu"], &[], "b"),
        make_cell(TestingScenario::Unit, ExecutionEnvironment::Ci, &["cpu"], &[], "c"),
    ]);
    let grid = BddGrid::from_rows(cells);
    let unit_rows = grid.rows_for_scenario(TestingScenario::Unit);
    assert_eq!(unit_rows.len(), 2);
}

#[test]
fn bdd_grid_rows_for_scenario_empty() {
    let cells = leak_cells(vec![make_cell(
        TestingScenario::Unit,
        ExecutionEnvironment::Local,
        &["cpu"],
        &[],
        "a",
    )]);
    let grid = BddGrid::from_rows(cells);
    assert!(grid.rows_for_scenario(TestingScenario::Performance).is_empty());
}

#[test]
fn bdd_grid_validate_returns_violations() {
    let cells = leak_cells(vec![make_cell(
        TestingScenario::Unit,
        ExecutionEnvironment::Local,
        &["cpu", "inference"],
        &[],
        "a",
    )]);
    let grid = BddGrid::from_rows(cells);
    let active = feature_set_from_names(&["cpu"]);
    let result = grid.validate(TestingScenario::Unit, ExecutionEnvironment::Local, &active);
    assert!(result.is_some());
    let (missing, _forbidden) = result.unwrap();
    assert!(missing.contains(BitnetFeature::Inference));
}

#[test]
fn bdd_grid_validate_returns_none_for_missing_cell() {
    let cells = leak_cells(vec![make_cell(
        TestingScenario::Unit,
        ExecutionEnvironment::Local,
        &["cpu"],
        &[],
        "a",
    )]);
    let grid = BddGrid::from_rows(cells);
    let active = feature_set_from_names(&["cpu"]);
    assert!(
        grid.validate(TestingScenario::Performance, ExecutionEnvironment::Ci, &active).is_none()
    );
}

#[test]
fn bdd_grid_rows_returns_all() {
    let cells = leak_cells(vec![
        make_cell(TestingScenario::Unit, ExecutionEnvironment::Local, &["cpu"], &[], "a"),
        make_cell(TestingScenario::Smoke, ExecutionEnvironment::Ci, &["cpu"], &[], "b"),
    ]);
    let grid = BddGrid::from_rows(cells);
    assert_eq!(grid.rows().len(), 2);
}

#[test]
fn bdd_grid_empty_rows() {
    let cells: &'static [BddCell] = leak_cells(vec![]);
    let grid = BddGrid::from_rows(cells);
    assert!(grid.rows().is_empty());
    assert!(grid.find(TestingScenario::Unit, ExecutionEnvironment::Local).is_none());
}
