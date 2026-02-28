//! Snapshot tests for `bitnet-feature-matrix` public API surface.
//!
//! Pins the Display format of scenario, environment, and feature enums plus
//! grid structure so that changes are visible in code review.

use bitnet_feature_matrix::{
    BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario, canonical_grid, feature_line,
};

// ---------------------------------------------------------------------------
// TestingScenario
// ---------------------------------------------------------------------------

#[test]
fn testing_scenario_display_all_variants() {
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
    let displays: Vec<String> = scenarios.iter().map(|s| s.to_string()).collect();
    insta::assert_debug_snapshot!("testing_scenario_display_variants", displays);
}

// ---------------------------------------------------------------------------
// ExecutionEnvironment
// ---------------------------------------------------------------------------

#[test]
fn execution_environment_display_all_variants() {
    let envs = [
        ExecutionEnvironment::Local,
        ExecutionEnvironment::Ci,
        ExecutionEnvironment::PreProduction,
        ExecutionEnvironment::Production,
    ];
    let displays: Vec<String> = envs.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!("execution_environment_display_variants", displays);
}

// ---------------------------------------------------------------------------
// BitnetFeature
// ---------------------------------------------------------------------------

#[test]
fn bitnet_feature_display_all_variants() {
    let features = [
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
        BitnetFeature::Python,
        BitnetFeature::Wasm,
        BitnetFeature::CrossValidation,
        BitnetFeature::Trace,
        BitnetFeature::Iq2sFfi,
        BitnetFeature::CppFfi,
        BitnetFeature::Fixtures,
        BitnetFeature::Reporting,
        BitnetFeature::Trend,
        BitnetFeature::IntegrationTests,
    ];
    let displays: Vec<String> = features.iter().map(|f| f.to_string()).collect();
    insta::assert_debug_snapshot!("bitnet_feature_display_variants", displays);
}

// ---------------------------------------------------------------------------
// FeatureSet
// ---------------------------------------------------------------------------

#[test]
fn feature_set_empty_labels() {
    let fs = FeatureSet::new();
    insta::assert_debug_snapshot!("feature_set_empty_labels", fs.labels());
}

#[test]
fn feature_set_cpu_only_labels() {
    let mut fs = FeatureSet::new();
    fs.insert(BitnetFeature::Cpu);
    insta::assert_debug_snapshot!("feature_set_cpu_only_labels", fs.labels());
}

#[test]
fn feature_set_multiple_labels_sorted() {
    let mut fs = FeatureSet::new();
    fs.insert(BitnetFeature::Wasm);
    fs.insert(BitnetFeature::Cpu);
    fs.insert(BitnetFeature::Inference);
    insta::assert_debug_snapshot!("feature_set_multiple_labels_sorted", fs.labels());
}

// ---------------------------------------------------------------------------
// Canonical grid
// ---------------------------------------------------------------------------

#[test]
fn canonical_grid_scenario_row_counts() {
    let grid = canonical_grid();
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
    let counts: Vec<String> =
        scenarios.iter().map(|s| format!("{}={}", s, grid.rows_for_scenario(*s).len())).collect();
    insta::assert_debug_snapshot!("canonical_grid_scenario_row_counts", counts);
}

// ---------------------------------------------------------------------------
// feature_line
// ---------------------------------------------------------------------------

#[test]
fn feature_line_is_non_empty_and_prefixed() {
    let line = feature_line();
    // feature_line content depends on compile-time flags, so we only pin the
    // structural invariant: it starts with "features:" and is non-empty.
    insta::assert_snapshot!(
        "feature_line_invariant",
        format!(
            "starts_with_features={} non_empty={}",
            line.starts_with("features:"),
            !line.is_empty()
        )
    );
}
