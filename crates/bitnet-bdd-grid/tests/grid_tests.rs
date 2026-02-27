//! Comprehensive integration tests for `bitnet-bdd-grid`.
//!
//! These tests verify curated grid structure, per-cell invariants, FeatureSet
//! semantics, scenario/environment parsing, and the BddGrid query API.

use bitnet_bdd_grid::{
    BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario, curated,
    feature_set_from_names,
};

// ── 1. Structural invariants ─────────────────────────────────────────────────

#[test]
fn curated_grid_is_non_empty() {
    assert!(!curated().rows().is_empty(), "curated() must return a non-empty grid");
}

#[test]
fn all_cells_have_at_least_one_required_feature() {
    for cell in curated().rows() {
        assert!(
            !cell.required_features.is_empty(),
            "Cell {:?}/{:?} has no required features — every cell must require at least one",
            cell.scenario,
            cell.environment,
        );
    }
}

#[test]
fn all_required_feature_labels_are_parseable() {
    // Every string returned by required_features.labels() must round-trip through
    // BitnetFeature::from_str, proving the canonical feature name list is self-consistent.
    for cell in curated().rows() {
        for label in cell.required_features.labels() {
            assert!(
                label.parse::<BitnetFeature>().is_ok(),
                "Label '{}' in cell {:?}/{:?} is not a valid BitnetFeature name",
                label,
                cell.scenario,
                cell.environment,
            );
        }
    }
}

#[test]
fn all_optional_feature_labels_are_parseable() {
    for cell in curated().rows() {
        for label in cell.optional_features.labels() {
            assert!(
                label.parse::<BitnetFeature>().is_ok(),
                "Optional label '{}' in cell {:?}/{:?} is not a valid BitnetFeature name",
                label,
                cell.scenario,
                cell.environment,
            );
        }
    }
}

#[test]
fn all_forbidden_feature_labels_are_parseable() {
    for cell in curated().rows() {
        for label in cell.forbidden_features.labels() {
            assert!(
                label.parse::<BitnetFeature>().is_ok(),
                "Forbidden label '{}' in cell {:?}/{:?} is not a valid BitnetFeature name",
                label,
                cell.scenario,
                cell.environment,
            );
        }
    }
}

// ── 2. Specific cell lookups ─────────────────────────────────────────────────

#[test]
fn cell_lookup_crossval_ci_exists() {
    let grid = curated();
    let cell = grid.find(TestingScenario::CrossValidation, ExecutionEnvironment::Ci);
    assert!(cell.is_some(), "CrossValidation/Ci cell must be in the curated grid");
}

#[test]
fn cell_lookup_endtoend_ci_exists() {
    let grid = curated();
    let cell = grid.find(TestingScenario::EndToEnd, ExecutionEnvironment::Ci);
    assert!(cell.is_some(), "EndToEnd/Ci cell must be in the curated grid");
}

#[test]
fn cell_lookup_debug_local_exists() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Debug, ExecutionEnvironment::Local);
    assert!(cell.is_some(), "Debug/Local cell must be in the curated grid");
}

#[test]
fn cell_lookup_minimal_local_exists() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Minimal, ExecutionEnvironment::Local);
    assert!(cell.is_some(), "Minimal/Local cell must be in the curated grid");
}

#[test]
fn cell_lookup_development_ci_exists() {
    let grid = curated();
    let cell = grid.find(TestingScenario::Development, ExecutionEnvironment::Ci);
    assert!(cell.is_some(), "Development/Ci cell must be in the curated grid");
}

#[test]
fn cell_lookup_nonexistent_returns_none() {
    // Production environment is not used in the curated grid.
    let grid = curated();
    let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Production);
    assert!(cell.is_none(), "Unit/Production cell should NOT exist in the curated grid");
}

// ── 3. Required-feature spot-checks ─────────────────────────────────────────

#[test]
fn unit_ci_cell_requires_cpu_feature() {
    let grid = curated();
    let cell =
        grid.find(TestingScenario::Unit, ExecutionEnvironment::Ci).expect("Unit/Ci must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::Cpu),
        "Unit/Ci cell must require the cpu feature for deterministic kernel tests"
    );
}

#[test]
fn integration_ci_cell_requires_quantization_feature() {
    let grid = curated();
    let cell = grid
        .find(TestingScenario::Integration, ExecutionEnvironment::Ci)
        .expect("Integration/Ci must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::Quantization),
        "Integration/Ci cell must require quantization for I2_S/QK256/TL1 coverage"
    );
}

#[test]
fn crossval_ci_cell_requires_crossval_feature() {
    let grid = curated();
    let cell = grid
        .find(TestingScenario::CrossValidation, ExecutionEnvironment::Ci)
        .expect("CrossValidation/Ci must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::CrossValidation),
        "CrossValidation/Ci cell must require the crossval feature"
    );
}

#[test]
fn endtoend_preprod_cell_requires_server_feature() {
    let grid = curated();
    let cell = grid
        .find(TestingScenario::EndToEnd, ExecutionEnvironment::PreProduction)
        .expect("EndToEnd/PreProduction must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::Server),
        "EndToEnd/PreProduction cell must require the server feature for health-check coverage"
    );
}

#[test]
fn development_ci_cell_requires_cli_feature() {
    let grid = curated();
    let cell = grid
        .find(TestingScenario::Development, ExecutionEnvironment::Ci)
        .expect("Development/Ci must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::Cli),
        "Development/Ci cell must require cli for prompt-template auto-detection tests"
    );
}

// ── 4. Forbidden-feature checks ───────────────────────────────────────────────

#[test]
fn unit_local_cell_forbids_cpp_ffi() {
    let grid = curated();
    let cell = grid
        .find(TestingScenario::Unit, ExecutionEnvironment::Local)
        .expect("Unit/Local must exist");
    assert!(
        cell.forbidden_features.contains(BitnetFeature::CppFfi),
        "Unit/Local must forbid cpp-ffi to keep fast-path isolation"
    );
}

#[test]
fn cell_with_forbidden_feature_not_supported_when_active() {
    let grid = curated();
    let cell = grid
        .find(TestingScenario::Unit, ExecutionEnvironment::Local)
        .expect("Unit/Local must exist");

    // Build a feature set that satisfies all required features but also activates
    // the forbidden cpp-ffi feature.
    let mut active = cell.required_features.clone();
    active.insert(BitnetFeature::CppFfi);
    assert!(
        !cell.supports(&active),
        "Cell must not support a feature set that includes a forbidden feature"
    );
}

#[test]
fn violations_detects_forbidden_feature_overlap() {
    let grid = curated();
    let cell = grid
        .find(TestingScenario::Unit, ExecutionEnvironment::Local)
        .expect("Unit/Local must exist");

    let mut active = cell.required_features.clone();
    active.insert(BitnetFeature::CppFfi); // forbidden

    let (missing, violated) = cell.violations(&active);
    assert!(missing.is_empty(), "No required features should be missing");
    assert!(
        !violated.is_empty(),
        "violations() must report the forbidden feature in the second tuple element"
    );
}

// ── 5. violations() / supports() semantics ───────────────────────────────────

#[test]
fn supports_true_when_all_required_present_and_no_forbidden() {
    let grid = curated();
    let cell = grid
        .find(TestingScenario::CrossValidation, ExecutionEnvironment::Ci)
        .expect("CrossValidation/Ci must exist");

    let active = cell.required_features.clone();
    assert!(cell.supports(&active), "Cell must support a feature set satisfying all required");
}

#[test]
fn violations_detects_missing_required_features() {
    let grid = curated();
    // EndToEnd/Ci requires multiple features; start with an empty active set.
    let cell = grid
        .find(TestingScenario::EndToEnd, ExecutionEnvironment::Ci)
        .expect("EndToEnd/Ci must exist");

    let active = FeatureSet::new();
    let (missing, violated) = cell.violations(&active);
    assert!(
        !missing.is_empty(),
        "EndToEnd/Ci must report missing required features for an empty active set"
    );
    assert!(violated.is_empty(), "An empty active set cannot trigger forbidden violations");
}

#[test]
fn validate_api_returns_violations_for_known_cell() {
    let grid = curated();
    let result =
        grid.validate(TestingScenario::Unit, ExecutionEnvironment::Local, &FeatureSet::new());
    assert!(result.is_some(), "validate() must return Some(...) for a known cell");
    let (missing, _) = result.unwrap();
    assert!(
        !missing.is_empty(),
        "Unit/Local must report missing required features when active set is empty"
    );
}

#[test]
fn validate_api_returns_none_for_nonexistent_cell() {
    let grid = curated();
    // Production environment has no grid cells.
    let result =
        grid.validate(TestingScenario::Debug, ExecutionEnvironment::Production, &FeatureSet::new());
    assert!(result.is_none(), "validate() must return None for an unknown scenario/env pair");
}

// ── 6. rows_for_scenario API ─────────────────────────────────────────────────

#[test]
fn rows_for_scenario_unit_returns_two_cells() {
    let grid = curated();
    let cells = grid.rows_for_scenario(TestingScenario::Unit);
    assert_eq!(cells.len(), 2, "Unit scenario must have exactly two grid cells (Local and Ci)");
}

#[test]
fn rows_for_scenario_performance_returns_two_cells() {
    let grid = curated();
    let cells = grid.rows_for_scenario(TestingScenario::Performance);
    assert_eq!(cells.len(), 2, "Performance scenario must have exactly two cells (Local and Ci)");
}

#[test]
fn rows_for_scenario_nonexistent_returns_empty() {
    // Production is not used in the curated grid.
    let no_prod = curated()
        .rows()
        .iter()
        .filter(|c| c.environment == ExecutionEnvironment::Production)
        .count();
    assert_eq!(no_prod, 0, "No curated cells should be in the Production environment");
}

// ── 7. FeatureSet helpers ────────────────────────────────────────────────────

#[test]
fn feature_set_from_names_builds_correct_set() {
    let set = feature_set_from_names(&["inference", "kernels", "tokenizers"]);
    assert!(set.contains(BitnetFeature::Inference));
    assert!(set.contains(BitnetFeature::Kernels));
    assert!(set.contains(BitnetFeature::Tokenizers));
    assert!(!set.contains(BitnetFeature::Gpu));
}

#[test]
fn feature_set_from_names_silently_ignores_unknown_names() {
    let set = feature_set_from_names(&["inference", "not-a-real-feature", "kernels"]);
    assert!(set.contains(BitnetFeature::Inference));
    assert!(set.contains(BitnetFeature::Kernels));
    // The unknown name must not be there (and no panic)
    assert_eq!(set.labels().len(), 2, "Unknown feature name should be silently dropped");
}

#[test]
fn feature_set_insert_and_contains_roundtrip() {
    let mut set = FeatureSet::new();
    assert!(!set.contains(BitnetFeature::Reporting));
    set.insert(BitnetFeature::Reporting);
    assert!(set.contains(BitnetFeature::Reporting));
}

#[test]
fn feature_set_iter_yields_all_inserted_features() {
    let set = feature_set_from_names(&["cpu", "gpu", "inference"]);
    let collected: Vec<_> = set.iter().copied().collect();
    assert_eq!(collected.len(), 3);
    assert!(collected.contains(&BitnetFeature::Cpu));
    assert!(collected.contains(&BitnetFeature::Gpu));
    assert!(collected.contains(&BitnetFeature::Inference));
}

// ── 8. TestingScenario and ExecutionEnvironment parsing ──────────────────────

#[test]
fn testing_scenario_all_variants_parse_from_canonical_display() {
    let variants = [
        (TestingScenario::Unit, "unit"),
        (TestingScenario::Integration, "integration"),
        (TestingScenario::EndToEnd, "e2e"),
        (TestingScenario::Performance, "performance"),
        (TestingScenario::CrossValidation, "crossval"),
        (TestingScenario::Smoke, "smoke"),
        (TestingScenario::Development, "development"),
        (TestingScenario::Debug, "debug"),
        (TestingScenario::Minimal, "minimal"),
    ];
    for (expected, s) in &variants {
        let parsed: TestingScenario = s.parse().unwrap_or_else(|_| panic!("failed to parse '{s}'"));
        assert_eq!(parsed, *expected, "parsing '{s}' should yield {expected:?}");
    }
}

#[test]
fn testing_scenario_alternate_aliases_parse() {
    assert_eq!("end-to-end".parse::<TestingScenario>(), Ok(TestingScenario::EndToEnd));
    assert_eq!("perf".parse::<TestingScenario>(), Ok(TestingScenario::Performance));
    assert_eq!("dev".parse::<TestingScenario>(), Ok(TestingScenario::Development));
    assert_eq!("min".parse::<TestingScenario>(), Ok(TestingScenario::Minimal));
    assert_eq!("cross-validation".parse::<TestingScenario>(), Ok(TestingScenario::CrossValidation));
}

#[test]
fn execution_environment_all_variants_parse() {
    let variants = [
        (ExecutionEnvironment::Local, "local"),
        (ExecutionEnvironment::Ci, "ci"),
        (ExecutionEnvironment::PreProduction, "pre-prod"),
        (ExecutionEnvironment::Production, "production"),
    ];
    for (expected, s) in &variants {
        let parsed: ExecutionEnvironment =
            s.parse().unwrap_or_else(|_| panic!("failed to parse env '{s}'"));
        assert_eq!(parsed, *expected, "parsing '{s}' should yield {expected:?}");
    }
}

#[test]
fn testing_scenario_unknown_string_errors() {
    assert!("totally-unknown".parse::<TestingScenario>().is_err());
}

#[test]
fn execution_environment_unknown_string_errors() {
    assert!("totally-unknown".parse::<ExecutionEnvironment>().is_err());
}

// ── 9. Display round-trips ───────────────────────────────────────────────────

#[test]
fn bitnet_feature_display_round_trips() {
    let features = [
        BitnetFeature::Cpu,
        BitnetFeature::Gpu,
        BitnetFeature::Inference,
        BitnetFeature::Kernels,
        BitnetFeature::Tokenizers,
        BitnetFeature::CrossValidation,
        BitnetFeature::Reporting,
        BitnetFeature::Fixtures,
        BitnetFeature::CppFfi,
        BitnetFeature::Server,
        BitnetFeature::Cli,
    ];
    for f in &features {
        let label = f.to_string();
        let parsed: BitnetFeature =
            label.parse().unwrap_or_else(|_| panic!("round-trip failed for {f:?}: '{label}'"));
        assert_eq!(parsed, *f, "Display→parse round-trip must be identity for {f:?}");
    }
}
