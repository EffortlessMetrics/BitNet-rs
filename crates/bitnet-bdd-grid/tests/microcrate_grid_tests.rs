//! BDD grid coverage tests for new microcrates: device-probe, logits, generation, engine-core.
//!
//! Each test asserts:
//! - The cell exists in the curated grid.
//! - Required features match the microcrate's known compile-time needs.
//! - The cell's intent is descriptive.
//! - supports() returns true when the required features are active.

use bitnet_bdd_grid::{
    BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario, curated,
    feature_set_from_names,
};

// ── device-probe: Smoke/Local ─────────────────────────────────────────────────

#[test]
fn device_probe_smoke_local_cell_exists() {
    let cell = curated().find(TestingScenario::Smoke, ExecutionEnvironment::Local);
    assert!(cell.is_some(), "Smoke/Local cell for device-probe must be in the curated grid");
}

#[test]
fn device_probe_smoke_local_requires_cpu_feature() {
    let cell = curated()
        .find(TestingScenario::Smoke, ExecutionEnvironment::Local)
        .expect("Smoke/Local must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::Cpu),
        "device-probe Smoke/Local must require the `cpu` feature for SIMD capability detection"
    );
}

#[test]
fn device_probe_smoke_local_requires_inference_feature() {
    let cell = curated()
        .find(TestingScenario::Smoke, ExecutionEnvironment::Local)
        .expect("Smoke/Local must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::Inference),
        "device-probe Smoke/Local must require `inference` to tie probe results to the engine"
    );
}

#[test]
fn device_probe_smoke_local_forbids_gpu() {
    let cell = curated()
        .find(TestingScenario::Smoke, ExecutionEnvironment::Local)
        .expect("Smoke/Local must exist");
    assert!(
        cell.forbidden_features.contains(BitnetFeature::Gpu),
        "device-probe Smoke/Local must forbid `gpu` — this cell targets CPU-only detection"
    );
}

#[test]
fn device_probe_smoke_local_supports_cpu_inference_active_set() {
    let cell = curated()
        .find(TestingScenario::Smoke, ExecutionEnvironment::Local)
        .expect("Smoke/Local must exist");
    let active = feature_set_from_names(&["cpu", "inference"]);
    assert!(
        cell.supports(&active),
        "device-probe Smoke/Local cell must support an active set of {{cpu, inference}}"
    );
}

#[test]
fn device_probe_smoke_local_does_not_support_gpu_active_set() {
    // Given the device-probe cell forbids gpu, a feature set containing gpu must be rejected.
    let cell = curated()
        .find(TestingScenario::Smoke, ExecutionEnvironment::Local)
        .expect("Smoke/Local must exist");
    let mut active = cell.required_features.clone();
    active.insert(BitnetFeature::Gpu);
    assert!(
        !cell.supports(&active),
        "device-probe Smoke/Local must not support a feature set containing the forbidden `gpu`"
    );
}

#[test]
fn device_probe_smoke_local_intent_mentions_simd() {
    let cell = curated()
        .find(TestingScenario::Smoke, ExecutionEnvironment::Local)
        .expect("Smoke/Local must exist");
    assert!(
        cell.intent.to_ascii_lowercase().contains("simd")
            || cell.intent.to_ascii_lowercase().contains("cpu"),
        "device-probe intent must mention SIMD or CPU capabilities; got: '{}'",
        cell.intent
    );
}

// ── logits: Integration/PreProduction ────────────────────────────────────────

#[test]
fn logits_integration_preprod_cell_exists() {
    let cell = curated().find(TestingScenario::Integration, ExecutionEnvironment::PreProduction);
    assert!(
        cell.is_some(),
        "Integration/PreProduction cell for logits must be in the curated grid"
    );
}

#[test]
fn logits_integration_preprod_requires_cpu_kernels_inference() {
    let cell = curated()
        .find(TestingScenario::Integration, ExecutionEnvironment::PreProduction)
        .expect("Integration/PreProduction must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::Cpu),
        "logits Integration/PreProduction must require `cpu`"
    );
    assert!(
        cell.required_features.contains(BitnetFeature::Kernels),
        "logits Integration/PreProduction must require `kernels` for SIMD dispatch"
    );
    assert!(
        cell.required_features.contains(BitnetFeature::Inference),
        "logits Integration/PreProduction must require `inference`"
    );
}

#[test]
fn logits_integration_preprod_supports_required_set() {
    let cell = curated()
        .find(TestingScenario::Integration, ExecutionEnvironment::PreProduction)
        .expect("Integration/PreProduction must exist");
    let active = cell.required_features.clone();
    assert!(
        cell.supports(&active),
        "logits Integration/PreProduction must support its own required feature set"
    );
}

#[test]
fn logits_integration_preprod_intent_mentions_logits_or_transforms() {
    let cell = curated()
        .find(TestingScenario::Integration, ExecutionEnvironment::PreProduction)
        .expect("Integration/PreProduction must exist");
    let lower = cell.intent.to_ascii_lowercase();
    assert!(
        lower.contains("logit") || lower.contains("transform") || lower.contains("top-k"),
        "logits Integration/PreProduction intent must describe transforms; got: '{}'",
        cell.intent
    );
}

// ── generation: Debug/PreProduction ──────────────────────────────────────────

#[test]
fn generation_debug_preprod_cell_exists() {
    let cell = curated().find(TestingScenario::Debug, ExecutionEnvironment::PreProduction);
    assert!(cell.is_some(), "Debug/PreProduction cell for generation must be in the curated grid");
}

#[test]
fn generation_debug_preprod_requires_inference() {
    let cell = curated()
        .find(TestingScenario::Debug, ExecutionEnvironment::PreProduction)
        .expect("Debug/PreProduction must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::Inference),
        "generation Debug/PreProduction must require `inference` for stop-criteria coverage"
    );
}

#[test]
fn generation_debug_preprod_optional_includes_trace() {
    let cell = curated()
        .find(TestingScenario::Debug, ExecutionEnvironment::PreProduction)
        .expect("Debug/PreProduction must exist");
    assert!(
        cell.optional_features.contains(BitnetFeature::Trace),
        "generation Debug/PreProduction should list `trace` as optional for diagnostics"
    );
}

#[test]
fn generation_debug_preprod_supports_inference_only_set() {
    let cell = curated()
        .find(TestingScenario::Debug, ExecutionEnvironment::PreProduction)
        .expect("Debug/PreProduction must exist");
    let active = feature_set_from_names(&["inference"]);
    assert!(
        cell.supports(&active),
        "generation Debug/PreProduction must support minimal {{inference}} feature set"
    );
}

#[test]
fn generation_debug_preprod_intent_mentions_stopping() {
    let cell = curated()
        .find(TestingScenario::Debug, ExecutionEnvironment::PreProduction)
        .expect("Debug/PreProduction must exist");
    let lower = cell.intent.to_ascii_lowercase();
    assert!(
        lower.contains("stop") || lower.contains("generation") || lower.contains("criteria"),
        "generation Debug/PreProduction intent must describe stopping; got: '{}'",
        cell.intent
    );
}

// ── engine-core: CrossValidation/PreProduction ───────────────────────────────

#[test]
fn engine_core_crossval_preprod_cell_exists() {
    let cell =
        curated().find(TestingScenario::CrossValidation, ExecutionEnvironment::PreProduction);
    assert!(
        cell.is_some(),
        "CrossValidation/PreProduction cell for engine-core must be in the curated grid"
    );
}

#[test]
fn engine_core_crossval_preprod_requires_inference_kernels() {
    let cell = curated()
        .find(TestingScenario::CrossValidation, ExecutionEnvironment::PreProduction)
        .expect("CrossValidation/PreProduction must exist");
    assert!(
        cell.required_features.contains(BitnetFeature::Inference),
        "engine-core CrossValidation/PreProduction must require `inference`"
    );
    assert!(
        cell.required_features.contains(BitnetFeature::Kernels),
        "engine-core CrossValidation/PreProduction must require `kernels` for kernel dispatch"
    );
}

#[test]
fn engine_core_crossval_preprod_optional_includes_crossval_and_fixtures() {
    let cell = curated()
        .find(TestingScenario::CrossValidation, ExecutionEnvironment::PreProduction)
        .expect("CrossValidation/PreProduction must exist");
    assert!(
        cell.optional_features.contains(BitnetFeature::CrossValidation),
        "engine-core CrossValidation/PreProduction should list `crossval` as optional"
    );
    assert!(
        cell.optional_features.contains(BitnetFeature::Fixtures),
        "engine-core CrossValidation/PreProduction should list `fixtures` as optional"
    );
}

#[test]
fn engine_core_crossval_preprod_supports_inference_kernels_set() {
    let cell = curated()
        .find(TestingScenario::CrossValidation, ExecutionEnvironment::PreProduction)
        .expect("CrossValidation/PreProduction must exist");
    let active = feature_set_from_names(&["inference", "kernels"]);
    assert!(
        cell.supports(&active),
        "engine-core CrossValidation/PreProduction must support {{inference, kernels}}"
    );
}

#[test]
fn engine_core_crossval_preprod_intent_mentions_session_or_contract() {
    let cell = curated()
        .find(TestingScenario::CrossValidation, ExecutionEnvironment::PreProduction)
        .expect("CrossValidation/PreProduction must exist");
    let lower = cell.intent.to_ascii_lowercase();
    assert!(
        lower.contains("session") || lower.contains("contract") || lower.contains("engine"),
        "engine-core intent must describe session contracts; got: '{}'",
        cell.intent
    );
}

// ── cross-cutting: total cell count reflects new additions ───────────────────

#[test]
fn curated_grid_has_at_least_twenty_two_cells() {
    // Original count was 18; we added 4 microcrate cells.
    let count = curated().rows().len();
    assert!(
        count >= 22,
        "curated grid should have at least 22 cells after microcrate additions; got {}",
        count
    );
}

#[test]
fn all_new_microcrate_cells_have_non_empty_intents() {
    let new_cells = [
        curated().find(TestingScenario::Smoke, ExecutionEnvironment::Local),
        curated().find(TestingScenario::Integration, ExecutionEnvironment::PreProduction),
        curated().find(TestingScenario::Debug, ExecutionEnvironment::PreProduction),
        curated().find(TestingScenario::CrossValidation, ExecutionEnvironment::PreProduction),
    ];
    for cell in new_cells.into_iter().flatten() {
        assert!(!cell.intent.is_empty(), "all microcrate cells must have non-empty intents");
    }
}

#[test]
fn all_new_microcrate_cells_have_disjoint_required_and_forbidden() {
    let new_cells = [
        curated().find(TestingScenario::Smoke, ExecutionEnvironment::Local),
        curated().find(TestingScenario::Integration, ExecutionEnvironment::PreProduction),
        curated().find(TestingScenario::Debug, ExecutionEnvironment::PreProduction),
        curated().find(TestingScenario::CrossValidation, ExecutionEnvironment::PreProduction),
    ];
    for cell in new_cells.into_iter().flatten() {
        let overlap = cell.required_features.forbidden_overlap(&cell.forbidden_features);
        assert!(
            overlap.is_empty(),
            "microcrate cell {:?}/{:?}: required∩forbidden = {:?}",
            cell.scenario,
            cell.environment,
            overlap.labels()
        );
    }
}

// ── GPU compile-only cell (no active GPU runtime required) ────────────────────

/// The engine-core CrossValidation/PreProduction cell must NOT require gpu/cuda,
/// keeping it buildable in CPU-only CI configurations.
#[test]
fn engine_core_crossval_preprod_does_not_require_gpu() {
    let cell = curated()
        .find(TestingScenario::CrossValidation, ExecutionEnvironment::PreProduction)
        .expect("CrossValidation/PreProduction must exist");
    assert!(
        !cell.required_features.contains(BitnetFeature::Gpu),
        "engine-core CrossValidation/PreProduction must not require gpu (compile-only path)"
    );
    assert!(
        !cell.required_features.contains(BitnetFeature::Cuda),
        "engine-core CrossValidation/PreProduction must not require cuda (compile-only path)"
    );
}

/// The logits Integration/PreProduction cell must NOT require gpu/cuda,
/// keeping pure math transforms testable in CPU-only builds.
#[test]
fn logits_integration_preprod_does_not_require_gpu() {
    let cell = curated()
        .find(TestingScenario::Integration, ExecutionEnvironment::PreProduction)
        .expect("Integration/PreProduction must exist");
    assert!(
        !cell.required_features.contains(BitnetFeature::Gpu),
        "logits Integration/PreProduction must not require gpu"
    );
}

// ── violations() API sanity ───────────────────────────────────────────────────

#[test]
fn device_probe_cell_reports_missing_features_for_empty_active_set() {
    let grid = curated();
    let result =
        grid.validate(TestingScenario::Smoke, ExecutionEnvironment::Local, &FeatureSet::new());
    assert!(result.is_some(), "validate() must return Some for Smoke/Local");
    let (missing, _) = result.unwrap();
    assert!(!missing.is_empty(), "Smoke/Local must report missing features for empty active set");
}

#[test]
fn generation_debug_cell_reports_no_violations_for_inference_set() {
    let grid = curated();
    let active = feature_set_from_names(&["inference"]);
    let result =
        grid.validate(TestingScenario::Debug, ExecutionEnvironment::PreProduction, &active);
    assert!(result.is_some());
    let (missing, violated) = result.unwrap();
    assert!(
        missing.is_empty() && violated.is_empty(),
        "generation Debug/PreProduction: unexpected violations for {{inference}} active set; \
         missing={:?}, violated={:?}",
        missing,
        violated
    );
}
