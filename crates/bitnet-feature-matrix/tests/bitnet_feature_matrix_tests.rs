//! Property and integration tests for `bitnet-feature-matrix`.
//!
//! The crate is a façade over `bitnet-runtime-profile-core`.  Tests verify the
//! grid, feature-label, and profile APIs maintain expected invariants.

use bitnet_feature_matrix::{
    BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario, active_features,
    active_profile, active_profile_for, canonical_grid, feature_labels, feature_line,
    validate_active_profile_for,
};
use proptest::prelude::*;

const SCENARIOS: [TestingScenario; 9] = [
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

const ENVS: [ExecutionEnvironment; 4] = [
    ExecutionEnvironment::Local,
    ExecutionEnvironment::Ci,
    ExecutionEnvironment::PreProduction,
    ExecutionEnvironment::Production,
];

// ── smoke tests ─────────────────────────────────────────────────────────────

#[test]
fn canonical_grid_is_non_empty() {
    let grid = canonical_grid();
    // Verify at least one scenario has grid rows.
    let has_rows = SCENARIOS.iter().any(|&s| !grid.rows_for_scenario(s).is_empty());
    assert!(has_rows, "canonical_grid must contain at least one BDD row");
}

#[test]
fn feature_labels_is_callable_and_returns_valid_strings() {
    // Returns active compile-time features; may be empty with --no-default-features.
    for label in feature_labels() {
        assert!(!label.is_empty(), "each feature label must be a non-empty string");
    }
}

#[test]
fn active_features_callable_and_returns_feature_set() {
    let _features: FeatureSet = active_features();
}

#[test]
fn active_profile_has_non_empty_scenario_and_env_strings() {
    let profile = active_profile();
    assert!(!profile.scenario.to_string().is_empty());
    assert!(!profile.environment.to_string().is_empty());
}

#[test]
fn active_profile_for_every_scenario_and_env_does_not_panic() {
    for scenario in SCENARIOS {
        for env in ENVS {
            let profile = active_profile_for(scenario, env);
            assert_eq!(profile.scenario, scenario);
            assert_eq!(profile.environment, env);
        }
    }
}

#[test]
fn feature_line_is_bounded_string() {
    let line = feature_line();
    assert!(line.len() <= 2048, "feature_line should not produce absurdly long output");
}

#[test]
fn validate_active_profile_for_unit_local_does_not_panic() {
    let _result = validate_active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Local);
}

// ── proptest invariants ──────────────────────────────────────────────────────

const FEATURES: [BitnetFeature; 20] = [
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

proptest! {
    /// Every `BitnetFeature` must display to a non-empty string and round-trip.
    #[test]
    fn bitnet_feature_display_roundtrip(idx in 0usize..20) {
        let f = FEATURES[idx];
        let displayed = f.to_string();
        prop_assert!(!displayed.is_empty());
        let parsed: BitnetFeature = displayed.parse().expect("display must round-trip via FromStr");
        prop_assert_eq!(f, parsed);
    }

    /// `active_profile_for` must preserve the inputs passed to it.
    #[test]
    fn active_profile_for_preserves_inputs(
        scenario_idx in 0usize..9,
        env_idx in 0usize..4,
    ) {
        let profile = active_profile_for(SCENARIOS[scenario_idx], ENVS[env_idx]);
        prop_assert_eq!(profile.scenario, SCENARIOS[scenario_idx]);
        prop_assert_eq!(profile.environment, ENVS[env_idx]);
    }

    /// A profile with an empty active feature set must never report forbidden violations.
    #[test]
    fn no_active_features_means_no_forbidden_violations(
        scenario_idx in 0usize..9,
        env_idx in 0usize..4,
    ) {
        let mut profile = active_profile_for(SCENARIOS[scenario_idx], ENVS[env_idx]);
        profile.features = FeatureSet::new();
        let (_missing, forbidden) = profile.violations();
        prop_assert!(
            forbidden.is_empty(),
            "empty active-feature set must never trigger forbidden violations"
        );
    }
}
