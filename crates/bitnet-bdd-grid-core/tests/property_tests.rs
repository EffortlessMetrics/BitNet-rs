//! Property-based tests for `bitnet-bdd-grid-core`.
//!
//! Tests `Display`/`FromStr` round-trips and `FeatureSet` invariants.

use bitnet_bdd_grid_core::{BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario};
use proptest::prelude::*;
use std::str::FromStr;

fn arb_scenario() -> impl Strategy<Value = TestingScenario> {
    prop_oneof![
        Just(TestingScenario::Unit),
        Just(TestingScenario::Integration),
        Just(TestingScenario::EndToEnd),
        Just(TestingScenario::Performance),
        Just(TestingScenario::CrossValidation),
        Just(TestingScenario::Smoke),
        Just(TestingScenario::Development),
        Just(TestingScenario::Debug),
        Just(TestingScenario::Minimal),
    ]
}

fn arb_environment() -> impl Strategy<Value = ExecutionEnvironment> {
    prop_oneof![
        Just(ExecutionEnvironment::Local),
        Just(ExecutionEnvironment::Ci),
        Just(ExecutionEnvironment::PreProduction),
        Just(ExecutionEnvironment::Production),
    ]
}

fn arb_feature() -> impl Strategy<Value = BitnetFeature> {
    prop_oneof![
        Just(BitnetFeature::Cpu),
        Just(BitnetFeature::Gpu),
        Just(BitnetFeature::Cuda),
        Just(BitnetFeature::Inference),
        Just(BitnetFeature::Kernels),
        Just(BitnetFeature::Tokenizers),
        Just(BitnetFeature::Quantization),
        Just(BitnetFeature::Cli),
        Just(BitnetFeature::Server),
        Just(BitnetFeature::Ffi),
        Just(BitnetFeature::Python),
        Just(BitnetFeature::Wasm),
        Just(BitnetFeature::CrossValidation),
        Just(BitnetFeature::Trace),
        Just(BitnetFeature::Iq2sFfi),
        Just(BitnetFeature::CppFfi),
        Just(BitnetFeature::Fixtures),
        Just(BitnetFeature::Reporting),
        Just(BitnetFeature::Trend),
        Just(BitnetFeature::IntegrationTests),
    ]
}

proptest! {
    /// `TestingScenario::Display` → `FromStr` is a round-trip (lossless).
    #[test]
    fn scenario_display_from_str_round_trip(scenario in arb_scenario()) {
        let label = scenario.to_string();
        let parsed = TestingScenario::from_str(&label);
        prop_assert!(parsed.is_ok(), "failed to parse '{}': {:?}", label, parsed);
        prop_assert_eq!(parsed.unwrap(), scenario);
    }

    /// `ExecutionEnvironment::Display` → `FromStr` is a round-trip.
    #[test]
    fn environment_display_from_str_round_trip(env in arb_environment()) {
        let label = env.to_string();
        let parsed = ExecutionEnvironment::from_str(&label);
        prop_assert!(parsed.is_ok(), "failed to parse '{}': {:?}", label, parsed);
        prop_assert_eq!(parsed.unwrap(), env);
    }

    /// `BitnetFeature::Display` → `FromStr` is a round-trip.
    #[test]
    fn feature_display_from_str_round_trip(feature in arb_feature()) {
        let label = feature.to_string();
        let parsed = BitnetFeature::from_str(&label);
        prop_assert!(parsed.is_ok(), "failed to parse '{}': {:?}", label, parsed);
        prop_assert_eq!(parsed.unwrap(), feature);
    }

    /// A feature inserted into `FeatureSet` is always `contains`-able.
    #[test]
    fn feature_set_insert_contains(feature in arb_feature()) {
        let mut set = FeatureSet::new();
        set.insert(feature);
        prop_assert!(set.contains(feature));
        prop_assert!(!set.is_empty());
    }

    /// `missing_required` on a superset of required is always empty.
    #[test]
    fn feature_set_superset_has_no_missing(feature in arb_feature()) {
        let mut active = FeatureSet::new();
        active.insert(feature);
        let mut required = FeatureSet::new();
        required.insert(feature);
        let missing = active.missing_required(&required);
        prop_assert!(missing.is_empty());
    }

    /// `FeatureSet::labels()` always contains the label of every inserted feature.
    #[test]
    fn feature_set_labels_contain_inserted(feature in arb_feature()) {
        let mut set = FeatureSet::new();
        set.insert(feature);
        let labels = set.labels();
        let expected_label = feature.to_string();
        prop_assert!(
            labels.contains(&expected_label),
            "expected '{}' in {:?}",
            expected_label,
            labels
        );
    }
}

#[test]
fn feature_set_empty_by_default() {
    let set = FeatureSet::new();
    assert!(set.is_empty());
    assert!(set.labels().is_empty());
}

#[test]
fn unknown_scenario_returns_err() {
    assert!(TestingScenario::from_str("not_a_scenario").is_err());
}

#[test]
fn unknown_environment_returns_err() {
    assert!(ExecutionEnvironment::from_str("not_an_env").is_err());
}

#[test]
fn unknown_feature_returns_err() {
    assert!(BitnetFeature::from_str("not_a_feature").is_err());
}
