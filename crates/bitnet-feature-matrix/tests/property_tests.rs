//! Property-based tests for `bitnet-feature-matrix`.
//!
//! Key invariants:
//! - `TestingScenario` and `ExecutionEnvironment`: Display→FromStr round-trip identity
//! - `BitnetFeature`: Display→FromStr round-trip for all 23 variants
//! - `FeatureSet`: set-algebra laws (missing_required, forbidden_overlap, satisfies)
//! - `BddGrid::canonical_grid()`: non-empty, lookup stability

use bitnet_feature_matrix::{
    BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario,
    canonical_grid,
};
use proptest::prelude::*;

// ── Arbitrary strategies ──────────────────────────────────────────────────────

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
        Just(BitnetFeature::Metal),
        Just(BitnetFeature::Vulkan),
        Just(BitnetFeature::Oneapi),
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

fn arb_feature_set() -> impl Strategy<Value = FeatureSet> {
    prop::collection::vec(arb_feature(), 0..8).prop_map(|features| {
        let mut set = FeatureSet::new();
        for f in features {
            set.insert(f);
        }
        set
    })
}

// ── TestingScenario round-trip ────────────────────────────────────────────────

proptest! {
    #[test]
    fn scenario_display_fromstr_roundtrip(s in arb_scenario()) {
        let display = s.to_string();
        let parsed: TestingScenario = display.parse().unwrap();
        prop_assert_eq!(s, parsed);
    }

    #[test]
    fn scenario_display_is_nonempty(s in arb_scenario()) {
        prop_assert!(!s.to_string().is_empty());
    }

    #[test]
    fn scenario_ordering_is_reflexive(s in arb_scenario()) {
        prop_assert!(s == s);
        prop_assert!(s <= s);
    }
}

// ── ExecutionEnvironment round-trip ───────────────────────────────────────────

proptest! {
    #[test]
    fn environment_display_fromstr_roundtrip(e in arb_environment()) {
        let display = e.to_string();
        let parsed: ExecutionEnvironment = display.parse().unwrap();
        prop_assert_eq!(e, parsed);
    }

    #[test]
    fn environment_display_is_nonempty(e in arb_environment()) {
        prop_assert!(!e.to_string().is_empty());
    }
}

// ── BitnetFeature round-trip ──────────────────────────────────────────────────

proptest! {
    #[test]
    fn feature_display_fromstr_roundtrip(f in arb_feature()) {
        let display = f.to_string();
        let parsed: BitnetFeature = display.parse().unwrap();
        prop_assert_eq!(f, parsed);
    }

    #[test]
    fn feature_display_is_nonempty(f in arb_feature()) {
        prop_assert!(!f.to_string().is_empty());
    }

    #[test]
    fn feature_fromstr_rejects_garbage(s in "[^a-z-]+") {
        let result = s.parse::<BitnetFeature>();
        prop_assert!(result.is_err());
    }
}

// ── FeatureSet algebra ────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn feature_set_insert_makes_contains_true(f in arb_feature()) {
        let mut set = FeatureSet::new();
        set.insert(f);
        prop_assert!(set.contains(f));
    }

    #[test]
    fn feature_set_empty_has_no_missing(required in arb_feature_set()) {
        let empty = FeatureSet::new();
        let missing = empty.missing_required(&required);
        // Empty set is missing everything in required
        for f in required.iter() {
            prop_assert!(missing.contains(*f));
        }
    }

    #[test]
    fn feature_set_satisfies_when_superset(
        required in arb_feature_set(),
        extra in arb_feature_set(),
    ) {
        let mut superset = FeatureSet::new();
        for f in required.iter() {
            superset.insert(*f);
        }
        for f in extra.iter() {
            superset.insert(*f);
        }
        let empty_forbidden = FeatureSet::new();
        prop_assert!(superset.satisfies(&required, &empty_forbidden));
    }

    #[test]
    fn feature_set_forbidden_overlap_is_symmetric_membership(
        a in arb_feature_set(),
        b in arb_feature_set(),
    ) {
        let overlap = a.forbidden_overlap(&b);
        // Every feature in the overlap must be in both sets
        for f in overlap.iter() {
            prop_assert!(a.contains(*f));
            prop_assert!(b.contains(*f));
        }
    }

    #[test]
    fn feature_set_labels_count_matches_iter(set in arb_feature_set()) {
        let labels = set.labels();
        let count = set.iter().count();
        prop_assert_eq!(labels.len(), count);
    }
}

// ── BddGrid ──────────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn canonical_grid_is_nonempty(_dummy in 0u8..1) {
        let grid = canonical_grid();
        prop_assert!(!grid.rows().is_empty());
    }

    #[test]
    fn canonical_grid_find_is_stable(
        scenario in arb_scenario(),
        env in arb_environment(),
    ) {
        let grid = canonical_grid();
        let a = grid.find(scenario, env);
        let b = grid.find(scenario, env);
        // Both calls return the same presence
        prop_assert_eq!(a.is_some(), b.is_some());
    }

    #[test]
    fn canonical_grid_rows_for_scenario_consistent(scenario in arb_scenario()) {
        let grid = canonical_grid();
        let rows = grid.rows_for_scenario(scenario);
        for row in &rows {
            prop_assert_eq!(row.scenario, scenario);
        }
    }
}

// ── FeatureSet from Display strings ──────────────────────────────────────────

proptest! {
    #[test]
    fn feature_set_from_display_roundtrip(features in prop::collection::vec(arb_feature(), 0..6)) {
        let mut set = FeatureSet::new();
        for f in &features {
            set.insert(*f);
        }
        // Every label should parse back to a valid BitnetFeature
        for label in set.labels() {
            let parsed: BitnetFeature = label.parse().unwrap();
            prop_assert!(set.contains(parsed));
        }
    }
}
