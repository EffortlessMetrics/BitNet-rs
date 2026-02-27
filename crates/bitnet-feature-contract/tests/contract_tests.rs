//! Comprehensive tests for `bitnet-feature-contract`.
//!
//! Covers `FeatureContractSnapshot` invariants, `feature_contract_snapshot` /
//! `feature_contract_drift` semantics, and the re-exported profile/grid API
//! from `bitnet-feature-matrix`.

use bitnet_feature_contract::{
    BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario, active_features,
    active_profile_for, canonical_grid, feature_contract_drift, feature_contract_snapshot,
    feature_labels, feature_line, validate_active_profile_for,
};

// ── FeatureContractSnapshot: construction and consistency ─────────────────────

#[test]
fn snapshot_empty_inputs_is_consistent() {
    let snap = feature_contract_snapshot([] as [&str; 0], [] as [&str; 0]);
    assert!(snap.is_consistent(), "empty policy + empty runtime must be consistent");
    assert!(snap.missing_from_runtime.is_empty());
    assert!(snap.extra_from_runtime.is_empty());
}

#[test]
fn snapshot_identical_sets_is_consistent() {
    let snap =
        feature_contract_snapshot(["cpu", "inference", "kernels"], ["cpu", "inference", "kernels"]);
    assert!(snap.is_consistent(), "identical policy and runtime must be consistent");
    assert!(snap.missing_from_runtime.is_empty());
    assert!(snap.extra_from_runtime.is_empty());
}

#[test]
fn snapshot_order_does_not_affect_consistency() {
    // Consistency is set-based; insertion order must not matter.
    let a = feature_contract_snapshot(["cpu", "gpu"], ["gpu", "cpu"]);
    let b = feature_contract_snapshot(["gpu", "cpu"], ["cpu", "gpu"]);
    assert!(a.is_consistent());
    assert!(b.is_consistent());
    assert_eq!(a.is_consistent(), b.is_consistent());
}

#[test]
fn snapshot_missing_feature_detected() {
    let snap = feature_contract_snapshot(["cpu", "kernels"], ["cpu"]);
    assert!(!snap.is_consistent());
    assert!(
        snap.missing_from_runtime.contains(&"kernels".to_string()),
        "expected 'kernels' in missing_from_runtime: {:?}",
        snap.missing_from_runtime
    );
    assert!(snap.extra_from_runtime.is_empty(), "no extra expected");
}

#[test]
fn snapshot_extra_feature_detected() {
    let snap = feature_contract_snapshot(["cpu"], ["cpu", "gpu"]);
    assert!(!snap.is_consistent());
    assert!(
        snap.extra_from_runtime.contains(&"gpu".to_string()),
        "expected 'gpu' in extra_from_runtime: {:?}",
        snap.extra_from_runtime
    );
    assert!(snap.missing_from_runtime.is_empty(), "no missing expected");
}

#[test]
fn snapshot_detects_both_missing_and_extra() {
    let snap = feature_contract_snapshot(["cpu", "kernels"], ["cpu", "inference"]);
    assert!(!snap.is_consistent());
    assert!(
        snap.missing_from_runtime.contains(&"kernels".to_string()),
        "expected 'kernels' in missing: {:?}",
        snap.missing_from_runtime
    );
    assert!(
        snap.extra_from_runtime.contains(&"inference".to_string()),
        "expected 'inference' in extra: {:?}",
        snap.extra_from_runtime
    );
}

#[test]
fn snapshot_missing_and_extra_are_disjoint() {
    let snap = feature_contract_snapshot(["cpu", "kernels", "trace"], ["cpu", "inference", "gpu"]);
    let missing: std::collections::BTreeSet<_> = snap.missing_from_runtime.iter().collect();
    let extra: std::collections::BTreeSet<_> = snap.extra_from_runtime.iter().collect();
    let overlap: Vec<_> = missing.intersection(&extra).collect();
    assert!(overlap.is_empty(), "missing and extra must be disjoint, found overlap: {overlap:?}");
}

#[test]
fn snapshot_preserves_input_order_for_policy() {
    // policy_features is stored in insertion order (Vec), not sorted.
    let snap = feature_contract_snapshot(["kernels", "cpu", "gpu"], ["kernels", "cpu", "gpu"]);
    assert_eq!(
        snap.policy_features,
        vec!["kernels".to_string(), "cpu".to_string(), "gpu".to_string()],
        "policy_features must preserve input order"
    );
}

#[test]
fn snapshot_preserves_input_order_for_runtime() {
    let snap = feature_contract_snapshot(["cpu"], ["inference", "cpu", "crossval"]);
    assert_eq!(
        snap.runtime_features,
        vec!["inference".to_string(), "cpu".to_string(), "crossval".to_string()],
        "runtime_features must preserve input order"
    );
}

#[test]
fn snapshot_clone_and_equality() {
    let snap = feature_contract_snapshot(["cpu", "gpu"], ["cpu"]);
    let cloned = snap.clone();
    assert_eq!(snap, cloned, "cloned snapshot must equal original");
    assert!(!cloned.is_consistent());
}

// ── feature_contract_drift ────────────────────────────────────────────────────

#[test]
fn drift_returns_none_when_policy_and_runtime_match() {
    let result = feature_contract_drift(["cpu", "inference"], ["inference", "cpu"]);
    assert!(result.is_none(), "drift must be None when sets are equal");
}

#[test]
fn drift_returns_some_when_policy_and_runtime_differ() {
    let result = feature_contract_drift(["cpu", "kernels"], ["cpu"]);
    assert!(result.is_some(), "drift must be Some when sets differ");
    let snap = result.unwrap();
    assert!(!snap.is_consistent());
}

#[test]
fn drift_snapshot_contents_match_direct_snapshot() {
    let (policy, runtime) = (vec!["cpu", "gpu", "trace"], vec!["cpu", "crossval"]);
    let from_drift = feature_contract_drift(policy.iter().copied(), runtime.iter().copied())
        .expect("sets differ so drift must be Some");
    let direct = feature_contract_snapshot(policy.iter().copied(), runtime.iter().copied());
    assert_eq!(
        from_drift.missing_from_runtime, direct.missing_from_runtime,
        "drift and snapshot must agree on missing"
    );
    assert_eq!(
        from_drift.extra_from_runtime, direct.extra_from_runtime,
        "drift and snapshot must agree on extra"
    );
}

// ── Re-exported feature-matrix API ───────────────────────────────────────────

#[test]
fn active_features_returns_valid_feature_set() {
    // active_features() is a compile-time snapshot; must always succeed.
    let features = active_features();
    // Cross-check: labels from FeatureSet must match feature_labels().
    let from_set = features.labels();
    let direct = feature_labels();
    assert_eq!(from_set, direct, "active_features().labels() must match feature_labels()");
}

#[test]
fn feature_line_always_has_features_prefix() {
    let line = feature_line();
    assert!(
        line.starts_with("features: "),
        "feature_line must start with 'features: ', got: {line:?}"
    );
}

#[test]
fn feature_labels_contains_cpu_when_compiled_with_cpu_feature() {
    #[cfg(feature = "cpu")]
    {
        let labels = feature_labels();
        assert!(
            labels.iter().any(|l| l == "cpu"),
            "expected 'cpu' in feature labels when --features cpu, got: {labels:?}"
        );
    }
    #[cfg(not(feature = "cpu"))]
    {} // no-op
}

#[test]
fn canonical_grid_is_nonempty() {
    let grid = canonical_grid();
    // The curated grid must contain at least one row.
    // We probe the Unit / Local pair which is always defined.
    let rows = grid.rows_for_scenario(TestingScenario::Unit);
    assert!(!rows.is_empty(), "canonical_grid must contain rows for Unit scenario");
}

#[test]
fn active_profile_for_known_scenario_env_is_constructed() {
    let profile = active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Ci);
    assert_eq!(profile.scenario, TestingScenario::Unit);
    assert_eq!(profile.environment, ExecutionEnvironment::Ci);
    // features are always populated (may be empty if no flags compiled in)
    let _ = profile.features.labels(); // must not panic
}

#[test]
fn validate_active_profile_for_unit_ci_returns_result() {
    // validate_active_profile_for returns Some only when the grid has a cell for the
    // given scenario/env pair.  We simply assert it does not panic and the return value
    // is coherent.
    let result = validate_active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Ci);
    if let Some((missing, forbidden)) = result {
        // Both FeatureSet values must be valid (labels may be empty).
        let _ = missing.labels();
        let _ = forbidden.labels();
    }
    // None is also valid when no grid cell exists for Unit/Ci.
}

#[test]
fn feature_set_operations_are_correct() {
    // Verify FeatureSet works correctly when constructed manually.
    let mut fs = FeatureSet::new();
    assert!(fs.is_empty());
    fs.insert(BitnetFeature::Cpu);
    assert!(!fs.is_empty());
    assert!(fs.contains(BitnetFeature::Cpu));
    assert!(!fs.contains(BitnetFeature::Gpu));
    let labels = fs.labels();
    assert_eq!(labels, vec!["cpu".to_string()]);
}

#[test]
fn feature_set_from_names_roundtrip() {
    let names = ["cpu", "inference", "kernels"];
    let fs = FeatureSet::from_names(names);
    let labels = fs.labels();
    for name in &names {
        assert!(
            labels.contains(&name.to_string()),
            "expected '{name}' in labels from FeatureSet::from_names, got {labels:?}"
        );
    }
}

// ── FeatureContractSnapshot: Debug formatting ─────────────────────────────────

#[test]
fn snapshot_debug_format_is_nonempty() {
    let snap = feature_contract_snapshot(["cpu"], ["cpu", "gpu"]);
    let debug = format!("{snap:?}");
    assert!(!debug.is_empty(), "Debug output must not be empty");
    assert!(debug.contains("FeatureContractSnapshot"), "expected type name in debug: {debug}");
}
