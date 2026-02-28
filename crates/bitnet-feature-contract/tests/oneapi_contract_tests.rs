//! Feature contract tests for oneapi feature combinations.
//!
//! Validates that the `oneapi` feature is properly represented in the
//! feature-contract infrastructure and that common feature combinations
//! involving `oneapi` are handled correctly by the drift/snapshot APIs.

use bitnet_feature_contract::{
    BitnetFeature, FeatureSet, feature_contract_drift, feature_contract_snapshot,
};

// ── oneapi in FeatureSet ─────────────────────────────────────────────────────

#[test]
fn feature_set_contains_oneapi_variant() {
    let mut fs = FeatureSet::new();
    fs.insert(BitnetFeature::Oneapi);
    assert!(fs.contains(BitnetFeature::Oneapi), "FeatureSet must support Oneapi");
    let labels = fs.labels();
    assert!(
        labels.iter().any(|l| l == "oneapi"),
        "expected 'oneapi' in labels, got: {labels:?}"
    );
}

#[test]
fn feature_set_from_names_includes_oneapi() {
    let fs = FeatureSet::from_names(["oneapi"]);
    assert!(fs.contains(BitnetFeature::Oneapi));
}

// ── oneapi alone ─────────────────────────────────────────────────────────────

#[test]
fn oneapi_alone_is_consistent_when_both_sides_match() {
    let snap = feature_contract_snapshot(["oneapi"], ["oneapi"]);
    assert!(snap.is_consistent(), "oneapi alone must be consistent when matched");
}

#[test]
fn oneapi_alone_drift_is_none_when_matched() {
    let drift = feature_contract_drift(["oneapi"], ["oneapi"]);
    assert!(drift.is_none(), "no drift when oneapi matches");
}

// ── oneapi + cpu: no conflict ────────────────────────────────────────────────

#[test]
fn oneapi_plus_cpu_is_consistent() {
    let snap = feature_contract_snapshot(["oneapi", "cpu"], ["oneapi", "cpu"]);
    assert!(snap.is_consistent(), "oneapi + cpu must not conflict");
    assert!(snap.missing_from_runtime.is_empty());
    assert!(snap.extra_from_runtime.is_empty());
}

#[test]
fn oneapi_plus_cpu_drift_is_none() {
    let drift = feature_contract_drift(["oneapi", "cpu"], ["cpu", "oneapi"]);
    assert!(drift.is_none(), "order must not affect oneapi + cpu consistency");
}

// ── oneapi + gpu: CUDA and OpenCL coexist ────────────────────────────────────

#[test]
fn oneapi_plus_gpu_is_consistent() {
    let snap = feature_contract_snapshot(["oneapi", "gpu"], ["oneapi", "gpu"]);
    assert!(snap.is_consistent(), "oneapi + gpu must coexist");
}

#[test]
fn oneapi_plus_gpu_plus_cuda_is_consistent() {
    let snap = feature_contract_snapshot(
        ["oneapi", "gpu", "cuda"],
        ["oneapi", "gpu", "cuda"],
    );
    assert!(snap.is_consistent(), "oneapi + gpu + cuda must coexist");
}

#[test]
fn oneapi_plus_gpu_detects_missing_oneapi() {
    let snap = feature_contract_snapshot(["oneapi", "gpu"], ["gpu"]);
    assert!(!snap.is_consistent());
    assert!(
        snap.missing_from_runtime.contains(&"oneapi".to_string()),
        "expected 'oneapi' in missing, got: {:?}",
        snap.missing_from_runtime
    );
}

// ── oneapi without cpu or gpu ────────────────────────────────────────────────

#[test]
fn oneapi_without_cpu_or_gpu_is_consistent() {
    let snap = feature_contract_snapshot(["oneapi"], ["oneapi"]);
    assert!(snap.is_consistent(), "oneapi alone (no cpu, no gpu) must be consistent");
}

#[test]
fn oneapi_only_policy_detects_extra_cpu() {
    let snap = feature_contract_snapshot(["oneapi"], ["oneapi", "cpu"]);
    assert!(!snap.is_consistent());
    assert!(
        snap.extra_from_runtime.contains(&"cpu".to_string()),
        "expected 'cpu' in extra, got: {:?}",
        snap.extra_from_runtime
    );
}

// ── oneapi feature set operations ────────────────────────────────────────────

#[test]
fn oneapi_feature_set_disjoint_from_cuda() {
    let mut oneapi_set = FeatureSet::new();
    oneapi_set.insert(BitnetFeature::Oneapi);

    let mut cuda_set = FeatureSet::new();
    cuda_set.insert(BitnetFeature::Cuda);

    assert!(!oneapi_set.contains(BitnetFeature::Cuda));
    assert!(!cuda_set.contains(BitnetFeature::Oneapi));
}

#[test]
fn oneapi_combined_feature_set() {
    let fs = FeatureSet::from_names(["oneapi", "cpu", "gpu", "cuda"]);
    assert!(fs.contains(BitnetFeature::Oneapi));
    assert!(fs.contains(BitnetFeature::Cpu));
    assert!(fs.contains(BitnetFeature::Gpu));
    assert!(fs.contains(BitnetFeature::Cuda));
    let labels = fs.labels();
    assert!(labels.len() >= 4, "expected at least 4 features, got: {labels:?}");
}

// ── drift detection for oneapi combinations ──────────────────────────────────

#[test]
fn drift_detects_oneapi_added_at_runtime() {
    let drift = feature_contract_drift(["cpu"], ["cpu", "oneapi"]);
    assert!(drift.is_some(), "adding oneapi at runtime is drift");
    let snap = drift.unwrap();
    assert!(snap.extra_from_runtime.contains(&"oneapi".to_string()));
}

#[test]
fn drift_detects_oneapi_removed_at_runtime() {
    let drift = feature_contract_drift(["cpu", "oneapi"], ["cpu"]);
    assert!(drift.is_some(), "removing oneapi at runtime is drift");
    let snap = drift.unwrap();
    assert!(snap.missing_from_runtime.contains(&"oneapi".to_string()));
}
