//! Edge-case tests for bitnet-feature-contract: FeatureContractSnapshot,
//! feature_contract_snapshot, feature_contract_drift.

use bitnet_feature_contract::{feature_contract_drift, feature_contract_snapshot};

// ---------------------------------------------------------------------------
// FeatureContractSnapshot — is_consistent
// ---------------------------------------------------------------------------

#[test]
fn snapshot_consistent_when_identical() {
    let snap = feature_contract_snapshot(vec!["cpu", "inference"], vec!["cpu", "inference"]);
    assert!(snap.is_consistent());
    assert!(snap.missing_from_runtime.is_empty());
    assert!(snap.extra_from_runtime.is_empty());
}

#[test]
fn snapshot_consistent_when_different_order() {
    let snap = feature_contract_snapshot(vec!["inference", "cpu"], vec!["cpu", "inference"]);
    assert!(snap.is_consistent());
}

#[test]
fn snapshot_not_consistent_when_missing() {
    let snap = feature_contract_snapshot(vec!["cpu", "inference"], vec!["cpu"]);
    assert!(!snap.is_consistent());
    assert_eq!(snap.missing_from_runtime, vec!["inference".to_string()]);
    assert!(snap.extra_from_runtime.is_empty());
}

#[test]
fn snapshot_not_consistent_when_extra() {
    let snap = feature_contract_snapshot(vec!["cpu"], vec!["cpu", "gpu"]);
    assert!(!snap.is_consistent());
    assert!(snap.missing_from_runtime.is_empty());
    assert_eq!(snap.extra_from_runtime, vec!["gpu".to_string()]);
}

#[test]
fn snapshot_not_consistent_both_missing_and_extra() {
    let snap = feature_contract_snapshot(vec!["cpu", "kernels"], vec!["cpu", "gpu"]);
    assert!(!snap.is_consistent());
    assert_eq!(snap.missing_from_runtime, vec!["kernels".to_string()]);
    assert_eq!(snap.extra_from_runtime, vec!["gpu".to_string()]);
}

#[test]
fn snapshot_empty_both() {
    let snap = feature_contract_snapshot(std::iter::empty::<&str>(), std::iter::empty::<&str>());
    assert!(snap.is_consistent());
    assert!(snap.policy_features.is_empty());
    assert!(snap.runtime_features.is_empty());
}

#[test]
fn snapshot_empty_policy() {
    let snap = feature_contract_snapshot(std::iter::empty::<&str>(), vec!["cpu"]);
    assert!(!snap.is_consistent());
    assert!(snap.missing_from_runtime.is_empty());
    assert_eq!(snap.extra_from_runtime, vec!["cpu".to_string()]);
}

#[test]
fn snapshot_empty_runtime() {
    let snap = feature_contract_snapshot(vec!["cpu"], std::iter::empty::<&str>());
    assert!(!snap.is_consistent());
    assert_eq!(snap.missing_from_runtime, vec!["cpu".to_string()]);
    assert!(snap.extra_from_runtime.is_empty());
}

#[test]
fn snapshot_preserves_input_order() {
    let snap = feature_contract_snapshot(vec!["z", "a", "m"], vec!["z", "a", "m"]);
    // policy_features preserves insertion order
    assert_eq!(snap.policy_features, vec!["z", "a", "m"]);
    assert_eq!(snap.runtime_features, vec!["z", "a", "m"]);
}

#[test]
fn snapshot_duplicates_in_input() {
    // Duplicates in input lists should be handled gracefully
    let snap = feature_contract_snapshot(vec!["cpu", "cpu"], vec!["cpu"]);
    // policy_features preserves raw input (including dupes)
    assert_eq!(snap.policy_features.len(), 2);
    // But set-based comparison should see them as consistent
    assert!(snap.is_consistent());
}

// ---------------------------------------------------------------------------
// FeatureContractSnapshot — Debug, Clone, PartialEq, Eq
// ---------------------------------------------------------------------------

#[test]
fn snapshot_debug() {
    let snap = feature_contract_snapshot(vec!["cpu"], vec!["cpu"]);
    let dbg = format!("{snap:?}");
    assert!(dbg.contains("FeatureContractSnapshot"));
}

#[test]
fn snapshot_clone() {
    let snap = feature_contract_snapshot(vec!["cpu", "gpu"], vec!["cpu"]);
    let cloned = snap.clone();
    assert_eq!(snap, cloned);
}

#[test]
fn snapshot_partial_eq() {
    let a = feature_contract_snapshot(vec!["cpu"], vec!["cpu"]);
    let b = feature_contract_snapshot(vec!["cpu"], vec!["cpu"]);
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// feature_contract_drift
// ---------------------------------------------------------------------------

#[test]
fn drift_none_when_aligned() {
    let result = feature_contract_drift(vec!["cpu", "inference"], vec!["inference", "cpu"]);
    assert!(result.is_none());
}

#[test]
fn drift_some_when_missing() {
    let result = feature_contract_drift(vec!["cpu", "kernels"], vec!["cpu"]);
    assert!(result.is_some());
    let snap = result.unwrap();
    assert_eq!(snap.missing_from_runtime, vec!["kernels".to_string()]);
}

#[test]
fn drift_some_when_extra() {
    let result = feature_contract_drift(vec!["cpu"], vec!["cpu", "wasm"]);
    assert!(result.is_some());
    let snap = result.unwrap();
    assert_eq!(snap.extra_from_runtime, vec!["wasm".to_string()]);
}

#[test]
fn drift_none_when_both_empty() {
    let result = feature_contract_drift(std::iter::empty::<&str>(), std::iter::empty::<&str>());
    assert!(result.is_none());
}

#[test]
fn drift_with_many_features() {
    let policy = vec!["cpu", "inference", "kernels", "tokenizers", "quantization"];
    let runtime = vec!["cpu", "inference", "kernels", "tokenizers", "quantization"];
    let result = feature_contract_drift(policy, runtime);
    assert!(result.is_none());
}

#[test]
fn drift_large_mismatch() {
    let policy = vec!["cpu", "inference", "kernels"];
    let runtime = vec!["gpu", "cuda", "metal"];
    let result = feature_contract_drift(policy, runtime);
    assert!(result.is_some());
    let snap = result.unwrap();
    assert_eq!(snap.missing_from_runtime.len(), 3);
    assert_eq!(snap.extra_from_runtime.len(), 3);
}
