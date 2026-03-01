//! Edge-case tests for bitnet-testing-policy-contract PolicyContract and helpers.

use bitnet_testing_policy_contract::{PolicyContract, drift_check, feature_contract_snapshot};

// ---------------------------------------------------------------------------
// PolicyContract: detect
// ---------------------------------------------------------------------------

#[test]
fn policy_contract_detect_returns_valid_snapshot() {
    let contract = PolicyContract::detect();
    let _ = contract.snapshot.summary();
}

#[test]
fn policy_contract_detect_has_context() {
    let contract = PolicyContract::detect();
    assert!(!contract.snapshot.context.scenario.to_string().is_empty());
    assert!(!contract.snapshot.context.environment.to_string().is_empty());
}

#[test]
fn policy_contract_detect_has_profile() {
    let contract = PolicyContract::detect();
    assert!(!contract.snapshot.active_profile.features.labels().is_empty());
}

// ---------------------------------------------------------------------------
// PolicyContract: Debug + Clone
// ---------------------------------------------------------------------------

#[test]
fn policy_contract_debug_not_empty() {
    let contract = PolicyContract::detect();
    let d = format!("{:?}", contract);
    assert!(d.contains("PolicyContract"));
}

#[test]
fn policy_contract_clone() {
    let contract = PolicyContract::detect();
    let contract2 = contract.clone();
    assert_eq!(contract.snapshot.summary(), contract2.snapshot.summary(),);
}

// ---------------------------------------------------------------------------
// PolicyContract: snapshot fields
// ---------------------------------------------------------------------------

#[test]
fn policy_contract_snapshot_is_compatible_callable() {
    let contract = PolicyContract::detect();
    let _ = contract.snapshot.is_compatible();
}

#[test]
fn policy_contract_snapshot_violations_callable() {
    let contract = PolicyContract::detect();
    let _ = contract.snapshot.violations();
}

// ---------------------------------------------------------------------------
// feature_contract_snapshot helper
// ---------------------------------------------------------------------------

#[test]
fn feature_contract_snapshot_returns_snapshot() {
    let snap = feature_contract_snapshot();
    // Should have policy and runtime features populated
    let _ = snap.is_consistent();
}

#[test]
fn feature_contract_snapshot_has_policy_features() {
    let snap = feature_contract_snapshot();
    // Policy features come from the BDD grid active profile
    assert!(!snap.policy_features.is_empty());
}

#[test]
fn feature_contract_snapshot_has_runtime_features() {
    let snap = feature_contract_snapshot();
    // Runtime features come from compile-time feature flags
    assert!(!snap.runtime_features.is_empty());
}

// ---------------------------------------------------------------------------
// drift_check helper
// ---------------------------------------------------------------------------

#[test]
fn drift_check_callable() {
    // May be None (aligned) or Some (drift)
    let _ = drift_check();
}

#[test]
fn drift_check_consistency() {
    let snap = feature_contract_snapshot();
    let drift = drift_check();
    // If snapshot is consistent, drift should be None; otherwise Some
    if snap.is_consistent() {
        assert!(drift.is_none());
    } else {
        assert!(drift.is_some());
    }
}

// ---------------------------------------------------------------------------
// Re-exported types from policy contract
// ---------------------------------------------------------------------------

#[test]
fn reexported_active_features_callable() {
    use bitnet_testing_policy_contract::active_features;
    let features = active_features();
    let _ = features.labels();
}

#[test]
fn reexported_active_runtime_features_callable() {
    use bitnet_testing_policy_contract::active_runtime_features;
    let features = active_runtime_features();
    let _ = features.labels();
}

#[test]
fn reexported_feature_line_callable() {
    use bitnet_testing_policy_contract::feature_line;
    let line = feature_line();
    assert!(!line.is_empty());
}

#[test]
fn reexported_runtime_feature_line_callable() {
    use bitnet_testing_policy_contract::runtime_feature_line;
    let line = runtime_feature_line();
    assert!(!line.is_empty());
}

#[test]
fn reexported_canonical_grid_callable() {
    use bitnet_testing_policy_contract::canonical_grid;
    let grid = canonical_grid();
    assert!(!grid.rows().is_empty());
}

#[test]
fn reexported_active_profile_summary_callable() {
    use bitnet_testing_policy_contract::active_profile_summary;
    let summary = active_profile_summary();
    assert!(!summary.is_empty());
}
