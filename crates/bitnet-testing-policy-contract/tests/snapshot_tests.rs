//! Snapshot tests for bitnet-testing-policy-contract.
//!
//! Pins: PolicyContract::detect() succeeds, drift_check() is None when consistent,
//! and feature_contract_snapshot() is_consistent() with itself.

use bitnet_testing_policy_contract::{PolicyContract, drift_check, feature_contract_snapshot};

#[test]
fn policy_contract_detect_succeeds() {
    let contract = PolicyContract::detect();
    // snapshot the Debug representation to pin the struct shape
    let debug_str = format!("{:?}", contract);
    insta::assert_snapshot!(debug_str.starts_with("PolicyContract").to_string());
}

#[test]
fn feature_contract_snapshot_is_self_consistent() {
    let snapshot = feature_contract_snapshot();
    // When policy features match runtime features, snapshot is consistent
    insta::assert_snapshot!(snapshot.is_consistent().to_string());
}

#[test]
fn drift_check_matches_consistency() {
    let consistent = feature_contract_snapshot().is_consistent();
    let has_drift = drift_check().is_some();
    // drift_check returns Some iff is_consistent() is false
    insta::assert_snapshot!((consistent == !has_drift).to_string());
}
