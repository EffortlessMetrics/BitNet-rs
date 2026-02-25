use bitnet_feature_contract::{FeatureContractSnapshot, feature_contract_snapshot};

#[test]
fn consistent_snapshot_when_policy_and_runtime_match() {
    let snap =
        feature_contract_snapshot(["cpu", "inference", "kernels"], ["cpu", "inference", "kernels"]);
    insta::assert_snapshot!(format!(
        "is_consistent={} missing={:?} extra={:?}",
        snap.is_consistent(),
        snap.missing_from_runtime,
        snap.extra_from_runtime
    ));
}

#[test]
fn inconsistent_snapshot_when_policy_has_extra_feature() {
    let snap = feature_contract_snapshot(["cpu", "inference", "kernels"], ["cpu"]);
    insta::assert_snapshot!(format!(
        "is_consistent={} missing={:?} extra={:?}",
        snap.is_consistent(),
        snap.missing_from_runtime,
        snap.extra_from_runtime
    ));
}

#[test]
fn inconsistent_snapshot_runtime_has_extra() {
    let snap: FeatureContractSnapshot = feature_contract_snapshot(["cpu"], ["cpu", "gpu", "cuda"]);
    insta::assert_snapshot!(format!(
        "is_consistent={} extra={:?}",
        snap.is_consistent(),
        snap.extra_from_runtime
    ));
}

#[test]
fn empty_both_sides_is_consistent() {
    let snap = feature_contract_snapshot::<[&str; 0], [&str; 0], &str, &str>([], []);
    insta::assert_snapshot!(format!("is_consistent={}", snap.is_consistent()));
}
