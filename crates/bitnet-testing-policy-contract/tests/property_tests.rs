use bitnet_testing_policy_contract::{
    PolicyContract, drift_check, feature_contract_snapshot, runtime_feature_labels,
    runtime_feature_line,
};
use proptest::prelude::*;

proptest! {
    /// `drift_check()` returns `None` iff `feature_contract_snapshot().is_consistent()`.
    #[test]
    fn drift_check_consistent_with_snapshot(_: ()) {
        let snapshot = feature_contract_snapshot();
        let drift = drift_check();
        prop_assert_eq!(
            drift.is_none(),
            snapshot.is_consistent(),
            "drift_check and feature_contract_snapshot disagree on consistency"
        );
    }

    /// `PolicyContract::detect()` never panics and has a non-empty snapshot summary.
    #[test]
    fn policy_contract_detect_never_panics(_: ()) {
        let _contract = PolicyContract::detect();
    }

    /// `runtime_feature_labels()` and `runtime_feature_line()` are mutually consistent.
    /// Every label must appear in the feature line.
    #[test]
    fn runtime_labels_appear_in_feature_line(_: ()) {
        let labels = runtime_feature_labels();
        let line = runtime_feature_line();
        for label in &labels {
            prop_assert!(
                line.contains(label.as_str()),
                "label {:?} absent from runtime_feature_line {:?}", label, line
            );
        }
    }

    /// `FeatureContractSnapshot::is_consistent()` holds iff missing and extra sets are empty.
    #[test]
    fn feature_contract_snapshot_consistency_iff_sets_empty(_: ()) {
        let snapshot = feature_contract_snapshot();
        let expected = snapshot.missing_from_runtime.is_empty()
            && snapshot.extra_from_runtime.is_empty();
        prop_assert_eq!(
            snapshot.is_consistent(),
            expected,
            "is_consistent() disagrees with actual set emptiness"
        );
    }
}
