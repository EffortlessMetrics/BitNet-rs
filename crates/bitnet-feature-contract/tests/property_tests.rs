//! Property-based tests for `bitnet-feature-contract`.
//!
//! Tests the set-diff semantics of `feature_contract_snapshot` and the
//! consistency predicate of `FeatureContractSnapshot`.

use bitnet_feature_contract::{feature_contract_drift, feature_contract_snapshot};
use proptest::prelude::*;

// ── Strategy ─────────────────────────────────────────────────────────────────

/// A small fixed-size vocabulary of feature name strings.
const FEATURE_VOCAB: &[&str] = &[
    "cpu", "gpu", "cuda", "inference", "kernels", "tokenizers",
    "tracing", "crossval", "ffi", "fixtures",
];

fn arb_feature_set() -> impl Strategy<Value = Vec<String>> {
    proptest::sample::subsequence(FEATURE_VOCAB, 0..=FEATURE_VOCAB.len())
        .prop_map(|v| v.into_iter().map(|s| s.to_string()).collect())
}

// ── Property tests ───────────────────────────────────────────────────────────

proptest! {
    /// Identical policy and runtime lists are always consistent.
    #[test]
    fn identical_lists_are_consistent(features in arb_feature_set()) {
        let snap = feature_contract_snapshot(features.clone(), features.clone());
        prop_assert!(snap.is_consistent(),
            "identical lists should be consistent; missing={:?}, extra={:?}",
            snap.missing_from_runtime, snap.extra_from_runtime);
    }

    /// When policy ⊆ runtime, there are no missing features.
    #[test]
    fn subset_policy_has_no_missing(
        all in arb_feature_set(),
        extra in arb_feature_set(),
    ) {
        // runtime = all + extra; policy = all  →  missing should be empty
        let mut runtime = all.clone();
        for f in &extra {
            if !runtime.contains(f) {
                runtime.push(f.clone());
            }
        }
        let snap = feature_contract_snapshot(all, runtime);
        prop_assert!(snap.missing_from_runtime.is_empty(),
            "policy ⊆ runtime should have no missing; got {:?}", snap.missing_from_runtime);
    }

    /// When runtime ⊆ policy, there are no extra features.
    #[test]
    fn subset_runtime_has_no_extra(
        all in arb_feature_set(),
        extra in arb_feature_set(),
    ) {
        // policy = all + extra; runtime = all  →  extra_from_runtime should be empty
        let mut policy = all.clone();
        for f in &extra {
            if !policy.contains(f) {
                policy.push(f.clone());
            }
        }
        let snap = feature_contract_snapshot(policy, all);
        prop_assert!(snap.extra_from_runtime.is_empty(),
            "runtime ⊆ policy should have no extra; got {:?}", snap.extra_from_runtime);
    }

    /// `missing + extra` is consistent with set-symmetric-difference.
    #[test]
    fn drift_agrees_with_symmetric_difference(
        policy in arb_feature_set(),
        runtime in arb_feature_set(),
    ) {
        use std::collections::BTreeSet;
        let snap = feature_contract_snapshot(policy.iter(), runtime.iter());

        let p: BTreeSet<_> = policy.iter().collect();
        let r: BTreeSet<_> = runtime.iter().collect();

        let expected_missing: BTreeSet<&String> = p.difference(&r).copied().collect();
        let expected_extra: BTreeSet<&String>   = r.difference(&p).copied().collect();

        let actual_missing: BTreeSet<&String> = snap.missing_from_runtime.iter().collect();
        let actual_extra: BTreeSet<&String>   = snap.extra_from_runtime.iter().collect();

        prop_assert_eq!(actual_missing, expected_missing);
        prop_assert_eq!(actual_extra,   expected_extra);
    }

    /// `feature_contract_drift` returns None iff policy == runtime (as sets).
    #[test]
    fn drift_returns_none_iff_consistent(
        policy in arb_feature_set(),
        runtime in arb_feature_set(),
    ) {
        use std::collections::BTreeSet;
        let drift = feature_contract_drift(policy.iter(), runtime.iter());
        let p: BTreeSet<_> = policy.iter().collect();
        let r: BTreeSet<_> = runtime.iter().collect();
        let consistent = p == r;
        if consistent {
            prop_assert!(drift.is_none(), "expected None for equal sets");
        } else {
            prop_assert!(drift.is_some(), "expected Some for differing sets");
        }
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[test]
fn empty_policy_and_runtime_is_consistent() {
    let snap = feature_contract_snapshot([] as [&str; 0], [] as [&str; 0]);
    assert!(snap.is_consistent());
}

#[test]
fn ordering_does_not_affect_consistency() {
    let a = feature_contract_snapshot(["cpu", "gpu"], ["gpu", "cpu"]);
    let b = feature_contract_snapshot(["gpu", "cpu"], ["cpu", "gpu"]);
    assert_eq!(a.is_consistent(), b.is_consistent());
    assert!(a.is_consistent());
}

#[test]
fn single_extra_feature_is_not_consistent() {
    let snap = feature_contract_snapshot(["cpu"], ["cpu", "gpu"]);
    assert!(!snap.is_consistent());
    assert!(snap.extra_from_runtime.contains(&"gpu".to_string()));
}
