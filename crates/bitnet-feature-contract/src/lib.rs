//! Compatibility shim over [`bitnet_feature_matrix`] for runtime and tooling.
//!
//! This crate keeps the existing `bitnet-feature-contract` public API stable for
//! consumers while the source-of-truth feature-matrix behavior now lives in
//! `bitnet-feature-matrix`.

use std::collections::BTreeSet;

pub use bitnet_feature_matrix::{
    ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature, ExecutionEnvironment,
    FeatureSet, TestingScenario, active_features, active_profile, active_profile_for,
    active_profile_summary, active_profile_violation_labels, canonical_grid, feature_labels,
    feature_line, validate_active_profile, validate_active_profile_for,
};

/// Snapshot of policy-vs-runtime feature alignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureContractSnapshot {
    /// Features required/expected by policy profile resolution.
    pub policy_features: Vec<String>,
    /// Features active in the runtime feature stack.
    pub runtime_features: Vec<String>,
    /// Features expected by policy but not active at runtime.
    pub missing_from_runtime: Vec<String>,
    /// Features active at runtime but not expected by policy.
    pub extra_from_runtime: Vec<String>,
}

impl FeatureContractSnapshot {
    /// Returns true when policy/runtime feature views are aligned.
    pub fn is_consistent(&self) -> bool {
        self.missing_from_runtime.is_empty() && self.extra_from_runtime.is_empty()
    }
}

/// Build a normalized feature-contract comparison from explicit feature lists.
pub fn feature_contract_snapshot<I, J, P, R>(
    policy_features: I,
    runtime_features: J,
) -> FeatureContractSnapshot
where
    I: IntoIterator<Item = P>,
    J: IntoIterator<Item = R>,
    P: AsRef<str>,
    R: AsRef<str>,
{
    let policy =
        policy_features.into_iter().map(|value| value.as_ref().to_string()).collect::<Vec<_>>();
    let runtime =
        runtime_features.into_iter().map(|value| value.as_ref().to_string()).collect::<Vec<_>>();

    let policy_set = policy.iter().cloned().collect::<BTreeSet<_>>();
    let runtime_set = runtime.iter().cloned().collect::<BTreeSet<_>>();

    let missing_from_runtime = policy_set.difference(&runtime_set).cloned().collect();
    let extra_from_runtime = runtime_set.difference(&policy_set).cloned().collect();

    FeatureContractSnapshot {
        policy_features: policy,
        runtime_features: runtime,
        missing_from_runtime,
        extra_from_runtime,
    }
}

/// Return a drift snapshot only when policy/runtime feature views disagree.
pub fn feature_contract_drift<I, J, P, R>(
    policy_features: I,
    runtime_features: J,
) -> Option<FeatureContractSnapshot>
where
    I: IntoIterator<Item = P>,
    J: IntoIterator<Item = R>,
    P: AsRef<str>,
    R: AsRef<str>,
{
    let snapshot = feature_contract_snapshot(policy_features, runtime_features);
    if snapshot.is_consistent() { None } else { Some(snapshot) }
}

#[cfg(test)]
mod tests {
    use super::{FeatureContractSnapshot, feature_contract_drift, feature_contract_snapshot};

    #[test]
    fn feature_contract_snapshot_detects_missing_and_extra_features() {
        let snapshot =
            feature_contract_snapshot(vec!["inference", "kernels"], vec!["inference", "gpu"]);

        assert_eq!(
            snapshot,
            FeatureContractSnapshot {
                policy_features: vec!["inference".to_string(), "kernels".to_string()],
                runtime_features: vec!["inference".to_string(), "gpu".to_string()],
                missing_from_runtime: vec!["kernels".to_string()],
                extra_from_runtime: vec!["gpu".to_string()],
            }
        );
    }

    #[test]
    fn feature_contract_drift_returns_none_when_aligned() {
        let none = feature_contract_drift(vec!["inference", "gpu"], vec!["gpu", "inference"]);
        assert!(none.is_none());
    }
}
