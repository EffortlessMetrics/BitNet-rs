//! Compatibility shim over [`bitnet_feature_matrix`] for runtime and tooling.
//!
//! This crate keeps the existing `bitnet-feature-contract` public API stable for
//! consumers while the source-of-truth feature-matrix behavior now lives in
//! `bitnet-feature-matrix`.

pub use bitnet_feature_matrix::{
    active_features, active_profile, active_profile_for, active_profile_summary, active_profile_violation_labels,
    canonical_grid, feature_labels, feature_line, validate_active_profile,
    validate_active_profile_for, ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature,
    ExecutionEnvironment, FeatureSet, TestingScenario,
};
