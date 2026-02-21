//! Shared runtime profile helpers for CLI/tests/tooling.
//!
//! The `bitnet-runtime-profile` crate intentionally keeps a lightweight façade API.
//! Implementation is now centralized in `bitnet_feature_matrix` to lock the profile
//! semantics and keep interoperability straightforward across crates.

pub use bitnet_feature_matrix::{
    active_features, active_profile, active_profile_for, active_profile_summary, active_profile_violation_labels,
    canonical_grid, feature_labels, feature_line, validate_active_profile,
    validate_active_profile_for, ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature,
    ExecutionEnvironment, FeatureSet, TestingScenario,
};
