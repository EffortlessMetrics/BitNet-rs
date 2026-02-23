//! Stable contract boundary for policy resolution and runtime feature flag drift checks.
//!
//! This crate is intentionally small and SRP-focused:
//! - It publishes a canonical policy snapshot shape (`PolicyContract`)
//! - It publishes a runtime-policy alignment snapshot (`FeatureContractSnapshot`)
//! - It centralizes feature drift checks used by tests, CLI, and tooling
//! - It keeps both policy-grid and runtime feature-flag semantics in one place

#![deny(unused_must_use)]

pub use bitnet_feature_contract::{
    FeatureContractSnapshot, feature_contract_drift as policy_runtime_feature_drift,
    feature_contract_snapshot as policy_runtime_feature_snapshot,
};
pub use bitnet_runtime_feature_flags::{
    active_features as active_runtime_features, feature_labels as runtime_feature_labels,
    feature_line as runtime_feature_line,
};
pub use bitnet_testing_policy_core::{
    ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature, ConfigurationContext,
    ExecutionEnvironment, FeatureSet, PlatformSettings, PolicySnapshot, ReportingProfile,
    ScenarioConfigManager, ScenarioType, TestConfigProfile, TestingScenario, active_features,
    active_profile, active_profile_for, active_profile_summary, active_profile_violation_labels,
    canonical_grid, feature_labels, feature_line, validate_active_profile,
    validate_active_profile_for, validate_context, validate_profile_for_context,
};

/// Canonical policy contract for a profile context.
#[derive(Debug, Clone)]
pub struct PolicyContract {
    /// Resolved policy profile for the active process context.
    pub snapshot: PolicySnapshot,
}

impl PolicyContract {
    /// Build policy contract from active process context.
    pub fn detect() -> Self {
        Self { snapshot: PolicySnapshot::detect() }
    }
}

/// Return normalized policy/runtime feature labels and drift.
pub fn feature_contract_snapshot() -> FeatureContractSnapshot {
    policy_runtime_feature_snapshot(active_features().labels(), active_runtime_features().labels())
}

/// Return feature drift only when policy/runtime features disagree.
pub fn drift_check() -> Option<FeatureContractSnapshot> {
    policy_runtime_feature_drift(active_features().labels(), active_runtime_features().labels())
}
