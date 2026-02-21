//! Interoperability façade for test policy + BDD grid + runtime feature flags.
//!
//! This crate intentionally keeps the compatibility surface for test configuration
//! policy resolution in one place. It combines:
//! - Grid/profile resolution and validation (`bitnet_testing_policy_kit`).
//! - Runtime feature-flag discovery (`bitnet_runtime_feature_flags`).
//!
//! The goal is to make BDD + feature-flag interactions easier to consume
//! across crates while keeping the API surface stable.

#![deny(unused_must_use)]

pub use bitnet_testing_policy_contract::{
    FeatureContractSnapshot, PolicyContract, active_runtime_features, drift_check,
    feature_contract_snapshot, runtime_feature_labels, runtime_feature_line,
};
pub use bitnet_testing_policy_kit::{
    ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature, ComparisonToleranceProfile,
    ConfigurationContext, CrossValidationProfile, EnvironmentType, ExecutionEnvironment,
    FeatureSet, FixtureProfile, PlatformSettings, QualityRequirements, ReportFormat,
    ReportingProfile, ResourceConstraints, ScenarioConfigManager, ScenarioType, TestConfigProfile,
    TestingScenario, TimeConstraints, active_features, active_profile, active_profile_for,
    active_profile_summary, active_profile_violation_labels, canonical_grid, feature_labels,
    feature_line, from_grid_environment, from_grid_scenario, to_grid_environment, to_grid_scenario,
    validate_active_profile, validate_active_profile_for,
    validate_active_profile_for as validate_explicit_profile, validate_profile_for_context,
};

/// Stable alias retained for readability in test-facing imports.
pub type Environment = ExecutionEnvironment;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_is_consistent_by_default() {
        let snapshot = feature_contract_snapshot();
        assert!(!snapshot.policy_features.is_empty() || snapshot.runtime_features.is_empty());
    }

    #[test]
    fn drift_check_matches_consistency() {
        match drift_check() {
            Some(snapshot) => assert!(!snapshot.is_consistent()),
            None => assert!(feature_contract_snapshot().is_consistent()),
        }
    }
}
