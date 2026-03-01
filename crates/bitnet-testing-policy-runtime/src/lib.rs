//! Runtime-oriented compatibility façade for policy-grid and feature-flag orchestration.
//!
//! This crate keeps the interop API ergonomic for test/runtime consumers while
//! adding a small, stable policy state object for diagnostics and interoperability.

#![deny(unused_must_use)]

pub use bitnet_testing_policy_interop::{
    ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature, ComparisonToleranceProfile,
    ConfigurationContext, CrossValidationProfile, EnvironmentType, ExecutionEnvironment,
    FeatureContractSnapshot, FeatureSet, FixtureProfile, PlatformSettings, PolicyContract,
    QualityRequirements, ReportFormat, ReportingProfile, ResourceConstraints,
    ScenarioConfigManager, ScenarioType, TestConfigProfile, TestingScenario, TimeConstraints,
    active_features, active_profile, active_profile_for, active_profile_summary,
    active_profile_violation_labels, active_runtime_features, canonical_grid, drift_check,
    feature_contract_snapshot, feature_labels, feature_line, from_grid_environment,
    from_grid_scenario, runtime_feature_labels, runtime_feature_line, to_grid_environment,
    to_grid_scenario, validate_active_profile, validate_active_profile_for,
    validate_active_profile_for as validate_explicit_profile, validate_profile_for_context,
};

pub use bitnet_testing_policy_runtime_core::{
    Environment, RuntimePolicyState, context_from_environment, detect_runtime_state,
    resolve_runtime_profile,
};
