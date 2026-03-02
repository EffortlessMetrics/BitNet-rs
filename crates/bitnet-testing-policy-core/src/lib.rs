#![deny(unused_must_use)]

//! Stable orchestration crate for test profile resolution.
//!
//! This crate composes:
//! - BDD-profile discovery and feature-flag validation (`bitnet-testing-profile`)
//! - Scenario/environment policy profiles (`bitnet-testing-scenarios-core`)
//! - Policy snapshot resolution (`bitnet-testing-policy-snapshot-core`)
//!
//! The result is a single place to reason about how active compile-time features,
//! grid constraints, and scenario configuration merge together.

use bitnet_testing_profile as profile;

pub use profile::{
    ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature, ExecutionEnvironment,
    FeatureSet, TestingScenario, active_features, active_profile, active_profile_for,
    active_profile_summary, active_profile_violation_labels, canonical_grid, feature_labels,
    feature_line, from_grid_environment, from_grid_scenario, to_grid_environment, to_grid_scenario,
    validate_active_profile, validate_active_profile_for, validate_profile_for_context,
};

pub use bitnet_testing_policy_snapshot_core::{
    PolicySnapshot, active_context, resolve_context_profile, snapshot_from_env, validate_context,
    validate_explicit_profile,
};

pub use bitnet_testing_scenarios_core::{
    ComparisonToleranceProfile, ConfigurationContext, CrossValidationProfile, EnvironmentType,
    FixtureProfile, PlatformSettings, QualityRequirements, ReportFormat, ReportingProfile,
    ResourceConstraints, ScenarioConfigManager, ScenarioType, TestConfigProfile, TimeConstraints,
};
