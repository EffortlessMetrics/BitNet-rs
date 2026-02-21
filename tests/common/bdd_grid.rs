//! BDD grid + feature-flag interop helpers for test configuration.
//!
//! This module delegates the canonical profile semantics to
//! `bitnet_runtime_profile`, then keeps the historical tests-side API shape
//! stable for callers that still depend on the `ScenarioConfigManager` enums.

use crate::config_scenarios::{EnvironmentType, ScenarioConfigManager, TestingScenario};
use bitnet_runtime_profile as feature_contract;
use bitnet_runtime_profile as bdd_grid;

pub use bitnet_runtime_profile::{
    validate_active_profile_for,
    active_profile_for,
    BddCell,
    BddGrid,
    BitnetFeature,
    ExecutionEnvironment,
    FeatureSet,
};

/// Canonical profile type from `bitnet_runtime_profile` preserved behind existing name.
pub type ActiveProfile = feature_contract::ActiveProfile;

/// Canonical curated grid.
pub const fn canonical_grid() -> BddGrid {
    feature_contract::canonical_grid()
}

/// Map local scenario enum to grid scenario.
pub const fn to_grid_scenario(scenario: TestingScenario) -> bdd_grid::TestingScenario {
    scenario
}

/// Map local environment enum to grid environment.
pub const fn to_grid_environment(
    environment: EnvironmentType,
) -> bdd_grid::ExecutionEnvironment {
    environment
}

/// Reconstruct legacy scenario enum from grid scenario.
pub const fn from_grid_scenario(scenario: bdd_grid::TestingScenario) -> TestingScenario {
    scenario
}

/// Reconstruct legacy environment enum from grid environment.
pub const fn from_grid_environment(
    environment: bdd_grid::ExecutionEnvironment,
) -> EnvironmentType {
    environment
}

/// Active feature set built from build-time feature flags.
pub fn active_features() -> FeatureSet {
    feature_contract::active_features()
}

/// Active profile inferred from environment + runtime context.
pub fn active_profile() -> ActiveProfile {
    feature_contract::active_profile()
}

/// Active profile violation labels `(missing, forbidden)` for the current process context.
pub fn active_profile_violation_labels() -> Option<(Vec<String>, Vec<String>)> {
    feature_contract::active_profile_violation_labels()
}

/// Short one-line active profile summary for diagnostics.
pub fn active_profile_summary() -> String {
    feature_contract::active_profile_summary()
}

/// Validate active profile against canonical grid row (if one exists).
pub fn validate_active_profile() -> Option<(FeatureSet, FeatureSet)> {
    let context = ScenarioConfigManager::context_from_environment();
    feature_contract::validate_active_profile_for(
        to_grid_scenario(context.scenario),
        to_grid_environment(context.environment),
    )
}

/// Validate an explicit scenario/environment with the active feature set.
pub fn validate_explicit_profile(
    scenario: bdd_grid::TestingScenario,
    environment: bdd_grid::ExecutionEnvironment,
) -> Option<(FeatureSet, FeatureSet)> {
    feature_contract::validate_active_profile_for(scenario, environment)
}
