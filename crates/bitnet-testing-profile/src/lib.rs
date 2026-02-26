//! Testing façade for BDD-grid and feature-flag profile contracts.
//!
//! This crate is intentionally narrow and stable:
//! - re-exports canonical profile types from `bitnet-runtime-profile`
//! - provides compatibility aliases used by the test harness
//! - exposes helper functions that keep conversions and validation calls explicit

pub use bitnet_runtime_profile::{
    ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature, ExecutionEnvironment,
    FeatureSet, TestingScenario, active_features, active_profile, active_profile_for,
    active_profile_summary, active_profile_violation_labels, canonical_grid, feature_labels,
    feature_line, validate_active_profile, validate_active_profile_for,
};

/// Compatibility alias for readability in test-facing code.
pub type EnvironmentType = ExecutionEnvironment;

/// Compatibility alias for readability in test-facing code.
pub type ScenarioType = TestingScenario;

/// Canonical no-op conversion kept for callers that still use the "grid" naming.
pub const fn to_grid_scenario(scenario: TestingScenario) -> TestingScenario {
    scenario
}

/// Canonical no-op conversion kept for callers that still use the "grid" naming.
pub const fn to_grid_environment(environment: ExecutionEnvironment) -> ExecutionEnvironment {
    environment
}

/// Canonical no-op conversion kept for callers that still use the "grid" naming.
pub const fn from_grid_scenario(scenario: TestingScenario) -> TestingScenario {
    scenario
}

/// Canonical no-op conversion kept for callers that still use the "grid" naming.
pub const fn from_grid_environment(environment: ExecutionEnvironment) -> ExecutionEnvironment {
    environment
}

/// Validate active profile against an explicit scenario/environment pair.
pub fn validate_explicit_profile(
    scenario: TestingScenario,
    environment: ExecutionEnvironment,
) -> Option<(FeatureSet, FeatureSet)> {
    validate_active_profile_for(scenario, environment)
}

/// Validate active profile against an explicit context.
pub fn validate_profile_for_context(context: ActiveContext) -> Option<(FeatureSet, FeatureSet)> {
    validate_active_profile_for(context.scenario, context.environment)
}

#[cfg(test)]
mod tests {
    use super::{
        ActiveContext, ExecutionEnvironment, TestingScenario, from_grid_environment,
        from_grid_scenario, to_grid_environment, to_grid_scenario, validate_active_profile_for,
        validate_explicit_profile, validate_profile_for_context,
    };

    #[test]
    fn conversion_helpers_roundtrip() {
        let scenario = TestingScenario::Unit;
        let environment = ExecutionEnvironment::Local;
        assert_eq!(to_grid_scenario(scenario), scenario);
        assert_eq!(to_grid_environment(environment), environment);
        assert_eq!(from_grid_scenario(scenario), scenario);
        assert_eq!(from_grid_environment(environment), environment);
    }

    #[test]
    fn validate_helpers_use_runtime_row_lookup() {
        assert!(
            validate_explicit_profile(TestingScenario::Unit, ExecutionEnvironment::Local).is_some()
        );
        // Construct context directly to avoid reading CI env vars (which would
        // change environment to Ci on GitHub Actions, and Unit/Ci is not a
        // supported BDD grid cell).
        let context = ActiveContext {
            scenario: TestingScenario::Unit,
            environment: ExecutionEnvironment::Local,
        };
        assert!(validate_profile_for_context(context).is_some());
        assert!(
            validate_active_profile_for(TestingScenario::Unit, ExecutionEnvironment::Local)
                .is_some()
        );
    }
}
