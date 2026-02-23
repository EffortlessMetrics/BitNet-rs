//! Interoperability façade for test policy + BDD-grid contracts.
//!
//! This crate intentionally keeps the BDD-grid and feature-gate contracts in one
//! small dependency surface for tests, tooling, and downstream users.
#![deny(unused_must_use)]

pub use bitnet_testing_policy::{
    ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature, ComparisonToleranceProfile,
    ConfigurationContext, CrossValidationProfile, ExecutionEnvironment, FeatureSet, FixtureProfile,
    PlatformSettings, QualityRequirements, ReportFormat, ReportingProfile, ResourceConstraints,
    ScenarioConfigManager, ScenarioType, TestConfigProfile, TestingScenario, TimeConstraints,
    active_features, active_profile, active_profile_for,
    active_profile_for as active_profile_for_context, active_profile_summary,
    active_profile_violation_labels, canonical_grid, feature_labels, feature_line,
    from_grid_environment, from_grid_scenario, to_grid_environment, to_grid_scenario,
    validate_active_profile, validate_active_profile_for, validate_context,
    validate_explicit_profile, validate_profile_for_context,
};

/// Compatibility aliases used by the existing test-facing API.
pub type EnvironmentType = ExecutionEnvironment;
pub type GridScenario = TestingScenario;
pub type GridEnvironment = ExecutionEnvironment;

/// Return the feature set currently active for the process context.
pub fn active_feature_labels() -> Vec<String> {
    active_features().labels()
}

/// Resolve the active policy profile using the default scenario/environment
/// inferred from environment variables.
pub fn active_profile_from_environment() -> ActiveProfile {
    let context = ScenarioConfigManager::context_from_environment();
    active_profile_for(context.scenario, context.environment)
}

/// Validate the active policy profile using environment-derived scenario context.
pub fn validate_active_profile_from_environment() -> Option<(FeatureSet, FeatureSet)> {
    let context = ScenarioConfigManager::context_from_environment();
    validate_profile_for_context(ActiveContext {
        scenario: context.scenario,
        environment: context.environment,
    })
}

/// Resolve active profile and scenario context together.
pub fn profile_snapshot() -> (ActiveProfile, ConfigurationContext, Option<(FeatureSet, FeatureSet)>)
{
    let context = ScenarioConfigManager::context_from_environment();
    let profile = active_profile_for(context.scenario, context.environment);
    let violations = validate_profile_for_context(ActiveContext {
        scenario: context.scenario,
        environment: context.environment,
    });
    (profile, context, violations)
}

/// Return merged scenario + environment profile config for diagnostics and orchestration.
pub fn resolve_active_profile() -> TestConfigProfile {
    ScenarioConfigManager::new()
        .get_context_config(&ScenarioConfigManager::context_from_environment())
}

/// Backward-compatible wrapper kept for feature-gated callers that still expect this
/// naming from older policy façades.
pub fn validate_profile() -> Option<(FeatureSet, FeatureSet)> {
    validate_active_profile()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_profile_helpers_return_values() {
        let (profile, context, violations) = profile_snapshot();
        assert_eq!(profile.scenario, context.scenario);
        assert_eq!(profile.environment, context.environment);
        if let Some((missing, forbidden)) = violations {
            assert_eq!(profile.violations(), (missing, forbidden));
        }
    }

    #[test]
    fn active_profile_from_environment_matches_context() {
        let profile = active_profile_from_environment();
        let context = ScenarioConfigManager::context_from_environment();
        assert_eq!(profile.scenario, context.scenario);
        assert_eq!(profile.environment, context.environment);
    }
}
