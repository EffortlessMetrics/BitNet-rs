#![deny(unused_must_use)]

//! Stable orchestration crate for test profile resolution.
//!
//! This crate composes:
//! - BDD-profile discovery and feature-flag validation (`bitnet-testing-profile`)
//! - Scenario/environment policy profiles (`bitnet-testing-scenarios-core`)
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

pub use bitnet_testing_scenarios_core::{
    ComparisonToleranceProfile, ConfigurationContext, CrossValidationProfile, EnvironmentType,
    FixtureProfile, PlatformSettings, QualityRequirements, ReportFormat, ReportingProfile,
    ResourceConstraints, ScenarioConfigManager, ScenarioType, TestConfigProfile, TimeConstraints,
};

/// Snapshot of policy+grid compatibility + merged policy config.
#[derive(Debug, Clone)]
pub struct PolicySnapshot {
    pub context: ConfigurationContext,
    pub active_profile: ActiveProfile,
    pub scenario_config: TestConfigProfile,
    pub environment_config: TestConfigProfile,
    pub resolved_config: TestConfigProfile,
}

impl PolicySnapshot {
    /// Resolve against the active process scenario/environment.
    pub fn detect() -> Self {
        Self::from_configuration_context(ScenarioConfigManager::context_from_environment())
    }

    /// Resolve against an explicit test configuration context.
    pub fn from_configuration_context(context: ConfigurationContext) -> Self {
        let manager = ScenarioConfigManager::new();
        let active_profile = active_profile_for(context.scenario, context.environment);
        let scenario_config = manager.get_scenario_config(&context.scenario);
        let environment_config = manager.get_environment_config(&context.environment);
        let resolved_config = manager.get_context_config(&context);

        Self { context, active_profile, scenario_config, environment_config, resolved_config }
    }

    /// Resolve using a compact runtime context.
    pub fn from_active_context(context: ActiveContext) -> Self {
        Self::from_configuration_context(ConfigurationContext {
            scenario: context.scenario,
            environment: context.environment,
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        })
    }

    /// True when a matching grid cell exists and feature constraints pass.
    pub fn is_compatible(&self) -> bool {
        match self.active_profile.cell {
            Some(_) => self
                .violations()
                .is_some_and(|(missing, forbidden)| missing.is_empty() && forbidden.is_empty()),
            None => false,
        }
    }

    /// Resolve violations for the active profile, if a grid row exists.
    pub fn violations(&self) -> Option<(FeatureSet, FeatureSet)> {
        self.active_profile.cell.map(|_| self.active_profile.violations())
    }

    /// Human-readable profile summary for diagnostics.
    pub fn summary(&self) -> String {
        match self.active_profile.cell {
            Some(cell) => format!(
                "scenario={}/environment={},required={},optional={},forbidden={},active={}",
                self.active_profile.scenario,
                self.active_profile.environment,
                cell.required_features.labels().join("+"),
                cell.optional_features.labels().join("+"),
                cell.forbidden_features.labels().join("+"),
                self.active_profile.features.labels().join("+"),
            ),
            None => format!(
                "scenario={}/environment={} (no matching grid cell)",
                self.active_profile.scenario, self.active_profile.environment
            ),
        }
    }
}

/// Build a policy snapshot for a context derived from runtime env vars.
pub fn snapshot_from_env() -> PolicySnapshot {
    PolicySnapshot::detect()
}

/// Resolve a configuration profile for a single context.
pub fn resolve_context_profile(context: &ConfigurationContext) -> TestConfigProfile {
    let manager = ScenarioConfigManager::new();
    manager.get_context_config(context)
}

/// Convert a rich configuration context into an active profile context.
pub fn active_context(context: &ConfigurationContext) -> ActiveContext {
    ActiveContext { scenario: context.scenario, environment: context.environment }
}

/// Resolve profile violations for a context that carries optional test policy metadata.
pub fn validate_context(context: &ConfigurationContext) -> Option<(FeatureSet, FeatureSet)> {
    validate_profile_for_context(active_context(context))
}

/// Re-exported helper retained for parity with older validation adapters.
pub fn validate_explicit_profile(
    scenario: TestingScenario,
    environment: ExecutionEnvironment,
) -> Option<(FeatureSet, FeatureSet)> {
    validate_active_profile_for(scenario, environment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_snapshot_for_env_is_populated() {
        let snapshot = snapshot_from_env();
        assert!(
            snapshot.context.scenario == TestingScenario::Unit
                || !snapshot.context.scenario.to_string().is_empty()
        );
    }

    #[test]
    fn resolve_context_profile_roundtrips() {
        let context = ConfigurationContext::default();
        let resolved = resolve_context_profile(&context);
        assert!(resolved.max_parallel_tests > 0);
        assert!(!resolved.reporting.formats.is_empty());
    }

    #[test]
    fn validate_explicit_profile_is_callable() {
        assert!(
            validate_explicit_profile(TestingScenario::Unit, ExecutionEnvironment::Local).is_some()
        );
    }
}
