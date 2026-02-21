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

/// Stable alias retained for test-facing imports.
pub type Environment = ExecutionEnvironment;

/// Resolved runtime view of policy + feature contract state.
#[derive(Debug, Clone)]
pub struct RuntimePolicyState {
    pub context: ConfigurationContext,
    pub active_profile: ActiveProfile,
    pub violations: Option<(FeatureSet, FeatureSet)>,
    pub feature_contract: FeatureContractSnapshot,
    pub feature_drift: Option<FeatureContractSnapshot>,
}

impl RuntimePolicyState {
    /// Resolve the runtime policy state from a provided policy context.
    pub fn from_context(context: &ConfigurationContext) -> Self {
        let active_context =
            ActiveContext { scenario: context.scenario, environment: context.environment };
        Self {
            context: context.clone(),
            active_profile: active_profile_for(context.scenario, context.environment),
            violations: validate_profile_for_context(active_context),
            feature_contract: feature_contract_snapshot(),
            feature_drift: drift_check(),
        }
    }

    /// Resolve the runtime policy state from environment variables.
    pub fn from_environment() -> Self {
        Self::from_context(&ScenarioConfigManager::context_from_environment())
    }

    /// Resolve and merge the full scenario/environment test configuration.
    pub fn resolved_profile_config(&self) -> TestConfigProfile {
        ScenarioConfigManager::new().get_context_config(&self.context)
    }

    /// Whether the active profile has no grid violations and resolves to an active cell.
    pub fn is_policy_compatible(&self) -> bool {
        self.violations
            .as_ref()
            .is_some_and(|(missing, forbidden)| missing.is_empty() && forbidden.is_empty())
    }

    /// Whether this state is governed by an explicit BDD-grid row.
    pub fn has_grid_cell(&self) -> bool {
        self.active_profile.cell.is_some()
    }

    /// Whether policy and runtime features are aligned.
    pub fn is_feature_contract_aligned(&self) -> bool {
        self.feature_contract.is_consistent()
    }

    /// Short human-readable summary for diagnostics.
    pub fn summary(&self) -> String {
        let compatibility = if self.is_feature_contract_aligned() {
            "feature-contract-aligned"
        } else {
            "feature-contract-drift"
        };
        let profile_summary = active_profile_summary();
        format!("{profile_summary} [{compatibility}]")
    }
}

/// Build the current runtime policy state from environment variables.
pub fn detect_runtime_state() -> RuntimePolicyState {
    RuntimePolicyState::from_environment()
}

/// Resolve a complete test configuration for a given policy context.
pub fn resolve_runtime_profile(context: &ConfigurationContext) -> TestConfigProfile {
    ScenarioConfigManager::new().get_context_config(context)
}

/// Return current context as inferred from `BITNET_*` style environment variables.
pub fn context_from_environment() -> ConfigurationContext {
    ScenarioConfigManager::context_from_environment()
}

#[cfg(test)]
mod tests {
    use super::{RuntimePolicyState, context_from_environment};

    #[test]
    fn runtime_state_can_be_detected() {
        let ctx = context_from_environment();
        let state = RuntimePolicyState::from_context(&ctx);
        assert_eq!(state.context.scenario, ctx.scenario);
        assert_eq!(state.context.environment, ctx.environment);
    }
}
