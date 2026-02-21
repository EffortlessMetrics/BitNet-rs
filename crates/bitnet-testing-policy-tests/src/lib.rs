//! Test-oriented policy compatibility façade.
//!
//! This microcrate composes BDD-grid policy resolution with runtime feature-contract
//! diagnostics into a single, stable surface for test tooling.

#![deny(unused_must_use)]

pub use bitnet_testing_policy_runtime::{
    ActiveContext, ActiveProfile, BddCell, BddGrid, BitnetFeature, ComparisonToleranceProfile,
    ConfigurationContext, CrossValidationProfile, EnvironmentType, ExecutionEnvironment,
    FeatureContractSnapshot, FeatureSet, FixtureProfile, PlatformSettings, PolicyContract,
    QualityRequirements, ReportFormat, ReportingProfile, ResourceConstraints, RuntimePolicyState,
    ScenarioConfigManager, ScenarioType, TestConfigProfile, TestingScenario, TimeConstraints,
    active_features, active_profile, active_profile_for,
    active_profile_for as active_profile_for_context, active_profile_summary,
    active_profile_violation_labels, active_runtime_features, canonical_grid,
    context_from_environment, detect_runtime_state, drift_check, feature_contract_snapshot,
    feature_labels, feature_line, from_grid_environment, from_grid_scenario,
    resolve_runtime_profile, runtime_feature_labels, runtime_feature_line, to_grid_environment,
    to_grid_scenario, validate_active_profile,
    validate_active_profile_for as validate_explicit_profile, validate_profile_for_context,
};

/// Canonical aliases for compatibility with older tests imports.
pub type GridScenario = TestingScenario;
pub type GridEnvironment = ExecutionEnvironment;

/// Stable, test-friendly policy snapshot with clear diagnostics.
#[derive(Debug, Clone)]
pub struct PolicyDiagnostics {
    inner: RuntimePolicyState,
}

impl PolicyDiagnostics {
    /// Capture diagnostics from the current process environment.
    pub fn current() -> Self {
        Self { inner: detect_runtime_state() }
    }

    /// Capture diagnostics for an explicit context.
    pub fn from_context(context: &ConfigurationContext) -> Self {
        Self { inner: RuntimePolicyState::from_context(context) }
    }

    /// Context used for this snapshot.
    pub fn context(&self) -> &ConfigurationContext {
        &self.inner.context
    }

    /// Active profile derived for the snapshot context.
    pub fn profile(&self) -> &ActiveProfile {
        &self.inner.active_profile
    }

    /// Violations for the active grid row, if one exists.
    pub fn violations(&self) -> Option<&(FeatureSet, FeatureSet)> {
        self.inner.violations.as_ref()
    }

    /// Resolved profile contract payload for orchestration and reporting.
    pub fn profile_config(&self) -> TestConfigProfile {
        resolve_runtime_profile(&self.inner.context)
    }

    /// Whether the active profile is grid-compatible.
    pub fn is_grid_compatible(&self) -> bool {
        self.inner
            .violations
            .as_ref()
            .is_some_and(|(missing, forbidden)| missing.is_empty() && forbidden.is_empty())
    }

    /// Whether runtime feature labels are contract-consistent.
    pub fn is_feature_contract_consistent(&self) -> bool {
        self.inner.is_feature_contract_aligned()
    }

    /// Compact snapshot summary for logs and diagnostics.
    pub fn summary(&self) -> String {
        self.inner.summary()
    }
}

/// Validate active profile for an explicit configuration context.
pub fn validate_active_profile_from_context(
    context: &ConfigurationContext,
) -> Option<(FeatureSet, FeatureSet)> {
    validate_profile_for_context(ActiveContext {
        scenario: context.scenario,
        environment: context.environment,
    })
}

/// Convenience constructor for callers that already carry a context.
#[inline]
pub fn diagnostics_for_context(context: &ConfigurationContext) -> PolicyDiagnostics {
    PolicyDiagnostics::from_context(context)
}

/// Convenience constructor for caller convenience in test harnesses.
#[inline]
pub fn diagnostics() -> PolicyDiagnostics {
    PolicyDiagnostics::current()
}

#[cfg(test)]
mod tests {
    use super::{diagnostics, validate_active_profile};

    #[test]
    fn compatibility_helpers_return_values() {
        let _ = diagnostics();
        let _ = validate_active_profile();
    }
}
