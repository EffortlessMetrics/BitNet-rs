//! Core runtime profile contracts shared by startup/runtime systems and tooling.
//!
//! This crate owns the canonical profile model and validation behavior.
//! Higher-level façade crates in the workspace re-export this API to keep a stable
//! downstream contract while allowing evolution of the underlying stack.

pub use bitnet_bdd_grid::BddGrid;
pub use bitnet_bdd_grid::{
    BddCell, BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario,
};
pub use bitnet_runtime_context::ActiveContext;
pub use bitnet_runtime_feature_flags::{active_features, feature_labels, feature_line};

/// Result of active-profile evaluation and row-level compatibility checks.
#[derive(Debug, Clone)]
pub struct ActiveProfile {
    /// Scenario under evaluation.
    pub scenario: TestingScenario,
    /// Environment under evaluation.
    pub environment: ExecutionEnvironment,
    /// Active features detected from compile-time feature flags.
    pub features: FeatureSet,
    /// Matching BDD cell if one is defined for this scenario/environment.
    pub cell: Option<&'static BddCell>,
}

impl ActiveProfile {
    /// Build a profile from an already-resolved runtime context.
    pub fn from_context(context: ActiveContext) -> Self {
        let features = active_features();
        Self {
            scenario: context.scenario,
            environment: context.environment,
            features,
            cell: canonical_grid().find(context.scenario, context.environment),
        }
    }

    /// True when the active feature set satisfies required/forbidden constraints.
    pub fn is_supported(&self) -> bool {
        match self.cell {
            Some(cell) => cell.supports(&self.features),
            None => false,
        }
    }

    /// Violations as `(missing_required, forbidden_active)` for this active cell.
    pub fn violations(&self) -> (FeatureSet, FeatureSet) {
        match self.cell {
            Some(cell) => cell.violations(&self.features),
            None => (FeatureSet::new(), FeatureSet::new()),
        }
    }

    /// Missing required feature labels.
    pub fn missing(&self) -> Vec<String> {
        self.violations().0.labels()
    }

    /// Forbidden feature labels currently active.
    pub fn forbidden(&self) -> Vec<String> {
        self.violations().1.labels()
    }
}

impl Default for ActiveProfile {
    fn default() -> Self {
        Self::from_context(ActiveContext::from_env())
    }
}

/// Canonical BDD grid used for all profile validation.
pub fn canonical_grid() -> BddGrid {
    bitnet_bdd_grid::curated()
}

/// Active profile for the current process context.
pub fn active_profile() -> ActiveProfile {
    ActiveProfile::from_context(ActiveContext::from_env())
}

/// Active profile for an explicit scenario/environment pair.
pub fn active_profile_for(
    scenario: TestingScenario,
    environment: ExecutionEnvironment,
) -> ActiveProfile {
    ActiveProfile::from_context(ActiveContext { scenario, environment })
}

/// Validate current context against canonical grid row (if present).
pub fn validate_active_profile() -> Option<(FeatureSet, FeatureSet)> {
    let context = ActiveContext::from_env();
    validate_active_profile_for(context.scenario, context.environment)
}

/// Validate explicit scenario/environment against canonical grid row (if present).
pub fn validate_active_profile_for(
    scenario: TestingScenario,
    environment: ExecutionEnvironment,
) -> Option<(FeatureSet, FeatureSet)> {
    let profile = active_profile_for(scenario, environment);
    if profile.cell.is_some() { Some(profile.violations()) } else { None }
}

/// Labels for missing/forbidden feature mismatches, if a grid row exists.
pub fn active_profile_violation_labels() -> Option<(Vec<String>, Vec<String>)> {
    let profile = active_profile();
    profile.cell?;
    let (missing, forbidden) = profile.violations();
    Some((missing.labels(), forbidden.labels()))
}

/// Human-readable profile summary used by runtime diagnostics.
pub fn active_profile_summary() -> String {
    let profile = active_profile();
    match profile.cell {
        Some(cell) => format!(
            "scenario={}/environment={},required={},optional={},forbidden={}",
            profile.scenario,
            profile.environment,
            cell.required_features.labels().join("+"),
            cell.optional_features.labels().join("+"),
            cell.forbidden_features.labels().join("+"),
        ),
        None => format!(
            "scenario={}/environment={} (no matching grid cell)",
            profile.scenario, profile.environment
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{ActiveContext, ExecutionEnvironment, TestingScenario};

    #[test]
    fn from_env_uses_bitnet_env_over_ci() {
        // SAFETY: test-only; serial execution prevents races.
        unsafe {
            std::env::set_var("BITNET_ENV", "production");
            std::env::set_var("CI", "true");
        }

        let context = ActiveContext::from_env();
        assert_eq!(context.scenario, TestingScenario::Unit);
        assert_eq!(context.environment, ExecutionEnvironment::Production);

        unsafe {
            std::env::remove_var("BITNET_ENV");
            std::env::remove_var("CI");
        }
    }

    #[test]
    fn from_env_uses_bitnet_test_env_when_bitnet_env_absent() {
        // SAFETY: test-only; serial execution prevents races.
        unsafe {
            std::env::set_var("BITNET_TEST_ENV", "pre-prod");
        }

        let context = ActiveContext::from_env();
        assert_eq!(context.environment, ExecutionEnvironment::PreProduction);

        unsafe {
            std::env::remove_var("BITNET_TEST_ENV");
        }
    }

    #[test]
    fn from_env_falls_back_to_ci_when_no_env_explicitly_set() {
        // SAFETY: test-only; serial execution prevents races.
        unsafe {
            std::env::set_var("CI", "1");
        }

        let context = ActiveContext::from_env();
        assert_eq!(context.environment, ExecutionEnvironment::Ci);

        unsafe {
            std::env::remove_var("CI");
        }
    }

    #[test]
    fn from_env_with_defaults_uses_component_defaults() {
        let context = ActiveContext::from_env_with_defaults(
            TestingScenario::Integration,
            ExecutionEnvironment::Local,
        );

        assert_eq!(context.scenario, TestingScenario::Integration);
        assert_eq!(context.environment, ExecutionEnvironment::Local);
    }

    #[test]
    fn from_env_with_defaults_prefers_ci_when_no_override() {
        // SAFETY: test-only; serial execution prevents races.
        unsafe {
            std::env::set_var("CI", "1");
        }

        let context = ActiveContext::from_env_with_defaults(
            TestingScenario::Integration,
            ExecutionEnvironment::Local,
        );
        assert_eq!(context.environment, ExecutionEnvironment::Ci);

        unsafe {
            std::env::remove_var("CI");
        }
    }
}
