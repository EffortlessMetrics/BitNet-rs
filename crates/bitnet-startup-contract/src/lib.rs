//! Startup contract primitives built from runtime profile + BDD grid.
//!
//! This crate is intentionally small and stable. It exposes the canonical startup
//! contract API used by CLI/server entrypoints while keeping profile detection
//! logic in `bitnet-runtime-profile`.

use anyhow::{anyhow, Result};

pub use bitnet_runtime_profile::{
    active_features,
    active_profile,
    active_profile_for,
    active_profile_summary,
    active_profile_violation_labels,
    canonical_grid,
    feature_labels,
    feature_line,
    validate_active_profile,
    validate_active_profile_for,
    ActiveContext,
    ActiveProfile,
    BddCell,
    BddGrid,
    BitnetFeature,
    ExecutionEnvironment,
    FeatureSet,
    TestingScenario,
};

/// Runtime component identity used for startup contract lookups.
#[derive(Debug, Clone, Copy)]
pub enum RuntimeComponent {
    /// CLI process (`bitnet`).
    Cli,
    /// HTTP server process (`bitnet-server`).
    Server,
    /// Test or benchmarking harness.
    Test,
    /// Any other component.
    Custom,
}

impl RuntimeComponent {
    fn default_scenario(self) -> TestingScenario {
        match self {
            Self::Cli => TestingScenario::Integration,
            Self::Server => TestingScenario::Integration,
            Self::Test => TestingScenario::CrossValidation,
            Self::Custom => TestingScenario::Unit,
        }
    }

    fn default_environment(self) -> ExecutionEnvironment {
        match self {
            Self::Test => ExecutionEnvironment::Ci,
            _ => ExecutionEnvironment::Local,
        }
    }

    /// Human-readable component name.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cli => "bitnet-cli",
            Self::Server => "bitnet-server",
            Self::Test => "test",
            Self::Custom => "custom",
        }
    }
}

/// Enforcement behavior when a contract is incompatible.
#[derive(Debug, Clone, Copy)]
pub enum ContractPolicy {
    /// Emit compatibility summary and continue startup.
    Observe,
    /// Fail startup when the active runtime profile is incompatible.
    Enforce,
}

/// Compatibility result for a resolved profile row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractState {
    /// A matching BDD row exists and feature constraints are satisfied.
    Compatible,
    /// No row in the curated grid for the active scenario/environment.
    UnknownGridCell,
    /// Required features are missing for the active row.
    MissingRequired,
    /// Forbidden features are enabled for the active row.
    ForbiddenActive,
}

/// Resolved startup contract result for a component.
#[derive(Debug)]
pub struct ProfileContract {
    component: RuntimeComponent,
    profile: ActiveProfile,
    cell: Option<&'static BddCell>,
    state: ContractState,
    policy: ContractPolicy,
    missing_required: Vec<String>,
    forbidden_active: Vec<String>,
    required: Vec<String>,
    optional: Vec<String>,
    forbidden: Vec<String>,
}

impl ProfileContract {
    /// Build a startup contract from an explicit scenario/environment context.
    pub fn with_context(component: RuntimeComponent, context: ActiveContext, policy: ContractPolicy) -> Self {
        let profile = active_profile_for(context.scenario, context.environment);
        let cell = profile.cell;
        let mut missing = Vec::new();
        let mut forbidden = Vec::new();
        let state = if let Some(cell) = cell {
            let (missing_features, forbidden_features) = profile.violations();
            missing.extend(missing_features.labels());
            forbidden.extend(forbidden_features.labels());

            if !forbidden.is_empty() {
                ContractState::ForbiddenActive
            } else if !missing.is_empty() {
                ContractState::MissingRequired
            } else {
                ContractState::Compatible
            }
        } else {
            ContractState::UnknownGridCell
        };

        Self {
            component,
            profile,
            cell,
            state,
            policy,
            missing_required: missing,
            forbidden_active: forbidden,
            required: cell.map(|cell| cell.required_features.labels()).unwrap_or_default(),
            optional: cell.map(|cell| cell.optional_features.labels()).unwrap_or_default(),
            forbidden: cell.map(|cell| cell.forbidden_features.labels()).unwrap_or_default(),
        }
    }

    /// Build a startup contract from process environment and component defaults.
    pub fn evaluate(component: RuntimeComponent, policy: ContractPolicy) -> Self {
        let base_context = ActiveContext::from_env_with_defaults(
            component.default_scenario(),
            component.default_environment(),
        );
        Self::with_context(component, base_context, policy)
    }

    /// Active scenario/environment context used for this contract.
    pub fn context(&self) -> ActiveContext {
        ActiveContext { scenario: self.profile.scenario, environment: self.profile.environment }
    }

    /// Which component emitted this contract.
    pub fn component(&self) -> RuntimeComponent {
        self.component
    }

    /// Contract compatibility state.
    pub fn state(&self) -> ContractState {
        self.state
    }

    /// Whether the active profile satisfies the discovered BDD cell.
    pub fn is_compatible(&self) -> bool {
        matches!(self.state, ContractState::Compatible)
    }

    /// Contract enforcement policy.
    pub fn policy(&self) -> ContractPolicy {
        self.policy
    }

    /// Missing required features list.
    pub fn missing_required(&self) -> &[String] {
        &self.missing_required
    }

    /// Active forbidden features list.
    pub fn forbidden_active(&self) -> &[String] {
        &self.forbidden_active
    }

    /// Required feature labels from the matching cell, if any.
    pub fn required_features(&self) -> &[String] {
        &self.required
    }

    /// Optional feature labels from the matching cell, if any.
    pub fn optional_features(&self) -> &[String] {
        &self.optional
    }

    /// Forbidden feature labels from the matching cell, if any.
    pub fn forbidden_features(&self) -> &[String] {
        &self.forbidden
    }

    /// Active feature labels on the process.
    pub fn active_features(&self) -> Vec<String> {
        self.profile.features.labels()
    }

    /// Human-readable summary suitable for logs and diagnostics.
    pub fn summary(&self) -> String {
        let state = match self.state {
            ContractState::Compatible => "compatible",
            ContractState::UnknownGridCell => "unknown-grid-cell",
            ContractState::MissingRequired => "missing-required",
            ContractState::ForbiddenActive => "forbidden-active",
        };

        let required = if self.required.is_empty() {
            "none".to_string()
        } else {
            self.required.join(", ")
        };
        let optional = if self.optional.is_empty() {
            "none".to_string()
        } else {
            self.optional.join(", ")
        };
        let forbidden = if self.forbidden.is_empty() {
            "none".to_string()
        } else {
            self.forbidden.join(", ")
        };
        let active = if self.profile.features.is_empty() {
            "none".to_string()
        } else {
            self.profile.features.labels().join(", ")
        };

        format!(
            "{} contract [{}]: scenario={} environment={} state={} required=[{}] optional=[{}] forbidden=[{}] active=[{}]",
            self.component.label(),
            self.policy_label(),
            self.context().scenario,
            self.context().environment,
            state,
            required,
            optional,
            forbidden,
            active
        )
    }

    fn policy_label(&self) -> &'static str {
        match self.policy {
            ContractPolicy::Observe => "observe",
            ContractPolicy::Enforce => "enforce",
        }
    }

    /// Enforce policy and return a copy of this contract if accepted.
    pub fn enforce(self) -> Result<Self> {
        match self.state {
            ContractState::Compatible => Ok(self),
            ContractState::UnknownGridCell => {
                if matches!(self.policy, ContractPolicy::Enforce) {
                    Err(anyhow!(
                        "No BDD grid row found for scenario/environment in {} contract",
                        self.component.label()
                    ))
                } else {
                    Ok(self)
                }
            }
            ContractState::MissingRequired => {
                if matches!(self.policy, ContractPolicy::Enforce) {
                    Err(anyhow!(
                        "{} startup contract is missing required features: {:?}",
                        self.component.label(),
                        self.missing_required
                    ))
                } else {
                    Ok(self)
                }
            }
            ContractState::ForbiddenActive => {
                if matches!(self.policy, ContractPolicy::Enforce) {
                    Err(anyhow!(
                        "{} startup contract has forbidden active features: {:?}",
                        self.component.label(),
                        self.forbidden_active
                    ))
                } else {
                    Ok(self)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RuntimeComponent, ContractPolicy, ProfileContract};

    #[test]
    fn evaluate_preserves_context_overrides() {
        std::env::set_var("BITNET_TEST_SCENARIO", "e2e");
        let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
        assert_eq!(contract.context().scenario.to_string(), "e2e");
        std::env::remove_var("BITNET_TEST_SCENARIO");
    }

    #[test]
    fn resolve_context_defaults_are_stable() {
        let contract = ProfileContract::evaluate(RuntimeComponent::Test, ContractPolicy::Observe);
        assert_eq!(contract.context().environment.to_string(), "ci");
    }
}
