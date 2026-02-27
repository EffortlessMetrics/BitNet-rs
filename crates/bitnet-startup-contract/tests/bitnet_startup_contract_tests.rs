//! Property and integration tests for `bitnet-startup-contract`.
//!
//! The crate is a façade over `bitnet-startup-contract-core`. Tests verify the
//! re-exported API surface is accessible and maintains expected invariants.

use bitnet_startup_contract::{
    ActiveContext, ContractPolicy, ContractState, ExecutionEnvironment, ProfileContract,
    RuntimeComponent, TestingScenario,
};
use proptest::prelude::*;

const COMPONENTS: [RuntimeComponent; 4] = [
    RuntimeComponent::Cli,
    RuntimeComponent::Server,
    RuntimeComponent::Test,
    RuntimeComponent::Custom,
];

// ── smoke tests ─────────────────────────────────────────────────────────────

#[test]
fn evaluate_returns_valid_contract_for_every_component() {
    for component in COMPONENTS {
        let contract = ProfileContract::evaluate(component, ContractPolicy::Observe);
        assert!(!contract.summary().is_empty());
        assert!(!contract.component().label().is_empty());
    }
}

#[test]
fn with_context_accepts_explicit_scenario_environment() {
    let context =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Custom, context, ContractPolicy::Observe);
    assert_eq!(contract.context().scenario, TestingScenario::Unit);
    assert_eq!(contract.context().environment, ExecutionEnvironment::Local);
}

#[test]
fn observe_policy_enforce_is_always_ok() {
    for component in COMPONENTS {
        let contract = ProfileContract::evaluate(component, ContractPolicy::Observe);
        assert!(contract.enforce().is_ok(), "Observe policy must never fail");
    }
}

#[test]
fn contract_feature_accessors_return_slices() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Test, ContractPolicy::Observe);
    let _req: &[String] = contract.required_features();
    let _opt: &[String] = contract.optional_features();
    let _fbd: &[String] = contract.forbidden_features();
    let _miss: &[String] = contract.missing_required();
    let _forb: &[String] = contract.forbidden_active();
}

// ── proptest invariants ──────────────────────────────────────────────────────

proptest! {
    /// `is_compatible()` must match `ContractState::Compatible`.
    #[test]
    fn is_compatible_matches_state(comp_idx in 0usize..4) {
        let contract = ProfileContract::evaluate(COMPONENTS[comp_idx], ContractPolicy::Observe);
        let expected = matches!(contract.state(), ContractState::Compatible);
        prop_assert_eq!(contract.is_compatible(), expected);
    }

    /// `summary()` must contain scenario and environment strings.
    #[test]
    fn summary_contains_scenario_and_environment(comp_idx in 0usize..4) {
        let contract = ProfileContract::evaluate(COMPONENTS[comp_idx], ContractPolicy::Observe);
        let summary = contract.summary();
        let ctx = contract.context();
        prop_assert!(summary.contains(&ctx.scenario.to_string()));
        prop_assert!(summary.contains(&ctx.environment.to_string()));
    }

    /// All active feature labels must be non-empty strings.
    #[test]
    fn active_features_elements_non_empty(comp_idx in 0usize..4) {
        let contract = ProfileContract::evaluate(COMPONENTS[comp_idx], ContractPolicy::Observe);
        for label in contract.active_features() {
            prop_assert!(!label.is_empty(), "feature label must not be empty");
        }
    }
}
