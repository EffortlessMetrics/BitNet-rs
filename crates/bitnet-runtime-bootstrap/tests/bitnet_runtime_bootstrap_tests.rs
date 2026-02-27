//! Property and integration tests for `bitnet-runtime-bootstrap`.
//!
//! The crate is a façade over `bitnet-startup-contract`, so these tests verify
//! that the re-exported types are accessible and behave correctly.

use bitnet_runtime_bootstrap::{ContractPolicy, ContractState, ProfileContract, RuntimeComponent};
use proptest::prelude::*;

// ── smoke tests ─────────────────────────────────────────────────────────────

#[test]
fn evaluate_observe_does_not_panic_for_all_components() {
    for component in [
        RuntimeComponent::Cli,
        RuntimeComponent::Server,
        RuntimeComponent::Test,
        RuntimeComponent::Custom,
    ] {
        let contract = ProfileContract::evaluate(component, ContractPolicy::Observe);
        assert!(!contract.summary().is_empty());
    }
}

#[test]
fn component_labels_are_unique_and_non_empty() {
    let labels: Vec<&str> = [
        RuntimeComponent::Cli,
        RuntimeComponent::Server,
        RuntimeComponent::Test,
        RuntimeComponent::Custom,
    ]
    .map(|c| c.label())
    .to_vec();

    for label in &labels {
        assert!(!label.is_empty(), "label must not be empty");
    }
    let unique: std::collections::HashSet<&&str> = labels.iter().collect();
    assert_eq!(unique.len(), labels.len(), "component labels must be unique");
}

#[test]
fn enforce_policy_observe_never_returns_err() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    assert!(contract.enforce().is_ok());
}

#[test]
fn contract_state_compatible_implies_is_compatible() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Test, ContractPolicy::Observe);
    let expected = matches!(contract.state(), ContractState::Compatible);
    assert_eq!(contract.is_compatible(), expected);
}

#[test]
fn active_features_list_is_accessible() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    let _features: Vec<String> = contract.active_features();
}

// ── proptest invariants ──────────────────────────────────────────────────────

const COMPONENTS: [RuntimeComponent; 4] = [
    RuntimeComponent::Cli,
    RuntimeComponent::Server,
    RuntimeComponent::Test,
    RuntimeComponent::Custom,
];

proptest! {
    /// `summary()` must always contain the component label.
    #[test]
    fn summary_contains_component_label(idx in 0usize..4) {
        let component = COMPONENTS[idx];
        let contract = ProfileContract::evaluate(component, ContractPolicy::Observe);
        prop_assert!(contract.summary().contains(component.label()));
    }

    /// Violation slices must be consistent with the contract state.
    #[test]
    fn violation_slices_consistent_with_state(idx in 0usize..4) {
        let contract = ProfileContract::evaluate(COMPONENTS[idx], ContractPolicy::Observe);
        match contract.state() {
            ContractState::Compatible => {
                prop_assert!(contract.missing_required().is_empty());
                prop_assert!(contract.forbidden_active().is_empty());
            }
            ContractState::MissingRequired => {
                prop_assert!(!contract.missing_required().is_empty());
            }
            ContractState::ForbiddenActive => {
                prop_assert!(!contract.forbidden_active().is_empty());
            }
            ContractState::UnknownGridCell => {}
        }
    }
}
