//! Edge-case tests for bitnet-startup-contract-core types and contract logic.

use bitnet_startup_contract_core::{
    ContractPolicy, ContractState, ProfileContract, RuntimeComponent,
};

// ---------------------------------------------------------------------------
// RuntimeComponent: labels and defaults
// ---------------------------------------------------------------------------

#[test]
fn component_cli_label() {
    assert_eq!(RuntimeComponent::Cli.label(), "bitnet-cli");
}

#[test]
fn component_server_label() {
    assert_eq!(RuntimeComponent::Server.label(), "bitnet-server");
}

#[test]
fn component_test_label() {
    assert_eq!(RuntimeComponent::Test.label(), "test");
}

#[test]
fn component_custom_label() {
    assert_eq!(RuntimeComponent::Custom.label(), "custom");
}

#[test]
fn component_debug_not_empty() {
    let d = format!("{:?}", RuntimeComponent::Cli);
    assert!(!d.is_empty());
    assert!(d.contains("Cli"));
}

#[test]
fn component_clone_copy() {
    let c = RuntimeComponent::Server;
    let c2 = c;
    assert_eq!(format!("{:?}", c), format!("{:?}", c2));
}

// ---------------------------------------------------------------------------
// ContractPolicy: Debug
// ---------------------------------------------------------------------------

#[test]
fn policy_observe_debug() {
    let d = format!("{:?}", ContractPolicy::Observe);
    assert!(d.contains("Observe"));
}

#[test]
fn policy_enforce_debug() {
    let d = format!("{:?}", ContractPolicy::Enforce);
    assert!(d.contains("Enforce"));
}

#[test]
fn policy_copy() {
    let p = ContractPolicy::Enforce;
    let p2 = p;
    assert_eq!(format!("{:?}", p), format!("{:?}", p2));
}

// ---------------------------------------------------------------------------
// ContractState: equality and Debug
// ---------------------------------------------------------------------------

#[test]
fn state_compatible_eq() {
    assert_eq!(ContractState::Compatible, ContractState::Compatible);
}

#[test]
fn state_unknown_grid_cell_eq() {
    assert_eq!(ContractState::UnknownGridCell, ContractState::UnknownGridCell);
}

#[test]
fn state_missing_required_eq() {
    assert_eq!(ContractState::MissingRequired, ContractState::MissingRequired);
}

#[test]
fn state_forbidden_active_eq() {
    assert_eq!(ContractState::ForbiddenActive, ContractState::ForbiddenActive);
}

#[test]
fn state_ne() {
    assert_ne!(ContractState::Compatible, ContractState::MissingRequired);
    assert_ne!(ContractState::UnknownGridCell, ContractState::ForbiddenActive);
}

#[test]
fn state_debug_compatible() {
    let d = format!("{:?}", ContractState::Compatible);
    assert!(d.contains("Compatible"));
}

#[test]
fn state_debug_unknown_grid_cell() {
    let d = format!("{:?}", ContractState::UnknownGridCell);
    assert!(d.contains("UnknownGridCell"));
}

#[test]
fn state_debug_missing_required() {
    let d = format!("{:?}", ContractState::MissingRequired);
    assert!(d.contains("MissingRequired"));
}

#[test]
fn state_debug_forbidden_active() {
    let d = format!("{:?}", ContractState::ForbiddenActive);
    assert!(d.contains("ForbiddenActive"));
}

#[test]
fn state_clone_copy() {
    let s = ContractState::Compatible;
    let s2 = s;
    assert_eq!(s, s2);
}

// ---------------------------------------------------------------------------
// ProfileContract: evaluate with Observe policy (safe, no env mutation)
// ---------------------------------------------------------------------------

#[test]
fn contract_evaluate_cli_observe_has_summary() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    let summary = contract.summary();
    assert!(summary.contains("bitnet-cli"));
    assert!(summary.contains("observe"));
}

#[test]
fn contract_evaluate_server_observe_has_summary() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Server, ContractPolicy::Observe);
    let summary = contract.summary();
    assert!(summary.contains("bitnet-server"));
}

#[test]
fn contract_evaluate_test_observe_has_ci_environment() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Test, ContractPolicy::Observe);
    let ctx = contract.context();
    // Test component defaults to CI environment
    assert_eq!(ctx.environment.to_string(), "ci");
}

#[test]
fn contract_evaluate_custom_observe_has_unit_scenario() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    let ctx = contract.context();
    // Custom component defaults to Unit scenario
    assert_eq!(ctx.scenario.to_string(), "unit");
}

#[test]
fn contract_state_is_deterministic() {
    let c1 = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    let c2 = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    assert_eq!(c1.state(), c2.state());
}

#[test]
fn contract_is_compatible_matches_state() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    if contract.state() == ContractState::Compatible {
        assert!(contract.is_compatible());
    } else {
        assert!(!contract.is_compatible());
    }
}

#[test]
fn contract_policy_accessor() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    match contract.policy() {
        ContractPolicy::Observe => {} // expected
        ContractPolicy::Enforce => panic!("should be observe"),
    }
}

#[test]
fn contract_enforce_policy_accessor() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Server, ContractPolicy::Enforce);
    match contract.policy() {
        ContractPolicy::Enforce => {} // expected
        ContractPolicy::Observe => panic!("should be enforce"),
    }
}

#[test]
fn contract_component_accessor() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    assert_eq!(contract.component().label(), "bitnet-cli");
}

#[test]
fn contract_missing_required_is_vec() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    // Just verifying the accessor works without panic
    let _missing = contract.missing_required();
}

#[test]
fn contract_forbidden_active_is_vec() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    let _forbidden = contract.forbidden_active();
}

#[test]
fn contract_required_features_is_vec() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    let _required = contract.required_features();
}

#[test]
fn contract_optional_features_is_vec() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    let _optional = contract.optional_features();
}

#[test]
fn contract_forbidden_features_is_vec() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    let _forbidden = contract.forbidden_features();
}

#[test]
fn contract_active_features_returns_labels() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    let active = contract.active_features();
    // Should be a Vec<String> of feature labels
    for label in &active {
        assert!(!label.is_empty());
    }
}

// ---------------------------------------------------------------------------
// ProfileContract: summary format validation
// ---------------------------------------------------------------------------

#[test]
fn contract_summary_contains_state() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    let summary = contract.summary();
    // Should contain one of the state strings
    let has_state = summary.contains("compatible")
        || summary.contains("unknown-grid-cell")
        || summary.contains("missing-required")
        || summary.contains("forbidden-active");
    assert!(has_state, "summary should contain state: {summary}");
}

#[test]
fn contract_summary_contains_scenario() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    let summary = contract.summary();
    assert!(summary.contains("scenario="), "summary should contain scenario=: {summary}");
}

#[test]
fn contract_summary_contains_environment() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    let summary = contract.summary();
    assert!(summary.contains("environment="), "summary should contain environment=: {summary}");
}

#[test]
fn contract_summary_contains_required() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    let summary = contract.summary();
    assert!(summary.contains("required="), "summary should contain required=: {summary}");
}

#[test]
fn contract_summary_contains_active() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    let summary = contract.summary();
    assert!(summary.contains("active="), "summary should contain active=: {summary}");
}

// ---------------------------------------------------------------------------
// ProfileContract: enforce behavior with Observe policy
// ---------------------------------------------------------------------------

#[test]
fn contract_enforce_observe_always_ok() {
    // Observe policy should never return Err even if incompatible
    let contract = ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    let result = contract.enforce();
    assert!(result.is_ok(), "observe policy should never fail enforce()");
}

#[test]
fn contract_enforce_observe_cli_ok() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    assert!(contract.enforce().is_ok());
}

#[test]
fn contract_enforce_observe_server_ok() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Server, ContractPolicy::Observe);
    assert!(contract.enforce().is_ok());
}

#[test]
fn contract_enforce_observe_test_ok() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Test, ContractPolicy::Observe);
    assert!(contract.enforce().is_ok());
}

// ---------------------------------------------------------------------------
// ProfileContract: Debug impl
// ---------------------------------------------------------------------------

#[test]
fn contract_debug_not_empty() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    let d = format!("{:?}", contract);
    assert!(!d.is_empty());
    assert!(d.contains("ProfileContract"));
}

// ---------------------------------------------------------------------------
// All components have consistent evaluate behavior
// ---------------------------------------------------------------------------

#[test]
fn all_components_produce_valid_summaries() {
    let components = [
        RuntimeComponent::Cli,
        RuntimeComponent::Server,
        RuntimeComponent::Test,
        RuntimeComponent::Custom,
    ];
    for comp in &components {
        let contract = ProfileContract::evaluate(*comp, ContractPolicy::Observe);
        let summary = contract.summary();
        assert!(!summary.is_empty(), "empty summary for {:?}", comp);
        assert!(
            summary.contains(comp.label()),
            "summary should contain component label for {:?}: {summary}",
            comp
        );
    }
}

#[test]
fn all_components_context_scenario_is_valid() {
    let components = [
        RuntimeComponent::Cli,
        RuntimeComponent::Server,
        RuntimeComponent::Test,
        RuntimeComponent::Custom,
    ];
    for comp in &components {
        let contract = ProfileContract::evaluate(*comp, ContractPolicy::Observe);
        let scenario_str = contract.context().scenario.to_string();
        assert!(!scenario_str.is_empty(), "empty scenario for {:?}", comp);
    }
}
