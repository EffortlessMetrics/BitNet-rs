//! Edge-case tests for bitnet-startup-contract-core: RuntimeComponent,
//! ContractPolicy, ContractState, ProfileContract.

use bitnet_startup_contract_core::{
    ActiveContext, ContractPolicy, ContractState, ExecutionEnvironment, ProfileContract,
    RuntimeComponent, TestingScenario,
};

// ---------------------------------------------------------------------------
// RuntimeComponent — label
// ---------------------------------------------------------------------------

#[test]
fn component_label_cli() {
    assert_eq!(RuntimeComponent::Cli.label(), "bitnet-cli");
}

#[test]
fn component_label_server() {
    assert_eq!(RuntimeComponent::Server.label(), "bitnet-server");
}

#[test]
fn component_label_test() {
    assert_eq!(RuntimeComponent::Test.label(), "test");
}

#[test]
fn component_label_custom() {
    assert_eq!(RuntimeComponent::Custom.label(), "custom");
}

#[test]
fn component_debug() {
    let dbg = format!("{:?}", RuntimeComponent::Cli);
    assert!(dbg.contains("Cli"));
}

#[test]
fn component_clone() {
    let c = RuntimeComponent::Server;
    let cloned = c;
    assert_eq!(cloned.label(), "bitnet-server");
}

// ---------------------------------------------------------------------------
// ContractPolicy — Debug, Clone, Copy
// ---------------------------------------------------------------------------

#[test]
fn policy_debug() {
    let dbg = format!("{:?}", ContractPolicy::Observe);
    assert!(dbg.contains("Observe"));
    let dbg = format!("{:?}", ContractPolicy::Enforce);
    assert!(dbg.contains("Enforce"));
}

// ---------------------------------------------------------------------------
// ContractState — Debug, Clone, Copy, PartialEq, Eq
// ---------------------------------------------------------------------------

#[test]
fn state_partial_eq() {
    assert_eq!(ContractState::Compatible, ContractState::Compatible);
    assert_ne!(ContractState::Compatible, ContractState::MissingRequired);
}

#[test]
fn state_all_variants_distinct() {
    let variants = [
        ContractState::Compatible,
        ContractState::UnknownGridCell,
        ContractState::MissingRequired,
        ContractState::ForbiddenActive,
    ];
    for i in 0..variants.len() {
        for j in 0..variants.len() {
            if i == j {
                assert_eq!(variants[i], variants[j]);
            } else {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }
}

#[test]
fn state_debug() {
    let dbg = format!("{:?}", ContractState::UnknownGridCell);
    assert!(dbg.contains("UnknownGridCell"));
}

// ---------------------------------------------------------------------------
// ProfileContract — with_context
// ---------------------------------------------------------------------------

#[test]
fn contract_with_context_has_component() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Cli, ctx, ContractPolicy::Observe);
    assert_eq!(contract.component().label(), "bitnet-cli");
}

#[test]
fn contract_summary_contains_component() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Server, ctx, ContractPolicy::Observe);
    let summary = contract.summary();
    assert!(summary.contains("bitnet-server"));
    assert!(summary.contains("unit"));
    assert!(summary.contains("local"));
}

#[test]
fn contract_summary_contains_policy_label() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Smoke, environment: ExecutionEnvironment::Ci };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Test, ctx, ContractPolicy::Enforce);
    let summary = contract.summary();
    assert!(summary.contains("enforce"));
}

#[test]
fn contract_compatible_is_compatible() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Custom, ctx, ContractPolicy::Observe);
    // Whether compatible depends on the canonical grid; at minimum the contract should resolve
    let _ = contract.is_compatible();
    let _ = contract.state();
}

#[test]
fn contract_enforce_compatible_returns_ok() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Custom, ctx, ContractPolicy::Enforce);
    // Even with enforce, if state is compatible or unknown-grid-cell + observe, enforce passes
    // We just check it doesn't panic
    let _ = contract.enforce();
}

#[test]
fn contract_observe_always_returns_ok() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Custom, ctx, ContractPolicy::Observe);
    // Observe policy never fails
    assert!(contract.enforce().is_ok());
}

#[test]
fn contract_context_roundtrip() {
    let ctx = ActiveContext {
        scenario: TestingScenario::Integration,
        environment: ExecutionEnvironment::Ci,
    };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Cli, ctx, ContractPolicy::Observe);
    let ctx_out = contract.context();
    assert_eq!(ctx_out.scenario, TestingScenario::Integration);
    assert_eq!(ctx_out.environment, ExecutionEnvironment::Ci);
}

#[test]
fn contract_accessors_dont_panic() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Smoke, environment: ExecutionEnvironment::Ci };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Server, ctx, ContractPolicy::Observe);
    let _ = contract.missing_required();
    let _ = contract.forbidden_active();
    let _ = contract.required_features();
    let _ = contract.optional_features();
    let _ = contract.forbidden_features();
    let _ = contract.active_features();
    let _ = contract.policy();
}

#[test]
fn contract_debug() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let contract =
        ProfileContract::with_context(RuntimeComponent::Custom, ctx, ContractPolicy::Observe);
    let dbg = format!("{contract:?}");
    assert!(dbg.contains("ProfileContract"));
}
