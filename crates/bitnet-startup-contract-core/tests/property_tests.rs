//! Property-based tests for `bitnet-startup-contract-core`.
//!
//! These tests verify invariants of `ProfileContract` and the `ContractState`
//! / `ContractPolicy` types without relying on any environment variables.

use bitnet_startup_contract_core::{
    ActiveContext, ContractPolicy, ContractState, ExecutionEnvironment, ProfileContract,
    RuntimeComponent, TestingScenario,
};
use proptest::prelude::*;

fn arb_component() -> impl Strategy<Value = RuntimeComponent> {
    prop_oneof![
        Just(RuntimeComponent::Cli),
        Just(RuntimeComponent::Server),
        Just(RuntimeComponent::Test),
        Just(RuntimeComponent::Custom),
    ]
}

fn arb_policy() -> impl Strategy<Value = ContractPolicy> {
    prop_oneof![Just(ContractPolicy::Observe), Just(ContractPolicy::Enforce),]
}

fn arb_scenario() -> impl Strategy<Value = TestingScenario> {
    prop_oneof![
        Just(TestingScenario::Unit),
        Just(TestingScenario::Integration),
        Just(TestingScenario::CrossValidation),
        Just(TestingScenario::EndToEnd),
    ]
}

fn arb_environment() -> impl Strategy<Value = ExecutionEnvironment> {
    prop_oneof![Just(ExecutionEnvironment::Local), Just(ExecutionEnvironment::Ci),]
}

proptest! {
    /// `ProfileContract::with_context` never panics for any component/policy/scenario/env
    /// combination, and the returned context always matches the input.
    #[test]
    fn with_context_context_round_trips(
        component in arb_component(),
        policy in arb_policy(),
        scenario in arb_scenario(),
        environment in arb_environment(),
    ) {
        let ctx = ActiveContext { scenario, environment };
        let contract = ProfileContract::with_context(component, ctx, policy);
        // The returned context must echo back what we passed in.
        prop_assert_eq!(contract.context().scenario, scenario);
        prop_assert_eq!(contract.context().environment, environment);
    }

    /// `summary()` always returns a non-empty string for any contract.
    #[test]
    fn summary_is_always_non_empty(
        component in arb_component(),
        policy in arb_policy(),
        scenario in arb_scenario(),
        environment in arb_environment(),
    ) {
        let ctx = ActiveContext { scenario, environment };
        let contract = ProfileContract::with_context(component, ctx, policy);
        let s = contract.summary();
        prop_assert!(!s.is_empty());
    }

    /// For `ContractPolicy::Observe`, `enforce()` always returns `Ok` (never fails).
    #[test]
    fn observe_policy_enforce_always_ok(
        component in arb_component(),
        scenario in arb_scenario(),
        environment in arb_environment(),
    ) {
        let ctx = ActiveContext { scenario, environment };
        let contract = ProfileContract::with_context(component, ctx, ContractPolicy::Observe);
        prop_assert!(contract.enforce().is_ok());
    }

    /// `is_compatible()` is consistent with `state() == ContractState::Compatible`.
    #[test]
    fn is_compatible_matches_state(
        component in arb_component(),
        policy in arb_policy(),
        scenario in arb_scenario(),
        environment in arb_environment(),
    ) {
        let ctx = ActiveContext { scenario, environment };
        let contract = ProfileContract::with_context(component, ctx, policy);
        let expected = matches!(contract.state(), ContractState::Compatible);
        prop_assert_eq!(contract.is_compatible(), expected);
    }
}

#[test]
fn component_labels_are_stable() {
    assert_eq!(RuntimeComponent::Cli.label(), "bitnet-cli");
    assert_eq!(RuntimeComponent::Server.label(), "bitnet-server");
    assert_eq!(RuntimeComponent::Test.label(), "test");
    assert_eq!(RuntimeComponent::Custom.label(), "custom");
}
