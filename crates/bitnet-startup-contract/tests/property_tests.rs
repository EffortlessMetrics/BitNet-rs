//! Property-based tests for `bitnet-startup-contract`.
//!
//! Key invariants:
//! - `RuntimeComponent::label()` is always a non-empty string.
//! - `ProfileContract::with_context` always produces a valid state.
//! - Compatible contracts have no missing-required or forbidden-active features.
//! - `ProfileContract::summary()` is always non-empty.
//! - `ContractState::Compatible` implies `is_compatible()`.

use bitnet_startup_contract::{
    ActiveContext, ContractPolicy, ContractState, ExecutionEnvironment, ProfileContract,
    RuntimeComponent, TestingScenario,
};
use proptest::prelude::*;

// ── Arbitrary strategies ──────────────────────────────────────────────────────

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
        Just(TestingScenario::EndToEnd),
        Just(TestingScenario::Performance),
        Just(TestingScenario::CrossValidation),
        Just(TestingScenario::Smoke),
        Just(TestingScenario::Development),
        Just(TestingScenario::Debug),
        Just(TestingScenario::Minimal),
    ]
}

fn arb_environment() -> impl Strategy<Value = ExecutionEnvironment> {
    prop_oneof![
        Just(ExecutionEnvironment::Local),
        Just(ExecutionEnvironment::Ci),
        Just(ExecutionEnvironment::PreProduction),
        Just(ExecutionEnvironment::Production),
    ]
}

// ── RuntimeComponent ──────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn component_label_is_nonempty(c in arb_component()) {
        prop_assert!(!c.label().is_empty());
    }

    #[test]
    fn component_label_is_stable(c in arb_component()) {
        prop_assert_eq!(c.label(), c.label());
    }
}

// ── ProfileContract construction ──────────────────────────────────────────────

proptest! {
    /// Contract construction never panics for any valid inputs.
    #[test]
    fn contract_construction_never_panics(
        component in arb_component(),
        scenario in arb_scenario(),
        env in arb_environment(),
        policy in arb_policy(),
    ) {
        let context = ActiveContext { scenario, environment: env };
        let contract = ProfileContract::with_context(component, context, policy);
        // Should always produce a valid contract
        let _ = contract.state();
    }

    /// Summary is always a non-empty string.
    #[test]
    fn contract_summary_is_nonempty(
        component in arb_component(),
        scenario in arb_scenario(),
        env in arb_environment(),
        policy in arb_policy(),
    ) {
        let context = ActiveContext { scenario, environment: env };
        let contract = ProfileContract::with_context(component, context, policy);
        prop_assert!(!contract.summary().is_empty());
    }

    /// Compatible state implies no missing-required and no forbidden-active.
    #[test]
    fn compatible_implies_no_violations(
        component in arb_component(),
        scenario in arb_scenario(),
        env in arb_environment(),
        policy in arb_policy(),
    ) {
        let context = ActiveContext { scenario, environment: env };
        let contract = ProfileContract::with_context(component, context, policy);
        if contract.state() == ContractState::Compatible {
            prop_assert!(contract.missing_required().is_empty());
            prop_assert!(contract.forbidden_active().is_empty());
            prop_assert!(contract.is_compatible());
        }
    }

    /// `is_compatible()` is true iff state is `Compatible`.
    #[test]
    fn is_compatible_matches_state(
        component in arb_component(),
        scenario in arb_scenario(),
        env in arb_environment(),
        policy in arb_policy(),
    ) {
        let context = ActiveContext { scenario, environment: env };
        let contract = ProfileContract::with_context(component, context, policy);
        prop_assert_eq!(
            contract.is_compatible(),
            contract.state() == ContractState::Compatible,
        );
    }
}

// ── Enforce policy ───────────────────────────────────────────────────────────

proptest! {
    /// Observe policy enforce() always succeeds (never returns Err).
    #[test]
    fn observe_enforce_always_succeeds(
        component in arb_component(),
        scenario in arb_scenario(),
        env in arb_environment(),
    ) {
        let context = ActiveContext { scenario, environment: env };
        let contract = ProfileContract::with_context(
            component, context, ContractPolicy::Observe,
        );
        prop_assert!(contract.enforce().is_ok());
    }

    /// Enforce policy on a compatible contract always succeeds.
    #[test]
    fn enforce_compatible_always_succeeds(
        component in arb_component(),
        scenario in arb_scenario(),
        env in arb_environment(),
    ) {
        let context = ActiveContext { scenario, environment: env };
        let contract = ProfileContract::with_context(
            component, context, ContractPolicy::Enforce,
        );
        if contract.state() == ContractState::Compatible {
            prop_assert!(contract.enforce().is_ok());
        }
    }
}
