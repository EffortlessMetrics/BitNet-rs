use bitnet_startup_contract_guard::{ContractPolicy, RuntimeComponent, StartupContractGuard};
use proptest::prelude::*;

fn arb_component() -> impl Strategy<Value = RuntimeComponent> {
    prop_oneof![
        Just(RuntimeComponent::Cli),
        Just(RuntimeComponent::Server),
        Just(RuntimeComponent::Test),
        Just(RuntimeComponent::Custom),
    ]
}

proptest! {
    /// `evaluate()` with Observe policy never returns an error for any component.
    #[test]
    fn evaluate_observe_never_errors(component in arb_component()) {
        let result = StartupContractGuard::evaluate(component, ContractPolicy::Observe);
        prop_assert!(result.is_ok(), "evaluate returned Err for {:?}: {:?}", component, result);
    }

    /// `is_compatible()` always delegates to the inner report contract state.
    #[test]
    fn is_compatible_delegates_to_report(component in arb_component()) {
        let guard = StartupContractGuard::evaluate(component, ContractPolicy::Observe)
            .expect("Observe policy must not fail");
        prop_assert_eq!(
            guard.is_compatible(),
            guard.report.contract.is_compatible(),
            "is_compatible() mismatch for {:?}", component
        );
    }

    /// `feature_line` snapshot is always non-empty.
    #[test]
    fn feature_line_snapshot_non_empty(component in arb_component()) {
        let guard = StartupContractGuard::evaluate(component, ContractPolicy::Observe)
            .expect("Observe policy must not fail");
        prop_assert!(
            !guard.feature_line.is_empty(),
            "feature_line was empty for {:?}", component
        );
    }

    /// `profile_summary` snapshot is always non-empty.
    #[test]
    fn profile_summary_snapshot_non_empty(component in arb_component()) {
        let guard = StartupContractGuard::evaluate(component, ContractPolicy::Observe)
            .expect("Observe policy must not fail");
        prop_assert!(
            !guard.profile_summary.is_empty(),
            "profile_summary was empty for {:?}", component
        );
    }
}
