use bitnet_startup_contract_diagnostics::{
    ContractPolicy, RuntimeComponent, StartupContractReport,
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

proptest! {
    /// `evaluate()` with Observe policy never returns an error for any component.
    #[test]
    fn evaluate_observe_never_errors(component in arb_component()) {
        let result = StartupContractReport::evaluate(component, ContractPolicy::Observe);
        prop_assert!(result.is_ok(), "evaluate returned Err for {:?}: {:?}", component, result);
    }

    /// `profile_summary()` always contains the "scenario=" and "environment=" keys.
    #[test]
    fn profile_summary_has_required_keys(component in arb_component()) {
        let report = StartupContractReport::evaluate(component, ContractPolicy::Observe)
            .expect("Observe policy must not fail");
        let summary = report.profile_summary();
        prop_assert!(
            summary.contains("scenario="),
            "profile_summary missing 'scenario=': {:?}", summary
        );
        prop_assert!(
            summary.contains("environment="),
            "profile_summary missing 'environment=': {:?}", summary
        );
    }

    /// `info` is always non-empty after a successful evaluation.
    #[test]
    fn info_is_non_empty_after_evaluate(component in arb_component()) {
        let report = StartupContractReport::evaluate(component, ContractPolicy::Observe)
            .expect("Observe policy must not fail");
        prop_assert!(
            !report.info.is_empty(),
            "info vec was empty for {:?}", component
        );
    }

    /// A compatible contract never produces warnings about non-compliance.
    #[test]
    fn compatible_contract_has_no_compliance_warnings(component in arb_component()) {
        let report = StartupContractReport::evaluate(component, ContractPolicy::Observe)
            .expect("Observe policy must not fail");
        if report.contract.is_compatible() {
            for warning in &report.warnings {
                prop_assert!(
                    !warning.contains("non-compliant"),
                    "compatible contract produced non-compliant warning: {:?}", warning
                );
            }
        }
    }
}
