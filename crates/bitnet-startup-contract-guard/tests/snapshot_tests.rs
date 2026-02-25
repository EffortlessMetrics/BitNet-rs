use bitnet_runtime_bootstrap::{ContractPolicy, RuntimeComponent};
use bitnet_startup_contract_guard::StartupContractGuard;

#[test]
fn startup_contract_guard_component_labels() {
    // Regression: component label strings must stay stable
    insta::assert_snapshot!({
        let labels = [
            RuntimeComponent::Cli.label(),
            RuntimeComponent::Server.label(),
            RuntimeComponent::Test.label(),
            RuntimeComponent::Custom.label(),
        ]
        .join("\n");
        labels
    });
}

#[test]
fn startup_contract_guard_is_compatible_with_observe() {
    let guard = StartupContractGuard::evaluate(RuntimeComponent::Test, ContractPolicy::Observe)
        .expect("Observe policy never fails");
    // A guard evaluated with Observe should always be compatible
    insta::assert_snapshot!(&guard.is_compatible().to_string());
}

#[test]
fn startup_contract_guard_feature_line_format() {
    let guard = StartupContractGuard::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe)
        .expect("Observe policy never fails");
    // feature_line must always start with "features:"
    insta::assert_snapshot!(&guard.feature_line[..9]);
}
