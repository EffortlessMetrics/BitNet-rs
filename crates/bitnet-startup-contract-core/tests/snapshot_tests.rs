use bitnet_startup_contract_core::{ContractPolicy, ProfileContract, RuntimeComponent};

#[test]
fn runtime_component_labels() {
    let labels: Vec<_> = [
        RuntimeComponent::Cli,
        RuntimeComponent::Server,
        RuntimeComponent::Test,
        RuntimeComponent::Custom,
    ]
    .iter()
    .map(|c| c.label())
    .collect();
    insta::assert_snapshot!(labels.join("\n"));
}

#[test]
fn test_component_observe_summary_contains_state() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Test, ContractPolicy::Observe);
    let summary = contract.summary();
    // Pin that summary is non-empty and contains key structural tokens
    insta::assert_snapshot!(format!(
        "has_component={} has_state={} has_scenario={}",
        summary.contains("test"),
        summary.contains("state="),
        summary.contains("scenario=")
    ));
}

#[test]
fn cli_component_observe_is_compatible_or_has_state() {
    let contract = ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    // Pin the compatible field value so it doesn't silently change
    insta::assert_snapshot!(format!("is_compatible={}", contract.is_compatible()));
}
