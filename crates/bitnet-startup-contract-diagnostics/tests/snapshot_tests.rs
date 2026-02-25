use bitnet_runtime_bootstrap::{ContractPolicy, RuntimeComponent};
use bitnet_startup_contract_diagnostics::StartupContractReport;

#[test]
fn startup_contract_report_profile_summary_format() {
    // Regression: profile_summary() must always contain field names
    let report = StartupContractReport::evaluate(RuntimeComponent::Test, ContractPolicy::Observe)
        .expect("Observe policy never fails");
    let summary = report.profile_summary();
    insta::assert_snapshot!(summary);
}

#[test]
fn startup_contract_report_info_non_empty() {
    let report = StartupContractReport::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe)
        .expect("Observe policy never fails");
    // info must always have at least 2 lines (contract summary + profile summary)
    insta::assert_snapshot!(&report.info.len().to_string());
}

#[test]
fn startup_contract_report_compatible_has_no_warnings() {
    let report = StartupContractReport::evaluate(RuntimeComponent::Test, ContractPolicy::Observe)
        .expect("Observe policy never fails");
    // A compatible contract should produce no warnings
    insta::assert_snapshot!(&report.warnings.len().to_string());
}
