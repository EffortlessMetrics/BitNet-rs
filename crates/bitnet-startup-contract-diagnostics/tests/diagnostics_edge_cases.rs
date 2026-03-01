//! Edge-case tests for bitnet-startup-contract-diagnostics report generation.

use bitnet_startup_contract_diagnostics::{
    ContractPolicy, RuntimeComponent, StartupContractReport,
};

// ---------------------------------------------------------------------------
// StartupContractReport: evaluate with Observe policy
// ---------------------------------------------------------------------------

#[test]
fn report_evaluate_cli_observe_ok() {
    let result = StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
    assert!(result.is_ok());
}

#[test]
fn report_evaluate_server_observe_ok() {
    let result = StartupContractReport::evaluate(RuntimeComponent::Server, ContractPolicy::Observe);
    assert!(result.is_ok());
}

#[test]
fn report_evaluate_test_observe_ok() {
    let result = StartupContractReport::evaluate(RuntimeComponent::Test, ContractPolicy::Observe);
    assert!(result.is_ok());
}

#[test]
fn report_evaluate_custom_observe_ok() {
    let result = StartupContractReport::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
    assert!(result.is_ok());
}

#[test]
fn report_info_is_non_empty() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    assert!(!report.info.is_empty(), "info messages should be populated");
}

#[test]
fn report_info_contains_summary() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let has_summary = report.info.iter().any(|line| line.contains("bitnet-cli"));
    assert!(has_summary, "info should contain component name");
}

#[test]
fn report_info_contains_profile_summary() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let has_profile = report.info.iter().any(|line| line.contains("Profile summary:"));
    assert!(has_profile, "info should contain 'Profile summary:'");
}

// ---------------------------------------------------------------------------
// StartupContractReport: profile_summary format
// ---------------------------------------------------------------------------

#[test]
fn report_profile_summary_contains_scenario() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let summary = report.profile_summary();
    assert!(summary.contains("scenario="), "should contain 'scenario=': {summary}");
}

#[test]
fn report_profile_summary_contains_environment() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let summary = report.profile_summary();
    assert!(summary.contains("environment="), "should contain 'environment=': {summary}");
}

#[test]
fn report_profile_summary_contains_required() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let summary = report.profile_summary();
    assert!(summary.contains("required="), "should contain 'required=': {summary}");
}

#[test]
fn report_profile_summary_contains_optional() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let summary = report.profile_summary();
    assert!(summary.contains("optional="), "should contain 'optional=': {summary}");
}

#[test]
fn report_profile_summary_contains_forbidden() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let summary = report.profile_summary();
    assert!(summary.contains("forbidden="), "should contain 'forbidden=': {summary}");
}

// ---------------------------------------------------------------------------
// StartupContractReport: contract field access
// ---------------------------------------------------------------------------

#[test]
fn report_contract_is_accessible() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    // Just verify we can access the contract field
    let _compatible = report.contract.is_compatible();
}

#[test]
fn report_contract_context_valid() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let ctx = report.contract.context();
    assert!(!ctx.scenario.to_string().is_empty());
    assert!(!ctx.environment.to_string().is_empty());
}

// ---------------------------------------------------------------------------
// All components produce consistent reports
// ---------------------------------------------------------------------------

#[test]
fn all_components_produce_reports() {
    let components = [
        RuntimeComponent::Cli,
        RuntimeComponent::Server,
        RuntimeComponent::Test,
        RuntimeComponent::Custom,
    ];
    for comp in &components {
        let result = StartupContractReport::evaluate(*comp, ContractPolicy::Observe);
        assert!(result.is_ok(), "evaluate failed for {:?}", comp);
        let report = result.unwrap();
        assert!(!report.info.is_empty(), "info empty for {:?}", comp);
    }
}

#[test]
fn all_components_profile_summaries_are_non_empty() {
    let components = [
        RuntimeComponent::Cli,
        RuntimeComponent::Server,
        RuntimeComponent::Test,
        RuntimeComponent::Custom,
    ];
    for comp in &components {
        let report = StartupContractReport::evaluate(*comp, ContractPolicy::Observe).unwrap();
        let summary = report.profile_summary();
        assert!(!summary.is_empty(), "empty profile summary for {:?}", comp);
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

#[test]
fn report_debug_not_empty() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let d = format!("{:?}", report);
    assert!(!d.is_empty());
    assert!(d.contains("StartupContractReport"));
}

// ---------------------------------------------------------------------------
// Warnings should be present or absent depending on compatibility
// ---------------------------------------------------------------------------

#[test]
fn report_warnings_consistent_with_compatibility() {
    let report =
        StartupContractReport::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    if report.contract.is_compatible() {
        // Compatible contracts should have no non-compliance warnings
        let has_non_compliant = report.warnings.iter().any(|w| w.contains("non-compliant"));
        assert!(!has_non_compliant, "compatible contract should not have non-compliant warnings");
    }
}
