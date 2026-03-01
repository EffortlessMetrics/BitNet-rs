//! Edge-case tests for bitnet-startup-contract-guard StartupContractGuard.

use bitnet_startup_contract_guard::{
    ContractPolicy, RuntimeComponent, StartupContractGuard, evaluate_and_emit,
};

// ---------------------------------------------------------------------------
// StartupContractGuard: evaluate with Observe
// ---------------------------------------------------------------------------

#[test]
fn guard_evaluate_cli_observe_succeeds() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    assert!(matches!(guard.component, RuntimeComponent::Cli));
    assert!(matches!(guard.policy, ContractPolicy::Observe));
}

#[test]
fn guard_evaluate_server_observe_succeeds() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Server, ContractPolicy::Observe).unwrap();
    assert!(matches!(guard.component, RuntimeComponent::Server));
}

#[test]
fn guard_evaluate_test_observe_succeeds() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Test, ContractPolicy::Observe).unwrap();
    assert!(matches!(guard.component, RuntimeComponent::Test));
}

// ---------------------------------------------------------------------------
// StartupContractGuard: evaluate with Enforce
// ---------------------------------------------------------------------------

#[test]
fn guard_evaluate_cli_enforce_succeeds() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Enforce).unwrap();
    assert!(matches!(guard.policy, ContractPolicy::Enforce));
}

#[test]
fn guard_evaluate_test_enforce_succeeds() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Test, ContractPolicy::Enforce).unwrap();
    assert!(matches!(guard.component, RuntimeComponent::Test));
    assert!(matches!(guard.policy, ContractPolicy::Enforce));
}

// ---------------------------------------------------------------------------
// StartupContractGuard: feature_line
// ---------------------------------------------------------------------------

#[test]
fn guard_feature_line_not_empty() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    assert!(!guard.feature_line.is_empty());
}

// ---------------------------------------------------------------------------
// StartupContractGuard: profile_summary
// ---------------------------------------------------------------------------

#[test]
fn guard_profile_summary_not_empty() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    assert!(!guard.profile_summary.is_empty());
}

// ---------------------------------------------------------------------------
// StartupContractGuard: is_compatible
// ---------------------------------------------------------------------------

#[test]
fn guard_is_compatible_callable() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    // Just verify it does not panic
    let _ = guard.is_compatible();
}

// ---------------------------------------------------------------------------
// StartupContractGuard: emit_to_tracing (no panic)
// ---------------------------------------------------------------------------

#[test]
fn guard_emit_to_tracing_observe_no_panic() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    guard.emit_to_tracing();
}

#[test]
fn guard_emit_to_tracing_enforce_no_panic() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Test, ContractPolicy::Enforce).unwrap();
    guard.emit_to_tracing();
}

// ---------------------------------------------------------------------------
// StartupContractGuard: report fields
// ---------------------------------------------------------------------------

#[test]
fn guard_report_info_populated() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    // Info should contain at least one diagnostic line
    assert!(!guard.report.info.is_empty());
}

#[test]
fn guard_report_contract_accessible() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let _ = guard.report.contract.summary();
}

// ---------------------------------------------------------------------------
// StartupContractGuard: profile_violations
// ---------------------------------------------------------------------------

#[test]
fn guard_profile_violations_is_option() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    // May be Some or None; just verify accessible
    match &guard.profile_violations {
        Some((missing, forbidden)) => {
            let _ = missing.len();
            let _ = forbidden.len();
        }
        None => {}
    }
}

// ---------------------------------------------------------------------------
// StartupContractGuard: Debug
// ---------------------------------------------------------------------------

#[test]
fn guard_debug_not_empty() {
    let guard =
        StartupContractGuard::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    let d = format!("{:?}", guard);
    assert!(d.contains("StartupContractGuard"));
}

// ---------------------------------------------------------------------------
// evaluate_and_emit convenience
// ---------------------------------------------------------------------------

#[test]
fn evaluate_and_emit_cli_observe_succeeds() {
    let guard = evaluate_and_emit(RuntimeComponent::Cli, ContractPolicy::Observe).unwrap();
    assert!(matches!(guard.component, RuntimeComponent::Cli));
}

#[test]
fn evaluate_and_emit_server_observe_succeeds() {
    let guard = evaluate_and_emit(RuntimeComponent::Server, ContractPolicy::Observe).unwrap();
    assert!(matches!(guard.component, RuntimeComponent::Server));
}

#[test]
fn evaluate_and_emit_test_enforce_succeeds() {
    let guard = evaluate_and_emit(RuntimeComponent::Test, ContractPolicy::Enforce).unwrap();
    assert!(matches!(guard.component, RuntimeComponent::Test));
    assert!(matches!(guard.policy, ContractPolicy::Enforce));
}

// ---------------------------------------------------------------------------
// All components × both policies
// ---------------------------------------------------------------------------

#[test]
fn all_component_policy_combinations() {
    let components = [RuntimeComponent::Cli, RuntimeComponent::Server, RuntimeComponent::Test];
    let policies = [ContractPolicy::Observe, ContractPolicy::Enforce];
    for component in &components {
        for policy in &policies {
            let guard = StartupContractGuard::evaluate(*component, *policy).unwrap();
            let _ = guard.is_compatible();
            guard.emit_to_tracing();
        }
    }
}
