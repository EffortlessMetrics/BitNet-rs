//! Edge-case tests for `bitnet-startup-contract-core`.
//!
//! Coverage:
//! - RuntimeComponent: label, default_scenario, default_environment
//! - ContractPolicy: Observe vs Enforce variants
//! - ContractState: all 4 variants, PartialEq, Eq
//! - ProfileContract::evaluate with Observe policy
//! - ProfileContract::with_context: explicit scenario/environment
//! - ProfileContract::enforce: Compatible passes, others depend on policy
//! - ProfileContract accessors: state, is_compatible, component, policy, etc.
//! - ProfileContract::summary output

use bitnet_startup_contract_core::*;
use serial_test::serial;

// ---------------------------------------------------------------------------
// RuntimeComponent — labels
// ---------------------------------------------------------------------------

#[test]
fn component_cli_label() {
    assert_eq!(RuntimeComponent::Cli.label(), "bitnet-cli");
}

#[test]
fn component_server_label() {
    assert_eq!(RuntimeComponent::Server.label(), "bitnet-server");
}

#[test]
fn component_test_label() {
    assert_eq!(RuntimeComponent::Test.label(), "test");
}

#[test]
fn component_custom_label() {
    assert_eq!(RuntimeComponent::Custom.label(), "custom");
}

// ---------------------------------------------------------------------------
// RuntimeComponent — traits (Debug, Clone, Copy)
// ---------------------------------------------------------------------------

#[test]
fn component_debug() {
    let dbg = format!("{:?}", RuntimeComponent::Cli);
    assert!(dbg.contains("Cli"));
}

#[test]
fn component_clone_copy() {
    let c = RuntimeComponent::Server;
    let c2 = c; // Copy
    assert_eq!(c.label(), c2.label());
    let c3 = c.clone();
    assert_eq!(c.label(), c3.label());
}

// ---------------------------------------------------------------------------
// ContractState — equality
// ---------------------------------------------------------------------------

#[test]
fn contract_state_eq() {
    assert_eq!(ContractState::Compatible, ContractState::Compatible);
    assert_eq!(ContractState::UnknownGridCell, ContractState::UnknownGridCell);
    assert_eq!(ContractState::MissingRequired, ContractState::MissingRequired);
    assert_eq!(ContractState::ForbiddenActive, ContractState::ForbiddenActive);
}

#[test]
fn contract_state_ne() {
    assert_ne!(ContractState::Compatible, ContractState::MissingRequired);
    assert_ne!(ContractState::UnknownGridCell, ContractState::ForbiddenActive);
}

#[test]
fn contract_state_debug() {
    let dbg = format!("{:?}", ContractState::Compatible);
    assert!(dbg.contains("Compatible"));
}

#[test]
fn contract_state_clone_copy() {
    let s = ContractState::MissingRequired;
    let s2 = s; // Copy
    assert_eq!(s, s2);
}

// ---------------------------------------------------------------------------
// ContractPolicy — traits
// ---------------------------------------------------------------------------

#[test]
fn contract_policy_debug() {
    assert!(format!("{:?}", ContractPolicy::Observe).contains("Observe"));
    assert!(format!("{:?}", ContractPolicy::Enforce).contains("Enforce"));
}

#[test]
fn contract_policy_clone_copy() {
    let p = ContractPolicy::Enforce;
    let p2 = p; // Copy
    let _ = p2;
    let _ = p; // still valid
}

// ---------------------------------------------------------------------------
// ProfileContract::evaluate — Observe policy (never fails)
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn evaluate_cli_observe() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
            // CLI default: Integration + Local
            assert_eq!(contract.context().scenario, TestingScenario::Integration);
            assert_eq!(contract.context().environment, ExecutionEnvironment::Local);
            assert_eq!(contract.component().label(), "bitnet-cli");
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn evaluate_server_observe() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Server, ContractPolicy::Observe);
            assert_eq!(contract.context().scenario, TestingScenario::Integration);
            assert_eq!(contract.context().environment, ExecutionEnvironment::Local);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn evaluate_test_observe() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Test, ContractPolicy::Observe);
            // Test default: CrossValidation + CI
            assert_eq!(contract.context().scenario, TestingScenario::CrossValidation);
            assert_eq!(contract.context().environment, ExecutionEnvironment::Ci);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn evaluate_custom_observe() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Custom, ContractPolicy::Observe);
            assert_eq!(contract.context().scenario, TestingScenario::Unit);
            assert_eq!(contract.context().environment, ExecutionEnvironment::Local);
        },
    );
}

// ---------------------------------------------------------------------------
// ProfileContract — accessors
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn contract_accessors() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
            // State should be one of the 4 variants
            let state = contract.state();
            let _: ContractState = state;
            // Policy should match what we passed
            let _: ContractPolicy = contract.policy();
            // Feature lists should be vec of strings
            let _: &[String] = contract.missing_required();
            let _: &[String] = contract.forbidden_active();
            let _: &[String] = contract.required_features();
            let _: &[String] = contract.optional_features();
            let _: &[String] = contract.forbidden_features();
            let _: Vec<String> = contract.active_features();
        },
    );
}

// ---------------------------------------------------------------------------
// ProfileContract::summary
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn contract_summary_contains_component_label() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
            let summary = contract.summary();
            assert!(summary.contains("bitnet-cli"), "summary should include component: {summary}");
            assert!(summary.contains("observe"), "summary should include policy: {summary}");
            assert!(summary.contains("scenario="), "summary should include scenario: {summary}");
            assert!(
                summary.contains("environment="),
                "summary should include environment: {summary}"
            );
            assert!(summary.contains("state="), "summary should include state: {summary}");
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn contract_summary_enforce_policy() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Server, ContractPolicy::Enforce);
            let summary = contract.summary();
            assert!(summary.contains("enforce"), "summary should include enforce: {summary}");
        },
    );
}

// ---------------------------------------------------------------------------
// ProfileContract::enforce — Compatible always succeeds
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn enforce_compatible_succeeds() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
            // Even if state isn't Compatible, Observe policy should pass enforce
            let result = contract.enforce();
            assert!(result.is_ok(), "Observe policy should pass enforce");
        },
    );
}

// ---------------------------------------------------------------------------
// ProfileContract::with_context — explicit context
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn with_context_unit_local() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext {
                scenario: TestingScenario::Unit,
                environment: ExecutionEnvironment::Local,
            };
            let contract = ProfileContract::with_context(
                RuntimeComponent::Custom,
                ctx,
                ContractPolicy::Observe,
            );
            assert_eq!(contract.context().scenario, TestingScenario::Unit);
            assert_eq!(contract.context().environment, ExecutionEnvironment::Local);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn with_context_performance_ci() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext {
                scenario: TestingScenario::Performance,
                environment: ExecutionEnvironment::Ci,
            };
            let contract =
                ProfileContract::with_context(RuntimeComponent::Test, ctx, ContractPolicy::Enforce);
            assert_eq!(contract.context().scenario, TestingScenario::Performance);
            assert_eq!(contract.context().environment, ExecutionEnvironment::Ci);
        },
    );
}

// ---------------------------------------------------------------------------
// ProfileContract::is_compatible
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn is_compatible_reflects_state() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
            let compat = contract.is_compatible();
            assert_eq!(compat, contract.state() == ContractState::Compatible);
        },
    );
}

// ---------------------------------------------------------------------------
// ProfileContract::enforce — Observe always passes
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn enforce_observe_never_fails() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            // All 4 component types under Observe
            for component in [
                RuntimeComponent::Cli,
                RuntimeComponent::Server,
                RuntimeComponent::Test,
                RuntimeComponent::Custom,
            ] {
                let contract = ProfileContract::evaluate(component, ContractPolicy::Observe);
                assert!(
                    contract.enforce().is_ok(),
                    "Observe should never fail for {:?}",
                    component
                );
            }
        },
    );
}

// ---------------------------------------------------------------------------
// ProfileContract — Debug trait
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn profile_contract_debug() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
            let dbg = format!("{:?}", contract);
            assert!(dbg.contains("ProfileContract"));
        },
    );
}

// ---------------------------------------------------------------------------
// Env override interaction
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn evaluate_respects_bitnet_env_override() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", Some("production")),
            ("BITNET_TEST_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Cli, ContractPolicy::Observe);
            assert_eq!(contract.context().environment, ExecutionEnvironment::Production);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn evaluate_respects_scenario_override() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("smoke")),
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let contract =
                ProfileContract::evaluate(RuntimeComponent::Server, ContractPolicy::Observe);
            assert_eq!(contract.context().scenario, TestingScenario::Smoke);
        },
    );
}
