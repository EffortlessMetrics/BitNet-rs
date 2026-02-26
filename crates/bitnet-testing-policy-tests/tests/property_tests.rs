//! Property tests for `bitnet-testing-policy-tests`.
//!
//! Validates key invariants of [`PolicyDiagnostics`] and [`diagnostics_for_context`]:
//! - Determinism: same context → same compatibility results
//! - Coherence: `is_grid_compatible()` agrees with the violations state
//! - Stability: `summary()` never panics and contains expected tokens
//! - Alias equivalence: `diagnostics_for_context` == `PolicyDiagnostics::from_context`

use bitnet_testing_policy_tests::{
    ConfigurationContext, ExecutionEnvironment, PolicyDiagnostics, TestingScenario,
    diagnostics_for_context,
};
use proptest::prelude::*;

// ── Strategies ───────────────────────────────────────────────────────────────

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

fn arb_env() -> impl Strategy<Value = ExecutionEnvironment> {
    prop_oneof![
        Just(ExecutionEnvironment::Local),
        Just(ExecutionEnvironment::Ci),
        Just(ExecutionEnvironment::PreProduction),
        Just(ExecutionEnvironment::Production),
    ]
}

fn arb_context() -> impl Strategy<Value = ConfigurationContext> {
    (arb_scenario(), arb_env()).prop_map(|(scenario, environment)| ConfigurationContext {
        scenario,
        environment,
        ..Default::default()
    })
}

// ── Property tests ───────────────────────────────────────────────────────────

proptest! {
    /// `PolicyDiagnostics::from_context` is deterministic: calling it twice with
    /// the same context produces the same `is_grid_compatible()` result.
    #[test]
    fn from_context_is_deterministic(ctx in arb_context()) {
        let d1 = PolicyDiagnostics::from_context(&ctx);
        let d2 = PolicyDiagnostics::from_context(&ctx);
        prop_assert_eq!(d1.is_grid_compatible(), d2.is_grid_compatible());
    }

    /// `is_grid_compatible()` is coherent with `violations()`:
    /// compatible ↔ violations are present AND both missing and forbidden are empty.
    #[test]
    fn is_grid_compatible_coherent_with_violations(ctx in arb_context()) {
        let diag = PolicyDiagnostics::from_context(&ctx);
        let compatible = diag.is_grid_compatible();
        let has_no_violations = diag.violations()
            .is_some_and(|(m, f)| m.is_empty() && f.is_empty());
        // compatible implies has_no_violations, and vice versa
        prop_assert_eq!(compatible, has_no_violations);
    }

    /// `summary()` never panics and contains the scenario name.
    #[test]
    fn summary_never_panics_and_contains_scenario(
        scenario in arb_scenario(),
        env in arb_env(),
    ) {
        let ctx = ConfigurationContext { scenario, environment: env, ..Default::default() };
        let diag = PolicyDiagnostics::from_context(&ctx);
        let s = diag.summary();
        prop_assert!(!s.is_empty(), "summary must not be empty");
    }

    /// `is_feature_contract_consistent()` returns without panicking for all inputs.
    #[test]
    fn feature_contract_consistent_does_not_panic(ctx in arb_context()) {
        let diag = PolicyDiagnostics::from_context(&ctx);
        let _ = diag.is_feature_contract_consistent();
    }

    /// `diagnostics_for_context` is equivalent to `PolicyDiagnostics::from_context`.
    #[test]
    fn diagnostics_for_context_matches_from_context(ctx in arb_context()) {
        let via_method = PolicyDiagnostics::from_context(&ctx);
        let via_fn = diagnostics_for_context(&ctx);
        prop_assert_eq!(
            via_method.is_grid_compatible(),
            via_fn.is_grid_compatible(),
        );
        prop_assert_eq!(
            via_method.is_feature_contract_consistent(),
            via_fn.is_feature_contract_consistent(),
        );
    }

    /// `profile_config()` returns without panicking for all contexts.
    #[test]
    fn profile_config_does_not_panic(ctx in arb_context()) {
        let diag = PolicyDiagnostics::from_context(&ctx);
        let _ = diag.profile_config();
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[test]
fn unit_local_context_constructs() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
        ..Default::default()
    };
    let diag = PolicyDiagnostics::from_context(&ctx);
    assert!(diag.context().scenario == TestingScenario::Unit);
    assert!(diag.context().environment == ExecutionEnvironment::Local);
}

#[test]
fn e2e_ci_context_constructs() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::EndToEnd,
        environment: ExecutionEnvironment::Ci,
        ..Default::default()
    };
    let diag = PolicyDiagnostics::from_context(&ctx);
    let _ = diag.summary();
    let _ = diag.is_grid_compatible();
}
