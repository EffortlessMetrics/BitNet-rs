//! Property and integration tests for `bitnet-runtime-context`.
//!
//! The crate is a façade over `bitnet-runtime-context-core`, which owns
//! `ActiveContext` and environment-variable parsing logic.

use bitnet_runtime_context::{ActiveContext, ExecutionEnvironment, TestingScenario};
use proptest::prelude::*;
use serial_test::serial;

// ── smoke tests ─────────────────────────────────────────────────────────────

#[test]
fn from_env_does_not_panic() {
    let ctx = ActiveContext::from_env();
    assert!(!ctx.scenario.to_string().is_empty());
    assert!(!ctx.environment.to_string().is_empty());
}

#[test]
fn default_equals_from_env() {
    let a = ActiveContext::from_env();
    let b = ActiveContext::default();
    assert_eq!(a.scenario, b.scenario);
    assert_eq!(a.environment, b.environment);
}

#[test]
#[serial(bitnet_env)]
fn from_env_with_defaults_uses_fallbacks_when_env_vars_absent() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", None),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env_with_defaults(
                TestingScenario::Smoke,
                ExecutionEnvironment::Local,
            );
            assert_eq!(ctx.scenario, TestingScenario::Smoke);
            assert_eq!(ctx.environment, ExecutionEnvironment::Local);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn bitnet_env_overrides_ci_flag() {
    temp_env::with_vars(
        [
            ("CI", Some("1")),
            ("BITNET_ENV", Some("production")),
            ("BITNET_TEST_ENV", None::<&str>),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Production);
        },
    );
}

// ── proptest invariants ──────────────────────────────────────────────────────

const SCENARIOS: [TestingScenario; 9] = [
    TestingScenario::Unit,
    TestingScenario::Integration,
    TestingScenario::EndToEnd,
    TestingScenario::Performance,
    TestingScenario::CrossValidation,
    TestingScenario::Smoke,
    TestingScenario::Development,
    TestingScenario::Debug,
    TestingScenario::Minimal,
];

const ENVS: [ExecutionEnvironment; 4] = [
    ExecutionEnvironment::Local,
    ExecutionEnvironment::Ci,
    ExecutionEnvironment::PreProduction,
    ExecutionEnvironment::Production,
];

proptest! {
    #[test]
    fn testing_scenario_display_roundtrip(idx in 0usize..9) {
        let s = SCENARIOS[idx];
        let displayed = s.to_string();
        let parsed: TestingScenario = displayed.parse().expect("display should be parseable");
        prop_assert_eq!(s, parsed);
    }

    #[test]
    fn execution_environment_display_roundtrip(idx in 0usize..4) {
        let e = ENVS[idx];
        let displayed = e.to_string();
        let parsed: ExecutionEnvironment = displayed.parse().expect("display should be parseable");
        prop_assert_eq!(e, parsed);
    }

    #[test]
    fn from_env_with_defaults_always_produces_valid_strings(
        scenario_idx in 0usize..9,
        env_idx in 0usize..4,
    ) {
        let ctx = ActiveContext::from_env_with_defaults(SCENARIOS[scenario_idx], ENVS[env_idx]);
        prop_assert!(!ctx.scenario.to_string().is_empty());
        prop_assert!(!ctx.environment.to_string().is_empty());
    }
}
