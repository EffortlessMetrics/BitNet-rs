//! Property-based tests for `bitnet-runtime-context`.
//!
//! Key invariants:
//! - `ActiveContext::from_env_with_defaults` respects the caller's defaults
//!   when no environment variables are set.
//! - Scenario and environment fields are always valid enum values.
//! - Default construction is deterministic.

use bitnet_runtime_context::{ActiveContext, ExecutionEnvironment, TestingScenario};
use proptest::prelude::*;
use serial_test::serial;

// ── Arbitrary strategies ──────────────────────────────────────────────────────

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

fn arb_environment() -> impl Strategy<Value = ExecutionEnvironment> {
    prop_oneof![
        Just(ExecutionEnvironment::Local),
        Just(ExecutionEnvironment::Ci),
        Just(ExecutionEnvironment::PreProduction),
        Just(ExecutionEnvironment::Production),
    ]
}

// ── from_env_with_defaults respects defaults ──────────────────────────────────

proptest! {
    /// When no override env vars are set, the returned context matches the
    /// caller-supplied defaults (modulo CI detection from the environment).
    #[test]
    #[serial(bitnet_env)]
    fn from_env_with_defaults_uses_supplied_scenario(
        scenario in arb_scenario(),
    ) {
        temp_env::with_vars(
            [
                ("BITNET_TEST_SCENARIO", None::<&str>),
                ("BITNET_ENV", None::<&str>),
                ("BITNET_TEST_ENV", None::<&str>),
                ("CI", None::<&str>),
                ("GITHUB_ACTIONS", None::<&str>),
            ],
            || {
                let ctx = ActiveContext::from_env_with_defaults(
                    scenario,
                    ExecutionEnvironment::Local,
                );
                prop_assert_eq!(ctx.scenario, scenario);
                Ok(())
            },
        )?;
    }

    /// When no override env vars are set, environment matches default.
    #[test]
    #[serial(bitnet_env)]
    fn from_env_with_defaults_uses_supplied_environment(
        env in arb_environment(),
    ) {
        temp_env::with_vars(
            [
                ("BITNET_TEST_SCENARIO", None::<&str>),
                ("BITNET_ENV", None::<&str>),
                ("BITNET_TEST_ENV", None::<&str>),
                ("CI", None::<&str>),
                ("GITHUB_ACTIONS", None::<&str>),
            ],
            || {
                let ctx = ActiveContext::from_env_with_defaults(
                    TestingScenario::Unit,
                    env,
                );
                prop_assert_eq!(ctx.environment, env);
                Ok(())
            },
        )?;
    }
}

// ── BITNET_TEST_SCENARIO override ─────────────────────────────────────────────

proptest! {
    /// Explicit BITNET_TEST_SCENARIO env var overrides any default.
    #[test]
    #[serial(bitnet_env)]
    fn env_var_overrides_default_scenario(
        default_scenario in arb_scenario(),
        override_scenario in arb_scenario(),
    ) {
        let override_str = override_scenario.to_string();
        temp_env::with_vars(
            [
                ("BITNET_TEST_SCENARIO", Some(override_str.as_str())),
                ("BITNET_ENV", None::<&str>),
                ("BITNET_TEST_ENV", None::<&str>),
                ("CI", None::<&str>),
                ("GITHUB_ACTIONS", None::<&str>),
            ],
            || {
                let ctx = ActiveContext::from_env_with_defaults(
                    default_scenario,
                    ExecutionEnvironment::Local,
                );
                prop_assert_eq!(ctx.scenario, override_scenario);
                Ok(())
            },
        )?;
    }
}

// ── ActiveContext Debug is non-empty ──────────────────────────────────────────

proptest! {
    #[test]
    #[serial(bitnet_env)]
    fn active_context_debug_is_nonempty(
        scenario in arb_scenario(),
        env in arb_environment(),
    ) {
        temp_env::with_vars(
            [
                ("BITNET_TEST_SCENARIO", None::<&str>),
                ("BITNET_ENV", None::<&str>),
                ("BITNET_TEST_ENV", None::<&str>),
                ("CI", None::<&str>),
                ("GITHUB_ACTIONS", None::<&str>),
            ],
            || {
                let ctx = ActiveContext::from_env_with_defaults(scenario, env);
                let debug = format!("{ctx:?}");
                prop_assert!(!debug.is_empty());
                Ok(())
            },
        )?;
    }
}
