//! Core runtime context resolution for BitNet profile contracts.
//!
//! This crate owns the low-level parsing and defaulting semantics used by
//! consumers that evaluate profiles and startup contracts.

use std::env;

pub use bitnet_bdd_grid_core::{ExecutionEnvironment, TestingScenario};

/// Runtime context selector for a scenario/environment pair.
#[derive(Debug, Clone, Copy)]
pub struct ActiveContext {
    /// Logical test scenario under evaluation.
    pub scenario: TestingScenario,
    /// Execution environment under evaluation.
    pub environment: ExecutionEnvironment,
}

impl Default for ActiveContext {
    fn default() -> Self {
        Self::from_env()
    }
}

impl ActiveContext {
    /// Resolve runtime context from environment variables.
    ///
    /// - `BITNET_TEST_SCENARIO` -> explicit scenario override.
    /// - `CI` / `GITHUB_ACTIONS` -> CI environment when no explicit environment is set.
    /// - `BITNET_ENV` / `BITNET_TEST_ENV` -> explicit environment override.
    pub fn from_env() -> Self {
        Self::from_env_with_defaults(TestingScenario::Unit, ExecutionEnvironment::Local)
    }

    /// Resolve runtime context from environment variables with explicit defaults.
    ///
    /// This preserves shared resolution semantics while allowing components to
    /// provide tailored default scenario/environment values.
    pub fn from_env_with_defaults(
        default_scenario: TestingScenario,
        default_environment: ExecutionEnvironment,
    ) -> Self {
        let scenario = env::var("BITNET_TEST_SCENARIO")
            .ok()
            .and_then(|value| value.parse::<TestingScenario>().ok())
            .unwrap_or(default_scenario);

        let env_in_ci = env::var("CI").is_ok() || env::var("GITHUB_ACTIONS").is_ok();
        let env_override = env::var("BITNET_ENV").ok().or_else(|| env::var("BITNET_TEST_ENV").ok());

        let environment = env_override
            .as_deref()
            .and_then(|value| value.parse::<ExecutionEnvironment>().ok())
            .or(if env_in_ci { Some(ExecutionEnvironment::Ci) } else { None })
            .unwrap_or(default_environment);

        Self { scenario, environment }
    }
}

#[cfg(test)]
mod tests {
    use super::{ActiveContext, ExecutionEnvironment, TestingScenario};
    use serial_test::serial;

    #[test]
    #[serial(bitnet_env)]
    fn from_env_uses_bitnet_env_over_ci() {
        temp_env::with_vars(
            [("BITNET_ENV", Some("production")), ("CI", Some("true")), ("BITNET_TEST_ENV", None)],
            || {
                let context = ActiveContext::from_env();
                assert_eq!(context.scenario, TestingScenario::Unit);
                assert_eq!(context.environment, ExecutionEnvironment::Production);
            },
        );
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_uses_bitnet_test_env_when_bitnet_env_absent() {
        temp_env::with_vars(
            [("BITNET_TEST_ENV", Some("pre-prod")), ("BITNET_ENV", None), ("CI", None)],
            || {
                let context = ActiveContext::from_env();
                assert_eq!(context.environment, ExecutionEnvironment::PreProduction);
            },
        );
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_falls_back_to_ci_when_no_env_explicitly_set() {
        temp_env::with_vars(
            [("CI", Some("1")), ("BITNET_ENV", None), ("BITNET_TEST_ENV", None)],
            || {
                let context = ActiveContext::from_env();
                assert_eq!(context.environment, ExecutionEnvironment::Ci);
            },
        );
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_with_defaults_uses_component_defaults() {
        // Clear CI env var so the default (Local) is respected.
        temp_env::with_vars(
            [("CI", None::<&str>), ("BITNET_ENV", None), ("BITNET_TEST_ENV", None)],
            || {
                let context = ActiveContext::from_env_with_defaults(
                    TestingScenario::Integration,
                    ExecutionEnvironment::Local,
                );
                assert_eq!(context.scenario, TestingScenario::Integration);
                assert_eq!(context.environment, ExecutionEnvironment::Local);
            },
        );
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_with_defaults_prefers_ci_when_no_override() {
        temp_env::with_vars(
            [("CI", Some("1")), ("BITNET_ENV", None), ("BITNET_TEST_ENV", None)],
            || {
                let context = ActiveContext::from_env_with_defaults(
                    TestingScenario::Integration,
                    ExecutionEnvironment::Local,
                );
                assert_eq!(context.environment, ExecutionEnvironment::Ci);
            },
        );
    }
}
