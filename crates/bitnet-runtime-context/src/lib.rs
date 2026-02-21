//! Runtime environment context resolution for feature/profile compatibility checks.

use std::env;

pub use bitnet_bdd_grid::{ExecutionEnvironment, TestingScenario};

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
        let env_override = env::var("BITNET_ENV").ok().or_else(|_| env::var("BITNET_TEST_ENV").ok());

        let environment = env_override
            .as_deref()
            .and_then(|value| value.parse::<ExecutionEnvironment>().ok())
            .or_else(|| {
                if env_in_ci {
                    Some(ExecutionEnvironment::Ci)
                } else {
                    None
                }
            })
            .unwrap_or(default_environment);

        Self { scenario, environment }
    }
}

#[cfg(test)]
mod tests {
    use super::{ActiveContext, ExecutionEnvironment, TestingScenario};

    #[test]
    fn from_env_uses_bitnet_env_over_ci() {
        std::env::set_var("BITNET_ENV", "production");
        std::env::set_var("CI", "true");

        let context = ActiveContext::from_env();
        assert_eq!(context.scenario, TestingScenario::Unit);
        assert_eq!(context.environment, ExecutionEnvironment::Production);

        std::env::remove_var("BITNET_ENV");
        std::env::remove_var("CI");
    }

    #[test]
    fn from_env_uses_bitnet_test_env_when_bitnet_env_absent() {
        std::env::set_var("BITNET_TEST_ENV", "pre-prod");

        let context = ActiveContext::from_env();
        assert_eq!(context.environment, ExecutionEnvironment::PreProduction);

        std::env::remove_var("BITNET_TEST_ENV");
    }

    #[test]
    fn from_env_falls_back_to_ci_when_no_env_explicitly_set() {
        std::env::set_var("CI", "1");

        let context = ActiveContext::from_env();
        assert_eq!(context.environment, ExecutionEnvironment::Ci);

        std::env::remove_var("CI");
    }

    #[test]
    fn from_env_with_defaults_uses_component_defaults() {
        let context = ActiveContext::from_env_with_defaults(
            TestingScenario::Integration,
            ExecutionEnvironment::Local,
        );

        assert_eq!(context.scenario, TestingScenario::Integration);
        assert_eq!(context.environment, ExecutionEnvironment::Local);
    }

    #[test]
    fn from_env_with_defaults_prefers_ci_when_no_override() {
        std::env::set_var("CI", "1");

        let context = ActiveContext::from_env_with_defaults(
            TestingScenario::Integration,
            ExecutionEnvironment::Local,
        );
        assert_eq!(context.environment, ExecutionEnvironment::Ci);

        std::env::remove_var("CI");
    }
}

