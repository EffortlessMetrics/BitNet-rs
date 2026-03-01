//! Edge-case tests for `bitnet-runtime-context-core`.
//!
//! Coverage:
//! - ActiveContext::from_env — env var priority (BITNET_ENV > BITNET_TEST_ENV > CI > default)
//! - ActiveContext::from_env_with_defaults — explicit defaults honored
//! - BITNET_TEST_SCENARIO env var — scenario override
//! - GITHUB_ACTIONS env var — CI detection fallback
//! - Invalid env values silently fall back to defaults
//! - Debug, Clone, Copy, Default traits

use bitnet_runtime_context_core::{ActiveContext, ExecutionEnvironment, TestingScenario};
use serial_test::serial;

// ---------------------------------------------------------------------------
// from_env — no env vars (clean slate)
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_clean_slate_uses_unit_local() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("BITNET_TEST_SCENARIO", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::Unit);
            assert_eq!(ctx.environment, ExecutionEnvironment::Local);
        },
    );
}

// ---------------------------------------------------------------------------
// from_env — BITNET_ENV takes priority
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_bitnet_env_production() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", Some("production")),
            ("BITNET_TEST_ENV", Some("local")),
            ("CI", Some("true")),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Production);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_bitnet_env_local() {
    temp_env::with_vars(
        [("BITNET_ENV", Some("local")), ("CI", Some("true")), ("BITNET_TEST_ENV", None::<&str>)],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Local);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_bitnet_env_preprod() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", Some("pre-prod")),
            ("CI", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::PreProduction);
        },
    );
}

// ---------------------------------------------------------------------------
// from_env — BITNET_TEST_ENV fallback (when BITNET_ENV absent)
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_test_env_staging() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", Some("staging")),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::PreProduction);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_test_env_ci() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", Some("ci")),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

// ---------------------------------------------------------------------------
// from_env — CI/GITHUB_ACTIONS detection
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_ci_env_var() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", Some("1")),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_github_actions_env_var() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", Some("true")),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_both_ci_and_github_actions() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", Some("true")),
            ("GITHUB_ACTIONS", Some("true")),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

// ---------------------------------------------------------------------------
// from_env — BITNET_TEST_SCENARIO
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_override_integration() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("integration")),
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::Integration);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_override_e2e() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("e2e")),
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::EndToEnd);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_override_perf() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("perf")),
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::Performance);
        },
    );
}

// ---------------------------------------------------------------------------
// from_env — invalid values fall back to defaults
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_invalid_scenario_falls_back() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("totally-invalid")),
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::Unit); // default
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_invalid_env_with_ci_set() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", Some("not-a-valid-env")),
            ("BITNET_TEST_ENV", None::<&str>),
            ("CI", Some("1")),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            // Invalid BITNET_ENV → falls through, CI=1 → CI
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_empty_string_env_falls_through() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", Some("")),
            ("BITNET_TEST_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            // Empty string is "set" but invalid → falls through to default
            assert_eq!(ctx.environment, ExecutionEnvironment::Local);
        },
    );
}

// ---------------------------------------------------------------------------
// from_env_with_defaults
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_with_defaults_uses_provided_defaults() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env_with_defaults(
                TestingScenario::Performance,
                ExecutionEnvironment::Production,
            );
            assert_eq!(ctx.scenario, TestingScenario::Performance);
            assert_eq!(ctx.environment, ExecutionEnvironment::Production);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_with_defaults_env_override_beats_default() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", Some("ci")),
            ("BITNET_TEST_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env_with_defaults(
                TestingScenario::Unit,
                ExecutionEnvironment::Production,
            );
            // BITNET_ENV="ci" should override the Production default
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_with_defaults_scenario_override_beats_default() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("smoke")),
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env_with_defaults(
                TestingScenario::Performance,
                ExecutionEnvironment::Local,
            );
            assert_eq!(ctx.scenario, TestingScenario::Smoke);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_with_defaults_ci_beats_explicit_default() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", Some("1")),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env_with_defaults(
                TestingScenario::Unit,
                ExecutionEnvironment::Local,
            );
            // CI=1 should override the Local default
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

// ---------------------------------------------------------------------------
// Traits — Debug, Clone, Copy, Default
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn active_context_is_copy() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            let ctx2 = ctx; // Copy
            assert_eq!(ctx.scenario, ctx2.scenario);
            assert_eq!(ctx.environment, ctx2.environment);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn active_context_clone() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            let ctx2 = ctx.clone();
            assert_eq!(ctx.scenario, ctx2.scenario);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn active_context_debug() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            let dbg = format!("{:?}", ctx);
            assert!(dbg.contains("ActiveContext"));
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn active_context_default_matches_from_env() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx1 = ActiveContext::from_env();
            let ctx2 = ActiveContext::default();
            assert_eq!(ctx1.scenario, ctx2.scenario);
            assert_eq!(ctx1.environment, ctx2.environment);
        },
    );
}

// ---------------------------------------------------------------------------
// Scenario aliases in env var
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_alias_crossval() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("crossval")),
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::CrossValidation);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_alias_dev() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("dev")),
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::Development);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_alias_min() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("min")),
            ("BITNET_ENV", None::<&str>),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::Minimal);
        },
    );
}

// ---------------------------------------------------------------------------
// Environment aliases in env var
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_env_alias_cicd() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", Some("cicd")),
            ("CI", None::<&str>),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_env_alias_prod() {
    temp_env::with_vars(
        [
            ("BITNET_ENV", Some("prod")),
            ("CI", None::<&str>),
            ("GITHUB_ACTIONS", None),
            ("BITNET_TEST_ENV", None),
            ("BITNET_TEST_SCENARIO", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Production);
        },
    );
}
