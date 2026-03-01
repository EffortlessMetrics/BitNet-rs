//! Edge-case tests for bitnet-runtime-context-core ActiveContext resolution.

use bitnet_runtime_context_core::{ActiveContext, ExecutionEnvironment, TestingScenario};
use serial_test::serial;

// ---------------------------------------------------------------------------
// ActiveContext: from_env defaults (with env fully cleared)
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_defaults_to_unit_local_when_no_env() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", None),
            ("BITNET_TEST_ENV", None),
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

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_override() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("integration")),
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::Integration);
            assert_eq!(ctx.environment, ExecutionEnvironment::Local);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_e2e_alias() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("e2e")),
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::EndToEnd);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_perf_alias() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("perf")),
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::Performance);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_scenario_crossval_alias() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("crossval")),
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.scenario, TestingScenario::CrossValidation);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_invalid_scenario_falls_back_to_default() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", Some("not-a-scenario")),
            ("BITNET_ENV", None::<&str>),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            // Invalid scenario should fall back to default (Unit)
            assert_eq!(ctx.scenario, TestingScenario::Unit);
        },
    );
}

// ---------------------------------------------------------------------------
// ActiveContext: environment resolution priority
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_bitnet_env_takes_priority_over_ci() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", Some("production")),
            ("BITNET_TEST_ENV", None),
            ("CI", Some("true")),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Production);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_bitnet_test_env_used_when_bitnet_env_absent() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", None),
            ("BITNET_TEST_ENV", Some("pre-prod")),
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
fn from_env_bitnet_env_priority_over_bitnet_test_env() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", Some("production")),
            ("BITNET_TEST_ENV", Some("ci")),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            // BITNET_ENV takes priority over BITNET_TEST_ENV
            assert_eq!(ctx.environment, ExecutionEnvironment::Production);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_ci_var_detected() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", None),
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
fn from_env_github_actions_detected() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", None),
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
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", None),
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

#[test]
#[serial(bitnet_env)]
fn from_env_invalid_env_falls_back_to_ci_or_local() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", Some("not-valid")),
            ("BITNET_TEST_ENV", None),
            ("CI", Some("1")),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            // Invalid BITNET_ENV should be ignored, CI takes effect
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_invalid_env_no_ci_falls_back_to_local() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", Some("not-valid")),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            // Invalid BITNET_ENV, no CI -> Local default
            assert_eq!(ctx.environment, ExecutionEnvironment::Local);
        },
    );
}

// ---------------------------------------------------------------------------
// ActiveContext: from_env_with_defaults
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_with_defaults_uses_provided_defaults() {
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
fn from_env_with_defaults_scenario_env_override_takes_priority() {
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
                ExecutionEnvironment::Production,
            );
            // Env override should win over provided default
            assert_eq!(ctx.scenario, TestingScenario::Smoke);
            assert_eq!(ctx.environment, ExecutionEnvironment::Production);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_with_defaults_ci_overrides_default_environment() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", None),
            ("BITNET_TEST_ENV", None),
            ("CI", Some("1")),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env_with_defaults(
                TestingScenario::Unit,
                ExecutionEnvironment::Production,
            );
            // CI should override the provided default environment
            assert_eq!(ctx.environment, ExecutionEnvironment::Ci);
        },
    );
}

// ---------------------------------------------------------------------------
// ActiveContext: Default impl
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn default_impl_matches_from_env() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", None),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let default_ctx = ActiveContext::default();
            let from_env = ActiveContext::from_env();
            assert_eq!(default_ctx.scenario, from_env.scenario);
            assert_eq!(default_ctx.environment, from_env.environment);
        },
    );
}

// ---------------------------------------------------------------------------
// ActiveContext: Debug/Clone/Copy
// ---------------------------------------------------------------------------

#[test]
fn active_context_debug() {
    let ctx =
        ActiveContext { scenario: TestingScenario::Unit, environment: ExecutionEnvironment::Local };
    let d = format!("{:?}", ctx);
    assert!(d.contains("ActiveContext"));
    assert!(d.contains("Unit"));
    assert!(d.contains("Local"));
}

#[test]
fn active_context_clone_copy() {
    let ctx = ActiveContext {
        scenario: TestingScenario::Integration,
        environment: ExecutionEnvironment::Ci,
    };
    let ctx2 = ctx;
    assert_eq!(ctx.scenario, ctx2.scenario);
    assert_eq!(ctx.environment, ctx2.environment);
}

#[test]
fn active_context_fields_directly_accessible() {
    let ctx = ActiveContext {
        scenario: TestingScenario::EndToEnd,
        environment: ExecutionEnvironment::PreProduction,
    };
    assert_eq!(ctx.scenario, TestingScenario::EndToEnd);
    assert_eq!(ctx.environment, ExecutionEnvironment::PreProduction);
}

// ---------------------------------------------------------------------------
// Env var aliases
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn from_env_staging_alias() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", Some("staging")),
            ("BITNET_TEST_ENV", None),
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
fn from_env_prod_alias() {
    temp_env::with_vars(
        [
            ("BITNET_TEST_SCENARIO", None::<&str>),
            ("BITNET_ENV", Some("prod")),
            ("BITNET_TEST_ENV", None),
            ("CI", None),
            ("GITHUB_ACTIONS", None),
        ],
        || {
            let ctx = ActiveContext::from_env();
            assert_eq!(ctx.environment, ExecutionEnvironment::Production);
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn from_env_all_scenario_values() {
    let scenarios = [
        ("unit", TestingScenario::Unit),
        ("integration", TestingScenario::Integration),
        ("e2e", TestingScenario::EndToEnd),
        ("performance", TestingScenario::Performance),
        ("crossval", TestingScenario::CrossValidation),
        ("smoke", TestingScenario::Smoke),
        ("development", TestingScenario::Development),
        ("debug", TestingScenario::Debug),
        ("minimal", TestingScenario::Minimal),
    ];
    for (name, expected) in &scenarios {
        temp_env::with_vars(
            [
                ("BITNET_TEST_SCENARIO", Some(*name)),
                ("BITNET_ENV", None::<&str>),
                ("BITNET_TEST_ENV", None),
                ("CI", None),
                ("GITHUB_ACTIONS", None),
            ],
            || {
                let ctx = ActiveContext::from_env();
                assert_eq!(ctx.scenario, *expected, "failed for scenario name: {name}");
            },
        );
    }
}
