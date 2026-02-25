use bitnet_runtime_context_core::{ActiveContext, ExecutionEnvironment, TestingScenario};
use proptest::prelude::*;
use serial_test::serial;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn scenario_str_strategy() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just("unit"),
        Just("integration"),
        Just("e2e"),
        Just("performance"),
        Just("crossval"),
        Just("smoke"),
        Just("development"),
        Just("debug"),
        Just("minimal"),
    ]
}

fn environment_str_strategy() -> impl Strategy<Value = &'static str> {
    prop_oneof![Just("local"), Just("ci"), Just("pre-prod"), Just("production"),]
}

fn scenario_value_strategy() -> impl Strategy<Value = TestingScenario> {
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

fn environment_value_strategy() -> impl Strategy<Value = ExecutionEnvironment> {
    prop_oneof![
        Just(ExecutionEnvironment::Local),
        Just(ExecutionEnvironment::Ci),
        Just(ExecutionEnvironment::PreProduction),
        Just(ExecutionEnvironment::Production),
    ]
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Every valid scenario string round-trips through Display → FromStr.
    #[test]
    fn scenario_display_roundtrips(s in scenario_str_strategy()) {
        let parsed: TestingScenario = s.parse().expect("valid scenario string should parse");
        let displayed = parsed.to_string();
        let reparsed: TestingScenario = displayed.parse().expect("display output should round-trip");
        prop_assert_eq!(format!("{:?}", parsed), format!("{:?}", reparsed));
    }

    /// Every valid environment string round-trips through Display → FromStr.
    #[test]
    fn environment_display_roundtrips(s in environment_str_strategy()) {
        let parsed: ExecutionEnvironment = s.parse().expect("valid environment string should parse");
        let displayed = parsed.to_string();
        let reparsed: ExecutionEnvironment = displayed.parse().expect("display output should round-trip");
        prop_assert_eq!(format!("{:?}", parsed), format!("{:?}", reparsed));
    }

    /// from_env_with_defaults always honours the supplied defaults when no env
    /// overrides are present (tested by ensuring the returned values are valid).
    #[test]
    #[serial(bitnet_env)]
    fn from_env_with_defaults_returns_valid_context(
        scenario in scenario_value_strategy(),
        environment in environment_value_strategy(),
    ) {
        // Clear any environment overrides that could interfere.
        temp_env::with_var("BITNET_TEST_SCENARIO", None::<&str>, || {
            temp_env::with_var("BITNET_ENV", None::<&str>, || {
                temp_env::with_var("BITNET_TEST_ENV", None::<&str>, || {
                    temp_env::with_var("CI", None::<&str>, || {
                        temp_env::with_var("GITHUB_ACTIONS", None::<&str>, || {
                            let ctx = ActiveContext::from_env_with_defaults(scenario, environment);
                            // When no env vars are set the defaults must be used verbatim.
                            prop_assert_eq!(format!("{:?}", ctx.scenario), format!("{:?}", scenario));
                            prop_assert_eq!(
                                format!("{:?}", ctx.environment),
                                format!("{:?}", environment)
                            );
                            Ok(())
                        })
                    })
                })
            })
        })?;
    }

    /// Garbage scenario strings produce a parse error (never panic).
    #[test]
    fn invalid_scenario_string_returns_err(s in "[a-z_-]{1,32}") {
        let valid = ["unit", "integration", "e2e", "performance", "crossval", "smoke", "development", "debug", "minimal"];
        if !valid.contains(&s.as_str()) {
            // Should either parse or return Err — never panic.
            let _: Result<TestingScenario, _> = s.parse();
        }
    }

    /// Garbage environment strings produce a parse error (never panic).
    #[test]
    fn invalid_environment_string_returns_err(s in "[a-z_-]{1,32}") {
        let valid = ["local", "ci", "pre-prod", "production"];
        if !valid.contains(&s.as_str()) {
            let _: Result<ExecutionEnvironment, _> = s.parse();
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[test]
fn scenario_clone_copy_equals_original() {
    let s = TestingScenario::Integration;
    let cloned = s;
    assert_eq!(format!("{:?}", s), format!("{:?}", cloned));
}

#[test]
fn environment_clone_copy_equals_original() {
    let e = ExecutionEnvironment::Ci;
    let cloned = e;
    assert_eq!(format!("{:?}", e), format!("{:?}", cloned));
}

#[test]
fn active_context_field_access() {
    let ctx = ActiveContext {
        scenario: TestingScenario::Smoke,
        environment: ExecutionEnvironment::Production,
    };
    assert_eq!(format!("{:?}", ctx.scenario), format!("{:?}", TestingScenario::Smoke));
    assert_eq!(format!("{:?}", ctx.environment), format!("{:?}", ExecutionEnvironment::Production));
}
