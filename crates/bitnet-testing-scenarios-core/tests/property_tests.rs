use bitnet_testing_scenarios_core::{
    EnvironmentType, ReportFormat, ScenarioConfigManager, TestConfigProfile, TestingScenario,
};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────────

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

fn arb_env() -> impl Strategy<Value = EnvironmentType> {
    prop_oneof![
        Just(EnvironmentType::Local),
        Just(EnvironmentType::Ci),
        Just(EnvironmentType::PreProduction),
        Just(EnvironmentType::Production),
    ]
}

// ── Property tests ───────────────────────────────────────────────────────────

proptest! {
    /// `ScenarioConfigManager::resolve()` always produces a positive parallelism count.
    #[test]
    fn resolved_parallelism_always_positive(
        scenario in arb_scenario(),
        env in arb_env(),
    ) {
        let manager = ScenarioConfigManager::default();
        let config = manager.resolve(&scenario, &env);
        prop_assert!(config.max_parallel_tests >= 1);
    }

    /// Resolved config always contains at least one report format.
    #[test]
    fn resolved_always_has_report_format(
        scenario in arb_scenario(),
        env in arb_env(),
    ) {
        let manager = ScenarioConfigManager::default();
        let config = manager.resolve(&scenario, &env);
        prop_assert!(!config.reporting.formats.is_empty());
    }

    /// `TestingScenario` Display→parse round-trip holds for all variants.
    #[test]
    fn scenario_display_parse_round_trip(scenario in arb_scenario()) {
        let s = scenario.to_string();
        let parsed: TestingScenario = s.parse().expect("valid display string should parse");
        prop_assert_eq!(parsed, scenario);
    }

    /// `EnvironmentType` Display→parse round-trip holds for all variants.
    #[test]
    fn env_type_display_parse_round_trip(env in arb_env()) {
        let s = env.to_string();
        let parsed: EnvironmentType = s.parse().expect("valid display string should parse");
        prop_assert_eq!(parsed, env);
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[test]
fn ci_environment_includes_junit_format() {
    let manager = ScenarioConfigManager::default();
    let config = manager.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);
    assert!(
        config.reporting.formats.contains(&ReportFormat::Junit),
        "CI should always include JUnit format for test result parsing"
    );
}

#[test]
fn ci_parallelism_is_at_least_two() {
    let manager = ScenarioConfigManager::default();
    let config = manager.resolve(&TestingScenario::Unit, &EnvironmentType::Ci);
    assert!(
        config.max_parallel_tests >= 2,
        "CI should allow at least 2 parallel tests, got {}",
        config.max_parallel_tests
    );
}

#[test]
fn unknown_scenario_parse_returns_error() {
    assert!("not-a-real-scenario".parse::<TestingScenario>().is_err());
}

#[test]
fn unknown_env_parse_returns_error() {
    assert!("not-a-real-env".parse::<EnvironmentType>().is_err());
}
