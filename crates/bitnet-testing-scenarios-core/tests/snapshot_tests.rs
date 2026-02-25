use bitnet_testing_scenarios_core::{EnvironmentType, ScenarioConfigManager, TestingScenario};

#[test]
fn scenario_descriptions_all_variants() {
    let descs: Vec<_> = ScenarioConfigManager::available_scenarios()
        .iter()
        .map(|s| format!("{s:?}: {}", ScenarioConfigManager::scenario_description(s)))
        .collect();
    insta::assert_snapshot!(descs.join("\n"));
}

#[test]
fn unit_scenario_config_timeout_secs() {
    let manager = ScenarioConfigManager::new();
    let cfg = manager.get_scenario_config(&TestingScenario::Unit);
    insta::assert_snapshot!(format!("timeout={}s", cfg.test_timeout.as_secs()));
}

#[test]
fn ci_environment_config_log_level() {
    let manager = ScenarioConfigManager::new();
    let cfg = manager.get_environment_config(&EnvironmentType::Ci);
    insta::assert_snapshot!(format!("log_level={}", cfg.log_level));
}

#[test]
fn available_scenarios_count() {
    let count = ScenarioConfigManager::available_scenarios().len();
    insta::assert_snapshot!(format!("count={count}"));
}
