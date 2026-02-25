use bitnet_testing_policy_core::{
    ActiveContext, ExecutionEnvironment, PolicySnapshot, TestingScenario, active_profile_summary,
};

#[test]
fn unit_local_policy_snapshot_summary_format() {
    let snap = PolicySnapshot::from_active_context(ActiveContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
    });
    let summary = snap.summary();
    insta::assert_snapshot!(format!(
        "has_scenario={} has_environment={} non_empty={}",
        summary.contains("scenario="),
        summary.contains("environment="),
        !summary.is_empty()
    ));
}

#[test]
fn active_profile_summary_contains_scenario() {
    let summary = active_profile_summary();
    // The function returns a stable summary — pin that it has key tokens
    insta::assert_snapshot!(format!(
        "non_empty={} has_scenario={}",
        !summary.is_empty(),
        summary.contains("scenario=")
    ));
}

#[test]
fn integration_ci_snapshot_no_cell_or_has_summary() {
    let snap = PolicySnapshot::from_active_context(ActiveContext {
        scenario: TestingScenario::Integration,
        environment: ExecutionEnvironment::Ci,
    });
    insta::assert_snapshot!(format!("summary_non_empty={}", !snap.summary().is_empty()));
}
