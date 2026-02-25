use bitnet_bdd_grid_core::{ExecutionEnvironment, TestingScenario};
use bitnet_runtime_context_core::ActiveContext;

#[test]
fn active_context_default_fields() {
    // Regression: default scenario/environment must stay stable
    let ctx =
        ActiveContext::from_env_with_defaults(TestingScenario::Unit, ExecutionEnvironment::Local);
    insta::assert_debug_snapshot!(ctx.scenario.to_string(), @r#""unit""#);
    insta::assert_debug_snapshot!(ctx.environment.to_string(), @r#""local""#);
}

#[test]
fn active_context_integration_ci_defaults() {
    let ctx = ActiveContext::from_env_with_defaults(
        TestingScenario::Integration,
        ExecutionEnvironment::Ci,
    );
    insta::assert_debug_snapshot!(ctx.scenario.to_string(), @r#""integration""#);
    insta::assert_debug_snapshot!(ctx.environment.to_string(), @r#""ci""#);
}

#[test]
fn active_context_scenario_variants_stable() {
    insta::assert_snapshot!("active_context_scenarios", {
        let mut out = String::new();
        for s in [TestingScenario::Unit, TestingScenario::Integration, TestingScenario::EndToEnd] {
            out.push_str(&format!("{}\n", s));
        }
        out
    });
}
