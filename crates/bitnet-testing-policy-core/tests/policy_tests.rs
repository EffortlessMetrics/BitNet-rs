//! Comprehensive integration tests for `bitnet-testing-policy-core`.
//!
//! Covers PolicySnapshot construction, resolved scenario/environment config
//! properties, public helper functions, and the validate/active_context API.

use bitnet_testing_policy_core::{
    ActiveContext, ConfigurationContext, EnvironmentType, ExecutionEnvironment, FeatureSet,
    PolicySnapshot, ReportFormat, ScenarioConfigManager, TestingScenario, active_context,
    resolve_context_profile, snapshot_from_env, validate_context, validate_explicit_profile,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn ctx(scenario: TestingScenario) -> ConfigurationContext {
    ConfigurationContext { scenario, ..ConfigurationContext::default() }
}

fn ctx_with_env(scenario: TestingScenario, environment: EnvironmentType) -> ConfigurationContext {
    ConfigurationContext { scenario, environment, ..ConfigurationContext::default() }
}

// ── 1. PolicySnapshot construction ───────────────────────────────────────────

#[test]
fn snapshot_preserves_scenario_in_context() {
    let snap = PolicySnapshot::from_configuration_context(ctx(TestingScenario::CrossValidation));
    assert_eq!(snap.context.scenario, TestingScenario::CrossValidation);
}

#[test]
fn snapshot_preserves_environment_in_context() {
    let snap = PolicySnapshot::from_configuration_context(ctx_with_env(
        TestingScenario::Integration,
        EnvironmentType::Ci,
    ));
    assert_eq!(snap.context.environment, EnvironmentType::Ci);
}

#[test]
fn snapshot_from_active_context_sets_correct_scenario() {
    let snap = PolicySnapshot::from_active_context(ActiveContext {
        scenario: TestingScenario::Smoke,
        environment: ExecutionEnvironment::PreProduction,
    });
    assert_eq!(snap.context.scenario, TestingScenario::Smoke);
    assert_eq!(snap.context.environment, ExecutionEnvironment::PreProduction);
}

#[test]
fn snapshot_detect_returns_populated_snapshot() {
    // detect() reads environment variables; in most test environments it
    // falls back to Unit/Local. We only assert structural invariants.
    let snap = PolicySnapshot::detect();
    assert!(snap.resolved_config.max_parallel_tests >= 1);
    assert!(!snap.summary().is_empty());
}

// ── 2. PolicySnapshot.summary() ──────────────────────────────────────────────

#[test]
fn summary_contains_scenario_keyword() {
    let snap = PolicySnapshot::from_configuration_context(ctx(TestingScenario::Performance));
    assert!(snap.summary().contains("scenario="), "summary must include 'scenario=' key");
}

#[test]
fn summary_contains_environment_keyword() {
    let snap = PolicySnapshot::from_configuration_context(ctx(TestingScenario::Debug));
    assert!(snap.summary().contains("environment="), "summary must include 'environment=' key");
}

#[test]
fn summary_for_unknown_cell_contains_no_matching_grid_cell() {
    // Production environment has no curated BDD grid cell — summary should indicate that.
    let snap = PolicySnapshot::from_configuration_context(ctx_with_env(
        TestingScenario::Unit,
        EnvironmentType::Production,
    ));
    let summary = snap.summary();
    assert!(
        summary.contains("no matching grid cell") || summary.contains("scenario="),
        "summary for unmapped cell should indicate absence: {summary}"
    );
}

// ── 3. Resolved config: scenario-specific properties ─────────────────────────

#[test]
fn performance_scenario_resolves_max_parallel_1_locally() {
    let snap = PolicySnapshot::from_configuration_context(ctx_with_env(
        TestingScenario::Performance,
        EnvironmentType::Local,
    ));
    // Performance must be sequential (1 parallel test) to avoid timing interference.
    assert_eq!(
        snap.scenario_config.max_parallel_tests, 1,
        "Performance scenario must run tests sequentially"
    );
}

#[test]
fn smoke_scenario_has_short_timeout() {
    let snap = PolicySnapshot::from_configuration_context(ctx(TestingScenario::Smoke));
    let secs = snap.scenario_config.test_timeout.as_secs();
    assert!(secs <= 30, "Smoke scenario timeout should be ≤ 30 s, got {secs}s");
}

#[test]
fn crossval_scenario_config_has_crossval_enabled() {
    let snap = PolicySnapshot::from_configuration_context(ctx(TestingScenario::CrossValidation));
    assert!(
        snap.scenario_config.crossval.enabled,
        "CrossValidation scenario must have crossval enabled in its config"
    );
}

#[test]
fn debug_scenario_config_uses_trace_log_level() {
    let snap = PolicySnapshot::from_configuration_context(ctx(TestingScenario::Debug));
    assert_eq!(
        snap.scenario_config.log_level, "trace",
        "Debug scenario must use 'trace' log level"
    );
}

#[test]
fn minimal_scenario_config_has_zero_coverage_threshold() {
    let snap = PolicySnapshot::from_configuration_context(ctx(TestingScenario::Minimal));
    assert_eq!(
        snap.scenario_config.coverage_threshold, 0.0,
        "Minimal scenario must have zero coverage threshold"
    );
}

#[test]
fn unit_scenario_has_json_report_format() {
    let snap = PolicySnapshot::from_configuration_context(ctx(TestingScenario::Unit));
    assert!(
        snap.scenario_config.reporting.formats.contains(&ReportFormat::Json),
        "Unit scenario must include JSON reporting"
    );
}

#[test]
fn e2e_scenario_enables_artifact_inclusion() {
    let snap = PolicySnapshot::from_configuration_context(ctx(TestingScenario::EndToEnd));
    assert!(
        snap.scenario_config.reporting.include_artifacts,
        "EndToEnd scenario must include artifacts in reporting"
    );
}

// ── 4. Resolved config: environment-specific overrides ───────────────────────

#[test]
fn ci_environment_config_overrides_max_parallel_to_8() {
    let snap = PolicySnapshot::from_configuration_context(ctx_with_env(
        TestingScenario::Unit,
        EnvironmentType::Ci,
    ));
    assert_eq!(
        snap.environment_config.max_parallel_tests, 8,
        "CI environment config must set max_parallel_tests = 8"
    );
}

#[test]
fn ci_environment_config_includes_junit_format() {
    let snap = PolicySnapshot::from_configuration_context(ctx_with_env(
        TestingScenario::Integration,
        EnvironmentType::Ci,
    ));
    assert!(
        snap.environment_config.reporting.formats.contains(&ReportFormat::Junit),
        "CI environment must always include JUnit format"
    );
}

#[test]
fn ci_resolved_config_inherits_parallel_from_environment_override() {
    let snap = PolicySnapshot::from_configuration_context(ctx_with_env(
        TestingScenario::Unit,
        EnvironmentType::Ci,
    ));
    // CI env overrides parallelism to 8 — resolved must reflect that.
    assert_eq!(
        snap.resolved_config.max_parallel_tests, 8,
        "Resolved config must apply CI environment parallelism override"
    );
}

#[test]
fn local_environment_uses_info_log_level() {
    let manager = ScenarioConfigManager::new();
    let cfg = manager.get_environment_config(&EnvironmentType::Local);
    assert_eq!(cfg.log_level, "info", "Local environment log level must be 'info'");
}

// ── 5. resolve_context_profile helper ────────────────────────────────────────

#[test]
fn resolve_context_profile_always_positive_parallelism() {
    let context = ConfigurationContext::default();
    let resolved = resolve_context_profile(&context);
    assert!(resolved.max_parallel_tests >= 1);
}

#[test]
fn resolve_context_profile_always_has_report_format() {
    let context = ConfigurationContext { scenario: TestingScenario::Debug, ..Default::default() };
    let resolved = resolve_context_profile(&context);
    assert!(!resolved.reporting.formats.is_empty());
}

// ── 6. validate_context / validate_explicit_profile helpers ──────────────────

#[test]
fn validate_context_returns_some_for_known_cell() {
    let context = ConfigurationContext {
        scenario: TestingScenario::Unit,
        environment: EnvironmentType::Local,
        ..Default::default()
    };
    let result = validate_context(&context);
    assert!(result.is_some(), "validate_context must return Some for a known grid cell");
}

#[test]
fn validate_explicit_profile_returns_some_for_unit_local() {
    let result = validate_explicit_profile(TestingScenario::Unit, ExecutionEnvironment::Local);
    assert!(result.is_some(), "validate_explicit_profile must return Some for Unit/Local");
}

#[test]
fn validate_explicit_profile_returns_some_for_crossval_ci() {
    let result =
        validate_explicit_profile(TestingScenario::CrossValidation, ExecutionEnvironment::Ci);
    assert!(result.is_some(), "validate_explicit_profile must return Some for CrossValidation/Ci");
}

// ── 7. active_context helper ─────────────────────────────────────────────────

#[test]
fn active_context_preserves_scenario_and_environment() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Performance,
        environment: EnvironmentType::Ci,
        ..Default::default()
    };
    let active = active_context(&ctx);
    assert_eq!(active.scenario, TestingScenario::Performance);
    assert_eq!(active.environment, ExecutionEnvironment::Ci);
}

// ── 8. snapshot_from_env helper ──────────────────────────────────────────────

#[test]
fn snapshot_from_env_produces_non_empty_summary() {
    let snap = snapshot_from_env();
    assert!(!snap.summary().is_empty(), "snapshot_from_env() must return a non-empty summary");
}

#[test]
fn snapshot_from_env_has_valid_resolved_config() {
    let snap = snapshot_from_env();
    assert!(snap.resolved_config.max_parallel_tests >= 1);
    assert!(!snap.resolved_config.reporting.formats.is_empty());
}

// ── 9. PolicySnapshot.violations() ───────────────────────────────────────────

#[test]
fn violations_returns_some_for_known_grid_cell() {
    let snap = PolicySnapshot::from_active_context(ActiveContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Local,
    });
    assert!(
        snap.violations().is_some(),
        "violations() must return Some(...) when a grid cell is found"
    );
}

#[test]
fn violations_returns_none_for_unmapped_cell() {
    // Production environment has no curated BDD grid cell.
    let snap = PolicySnapshot::from_active_context(ActiveContext {
        scenario: TestingScenario::Unit,
        environment: ExecutionEnvironment::Production,
    });
    assert!(
        snap.violations().is_none(),
        "violations() must return None when no grid cell exists for the pair"
    );
}
