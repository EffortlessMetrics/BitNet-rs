//! Edge-case tests for `bitnet-testing-scenarios-profile-core` data model and defaults.

use bitnet_testing_scenarios_profile_core::{
    ComparisonToleranceProfile, ConfigurationContext, CrossValidationProfile, EnvironmentType,
    FixtureProfile, PlatformSettings, QualityRequirements, ReportFormat, ReportingProfile,
    ResourceConstraints, TestConfigProfile, TestingScenario, TimeConstraints,
};
use std::path::PathBuf;
use std::time::Duration;

// ---------------------------------------------------------------------------
// ConfigurationContext defaults
// ---------------------------------------------------------------------------

#[test]
fn configuration_context_default_is_unit_local() {
    let ctx = ConfigurationContext::default();
    assert!(matches!(ctx.scenario, TestingScenario::Unit));
    assert!(matches!(ctx.environment, EnvironmentType::Local));
    assert!(ctx.resource_constraints.is_none());
    assert!(ctx.time_constraints.is_none());
    assert!(ctx.quality_requirements.is_none());
    assert!(ctx.platform_settings.is_none());
}

#[test]
fn configuration_context_all_fields_settable() {
    let ctx = ConfigurationContext {
        scenario: TestingScenario::Performance,
        environment: EnvironmentType::Ci,
        resource_constraints: Some(ResourceConstraints {
            max_memory_mb: Some(8192),
            max_cpu_cores: Some(4),
            max_disk_gb: Some(100),
        }),
        time_constraints: Some(TimeConstraints {
            max_total_duration: Some(Duration::from_secs(3600)),
            max_test_duration: Some(Duration::from_secs(600)),
        }),
        quality_requirements: Some(QualityRequirements {
            min_coverage: Some(0.9),
            max_flakiness: Some(0.01),
            required_passes: Some(3),
        }),
        platform_settings: Some(PlatformSettings {
            os: Some("linux".to_string()),
            arch: Some("x86_64".to_string()),
            features: vec!["avx2".to_string()],
        }),
    };
    assert!(matches!(ctx.scenario, TestingScenario::Performance));
    assert!(ctx.resource_constraints.unwrap().max_memory_mb == Some(8192));
}

// ---------------------------------------------------------------------------
// TestConfigProfile defaults
// ---------------------------------------------------------------------------

#[test]
fn test_config_profile_default_timeout() {
    let cfg = TestConfigProfile::default();
    assert_eq!(cfg.test_timeout, Duration::from_secs(300));
}

#[test]
fn test_config_profile_default_log_level() {
    let cfg = TestConfigProfile::default();
    assert_eq!(cfg.log_level, "info");
}

#[test]
fn test_config_profile_default_coverage_threshold() {
    let cfg = TestConfigProfile::default();
    assert!((cfg.coverage_threshold - 0.9).abs() < f64::EPSILON);
}

#[test]
fn test_config_profile_default_cache_dir() {
    let cfg = TestConfigProfile::default();
    assert_eq!(cfg.cache_dir, PathBuf::from("tests/cache"));
}

#[test]
fn test_config_profile_default_parallel_at_least_1() {
    let cfg = TestConfigProfile::default();
    assert!(cfg.max_parallel_tests >= 1);
}

// ---------------------------------------------------------------------------
// ReportingProfile defaults
// ---------------------------------------------------------------------------

#[test]
fn reporting_profile_default_output_dir() {
    let rp = ReportingProfile::default();
    assert_eq!(rp.output_dir, PathBuf::from("test-reports"));
}

#[test]
fn reporting_profile_default_formats() {
    let rp = ReportingProfile::default();
    assert!(rp.formats.contains(&ReportFormat::Html));
    assert!(rp.formats.contains(&ReportFormat::Json));
}

#[test]
fn reporting_profile_default_flags() {
    let rp = ReportingProfile::default();
    assert!(rp.include_artifacts);
    assert!(rp.generate_coverage);
    assert!(rp.generate_performance);
    assert!(!rp.upload_reports);
}

// ---------------------------------------------------------------------------
// FixtureProfile defaults
// ---------------------------------------------------------------------------

#[test]
fn fixture_profile_default_auto_download() {
    let fp = FixtureProfile::default();
    assert!(fp.auto_download);
}

#[test]
fn fixture_profile_default_cache_size() {
    let fp = FixtureProfile::default();
    assert_eq!(fp.max_cache_size, 10 * 1024 * 1024 * 1024); // 10 GB
}

#[test]
fn fixture_profile_default_cleanup_interval() {
    let fp = FixtureProfile::default();
    assert_eq!(fp.cleanup_interval, Duration::from_secs(24 * 60 * 60)); // 1 day
}

#[test]
fn fixture_profile_default_download_timeout() {
    let fp = FixtureProfile::default();
    assert_eq!(fp.download_timeout, Duration::from_secs(300));
}

#[test]
fn fixture_profile_default_no_base_url() {
    let fp = FixtureProfile::default();
    assert!(fp.base_url.is_none());
}

// ---------------------------------------------------------------------------
// ComparisonToleranceProfile defaults
// ---------------------------------------------------------------------------

#[test]
fn comparison_tolerance_default_accuracy() {
    let ctp = ComparisonToleranceProfile::default();
    assert!(ctp.min_token_accuracy > 0.99);
}

#[test]
fn comparison_tolerance_default_divergence() {
    let ctp = ComparisonToleranceProfile::default();
    assert!(ctp.max_probability_divergence <= 1e-5);
}

#[test]
fn comparison_tolerance_default_regression() {
    let ctp = ComparisonToleranceProfile::default();
    assert!(ctp.max_performance_regression <= 0.2);
}

#[test]
fn comparison_tolerance_default_numerical() {
    let ctp = ComparisonToleranceProfile::default();
    assert!(ctp.numerical_tolerance <= 1e-5);
}

// ---------------------------------------------------------------------------
// CrossValidationProfile defaults
// ---------------------------------------------------------------------------

#[test]
fn crossval_profile_default_disabled() {
    let cvp = CrossValidationProfile::default();
    assert!(!cvp.enabled);
}

#[test]
fn crossval_profile_default_test_cases() {
    let cvp = CrossValidationProfile::default();
    assert!(!cvp.test_cases.is_empty());
    assert!(cvp.test_cases.contains(&"basic_inference".to_string()));
}

#[test]
fn crossval_profile_default_comparison_flags() {
    let cvp = CrossValidationProfile::default();
    assert!(cvp.performance_comparison);
    assert!(cvp.accuracy_comparison);
}

#[test]
fn crossval_profile_default_no_cpp_binary() {
    let cvp = CrossValidationProfile::default();
    assert!(cvp.cpp_binary_path.is_none());
}

// ---------------------------------------------------------------------------
// ReportFormat equality
// ---------------------------------------------------------------------------

#[test]
fn report_format_equality() {
    assert_eq!(ReportFormat::Html, ReportFormat::Html);
    assert_ne!(ReportFormat::Html, ReportFormat::Json);
    assert_ne!(ReportFormat::Junit, ReportFormat::Csv);
}

// ---------------------------------------------------------------------------
// Struct cloneability
// ---------------------------------------------------------------------------

#[test]
fn test_config_profile_cloneable() {
    let cfg = TestConfigProfile::default();
    let cfg2 = cfg.clone();
    assert_eq!(cfg.max_parallel_tests, cfg2.max_parallel_tests);
    assert_eq!(cfg.test_timeout, cfg2.test_timeout);
}

#[test]
fn configuration_context_cloneable() {
    let ctx = ConfigurationContext::default();
    let ctx2 = ctx.clone();
    assert!(matches!(ctx2.scenario, TestingScenario::Unit));
}

#[test]
fn resource_constraints_default_all_none() {
    let rc = ResourceConstraints::default();
    assert!(rc.max_memory_mb.is_none());
    assert!(rc.max_cpu_cores.is_none());
    assert!(rc.max_disk_gb.is_none());
}

#[test]
fn time_constraints_default_all_none() {
    let tc = TimeConstraints::default();
    assert!(tc.max_total_duration.is_none());
    assert!(tc.max_test_duration.is_none());
}

#[test]
fn quality_requirements_default_all_none() {
    let qr = QualityRequirements::default();
    assert!(qr.min_coverage.is_none());
    assert!(qr.max_flakiness.is_none());
    assert!(qr.required_passes.is_none());
}

#[test]
fn platform_settings_default_empty() {
    let ps = PlatformSettings::default();
    assert!(ps.os.is_none());
    assert!(ps.arch.is_none());
    assert!(ps.features.is_empty());
}

// ---------------------------------------------------------------------------
// ScenarioType alias resolves
// ---------------------------------------------------------------------------

#[test]
fn scenario_type_alias_works() {
    use bitnet_testing_scenarios_profile_core::ScenarioType;
    let _s: ScenarioType = TestingScenario::Unit;
}
