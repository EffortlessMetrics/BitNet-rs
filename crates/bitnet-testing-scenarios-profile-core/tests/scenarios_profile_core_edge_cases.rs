//! Edge-case tests for bitnet-testing-scenarios-profile-core data model.

use bitnet_testing_scenarios_profile_core::{
    ComparisonToleranceProfile, ConfigurationContext, CrossValidationProfile, FixtureProfile,
    PlatformSettings, QualityRequirements, ReportFormat, ReportingProfile, ResourceConstraints,
    TestConfigProfile, TimeConstraints,
};
use std::path::PathBuf;
use std::time::Duration;

// ---------------------------------------------------------------------------
// ConfigurationContext: Default
// ---------------------------------------------------------------------------

#[test]
fn config_context_default_is_unit_local() {
    let ctx = ConfigurationContext::default();
    assert_eq!(ctx.scenario.to_string(), "unit");
    assert_eq!(ctx.environment.to_string(), "local");
    assert!(ctx.resource_constraints.is_none());
    assert!(ctx.time_constraints.is_none());
    assert!(ctx.quality_requirements.is_none());
    assert!(ctx.platform_settings.is_none());
}

#[test]
fn config_context_clone() {
    let ctx = ConfigurationContext::default();
    let ctx2 = ctx.clone();
    assert_eq!(ctx.scenario.to_string(), ctx2.scenario.to_string());
}

#[test]
fn config_context_debug() {
    let ctx = ConfigurationContext::default();
    let d = format!("{:?}", ctx);
    assert!(d.contains("ConfigurationContext"));
}

// ---------------------------------------------------------------------------
// ResourceConstraints: Default
// ---------------------------------------------------------------------------

#[test]
fn resource_constraints_default_all_none() {
    let r = ResourceConstraints::default();
    assert!(r.max_memory_mb.is_none());
    assert!(r.max_cpu_cores.is_none());
    assert!(r.max_disk_gb.is_none());
}

#[test]
fn resource_constraints_custom() {
    let r = ResourceConstraints {
        max_memory_mb: Some(4096),
        max_cpu_cores: Some(8),
        max_disk_gb: Some(100),
    };
    assert_eq!(r.max_memory_mb, Some(4096));
    assert_eq!(r.max_cpu_cores, Some(8));
    assert_eq!(r.max_disk_gb, Some(100));
}

// ---------------------------------------------------------------------------
// TimeConstraints: Default
// ---------------------------------------------------------------------------

#[test]
fn time_constraints_default_all_none() {
    let t = TimeConstraints::default();
    assert!(t.max_total_duration.is_none());
    assert!(t.max_test_duration.is_none());
}

#[test]
fn time_constraints_custom() {
    let t = TimeConstraints {
        max_total_duration: Some(Duration::from_secs(600)),
        max_test_duration: Some(Duration::from_secs(60)),
    };
    assert_eq!(t.max_total_duration, Some(Duration::from_secs(600)));
    assert_eq!(t.max_test_duration, Some(Duration::from_secs(60)));
}

// ---------------------------------------------------------------------------
// QualityRequirements: Default
// ---------------------------------------------------------------------------

#[test]
fn quality_requirements_default_all_none() {
    let q = QualityRequirements::default();
    assert!(q.min_coverage.is_none());
    assert!(q.max_flakiness.is_none());
    assert!(q.required_passes.is_none());
}

#[test]
fn quality_requirements_custom() {
    let q = QualityRequirements {
        min_coverage: Some(0.95),
        max_flakiness: Some(0.01),
        required_passes: Some(3),
    };
    assert!((q.min_coverage.unwrap() - 0.95).abs() < 1e-10);
    assert_eq!(q.required_passes, Some(3));
}

// ---------------------------------------------------------------------------
// PlatformSettings: Default
// ---------------------------------------------------------------------------

#[test]
fn platform_settings_default_all_none_or_empty() {
    let p = PlatformSettings::default();
    assert!(p.os.is_none());
    assert!(p.arch.is_none());
    assert!(p.features.is_empty());
}

#[test]
fn platform_settings_custom() {
    let p = PlatformSettings {
        os: Some("linux".to_string()),
        arch: Some("x86_64".to_string()),
        features: vec!["avx2".to_string(), "cuda".to_string()],
    };
    assert_eq!(p.os.as_deref(), Some("linux"));
    assert_eq!(p.features.len(), 2);
}

// ---------------------------------------------------------------------------
// ReportFormat: variants
// ---------------------------------------------------------------------------

#[test]
fn report_format_eq() {
    assert_eq!(ReportFormat::Html, ReportFormat::Html);
    assert_eq!(ReportFormat::Json, ReportFormat::Json);
    assert_ne!(ReportFormat::Html, ReportFormat::Json);
}

#[test]
fn report_format_all_variants_debug() {
    let formats = [
        ReportFormat::Html,
        ReportFormat::Json,
        ReportFormat::Junit,
        ReportFormat::Markdown,
        ReportFormat::Csv,
    ];
    for f in &formats {
        let d = format!("{:?}", f);
        assert!(!d.is_empty());
    }
}

#[test]
fn report_format_clone() {
    let f = ReportFormat::Junit;
    let f2 = f.clone();
    assert_eq!(f, f2);
}

// ---------------------------------------------------------------------------
// ReportingProfile: Default
// ---------------------------------------------------------------------------

#[test]
fn reporting_profile_default_has_html_json() {
    let r = ReportingProfile::default();
    assert!(r.formats.contains(&ReportFormat::Html));
    assert!(r.formats.contains(&ReportFormat::Json));
}

#[test]
fn reporting_profile_default_output_dir() {
    let r = ReportingProfile::default();
    assert_eq!(r.output_dir, PathBuf::from("test-reports"));
}

#[test]
fn reporting_profile_default_flags() {
    let r = ReportingProfile::default();
    assert!(r.include_artifacts);
    assert!(r.generate_coverage);
    assert!(r.generate_performance);
    assert!(!r.upload_reports);
}

// ---------------------------------------------------------------------------
// FixtureProfile: Default
// ---------------------------------------------------------------------------

#[test]
fn fixture_profile_default_auto_download() {
    let f = FixtureProfile::default();
    assert!(f.auto_download);
}

#[test]
fn fixture_profile_default_cache_size_10gb() {
    let f = FixtureProfile::default();
    assert_eq!(f.max_cache_size, 10 * 1024 * 1024 * 1024);
}

#[test]
fn fixture_profile_default_cleanup_interval_24h() {
    let f = FixtureProfile::default();
    assert_eq!(f.cleanup_interval, Duration::from_secs(86400));
}

#[test]
fn fixture_profile_default_download_timeout_5m() {
    let f = FixtureProfile::default();
    assert_eq!(f.download_timeout, Duration::from_secs(300));
}

#[test]
fn fixture_profile_default_no_base_url() {
    let f = FixtureProfile::default();
    assert!(f.base_url.is_none());
}

// ---------------------------------------------------------------------------
// ComparisonToleranceProfile: Default
// ---------------------------------------------------------------------------

#[test]
fn tolerance_profile_default_high_accuracy() {
    let t = ComparisonToleranceProfile::default();
    assert!(t.min_token_accuracy > 0.999);
    assert!(t.max_probability_divergence < 1e-5);
    assert!(t.numerical_tolerance < 1e-5);
}

#[test]
fn tolerance_profile_default_max_regression() {
    let t = ComparisonToleranceProfile::default();
    assert!((t.max_performance_regression - 0.1).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// CrossValidationProfile: Default
// ---------------------------------------------------------------------------

#[test]
fn crossval_profile_default_disabled() {
    let cv = CrossValidationProfile::default();
    assert!(!cv.enabled);
}

#[test]
fn crossval_profile_default_test_cases() {
    let cv = CrossValidationProfile::default();
    assert_eq!(cv.test_cases.len(), 3);
    assert!(cv.test_cases.contains(&"basic_inference".to_string()));
    assert!(cv.test_cases.contains(&"tokenization".to_string()));
    assert!(cv.test_cases.contains(&"model_loading".to_string()));
}

#[test]
fn crossval_profile_default_comparisons_enabled() {
    let cv = CrossValidationProfile::default();
    assert!(cv.performance_comparison);
    assert!(cv.accuracy_comparison);
}

#[test]
fn crossval_profile_default_no_cpp_binary() {
    let cv = CrossValidationProfile::default();
    assert!(cv.cpp_binary_path.is_none());
}

// ---------------------------------------------------------------------------
// TestConfigProfile: Default
// ---------------------------------------------------------------------------

#[test]
fn test_config_profile_default_parallel_positive() {
    let config = TestConfigProfile::default();
    assert!(config.max_parallel_tests > 0);
}

#[test]
fn test_config_profile_default_timeout_5m() {
    let config = TestConfigProfile::default();
    assert_eq!(config.test_timeout, Duration::from_secs(300));
}

#[test]
fn test_config_profile_default_cache_dir() {
    let config = TestConfigProfile::default();
    assert_eq!(config.cache_dir, PathBuf::from("tests/cache"));
}

#[test]
fn test_config_profile_default_log_level_info() {
    let config = TestConfigProfile::default();
    assert_eq!(config.log_level, "info");
}

#[test]
fn test_config_profile_default_coverage_threshold() {
    let config = TestConfigProfile::default();
    assert!((config.coverage_threshold - 0.9).abs() < 1e-10);
}

#[test]
fn test_config_profile_default_reporting_has_formats() {
    let config = TestConfigProfile::default();
    assert!(!config.reporting.formats.is_empty());
}

#[test]
fn test_config_profile_clone() {
    let config = TestConfigProfile::default();
    let config2 = config.clone();
    assert_eq!(config.max_parallel_tests, config2.max_parallel_tests);
    assert_eq!(config.log_level, config2.log_level);
}

#[test]
fn test_config_profile_debug() {
    let config = TestConfigProfile::default();
    let d = format!("{:?}", config);
    assert!(d.contains("TestConfigProfile"));
}

// ---------------------------------------------------------------------------
// ConfigurationContext: with constraints
// ---------------------------------------------------------------------------

#[test]
fn config_context_with_all_constraints() {
    use bitnet_testing_scenarios_profile_core::TestingScenario;

    let ctx = ConfigurationContext {
        scenario: TestingScenario::Integration,
        environment: bitnet_testing_scenarios_profile_core::EnvironmentType::Ci,
        resource_constraints: Some(ResourceConstraints {
            max_memory_mb: Some(8192),
            max_cpu_cores: Some(4),
            max_disk_gb: Some(50),
        }),
        time_constraints: Some(TimeConstraints {
            max_total_duration: Some(Duration::from_secs(1800)),
            max_test_duration: Some(Duration::from_secs(120)),
        }),
        quality_requirements: Some(QualityRequirements {
            min_coverage: Some(0.8),
            max_flakiness: Some(0.05),
            required_passes: Some(2),
        }),
        platform_settings: Some(PlatformSettings {
            os: Some("windows".to_string()),
            arch: Some("aarch64".to_string()),
            features: vec!["neon".to_string()],
        }),
    };
    assert!(ctx.resource_constraints.is_some());
    assert!(ctx.time_constraints.is_some());
    assert!(ctx.quality_requirements.is_some());
    assert!(ctx.platform_settings.is_some());
}
