use bitnet_testing_scenarios_profile_core::{
    ComparisonToleranceProfile, ConfigurationContext, CrossValidationProfile, FixtureProfile,
    ReportFormat, ReportingProfile, TestConfigProfile,
};

#[test]
fn configuration_context_default() {
    let ctx = ConfigurationContext::default();
    insta::assert_debug_snapshot!(ctx);
}

#[test]
fn reporting_profile_default_formats() {
    let profile = ReportingProfile::default();
    // Default formats must stay stable — changes break CI reporting
    insta::assert_snapshot!({
        profile.formats.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>().join(",")
    });
}

#[test]
fn fixture_profile_default_fields() {
    let profile = FixtureProfile::default();
    insta::assert_debug_snapshot!(profile);
}

#[test]
fn comparison_tolerance_profile_defaults() {
    let profile = ComparisonToleranceProfile::default();
    insta::assert_debug_snapshot!(profile);
}

#[test]
fn cross_validation_profile_defaults() {
    let profile = CrossValidationProfile::default();
    insta::assert_debug_snapshot!(profile);
}

#[test]
fn report_format_variants_debug_stable() {
    insta::assert_snapshot!({
        let variants = [
            ReportFormat::Html,
            ReportFormat::Json,
            ReportFormat::Junit,
            ReportFormat::Markdown,
            ReportFormat::Csv,
        ];
        variants.iter().map(|f| format!("{:?}", f)).collect::<Vec<_>>().join("\n")
    });
}

#[test]
fn test_config_profile_defaults() {
    let profile = TestConfigProfile::default();
    // max_parallel_tests is CPU/env-dependent — normalize it for a stable snapshot.
    insta::with_settings!({
        filters => vec![(r"max_parallel_tests: \d+", "max_parallel_tests: <CPU_DEPENDENT>")]
    }, {
        insta::assert_debug_snapshot!(profile);
    });
}
