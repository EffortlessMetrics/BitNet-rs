use bitnet_testing_scenarios_profile_core::{
    ComparisonToleranceProfile, CrossValidationProfile, FixtureProfile, ReportFormat,
    ReportingProfile, TestConfigProfile,
};
use proptest::prelude::*;

// ── ComparisonToleranceProfile properties ────────────────────────────────────

proptest! {
    /// min_token_accuracy default must be in (0.0, 1.0] (a valid probability threshold).
    #[test]
    fn comparison_tolerance_default_accuracy_valid(_dummy in 0u8..1) {
        let tol = ComparisonToleranceProfile::default();
        prop_assert!(tol.min_token_accuracy > 0.0);
        prop_assert!(tol.min_token_accuracy <= 1.0);
    }

    /// max_probability_divergence default must be positive and finite.
    #[test]
    fn comparison_tolerance_default_divergence_positive(_dummy in 0u8..1) {
        let tol = ComparisonToleranceProfile::default();
        prop_assert!(tol.max_probability_divergence > 0.0);
        prop_assert!(tol.max_probability_divergence.is_finite());
    }

    /// numerical_tolerance default must be positive and finite.
    #[test]
    fn comparison_tolerance_default_numerical_tolerance_positive(_dummy in 0u8..1) {
        let tol = ComparisonToleranceProfile::default();
        prop_assert!(tol.numerical_tolerance > 0.0);
        prop_assert!(tol.numerical_tolerance.is_finite());
    }

    /// max_performance_regression default must be in (0.0, 1.0].
    #[test]
    fn comparison_tolerance_default_regression_valid(_dummy in 0u8..1) {
        let tol = ComparisonToleranceProfile::default();
        prop_assert!(tol.max_performance_regression > 0.0);
        prop_assert!(tol.max_performance_regression <= 1.0);
    }
}

// ── FixtureProfile properties ────────────────────────────────────────────────

proptest! {
    /// max_cache_size default must be positive (at least 1 byte).
    #[test]
    fn fixture_profile_default_cache_size_positive(_dummy in 0u8..1) {
        let fp = FixtureProfile::default();
        prop_assert!(fp.max_cache_size > 0);
    }

    /// download_timeout default must be non-zero.
    #[test]
    fn fixture_profile_default_timeout_nonzero(_dummy in 0u8..1) {
        let fp = FixtureProfile::default();
        prop_assert!(!fp.download_timeout.is_zero());
    }

    /// cleanup_interval default must be non-zero.
    #[test]
    fn fixture_profile_default_cleanup_interval_nonzero(_dummy in 0u8..1) {
        let fp = FixtureProfile::default();
        prop_assert!(!fp.cleanup_interval.is_zero());
    }
}

// ── ReportingProfile properties ──────────────────────────────────────────────

proptest! {
    /// ReportingProfile default must have at least one format.
    #[test]
    fn reporting_profile_default_has_formats(_dummy in 0u8..1) {
        let rp = ReportingProfile::default();
        prop_assert!(!rp.formats.is_empty());
    }

    /// output_dir default must be non-empty.
    #[test]
    fn reporting_profile_default_output_dir_nonempty(_dummy in 0u8..1) {
        let rp = ReportingProfile::default();
        let dir_str = rp.output_dir.to_str().unwrap_or("");
        prop_assert!(!dir_str.is_empty());
    }
}

// ── CrossValidationProfile properties ────────────────────────────────────────

proptest! {
    /// CrossValidationProfile default must have at least one test case.
    #[test]
    fn crossval_profile_default_has_test_cases(_dummy in 0u8..1) {
        let cv = CrossValidationProfile::default();
        prop_assert!(!cv.test_cases.is_empty());
    }

    /// CrossValidationProfile default tolerance must satisfy accuracy invariant.
    #[test]
    fn crossval_profile_default_tolerance_valid(_dummy in 0u8..1) {
        let cv = CrossValidationProfile::default();
        prop_assert!(cv.tolerance.min_token_accuracy > 0.0);
        prop_assert!(cv.tolerance.min_token_accuracy <= 1.0);
    }
}

// ── TestConfigProfile properties ─────────────────────────────────────────────

proptest! {
    /// max_parallel_tests default must be at least 1 (never 0).
    #[test]
    fn test_config_profile_default_parallelism_positive(_dummy in 0u8..1) {
        let tcp = TestConfigProfile::default();
        prop_assert!(tcp.max_parallel_tests >= 1);
    }

    /// test_timeout default must be non-zero.
    #[test]
    fn test_config_profile_default_timeout_nonzero(_dummy in 0u8..1) {
        let tcp = TestConfigProfile::default();
        prop_assert!(!tcp.test_timeout.is_zero());
    }

    /// coverage_threshold default must be in (0.0, 1.0].
    #[test]
    fn test_config_profile_default_coverage_threshold_valid(_dummy in 0u8..1) {
        let tcp = TestConfigProfile::default();
        prop_assert!(tcp.coverage_threshold > 0.0);
        prop_assert!(tcp.coverage_threshold <= 1.0);
    }

    /// log_level default must be non-empty.
    #[test]
    fn test_config_profile_default_log_level_nonempty(_dummy in 0u8..1) {
        let tcp = TestConfigProfile::default();
        prop_assert!(!tcp.log_level.is_empty());
    }
}

// ── ReportFormat unit tests ──────────────────────────────────────────────────

#[test]
fn report_format_variants_are_distinct() {
    let formats = vec![
        ReportFormat::Html,
        ReportFormat::Json,
        ReportFormat::Junit,
        ReportFormat::Markdown,
        ReportFormat::Csv,
    ];
    // All variants are PartialEq; verify they're distinct
    for (i, a) in formats.iter().enumerate() {
        for (j, b) in formats.iter().enumerate() {
            if i == j {
                assert_eq!(a, b, "same variant should be equal");
            } else {
                assert_ne!(a, b, "different variants should not be equal");
            }
        }
    }
}

#[test]
fn reporting_profile_default_contains_html_and_json() {
    let rp = ReportingProfile::default();
    assert!(rp.formats.contains(&ReportFormat::Html));
    assert!(rp.formats.contains(&ReportFormat::Json));
}
