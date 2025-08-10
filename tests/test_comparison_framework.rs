use bitnet_tests::cross_validation::implementation::InferenceConfig;
use bitnet_tests::cross_validation::{
    comparison::{test_cases, ComparisonTestCase, ComparisonTolerance},
    test_implementation::MockImplementation,
};

#[test]
fn test_comparison_tolerance_default() {
    let tolerance = ComparisonTolerance::default();
    assert_eq!(tolerance.min_token_accuracy, 0.95);
    assert_eq!(tolerance.max_probability_divergence, 0.1);
    assert_eq!(tolerance.max_performance_regression, 2.0);
    assert_eq!(tolerance.float_tolerance, 1e-6);
}

#[test]
fn test_comparison_test_case_creation() {
    let test_case = ComparisonTestCase::new("test", "Hello world", InferenceConfig::default())
        .with_token_range(5, 15)
        .with_description("Test case");

    assert_eq!(test_case.name, "test");
    assert_eq!(test_case.input, "Hello world");
    assert_eq!(test_case.expected_min_tokens, Some(5));
    assert_eq!(test_case.expected_max_tokens, Some(15));
    assert_eq!(test_case.description, "Test case");
}

#[test]
fn test_basic_test_cases_creation() {
    let test_cases = test_cases::create_basic_test_cases();
    assert!(!test_cases.is_empty());
    assert!(test_cases.iter().any(|tc| tc.name == "simple_greeting"));
    assert!(test_cases.iter().any(|tc| tc.name == "short_completion"));
}

#[test]
fn test_mock_implementation_creation() {
    let mock = MockImplementation::new("test".to_string(), "1.0".to_string());
    assert_eq!(mock.implementation_name(), "test");
    assert_eq!(mock.implementation_version(), "1.0");
}
