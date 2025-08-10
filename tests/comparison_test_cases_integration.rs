use bitnet_tests::cross_validation::{
    test_suites, ComparisonTestCaseRegistry, ComparisonTestRunner, TestCaseCategory,
};
use bitnet_tests::data::models::ModelSize;

/// Integration test for the comprehensive comparison test cases
#[tokio::test]
async fn test_comparison_test_case_registry() {
    let registry = ComparisonTestCaseRegistry::new();

    // Test that all categories have test cases
    assert!(!registry.by_category(TestCaseCategory::Basic).is_empty());
    assert!(!registry.by_category(TestCaseCategory::EdgeCase).is_empty());
    assert!(!registry
        .by_category(TestCaseCategory::Performance)
        .is_empty());
    assert!(!registry
        .by_category(TestCaseCategory::Regression)
        .is_empty());
    assert!(!registry
        .by_category(TestCaseCategory::FormatCompatibility)
        .is_empty());
    assert!(!registry.by_category(TestCaseCategory::ModelSize).is_empty());

    // Test that all model sizes have test cases
    assert!(!registry.by_model_size(ModelSize::Tiny).is_empty());
    assert!(!registry.by_model_size(ModelSize::Small).is_empty());
    assert!(!registry.by_model_size(ModelSize::Medium).is_empty());

    // Test specific test cases exist
    assert!(registry.get("basic_greeting").is_some());
    assert!(registry.get("edge_empty_input").is_some());
    assert!(registry.get("perf_throughput").is_some());
    assert!(registry
        .get("regression_tokenization_consistency")
        .is_some());

    println!("✓ Test case registry validation passed");
}

#[tokio::test]
async fn test_basic_test_suite() {
    let basic_suite = test_suites::create_basic_suite();

    assert!(!basic_suite.is_empty());
    assert!(basic_suite.len() >= 5); // Should have at least 5 basic tests

    // Verify specific test cases
    assert!(basic_suite.iter().any(|tc| tc.name == "basic_greeting"));
    assert!(basic_suite.iter().any(|tc| tc.name == "basic_completion"));
    assert!(basic_suite.iter().any(|tc| tc.name == "basic_qa"));
    assert!(basic_suite.iter().any(|tc| tc.name == "basic_code"));
    assert!(basic_suite.iter().any(|tc| tc.name == "basic_math"));

    // Verify test case properties
    for test_case in &basic_suite {
        assert!(!test_case.name.is_empty());
        assert!(!test_case.description.is_empty());
        assert!(test_case.expected_min_tokens.is_some());
        assert!(test_case.expected_max_tokens.is_some());

        // Basic tests should be deterministic (temperature = 0.0)
        if test_case.name.starts_with("basic_") {
            assert_eq!(test_case.config.temperature, 0.0);
        }
    }

    println!("✓ Basic test suite validation passed");
}

#[tokio::test]
async fn test_edge_case_suite() {
    let edge_suite = test_suites::create_edge_case_suite();

    assert!(!edge_suite.is_empty());
    assert!(edge_suite.len() >= 6); // Should have at least 6 edge case tests

    // Verify specific edge cases
    assert!(edge_suite.iter().any(|tc| tc.name == "edge_empty_input"));
    assert!(edge_suite.iter().any(|tc| tc.name == "edge_single_char"));
    assert!(edge_suite.iter().any(|tc| tc.name == "edge_special_chars"));
    assert!(edge_suite
        .iter()
        .any(|tc| tc.name == "edge_very_long_input"));
    assert!(edge_suite.iter().any(|tc| tc.name == "edge_multilingual"));

    // Test empty input case specifically
    let empty_test = edge_suite
        .iter()
        .find(|tc| tc.name == "edge_empty_input")
        .unwrap();
    assert_eq!(empty_test.input, "");
    assert_eq!(empty_test.expected_min_tokens, Some(0));

    // Test very long input case
    let long_test = edge_suite
        .iter()
        .find(|tc| tc.name == "edge_very_long_input")
        .unwrap();
    assert!(long_test.input.len() > 1000); // Should be very long

    println!("✓ Edge case test suite validation passed");
}

#[tokio::test]
async fn test_performance_suite() {
    let perf_suite = test_suites::create_performance_suite();

    assert!(!perf_suite.is_empty());
    assert!(perf_suite.len() >= 4); // Should have at least 4 performance tests

    // Verify specific performance tests
    assert!(perf_suite.iter().any(|tc| tc.name == "perf_throughput"));
    assert!(perf_suite
        .iter()
        .any(|tc| tc.name == "perf_long_generation"));
    assert!(perf_suite
        .iter()
        .any(|tc| tc.name == "perf_batch_simulation"));
    assert!(perf_suite.iter().any(|tc| tc.name == "perf_memory_stress"));

    // Test long generation case specifically
    let long_gen = perf_suite
        .iter()
        .find(|tc| tc.name == "perf_long_generation")
        .unwrap();
    assert_eq!(long_gen.config.max_tokens, 500);
    assert_eq!(long_gen.config.temperature, 0.3);
    assert_eq!(long_gen.expected_max_tokens, Some(500));

    // Test high temperature creativity
    let creativity = perf_suite
        .iter()
        .find(|tc| tc.name == "perf_high_temp_creativity")
        .unwrap();
    assert_eq!(creativity.config.temperature, 0.9);
    assert_eq!(creativity.config.top_p, 0.9);

    println!("✓ Performance test suite validation passed");
}

#[tokio::test]
async fn test_regression_suite() {
    let regression_suite = test_suites::create_regression_suite();

    assert!(!regression_suite.is_empty());
    assert!(regression_suite.len() >= 4); // Should have at least 4 regression tests

    // Verify specific regression tests
    assert!(regression_suite
        .iter()
        .any(|tc| tc.name == "regression_tokenization_consistency"));
    assert!(regression_suite
        .iter()
        .any(|tc| tc.name == "regression_memory_management"));
    assert!(regression_suite
        .iter()
        .any(|tc| tc.name == "regression_float_precision"));
    assert!(regression_suite
        .iter()
        .any(|tc| tc.name == "regression_context_window"));

    // Test stop token handling
    let stop_tokens = regression_suite
        .iter()
        .find(|tc| tc.name == "regression_stop_tokens")
        .unwrap();
    assert!(!stop_tokens.config.stop_tokens.is_empty());
    assert_eq!(stop_tokens.config.stop_tokens[0], "STOP");

    println!("✓ Regression test suite validation passed");
}

#[tokio::test]
async fn test_format_compatibility_suite() {
    let format_suite = test_suites::create_format_compatibility_suite();

    assert!(!format_suite.is_empty());
    assert!(format_suite.len() >= 3); // Should have at least 3 format tests

    // Verify specific format tests
    assert!(format_suite
        .iter()
        .any(|tc| tc.name == "format_gguf_compatibility"));
    assert!(format_suite
        .iter()
        .any(|tc| tc.name == "format_safetensors_compatibility"));
    assert!(format_suite
        .iter()
        .any(|tc| tc.name == "format_quantization_compatibility"));

    println!("✓ Format compatibility test suite validation passed");
}

#[tokio::test]
async fn test_model_size_suite() {
    let size_suite = test_suites::create_model_size_suite();

    assert!(!size_suite.is_empty());
    assert!(size_suite.len() >= 4); // Should have at least 4 model size tests

    // Verify specific model size tests
    assert!(size_suite
        .iter()
        .any(|tc| tc.name == "size_tiny_model_limits"));
    assert!(size_suite
        .iter()
        .any(|tc| tc.name == "size_small_model_scaling"));
    assert!(size_suite
        .iter()
        .any(|tc| tc.name == "size_medium_model_capabilities"));
    assert!(size_suite
        .iter()
        .any(|tc| tc.name == "size_large_model_stress"));

    // Test large model stress test
    let large_test = size_suite
        .iter()
        .find(|tc| tc.name == "size_large_model_stress")
        .unwrap();
    assert_eq!(large_test.config.max_tokens, 500);
    assert_eq!(large_test.expected_max_tokens, Some(500));

    println!("✓ Model size test suite validation passed");
}

#[tokio::test]
async fn test_smoke_test_suite() {
    let smoke_suite = test_suites::create_smoke_test_suite();

    assert_eq!(smoke_suite.len(), 3); // Should be exactly 3 tests for quick validation

    // All smoke tests should be from basic category
    for test_case in &smoke_suite {
        assert!(test_case.name.starts_with("basic_"));
        assert_eq!(test_case.config.temperature, 0.0); // Should be deterministic
    }

    println!("✓ Smoke test suite validation passed");
}

#[tokio::test]
async fn test_comprehensive_suite() {
    let comprehensive_suite = test_suites::create_comprehensive_suite();

    assert!(!comprehensive_suite.is_empty());
    assert!(comprehensive_suite.len() >= 20); // Should have many tests

    // Should include tests from all categories
    let categories = [
        "basic_",
        "edge_",
        "perf_",
        "regression_",
        "format_",
        "size_",
    ];

    for category_prefix in &categories {
        assert!(
            comprehensive_suite
                .iter()
                .any(|tc| tc.name.starts_with(category_prefix)),
            "Missing tests for category: {}",
            category_prefix
        );
    }

    println!("✓ Comprehensive test suite validation passed");
}

#[tokio::test]
async fn test_model_size_filtering() {
    let registry = ComparisonTestCaseRegistry::new();

    let tiny_tests = registry.by_model_size(ModelSize::Tiny);
    let small_tests = registry.by_model_size(ModelSize::Small);
    let medium_tests = registry.by_model_size(ModelSize::Medium);

    assert!(!tiny_tests.is_empty());
    assert!(!small_tests.is_empty());
    assert!(!medium_tests.is_empty());

    // Verify that tiny tests are appropriate for tiny models
    for test in tiny_tests {
        // Tiny model tests should have reasonable token limits
        if let Some(max_tokens) = test.expected_max_tokens {
            assert!(
                max_tokens <= 50,
                "Tiny model test '{}' has too many expected tokens: {}",
                test.name,
                max_tokens
            );
        }
    }

    // Verify that medium tests can have larger token counts
    for test in medium_tests {
        if test.name.contains("long") || test.name.contains("stress") {
            if let Some(max_tokens) = test.expected_max_tokens {
                assert!(
                    max_tokens >= 50,
                    "Medium model test '{}' should allow more tokens: {}",
                    test.name,
                    max_tokens
                );
            }
        }
    }

    println!("✓ Model size filtering validation passed");
}

#[tokio::test]
async fn test_suite_for_model_size() {
    let tiny_suite = test_suites::create_suite_for_model_size(ModelSize::Tiny);
    let small_suite = test_suites::create_suite_for_model_size(ModelSize::Small);
    let medium_suite = test_suites::create_suite_for_model_size(ModelSize::Medium);

    assert!(!tiny_suite.is_empty());
    assert!(!small_suite.is_empty());
    assert!(!medium_suite.is_empty());

    // Tiny suite should include basic tests
    assert!(tiny_suite.iter().any(|tc| tc.name == "basic_greeting"));
    assert!(tiny_suite.iter().any(|tc| tc.name == "edge_empty_input"));

    // Medium suite should include performance tests
    assert!(medium_suite
        .iter()
        .any(|tc| tc.name == "perf_long_generation"));
    assert!(medium_suite
        .iter()
        .any(|tc| tc.name == "edge_very_long_input"));

    println!("✓ Model size specific suites validation passed");
}

#[tokio::test]
async fn test_test_case_properties() {
    let registry = ComparisonTestCaseRegistry::new();
    let all_tests = registry.all();

    for test_case in all_tests {
        // All test cases should have required properties
        assert!(!test_case.name.is_empty(), "Test case has empty name");
        assert!(
            !test_case.description.is_empty(),
            "Test case '{}' has empty description",
            test_case.name
        );

        // Config should be valid
        assert!(
            test_case.config.max_tokens > 0,
            "Test case '{}' has invalid max_tokens",
            test_case.name
        );
        assert!(
            test_case.config.temperature >= 0.0,
            "Test case '{}' has negative temperature",
            test_case.name
        );
        assert!(
            test_case.config.temperature <= 2.0,
            "Test case '{}' has too high temperature",
            test_case.name
        );

        // Token ranges should be sensible
        if let (Some(min), Some(max)) =
            (test_case.expected_min_tokens, test_case.expected_max_tokens)
        {
            assert!(
                min <= max,
                "Test case '{}' has min_tokens > max_tokens",
                test_case.name
            );
            assert!(
                max <= test_case.config.max_tokens,
                "Test case '{}' expects more tokens than config allows",
                test_case.name
            );
        }

        // Name should match expected patterns
        let valid_prefixes = [
            "basic_",
            "edge_",
            "perf_",
            "regression_",
            "format_",
            "size_",
        ];
        assert!(
            valid_prefixes
                .iter()
                .any(|prefix| test_case.name.starts_with(prefix)),
            "Test case '{}' doesn't follow naming convention",
            test_case.name
        );
    }

    println!("✓ Test case properties validation passed");
}

#[tokio::test]
async fn test_comparison_test_runner_creation() {
    // Test that we can create a test runner
    let runner_result = ComparisonTestRunner::new().await;
    assert!(
        runner_result.is_ok(),
        "Failed to create test runner: {:?}",
        runner_result.err()
    );

    let runner = runner_result.unwrap();
    assert_eq!(runner.get_results().len(), 0);

    // Test summary statistics for empty runner
    let stats = runner.get_summary_statistics();
    assert_eq!(stats.total_test_suites, 0);
    assert_eq!(stats.total_test_cases, 0);
    assert_eq!(stats.success_rate, 0.0);

    println!("✓ Test runner creation validation passed");
}

/// This test demonstrates the complete test case structure and validates
/// that all required test scenarios are covered
#[tokio::test]
async fn test_complete_test_coverage() {
    let registry = ComparisonTestCaseRegistry::new();

    // Count tests by category
    let basic_count = registry.by_category(TestCaseCategory::Basic).len();
    let edge_count = registry.by_category(TestCaseCategory::EdgeCase).len();
    let perf_count = registry.by_category(TestCaseCategory::Performance).len();
    let regression_count = registry.by_category(TestCaseCategory::Regression).len();
    let format_count = registry
        .by_category(TestCaseCategory::FormatCompatibility)
        .len();
    let size_count = registry.by_category(TestCaseCategory::ModelSize).len();

    println!("Test Coverage Summary:");
    println!("  Basic functionality: {} tests", basic_count);
    println!("  Edge cases: {} tests", edge_count);
    println!("  Performance: {} tests", perf_count);
    println!("  Regression: {} tests", regression_count);
    println!("  Format compatibility: {} tests", format_count);
    println!("  Model size variations: {} tests", size_count);

    let total_tests =
        basic_count + edge_count + perf_count + regression_count + format_count + size_count;
    println!("  Total: {} tests", total_tests);

    // Ensure we have comprehensive coverage
    assert!(basic_count >= 5, "Need at least 5 basic tests");
    assert!(edge_count >= 6, "Need at least 6 edge case tests");
    assert!(perf_count >= 4, "Need at least 4 performance tests");
    assert!(regression_count >= 4, "Need at least 4 regression tests");
    assert!(
        format_count >= 3,
        "Need at least 3 format compatibility tests"
    );
    assert!(size_count >= 4, "Need at least 4 model size tests");

    assert!(
        total_tests >= 26,
        "Need at least 26 total tests for comprehensive coverage"
    );

    println!("✓ Complete test coverage validation passed");
}
