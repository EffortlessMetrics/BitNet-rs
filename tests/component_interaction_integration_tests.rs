//! # Component Interaction Integration Tests
//!
//! This test file runs the component interaction test suite to validate
//! cross-crate component interactions, data flow, configuration propagation,
//! error handling, and resource management.

use bitnet_tests::{TestConfig, TestHarness, TestStatus};
use bitnet_tests::integration::component_interaction_tests::{ComponentInteractionTestSuite, CrossCrateDataFlowTest, ConfigurationPropagationTest, ErrorHandlingAndRecoveryTest, ResourceSharingTest};
use log::LevelFilter;
use env_logger;

#[tokio::test]
async fn run_component_interaction_tests() {
    // Initialize logging for test debugging
    let _ = env_logger::builder().filter_level(LevelFilter::Debug).is_test(true).try_init();

    // Create test configuration
    let config = TestConfig {
        max_parallel_tests: 2, // Limit parallelism for component interaction tests
        test_timeout: std::time::Duration::from_secs(300),
        log_level: "debug".to_string(),
        coverage_threshold: 0.8,
        ..Default::default()
    };

    // Create test harness
    let harness = TestHarness::new(config).await.expect("Failed to create test harness");

    // Run component interaction test suite
    let suite = ComponentInteractionTestSuite;
    let result = harness
        .run_test_suite(suite)
        .await
        .expect("Failed to run component interaction test suite");

    // Verify test results
    println!("Component Interaction Test Results:");
    println!("  Total tests: {}", result.summary.total_tests);
    println!("  Passed: {}", result.summary.passed);
    println!("  Failed: {}", result.summary.failed);
    println!("  Success rate: {:.2}%", result.summary.success_rate * 100.0);
    println!("  Total duration: {:?}", result.summary.total_duration);

    // Assert all tests passed
    assert_eq!(result.summary.failed, 0, "Some component interaction tests failed");
    assert!(result.summary.passed > 0, "No component interaction tests were run");
    assert!(
        result.summary.success_rate >= 0.8,
        "Success rate too low: {:.2}%",
        result.summary.success_rate * 100.0
    );

    // Print detailed results for each test
    for test_result in &result.test_results {
        println!("\nTest: {}", test_result.test_name);
        println!("  Status: {:?}", test_result.status);
        println!("  Duration: {:?}", test_result.duration);

        if let Some(error) = &test_result.error {
            println!("  Error: {}", error);
        }

        // Print custom metrics
        if !test_result.metrics.custom_metrics.is_empty() {
            println!("  Metrics:");
            for (key, value) in &test_result.metrics.custom_metrics {
                println!("    {}: {}", key, value);
            }
        }
    }
}

#[tokio::test]
async fn test_cross_crate_data_flow_validation() {
    let _ = env_logger::builder().filter_level(LevelFilter::Debug).is_test(true).try_init();

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    let test_case = CrossCrateDataFlowTest;
    let result = harness.run_single_test(Box::new(test_case)).await.unwrap();

    assert!(matches!(result.status, TestStatus::Passed));

    // Verify specific metrics for data flow
    let metrics = &result.metrics.custom_metrics;
    assert!(metrics.get("tokenizer_encode_calls").unwrap_or(&0.0) > &0.0);
    assert!(metrics.get("tokenizer_decode_calls").unwrap_or(&0.0) > &0.0);
    assert!(metrics.get("model_forward_calls").unwrap_or(&0.0) > &0.0);
}

#[tokio::test]
async fn test_configuration_propagation_validation() {
    let _ = env_logger::builder().filter_level(LevelFilter::Debug).is_test(true).try_init();

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    let test_case = ConfigurationPropagationTest;
    let result = harness.run_single_test(Box::new(test_case)).await.unwrap();

    assert!(matches!(result.status, TestStatus::Passed));

    // Verify configuration metrics
    let metrics = &result.metrics.custom_metrics;
    assert!(metrics.get("model_vocab_size").unwrap_or(&0.0) > &0.0);
    assert!(metrics.get("configurations_tested").unwrap_or(&0.0) >= &3.0);
}

#[tokio::test]
async fn test_error_handling_and_recovery_validation() {
    let _ = env_logger::builder().filter_level(LevelFilter::Debug).is_test(true).try_init();

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    let test_case = ErrorHandlingAndRecoveryTest;
    let result = harness.run_single_test(Box::new(test_case)).await.unwrap();

    assert!(matches!(result.status, TestStatus::Passed));

    // Verify error handling metrics
    let metrics = &result.metrics.custom_metrics;
    assert!(metrics.get("error_scenarios_tested").unwrap_or(&0.0) > &0.0);
    assert!(metrics.get("recovery_scenarios_successful").unwrap_or(&0.0) >= &0.0);
}

#[tokio::test]
async fn test_resource_sharing_validation() {
    let _ = env_logger::builder().filter_level(LevelFilter::Debug).is_test(true).try_init();

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    let test_case = ResourceSharingTest;
    let result = harness.run_single_test(Box::new(test_case)).await.unwrap();

    assert!(matches!(result.status, TestStatus::Passed));

    // Verify resource sharing metrics
    let metrics = &result.metrics.custom_metrics;
    assert!(metrics.get("shared_model_accesses").unwrap_or(&0.0) > &0.0);
    assert!(metrics.get("resource_sharing_scenarios").unwrap_or(&0.0) >= &2.0);
}
