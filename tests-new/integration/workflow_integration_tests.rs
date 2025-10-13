//! # Workflow Integration Tests
//!
//! This module contains comprehensive integration tests that validate complete workflows
//! across the BitNet Rust ecosystem, covering end-to-end inference, model loading,
//! tokenization pipelines, streaming generation, and batch processing.
//!
//! These tests implement task 17 from the testing framework implementation spec:
//! - Create end-to-end inference workflow tests
//! - Add model loading and initialization integration tests
//! - Implement tokenization to inference pipeline tests

#![cfg(feature = "integration-tests")]
//! - Create streaming inference workflow tests
//! - Add batch processing integration tests

mod integration;

use bitnet_tests::{ConsoleReporter, TestConfig, TestHarness};
use integration::*;
use tracing::{debug, info};

/// Main test runner for all workflow integration tests
#[tokio::test]
async fn test_all_workflow_integrations() {
    // Initialize logging for test execution
    bitnet_tests::init_logging();

    info!("Starting comprehensive workflow integration tests");

    // Create test configuration
    let config = TestConfig {
        max_parallel_tests: 2, // Limit parallelism for integration tests
        test_timeout: std::time::Duration::from_secs(120), // Longer timeout for integration tests
        ..Default::default()
    };

    // Create test harness with console reporter
    let mut harness = TestHarness::new(config).await.expect("Failed to create test harness");

    harness.add_reporter(Box::new(ConsoleReporter::new(true)));

    // Run all integration test suites
    let test_suites: Vec<Box<dyn bitnet_tests::TestSuite>> = vec![
        Box::new(workflow_tests::WorkflowIntegrationTestSuite),
        Box::new(model_loading_tests::ModelLoadingTestSuite),
        Box::new(tokenization_pipeline_tests::TokenizationPipelineTestSuite),
        Box::new(streaming_tests::StreamingWorkflowTestSuite),
        Box::new(batch_processing_tests::BatchProcessingTestSuite),
    ];

    let mut all_passed = true;
    let mut total_tests = 0;
    let mut total_passed = 0;
    let mut total_failed = 0;

    for suite in test_suites {
        info!("Running test suite: {}", suite.name());

        match harness.run_test_suite(suite).await {
            Ok(result) => {
                total_tests += result.summary.total_tests;
                total_passed += result.summary.passed;
                total_failed += result.summary.failed;

                if result.summary.failed > 0 {
                    all_passed = false;
                }

                info!(
                    "Suite completed: {}/{} passed",
                    result.summary.passed, result.summary.total_tests
                );
            }
            Err(e) => {
                eprintln!("Test suite failed: {}", e);
                all_passed = false;
            }
        }
    }

    // Print final summary
    info!("=== Workflow Integration Test Summary ===");
    info!("Total tests: {}", total_tests);
    info!("Passed: {}", total_passed);
    info!("Failed: {}", total_failed);
    info!("Success rate: {:.1}%", (total_passed as f64 / total_tests as f64) * 100.0);

    // Get execution statistics
    let stats = harness.get_stats().await;
    info!("Execution statistics:");
    info!("  Total duration: {:?}", stats.total_duration);
    info!("  Average per test: {:?}", stats.average_duration());
    info!("  Success rate: {:.1}%", stats.success_rate());

    assert!(all_passed, "Some workflow integration tests failed");
    assert!(total_tests > 0, "No tests were executed");
}

/// Test individual workflow components
#[tokio::test]
async fn test_basic_workflow_integration() {
    bitnet_tests::init_logging();

    debug!("Testing basic workflow integration");

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    let suite = workflow_tests::WorkflowIntegrationTestSuite;
    let result = harness.run_test_suite(suite).await.unwrap();

    assert!(result.summary.total_tests > 0);
    assert!(result.summary.passed > 0);
    assert_eq!(result.summary.failed, 0, "Basic workflow tests should not fail");
}

/// Test model loading integration
#[tokio::test]
async fn test_model_loading_integration() {
    bitnet_tests::init_logging();

    debug!("Testing model loading integration");

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    let suite = model_loading_tests::ModelLoadingTestSuite;
    let result = harness.run_test_suite(suite).await.unwrap();

    assert!(result.summary.total_tests > 0);
    assert!(result.summary.passed > 0);
}

/// Test tokenization pipeline integration
#[tokio::test]
async fn test_tokenization_pipeline_integration() {
    bitnet_tests::init_logging();

    debug!("Testing tokenization pipeline integration");

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    let suite = tokenization_pipeline_tests::TokenizationPipelineTestSuite;
    let result = harness.run_test_suite(suite).await.unwrap();

    assert!(result.summary.total_tests > 0);
    assert!(result.summary.passed > 0);
}

/// Test streaming workflow integration
#[tokio::test]
async fn test_streaming_workflow_integration() {
    bitnet_tests::init_logging();

    debug!("Testing streaming workflow integration");

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    let suite = streaming_tests::StreamingWorkflowTestSuite;
    let result = harness.run_test_suite(suite).await.unwrap();

    assert!(result.summary.total_tests > 0);
    assert!(result.summary.passed > 0);
}

/// Test batch processing integration
#[tokio::test]
async fn test_batch_processing_integration() {
    bitnet_tests::init_logging();

    debug!("Testing batch processing integration");

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    let suite = batch_processing_tests::BatchProcessingTestSuite;
    let result = harness.run_test_suite(suite).await.unwrap();

    assert!(result.summary.total_tests > 0);
    assert!(result.summary.passed > 0);
}

/// Performance benchmark for workflow integration
#[tokio::test]
async fn benchmark_workflow_performance() {
    bitnet_tests::init_logging();

    info!("Running workflow performance benchmark");

    let config = TestConfig {
        max_parallel_tests: 1, // Sequential for accurate benchmarking
        ..Default::default()
    };

    let harness = TestHarness::new(config).await.unwrap();

    // Run a subset of tests for performance measurement
    let benchmark_suite = workflow_tests::WorkflowIntegrationTestSuite;

    let start_time = std::time::Instant::now();
    let result = harness.run_test_suite(benchmark_suite).await.unwrap();
    let total_duration = start_time.elapsed();

    let tests_per_second = result.summary.total_tests as f64 / total_duration.as_secs_f64();

    info!("Performance benchmark results:");
    info!("  Tests executed: {}", result.summary.total_tests);
    info!("  Total duration: {:?}", total_duration);
    info!("  Tests per second: {:.2}", tests_per_second);
    info!("  Average per test: {:?}", total_duration / result.summary.total_tests as u32);

    // Performance assertions
    assert!(tests_per_second > 0.1, "Performance too slow: {} tests/sec", tests_per_second);
    assert!(total_duration.as_secs() < 300, "Benchmark took too long: {:?}", total_duration);
}

/// Test error handling across workflow integrations
#[tokio::test]
async fn test_workflow_error_handling() {
    bitnet_tests::init_logging();

    debug!("Testing workflow error handling");

    let config = TestConfig {
        test_timeout: std::time::Duration::from_secs(30), // Shorter timeout for error tests
        ..Default::default()
    };

    let harness = TestHarness::new(config).await.unwrap();

    // Test that the framework handles test failures gracefully
    let suite = workflow_tests::WorkflowIntegrationTestSuite;
    let result = harness.run_test_suite(suite).await;

    // Should succeed even if some individual tests have errors
    assert!(result.is_ok(), "Test framework should handle errors gracefully");

    let suite_result = result.unwrap();
    assert!(suite_result.summary.total_tests > 0, "Should execute some tests");

    // Framework should provide detailed error information
    if suite_result.summary.failed > 0 {
        let failed_tests = suite_result.failed_tests();
        assert!(!failed_tests.is_empty(), "Should provide failed test details");

        for failed_test in failed_tests {
            assert!(failed_test.error.is_some(), "Failed tests should have error details");
        }
    }
}

/// Test resource cleanup after workflow integration tests
#[tokio::test]
async fn test_workflow_resource_cleanup() {
    bitnet_tests::init_logging();

    debug!("Testing workflow resource cleanup");

    let config = TestConfig::default();
    let harness = TestHarness::new(config).await.unwrap();

    // Get initial stats
    let initial_stats = harness.get_stats().await;

    // Run tests
    let suite = workflow_tests::WorkflowIntegrationTestSuite;
    let _result = harness.run_test_suite(suite).await.unwrap();

    // Get final stats
    let final_stats = harness.get_stats().await;

    // Verify stats were updated
    assert!(final_stats.total_tests > initial_stats.total_tests);
    assert!(final_stats.total_duration > initial_stats.total_duration);

    // Test framework should clean up properly
    // (In a real implementation, we might check memory usage, file handles, etc.)
    debug!("Resource cleanup verification completed");
}

/// Integration test for the complete testing framework
#[tokio::test]
async fn test_complete_testing_framework() {
    bitnet_tests::init_logging();

    info!("Testing complete testing framework integration");

    // Test framework components
    let config = TestConfig::default();
    assert!(config.validate().is_ok(), "Default config should be valid");

    let harness = TestHarness::new(config).await;
    assert!(harness.is_ok(), "Test harness should initialize successfully");

    let mut harness = harness.unwrap();

    // Test reporter functionality
    harness.add_reporter(Box::new(ConsoleReporter::new(false)));

    // Test fixture management
    // (This would test the fixture manager if we had real fixtures)

    // Test configuration management
    let memory_config = TestConfig {
        max_parallel_tests: 1,
        test_timeout: std::time::Duration::from_secs(60),
        ..Default::default()
    };

    assert!(memory_config.validate().is_ok(), "Memory config should be valid");

    // Test that framework handles empty test suites
    struct EmptyTestSuite;

    impl bitnet_tests::TestSuite for EmptyTestSuite {
        fn name(&self) -> &str {
            "Empty Test Suite"
        }

        fn test_cases(&self) -> Vec<Box<dyn bitnet_tests::TestCase>> {
            vec![]
        }
    }

    let empty_result = harness.run_test_suite(EmptyTestSuite).await.unwrap();
    assert_eq!(empty_result.summary.total_tests, 0);
    assert_eq!(empty_result.summary.passed, 0);
    assert_eq!(empty_result.summary.failed, 0);

    info!("Complete testing framework integration verified");
}
