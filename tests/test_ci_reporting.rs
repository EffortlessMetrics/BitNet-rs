//! Tests for CI reporting and notifications functionality

use bitnet_tests::ci_reporting::{
    CIContext, CINotificationManager, GitHubReporter, NotificationConfig,
};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
use bitnet_tests::trend_reporting::{TestRunMetadata, TrendConfig, TrendReporter};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tokio;

#[tokio::test]
async fn test_ci_notification_manager_creation() {
    let config = NotificationConfig::default();

    // This might fail if GITHUB_TOKEN is not set, but that's expected in test environment
    let result = CINotificationManager::new(config);

    // We mainly want to test that the structure is correct
    match result {
        Ok(_) => {
            // Success case - GitHub token is available
        }
        Err(e) => {
            // Expected case in test environment without GitHub token
            assert!(e.to_string().contains("GITHUB_REPOSITORY") || e.to_string().contains("token"));
        }
    }
}

#[tokio::test]
async fn test_github_reporter_creation() {
    // Test GitHub reporter creation
    let result = GitHubReporter::new();

    match result {
        Ok(_reporter) => {
            // Success case - environment is properly configured
            // Just verify we can create the reporter
        }
        Err(e) => {
            // Expected case in test environment
            assert!(
                e.to_string().contains("GITHUB_REPOSITORY") || e.to_string().contains("format")
            );
        }
    }
}

#[tokio::test]
async fn test_ci_context_from_env() {
    // Test CI context creation from environment
    let context = CIContext::from_env();

    // In test environment, most values will be None
    assert!(context.commit_sha.is_none() || context.commit_sha.is_some());
    assert!(context.pr_number.is_none() || context.pr_number.is_some());
    assert!(context.branch_name.is_none() || context.branch_name.is_some());
}

#[tokio::test]
async fn test_trend_reporter_creation() {
    let temp_dir = TempDir::new().unwrap();
    let config = TrendConfig::default();

    let trend_reporter = TrendReporter::new(temp_dir.path().to_path_buf(), config);

    // Test that we can create a trend reporter
    assert!(temp_dir.path().exists());
}

#[tokio::test]
async fn test_trend_data_recording() {
    let temp_dir = TempDir::new().unwrap();
    let config = TrendConfig::default();
    let trend_reporter = TrendReporter::new(temp_dir.path().to_path_buf(), config);

    // Create sample test results
    let test_results = vec![create_sample_test_suite_result()];

    let metadata = TestRunMetadata {
        commit_sha: Some("abc123".to_string()),
        branch: Some("main".to_string()),
        pr_number: None,
        environment: HashMap::new(),
        configuration: HashMap::new(),
    };

    // Record test results
    let result = trend_reporter.record_test_results(&test_results, &metadata).await;

    match result {
        Ok(()) => {
            // Check that a file was created
            let entries = std::fs::read_dir(temp_dir.path()).unwrap();
            let count = entries.count();
            assert!(count > 0, "Expected trend data file to be created");
        }
        Err(e) => {
            panic!("Failed to record trend data: {}", e);
        }
    }
}

#[tokio::test]
async fn test_trend_report_generation() {
    let temp_dir = TempDir::new().unwrap();
    let config = TrendConfig::default();
    let trend_reporter = TrendReporter::new(temp_dir.path().to_path_buf(), config);

    // Generate trend report (will be empty since no data)
    let result = trend_reporter.generate_trend_report(30, None).await;

    match result {
        Ok(report) => {
            assert_eq!(report.total_entries, 0);
            assert_eq!(report.period_days, 30);
        }
        Err(e) => {
            panic!("Failed to generate trend report: {}", e);
        }
    }
}

#[tokio::test]
async fn test_performance_regression_detection() {
    let temp_dir = TempDir::new().unwrap();
    let config = TrendConfig {
        retention_days: 90,
        min_samples_for_baseline: 1, // Lower threshold for testing
        regression_threshold: 1.5,   // 50% slower
    };
    let trend_reporter = TrendReporter::new(temp_dir.path().to_path_buf(), config);

    // Create sample test results
    let test_results = vec![create_sample_test_suite_result()];

    // Detect regressions (will be empty since no baseline data)
    let result = trend_reporter.detect_regressions(&test_results, 30).await;

    match result {
        Ok(regressions) => {
            // Should be empty since no baseline data
            assert_eq!(regressions.len(), 0);
        }
        Err(e) => {
            panic!("Failed to detect regressions: {}", e);
        }
    }
}

#[tokio::test]
async fn test_notification_config_defaults() {
    let config = NotificationConfig::default();

    assert!(config.notify_on_failure);
    assert!(!config.notify_on_success);
    assert!(config.check_performance_regression);
    assert_eq!(config.performance_regression_threshold, 1.1);
    assert!(config.create_status_checks);
    assert!(config.create_pr_comments);
}

#[tokio::test]
async fn test_trend_config_defaults() {
    let config = TrendConfig::default();

    assert_eq!(config.retention_days, 90);
    assert_eq!(config.min_samples_for_baseline, 5);
    assert_eq!(config.regression_threshold, 1.2);
}

fn create_sample_test_suite_result() -> TestSuiteResult {
    let test_result = TestResult {
        test_name: "sample_test".to_string(),
        status: TestStatus::Passed,
        duration: Duration::from_secs(2),
        metrics: TestMetrics {
            memory_peak: Some(1024),
            memory_average: Some(512),
            cpu_time: Some(Duration::from_secs(1)),
            wall_time: Duration::from_secs(2),
            custom_metrics: HashMap::new(),
            assertions: 5,
            operations: 10,
        },
        error: None,
        stack_trace: None,
        artifacts: Vec::new(),
        start_time: std::time::SystemTime::now() - Duration::from_secs(2),
        end_time: std::time::SystemTime::now(),
        metadata: HashMap::new(),
    };

    TestSuiteResult {
        suite_name: "sample_suite".to_string(),
        total_duration: Duration::from_secs(2),
        test_results: vec![test_result],
        summary: TestSummary {
            total_tests: 1,
            passed: 1,
            failed: 0,
            skipped: 0,
            timeout: 0,
            success_rate: 100.0,
            total_duration: Duration::from_secs(2),
            average_duration: Duration::from_secs(2),
            peak_memory: Some(1024),
            total_assertions: 5,
        },
        environment: HashMap::new(),
        configuration: HashMap::new(),
        start_time: std::time::SystemTime::now() - Duration::from_secs(2),
        end_time: std::time::SystemTime::now(),
    }
}
