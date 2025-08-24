#![cfg(feature = "integration-tests")]
//! Test for the reporting system implementation
#![cfg(feature = "reporting")]

use bitnet_tests::reporting::{ReportConfig, ReportFormat, ReportingManager};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;

#[tokio::test]
async fn test_reporting_system_basic() {
    // Create a temporary directory for test output
    let temp_dir = TempDir::new().unwrap();

    // Create test configuration
    let config = ReportConfig {
        output_dir: temp_dir.path().to_path_buf(),
        formats: vec![ReportFormat::Json, ReportFormat::Html, ReportFormat::Markdown],
        include_artifacts: false,
        generate_coverage: false,
        interactive_html: true,
    };

    // Create reporting manager
    let manager = ReportingManager::new(config);

    // Create sample test results
    let test_results = vec![
        TestResult {
            test_name: "test_sample_pass".to_string(),
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
        },
        TestResult {
            test_name: "test_sample_fail".to_string(),
            status: TestStatus::Failed,
            duration: Duration::from_secs(3),
            metrics: TestMetrics {
                memory_peak: Some(2048),
                memory_average: Some(1024),
                cpu_time: Some(Duration::from_secs(2)),
                wall_time: Duration::from_secs(3),
                custom_metrics: HashMap::new(),
                assertions: 3,
                operations: 5,
            },
            error: Some("Sample test failure".to_string()),
            stack_trace: None,
            artifacts: Vec::new(),
            start_time: std::time::SystemTime::now() - Duration::from_secs(3),
            end_time: std::time::SystemTime::now(),
            metadata: HashMap::new(),
        },
    ];

    let suite_result = TestSuiteResult {
        suite_name: "sample_test_suite".to_string(),
        total_duration: Duration::from_secs(5),
        test_results,
        summary: TestSummary {
            total_tests: 2,
            passed: 1,
            failed: 1,
            skipped: 0,
            timeout: 0,
            success_rate: 50.0,
            total_duration: Duration::from_secs(5),
            average_duration: Duration::from_millis(2500),
            peak_memory: Some(2048),
            total_assertions: 8,
        },
        environment: HashMap::new(),
        configuration: HashMap::new(),
        start_time: std::time::SystemTime::now() - Duration::from_secs(5),
        end_time: std::time::SystemTime::now(),
    };

    // Generate reports
    let results = manager.generate_all_reports(&[suite_result]).await.unwrap();

    // Verify reports were generated
    assert_eq!(results.len(), 3); // JSON, HTML, Markdown

    // Check that files exist
    for result in &results {
        assert!(result.output_path.exists());
        assert!(result.size_bytes > 0);
    }

    // Verify summary report was created
    let summary_path = temp_dir.path().join("report_summary.md");
    assert!(summary_path.exists());
}
