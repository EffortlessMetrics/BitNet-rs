//! Minimal test for the reporting system implementation

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tempfile::TempDir;

// Import only the reporting types we need
use bitnet_tests::reporting::{ReportConfig, ReportFormat, ReportingManager};
use bitnet_tests::results::{
    TestArtifact, TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary,
};

#[tokio::test]
async fn test_json_reporter_basic() {
    // Create a temporary directory for test output
    let temp_dir = TempDir::new().unwrap();

    // Create test configuration for JSON only
    let config = ReportConfig {
        output_dir: temp_dir.path().to_path_buf(),
        formats: vec![ReportFormat::Json],
        include_artifacts: false,
        generate_coverage: false,
        interactive_html: false,
    };

    // Create reporting manager
    let manager = ReportingManager::new(config);

    // Create sample test results
    let test_results = vec![TestResult {
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
        start_time: SystemTime::now() - Duration::from_secs(2),
        end_time: SystemTime::now(),
        metadata: HashMap::new(),
    }];

    let suite_result = TestSuiteResult {
        suite_name: "sample_test_suite".to_string(),
        total_duration: Duration::from_secs(2),
        test_results,
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
        start_time: SystemTime::now() - Duration::from_secs(2),
        end_time: SystemTime::now(),
    };

    // Generate reports
    let results = manager.generate_all_reports(&[suite_result]).await.unwrap();

    // Verify reports were generated
    assert_eq!(results.len(), 1); // JSON only

    // Check that file exists
    let result = &results[0];
    assert!(result.output_path.exists());
    assert!(result.size_bytes > 0);

    // Verify it's a JSON file
    assert_eq!(result.output_path.extension().unwrap(), "json");

    // Verify summary report was created
    let summary_path = temp_dir.path().join("report_summary.md");
    assert!(summary_path.exists());
}
