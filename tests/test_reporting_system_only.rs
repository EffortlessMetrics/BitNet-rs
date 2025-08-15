//! Test only the reporting system functionality
//!
//! This test focuses specifically on the reporting system without
//! dependencies on other modules that may have compilation issues.

use bitnet_tests::reporting::{
    formats::{HtmlReporter, JsonReporter, JunitReporter, MarkdownReporter},
    ReportConfig, ReportFormat, ReportingManager, TestReporter,
};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;

/// Create simple test data for reporting
fn create_simple_test_data() -> Vec<TestSuiteResult> {
    vec![TestSuiteResult {
        suite_name: "simple_test_suite".to_string(),
        total_duration: Duration::from_secs(5),
        test_results: vec![
            TestResult {
                test_name: "test_pass".to_string(),
                status: TestStatus::Passed,
                duration: Duration::from_secs(2),
                metrics: TestMetrics {
                    memory_peak: Some(1024),
                    memory_average: Some(512),
                    cpu_time: Some(Duration::from_secs(1)),
                    wall_time: Duration::from_secs(2),
                    custom_metrics: HashMap::new(),
                    assertions: 3,
                    operations: 5,
                },
                error: None,
                stack_trace: None,
                artifacts: Vec::new(),
                start_time: std::time::SystemTime::now() - Duration::from_secs(5),
                end_time: std::time::SystemTime::now() - Duration::from_secs(3),
                metadata: HashMap::new(),
            },
            TestResult {
                test_name: "test_fail".to_string(),
                status: TestStatus::Failed,
                duration: Duration::from_secs(3),
                metrics: TestMetrics {
                    memory_peak: Some(2048),
                    memory_average: Some(1024),
                    cpu_time: Some(Duration::from_secs(2)),
                    wall_time: Duration::from_secs(3),
                    custom_metrics: HashMap::new(),
                    assertions: 2,
                    operations: 3,
                },
                error: Some("Test assertion failed".to_string()),
                stack_trace: None,
                artifacts: Vec::new(),
                start_time: std::time::SystemTime::now() - Duration::from_secs(3),
                end_time: std::time::SystemTime::now(),
                metadata: HashMap::new(),
            },
        ],
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
            total_assertions: 5,
        },
        environment: HashMap::new(),
        configuration: HashMap::new(),
        start_time: std::time::SystemTime::now() - Duration::from_secs(5),
        end_time: std::time::SystemTime::now(),
    }]
}

#[tokio::test]
async fn test_json_reporter() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_report.json");

    let reporter = JsonReporter::new();
    let test_data = create_simple_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    assert_eq!(result.format, ReportFormat::Json);
    assert!(output_path.exists());
    assert!(result.size_bytes > 0);

    let content = fs::read_to_string(&output_path).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Verify JSON structure
    assert!(parsed["metadata"].is_object());
    assert!(parsed["test_suites"].is_array());
    assert!(parsed["summary"].is_object());

    println!("✅ JSON report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_html_reporter() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_report.html");

    let reporter = HtmlReporter::new(true);
    let test_data = create_simple_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    assert_eq!(result.format, ReportFormat::Html);
    assert!(output_path.exists());
    assert!(result.size_bytes > 0);

    let content = fs::read_to_string(&output_path).await.unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("BitNet.rs Test Report"));

    println!("✅ HTML report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_markdown_reporter() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_report.md");

    let reporter = MarkdownReporter::new();
    let test_data = create_simple_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    assert_eq!(result.format, ReportFormat::Markdown);
    assert!(output_path.exists());
    assert!(result.size_bytes > 0);

    let content = fs::read_to_string(&output_path).await.unwrap();
    assert!(content.contains("# BitNet.rs Test Report"));
    assert!(content.contains("## Summary"));

    println!("✅ Markdown report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_junit_reporter() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_report.xml");

    let reporter = JunitReporter::new();
    let test_data = create_simple_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    assert_eq!(result.format, ReportFormat::Junit);
    assert!(output_path.exists());
    assert!(result.size_bytes > 0);

    let content = fs::read_to_string(&output_path).await.unwrap();
    assert!(content.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assert!(content.contains("<testsuites"));

    println!("✅ JUnit XML report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_reporting_manager() {
    let temp_dir = TempDir::new().unwrap();

    let config = ReportConfig {
        output_dir: temp_dir.path().to_path_buf(),
        formats: vec![ReportFormat::Json, ReportFormat::Html],
        include_artifacts: false,
        generate_coverage: false,
        interactive_html: true,
    };

    let manager = ReportingManager::new(config);
    let test_data = create_simple_test_data();

    let results = manager.generate_all_reports(&test_data).await.unwrap();

    assert_eq!(results.len(), 2); // JSON and HTML

    // Verify files were created
    assert!(temp_dir.path().join("test_report.json").exists());
    assert!(temp_dir.path().join("test_report.html").exists());
    assert!(temp_dir.path().join("report_summary.md").exists());

    println!("✅ ReportingManager generated {} reports successfully", results.len());
}
