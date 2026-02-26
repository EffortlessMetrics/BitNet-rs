#![cfg(feature = "integration-tests")]
//! Basic test for the reporting system
//!
//! This test validates that the core reporting functionality works
//! with minimal dependencies and simple test data.

use bitnet_tests::reporting::formats::{HtmlReporter, JsonReporter, MarkdownReporter};
use bitnet_tests::reporting::{ReportFormat, TestReporter};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
use bitnet_tests::units::{BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;

/// Create simple test data for basic reporting
fn create_simple_test_data() -> Vec<TestSuiteResult> {
    vec![TestSuiteResult {
        suite_name: "basic_test_suite".to_string(),
        total_duration: Duration::from_secs(5),
        test_results: vec![
            TestResult {
                test_name: "test_pass".to_string(),
                status: TestStatus::Passed,
                duration: Duration::from_secs(2),
                metrics: TestMetrics {
                    memory_peak: Some(BYTES_PER_MB),          // 1 MB
                    memory_average: Some(512 * BYTES_PER_KB), // 512 KB
                    cpu_time: Some(Duration::from_millis(1500)),
                    wall_time: Duration::from_secs(2),
                    custom_metrics: HashMap::new(),
                    assertions: 3,
                    operations: 5,
                },
                error: None,
                stack_trace: None,
                artifacts: Vec::new(),
                start_time: std::time::SystemTime::now() - Duration::from_secs(2),
                end_time: std::time::SystemTime::now(),
                metadata: HashMap::new(),
            },
            TestResult {
                test_name: "test_fail".to_string(),
                status: TestStatus::Failed,
                duration: Duration::from_secs(3),
                metrics: TestMetrics {
                    memory_peak: Some(2048 * BYTES_PER_KB), // 2 MB
                    memory_average: Some(BYTES_PER_MB),     // 1 MB
                    cpu_time: Some(Duration::from_millis(2500)),
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
            peak_memory: Some(2048 * BYTES_PER_KB),
            total_assertions: 5,
        },
        environment: HashMap::new(),
        configuration: HashMap::new(),
        start_time: std::time::SystemTime::now() - Duration::from_secs(5),
        end_time: std::time::SystemTime::now(),
    }]
}

#[tokio::test]
async fn test_html_basic_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("basic_report.html");

    let reporter = HtmlReporter::new(false); // Non-interactive mode
    let test_data = create_simple_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify report was generated
    assert_eq!(result.format, ReportFormat::Html);
    assert!(output_path.exists());
    assert!(result.size_bytes > 100);

    // Verify HTML content
    let content = fs::read_to_string(&output_path).await.unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("BitNet-rs Test Report"));
    assert!(content.contains("basic_test_suite"));
    assert!(content.contains("test_pass"));
    assert!(content.contains("test_fail"));

    println!("✅ HTML basic report generated: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_json_basic_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("basic_report.json");

    let reporter = JsonReporter::new();
    let test_data = create_simple_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify report was generated
    assert_eq!(result.format, ReportFormat::Json);
    assert!(output_path.exists());
    assert!(result.size_bytes > 50);

    // Verify JSON content
    let content = fs::read_to_string(&output_path).await.unwrap();
    let json_value: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert!(json_value["metadata"].is_object());
    assert!(json_value["test_suites"].is_array());
    assert!(json_value["summary"].is_object());
    assert_eq!(json_value["summary"]["total_tests"], 2);
    assert_eq!(json_value["summary"]["total_passed"], 1);
    assert_eq!(json_value["summary"]["total_failed"], 1);

    println!("✅ JSON basic report generated: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_markdown_basic_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("basic_report.md");

    let reporter = MarkdownReporter::new();
    let test_data = create_simple_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify report was generated
    assert_eq!(result.format, ReportFormat::Markdown);
    assert!(output_path.exists());
    assert!(result.size_bytes > 50);

    // Verify Markdown content
    let content = fs::read_to_string(&output_path).await.unwrap();
    assert!(content.contains("# BitNet-rs Test Report"));
    assert!(content.contains("## Summary"));
    assert!(content.contains("basic_test_suite"));
    assert!(content.contains("test_pass"));
    assert!(content.contains("test_fail"));
    assert!(content.contains("✅"));
    assert!(content.contains("❌"));

    println!("✅ Markdown basic report generated: {} bytes", result.size_bytes);
}
