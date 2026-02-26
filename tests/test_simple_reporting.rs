#![cfg(feature = "integration-tests")]
//! Simple test for the core reporting system functionality

use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;

// Import only the essential types we need
use bitnet_tests::reporting::formats::{HtmlReporter, JsonReporter, MarkdownReporter};
use bitnet_tests::reporting::{ReportFormat, TestReporter};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};

/// Create minimal test data for reporting
fn create_minimal_test_data() -> Vec<TestSuiteResult> {
    vec![TestSuiteResult {
        suite_name: "simple_test_suite".to_string(),
        total_duration: Duration::from_secs(3),
        test_results: vec![
            TestResult {
                test_name: "test_simple_pass".to_string(),
                status: TestStatus::Passed,
                duration: Duration::from_secs(1),
                metrics: TestMetrics {
                    memory_peak: Some(1024),
                    memory_average: Some(512),
                    cpu_time: Some(Duration::from_millis(800)),
                    wall_time: Duration::from_secs(1),
                    custom_metrics: HashMap::new(),
                    assertions: 2,
                    operations: 3,
                },
                error: None,
                stack_trace: None,
                artifacts: Vec::new(),
                start_time: std::time::SystemTime::now() - Duration::from_secs(1),
                end_time: std::time::SystemTime::now(),
                metadata: HashMap::new(),
            },
            TestResult {
                test_name: "test_simple_fail".to_string(),
                status: TestStatus::Failed,
                duration: Duration::from_secs(2),
                metrics: TestMetrics {
                    memory_peak: Some(2048),
                    memory_average: Some(1024),
                    cpu_time: Some(Duration::from_millis(1500)),
                    wall_time: Duration::from_secs(2),
                    custom_metrics: HashMap::new(),
                    assertions: 1,
                    operations: 2,
                },
                error: Some("Simple test failed".to_string()),
                stack_trace: None,
                artifacts: Vec::new(),
                start_time: std::time::SystemTime::now() - Duration::from_secs(2),
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
            total_duration: Duration::from_secs(3),
            average_duration: Duration::from_millis(1500),
            peak_memory: Some(2048),
            total_assertions: 3,
        },
        environment: HashMap::new(),
        configuration: HashMap::new(),
        start_time: std::time::SystemTime::now() - Duration::from_secs(3),
        end_time: std::time::SystemTime::now(),
    }]
}

#[tokio::test]
async fn test_html_report_simple() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("simple_report.html");

    let reporter = HtmlReporter::new(false); // Non-interactive
    let test_data = create_minimal_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify basic properties
    assert_eq!(result.format, ReportFormat::Html);
    assert!(output_path.exists());
    assert!(result.size_bytes > 0);

    // Verify HTML content contains expected elements
    let content = fs::read_to_string(&output_path).await.unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("BitNet-rs Test Report"));
    assert!(content.contains("simple_test_suite"));
    assert!(content.contains("test_simple_pass"));
    assert!(content.contains("test_simple_fail"));

    println!("✅ HTML report: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_json_report_simple() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("simple_report.json");

    let reporter = JsonReporter::new();
    let test_data = create_minimal_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify basic properties
    assert_eq!(result.format, ReportFormat::Json);
    assert!(output_path.exists());
    assert!(result.size_bytes > 0);

    // Verify JSON structure
    let content = fs::read_to_string(&output_path).await.unwrap();
    let json_value: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert!(json_value["metadata"].is_object());
    assert!(json_value["test_suites"].is_array());
    assert!(json_value["summary"].is_object());
    assert_eq!(json_value["summary"]["total_tests"], 2);
    assert_eq!(json_value["summary"]["total_passed"], 1);
    assert_eq!(json_value["summary"]["total_failed"], 1);

    println!("✅ JSON report: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_markdown_report_simple() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("simple_report.md");

    let reporter = MarkdownReporter::new();
    let test_data = create_minimal_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify basic properties
    assert_eq!(result.format, ReportFormat::Markdown);
    assert!(output_path.exists());
    assert!(result.size_bytes > 0);

    // Verify Markdown structure
    let content = fs::read_to_string(&output_path).await.unwrap();
    assert!(content.contains("# BitNet-rs Test Report"));
    assert!(content.contains("## Summary"));
    assert!(content.contains("simple_test_suite"));
    assert!(content.contains("test_simple_pass"));
    assert!(content.contains("test_simple_fail"));
    assert!(content.contains("✅"));
    assert!(content.contains("❌"));

    println!("✅ Markdown report: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_all_formats_consistency() {
    let temp_dir = TempDir::new().unwrap();
    let test_data = create_minimal_test_data();

    // Generate all three formats
    let html_path = temp_dir.path().join("consistency.html");
    let json_path = temp_dir.path().join("consistency.json");
    let md_path = temp_dir.path().join("consistency.md");

    let html_reporter = HtmlReporter::new(false);
    let json_reporter = JsonReporter::new();
    let md_reporter = MarkdownReporter::new();

    let html_result = html_reporter.generate_report(&test_data, &html_path).await.unwrap();
    let json_result = json_reporter.generate_report(&test_data, &json_path).await.unwrap();
    let md_result = md_reporter.generate_report(&test_data, &md_path).await.unwrap();

    // All should be generated successfully
    assert!(html_result.size_bytes > 0);
    assert!(json_result.size_bytes > 0);
    assert!(md_result.size_bytes > 0);

    // Read all contents
    let html_content = fs::read_to_string(&html_path).await.unwrap();
    let json_content = fs::read_to_string(&json_path).await.unwrap();
    let md_content = fs::read_to_string(&md_path).await.unwrap();

    // All should contain the same test names
    let test_names = ["test_simple_pass", "test_simple_fail"];
    for test_name in &test_names {
        assert!(html_content.contains(test_name));
        assert!(json_content.contains(test_name));
        assert!(md_content.contains(test_name));
    }

    // All should contain the suite name
    assert!(html_content.contains("simple_test_suite"));
    assert!(json_content.contains("simple_test_suite"));
    assert!(md_content.contains("simple_test_suite"));

    println!(
        "✅ All formats consistent: HTML={}, JSON={}, MD={} bytes",
        html_result.size_bytes, json_result.size_bytes, md_result.size_bytes
    );
}
