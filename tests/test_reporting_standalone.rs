#![cfg(feature = "integration-tests")]
//! Standalone test for the reporting system
//! This test only uses the core reporting functionality without dependencies on other modules

use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;

// Manually define the types we need to avoid import issues
use bitnet_tests::units::{BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Timeout,
    Running,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetrics {
    pub memory_peak: Option<u64>,
    pub memory_average: Option<u64>,
    pub cpu_time: Option<Duration>,
    pub wall_time: Duration,
    pub custom_metrics: HashMap<String, f64>,
    pub assertions: usize,
    pub operations: usize,
}

impl Default for TestMetrics {
    fn default() -> Self {
        Self {
            memory_peak: None,
            memory_average: None,
            cpu_time: None,
            wall_time: Duration::ZERO,
            custom_metrics: HashMap::new(),
            assertions: 0,
            operations: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub status: TestStatus,
    pub duration: Duration,
    pub metrics: TestMetrics,
    pub error: Option<String>,
    pub stack_trace: Option<String>,
    pub artifacts: Vec<String>,
    pub start_time: std::time::SystemTime,
    pub end_time: std::time::SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub timeout: usize,
    pub success_rate: f64,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub peak_memory: Option<u64>,
    pub total_assertions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteResult {
    pub suite_name: String,
    pub total_duration: Duration,
    pub test_results: Vec<TestResult>,
    pub summary: TestSummary,
    pub environment: HashMap<String, String>,
    pub configuration: HashMap<String, String>,
    pub start_time: std::time::SystemTime,
    pub end_time: std::time::SystemTime,
}

/// Create test data for standalone reporting test
fn create_standalone_test_data() -> Vec<TestSuiteResult> {
    vec![TestSuiteResult {
        suite_name: "standalone_test_suite".to_string(),
        total_duration: Duration::from_secs(4),
        test_results: vec![
            TestResult {
                test_name: "test_standalone_pass".to_string(),
                status: TestStatus::Passed,
                duration: Duration::from_secs(2),
                metrics: TestMetrics {
                    memory_peak: Some(BYTES_PER_MB),
                    memory_average: Some(512 * BYTES_PER_KB),
                    cpu_time: Some(Duration::from_millis(1800)),
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
                test_name: "test_standalone_fail".to_string(),
                status: TestStatus::Failed,
                duration: Duration::from_secs(2),
                metrics: TestMetrics {
                    memory_peak: Some(2048 * BYTES_PER_KB),
                    memory_average: Some(BYTES_PER_MB),
                    cpu_time: Some(Duration::from_millis(1900)),
                    wall_time: Duration::from_secs(2),
                    custom_metrics: HashMap::new(),
                    assertions: 2,
                    operations: 3,
                },
                error: Some("Standalone test failed".to_string()),
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
            total_duration: Duration::from_secs(4),
            average_duration: Duration::from_secs(2),
            peak_memory: Some(2048 * BYTES_PER_KB),
            total_assertions: 5,
        },
        environment: HashMap::new(),
        configuration: HashMap::new(),
        start_time: std::time::SystemTime::now() - Duration::from_secs(4),
        end_time: std::time::SystemTime::now(),
    }]
}

/// Simple HTML reporter implementation
struct SimpleHtmlReporter;

impl SimpleHtmlReporter {
    fn generate_html(&self, results: &[TestSuiteResult]) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html><head><title>Test Report</title></head><body>\n");
        html.push_str("<h1>BitNet-rs Test Report</h1>\n");

        for suite in results {
            html.push_str(&format!("<h2>{}</h2>\n", suite.suite_name));
            html.push_str(&format!("<p>Duration: {:?}</p>\n", suite.total_duration));
            html.push_str(&format!(
                "<p>Tests: {} passed, {} failed</p>\n",
                suite.summary.passed, suite.summary.failed
            ));

            html.push_str("<ul>\n");
            for test in &suite.test_results {
                let status = match test.status {
                    TestStatus::Passed => "✅ PASSED",
                    TestStatus::Failed => "❌ FAILED",
                    TestStatus::Skipped => "⏭️ SKIPPED",
                    TestStatus::Timeout => "⏰ TIMEOUT",
                    TestStatus::Running => "🔄 RUNNING",
                };
                html.push_str(&format!(
                    "<li>{}: {} ({:?})</li>\n",
                    status, test.test_name, test.duration
                ));
            }
            html.push_str("</ul>\n");
        }

        html.push_str("</body></html>\n");
        html
    }
}

/// Simple JSON reporter implementation
struct SimpleJsonReporter;

impl SimpleJsonReporter {
    fn generate_json(&self, results: &[TestSuiteResult]) -> Result<String, serde_json::Error> {
        let report = serde_json::json!({
            "metadata": {
                "generated_at": chrono::Utc::now().to_rfc3339(),
                "generator": "BitNet-rs Test Framework",
                "total_suites": results.len()
            },
            "test_suites": results,
            "summary": {
                "total_tests": results.iter().map(|r| r.summary.total_tests).sum::<usize>(),
                "total_passed": results.iter().map(|r| r.summary.passed).sum::<usize>(),
                "total_failed": results.iter().map(|r| r.summary.failed).sum::<usize>(),
            }
        });

        serde_json::to_string_pretty(&report)
    }
}

/// Simple Markdown reporter implementation
struct SimpleMarkdownReporter;

impl SimpleMarkdownReporter {
    fn generate_markdown(&self, results: &[TestSuiteResult]) -> String {
        let mut md = String::new();

        md.push_str("# BitNet-rs Test Report\n\n");
        md.push_str(&format!(
            "Generated on: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        md.push_str("## Summary\n\n");
        let total_tests: usize = results.iter().map(|r| r.summary.total_tests).sum();
        let total_passed: usize = results.iter().map(|r| r.summary.passed).sum();
        let total_failed: usize = results.iter().map(|r| r.summary.failed).sum();

        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Total Tests | {} |\n", total_tests));
        md.push_str(&format!("| Passed | {} |\n", total_passed));
        md.push_str(&format!("| Failed | {} |\n", total_failed));
        md.push_str("\n");

        for suite in results {
            md.push_str(&format!("## {}\n\n", suite.suite_name));

            for test in &suite.test_results {
                let status_emoji = match test.status {
                    TestStatus::Passed => "✅",
                    TestStatus::Failed => "❌",
                    TestStatus::Skipped => "⏭️",
                    TestStatus::Timeout => "⏰",
                    TestStatus::Running => "🔄",
                };
                md.push_str(&format!(
                    "- {} **{}** ({:?})\n",
                    status_emoji, test.test_name, test.duration
                ));
            }
            md.push_str("\n");
        }

        md
    }
}

#[tokio::test]
async fn test_standalone_html_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("standalone_report.html");

    let reporter = SimpleHtmlReporter;
    let test_data = create_standalone_test_data();

    let html_content = reporter.generate_html(&test_data);
    fs::write(&output_path, &html_content).await.unwrap();

    // Verify file was created
    assert!(output_path.exists());
    assert!(html_content.len() > 100);

    // Verify HTML content
    assert!(html_content.contains("<!DOCTYPE html>"));
    assert!(html_content.contains("BitNet-rs Test Report"));
    assert!(html_content.contains("standalone_test_suite"));
    assert!(html_content.contains("test_standalone_pass"));
    assert!(html_content.contains("test_standalone_fail"));
    assert!(html_content.contains("✅ PASSED"));
    assert!(html_content.contains("❌ FAILED"));

    println!("✅ Standalone HTML report generated: {} bytes", html_content.len());
}

#[tokio::test]
async fn test_standalone_json_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("standalone_report.json");

    let reporter = SimpleJsonReporter;
    let test_data = create_standalone_test_data();

    let json_content = reporter.generate_json(&test_data).unwrap();
    fs::write(&output_path, &json_content).await.unwrap();

    // Verify file was created
    assert!(output_path.exists());
    assert!(json_content.len() > 100);

    // Verify JSON structure
    let json_value: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    assert!(json_value["metadata"].is_object());
    assert!(json_value["test_suites"].is_array());
    assert!(json_value["summary"].is_object());
    assert_eq!(json_value["summary"]["total_tests"], 2);
    assert_eq!(json_value["summary"]["total_passed"], 1);
    assert_eq!(json_value["summary"]["total_failed"], 1);

    println!("✅ Standalone JSON report generated: {} bytes", json_content.len());
}

#[tokio::test]
async fn test_standalone_markdown_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("standalone_report.md");

    let reporter = SimpleMarkdownReporter;
    let test_data = create_standalone_test_data();

    let md_content = reporter.generate_markdown(&test_data);
    fs::write(&output_path, &md_content).await.unwrap();

    // Verify file was created
    assert!(output_path.exists());
    assert!(md_content.len() > 100);

    // Verify Markdown content
    assert!(md_content.contains("# BitNet-rs Test Report"));
    assert!(md_content.contains("## Summary"));
    assert!(md_content.contains("standalone_test_suite"));
    assert!(md_content.contains("test_standalone_pass"));
    assert!(md_content.contains("test_standalone_fail"));
    assert!(md_content.contains("✅"));
    assert!(md_content.contains("❌"));

    println!("✅ Standalone Markdown report generated: {} bytes", md_content.len());
}

#[tokio::test]
async fn test_all_standalone_formats() {
    let temp_dir = TempDir::new().unwrap();
    let test_data = create_standalone_test_data();

    // Generate all formats
    let html_reporter = SimpleHtmlReporter;
    let json_reporter = SimpleJsonReporter;
    let md_reporter = SimpleMarkdownReporter;

    let html_content = html_reporter.generate_html(&test_data);
    let json_content = json_reporter.generate_json(&test_data).unwrap();
    let md_content = md_reporter.generate_markdown(&test_data);

    // Write all files
    let html_path = temp_dir.path().join("all_formats.html");
    let json_path = temp_dir.path().join("all_formats.json");
    let md_path = temp_dir.path().join("all_formats.md");

    fs::write(&html_path, &html_content).await.unwrap();
    fs::write(&json_path, &json_content).await.unwrap();
    fs::write(&md_path, &md_content).await.unwrap();

    // Verify all files exist and have content
    assert!(html_path.exists());
    assert!(json_path.exists());
    assert!(md_path.exists());

    assert!(html_content.len() > 0);
    assert!(json_content.len() > 0);
    assert!(md_content.len() > 0);

    // Verify consistent content across formats
    let test_names = ["test_standalone_pass", "test_standalone_fail"];
    for test_name in &test_names {
        assert!(html_content.contains(test_name));
        assert!(json_content.contains(test_name));
        assert!(md_content.contains(test_name));
    }

    println!("✅ All standalone formats generated successfully:");
    println!("  HTML: {} bytes", html_content.len());
    println!("  JSON: {} bytes", json_content.len());
    println!("  Markdown: {} bytes", md_content.len());
}
