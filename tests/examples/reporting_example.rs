//! Example demonstrating the BitNet.rs reporting system
//!
//! This example shows how to use the comprehensive reporting system
//! to generate HTML, JSON, JUnit XML, and Markdown reports.

use bitnet_tests::reporting::{
    formats::{HtmlReporter, JsonReporter, JunitReporter, MarkdownReporter},
    ReportConfig, ReportFormat, ReportingManager, TestReporter,
};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs;

/// Create example test data
fn create_example_test_data() -> Vec<TestSuiteResult> {
    vec![TestSuiteResult {
        suite_name: "example_test_suite".to_string(),
        total_duration: Duration::from_secs(8),
        test_results: vec![
            TestResult {
                test_name: "test_successful_operation".to_string(),
                status: TestStatus::Passed,
                duration: Duration::from_secs(3),
                metrics: TestMetrics {
                    memory_peak: Some(BYTES_PER_MB),   // 1MB
                    memory_average: Some(512 * 1024), // 512KB
                    cpu_time: Some(Duration::from_secs(2)),
                    wall_time: Duration::from_secs(3),
                    custom_metrics: {
                        let mut metrics = HashMap::new();
                        metrics.insert("operations_per_second".to_string(), 1000.0);
                        metrics.insert("accuracy".to_string(), 0.98);
                        metrics
                    },
                    assertions: 5,
                    operations: 10,
                },
                error: None,
                stack_trace: None,
                artifacts: Vec::new(),
                start_time: std::time::SystemTime::now() - Duration::from_secs(8),
                end_time: std::time::SystemTime::now() - Duration::from_secs(5),
                metadata: HashMap::new(),
            },
            TestResult {
                test_name: "test_error_handling".to_string(),
                status: TestStatus::Failed,
                duration: Duration::from_secs(5),
                metrics: TestMetrics {
                    memory_peak: Some(2048 * 1024),    // 2MB
                    memory_average: Some(BYTES_PER_MB), // 1MB
                    cpu_time: Some(Duration::from_secs(4)),
                    wall_time: Duration::from_secs(5),
                    custom_metrics: HashMap::new(),
                    assertions: 3,
                    operations: 5,
                },
                error: Some("Expected error condition was not handled correctly".to_string()),
                stack_trace: Some("at test_error_handling (example.rs:42)".to_string()),
                artifacts: Vec::new(),
                start_time: std::time::SystemTime::now() - Duration::from_secs(5),
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
            total_duration: Duration::from_secs(8),
            average_duration: Duration::from_secs(4),
            peak_memory: Some(2048 * 1024),
            total_assertions: 8,
        },
        environment: {
            let mut env = HashMap::new();
            env.insert("rust_version".to_string(), "1.75.0".to_string());
            env.insert("os".to_string(), "Windows".to_string());
            env
        },
        configuration: {
            let mut config = HashMap::new();
            config.insert("mode".to_string(), "example".to_string());
            config
        },
        start_time: std::time::SystemTime::now() - Duration::from_secs(8),
        end_time: std::time::SystemTime::now(),
    }]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("BitNet.rs Reporting System Example");
    println!("==================================\n");

    // Create output directory
    let output_dir = PathBuf::from("example_reports");
    fs::create_dir_all(&output_dir).await?;

    let test_data = create_example_test_data();

    // Example 1: Generate individual reports
    println!("1. Generating individual reports...");

    // JSON Report
    let json_reporter = JsonReporter::new();
    let json_result =
        json_reporter.generate_report(&test_data, &output_dir.join("example_report.json")).await?;
    println!("   ✅ JSON: {} bytes", json_result.size_bytes);

    // HTML Report (interactive)
    let html_reporter = HtmlReporter::new(true);
    let html_result =
        html_reporter.generate_report(&test_data, &output_dir.join("example_report.html")).await?;
    println!("   ✅ HTML: {} bytes", html_result.size_bytes);

    // JUnit XML Report
    let junit_reporter = JunitReporter::new();
    let junit_result =
        junit_reporter.generate_report(&test_data, &output_dir.join("example_report.xml")).await?;
    println!("   ✅ JUnit XML: {} bytes", junit_result.size_bytes);

    // Markdown Report
    let markdown_reporter = MarkdownReporter::new();
    let markdown_result = markdown_reporter
        .generate_report(&test_data, &output_dir.join("example_report.md"))
        .await?;
    println!("   ✅ Markdown: {} bytes", markdown_result.size_bytes);

    println!();

    // Example 2: Use ReportingManager for multiple formats
    println!("2. Using ReportingManager for multiple formats...");

    let manager_dir = output_dir.join("manager_output");
    let config = ReportConfig {
        output_dir: manager_dir.clone(),
        formats: vec![
            ReportFormat::Html,
            ReportFormat::Json,
            ReportFormat::Junit,
            ReportFormat::Markdown,
        ],
        include_artifacts: true,
        generate_coverage: false,
        interactive_html: true,
    };

    let manager = ReportingManager::new(config);
    let results = manager.generate_all_reports(&test_data).await?;

    for result in results {
        println!("   ✅ {:?}: {} bytes", result.format, result.size_bytes);
    }

    println!();
    println!("📁 Reports saved to: {}", output_dir.display());
    println!("   • Individual reports: example_report.*");
    println!("   • Manager reports: manager_output/test_report.*");
    println!("   • Summary: manager_output/report_summary.md");

    // Display a sample of the generated content
    println!("\n3. Sample content from JSON report:");
    let json_content = fs::read_to_string(output_dir.join("example_report.json")).await?;
    let json_parsed: serde_json::Value = serde_json::from_str(&json_content)?;
    println!("   Generator: {}", json_parsed["metadata"]["generator"]);
    println!("   Total Tests: {}", json_parsed["summary"]["total_tests"]);
    println!(
        "   Success Rate: {:.1}%",
        json_parsed["summary"]["overall_success_rate"].as_f64().unwrap() * 100.0
    );

    println!("\n🎉 Example completed successfully!");
    println!("Open the HTML report in your browser to see the interactive features.");

    Ok(())
}
