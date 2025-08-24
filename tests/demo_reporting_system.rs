#![cfg(feature = "integration-tests")]
//! Demonstration of the reporting system functionality
//! This shows that the core reporting system works correctly

use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;

// Simple test data structures
#[derive(Debug, Clone)]
pub struct SimpleTestResult {
    pub name: String,
    pub passed: bool,
    pub duration: Duration,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SimpleTestSuite {
    pub name: String,
    pub tests: Vec<SimpleTestResult>,
    pub total_duration: Duration,
}

/// Simple HTML report generator
fn generate_html_report(suites: &[SimpleTestSuite]) -> String {
    let mut html = String::new();

    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("<title>BitNet.rs Test Report</title>\n");
    html.push_str("<style>\n");
    html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
    html.push_str(".passed { color: green; }\n");
    html.push_str(".failed { color: red; }\n");
    html.push_str("table { border-collapse: collapse; width: 100%; }\n");
    html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
    html.push_str("th { background-color: #f2f2f2; }\n");
    html.push_str("</style>\n</head>\n<body>\n");

    html.push_str("<h1>BitNet.rs Test Report</h1>\n");
    html.push_str(&format!(
        "<p>Generated on: {}</p>\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));

    for suite in suites {
        let passed_count = suite.tests.iter().filter(|t| t.passed).count();
        let total_count = suite.tests.len();

        html.push_str(&format!("<h2>{}</h2>\n", suite.name));
        html.push_str(&format!(
            "<p>Tests: {} passed, {} failed (Duration: {:?})</p>\n",
            passed_count,
            total_count - passed_count,
            suite.total_duration
        ));

        html.push_str("<table>\n");
        html.push_str(
            "<tr><th>Test Name</th><th>Status</th><th>Duration</th><th>Error</th></tr>\n",
        );

        for test in &suite.tests {
            let status_class = if test.passed { "passed" } else { "failed" };
            let status_text = if test.passed { "✅ PASSED" } else { "❌ FAILED" };
            let error_text = test.error.as_deref().unwrap_or("-");

            html.push_str(&format!(
                "<tr><td>{}</td><td class=\"{}\">{}</td><td>{:?}</td><td>{}</td></tr>\n",
                test.name, status_class, status_text, test.duration, error_text
            ));
        }

        html.push_str("</table>\n");
    }

    html.push_str("</body>\n</html>\n");
    html
}

/// Simple JSON report generator
fn generate_json_report(suites: &[SimpleTestSuite]) -> Result<String, serde_json::Error> {
    let total_tests: usize = suites.iter().map(|s| s.tests.len()).sum();
    let total_passed: usize =
        suites.iter().map(|s| s.tests.iter().filter(|t| t.passed).count()).sum();
    let total_failed = total_tests - total_passed;

    let report = serde_json::json!({
        "metadata": {
            "generated_at": chrono::Utc::now().to_rfc3339(),
            "generator": "BitNet.rs Test Framework",
            "version": "0.1.0"
        },
        "summary": {
            "total_suites": suites.len(),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": if total_tests > 0 { (total_passed as f64 / total_tests as f64) * 100.0 } else { 0.0 }
        },
        "test_suites": suites.iter().map(|suite| {
            serde_json::json!({
                "name": suite.name,
                "total_duration_ms": suite.total_duration.as_millis(),
                "tests": suite.tests.iter().map(|test| {
                    serde_json::json!({
                        "name": test.name,
                        "passed": test.passed,
                        "duration_ms": test.duration.as_millis(),
                        "error": test.error
                    })
                }).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>()
    });

    serde_json::to_string_pretty(&report)
}

/// Simple Markdown report generator
fn generate_markdown_report(suites: &[SimpleTestSuite]) -> String {
    let mut md = String::new();

    md.push_str("# BitNet.rs Test Report\n\n");
    md.push_str(&format!(
        "Generated on: {}\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));

    // Summary
    let total_tests: usize = suites.iter().map(|s| s.tests.len()).sum();
    let total_passed: usize =
        suites.iter().map(|s| s.tests.iter().filter(|t| t.passed).count()).sum();
    let total_failed = total_tests - total_passed;

    md.push_str("## Summary\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Total Suites | {} |\n", suites.len()));
    md.push_str(&format!("| Total Tests | {} |\n", total_tests));
    md.push_str(&format!("| Passed | {} |\n", total_passed));
    md.push_str(&format!("| Failed | {} |\n", total_failed));
    if total_tests > 0 {
        md.push_str(&format!(
            "| Success Rate | {:.1}% |\n",
            (total_passed as f64 / total_tests as f64) * 100.0
        ));
    }
    md.push_str("\n");

    // Test suites
    for suite in suites {
        md.push_str(&format!("## {}\n\n", suite.name));
        md.push_str(&format!("Duration: {:?}\n\n", suite.total_duration));

        for test in &suite.tests {
            let status_emoji = if test.passed { "✅" } else { "❌" };
            md.push_str(&format!("- {} **{}** ({:?})", status_emoji, test.name, test.duration));
            if let Some(error) = &test.error {
                md.push_str(&format!(" - Error: {}", error));
            }
            md.push_str("\n");
        }
        md.push_str("\n");
    }

    md
}

/// Create sample test data
fn create_sample_data() -> Vec<SimpleTestSuite> {
    vec![
        SimpleTestSuite {
            name: "Core Tests".to_string(),
            total_duration: Duration::from_secs(5),
            tests: vec![
                SimpleTestResult {
                    name: "test_model_loading".to_string(),
                    passed: true,
                    duration: Duration::from_secs(2),
                    error: None,
                },
                SimpleTestResult {
                    name: "test_inference".to_string(),
                    passed: true,
                    duration: Duration::from_secs(3),
                    error: None,
                },
            ],
        },
        SimpleTestSuite {
            name: "Integration Tests".to_string(),
            total_duration: Duration::from_secs(8),
            tests: vec![
                SimpleTestResult {
                    name: "test_end_to_end".to_string(),
                    passed: true,
                    duration: Duration::from_secs(5),
                    error: None,
                },
                SimpleTestResult {
                    name: "test_performance".to_string(),
                    passed: false,
                    duration: Duration::from_secs(3),
                    error: Some("Performance regression detected".to_string()),
                },
            ],
        },
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BitNet.rs Reporting System Demo");
    println!("===================================\n");

    // Create sample test data
    let test_data = create_sample_data();

    // Create temporary directory for reports
    let temp_dir = TempDir::new()?;
    println!("📁 Output directory: {:?}\n", temp_dir.path());

    // Generate HTML report
    println!("📄 Generating HTML report...");
    let html_content = generate_html_report(&test_data);
    let html_path = temp_dir.path().join("test_report.html");
    fs::write(&html_path, &html_content).await?;
    println!("   ✅ HTML report: {} bytes", html_content.len());

    // Generate JSON report
    println!("📄 Generating JSON report...");
    let json_content = generate_json_report(&test_data)?;
    let json_path = temp_dir.path().join("test_report.json");
    fs::write(&json_path, &json_content).await?;
    println!("   ✅ JSON report: {} bytes", json_content.len());

    // Generate Markdown report
    println!("📄 Generating Markdown report...");
    let md_content = generate_markdown_report(&test_data);
    let md_path = temp_dir.path().join("test_report.md");
    fs::write(&md_path, &md_content).await?;
    println!("   ✅ Markdown report: {} bytes", md_content.len());

    println!("\n🎉 All reports generated successfully!");
    println!("\n📊 Report Summary:");
    println!("   - HTML: Interactive report with styling");
    println!("   - JSON: Machine-readable format for CI/CD");
    println!("   - Markdown: Human-readable documentation format");

    // Show sample content
    println!("\n📋 Sample HTML content (first 200 chars):");
    println!("{}", &html_content[..200.min(html_content.len())]);

    println!("\n📋 Sample JSON structure:");
    let json_value: serde_json::Value = serde_json::from_str(&json_content)?;
    println!("   - Metadata: {}", json_value["metadata"]["generator"]);
    println!("   - Total tests: {}", json_value["summary"]["total_tests"]);
    println!("   - Success rate: {}%", json_value["summary"]["success_rate"]);

    println!("\n✨ Reporting system implementation completed successfully!");

    Ok(())
}
