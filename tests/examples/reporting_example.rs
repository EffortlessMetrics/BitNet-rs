//! Example demonstrating the BitNet.rs reporting system
//!
//! This example shows how to use the comprehensive reporting system
//! to generate HTML, JSON, JUnit XML, and Markdown reports.

#[cfg(feature = "reporting")]
mod reporting_example {
    use bitnet_tests::reporting::{
        ReportConfig, ReportFormat, ReportingManager, TestReporter,
        formats::{HtmlReporter, JsonReporter, JunitReporter, MarkdownReporter},
    };
    use bitnet_tests::results::{
        TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary,
    };
    use bitnet_tests::units::{BYTES_PER_KB, BYTES_PER_MB};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Duration;
    use tokio::fs;

    /// Helper function to create test results with less boilerplate
    #[allow(dead_code)]
    fn example_result(
        name: &str,
        status: TestStatus,
        dur_ms: u64,
        error: Option<&str>,
    ) -> TestResult {
        use std::time::SystemTime;
        TestResult {
            test_name: name.into(),
            status,
            duration: Duration::from_millis(dur_ms),
            metrics: TestMetrics::default(),
            error: error.map(|e| e.into()),
            stack_trace: error.map(|e| format!("{}\nStack trace...\n", e)),
            artifacts: Vec::new(),
            start_time: SystemTime::now(),
            end_time: SystemTime::now() + Duration::from_millis(dur_ms.max(1)),
            metadata: Default::default(),
        }
    }

    /// Create example test data
    pub fn create_example_test_data() -> Vec<TestSuiteResult> {
        vec![TestSuiteResult {
            suite_name: "example_test_suite".to_string(),
            total_duration: Duration::from_secs(8),
            test_results: vec![
                TestResult {
                    test_name: "test_successful_operation".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(3),
                    metrics: TestMetrics {
                        memory_peak: Some(BYTES_PER_MB),          // 1MB
                        memory_average: Some(512 * BYTES_PER_KB), // 512KB
                        cpu_time: Some(Duration::from_secs(2)),
                        wall_time: Duration::from_secs(3),
                        assertions: 42,
                        operations: 1000,
                        custom_metrics: HashMap::from([
                            ("throughput".to_string(), 333.33),
                            ("latency_p99".to_string(), 45.5),
                        ]),
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now(),
                    end_time: std::time::SystemTime::now() + Duration::from_secs(3),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_failed_assertion".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_millis(500),
                    metrics: TestMetrics {
                        memory_peak: Some(2 * BYTES_PER_MB),
                        memory_average: Some(BYTES_PER_MB),
                        cpu_time: Some(Duration::from_millis(400)),
                        wall_time: Duration::from_millis(500),
                        assertions: 10,
                        operations: 50,
                        custom_metrics: HashMap::new(),
                    },
                    error: Some("Assertion failed: expected 42, got 0".to_string()),
                    stack_trace: Some(
                        "ERROR: Assertion failed at line 42\nStack trace...\n".to_string(),
                    ),
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now(),
                    end_time: std::time::SystemTime::now() + Duration::from_millis(500),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_skipped_conditionally".to_string(),
                    status: TestStatus::Skipped,
                    duration: Duration::ZERO,
                    metrics: TestMetrics::default(),
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now(),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
            ],
            summary: TestSummary {
                total_tests: 3,
                passed: 1,
                failed: 1,
                skipped: 1,
                timeout: 0,
                success_rate: 33.33,
                total_duration: Duration::from_secs(8),
                average_duration: Duration::from_secs(2),
                peak_memory: Some(2 * BYTES_PER_MB),
                total_assertions: 52,
            },
            environment: HashMap::from([
                ("rust_version".to_string(), "1.75.0".to_string()),
                ("os".to_string(), "Linux".to_string()),
                ("arch".to_string(), "x86_64".to_string()),
            ]),
            configuration: HashMap::from([
                ("parallel_tests".to_string(), "8".to_string()),
                ("test_timeout".to_string(), "60s".to_string()),
            ]),
            start_time: std::time::SystemTime::now(),
            end_time: std::time::SystemTime::now() + Duration::from_secs(8),
        }]
    }

    pub async fn run_example() -> Result<(), Box<dyn std::error::Error>> {
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
        let json_result = json_reporter
            .generate_report(&test_data, &output_dir.join("example_report.json"))
            .await?;
        println!("   ✅ JSON: {} bytes", json_result.size_bytes);

        // HTML Report (interactive)
        let html_reporter = HtmlReporter::new(true);
        let html_result = html_reporter
            .generate_report(&test_data, &output_dir.join("example_report.html"))
            .await?;
        println!("   ✅ HTML: {} bytes", html_result.size_bytes);

        // JUnit XML Report
        let junit_reporter = JunitReporter::new();
        let junit_result = junit_reporter
            .generate_report(&test_data, &output_dir.join("example_report.xml"))
            .await?;
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
            generate_coverage: false,
            include_artifacts: true,
            interactive_html: true,
        };

        let manager = ReportingManager::new(config);
        let results = manager.generate_all_reports(&test_data).await?;

        println!("   Generated {} reports:", results.len());
        for result in &results {
            println!("   ✅ {}: {} bytes", result.format, result.size_bytes);
        }

        println!();

        // Example 3: Generate with coverage data
        println!("3. Generating report with coverage data...");

        let coverage_config = ReportConfig {
            output_dir: output_dir.join("coverage_output"),
            formats: vec![ReportFormat::Html],
            generate_coverage: true,
            include_artifacts: true,
            interactive_html: true,
        };

        let coverage_manager = ReportingManager::new(coverage_config);
        let coverage_results = coverage_manager.generate_all_reports(&test_data).await?;

        for result in &coverage_results {
            println!("   ✅ {} with coverage: {} bytes", result.format, result.size_bytes);
        }

        println!();
        println!("✅ All reports generated successfully!");
        println!("📁 Reports saved to: {}", output_dir.display());

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "reporting")]
    {
        reporting_example::run_example().await?;
    }

    #[cfg(not(feature = "reporting"))]
    {
        println!("Reporting example requires the 'reporting' feature to be enabled.");
        println!("Run with: cargo run --example reporting_example --features reporting");
    }

    Ok(())
}
