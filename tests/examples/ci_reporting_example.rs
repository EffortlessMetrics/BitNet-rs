//! Example demonstrating CI reporting and notifications functionality
//!
//! This example shows how to use the CI reporting system to:
//! - Process test results
//! - Generate status checks
//! - Create PR comments
//! - Track performance trends
//! - Detect regressions

#[cfg(all(feature = "reporting", feature = "trend"))]
mod ci_example {
    use bitnet_tests::BYTES_PER_MB;
    use bitnet_tests::ci_reporting::{CIContext, CINotificationManager, NotificationConfig};
    use bitnet_tests::results::{
        TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary,
    };
    use bitnet_tests::trend_reporting::{TestRunMetadata, TrendConfig, TrendReporter};
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;

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

    pub async fn run_example() -> Result<(), Box<dyn std::error::Error>> {
        // Initialize logging
        tracing_subscriber::fmt::init();

        println!("🚀 BitNet.rs CI Reporting Example");
        println!("==================================\n");

        // Create sample test results
        let test_results = create_sample_test_results();
        println!("📊 Created {} test suites with sample results", test_results.len());

        // Example 1: Basic CI Notification Processing
        println!("\n1️⃣ Processing CI Notifications");
        process_ci_notifications(&test_results).await?;

        // Example 2: Performance Trend Analysis
        println!("\n2️⃣ Analyzing Performance Trends");
        analyze_performance_trends().await?;

        // Example 3: PR Status Checks
        println!("\n3️⃣ Generating PR Status Checks");
        generate_pr_status_checks(&test_results).await?;

        println!("\n✅ CI Reporting Example Complete!");
        Ok(())
    }

    fn create_sample_test_results() -> Vec<TestSuiteResult> {
        vec![TestSuiteResult {
            suite_name: "unit_tests".to_string(),
            total_duration: Duration::from_secs(30),
            test_results: vec![
                TestResult {
                    test_name: "test_bitnet_quantization".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(5),
                    metrics: TestMetrics {
                        memory_peak: Some(100 * BYTES_PER_MB), // 100MB
                        memory_average: Some(50 * BYTES_PER_MB),
                        cpu_time: Some(Duration::from_secs(4)),
                        wall_time: Duration::from_secs(5),
                        assertions: 150,
                        operations: 10000,
                        custom_metrics: HashMap::from([
                            ("accuracy".to_string(), 0.9995),
                            ("throughput_ops_sec".to_string(), 2000.0),
                        ]),
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now(),
                    end_time: std::time::SystemTime::now() + Duration::from_secs(5),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_model_loading".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_secs(2),
                    metrics: TestMetrics::default(),
                    error: Some("Failed to load model: file not found".to_string()),
                    stack_trace: Some("Error: Model file missing\nStack trace...\n".to_string()),
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now(),
                    end_time: std::time::SystemTime::now() + Duration::from_secs(2),
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
                total_duration: Duration::from_secs(30),
                average_duration: Duration::from_secs(3),
                peak_memory: Some(100 * BYTES_PER_MB),
                total_assertions: 150,
            },
            environment: HashMap::from([
                ("ci_provider".to_string(), "github_actions".to_string()),
                ("runner_os".to_string(), "ubuntu-22.04".to_string()),
            ]),
            configuration: HashMap::new(),
            start_time: std::time::SystemTime::now(),
            end_time: std::time::SystemTime::now() + Duration::from_secs(30),
        }]
    }

    async fn process_ci_notifications(
        test_results: &[TestSuiteResult],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create CI context
        let context = CIContext {
            commit_sha: Some("abc123def456".to_string()),
            pr_number: Some(42),
            branch_name: Some("feature/ci-improvements".to_string()),
            workflow_run_id: Some("12345".to_string()),
            actor: Some("ci-bot".to_string()),
        };

        // Configure notifications
        let config = NotificationConfig {
            notify_on_failure: true,
            notify_on_success: false,
            check_performance_regression: true,
            performance_regression_threshold: 1.1,
            create_status_checks: true,
            create_pr_comments: true,
        };

        // Create notification manager
        let manager = CINotificationManager::new(config)?;

        // Process test results
        manager.process_test_results(test_results, &context).await?;

        println!("  📬 Generated notifications successfully");

        Ok(())
    }

    async fn analyze_performance_trends() -> Result<(), Box<dyn std::error::Error>> {
        // Create temporary directory for trend data
        let temp_dir = TempDir::new()?;

        // Configure trend reporter
        let config = TrendConfig {
            retention_days: 30,
            min_samples_for_baseline: 5,
            regression_threshold: 1.10, // 10% performance regression
        };

        let reporter = TrendReporter::new(temp_dir.path().to_path_buf(), config);

        // Create sample historical data
        for i in 0..5 {
            let metadata = TestRunMetadata {
                commit_sha: Some(format!("commit_{}", i)),
                branch: Some("main".to_string()),
                pr_number: None,
                environment: HashMap::from([("version".to_string(), format!("v1.{}", i))]),
                configuration: HashMap::new(),
            };

            let results = create_sample_test_results();
            reporter.record_test_results(&results, &metadata).await?;
        }

        // Generate trend report
        let report = reporter.generate_trend_report(30, Some("main")).await?;

        println!("  📈 Performance Trend Analysis:");
        println!("     - Report period: {} days", report.period_days);
        println!("     - Total entries: {}", report.total_entries);

        Ok(())
    }

    async fn generate_pr_status_checks(
        test_results: &[TestSuiteResult],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Calculate overall status
        let total_tests: usize = test_results.iter().map(|r| r.summary.total_tests).sum();
        let total_passed: usize = test_results.iter().map(|r| r.summary.passed).sum();
        let total_failed: usize = test_results.iter().map(|r| r.summary.failed).sum();

        let status = if total_failed == 0 { "success" } else { "failure" };

        println!("  ✅ PR Status Check:");
        println!("     - Status: {}", status);
        println!("     - Tests: {}/{} passed", total_passed, total_tests);

        // Generate detailed status message
        let status_message = format!(
            "Test Results: {} passed, {} failed, {} total",
            total_passed, total_failed, total_tests
        );

        println!("     - Message: {}", status_message);

        // Generate annotations for failed tests
        for suite in test_results {
            for test in &suite.test_results {
                if test.status == TestStatus::Failed {
                    println!(
                        "  ❌ Annotation: {} - {}",
                        test.test_name,
                        test.error.as_ref().unwrap_or(&"Unknown error".to_string())
                    );
                }
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(feature = "reporting", feature = "trend"))]
    {
        ci_example::run_example().await?;
    }

    #[cfg(not(all(feature = "reporting", feature = "trend")))]
    {
        println!(
            "CI Reporting example requires both 'reporting' and 'trend' features to be enabled."
        );
        println!("Run with: cargo run --example ci_reporting_example --features reporting,trend");
    }

    Ok(())
}
