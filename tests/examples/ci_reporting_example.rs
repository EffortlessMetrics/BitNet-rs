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
    use bitnet_tests::ci_reporting::{CIContext, CINotificationManager, NotificationConfig};
    use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
    use bitnet_tests::trend_reporting::{TestRunMetadata, TrendConfig, TrendReporter};
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;

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
        vec![
            TestSuiteResult {
                suite_name: "unit_tests".to_string(),
                total_duration: Duration::from_secs(30),
                test_results: vec![
                    TestResult {
                        test_name: "test_bitnet_quantization".to_string(),
                        status: TestStatus::Passed,
                        duration: Duration::from_secs(5),
                        metrics: TestMetrics {
                            memory_peak: Some(100 * 1024 * 1024), // 100MB
                            memory_average: Some(50 * 1024 * 1024),
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
                        stdout: Some("All tests passed".to_string()),
                        stderr: None,
                        timestamp: std::time::SystemTime::now(),
                    },
                    TestResult {
                        test_name: "test_model_loading".to_string(),
                        status: TestStatus::Failed,
                        duration: Duration::from_secs(2),
                        metrics: TestMetrics::default(),
                        error: Some("Failed to load model: file not found".to_string()),
                        stdout: None,
                        stderr: Some("Error: Model file missing".to_string()),
                        timestamp: std::time::SystemTime::now(),
                    },
                ],
                summary: TestSummary {
                    total_tests: 2,
                    passed: 1,
                    failed: 1,
                    skipped: 0,
                    success_rate: 50.0,
                },
                environment: HashMap::from([
                    ("ci_provider".to_string(), "github_actions".to_string()),
                    ("runner_os".to_string(), "ubuntu-22.04".to_string()),
                ]),
                configuration: HashMap::new(),
                metadata: HashMap::from([
                    ("commit_sha".to_string(), "abc123def456".to_string()),
                    ("pr_number".to_string(), "42".to_string()),
                ]),
            },
        ]
    }

    async fn process_ci_notifications(
        test_results: &[TestSuiteResult],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create CI context
        let context = CIContext {
            provider: "github_actions".to_string(),
            repository: "anthropics/bitnet-rs".to_string(),
            branch: "feature/ci-improvements".to_string(),
            commit_sha: "abc123def456".to_string(),
            pr_number: Some(42),
            workflow_name: "CI Tests".to_string(),
            job_name: "test-suite".to_string(),
            run_id: "12345".to_string(),
            run_attempt: 1,
        };

        // Configure notifications
        let config = NotificationConfig {
            enable_status_checks: true,
            enable_pr_comments: true,
            enable_annotations: true,
            enable_slack: false,
            slack_webhook_url: None,
            notification_threshold: TestStatus::Failed,
            include_performance_metrics: true,
            include_coverage: false,
        };

        // Create notification manager
        let manager = CINotificationManager::new(context, config);

        // Process test results
        let notification_results = manager.process_test_results(test_results).await?;

        println!("  📬 Generated {} notifications", notification_results.len());
        for result in &notification_results {
            println!("     - {}: {}", result.notification_type, result.status);
        }

        Ok(())
    }

    async fn analyze_performance_trends() -> Result<(), Box<dyn std::error::Error>> {
        // Create temporary directory for trend data
        let temp_dir = TempDir::new()?;

        // Configure trend reporter
        let config = TrendConfig {
            data_dir: temp_dir.path().to_path_buf(),
            retention_days: 30,
            enable_regression_detection: true,
            regression_threshold: 0.10, // 10% performance regression
            enable_outlier_detection: true,
            outlier_std_devs: 3.0,
            metrics_to_track: vec![
                "duration".to_string(),
                "memory_peak".to_string(),
                "throughput_ops_sec".to_string(),
            ],
        };

        let reporter = TrendReporter::new(config);

        // Create sample historical data
        for i in 0..5 {
            let metadata = TestRunMetadata {
                run_id: format!("run_{}", i),
                timestamp: std::time::SystemTime::now(),
                commit_sha: format!("commit_{}", i),
                branch: "main".to_string(),
                tags: HashMap::from([("version".to_string(), format!("v1.{}", i))]),
            };

            let results = create_sample_test_results();
            reporter.record_test_run(&metadata, &results).await?;
        }

        // Analyze trends
        let trends = reporter.analyze_trends("test_bitnet_quantization").await?;

        if let Some(trend) = trends {
            println!("  📈 Performance Trend Analysis:");
            println!("     - Average duration: {:.2}s", trend.average_duration.as_secs_f64());
            println!("     - Duration trend: {:+.2}%", trend.duration_trend * 100.0);
            println!("     - Memory trend: {:+.2}%", trend.memory_trend * 100.0);

            if !trend.regressions.is_empty() {
                println!("  ⚠️  Detected {} regressions!", trend.regressions.len());
            }
        }

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
        println!("CI Reporting example requires both 'reporting' and 'trend' features to be enabled.");
        println!("Run with: cargo run --example ci_reporting_example --features reporting,trend");
    }
    
    Ok(())
}