//! Example demonstrating CI reporting and notifications functionality
//!
//! This example shows how to use the CI reporting system to:
//! - Process test results
//! - Generate status checks
//! - Create PR comments
//! - Track performance trends
//! - Detect regressions

use bitnet_tests::ci_reporting::{CIContext, CINotificationManager, NotificationConfig};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
use bitnet_tests::trend_reporting::{TestRunMetadata, TrendConfig, TrendReporter};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 BitNet.rs CI Reporting Example");
    println!("==================================\n");

    // Create sample test results
    let test_results = create_sample_test_results();
    println!("📊 Created {} test suites with sample results", test_results.len());

    // Example 1: Basic CI Notification Processing
    println!("\n1️⃣ Processing CI Notifications");
    println!("------------------------------");

    let notification_config = NotificationConfig {
        notify_on_failure: true,
        notify_on_success: false,
        check_performance_regression: true,
        performance_regression_threshold: 1.2,
        create_status_checks: true,
        create_pr_comments: true,
    };

    // Create CI context (simulating GitHub Actions environment)
    let ci_context = CIContext {
        commit_sha: Some("abc123def456".to_string()),
        pr_number: Some(42),
        branch_name: Some("feature/ci-reporting".to_string()),
        workflow_run_id: Some("12345".to_string()),
        actor: Some("developer".to_string()),
    };

    // Note: In a real environment, this would require GITHUB_TOKEN
    match CINotificationManager::new(notification_config) {
        Ok(notification_manager) => {
            println!("✅ Created CI notification manager");

            // Process test results (this would normally send to GitHub)
            match notification_manager.process_test_results(&test_results, &ci_context).await {
                Ok(()) => println!("✅ Processed test results for CI notifications"),
                Err(e) => {
                    println!("⚠️  CI notification processing failed (expected in test env): {}", e)
                }
            }
        }
        Err(e) => {
            println!("⚠️  Could not create CI notification manager (expected in test env): {}", e);
            println!("   This is normal when GITHUB_TOKEN is not available");
        }
    }

    // Example 2: Trend Reporting
    println!("\n2️⃣ Trend Reporting");
    println!("------------------");

    let temp_dir = TempDir::new()?;
    let trend_config = TrendConfig::default();
    let trend_reporter = TrendReporter::new(temp_dir.path().to_path_buf(), trend_config);

    // Record test results for trend analysis
    let metadata = TestRunMetadata {
        commit_sha: Some("abc123def456".to_string()),
        branch: Some("feature/ci-reporting".to_string()),
        pr_number: Some(42),
        environment: create_sample_environment(),
        configuration: create_sample_configuration(),
    };

    match trend_reporter.record_test_results(&test_results, &metadata).await {
        Ok(()) => println!("✅ Recorded test results for trend analysis"),
        Err(e) => println!("❌ Failed to record trend data: {}", e),
    }

    // Generate trend report
    match trend_reporter.generate_trend_report(30, Some("main")).await {
        Ok(report) => {
            println!("✅ Generated trend report:");
            println!("   - Period: {} days", report.period_days);
            println!("   - Total entries: {}", report.total_entries);
            println!("   - Overall stability: {:.1}%", report.analysis.overall_stability * 100.0);
            println!("   - Performance trend: {:?}", report.analysis.performance_trend);
        }
        Err(e) => println!("❌ Failed to generate trend report: {}", e),
    }

    // Example 3: Performance Regression Detection
    println!("\n3️⃣ Performance Regression Detection");
    println!("-----------------------------------");

    match trend_reporter.detect_regressions(&test_results, 30).await {
        Ok(regressions) => {
            if regressions.is_empty() {
                println!("✅ No performance regressions detected");
            } else {
                println!("⚠️  Detected {} performance regressions:", regressions.len());
                for regression in &regressions {
                    println!(
                        "   - {}: {:.1}% slower",
                        regression.test_name, regression.regression_percent
                    );
                }
            }
        }
        Err(e) => println!("❌ Failed to detect regressions: {}", e),
    }

    // Example 4: Performance Trends for Specific Tests
    println!("\n4️⃣ Performance Trends");
    println!("---------------------");

    let key_tests = vec![
        "test_inference_performance".to_string(),
        "test_model_loading".to_string(),
        "test_tokenization".to_string(),
    ];

    match trend_reporter.get_performance_trends(&key_tests, 30).await {
        Ok(trends) => {
            println!("✅ Retrieved performance trends for {} tests", trends.len());
            for (test_name, data_points) in &trends {
                println!("   - {}: {} data points", test_name, data_points.len());
            }
        }
        Err(e) => println!("❌ Failed to get performance trends: {}", e),
    }

    println!("\n🎉 CI Reporting Example Completed!");
    println!("\nIn a real CI environment, this would:");
    println!("• Create GitHub status checks for each test suite");
    println!("• Post detailed comments on pull requests");
    println!("• Send notifications on test failures");
    println!("• Track performance trends over time");
    println!("• Alert on performance regressions");
    println!("• Generate HTML trend reports");

    Ok(())
}

fn create_sample_test_results() -> Vec<TestSuiteResult> {
    vec![
        create_test_suite("unit_tests", 50, 48, 2),
        create_test_suite("integration_tests", 25, 24, 1),
        create_test_suite("performance_tests", 10, 10, 0),
    ]
}

fn create_test_suite(name: &str, total: usize, passed: usize, failed: usize) -> TestSuiteResult {
    let mut test_results = Vec::new();

    // Create passed tests
    for i in 0..passed {
        test_results.push(TestResult {
            test_name: format!("{}::test_{}", name, i),
            status: TestStatus::Passed,
            duration: Duration::from_millis(100 + (i * 10) as u64),
            metrics: TestMetrics {
                memory_peak: Some(1024 * (i + 1) as u64),
                memory_average: Some(512 * (i + 1) as u64),
                cpu_time: Some(Duration::from_millis(50 + (i * 5) as u64)),
                wall_time: Duration::from_millis(100 + (i * 10) as u64),
                custom_metrics: HashMap::new(),
                assertions: 5,
                operations: 10,
            },
            error: None,
            stack_trace: None,
            artifacts: Vec::new(),
            start_time: std::time::SystemTime::now() - Duration::from_secs(60),
            end_time: std::time::SystemTime::now() - Duration::from_secs(50),
            metadata: HashMap::new(),
        });
    }

    // Create failed tests
    for i in 0..failed {
        test_results.push(TestResult {
            test_name: format!("{}::test_failed_{}", name, i),
            status: TestStatus::Failed,
            duration: Duration::from_millis(200 + (i * 20) as u64),
            metrics: TestMetrics {
                memory_peak: Some(2048 * (i + 1) as u64),
                memory_average: Some(1024 * (i + 1) as u64),
                cpu_time: Some(Duration::from_millis(100 + (i * 10) as u64)),
                wall_time: Duration::from_millis(200 + (i * 20) as u64),
                custom_metrics: HashMap::new(),
                assertions: 3,
                operations: 5,
            },
            error: Some(format!("Test failure reason {}", i)),
            stack_trace: Some(format!("Stack trace for failed test {}", i)),
            artifacts: Vec::new(),
            start_time: std::time::SystemTime::now() - Duration::from_secs(60),
            end_time: std::time::SystemTime::now() - Duration::from_secs(50),
            metadata: HashMap::new(),
        });
    }

    let total_duration: Duration = test_results.iter().map(|t| t.duration).sum();
    let success_rate = if total > 0 { (passed as f64 / total as f64) * 100.0 } else { 100.0 };

    TestSuiteResult {
        suite_name: name.to_string(),
        total_duration,
        test_results,
        summary: TestSummary {
            total_tests: total,
            passed,
            failed,
            skipped: 0,
            timeout: 0,
            success_rate,
            total_duration,
            average_duration: if total > 0 {
                Duration::from_nanos(
                    (total_duration.as_nanos() / total as u128).try_into().unwrap_or(u64::MAX),
                )
            } else {
                Duration::from_secs(0)
            },
            peak_memory: Some(2048),
            total_assertions: passed * 5 + failed * 3,
        },
        environment: create_sample_environment(),
        configuration: create_sample_configuration(),
        start_time: std::time::SystemTime::now() - Duration::from_secs(60),
        end_time: std::time::SystemTime::now() - Duration::from_secs(50),
    }
}

fn create_sample_environment() -> HashMap<String, String> {
    let mut env = HashMap::new();
    env.insert("os".to_string(), "ubuntu-latest".to_string());
    env.insert("arch".to_string(), "x86_64".to_string());
    env.insert("rust_version".to_string(), "1.75.0".to_string());
    env
}

fn create_sample_configuration() -> HashMap<String, String> {
    let mut config = HashMap::new();
    config.insert("features".to_string(), "cpu,gpu".to_string());
    config.insert("profile".to_string(), "release".to_string());
    config
}
