//! CI report generation tool
//!
//! This tool collects test results from various sources and generates
//! comprehensive CI reports including status checks and PR comments.

use anyhow::{Context, Result};
use bitnet_tests::ci_reporting::{CIContext, CINotificationManager, NotificationConfig};
use bitnet_tests::results::{TestResult, TestStatus, TestSuiteResult, TestSummary};
use bitnet_tests::trend_reporting::{TestRunMetadata, TrendConfig, TrendReporter};
use clap::Parser;
use serde_json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::fs;
use tracing::{error, info, warn};

#[derive(Parser)]
#[command(name = "generate_ci_report")]
#[command(about = "Generate CI reports from test results")]
struct Args {
    /// Directory containing test result files
    #[arg(long)]
    results_dir: PathBuf,

    /// Output directory for CI reports
    #[arg(long)]
    output_dir: PathBuf,

    /// Directory for trend data storage
    #[arg(long)]
    trend_data_dir: PathBuf,

    /// Commit SHA
    #[arg(long)]
    commit_sha: Option<String>,

    /// Pull request number
    #[arg(long)]
    pr_number: Option<String>,

    /// Branch name
    #[arg(long)]
    branch: Option<String>,

    /// Workflow run ID
    #[arg(long)]
    workflow_run_id: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Starting CI report generation");
    info!("Results directory: {:?}", args.results_dir);
    info!("Output directory: {:?}", args.output_dir);

    // Ensure output directory exists
    fs::create_dir_all(&args.output_dir).await?;

    // Load test results
    let test_results = load_test_results(&args.results_dir).await?;
    info!("Loaded {} test suites", test_results.len());

    // Create CI context
    let ci_context = CIContext {
        commit_sha: args.commit_sha.clone(),
        pr_number: args.pr_number.as_ref().and_then(|s| s.parse().ok()),
        branch_name: args.branch.clone(),
        workflow_run_id: args.workflow_run_id.clone(),
        actor: std::env::var("GITHUB_ACTOR").ok(),
    };

    // Generate CI reports
    let notification_config = NotificationConfig::default();
    let notification_manager = CINotificationManager::new(notification_config)?;

    // Process test results for notifications
    if let Err(e) = notification_manager.process_test_results(&test_results, &ci_context).await {
        error!("Failed to process test results for notifications: {}", e);
    }

    // Generate status checks file
    generate_status_checks_file(&test_results, &args.output_dir).await?;

    // Generate PR comment file
    if ci_context.pr_number.is_some() {
        generate_pr_comment_file(&test_results, &args.output_dir).await?;
    }

    // Save test results as JSON
    save_test_results_json(&test_results, &args.output_dir).await?;

    // Record trend data
    let trend_config = TrendConfig::default();
    let trend_reporter = TrendReporter::new(args.trend_data_dir, trend_config);

    let metadata = TestRunMetadata {
        commit_sha: args.commit_sha,
        branch: args.branch,
        pr_number: ci_context.pr_number,
        environment: collect_environment_info(),
        configuration: collect_configuration_info(),
    };

    if let Err(e) = trend_reporter.record_test_results(&test_results, &metadata).await {
        warn!("Failed to record trend data: {}", e);
    }

    info!("CI report generation completed successfully");
    Ok(())
}

async fn load_test_results(results_dir: &Path) -> Result<Vec<TestSuiteResult>> {
    let mut test_results = Vec::new();

    if !results_dir.exists() {
        warn!("Results directory does not exist: {:?}", results_dir);
        return Ok(test_results);
    }

    let mut entries = fs::read_dir(results_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            match load_json_test_results(&path).await {
                Ok(mut results) => {
                    test_results.append(&mut results);
                }
                Err(e) => {
                    warn!("Failed to load test results from {:?}: {}", path, e);
                }
            }
        } else if path.extension().and_then(|s| s.to_str()) == Some("xml") {
            match load_junit_test_results(&path).await {
                Ok(results) => {
                    test_results.push(results);
                }
                Err(e) => {
                    warn!("Failed to load JUnit results from {:?}: {}", path, e);
                }
            }
        }
    }

    Ok(test_results)
}

async fn load_json_test_results(path: &Path) -> Result<Vec<TestSuiteResult>> {
    let content = fs::read_to_string(path).await?;
    let results: Vec<TestSuiteResult> =
        serde_json::from_str(&content).context("Failed to parse JSON test results")?;
    Ok(results)
}

async fn load_junit_test_results(path: &Path) -> Result<TestSuiteResult> {
    // Simple JUnit XML parsing - in a real implementation, you'd use a proper XML parser
    let content = fs::read_to_string(path).await?;

    // Extract basic information from JUnit XML
    let suite_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();

    // Parse test count from XML (simplified)
    let total_tests = content.matches("<testcase").count();
    let failed_tests = content.matches("failure>").count() + content.matches("error>").count();
    let passed_tests = total_tests - failed_tests;

    let summary = TestSummary {
        total_tests,
        passed: passed_tests,
        failed: failed_tests,
        skipped: 0,
        timeout: 0,
        success_rate: if total_tests > 0 {
            (passed_tests as f64 / total_tests as f64) * 100.0
        } else {
            100.0
        },
        total_duration: Duration::from_secs(0), // Would need to parse from XML
        average_duration: Duration::from_secs(0),
        peak_memory: None,
        total_assertions: 0,
    };

    Ok(TestSuiteResult {
        suite_name,
        total_duration: Duration::from_secs(0),
        test_results: Vec::new(), // Would need to parse individual tests
        summary,
        environment: HashMap::new(),
        configuration: HashMap::new(),
        start_time: std::time::SystemTime::now(),
        end_time: std::time::SystemTime::now(),
    })
}

async fn generate_status_checks_file(
    test_results: &[TestSuiteResult],
    output_dir: &PathBuf,
) -> Result<()> {
    let mut status_checks = Vec::new();

    // Overall status
    let total_tests: usize = test_results.iter().map(|s| s.summary.total_tests).sum();
    let failed_tests: usize = test_results.iter().map(|s| s.summary.failed).sum();
    let passed_tests = total_tests - failed_tests;

    let overall_state = if failed_tests > 0 { "failure" } else { "success" };
    let description = format!("{}/{} tests passed", passed_tests, total_tests);

    status_checks.push(serde_json::json!({
        "context": "bitnet-rs/tests",
        "state": overall_state,
        "description": description,
        "target_url": null
    }));

    // Individual suite status checks
    for suite in test_results {
        let state = if suite.summary.failed > 0 { "failure" } else { "success" };
        let description =
            format!("{}/{} tests passed", suite.summary.passed, suite.summary.total_tests);
        let context = format!("bitnet-rs/tests/{}", suite.suite_name);

        status_checks.push(serde_json::json!({
            "context": context,
            "state": state,
            "description": description,
            "target_url": null
        }));
    }

    let status_file = output_dir.join("status-checks.json");
    let json_content = serde_json::to_string_pretty(&status_checks)?;
    fs::write(status_file, json_content).await?;

    Ok(())
}

async fn generate_pr_comment_file(
    test_results: &[TestSuiteResult],
    output_dir: &PathBuf,
) -> Result<()> {
    let mut comment = String::new();

    comment.push_str("<!-- BitNet.rs Test Results -->\n\n");

    // Calculate overall statistics
    let total_tests: usize = test_results.iter().map(|s| s.summary.total_tests).sum();
    let failed_tests: usize = test_results.iter().map(|s| s.summary.failed).sum();
    let passed_tests = total_tests - failed_tests;
    let success_rate =
        if total_tests > 0 { (passed_tests as f64 / total_tests as f64) * 100.0 } else { 100.0 };

    // Header
    if failed_tests > 0 {
        comment.push_str("## ❌ Test Results - Some tests failed\n\n");
    } else {
        comment.push_str("## ✅ Test Results - All tests passed\n\n");
    }

    // Summary table
    comment.push_str("| Metric | Value |\n");
    comment.push_str("|--------|-------|\n");
    comment.push_str(&format!("| Total Tests | {} |\n", total_tests));
    comment.push_str(&format!("| Passed | {} |\n", passed_tests));
    comment.push_str(&format!("| Failed | {} |\n", failed_tests));
    comment.push_str(&format!("| Success Rate | {:.1}% |\n", success_rate));

    // Suite breakdown
    if test_results.len() > 1 {
        comment.push_str("\n### Test Suite Breakdown\n\n");
        comment.push_str("| Suite | Status | Tests | Duration |\n");
        comment.push_str("|-------|--------|-------|----------|\n");

        for suite in test_results {
            let status = if suite.summary.failed > 0 { "❌" } else { "✅" };
            comment.push_str(&format!(
                "| {} | {} | {}/{} | {:.2}s |\n",
                suite.suite_name,
                status,
                suite.summary.passed,
                suite.summary.total_tests,
                suite.total_duration.as_secs_f64()
            ));
        }
    }

    // Failed tests details
    if failed_tests > 0 {
        comment.push_str("\n### Failed Tests\n\n");

        for suite in test_results {
            let failed_tests: Vec<&TestResult> = suite
                .test_results
                .iter()
                .filter(|test| matches!(test.status, TestStatus::Failed))
                .collect();

            if !failed_tests.is_empty() {
                comment.push_str(&format!("**{}**\n", suite.suite_name));
                for test in failed_tests {
                    comment.push_str(&format!(
                        "- `{}`: {}\n",
                        test.test_name,
                        test.error.as_deref().unwrap_or("Unknown error")
                    ));
                }
                comment.push('\n');
            }
        }
    }

    comment.push_str("\n---\n");
    comment.push_str(&format!(
        "*Generated at {}*",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));

    let comment_file = output_dir.join("pr-comment.md");
    fs::write(comment_file, comment).await?;

    Ok(())
}

async fn save_test_results_json(
    test_results: &[TestSuiteResult],
    output_dir: &PathBuf,
) -> Result<()> {
    let json_content = serde_json::to_string_pretty(test_results)?;
    let json_file = output_dir.join("test-results.json");
    fs::write(json_file, json_content).await?;
    Ok(())
}

fn collect_environment_info() -> HashMap<String, String> {
    let mut env = HashMap::new();

    if let Ok(os) = std::env::var("RUNNER_OS") {
        env.insert("os".to_string(), os);
    }

    if let Ok(arch) = std::env::var("RUNNER_ARCH") {
        env.insert("arch".to_string(), arch);
    }

    if let Ok(rust_version) = std::env::var("RUSTC_VERSION") {
        env.insert("rust_version".to_string(), rust_version);
    }

    env
}

fn collect_configuration_info() -> HashMap<String, String> {
    let mut config = HashMap::new();

    if let Ok(features) = std::env::var("CARGO_BUILD_FEATURES") {
        config.insert("features".to_string(), features);
    }

    if let Ok(profile) = std::env::var("CARGO_BUILD_PROFILE") {
        config.insert("profile".to_string(), profile);
    }

    config
}
