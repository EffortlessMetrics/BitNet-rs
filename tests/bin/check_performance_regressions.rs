//! Performance regression detection tool
//!
//! This tool analyzes current test results against historical trends
//! to detect performance regressions.

use anyhow::{Context, Result};
use bitnet_tests::results::TestSuiteResult;
use bitnet_tests::trend_reporting::{TrendConfig, TrendReporter};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "check_performance_regressions")]
#[command(about = "Check for performance regressions in test results")]
struct Args {
    /// Path to current test results JSON file
    #[arg(long)]
    current_results: PathBuf,

    /// Directory containing trend data
    #[arg(long)]
    trend_data: PathBuf,

    /// Output file for regression report
    #[arg(long)]
    output: PathBuf,

    /// Regression threshold (e.g., 1.2 for 20% slower)
    #[arg(long, default_value = "1.2")]
    threshold: f64,

    /// Minimum number of historical samples required
    #[arg(long, default_value = "5")]
    min_samples: usize,

    /// Number of days to look back for baseline
    #[arg(long, default_value = "30")]
    lookback_days: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegressionReport {
    test_name: String,
    suite_name: String,
    current_duration_ms: f64,
    baseline_duration_ms: f64,
    regression_percent: f64,
    confidence: f64,
    sample_size: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Starting performance regression check");
    info!("Current results: {:?}", args.current_results);
    info!("Trend data: {:?}", args.trend_data);
    info!("Threshold: {}x", args.threshold);

    // Load current test results
    let current_results = load_current_results(&args.current_results).await?;
    info!("Loaded {} test suites from current results", current_results.len());

    // Create trend reporter
    let trend_config = TrendConfig {
        retention_days: 90,
        min_samples_for_baseline: args.min_samples,
        regression_threshold: args.threshold,
    };
    let trend_reporter = TrendReporter::new(args.trend_data, trend_config);

    // Detect regressions
    let regressions =
        trend_reporter.detect_regressions(&current_results, args.lookback_days).await?;

    info!("Detected {} potential regressions", regressions.len());

    // Convert to report format
    let regression_reports: Vec<RegressionReport> = regressions
        .into_iter()
        .map(|reg| RegressionReport {
            test_name: reg.test_name,
            suite_name: reg.suite_name,
            current_duration_ms: reg.current_duration.as_secs_f64() * 1000.0,
            baseline_duration_ms: reg.baseline_duration.as_secs_f64() * 1000.0,
            regression_percent: reg.regression_percent,
            confidence: reg.confidence,
            sample_size: reg.sample_size,
        })
        .collect();

    // Save regression report
    let json_content = serde_json::to_string_pretty(&regression_reports)?;
    fs::write(&args.output, json_content).await?;

    if !regression_reports.is_empty() {
        warn!("Performance regressions detected:");
        for report in &regression_reports {
            warn!(
                "  {}: {:.1}% slower ({:.1}ms -> {:.1}ms, confidence: {:.2})",
                report.test_name,
                report.regression_percent,
                report.baseline_duration_ms,
                report.current_duration_ms,
                report.confidence
            );
        }

        // Exit with error code to indicate regressions found
        std::process::exit(1);
    } else {
        info!("No performance regressions detected");
    }

    Ok(())
}

async fn load_current_results(path: &Path) -> Result<Vec<TestSuiteResult>> {
    let content = fs::read_to_string(path).await.context("Failed to read current results file")?;

    let results: Vec<TestSuiteResult> =
        serde_json::from_str(&content).context("Failed to parse current results JSON")?;

    Ok(results)
}
