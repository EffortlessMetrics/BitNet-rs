//! Trend analysis generation tool
//!
//! This tool generates comprehensive trend analysis reports
//! from historical test data.

use anyhow::Result;
use bitnet_tests::trend_reporting::{TrendConfig, TrendReporter};
use clap::Parser;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::info;

#[derive(Parser)]
#[command(name = "generate_trend_analysis")]
#[command(about = "Generate trend analysis reports from historical test data")]
struct Args {
    /// Directory containing trend data
    #[arg(long)]
    trend_data: PathBuf,

    /// Output directory for trend reports
    #[arg(long)]
    output_dir: PathBuf,

    /// Number of days to analyze
    #[arg(long, default_value = "30")]
    days_back: u32,

    /// Branch to filter by (optional)
    #[arg(long)]
    branch: Option<String>,

    /// Generate HTML report
    #[arg(long, default_value = "true")]
    html: bool,

    /// Generate JSON report
    #[arg(long, default_value = "true")]
    json: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Starting trend analysis generation");
    info!("Trend data: {:?}", args.trend_data);
    info!("Output directory: {:?}", args.output_dir);
    info!("Days back: {}", args.days_back);

    // Ensure output directory exists
    fs::create_dir_all(&args.output_dir).await?;

    // Create trend reporter
    let trend_config = TrendConfig::default();
    let trend_reporter = TrendReporter::new(args.trend_data, trend_config);

    // Generate trend report
    let report =
        trend_reporter.generate_trend_report(args.days_back, args.branch.as_deref()).await?;

    info!("Generated trend report with {} entries", report.total_entries);

    // Generate JSON report
    if args.json {
        let json_content = serde_json::to_string_pretty(&report)?;
        let json_path = args.output_dir.join("trend-analysis.json");
        fs::write(&json_path, json_content).await?;
        info!("Generated JSON report: {:?}", json_path);
    }

    // Generate HTML report
    if args.html {
        let html_path = args.output_dir.join("trend-analysis.html");
        trend_reporter.generate_html_report(&report, &html_path).await?;
        info!("Generated HTML report: {:?}", html_path);
    }

    // Generate performance trend data for key tests
    let key_tests = vec![
        "test_inference_performance".to_string(),
        "test_model_loading_time".to_string(),
        "test_tokenization_speed".to_string(),
        "test_memory_usage".to_string(),
    ];

    let performance_trends =
        trend_reporter.get_performance_trends(&key_tests, args.days_back).await?;

    // Save performance trends as JSON
    let trends_json = serde_json::to_string_pretty(&performance_trends)?;
    let trends_path = args.output_dir.join("performance-trends.json");
    fs::write(&trends_path, trends_json).await?;
    info!("Generated performance trends: {:?}", trends_path);

    // Generate summary statistics
    generate_summary_report(&report, &args.output_dir).await?;

    info!("Trend analysis generation completed successfully");
    Ok(())
}

async fn generate_summary_report(
    report: &bitnet_tests::trend_reporting::TrendReport,
    output_dir: &Path,
) -> Result<()> {
    let mut summary = String::new();

    summary.push_str("# BitNet-rs Test Trend Summary\n\n");
    summary.push_str(&format!("**Analysis Period:** {} days\n", report.period_days));
    summary.push_str(&format!("**Total Entries:** {}\n", report.total_entries));
    summary.push_str(&format!(
        "**Generated:** {}\n\n",
        report.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
    ));

    if let Some(branch) = &report.branch_filter {
        summary.push_str(&format!("**Branch:** {}\n\n", branch));
    }

    // Overall stability
    summary.push_str("## Overall Stability\n\n");
    summary.push_str(&format!(
        "- **Stability Score:** {:.1}%\n",
        report.analysis.overall_stability * 100.0
    ));
    summary
        .push_str(&format!("- **Performance Trend:** {:?}\n\n", report.analysis.performance_trend));

    // Suite summary
    summary.push_str("## Test Suite Summary\n\n");
    summary.push_str("| Suite | Success Rate | Avg Duration | Stability |\n");
    summary.push_str("|-------|--------------|--------------|----------|\n");

    for (suite_name, trend) in &report.analysis.suite_trends {
        summary.push_str(&format!(
            "| {} | {:.1}% | {:.2}s | {:?} |\n",
            suite_name,
            trend.average_success_rate,
            trend.average_duration.as_secs_f64(),
            trend.stability
        ));
    }

    // Top issues
    summary.push_str("\n## Key Insights\n\n");

    let unstable_suites: Vec<_> = report
        .analysis
        .suite_trends
        .iter()
        .filter(|(_, trend)| {
            matches!(
                trend.stability,
                bitnet_tests::trend_reporting::TestStability::Unstable
                    | bitnet_tests::trend_reporting::TestStability::Flaky
            )
        })
        .collect();

    if !unstable_suites.is_empty() {
        summary.push_str("### Unstable Test Suites\n\n");
        for (suite_name, trend) in &unstable_suites {
            summary.push_str(&format!(
                "- **{}**: {:.1}% success rate ({:?})\n",
                suite_name, trend.average_success_rate, trend.stability
            ));
        }
        summary.push('\n');
    }

    let degrading_tests: Vec<_> = report
        .analysis
        .test_trends
        .iter()
        .filter(|(_, trend)| {
            matches!(
                trend.performance_trend,
                bitnet_tests::trend_reporting::PerformanceTrend::Degrading
            )
        })
        .collect();

    if !degrading_tests.is_empty() {
        summary.push_str("### Performance Degradations\n\n");
        for (test_name, trend) in &degrading_tests {
            summary.push_str(&format!(
                "- **{}**: Average {:.2}s (degrading trend)\n",
                test_name,
                trend.average_duration.as_secs_f64()
            ));
        }
        summary.push('\n');
    }

    // Recommendations
    summary.push_str("## Recommendations\n\n");

    if report.analysis.overall_stability < 0.95 {
        summary.push_str("- 🔴 **Overall stability is below 95%** - investigate failing tests\n");
    }

    if matches!(
        report.analysis.performance_trend,
        bitnet_tests::trend_reporting::PerformanceTrend::Degrading
    ) {
        summary.push_str(
            "- 🔴 **Performance is degrading** - review recent changes for performance impact\n",
        );
    }

    if !unstable_suites.is_empty() {
        summary.push_str("- 🟡 **Some test suites are unstable** - investigate flaky tests\n");
    }

    if !degrading_tests.is_empty() {
        summary
            .push_str("- 🟡 **Some tests show performance degradation** - optimize slow tests\n");
    }

    if report.analysis.overall_stability >= 0.95
        && matches!(
            report.analysis.performance_trend,
            bitnet_tests::trend_reporting::PerformanceTrend::Stable
                | bitnet_tests::trend_reporting::PerformanceTrend::Improving
        )
    {
        summary.push_str(
            "- ✅ **Test suite is stable and performing well** - keep up the good work!\n",
        );
    }

    let summary_path = output_dir.join("trend-summary.md");
    fs::write(&summary_path, summary).await?;
    info!("Generated trend summary: {:?}", summary_path);

    Ok(())
}
