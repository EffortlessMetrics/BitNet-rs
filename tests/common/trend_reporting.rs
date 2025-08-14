//! Test trend reporting and analysis
//!
//! This module provides functionality to track test results over time,
//! analyze trends, and generate historical reports.

use crate::results::TestSuiteResult;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::fs;
use tracing::{debug, info, warn};

/// Test trend analyzer and reporter
pub struct TrendReporter {
    storage_path: PathBuf,
    config: TrendConfig,
}

impl TrendReporter {
    pub fn new(storage_path: PathBuf, config: TrendConfig) -> Self {
        Self { storage_path, config }
    }

    /// Record test results for trend analysis
    pub async fn record_test_results(
        &self,
        results: &[TestSuiteResult],
        metadata: &TestRunMetadata,
    ) -> Result<()> {
        info!("Recording test results for trend analysis");

        // Ensure storage directory exists
        fs::create_dir_all(&self.storage_path).await?;

        // Create trend entry
        let trend_entry = TrendEntry {
            timestamp: Utc::now(),
            commit_sha: metadata.commit_sha.clone(),
            branch: metadata.branch.clone(),
            pr_number: metadata.pr_number,
            results: results.to_vec(),
            environment: metadata.environment.clone(),
            configuration: metadata.configuration.clone(),
        };

        // Save to file
        let filename = format!("trend_{}.json", trend_entry.timestamp.format("%Y%m%d_%H%M%S"));
        let file_path = self.storage_path.join(filename);

        let json_data = serde_json::to_string_pretty(&trend_entry)?;
        fs::write(&file_path, json_data).await?;

        // Clean up old entries if needed
        self.cleanup_old_entries().await?;

        debug!("Recorded trend entry: {:?}", file_path);
        Ok(())
    }

    /// Generate trend analysis report
    pub async fn generate_trend_report(
        &self,
        days_back: u32,
        branch_filter: Option<&str>,
    ) -> Result<TrendReport> {
        info!("Generating trend report for {} days", days_back);

        let entries = self.load_trend_entries(days_back, branch_filter).await?;

        if entries.is_empty() {
            warn!("No trend data found for the specified period");
            return Ok(TrendReport::empty());
        }

        let analysis = self.analyze_trends(&entries)?;

        Ok(TrendReport {
            period_days: days_back,
            branch_filter: branch_filter.map(|s| s.to_string()),
            total_entries: entries.len(),
            analysis,
            generated_at: Utc::now(),
        })
    }

    /// Get performance trends for specific tests
    pub async fn get_performance_trends(
        &self,
        test_names: &[String],
        days_back: u32,
    ) -> Result<HashMap<String, Vec<PerformanceDataPoint>>> {
        let entries = self.load_trend_entries(days_back, None).await?;
        let mut trends = HashMap::new();

        for test_name in test_names {
            let mut data_points = Vec::new();

            for entry in &entries {
                for suite in &entry.results {
                    for test in &suite.test_results {
                        if test.test_name == *test_name {
                            data_points.push(PerformanceDataPoint {
                                timestamp: entry.timestamp,
                                duration: test.duration,
                                memory_peak: test.metrics.memory_peak,
                                commit_sha: entry.commit_sha.clone(),
                                branch: entry.branch.clone(),
                            });
                            break;
                        }
                    }
                }
            }

            data_points.sort_by_key(|dp| dp.timestamp);
            trends.insert(test_name.clone(), data_points);
        }

        Ok(trends)
    }

    /// Detect performance regressions based on trends
    pub async fn detect_regressions(
        &self,
        current_results: &[TestSuiteResult],
        lookback_days: u32,
    ) -> Result<Vec<RegressionDetection>> {
        info!("Detecting performance regressions");

        let historical_entries = self.load_trend_entries(lookback_days, None).await?;
        let mut regressions = Vec::new();

        // Build performance baselines from historical data
        let baselines = self.calculate_performance_baselines(&historical_entries)?;

        for suite in current_results {
            for test in &suite.test_results {
                if let Some(baseline) = baselines.get(&test.test_name) {
                    let current_duration = test.duration;
                    let regression_threshold =
                        baseline.average_duration.mul_f64(self.config.regression_threshold);

                    if current_duration > regression_threshold {
                        let regression_percent = (current_duration.as_secs_f64()
                            / baseline.average_duration.as_secs_f64()
                            - 1.0)
                            * 100.0;

                        regressions.push(RegressionDetection {
                            test_name: test.test_name.clone(),
                            suite_name: suite.suite_name.clone(),
                            current_duration,
                            baseline_duration: baseline.average_duration,
                            regression_percent,
                            confidence: baseline.confidence,
                            sample_size: baseline.sample_size,
                        });
                    }
                }
            }
        }

        info!("Detected {} potential regressions", regressions.len());
        Ok(regressions)
    }

    /// Generate HTML trend report
    pub async fn generate_html_report(
        &self,
        report: &TrendReport,
        output_path: &Path,
    ) -> Result<()> {
        let html_content = self.generate_html_content(report)?;
        fs::write(output_path, html_content).await?;
        info!("Generated HTML trend report: {:?}", output_path);
        Ok(())
    }

    async fn load_trend_entries(
        &self,
        days_back: u32,
        branch_filter: Option<&str>,
    ) -> Result<Vec<TrendEntry>> {
        let cutoff_date = Utc::now() - chrono::Duration::days(days_back as i64);
        let mut entries = Vec::new();

        let mut dir_entries = fs::read_dir(&self.storage_path).await?;
        while let Some(entry) = dir_entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(content) = fs::read_to_string(&path).await {
                    if let Ok(trend_entry) = serde_json::from_str::<TrendEntry>(&content) {
                        // Filter by date
                        if trend_entry.timestamp < cutoff_date {
                            continue;
                        }

                        // Filter by branch if specified
                        if let Some(branch) = branch_filter {
                            if trend_entry.branch.as_deref() != Some(branch) {
                                continue;
                            }
                        }

                        entries.push(trend_entry);
                    }
                }
            }
        }

        entries.sort_by_key(|e| e.timestamp);
        Ok(entries)
    }

    fn analyze_trends(&self, entries: &[TrendEntry]) -> Result<TrendAnalysis> {
        let mut test_trends = HashMap::new();
        let mut suite_trends = HashMap::new();

        // Collect data points for each test
        for entry in entries {
            for suite in &entry.results {
                // Suite-level trends
                let suite_stats =
                    suite_trends.entry(suite.suite_name.clone()).or_insert_with(Vec::new);

                suite_stats.push(SuiteDataPoint {
                    timestamp: entry.timestamp,
                    total_tests: suite.summary.total_tests,
                    passed: suite.summary.passed,
                    failed: suite.summary.failed,
                    duration: suite.total_duration,
                    success_rate: suite.summary.success_rate,
                });

                // Test-level trends
                for test in &suite.test_results {
                    let test_stats =
                        test_trends.entry(test.test_name.clone()).or_insert_with(Vec::new);

                    test_stats.push(TestDataPoint {
                        timestamp: entry.timestamp,
                        duration: test.duration,
                        status: test.status.clone(),
                        memory_peak: test.metrics.memory_peak,
                    });
                }
            }
        }

        // Calculate trends
        let mut test_trend_analysis = HashMap::new();
        for (test_name, data_points) in test_trends {
            let analysis = self.calculate_test_trend(&data_points)?;
            test_trend_analysis.insert(test_name, analysis);
        }

        let mut suite_trend_analysis = HashMap::new();
        for (suite_name, data_points) in suite_trends {
            let analysis = self.calculate_suite_trend(&data_points)?;
            suite_trend_analysis.insert(suite_name, analysis);
        }

        Ok(TrendAnalysis {
            test_trends: test_trend_analysis,
            suite_trends: suite_trend_analysis,
            overall_stability: self.calculate_overall_stability(entries),
            performance_trend: self.calculate_performance_trend(entries),
        })
    }

    fn calculate_test_trend(&self, data_points: &[TestDataPoint]) -> Result<TestTrend> {
        if data_points.is_empty() {
            return Ok(TestTrend::default());
        }

        let durations: Vec<f64> = data_points.iter().map(|dp| dp.duration.as_secs_f64()).collect();

        let success_count = data_points
            .iter()
            .filter(|dp| matches!(dp.status, crate::results::TestStatus::Passed))
            .count();

        let success_rate = success_count as f64 / data_points.len() as f64;

        // Calculate linear regression for performance trend
        let performance_slope = self.calculate_linear_regression_slope(&durations)?;

        Ok(TestTrend {
            sample_size: data_points.len(),
            success_rate,
            average_duration: Duration::from_secs_f64(
                durations.iter().sum::<f64>() / durations.len() as f64,
            ),
            performance_trend: if performance_slope > 0.01 {
                PerformanceTrend::Degrading
            } else if performance_slope < -0.01 {
                PerformanceTrend::Improving
            } else {
                PerformanceTrend::Stable
            },
            stability: if success_rate > 0.95 {
                TestStability::Stable
            } else if success_rate > 0.8 {
                TestStability::Unstable
            } else {
                TestStability::Flaky
            },
        })
    }

    fn calculate_suite_trend(&self, data_points: &[SuiteDataPoint]) -> Result<SuiteTrend> {
        if data_points.is_empty() {
            return Ok(SuiteTrend::default());
        }

        let success_rates: Vec<f64> = data_points.iter().map(|dp| dp.success_rate).collect();

        let durations: Vec<f64> = data_points.iter().map(|dp| dp.duration.as_secs_f64()).collect();

        let avg_success_rate = success_rates.iter().sum::<f64>() / success_rates.len() as f64;
        let avg_duration =
            Duration::from_secs_f64(durations.iter().sum::<f64>() / durations.len() as f64);

        Ok(SuiteTrend {
            sample_size: data_points.len(),
            average_success_rate: avg_success_rate,
            average_duration: avg_duration,
            stability: if avg_success_rate > 0.95 {
                TestStability::Stable
            } else if avg_success_rate > 0.8 {
                TestStability::Unstable
            } else {
                TestStability::Flaky
            },
        })
    }

    fn calculate_overall_stability(&self, entries: &[TrendEntry]) -> f64 {
        if entries.is_empty() {
            return 1.0;
        }

        let total_success_rate: f64 = entries
            .iter()
            .flat_map(|entry| &entry.results)
            .map(|suite| suite.summary.success_rate)
            .sum();

        let total_suites: usize = entries.iter().map(|entry| entry.results.len()).sum();

        if total_suites > 0 {
            total_success_rate / (total_suites as f64 * 100.0)
        } else {
            1.0
        }
    }

    fn calculate_performance_trend(&self, entries: &[TrendEntry]) -> PerformanceTrend {
        if entries.len() < 2 {
            return PerformanceTrend::Stable;
        }

        let durations: Vec<f64> = entries
            .iter()
            .flat_map(|entry| &entry.results)
            .map(|suite| suite.total_duration.as_secs_f64())
            .collect();

        if let Ok(slope) = self.calculate_linear_regression_slope(&durations) {
            if slope > 0.1 {
                PerformanceTrend::Degrading
            } else if slope < -0.1 {
                PerformanceTrend::Improving
            } else {
                PerformanceTrend::Stable
            }
        } else {
            PerformanceTrend::Stable
        }
    }

    fn calculate_linear_regression_slope(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 2 {
            return Ok(0.0);
        }

        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(values.iter()).map(|(x, y)| x * y).sum();
        let sum_x_squared: f64 = x_values.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x);
        Ok(slope)
    }

    fn calculate_performance_baselines(
        &self,
        entries: &[TrendEntry],
    ) -> Result<HashMap<String, PerformanceBaseline>> {
        let mut test_durations: HashMap<String, Vec<Duration>> = HashMap::new();

        // Collect all durations for each test
        for entry in entries {
            for suite in &entry.results {
                for test in &suite.test_results {
                    test_durations.entry(test.test_name.clone()).or_default().push(test.duration);
                }
            }
        }

        let mut baselines = HashMap::new();

        for (test_name, durations) in test_durations {
            if durations.len() >= self.config.min_samples_for_baseline {
                let total_secs: f64 = durations.iter().map(|d| d.as_secs_f64()).sum();
                let average_duration = Duration::from_secs_f64(total_secs / durations.len() as f64);

                // Calculate confidence based on sample size and variance
                let variance = self.calculate_variance(&durations, average_duration);
                let confidence = self.calculate_confidence(durations.len(), variance);

                baselines.insert(
                    test_name,
                    PerformanceBaseline {
                        average_duration,
                        sample_size: durations.len(),
                        confidence,
                    },
                );
            }
        }

        Ok(baselines)
    }

    fn calculate_variance(&self, durations: &[Duration], average: Duration) -> f64 {
        let avg_secs = average.as_secs_f64();
        let variance_sum: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - avg_secs;
                diff * diff
            })
            .sum();

        variance_sum / durations.len() as f64
    }

    fn calculate_confidence(&self, sample_size: usize, variance: f64) -> f64 {
        // Simple confidence calculation based on sample size and variance
        let size_factor = (sample_size as f64).ln() / 10.0;
        let variance_factor = 1.0 / (1.0 + variance);
        (size_factor * variance_factor).min(1.0).max(0.0)
    }

    async fn cleanup_old_entries(&self) -> Result<()> {
        let cutoff_date = Utc::now() - chrono::Duration::days(self.config.retention_days as i64);
        let mut removed_count = 0;

        let mut dir_entries = fs::read_dir(&self.storage_path).await?;
        while let Some(entry) = dir_entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(content) = fs::read_to_string(&path).await {
                    if let Ok(trend_entry) = serde_json::from_str::<TrendEntry>(&content) {
                        if trend_entry.timestamp < cutoff_date {
                            if let Err(e) = fs::remove_file(&path).await {
                                warn!("Failed to remove old trend entry {:?}: {}", path, e);
                            } else {
                                removed_count += 1;
                            }
                        }
                    }
                }
            }
        }

        if removed_count > 0 {
            info!("Cleaned up {} old trend entries", removed_count);
        }

        Ok(())
    }

    fn generate_html_content(&self, report: &TrendReport) -> Result<String> {
        // Simple HTML template - in a real implementation, you might use a templating engine
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>BitNet.rs Test Trend Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(include_str!("../assets/trend_report.css"));
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        html.push_str("<h1>BitNet.rs Test Trend Report</h1>\n");
        html.push_str(&format!(
            "<p>Generated: {}</p>\n",
            report.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        html.push_str(&format!("<p>Period: {} days</p>\n", report.period_days));
        html.push_str(&format!("<p>Total entries: {}</p>\n", report.total_entries));

        if let Some(branch) = &report.branch_filter {
            html.push_str(&format!("<p>Branch: {}</p>\n", branch));
        }

        // Overall stability
        html.push_str("<h2>Overall Stability</h2>\n");
        html.push_str(&format!(
            "<p>Stability: {:.1}%</p>\n",
            report.analysis.overall_stability * 100.0
        ));
        html.push_str(&format!(
            "<p>Performance Trend: {:?}</p>\n",
            report.analysis.performance_trend
        ));

        // Suite trends
        html.push_str("<h2>Test Suite Trends</h2>\n");
        html.push_str("<table>\n");
        html.push_str(
            "<tr><th>Suite</th><th>Success Rate</th><th>Avg Duration</th><th>Stability</th></tr>\n",
        );

        for (suite_name, trend) in &report.analysis.suite_trends {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.1}%</td><td>{:.2}s</td><td>{:?}</td></tr>\n",
                suite_name,
                trend.average_success_rate,
                trend.average_duration.as_secs_f64(),
                trend.stability
            ));
        }
        html.push_str("</table>\n");

        html.push_str("</body>\n</html>");

        Ok(html)
    }
}

// Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendConfig {
    pub retention_days: u32,
    pub min_samples_for_baseline: usize,
    pub regression_threshold: f64,
}

impl Default for TrendConfig {
    fn default() -> Self {
        Self {
            retention_days: 90,
            min_samples_for_baseline: 5,
            regression_threshold: 1.2, // 20% slower
        }
    }
}

// Data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendEntry {
    pub timestamp: DateTime<Utc>,
    pub commit_sha: Option<String>,
    pub branch: Option<String>,
    pub pr_number: Option<u64>,
    pub results: Vec<TestSuiteResult>,
    pub environment: HashMap<String, String>,
    pub configuration: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct TestRunMetadata {
    pub commit_sha: Option<String>,
    pub branch: Option<String>,
    pub pr_number: Option<u64>,
    pub environment: HashMap<String, String>,
    pub configuration: HashMap<String, String>,
}

#[derive(Debug)]
pub struct TrendReport {
    pub period_days: u32,
    pub branch_filter: Option<String>,
    pub total_entries: usize,
    pub analysis: TrendAnalysis,
    pub generated_at: DateTime<Utc>,
}

impl TrendReport {
    fn empty() -> Self {
        Self {
            period_days: 0,
            branch_filter: None,
            total_entries: 0,
            analysis: TrendAnalysis::default(),
            generated_at: Utc::now(),
        }
    }
}

#[derive(Debug, Default)]
pub struct TrendAnalysis {
    pub test_trends: HashMap<String, TestTrend>,
    pub suite_trends: HashMap<String, SuiteTrend>,
    pub overall_stability: f64,
    pub performance_trend: PerformanceTrend,
}

#[derive(Debug, Default)]
pub struct TestTrend {
    pub sample_size: usize,
    pub success_rate: f64,
    pub average_duration: Duration,
    pub performance_trend: PerformanceTrend,
    pub stability: TestStability,
}

#[derive(Debug, Default)]
pub struct SuiteTrend {
    pub sample_size: usize,
    pub average_success_rate: f64,
    pub average_duration: Duration,
    pub stability: TestStability,
}

#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
    pub memory_peak: Option<u64>,
    pub commit_sha: Option<String>,
    pub branch: Option<String>,
}

#[derive(Debug)]
struct TestDataPoint {
    timestamp: DateTime<Utc>,
    duration: Duration,
    status: crate::results::TestStatus,
    memory_peak: Option<u64>,
}

#[derive(Debug)]
struct SuiteDataPoint {
    timestamp: DateTime<Utc>,
    total_tests: usize,
    passed: usize,
    failed: usize,
    duration: Duration,
    success_rate: f64,
}

#[derive(Debug)]
pub struct RegressionDetection {
    pub test_name: String,
    pub suite_name: String,
    pub current_duration: Duration,
    pub baseline_duration: Duration,
    pub regression_percent: f64,
    pub confidence: f64,
    pub sample_size: usize,
}

#[derive(Debug)]
struct PerformanceBaseline {
    average_duration: Duration,
    sample_size: usize,
    confidence: f64,
}

#[derive(Debug, Clone, Default)]
pub enum PerformanceTrend {
    Improving,
    #[default]
    Stable,
    Degrading,
}

#[derive(Debug, Clone, Default)]
pub enum TestStability {
    #[default]
    Stable,
    Unstable,
    Flaky,
}
