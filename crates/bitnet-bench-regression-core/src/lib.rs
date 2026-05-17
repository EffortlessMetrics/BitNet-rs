#![allow(clippy::module_name_repetitions)]
//! Regression detection by comparing baseline vs current benchmark receipts.

use bitnet_bench_receipts::BenchReceipt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// The outcome of comparing a single kernel's performance against its baseline.
#[derive(Debug, Clone)]
pub struct RegressionResult {
    pub kernel: String,
    pub baseline_us: u64,
    pub current_us: u64,
    pub change_pct: f64,
    pub is_regression: bool,
}

/// Compares current benchmark receipts against a baseline set.
pub struct RegressionDetector;

impl RegressionDetector {
    /// Check for regressions by matching kernels by name.
    ///
    /// A regression is flagged when `current_us` exceeds `baseline_us` by more
    /// than `threshold_pct` percent. Kernels present in `current` but absent
    /// from `baseline` are reported with zero baseline and no regression flag.
    #[must_use]
    pub fn check(
        baseline: &[BenchReceipt],
        current: &[BenchReceipt],
        threshold_pct: f64,
    ) -> Vec<RegressionResult> {
        let baseline_map: HashMap<&str, u64> =
            baseline.iter().map(|r| (r.kernel_name.as_str(), r.elapsed_us)).collect();

        current
            .iter()
            .map(|r| {
                let current_us = r.elapsed_us;
                match baseline_map.get(r.kernel_name.as_str()) {
                    Some(&baseline_us) if baseline_us > 0 => {
                        let change_pct =
                            ((current_us as f64 - baseline_us as f64) / baseline_us as f64) * 100.0;
                        RegressionResult {
                            kernel: r.kernel_name.clone(),
                            baseline_us,
                            current_us,
                            change_pct,
                            is_regression: change_pct > threshold_pct,
                        }
                    }
                    _ => RegressionResult {
                        kernel: r.kernel_name.clone(),
                        baseline_us: 0,
                        current_us,
                        change_pct: 0.0,
                        is_regression: false,
                    },
                }
            })
            .collect()
    }
}

/// One normalized timing-style benchmark metric extracted from JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub name: String,
    pub value: f64,
}

/// Classification for one benchmark metric comparison.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceStatus {
    New,
    Removed,
    Maintained,
    Regression,
    Improvement,
}

/// Comparison details for a single metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceSummaryEntry {
    pub status: PerformanceStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_percent: Option<f64>,
}

/// A metric that regressed beyond the allowed ratio threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceRegression {
    pub benchmark: String,
    pub baseline: f64,
    pub current: f64,
    pub ratio: f64,
    pub change_percent: f64,
}

/// A metric that improved beyond the configured improvement ratio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceImprovement {
    pub benchmark: String,
    pub baseline: f64,
    pub current: f64,
    pub ratio: f64,
    pub change_percent: f64,
}

/// Full performance comparison result used by release-validation tooling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceComparison {
    pub passed: bool,
    pub regressions: Vec<PerformanceRegression>,
    pub improvements: Vec<PerformanceImprovement>,
    pub summary: BTreeMap<String, PerformanceSummaryEntry>,
}

impl PerformanceComparison {
    /// Render a deterministic Markdown report.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut report = Vec::new();
        report.push("# Performance Comparison Report".to_string());
        report.push(String::new());
        report.push(if self.passed {
            "✅ **Performance validation PASSED**".to_string()
        } else {
            "❌ **Performance validation FAILED**".to_string()
        });
        report.push(String::new());
        report.push("## Summary".to_string());
        report.push(format!("- Total benchmarks: {}", self.summary.len()));
        report.push(format!("- Regressions: {}", self.regressions.len()));
        report.push(format!("- Improvements: {}", self.improvements.len()));
        report.push(String::new());

        if !self.regressions.is_empty() {
            report.push("## ❌ Performance Regressions".to_string());
            for regression in &self.regressions {
                report.push(format!(
                    "- **{}**: {:+.2}% ({:.3}s → {:.3}s)",
                    regression.benchmark,
                    regression.change_percent,
                    regression.baseline,
                    regression.current
                ));
            }
            report.push(String::new());
        }

        if !self.improvements.is_empty() {
            report.push("## ✅ Performance Improvements".to_string());
            for improvement in &self.improvements {
                report.push(format!(
                    "- **{}**: {:+.2}% ({:.3}s → {:.3}s)",
                    improvement.benchmark,
                    improvement.change_percent,
                    improvement.baseline,
                    improvement.current
                ));
            }
            report.push(String::new());
        }

        report.push("## Detailed Results".to_string());
        for (benchmark, data) in &self.summary {
            let status_icon = match data.status {
                PerformanceStatus::Regression => "❌",
                PerformanceStatus::Improvement => "✅",
                PerformanceStatus::Maintained => "➖",
                PerformanceStatus::New => "🆕",
                PerformanceStatus::Removed => "🗑️",
            };
            if let Some(change_percent) = data.change_percent {
                report.push(format!("- {status_icon} **{benchmark}**: {change_percent:+.2}%"));
            } else {
                report.push(format!("- {status_icon} **{benchmark}**: {:?}", data.status));
            }
        }

        report.join("\n")
    }
}

/// Extract legacy benchmark metrics from the release-validation JSON shape.
///
/// Accepted entries are objects inside a top-level `benchmarks` array. The
/// metric value is selected in the same priority order as the historical Python
/// helper: `value`, then `mean`, then `median`.
#[must_use]
pub fn extract_legacy_benchmark_metrics(results: &Value) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    let Some(benchmarks) = results.get("benchmarks").and_then(Value::as_array) else {
        return metrics;
    };

    for benchmark in benchmarks {
        let Some(name) = benchmark.get("name").and_then(Value::as_str) else {
            continue;
        };
        let value = benchmark
            .get("value")
            .or_else(|| benchmark.get("mean"))
            .or_else(|| benchmark.get("median"))
            .and_then(Value::as_f64);
        if let Some(value) = value {
            metrics.insert(name.to_string(), value);
        }
    }

    metrics
}

/// Merge metrics from inference and kernel benchmark result files.
#[must_use]
pub fn merge_benchmark_metrics(
    inference: BTreeMap<String, f64>,
    kernels: BTreeMap<String, f64>,
) -> BTreeMap<String, f64> {
    inference.into_iter().chain(kernels).collect()
}

/// Compare timing metrics with the historical ratio threshold semantics.
///
/// Lower values are better. A `threshold_ratio` of `0.95` allows a regression
/// up to `1 / 0.95` (about 5.26%) and records improvements below `0.95`.
#[must_use]
pub fn compare_performance_metrics(
    baseline_metrics: &BTreeMap<String, f64>,
    current_metrics: &BTreeMap<String, f64>,
    threshold_ratio: f64,
) -> PerformanceComparison {
    let mut comparison = PerformanceComparison {
        passed: true,
        regressions: Vec::new(),
        improvements: Vec::new(),
        summary: BTreeMap::new(),
    };

    let all_benchmarks: BTreeSet<String> =
        baseline_metrics.keys().chain(current_metrics.keys()).cloned().collect();

    for benchmark in all_benchmarks {
        match (baseline_metrics.get(&benchmark), current_metrics.get(&benchmark)) {
            (None, Some(&current)) => {
                comparison.summary.insert(
                    benchmark,
                    PerformanceSummaryEntry {
                        status: PerformanceStatus::New,
                        baseline: None,
                        current: Some(current),
                        ratio: None,
                        change_percent: None,
                    },
                );
            }
            (Some(&baseline), None) => {
                comparison.summary.insert(
                    benchmark,
                    PerformanceSummaryEntry {
                        status: PerformanceStatus::Removed,
                        baseline: Some(baseline),
                        current: None,
                        ratio: None,
                        change_percent: None,
                    },
                );
            }
            (Some(&baseline), Some(&current)) if baseline > 0.0 => {
                let ratio = current / baseline;
                let change_percent = (ratio - 1.0) * 100.0;
                let status = if ratio > (1.0 / threshold_ratio) {
                    comparison.passed = false;
                    comparison.regressions.push(PerformanceRegression {
                        benchmark: benchmark.clone(),
                        baseline,
                        current,
                        ratio,
                        change_percent,
                    });
                    PerformanceStatus::Regression
                } else if ratio < threshold_ratio {
                    comparison.improvements.push(PerformanceImprovement {
                        benchmark: benchmark.clone(),
                        baseline,
                        current,
                        ratio,
                        change_percent,
                    });
                    PerformanceStatus::Improvement
                } else {
                    PerformanceStatus::Maintained
                };
                comparison.summary.insert(
                    benchmark,
                    PerformanceSummaryEntry {
                        status,
                        baseline: Some(baseline),
                        current: Some(current),
                        ratio: Some(ratio),
                        change_percent: Some(change_percent),
                    },
                );
            }
            (Some(&baseline), Some(&current)) => {
                comparison.summary.insert(
                    benchmark,
                    PerformanceSummaryEntry {
                        status: PerformanceStatus::Maintained,
                        baseline: Some(baseline),
                        current: Some(current),
                        ratio: None,
                        change_percent: None,
                    },
                );
            }
            (None, None) => unreachable!("benchmark set is built from existing keys"),
        }
    }

    comparison
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn receipt(name: &str, elapsed_us: u64) -> BenchReceipt {
        BenchReceipt::new(name, [256, 1, 1], [1, 1, 1], elapsed_us, 0.0, 0, "", "")
    }

    #[test]
    fn test_no_regression_when_faster() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 900)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_regression);
    }

    #[test]
    fn test_no_regression_within_threshold() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 1050)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(!results[0].is_regression);
        assert!((results[0].change_pct - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_regression_above_threshold() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 1200)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(results[0].is_regression);
        assert!((results[0].change_pct - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_exact_threshold_not_regression() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 1100)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(!results[0].is_regression);
    }

    #[test]
    fn test_new_kernel_no_baseline() {
        let baseline = vec![];
        let current = vec![receipt("new_kernel", 500)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_regression);
        assert_eq!(results[0].baseline_us, 0);
    }

    #[test]
    fn test_multiple_kernels_mixed() {
        let baseline = vec![receipt("fast", 100), receipt("slow", 1000)];
        let current = vec![receipt("fast", 90), receipt("slow", 1500)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(!results[0].is_regression);
        assert!(results[1].is_regression);
    }

    #[test]
    fn test_empty_inputs() {
        let results = RegressionDetector::check(&[], &[], 10.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_change_pct_negative_for_improvement() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 800)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(results[0].change_pct < 0.0);
        assert!((results[0].change_pct - (-20.0)).abs() < 0.01);
    }

    #[test]
    fn test_zero_baseline_no_panic() {
        let baseline = vec![receipt("k", 0)];
        let current = vec![receipt("k", 100)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(!results[0].is_regression);
        assert_eq!(results[0].baseline_us, 0);
    }

    #[test]
    fn extracts_legacy_benchmark_array_values_by_priority() {
        let input = json!({
            "benchmarks": [
                {"name": "value_case", "value": 1.0, "mean": 2.0},
                {"name": "mean_case", "mean": 3.0, "median": 4.0},
                {"name": "median_case", "median": 5.0},
                {"name": "ignored_without_value"}
            ]
        });

        let metrics = extract_legacy_benchmark_metrics(&input);
        assert_eq!(metrics.get("value_case"), Some(&1.0));
        assert_eq!(metrics.get("mean_case"), Some(&3.0));
        assert_eq!(metrics.get("median_case"), Some(&5.0));
        assert!(!metrics.contains_key("ignored_without_value"));
    }

    #[test]
    fn compares_legacy_metrics_with_deterministic_statuses() {
        let baseline = BTreeMap::from([
            ("faster".to_string(), 100.0),
            ("stable".to_string(), 100.0),
            ("slower".to_string(), 100.0),
            ("removed".to_string(), 100.0),
        ]);
        let current = BTreeMap::from([
            ("faster".to_string(), 90.0),
            ("stable".to_string(), 102.0),
            ("slower".to_string(), 110.0),
            ("new".to_string(), 50.0),
        ]);

        let comparison = compare_performance_metrics(&baseline, &current, 0.95);

        assert!(!comparison.passed);
        assert_eq!(comparison.regressions[0].benchmark, "slower");
        assert_eq!(comparison.improvements[0].benchmark, "faster");
        assert_eq!(comparison.summary["new"].status, PerformanceStatus::New);
        assert_eq!(comparison.summary["removed"].status, PerformanceStatus::Removed);
        assert_eq!(comparison.summary["stable"].status, PerformanceStatus::Maintained);
    }

    #[test]
    fn markdown_rendering_does_not_require_missing_change_field() {
        let baseline = BTreeMap::from([("new_only".to_string(), 1.0)]);
        let current = BTreeMap::new();
        let comparison = compare_performance_metrics(&baseline, &current, 0.95);
        let report = comparison.to_markdown();
        assert!(report.contains("Performance Comparison Report"));
        assert!(report.contains("🗑️ **new_only**"));
    }
}
