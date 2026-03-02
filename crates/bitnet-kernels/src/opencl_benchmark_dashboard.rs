//! Benchmark result collection and dashboard data generation for Intel Arc A770.
//!
//! Collects timing data from kernel runs, computes statistics, generates
//! comparison reports between backends (CPU vs OpenCL) and across runs. All
//! implementations are CPU reference code — no OpenCL runtime required.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// BenchmarkResult — single kernel timing
// ---------------------------------------------------------------------------

/// A single benchmark measurement from one kernel invocation.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the kernel being benchmarked.
    pub kernel_name: String,
    /// Wall-clock duration in microseconds.
    pub duration_us: f64,
    /// Achieved throughput in GFLOP/s (if applicable).
    pub gflops: Option<f64>,
    /// Achieved memory bandwidth in GB/s (if applicable).
    pub bandwidth_gbps: Option<f64>,
    /// Unix timestamp (seconds since epoch) when this sample was recorded.
    pub timestamp: f64,
}

impl BenchmarkResult {
    /// Create a new benchmark result with only timing data.
    pub fn new(kernel_name: impl Into<String>, duration_us: f64, timestamp: f64) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            duration_us,
            gflops: None,
            bandwidth_gbps: None,
            timestamp,
        }
    }

    /// Create a result with full throughput metrics.
    pub fn with_metrics(
        kernel_name: impl Into<String>,
        duration_us: f64,
        gflops: f64,
        bandwidth_gbps: f64,
        timestamp: f64,
    ) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            duration_us,
            gflops: Some(gflops),
            bandwidth_gbps: Some(bandwidth_gbps),
            timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// BenchmarkSuite — named collection of results
// ---------------------------------------------------------------------------

/// A named collection of benchmark results with associated metadata.
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    /// Human-readable suite name (e.g. "matmul_i2s_1024x1024").
    pub name: String,
    /// Individual measurements.
    pub results: Vec<BenchmarkResult>,
    /// Hardware description string (e.g. "Intel Arc A770 16GB").
    pub hardware_info: String,
    /// Model description string (e.g. "BitNet-2B-4T QK256").
    pub model_info: String,
}

impl BenchmarkSuite {
    pub fn new(
        name: impl Into<String>,
        hardware_info: impl Into<String>,
        model_info: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            results: Vec::new(),
            hardware_info: hardware_info.into(),
            model_info: model_info.into(),
        }
    }

    /// Add a single result to the suite.
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Compute statistics over the duration_us values.
    pub fn duration_stats(&self) -> Option<StatsSummary> {
        let durations: Vec<f64> = self.results.iter().map(|r| r.duration_us).collect();
        StatsSummary::compute(&durations)
    }

    /// Compute statistics over gflops values (ignoring `None` entries).
    pub fn gflops_stats(&self) -> Option<StatsSummary> {
        let vals: Vec<f64> = self.results.iter().filter_map(|r| r.gflops).collect();
        StatsSummary::compute(&vals)
    }
}

// ---------------------------------------------------------------------------
// StatsSummary — descriptive statistics
// ---------------------------------------------------------------------------

/// Descriptive statistics for a collection of f64 samples.
#[derive(Debug, Clone)]
pub struct StatsSummary {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub std_dev: f64,
    /// Number of samples used to compute these stats.
    pub count: usize,
}

impl StatsSummary {
    /// Compute statistics from a slice of samples.
    ///
    /// Returns `None` if the slice is empty or contains only non-finite values.
    pub fn compute(values: &[f64]) -> Option<Self> {
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return None;
        }
        let mut sorted = finite.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count = sorted.len();
        let min = sorted[0];
        let max = sorted[count - 1];
        let mean = sorted.iter().sum::<f64>() / count as f64;
        let median = percentile_sorted(&sorted, 50.0);
        let p95 = percentile_sorted(&sorted, 95.0);
        let p99 = percentile_sorted(&sorted, 99.0);

        let variance = if count > 1 {
            sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        Some(Self { min, max, mean, median, p95, p99, std_dev, count })
    }
}

impl fmt::Display for StatsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n={} min={:.1} max={:.1} mean={:.1} median={:.1} p95={:.1} p99={:.1} σ={:.1}",
            self.count,
            self.min,
            self.max,
            self.mean,
            self.median,
            self.p95,
            self.p99,
            self.std_dev,
        )
    }
}

/// Compute the `p`-th percentile from an already-sorted slice using linear interpolation.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=100.0).contains(&p));

    if sorted.len() == 1 {
        return sorted[0];
    }

    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;

    if lo == hi { sorted[lo] } else { sorted[lo] * (1.0 - frac) + sorted[hi] * frac }
}

// ---------------------------------------------------------------------------
// ComparisonReport — CPU vs GPU
// ---------------------------------------------------------------------------

/// Side-by-side comparison of CPU and GPU benchmark statistics.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Kernel or suite being compared.
    pub label: String,
    /// CPU-side statistics (duration in µs).
    pub cpu_stats: StatsSummary,
    /// GPU-side statistics (duration in µs).
    pub gpu_stats: StatsSummary,
    /// Speedup ratio: `cpu_mean / gpu_mean`.
    pub speedup: f64,
    /// Efficiency percentage: `(speedup / theoretical_max) * 100`.
    pub efficiency_pct: f64,
}

impl ComparisonReport {
    /// Build a comparison from two stats summaries.
    ///
    /// `theoretical_max_speedup` is used to compute `efficiency_pct`. Pass `1.0`
    /// if no theoretical limit is known (efficiency will equal speedup × 100).
    pub fn new(
        label: impl Into<String>,
        cpu_stats: StatsSummary,
        gpu_stats: StatsSummary,
        theoretical_max_speedup: f64,
    ) -> Self {
        let speedup = if gpu_stats.mean > 0.0 { cpu_stats.mean / gpu_stats.mean } else { 0.0 };
        let theoretical = if theoretical_max_speedup > 0.0 { theoretical_max_speedup } else { 1.0 };
        let efficiency_pct = (speedup / theoretical) * 100.0;
        Self { label: label.into(), cpu_stats, gpu_stats, speedup, efficiency_pct }
    }
}

impl fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: speedup={:.2}x efficiency={:.1}% (cpu_mean={:.1}µs gpu_mean={:.1}µs)",
            self.label, self.speedup, self.efficiency_pct, self.cpu_stats.mean, self.gpu_stats.mean,
        )
    }
}

// ---------------------------------------------------------------------------
// TrendAnalysis — regression detection
// ---------------------------------------------------------------------------

/// Direction of a performance trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Improving,
    Regressing,
    Stable,
}

impl fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Improving => write!(f, "improving"),
            Self::Regressing => write!(f, "regressing"),
            Self::Stable => write!(f, "stable"),
        }
    }
}

/// Trend analysis for a sequence of benchmark runs.
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Whether a statistically significant regression was detected.
    pub regression_detected: bool,
    /// Overall trend direction.
    pub trend_direction: TrendDirection,
    /// Confidence level (0.0–1.0) in the trend assessment.
    pub confidence: f64,
    /// Percentage change from first to last window mean.
    pub pct_change: f64,
}

impl TrendAnalysis {
    /// Analyse a time-ordered sequence of duration measurements.
    ///
    /// Splits the samples into a *baseline* (first half) and a *recent*
    /// (second half) window, then compares their means.
    ///
    /// `regression_threshold` is a fractional increase (e.g. 0.05 for 5%) above
    /// which a regression is flagged.
    pub fn analyse(values: &[f64], regression_threshold: f64) -> Option<Self> {
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.len() < 2 {
            return None;
        }

        let mid = finite.len() / 2;
        let baseline_mean = finite[..mid].iter().sum::<f64>() / mid as f64;
        let recent_mean = finite[mid..].iter().sum::<f64>() / (finite.len() - mid) as f64;

        if baseline_mean == 0.0 {
            return Some(Self {
                regression_detected: false,
                trend_direction: TrendDirection::Stable,
                confidence: 0.0,
                pct_change: 0.0,
            });
        }

        let pct_change = (recent_mean - baseline_mean) / baseline_mean;

        // For durations, an increase is a regression.
        let regression_detected = pct_change > regression_threshold;
        let trend_direction = if pct_change > regression_threshold {
            TrendDirection::Regressing
        } else if pct_change < -regression_threshold {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        };

        // Simple confidence: ratio of |change| to threshold, clamped to [0, 1].
        let confidence = if regression_threshold > 0.0 {
            (pct_change.abs() / regression_threshold).min(1.0)
        } else {
            1.0
        };

        Some(Self { regression_detected, trend_direction, confidence, pct_change })
    }
}

impl fmt::Display for TrendAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "trend={} change={:+.1}% confidence={:.0}%{}",
            self.trend_direction,
            self.pct_change * 100.0,
            self.confidence * 100.0,
            if self.regression_detected { " ⚠ REGRESSION" } else { "" },
        )
    }
}

// ---------------------------------------------------------------------------
// DashboardData — top-level aggregate
// ---------------------------------------------------------------------------

/// Top-level dashboard payload combining all benchmark data.
#[derive(Debug, Clone)]
pub struct DashboardData {
    pub suites: Vec<BenchmarkSuite>,
    pub comparisons: Vec<ComparisonReport>,
    pub trends: Vec<TrendAnalysis>,
    /// ISO-8601 timestamp string when this dashboard was generated.
    pub generated_at: String,
}

impl DashboardData {
    pub fn new(generated_at: impl Into<String>) -> Self {
        Self {
            suites: Vec::new(),
            comparisons: Vec::new(),
            trends: Vec::new(),
            generated_at: generated_at.into(),
        }
    }

    pub fn add_suite(&mut self, suite: BenchmarkSuite) {
        self.suites.push(suite);
    }

    pub fn add_comparison(&mut self, report: ComparisonReport) {
        self.comparisons.push(report);
    }

    pub fn add_trend(&mut self, trend: TrendAnalysis) {
        self.trends.push(trend);
    }
}

// ---------------------------------------------------------------------------
// ReportFormatter — markdown / JSON output
// ---------------------------------------------------------------------------

/// Generates human-readable and machine-readable reports from [`DashboardData`].
pub struct ReportFormatter;

impl ReportFormatter {
    /// Render a full dashboard as Markdown.
    pub fn to_markdown(data: &DashboardData) -> String {
        let mut out = String::new();
        out.push_str("# Benchmark Dashboard\n\n");
        out.push_str(&format!("Generated: {}\n\n", data.generated_at));

        // Suites
        if !data.suites.is_empty() {
            out.push_str("## Suites\n\n");
            for suite in &data.suites {
                out.push_str(&format!(
                    "### {}\n\nHardware: {} | Model: {} | Samples: {}\n\n",
                    suite.name,
                    suite.hardware_info,
                    suite.model_info,
                    suite.results.len(),
                ));
                if let Some(stats) = suite.duration_stats() {
                    out.push_str(&format!(
                        "| Metric | Value |\n|--------|-------|\n\
                         | Min | {:.1} µs |\n| Max | {:.1} µs |\n\
                         | Mean | {:.1} µs |\n| Median | {:.1} µs |\n\
                         | P95 | {:.1} µs |\n| P99 | {:.1} µs |\n\
                         | Std Dev | {:.1} µs |\n\n",
                        stats.min,
                        stats.max,
                        stats.mean,
                        stats.median,
                        stats.p95,
                        stats.p99,
                        stats.std_dev,
                    ));
                }
            }
        }

        // Comparisons
        if !data.comparisons.is_empty() {
            out.push_str("## CPU vs GPU Comparisons\n\n");
            out.push_str("| Kernel | CPU Mean (µs) | GPU Mean (µs) | Speedup | Efficiency |\n");
            out.push_str("|--------|--------------|--------------|---------|------------|\n");
            for cmp in &data.comparisons {
                out.push_str(&format!(
                    "| {} | {:.1} | {:.1} | {:.2}x | {:.1}% |\n",
                    cmp.label,
                    cmp.cpu_stats.mean,
                    cmp.gpu_stats.mean,
                    cmp.speedup,
                    cmp.efficiency_pct,
                ));
            }
            out.push('\n');
        }

        // Trends
        if !data.trends.is_empty() {
            out.push_str("## Trends\n\n");
            for (i, trend) in data.trends.iter().enumerate() {
                out.push_str(&format!("- Run {}: {}\n", i + 1, trend));
            }
            out.push('\n');
        }

        out
    }

    /// Render a full dashboard as JSON.
    pub fn to_json(data: &DashboardData) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"generated_at\": \"{}\",\n", data.generated_at));

        // Suites
        out.push_str("  \"suites\": [\n");
        for (i, suite) in data.suites.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"name\": \"{}\",\n", suite.name));
            out.push_str(&format!("      \"hardware_info\": \"{}\",\n", suite.hardware_info));
            out.push_str(&format!("      \"model_info\": \"{}\",\n", suite.model_info));
            out.push_str(&format!("      \"sample_count\": {}", suite.results.len()));
            if let Some(stats) = suite.duration_stats() {
                out.push_str(",\n");
                Self::append_stats_json(&mut out, "duration_stats", &stats, 6);
            } else {
                out.push('\n');
            }
            out.push_str("    }");
            if i + 1 < data.suites.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ],\n");

        // Comparisons
        out.push_str("  \"comparisons\": [\n");
        for (i, cmp) in data.comparisons.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"label\": \"{}\",\n", cmp.label));
            out.push_str(&format!("      \"speedup\": {:.4},\n", cmp.speedup));
            out.push_str(&format!("      \"efficiency_pct\": {:.2}\n", cmp.efficiency_pct));
            out.push_str("    }");
            if i + 1 < data.comparisons.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ],\n");

        // Trends
        out.push_str("  \"trends\": [\n");
        for (i, trend) in data.trends.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!(
                "      \"regression_detected\": {},\n",
                trend.regression_detected
            ));
            out.push_str(&format!("      \"trend_direction\": \"{}\",\n", trend.trend_direction));
            out.push_str(&format!("      \"confidence\": {:.4},\n", trend.confidence));
            out.push_str(&format!("      \"pct_change\": {:.4}\n", trend.pct_change));
            out.push_str("    }");
            if i + 1 < data.trends.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]\n");

        out.push('}');
        out
    }

    fn append_stats_json(out: &mut String, key: &str, stats: &StatsSummary, indent: usize) {
        let pad: String = " ".repeat(indent);
        out.push_str(&format!("{pad}\"{key}\": {{\n"));
        out.push_str(&format!("{pad}  \"min\": {:.4},\n", stats.min));
        out.push_str(&format!("{pad}  \"max\": {:.4},\n", stats.max));
        out.push_str(&format!("{pad}  \"mean\": {:.4},\n", stats.mean));
        out.push_str(&format!("{pad}  \"median\": {:.4},\n", stats.median));
        out.push_str(&format!("{pad}  \"p95\": {:.4},\n", stats.p95));
        out.push_str(&format!("{pad}  \"p99\": {:.4},\n", stats.p99));
        out.push_str(&format!("{pad}  \"std_dev\": {:.4},\n", stats.std_dev));
        out.push_str(&format!("{pad}  \"count\": {}\n", stats.count));
        out.push_str(&format!("{pad}}}\n"));
    }

    /// Render only statistics as a compact one-line summary.
    pub fn stats_oneliner(label: &str, stats: &StatsSummary) -> String {
        format!(
            "{}: mean={:.1}µs median={:.1}µs p95={:.1}µs (n={})",
            label, stats.mean, stats.median, stats.p95, stats.count,
        )
    }

    /// Build a [`DashboardData`] from named CPU and GPU suites with auto-comparison.
    pub fn build_dashboard(
        cpu_suites: Vec<BenchmarkSuite>,
        gpu_suites: Vec<BenchmarkSuite>,
        generated_at: impl Into<String>,
    ) -> DashboardData {
        let mut dashboard = DashboardData::new(generated_at);

        // Index GPU suites by name for matching.
        let gpu_by_name: HashMap<&str, &BenchmarkSuite> =
            gpu_suites.iter().map(|s| (s.name.as_str(), s)).collect();

        for cpu_suite in &cpu_suites {
            dashboard.add_suite(cpu_suite.clone());
            if let Some(gpu_suite) = gpu_by_name.get(cpu_suite.name.as_str()) {
                dashboard.add_suite((*gpu_suite).clone());
                if let (Some(cpu_stats), Some(gpu_stats)) =
                    (cpu_suite.duration_stats(), gpu_suite.duration_stats())
                {
                    let report = ComparisonReport::new(&cpu_suite.name, cpu_stats, gpu_stats, 1.0);
                    dashboard.add_comparison(report);
                }
            }
        }

        // Add remaining GPU-only suites.
        for gpu_suite in &gpu_suites {
            if !cpu_suites.iter().any(|c| c.name == gpu_suite.name) {
                dashboard.add_suite(gpu_suite.clone());
            }
        }

        dashboard
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn make_results(kernel: &str, durations: &[f64]) -> Vec<BenchmarkResult> {
        durations
            .iter()
            .enumerate()
            .map(|(i, &d)| BenchmarkResult::new(kernel, d, i as f64))
            .collect()
    }

    fn make_suite(name: &str, durations: &[f64]) -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new(name, "Intel Arc A770 16GB", "BitNet-2B-4T QK256");
        for r in make_results(name, durations) {
            suite.add_result(r);
        }
        suite
    }

    // =====================================================================
    // Statistics calculation
    // =====================================================================

    #[test]
    fn test_stats_mean() {
        let stats = StatsSummary::compute(&[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        assert!((stats.mean - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_median_odd() {
        let stats = StatsSummary::compute(&[5.0, 1.0, 3.0]).unwrap();
        assert!((stats.median - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_median_even() {
        let stats = StatsSummary::compute(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((stats.median - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_p95() {
        // 20 evenly spaced values 1..=20
        let vals: Vec<f64> = (1..=20).map(|v| v as f64).collect();
        let stats = StatsSummary::compute(&vals).unwrap();
        // P95 with linear interpolation: rank = 0.95 * 19 = 18.05
        let expected = 19.0 * 0.95 + 20.0 * 0.05;
        assert!((stats.p95 - expected).abs() < 1e-9);
    }

    #[test]
    fn test_stats_p99() {
        let vals: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        let stats = StatsSummary::compute(&vals).unwrap();
        // rank = 0.99 * 99 = 98.01
        let expected = 99.0 * 0.99 + 100.0 * 0.01;
        assert!((stats.p99 - expected).abs() < 1e-9);
    }

    #[test]
    fn test_stats_std_dev() {
        // [2, 4, 4, 4, 5, 5, 7, 9] — classic example
        let vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = StatsSummary::compute(&vals).unwrap();
        let expected_mean = 5.0;
        assert!((stats.mean - expected_mean).abs() < 1e-9);
        // sample variance = sum((x-mean)^2) / (n-1)
        let var: f64 = vals.iter().map(|v| (v - expected_mean).powi(2)).sum::<f64>() / 7.0;
        assert!((stats.std_dev - var.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_stats_min_max() {
        let stats = StatsSummary::compute(&[42.0, 7.0, 99.0, 1.0]).unwrap();
        assert!((stats.min - 1.0).abs() < 1e-9);
        assert!((stats.max - 99.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_count() {
        let stats = StatsSummary::compute(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(stats.count, 3);
    }

    // =====================================================================
    // Empty / single result handling
    // =====================================================================

    #[test]
    fn test_stats_empty() {
        assert!(StatsSummary::compute(&[]).is_none());
    }

    #[test]
    fn test_stats_single_value() {
        let stats = StatsSummary::compute(&[42.0]).unwrap();
        assert!((stats.min - 42.0).abs() < 1e-9);
        assert!((stats.max - 42.0).abs() < 1e-9);
        assert!((stats.mean - 42.0).abs() < 1e-9);
        assert!((stats.median - 42.0).abs() < 1e-9);
        assert!((stats.p95 - 42.0).abs() < 1e-9);
        assert!((stats.p99 - 42.0).abs() < 1e-9);
        assert!((stats.std_dev - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_two_values() {
        let stats = StatsSummary::compute(&[10.0, 20.0]).unwrap();
        assert!((stats.mean - 15.0).abs() < 1e-9);
        assert!((stats.median - 15.0).abs() < 1e-9);
    }

    // =====================================================================
    // Edge cases: identical times, zero duration, NaN values
    // =====================================================================

    #[test]
    fn test_stats_identical_values() {
        let stats = StatsSummary::compute(&[5.0, 5.0, 5.0, 5.0]).unwrap();
        assert!((stats.min - 5.0).abs() < 1e-9);
        assert!((stats.max - 5.0).abs() < 1e-9);
        assert!((stats.mean - 5.0).abs() < 1e-9);
        assert!((stats.std_dev).abs() < 1e-9);
    }

    #[test]
    fn test_stats_zero_duration() {
        let stats = StatsSummary::compute(&[0.0, 0.0, 0.0]).unwrap();
        assert!((stats.mean).abs() < 1e-9);
        assert!((stats.std_dev).abs() < 1e-9);
    }

    #[test]
    fn test_stats_nan_filtered() {
        let stats = StatsSummary::compute(&[f64::NAN, 10.0, 20.0, f64::NAN]).unwrap();
        assert_eq!(stats.count, 2);
        assert!((stats.mean - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_all_nan() {
        assert!(StatsSummary::compute(&[f64::NAN, f64::NAN]).is_none());
    }

    #[test]
    fn test_stats_infinity_filtered() {
        let stats = StatsSummary::compute(&[f64::INFINITY, 10.0, f64::NEG_INFINITY, 20.0]).unwrap();
        assert_eq!(stats.count, 2);
        assert!((stats.mean - 15.0).abs() < 1e-9);
    }

    // =====================================================================
    // Speedup computation
    // =====================================================================

    #[test]
    fn test_speedup_basic() {
        let cpu = StatsSummary::compute(&[100.0, 100.0, 100.0]).unwrap();
        let gpu = StatsSummary::compute(&[10.0, 10.0, 10.0]).unwrap();
        let report = ComparisonReport::new("matmul", cpu, gpu, 1.0);
        assert!((report.speedup - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_speedup_slower_gpu() {
        let cpu = StatsSummary::compute(&[10.0, 10.0]).unwrap();
        let gpu = StatsSummary::compute(&[100.0, 100.0]).unwrap();
        let report = ComparisonReport::new("matmul", cpu, gpu, 1.0);
        assert!((report.speedup - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_speedup_equal() {
        let cpu = StatsSummary::compute(&[50.0, 50.0]).unwrap();
        let gpu = StatsSummary::compute(&[50.0, 50.0]).unwrap();
        let report = ComparisonReport::new("matmul", cpu, gpu, 1.0);
        assert!((report.speedup - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_speedup_zero_gpu_mean() {
        let cpu = StatsSummary::compute(&[100.0]).unwrap();
        let gpu = StatsSummary::compute(&[0.0]).unwrap();
        let report = ComparisonReport::new("matmul", cpu, gpu, 1.0);
        assert!((report.speedup).abs() < 1e-9); // 0.0 — avoid division by zero
    }

    #[test]
    fn test_efficiency_with_theoretical_max() {
        let cpu = StatsSummary::compute(&[100.0, 100.0]).unwrap();
        let gpu = StatsSummary::compute(&[10.0, 10.0]).unwrap();
        // Theoretical max 20x → achieved 10x → efficiency 50%
        let report = ComparisonReport::new("matmul", cpu, gpu, 20.0);
        assert!((report.efficiency_pct - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_comparison_display() {
        let cpu = StatsSummary::compute(&[100.0, 100.0]).unwrap();
        let gpu = StatsSummary::compute(&[25.0, 25.0]).unwrap();
        let report = ComparisonReport::new("gemm", cpu, gpu, 1.0);
        let display = format!("{}", report);
        assert!(display.contains("gemm"));
        assert!(display.contains("4.00x"));
    }

    // =====================================================================
    // Trend analysis
    // =====================================================================

    #[test]
    fn test_trend_stable() {
        let vals = vec![100.0, 100.0, 100.0, 100.0];
        let trend = TrendAnalysis::analyse(&vals, 0.05).unwrap();
        assert_eq!(trend.trend_direction, TrendDirection::Stable);
        assert!(!trend.regression_detected);
    }

    #[test]
    fn test_trend_improving() {
        // Durations decrease → improvement
        let vals = vec![100.0, 100.0, 50.0, 50.0];
        let trend = TrendAnalysis::analyse(&vals, 0.05).unwrap();
        assert_eq!(trend.trend_direction, TrendDirection::Improving);
        assert!(!trend.regression_detected);
    }

    #[test]
    fn test_trend_regressing() {
        // Durations increase → regression
        let vals = vec![100.0, 100.0, 200.0, 200.0];
        let trend = TrendAnalysis::analyse(&vals, 0.05).unwrap();
        assert_eq!(trend.trend_direction, TrendDirection::Regressing);
        assert!(trend.regression_detected);
    }

    #[test]
    fn test_trend_regression_below_threshold() {
        // 3% increase with 5% threshold → stable
        let vals = vec![100.0, 100.0, 103.0, 103.0];
        let trend = TrendAnalysis::analyse(&vals, 0.05).unwrap();
        assert_eq!(trend.trend_direction, TrendDirection::Stable);
        assert!(!trend.regression_detected);
    }

    #[test]
    fn test_trend_insufficient_data() {
        assert!(TrendAnalysis::analyse(&[100.0], 0.05).is_none());
    }

    #[test]
    fn test_trend_empty() {
        assert!(TrendAnalysis::analyse(&[], 0.05).is_none());
    }

    #[test]
    fn test_trend_nan_filtered() {
        let vals = vec![f64::NAN, 100.0, 100.0, f64::NAN, 200.0, 200.0];
        let trend = TrendAnalysis::analyse(&vals, 0.05).unwrap();
        assert!(trend.regression_detected);
    }

    #[test]
    fn test_trend_confidence_high_change() {
        let vals = vec![100.0, 100.0, 500.0, 500.0];
        let trend = TrendAnalysis::analyse(&vals, 0.05).unwrap();
        assert!((trend.confidence - 1.0).abs() < 1e-9); // clamped to 1.0
    }

    #[test]
    fn test_trend_display() {
        let vals = vec![100.0, 100.0, 110.0, 110.0];
        let trend = TrendAnalysis::analyse(&vals, 0.05).unwrap();
        let display = format!("{}", trend);
        assert!(display.contains("regressing"));
        assert!(display.contains("REGRESSION"));
    }

    #[test]
    fn test_trend_zero_baseline() {
        let vals = vec![0.0, 0.0, 100.0, 100.0];
        let trend = TrendAnalysis::analyse(&vals, 0.05).unwrap();
        // Cannot compute meaningful pct_change from zero baseline → stable fallback
        assert_eq!(trend.trend_direction, TrendDirection::Stable);
    }

    // =====================================================================
    // BenchmarkSuite
    // =====================================================================

    #[test]
    fn test_suite_duration_stats() {
        let suite = make_suite("test_kernel", &[10.0, 20.0, 30.0]);
        let stats = suite.duration_stats().unwrap();
        assert!((stats.mean - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_suite_empty_stats() {
        let suite = BenchmarkSuite::new("empty", "hw", "model");
        assert!(suite.duration_stats().is_none());
    }

    #[test]
    fn test_suite_gflops_stats() {
        let mut suite = BenchmarkSuite::new("test", "hw", "model");
        suite.add_result(BenchmarkResult::with_metrics("k", 100.0, 5.0, 10.0, 0.0));
        suite.add_result(BenchmarkResult::with_metrics("k", 100.0, 15.0, 20.0, 1.0));
        let gstats = suite.gflops_stats().unwrap();
        assert!((gstats.mean - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_suite_gflops_none_ignored() {
        let mut suite = BenchmarkSuite::new("test", "hw", "model");
        suite.add_result(BenchmarkResult::new("k", 100.0, 0.0)); // no gflops
        suite.add_result(BenchmarkResult::with_metrics("k", 100.0, 8.0, 10.0, 1.0));
        let gstats = suite.gflops_stats().unwrap();
        assert_eq!(gstats.count, 1);
    }

    // =====================================================================
    // Report formatting — Markdown
    // =====================================================================

    #[test]
    fn test_markdown_contains_header() {
        let data = DashboardData::new("2025-01-01T00:00:00Z");
        let md = ReportFormatter::to_markdown(&data);
        assert!(md.contains("# Benchmark Dashboard"));
        assert!(md.contains("2025-01-01T00:00:00Z"));
    }

    #[test]
    fn test_markdown_suite_table() {
        let mut data = DashboardData::new("now");
        data.add_suite(make_suite("gemm", &[10.0, 20.0, 30.0]));
        let md = ReportFormatter::to_markdown(&data);
        assert!(md.contains("### gemm"));
        assert!(md.contains("Mean"));
        assert!(md.contains("Intel Arc A770 16GB"));
    }

    #[test]
    fn test_markdown_comparison_table() {
        let cpu = StatsSummary::compute(&[100.0, 100.0]).unwrap();
        let gpu = StatsSummary::compute(&[25.0, 25.0]).unwrap();
        let mut data = DashboardData::new("now");
        data.add_comparison(ComparisonReport::new("matmul", cpu, gpu, 1.0));
        let md = ReportFormatter::to_markdown(&data);
        assert!(md.contains("CPU vs GPU"));
        assert!(md.contains("matmul"));
        assert!(md.contains("4.00x"));
    }

    // =====================================================================
    // Report formatting — JSON
    // =====================================================================

    #[test]
    fn test_json_structure() {
        let mut data = DashboardData::new("2025-01-01T00:00:00Z");
        data.add_suite(make_suite("gemm", &[10.0, 20.0]));
        let json = ReportFormatter::to_json(&data);
        assert!(json.contains("\"generated_at\""));
        assert!(json.contains("\"suites\""));
        assert!(json.contains("\"comparisons\""));
        assert!(json.contains("\"trends\""));
        assert!(json.contains("\"gemm\""));
    }

    #[test]
    fn test_json_stats_present() {
        let mut data = DashboardData::new("now");
        data.add_suite(make_suite("k", &[10.0, 20.0, 30.0]));
        let json = ReportFormatter::to_json(&data);
        assert!(json.contains("\"duration_stats\""));
        assert!(json.contains("\"mean\""));
        assert!(json.contains("\"p95\""));
    }

    #[test]
    fn test_json_empty_dashboard() {
        let data = DashboardData::new("now");
        let json = ReportFormatter::to_json(&data);
        assert!(json.contains("\"suites\": ["));
        assert!(json.contains("\"comparisons\": ["));
        assert!(json.contains("\"trends\": ["));
    }

    // =====================================================================
    // Property-style tests: stats invariants
    // =====================================================================

    #[test]
    fn test_property_min_le_median_le_max() {
        for seed in 0..20 {
            let vals: Vec<f64> = (0..50).map(|i| ((i * 7 + seed) % 100) as f64).collect();
            let stats = StatsSummary::compute(&vals).unwrap();
            assert!(stats.min <= stats.median, "min <= median failed for seed {seed}");
            assert!(stats.median <= stats.max, "median <= max failed for seed {seed}");
        }
    }

    #[test]
    fn test_property_min_le_mean_le_max() {
        for seed in 0..20 {
            let vals: Vec<f64> = (0..50).map(|i| ((i * 13 + seed) % 200) as f64).collect();
            let stats = StatsSummary::compute(&vals).unwrap();
            assert!(stats.min <= stats.mean, "min <= mean failed for seed {seed}");
            assert!(stats.mean <= stats.max, "mean <= max failed for seed {seed}");
        }
    }

    #[test]
    fn test_property_percentile_ordering() {
        let vals: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        let stats = StatsSummary::compute(&vals).unwrap();
        assert!(stats.min <= stats.median);
        assert!(stats.median <= stats.p95);
        assert!(stats.p95 <= stats.p99);
        assert!(stats.p99 <= stats.max);
    }

    #[test]
    fn test_property_std_dev_non_negative() {
        for multiplier in [1, 7, 13, 37] {
            let vals: Vec<f64> = (0..30).map(|i| (i * multiplier % 50) as f64).collect();
            let stats = StatsSummary::compute(&vals).unwrap();
            assert!(stats.std_dev >= 0.0, "std_dev negative for multiplier {multiplier}");
        }
    }

    // =====================================================================
    // Large result set performance
    // =====================================================================

    #[test]
    fn test_large_result_set() {
        let vals: Vec<f64> = (0..10_000).map(|i| (i as f64).sin().abs() * 1000.0).collect();
        let stats = StatsSummary::compute(&vals).unwrap();
        assert_eq!(stats.count, 10_000);
        assert!(stats.min >= 0.0);
        assert!(stats.p99 >= stats.p95);
    }

    // =====================================================================
    // Dashboard builder
    // =====================================================================

    #[test]
    fn test_build_dashboard_matching_suites() {
        let cpu = vec![make_suite("matmul", &[100.0, 100.0])];
        let gpu = vec![make_suite("matmul", &[25.0, 25.0])];
        let dash = ReportFormatter::build_dashboard(cpu, gpu, "now");
        assert_eq!(dash.suites.len(), 2);
        assert_eq!(dash.comparisons.len(), 1);
        assert!((dash.comparisons[0].speedup - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_dashboard_no_gpu() {
        let cpu = vec![make_suite("matmul", &[100.0])];
        let dash = ReportFormatter::build_dashboard(cpu, vec![], "now");
        assert_eq!(dash.suites.len(), 1);
        assert!(dash.comparisons.is_empty());
    }

    #[test]
    fn test_build_dashboard_extra_gpu_suite() {
        let cpu = vec![make_suite("gemm", &[100.0])];
        let gpu = vec![make_suite("gemm", &[25.0]), make_suite("attention", &[50.0])];
        let dash = ReportFormatter::build_dashboard(cpu, gpu, "now");
        // gemm(cpu) + gemm(gpu) + attention(gpu)
        assert_eq!(dash.suites.len(), 3);
        assert_eq!(dash.comparisons.len(), 1);
    }

    // =====================================================================
    // Stats oneliner
    // =====================================================================

    #[test]
    fn test_stats_oneliner() {
        let stats = StatsSummary::compute(&[10.0, 20.0, 30.0]).unwrap();
        let line = ReportFormatter::stats_oneliner("gemm", &stats);
        assert!(line.starts_with("gemm:"));
        assert!(line.contains("mean="));
        assert!(line.contains("n=3"));
    }

    // =====================================================================
    // Display impls
    // =====================================================================

    #[test]
    fn test_stats_display() {
        let stats = StatsSummary::compute(&[10.0, 20.0, 30.0]).unwrap();
        let display = format!("{}", stats);
        assert!(display.contains("n=3"));
        assert!(display.contains("σ="));
    }

    #[test]
    fn test_trend_direction_display() {
        assert_eq!(format!("{}", TrendDirection::Improving), "improving");
        assert_eq!(format!("{}", TrendDirection::Regressing), "regressing");
        assert_eq!(format!("{}", TrendDirection::Stable), "stable");
    }
}
