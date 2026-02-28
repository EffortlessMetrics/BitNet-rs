//! Benchmark result analysis and regression detection.
//!
//! Parses Criterion/bencher output, compares runs, and flags regressions
//! that exceed a configurable threshold.

use std::fmt;

// ── Result types ──────────────────────────────────────────────────────────

/// Unit for throughput measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThroughputUnit {
    /// Bytes processed per second.
    BytesPerSec,
    /// Elements processed per second.
    ElementsPerSec,
    /// Tokens generated per second.
    TokensPerSec,
}

impl fmt::Display for ThroughputUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BytesPerSec => write!(f, "bytes/s"),
            Self::ElementsPerSec => write!(f, "elem/s"),
            Self::TokensPerSec => write!(f, "tok/s"),
        }
    }
}

/// Throughput measurement for a benchmark.
#[derive(Debug, Clone)]
pub struct Throughput {
    pub value: f64,
    pub unit: ThroughputUnit,
}

impl fmt::Display for Throughput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} {}", self.value, self.unit)
    }
}

/// A single benchmark measurement.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: u64,
    pub time_ns: u64,
    pub throughput: Option<Throughput>,
}

impl BenchmarkResult {
    /// Average time per iteration in nanoseconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn time_per_iter_ns(&self) -> f64 {
        if self.iterations == 0 {
            return 0.0;
        }
        self.time_ns as f64 / self.iterations as f64
    }

    /// Returns throughput if available, or computes it from iterations
    /// and time as elements-per-second.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput(&self) -> Option<Throughput> {
        if let Some(ref tp) = self.throughput {
            return Some(tp.clone());
        }
        if self.time_ns == 0 {
            return None;
        }
        let secs = self.time_ns as f64 / 1_000_000_000.0;
        Some(Throughput {
            value: self.iterations as f64 / secs,
            unit: ThroughputUnit::ElementsPerSec,
        })
    }
}

// ── Comparison ────────────────────────────────────────────────────────────

/// Side-by-side comparison of two benchmark measurements.
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub name: String,
    pub baseline: BenchmarkResult,
    pub current: BenchmarkResult,
    /// Percentage change (positive = slower, negative = faster).
    pub change_percent: f64,
    /// `true` when the slowdown exceeds the threshold.
    pub regression: bool,
    /// Threshold that was used for detection.
    pub threshold: f64,
}

impl BenchmarkComparison {
    /// Build a comparison from a baseline and current result.
    ///
    /// `threshold` is the maximum acceptable slowdown as a fraction
    /// (e.g. `0.05` for 5 %).
    pub fn from_pair(baseline: BenchmarkResult, current: BenchmarkResult, threshold: f64) -> Self {
        let base_ns = baseline.time_per_iter_ns();
        let curr_ns = current.time_per_iter_ns();
        let change_percent =
            if base_ns == 0.0 { 0.0 } else { (curr_ns - base_ns) / base_ns * 100.0 };
        let regression = change_percent > threshold * 100.0;
        Self {
            name: baseline.name.clone(),
            baseline,
            current,
            change_percent,
            regression,
            threshold,
        }
    }
}

impl fmt::Display for BenchmarkComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction = if self.change_percent > 0.0 { "slower" } else { "faster" };
        let marker = if self.regression { " ⚠ REGRESSION" } else { "" };
        write!(f, "{}: {:.2}% {}{marker}", self.name, self.change_percent.abs(), direction,)
    }
}

// ── Suite ─────────────────────────────────────────────────────────────────

/// A collection of benchmark results from a single run.
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    pub results: Vec<BenchmarkResult>,
    pub timestamp: u64,
    pub git_sha: String,
}

impl BenchmarkSuite {
    /// Human-readable summary of all results.
    pub fn summary(&self) -> String {
        if self.results.is_empty() {
            return format!(
                "Benchmark suite (sha: {}, ts: {}): no results",
                self.git_sha, self.timestamp,
            );
        }
        let mut lines = vec![format!(
            "Benchmark suite (sha: {}, ts: {}, {} results):",
            self.git_sha,
            self.timestamp,
            self.results.len(),
        )];
        for r in &self.results {
            lines.push(format!(
                "  {}: {:.2} ns/iter ({} iters, {} ns total)",
                r.name,
                r.time_per_iter_ns(),
                r.iterations,
                r.time_ns,
            ));
        }
        lines.join("\n")
    }
}

// ── Regression detector ───────────────────────────────────────────────────

/// Detects performance regressions between benchmark suites.
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    /// Maximum acceptable slowdown as a fraction (e.g. `0.05` for 5 %).
    pub threshold_percent: f64,
    /// Historical suites for trend analysis.
    pub history: Vec<BenchmarkSuite>,
}

impl RegressionDetector {
    /// Create a detector with the given threshold.
    pub const fn new(threshold_percent: f64) -> Self {
        Self { threshold_percent, history: Vec::new() }
    }

    /// Compare a baseline suite against a current suite.
    ///
    /// Only benchmarks present in **both** suites are compared.
    pub fn detect(
        &self,
        baseline: &BenchmarkSuite,
        current: &BenchmarkSuite,
    ) -> Vec<BenchmarkComparison> {
        let mut comparisons = Vec::new();
        for base_result in &baseline.results {
            if let Some(curr_result) = current.results.iter().find(|r| r.name == base_result.name) {
                comparisons.push(BenchmarkComparison::from_pair(
                    base_result.clone(),
                    curr_result.clone(),
                    self.threshold_percent,
                ));
            }
        }
        comparisons
    }

    /// Push a suite into the history.
    pub fn add_to_history(&mut self, suite: BenchmarkSuite) {
        self.history.push(suite);
    }

    /// Return the average time-per-iter for a given benchmark across
    /// the historical suites. Returns `None` if the benchmark was never
    /// recorded.
    pub fn historical_average_ns(&self, benchmark_name: &str) -> Option<f64> {
        let values: Vec<f64> = self
            .history
            .iter()
            .flat_map(|s| &s.results)
            .filter(|r| r.name == benchmark_name)
            .map(BenchmarkResult::time_per_iter_ns)
            .collect();
        if values.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }

    /// Detect regressions in `current` relative to the historical
    /// average rather than a single baseline.
    pub fn detect_from_history(&self, current: &BenchmarkSuite) -> Vec<BenchmarkComparison> {
        let mut comparisons = Vec::new();
        for result in &current.results {
            if let Some(avg_ns) = self.historical_average_ns(&result.name) {
                let baseline = BenchmarkResult {
                    name: result.name.clone(),
                    iterations: 1,
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    time_ns: avg_ns as u64,
                    throughput: None,
                };
                comparisons.push(BenchmarkComparison::from_pair(
                    baseline,
                    result.clone(),
                    self.threshold_percent,
                ));
            }
        }
        comparisons
    }
}

// ── Bencher output parser ─────────────────────────────────────────────────

/// Parse a single bencher-format line.
///
/// Expected format:
/// ```text
/// test bench_name ... bench:      1,234 ns/iter (+/- 56)
/// ```
pub fn parse_bencher_line(line: &str) -> Option<BenchmarkResult> {
    let line = line.trim();
    if !line.starts_with("test ") || !line.contains("bench:") {
        return None;
    }

    // Extract name: between "test " and " ... bench:"
    let name_end = line.find(" ... bench:")?;
    let name = line[5..name_end].trim().to_string();

    // Extract ns value: after "bench:" and before "ns/iter"
    let bench_start = line.find("bench:")? + 6;
    let ns_end = line.find("ns/iter")?;
    let ns_str: String = line[bench_start..ns_end].chars().filter(char::is_ascii_digit).collect();
    let time_ns: u64 = ns_str.parse().ok()?;

    Some(BenchmarkResult { name, iterations: 1, time_ns, throughput: None })
}

/// Parse multiple lines of bencher output into a [`BenchmarkSuite`].
pub fn parse_bencher_output(output: &str, git_sha: &str, timestamp: u64) -> BenchmarkSuite {
    let results: Vec<BenchmarkResult> = output.lines().filter_map(parse_bencher_line).collect();
    BenchmarkSuite { results, timestamp, git_sha: git_sha.to_string() }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────

    fn result(name: &str, iters: u64, time_ns: u64) -> BenchmarkResult {
        BenchmarkResult { name: name.to_string(), iterations: iters, time_ns, throughput: None }
    }

    fn result_with_tp(
        name: &str,
        iters: u64,
        time_ns: u64,
        tp_value: f64,
        tp_unit: ThroughputUnit,
    ) -> BenchmarkResult {
        BenchmarkResult {
            name: name.to_string(),
            iterations: iters,
            time_ns,
            throughput: Some(Throughput { value: tp_value, unit: tp_unit }),
        }
    }

    fn suite(results: Vec<BenchmarkResult>) -> BenchmarkSuite {
        BenchmarkSuite { results, timestamp: 1_700_000_000, git_sha: "abc123".to_string() }
    }

    // ── BenchmarkResult ──────────────────────────────────────────────

    #[test]
    fn time_per_iter_basic() {
        let r = result("foo", 100, 1_000);
        assert!((r.time_per_iter_ns() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn time_per_iter_zero_iterations() {
        let r = result("foo", 0, 1_000);
        assert!((r.time_per_iter_ns() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn time_per_iter_single_iteration() {
        let r = result("foo", 1, 500);
        assert!((r.time_per_iter_ns() - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn throughput_computed_from_iters() {
        let r = result("foo", 1_000_000, 1_000_000_000);
        let tp = r.throughput().unwrap();
        assert_eq!(tp.unit, ThroughputUnit::ElementsPerSec);
        assert!((tp.value - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn throughput_explicit() {
        let r = result_with_tp("foo", 10, 100, 42.5, ThroughputUnit::TokensPerSec);
        let tp = r.throughput().unwrap();
        assert_eq!(tp.unit, ThroughputUnit::TokensPerSec);
        assert!((tp.value - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn throughput_none_when_zero_time() {
        let r = result("foo", 0, 0);
        assert!(r.throughput().is_none());
    }

    #[test]
    fn throughput_display_bytes() {
        let tp = Throughput { value: 1024.0, unit: ThroughputUnit::BytesPerSec };
        assert_eq!(format!("{tp}"), "1024.00 bytes/s");
    }

    #[test]
    fn throughput_display_elements() {
        let tp = Throughput { value: 500.0, unit: ThroughputUnit::ElementsPerSec };
        assert_eq!(format!("{tp}"), "500.00 elem/s");
    }

    #[test]
    fn throughput_display_tokens() {
        let tp = Throughput { value: 12.34, unit: ThroughputUnit::TokensPerSec };
        assert_eq!(format!("{tp}"), "12.34 tok/s");
    }

    #[test]
    fn throughput_unit_display() {
        assert_eq!(format!("{}", ThroughputUnit::BytesPerSec), "bytes/s");
        assert_eq!(format!("{}", ThroughputUnit::ElementsPerSec), "elem/s");
        assert_eq!(format!("{}", ThroughputUnit::TokensPerSec), "tok/s");
    }

    // ── BenchmarkComparison ──────────────────────────────────────────

    #[test]
    fn comparison_faster_is_not_regression() {
        let base = result("bench_a", 1, 1_000);
        let curr = result("bench_a", 1, 800);
        let cmp = BenchmarkComparison::from_pair(base, curr, 0.05);
        assert!(!cmp.regression);
        assert!(cmp.change_percent < 0.0);
    }

    #[test]
    fn comparison_10pct_slower_is_regression_at_5pct_threshold() {
        let base = result("bench_a", 1, 1_000);
        let curr = result("bench_a", 1, 1_100);
        let cmp = BenchmarkComparison::from_pair(base, curr, 0.05);
        assert!(cmp.regression);
        assert!((cmp.change_percent - 10.0).abs() < 0.1);
    }

    #[test]
    fn comparison_4pct_slower_is_not_regression_at_5pct_threshold() {
        let base = result("bench_a", 1, 1_000);
        let curr = result("bench_a", 1, 1_040);
        let cmp = BenchmarkComparison::from_pair(base, curr, 0.05);
        assert!(!cmp.regression);
    }

    #[test]
    fn comparison_exact_threshold_is_not_regression() {
        let base = result("bench_a", 1, 1_000);
        let curr = result("bench_a", 1, 1_050);
        let cmp = BenchmarkComparison::from_pair(base, curr, 0.05);
        assert!(!cmp.regression);
    }

    #[test]
    fn comparison_just_over_threshold_is_regression() {
        let base = result("bench_a", 1, 1_000);
        let curr = result("bench_a", 1, 1_051);
        let cmp = BenchmarkComparison::from_pair(base, curr, 0.05);
        assert!(cmp.regression);
    }

    #[test]
    fn comparison_zero_baseline_no_panic() {
        let base = result("bench_a", 0, 0);
        let curr = result("bench_a", 1, 100);
        let cmp = BenchmarkComparison::from_pair(base, curr, 0.05);
        assert!(!cmp.regression);
        assert!((cmp.change_percent - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn comparison_identical_is_not_regression() {
        let base = result("bench_a", 1, 1_000);
        let curr = result("bench_a", 1, 1_000);
        let cmp = BenchmarkComparison::from_pair(base, curr, 0.05);
        assert!(!cmp.regression);
        assert!((cmp.change_percent - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn comparison_display_regression() {
        let base = result("bench_a", 1, 1_000);
        let curr = result("bench_a", 1, 1_200);
        let cmp = BenchmarkComparison::from_pair(base, curr, 0.05);
        let s = format!("{cmp}");
        assert!(s.contains("REGRESSION"));
        assert!(s.contains("slower"));
    }

    #[test]
    fn comparison_display_improvement() {
        let base = result("bench_a", 1, 1_000);
        let curr = result("bench_a", 1, 900);
        let cmp = BenchmarkComparison::from_pair(base, curr, 0.05);
        let s = format!("{cmp}");
        assert!(!s.contains("REGRESSION"));
        assert!(s.contains("faster"));
    }

    // ── BenchmarkSuite ───────────────────────────────────────────────

    #[test]
    fn suite_summary_with_results() {
        let s = suite(vec![result("alpha", 100, 10_000), result("beta", 50, 5_000)]);
        let summary = s.summary();
        assert!(summary.contains("2 results"));
        assert!(summary.contains("alpha"));
        assert!(summary.contains("beta"));
        assert!(summary.contains("abc123"));
    }

    #[test]
    fn suite_summary_empty() {
        let s = suite(vec![]);
        let summary = s.summary();
        assert!(summary.contains("no results"));
        assert!(summary.contains("abc123"));
    }

    #[test]
    fn suite_summary_single_result() {
        let s = suite(vec![result("only", 1, 42)]);
        let summary = s.summary();
        assert!(summary.contains("1 results"));
        assert!(summary.contains("only"));
    }

    // ── RegressionDetector ───────────────────────────────────────────

    #[test]
    fn detector_no_overlap_yields_empty() {
        let detector = RegressionDetector::new(0.05);
        let base = suite(vec![result("alpha", 1, 100)]);
        let curr = suite(vec![result("beta", 1, 200)]);
        let comparisons = detector.detect(&base, &curr);
        assert!(comparisons.is_empty());
    }

    #[test]
    fn detector_finds_regression() {
        let detector = RegressionDetector::new(0.05);
        let base = suite(vec![result("bench", 1, 1_000)]);
        let curr = suite(vec![result("bench", 1, 1_200)]);
        let comparisons = detector.detect(&base, &curr);
        assert_eq!(comparisons.len(), 1);
        assert!(comparisons[0].regression);
    }

    #[test]
    fn detector_finds_no_regression_when_faster() {
        let detector = RegressionDetector::new(0.05);
        let base = suite(vec![result("bench", 1, 1_000)]);
        let curr = suite(vec![result("bench", 1, 900)]);
        let comparisons = detector.detect(&base, &curr);
        assert_eq!(comparisons.len(), 1);
        assert!(!comparisons[0].regression);
    }

    #[test]
    fn detector_multiple_benchmarks() {
        let detector = RegressionDetector::new(0.05);
        let base = suite(vec![result("a", 1, 1_000), result("b", 1, 2_000), result("c", 1, 500)]);
        let curr = suite(vec![result("a", 1, 1_200), result("b", 1, 1_900), result("c", 1, 600)]);
        let comparisons = detector.detect(&base, &curr);
        assert_eq!(comparisons.len(), 3);
        // "a" regressed (20%), "b" improved, "c" regressed (20%)
        assert!(comparisons[0].regression);
        assert!(!comparisons[1].regression);
        assert!(comparisons[2].regression);
    }

    #[test]
    fn detector_empty_suites() {
        let detector = RegressionDetector::new(0.05);
        let base = suite(vec![]);
        let curr = suite(vec![]);
        assert!(detector.detect(&base, &curr).is_empty());
    }

    #[test]
    fn detector_empty_baseline() {
        let detector = RegressionDetector::new(0.05);
        let base = suite(vec![]);
        let curr = suite(vec![result("x", 1, 100)]);
        assert!(detector.detect(&base, &curr).is_empty());
    }

    #[test]
    fn detector_empty_current() {
        let detector = RegressionDetector::new(0.05);
        let base = suite(vec![result("x", 1, 100)]);
        let curr = suite(vec![]);
        assert!(detector.detect(&base, &curr).is_empty());
    }

    #[test]
    fn detector_custom_threshold() {
        let detector = RegressionDetector::new(0.20);
        let base = suite(vec![result("bench", 1, 1_000)]);
        let curr = suite(vec![result("bench", 1, 1_150)]);
        let comparisons = detector.detect(&base, &curr);
        assert!(!comparisons[0].regression); // 15% < 20% threshold
    }

    #[test]
    fn detector_strict_threshold() {
        let detector = RegressionDetector::new(0.01);
        let base = suite(vec![result("bench", 1, 1_000)]);
        let curr = suite(vec![result("bench", 1, 1_020)]);
        let comparisons = detector.detect(&base, &curr);
        assert!(comparisons[0].regression); // 2% > 1% threshold
    }

    // ── Historical trend ─────────────────────────────────────────────

    #[test]
    fn historical_average_empty() {
        let detector = RegressionDetector::new(0.05);
        assert!(detector.historical_average_ns("any").is_none());
    }

    #[test]
    fn historical_average_single() {
        let mut detector = RegressionDetector::new(0.05);
        detector.add_to_history(suite(vec![result("bench", 1, 100)]));
        let avg = detector.historical_average_ns("bench").unwrap();
        assert!((avg - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn historical_average_multiple() {
        let mut detector = RegressionDetector::new(0.05);
        detector.add_to_history(suite(vec![result("bench", 1, 100)]));
        detector.add_to_history(suite(vec![result("bench", 1, 200)]));
        detector.add_to_history(suite(vec![result("bench", 1, 300)]));
        let avg = detector.historical_average_ns("bench").unwrap();
        assert!((avg - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn historical_average_missing_benchmark() {
        let mut detector = RegressionDetector::new(0.05);
        detector.add_to_history(suite(vec![result("alpha", 1, 100)]));
        assert!(detector.historical_average_ns("beta").is_none());
    }

    #[test]
    fn detect_from_history_regression() {
        let mut detector = RegressionDetector::new(0.05);
        detector.add_to_history(suite(vec![result("bench", 1, 100)]));
        detector.add_to_history(suite(vec![result("bench", 1, 100)]));
        let current = suite(vec![result("bench", 1, 120)]);
        let comparisons = detector.detect_from_history(&current);
        assert_eq!(comparisons.len(), 1);
        assert!(comparisons[0].regression);
    }

    #[test]
    fn detect_from_history_no_history() {
        let detector = RegressionDetector::new(0.05);
        let current = suite(vec![result("bench", 1, 100)]);
        assert!(detector.detect_from_history(&current).is_empty());
    }

    // ── Bencher output parsing ───────────────────────────────────────

    #[test]
    fn parse_bencher_simple_line() {
        let line = "test bench_add ... bench:         123 ns/iter (+/- 10)";
        let r = parse_bencher_line(line).unwrap();
        assert_eq!(r.name, "bench_add");
        assert_eq!(r.time_ns, 123);
    }

    #[test]
    fn parse_bencher_with_commas() {
        let line = "test bench_sort ... bench:       1,234 ns/iter (+/- 56)";
        let r = parse_bencher_line(line).unwrap();
        assert_eq!(r.name, "bench_sort");
        assert_eq!(r.time_ns, 1234);
    }

    #[test]
    fn parse_bencher_large_number() {
        let line = "test bench_big ... bench:   1,234,567 ns/iter (+/- 100)";
        let r = parse_bencher_line(line).unwrap();
        assert_eq!(r.name, "bench_big");
        assert_eq!(r.time_ns, 1_234_567);
    }

    #[test]
    fn parse_bencher_non_bench_line() {
        assert!(parse_bencher_line("running 5 tests").is_none());
        assert!(parse_bencher_line("").is_none());
        assert!(parse_bencher_line("test result: ok").is_none());
    }

    #[test]
    fn parse_bencher_multiline_output() {
        let output = "\
running 3 tests
test bench_a ... bench:         100 ns/iter (+/- 5)
test bench_b ... bench:       2,000 ns/iter (+/- 50)
test bench_c ... bench:      30,000 ns/iter (+/- 200)

test result: ok. 0 passed; 0 failed; 0 ignored; 3 measured; 0 filtered out
";
        let s = parse_bencher_output(output, "def456", 1_700_000_000);
        assert_eq!(s.results.len(), 3);
        assert_eq!(s.results[0].name, "bench_a");
        assert_eq!(s.results[0].time_ns, 100);
        assert_eq!(s.results[1].name, "bench_b");
        assert_eq!(s.results[1].time_ns, 2000);
        assert_eq!(s.results[2].name, "bench_c");
        assert_eq!(s.results[2].time_ns, 30_000);
        assert_eq!(s.git_sha, "def456");
    }

    #[test]
    fn parse_bencher_empty_output() {
        let s = parse_bencher_output("", "abc", 0);
        assert!(s.results.is_empty());
    }

    #[test]
    fn parse_bencher_whitespace_in_name() {
        let line = "test my::nested::bench ... bench:         42 ns/iter (+/- 1)";
        let r = parse_bencher_line(line).unwrap();
        assert_eq!(r.name, "my::nested::bench");
        assert_eq!(r.time_ns, 42);
    }

    #[test]
    fn parse_bencher_preserves_order() {
        let output = "\
test z_bench ... bench:         10 ns/iter (+/- 1)
test a_bench ... bench:         20 ns/iter (+/- 1)
";
        let s = parse_bencher_output(output, "sha", 0);
        assert_eq!(s.results[0].name, "z_bench");
        assert_eq!(s.results[1].name, "a_bench");
    }

    // ── Round-trip: parse then detect ────────────────────────────────

    #[test]
    fn round_trip_parse_and_detect() {
        let baseline_output = "\
test bench_x ... bench:       1,000 ns/iter (+/- 10)
test bench_y ... bench:       2,000 ns/iter (+/- 20)
";
        let current_output = "\
test bench_x ... bench:       1,200 ns/iter (+/- 10)
test bench_y ... bench:       1,900 ns/iter (+/- 20)
";
        let base = parse_bencher_output(baseline_output, "base", 1_000_000);
        let curr = parse_bencher_output(current_output, "curr", 2_000_000);
        let detector = RegressionDetector::new(0.05);
        let comparisons = detector.detect(&base, &curr);
        assert_eq!(comparisons.len(), 2);
        // bench_x: 1000 → 1200 = 20% slower → regression
        assert!(comparisons[0].regression);
        // bench_y: 2000 → 1900 = 5% faster → not a regression
        assert!(!comparisons[1].regression);
    }
}
