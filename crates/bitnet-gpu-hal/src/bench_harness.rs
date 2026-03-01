//! Benchmark harness for GPU kernel performance measurement.
//!
//! Provides [`BenchmarkHarness`] for orchestrating GPU kernel benchmarks with
//! statistical timing, throughput metrics, regression detection, multi-format
//! reporting, and A/B comparison.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

// ── Configuration ────────────────────────────────────────────────────────

/// Statistical and iteration parameters for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations (results discarded).
    pub warmup_iterations: u32,
    /// Number of measurement iterations.
    pub measurement_iterations: u32,
    /// Statistical confidence level (0.0–1.0, e.g. 0.95 for 95%).
    pub confidence_level: f64,
    /// Minimum duration per benchmark (guards against too-fast kernels).
    pub min_duration: Duration,
    /// Whether to discard statistical outliers.
    pub outlier_removal: bool,
    /// Outlier threshold in standard deviations.
    pub outlier_sigma: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            confidence_level: 0.95,
            min_duration: Duration::from_millis(100),
            outlier_removal: true,
            outlier_sigma: 3.0,
        }
    }
}

impl BenchmarkConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), String> {
        if self.measurement_iterations == 0 {
            return Err("measurement_iterations must be > 0".into());
        }
        if !(0.0..=1.0).contains(&self.confidence_level) {
            return Err("confidence_level must be in [0.0, 1.0]".into());
        }
        if self.outlier_sigma <= 0.0 {
            return Err("outlier_sigma must be > 0".into());
        }
        Ok(())
    }
}

// ── Timing Result ────────────────────────────────────────────────────────

/// Statistical summary of benchmark timing measurements (all in nanoseconds).
#[derive(Debug, Clone)]
pub struct TimingResult {
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: f64,
    pub median_ns: u64,
    pub std_dev_ns: f64,
    pub p99_ns: u64,
    /// Raw sample values (sorted).
    pub samples: Vec<u64>,
}

impl TimingResult {
    /// Compute statistics from raw nanosecond samples.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn from_samples(mut samples: Vec<u64>) -> Self {
        assert!(!samples.is_empty(), "need at least one sample");
        samples.sort_unstable();
        let len = samples.len();
        let lo = samples[0];
        let hi = samples[len - 1];
        let sum: u128 = samples.iter().map(|&s| u128::from(s)).sum();
        let avg = sum as f64 / len as f64;
        let mid = if len.is_multiple_of(2) {
            samples[len / 2 - 1].midpoint(samples[len / 2])
        } else {
            samples[len / 2]
        };
        let variance: f64 = samples
            .iter()
            .map(|&s| {
                let diff = s as f64 - avg;
                diff.mul_add(diff, 0.0)
            })
            .sum::<f64>()
            / len as f64;
        let dev = variance.sqrt();
        let p99_idx = ((len as f64) * 0.99).ceil() as usize;
        let p99 = samples[p99_idx.min(len) - 1];
        Self {
            min_ns: lo,
            max_ns: hi,
            mean_ns: avg,
            median_ns: mid,
            std_dev_ns: dev,
            p99_ns: p99,
            samples,
        }
    }

    /// Remove outliers beyond `sigma` standard deviations from the mean.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn remove_outliers(&self, sigma: f64) -> Self {
        let lower = (-sigma).mul_add(self.std_dev_ns, self.mean_ns);
        let upper = sigma.mul_add(self.std_dev_ns, self.mean_ns);
        let filtered: Vec<u64> = self
            .samples
            .iter()
            .copied()
            .filter(|&s| (s as f64) >= lower && (s as f64) <= upper)
            .collect();
        if filtered.is_empty() {
            return self.clone();
        }
        Self::from_samples(filtered)
    }

    /// Coefficient of variation (`std_dev` / mean).
    pub fn cv(&self) -> f64 {
        if self.mean_ns.abs() < f64::EPSILON {
            return 0.0;
        }
        self.std_dev_ns / self.mean_ns
    }
}

// ── Throughput Metric ────────────────────────────────────────────────────

/// Throughput measurements for a benchmark.
#[derive(Debug, Clone)]
pub struct ThroughputMetric {
    /// Tokens processed per second.
    pub tokens_per_sec: f64,
    /// Floating-point operations per second.
    pub flops: f64,
    /// Memory bandwidth in GB/s.
    pub bandwidth_gbs: f64,
}

impl ThroughputMetric {
    /// Compute throughput given operation counts and elapsed time.
    #[allow(clippy::cast_precision_loss, clippy::similar_names)]
    pub fn compute(
        token_count: u64,
        flop_count: u64,
        bytes_transferred: u64,
        elapsed_ns: f64,
    ) -> Self {
        let elapsed_s = elapsed_ns / 1e9;
        Self {
            tokens_per_sec: if elapsed_s > 0.0 { token_count as f64 / elapsed_s } else { 0.0 },
            flops: if elapsed_s > 0.0 { flop_count as f64 / elapsed_s } else { 0.0 },
            bandwidth_gbs: if elapsed_s > 0.0 {
                bytes_transferred as f64 / elapsed_s / 1e9
            } else {
                0.0
            },
        }
    }

    /// Scale throughput by a multiplier (e.g. for batch size).
    #[must_use]
    pub const fn scale(&self, factor: f64) -> Self {
        Self {
            tokens_per_sec: self.tokens_per_sec * factor,
            flops: self.flops * factor,
            bandwidth_gbs: self.bandwidth_gbs * factor,
        }
    }
}

// ── Benchmark Case ───────────────────────────────────────────────────────

/// Describes a single benchmark to execute.
#[derive(Clone)]
pub struct BenchmarkCase {
    /// Human-readable name.
    pub name: String,
    /// Category tag for grouping (e.g. "matmul", "attention").
    pub category: String,
    /// Setup function called once before warmup.
    pub setup: Option<fn() -> HashMap<String, String>>,
    /// The function under measurement. Receives context from setup.
    pub run: fn(&HashMap<String, String>),
    /// Teardown function called after all iterations.
    pub teardown: Option<fn(&HashMap<String, String>)>,
    /// Expected operation counts for throughput calculation.
    pub workload: WorkloadSpec,
}

impl std::fmt::Debug for BenchmarkCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BenchmarkCase")
            .field("name", &self.name)
            .field("category", &self.category)
            .field("workload", &self.workload)
            .finish_non_exhaustive()
    }
}

impl BenchmarkCase {
    /// Create a minimal benchmark case.
    pub fn new(name: impl Into<String>, run: fn(&HashMap<String, String>)) -> Self {
        Self {
            name: name.into(),
            category: "default".into(),
            setup: None,
            run,
            teardown: None,
            workload: WorkloadSpec::default(),
        }
    }

    /// Builder: set the category.
    #[must_use]
    pub fn with_category(mut self, cat: impl Into<String>) -> Self {
        self.category = cat.into();
        self
    }

    /// Builder: set the setup function.
    #[must_use]
    pub fn with_setup(mut self, f: fn() -> HashMap<String, String>) -> Self {
        self.setup = Some(f);
        self
    }

    /// Builder: set the teardown function.
    #[must_use]
    pub fn with_teardown(mut self, f: fn(&HashMap<String, String>)) -> Self {
        self.teardown = Some(f);
        self
    }

    /// Builder: set the workload spec.
    #[must_use]
    pub const fn with_workload(mut self, w: WorkloadSpec) -> Self {
        self.workload = w;
        self
    }
}

/// Describes expected operation counts for throughput calculation.
#[derive(Debug, Clone, Default)]
pub struct WorkloadSpec {
    pub token_count: u64,
    pub flop_count: u64,
    pub bytes_transferred: u64,
}

// ── Benchmark Result ─────────────────────────────────────────────────────

/// Full result for one benchmark case.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub category: String,
    pub timing: TimingResult,
    pub throughput: Option<ThroughputMetric>,
    pub timestamp: Duration,
}

// ── Benchmark Suite ──────────────────────────────────────────────────────

/// A named collection of benchmark cases.
#[derive(Debug)]
pub struct BenchmarkSuite {
    pub name: String,
    pub description: String,
    pub cases: Vec<BenchmarkCase>,
    pub config: BenchmarkConfig,
}

impl BenchmarkSuite {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            cases: Vec::new(),
            config: BenchmarkConfig::default(),
        }
    }

    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    #[must_use]
    pub const fn with_config(mut self, config: BenchmarkConfig) -> Self {
        self.config = config;
        self
    }

    pub fn add_case(&mut self, case: BenchmarkCase) {
        self.cases.push(case);
    }

    /// Filter cases by category.
    pub fn cases_in_category(&self, cat: &str) -> Vec<&BenchmarkCase> {
        self.cases.iter().filter(|c| c.category == cat).collect()
    }

    /// List unique categories.
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self.cases.iter().map(|c| c.category.clone()).collect();
        cats.sort();
        cats.dedup();
        cats
    }
}

// ── Regression Detector ──────────────────────────────────────────────────

/// Outcome of a regression check.
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionVerdict {
    /// Performance improved by the given percentage.
    Improved(f64),
    /// No significant change.
    Stable,
    /// Performance regressed by the given percentage.
    Regressed(f64),
}

/// Compares benchmark results against a stored baseline.
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    /// Maximum allowed regression percentage before flagging.
    pub threshold_pct: f64,
    /// Minimum improvement percentage to count as improved.
    pub improvement_pct: f64,
    /// Baselines keyed by benchmark name (`mean_ns`).
    pub baselines: HashMap<String, f64>,
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self { threshold_pct: 5.0, improvement_pct: 5.0, baselines: HashMap::new() }
    }
}

impl RegressionDetector {
    #[must_use]
    pub fn new(threshold_pct: f64, improvement_pct: f64) -> Self {
        Self { threshold_pct, improvement_pct, baselines: HashMap::new() }
    }

    /// Register a baseline `mean_ns` for a named benchmark.
    pub fn set_baseline(&mut self, name: impl Into<String>, mean_ns: f64) {
        self.baselines.insert(name.into(), mean_ns);
    }

    /// Compare a result against its baseline.
    pub fn check(&self, result: &BenchmarkResult) -> Option<RegressionVerdict> {
        let baseline = self.baselines.get(&result.name)?;
        if *baseline == 0.0 {
            return Some(RegressionVerdict::Stable);
        }
        let change_pct = (result.timing.mean_ns - baseline) / baseline * 100.0;
        if change_pct > self.threshold_pct {
            Some(RegressionVerdict::Regressed(change_pct))
        } else if change_pct < -self.improvement_pct {
            Some(RegressionVerdict::Improved(-change_pct))
        } else {
            Some(RegressionVerdict::Stable)
        }
    }

    /// Check all results, returning verdicts keyed by name.
    pub fn check_all(&self, results: &[BenchmarkResult]) -> HashMap<String, RegressionVerdict> {
        results.iter().filter_map(|r| self.check(r).map(|v| (r.name.clone(), v))).collect()
    }

    /// Returns `true` if any result regressed.
    pub fn has_regressions(&self, results: &[BenchmarkResult]) -> bool {
        self.check_all(results).values().any(|v| matches!(v, RegressionVerdict::Regressed(_)))
    }
}

// ── Benchmark Reporter ───────────────────────────────────────────────────

/// Output format for reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Text,
    Json,
    Csv,
    Markdown,
}

/// Generates formatted benchmark reports.
#[derive(Debug, Clone)]
pub struct BenchmarkReporter {
    pub format: ReportFormat,
    pub include_samples: bool,
}

impl BenchmarkReporter {
    #[must_use]
    pub const fn new(format: ReportFormat) -> Self {
        Self { format, include_samples: false }
    }

    #[must_use]
    pub const fn with_samples(mut self, include: bool) -> Self {
        self.include_samples = include;
        self
    }

    /// Render a set of results into the configured format.
    pub fn render(&self, results: &[BenchmarkResult]) -> String {
        match self.format {
            ReportFormat::Text => render_text(results),
            ReportFormat::Json => render_json(results, self.include_samples),
            ReportFormat::Csv => render_csv(results),
            ReportFormat::Markdown => render_markdown(results),
        }
    }
}

fn render_text(results: &[BenchmarkResult]) -> String {
    let mut out = String::from("Benchmark Results\n");
    out.push_str(&"=".repeat(60));
    out.push('\n');
    for r in results {
        let _ = writeln!(
            out,
            "{} [{}]: mean={:.1}ns median={}ns min={}ns max={}ns \
             std_dev={:.1}ns p99={}ns",
            r.name,
            r.category,
            r.timing.mean_ns,
            r.timing.median_ns,
            r.timing.min_ns,
            r.timing.max_ns,
            r.timing.std_dev_ns,
            r.timing.p99_ns,
        );
        if let Some(ref tp) = r.throughput {
            let _ = writeln!(
                out,
                "  throughput: {:.1} tok/s, {:.1} GFLOPS, {:.2} GB/s",
                tp.tokens_per_sec,
                tp.flops / 1e9,
                tp.bandwidth_gbs,
            );
        }
    }
    out
}

fn render_json(results: &[BenchmarkResult], include_samples: bool) -> String {
    let mut out = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        let _ = write!(
            out,
            "  {{\"name\":\"{}\",\"category\":\"{}\",\"mean_ns\":{:.1},\
             \"median_ns\":{},\"min_ns\":{},\"max_ns\":{},\
             \"std_dev_ns\":{:.1},\"p99_ns\":{}",
            r.name,
            r.category,
            r.timing.mean_ns,
            r.timing.median_ns,
            r.timing.min_ns,
            r.timing.max_ns,
            r.timing.std_dev_ns,
            r.timing.p99_ns,
        );
        if let Some(ref tp) = r.throughput {
            let _ = write!(
                out,
                ",\"tokens_per_sec\":{:.1},\"flops\":{:.1},\
                 \"bandwidth_gbs\":{:.2}",
                tp.tokens_per_sec, tp.flops, tp.bandwidth_gbs,
            );
        }
        if include_samples {
            let s: Vec<String> = r.timing.samples.iter().map(ToString::to_string).collect();
            let _ = write!(out, ",\"samples\":[{}]", s.join(","));
        }
        out.push('}');
        if i + 1 < results.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push(']');
    out
}

fn render_csv(results: &[BenchmarkResult]) -> String {
    let mut out = String::from("name,category,mean_ns,median_ns,min_ns,max_ns,std_dev_ns,p99_ns\n");
    for r in results {
        let _ = writeln!(
            out,
            "{},{},{:.1},{},{},{},{:.1},{}",
            r.name,
            r.category,
            r.timing.mean_ns,
            r.timing.median_ns,
            r.timing.min_ns,
            r.timing.max_ns,
            r.timing.std_dev_ns,
            r.timing.p99_ns,
        );
    }
    out
}

fn render_markdown(results: &[BenchmarkResult]) -> String {
    let mut out = String::from("# Benchmark Results\n\n");
    out.push_str(
        "| Name | Category | Mean (ns) | Median (ns) \
         | Min (ns) | Max (ns) | Std Dev | P99 (ns) |\n",
    );
    out.push_str(
        "|------|----------|-----------|-------------|\
         ----------|----------|---------|----------|\n",
    );
    for r in results {
        let _ = writeln!(
            out,
            "| {} | {} | {:.1} | {} | {} | {} | {:.1} | {} |",
            r.name,
            r.category,
            r.timing.mean_ns,
            r.timing.median_ns,
            r.timing.min_ns,
            r.timing.max_ns,
            r.timing.std_dev_ns,
            r.timing.p99_ns,
        );
    }
    out
}

// ── Profiler Integration ─────────────────────────────────────────────────

/// Marker event type for GPU profilers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfilerEvent {
    /// Begin a named region.
    RangeStart(String),
    /// End the current region.
    RangeEnd(String),
    /// Instantaneous marker.
    Marker(String),
}

/// Collects profiler hooks for GPU kernel instrumentation.
#[derive(Debug, Clone)]
pub struct ProfilerIntegration {
    /// Whether profiling is enabled.
    pub enabled: bool,
    /// Recorded events for validation/testing.
    events: Vec<ProfilerEvent>,
    /// Active range stack for nesting validation.
    active_ranges: Vec<String>,
}

impl Default for ProfilerIntegration {
    fn default() -> Self {
        Self { enabled: true, events: Vec::new(), active_ranges: Vec::new() }
    }
}

impl ProfilerIntegration {
    #[must_use]
    pub const fn new(enabled: bool) -> Self {
        Self { enabled, events: Vec::new(), active_ranges: Vec::new() }
    }

    /// Begin a named profiler range.
    pub fn range_start(&mut self, name: impl Into<String>) {
        if !self.enabled {
            return;
        }
        let n = name.into();
        self.events.push(ProfilerEvent::RangeStart(n.clone()));
        self.active_ranges.push(n);
    }

    /// End the most recent profiler range.
    pub fn range_end(&mut self, name: impl Into<String>) {
        if !self.enabled {
            return;
        }
        let n = name.into();
        self.events.push(ProfilerEvent::RangeEnd(n.clone()));
        if let Some(pos) = self.active_ranges.iter().rposition(|r| r == &n) {
            self.active_ranges.remove(pos);
        }
    }

    /// Emit an instantaneous marker.
    pub fn marker(&mut self, name: impl Into<String>) {
        if !self.enabled {
            return;
        }
        self.events.push(ProfilerEvent::Marker(name.into()));
    }

    /// Return all recorded events.
    pub fn events(&self) -> &[ProfilerEvent] {
        &self.events
    }

    /// Return currently active (unclosed) ranges.
    pub fn active_ranges(&self) -> &[String] {
        &self.active_ranges
    }

    /// Check if all ranges have been closed.
    pub const fn all_ranges_closed(&self) -> bool {
        self.active_ranges.is_empty()
    }

    /// Reset recorded state.
    pub fn reset(&mut self) {
        self.events.clear();
        self.active_ranges.clear();
    }
}

// ── Benchmark Comparator ─────────────────────────────────────────────────

/// Result of comparing two benchmark runs (A vs B).
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub name: String,
    pub a_mean_ns: f64,
    pub b_mean_ns: f64,
    /// Speedup of B over A (>1.0 means B is faster).
    pub speedup: f64,
    /// Percentage change (negative = B is faster).
    pub change_pct: f64,
    /// Whether the difference is statistically significant.
    pub significant: bool,
}

/// Compares two benchmark runs for A/B analysis.
#[derive(Debug, Clone)]
pub struct BenchmarkComparator {
    /// Minimum percentage change to consider significant.
    pub significance_threshold: f64,
    /// Minimum sample overlap required.
    pub min_samples: usize,
}

impl Default for BenchmarkComparator {
    fn default() -> Self {
        Self { significance_threshold: 5.0, min_samples: 10 }
    }
}

impl BenchmarkComparator {
    #[must_use]
    pub const fn new(significance_threshold: f64) -> Self {
        Self { significance_threshold, min_samples: 10 }
    }

    /// Compare two result sets, matching by name.
    pub fn compare(
        &self,
        a_results: &[BenchmarkResult],
        b_results: &[BenchmarkResult],
    ) -> Vec<ComparisonResult> {
        let b_map: HashMap<&str, &BenchmarkResult> =
            b_results.iter().map(|r| (r.name.as_str(), r)).collect();
        a_results
            .iter()
            .filter_map(|a| {
                let b = b_map.get(a.name.as_str())?;
                Some(self.compare_pair(a, b))
            })
            .collect()
    }

    /// Compare a single pair of results.
    pub fn compare_pair(&self, a: &BenchmarkResult, b: &BenchmarkResult) -> ComparisonResult {
        let a_mean = a.timing.mean_ns;
        let b_mean = b.timing.mean_ns;
        let change_pct = if a_mean > 0.0 { (b_mean - a_mean) / a_mean * 100.0 } else { 0.0 };
        let speedup = if b_mean > 0.0 { a_mean / b_mean } else { 0.0 };
        let significant = change_pct.abs() >= self.significance_threshold
            && a.timing.samples.len() >= self.min_samples
            && b.timing.samples.len() >= self.min_samples;
        ComparisonResult {
            name: a.name.clone(),
            a_mean_ns: a_mean,
            b_mean_ns: b_mean,
            speedup,
            change_pct,
            significant,
        }
    }

    /// Render a comparison as a text table.
    pub fn render_comparison(comparisons: &[ComparisonResult]) -> String {
        let mut out = String::from("A/B Comparison\n");
        out.push_str(&"-".repeat(80));
        out.push('\n');
        let _ = writeln!(
            out,
            "{:<30} {:>12} {:>12} {:>8} {:>8} Sig?",
            "Benchmark", "A (ns)", "B (ns)", "Change", "Speedup"
        );
        out.push_str(&"-".repeat(80));
        out.push('\n');
        for c in comparisons {
            let _ = writeln!(
                out,
                "{:<30} {:>12.1} {:>12.1} {:>+7.1}% {:>7.2}x {}",
                c.name,
                c.a_mean_ns,
                c.b_mean_ns,
                c.change_pct,
                c.speedup,
                if c.significant { "*" } else { "" },
            );
        }
        out
    }
}

// ── Benchmark Harness ────────────────────────────────────────────────────

/// Result of a full harness run.
#[derive(Debug)]
pub struct HarnessRunResult {
    pub suite_name: String,
    pub results: Vec<BenchmarkResult>,
    pub total_elapsed: Duration,
    pub regressions: HashMap<String, RegressionVerdict>,
}

/// Orchestrator: configure → discover → warmup → measure → report.
#[derive(Debug)]
pub struct BenchmarkHarness {
    pub config: BenchmarkConfig,
    pub profiler: ProfilerIntegration,
    pub detector: Option<RegressionDetector>,
    pub reporter: BenchmarkReporter,
}

impl Default for BenchmarkHarness {
    fn default() -> Self {
        Self {
            config: BenchmarkConfig::default(),
            profiler: ProfilerIntegration::default(),
            reporter: BenchmarkReporter::new(ReportFormat::Text),
            detector: None,
        }
    }
}

impl BenchmarkHarness {
    #[must_use]
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config, ..Default::default() }
    }

    #[must_use]
    pub fn with_profiler(mut self, profiler: ProfilerIntegration) -> Self {
        self.profiler = profiler;
        self
    }

    #[must_use]
    pub fn with_regression_detector(mut self, det: RegressionDetector) -> Self {
        self.detector = Some(det);
        self
    }

    #[must_use]
    pub const fn with_reporter(mut self, reporter: BenchmarkReporter) -> Self {
        self.reporter = reporter;
        self
    }

    /// Run a full benchmark suite.
    #[allow(clippy::cast_possible_truncation)]
    pub fn run(&mut self, suite: &BenchmarkSuite) -> HarnessRunResult {
        let overall_start = Instant::now();
        let mut results = Vec::with_capacity(suite.cases.len());

        for case in &suite.cases {
            let r = self.run_case(case, &suite.config);
            results.push(r);
        }

        let regressions =
            self.detector.as_ref().map_or_else(HashMap::new, |d| d.check_all(&results));

        HarnessRunResult {
            suite_name: suite.name.clone(),
            results,
            total_elapsed: overall_start.elapsed(),
            regressions,
        }
    }

    /// Run a single benchmark case.
    #[allow(clippy::cast_possible_truncation)]
    fn run_case(&mut self, case: &BenchmarkCase, config: &BenchmarkConfig) -> BenchmarkResult {
        let ctx = case.setup.map_or_else(HashMap::new, |f| f());

        self.profiler.range_start(format!("warmup:{}", case.name));
        for _ in 0..config.warmup_iterations {
            (case.run)(&ctx);
        }
        self.profiler.range_end(format!("warmup:{}", case.name));

        self.profiler.range_start(format!("measure:{}", case.name));
        let mut samples = Vec::with_capacity(config.measurement_iterations as usize);
        for _ in 0..config.measurement_iterations {
            let start = Instant::now();
            (case.run)(&ctx);
            samples.push(start.elapsed().as_nanos() as u64);
        }
        self.profiler.range_end(format!("measure:{}", case.name));

        if let Some(teardown) = case.teardown {
            teardown(&ctx);
        }

        let mut timing = TimingResult::from_samples(samples);
        if config.outlier_removal {
            timing = timing.remove_outliers(config.outlier_sigma);
        }

        let throughput = if case.workload.flop_count > 0
            || case.workload.token_count > 0
            || case.workload.bytes_transferred > 0
        {
            Some(ThroughputMetric::compute(
                case.workload.token_count,
                case.workload.flop_count,
                case.workload.bytes_transferred,
                timing.mean_ns,
            ))
        } else {
            None
        };

        BenchmarkResult {
            name: case.name.clone(),
            category: case.category.clone(),
            timing,
            throughput,
            timestamp: Duration::from_nanos(0),
        }
    }

    /// Generate a report from results.
    pub fn report(&self, results: &[BenchmarkResult]) -> String {
        self.reporter.render(results)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- BenchmarkConfig tests ------------------------------------------------

    #[test]
    fn test_config_default_values() {
        let c = BenchmarkConfig::default();
        assert_eq!(c.warmup_iterations, 10);
        assert_eq!(c.measurement_iterations, 100);
        assert!((c.confidence_level - 0.95).abs() < f64::EPSILON);
        assert!(c.outlier_removal);
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(BenchmarkConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_iterations() {
        let c = BenchmarkConfig { measurement_iterations: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_validate_confidence_too_high() {
        let c = BenchmarkConfig { confidence_level: 1.5, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_validate_confidence_negative() {
        let c = BenchmarkConfig { confidence_level: -0.1, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_validate_sigma_zero() {
        let c = BenchmarkConfig { outlier_sigma: 0.0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_validate_sigma_negative() {
        let c = BenchmarkConfig { outlier_sigma: -1.0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_min_duration() {
        let c = BenchmarkConfig::default();
        assert_eq!(c.min_duration, Duration::from_millis(100));
    }

    // -- TimingResult tests ---------------------------------------------------

    #[test]
    fn test_timing_single_sample() {
        let t = TimingResult::from_samples(vec![1000]);
        assert_eq!(t.min_ns, 1000);
        assert_eq!(t.max_ns, 1000);
        assert_eq!(t.median_ns, 1000);
        assert!((t.mean_ns - 1000.0).abs() < f64::EPSILON);
        assert!(t.std_dev_ns < f64::EPSILON);
    }

    #[test]
    fn test_timing_two_samples() {
        let t = TimingResult::from_samples(vec![100, 200]);
        assert_eq!(t.min_ns, 100);
        assert_eq!(t.max_ns, 200);
        assert_eq!(t.median_ns, 150);
        assert!((t.mean_ns - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_timing_odd_samples_median() {
        let t = TimingResult::from_samples(vec![10, 20, 30]);
        assert_eq!(t.median_ns, 20);
    }

    #[test]
    fn test_timing_even_samples_median() {
        let t = TimingResult::from_samples(vec![10, 20, 30, 40]);
        assert_eq!(t.median_ns, 25);
    }

    #[test]
    fn test_timing_sorted() {
        let t = TimingResult::from_samples(vec![300, 100, 200]);
        assert_eq!(t.samples, vec![100, 200, 300]);
    }

    #[test]
    fn test_timing_p99_small() {
        let t = TimingResult::from_samples(vec![10, 20, 30, 40, 50]);
        assert_eq!(t.p99_ns, 50);
    }

    #[test]
    fn test_timing_p99_large() {
        let samples: Vec<u64> = (1..=100).collect();
        let t = TimingResult::from_samples(samples);
        assert_eq!(t.p99_ns, 99);
    }

    #[test]
    fn test_timing_std_dev() {
        let t = TimingResult::from_samples(vec![100, 100, 100]);
        assert!(t.std_dev_ns < f64::EPSILON);
    }

    #[test]
    fn test_timing_std_dev_nonzero() {
        let t = TimingResult::from_samples(vec![100, 200]);
        assert!(t.std_dev_ns > 0.0);
    }

    #[test]
    fn test_timing_cv_zero() {
        let t = TimingResult::from_samples(vec![100, 100, 100]);
        assert!(t.cv() < f64::EPSILON);
    }

    #[test]
    fn test_timing_cv_positive() {
        let t = TimingResult::from_samples(vec![100, 200, 300]);
        assert!(t.cv() > 0.0);
    }

    #[test]
    fn test_timing_remove_outliers() {
        let mut samples: Vec<u64> = vec![100; 98];
        samples.push(100_000); // outlier
        samples.push(1); // outlier
        let t = TimingResult::from_samples(samples);
        let filtered = t.remove_outliers(2.0);
        assert!(filtered.max_ns < 100_000);
    }

    #[test]
    fn test_timing_remove_outliers_keeps_all_when_similar() {
        let t = TimingResult::from_samples(vec![100, 101, 102, 103]);
        let filtered = t.remove_outliers(3.0);
        assert_eq!(filtered.samples.len(), 4);
    }

    #[test]
    fn test_timing_remove_outliers_empty_fallback() {
        let t = TimingResult::from_samples(vec![1, 1_000_000]);
        // With extremely tight sigma, might filter everything—fallback
        let filtered = t.remove_outliers(0.001);
        assert!(!filtered.samples.is_empty());
    }

    #[test]
    #[should_panic(expected = "need at least one sample")]
    fn test_timing_empty_samples_panics() {
        TimingResult::from_samples(vec![]);
    }

    // -- ThroughputMetric tests -----------------------------------------------

    #[test]
    fn test_throughput_basic() {
        let tp = ThroughputMetric::compute(1000, 1_000_000, 1_000_000_000, 1e9);
        assert!((tp.tokens_per_sec - 1000.0).abs() < 1e-6);
        assert!((tp.flops - 1_000_000.0).abs() < 1e-6);
        assert!((tp.bandwidth_gbs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_throughput_zero_elapsed() {
        let tp = ThroughputMetric::compute(100, 100, 100, 0.0);
        assert_eq!(tp.tokens_per_sec, 0.0);
        assert_eq!(tp.flops, 0.0);
        assert_eq!(tp.bandwidth_gbs, 0.0);
    }

    #[test]
    fn test_throughput_scale() {
        let tp = ThroughputMetric::compute(100, 100, 100, 1e9);
        let scaled = tp.scale(2.0);
        assert!((scaled.tokens_per_sec - tp.tokens_per_sec * 2.0).abs() < 1e-6);
        assert!((scaled.flops - tp.flops * 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_throughput_scale_zero() {
        let tp = ThroughputMetric::compute(100, 100, 100, 1e9);
        let scaled = tp.scale(0.0);
        assert_eq!(scaled.tokens_per_sec, 0.0);
    }

    #[test]
    fn test_throughput_high_flops() {
        let tp = ThroughputMetric::compute(0, 1_000_000_000, 0, 1e9);
        assert!((tp.flops - 1e9).abs() < 1e-3);
    }

    #[test]
    fn test_throughput_high_bandwidth() {
        let tp = ThroughputMetric::compute(0, 0, 10_000_000_000, 1e9);
        assert!((tp.bandwidth_gbs - 10.0).abs() < 1e-6);
    }

    // -- WorkloadSpec tests ---------------------------------------------------

    #[test]
    fn test_workload_default() {
        let w = WorkloadSpec::default();
        assert_eq!(w.token_count, 0);
        assert_eq!(w.flop_count, 0);
        assert_eq!(w.bytes_transferred, 0);
    }

    // -- BenchmarkCase tests --------------------------------------------------

    fn noop_run(_ctx: &HashMap<String, String>) {}

    fn setup_fn() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("key".into(), "value".into());
        m
    }

    fn teardown_fn(_ctx: &HashMap<String, String>) {}

    #[test]
    fn test_case_new() {
        let c = BenchmarkCase::new("test", noop_run);
        assert_eq!(c.name, "test");
        assert_eq!(c.category, "default");
        assert!(c.setup.is_none());
        assert!(c.teardown.is_none());
    }

    #[test]
    fn test_case_with_category() {
        let c = BenchmarkCase::new("t", noop_run).with_category("matmul");
        assert_eq!(c.category, "matmul");
    }

    #[test]
    fn test_case_with_setup() {
        let c = BenchmarkCase::new("t", noop_run).with_setup(setup_fn);
        assert!(c.setup.is_some());
        let ctx = (c.setup.unwrap())();
        assert_eq!(ctx.get("key").unwrap(), "value");
    }

    #[test]
    fn test_case_with_teardown() {
        let c = BenchmarkCase::new("t", noop_run).with_teardown(teardown_fn);
        assert!(c.teardown.is_some());
    }

    #[test]
    fn test_case_with_workload() {
        let w = WorkloadSpec { token_count: 42, flop_count: 100, bytes_transferred: 200 };
        let c = BenchmarkCase::new("t", noop_run).with_workload(w);
        assert_eq!(c.workload.token_count, 42);
    }

    #[test]
    fn test_case_debug_format() {
        let c = BenchmarkCase::new("debug_test", noop_run);
        let dbg = format!("{c:?}");
        assert!(dbg.contains("debug_test"));
    }

    #[test]
    fn test_case_builder_chain() {
        let c = BenchmarkCase::new("chain", noop_run)
            .with_category("attn")
            .with_setup(setup_fn)
            .with_teardown(teardown_fn)
            .with_workload(WorkloadSpec { token_count: 10, flop_count: 20, bytes_transferred: 30 });
        assert_eq!(c.category, "attn");
        assert!(c.setup.is_some());
        assert!(c.teardown.is_some());
        assert_eq!(c.workload.token_count, 10);
    }

    // -- BenchmarkSuite tests -------------------------------------------------

    #[test]
    fn test_suite_new() {
        let s = BenchmarkSuite::new("gpu");
        assert_eq!(s.name, "gpu");
        assert!(s.cases.is_empty());
    }

    #[test]
    fn test_suite_with_description() {
        let s = BenchmarkSuite::new("s").with_description("desc");
        assert_eq!(s.description, "desc");
    }

    #[test]
    fn test_suite_add_case() {
        let mut s = BenchmarkSuite::new("s");
        s.add_case(BenchmarkCase::new("a", noop_run));
        s.add_case(BenchmarkCase::new("b", noop_run));
        assert_eq!(s.cases.len(), 2);
    }

    #[test]
    fn test_suite_categories() {
        let mut s = BenchmarkSuite::new("s");
        s.add_case(BenchmarkCase::new("a", noop_run).with_category("x"));
        s.add_case(BenchmarkCase::new("b", noop_run).with_category("y"));
        s.add_case(BenchmarkCase::new("c", noop_run).with_category("x"));
        let cats = s.categories();
        assert_eq!(cats, vec!["x", "y"]);
    }

    #[test]
    fn test_suite_cases_in_category() {
        let mut s = BenchmarkSuite::new("s");
        s.add_case(BenchmarkCase::new("a", noop_run).with_category("matmul"));
        s.add_case(BenchmarkCase::new("b", noop_run).with_category("attn"));
        s.add_case(BenchmarkCase::new("c", noop_run).with_category("matmul"));
        assert_eq!(s.cases_in_category("matmul").len(), 2);
        assert_eq!(s.cases_in_category("attn").len(), 1);
        assert_eq!(s.cases_in_category("none").len(), 0);
    }

    #[test]
    fn test_suite_with_config() {
        let cfg = BenchmarkConfig { warmup_iterations: 5, ..Default::default() };
        let s = BenchmarkSuite::new("s").with_config(cfg);
        assert_eq!(s.config.warmup_iterations, 5);
    }

    // -- RegressionDetector tests ---------------------------------------------

    fn make_result(name: &str, mean: f64) -> BenchmarkResult {
        BenchmarkResult {
            name: name.into(),
            category: "test".into(),
            timing: TimingResult {
                min_ns: mean as u64,
                max_ns: mean as u64,
                mean_ns: mean,
                median_ns: mean as u64,
                std_dev_ns: 0.0,
                p99_ns: mean as u64,
                samples: vec![mean as u64],
            },
            throughput: None,
            timestamp: Duration::ZERO,
        }
    }

    #[test]
    fn test_regression_default() {
        let d = RegressionDetector::default();
        assert_eq!(d.threshold_pct, 5.0);
        assert_eq!(d.improvement_pct, 5.0);
    }

    #[test]
    fn test_regression_stable() {
        let mut d = RegressionDetector::default();
        d.set_baseline("k", 1000.0);
        let r = make_result("k", 1020.0);
        assert_eq!(d.check(&r), Some(RegressionVerdict::Stable));
    }

    #[test]
    fn test_regression_regressed() {
        let mut d = RegressionDetector::new(5.0, 5.0);
        d.set_baseline("k", 1000.0);
        let r = make_result("k", 1100.0);
        match d.check(&r) {
            Some(RegressionVerdict::Regressed(pct)) => assert!(pct > 5.0),
            other => panic!("expected Regressed, got {other:?}"),
        }
    }

    #[test]
    fn test_regression_improved() {
        let mut d = RegressionDetector::new(5.0, 5.0);
        d.set_baseline("k", 1000.0);
        let r = make_result("k", 900.0);
        match d.check(&r) {
            Some(RegressionVerdict::Improved(pct)) => assert!(pct > 5.0),
            other => panic!("expected Improved, got {other:?}"),
        }
    }

    #[test]
    fn test_regression_no_baseline() {
        let d = RegressionDetector::default();
        let r = make_result("k", 1000.0);
        assert_eq!(d.check(&r), None);
    }

    #[test]
    fn test_regression_zero_baseline() {
        let mut d = RegressionDetector::default();
        d.set_baseline("k", 0.0);
        let r = make_result("k", 100.0);
        assert_eq!(d.check(&r), Some(RegressionVerdict::Stable));
    }

    #[test]
    fn test_regression_check_all() {
        let mut d = RegressionDetector::new(5.0, 5.0);
        d.set_baseline("a", 1000.0);
        d.set_baseline("b", 1000.0);
        let results = vec![make_result("a", 1200.0), make_result("b", 800.0)];
        let verdicts = d.check_all(&results);
        assert!(matches!(verdicts.get("a"), Some(RegressionVerdict::Regressed(_))));
        assert!(matches!(verdicts.get("b"), Some(RegressionVerdict::Improved(_))));
    }

    #[test]
    fn test_regression_has_regressions_true() {
        let mut d = RegressionDetector::new(5.0, 5.0);
        d.set_baseline("a", 1000.0);
        let results = vec![make_result("a", 1200.0)];
        assert!(d.has_regressions(&results));
    }

    #[test]
    fn test_regression_has_regressions_false() {
        let mut d = RegressionDetector::new(5.0, 5.0);
        d.set_baseline("a", 1000.0);
        let results = vec![make_result("a", 1010.0)];
        assert!(!d.has_regressions(&results));
    }

    #[test]
    fn test_regression_boundary_exact_threshold() {
        let mut d = RegressionDetector::new(5.0, 5.0);
        d.set_baseline("k", 1000.0);
        let r = make_result("k", 1050.0);
        assert_eq!(d.check(&r), Some(RegressionVerdict::Stable));
    }

    #[test]
    fn test_regression_just_over_threshold() {
        let mut d = RegressionDetector::new(5.0, 5.0);
        d.set_baseline("k", 1000.0);
        let r = make_result("k", 1050.1);
        match d.check(&r) {
            Some(RegressionVerdict::Regressed(_)) => {}
            other => panic!("expected Regressed, got {other:?}"),
        }
    }

    // -- BenchmarkReporter tests ----------------------------------------------

    fn sample_results() -> Vec<BenchmarkResult> {
        vec![BenchmarkResult {
            name: "kernel_a".into(),
            category: "matmul".into(),
            timing: TimingResult::from_samples(vec![100, 110, 120, 130, 140]),
            throughput: Some(ThroughputMetric {
                tokens_per_sec: 1000.0,
                flops: 1e9,
                bandwidth_gbs: 10.0,
            }),
            timestamp: Duration::ZERO,
        }]
    }

    #[test]
    fn test_reporter_text() {
        let r = BenchmarkReporter::new(ReportFormat::Text);
        let out = r.render(&sample_results());
        assert!(out.contains("kernel_a"));
        assert!(out.contains("matmul"));
        assert!(out.contains("mean="));
    }

    #[test]
    fn test_reporter_json() {
        let r = BenchmarkReporter::new(ReportFormat::Json);
        let out = r.render(&sample_results());
        assert!(out.contains("\"name\":\"kernel_a\""));
        assert!(out.contains("\"mean_ns\":"));
    }

    #[test]
    fn test_reporter_csv() {
        let r = BenchmarkReporter::new(ReportFormat::Csv);
        let out = r.render(&sample_results());
        assert!(out.starts_with("name,category,mean_ns"));
        assert!(out.contains("kernel_a,matmul,"));
    }

    #[test]
    fn test_reporter_markdown() {
        let r = BenchmarkReporter::new(ReportFormat::Markdown);
        let out = r.render(&sample_results());
        assert!(out.contains("# Benchmark Results"));
        assert!(out.contains("| kernel_a |"));
    }

    #[test]
    fn test_reporter_json_with_samples() {
        let r = BenchmarkReporter::new(ReportFormat::Json).with_samples(true);
        let out = r.render(&sample_results());
        assert!(out.contains("\"samples\":["));
    }

    #[test]
    fn test_reporter_json_without_samples() {
        let r = BenchmarkReporter::new(ReportFormat::Json).with_samples(false);
        let out = r.render(&sample_results());
        assert!(!out.contains("\"samples\""));
    }

    #[test]
    fn test_reporter_empty_results() {
        let r = BenchmarkReporter::new(ReportFormat::Text);
        let out = r.render(&[]);
        assert!(out.contains("Benchmark Results"));
    }

    #[test]
    fn test_reporter_csv_header() {
        let r = BenchmarkReporter::new(ReportFormat::Csv);
        let out = r.render(&[]);
        let header = out.lines().next().unwrap();
        assert_eq!(header, "name,category,mean_ns,median_ns,min_ns,max_ns,std_dev_ns,p99_ns");
    }

    #[test]
    fn test_reporter_text_throughput() {
        let r = BenchmarkReporter::new(ReportFormat::Text);
        let out = r.render(&sample_results());
        assert!(out.contains("tok/s"));
        assert!(out.contains("GFLOPS"));
        assert!(out.contains("GB/s"));
    }

    #[test]
    fn test_reporter_markdown_table_header() {
        let r = BenchmarkReporter::new(ReportFormat::Markdown);
        let out = r.render(&sample_results());
        assert!(out.contains("| Name |"));
        assert!(out.contains("|------|"));
    }

    // -- ProfilerIntegration tests --------------------------------------------

    #[test]
    fn test_profiler_default_enabled() {
        let p = ProfilerIntegration::default();
        assert!(p.enabled);
        assert!(p.events().is_empty());
    }

    #[test]
    fn test_profiler_disabled() {
        let mut p = ProfilerIntegration::new(false);
        p.range_start("x");
        p.marker("y");
        p.range_end("x");
        assert!(p.events().is_empty());
    }

    #[test]
    fn test_profiler_range_start_end() {
        let mut p = ProfilerIntegration::default();
        p.range_start("kernel");
        assert_eq!(p.active_ranges(), &["kernel"]);
        p.range_end("kernel");
        assert!(p.all_ranges_closed());
    }

    #[test]
    fn test_profiler_marker() {
        let mut p = ProfilerIntegration::default();
        p.marker("iteration_0");
        assert_eq!(p.events().len(), 1);
        assert_eq!(p.events()[0], ProfilerEvent::Marker("iteration_0".into()));
    }

    #[test]
    fn test_profiler_nested_ranges() {
        let mut p = ProfilerIntegration::default();
        p.range_start("outer");
        p.range_start("inner");
        assert_eq!(p.active_ranges().len(), 2);
        p.range_end("inner");
        assert_eq!(p.active_ranges(), &["outer"]);
        p.range_end("outer");
        assert!(p.all_ranges_closed());
    }

    #[test]
    fn test_profiler_event_sequence() {
        let mut p = ProfilerIntegration::default();
        p.range_start("a");
        p.marker("m");
        p.range_end("a");
        assert_eq!(p.events().len(), 3);
        assert_eq!(p.events()[0], ProfilerEvent::RangeStart("a".into()));
        assert_eq!(p.events()[1], ProfilerEvent::Marker("m".into()));
        assert_eq!(p.events()[2], ProfilerEvent::RangeEnd("a".into()));
    }

    #[test]
    fn test_profiler_reset() {
        let mut p = ProfilerIntegration::default();
        p.range_start("x");
        p.marker("y");
        p.reset();
        assert!(p.events().is_empty());
        assert!(p.active_ranges().is_empty());
    }

    #[test]
    fn test_profiler_unclosed_range() {
        let mut p = ProfilerIntegration::default();
        p.range_start("leak");
        assert!(!p.all_ranges_closed());
    }

    // -- BenchmarkComparator tests --------------------------------------------

    fn make_result_with_samples(name: &str, samples: Vec<u64>) -> BenchmarkResult {
        BenchmarkResult {
            name: name.into(),
            category: "test".into(),
            timing: TimingResult::from_samples(samples),
            throughput: None,
            timestamp: Duration::ZERO,
        }
    }

    #[test]
    fn test_comparator_default() {
        let c = BenchmarkComparator::default();
        assert_eq!(c.significance_threshold, 5.0);
        assert_eq!(c.min_samples, 10);
    }

    #[test]
    fn test_comparator_faster_b() {
        let a = make_result_with_samples("k", vec![200; 20]);
        let b = make_result_with_samples("k", vec![100; 20]);
        let c = BenchmarkComparator::new(5.0);
        let cmp = c.compare_pair(&a, &b);
        assert!(cmp.speedup > 1.0);
        assert!(cmp.change_pct < 0.0);
        assert!(cmp.significant);
    }

    #[test]
    fn test_comparator_slower_b() {
        let a = make_result_with_samples("k", vec![100; 20]);
        let b = make_result_with_samples("k", vec![200; 20]);
        let c = BenchmarkComparator::new(5.0);
        let cmp = c.compare_pair(&a, &b);
        assert!(cmp.speedup < 1.0);
        assert!(cmp.change_pct > 0.0);
    }

    #[test]
    fn test_comparator_same_performance() {
        let a = make_result_with_samples("k", vec![100; 20]);
        let b = make_result_with_samples("k", vec![100; 20]);
        let c = BenchmarkComparator::new(5.0);
        let cmp = c.compare_pair(&a, &b);
        assert!((cmp.change_pct).abs() < f64::EPSILON);
        assert!(!cmp.significant);
    }

    #[test]
    fn test_comparator_not_significant_few_samples() {
        let a = make_result_with_samples("k", vec![200]);
        let b = make_result_with_samples("k", vec![100]);
        let c = BenchmarkComparator::new(5.0);
        let cmp = c.compare_pair(&a, &b);
        assert!(!cmp.significant); // too few samples
    }

    #[test]
    fn test_comparator_compare_multiple() {
        let a = vec![
            make_result_with_samples("x", vec![100; 20]),
            make_result_with_samples("y", vec![200; 20]),
        ];
        let b = vec![
            make_result_with_samples("x", vec![90; 20]),
            make_result_with_samples("y", vec![220; 20]),
        ];
        let c = BenchmarkComparator::new(5.0);
        let results = c.compare(&a, &b);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_comparator_missing_in_b() {
        let a = vec![make_result_with_samples("x", vec![100; 20])];
        let b: Vec<BenchmarkResult> = vec![];
        let c = BenchmarkComparator::new(5.0);
        let results = c.compare(&a, &b);
        assert!(results.is_empty());
    }

    #[test]
    fn test_comparator_render() {
        let a = make_result_with_samples("k", vec![200; 20]);
        let b = make_result_with_samples("k", vec![100; 20]);
        let c = BenchmarkComparator::new(5.0);
        let cmp = c.compare_pair(&a, &b);
        let out = BenchmarkComparator::render_comparison(&[cmp]);
        assert!(out.contains("A/B Comparison"));
        assert!(out.contains("k"));
    }

    #[test]
    fn test_comparator_render_empty() {
        let out = BenchmarkComparator::render_comparison(&[]);
        assert!(out.contains("A/B Comparison"));
    }

    #[test]
    fn test_comparator_speedup_value() {
        let a = make_result_with_samples("k", vec![200; 20]);
        let b = make_result_with_samples("k", vec![100; 20]);
        let c = BenchmarkComparator::new(5.0);
        let cmp = c.compare_pair(&a, &b);
        assert!((cmp.speedup - 2.0).abs() < 0.01);
    }

    // -- BenchmarkHarness tests -----------------------------------------------

    #[test]
    fn test_harness_default() {
        let h = BenchmarkHarness::default();
        assert_eq!(h.config.warmup_iterations, 10);
        assert!(h.detector.is_none());
    }

    #[test]
    fn test_harness_with_config() {
        let cfg = BenchmarkConfig {
            warmup_iterations: 3,
            measurement_iterations: 5,
            ..Default::default()
        };
        let h = BenchmarkHarness::new(cfg);
        assert_eq!(h.config.warmup_iterations, 3);
    }

    #[test]
    fn test_harness_with_profiler() {
        let p = ProfilerIntegration::new(true);
        let h = BenchmarkHarness::default().with_profiler(p);
        assert!(h.profiler.enabled);
    }

    #[test]
    fn test_harness_with_reporter() {
        let r = BenchmarkReporter::new(ReportFormat::Json);
        let h = BenchmarkHarness::default().with_reporter(r);
        assert_eq!(h.reporter.format, ReportFormat::Json);
    }

    #[test]
    fn test_harness_with_regression_detector() {
        let d = RegressionDetector::default();
        let h = BenchmarkHarness::default().with_regression_detector(d);
        assert!(h.detector.is_some());
    }

    #[test]
    fn test_harness_run_empty_suite() {
        let mut h = BenchmarkHarness::default();
        let suite = BenchmarkSuite::new("empty");
        let result = h.run(&suite);
        assert!(result.results.is_empty());
        assert_eq!(result.suite_name, "empty");
    }

    #[test]
    fn test_harness_run_single_case() {
        let cfg = BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 5,
            outlier_removal: false,
            ..Default::default()
        };
        let mut h = BenchmarkHarness::new(cfg.clone());
        let mut suite = BenchmarkSuite::new("s").with_config(cfg);
        suite.add_case(BenchmarkCase::new("noop", noop_run));
        let result = h.run(&suite);
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].name, "noop");
    }

    #[test]
    fn test_harness_run_with_setup_teardown() {
        let cfg = BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 3,
            outlier_removal: false,
            ..Default::default()
        };
        let mut h = BenchmarkHarness::new(cfg.clone());
        let mut suite = BenchmarkSuite::new("s").with_config(cfg);
        suite.add_case(
            BenchmarkCase::new("with_ctx", noop_run)
                .with_setup(setup_fn)
                .with_teardown(teardown_fn),
        );
        let result = h.run(&suite);
        assert_eq!(result.results.len(), 1);
    }

    #[test]
    fn test_harness_run_with_workload() {
        let cfg = BenchmarkConfig {
            warmup_iterations: 0,
            measurement_iterations: 3,
            outlier_removal: false,
            ..Default::default()
        };
        let mut h = BenchmarkHarness::new(cfg.clone());
        let mut suite = BenchmarkSuite::new("s").with_config(cfg);
        suite.add_case(BenchmarkCase::new("work", noop_run).with_workload(WorkloadSpec {
            token_count: 100,
            flop_count: 1000,
            bytes_transferred: 4096,
        }));
        let result = h.run(&suite);
        assert!(result.results[0].throughput.is_some());
    }

    #[test]
    fn test_harness_profiler_events_after_run() {
        let cfg = BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 2,
            outlier_removal: false,
            ..Default::default()
        };
        let mut h = BenchmarkHarness::new(cfg.clone());
        let mut suite = BenchmarkSuite::new("s").with_config(cfg);
        suite.add_case(BenchmarkCase::new("x", noop_run));
        h.run(&suite);
        assert!(!h.profiler.events().is_empty());
        assert!(h.profiler.all_ranges_closed());
    }

    #[test]
    fn test_harness_regression_detection() {
        let cfg = BenchmarkConfig {
            warmup_iterations: 0,
            measurement_iterations: 3,
            outlier_removal: false,
            ..Default::default()
        };
        let mut det = RegressionDetector::new(5.0, 5.0);
        det.set_baseline("noop", 0.0);
        let mut h = BenchmarkHarness::new(cfg.clone()).with_regression_detector(det);
        let mut suite = BenchmarkSuite::new("s").with_config(cfg);
        suite.add_case(BenchmarkCase::new("noop", noop_run));
        let result = h.run(&suite);
        assert!(result.regressions.contains_key("noop"));
    }

    #[test]
    fn test_harness_report_output() {
        let h = BenchmarkHarness::default();
        let results = sample_results();
        let report = h.report(&results);
        assert!(report.contains("kernel_a"));
    }

    #[test]
    fn test_harness_multiple_cases() {
        let cfg = BenchmarkConfig {
            warmup_iterations: 0,
            measurement_iterations: 2,
            outlier_removal: false,
            ..Default::default()
        };
        let mut h = BenchmarkHarness::new(cfg.clone());
        let mut suite = BenchmarkSuite::new("multi").with_config(cfg);
        suite.add_case(BenchmarkCase::new("a", noop_run));
        suite.add_case(BenchmarkCase::new("b", noop_run));
        suite.add_case(BenchmarkCase::new("c", noop_run));
        let result = h.run(&suite);
        assert_eq!(result.results.len(), 3);
    }

    #[test]
    fn test_harness_total_elapsed() {
        let cfg = BenchmarkConfig {
            warmup_iterations: 0,
            measurement_iterations: 1,
            outlier_removal: false,
            ..Default::default()
        };
        let mut h = BenchmarkHarness::new(cfg.clone());
        let mut suite = BenchmarkSuite::new("s").with_config(cfg);
        suite.add_case(BenchmarkCase::new("t", noop_run));
        let result = h.run(&suite);
        assert!(result.total_elapsed.as_nanos() > 0);
    }

    #[test]
    fn test_harness_no_throughput_without_workload() {
        let cfg = BenchmarkConfig {
            warmup_iterations: 0,
            measurement_iterations: 2,
            outlier_removal: false,
            ..Default::default()
        };
        let mut h = BenchmarkHarness::new(cfg.clone());
        let mut suite = BenchmarkSuite::new("s").with_config(cfg);
        suite.add_case(BenchmarkCase::new("bare", noop_run));
        let result = h.run(&suite);
        assert!(result.results[0].throughput.is_none());
    }
}
