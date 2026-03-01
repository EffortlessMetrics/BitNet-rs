//! Cross-backend benchmark suite for GPU HAL operations.
//!
//! Provides automated benchmarking with regression detection, latency
//! histograms, and structured reporting across multiple backends.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ── Operation kinds ───────────────────────────────────────────────────────

/// Categories of operations that can be benchmarked.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize,
)]
pub enum OpKind {
    Matmul,
    Elementwise,
    Reduce,
    Softmax,
    Attention,
    Quantize,
    Dequantize,
    Tokenize,
    RmsNorm,
    Rope,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Matmul => "matmul",
            Self::Elementwise => "elementwise",
            Self::Reduce => "reduce",
            Self::Softmax => "softmax",
            Self::Attention => "attention",
            Self::Quantize => "quantize",
            Self::Dequantize => "dequantize",
            Self::Tokenize => "tokenize",
            Self::RmsNorm => "rms_norm",
            Self::Rope => "rope",
        };
        f.write_str(s)
    }
}

// ── Configuration ─────────────────────────────────────────────────────────

/// Configuration for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    /// Number of warm-up iterations before measurement.
    pub warmup_iterations: usize,
    /// Number of measured iterations.
    pub measurement_iterations: usize,
    /// Which backends to include.
    pub backends_to_test: Vec<String>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            measurement_iterations: 10,
            backends_to_test: vec!["cpu".into()],
        }
    }
}

// ── Benchmark case ────────────────────────────────────────────────────────

/// A single benchmark scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCase {
    /// Human-readable name.
    pub name: String,
    /// Input sizes to exercise (e.g. matrix dimensions).
    pub input_sizes: Vec<usize>,
    /// The operation category being tested.
    pub expected_op: OpKind,
}

impl BenchmarkCase {
    pub fn new(
        name: impl Into<String>,
        input_sizes: Vec<usize>,
        expected_op: OpKind,
    ) -> Self {
        Self { name: name.into(), input_sizes, expected_op }
    }
}

// ── Benchmark result ──────────────────────────────────────────────────────

/// Result of running a single benchmark case on one backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Backend that produced this result.
    pub backend: String,
    /// Name of the benchmark case.
    pub case_name: String,
    /// Elapsed wall-clock time in nanoseconds.
    pub elapsed_ns: u64,
    /// Throughput in giga-operations per second (0.0 when N/A).
    pub throughput_gops: f64,
    /// Peak memory usage in bytes (0 when unavailable).
    pub memory_peak_bytes: u64,
    /// Per-iteration latency samples in nanoseconds.
    pub latency_samples_ns: Vec<u64>,
}

// ── Latency histogram ─────────────────────────────────────────────────────

/// Percentile-based latency summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatencyHistogram {
    pub p50_ns: u64,
    pub p90_ns: u64,
    pub p99_ns: u64,
    pub max_ns: u64,
    pub min_ns: u64,
    pub mean_ns: f64,
    pub count: usize,
}

impl LatencyHistogram {
    /// Build a histogram from raw nanosecond samples.
    ///
    /// Returns `None` if the input is empty.
    pub fn from_samples(samples: &[u64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = samples.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        let mean_ns = {
            #[allow(clippy::cast_precision_loss)]
            let v = sorted.iter().copied().sum::<u64>() as f64
                / n as f64;
            v
        };
        Some(Self {
            p50_ns: percentile(&sorted, 50.0),
            p90_ns: percentile(&sorted, 90.0),
            p99_ns: percentile(&sorted, 99.0),
            max_ns: sorted[n - 1],
            min_ns: sorted[0],
            mean_ns,
            count: n,
        })
    }
}

/// Nearest-rank percentile on a **sorted** slice.
fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rank = (pct / 100.0 * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

impl fmt::Display for LatencyHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "P50={} P90={} P99={} max={} (n={})",
            format_ns(self.p50_ns),
            format_ns(self.p90_ns),
            format_ns(self.p99_ns),
            format_ns(self.max_ns),
            self.count,
        )
    }
}

/// Human-readable nanosecond formatting.
#[allow(clippy::cast_precision_loss)]
fn format_ns(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2}s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.2}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2}µs", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

// ── Comparison across backends ────────────────────────────────────────────

/// Pairwise comparison of two backends on the same case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchComparison {
    pub case_name: String,
    pub baseline_backend: String,
    pub candidate_backend: String,
    /// `candidate_elapsed / baseline_elapsed`.
    /// Values < 1.0 mean candidate is faster.
    pub speedup_ratio: f64,
    pub baseline_ns: u64,
    pub candidate_ns: u64,
}

impl BenchComparison {
    pub fn from_results(
        baseline: &BenchResult,
        candidate: &BenchResult,
    ) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let speedup_ratio = if baseline.elapsed_ns == 0 {
            0.0
        } else {
            candidate.elapsed_ns as f64 / baseline.elapsed_ns as f64
        };
        Self {
            case_name: baseline.case_name.clone(),
            baseline_backend: baseline.backend.clone(),
            candidate_backend: candidate.backend.clone(),
            speedup_ratio,
            baseline_ns: baseline.elapsed_ns,
            candidate_ns: candidate.elapsed_ns,
        }
    }
}

// ── Regression detection ──────────────────────────────────────────────────

/// Detects performance regressions against a saved baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetector {
    /// Maximum allowed slowdown as a fraction (e.g. 0.10 = 10%).
    pub threshold_pct: f64,
    /// Saved baseline results keyed by `(backend, case_name)`.
    baselines: HashMap<String, u64>,
}

/// Outcome of checking one result against its baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionStatus {
    /// Within threshold.
    Ok { delta_pct: f64 },
    /// Slower than the threshold permits.
    Regression { delta_pct: f64 },
    /// Faster than baseline (negative delta).
    Improvement { delta_pct: f64 },
    /// No baseline exists for this case.
    NoBaseline,
}

impl RegressionDetector {
    /// Create a detector with the given threshold (0.0–1.0).
    pub fn new(threshold_pct: f64) -> Self {
        Self { threshold_pct, baselines: HashMap::new() }
    }

    /// Register a baseline timing.
    pub fn set_baseline(
        &mut self,
        backend: &str,
        case_name: &str,
        elapsed_ns: u64,
    ) {
        let key = format!("{backend}::{case_name}");
        self.baselines.insert(key, elapsed_ns);
    }

    /// Load baselines from JSON bytes.
    pub fn load_baselines(
        &mut self,
        json: &[u8],
    ) -> Result<usize, serde_json::Error> {
        let map: HashMap<String, u64> = serde_json::from_slice(json)?;
        let count = map.len();
        self.baselines.extend(map);
        Ok(count)
    }

    /// Save current baselines to JSON.
    pub fn save_baselines(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.baselines)
    }

    /// Check a result against the stored baseline.
    pub fn check(
        &self,
        backend: &str,
        case_name: &str,
        elapsed_ns: u64,
    ) -> RegressionStatus {
        let key = format!("{backend}::{case_name}");
        let Some(&baseline_ns) = self.baselines.get(&key) else {
            return RegressionStatus::NoBaseline;
        };
        if baseline_ns == 0 {
            return RegressionStatus::Ok { delta_pct: 0.0 };
        }
        #[allow(clippy::cast_precision_loss)]
        let delta_pct =
            (elapsed_ns as f64 - baseline_ns as f64) / baseline_ns as f64;
        if delta_pct > self.threshold_pct {
            RegressionStatus::Regression { delta_pct }
        } else if delta_pct < -self.threshold_pct {
            RegressionStatus::Improvement { delta_pct }
        } else {
            RegressionStatus::Ok { delta_pct }
        }
    }

    /// Number of baselines stored.
    pub fn baseline_count(&self) -> usize {
        self.baselines.len()
    }
}

// ── Report ────────────────────────────────────────────────────────────────

/// Output format for benchmark reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Json,
    Csv,
    Table,
}

/// Structured benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub results: Vec<BenchResult>,
    pub comparisons: Vec<BenchComparison>,
    pub histograms: HashMap<String, LatencyHistogram>,
    pub regressions: Vec<(String, RegressionStatus)>,
}

impl BenchmarkReport {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            comparisons: Vec::new(),
            histograms: HashMap::new(),
            regressions: Vec::new(),
        }
    }

    /// Render the report in the requested format.
    pub fn render(&self, format: ReportFormat) -> String {
        match format {
            ReportFormat::Json => self.render_json(),
            ReportFormat::Csv => self.render_csv(),
            ReportFormat::Table => self.render_table(),
        }
    }

    fn render_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    fn render_csv(&self) -> String {
        let mut out =
            String::from("backend,case,elapsed_ns,throughput_gops\n");
        for r in &self.results {
            let _ = writeln!(
                out,
                "{},{},{},{:.4}",
                r.backend, r.case_name, r.elapsed_ns, r.throughput_gops,
            );
        }
        out
    }

    fn render_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:<12} {:<24} {:>14} {:>12}",
            "Backend", "Case", "Elapsed", "GOPS",
        );
        out.push_str(&"-".repeat(66));
        out.push('\n');
        for r in &self.results {
            let _ = writeln!(
                out,
                "{:<12} {:<24} {:>14} {:>12.4}",
                r.backend,
                r.case_name,
                format_ns(r.elapsed_ns),
                r.throughput_gops,
            );
        }
        if !self.comparisons.is_empty() {
            out.push('\n');
            let _ = writeln!(
                out,
                "{:<24} {:<12} {:<12} {:>10}",
                "Case", "Baseline", "Candidate", "Ratio",
            );
            out.push_str(&"-".repeat(62));
            out.push('\n');
            for c in &self.comparisons {
                let _ = writeln!(
                    out,
                    "{:<24} {:<12} {:<12} {:>10.3}",
                    c.case_name,
                    c.baseline_backend,
                    c.candidate_backend,
                    c.speedup_ratio,
                );
            }
        }
        out
    }
}

impl Default for BenchmarkReport {
    fn default() -> Self {
        Self::new()
    }
}

// ── Standard scenarios ────────────────────────────────────────────────────

/// Return the suite of standard benchmark scenarios.
pub fn standard_scenarios() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase::new(
            "matmul_128x128",
            vec![128, 128],
            OpKind::Matmul,
        ),
        BenchmarkCase::new(
            "matmul_512x512",
            vec![512, 512],
            OpKind::Matmul,
        ),
        BenchmarkCase::new(
            "matmul_1024x1024",
            vec![1024, 1024],
            OpKind::Matmul,
        ),
        BenchmarkCase::new(
            "matmul_2048x2048",
            vec![2048, 2048],
            OpKind::Matmul,
        ),
        BenchmarkCase::new(
            "attention_seq64_dim64",
            vec![64, 64],
            OpKind::Attention,
        ),
        BenchmarkCase::new(
            "attention_seq256_dim128",
            vec![256, 128],
            OpKind::Attention,
        ),
        BenchmarkCase::new(
            "softmax_1024",
            vec![1024],
            OpKind::Softmax,
        ),
        BenchmarkCase::new(
            "softmax_32768",
            vec![32768],
            OpKind::Softmax,
        ),
        BenchmarkCase::new(
            "quantize_4096",
            vec![4096],
            OpKind::Quantize,
        ),
        BenchmarkCase::new(
            "quantize_16384",
            vec![16384],
            OpKind::Quantize,
        ),
        BenchmarkCase::new(
            "tokenize_128",
            vec![128],
            OpKind::Tokenize,
        ),
        BenchmarkCase::new(
            "elementwise_8192",
            vec![8192],
            OpKind::Elementwise,
        ),
        BenchmarkCase::new(
            "reduce_8192",
            vec![8192],
            OpKind::Reduce,
        ),
        BenchmarkCase::new(
            "rms_norm_4096",
            vec![4096],
            OpKind::RmsNorm,
        ),
        BenchmarkCase::new("rope_dim128", vec![128], OpKind::Rope),
    ]
}

// ── Workload simulators ───────────────────────────────────────────────────

/// Simulate a matmul workload (dot products of size `n`).
#[allow(clippy::cast_possible_truncation)]
fn simulate_matmul(sizes: &[usize]) -> u64 {
    let n = sizes.first().copied().unwrap_or(128);
    let a = vec![1.0_f32; n * n];
    let b = vec![1.0_f32; n * n];
    let mut c = vec![0.0_f32; n * n];
    let start = Instant::now();
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    let _ = c; // prevent optimizing away
    start.elapsed().as_nanos() as u64
}

/// Simulate a softmax workload.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn simulate_softmax(sizes: &[usize]) -> u64 {
    let n = sizes.first().copied().unwrap_or(1024);
    let mut logits: Vec<f32> =
        (0..n).map(|i| (i as f32) * 0.01).collect();
    let start = Instant::now();
    crate::softmax(&mut logits);
    start.elapsed().as_nanos() as u64
}

/// Simulate an elementwise workload (scale + bias).
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn simulate_elementwise(sizes: &[usize]) -> u64 {
    let n = sizes.first().copied().unwrap_or(8192);
    let mut data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let start = Instant::now();
    for v in &mut data {
        *v = (*v).mul_add(2.0, 1.0);
    }
    let _ = data;
    start.elapsed().as_nanos() as u64
}

/// Simulate a reduce (sum) workload.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn simulate_reduce(sizes: &[usize]) -> u64 {
    let n = sizes.first().copied().unwrap_or(8192);
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let start = Instant::now();
    let _sum: f32 = data.iter().sum();
    start.elapsed().as_nanos() as u64
}

/// Simulate a quantize workload.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn simulate_quantize(sizes: &[usize]) -> u64 {
    let n = sizes.first().copied().unwrap_or(4096);
    let values: Vec<f32> =
        (0..n).map(|i| (i as f32) * 0.001).collect();
    let start = Instant::now();
    let _ = crate::ternary_quantize(&values);
    start.elapsed().as_nanos() as u64
}

/// Simulate attention workload.
#[allow(clippy::cast_possible_truncation)]
fn simulate_attention(sizes: &[usize]) -> u64 {
    let seq_len = sizes.first().copied().unwrap_or(64);
    let head_dim = sizes.get(1).copied().unwrap_or(64);
    let total = seq_len * head_dim;
    let q = vec![1.0_f32; total];
    let k = vec![1.0_f32; total];
    let v = vec![1.0_f32; total];
    let start = Instant::now();
    let _ = crate::attention_forward(&q, &k, &v, seq_len, head_dim);
    start.elapsed().as_nanos() as u64
}

/// Simulate RMS norm workload.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn simulate_rms_norm(sizes: &[usize]) -> u64 {
    let n = sizes.first().copied().unwrap_or(4096);
    let mut x: Vec<f32> =
        (0..n).map(|i| i as f32 * 0.01).collect();
    let w = vec![1.0_f32; n];
    let start = Instant::now();
    crate::rms_norm(&mut x, &w, 1e-5);
    start.elapsed().as_nanos() as u64
}

/// Simulate `RoPE` workload.
#[allow(clippy::cast_possible_truncation)]
fn simulate_rope(sizes: &[usize]) -> u64 {
    let dim = sizes.first().copied().unwrap_or(128);
    let seq_len = 64;
    let start = Instant::now();
    let _ = crate::build_rope_tables(dim, seq_len, 10000.0);
    start.elapsed().as_nanos() as u64
}

/// Dispatch a workload simulation by `OpKind`.
fn run_workload(op: OpKind, sizes: &[usize]) -> u64 {
    match op {
        OpKind::Matmul => simulate_matmul(sizes),
        OpKind::Softmax => simulate_softmax(sizes),
        OpKind::Elementwise | OpKind::Tokenize => {
            simulate_elementwise(sizes)
        }
        OpKind::Reduce => simulate_reduce(sizes),
        OpKind::Quantize | OpKind::Dequantize => {
            simulate_quantize(sizes)
        }
        OpKind::Attention => simulate_attention(sizes),
        OpKind::RmsNorm => simulate_rms_norm(sizes),
        OpKind::Rope => simulate_rope(sizes),
    }
}

// ── BenchmarkSuite ────────────────────────────────────────────────────────

/// Orchestrates cross-backend benchmark execution.
pub struct BenchmarkSuite {
    config: BenchConfig,
    cases: Vec<BenchmarkCase>,
    detector: RegressionDetector,
}

impl BenchmarkSuite {
    /// Create a new suite with the given config.
    pub fn new(config: BenchConfig) -> Self {
        Self {
            config,
            cases: Vec::new(),
            detector: RegressionDetector::new(0.10),
        }
    }

    /// Create a suite with default standard scenarios.
    pub fn with_standard_scenarios(config: BenchConfig) -> Self {
        Self {
            config,
            cases: standard_scenarios(),
            detector: RegressionDetector::new(0.10),
        }
    }

    /// Add a benchmark case.
    pub fn add_case(&mut self, case: BenchmarkCase) {
        self.cases.push(case);
    }

    /// Number of registered cases.
    pub const fn case_count(&self) -> usize {
        self.cases.len()
    }

    /// Access the regression detector.
    pub const fn detector(&self) -> &RegressionDetector {
        &self.detector
    }

    /// Mutable access to the regression detector.
    pub const fn detector_mut(&mut self) -> &mut RegressionDetector {
        &mut self.detector
    }

    /// Set regression threshold.
    pub const fn set_regression_threshold(&mut self, pct: f64) {
        self.detector.threshold_pct = pct;
    }

    /// Run all benchmark cases on all configured backends.
    pub fn run(&self) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();

        for case in &self.cases {
            for backend in &self.config.backends_to_test {
                let result =
                    self.run_single(backend, case);

                // Build histogram
                if let Some(hist) = LatencyHistogram::from_samples(
                    &result.latency_samples_ns,
                ) {
                    let key =
                        format!("{}::{}", result.backend, result.case_name);
                    report.histograms.insert(key, hist);
                }

                // Check regression
                let status = self.detector.check(
                    &result.backend,
                    &result.case_name,
                    result.elapsed_ns,
                );
                let key =
                    format!("{}::{}", result.backend, result.case_name);
                report.regressions.push((key, status));

                report.results.push(result);
            }
        }

        // Build comparisons if >1 backend
        if self.config.backends_to_test.len() >= 2 {
            let baseline_backend = &self.config.backends_to_test[0];
            for case in &self.cases {
                let baseline_result = report.results.iter().find(|r| {
                    r.backend == *baseline_backend
                        && r.case_name == case.name
                });
                for other in &self.config.backends_to_test[1..] {
                    let cand_result = report.results.iter().find(|r| {
                        r.backend == *other && r.case_name == case.name
                    });
                    if let (Some(b), Some(c)) =
                        (baseline_result, cand_result)
                    {
                        report.comparisons.push(
                            BenchComparison::from_results(b, c),
                        );
                    }
                }
            }
        }

        report
    }

    /// Run a single case on one backend.
    fn run_single(
        &self,
        backend: &str,
        case: &BenchmarkCase,
    ) -> BenchResult {
        // Warm-up
        for _ in 0..self.config.warmup_iterations {
            let _ = run_workload(case.expected_op, &case.input_sizes);
        }

        // Measurement
        let mut samples = Vec::with_capacity(
            self.config.measurement_iterations,
        );
        let overall_start = Instant::now();
        for _ in 0..self.config.measurement_iterations {
            let ns = run_workload(case.expected_op, &case.input_sizes);
            samples.push(ns);
        }
        #[allow(clippy::cast_possible_truncation)]
        let total_ns = overall_start.elapsed().as_nanos() as u64;

        // Compute throughput for matmul (2*N^3 FLOPs)
        #[allow(clippy::cast_precision_loss)]
        let throughput_gops = if case.expected_op == OpKind::Matmul {
            let n =
                case.input_sizes.first().copied().unwrap_or(1) as f64;
            let flops = 2.0 * n * n * n;
            let iters = self.config.measurement_iterations as f64;
            let total_flops = flops * iters;
            let secs = total_ns as f64 / 1e9;
            if secs > 0.0 { total_flops / secs / 1e9 } else { 0.0 }
        } else {
            0.0
        };

        BenchResult {
            backend: backend.to_string(),
            case_name: case.name.clone(),
            elapsed_ns: total_ns,
            throughput_gops,
            memory_peak_bytes: 0,
            latency_samples_ns: samples,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- BenchConfig tests --

    #[test]
    fn config_default_has_cpu() {
        let cfg = BenchConfig::default();
        assert_eq!(cfg.backends_to_test, vec!["cpu"]);
        assert!(cfg.warmup_iterations > 0);
        assert!(cfg.measurement_iterations > 0);
    }

    #[test]
    fn config_custom_backends() {
        let cfg = BenchConfig {
            warmup_iterations: 5,
            measurement_iterations: 20,
            backends_to_test: vec!["cpu".into(), "cuda".into()],
        };
        assert_eq!(cfg.backends_to_test.len(), 2);
        assert_eq!(cfg.measurement_iterations, 20);
    }

    #[test]
    fn config_serialization_roundtrip() {
        let cfg = BenchConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: BenchConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.warmup_iterations, cfg2.warmup_iterations);
    }

    // -- OpKind tests --

    #[test]
    fn op_kind_display() {
        assert_eq!(OpKind::Matmul.to_string(), "matmul");
        assert_eq!(OpKind::Softmax.to_string(), "softmax");
        assert_eq!(OpKind::Attention.to_string(), "attention");
        assert_eq!(OpKind::RmsNorm.to_string(), "rms_norm");
    }

    #[test]
    fn op_kind_equality() {
        assert_eq!(OpKind::Matmul, OpKind::Matmul);
        assert_ne!(OpKind::Matmul, OpKind::Reduce);
    }

    #[test]
    fn op_kind_hash() {
        let mut map = HashMap::new();
        map.insert(OpKind::Matmul, 42);
        assert_eq!(map.get(&OpKind::Matmul), Some(&42));
    }

    // -- BenchmarkCase tests --

    #[test]
    fn case_construction() {
        let c =
            BenchmarkCase::new("test_case", vec![64, 64], OpKind::Matmul);
        assert_eq!(c.name, "test_case");
        assert_eq!(c.input_sizes, vec![64, 64]);
        assert_eq!(c.expected_op, OpKind::Matmul);
    }

    #[test]
    fn case_serialization_roundtrip() {
        let c =
            BenchmarkCase::new("matmul_64", vec![64], OpKind::Matmul);
        let json = serde_json::to_string(&c).unwrap();
        let c2: BenchmarkCase = serde_json::from_str(&json).unwrap();
        assert_eq!(c.name, c2.name);
    }

    // -- LatencyHistogram tests --

    #[test]
    fn histogram_empty_returns_none() {
        assert!(LatencyHistogram::from_samples(&[]).is_none());
    }

    #[test]
    fn histogram_single_sample() {
        let h = LatencyHistogram::from_samples(&[1000]).unwrap();
        assert_eq!(h.p50_ns, 1000);
        assert_eq!(h.p90_ns, 1000);
        assert_eq!(h.p99_ns, 1000);
        assert_eq!(h.max_ns, 1000);
        assert_eq!(h.min_ns, 1000);
        assert_eq!(h.count, 1);
    }

    #[test]
    fn histogram_sorted_samples() {
        let samples: Vec<u64> = (1..=100).collect();
        let h = LatencyHistogram::from_samples(&samples).unwrap();
        assert_eq!(h.min_ns, 1);
        assert_eq!(h.max_ns, 100);
        assert_eq!(h.count, 100);
        assert_eq!(h.p50_ns, 50);
        assert_eq!(h.p90_ns, 90);
        assert_eq!(h.p99_ns, 99);
    }

    #[test]
    fn histogram_unsorted_samples() {
        let samples = vec![50, 10, 90, 30, 70];
        let h = LatencyHistogram::from_samples(&samples).unwrap();
        assert_eq!(h.min_ns, 10);
        assert_eq!(h.max_ns, 90);
        assert_eq!(h.count, 5);
    }

    #[test]
    fn histogram_mean_calculation() {
        let samples = vec![10, 20, 30];
        let h = LatencyHistogram::from_samples(&samples).unwrap();
        assert!((h.mean_ns - 20.0).abs() < 0.01);
    }

    #[test]
    fn histogram_display_formatting() {
        let h = LatencyHistogram::from_samples(&[1000]).unwrap();
        let s = h.to_string();
        assert!(s.contains("P50="));
        assert!(s.contains("P90="));
        assert!(s.contains("P99="));
    }

    #[test]
    fn histogram_large_values() {
        let samples = vec![1_000_000_000, 2_000_000_000];
        let h = LatencyHistogram::from_samples(&samples).unwrap();
        assert_eq!(h.min_ns, 1_000_000_000);
        assert_eq!(h.max_ns, 2_000_000_000);
    }

    #[test]
    fn histogram_all_same() {
        let samples = vec![500; 50];
        let h = LatencyHistogram::from_samples(&samples).unwrap();
        assert_eq!(h.p50_ns, 500);
        assert_eq!(h.p90_ns, 500);
        assert_eq!(h.max_ns, 500);
    }

    // -- format_ns tests --

    #[test]
    fn format_ns_nanoseconds() {
        assert_eq!(format_ns(42), "42ns");
    }

    #[test]
    fn format_ns_microseconds() {
        let s = format_ns(5_500);
        assert!(s.contains("µs"));
    }

    #[test]
    fn format_ns_milliseconds() {
        let s = format_ns(2_500_000);
        assert!(s.contains("ms"));
    }

    #[test]
    fn format_ns_seconds() {
        let s = format_ns(1_500_000_000);
        assert!(s.contains('s'));
    }

    // -- BenchComparison tests --

    #[test]
    fn comparison_from_results() {
        let b = BenchResult {
            backend: "cpu".into(),
            case_name: "matmul".into(),
            elapsed_ns: 1000,
            throughput_gops: 0.0,
            memory_peak_bytes: 0,
            latency_samples_ns: vec![],
        };
        let c = BenchResult {
            backend: "cuda".into(),
            case_name: "matmul".into(),
            elapsed_ns: 500,
            throughput_gops: 0.0,
            memory_peak_bytes: 0,
            latency_samples_ns: vec![],
        };
        let cmp = BenchComparison::from_results(&b, &c);
        assert_eq!(cmp.speedup_ratio, 0.5);
        assert_eq!(cmp.baseline_backend, "cpu");
        assert_eq!(cmp.candidate_backend, "cuda");
    }

    #[test]
    fn comparison_zero_baseline() {
        let b = BenchResult {
            backend: "cpu".into(),
            case_name: "x".into(),
            elapsed_ns: 0,
            throughput_gops: 0.0,
            memory_peak_bytes: 0,
            latency_samples_ns: vec![],
        };
        let c = BenchResult {
            backend: "cuda".into(),
            case_name: "x".into(),
            elapsed_ns: 100,
            throughput_gops: 0.0,
            memory_peak_bytes: 0,
            latency_samples_ns: vec![],
        };
        let cmp = BenchComparison::from_results(&b, &c);
        assert_eq!(cmp.speedup_ratio, 0.0);
    }

    #[test]
    fn comparison_equal_times() {
        let b = BenchResult {
            backend: "a".into(),
            case_name: "t".into(),
            elapsed_ns: 100,
            throughput_gops: 0.0,
            memory_peak_bytes: 0,
            latency_samples_ns: vec![],
        };
        let cmp = BenchComparison::from_results(&b, &b);
        assert!((cmp.speedup_ratio - 1.0).abs() < f64::EPSILON);
    }

    // -- RegressionDetector tests --

    #[test]
    fn detector_no_baseline() {
        let det = RegressionDetector::new(0.10);
        let s = det.check("cpu", "matmul", 100);
        assert_eq!(s, RegressionStatus::NoBaseline);
    }

    #[test]
    fn detector_within_threshold() {
        let mut det = RegressionDetector::new(0.10);
        det.set_baseline("cpu", "matmul", 1000);
        let s = det.check("cpu", "matmul", 1050);
        assert!(matches!(s, RegressionStatus::Ok { .. }));
    }

    #[test]
    fn detector_regression() {
        let mut det = RegressionDetector::new(0.10);
        det.set_baseline("cpu", "matmul", 1000);
        let s = det.check("cpu", "matmul", 1200);
        assert!(matches!(s, RegressionStatus::Regression { .. }));
    }

    #[test]
    fn detector_improvement() {
        let mut det = RegressionDetector::new(0.10);
        det.set_baseline("cpu", "matmul", 1000);
        let s = det.check("cpu", "matmul", 800);
        assert!(matches!(s, RegressionStatus::Improvement { .. }));
    }

    #[test]
    fn detector_exact_threshold() {
        let mut det = RegressionDetector::new(0.10);
        det.set_baseline("cpu", "matmul", 1000);
        // Exactly at threshold boundary
        let s = det.check("cpu", "matmul", 1100);
        assert!(matches!(s, RegressionStatus::Ok { .. }));
    }

    #[test]
    fn detector_zero_baseline_is_ok() {
        let mut det = RegressionDetector::new(0.10);
        det.set_baseline("cpu", "matmul", 0);
        let s = det.check("cpu", "matmul", 500);
        assert!(matches!(s, RegressionStatus::Ok { .. }));
    }

    #[test]
    fn detector_save_load_roundtrip() {
        let mut det = RegressionDetector::new(0.10);
        det.set_baseline("cpu", "matmul_128", 5000);
        det.set_baseline("cpu", "softmax_1k", 2000);
        let json = det.save_baselines().unwrap();
        let mut det2 = RegressionDetector::new(0.10);
        let count = det2.load_baselines(json.as_bytes()).unwrap();
        assert_eq!(count, 2);
        let s = det2.check("cpu", "matmul_128", 5000);
        assert!(matches!(s, RegressionStatus::Ok { .. }));
    }

    #[test]
    fn detector_baseline_count() {
        let mut det = RegressionDetector::new(0.10);
        assert_eq!(det.baseline_count(), 0);
        det.set_baseline("cpu", "a", 100);
        det.set_baseline("cpu", "b", 200);
        assert_eq!(det.baseline_count(), 2);
    }

    #[test]
    fn detector_overwrite_baseline() {
        let mut det = RegressionDetector::new(0.10);
        det.set_baseline("cpu", "case", 1000);
        det.set_baseline("cpu", "case", 2000);
        assert_eq!(det.baseline_count(), 1);
        // New baseline is 2000, so 2100 should be ok (5%)
        let s = det.check("cpu", "case", 2100);
        assert!(matches!(s, RegressionStatus::Ok { .. }));
    }

    #[test]
    fn detector_different_backends_are_independent() {
        let mut det = RegressionDetector::new(0.10);
        det.set_baseline("cpu", "case", 1000);
        det.set_baseline("cuda", "case", 500);
        assert_eq!(det.baseline_count(), 2);
        let s = det.check("cuda", "case", 550);
        assert!(matches!(s, RegressionStatus::Ok { .. }));
    }

    #[test]
    fn detector_delta_pct_accuracy() {
        let mut det = RegressionDetector::new(0.20);
        det.set_baseline("cpu", "case", 1000);
        match det.check("cpu", "case", 1150) {
            RegressionStatus::Ok { delta_pct } => {
                assert!((delta_pct - 0.15).abs() < 0.001);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    // -- BenchmarkReport tests --

    #[test]
    fn report_new_is_empty() {
        let r = BenchmarkReport::new();
        assert!(r.results.is_empty());
        assert!(r.comparisons.is_empty());
        assert!(r.histograms.is_empty());
    }

    #[test]
    fn report_default_is_empty() {
        let r = BenchmarkReport::default();
        assert!(r.results.is_empty());
    }

    #[test]
    fn report_render_json() {
        let r = BenchmarkReport::new();
        let json = r.render(ReportFormat::Json);
        assert!(json.contains("results"));
    }

    #[test]
    fn report_render_csv_header() {
        let r = BenchmarkReport::new();
        let csv = r.render(ReportFormat::Csv);
        assert!(csv.starts_with("backend,case,elapsed_ns,"));
    }

    #[test]
    fn report_render_csv_with_data() {
        let mut r = BenchmarkReport::new();
        r.results.push(BenchResult {
            backend: "cpu".into(),
            case_name: "matmul".into(),
            elapsed_ns: 1000,
            throughput_gops: 1.5,
            memory_peak_bytes: 0,
            latency_samples_ns: vec![],
        });
        let csv = r.render(ReportFormat::Csv);
        assert!(csv.contains("cpu,matmul,1000,1.5000"));
    }

    #[test]
    fn report_render_table_header() {
        let r = BenchmarkReport::new();
        let tbl = r.render(ReportFormat::Table);
        assert!(tbl.contains("Backend"));
        assert!(tbl.contains("Case"));
        assert!(tbl.contains("GOPS"));
    }

    #[test]
    fn report_render_table_with_data() {
        let mut r = BenchmarkReport::new();
        r.results.push(BenchResult {
            backend: "cpu".into(),
            case_name: "softmax".into(),
            elapsed_ns: 500,
            throughput_gops: 0.0,
            memory_peak_bytes: 0,
            latency_samples_ns: vec![],
        });
        let tbl = r.render(ReportFormat::Table);
        assert!(tbl.contains("cpu"));
        assert!(tbl.contains("softmax"));
    }

    #[test]
    fn report_render_table_with_comparisons() {
        let mut r = BenchmarkReport::new();
        r.comparisons.push(BenchComparison {
            case_name: "matmul".into(),
            baseline_backend: "cpu".into(),
            candidate_backend: "cuda".into(),
            speedup_ratio: 0.5,
            baseline_ns: 1000,
            candidate_ns: 500,
        });
        let tbl = r.render(ReportFormat::Table);
        assert!(tbl.contains("Baseline"));
        assert!(tbl.contains("Candidate"));
    }

    #[test]
    fn report_json_roundtrip() {
        let mut r = BenchmarkReport::new();
        r.results.push(BenchResult {
            backend: "cpu".into(),
            case_name: "test".into(),
            elapsed_ns: 42,
            throughput_gops: 1.0,
            memory_peak_bytes: 0,
            latency_samples_ns: vec![10, 20],
        });
        let json = r.render(ReportFormat::Json);
        let r2: BenchmarkReport =
            serde_json::from_str(&json).unwrap();
        assert_eq!(r2.results.len(), 1);
        assert_eq!(r2.results[0].elapsed_ns, 42);
    }

    // -- standard_scenarios tests --

    #[test]
    fn standard_scenarios_not_empty() {
        let s = standard_scenarios();
        assert!(!s.is_empty());
    }

    #[test]
    fn standard_scenarios_have_matmul() {
        let s = standard_scenarios();
        assert!(s.iter().any(|c| c.expected_op == OpKind::Matmul));
    }

    #[test]
    fn standard_scenarios_have_attention() {
        let s = standard_scenarios();
        assert!(s.iter().any(|c| c.expected_op == OpKind::Attention));
    }

    #[test]
    fn standard_scenarios_have_unique_names() {
        let s = standard_scenarios();
        let mut names: Vec<&str> =
            s.iter().map(|c| c.name.as_str()).collect();
        let orig_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), orig_len);
    }

    #[test]
    fn standard_scenarios_sizes_nonempty() {
        let s = standard_scenarios();
        for case in &s {
            assert!(
                !case.input_sizes.is_empty(),
                "case {} has empty sizes",
                case.name,
            );
        }
    }

    // -- BenchmarkSuite tests --

    #[test]
    fn suite_new_empty() {
        let s = BenchmarkSuite::new(BenchConfig::default());
        assert_eq!(s.case_count(), 0);
    }

    #[test]
    fn suite_with_standard_scenarios() {
        let s = BenchmarkSuite::with_standard_scenarios(
            BenchConfig::default(),
        );
        assert!(s.case_count() > 0);
    }

    #[test]
    fn suite_add_case() {
        let mut s = BenchmarkSuite::new(BenchConfig::default());
        s.add_case(BenchmarkCase::new(
            "custom",
            vec![32],
            OpKind::Reduce,
        ));
        assert_eq!(s.case_count(), 1);
    }

    #[test]
    fn suite_run_empty() {
        let s = BenchmarkSuite::new(BenchConfig::default());
        let r = s.run();
        assert!(r.results.is_empty());
    }

    #[test]
    fn suite_run_single_case() {
        let cfg = BenchConfig {
            warmup_iterations: 1,
            measurement_iterations: 3,
            backends_to_test: vec!["cpu".into()],
        };
        let mut s = BenchmarkSuite::new(cfg);
        s.add_case(BenchmarkCase::new(
            "softmax_small",
            vec![64],
            OpKind::Softmax,
        ));
        let r = s.run();
        assert_eq!(r.results.len(), 1);
        assert!(r.results[0].elapsed_ns > 0);
        assert_eq!(r.results[0].latency_samples_ns.len(), 3);
    }

    #[test]
    fn suite_run_produces_histograms() {
        let cfg = BenchConfig {
            warmup_iterations: 1,
            measurement_iterations: 5,
            backends_to_test: vec!["cpu".into()],
        };
        let mut s = BenchmarkSuite::new(cfg);
        s.add_case(BenchmarkCase::new(
            "reduce_small",
            vec![128],
            OpKind::Reduce,
        ));
        let r = s.run();
        assert_eq!(r.histograms.len(), 1);
        let hist = r.histograms.get("cpu::reduce_small").unwrap();
        assert_eq!(hist.count, 5);
    }

    #[test]
    fn suite_run_multi_backend() {
        let cfg = BenchConfig {
            warmup_iterations: 1,
            measurement_iterations: 2,
            backends_to_test: vec!["cpu".into(), "mock_gpu".into()],
        };
        let mut s = BenchmarkSuite::new(cfg);
        s.add_case(BenchmarkCase::new(
            "elem_tiny",
            vec![16],
            OpKind::Elementwise,
        ));
        let r = s.run();
        assert_eq!(r.results.len(), 2);
        assert_eq!(r.comparisons.len(), 1);
    }

    #[test]
    fn suite_set_regression_threshold() {
        let mut s = BenchmarkSuite::new(BenchConfig::default());
        s.set_regression_threshold(0.05);
        assert!((s.detector().threshold_pct - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn suite_detector_integration() {
        let cfg = BenchConfig {
            warmup_iterations: 1,
            measurement_iterations: 2,
            backends_to_test: vec!["cpu".into()],
        };
        let mut s = BenchmarkSuite::new(cfg);
        s.add_case(BenchmarkCase::new(
            "elem_tiny",
            vec![16],
            OpKind::Elementwise,
        ));
        // No baselines → all NoBaseline
        let r = s.run();
        assert!(r.regressions.iter().all(|(_, st)| {
            matches!(st, RegressionStatus::NoBaseline)
        }));
    }

    #[test]
    fn suite_matmul_throughput_positive() {
        let cfg = BenchConfig {
            warmup_iterations: 0,
            measurement_iterations: 2,
            backends_to_test: vec!["cpu".into()],
        };
        let mut s = BenchmarkSuite::new(cfg);
        s.add_case(BenchmarkCase::new(
            "matmul_tiny",
            vec![16, 16],
            OpKind::Matmul,
        ));
        let r = s.run();
        assert!(r.results[0].throughput_gops > 0.0);
    }

    #[test]
    fn suite_non_matmul_throughput_zero() {
        let cfg = BenchConfig {
            warmup_iterations: 0,
            measurement_iterations: 2,
            backends_to_test: vec!["cpu".into()],
        };
        let mut s = BenchmarkSuite::new(cfg);
        s.add_case(BenchmarkCase::new(
            "reduce_tiny",
            vec![64],
            OpKind::Reduce,
        ));
        let r = s.run();
        assert!((r.results[0].throughput_gops - 0.0).abs() < f64::EPSILON);
    }

    // -- Workload simulation tests --

    #[test]
    fn workload_matmul_runs() {
        let ns = run_workload(OpKind::Matmul, &[8, 8]);
        assert!(ns > 0);
    }

    #[test]
    fn workload_softmax_runs() {
        let ns = run_workload(OpKind::Softmax, &[64]);
        assert!(ns > 0);
    }

    #[test]
    fn workload_elementwise_runs() {
        let ns = run_workload(OpKind::Elementwise, &[128]);
        assert!(ns > 0);
    }

    #[test]
    fn workload_reduce_runs() {
        let ns = run_workload(OpKind::Reduce, &[128]);
        assert!(ns > 0);
    }

    #[test]
    fn workload_quantize_runs() {
        let ns = run_workload(OpKind::Quantize, &[128]);
        assert!(ns > 0);
    }

    #[test]
    fn workload_attention_runs() {
        let ns = run_workload(OpKind::Attention, &[8, 8]);
        assert!(ns > 0);
    }

    #[test]
    fn workload_rms_norm_runs() {
        let ns = run_workload(OpKind::RmsNorm, &[64]);
        assert!(ns > 0);
    }

    #[test]
    fn workload_rope_runs() {
        let ns = run_workload(OpKind::Rope, &[64]);
        assert!(ns > 0);
    }

    #[test]
    fn workload_tokenize_runs() {
        let ns = run_workload(OpKind::Tokenize, &[64]);
        assert!(ns > 0);
    }

    #[test]
    fn workload_dequantize_runs() {
        let ns = run_workload(OpKind::Dequantize, &[128]);
        assert!(ns > 0);
    }

    // -- percentile helper tests --

    #[test]
    fn percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0);
    }

    #[test]
    fn percentile_single() {
        assert_eq!(percentile(&[42], 50.0), 42);
        assert_eq!(percentile(&[42], 99.0), 42);
    }

    #[test]
    fn percentile_two_elements() {
        assert_eq!(percentile(&[10, 20], 50.0), 10);
        assert_eq!(percentile(&[10, 20], 100.0), 20);
    }
}
