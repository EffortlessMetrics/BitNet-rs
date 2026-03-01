//! Module stub - implementation pending merge from feature branch
//! Cross-backend performance comparison and analysis.
//!
//! Provides infrastructure for benchmarking operations across GPU/CPU backends,
//! computing speedup ratios, analyzing memory efficiency, measuring scalability,
//! and generating comparison reports in multiple formats.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::missing_const_for_fn
)]

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Backend & metric identifiers
// ---------------------------------------------------------------------------

/// Identifies a compute backend for comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BackendId {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    OpenCl,
    Custom(String),
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda => write!(f, "CUDA"),
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
            Self::OpenCl => write!(f, "OpenCL"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

/// Which metric to collect during a benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricKind {
    Latency,
    Throughput,
    PeakMemory,
    MemoryBandwidth,
    ComputeUtilization,
    ArithmeticIntensity,
}

/// Operation type being benchmarked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationType {
    MatMul,
    Attention,
    Softmax,
    LayerNorm,
    Quantize,
    Dequantize,
    RoPE,
    Custom(String),
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMul => write!(f, "MatMul"),
            Self::Attention => write!(f, "Attention"),
            Self::Softmax => write!(f, "Softmax"),
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::Quantize => write!(f, "Quantize"),
            Self::Dequantize => write!(f, "Dequantize"),
            Self::RoPE => write!(f, "RoPE"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Top-level configuration for a cross-backend comparison run.
#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    /// Backends to include in the comparison.
    pub backends: Vec<BackendId>,
    /// Number of timed iterations per benchmark.
    pub iterations: usize,
    /// Number of warmup iterations before timing begins.
    pub warmup_iterations: usize,
    /// Metrics to collect.
    pub metrics: Vec<MetricKind>,
    /// Optional timeout per individual benchmark invocation.
    pub timeout: Option<Duration>,
    /// Reference backend for relative speedup (defaults to first).
    pub reference_backend: Option<BackendId>,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            backends: vec![BackendId::Cpu, BackendId::Cuda],
            iterations: 100,
            warmup_iterations: 10,
            metrics: vec![MetricKind::Latency, MetricKind::Throughput],
            timeout: Some(Duration::from_secs(60)),
            reference_backend: None,
        }
    }
}

impl ComparisonConfig {
    /// Create a new config with the given backends.
    pub fn new(backends: Vec<BackendId>) -> Self {
        Self { backends, ..Default::default() }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.backends.is_empty() {
            return Err("at least one backend is required".into());
        }
        if self.backends.len() < 2 {
            return Err("at least two backends are required for comparison".into());
        }
        if self.iterations == 0 {
            return Err("iterations must be > 0".into());
        }
        if self.reference_backend.as_ref().is_some_and(|rb| !self.backends.contains(rb)) {
            return Err(format!(
                "reference backend {} not in backend list",
                self.reference_backend.as_ref().unwrap()
            ));
        }
        Ok(())
    }

    /// Return the reference backend (explicit or first in list).
    pub fn effective_reference(&self) -> &BackendId {
        self.reference_backend.as_ref().unwrap_or(&self.backends[0])
    }
}

// ---------------------------------------------------------------------------
// Single-backend benchmark result
// ---------------------------------------------------------------------------

/// Timing and resource statistics from a single backend benchmark run.
#[derive(Debug, Clone)]
pub struct BackendBenchmark {
    pub backend: BackendId,
    pub operation: OperationType,
    /// Per-iteration durations (after warmup).
    pub durations: Vec<Duration>,
    /// Operations per second.
    pub throughput_ops: f64,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Bytes transferred during the benchmark.
    pub memory_bandwidth_bytes: u64,
    /// Compute utilization 0.0–1.0.
    pub compute_utilization: f64,
}

impl BackendBenchmark {
    /// Create from a set of durations, deriving throughput automatically.
    pub fn from_durations(
        backend: BackendId,
        operation: OperationType,
        durations: Vec<Duration>,
    ) -> Self {
        let throughput = if durations.is_empty() {
            0.0
        } else {
            let total: Duration = durations.iter().sum();
            let secs = total.as_secs_f64();
            if secs > 0.0 { durations.len() as f64 / secs } else { 0.0 }
        };
        Self {
            backend,
            operation,
            durations,
            throughput_ops: throughput,
            peak_memory_bytes: 0,
            memory_bandwidth_bytes: 0,
            compute_utilization: 0.0,
        }
    }

    /// Mean latency across all timed iterations.
    pub fn mean_latency(&self) -> Duration {
        if self.durations.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.durations.iter().sum();
        #[allow(clippy::cast_possible_truncation)]
        let count = self.durations.len() as u32;
        total / count
    }

    /// Median latency.
    pub fn median_latency(&self) -> Duration {
        if self.durations.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted: Vec<Duration> = self.durations.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    /// Standard deviation of latency in seconds.
    pub fn stddev_secs(&self) -> f64 {
        if self.durations.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_latency().as_secs_f64();
        let variance: f64 = self
            .durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.durations.len() - 1) as f64;
        variance.sqrt()
    }

    /// The p-th percentile latency (0–100).
    pub fn percentile(&self, p: f64) -> Duration {
        if self.durations.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted: Vec<Duration> = self.durations.clone();
        sorted.sort();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Coefficient of variation (stddev / mean).
    pub fn cv(&self) -> f64 {
        let mean = self.mean_latency().as_secs_f64();
        if mean == 0.0 {
            return 0.0;
        }
        self.stddev_secs() / mean
    }
}

// ---------------------------------------------------------------------------
// Operation benchmark
// ---------------------------------------------------------------------------

/// Holds benchmark results for a single operation across multiple backends.
#[derive(Debug, Clone)]
pub struct OperationBenchmark {
    pub operation: OperationType,
    pub results: HashMap<BackendId, BackendBenchmark>,
    /// Problem size descriptor (e.g. "M=1024 N=1024 K=1024").
    pub problem_size: String,
}

impl OperationBenchmark {
    pub fn new(operation: OperationType, problem_size: impl Into<String>) -> Self {
        Self { operation, results: HashMap::new(), problem_size: problem_size.into() }
    }

    /// Insert a result for a backend.
    pub fn add_result(&mut self, benchmark: BackendBenchmark) {
        self.results.insert(benchmark.backend.clone(), benchmark);
    }

    /// Return the fastest backend by mean latency.
    pub fn fastest_backend(&self) -> Option<&BackendId> {
        self.results.iter().min_by_key(|(_, b)| b.mean_latency()).map(|(id, _)| id)
    }

    /// Return the backend with highest throughput.
    pub fn highest_throughput_backend(&self) -> Option<&BackendId> {
        self.results
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.throughput_ops.partial_cmp(&b.throughput_ops).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id)
    }
}

// ---------------------------------------------------------------------------
// Comparison result
// ---------------------------------------------------------------------------

/// Side-by-side comparison of all operation benchmarks.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub config: ComparisonConfig,
    pub benchmarks: Vec<OperationBenchmark>,
    /// Wall-clock time for the entire comparison run.
    pub total_wall_time: Duration,
}

impl ComparisonResult {
    pub fn new(config: ComparisonConfig) -> Self {
        Self { config, benchmarks: Vec::new(), total_wall_time: Duration::ZERO }
    }

    pub fn add_benchmark(&mut self, benchmark: OperationBenchmark) {
        self.benchmarks.push(benchmark);
    }

    /// Get the benchmark for a specific operation.
    pub fn get_operation(&self, op: &OperationType) -> Option<&OperationBenchmark> {
        self.benchmarks.iter().find(|b| &b.operation == op)
    }

    /// Summary: for each operation, which backend was fastest.
    pub fn winners(&self) -> HashMap<OperationType, BackendId> {
        let mut map = HashMap::new();
        for bench in &self.benchmarks {
            if let Some(winner) = bench.fastest_backend() {
                map.insert(bench.operation.clone(), winner.clone());
            }
        }
        map
    }
}

// ---------------------------------------------------------------------------
// Speedup calculator
// ---------------------------------------------------------------------------

/// Computes relative speedup between a reference and target backend.
#[derive(Debug, Clone)]
pub struct SpeedupCalculator {
    pub reference: BackendId,
    pub target: BackendId,
}

/// Result of a speedup calculation for one operation.
#[derive(Debug, Clone)]
pub struct SpeedupResult {
    pub operation: OperationType,
    pub reference_backend: BackendId,
    pub target_backend: BackendId,
    /// > 1.0 means target is faster.
    pub speedup: f64,
    /// Absolute time saved per invocation.
    pub time_saved: Duration,
}

impl SpeedupCalculator {
    pub const fn new(reference: BackendId, target: BackendId) -> Self {
        Self { reference, target }
    }

    /// Compute speedup for a single operation benchmark.
    pub fn compute(&self, benchmark: &OperationBenchmark) -> Option<SpeedupResult> {
        let ref_bench = benchmark.results.get(&self.reference)?;
        let tgt_bench = benchmark.results.get(&self.target)?;
        let ref_mean = ref_bench.mean_latency().as_secs_f64();
        let tgt_mean = tgt_bench.mean_latency().as_secs_f64();
        if tgt_mean == 0.0 {
            return None;
        }
        let speedup = ref_mean / tgt_mean;
        let time_saved = ref_bench.mean_latency().saturating_sub(tgt_bench.mean_latency());
        Some(SpeedupResult {
            operation: benchmark.operation.clone(),
            reference_backend: self.reference.clone(),
            target_backend: self.target.clone(),
            speedup,
            time_saved,
        })
    }

    /// Compute speedup for all operations in a comparison result.
    pub fn compute_all(&self, result: &ComparisonResult) -> Vec<SpeedupResult> {
        result.benchmarks.iter().filter_map(|b| self.compute(b)).collect()
    }

    /// Geometric mean speedup across all operations.
    pub fn geometric_mean_speedup(speedups: &[SpeedupResult]) -> f64 {
        if speedups.is_empty() {
            return 1.0;
        }
        let product: f64 = speedups.iter().map(|s| s.speedup).product();
        product.powf(1.0 / speedups.len() as f64)
    }
}

// ---------------------------------------------------------------------------
// Memory efficiency
// ---------------------------------------------------------------------------

/// Memory usage snapshot for a backend.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub backend: BackendId,
    pub peak_bytes: u64,
    pub allocated_bytes: u64,
    pub bandwidth_bytes_per_sec: f64,
}

/// Compares memory usage across backends.
#[derive(Debug, Clone, Default)]
pub struct MemoryEfficiency {
    pub snapshots: Vec<MemorySnapshot>,
}

impl MemoryEfficiency {
    #[must_use]
    pub const fn new() -> Self {
        Self { snapshots: Vec::new() }
    }

    pub fn add_snapshot(&mut self, snapshot: MemorySnapshot) {
        self.snapshots.push(snapshot);
    }

    /// Return the backend with lowest peak memory.
    pub fn most_memory_efficient(&self) -> Option<&BackendId> {
        self.snapshots.iter().min_by_key(|s| s.peak_bytes).map(|s| &s.backend)
    }

    /// Ratio of peak memory: target / reference. < 1.0 means target uses less.
    pub fn memory_ratio(&self, reference: &BackendId, target: &BackendId) -> Option<f64> {
        let ref_snap = self.snapshots.iter().find(|s| &s.backend == reference)?;
        let tgt_snap = self.snapshots.iter().find(|s| &s.backend == target)?;
        if ref_snap.peak_bytes == 0 {
            return None;
        }
        Some(tgt_snap.peak_bytes as f64 / ref_snap.peak_bytes as f64)
    }

    /// Bandwidth efficiency ratio: target / reference.
    pub fn bandwidth_ratio(&self, reference: &BackendId, target: &BackendId) -> Option<f64> {
        let ref_snap = self.snapshots.iter().find(|s| &s.backend == reference)?;
        let tgt_snap = self.snapshots.iter().find(|s| &s.backend == target)?;
        if ref_snap.bandwidth_bytes_per_sec == 0.0 {
            return None;
        }
        Some(tgt_snap.bandwidth_bytes_per_sec / ref_snap.bandwidth_bytes_per_sec)
    }

    /// Total memory across all backends.
    pub fn total_peak_bytes(&self) -> u64 {
        self.snapshots.iter().map(|s| s.peak_bytes).sum()
    }
}

// ---------------------------------------------------------------------------
// Scalability analyzer
// ---------------------------------------------------------------------------

/// A single data point in a scalability sweep.
#[derive(Debug, Clone)]
pub struct ScalabilityPoint {
    pub parameter_value: usize,
    pub latency: Duration,
    pub throughput_ops: f64,
    pub peak_memory_bytes: u64,
}

/// Dimension to sweep when analyzing scalability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalabilityDimension {
    BatchSize,
    SequenceLength,
    HiddenDimension,
    HeadCount,
}

impl fmt::Display for ScalabilityDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchSize => write!(f, "batch_size"),
            Self::SequenceLength => write!(f, "seq_len"),
            Self::HiddenDimension => write!(f, "hidden_dim"),
            Self::HeadCount => write!(f, "head_count"),
        }
    }
}

/// Measures how performance scales with a parameter for a given backend.
#[derive(Debug, Clone)]
pub struct ScalabilityAnalyzer {
    pub backend: BackendId,
    pub dimension: ScalabilityDimension,
    pub points: Vec<ScalabilityPoint>,
}

/// Scaling behaviour classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingBehaviour {
    /// O(1) – constant.
    Constant,
    /// O(n) – linear.
    Linear,
    /// O(n log n) – log-linear.
    LogLinear,
    /// O(n²) – quadratic.
    Quadratic,
    /// Could not determine.
    Unknown,
}

impl ScalabilityAnalyzer {
    pub fn new(backend: BackendId, dimension: ScalabilityDimension) -> Self {
        Self { backend, dimension, points: Vec::new() }
    }

    pub fn add_point(&mut self, point: ScalabilityPoint) {
        self.points.push(point);
    }

    /// Compute the scaling factor between the first and last points.
    pub fn scaling_factor(&self) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }
        let first_lat = self.points.first()?.latency.as_secs_f64();
        let last_lat = self.points.last()?.latency.as_secs_f64();
        if first_lat == 0.0 {
            return None;
        }
        Some(last_lat / first_lat)
    }

    /// Compute the parameter ratio between first and last points.
    pub fn parameter_ratio(&self) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }
        let first_val = self.points.first()?.parameter_value as f64;
        let last_val = self.points.last()?.parameter_value as f64;
        if first_val == 0.0 {
            return None;
        }
        Some(last_val / first_val)
    }

    /// Heuristic classification of scaling behaviour.
    pub fn classify_scaling(&self) -> ScalingBehaviour {
        let (Some(sf), Some(pr)) = (self.scaling_factor(), self.parameter_ratio()) else {
            return ScalingBehaviour::Unknown;
        };
        if pr <= 1.0 {
            return ScalingBehaviour::Unknown;
        }
        let exponent = sf.log(pr);
        if exponent < 0.3 {
            ScalingBehaviour::Constant
        } else if exponent < 1.3 {
            ScalingBehaviour::Linear
        } else if exponent < 1.7 {
            ScalingBehaviour::LogLinear
        } else {
            ScalingBehaviour::Quadratic
        }
    }

    /// Whether throughput increased with the parameter (good scaling).
    pub fn throughput_scales_positively(&self) -> bool {
        if self.points.len() < 2 {
            return false;
        }
        let first_tp = self.points.first().unwrap().throughput_ops;
        let last_tp = self.points.last().unwrap().throughput_ops;
        last_tp >= first_tp
    }

    /// Linear regression slope of latency vs parameter value.
    pub fn latency_slope(&self) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }
        let n = self.points.len() as f64;
        let sum_x: f64 = self.points.iter().map(|p| p.parameter_value as f64).sum();
        let sum_y: f64 = self.points.iter().map(|p| p.latency.as_secs_f64()).sum();
        let sum_xy: f64 =
            self.points.iter().map(|p| p.parameter_value as f64 * p.latency.as_secs_f64()).sum();
        let sum_xx: f64 = self.points.iter().map(|p| (p.parameter_value as f64).powi(2)).sum();
        let denom = n.mul_add(sum_xx, -(sum_x * sum_x));
        if denom.abs() < f64::EPSILON {
            return None;
        }
        Some(n.mul_add(sum_xy, -(sum_x * sum_y)) / denom)
    }
}

// ---------------------------------------------------------------------------
// Roofline model
// ---------------------------------------------------------------------------

/// A single point on the roofline chart.
#[derive(Debug, Clone)]
pub struct RooflinePoint {
    pub operation: OperationType,
    pub backend: BackendId,
    /// FLOPs / byte transferred.
    pub arithmetic_intensity: f64,
    /// Achieved GFLOP/s.
    pub achieved_gflops: f64,
}

/// Roofline model for a backend: peak compute + peak memory bandwidth.
#[derive(Debug, Clone)]
pub struct RooflineModel {
    pub backend: BackendId,
    /// Peak GFLOP/s of the device.
    pub peak_gflops: f64,
    /// Peak memory bandwidth in GB/s.
    pub peak_bandwidth_gbs: f64,
    /// Measured data points.
    pub points: Vec<RooflinePoint>,
}

impl RooflineModel {
    pub fn new(backend: BackendId, peak_gflops: f64, peak_bandwidth_gbs: f64) -> Self {
        Self { backend, peak_gflops, peak_bandwidth_gbs, points: Vec::new() }
    }

    pub fn add_point(&mut self, point: RooflinePoint) {
        self.points.push(point);
    }

    /// The ridge point: arithmetic intensity where compute-bound meets
    /// memory-bound.
    pub fn ridge_point(&self) -> f64 {
        if self.peak_bandwidth_gbs == 0.0 {
            return 0.0;
        }
        self.peak_gflops / self.peak_bandwidth_gbs
    }

    /// Theoretical peak at a given arithmetic intensity.
    pub fn theoretical_peak(&self, ai: f64) -> f64 {
        let memory_bound = ai * self.peak_bandwidth_gbs;
        memory_bound.min(self.peak_gflops)
    }

    /// Efficiency of a point: achieved / theoretical.
    pub fn efficiency(&self, point: &RooflinePoint) -> f64 {
        let peak = self.theoretical_peak(point.arithmetic_intensity);
        if peak == 0.0 {
            return 0.0;
        }
        point.achieved_gflops / peak
    }

    /// Whether a point is memory-bound (below ridge point).
    pub fn is_memory_bound(&self, point: &RooflinePoint) -> bool {
        point.arithmetic_intensity < self.ridge_point()
    }

    /// Average efficiency across all points.
    pub fn average_efficiency(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.points.iter().map(|p| self.efficiency(p)).sum();
        sum / self.points.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Report format
// ---------------------------------------------------------------------------

/// Output format for comparison reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Markdown,
    Json,
    Csv,
}

// ---------------------------------------------------------------------------
// Comparison reporter
// ---------------------------------------------------------------------------

/// Generates formatted comparison reports.
#[derive(Debug)]
pub struct ComparisonReporter {
    pub format: ReportFormat,
    pub include_stddev: bool,
    pub include_percentiles: bool,
}

impl Default for ComparisonReporter {
    fn default() -> Self {
        Self { format: ReportFormat::Markdown, include_stddev: true, include_percentiles: true }
    }
}

impl ComparisonReporter {
    pub const fn new(format: ReportFormat) -> Self {
        Self { format, include_stddev: true, include_percentiles: true }
    }

    /// Render a full comparison result to string.
    pub fn render(&self, result: &ComparisonResult) -> String {
        match self.format {
            ReportFormat::Markdown => self.render_markdown(result),
            ReportFormat::Json => self.render_json(result),
            ReportFormat::Csv => self.render_csv(result),
        }
    }

    fn render_markdown(&self, result: &ComparisonResult) -> String {
        let mut out = String::new();
        out.push_str("# Performance Comparison\n\n");
        let _ = write!(out, "Total wall time: {:.2}s\n\n", result.total_wall_time.as_secs_f64());

        for bench in &result.benchmarks {
            let _ = write!(out, "## {} ({})\n\n", bench.operation, bench.problem_size);
            out.push_str("| Backend | Mean (ms) | Median (ms) |");
            if self.include_stddev {
                out.push_str(" Stddev (ms) |");
            }
            if self.include_percentiles {
                out.push_str(" P99 (ms) |");
            }
            out.push_str(" Throughput (ops/s) |\n");

            out.push_str("|---------|-----------|-------------|");
            if self.include_stddev {
                out.push_str("-------------|");
            }
            if self.include_percentiles {
                out.push_str("-----------|");
            }
            out.push_str("--------------------|\n");

            let mut entries: Vec<_> = bench.results.iter().collect();
            entries.sort_by_key(|(id, _)| format!("{id}"));

            for (id, b) in &entries {
                let _ = write!(
                    out,
                    "| {id} | {:.3} | {:.3} |",
                    b.mean_latency().as_secs_f64() * 1000.0,
                    b.median_latency().as_secs_f64() * 1000.0,
                );
                if self.include_stddev {
                    let _ = write!(out, " {:.3} |", b.stddev_secs() * 1000.0);
                }
                if self.include_percentiles {
                    let _ = write!(out, " {:.3} |", b.percentile(99.0).as_secs_f64() * 1000.0);
                }
                let _ = writeln!(out, " {:.1} |", b.throughput_ops);
            }
            out.push('\n');
        }
        out
    }

    fn render_json(&self, result: &ComparisonResult) -> String {
        let _ = self; // dispatched via self.format
        let mut out = String::from("{\n  \"benchmarks\": [\n");
        for (i, bench) in result.benchmarks.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            let _ = write!(
                out,
                "    {{\"operation\": \"{}\", \"problem_size\": \"{}\"",
                bench.operation, bench.problem_size
            );

            let mut entries: Vec<_> = bench.results.iter().collect();
            entries.sort_by_key(|(id, _)| format!("{id}"));

            out.push_str(", \"results\": {");
            for (j, (id, b)) in entries.iter().enumerate() {
                if j > 0 {
                    out.push_str(", ");
                }
                let _ = write!(
                    out,
                    "\"{id}\": {{\"mean_ms\": {:.3}, \"throughput_ops\": {:.1}}}",
                    b.mean_latency().as_secs_f64() * 1000.0,
                    b.throughput_ops
                );
            }
            out.push_str("}}");
        }
        out.push_str("\n  ],\n");
        let _ = writeln!(
            out,
            "  \"total_wall_time_secs\": {:.2}",
            result.total_wall_time.as_secs_f64()
        );
        out.push_str("}\n");
        out
    }

    fn render_csv(&self, result: &ComparisonResult) -> String {
        let _ = self; // dispatched via self.format
        let mut out =
            String::from("operation,problem_size,backend,mean_ms,median_ms,throughput_ops\n");
        for bench in &result.benchmarks {
            let mut entries: Vec<_> = bench.results.iter().collect();
            entries.sort_by_key(|(id, _)| format!("{id}"));

            for (id, b) in &entries {
                let _ = writeln!(
                    out,
                    "{},{},{id},{:.3},{:.3},{:.1}",
                    bench.operation,
                    bench.problem_size,
                    b.mean_latency().as_secs_f64() * 1000.0,
                    b.median_latency().as_secs_f64() * 1000.0,
                    b.throughput_ops,
                );
            }
        }
        out
    }

    /// Render a speedup summary table in markdown.
    pub fn render_speedup_table(&self, speedups: &[SpeedupResult]) -> String {
        let mut out = String::from("| Operation | Speedup | Time Saved |\n");
        out.push_str("|-----------|---------|------------|\n");
        for s in speedups {
            let _ = writeln!(
                out,
                "| {} | {:.2}x | {:.3} ms |",
                s.operation,
                s.speedup,
                s.time_saved.as_secs_f64() * 1000.0,
            );
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Performance comparison orchestrator
// ---------------------------------------------------------------------------

/// Orchestrates a full cross-backend benchmark comparison.
///
/// Workflow: configure → benchmark all → compare → report.
#[derive(Debug)]
pub struct PerformanceComparison {
    pub config: ComparisonConfig,
    pub result: ComparisonResult,
    pub reporter: ComparisonReporter,
    pub speedup_calculator: Option<SpeedupCalculator>,
    pub memory_efficiency: MemoryEfficiency,
    pub scalability_analyzers: Vec<ScalabilityAnalyzer>,
    pub roofline_models: Vec<RooflineModel>,
}

impl PerformanceComparison {
    /// Create from configuration.
    pub fn new(config: ComparisonConfig) -> Result<Self, String> {
        config.validate()?;
        let reference = config.effective_reference().clone();
        let result = ComparisonResult::new(config.clone());
        let speedup_calculator = config
            .backends
            .iter()
            .find(|b| *b != &reference)
            .map(|target| SpeedupCalculator::new(reference, target.clone()));
        Ok(Self {
            config,
            result,
            reporter: ComparisonReporter::default(),
            speedup_calculator,
            memory_efficiency: MemoryEfficiency::new(),
            scalability_analyzers: Vec::new(),
            roofline_models: Vec::new(),
        })
    }

    /// Register an operation benchmark.
    pub fn add_benchmark(&mut self, benchmark: OperationBenchmark) {
        self.result.add_benchmark(benchmark);
    }

    /// Register a scalability analyzer.
    pub fn add_scalability(&mut self, analyzer: ScalabilityAnalyzer) {
        self.scalability_analyzers.push(analyzer);
    }

    /// Register a roofline model.
    pub fn add_roofline(&mut self, model: RooflineModel) {
        self.roofline_models.push(model);
    }

    /// Compute speedups for all registered benchmarks.
    pub fn compute_speedups(&self) -> Vec<SpeedupResult> {
        self.speedup_calculator
            .as_ref()
            .map_or_else(Vec::new, |calc| calc.compute_all(&self.result))
    }

    /// Generate the full report.
    pub fn report(&self) -> String {
        let mut out = self.reporter.render(&self.result);
        let speedups = self.compute_speedups();
        if !speedups.is_empty() {
            out.push_str("## Speedup Summary\n\n");
            out.push_str(&self.reporter.render_speedup_table(&speedups));
            let _ = write!(
                out,
                "\nGeometric mean speedup: {:.2}x\n",
                SpeedupCalculator::geometric_mean_speedup(&speedups),
            );
        }
        out
    }

    /// Get the winners map.
    pub fn winners(&self) -> HashMap<OperationType, BackendId> {
        self.result.winners()
    }

    /// Set the report format.
    pub fn set_format(&mut self, format: ReportFormat) {
        self.reporter.format = format;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Helpers ---------------------------------------------------------------

    fn dur_ms(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    fn sample_durations(ms_values: &[u64]) -> Vec<Duration> {
        ms_values.iter().map(|&m| dur_ms(m)).collect()
    }

    fn make_bench(backend: BackendId, op: OperationType, ms_values: &[u64]) -> BackendBenchmark {
        BackendBenchmark::from_durations(backend, op, sample_durations(ms_values))
    }

    fn make_op_bench(op: OperationType, pairs: &[(BackendId, &[u64])]) -> OperationBenchmark {
        let mut ob = OperationBenchmark::new(op, "test");
        for (backend, ms) in pairs {
            ob.add_result(make_bench(backend.clone(), ob.operation.clone(), ms));
        }
        ob
    }

    fn default_two_backend_config() -> ComparisonConfig {
        ComparisonConfig::new(vec![BackendId::Cpu, BackendId::Cuda])
    }

    // === BackendId =========================================================

    #[test]
    fn backend_id_display() {
        assert_eq!(BackendId::Cpu.to_string(), "CPU");
        assert_eq!(BackendId::Cuda.to_string(), "CUDA");
        assert_eq!(BackendId::Vulkan.to_string(), "Vulkan");
        assert_eq!(BackendId::Metal.to_string(), "Metal");
        assert_eq!(BackendId::OpenCl.to_string(), "OpenCL");
        assert_eq!(BackendId::Custom("SYCL".into()).to_string(), "SYCL");
    }

    #[test]
    fn backend_id_equality() {
        assert_eq!(BackendId::Cpu, BackendId::Cpu);
        assert_ne!(BackendId::Cpu, BackendId::Cuda);
        assert_eq!(BackendId::Custom("A".into()), BackendId::Custom("A".into()));
    }

    // === OperationType =====================================================

    #[test]
    fn operation_type_display() {
        assert_eq!(OperationType::MatMul.to_string(), "MatMul");
        assert_eq!(OperationType::Attention.to_string(), "Attention");
        assert_eq!(OperationType::Custom("GEMV".into()).to_string(), "GEMV");
    }

    // === ComparisonConfig ==================================================

    #[test]
    fn config_default_has_two_backends() {
        let cfg = ComparisonConfig::default();
        assert_eq!(cfg.backends.len(), 2);
        assert_eq!(cfg.iterations, 100);
        assert_eq!(cfg.warmup_iterations, 10);
    }

    #[test]
    fn config_validate_ok() {
        let cfg = default_two_backend_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validate_empty_backends() {
        let cfg = ComparisonConfig::new(vec![]);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_single_backend() {
        let cfg = ComparisonConfig::new(vec![BackendId::Cpu]);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("at least two"));
    }

    #[test]
    fn config_validate_zero_iterations() {
        let mut cfg = default_two_backend_config();
        cfg.iterations = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_bad_reference() {
        let mut cfg = default_two_backend_config();
        cfg.reference_backend = Some(BackendId::Vulkan);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("Vulkan"));
    }

    #[test]
    fn config_effective_reference_default() {
        let cfg = default_two_backend_config();
        assert_eq!(cfg.effective_reference(), &BackendId::Cpu);
    }

    #[test]
    fn config_effective_reference_explicit() {
        let mut cfg = default_two_backend_config();
        cfg.reference_backend = Some(BackendId::Cuda);
        assert_eq!(cfg.effective_reference(), &BackendId::Cuda);
    }

    // === BackendBenchmark ==================================================

    #[test]
    fn bench_from_durations_throughput() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[100, 100, 100, 100, 100]);
        // 5 iterations, 500 ms total → 10 ops/s
        assert!((b.throughput_ops - 10.0).abs() < 0.5);
    }

    #[test]
    fn bench_mean_latency() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[10, 20, 30]);
        assert_eq!(b.mean_latency(), dur_ms(20));
    }

    #[test]
    fn bench_mean_latency_empty() {
        let b = BackendBenchmark::from_durations(BackendId::Cpu, OperationType::MatMul, vec![]);
        assert_eq!(b.mean_latency(), Duration::ZERO);
    }

    #[test]
    fn bench_median_latency_odd() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[30, 10, 20]);
        assert_eq!(b.median_latency(), dur_ms(20));
    }

    #[test]
    fn bench_median_latency_even() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[10, 30, 20, 40]);
        // Sorted: 10,20,30,40 → idx 2 → 30
        assert_eq!(b.median_latency(), dur_ms(30));
    }

    #[test]
    fn bench_median_empty() {
        let b = BackendBenchmark::from_durations(BackendId::Cpu, OperationType::MatMul, vec![]);
        assert_eq!(b.median_latency(), Duration::ZERO);
    }

    #[test]
    fn bench_stddev_zero_for_uniform() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[100, 100, 100]);
        assert!(b.stddev_secs() < 1e-9);
    }

    #[test]
    fn bench_stddev_single() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[100]);
        assert_eq!(b.stddev_secs(), 0.0);
    }

    #[test]
    fn bench_percentile_p0() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[10, 20, 30, 40, 50]);
        assert_eq!(b.percentile(0.0), dur_ms(10));
    }

    #[test]
    fn bench_percentile_p100() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[10, 20, 30, 40, 50]);
        assert_eq!(b.percentile(100.0), dur_ms(50));
    }

    #[test]
    fn bench_percentile_p50() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[10, 20, 30, 40, 50]);
        assert_eq!(b.percentile(50.0), dur_ms(30));
    }

    #[test]
    fn bench_percentile_empty() {
        let b = BackendBenchmark::from_durations(BackendId::Cpu, OperationType::MatMul, vec![]);
        assert_eq!(b.percentile(50.0), Duration::ZERO);
    }

    #[test]
    fn bench_cv_zero_for_uniform() {
        let b = make_bench(BackendId::Cpu, OperationType::MatMul, &[100, 100, 100]);
        assert!(b.cv() < 1e-9);
    }

    #[test]
    fn bench_cv_empty() {
        let b = BackendBenchmark::from_durations(BackendId::Cpu, OperationType::MatMul, vec![]);
        assert_eq!(b.cv(), 0.0);
    }

    // === OperationBenchmark ================================================

    #[test]
    fn op_bench_add_and_get() {
        let ob = make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[10, 20]), (BackendId::Cuda, &[5, 10])],
        );
        assert_eq!(ob.results.len(), 2);
    }

    #[test]
    fn op_bench_fastest() {
        let ob = make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100, 100]), (BackendId::Cuda, &[10, 10])],
        );
        assert_eq!(ob.fastest_backend(), Some(&BackendId::Cuda));
    }

    #[test]
    fn op_bench_highest_throughput() {
        let ob = make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100, 100]), (BackendId::Cuda, &[10, 10])],
        );
        assert_eq!(ob.highest_throughput_backend(), Some(&BackendId::Cuda));
    }

    #[test]
    fn op_bench_empty() {
        let ob = OperationBenchmark::new(OperationType::Softmax, "empty");
        assert!(ob.fastest_backend().is_none());
        assert!(ob.highest_throughput_backend().is_none());
    }

    // === ComparisonResult ==================================================

    #[test]
    fn comparison_result_add_and_get() {
        let mut cr = ComparisonResult::new(default_two_backend_config());
        let ob = make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[50]), (BackendId::Cuda, &[10])],
        );
        cr.add_benchmark(ob);
        assert!(cr.get_operation(&OperationType::MatMul).is_some());
        assert!(cr.get_operation(&OperationType::Softmax).is_none());
    }

    #[test]
    fn comparison_result_winners() {
        let mut cr = ComparisonResult::new(default_two_backend_config());
        cr.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[50]), (BackendId::Cuda, &[10])],
        ));
        cr.add_benchmark(make_op_bench(
            OperationType::Softmax,
            &[(BackendId::Cpu, &[5]), (BackendId::Cuda, &[20])],
        ));
        let w = cr.winners();
        assert_eq!(w[&OperationType::MatMul], BackendId::Cuda);
        assert_eq!(w[&OperationType::Softmax], BackendId::Cpu);
    }

    #[test]
    fn comparison_result_empty_winners() {
        let cr = ComparisonResult::new(default_two_backend_config());
        assert!(cr.winners().is_empty());
    }

    // === SpeedupCalculator =================================================

    #[test]
    fn speedup_basic() {
        let calc = SpeedupCalculator::new(BackendId::Cpu, BackendId::Cuda);
        let ob = make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[25])],
        );
        let s = calc.compute(&ob).unwrap();
        assert!((s.speedup - 4.0).abs() < 0.01);
    }

    #[test]
    fn speedup_slower_than_reference() {
        let calc = SpeedupCalculator::new(BackendId::Cpu, BackendId::Cuda);
        let ob = make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[10]), (BackendId::Cuda, &[100])],
        );
        let s = calc.compute(&ob).unwrap();
        assert!(s.speedup < 1.0);
        assert_eq!(s.time_saved, Duration::ZERO);
    }

    #[test]
    fn speedup_missing_backend() {
        let calc = SpeedupCalculator::new(BackendId::Cpu, BackendId::Vulkan);
        let ob = make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[25])],
        );
        assert!(calc.compute(&ob).is_none());
    }

    #[test]
    fn speedup_compute_all() {
        let calc = SpeedupCalculator::new(BackendId::Cpu, BackendId::Cuda);
        let mut cr = ComparisonResult::new(default_two_backend_config());
        cr.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[50])],
        ));
        cr.add_benchmark(make_op_bench(
            OperationType::Softmax,
            &[(BackendId::Cpu, &[200]), (BackendId::Cuda, &[50])],
        ));
        let speedups = calc.compute_all(&cr);
        assert_eq!(speedups.len(), 2);
    }

    #[test]
    fn speedup_geometric_mean() {
        // 2x and 8x → geometric mean = 4x
        let speedups = vec![
            SpeedupResult {
                operation: OperationType::MatMul,
                reference_backend: BackendId::Cpu,
                target_backend: BackendId::Cuda,
                speedup: 2.0,
                time_saved: dur_ms(50),
            },
            SpeedupResult {
                operation: OperationType::Softmax,
                reference_backend: BackendId::Cpu,
                target_backend: BackendId::Cuda,
                speedup: 8.0,
                time_saved: dur_ms(150),
            },
        ];
        let gm = SpeedupCalculator::geometric_mean_speedup(&speedups);
        assert!((gm - 4.0).abs() < 0.01);
    }

    #[test]
    fn speedup_geometric_mean_empty() {
        let gm = SpeedupCalculator::geometric_mean_speedup(&[]);
        assert!((gm - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn speedup_time_saved_positive() {
        let calc = SpeedupCalculator::new(BackendId::Cpu, BackendId::Cuda);
        let ob = make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[40])],
        );
        let s = calc.compute(&ob).unwrap();
        assert_eq!(s.time_saved, dur_ms(60));
    }

    // === MemoryEfficiency ==================================================

    #[test]
    fn memory_efficiency_most_efficient() {
        let mut me = MemoryEfficiency::new();
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cpu,
            peak_bytes: 1_000_000,
            allocated_bytes: 800_000,
            bandwidth_bytes_per_sec: 10e9,
        });
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cuda,
            peak_bytes: 500_000,
            allocated_bytes: 400_000,
            bandwidth_bytes_per_sec: 300e9,
        });
        assert_eq!(me.most_memory_efficient(), Some(&BackendId::Cuda));
    }

    #[test]
    fn memory_ratio() {
        let mut me = MemoryEfficiency::new();
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cpu,
            peak_bytes: 1000,
            allocated_bytes: 800,
            bandwidth_bytes_per_sec: 10.0,
        });
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cuda,
            peak_bytes: 500,
            allocated_bytes: 400,
            bandwidth_bytes_per_sec: 100.0,
        });
        let r = me.memory_ratio(&BackendId::Cpu, &BackendId::Cuda).unwrap();
        assert!((r - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn memory_ratio_missing() {
        let me = MemoryEfficiency::new();
        assert!(me.memory_ratio(&BackendId::Cpu, &BackendId::Cuda).is_none());
    }

    #[test]
    fn memory_ratio_zero_ref() {
        let mut me = MemoryEfficiency::new();
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cpu,
            peak_bytes: 0,
            allocated_bytes: 0,
            bandwidth_bytes_per_sec: 0.0,
        });
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cuda,
            peak_bytes: 500,
            allocated_bytes: 400,
            bandwidth_bytes_per_sec: 100.0,
        });
        assert!(me.memory_ratio(&BackendId::Cpu, &BackendId::Cuda).is_none());
    }

    #[test]
    fn bandwidth_ratio() {
        let mut me = MemoryEfficiency::new();
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cpu,
            peak_bytes: 1000,
            allocated_bytes: 800,
            bandwidth_bytes_per_sec: 50.0,
        });
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cuda,
            peak_bytes: 500,
            allocated_bytes: 400,
            bandwidth_bytes_per_sec: 200.0,
        });
        let r = me.bandwidth_ratio(&BackendId::Cpu, &BackendId::Cuda).unwrap();
        assert!((r - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bandwidth_ratio_zero_ref() {
        let mut me = MemoryEfficiency::new();
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cpu,
            peak_bytes: 1000,
            allocated_bytes: 800,
            bandwidth_bytes_per_sec: 0.0,
        });
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cuda,
            peak_bytes: 500,
            allocated_bytes: 400,
            bandwidth_bytes_per_sec: 200.0,
        });
        assert!(me.bandwidth_ratio(&BackendId::Cpu, &BackendId::Cuda).is_none());
    }

    #[test]
    fn memory_total_peak() {
        let mut me = MemoryEfficiency::new();
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cpu,
            peak_bytes: 1000,
            allocated_bytes: 0,
            bandwidth_bytes_per_sec: 0.0,
        });
        me.add_snapshot(MemorySnapshot {
            backend: BackendId::Cuda,
            peak_bytes: 2000,
            allocated_bytes: 0,
            bandwidth_bytes_per_sec: 0.0,
        });
        assert_eq!(me.total_peak_bytes(), 3000);
    }

    #[test]
    fn memory_efficiency_empty() {
        let me = MemoryEfficiency::new();
        assert!(me.most_memory_efficient().is_none());
        assert_eq!(me.total_peak_bytes(), 0);
    }

    #[test]
    fn memory_default_trait() {
        let me = MemoryEfficiency::default();
        assert!(me.snapshots.is_empty());
    }

    // === ScalabilityAnalyzer ===============================================

    fn make_scalability(points: &[(usize, u64, f64)]) -> ScalabilityAnalyzer {
        let mut sa = ScalabilityAnalyzer::new(BackendId::Cuda, ScalabilityDimension::BatchSize);
        for &(param, lat_ms, thr) in points {
            sa.add_point(ScalabilityPoint {
                parameter_value: param,
                latency: dur_ms(lat_ms),
                throughput_ops: thr,
                peak_memory_bytes: 0,
            });
        }
        sa
    }

    #[test]
    fn scalability_linear() {
        let sa =
            make_scalability(&[(1, 10, 100.0), (2, 20, 100.0), (4, 40, 100.0), (8, 80, 100.0)]);
        assert_eq!(sa.classify_scaling(), ScalingBehaviour::Linear);
    }

    #[test]
    fn scalability_quadratic() {
        let sa = make_scalability(&[(1, 10, 100.0), (2, 40, 25.0), (4, 160, 6.25)]);
        assert_eq!(sa.classify_scaling(), ScalingBehaviour::Quadratic);
    }

    #[test]
    fn scalability_constant() {
        let sa = make_scalability(&[(1, 10, 100.0), (2, 10, 200.0), (4, 10, 400.0)]);
        assert_eq!(sa.classify_scaling(), ScalingBehaviour::Constant);
    }

    #[test]
    fn scalability_unknown_single_point() {
        let sa = make_scalability(&[(1, 10, 100.0)]);
        assert_eq!(sa.classify_scaling(), ScalingBehaviour::Unknown);
    }

    #[test]
    fn scalability_unknown_empty() {
        let sa = ScalabilityAnalyzer::new(BackendId::Cpu, ScalabilityDimension::SequenceLength);
        assert_eq!(sa.classify_scaling(), ScalingBehaviour::Unknown);
    }

    #[test]
    fn scalability_factor() {
        let sa = make_scalability(&[(1, 10, 100.0), (4, 40, 25.0)]);
        assert!((sa.scaling_factor().unwrap() - 4.0).abs() < 0.01);
    }

    #[test]
    fn scalability_factor_none() {
        let sa = make_scalability(&[(1, 10, 100.0)]);
        assert!(sa.scaling_factor().is_none());
    }

    #[test]
    fn scalability_parameter_ratio() {
        let sa = make_scalability(&[(2, 10, 0.0), (8, 40, 0.0)]);
        assert!((sa.parameter_ratio().unwrap() - 4.0).abs() < 0.01);
    }

    #[test]
    fn scalability_parameter_ratio_none() {
        let sa = make_scalability(&[]);
        assert!(sa.parameter_ratio().is_none());
    }

    #[test]
    fn scalability_throughput_positive() {
        let sa = make_scalability(&[(1, 10, 100.0), (2, 15, 133.0)]);
        assert!(sa.throughput_scales_positively());
    }

    #[test]
    fn scalability_throughput_negative() {
        let sa = make_scalability(&[(1, 10, 100.0), (2, 40, 50.0)]);
        assert!(!sa.throughput_scales_positively());
    }

    #[test]
    fn scalability_throughput_empty() {
        let sa = make_scalability(&[]);
        assert!(!sa.throughput_scales_positively());
    }

    #[test]
    fn scalability_latency_slope() {
        let sa = make_scalability(&[(1, 10, 0.0), (2, 20, 0.0), (3, 30, 0.0), (4, 40, 0.0)]);
        let slope = sa.latency_slope().unwrap();
        // 10 ms = 0.01 s per unit
        assert!((slope - 0.01).abs() < 0.001);
    }

    #[test]
    fn scalability_latency_slope_none() {
        let sa = make_scalability(&[(1, 10, 0.0)]);
        assert!(sa.latency_slope().is_none());
    }

    #[test]
    fn scalability_dimension_display() {
        assert_eq!(ScalabilityDimension::BatchSize.to_string(), "batch_size");
        assert_eq!(ScalabilityDimension::SequenceLength.to_string(), "seq_len");
        assert_eq!(ScalabilityDimension::HiddenDimension.to_string(), "hidden_dim");
        assert_eq!(ScalabilityDimension::HeadCount.to_string(), "head_count");
    }

    // === RooflineModel =====================================================

    #[test]
    fn roofline_ridge_point() {
        let rm = RooflineModel::new(BackendId::Cuda, 1000.0, 500.0);
        assert!((rm.ridge_point() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn roofline_ridge_point_zero_bw() {
        let rm = RooflineModel::new(BackendId::Cuda, 1000.0, 0.0);
        assert_eq!(rm.ridge_point(), 0.0);
    }

    #[test]
    fn roofline_theoretical_peak_memory_bound() {
        let rm = RooflineModel::new(BackendId::Cuda, 1000.0, 500.0);
        assert!((rm.theoretical_peak(1.0) - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn roofline_theoretical_peak_compute_bound() {
        let rm = RooflineModel::new(BackendId::Cuda, 1000.0, 500.0);
        assert!((rm.theoretical_peak(10.0) - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn roofline_efficiency() {
        let rm = RooflineModel::new(BackendId::Cuda, 1000.0, 500.0);
        let pt = RooflinePoint {
            operation: OperationType::MatMul,
            backend: BackendId::Cuda,
            arithmetic_intensity: 1.0,
            achieved_gflops: 250.0,
        };
        assert!((rm.efficiency(&pt) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn roofline_is_memory_bound() {
        let rm = RooflineModel::new(BackendId::Cuda, 1000.0, 500.0);
        let mb_point = RooflinePoint {
            operation: OperationType::Softmax,
            backend: BackendId::Cuda,
            arithmetic_intensity: 0.5,
            achieved_gflops: 100.0,
        };
        let cb_point = RooflinePoint {
            operation: OperationType::MatMul,
            backend: BackendId::Cuda,
            arithmetic_intensity: 10.0,
            achieved_gflops: 800.0,
        };
        assert!(rm.is_memory_bound(&mb_point));
        assert!(!rm.is_memory_bound(&cb_point));
    }

    #[test]
    fn roofline_average_efficiency() {
        let mut rm = RooflineModel::new(BackendId::Cuda, 1000.0, 500.0);
        rm.add_point(RooflinePoint {
            operation: OperationType::MatMul,
            backend: BackendId::Cuda,
            arithmetic_intensity: 1.0,
            achieved_gflops: 250.0, // eff = 0.5
        });
        rm.add_point(RooflinePoint {
            operation: OperationType::Softmax,
            backend: BackendId::Cuda,
            arithmetic_intensity: 10.0,
            achieved_gflops: 1000.0, // eff = 1.0
        });
        assert!((rm.average_efficiency() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn roofline_average_efficiency_empty() {
        let rm = RooflineModel::new(BackendId::Cuda, 1000.0, 500.0);
        assert_eq!(rm.average_efficiency(), 0.0);
    }

    #[test]
    fn roofline_efficiency_zero_peak() {
        let rm = RooflineModel::new(BackendId::Cuda, 0.0, 0.0);
        let pt = RooflinePoint {
            operation: OperationType::MatMul,
            backend: BackendId::Cuda,
            arithmetic_intensity: 1.0,
            achieved_gflops: 100.0,
        };
        assert_eq!(rm.efficiency(&pt), 0.0);
    }

    // === ComparisonReporter ================================================

    #[test]
    fn reporter_markdown_contains_header() {
        let reporter = ComparisonReporter::new(ReportFormat::Markdown);
        let cr = ComparisonResult::new(default_two_backend_config());
        let out = reporter.render(&cr);
        assert!(out.contains("# Performance Comparison"));
    }

    #[test]
    fn reporter_markdown_with_data() {
        let reporter = ComparisonReporter::new(ReportFormat::Markdown);
        let mut cr = ComparisonResult::new(default_two_backend_config());
        cr.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[25])],
        ));
        let out = reporter.render(&cr);
        assert!(out.contains("MatMul"));
        assert!(out.contains("CPU"));
        assert!(out.contains("CUDA"));
    }

    #[test]
    fn reporter_json_valid_structure() {
        let reporter = ComparisonReporter::new(ReportFormat::Json);
        let mut cr = ComparisonResult::new(default_two_backend_config());
        cr.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[50]), (BackendId::Cuda, &[10])],
        ));
        let out = reporter.render(&cr);
        assert!(out.contains("\"benchmarks\""));
        assert!(out.contains("\"MatMul\""));
    }

    #[test]
    fn reporter_csv_header() {
        let reporter = ComparisonReporter::new(ReportFormat::Csv);
        let cr = ComparisonResult::new(default_two_backend_config());
        let out = reporter.render(&cr);
        assert!(out.starts_with("operation,problem_size,backend,"));
    }

    #[test]
    fn reporter_csv_with_data() {
        let reporter = ComparisonReporter::new(ReportFormat::Csv);
        let mut cr = ComparisonResult::new(default_two_backend_config());
        cr.add_benchmark(make_op_bench(OperationType::Softmax, &[(BackendId::Cpu, &[20])]));
        let out = reporter.render(&cr);
        assert!(out.contains("Softmax"));
        assert!(out.contains("CPU"));
    }

    #[test]
    fn reporter_speedup_table() {
        let reporter = ComparisonReporter::default();
        let speedups = vec![SpeedupResult {
            operation: OperationType::MatMul,
            reference_backend: BackendId::Cpu,
            target_backend: BackendId::Cuda,
            speedup: 4.0,
            time_saved: dur_ms(75),
        }];
        let tbl = reporter.render_speedup_table(&speedups);
        assert!(tbl.contains("4.00x"));
        assert!(tbl.contains("MatMul"));
    }

    #[test]
    fn reporter_default() {
        let r = ComparisonReporter::default();
        assert_eq!(r.format, ReportFormat::Markdown);
        assert!(r.include_stddev);
        assert!(r.include_percentiles);
    }

    #[test]
    fn reporter_no_stddev() {
        let mut reporter = ComparisonReporter::new(ReportFormat::Markdown);
        reporter.include_stddev = false;
        let mut cr = ComparisonResult::new(default_two_backend_config());
        cr.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[25])],
        ));
        let out = reporter.render(&cr);
        assert!(!out.contains("Stddev"));
    }

    // === PerformanceComparison (orchestrator) ==============================

    #[test]
    fn orchestrator_new_valid() {
        let pc = PerformanceComparison::new(default_two_backend_config());
        assert!(pc.is_ok());
    }

    #[test]
    fn orchestrator_new_invalid() {
        let cfg = ComparisonConfig::new(vec![]);
        assert!(PerformanceComparison::new(cfg).is_err());
    }

    #[test]
    fn orchestrator_add_benchmark() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        pc.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[25])],
        ));
        assert_eq!(pc.result.benchmarks.len(), 1);
    }

    #[test]
    fn orchestrator_compute_speedups() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        pc.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[25])],
        ));
        let speedups = pc.compute_speedups();
        assert_eq!(speedups.len(), 1);
        assert!((speedups[0].speedup - 4.0).abs() < 0.01);
    }

    #[test]
    fn orchestrator_report_contains_speedup() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        pc.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[25])],
        ));
        let report = pc.report();
        assert!(report.contains("Speedup Summary"));
        assert!(report.contains("Geometric mean"));
    }

    #[test]
    fn orchestrator_winners() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        pc.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[10])],
        ));
        let w = pc.winners();
        assert_eq!(w[&OperationType::MatMul], BackendId::Cuda);
    }

    #[test]
    fn orchestrator_set_format() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        pc.set_format(ReportFormat::Json);
        assert_eq!(pc.reporter.format, ReportFormat::Json);
    }

    #[test]
    fn orchestrator_add_scalability() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        let sa = ScalabilityAnalyzer::new(BackendId::Cuda, ScalabilityDimension::BatchSize);
        pc.add_scalability(sa);
        assert_eq!(pc.scalability_analyzers.len(), 1);
    }

    #[test]
    fn orchestrator_add_roofline() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        let rm = RooflineModel::new(BackendId::Cuda, 1000.0, 500.0);
        pc.add_roofline(rm);
        assert_eq!(pc.roofline_models.len(), 1);
    }

    #[test]
    fn orchestrator_empty_speedups() {
        let pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        assert!(pc.compute_speedups().is_empty());
    }

    #[test]
    fn orchestrator_report_empty() {
        let pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        let report = pc.report();
        assert!(report.contains("# Performance Comparison"));
    }

    #[test]
    fn orchestrator_multiple_operations() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        pc.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[100]), (BackendId::Cuda, &[25])],
        ));
        pc.add_benchmark(make_op_bench(
            OperationType::Attention,
            &[(BackendId::Cpu, &[200]), (BackendId::Cuda, &[40])],
        ));
        pc.add_benchmark(make_op_bench(
            OperationType::Softmax,
            &[(BackendId::Cpu, &[30]), (BackendId::Cuda, &[10])],
        ));
        let speedups = pc.compute_speedups();
        assert_eq!(speedups.len(), 3);
        let report = pc.report();
        assert!(report.contains("MatMul"));
        assert!(report.contains("Attention"));
        assert!(report.contains("Softmax"));
    }

    #[test]
    fn orchestrator_json_report() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        pc.set_format(ReportFormat::Json);
        pc.add_benchmark(make_op_bench(
            OperationType::MatMul,
            &[(BackendId::Cpu, &[50]), (BackendId::Cuda, &[10])],
        ));
        let report = pc.report();
        assert!(report.contains("\"benchmarks\""));
    }

    #[test]
    fn orchestrator_csv_report() {
        let mut pc = PerformanceComparison::new(default_two_backend_config()).unwrap();
        pc.set_format(ReportFormat::Csv);
        pc.add_benchmark(make_op_bench(
            OperationType::LayerNorm,
            &[(BackendId::Cpu, &[15]), (BackendId::Cuda, &[5])],
        ));
        let report = pc.report();
        assert!(report.contains("LayerNorm"));
    }
}
