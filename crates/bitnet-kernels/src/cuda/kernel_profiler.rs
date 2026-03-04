//! CUDA kernel profiling framework for BitNet inference.
//!
//! Provides GPU event-based timing, execution statistics, memory bandwidth
//! measurement, arithmetic intensity analysis, occupancy tracking, hierarchical
//! profiling scopes, report generation, and throughput regression detection.
//!
//! All types compile on CPU builds (using `std::time` for timing) so that
//! profiling call-sites are always available without feature-flag churn.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── GpuEvent ─────────────────────────────────────────────────────────

/// A GPU event-based timing pair (start/stop).
///
/// On CPU builds this wraps `std::time::Instant`; on real GPU builds the
/// same API would wrap CUDA event handles.
#[derive(Debug, Clone)]
pub struct GpuEvent {
    /// Human-readable label.
    pub label: String,
    /// Stream id this event is associated with.
    pub stream_id: u32,
    start: Instant,
    end: Option<Instant>,
}

impl GpuEvent {
    /// Record the start of a GPU event on the given stream.
    pub fn record_start(label: impl Into<String>, stream_id: u32) -> Self {
        Self { label: label.into(), stream_id, start: Instant::now(), end: None }
    }

    /// Record the stop of this GPU event.
    pub fn record_stop(&mut self) -> Duration {
        let now = Instant::now();
        self.end = Some(now);
        now.duration_since(self.start)
    }

    /// Elapsed time between start and stop (or since start if not stopped).
    pub fn elapsed_ms(&self) -> f64 {
        let d = match self.end {
            Some(e) => e.duration_since(self.start),
            None => self.start.elapsed(),
        };
        d.as_secs_f64() * 1e3
    }

    /// Whether the stop event has been recorded.
    pub fn is_complete(&self) -> bool {
        self.end.is_some()
    }

    /// Raw duration.
    pub fn duration(&self) -> Duration {
        match self.end {
            Some(e) => e.duration_since(self.start),
            None => self.start.elapsed(),
        }
    }
}

// ── KernelExecStats ──────────────────────────────────────────────────

/// Aggregated execution statistics for a named kernel.
#[derive(Debug, Clone)]
pub struct KernelExecStats {
    /// Kernel name.
    pub name: String,
    /// All recorded latencies.
    latencies: Vec<Duration>,
}

impl KernelExecStats {
    /// Create a new stats tracker for the named kernel.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), latencies: Vec::new() }
    }

    /// Record a latency sample.
    pub fn record(&mut self, d: Duration) {
        self.latencies.push(d);
    }

    /// Number of recorded samples.
    pub fn count(&self) -> usize {
        self.latencies.len()
    }

    /// Minimum latency.
    pub fn min(&self) -> Option<Duration> {
        self.latencies.iter().min().copied()
    }

    /// Maximum latency.
    pub fn max(&self) -> Option<Duration> {
        self.latencies.iter().max().copied()
    }

    /// Mean latency.
    pub fn mean(&self) -> Option<Duration> {
        if self.latencies.is_empty() {
            return None;
        }
        let total: Duration = self.latencies.iter().sum();
        Some(total / self.latencies.len() as u32)
    }

    /// Standard deviation of latency in seconds.
    pub fn stddev_secs(&self) -> Option<f64> {
        let n = self.latencies.len();
        if n == 0 {
            return None;
        }
        let mean_s = self.mean()?.as_secs_f64();
        let variance = self
            .latencies
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean_s;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        Some(variance.sqrt())
    }

    /// Median latency.
    pub fn median(&self) -> Option<Duration> {
        if self.latencies.is_empty() {
            return None;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort();
        Some(sorted[sorted.len() / 2])
    }

    /// 95th-percentile latency.
    pub fn p95(&self) -> Option<Duration> {
        if self.latencies.is_empty() {
            return None;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    /// 99th-percentile latency.
    pub fn p99(&self) -> Option<Duration> {
        if self.latencies.is_empty() {
            return None;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64) * 0.99).ceil() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    /// Clear all recorded samples.
    pub fn clear(&mut self) {
        self.latencies.clear();
    }
}

// ── MemoryBandwidth ──────────────────────────────────────────────────

/// Memory bandwidth measurement for a kernel execution.
#[derive(Debug, Clone, Copy)]
pub struct MemoryBandwidth {
    /// Bytes read from global memory.
    pub bytes_read: u64,
    /// Bytes written to global memory.
    pub bytes_written: u64,
    /// Kernel execution duration.
    pub duration: Duration,
    /// Theoretical peak bandwidth in bytes/sec.
    pub theoretical_peak_bps: f64,
}

impl MemoryBandwidth {
    /// Create a new bandwidth measurement.
    pub fn new(
        bytes_read: u64,
        bytes_written: u64,
        duration: Duration,
        theoretical_peak_bps: f64,
    ) -> Self {
        Self { bytes_read, bytes_written, duration, theoretical_peak_bps }
    }

    /// Total bytes transferred.
    pub fn total_bytes(&self) -> u64 {
        self.bytes_read + self.bytes_written
    }

    /// Effective bandwidth in bytes/sec.
    pub fn effective_bps(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_bytes() as f64 / secs
    }

    /// Effective bandwidth in GB/s.
    pub fn effective_gb_s(&self) -> f64 {
        self.effective_bps() / 1e9
    }

    /// Bandwidth efficiency as a fraction in `[0.0, 1.0]`.
    pub fn efficiency(&self) -> f64 {
        if self.theoretical_peak_bps <= 0.0 {
            return 0.0;
        }
        (self.effective_bps() / self.theoretical_peak_bps).min(1.0)
    }

    /// Gap between theoretical and effective in GB/s.
    pub fn gap_gb_s(&self) -> f64 {
        (self.theoretical_peak_bps / 1e9) - self.effective_gb_s()
    }
}

// ── ArithmeticIntensity ──────────────────────────────────────────────

/// Arithmetic intensity analysis (FLOP/byte ratio).
#[derive(Debug, Clone, Copy)]
pub struct ArithmeticIntensity {
    /// Total floating-point operations.
    pub flops: u64,
    /// Total bytes transferred (read + written).
    pub bytes_transferred: u64,
}

impl ArithmeticIntensity {
    /// Create a new arithmetic intensity measurement.
    pub fn new(flops: u64, bytes_transferred: u64) -> Self {
        Self { flops, bytes_transferred }
    }

    /// FLOP/byte ratio.
    pub fn intensity(&self) -> f64 {
        if self.bytes_transferred == 0 {
            return 0.0;
        }
        self.flops as f64 / self.bytes_transferred as f64
    }

    /// Whether the kernel is compute-bound given the machine balance point.
    ///
    /// The balance point is peak FLOP/s divided by peak bytes/s.
    pub fn is_compute_bound(&self, machine_balance: f64) -> bool {
        self.intensity() >= machine_balance
    }

    /// Whether the kernel is memory-bound given the machine balance point.
    pub fn is_memory_bound(&self, machine_balance: f64) -> bool {
        self.intensity() < machine_balance
    }

    /// Roofline model: achievable FLOP/s for the given peak compute and bandwidth.
    pub fn roofline_peak(&self, peak_flops: f64, peak_bandwidth: f64) -> f64 {
        if peak_bandwidth <= 0.0 {
            return 0.0;
        }
        let mem_roof = self.intensity() * peak_bandwidth;
        peak_flops.min(mem_roof)
    }
}

impl fmt::Display for ArithmeticIntensity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} FLOP/B", self.intensity())
    }
}

// ── OccupancyTracker ─────────────────────────────────────────────────

/// Tracks achieved vs theoretical occupancy for kernel launches.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyRecord {
    /// Kernel name (static for Copy, use index in practice).
    pub theoretical: f64,
    /// Achieved occupancy as a fraction in `[0.0, 1.0]`.
    pub achieved: f64,
    /// Active warps per SM.
    pub active_warps: u32,
    /// Maximum warps per SM.
    pub max_warps: u32,
    /// Threads per block used.
    pub threads_per_block: u32,
    /// Shared memory per block in bytes.
    pub shared_mem_bytes: u32,
    /// Registers per thread.
    pub registers_per_thread: u32,
}

impl OccupancyRecord {
    /// Create a new occupancy record.
    pub fn new(
        theoretical: f64,
        achieved: f64,
        active_warps: u32,
        max_warps: u32,
        threads_per_block: u32,
        shared_mem_bytes: u32,
        registers_per_thread: u32,
    ) -> Self {
        Self {
            theoretical: theoretical.clamp(0.0, 1.0),
            achieved: achieved.clamp(0.0, 1.0),
            active_warps,
            max_warps,
            threads_per_block,
            shared_mem_bytes,
            registers_per_thread,
        }
    }

    /// Gap between theoretical and achieved occupancy.
    pub fn gap(&self) -> f64 {
        (self.theoretical - self.achieved).max(0.0)
    }

    /// Whether achieved is within the given tolerance of theoretical.
    pub fn is_near_theoretical(&self, tolerance: f64) -> bool {
        self.gap() <= tolerance
    }
}

/// Collects occupancy records across kernel launches.
#[derive(Debug, Clone, Default)]
pub struct OccupancyTracker {
    records: HashMap<String, Vec<OccupancyRecord>>,
}

impl OccupancyTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an occupancy measurement for a named kernel.
    pub fn record(&mut self, kernel_name: impl Into<String>, record: OccupancyRecord) {
        self.records.entry(kernel_name.into()).or_default().push(record);
    }

    /// Get all records for a named kernel.
    pub fn records_for(&self, kernel_name: &str) -> &[OccupancyRecord] {
        self.records.get(kernel_name).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Mean achieved occupancy for a named kernel.
    pub fn mean_achieved(&self, kernel_name: &str) -> Option<f64> {
        let recs = self.records.get(kernel_name)?;
        if recs.is_empty() {
            return None;
        }
        Some(recs.iter().map(|r| r.achieved).sum::<f64>() / recs.len() as f64)
    }

    /// Number of tracked kernels.
    pub fn kernel_count(&self) -> usize {
        self.records.len()
    }

    /// All tracked kernel names.
    pub fn kernel_names(&self) -> Vec<&str> {
        self.records.keys().map(|s| s.as_str()).collect()
    }

    /// Total number of records across all kernels.
    pub fn total_records(&self) -> usize {
        self.records.values().map(|v| v.len()).sum()
    }
}

// ── ProfilingScope ───────────────────────────────────────────────────

/// A hierarchical profiling scope that can nest sub-scopes.
#[derive(Debug, Clone)]
pub struct ProfilingScope {
    /// Scope name.
    pub name: String,
    /// Depth in the hierarchy (0 = root).
    pub depth: u32,
    /// Duration of this scope.
    pub duration: Duration,
    /// Child scopes.
    pub children: Vec<ProfilingScope>,
    /// Kernel executions directly in this scope.
    pub kernel_count: usize,
}

impl ProfilingScope {
    /// Create a completed scope with no children.
    pub fn leaf(name: impl Into<String>, depth: u32, duration: Duration) -> Self {
        Self { name: name.into(), depth, duration, children: Vec::new(), kernel_count: 0 }
    }

    /// Create a scope with children.
    pub fn with_children(
        name: impl Into<String>,
        depth: u32,
        duration: Duration,
        children: Vec<ProfilingScope>,
    ) -> Self {
        Self { name: name.into(), depth, duration, children, kernel_count: 0 }
    }

    /// Set the kernel count for this scope.
    pub fn with_kernel_count(mut self, count: usize) -> Self {
        self.kernel_count = count;
        self
    }

    /// Total time spent in child scopes.
    pub fn children_duration(&self) -> Duration {
        self.children.iter().map(|c| c.duration).sum()
    }

    /// Time in this scope NOT accounted for by children.
    pub fn self_time(&self) -> Duration {
        self.duration.saturating_sub(self.children_duration())
    }

    /// Total number of descendants (recursive).
    pub fn descendant_count(&self) -> usize {
        self.children.len() + self.children.iter().map(|c| c.descendant_count()).sum::<usize>()
    }

    /// Maximum depth in the subtree rooted at this scope.
    pub fn max_depth(&self) -> u32 {
        if self.children.is_empty() {
            self.depth
        } else {
            self.children.iter().map(|c| c.max_depth()).max().unwrap_or(self.depth)
        }
    }

    /// Flatten the scope tree into a depth-first list.
    pub fn flatten(&self) -> Vec<&ProfilingScope> {
        let mut result = vec![self];
        for child in &self.children {
            result.extend(child.flatten());
        }
        result
    }
}

/// Builder for constructing hierarchical profiling scope trees.
#[derive(Debug)]
pub struct ScopeBuilder {
    stack: Vec<(String, Instant, Vec<ProfilingScope>, usize)>,
}

impl ScopeBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    /// Enter a new profiling scope.
    pub fn enter(&mut self, name: impl Into<String>) {
        self.stack.push((name.into(), Instant::now(), Vec::new(), 0));
    }

    /// Increment the kernel count for the current scope.
    pub fn record_kernel(&mut self) {
        if let Some(top) = self.stack.last_mut() {
            top.3 += 1;
        }
    }

    /// Exit the current scope and return the completed scope.
    pub fn exit(&mut self) -> Option<ProfilingScope> {
        let (name, start, children, kernel_count) = self.stack.pop()?;
        let duration = start.elapsed();
        let depth = self.stack.len() as u32;
        let scope = ProfilingScope { name, depth, duration, children, kernel_count };
        // If there's a parent scope, attach this as a child.
        if let Some(parent) = self.stack.last_mut() {
            parent.2.push(scope.clone());
        }
        Some(scope)
    }

    /// Current nesting depth.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Whether the builder has no open scopes.
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

impl Default for ScopeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── ProfilingReport ──────────────────────────────────────────────────

/// Output format for profiling reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// Human-readable text.
    Text,
    /// JSON format.
    Json,
}

/// A profiling report generated from collected data.
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    /// Title of the report.
    pub title: String,
    /// Per-kernel execution stats.
    pub kernel_stats: Vec<KernelStatsSummary>,
    /// Total profiling duration.
    pub total_duration: Duration,
    /// Hierarchical scopes (if any).
    pub scopes: Vec<ProfilingScope>,
    /// Regression results (if any).
    pub regressions: Vec<RegressionResult>,
}

/// Summary of a single kernel's stats for reporting.
#[derive(Debug, Clone)]
pub struct KernelStatsSummary {
    /// Kernel name.
    pub name: String,
    /// Number of invocations.
    pub count: usize,
    /// Mean latency in microseconds.
    pub mean_us: f64,
    /// Min latency in microseconds.
    pub min_us: f64,
    /// Max latency in microseconds.
    pub max_us: f64,
    /// Standard deviation in microseconds.
    pub stddev_us: f64,
    /// Optional bandwidth efficiency.
    pub bandwidth_efficiency: Option<f64>,
    /// Optional arithmetic intensity.
    pub arithmetic_intensity: Option<f64>,
    /// Optional occupancy achieved.
    pub occupancy_achieved: Option<f64>,
}

impl ProfilingReport {
    /// Create a new empty report.
    pub fn new(title: impl Into<String>, total_duration: Duration) -> Self {
        Self {
            title: title.into(),
            kernel_stats: Vec::new(),
            total_duration,
            scopes: Vec::new(),
            regressions: Vec::new(),
        }
    }

    /// Add kernel stats to the report.
    pub fn add_kernel_stats(&mut self, stats: KernelStatsSummary) {
        self.kernel_stats.push(stats);
    }

    /// Add a scope to the report.
    pub fn add_scope(&mut self, scope: ProfilingScope) {
        self.scopes.push(scope);
    }

    /// Add a regression result.
    pub fn add_regression(&mut self, regression: RegressionResult) {
        self.regressions.push(regression);
    }

    /// Render the report as text.
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("=== {} ===\n", self.title));
        out.push_str(&format!(
            "Total duration: {:.3} ms\n\n",
            self.total_duration.as_secs_f64() * 1e3
        ));

        if !self.kernel_stats.is_empty() {
            out.push_str("Kernel Statistics:\n");
            out.push_str(&format!(
                "{:<30} {:>6} {:>12} {:>12} {:>12} {:>12}\n",
                "Name", "Count", "Mean(µs)", "Min(µs)", "Max(µs)", "StdDev(µs)"
            ));
            out.push_str(&"-".repeat(90));
            out.push('\n');
            for ks in &self.kernel_stats {
                out.push_str(&format!(
                    "{:<30} {:>6} {:>12.2} {:>12.2} {:>12.2} {:>12.2}\n",
                    ks.name, ks.count, ks.mean_us, ks.min_us, ks.max_us, ks.stddev_us
                ));
                if let Some(bw) = ks.bandwidth_efficiency {
                    out.push_str(&format!("  Bandwidth efficiency: {:.1}%\n", bw * 100.0));
                }
                if let Some(ai) = ks.arithmetic_intensity {
                    out.push_str(&format!("  Arithmetic intensity: {:.2} FLOP/B\n", ai));
                }
                if let Some(occ) = ks.occupancy_achieved {
                    out.push_str(&format!("  Occupancy achieved: {:.1}%\n", occ * 100.0));
                }
            }
            out.push('\n');
        }

        if !self.scopes.is_empty() {
            out.push_str("Profiling Scopes:\n");
            for scope in &self.scopes {
                render_scope_text(&mut out, scope);
            }
            out.push('\n');
        }

        if !self.regressions.is_empty() {
            out.push_str("Regression Results:\n");
            for r in &self.regressions {
                let status = if r.is_regression { "REGRESSION" } else { "OK" };
                out.push_str(&format!(
                    "  {} [{}]: baseline={:.2}µs current={:.2}µs change={:+.1}%\n",
                    r.kernel_name, status, r.baseline_mean_us, r.current_mean_us, r.change_percent
                ));
            }
        }

        out
    }

    /// Render the report as JSON.
    pub fn render_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str(&format!("  \"title\": \"{}\",\n", escape_json_str(&self.title)));
        json.push_str(&format!(
            "  \"total_duration_ms\": {:.3},\n",
            self.total_duration.as_secs_f64() * 1e3
        ));

        json.push_str("  \"kernel_stats\": [\n");
        for (i, ks) in self.kernel_stats.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str("    {\n");
            json.push_str(&format!("      \"name\": \"{}\",\n", escape_json_str(&ks.name)));
            json.push_str(&format!("      \"count\": {},\n", ks.count));
            json.push_str(&format!("      \"mean_us\": {:.2},\n", ks.mean_us));
            json.push_str(&format!("      \"min_us\": {:.2},\n", ks.min_us));
            json.push_str(&format!("      \"max_us\": {:.2},\n", ks.max_us));
            json.push_str(&format!("      \"stddev_us\": {:.2}", ks.stddev_us));
            if let Some(bw) = ks.bandwidth_efficiency {
                json.push_str(&format!(",\n      \"bandwidth_efficiency\": {:.4}", bw));
            }
            if let Some(ai) = ks.arithmetic_intensity {
                json.push_str(&format!(",\n      \"arithmetic_intensity\": {:.4}", ai));
            }
            if let Some(occ) = ks.occupancy_achieved {
                json.push_str(&format!(",\n      \"occupancy_achieved\": {:.4}", occ));
            }
            json.push_str("\n    }");
        }
        json.push_str("\n  ],\n");

        json.push_str("  \"regressions\": [\n");
        for (i, r) in self.regressions.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str("    {\n");
            json.push_str(&format!(
                "      \"kernel_name\": \"{}\",\n",
                escape_json_str(&r.kernel_name)
            ));
            json.push_str(&format!("      \"baseline_mean_us\": {:.2},\n", r.baseline_mean_us));
            json.push_str(&format!("      \"current_mean_us\": {:.2},\n", r.current_mean_us));
            json.push_str(&format!("      \"change_percent\": {:.2},\n", r.change_percent));
            json.push_str(&format!("      \"is_regression\": {}\n", r.is_regression));
            json.push_str("    }");
        }
        json.push_str("\n  ]\n");

        json.push('}');
        json
    }

    /// Render the report in the specified format.
    pub fn render(&self, format: ReportFormat) -> String {
        match format {
            ReportFormat::Text => self.render_text(),
            ReportFormat::Json => self.render_json(),
        }
    }

    /// Whether any regressions were detected.
    pub fn has_regressions(&self) -> bool {
        self.regressions.iter().any(|r| r.is_regression)
    }
}

fn render_scope_text(out: &mut String, scope: &ProfilingScope) {
    let indent = "  ".repeat(scope.depth as usize);
    out.push_str(&format!(
        "{}{} — {:.3} ms (self: {:.3} ms, kernels: {})\n",
        indent,
        scope.name,
        scope.duration.as_secs_f64() * 1e3,
        scope.self_time().as_secs_f64() * 1e3,
        scope.kernel_count,
    ));
    for child in &scope.children {
        render_scope_text(out, child);
    }
}

/// Minimal JSON string escaping.
fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ── RegressionDetector ───────────────────────────────────────────────

/// Baseline throughput entry for a kernel.
#[derive(Debug, Clone)]
pub struct BaselineEntry {
    /// Kernel name.
    pub kernel_name: String,
    /// Baseline mean latency in microseconds.
    pub mean_us: f64,
    /// Baseline standard deviation in microseconds.
    pub stddev_us: f64,
}

impl BaselineEntry {
    /// Create a new baseline entry.
    pub fn new(kernel_name: impl Into<String>, mean_us: f64, stddev_us: f64) -> Self {
        Self { kernel_name: kernel_name.into(), mean_us, stddev_us }
    }
}

/// Result of a regression check for a single kernel.
#[derive(Debug, Clone)]
pub struct RegressionResult {
    /// Kernel name.
    pub kernel_name: String,
    /// Baseline mean latency in µs.
    pub baseline_mean_us: f64,
    /// Current mean latency in µs.
    pub current_mean_us: f64,
    /// Percentage change from baseline (positive = slower).
    pub change_percent: f64,
    /// Whether this is classified as a regression.
    pub is_regression: bool,
}

/// Detects throughput regressions by comparing current measurements against
/// a stored baseline.
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    /// Baseline entries keyed by kernel name.
    baselines: HashMap<String, BaselineEntry>,
    /// Percentage threshold above which a slowdown is a regression.
    pub threshold_percent: f64,
    /// Minimum number of samples required to compare.
    pub min_samples: usize,
}

impl RegressionDetector {
    /// Create a new detector with the given threshold.
    ///
    /// `threshold_percent` is the percentage increase in latency that
    /// constitutes a regression (e.g., 10.0 means >10% slower).
    pub fn new(threshold_percent: f64) -> Self {
        Self { baselines: HashMap::new(), threshold_percent, min_samples: 1 }
    }

    /// Set the minimum number of samples required.
    pub fn with_min_samples(mut self, min: usize) -> Self {
        self.min_samples = min;
        self
    }

    /// Register a baseline for a kernel.
    pub fn add_baseline(&mut self, entry: BaselineEntry) {
        self.baselines.insert(entry.kernel_name.clone(), entry);
    }

    /// Check a kernel's current stats against its baseline.
    pub fn check(&self, stats: &KernelExecStats) -> Option<RegressionResult> {
        let baseline = self.baselines.get(&stats.name)?;
        if stats.count() < self.min_samples {
            return None;
        }
        let current_mean_us = stats.mean()?.as_secs_f64() * 1e6;
        let change_percent = if baseline.mean_us > 0.0 {
            ((current_mean_us - baseline.mean_us) / baseline.mean_us) * 100.0
        } else {
            0.0
        };
        let is_regression = change_percent > self.threshold_percent;
        Some(RegressionResult {
            kernel_name: stats.name.clone(),
            baseline_mean_us: baseline.mean_us,
            current_mean_us,
            change_percent,
            is_regression,
        })
    }

    /// Check all kernels in a stats map.
    pub fn check_all(&self, stats_map: &HashMap<String, KernelExecStats>) -> Vec<RegressionResult> {
        let mut results = Vec::new();
        for stats in stats_map.values() {
            if let Some(r) = self.check(stats) {
                results.push(r);
            }
        }
        results.sort_by(|a, b| a.kernel_name.cmp(&b.kernel_name));
        results
    }

    /// Number of registered baselines.
    pub fn baseline_count(&self) -> usize {
        self.baselines.len()
    }

    /// Whether a baseline exists for the given kernel.
    pub fn has_baseline(&self, kernel_name: &str) -> bool {
        self.baselines.contains_key(kernel_name)
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new(10.0)
    }
}

// ── KernelProfiler (top-level orchestrator) ──────────────────────────

/// Top-level CUDA kernel profiler that orchestrates event timing,
/// statistics collection, bandwidth measurement, and regression detection.
#[derive(Debug)]
pub struct KernelProfiler {
    /// Per-kernel execution statistics.
    stats: HashMap<String, KernelExecStats>,
    /// Occupancy tracker.
    occupancy: OccupancyTracker,
    /// Active GPU events (not yet stopped).
    active_events: HashMap<String, GpuEvent>,
    /// Whether profiling is enabled.
    enabled: bool,
    /// Profiling start time.
    start_time: Instant,
}

impl KernelProfiler {
    /// Create a new enabled profiler.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            occupancy: OccupancyTracker::new(),
            active_events: HashMap::new(),
            enabled: true,
            start_time: Instant::now(),
        }
    }

    /// Create a disabled profiler (no-op for all operations).
    pub fn disabled() -> Self {
        Self {
            stats: HashMap::new(),
            occupancy: OccupancyTracker::new(),
            active_events: HashMap::new(),
            enabled: false,
            start_time: Instant::now(),
        }
    }

    /// Whether profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable profiling.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Start timing a kernel execution.
    pub fn start_event(&mut self, kernel_name: impl Into<String>, stream_id: u32) {
        if !self.enabled {
            return;
        }
        let name = kernel_name.into();
        let event = GpuEvent::record_start(&name, stream_id);
        self.active_events.insert(name, event);
    }

    /// Stop timing a kernel execution and record the latency.
    pub fn stop_event(&mut self, kernel_name: &str) -> Option<Duration> {
        if !self.enabled {
            return None;
        }
        let mut event = self.active_events.remove(kernel_name)?;
        let duration = event.record_stop();
        self.stats
            .entry(kernel_name.to_string())
            .or_insert_with(|| KernelExecStats::new(kernel_name))
            .record(duration);
        Some(duration)
    }

    /// Directly record a latency for a kernel (no event pair needed).
    pub fn record_latency(&mut self, kernel_name: impl Into<String>, duration: Duration) {
        if !self.enabled {
            return;
        }
        let name = kernel_name.into();
        self.stats
            .entry(name.clone())
            .or_insert_with(|| KernelExecStats::new(&name))
            .record(duration);
    }

    /// Record occupancy for a kernel.
    pub fn record_occupancy(&mut self, kernel_name: impl Into<String>, record: OccupancyRecord) {
        if !self.enabled {
            return;
        }
        self.occupancy.record(kernel_name, record);
    }

    /// Get stats for a specific kernel.
    pub fn stats(&self, kernel_name: &str) -> Option<&KernelExecStats> {
        self.stats.get(kernel_name)
    }

    /// Get the occupancy tracker.
    pub fn occupancy_tracker(&self) -> &OccupancyTracker {
        &self.occupancy
    }

    /// All tracked kernel names.
    pub fn kernel_names(&self) -> Vec<&str> {
        self.stats.keys().map(|s| s.as_str()).collect()
    }

    /// Total number of kernel invocations recorded.
    pub fn total_invocations(&self) -> usize {
        self.stats.values().map(|s| s.count()).sum()
    }

    /// Number of active (unstopped) events.
    pub fn active_event_count(&self) -> usize {
        self.active_events.len()
    }

    /// Total wall-clock time since profiler creation.
    pub fn wall_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Generate a profiling report.
    pub fn report(&self, title: impl Into<String>) -> ProfilingReport {
        let mut report = ProfilingReport::new(title, self.wall_time());
        for (name, stats) in &self.stats {
            if let (Some(mean), Some(min), Some(max), Some(stddev)) =
                (stats.mean(), stats.min(), stats.max(), stats.stddev_secs())
            {
                let occ = self.occupancy.mean_achieved(name);
                report.add_kernel_stats(KernelStatsSummary {
                    name: name.clone(),
                    count: stats.count(),
                    mean_us: mean.as_secs_f64() * 1e6,
                    min_us: min.as_secs_f64() * 1e6,
                    max_us: max.as_secs_f64() * 1e6,
                    stddev_us: stddev * 1e6,
                    bandwidth_efficiency: None,
                    arithmetic_intensity: None,
                    occupancy_achieved: occ,
                });
            }
        }
        report
    }

    /// Reset all collected data.
    pub fn reset(&mut self) {
        self.stats.clear();
        self.occupancy = OccupancyTracker::new();
        self.active_events.clear();
        self.start_time = Instant::now();
    }

    /// Reference to the raw stats map.
    pub fn stats_map(&self) -> &HashMap<String, KernelExecStats> {
        &self.stats
    }
}

impl Default for KernelProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // ── GpuEvent tests ───────────────────────────────────────────────

    #[test]
    fn gpu_event_record_start() {
        let ev = GpuEvent::record_start("gemm", 0);
        assert_eq!(ev.label, "gemm");
        assert_eq!(ev.stream_id, 0);
        assert!(!ev.is_complete());
    }

    #[test]
    fn gpu_event_record_stop() {
        let mut ev = GpuEvent::record_start("softmax", 1);
        thread::sleep(Duration::from_millis(5));
        let d = ev.record_stop();
        assert!(d >= Duration::from_millis(1));
        assert!(ev.is_complete());
    }

    #[test]
    fn gpu_event_elapsed_ms_before_stop() {
        let ev = GpuEvent::record_start("k", 0);
        thread::sleep(Duration::from_millis(5));
        assert!(ev.elapsed_ms() >= 1.0);
    }

    #[test]
    fn gpu_event_elapsed_ms_after_stop() {
        let mut ev = GpuEvent::record_start("k", 0);
        thread::sleep(Duration::from_millis(5));
        ev.record_stop();
        let ms1 = ev.elapsed_ms();
        thread::sleep(Duration::from_millis(10));
        let ms2 = ev.elapsed_ms();
        assert!((ms1 - ms2).abs() < 0.1);
    }

    #[test]
    fn gpu_event_duration_returns_elapsed() {
        let mut ev = GpuEvent::record_start("k", 0);
        thread::sleep(Duration::from_millis(5));
        ev.record_stop();
        assert!(ev.duration() >= Duration::from_millis(1));
    }

    #[test]
    fn gpu_event_stream_id_preserved() {
        let ev = GpuEvent::record_start("k", 42);
        assert_eq!(ev.stream_id, 42);
    }

    // ── KernelExecStats tests ────────────────────────────────────────

    #[test]
    fn exec_stats_new_empty() {
        let s = KernelExecStats::new("gemm");
        assert_eq!(s.name, "gemm");
        assert_eq!(s.count(), 0);
        assert!(s.min().is_none());
        assert!(s.max().is_none());
        assert!(s.mean().is_none());
        assert!(s.stddev_secs().is_none());
    }

    #[test]
    fn exec_stats_record_and_count() {
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_millis(10));
        s.record(Duration::from_millis(20));
        assert_eq!(s.count(), 2);
    }

    #[test]
    fn exec_stats_min_max() {
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_millis(10));
        s.record(Duration::from_millis(30));
        s.record(Duration::from_millis(20));
        assert_eq!(s.min().unwrap(), Duration::from_millis(10));
        assert_eq!(s.max().unwrap(), Duration::from_millis(30));
    }

    #[test]
    fn exec_stats_mean() {
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_millis(10));
        s.record(Duration::from_millis(20));
        s.record(Duration::from_millis(30));
        assert_eq!(s.mean().unwrap(), Duration::from_millis(20));
    }

    #[test]
    fn exec_stats_stddev_zero_for_identical() {
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_millis(10));
        s.record(Duration::from_millis(10));
        s.record(Duration::from_millis(10));
        assert!(s.stddev_secs().unwrap() < 1e-9);
    }

    #[test]
    fn exec_stats_stddev_positive_for_varied() {
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_millis(10));
        s.record(Duration::from_millis(30));
        assert!(s.stddev_secs().unwrap() > 0.0);
    }

    #[test]
    fn exec_stats_median_odd() {
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_millis(30));
        s.record(Duration::from_millis(10));
        s.record(Duration::from_millis(20));
        assert_eq!(s.median().unwrap(), Duration::from_millis(20));
    }

    #[test]
    fn exec_stats_median_even() {
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_millis(10));
        s.record(Duration::from_millis(20));
        // With 2 elements, median picks index 1 (upper middle).
        assert_eq!(s.median().unwrap(), Duration::from_millis(20));
    }

    #[test]
    fn exec_stats_p95() {
        let mut s = KernelExecStats::new("k");
        for i in 0..100 {
            s.record(Duration::from_micros(i * 10));
        }
        let p95 = s.p95().unwrap();
        assert!(p95 >= Duration::from_micros(900));
    }

    #[test]
    fn exec_stats_p99() {
        let mut s = KernelExecStats::new("k");
        for i in 0..100 {
            s.record(Duration::from_micros(i * 10));
        }
        let p99 = s.p99().unwrap();
        assert!(p99 >= Duration::from_micros(950));
    }

    #[test]
    fn exec_stats_p95_empty() {
        let s = KernelExecStats::new("k");
        assert!(s.p95().is_none());
    }

    #[test]
    fn exec_stats_p99_empty() {
        let s = KernelExecStats::new("k");
        assert!(s.p99().is_none());
    }

    #[test]
    fn exec_stats_clear() {
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_millis(10));
        s.clear();
        assert_eq!(s.count(), 0);
        assert!(s.mean().is_none());
    }

    #[test]
    fn exec_stats_single_sample() {
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_millis(42));
        assert_eq!(s.count(), 1);
        assert_eq!(s.min().unwrap(), Duration::from_millis(42));
        assert_eq!(s.max().unwrap(), Duration::from_millis(42));
        assert_eq!(s.mean().unwrap(), Duration::from_millis(42));
        assert!(s.stddev_secs().unwrap() < 1e-9);
        assert_eq!(s.median().unwrap(), Duration::from_millis(42));
    }

    // ── MemoryBandwidth tests ────────────────────────────────────────

    #[test]
    fn bandwidth_total_bytes() {
        let bw = MemoryBandwidth::new(100, 50, Duration::from_secs(1), 1e9);
        assert_eq!(bw.total_bytes(), 150);
    }

    #[test]
    fn bandwidth_effective_bps() {
        let bw = MemoryBandwidth::new(500_000_000, 500_000_000, Duration::from_secs(1), 2e9);
        assert!((bw.effective_bps() - 1e9).abs() < 1e3);
    }

    #[test]
    fn bandwidth_effective_gb_s() {
        let bw = MemoryBandwidth::new(1_000_000_000, 0, Duration::from_secs(1), 2e9);
        assert!((bw.effective_gb_s() - 1.0).abs() < 0.01);
    }

    #[test]
    fn bandwidth_efficiency() {
        let bw = MemoryBandwidth::new(500_000_000, 500_000_000, Duration::from_secs(1), 2e9);
        assert!((bw.efficiency() - 0.5).abs() < 0.01);
    }

    #[test]
    fn bandwidth_efficiency_clamped() {
        let bw = MemoryBandwidth::new(2_000_000_000, 0, Duration::from_secs(1), 1e9);
        assert!((bw.efficiency() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bandwidth_zero_duration() {
        let bw = MemoryBandwidth::new(100, 100, Duration::ZERO, 1e9);
        assert_eq!(bw.effective_bps(), 0.0);
    }

    #[test]
    fn bandwidth_zero_peak() {
        let bw = MemoryBandwidth::new(100, 100, Duration::from_secs(1), 0.0);
        assert_eq!(bw.efficiency(), 0.0);
    }

    #[test]
    fn bandwidth_gap_gb_s() {
        let bw = MemoryBandwidth::new(500_000_000, 0, Duration::from_secs(1), 1e9);
        assert!((bw.gap_gb_s() - 0.5).abs() < 0.01);
    }

    // ── ArithmeticIntensity tests ────────────────────────────────────

    #[test]
    fn arithmetic_intensity_basic() {
        let ai = ArithmeticIntensity::new(1000, 100);
        assert!((ai.intensity() - 10.0).abs() < 0.01);
    }

    #[test]
    fn arithmetic_intensity_zero_bytes() {
        let ai = ArithmeticIntensity::new(1000, 0);
        assert_eq!(ai.intensity(), 0.0);
    }

    #[test]
    fn arithmetic_intensity_compute_bound() {
        let ai = ArithmeticIntensity::new(1000, 100);
        // Intensity = 10, balance point = 5 → compute-bound
        assert!(ai.is_compute_bound(5.0));
        assert!(!ai.is_memory_bound(5.0));
    }

    #[test]
    fn arithmetic_intensity_memory_bound() {
        let ai = ArithmeticIntensity::new(100, 1000);
        // Intensity = 0.1, balance point = 5 → memory-bound
        assert!(ai.is_memory_bound(5.0));
        assert!(!ai.is_compute_bound(5.0));
    }

    #[test]
    fn arithmetic_intensity_roofline_compute_limited() {
        let ai = ArithmeticIntensity::new(10000, 100);
        // Intensity = 100, peak_flops = 1000, peak_bw = 100
        // mem_roof = 100 * 100 = 10000 → min(1000, 10000) = 1000
        let peak = ai.roofline_peak(1000.0, 100.0);
        assert!((peak - 1000.0).abs() < 0.01);
    }

    #[test]
    fn arithmetic_intensity_roofline_memory_limited() {
        let ai = ArithmeticIntensity::new(100, 1000);
        // Intensity = 0.1, peak_flops = 1000, peak_bw = 100
        // mem_roof = 0.1 * 100 = 10 → min(1000, 10) = 10
        let peak = ai.roofline_peak(1000.0, 100.0);
        assert!((peak - 10.0).abs() < 0.01);
    }

    #[test]
    fn arithmetic_intensity_roofline_zero_bw() {
        let ai = ArithmeticIntensity::new(100, 100);
        assert_eq!(ai.roofline_peak(1000.0, 0.0), 0.0);
    }

    #[test]
    fn arithmetic_intensity_display() {
        let ai = ArithmeticIntensity::new(1000, 100);
        let s = format!("{ai}");
        assert!(s.contains("FLOP/B"));
    }

    #[test]
    fn arithmetic_intensity_at_balance_point() {
        let ai = ArithmeticIntensity::new(500, 100);
        // Intensity = 5.0, exactly at balance point → compute-bound
        assert!(ai.is_compute_bound(5.0));
        assert!(!ai.is_memory_bound(5.0));
    }

    // ── OccupancyRecord tests ────────────────────────────────────────

    #[test]
    fn occupancy_record_new() {
        let r = OccupancyRecord::new(0.75, 0.60, 48, 64, 256, 4096, 32);
        assert!((r.theoretical - 0.75).abs() < f64::EPSILON);
        assert!((r.achieved - 0.60).abs() < f64::EPSILON);
        assert_eq!(r.active_warps, 48);
        assert_eq!(r.max_warps, 64);
    }

    #[test]
    fn occupancy_record_clamps() {
        let r = OccupancyRecord::new(1.5, -0.1, 64, 64, 256, 0, 0);
        assert!((r.theoretical - 1.0).abs() < f64::EPSILON);
        assert!((r.achieved - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn occupancy_record_gap() {
        let r = OccupancyRecord::new(0.75, 0.50, 32, 64, 256, 0, 0);
        assert!((r.gap() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn occupancy_record_gap_zero_when_achieved_matches() {
        let r = OccupancyRecord::new(0.75, 0.75, 48, 64, 256, 0, 0);
        assert!(r.gap() < f64::EPSILON);
    }

    #[test]
    fn occupancy_record_is_near_theoretical() {
        let r = OccupancyRecord::new(0.75, 0.72, 46, 64, 256, 0, 0);
        assert!(r.is_near_theoretical(0.05));
        assert!(!r.is_near_theoretical(0.01));
    }

    #[test]
    fn occupancy_record_gap_non_negative() {
        let r = OccupancyRecord::new(0.50, 0.75, 48, 64, 256, 0, 0);
        assert!(r.gap() >= 0.0);
    }

    // ── OccupancyTracker tests ───────────────────────────────────────

    #[test]
    fn tracker_new_empty() {
        let t = OccupancyTracker::new();
        assert_eq!(t.kernel_count(), 0);
        assert_eq!(t.total_records(), 0);
    }

    #[test]
    fn tracker_record_and_retrieve() {
        let mut t = OccupancyTracker::new();
        let r = OccupancyRecord::new(0.75, 0.60, 48, 64, 256, 0, 0);
        t.record("gemm", r);
        assert_eq!(t.records_for("gemm").len(), 1);
        assert_eq!(t.kernel_count(), 1);
    }

    #[test]
    fn tracker_mean_achieved() {
        let mut t = OccupancyTracker::new();
        t.record("k", OccupancyRecord::new(0.75, 0.60, 48, 64, 256, 0, 0));
        t.record("k", OccupancyRecord::new(0.75, 0.80, 48, 64, 256, 0, 0));
        let mean = t.mean_achieved("k").unwrap();
        assert!((mean - 0.70).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_mean_achieved_missing() {
        let t = OccupancyTracker::new();
        assert!(t.mean_achieved("missing").is_none());
    }

    #[test]
    fn tracker_kernel_names() {
        let mut t = OccupancyTracker::new();
        let r = OccupancyRecord::new(0.5, 0.5, 32, 64, 128, 0, 0);
        t.record("a", r);
        t.record("b", r);
        let mut names = t.kernel_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn tracker_total_records() {
        let mut t = OccupancyTracker::new();
        let r = OccupancyRecord::new(0.5, 0.5, 32, 64, 128, 0, 0);
        t.record("a", r);
        t.record("a", r);
        t.record("b", r);
        assert_eq!(t.total_records(), 3);
    }

    #[test]
    fn tracker_records_for_missing() {
        let t = OccupancyTracker::new();
        assert!(t.records_for("none").is_empty());
    }

    // ── ProfilingScope tests ─────────────────────────────────────────

    #[test]
    fn scope_leaf() {
        let s = ProfilingScope::leaf("gemm", 0, Duration::from_millis(10));
        assert_eq!(s.name, "gemm");
        assert_eq!(s.depth, 0);
        assert!(s.children.is_empty());
        assert_eq!(s.kernel_count, 0);
    }

    #[test]
    fn scope_with_children() {
        let child = ProfilingScope::leaf("inner", 1, Duration::from_millis(5));
        let parent =
            ProfilingScope::with_children("outer", 0, Duration::from_millis(10), vec![child]);
        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.children[0].name, "inner");
    }

    #[test]
    fn scope_with_kernel_count() {
        let s = ProfilingScope::leaf("k", 0, Duration::from_millis(1)).with_kernel_count(5);
        assert_eq!(s.kernel_count, 5);
    }

    #[test]
    fn scope_children_duration() {
        let c1 = ProfilingScope::leaf("c1", 1, Duration::from_millis(3));
        let c2 = ProfilingScope::leaf("c2", 1, Duration::from_millis(7));
        let parent = ProfilingScope::with_children("p", 0, Duration::from_millis(15), vec![c1, c2]);
        assert_eq!(parent.children_duration(), Duration::from_millis(10));
    }

    #[test]
    fn scope_self_time() {
        let child = ProfilingScope::leaf("c", 1, Duration::from_millis(6));
        let parent = ProfilingScope::with_children("p", 0, Duration::from_millis(10), vec![child]);
        assert_eq!(parent.self_time(), Duration::from_millis(4));
    }

    #[test]
    fn scope_self_time_no_children() {
        let s = ProfilingScope::leaf("k", 0, Duration::from_millis(10));
        assert_eq!(s.self_time(), Duration::from_millis(10));
    }

    #[test]
    fn scope_self_time_saturates() {
        // Children duration exceeds parent (shouldn't happen, but must not panic).
        let child = ProfilingScope::leaf("c", 1, Duration::from_millis(20));
        let parent = ProfilingScope::with_children("p", 0, Duration::from_millis(10), vec![child]);
        assert_eq!(parent.self_time(), Duration::ZERO);
    }

    #[test]
    fn scope_descendant_count() {
        let gc = ProfilingScope::leaf("gc", 2, Duration::from_millis(1));
        let child = ProfilingScope::with_children("c", 1, Duration::from_millis(5), vec![gc]);
        let parent = ProfilingScope::with_children("p", 0, Duration::from_millis(10), vec![child]);
        assert_eq!(parent.descendant_count(), 2);
    }

    #[test]
    fn scope_max_depth_leaf() {
        let s = ProfilingScope::leaf("k", 0, Duration::from_millis(1));
        assert_eq!(s.max_depth(), 0);
    }

    #[test]
    fn scope_max_depth_nested() {
        let gc = ProfilingScope::leaf("gc", 2, Duration::from_millis(1));
        let child = ProfilingScope::with_children("c", 1, Duration::from_millis(5), vec![gc]);
        let root = ProfilingScope::with_children("root", 0, Duration::from_millis(10), vec![child]);
        assert_eq!(root.max_depth(), 2);
    }

    #[test]
    fn scope_flatten() {
        let child = ProfilingScope::leaf("c", 1, Duration::from_millis(5));
        let root = ProfilingScope::with_children("r", 0, Duration::from_millis(10), vec![child]);
        let flat = root.flatten();
        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0].name, "r");
        assert_eq!(flat[1].name, "c");
    }

    #[test]
    fn scope_flatten_deep() {
        let gc = ProfilingScope::leaf("gc", 2, Duration::from_millis(1));
        let child = ProfilingScope::with_children("c", 1, Duration::from_millis(5), vec![gc]);
        let root = ProfilingScope::with_children("r", 0, Duration::from_millis(10), vec![child]);
        let flat = root.flatten();
        assert_eq!(flat.len(), 3);
        assert_eq!(flat[2].name, "gc");
    }

    // ── ScopeBuilder tests ───────────────────────────────────────────

    #[test]
    fn scope_builder_empty() {
        let b = ScopeBuilder::new();
        assert!(b.is_empty());
        assert_eq!(b.depth(), 0);
    }

    #[test]
    fn scope_builder_default() {
        let b = ScopeBuilder::default();
        assert!(b.is_empty());
    }

    #[test]
    fn scope_builder_enter_exit() {
        let mut b = ScopeBuilder::new();
        b.enter("root");
        assert_eq!(b.depth(), 1);
        let scope = b.exit().unwrap();
        assert_eq!(scope.name, "root");
        assert!(b.is_empty());
    }

    #[test]
    fn scope_builder_nested() {
        let mut b = ScopeBuilder::new();
        b.enter("outer");
        b.enter("inner");
        assert_eq!(b.depth(), 2);
        let inner = b.exit().unwrap();
        assert_eq!(inner.name, "inner");
        assert_eq!(inner.depth, 1);
        let outer = b.exit().unwrap();
        assert_eq!(outer.name, "outer");
        assert_eq!(outer.children.len(), 1);
        assert_eq!(outer.children[0].name, "inner");
    }

    #[test]
    fn scope_builder_record_kernel() {
        let mut b = ScopeBuilder::new();
        b.enter("scope");
        b.record_kernel();
        b.record_kernel();
        let scope = b.exit().unwrap();
        assert_eq!(scope.kernel_count, 2);
    }

    #[test]
    fn scope_builder_exit_empty_returns_none() {
        let mut b = ScopeBuilder::new();
        assert!(b.exit().is_none());
    }

    #[test]
    fn scope_builder_three_levels() {
        let mut b = ScopeBuilder::new();
        b.enter("l0");
        b.enter("l1");
        b.enter("l2");
        b.record_kernel();
        let l2 = b.exit().unwrap();
        assert_eq!(l2.depth, 2);
        assert_eq!(l2.kernel_count, 1);
        let l1 = b.exit().unwrap();
        assert_eq!(l1.children.len(), 1);
        let l0 = b.exit().unwrap();
        assert_eq!(l0.children.len(), 1);
        assert_eq!(l0.children[0].children.len(), 1);
    }

    // ── ProfilingReport tests ────────────────────────────────────────

    #[test]
    fn report_new_empty() {
        let r = ProfilingReport::new("Test", Duration::from_secs(1));
        assert_eq!(r.title, "Test");
        assert!(r.kernel_stats.is_empty());
        assert!(!r.has_regressions());
    }

    #[test]
    fn report_render_text_contains_title() {
        let r = ProfilingReport::new("My Report", Duration::from_secs(1));
        let text = r.render_text();
        assert!(text.contains("My Report"));
    }

    #[test]
    fn report_render_text_with_stats() {
        let mut r = ProfilingReport::new("Report", Duration::from_secs(1));
        r.add_kernel_stats(KernelStatsSummary {
            name: "gemm".to_string(),
            count: 10,
            mean_us: 100.0,
            min_us: 80.0,
            max_us: 120.0,
            stddev_us: 10.0,
            bandwidth_efficiency: Some(0.75),
            arithmetic_intensity: Some(12.5),
            occupancy_achieved: Some(0.85),
        });
        let text = r.render_text();
        assert!(text.contains("gemm"));
        assert!(text.contains("Bandwidth efficiency"));
        assert!(text.contains("Arithmetic intensity"));
        assert!(text.contains("Occupancy achieved"));
    }

    #[test]
    fn report_render_text_with_scopes() {
        let mut r = ProfilingReport::new("Report", Duration::from_secs(1));
        let child = ProfilingScope::leaf("inner", 1, Duration::from_millis(5));
        let scope =
            ProfilingScope::with_children("outer", 0, Duration::from_millis(10), vec![child]);
        r.add_scope(scope);
        let text = r.render_text();
        assert!(text.contains("outer"));
        assert!(text.contains("inner"));
        assert!(text.contains("Profiling Scopes"));
    }

    #[test]
    fn report_render_json_valid_structure() {
        let r = ProfilingReport::new("Test", Duration::from_millis(500));
        let json = r.render_json();
        assert!(json.contains("\"title\": \"Test\""));
        assert!(json.contains("\"kernel_stats\""));
        assert!(json.contains("\"regressions\""));
    }

    #[test]
    fn report_render_json_with_stats() {
        let mut r = ProfilingReport::new("JSON Report", Duration::from_secs(1));
        r.add_kernel_stats(KernelStatsSummary {
            name: "softmax".to_string(),
            count: 5,
            mean_us: 50.0,
            min_us: 40.0,
            max_us: 60.0,
            stddev_us: 5.0,
            bandwidth_efficiency: None,
            arithmetic_intensity: None,
            occupancy_achieved: None,
        });
        let json = r.render_json();
        assert!(json.contains("\"name\": \"softmax\""));
        assert!(json.contains("\"count\": 5"));
    }

    #[test]
    fn report_render_json_with_regression() {
        let mut r = ProfilingReport::new("Reg", Duration::from_secs(1));
        r.add_regression(RegressionResult {
            kernel_name: "gemm".to_string(),
            baseline_mean_us: 100.0,
            current_mean_us: 120.0,
            change_percent: 20.0,
            is_regression: true,
        });
        let json = r.render_json();
        assert!(json.contains("\"is_regression\": true"));
    }

    #[test]
    fn report_render_format_dispatch() {
        let r = ProfilingReport::new("F", Duration::from_secs(1));
        let text = r.render(ReportFormat::Text);
        let json = r.render(ReportFormat::Json);
        assert!(text.contains("==="));
        assert!(json.contains('{'));
    }

    #[test]
    fn report_has_regressions_false() {
        let mut r = ProfilingReport::new("R", Duration::from_secs(1));
        r.add_regression(RegressionResult {
            kernel_name: "k".to_string(),
            baseline_mean_us: 100.0,
            current_mean_us: 105.0,
            change_percent: 5.0,
            is_regression: false,
        });
        assert!(!r.has_regressions());
    }

    #[test]
    fn report_has_regressions_true() {
        let mut r = ProfilingReport::new("R", Duration::from_secs(1));
        r.add_regression(RegressionResult {
            kernel_name: "k".to_string(),
            baseline_mean_us: 100.0,
            current_mean_us: 130.0,
            change_percent: 30.0,
            is_regression: true,
        });
        assert!(r.has_regressions());
    }

    #[test]
    fn report_render_text_with_regression() {
        let mut r = ProfilingReport::new("Reg Report", Duration::from_secs(1));
        r.add_regression(RegressionResult {
            kernel_name: "gemm".to_string(),
            baseline_mean_us: 100.0,
            current_mean_us: 115.0,
            change_percent: 15.0,
            is_regression: true,
        });
        let text = r.render_text();
        assert!(text.contains("REGRESSION"));
        assert!(text.contains("gemm"));
    }

    // ── BaselineEntry tests ──────────────────────────────────────────

    #[test]
    fn baseline_entry_new() {
        let e = BaselineEntry::new("gemm", 100.0, 5.0);
        assert_eq!(e.kernel_name, "gemm");
        assert!((e.mean_us - 100.0).abs() < f64::EPSILON);
        assert!((e.stddev_us - 5.0).abs() < f64::EPSILON);
    }

    // ── RegressionDetector tests ─────────────────────────────────────

    #[test]
    fn regression_detector_default() {
        let d = RegressionDetector::default();
        assert!((d.threshold_percent - 10.0).abs() < f64::EPSILON);
        assert_eq!(d.baseline_count(), 0);
    }

    #[test]
    fn regression_detector_new() {
        let d = RegressionDetector::new(5.0);
        assert!((d.threshold_percent - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn regression_detector_add_baseline() {
        let mut d = RegressionDetector::new(10.0);
        d.add_baseline(BaselineEntry::new("gemm", 100.0, 5.0));
        assert_eq!(d.baseline_count(), 1);
        assert!(d.has_baseline("gemm"));
        assert!(!d.has_baseline("softmax"));
    }

    #[test]
    fn regression_detector_no_baseline_returns_none() {
        let d = RegressionDetector::new(10.0);
        let mut s = KernelExecStats::new("gemm");
        s.record(Duration::from_micros(100));
        assert!(d.check(&s).is_none());
    }

    #[test]
    fn regression_detector_detects_regression() {
        let mut d = RegressionDetector::new(10.0);
        d.add_baseline(BaselineEntry::new("gemm", 100.0, 5.0));
        let mut s = KernelExecStats::new("gemm");
        // 120µs is 20% slower than 100µs baseline
        s.record(Duration::from_micros(120));
        let r = d.check(&s).unwrap();
        assert!(r.is_regression);
        assert!(r.change_percent > 10.0);
    }

    #[test]
    fn regression_detector_no_regression() {
        let mut d = RegressionDetector::new(10.0);
        d.add_baseline(BaselineEntry::new("gemm", 100.0, 5.0));
        let mut s = KernelExecStats::new("gemm");
        s.record(Duration::from_micros(105));
        let r = d.check(&s).unwrap();
        assert!(!r.is_regression);
    }

    #[test]
    fn regression_detector_faster_is_not_regression() {
        let mut d = RegressionDetector::new(10.0);
        d.add_baseline(BaselineEntry::new("gemm", 100.0, 5.0));
        let mut s = KernelExecStats::new("gemm");
        s.record(Duration::from_micros(80));
        let r = d.check(&s).unwrap();
        assert!(!r.is_regression);
        assert!(r.change_percent < 0.0);
    }

    #[test]
    fn regression_detector_min_samples() {
        let mut d = RegressionDetector::new(10.0).with_min_samples(3);
        d.add_baseline(BaselineEntry::new("gemm", 100.0, 5.0));
        let mut s = KernelExecStats::new("gemm");
        s.record(Duration::from_micros(200));
        // Only 1 sample, needs 3 → returns None.
        assert!(d.check(&s).is_none());
        s.record(Duration::from_micros(200));
        s.record(Duration::from_micros(200));
        assert!(d.check(&s).is_some());
    }

    #[test]
    fn regression_detector_check_all() {
        let mut d = RegressionDetector::new(10.0);
        d.add_baseline(BaselineEntry::new("gemm", 100.0, 5.0));
        d.add_baseline(BaselineEntry::new("softmax", 50.0, 2.0));

        let mut stats_map = HashMap::new();
        let mut s1 = KernelExecStats::new("gemm");
        s1.record(Duration::from_micros(120));
        stats_map.insert("gemm".to_string(), s1);
        let mut s2 = KernelExecStats::new("softmax");
        s2.record(Duration::from_micros(52));
        stats_map.insert("softmax".to_string(), s2);

        let results = d.check_all(&stats_map);
        assert_eq!(results.len(), 2);
        // Results are sorted by kernel name.
        assert_eq!(results[0].kernel_name, "gemm");
        assert!(results[0].is_regression);
        assert_eq!(results[1].kernel_name, "softmax");
        assert!(!results[1].is_regression);
    }

    #[test]
    fn regression_detector_zero_baseline_no_div_by_zero() {
        let mut d = RegressionDetector::new(10.0);
        d.add_baseline(BaselineEntry::new("k", 0.0, 0.0));
        let mut s = KernelExecStats::new("k");
        s.record(Duration::from_micros(100));
        let r = d.check(&s).unwrap();
        assert!((r.change_percent - 0.0).abs() < f64::EPSILON);
    }

    // ── KernelProfiler tests ─────────────────────────────────────────

    #[test]
    fn profiler_new_enabled() {
        let p = KernelProfiler::new();
        assert!(p.is_enabled());
        assert_eq!(p.total_invocations(), 0);
        assert_eq!(p.active_event_count(), 0);
    }

    #[test]
    fn profiler_default() {
        let p = KernelProfiler::default();
        assert!(p.is_enabled());
    }

    #[test]
    fn profiler_disabled() {
        let p = KernelProfiler::disabled();
        assert!(!p.is_enabled());
    }

    #[test]
    fn profiler_enable_disable() {
        let mut p = KernelProfiler::new();
        p.disable();
        assert!(!p.is_enabled());
        p.enable();
        assert!(p.is_enabled());
    }

    #[test]
    fn profiler_start_stop_event() {
        let mut p = KernelProfiler::new();
        p.start_event("gemm", 0);
        assert_eq!(p.active_event_count(), 1);
        thread::sleep(Duration::from_millis(5));
        let d = p.stop_event("gemm").unwrap();
        assert!(d >= Duration::from_millis(1));
        assert_eq!(p.active_event_count(), 0);
        assert_eq!(p.total_invocations(), 1);
    }

    #[test]
    fn profiler_stop_unknown_event() {
        let mut p = KernelProfiler::new();
        assert!(p.stop_event("nonexistent").is_none());
    }

    #[test]
    fn profiler_disabled_ignores_events() {
        let mut p = KernelProfiler::disabled();
        p.start_event("k", 0);
        assert_eq!(p.active_event_count(), 0);
        assert!(p.stop_event("k").is_none());
    }

    #[test]
    fn profiler_record_latency() {
        let mut p = KernelProfiler::new();
        p.record_latency("softmax", Duration::from_micros(42));
        p.record_latency("softmax", Duration::from_micros(50));
        let stats = p.stats("softmax").unwrap();
        assert_eq!(stats.count(), 2);
    }

    #[test]
    fn profiler_record_latency_disabled() {
        let mut p = KernelProfiler::disabled();
        p.record_latency("k", Duration::from_micros(100));
        assert!(p.stats("k").is_none());
    }

    #[test]
    fn profiler_record_occupancy() {
        let mut p = KernelProfiler::new();
        let r = OccupancyRecord::new(0.75, 0.60, 48, 64, 256, 0, 0);
        p.record_occupancy("gemm", r);
        let tracker = p.occupancy_tracker();
        assert_eq!(tracker.records_for("gemm").len(), 1);
    }

    #[test]
    fn profiler_record_occupancy_disabled() {
        let mut p = KernelProfiler::disabled();
        let r = OccupancyRecord::new(0.75, 0.60, 48, 64, 256, 0, 0);
        p.record_occupancy("gemm", r);
        assert_eq!(p.occupancy_tracker().total_records(), 0);
    }

    #[test]
    fn profiler_kernel_names() {
        let mut p = KernelProfiler::new();
        p.record_latency("a", Duration::from_micros(10));
        p.record_latency("b", Duration::from_micros(20));
        let mut names = p.kernel_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn profiler_total_invocations() {
        let mut p = KernelProfiler::new();
        p.record_latency("a", Duration::from_micros(10));
        p.record_latency("a", Duration::from_micros(20));
        p.record_latency("b", Duration::from_micros(30));
        assert_eq!(p.total_invocations(), 3);
    }

    #[test]
    fn profiler_wall_time() {
        let p = KernelProfiler::new();
        thread::sleep(Duration::from_millis(5));
        assert!(p.wall_time() >= Duration::from_millis(1));
    }

    #[test]
    fn profiler_reset() {
        let mut p = KernelProfiler::new();
        p.record_latency("k", Duration::from_micros(100));
        p.start_event("x", 0);
        let r = OccupancyRecord::new(0.5, 0.5, 32, 64, 128, 0, 0);
        p.record_occupancy("k", r);
        p.reset();
        assert_eq!(p.total_invocations(), 0);
        assert_eq!(p.active_event_count(), 0);
        assert_eq!(p.occupancy_tracker().total_records(), 0);
    }

    #[test]
    fn profiler_report_generation() {
        let mut p = KernelProfiler::new();
        p.record_latency("gemm", Duration::from_micros(100));
        p.record_latency("gemm", Duration::from_micros(120));
        p.record_latency("softmax", Duration::from_micros(50));
        let report = p.report("Test Report");
        assert_eq!(report.title, "Test Report");
        assert_eq!(report.kernel_stats.len(), 2);
    }

    #[test]
    fn profiler_report_text_output() {
        let mut p = KernelProfiler::new();
        p.record_latency("gemm", Duration::from_micros(100));
        let report = p.report("Text");
        let text = report.render_text();
        assert!(text.contains("gemm"));
        assert!(text.contains("Text"));
    }

    #[test]
    fn profiler_report_json_output() {
        let mut p = KernelProfiler::new();
        p.record_latency("gemm", Duration::from_micros(100));
        let report = p.report("JSON");
        let json = report.render_json();
        assert!(json.contains("\"name\": \"gemm\""));
    }

    #[test]
    fn profiler_stats_map() {
        let mut p = KernelProfiler::new();
        p.record_latency("k", Duration::from_micros(10));
        assert!(p.stats_map().contains_key("k"));
    }

    #[test]
    fn profiler_report_includes_occupancy() {
        let mut p = KernelProfiler::new();
        p.record_latency("gemm", Duration::from_micros(100));
        let r = OccupancyRecord::new(0.75, 0.60, 48, 64, 256, 0, 0);
        p.record_occupancy("gemm", r);
        let report = p.report("Occ");
        let ks = report.kernel_stats.iter().find(|s| s.name == "gemm").unwrap();
        assert!(ks.occupancy_achieved.is_some());
    }

    // ── Integration / end-to-end tests ───────────────────────────────

    #[test]
    fn e2e_full_profiling_workflow() {
        let mut profiler = KernelProfiler::new();

        // Simulate profiling a GEMM kernel.
        profiler.start_event("gemm", 0);
        thread::sleep(Duration::from_millis(2));
        profiler.stop_event("gemm");

        // Record additional latencies.
        for i in 0..5 {
            profiler.record_latency("gemm", Duration::from_micros(100 + i * 10));
        }

        // Record occupancy.
        let occ = OccupancyRecord::new(0.75, 0.70, 48, 64, 256, 4096, 32);
        profiler.record_occupancy("gemm", occ);

        // Check stats.
        let stats = profiler.stats("gemm").unwrap();
        assert_eq!(stats.count(), 6);
        assert!(stats.min().is_some());
        assert!(stats.stddev_secs().unwrap() >= 0.0);

        // Generate report.
        let report = profiler.report("E2E Test");
        let text = report.render_text();
        assert!(text.contains("gemm"));
        let json = report.render_json();
        assert!(json.contains("gemm"));
    }

    #[test]
    fn e2e_regression_detection_workflow() {
        let mut profiler = KernelProfiler::new();
        for _ in 0..10 {
            profiler.record_latency("gemm", Duration::from_micros(120));
        }
        for _ in 0..10 {
            profiler.record_latency("softmax", Duration::from_micros(52));
        }

        let mut detector = RegressionDetector::new(10.0);
        detector.add_baseline(BaselineEntry::new("gemm", 100.0, 5.0));
        detector.add_baseline(BaselineEntry::new("softmax", 50.0, 2.0));

        let results = detector.check_all(profiler.stats_map());
        assert_eq!(results.len(), 2);

        let gemm_result = results.iter().find(|r| r.kernel_name == "gemm").unwrap();
        assert!(gemm_result.is_regression);

        let softmax_result = results.iter().find(|r| r.kernel_name == "softmax").unwrap();
        assert!(!softmax_result.is_regression);

        let mut report = profiler.report("Regression Test");
        for r in results {
            report.add_regression(r);
        }
        assert!(report.has_regressions());
    }

    #[test]
    fn e2e_hierarchical_scoping() {
        let mut builder = ScopeBuilder::new();
        builder.enter("forward_pass");
        builder.enter("attention");
        builder.record_kernel();
        builder.record_kernel();
        let _attention = builder.exit().unwrap();
        builder.enter("ffn");
        builder.record_kernel();
        let _ffn = builder.exit().unwrap();
        let root = builder.exit().unwrap();
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].name, "attention");
        assert_eq!(root.children[0].kernel_count, 2);
        assert_eq!(root.children[1].name, "ffn");
        assert_eq!(root.children[1].kernel_count, 1);
    }

    #[test]
    fn e2e_bandwidth_and_intensity() {
        let bw =
            MemoryBandwidth::new(2_000_000_000, 1_000_000_000, Duration::from_millis(10), 900e9);
        let ai = ArithmeticIntensity::new(50_000_000_000, bw.total_bytes());
        let machine_balance = 900e9 / 312e12; // ~2.88
        assert!(ai.intensity() > machine_balance);
        assert!(ai.is_compute_bound(machine_balance));
        let peak = ai.roofline_peak(312e12, 900e9);
        assert!(peak > 0.0);
    }

    #[test]
    fn e2e_multiple_streams() {
        let mut profiler = KernelProfiler::new();
        profiler.start_event("gemm_s0", 0);
        profiler.start_event("softmax_s1", 1);
        assert_eq!(profiler.active_event_count(), 2);
        profiler.stop_event("gemm_s0");
        profiler.stop_event("softmax_s1");
        assert_eq!(profiler.active_event_count(), 0);
        assert_eq!(profiler.total_invocations(), 2);
    }

    #[test]
    fn escape_json_str_basic() {
        assert_eq!(escape_json_str("hello"), "hello");
        assert_eq!(escape_json_str("a\"b"), "a\\\"b");
        assert_eq!(escape_json_str("a\\b"), "a\\\\b");
        assert_eq!(escape_json_str("a\nb"), "a\\nb");
    }
}
