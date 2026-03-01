//! Module stub - implementation pending merge from feature branch
//! Continuous profiling with CPU/GPU/memory/latency tracking.
//!
//! Provides [`ContinuousProfiler`] for production profiling with
//! configurable collectors, flame-graph generation, hotspot detection,
//! latency/throughput tracking, historical trend analysis, and
//! Prometheus/JSON/pprof export.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_const_for_fn,
    clippy::similar_names,
    clippy::suspicious_operation_groupings,
    clippy::suboptimal_flops,
    clippy::needless_pass_by_value,
    clippy::format_push_string,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_wraps,
    clippy::return_self_not_must_use,
    clippy::unnecessary_literal_bound,
    clippy::collapsible_if,
    clippy::cast_lossless
)]

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ── Configuration ─────────────────────────────────────────────────────────

/// Which collector types are enabled.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollectorKind {
    CpuTime,
    Memory,
    GpuKernel,
    Io,
}

/// Configuration for the continuous profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Samples per second. Clamped to `[1, 10_000]`.
    pub sampling_rate_hz: u32,
    /// Maximum number of samples held in the ring buffer.
    pub buffer_size: usize,
    /// Seconds between automatic exports (0 = manual only).
    pub export_interval_secs: u64,
    /// Collector types that are active.
    pub enabled_collectors: Vec<CollectorKind>,
    /// Maximum acceptable profiling overhead as a fraction (0.0–1.0).
    pub max_overhead_fraction: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            sampling_rate_hz: 100,
            buffer_size: 65_536,
            export_interval_secs: 60,
            enabled_collectors: vec![CollectorKind::CpuTime, CollectorKind::Memory],
            max_overhead_fraction: 0.01,
        }
    }
}

impl ProfilerConfig {
    /// Effective sampling rate after clamping.
    pub fn effective_rate(&self) -> u32 {
        self.sampling_rate_hz.clamp(1, 10_000)
    }

    /// Duration between two consecutive samples.
    pub fn sample_interval(&self) -> Duration {
        Duration::from_secs_f64(1.0 / f64::from(self.effective_rate()))
    }

    /// Whether the given collector kind is enabled.
    pub fn is_collector_enabled(&self, kind: &CollectorKind) -> bool {
        self.enabled_collectors.contains(kind)
    }
}

// ── Profile sample ────────────────────────────────────────────────────────

/// A single profiling sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSample {
    /// Microseconds since UNIX epoch.
    pub timestamp_us: u64,
    /// Name of the collector that produced this sample.
    pub collector_name: String,
    /// Measured value (meaning depends on collector).
    pub value: f64,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl ProfileSample {
    pub fn new(collector_name: impl Into<String>, value: f64) -> Self {
        Self {
            timestamp_us: now_us(),
            collector_name: collector_name.into(),
            value,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

fn now_us() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        #[allow(clippy::cast_possible_truncation)]
        let us = d.as_micros() as u64;
        us
    })
}

// ── ProfileCollector trait ─────────────────────────────────────────────────

/// A pluggable source of profiling samples.
pub trait ProfileCollector: fmt::Debug + Send {
    /// Human-readable collector name.
    fn name(&self) -> &str;
    /// Gather one round of samples.
    fn collect(&mut self) -> Vec<ProfileSample>;
    /// Estimated per-call overhead in nanoseconds.
    fn overhead_ns(&self) -> u64;
}

// ── Built-in collectors ───────────────────────────────────────────────────

/// Collects simulated CPU-time samples.
#[derive(Debug)]
pub struct CpuTimeCollector {
    /// Tracks cumulative simulated CPU microseconds.
    cumulative_us: u64,
}

impl CpuTimeCollector {
    pub fn new() -> Self {
        Self { cumulative_us: 0 }
    }
}

impl Default for CpuTimeCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileCollector for CpuTimeCollector {
    fn name(&self) -> &str {
        "cpu_time"
    }

    fn collect(&mut self) -> Vec<ProfileSample> {
        // In a real implementation this would read /proc/stat or
        // platform counters. Here we simulate a small increment.
        self.cumulative_us += 100;
        vec![ProfileSample::new("cpu_time", self.cumulative_us as f64)]
    }

    fn overhead_ns(&self) -> u64 {
        500 // ~0.5 µs
    }
}

/// Collects simulated memory-usage samples.
#[derive(Debug)]
pub struct MemoryCollector {
    current_bytes: u64,
}

impl MemoryCollector {
    pub fn new() -> Self {
        Self { current_bytes: 0 }
    }

    /// Manually set reported memory usage (useful for testing).
    pub fn set_usage(&mut self, bytes: u64) {
        self.current_bytes = bytes;
    }
}

impl Default for MemoryCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileCollector for MemoryCollector {
    fn name(&self) -> &str {
        "memory"
    }

    fn collect(&mut self) -> Vec<ProfileSample> {
        vec![ProfileSample::new("memory", self.current_bytes as f64).with_metadata("unit", "bytes")]
    }

    fn overhead_ns(&self) -> u64 {
        200
    }
}

/// Collects simulated GPU kernel execution samples.
#[derive(Debug)]
pub struct GpuKernelCollector {
    kernel_launches: u64,
    total_kernel_ns: u64,
}

impl GpuKernelCollector {
    pub fn new() -> Self {
        Self { kernel_launches: 0, total_kernel_ns: 0 }
    }

    /// Record a kernel launch for the next collection round.
    pub fn record_launch(&mut self, duration_ns: u64) {
        self.kernel_launches += 1;
        self.total_kernel_ns += duration_ns;
    }
}

impl Default for GpuKernelCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileCollector for GpuKernelCollector {
    fn name(&self) -> &str {
        "gpu_kernel"
    }

    fn collect(&mut self) -> Vec<ProfileSample> {
        let sample = ProfileSample::new("gpu_kernel", self.total_kernel_ns as f64)
            .with_metadata("launches", self.kernel_launches.to_string());
        // Reset counters after collection.
        self.kernel_launches = 0;
        self.total_kernel_ns = 0;
        vec![sample]
    }

    fn overhead_ns(&self) -> u64 {
        1_000 // ~1 µs
    }
}

/// Collects simulated I/O throughput samples.
#[derive(Debug)]
pub struct IoCollector {
    bytes_read: u64,
    bytes_written: u64,
}

impl IoCollector {
    pub fn new() -> Self {
        Self { bytes_read: 0, bytes_written: 0 }
    }

    pub fn record_read(&mut self, bytes: u64) {
        self.bytes_read += bytes;
    }

    pub fn record_write(&mut self, bytes: u64) {
        self.bytes_written += bytes;
    }
}

impl Default for IoCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileCollector for IoCollector {
    fn name(&self) -> &str {
        "io"
    }

    fn collect(&mut self) -> Vec<ProfileSample> {
        let read =
            ProfileSample::new("io", self.bytes_read as f64).with_metadata("direction", "read");
        let write =
            ProfileSample::new("io", self.bytes_written as f64).with_metadata("direction", "write");
        self.bytes_read = 0;
        self.bytes_written = 0;
        vec![read, write]
    }

    fn overhead_ns(&self) -> u64 {
        300
    }
}

// ── Ring buffer ───────────────────────────────────────────────────────────

/// Fixed-capacity ring buffer of [`ProfileSample`]s.
#[derive(Debug)]
pub struct SampleRingBuffer {
    buf: Vec<Option<ProfileSample>>,
    head: usize,
    len: usize,
}

impl SampleRingBuffer {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self { buf: (0..capacity).map(|_| None).collect(), head: 0, len: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, sample: ProfileSample) {
        self.buf[self.head] = Some(sample);
        self.head = (self.head + 1) % self.buf.len();
        if self.len < self.buf.len() {
            self.len += 1;
        }
    }

    /// Iterate over samples in insertion order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &ProfileSample> {
        let cap = self.buf.len();
        let start = if self.len < cap { 0 } else { self.head };
        (0..self.len).filter_map(move |i| {
            let idx = (start + i) % cap;
            self.buf[idx].as_ref()
        })
    }

    /// Drain all samples, returning them oldest-first.
    pub fn drain(&mut self) -> Vec<ProfileSample> {
        let samples: Vec<_> = self.iter().cloned().collect();
        self.head = 0;
        self.len = 0;
        for slot in &mut self.buf {
            *slot = None;
        }
        samples
    }
}

// ── Flame graph builder ───────────────────────────────────────────────────

/// A single frame in a call stack.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StackFrame {
    pub function_name: String,
    pub module: String,
}

impl fmt::Display for StackFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.module.is_empty() {
            write!(f, "{}", self.function_name)
        } else {
            write!(f, "{}::{}", self.module, self.function_name)
        }
    }
}

/// Accumulated folded-stack entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldedStack {
    /// Semicolon-delimited stack path.
    pub stack: String,
    /// Total sample count.
    pub count: u64,
}

/// Builds flame graphs from profiling samples.
///
/// Samples are expected to carry a `"stack"` metadata key whose value
/// is a semicolon-separated folded stack (e.g. `"main;foo;bar"`).
#[derive(Debug, Default)]
pub struct FlameGraphBuilder {
    folded: HashMap<String, u64>,
}

impl FlameGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a batch of samples.
    pub fn add_samples(&mut self, samples: &[ProfileSample]) {
        for s in samples {
            if let Some(stack) = s.metadata.get("stack") {
                if !stack.is_empty() {
                    *self.folded.entry(stack.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    /// Add a single folded stack string with a count.
    pub fn add_folded(&mut self, stack: &str, count: u64) {
        if !stack.is_empty() {
            *self.folded.entry(stack.to_string()).or_insert(0) += count;
        }
    }

    /// Number of distinct stacks recorded.
    pub fn unique_stacks(&self) -> usize {
        self.folded.len()
    }

    /// Total sample count across all stacks.
    pub fn total_samples(&self) -> u64 {
        self.folded.values().sum()
    }

    /// Produce sorted folded-stack entries.
    pub fn build(&self) -> Vec<FoldedStack> {
        let mut entries: Vec<FoldedStack> = self
            .folded
            .iter()
            .map(|(stack, &count)| FoldedStack { stack: stack.clone(), count })
            .collect();
        entries.sort_by(|a, b| b.count.cmp(&a.count));
        entries
    }

    /// Render as folded-stack text (compatible with `flamegraph.pl`).
    pub fn render_folded(&self) -> String {
        let entries = self.build();
        let mut out = String::new();
        for e in &entries {
            out.push_str(&format!("{} {}\n", e.stack, e.count));
        }
        out
    }

    pub fn clear(&mut self) {
        self.folded.clear();
    }
}

// ── Hotspot detection ─────────────────────────────────────────────────────

/// A detected performance hotspot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hotspot {
    pub function_name: String,
    /// Percentage of total profiling time.
    pub total_time_pct: f64,
    pub call_count: u64,
    /// Average call duration in nanoseconds.
    pub avg_time_ns: f64,
}

/// Identifies the most expensive functions from folded stacks.
#[derive(Debug)]
pub struct HotspotDetector {
    /// Minimum percentage to report as a hotspot.
    threshold_pct: f64,
    /// Maximum number of hotspots to return.
    max_hotspots: usize,
}

impl HotspotDetector {
    pub fn new(threshold_pct: f64, max_hotspots: usize) -> Self {
        Self { threshold_pct: threshold_pct.max(0.0), max_hotspots: max_hotspots.max(1) }
    }

    /// Detect hotspots from a set of [`FoldedStack`] entries.
    ///
    /// Each leaf function (last element of the semicolon-delimited stack)
    /// accumulates the sample count. The result is sorted by
    /// `total_time_pct` descending.
    pub fn detect(&self, stacks: &[FoldedStack]) -> Vec<Hotspot> {
        let total: u64 = stacks.iter().map(|s| s.count).sum();
        if total == 0 {
            return Vec::new();
        }

        let mut leaf_counts: HashMap<String, u64> = HashMap::new();
        for entry in stacks {
            let leaf = entry.stack.rsplit(';').next().unwrap_or(&entry.stack).to_string();
            *leaf_counts.entry(leaf).or_insert(0) += entry.count;
        }

        let mut hotspots: Vec<Hotspot> = leaf_counts
            .into_iter()
            .map(|(name, count)| {
                let pct = (count as f64 / total as f64) * 100.0;
                Hotspot {
                    function_name: name,
                    total_time_pct: pct,
                    call_count: count,
                    avg_time_ns: 0.0, // needs duration info
                }
            })
            .filter(|h| h.total_time_pct >= self.threshold_pct)
            .collect();

        hotspots.sort_by(|a, b| b.total_time_pct.partial_cmp(&a.total_time_pct).unwrap());
        hotspots.truncate(self.max_hotspots);
        hotspots
    }

    /// Detect hotspots from raw samples carrying `"stack"` and
    /// `"duration_ns"` metadata.
    pub fn detect_from_samples(&self, samples: &[ProfileSample]) -> Vec<Hotspot> {
        let mut leaf_times: HashMap<String, (u64, f64)> = HashMap::new();

        for s in samples {
            if let Some(stack) = s.metadata.get("stack") {
                let leaf = stack.rsplit(';').next().unwrap_or(stack).to_string();
                let dur: f64 =
                    s.metadata.get("duration_ns").and_then(|v| v.parse().ok()).unwrap_or(0.0);
                let entry = leaf_times.entry(leaf).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += dur;
            }
        }

        let total_time: f64 = leaf_times.values().map(|(_, t)| *t).sum();
        if total_time == 0.0 {
            return Vec::new();
        }

        let mut hotspots: Vec<Hotspot> = leaf_times
            .into_iter()
            .map(|(name, (count, time))| {
                let pct = (time / total_time) * 100.0;
                Hotspot {
                    function_name: name,
                    total_time_pct: pct,
                    call_count: count,
                    avg_time_ns: if count > 0 { time / count as f64 } else { 0.0 },
                }
            })
            .filter(|h| h.total_time_pct >= self.threshold_pct)
            .collect();

        hotspots.sort_by(|a, b| b.total_time_pct.partial_cmp(&a.total_time_pct).unwrap());
        hotspots.truncate(self.max_hotspots);
        hotspots
    }
}

impl Default for HotspotDetector {
    fn default() -> Self {
        Self::new(1.0, 20)
    }
}

// ── Profile history ───────────────────────────────────────────────────────

/// A snapshot of profiling state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSnapshot {
    pub timestamp_us: u64,
    pub sample_count: usize,
    pub hotspots: Vec<Hotspot>,
    pub summary: HashMap<String, f64>,
}

/// Ring buffer of [`ProfileSnapshot`]s for trend analysis.
#[derive(Debug)]
pub struct ProfileHistory {
    snapshots: Vec<Option<ProfileSnapshot>>,
    head: usize,
    len: usize,
}

impl ProfileHistory {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self { snapshots: (0..capacity).map(|_| None).collect(), head: 0, len: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.snapshots.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, snapshot: ProfileSnapshot) {
        self.snapshots[self.head] = Some(snapshot);
        self.head = (self.head + 1) % self.snapshots.len();
        if self.len < self.snapshots.len() {
            self.len += 1;
        }
    }

    /// Iterate over snapshots oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = &ProfileSnapshot> {
        let cap = self.snapshots.len();
        let start = if self.len < cap { 0 } else { self.head };
        (0..self.len).filter_map(move |i| {
            let idx = (start + i) % cap;
            self.snapshots[idx].as_ref()
        })
    }

    /// Latest snapshot, if any.
    pub fn latest(&self) -> Option<&ProfileSnapshot> {
        if self.len == 0 {
            return None;
        }
        let idx = if self.head == 0 { self.snapshots.len() - 1 } else { self.head - 1 };
        self.snapshots[idx].as_ref()
    }
}

// ── Profile exporters ─────────────────────────────────────────────────────

/// Format-agnostic profile exporter.
pub trait ProfileExporter: fmt::Debug + Send {
    /// Export samples to a byte buffer.
    fn export(&self, samples: &[ProfileSample]) -> Vec<u8>;
    /// File extension hint.
    fn extension(&self) -> &str;
}

/// Exports samples as new-line-delimited JSON.
#[derive(Debug, Default)]
pub struct JsonExporter;

impl JsonExporter {
    pub fn new() -> Self {
        Self
    }
}

impl ProfileExporter for JsonExporter {
    fn export(&self, samples: &[ProfileSample]) -> Vec<u8> {
        let mut out = Vec::new();
        for s in samples {
            if let Ok(line) = serde_json::to_string(s) {
                out.extend_from_slice(line.as_bytes());
                out.push(b'\n');
            }
        }
        out
    }

    fn extension(&self) -> &str {
        "jsonl"
    }
}

/// Exports samples in folded-stack format for flame graph tools.
#[derive(Debug, Default)]
pub struct FoldedStackExporter;

impl FoldedStackExporter {
    pub fn new() -> Self {
        Self
    }
}

impl ProfileExporter for FoldedStackExporter {
    fn export(&self, samples: &[ProfileSample]) -> Vec<u8> {
        let mut builder = FlameGraphBuilder::new();
        builder.add_samples(samples);
        builder.render_folded().into_bytes()
    }

    fn extension(&self) -> &str {
        "folded"
    }
}

// ── Trend analysis ────────────────────────────────────────────────────────

/// Result of trend analysis over a metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendResult {
    pub metric_name: String,
    /// True when the metric appears to be degrading.
    pub degrading: bool,
    /// Slope of linear regression (units per snapshot).
    pub slope: f64,
    /// Percentage change from first to last snapshot.
    pub pct_change: f64,
    pub data_points: usize,
}

/// Detects performance degradation trends from [`ProfileHistory`].
#[derive(Debug)]
pub struct TrendAnalyzer {
    /// Minimum percentage increase to flag as degrading.
    degradation_threshold_pct: f64,
    /// Minimum data points required to analyse.
    min_data_points: usize,
}

impl TrendAnalyzer {
    pub fn new(degradation_threshold_pct: f64, min_data_points: usize) -> Self {
        Self {
            degradation_threshold_pct: degradation_threshold_pct.max(0.0),
            min_data_points: min_data_points.max(2),
        }
    }

    /// Analyse a named metric across history snapshots.
    pub fn analyze(&self, history: &ProfileHistory, metric_name: &str) -> Option<TrendResult> {
        let values: Vec<f64> =
            history.iter().filter_map(|snap| snap.summary.get(metric_name).copied()).collect();
        if values.len() < self.min_data_points {
            return None;
        }
        let slope = linear_regression_slope(&values);
        let first = values[0];
        let last = *values.last().unwrap();
        let pct_change =
            if first.abs() > f64::EPSILON { ((last - first) / first) * 100.0 } else { 0.0 };
        Some(TrendResult {
            metric_name: metric_name.to_string(),
            degrading: pct_change > self.degradation_threshold_pct,
            slope,
            pct_change,
            data_points: values.len(),
        })
    }

    /// Analyse all metrics present in history.
    pub fn analyze_all(&self, history: &ProfileHistory) -> Vec<TrendResult> {
        let mut keys: Vec<String> = Vec::new();
        for snap in history.iter() {
            for k in snap.summary.keys() {
                if !keys.contains(k) {
                    keys.push(k.clone());
                }
            }
        }
        keys.iter().filter_map(|k| self.analyze(history, k)).collect()
    }
}

impl Default for TrendAnalyzer {
    fn default() -> Self {
        Self::new(10.0, 3)
    }
}

/// Simple linear-regression slope over evenly-spaced values.
fn linear_regression_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..values.len()).map(|i| (i as f64) * (i as f64)).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }
    (n * sum_xy - sum_x * sum_y) / denom
}

// ── Overhead estimation ───────────────────────────────────────────────────

/// Estimated profiling overhead for a set of collectors and config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadEstimate {
    /// Total estimated nanoseconds per sample cycle.
    pub per_cycle_ns: u64,
    /// Estimated overhead fraction (0.0–1.0).
    pub fraction: f64,
    /// Whether the overhead is within the configured budget.
    pub within_budget: bool,
}

/// Estimate profiling overhead given collectors' declared costs.
pub fn estimate_overhead(
    config: &ProfilerConfig,
    collector_overheads_ns: &[u64],
) -> OverheadEstimate {
    let per_cycle: u64 = collector_overheads_ns.iter().sum();
    let interval_ns = config.sample_interval().as_nanos() as u64;
    let fraction = if interval_ns > 0 { per_cycle as f64 / interval_ns as f64 } else { 1.0 };
    OverheadEstimate {
        per_cycle_ns: per_cycle,
        fraction,
        within_budget: fraction <= config.max_overhead_fraction,
    }
}

// ── Latency tracker ───────────────────────────────────────────────────────

/// Tracks per-token latency, time-to-first-token, and inter-token latency.
#[derive(Debug)]
pub struct LatencyTracker {
    /// Time-to-first-token samples in nanoseconds.
    ttft_samples_ns: Vec<u64>,
    /// Per-token latencies in nanoseconds.
    token_latencies_ns: Vec<u64>,
    /// Inter-token latencies in nanoseconds.
    inter_token_ns: Vec<u64>,
    /// Timestamp of the last recorded token (for inter-token calc).
    last_token_time: Option<Instant>,
    /// Histogram bucket boundaries (nanoseconds).
    bucket_boundaries_ns: Vec<u64>,
}

impl LatencyTracker {
    /// Create a new tracker with default histogram buckets.
    pub fn new() -> Self {
        Self {
            ttft_samples_ns: Vec::new(),
            token_latencies_ns: Vec::new(),
            inter_token_ns: Vec::new(),
            last_token_time: None,
            bucket_boundaries_ns: vec![
                500_000,     // 0.5 ms
                1_000_000,   // 1 ms
                5_000_000,   // 5 ms
                10_000_000,  // 10 ms
                50_000_000,  // 50 ms
                100_000_000, // 100 ms
                500_000_000, // 500 ms
            ],
        }
    }

    /// Create a tracker with custom histogram bucket boundaries.
    pub fn with_buckets(boundaries_ns: Vec<u64>) -> Self {
        let mut s = Self::new();
        s.bucket_boundaries_ns = boundaries_ns;
        s
    }

    /// Record a time-to-first-token observation.
    pub fn record_ttft(&mut self, ns: u64) {
        self.ttft_samples_ns.push(ns);
    }

    /// Record a per-token latency and compute inter-token gap.
    pub fn record_token(&mut self, latency_ns: u64) {
        self.token_latencies_ns.push(latency_ns);
        let now = Instant::now();
        if let Some(prev) = self.last_token_time {
            let gap = now.duration_since(prev).as_nanos() as u64;
            self.inter_token_ns.push(gap);
        }
        self.last_token_time = Some(now);
    }

    /// Record a token with an explicit inter-token gap (for testing).
    pub fn record_token_with_gap(&mut self, latency_ns: u64, gap_ns: u64) {
        self.token_latencies_ns.push(latency_ns);
        self.inter_token_ns.push(gap_ns);
    }

    /// Average time-to-first-token in nanoseconds, or `None` if empty.
    pub fn avg_ttft_ns(&self) -> Option<f64> {
        if self.ttft_samples_ns.is_empty() {
            return None;
        }
        let sum: u64 = self.ttft_samples_ns.iter().sum();
        Some(sum as f64 / self.ttft_samples_ns.len() as f64)
    }

    /// P50 of per-token latencies in nanoseconds.
    pub fn p50_token_latency_ns(&self) -> Option<u64> {
        percentile(&self.token_latencies_ns, 50)
    }

    /// P99 of per-token latencies in nanoseconds.
    pub fn p99_token_latency_ns(&self) -> Option<u64> {
        percentile(&self.token_latencies_ns, 99)
    }

    /// P50 of inter-token latencies in nanoseconds.
    pub fn p50_inter_token_ns(&self) -> Option<u64> {
        percentile(&self.inter_token_ns, 50)
    }

    /// Compute a histogram over per-token latencies.
    pub fn token_latency_histogram(&self) -> Vec<(u64, usize)> {
        histogram(&self.token_latencies_ns, &self.bucket_boundaries_ns)
    }

    /// Total number of token latency observations.
    pub fn token_count(&self) -> usize {
        self.token_latencies_ns.len()
    }

    /// Total number of TTFT observations.
    pub fn ttft_count(&self) -> usize {
        self.ttft_samples_ns.len()
    }

    /// Reset all tracked data.
    pub fn reset(&mut self) {
        self.ttft_samples_ns.clear();
        self.token_latencies_ns.clear();
        self.inter_token_ns.clear();
        self.last_token_time = None;
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the `pct`-th percentile from a slice of values.
fn percentile(values: &[u64], pct: u32) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((pct as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}

/// Build a histogram: for each boundary, count values ≤ boundary.
fn histogram(values: &[u64], boundaries: &[u64]) -> Vec<(u64, usize)> {
    boundaries.iter().map(|&b| (b, values.iter().filter(|&&v| v <= b).count())).collect()
}

// ── Throughput tracker ────────────────────────────────────────────────────

/// Tracks tokens/sec, requests/sec, and batch throughput.
#[derive(Debug)]
pub struct ThroughputTracker {
    /// Total tokens generated.
    total_tokens: u64,
    /// Total requests completed.
    total_requests: u64,
    /// Per-batch token counts (for batch throughput).
    batch_token_counts: Vec<u64>,
    /// Tracking start time.
    started_at: Option<Instant>,
}

impl ThroughputTracker {
    pub fn new() -> Self {
        Self {
            total_tokens: 0,
            total_requests: 0,
            batch_token_counts: Vec::new(),
            started_at: None,
        }
    }

    /// Start (or restart) the throughput window.
    pub fn start(&mut self) {
        self.started_at = Some(Instant::now());
    }

    /// Record generation of `n` tokens.
    pub fn record_tokens(&mut self, n: u64) {
        self.total_tokens += n;
    }

    /// Record completion of one request.
    pub fn record_request(&mut self) {
        self.total_requests += 1;
    }

    /// Record a batch of `n` tokens.
    pub fn record_batch(&mut self, tokens: u64) {
        self.batch_token_counts.push(tokens);
        self.total_tokens += tokens;
    }

    /// Elapsed time since `start()`.
    pub fn elapsed(&self) -> Duration {
        self.started_at.map_or(Duration::ZERO, |t| t.elapsed())
    }

    /// Tokens per second (returns 0.0 if not started or zero elapsed).
    pub fn tokens_per_sec(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs > 0.0 { self.total_tokens as f64 / secs } else { 0.0 }
    }

    /// Tokens per second using an explicit duration (for testing).
    pub fn tokens_per_sec_with_duration(&self, elapsed: Duration) -> f64 {
        let secs = elapsed.as_secs_f64();
        if secs > 0.0 { self.total_tokens as f64 / secs } else { 0.0 }
    }

    /// Requests per second using an explicit duration.
    pub fn requests_per_sec_with_duration(&self, elapsed: Duration) -> f64 {
        let secs = elapsed.as_secs_f64();
        if secs > 0.0 { self.total_requests as f64 / secs } else { 0.0 }
    }

    /// Average batch size.
    pub fn avg_batch_size(&self) -> f64 {
        if self.batch_token_counts.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.batch_token_counts.iter().sum();
        sum as f64 / self.batch_token_counts.len() as f64
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    pub fn batch_count(&self) -> usize {
        self.batch_token_counts.len()
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.total_tokens = 0;
        self.total_requests = 0;
        self.batch_token_counts.clear();
        self.started_at = None;
    }
}

impl Default for ThroughputTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Prometheus exporter ───────────────────────────────────────────────────

/// Exports samples in Prometheus text exposition format.
#[derive(Debug, Default)]
pub struct PrometheusExporter {
    /// Optional prefix for metric names.
    prefix: String,
}

impl PrometheusExporter {
    pub fn new() -> Self {
        Self { prefix: String::new() }
    }

    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        Self { prefix: prefix.into() }
    }
}

impl ProfileExporter for PrometheusExporter {
    fn export(&self, samples: &[ProfileSample]) -> Vec<u8> {
        let mut out = String::new();
        for s in samples {
            let name = if self.prefix.is_empty() {
                s.collector_name.clone()
            } else {
                format!("{}_{}", self.prefix, s.collector_name)
            };
            let labels: String = if s.metadata.is_empty() {
                String::new()
            } else {
                let pairs: Vec<String> =
                    s.metadata.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
                format!("{{{}}}", pairs.join(","))
            };
            out.push_str(&format!(
                "{name}{labels} {value} {ts}\n",
                value = s.value,
                ts = s.timestamp_us / 1000, // Prometheus uses ms
            ));
        }
        out.into_bytes()
    }

    fn extension(&self) -> &str {
        "prom"
    }
}

// ── Pprof exporter ────────────────────────────────────────────────────────

/// Exports samples in a simplified pprof-compatible text format.
///
/// Real pprof uses protobuf; this produces a human-readable approximation
/// that captures the same logical structure (sample type, stacks, values).
#[derive(Debug, Default)]
pub struct PprofExporter;

impl PprofExporter {
    pub fn new() -> Self {
        Self
    }
}

impl ProfileExporter for PprofExporter {
    fn export(&self, samples: &[ProfileSample]) -> Vec<u8> {
        let mut out = String::new();
        out.push_str("--- pprof-text ---\n");
        out.push_str("sample_type: count/value\n\n");
        for s in samples {
            let stack =
                s.metadata.get("stack").cloned().unwrap_or_else(|| s.collector_name.clone());
            out.push_str(&format!("{stack} {value}\n", value = s.value,));
        }
        out.into_bytes()
    }

    fn extension(&self) -> &str {
        "pprof.txt"
    }
}

// ── Continuous profiler ───────────────────────────────────────────────────

/// Always-on profiler that samples registered collectors at a
/// configurable rate and maintains a ring buffer of recent samples.
#[derive(Debug)]
pub struct ContinuousProfiler {
    config: ProfilerConfig,
    collectors: Vec<Box<dyn ProfileCollector>>,
    buffer: SampleRingBuffer,
    total_samples_collected: u64,
    started_at: Option<Instant>,
    paused: bool,
}

impl ContinuousProfiler {
    pub fn new(config: ProfilerConfig) -> Self {
        let buffer = SampleRingBuffer::new(config.buffer_size);
        Self {
            config,
            collectors: Vec::new(),
            buffer,
            total_samples_collected: 0,
            started_at: None,
            paused: false,
        }
    }

    /// Register a collector.
    pub fn add_collector(&mut self, collector: Box<dyn ProfileCollector>) {
        self.collectors.push(collector);
    }

    /// Start (or restart) the profiler.
    pub fn start(&mut self) {
        self.started_at = Some(Instant::now());
        self.paused = false;
    }

    /// Pause collection without resetting state.
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resume after pause.
    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn is_running(&self) -> bool {
        self.started_at.is_some() && !self.paused
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Perform one collection cycle across all registered collectors.
    ///
    /// Returns the number of samples gathered in this cycle.
    pub fn tick(&mut self) -> usize {
        if !self.is_running() {
            return 0;
        }
        let mut count = 0;
        // Collect samples from each collector into a temp vec to
        // satisfy the borrow checker (collectors borrows &mut self).
        let mut all_samples: Vec<ProfileSample> = Vec::new();
        for c in &mut self.collectors {
            all_samples.extend(c.collect());
        }
        for s in all_samples {
            self.buffer.push(s);
            count += 1;
            self.total_samples_collected += 1;
        }
        count
    }

    /// Total samples collected since creation.
    pub fn total_samples(&self) -> u64 {
        self.total_samples_collected
    }

    /// Number of samples currently in the ring buffer.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    /// Read-only access to the config.
    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }

    /// Drain all buffered samples.
    pub fn drain(&mut self) -> Vec<ProfileSample> {
        self.buffer.drain()
    }

    /// Snapshot current buffer for iteration.
    pub fn snapshot_samples(&self) -> Vec<ProfileSample> {
        self.buffer.iter().cloned().collect()
    }

    /// Estimate current overhead.
    pub fn estimate_overhead(&self) -> OverheadEstimate {
        let overheads: Vec<u64> = self.collectors.iter().map(|c| c.overhead_ns()).collect();
        estimate_overhead(&self.config, &overheads)
    }

    /// Number of registered collectors.
    pub fn collector_count(&self) -> usize {
        self.collectors.len()
    }

    /// Elapsed time since start (or zero if not started).
    pub fn elapsed(&self) -> Duration {
        self.started_at.map_or(Duration::ZERO, |t| t.elapsed())
    }

    /// Dynamically adjust sampling rate (clamped to valid range).
    pub fn set_sampling_rate(&mut self, hz: u32) {
        self.config.sampling_rate_hz = hz.clamp(1, 10_000);
    }

    /// Build a [`ProfileSnapshot`] from the current buffer state.
    pub fn take_snapshot(&mut self, detector: &HotspotDetector) -> ProfileSnapshot {
        let samples = self.snapshot_samples();
        let mut builder = FlameGraphBuilder::new();
        builder.add_samples(&samples);
        let stacks = builder.build();
        let hotspots = detector.detect(&stacks);

        // Summarise per-collector averages.
        let mut summary = HashMap::new();
        let mut counts: HashMap<String, (f64, u64)> = HashMap::new();
        for s in &samples {
            let e = counts.entry(s.collector_name.clone()).or_default();
            e.0 += s.value;
            e.1 += 1;
        }
        for (name, (total, count)) in &counts {
            summary.insert(name.clone(), *total / *count as f64);
        }

        ProfileSnapshot { timestamp_us: now_us(), sample_count: samples.len(), hotspots, summary }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProfilerConfig ────────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = ProfilerConfig::default();
        assert_eq!(cfg.sampling_rate_hz, 100);
        assert_eq!(cfg.buffer_size, 65_536);
        assert_eq!(cfg.export_interval_secs, 60);
        assert_eq!(cfg.enabled_collectors.len(), 2);
    }

    #[test]
    fn config_effective_rate_clamping() {
        let mut cfg = ProfilerConfig::default();
        cfg.sampling_rate_hz = 0;
        assert_eq!(cfg.effective_rate(), 1);
        cfg.sampling_rate_hz = 999_999;
        assert_eq!(cfg.effective_rate(), 10_000);
        cfg.sampling_rate_hz = 500;
        assert_eq!(cfg.effective_rate(), 500);
    }

    #[test]
    fn config_sample_interval() {
        let mut cfg = ProfilerConfig::default();
        cfg.sampling_rate_hz = 1000;
        assert_eq!(cfg.sample_interval(), Duration::from_millis(1));
    }

    #[test]
    fn config_is_collector_enabled() {
        let cfg = ProfilerConfig::default();
        assert!(cfg.is_collector_enabled(&CollectorKind::CpuTime));
        assert!(cfg.is_collector_enabled(&CollectorKind::Memory));
        assert!(!cfg.is_collector_enabled(&CollectorKind::GpuKernel));
    }

    // ── ProfileSample ─────────────────────────────────────────────────

    #[test]
    fn sample_creation() {
        let s = ProfileSample::new("test", 42.0);
        assert_eq!(s.collector_name, "test");
        assert!((s.value - 42.0).abs() < f64::EPSILON);
        assert!(s.timestamp_us > 0);
    }

    #[test]
    fn sample_with_metadata() {
        let s = ProfileSample::new("test", 1.0).with_metadata("key", "val");
        assert_eq!(s.metadata.get("key").unwrap(), "val");
    }

    #[test]
    fn sample_multiple_metadata() {
        let s = ProfileSample::new("x", 0.0).with_metadata("a", "1").with_metadata("b", "2");
        assert_eq!(s.metadata.len(), 2);
    }

    // ── CpuTimeCollector ──────────────────────────────────────────────

    #[test]
    fn cpu_collector_name() {
        let c = CpuTimeCollector::new();
        assert_eq!(c.name(), "cpu_time");
    }

    #[test]
    fn cpu_collector_accumulates() {
        let mut c = CpuTimeCollector::new();
        let s1 = c.collect();
        let s2 = c.collect();
        assert!(s2[0].value > s1[0].value);
    }

    #[test]
    fn cpu_collector_overhead() {
        let c = CpuTimeCollector::new();
        assert!(c.overhead_ns() < 10_000);
    }

    // ── MemoryCollector ───────────────────────────────────────────────

    #[test]
    fn memory_collector_default_zero() {
        let mut c = MemoryCollector::new();
        let s = c.collect();
        assert!((s[0].value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn memory_collector_set_usage() {
        let mut c = MemoryCollector::new();
        c.set_usage(1024);
        let s = c.collect();
        assert!((s[0].value - 1024.0).abs() < f64::EPSILON);
    }

    #[test]
    fn memory_collector_metadata_unit() {
        let mut c = MemoryCollector::new();
        let s = c.collect();
        assert_eq!(s[0].metadata.get("unit").unwrap(), "bytes");
    }

    // ── GpuKernelCollector ────────────────────────────────────────────

    #[test]
    fn gpu_kernel_collector_empty() {
        let mut c = GpuKernelCollector::new();
        let s = c.collect();
        assert!((s[0].value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gpu_kernel_collector_records() {
        let mut c = GpuKernelCollector::new();
        c.record_launch(1000);
        c.record_launch(2000);
        let s = c.collect();
        assert!((s[0].value - 3000.0).abs() < f64::EPSILON);
        assert_eq!(s[0].metadata.get("launches").unwrap(), "2");
    }

    #[test]
    fn gpu_kernel_collector_resets_after_collect() {
        let mut c = GpuKernelCollector::new();
        c.record_launch(500);
        let _ = c.collect();
        let s = c.collect();
        assert!((s[0].value - 0.0).abs() < f64::EPSILON);
    }

    // ── IoCollector ───────────────────────────────────────────────────

    #[test]
    fn io_collector_read_write() {
        let mut c = IoCollector::new();
        c.record_read(100);
        c.record_write(200);
        let samples = c.collect();
        assert_eq!(samples.len(), 2);
        assert!((samples[0].value - 100.0).abs() < f64::EPSILON);
        assert!((samples[1].value - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn io_collector_resets_after_collect() {
        let mut c = IoCollector::new();
        c.record_read(50);
        let _ = c.collect();
        let s = c.collect();
        assert!((s[0].value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn io_collector_metadata_direction() {
        let mut c = IoCollector::new();
        let s = c.collect();
        assert_eq!(s[0].metadata.get("direction").unwrap(), "read");
        assert_eq!(s[1].metadata.get("direction").unwrap(), "write");
    }

    // ── SampleRingBuffer ──────────────────────────────────────────────

    #[test]
    fn ring_buffer_empty() {
        let buf = SampleRingBuffer::new(8);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 8);
    }

    #[test]
    fn ring_buffer_push_and_len() {
        let mut buf = SampleRingBuffer::new(4);
        buf.push(ProfileSample::new("a", 1.0));
        buf.push(ProfileSample::new("b", 2.0));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn ring_buffer_wraps() {
        let mut buf = SampleRingBuffer::new(3);
        for i in 0..5 {
            buf.push(ProfileSample::new("x", i as f64));
        }
        assert_eq!(buf.len(), 3);
        let values: Vec<f64> = buf.iter().map(|s| s.value).collect();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn ring_buffer_drain() {
        let mut buf = SampleRingBuffer::new(4);
        buf.push(ProfileSample::new("a", 1.0));
        buf.push(ProfileSample::new("b", 2.0));
        let drained = buf.drain();
        assert_eq!(drained.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn ring_buffer_min_capacity() {
        let buf = SampleRingBuffer::new(0);
        assert_eq!(buf.capacity(), 1);
    }

    #[test]
    fn ring_buffer_iter_partial_fill() {
        let mut buf = SampleRingBuffer::new(8);
        buf.push(ProfileSample::new("a", 10.0));
        let vals: Vec<f64> = buf.iter().map(|s| s.value).collect();
        assert_eq!(vals, vec![10.0]);
    }

    // ── FlameGraphBuilder ─────────────────────────────────────────────

    #[test]
    fn flamegraph_empty() {
        let builder = FlameGraphBuilder::new();
        assert_eq!(builder.unique_stacks(), 0);
        assert_eq!(builder.total_samples(), 0);
        assert!(builder.build().is_empty());
    }

    #[test]
    fn flamegraph_add_folded() {
        let mut builder = FlameGraphBuilder::new();
        builder.add_folded("main;foo;bar", 5);
        builder.add_folded("main;foo;baz", 3);
        assert_eq!(builder.unique_stacks(), 2);
        assert_eq!(builder.total_samples(), 8);
    }

    #[test]
    fn flamegraph_add_samples_with_stack() {
        let mut builder = FlameGraphBuilder::new();
        let s = ProfileSample::new("cpu", 1.0).with_metadata("stack", "main;compute");
        builder.add_samples(&[s.clone(), s]);
        assert_eq!(builder.total_samples(), 2);
    }

    #[test]
    fn flamegraph_ignores_empty_stack() {
        let mut builder = FlameGraphBuilder::new();
        builder.add_folded("", 10);
        assert_eq!(builder.unique_stacks(), 0);
    }

    #[test]
    fn flamegraph_ignores_missing_stack_key() {
        let mut builder = FlameGraphBuilder::new();
        let s = ProfileSample::new("cpu", 1.0);
        builder.add_samples(&[s]);
        assert_eq!(builder.unique_stacks(), 0);
    }

    #[test]
    fn flamegraph_build_sorted_descending() {
        let mut builder = FlameGraphBuilder::new();
        builder.add_folded("a;b", 1);
        builder.add_folded("a;c", 5);
        builder.add_folded("a;d", 3);
        let entries = builder.build();
        assert_eq!(entries[0].count, 5);
        assert_eq!(entries[1].count, 3);
        assert_eq!(entries[2].count, 1);
    }

    #[test]
    fn flamegraph_render_folded() {
        let mut builder = FlameGraphBuilder::new();
        builder.add_folded("main;foo", 2);
        let text = builder.render_folded();
        assert!(text.contains("main;foo 2"));
    }

    #[test]
    fn flamegraph_clear() {
        let mut builder = FlameGraphBuilder::new();
        builder.add_folded("a;b", 5);
        builder.clear();
        assert_eq!(builder.unique_stacks(), 0);
    }

    #[test]
    fn flamegraph_accumulates_same_stack() {
        let mut builder = FlameGraphBuilder::new();
        builder.add_folded("a;b", 3);
        builder.add_folded("a;b", 7);
        assert_eq!(builder.unique_stacks(), 1);
        assert_eq!(builder.total_samples(), 10);
    }

    // ── HotspotDetector ───────────────────────────────────────────────

    #[test]
    fn hotspot_empty_input() {
        let d = HotspotDetector::default();
        let result = d.detect(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn hotspot_detects_leaf_functions() {
        let d = HotspotDetector::new(0.0, 10);
        let stacks = vec![
            FoldedStack { stack: "main;compute".into(), count: 8 },
            FoldedStack { stack: "main;io".into(), count: 2 },
        ];
        let hs = d.detect(&stacks);
        assert_eq!(hs.len(), 2);
        assert_eq!(hs[0].function_name, "compute");
    }

    #[test]
    fn hotspot_threshold_filters() {
        let d = HotspotDetector::new(50.0, 10);
        let stacks = vec![
            FoldedStack { stack: "main;big".into(), count: 90 },
            FoldedStack { stack: "main;small".into(), count: 10 },
        ];
        let hs = d.detect(&stacks);
        assert_eq!(hs.len(), 1);
        assert_eq!(hs[0].function_name, "big");
    }

    #[test]
    fn hotspot_max_limit() {
        let d = HotspotDetector::new(0.0, 2);
        let stacks = vec![
            FoldedStack { stack: "a".into(), count: 5 },
            FoldedStack { stack: "b".into(), count: 3 },
            FoldedStack { stack: "c".into(), count: 1 },
        ];
        let hs = d.detect(&stacks);
        assert_eq!(hs.len(), 2);
    }

    #[test]
    fn hotspot_percentage_correct() {
        let d = HotspotDetector::new(0.0, 10);
        let stacks = vec![
            FoldedStack { stack: "a".into(), count: 50 },
            FoldedStack { stack: "b".into(), count: 50 },
        ];
        let hs = d.detect(&stacks);
        assert!((hs[0].total_time_pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn hotspot_from_samples_with_duration() {
        let d = HotspotDetector::new(0.0, 10);
        let samples = vec![
            ProfileSample::new("cpu", 1.0)
                .with_metadata("stack", "main;foo")
                .with_metadata("duration_ns", "1000"),
            ProfileSample::new("cpu", 1.0)
                .with_metadata("stack", "main;bar")
                .with_metadata("duration_ns", "3000"),
        ];
        let hs = d.detect_from_samples(&samples);
        assert_eq!(hs[0].function_name, "bar");
    }

    #[test]
    fn hotspot_from_samples_avg_time() {
        let d = HotspotDetector::new(0.0, 10);
        let samples = vec![
            ProfileSample::new("cpu", 1.0)
                .with_metadata("stack", "main;foo")
                .with_metadata("duration_ns", "1000"),
            ProfileSample::new("cpu", 1.0)
                .with_metadata("stack", "main;foo")
                .with_metadata("duration_ns", "3000"),
        ];
        let hs = d.detect_from_samples(&samples);
        assert!((hs[0].avg_time_ns - 2000.0).abs() < 0.01);
    }

    #[test]
    fn hotspot_from_samples_empty() {
        let d = HotspotDetector::default();
        assert!(d.detect_from_samples(&[]).is_empty());
    }

    // ── ProfileHistory ────────────────────────────────────────────────

    #[test]
    fn history_empty() {
        let h = ProfileHistory::new(5);
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert!(h.latest().is_none());
    }

    #[test]
    fn history_push_and_latest() {
        let mut h = ProfileHistory::new(5);
        h.push(ProfileSnapshot {
            timestamp_us: 1,
            sample_count: 10,
            hotspots: vec![],
            summary: HashMap::new(),
        });
        assert_eq!(h.len(), 1);
        assert_eq!(h.latest().unwrap().timestamp_us, 1);
    }

    #[test]
    fn history_wraps() {
        let mut h = ProfileHistory::new(3);
        for i in 0..5 {
            h.push(ProfileSnapshot {
                timestamp_us: i,
                sample_count: 0,
                hotspots: vec![],
                summary: HashMap::new(),
            });
        }
        assert_eq!(h.len(), 3);
        let ts: Vec<u64> = h.iter().map(|s| s.timestamp_us).collect();
        assert_eq!(ts, vec![2, 3, 4]);
    }

    #[test]
    fn history_latest_after_wrap() {
        let mut h = ProfileHistory::new(2);
        h.push(ProfileSnapshot {
            timestamp_us: 10,
            sample_count: 0,
            hotspots: vec![],
            summary: HashMap::new(),
        });
        h.push(ProfileSnapshot {
            timestamp_us: 20,
            sample_count: 0,
            hotspots: vec![],
            summary: HashMap::new(),
        });
        h.push(ProfileSnapshot {
            timestamp_us: 30,
            sample_count: 0,
            hotspots: vec![],
            summary: HashMap::new(),
        });
        assert_eq!(h.latest().unwrap().timestamp_us, 30);
    }

    // ── TrendAnalyzer ─────────────────────────────────────────────────

    fn make_history_with_metric(values: &[f64], metric: &str) -> ProfileHistory {
        let mut h = ProfileHistory::new(values.len());
        for (i, &v) in values.iter().enumerate() {
            let mut summary = HashMap::new();
            summary.insert(metric.to_string(), v);
            h.push(ProfileSnapshot {
                timestamp_us: i as u64,
                sample_count: 0,
                hotspots: vec![],
                summary,
            });
        }
        h
    }

    #[test]
    fn trend_not_enough_data() {
        let analyzer = TrendAnalyzer::new(10.0, 3);
        let h = make_history_with_metric(&[1.0], "cpu");
        assert!(analyzer.analyze(&h, "cpu").is_none());
    }

    #[test]
    fn trend_stable() {
        let analyzer = TrendAnalyzer::new(10.0, 3);
        let h = make_history_with_metric(&[100.0, 100.0, 100.0], "cpu");
        let r = analyzer.analyze(&h, "cpu").unwrap();
        assert!(!r.degrading);
        assert!(r.slope.abs() < 0.01);
    }

    #[test]
    fn trend_degrading() {
        let analyzer = TrendAnalyzer::new(10.0, 3);
        let h = make_history_with_metric(&[100.0, 120.0, 150.0], "cpu");
        let r = analyzer.analyze(&h, "cpu").unwrap();
        assert!(r.degrading);
        assert!(r.pct_change > 10.0);
    }

    #[test]
    fn trend_improving() {
        let analyzer = TrendAnalyzer::new(10.0, 3);
        let h = make_history_with_metric(&[150.0, 120.0, 100.0], "cpu");
        let r = analyzer.analyze(&h, "cpu").unwrap();
        assert!(!r.degrading);
        assert!(r.pct_change < 0.0);
    }

    #[test]
    fn trend_missing_metric() {
        let analyzer = TrendAnalyzer::new(10.0, 3);
        let h = make_history_with_metric(&[1.0, 2.0, 3.0], "cpu");
        assert!(analyzer.analyze(&h, "gpu").is_none());
    }

    #[test]
    fn trend_analyze_all() {
        let analyzer = TrendAnalyzer::new(0.0, 2);
        let mut h = ProfileHistory::new(3);
        for i in 0..3 {
            let mut summary = HashMap::new();
            summary.insert("cpu".into(), 100.0 + i as f64 * 10.0);
            summary.insert("mem".into(), 200.0);
            h.push(ProfileSnapshot { timestamp_us: i, sample_count: 0, hotspots: vec![], summary });
        }
        let results = analyzer.analyze_all(&h);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn trend_pct_change_calculation() {
        let analyzer = TrendAnalyzer::new(0.0, 2);
        let h = make_history_with_metric(&[100.0, 200.0], "m");
        let r = analyzer.analyze(&h, "m").unwrap();
        assert!((r.pct_change - 100.0).abs() < 0.01);
    }

    #[test]
    fn trend_slope_positive() {
        let analyzer = TrendAnalyzer::new(0.0, 2);
        let h = make_history_with_metric(&[1.0, 2.0, 3.0], "m");
        let r = analyzer.analyze(&h, "m").unwrap();
        assert!(r.slope > 0.0);
    }

    // ── Overhead estimation ───────────────────────────────────────────

    #[test]
    fn overhead_within_budget() {
        let cfg = ProfilerConfig {
            sampling_rate_hz: 100,
            max_overhead_fraction: 0.01,
            ..ProfilerConfig::default()
        };
        // 100 Hz → 10 ms interval = 10_000_000 ns; 1000 ns total
        let est = estimate_overhead(&cfg, &[500, 300, 200]);
        assert!(est.within_budget);
        assert!(est.fraction < 0.01);
    }

    #[test]
    fn overhead_exceeds_budget() {
        let cfg = ProfilerConfig {
            sampling_rate_hz: 10_000,
            max_overhead_fraction: 0.0001,
            ..ProfilerConfig::default()
        };
        // 10 kHz → 100 µs interval; 50 µs overhead → 50%
        let est = estimate_overhead(&cfg, &[50_000]);
        assert!(!est.within_budget);
    }

    #[test]
    fn overhead_per_cycle_sum() {
        let cfg = ProfilerConfig::default();
        let est = estimate_overhead(&cfg, &[100, 200, 300]);
        assert_eq!(est.per_cycle_ns, 600);
    }

    // ── Exporters ─────────────────────────────────────────────────────

    #[test]
    fn json_exporter_produces_jsonl() {
        let exp = JsonExporter::new();
        let samples = vec![ProfileSample::new("cpu", 42.0)];
        let bytes = exp.export(&samples);
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("\"collector_name\":\"cpu\""));
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn json_exporter_extension() {
        assert_eq!(JsonExporter::new().extension(), "jsonl");
    }

    #[test]
    fn folded_exporter_produces_folded() {
        let exp = FoldedStackExporter::new();
        let s = ProfileSample::new("cpu", 1.0).with_metadata("stack", "main;foo;bar");
        let bytes = exp.export(&[s.clone(), s]);
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("main;foo;bar 2"));
    }

    #[test]
    fn folded_exporter_extension() {
        assert_eq!(FoldedStackExporter::new().extension(), "folded");
    }

    // ── ContinuousProfiler integration ────────────────────────────────

    #[test]
    fn profiler_creation() {
        let p = ContinuousProfiler::new(ProfilerConfig::default());
        assert!(!p.is_running());
        assert_eq!(p.total_samples(), 0);
        assert_eq!(p.collector_count(), 0);
    }

    #[test]
    fn profiler_start_stop() {
        let mut p = ContinuousProfiler::new(ProfilerConfig::default());
        p.start();
        assert!(p.is_running());
        p.pause();
        assert!(!p.is_running());
        assert!(p.is_paused());
        p.resume();
        assert!(p.is_running());
    }

    #[test]
    fn profiler_tick_not_running() {
        let mut p = ContinuousProfiler::new(ProfilerConfig::default());
        assert_eq!(p.tick(), 0);
    }

    #[test]
    fn profiler_tick_collects() {
        let mut p = ContinuousProfiler::new(ProfilerConfig::default());
        p.add_collector(Box::new(CpuTimeCollector::new()));
        p.start();
        let n = p.tick();
        assert!(n > 0);
        assert_eq!(p.total_samples(), n as u64);
    }

    #[test]
    fn profiler_multiple_collectors() {
        let mut p = ContinuousProfiler::new(ProfilerConfig::default());
        p.add_collector(Box::new(CpuTimeCollector::new()));
        p.add_collector(Box::new(MemoryCollector::new()));
        p.start();
        let n = p.tick();
        assert_eq!(n, 2); // one sample per collector
    }

    #[test]
    fn profiler_drain() {
        let mut p = ContinuousProfiler::new(ProfilerConfig::default());
        p.add_collector(Box::new(CpuTimeCollector::new()));
        p.start();
        p.tick();
        let drained = p.drain();
        assert!(!drained.is_empty());
        assert_eq!(p.buffered_samples(), 0);
    }

    #[test]
    fn profiler_snapshot_samples() {
        let mut p = ContinuousProfiler::new(ProfilerConfig::default());
        p.add_collector(Box::new(CpuTimeCollector::new()));
        p.start();
        p.tick();
        let snap = p.snapshot_samples();
        assert!(!snap.is_empty());
        // Buffer not drained.
        assert_eq!(p.buffered_samples(), snap.len());
    }

    #[test]
    fn profiler_overhead_estimate() {
        let mut p = ContinuousProfiler::new(ProfilerConfig::default());
        p.add_collector(Box::new(CpuTimeCollector::new()));
        let est = p.estimate_overhead();
        assert!(est.within_budget);
    }

    #[test]
    fn profiler_set_sampling_rate() {
        let mut p = ContinuousProfiler::new(ProfilerConfig::default());
        p.set_sampling_rate(500);
        assert_eq!(p.config().effective_rate(), 500);
    }

    #[test]
    fn profiler_elapsed_zero_before_start() {
        let p = ContinuousProfiler::new(ProfilerConfig::default());
        assert_eq!(p.elapsed(), Duration::ZERO);
    }

    #[test]
    fn profiler_take_snapshot() {
        let mut p = ContinuousProfiler::new(ProfilerConfig::default());
        p.add_collector(Box::new(CpuTimeCollector::new()));
        p.start();
        p.tick();
        let snap = p.take_snapshot(&HotspotDetector::default());
        assert!(snap.sample_count > 0);
    }

    // ── StackFrame display ────────────────────────────────────────────

    #[test]
    fn stack_frame_display_with_module() {
        let f = StackFrame { function_name: "foo".into(), module: "bar".into() };
        assert_eq!(f.to_string(), "bar::foo");
    }

    #[test]
    fn stack_frame_display_no_module() {
        let f = StackFrame { function_name: "foo".into(), module: String::new() };
        assert_eq!(f.to_string(), "foo");
    }

    // ── linear_regression_slope ───────────────────────────────────────

    #[test]
    fn regression_slope_flat() {
        assert!(linear_regression_slope(&[5.0, 5.0, 5.0]).abs() < 0.01);
    }

    #[test]
    fn regression_slope_increasing() {
        let slope = linear_regression_slope(&[1.0, 2.0, 3.0, 4.0]);
        assert!((slope - 1.0).abs() < 0.01);
    }

    #[test]
    fn regression_slope_single_point() {
        assert!((linear_regression_slope(&[42.0]) - 0.0).abs() < f64::EPSILON);
    }

    // ── Collector trait object ─────────────────────────────────────────

    #[test]
    fn collector_trait_object() {
        let mut collectors: Vec<Box<dyn ProfileCollector>> = vec![
            Box::new(CpuTimeCollector::new()),
            Box::new(MemoryCollector::new()),
            Box::new(GpuKernelCollector::new()),
            Box::new(IoCollector::new()),
        ];
        let names: Vec<&str> = collectors.iter().map(|c| c.name()).collect();
        assert_eq!(names, vec!["cpu_time", "memory", "gpu_kernel", "io"]);
        for c in &mut collectors {
            assert!(!c.collect().is_empty());
        }
    }

    // ── Exporter trait object ─────────────────────────────────────────

    #[test]
    fn exporter_trait_object() {
        let exporters: Vec<Box<dyn ProfileExporter>> =
            vec![Box::new(JsonExporter::new()), Box::new(FoldedStackExporter::new())];
        assert_eq!(exporters[0].extension(), "jsonl");
        assert_eq!(exporters[1].extension(), "folded");
    }

    // ── LatencyTracker ────────────────────────────────────────────────

    #[test]
    fn latency_tracker_empty() {
        let t = LatencyTracker::new();
        assert!(t.avg_ttft_ns().is_none());
        assert!(t.p50_token_latency_ns().is_none());
        assert!(t.p99_token_latency_ns().is_none());
        assert!(t.p50_inter_token_ns().is_none());
        assert_eq!(t.token_count(), 0);
        assert_eq!(t.ttft_count(), 0);
    }

    #[test]
    fn latency_tracker_ttft() {
        let mut t = LatencyTracker::new();
        t.record_ttft(1_000_000);
        t.record_ttft(3_000_000);
        assert_eq!(t.ttft_count(), 2);
        let avg = t.avg_ttft_ns().unwrap();
        assert!((avg - 2_000_000.0).abs() < 0.01);
    }

    #[test]
    fn latency_tracker_token_latency() {
        let mut t = LatencyTracker::new();
        for ns in [1000, 2000, 3000, 4000, 5000] {
            t.record_token_with_gap(ns, 100);
        }
        assert_eq!(t.token_count(), 5);
        assert_eq!(t.p50_token_latency_ns(), Some(3000));
    }

    #[test]
    fn latency_tracker_p99() {
        let mut t = LatencyTracker::new();
        for i in 1..=100 {
            t.record_token_with_gap(i * 100, 10);
        }
        let p99 = t.p99_token_latency_ns().unwrap();
        assert!(p99 >= 9900);
    }

    #[test]
    fn latency_tracker_inter_token() {
        let mut t = LatencyTracker::new();
        t.record_token_with_gap(100, 500);
        t.record_token_with_gap(200, 1000);
        t.record_token_with_gap(300, 1500);
        let p50 = t.p50_inter_token_ns().unwrap();
        assert_eq!(p50, 1000);
    }

    #[test]
    fn latency_tracker_histogram() {
        let mut t = LatencyTracker::with_buckets(vec![100, 500, 1000]);
        t.record_token_with_gap(50, 10);
        t.record_token_with_gap(200, 10);
        t.record_token_with_gap(800, 10);
        t.record_token_with_gap(1500, 10);
        let hist = t.token_latency_histogram();
        assert_eq!(hist.len(), 3);
        assert_eq!(hist[0], (100, 1)); // ≤100: 50
        assert_eq!(hist[1], (500, 2)); // ≤500: 50,200
        assert_eq!(hist[2], (1000, 3)); // ≤1000: 50,200,800
    }

    #[test]
    fn latency_tracker_reset() {
        let mut t = LatencyTracker::new();
        t.record_ttft(1000);
        t.record_token_with_gap(500, 100);
        t.reset();
        assert_eq!(t.token_count(), 0);
        assert_eq!(t.ttft_count(), 0);
        assert!(t.p50_inter_token_ns().is_none());
    }

    #[test]
    fn latency_tracker_custom_buckets() {
        let t = LatencyTracker::with_buckets(vec![10, 20, 30]);
        let hist = t.token_latency_histogram();
        assert_eq!(hist.len(), 3);
        // All counts zero with no data.
        assert!(hist.iter().all(|&(_, c)| c == 0));
    }

    #[test]
    fn latency_tracker_single_sample_percentile() {
        let mut t = LatencyTracker::new();
        t.record_token_with_gap(42, 10);
        assert_eq!(t.p50_token_latency_ns(), Some(42));
        assert_eq!(t.p99_token_latency_ns(), Some(42));
    }

    // ── ThroughputTracker ─────────────────────────────────────────────

    #[test]
    fn throughput_tracker_empty() {
        let t = ThroughputTracker::new();
        assert_eq!(t.total_tokens(), 0);
        assert_eq!(t.total_requests(), 0);
        assert_eq!(t.batch_count(), 0);
        assert!((t.avg_batch_size() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn throughput_tracker_tokens() {
        let mut t = ThroughputTracker::new();
        t.record_tokens(10);
        t.record_tokens(20);
        assert_eq!(t.total_tokens(), 30);
    }

    #[test]
    fn throughput_tracker_requests() {
        let mut t = ThroughputTracker::new();
        t.record_request();
        t.record_request();
        t.record_request();
        assert_eq!(t.total_requests(), 3);
    }

    #[test]
    fn throughput_tracker_batch() {
        let mut t = ThroughputTracker::new();
        t.record_batch(10);
        t.record_batch(20);
        t.record_batch(30);
        assert_eq!(t.batch_count(), 3);
        assert!((t.avg_batch_size() - 20.0).abs() < f64::EPSILON);
        assert_eq!(t.total_tokens(), 60);
    }

    #[test]
    fn throughput_tracker_tokens_per_sec() {
        let mut t = ThroughputTracker::new();
        t.record_tokens(100);
        let tps = t.tokens_per_sec_with_duration(Duration::from_secs(2));
        assert!((tps - 50.0).abs() < 0.01);
    }

    #[test]
    fn throughput_tracker_requests_per_sec() {
        let mut t = ThroughputTracker::new();
        for _ in 0..10 {
            t.record_request();
        }
        let rps = t.requests_per_sec_with_duration(Duration::from_secs(5));
        assert!((rps - 2.0).abs() < 0.01);
    }

    #[test]
    fn throughput_tracker_zero_duration() {
        let t = ThroughputTracker::new();
        assert!((t.tokens_per_sec_with_duration(Duration::ZERO)).abs() < f64::EPSILON);
        assert!((t.requests_per_sec_with_duration(Duration::ZERO)).abs() < f64::EPSILON);
    }

    #[test]
    fn throughput_tracker_reset() {
        let mut t = ThroughputTracker::new();
        t.record_tokens(50);
        t.record_request();
        t.record_batch(10);
        t.reset();
        assert_eq!(t.total_tokens(), 0);
        assert_eq!(t.total_requests(), 0);
        assert_eq!(t.batch_count(), 0);
    }

    #[test]
    fn throughput_tracker_elapsed_not_started() {
        let t = ThroughputTracker::new();
        assert_eq!(t.elapsed(), Duration::ZERO);
    }

    #[test]
    fn throughput_tracker_elapsed_after_start() {
        let mut t = ThroughputTracker::new();
        t.start();
        // Just verify it's non-panicking and ≥ ZERO.
        assert!(t.elapsed() >= Duration::ZERO);
    }

    // ── PrometheusExporter ────────────────────────────────────────────

    #[test]
    fn prometheus_exporter_basic() {
        let exp = PrometheusExporter::new();
        let s = ProfileSample::new("cpu_time", 42.0);
        let out = exp.export(&[s]);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("cpu_time"));
        assert!(text.contains("42"));
    }

    #[test]
    fn prometheus_exporter_with_prefix() {
        let exp = PrometheusExporter::with_prefix("bitnet");
        let s = ProfileSample::new("cpu_time", 1.0);
        let out = exp.export(&[s]);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("bitnet_cpu_time"));
    }

    #[test]
    fn prometheus_exporter_with_labels() {
        let exp = PrometheusExporter::new();
        let s = ProfileSample::new("mem", 1024.0).with_metadata("unit", "bytes");
        let out = exp.export(&[s]);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("unit=\"bytes\""));
    }

    #[test]
    fn prometheus_exporter_extension() {
        let exp = PrometheusExporter::new();
        assert_eq!(exp.extension(), "prom");
    }

    #[test]
    fn prometheus_exporter_empty() {
        let exp = PrometheusExporter::new();
        let out = exp.export(&[]);
        assert!(out.is_empty());
    }

    // ── PprofExporter ─────────────────────────────────────────────────

    #[test]
    fn pprof_exporter_basic() {
        let exp = PprofExporter::new();
        let s = ProfileSample::new("cpu", 99.0).with_metadata("stack", "main;foo;bar");
        let out = exp.export(&[s]);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("--- pprof-text ---"));
        assert!(text.contains("main;foo;bar 99"));
    }

    #[test]
    fn pprof_exporter_falls_back_to_collector_name() {
        let exp = PprofExporter::new();
        let s = ProfileSample::new("gpu_kernel", 7.5);
        let out = exp.export(&[s]);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("gpu_kernel 7.5"));
    }

    #[test]
    fn pprof_exporter_extension() {
        let exp = PprofExporter::new();
        assert_eq!(exp.extension(), "pprof.txt");
    }

    #[test]
    fn pprof_exporter_empty() {
        let exp = PprofExporter::new();
        let out = exp.export(&[]);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("--- pprof-text ---"));
    }

    // ── All exporters as trait objects ─────────────────────────────────

    #[test]
    fn all_exporters_trait_object() {
        let exporters: Vec<Box<dyn ProfileExporter>> = vec![
            Box::new(JsonExporter::new()),
            Box::new(FoldedStackExporter::new()),
            Box::new(PrometheusExporter::new()),
            Box::new(PprofExporter::new()),
        ];
        let extensions: Vec<&str> = exporters.iter().map(|e| e.extension()).collect();
        assert_eq!(extensions, vec!["jsonl", "folded", "prom", "pprof.txt"]);
    }

    // ── Percentile / histogram helpers ────────────────────────────────

    #[test]
    fn percentile_empty() {
        assert!(percentile(&[], 50).is_none());
    }

    #[test]
    fn percentile_single_value() {
        assert_eq!(percentile(&[42], 50), Some(42));
        assert_eq!(percentile(&[42], 99), Some(42));
    }

    #[test]
    fn histogram_empty_values() {
        let h = histogram(&[], &[100, 200]);
        assert_eq!(h, vec![(100, 0), (200, 0)]);
    }

    #[test]
    fn histogram_all_below() {
        let h = histogram(&[1, 2, 3], &[10]);
        assert_eq!(h, vec![(10, 3)]);
    }
}
