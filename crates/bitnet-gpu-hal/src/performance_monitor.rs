//! Performance monitoring for GPU/CPU inference pipelines.
//!
//! Provides throughput tracking, latency histograms, memory monitoring,
//! utilisation metrics, bottleneck detection, alerting, and multi-format
//! export (Prometheus, JSON, CSV).
//!
//! All types are `Send + Sync`-safe by design (no interior mutability tricks)
//! and work on CPU-only builds as a reference baseline.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ── MonitorConfig ────────────────────────────────────────────────────────────

/// Configuration for the performance monitoring engine.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// How often metrics are sampled (advisory).
    pub sample_interval: Duration,
    /// Maximum number of data-points kept in rolling windows.
    pub history_size: usize,
    /// Throughput threshold (tokens/sec) below which an alert fires.
    pub throughput_alert_threshold: f64,
    /// Latency threshold (ms) above which an alert fires (p99).
    pub latency_alert_threshold_ms: f64,
    /// Memory usage percentage above which an alert fires.
    pub memory_alert_threshold_pct: f64,
    /// GPU utilisation percentage below which an alert fires.
    pub utilization_alert_threshold_pct: f64,
    /// Whether alerting is enabled.
    pub alerts_enabled: bool,
    /// Label applied to all exported metrics.
    pub instance_label: String,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_millis(100),
            history_size: 1000,
            throughput_alert_threshold: 1.0,
            latency_alert_threshold_ms: 5000.0,
            memory_alert_threshold_pct: 90.0,
            utilization_alert_threshold_pct: 10.0,
            alerts_enabled: true,
            instance_label: "bitnet-gpu-hal".into(),
        }
    }
}

impl MonitorConfig {
    /// Create a new config with the given history size.
    pub fn with_history_size(mut self, size: usize) -> Self {
        self.history_size = size;
        self
    }

    /// Create a new config with the given sample interval.
    pub fn with_sample_interval(mut self, interval: Duration) -> Self {
        self.sample_interval = interval;
        self
    }

    /// Validate configuration, returning an error description on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.history_size == 0 {
            return Err("history_size must be > 0".into());
        }
        if self.throughput_alert_threshold < 0.0 {
            return Err("throughput_alert_threshold must be >= 0".into());
        }
        if self.latency_alert_threshold_ms < 0.0 {
            return Err("latency_alert_threshold_ms must be >= 0".into());
        }
        Ok(())
    }
}

// ── ThroughputTracker ────────────────────────────────────────────────────────

/// Timestamped throughput sample.
#[derive(Debug, Clone, Copy)]
pub struct ThroughputSample {
    /// Tokens generated in this interval.
    pub tokens: u64,
    /// Batches processed in this interval.
    pub batches: u64,
    /// Wall-clock timestamp.
    pub timestamp: Instant,
}

/// Tracks tokens/sec and batches/sec with a rolling window.
#[derive(Debug, Clone)]
pub struct ThroughputTracker {
    samples: VecDeque<ThroughputSample>,
    max_samples: usize,
    total_tokens: u64,
    total_batches: u64,
    start_time: Instant,
}

impl ThroughputTracker {
    /// Create a new tracker with the given window size.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            total_tokens: 0,
            total_batches: 0,
            start_time: Instant::now(),
        }
    }

    /// Record a throughput sample.
    pub fn record(&mut self, tokens: u64, batches: u64) {
        self.total_tokens += tokens;
        self.total_batches += batches;
        let sample = ThroughputSample { tokens, batches, timestamp: Instant::now() };
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Instantaneous tokens/sec from the most recent sample pair.
    pub fn tokens_per_sec(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let newest = self.samples.back().unwrap();
        let oldest = self.samples.front().unwrap();
        let elapsed = newest.timestamp.duration_since(oldest.timestamp).as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        let total: u64 = self.samples.iter().map(|s| s.tokens).sum();
        // Subtract oldest since the window starts *at* oldest.
        let windowed = total.saturating_sub(oldest.tokens);
        windowed as f64 / elapsed
    }

    /// Instantaneous batches/sec from the rolling window.
    pub fn batches_per_sec(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let newest = self.samples.back().unwrap();
        let oldest = self.samples.front().unwrap();
        let elapsed = newest.timestamp.duration_since(oldest.timestamp).as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        let total: u64 = self.samples.iter().map(|s| s.batches).sum();
        let windowed = total.saturating_sub(oldest.batches);
        windowed as f64 / elapsed
    }

    /// Rolling average tokens/sec over the entire lifetime.
    pub fn lifetime_tokens_per_sec(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        self.total_tokens as f64 / elapsed
    }

    /// Total tokens recorded.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Total batches recorded.
    pub fn total_batches(&self) -> u64 {
        self.total_batches
    }

    /// Number of samples currently held.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Reset all counters and history.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.total_tokens = 0;
        self.total_batches = 0;
        self.start_time = Instant::now();
    }
}

// ── LatencyTracker ───────────────────────────────────────────────────────────

/// Histogram bucket boundary (upper-bound inclusive, microseconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BucketBound(pub u64);

/// Tracks request latencies with percentile computation via a histogram.
///
/// Internally stores every observation in a sorted structure (for small-N
/// accuracy) *and* fixed-bucket histogram counters (for Prometheus export).
#[derive(Debug, Clone)]
pub struct LatencyTracker {
    /// Individual observations (microseconds), kept sorted.
    observations: Vec<u64>,
    /// Maximum observations kept (oldest evicted on overflow).
    max_observations: usize,
    /// Fixed histogram buckets: upper-bound-µs → count.
    buckets: BTreeMap<u64, u64>,
    /// Running sum (µs) for mean computation.
    sum_us: u64,
    /// Total observation count (may exceed `observations.len()` after eviction).
    count: u64,
}

impl LatencyTracker {
    /// Default histogram bucket boundaries (microseconds).
    pub const DEFAULT_BOUNDS_US: &[u64] = &[
        100,       // 0.1 ms
        500,       // 0.5 ms
        1_000,     // 1 ms
        5_000,     // 5 ms
        10_000,    // 10 ms
        25_000,    // 25 ms
        50_000,    // 50 ms
        100_000,   // 100 ms
        250_000,   // 250 ms
        500_000,   // 500 ms
        1_000_000, // 1 s
        5_000_000, // 5 s
    ];

    /// Create a tracker with default bucket boundaries.
    pub fn new(max_observations: usize) -> Self {
        Self::with_buckets(max_observations, Self::DEFAULT_BOUNDS_US)
    }

    /// Create a tracker with custom bucket boundaries (microseconds).
    pub fn with_buckets(max_observations: usize, bounds: &[u64]) -> Self {
        let mut buckets = BTreeMap::new();
        for &b in bounds {
            buckets.insert(b, 0);
        }
        // +Inf bucket
        buckets.insert(u64::MAX, 0);
        Self {
            observations: Vec::with_capacity(max_observations),
            max_observations,
            buckets,
            sum_us: 0,
            count: 0,
        }
    }

    /// Record a latency observation.
    pub fn record(&mut self, duration: Duration) {
        let us = duration.as_micros() as u64;
        self.sum_us = self.sum_us.saturating_add(us);
        self.count += 1;

        // Update histogram buckets (cumulative).
        for (bound, cnt) in &mut self.buckets {
            if us <= *bound {
                *cnt += 1;
            }
        }

        // Store individual observation for percentile queries.
        if self.observations.len() >= self.max_observations {
            self.observations.remove(0);
        }
        let pos = self.observations.partition_point(|&x| x < us);
        self.observations.insert(pos, us);
    }

    /// Record a latency in milliseconds.
    pub fn record_ms(&mut self, ms: f64) {
        self.record(Duration::from_secs_f64(ms / 1000.0));
    }

    /// Compute percentile (0.0–1.0) from stored observations.
    pub fn percentile(&self, p: f64) -> Duration {
        if self.observations.is_empty() {
            return Duration::ZERO;
        }
        let p = p.clamp(0.0, 1.0);
        let idx = ((self.observations.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(self.observations.len() - 1);
        Duration::from_micros(self.observations[idx])
    }

    /// P50 latency.
    pub fn p50(&self) -> Duration {
        self.percentile(0.50)
    }
    /// P90 latency.
    pub fn p90(&self) -> Duration {
        self.percentile(0.90)
    }
    /// P95 latency.
    pub fn p95(&self) -> Duration {
        self.percentile(0.95)
    }
    /// P99 latency.
    pub fn p99(&self) -> Duration {
        self.percentile(0.99)
    }

    /// Mean latency.
    pub fn mean(&self) -> Duration {
        if self.count == 0 {
            return Duration::ZERO;
        }
        Duration::from_micros(self.sum_us / self.count)
    }

    /// Minimum observed latency.
    pub fn min(&self) -> Duration {
        self.observations.first().map(|&v| Duration::from_micros(v)).unwrap_or(Duration::ZERO)
    }

    /// Maximum observed latency.
    pub fn max(&self) -> Duration {
        self.observations.last().map(|&v| Duration::from_micros(v)).unwrap_or(Duration::ZERO)
    }

    /// Total observation count.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Snapshot of histogram buckets (upper-bound-µs → cumulative count).
    pub fn histogram_buckets(&self) -> &BTreeMap<u64, u64> {
        &self.buckets
    }

    /// Reset all data.
    pub fn reset(&mut self) {
        self.observations.clear();
        for cnt in self.buckets.values_mut() {
            *cnt = 0;
        }
        self.sum_us = 0;
        self.count = 0;
    }
}

// ── MemoryTracker ────────────────────────────────────────────────────────────

/// Snapshot of memory usage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemorySnapshot {
    /// Bytes currently allocated.
    pub allocated_bytes: u64,
    /// Peak bytes allocated so far.
    pub peak_bytes: u64,
    /// Total capacity of the allocator / device.
    pub total_bytes: u64,
    /// Number of live allocations.
    pub allocation_count: u64,
    /// Timestamp of the snapshot.
    pub timestamp: Instant,
}

/// Tracks memory usage, peak, and fragmentation over time.
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    snapshots: VecDeque<MemorySnapshot>,
    max_snapshots: usize,
    current_allocated: u64,
    peak_allocated: u64,
    total_capacity: u64,
    allocation_count: u64,
}

impl MemoryTracker {
    /// Create a tracker with the given total capacity.
    pub fn new(total_capacity_bytes: u64, max_snapshots: usize) -> Self {
        Self {
            snapshots: VecDeque::with_capacity(max_snapshots),
            max_snapshots,
            current_allocated: 0,
            peak_allocated: 0,
            total_capacity: total_capacity_bytes,
            allocation_count: 0,
        }
    }

    /// Record an allocation of `bytes`.
    pub fn allocate(&mut self, bytes: u64) {
        self.current_allocated = self.current_allocated.saturating_add(bytes);
        if self.current_allocated > self.peak_allocated {
            self.peak_allocated = self.current_allocated;
        }
        self.allocation_count += 1;
        self.push_snapshot();
    }

    /// Record a deallocation of `bytes`.
    pub fn deallocate(&mut self, bytes: u64) {
        self.current_allocated = self.current_allocated.saturating_sub(bytes);
        self.allocation_count = self.allocation_count.saturating_sub(1);
        self.push_snapshot();
    }

    /// Set the current allocation state directly (e.g. from a device query).
    pub fn set_usage(&mut self, allocated: u64, alloc_count: u64) {
        self.current_allocated = allocated;
        if allocated > self.peak_allocated {
            self.peak_allocated = allocated;
        }
        self.allocation_count = alloc_count;
        self.push_snapshot();
    }

    fn push_snapshot(&mut self) {
        let snap = MemorySnapshot {
            allocated_bytes: self.current_allocated,
            peak_bytes: self.peak_allocated,
            total_bytes: self.total_capacity,
            allocation_count: self.allocation_count,
            timestamp: Instant::now(),
        };
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snap);
    }

    /// Current allocation in bytes.
    pub fn current_allocated(&self) -> u64 {
        self.current_allocated
    }

    /// Peak allocation observed.
    pub fn peak_allocated(&self) -> u64 {
        self.peak_allocated
    }

    /// Usage as a percentage of total capacity.
    pub fn usage_pct(&self) -> f64 {
        if self.total_capacity == 0 {
            return 0.0;
        }
        (self.current_allocated as f64 / self.total_capacity as f64) * 100.0
    }

    /// Estimated fragmentation ratio (0.0 = no fragmentation, 1.0 = fully
    /// fragmented). Approximation: `1.0 - (allocated / peak)` when peak > 0.
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.peak_allocated == 0 {
            return 0.0;
        }
        1.0 - (self.current_allocated as f64 / self.peak_allocated as f64)
    }

    /// Total capacity.
    pub fn total_capacity(&self) -> u64 {
        self.total_capacity
    }

    /// Number of live allocations.
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count
    }

    /// Most recent snapshot, if any.
    pub fn latest_snapshot(&self) -> Option<&MemorySnapshot> {
        self.snapshots.back()
    }

    /// Number of snapshots stored.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.snapshots.clear();
        self.current_allocated = 0;
        self.peak_allocated = 0;
        self.allocation_count = 0;
    }
}

// ── DeviceUtilization ────────────────────────────────────────────────────────

/// A single utilisation sample.
#[derive(Debug, Clone, Copy)]
pub struct UtilizationSample {
    /// Compute utilisation (0.0–100.0 %).
    pub compute_pct: f64,
    /// Memory-controller utilisation (0.0–100.0 %).
    pub memory_pct: f64,
    /// Transfer / bus utilisation (0.0–100.0 %).
    pub transfer_pct: f64,
    /// Timestamp of the reading.
    pub timestamp: Instant,
}

/// Rolling tracker for device utilisation percentages.
#[derive(Debug, Clone)]
pub struct DeviceUtilization {
    samples: VecDeque<UtilizationSample>,
    max_samples: usize,
    device_name: String,
}

impl DeviceUtilization {
    /// Create a tracker for the named device.
    pub fn new(device_name: &str, max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            device_name: device_name.to_string(),
        }
    }

    /// Record a utilisation sample.
    pub fn record(&mut self, compute_pct: f64, memory_pct: f64, transfer_pct: f64) {
        let sample = UtilizationSample {
            compute_pct: compute_pct.clamp(0.0, 100.0),
            memory_pct: memory_pct.clamp(0.0, 100.0),
            transfer_pct: transfer_pct.clamp(0.0, 100.0),
            timestamp: Instant::now(),
        };
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Average compute utilisation over the window.
    pub fn avg_compute(&self) -> f64 {
        Self::avg_field(&self.samples, |s| s.compute_pct)
    }

    /// Average memory-controller utilisation over the window.
    pub fn avg_memory(&self) -> f64 {
        Self::avg_field(&self.samples, |s| s.memory_pct)
    }

    /// Average transfer utilisation over the window.
    pub fn avg_transfer(&self) -> f64 {
        Self::avg_field(&self.samples, |s| s.transfer_pct)
    }

    /// Most recent sample, if any.
    pub fn latest(&self) -> Option<&UtilizationSample> {
        self.samples.back()
    }

    /// Device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Number of samples stored.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Reset all history.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    fn avg_field(samples: &VecDeque<UtilizationSample>, f: fn(&UtilizationSample) -> f64) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|s| f(s)).sum();
        sum / samples.len() as f64
    }
}

// ── BottleneckDetector ───────────────────────────────────────────────────────

/// Classification of performance bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BottleneckKind {
    /// Compute-bound (ALU saturation).
    Compute,
    /// Memory-bandwidth-bound.
    MemoryBandwidth,
    /// Data-transfer-bound (host ↔ device).
    Transfer,
    /// No bottleneck detected.
    None,
}

impl fmt::Display for BottleneckKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compute => write!(f, "compute-bound"),
            Self::MemoryBandwidth => write!(f, "memory-bandwidth-bound"),
            Self::Transfer => write!(f, "transfer-bound"),
            Self::None => write!(f, "none"),
        }
    }
}

/// Result of bottleneck analysis.
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// Identified bottleneck kind.
    pub kind: BottleneckKind,
    /// Confidence score (0.0–1.0).
    pub confidence: f64,
    /// Human-readable explanation.
    pub description: String,
    /// Suggested mitigation.
    pub recommendation: String,
}

/// Identifies performance bottlenecks from utilisation data.
#[derive(Debug, Clone)]
pub struct BottleneckDetector {
    /// Threshold above which a resource is considered saturated (%).
    pub saturation_threshold: f64,
    /// Threshold below which a resource is considered idle (%).
    pub idle_threshold: f64,
    /// History of analyses.
    analyses: Vec<BottleneckAnalysis>,
    max_analyses: usize,
}

impl BottleneckDetector {
    /// Create a detector with default thresholds.
    pub fn new(max_analyses: usize) -> Self {
        Self {
            saturation_threshold: 80.0,
            idle_threshold: 20.0,
            analyses: Vec::new(),
            max_analyses,
        }
    }

    /// Analyse utilisation data and return a bottleneck classification.
    pub fn analyse(&mut self, utilization: &DeviceUtilization) -> BottleneckAnalysis {
        let compute = utilization.avg_compute();
        let memory = utilization.avg_memory();
        let transfer = utilization.avg_transfer();

        let analysis = if compute >= self.saturation_threshold && memory < self.saturation_threshold
        {
            BottleneckAnalysis {
                kind: BottleneckKind::Compute,
                confidence: (compute / 100.0).min(1.0),
                description: format!(
                    "Compute utilisation high ({compute:.1}%), \
                     memory utilisation moderate ({memory:.1}%)"
                ),
                recommendation: "Consider reducing model complexity or enabling \
                     kernel fusion"
                    .into(),
            }
        } else if memory >= self.saturation_threshold && compute < self.saturation_threshold {
            BottleneckAnalysis {
                kind: BottleneckKind::MemoryBandwidth,
                confidence: (memory / 100.0).min(1.0),
                description: format!(
                    "Memory utilisation high ({memory:.1}%), \
                     compute utilisation moderate ({compute:.1}%)"
                ),
                recommendation: "Consider quantisation or reducing batch size to \
                     lower memory pressure"
                    .into(),
            }
        } else if transfer >= self.saturation_threshold {
            BottleneckAnalysis {
                kind: BottleneckKind::Transfer,
                confidence: (transfer / 100.0).min(1.0),
                description: format!("Transfer utilisation high ({transfer:.1}%)"),
                recommendation: "Consider pinned memory or overlapping transfers \
                     with compute"
                    .into(),
            }
        } else {
            BottleneckAnalysis {
                kind: BottleneckKind::None,
                confidence: 1.0,
                description: "No bottleneck detected".into(),
                recommendation: "System is balanced".into(),
            }
        };

        if self.analyses.len() >= self.max_analyses {
            self.analyses.remove(0);
        }
        self.analyses.push(analysis.clone());
        analysis
    }

    /// Latest analysis, if any.
    pub fn latest(&self) -> Option<&BottleneckAnalysis> {
        self.analyses.last()
    }

    /// Full history of analyses.
    pub fn history(&self) -> &[BottleneckAnalysis] {
        &self.analyses
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.analyses.clear();
    }
}

// ── AlertManager ─────────────────────────────────────────────────────────────

/// Severity level for alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    /// Informational only.
    Info,
    /// Potential performance degradation.
    Warning,
    /// Severe performance degradation.
    Critical,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A fired alert.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Unique alert id (monotonically increasing).
    pub id: u64,
    /// Severity of the alert.
    pub severity: AlertSeverity,
    /// Metric that triggered the alert.
    pub metric: String,
    /// Current value of the metric.
    pub value: f64,
    /// Threshold that was exceeded.
    pub threshold: f64,
    /// Human-readable message.
    pub message: String,
    /// Timestamp when the alert fired (millis since UNIX epoch).
    pub timestamp_ms: u64,
    /// Whether the alert has been acknowledged.
    pub acknowledged: bool,
}

/// Manages alert evaluation and history.
#[derive(Debug, Clone)]
pub struct AlertManager {
    alerts: Vec<Alert>,
    max_alerts: usize,
    next_id: u64,
    enabled: bool,
}

impl AlertManager {
    /// Create a new alert manager.
    pub fn new(max_alerts: usize) -> Self {
        Self { alerts: Vec::new(), max_alerts, next_id: 1, enabled: true }
    }

    /// Enable or disable alert firing.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether alerting is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Evaluate a metric value against a threshold.
    /// `exceeds_when_above`: if true, alert fires when `value > threshold`.
    /// If false, fires when `value < threshold`.
    pub fn evaluate(
        &mut self,
        metric: &str,
        value: f64,
        threshold: f64,
        exceeds_when_above: bool,
        severity: AlertSeverity,
    ) -> Option<&Alert> {
        if !self.enabled {
            return None;
        }
        let triggered = if exceeds_when_above { value > threshold } else { value < threshold };
        if !triggered {
            return None;
        }

        let direction = if exceeds_when_above { "above" } else { "below" };
        let alert = Alert {
            id: self.next_id,
            severity,
            metric: metric.to_string(),
            value,
            threshold,
            message: format!(
                "{metric} is {direction} threshold: \
                 {value:.2} (threshold: {threshold:.2})"
            ),
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            acknowledged: false,
        };
        self.next_id += 1;
        if self.alerts.len() >= self.max_alerts {
            self.alerts.remove(0);
        }
        self.alerts.push(alert);
        self.alerts.last()
    }

    /// Acknowledge an alert by id.
    pub fn acknowledge(&mut self, id: u64) -> bool {
        if let Some(alert) = self.alerts.iter_mut().find(|a| a.id == id) {
            alert.acknowledged = true;
            true
        } else {
            false
        }
    }

    /// All active (unacknowledged) alerts.
    pub fn active_alerts(&self) -> Vec<&Alert> {
        self.alerts.iter().filter(|a| !a.acknowledged).collect()
    }

    /// Full alert history.
    pub fn all_alerts(&self) -> &[Alert] {
        &self.alerts
    }

    /// Number of unacknowledged alerts.
    pub fn active_count(&self) -> usize {
        self.alerts.iter().filter(|a| !a.acknowledged).count()
    }

    /// Reset all alerts.
    pub fn reset(&mut self) {
        self.alerts.clear();
        self.next_id = 1;
    }
}

// ── MetricsExporter ──────────────────────────────────────────────────────────

/// Supported export format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Prometheus text exposition format.
    Prometheus,
    /// JSON object.
    Json,
    /// CSV table.
    Csv,
}

/// Collects typed metric values for export.
#[derive(Debug, Clone)]
pub struct MetricValue {
    pub name: String,
    pub labels: HashMap<String, String>,
    pub value: f64,
    pub help: String,
}

/// Exports metrics in multiple formats.
#[derive(Debug, Clone)]
pub struct MetricsExporter {
    instance_label: String,
    metrics: Vec<MetricValue>,
}

impl MetricsExporter {
    /// Create an exporter with the given instance label.
    pub fn new(instance_label: &str) -> Self {
        Self { instance_label: instance_label.to_string(), metrics: Vec::new() }
    }

    /// Clear all registered metrics.
    pub fn clear(&mut self) {
        self.metrics.clear();
    }

    /// Register a metric value.
    pub fn register(
        &mut self,
        name: &str,
        value: f64,
        help: &str,
        labels: HashMap<String, String>,
    ) {
        self.metrics.push(MetricValue {
            name: name.to_string(),
            labels,
            value,
            help: help.to_string(),
        });
    }

    /// Export all metrics in the given format.
    pub fn export(&self, format: ExportFormat) -> String {
        match format {
            ExportFormat::Prometheus => self.export_prometheus(),
            ExportFormat::Json => self.export_json(),
            ExportFormat::Csv => self.export_csv(),
        }
    }

    fn export_prometheus(&self) -> String {
        let mut out = String::new();
        for m in &self.metrics {
            out.push_str(&format!("# HELP {} {}\n", m.name, m.help));
            out.push_str(&format!("# TYPE {} gauge\n", m.name));
            let mut label_parts: Vec<String> =
                vec![format!("instance=\"{}\"", self.instance_label)];
            for (k, v) in &m.labels {
                label_parts.push(format!("{k}=\"{v}\""));
            }
            let labels = label_parts.join(",");
            out.push_str(&format!("{}{{{}}} {}\n", m.name, labels, m.value));
        }
        out
    }

    fn export_json(&self) -> String {
        let mut entries: Vec<String> = Vec::new();
        for m in &self.metrics {
            let mut label_entries: Vec<String> = Vec::new();
            for (k, v) in &m.labels {
                label_entries.push(format!("\"{}\":\"{}\"", k, v));
            }
            label_entries.push(format!("\"instance\":\"{}\"", self.instance_label));
            let labels_json = label_entries.join(",");
            entries.push(format!(
                "{{\"name\":\"{}\",\"value\":{},\"labels\":{{{}}},\
                 \"help\":\"{}\"}}",
                m.name, m.value, labels_json, m.help
            ));
        }
        format!("[{}]", entries.join(","))
    }

    fn export_csv(&self) -> String {
        let mut out = String::from("name,value,instance,help\n");
        for m in &self.metrics {
            out.push_str(&format!("{},{},{},{}\n", m.name, m.value, self.instance_label, m.help));
        }
        out
    }

    /// Number of registered metrics.
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }
}

// ── PerformanceReport ────────────────────────────────────────────────────────

/// Comprehensive performance report.
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Tokens per second (rolling).
    pub tokens_per_sec: f64,
    /// Batches per second (rolling).
    pub batches_per_sec: f64,
    /// Total tokens generated.
    pub total_tokens: u64,
    /// P50 latency.
    pub p50_latency: Duration,
    /// P90 latency.
    pub p90_latency: Duration,
    /// P95 latency.
    pub p95_latency: Duration,
    /// P99 latency.
    pub p99_latency: Duration,
    /// Mean latency.
    pub mean_latency: Duration,
    /// Memory allocated (bytes).
    pub memory_allocated: u64,
    /// Memory peak (bytes).
    pub memory_peak: u64,
    /// Memory usage percentage.
    pub memory_usage_pct: f64,
    /// Fragmentation ratio.
    pub fragmentation: f64,
    /// Average compute utilisation.
    pub avg_compute_pct: f64,
    /// Average memory-controller utilisation.
    pub avg_memory_util_pct: f64,
    /// Detected bottleneck.
    pub bottleneck: BottleneckKind,
    /// Number of active alerts.
    pub active_alerts: usize,
    /// Report generation timestamp (millis since epoch).
    pub timestamp_ms: u64,
}

impl PerformanceReport {
    /// Format the report as a human-readable multi-line string.
    pub fn summary(&self) -> String {
        format!(
            "=== Performance Report ===\n\
             Throughput: {:.2} tok/s, {:.2} batch/s (total: {} tokens)\n\
             Latency: p50={:.2}ms p90={:.2}ms p95={:.2}ms p99={:.2}ms \
             mean={:.2}ms\n\
             Memory: {:.1} MB allocated, {:.1} MB peak ({:.1}% used, \
             {:.2} frag)\n\
             Utilisation: compute={:.1}% memory={:.1}%\n\
             Bottleneck: {}\n\
             Active alerts: {}",
            self.tokens_per_sec,
            self.batches_per_sec,
            self.total_tokens,
            self.p50_latency.as_secs_f64() * 1000.0,
            self.p90_latency.as_secs_f64() * 1000.0,
            self.p95_latency.as_secs_f64() * 1000.0,
            self.p99_latency.as_secs_f64() * 1000.0,
            self.mean_latency.as_secs_f64() * 1000.0,
            self.memory_allocated as f64 / 1_048_576.0,
            self.memory_peak as f64 / 1_048_576.0,
            self.memory_usage_pct,
            self.fragmentation,
            self.avg_compute_pct,
            self.avg_memory_util_pct,
            self.bottleneck,
            self.active_alerts,
        )
    }
}

impl fmt::Display for PerformanceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ── PerformanceMonitorEngine ─────────────────────────────────────────────────

/// Unified performance monitoring engine.
///
/// Combines all trackers, alert evaluation, bottleneck detection, and export
/// into a single entry-point.  CPU reference implementation — works without
/// any GPU runtime.
#[derive(Debug, Clone)]
pub struct PerformanceMonitorEngine {
    config: MonitorConfig,
    throughput: ThroughputTracker,
    latency: LatencyTracker,
    memory: MemoryTracker,
    utilization: DeviceUtilization,
    bottleneck: BottleneckDetector,
    alerts: AlertManager,
    exporter: MetricsExporter,
    /// Monotonically increasing tick counter.
    tick: u64,
}

impl PerformanceMonitorEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: MonitorConfig) -> Self {
        let history = config.history_size;
        let label = config.instance_label.clone();
        Self {
            throughput: ThroughputTracker::new(history),
            latency: LatencyTracker::new(history),
            memory: MemoryTracker::new(0, history),
            utilization: DeviceUtilization::new("cpu", history),
            bottleneck: BottleneckDetector::new(history),
            alerts: AlertManager::new(history),
            exporter: MetricsExporter::new(&label),
            tick: 0,
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_engine() -> Self {
        Self::new(MonitorConfig::default())
    }

    /// Set the total memory capacity (needed for percentage calculations).
    pub fn set_memory_capacity(&mut self, bytes: u64) {
        self.memory = MemoryTracker::new(bytes, self.config.history_size);
    }

    /// Set the device name for utilisation tracking.
    pub fn set_device_name(&mut self, name: &str) {
        self.utilization = DeviceUtilization::new(name, self.config.history_size);
    }

    // ── Recording ────────────────────────────────────────────────────

    /// Record a throughput sample.
    pub fn record_throughput(&mut self, tokens: u64, batches: u64) {
        self.throughput.record(tokens, batches);
    }

    /// Record a latency observation.
    pub fn record_latency(&mut self, duration: Duration) {
        self.latency.record(duration);
    }

    /// Record a memory allocation.
    pub fn record_allocation(&mut self, bytes: u64) {
        self.memory.allocate(bytes);
    }

    /// Record a memory deallocation.
    pub fn record_deallocation(&mut self, bytes: u64) {
        self.memory.deallocate(bytes);
    }

    /// Record a device utilisation sample.
    pub fn record_utilization(&mut self, compute_pct: f64, memory_pct: f64, transfer_pct: f64) {
        self.utilization.record(compute_pct, memory_pct, transfer_pct);
    }

    // ── Evaluation ───────────────────────────────────────────────────

    /// Run one evaluation tick: check alerts and update bottleneck analysis.
    pub fn tick(&mut self) {
        self.tick += 1;

        // Throughput alert (fires when *below* threshold).
        let tps = self.throughput.tokens_per_sec();
        if self.config.alerts_enabled && self.throughput.sample_count() >= 2 {
            self.alerts.evaluate(
                "tokens_per_sec",
                tps,
                self.config.throughput_alert_threshold,
                false,
                AlertSeverity::Warning,
            );
        }

        // Latency alert (fires when *above* threshold).
        let p99_ms = self.latency.p99().as_secs_f64() * 1000.0;
        if self.config.alerts_enabled && self.latency.count() > 0 {
            self.alerts.evaluate(
                "p99_latency_ms",
                p99_ms,
                self.config.latency_alert_threshold_ms,
                true,
                AlertSeverity::Critical,
            );
        }

        // Memory alert (fires when *above* threshold).
        let mem_pct = self.memory.usage_pct();
        if self.config.alerts_enabled && self.memory.total_capacity() > 0 {
            self.alerts.evaluate(
                "memory_usage_pct",
                mem_pct,
                self.config.memory_alert_threshold_pct,
                true,
                AlertSeverity::Critical,
            );
        }

        // Utilisation alert (fires when *below* threshold).
        let compute = self.utilization.avg_compute();
        if self.config.alerts_enabled && self.utilization.sample_count() > 0 {
            self.alerts.evaluate(
                "compute_utilization_pct",
                compute,
                self.config.utilization_alert_threshold_pct,
                false,
                AlertSeverity::Info,
            );
        }

        // Bottleneck detection.
        self.bottleneck.analyse(&self.utilization);
    }

    // ── Reporting ────────────────────────────────────────────────────

    /// Generate a comprehensive performance report.
    pub fn report(&self) -> PerformanceReport {
        let bottleneck = self.bottleneck.latest().map(|a| a.kind).unwrap_or(BottleneckKind::None);
        PerformanceReport {
            tokens_per_sec: self.throughput.tokens_per_sec(),
            batches_per_sec: self.throughput.batches_per_sec(),
            total_tokens: self.throughput.total_tokens(),
            p50_latency: self.latency.p50(),
            p90_latency: self.latency.p90(),
            p95_latency: self.latency.p95(),
            p99_latency: self.latency.p99(),
            mean_latency: self.latency.mean(),
            memory_allocated: self.memory.current_allocated(),
            memory_peak: self.memory.peak_allocated(),
            memory_usage_pct: self.memory.usage_pct(),
            fragmentation: self.memory.fragmentation_ratio(),
            avg_compute_pct: self.utilization.avg_compute(),
            avg_memory_util_pct: self.utilization.avg_memory(),
            bottleneck,
            active_alerts: self.alerts.active_count(),
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Export current metrics in the given format.
    pub fn export(&mut self, format: ExportFormat) -> String {
        self.refresh_exporter();
        self.exporter.export(format)
    }

    fn refresh_exporter(&mut self) {
        self.exporter.clear();
        let empty = HashMap::new();
        self.exporter.register(
            "bitnet_tokens_per_sec",
            self.throughput.tokens_per_sec(),
            "Tokens generated per second",
            empty.clone(),
        );
        self.exporter.register(
            "bitnet_batches_per_sec",
            self.throughput.batches_per_sec(),
            "Batches processed per second",
            empty.clone(),
        );
        self.exporter.register(
            "bitnet_p50_latency_ms",
            self.latency.p50().as_secs_f64() * 1000.0,
            "P50 latency in milliseconds",
            empty.clone(),
        );
        self.exporter.register(
            "bitnet_p99_latency_ms",
            self.latency.p99().as_secs_f64() * 1000.0,
            "P99 latency in milliseconds",
            empty.clone(),
        );
        self.exporter.register(
            "bitnet_memory_allocated_bytes",
            self.memory.current_allocated() as f64,
            "Currently allocated memory in bytes",
            empty.clone(),
        );
        self.exporter.register(
            "bitnet_memory_peak_bytes",
            self.memory.peak_allocated() as f64,
            "Peak memory allocation in bytes",
            empty.clone(),
        );
        self.exporter.register(
            "bitnet_memory_usage_pct",
            self.memory.usage_pct(),
            "Memory usage percentage",
            empty.clone(),
        );
        self.exporter.register(
            "bitnet_compute_utilization_pct",
            self.utilization.avg_compute(),
            "Average compute utilisation",
            empty.clone(),
        );
        self.exporter.register(
            "bitnet_active_alerts",
            self.alerts.active_count() as f64,
            "Number of active alerts",
            empty,
        );
    }

    // ── Accessors ────────────────────────────────────────────────────

    /// Access the throughput tracker.
    pub fn throughput(&self) -> &ThroughputTracker {
        &self.throughput
    }

    /// Access the latency tracker.
    pub fn latency(&self) -> &LatencyTracker {
        &self.latency
    }

    /// Access the memory tracker.
    pub fn memory(&self) -> &MemoryTracker {
        &self.memory
    }

    /// Access the utilisation tracker.
    pub fn utilization(&self) -> &DeviceUtilization {
        &self.utilization
    }

    /// Access the bottleneck detector.
    pub fn bottleneck(&self) -> &BottleneckDetector {
        &self.bottleneck
    }

    /// Access the alert manager.
    pub fn alerts(&self) -> &AlertManager {
        &self.alerts
    }

    /// Access the alert manager mutably.
    pub fn alerts_mut(&mut self) -> &mut AlertManager {
        &mut self.alerts
    }

    /// Current tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick
    }

    /// Current configuration.
    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }

    /// Reset all trackers, alerts, and counters.
    pub fn reset(&mut self) {
        self.throughput.reset();
        self.latency.reset();
        self.memory.reset();
        self.utilization.reset();
        self.bottleneck.reset();
        self.alerts.reset();
        self.exporter.clear();
        self.tick = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── MonitorConfig ────────────────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        let cfg = MonitorConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_zero_history_invalid() {
        let mut cfg = MonitorConfig::default();
        cfg.history_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_negative_throughput_threshold_invalid() {
        let mut cfg = MonitorConfig::default();
        cfg.throughput_alert_threshold = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_negative_latency_threshold_invalid() {
        let mut cfg = MonitorConfig::default();
        cfg.latency_alert_threshold_ms = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_builder_history_size() {
        let cfg = MonitorConfig::default().with_history_size(500);
        assert_eq!(cfg.history_size, 500);
    }

    #[test]
    fn config_builder_sample_interval() {
        let cfg = MonitorConfig::default().with_sample_interval(Duration::from_secs(1));
        assert_eq!(cfg.sample_interval, Duration::from_secs(1));
    }

    #[test]
    fn config_default_alerts_enabled() {
        let cfg = MonitorConfig::default();
        assert!(cfg.alerts_enabled);
    }

    // ── ThroughputTracker ────────────────────────────────────────────

    #[test]
    fn throughput_new_empty() {
        let t = ThroughputTracker::new(10);
        assert_eq!(t.total_tokens(), 0);
        assert_eq!(t.total_batches(), 0);
        assert_eq!(t.sample_count(), 0);
    }

    #[test]
    fn throughput_record_updates_totals() {
        let mut t = ThroughputTracker::new(10);
        t.record(50, 2);
        assert_eq!(t.total_tokens(), 50);
        assert_eq!(t.total_batches(), 2);
        assert_eq!(t.sample_count(), 1);
    }

    #[test]
    fn throughput_multiple_records() {
        let mut t = ThroughputTracker::new(10);
        t.record(10, 1);
        t.record(20, 2);
        t.record(30, 3);
        assert_eq!(t.total_tokens(), 60);
        assert_eq!(t.total_batches(), 6);
        assert_eq!(t.sample_count(), 3);
    }

    #[test]
    fn throughput_window_eviction() {
        let mut t = ThroughputTracker::new(3);
        for i in 0..5 {
            t.record(i + 1, 1);
        }
        assert_eq!(t.sample_count(), 3);
        assert_eq!(t.total_tokens(), 15);
    }

    #[test]
    fn throughput_tokens_per_sec_empty() {
        let t = ThroughputTracker::new(10);
        assert_eq!(t.tokens_per_sec(), 0.0);
    }

    #[test]
    fn throughput_tokens_per_sec_single() {
        let mut t = ThroughputTracker::new(10);
        t.record(100, 1);
        // Single sample → 0
        assert_eq!(t.tokens_per_sec(), 0.0);
    }

    #[test]
    fn throughput_batches_per_sec_empty() {
        let t = ThroughputTracker::new(10);
        assert_eq!(t.batches_per_sec(), 0.0);
    }

    #[test]
    fn throughput_reset() {
        let mut t = ThroughputTracker::new(10);
        t.record(100, 5);
        t.reset();
        assert_eq!(t.total_tokens(), 0);
        assert_eq!(t.total_batches(), 0);
        assert_eq!(t.sample_count(), 0);
    }

    #[test]
    fn throughput_lifetime_tokens_per_sec() {
        let mut t = ThroughputTracker::new(10);
        t.record(1000, 10);
        // Just verify it returns a non-negative number.
        assert!(t.lifetime_tokens_per_sec() >= 0.0);
    }

    #[test]
    fn throughput_with_time_gap() {
        let mut t = ThroughputTracker::new(100);
        t.record(10, 1);
        std::thread::sleep(Duration::from_millis(20));
        t.record(20, 2);
        // With a gap, we should get a non-zero rate.
        assert!(t.tokens_per_sec() > 0.0);
        assert!(t.batches_per_sec() > 0.0);
    }

    // ── LatencyTracker ───────────────────────────────────────────────

    #[test]
    fn latency_new_empty() {
        let l = LatencyTracker::new(100);
        assert_eq!(l.count(), 0);
        assert_eq!(l.mean(), Duration::ZERO);
        assert_eq!(l.p50(), Duration::ZERO);
    }

    #[test]
    fn latency_record_single() {
        let mut l = LatencyTracker::new(100);
        l.record(Duration::from_millis(10));
        assert_eq!(l.count(), 1);
        assert_eq!(l.p50(), Duration::from_millis(10));
    }

    #[test]
    fn latency_record_ms() {
        let mut l = LatencyTracker::new(100);
        l.record_ms(5.0);
        assert_eq!(l.count(), 1);
        // Should be approximately 5ms.
        let p = l.p50().as_micros();
        assert!(p >= 4900 && p <= 5100);
    }

    #[test]
    fn latency_percentiles_sorted() {
        let mut l = LatencyTracker::new(1000);
        for i in 1..=100 {
            l.record(Duration::from_millis(i));
        }
        assert!(l.p50() <= l.p90());
        assert!(l.p90() <= l.p95());
        assert!(l.p95() <= l.p99());
    }

    #[test]
    fn latency_p50_accuracy() {
        let mut l = LatencyTracker::new(1000);
        for i in 1..=100 {
            l.record(Duration::from_millis(i));
        }
        let p50_ms = l.p50().as_millis();
        // P50 of 1..=100 should be around 50.
        assert!((45..=55).contains(&p50_ms), "p50={p50_ms}ms, expected ~50ms");
    }

    #[test]
    fn latency_p99_accuracy() {
        let mut l = LatencyTracker::new(1000);
        for i in 1..=100 {
            l.record(Duration::from_millis(i));
        }
        let p99_ms = l.p99().as_millis();
        assert!((95..=100).contains(&p99_ms), "p99={p99_ms}ms, expected ~99ms");
    }

    #[test]
    fn latency_mean() {
        let mut l = LatencyTracker::new(1000);
        l.record(Duration::from_millis(10));
        l.record(Duration::from_millis(20));
        l.record(Duration::from_millis(30));
        let mean_ms = l.mean().as_millis();
        assert_eq!(mean_ms, 20);
    }

    #[test]
    fn latency_min_max() {
        let mut l = LatencyTracker::new(1000);
        l.record(Duration::from_millis(5));
        l.record(Duration::from_millis(50));
        l.record(Duration::from_millis(500));
        assert_eq!(l.min().as_millis(), 5);
        assert_eq!(l.max().as_millis(), 500);
    }

    #[test]
    fn latency_histogram_buckets() {
        let mut l = LatencyTracker::new(100);
        l.record(Duration::from_micros(50)); // 50 µs
        l.record(Duration::from_millis(2)); // 2000 µs
        let buckets = l.histogram_buckets();
        // 50µs is <= 100µs bucket
        assert!(*buckets.get(&100).unwrap() >= 1);
        // Both are <= 5000µs bucket
        assert!(*buckets.get(&5_000).unwrap() >= 2);
    }

    #[test]
    fn latency_eviction() {
        let mut l = LatencyTracker::new(5);
        for i in 0..10 {
            l.record(Duration::from_millis(i));
        }
        // Total count tracks all, but observation vector is capped.
        assert_eq!(l.count(), 10);
    }

    #[test]
    fn latency_reset() {
        let mut l = LatencyTracker::new(100);
        l.record(Duration::from_millis(10));
        l.reset();
        assert_eq!(l.count(), 0);
        assert_eq!(l.mean(), Duration::ZERO);
    }

    #[test]
    fn latency_custom_buckets() {
        let mut l = LatencyTracker::with_buckets(100, &[1000, 10_000]);
        l.record(Duration::from_micros(500));
        l.record(Duration::from_micros(5_000));
        let buckets = l.histogram_buckets();
        assert_eq!(*buckets.get(&1000).unwrap(), 1);
        assert_eq!(*buckets.get(&10_000).unwrap(), 2);
    }

    #[test]
    fn latency_percentile_clamp() {
        let mut l = LatencyTracker::new(100);
        l.record(Duration::from_millis(10));
        // Out-of-range percentile should be clamped.
        let _ = l.percentile(1.5);
        let _ = l.percentile(-0.5);
    }

    // ── MemoryTracker ────────────────────────────────────────────────

    #[test]
    fn memory_new_defaults() {
        let m = MemoryTracker::new(1024 * 1024, 100);
        assert_eq!(m.current_allocated(), 0);
        assert_eq!(m.peak_allocated(), 0);
        assert_eq!(m.total_capacity(), 1024 * 1024);
    }

    #[test]
    fn memory_allocate() {
        let mut m = MemoryTracker::new(1000, 100);
        m.allocate(100);
        assert_eq!(m.current_allocated(), 100);
        assert_eq!(m.peak_allocated(), 100);
        assert_eq!(m.allocation_count(), 1);
    }

    #[test]
    fn memory_deallocate() {
        let mut m = MemoryTracker::new(1000, 100);
        m.allocate(100);
        m.deallocate(60);
        assert_eq!(m.current_allocated(), 40);
        assert_eq!(m.peak_allocated(), 100);
    }

    #[test]
    fn memory_deallocate_saturates() {
        let mut m = MemoryTracker::new(1000, 100);
        m.allocate(50);
        m.deallocate(100);
        assert_eq!(m.current_allocated(), 0);
    }

    #[test]
    fn memory_peak_tracks_maximum() {
        let mut m = MemoryTracker::new(10_000, 100);
        m.allocate(500);
        m.allocate(500);
        m.deallocate(800);
        assert_eq!(m.peak_allocated(), 1000);
    }

    #[test]
    fn memory_usage_pct() {
        let mut m = MemoryTracker::new(1000, 100);
        m.allocate(250);
        let pct = m.usage_pct();
        assert!((pct - 25.0).abs() < 0.01);
    }

    #[test]
    fn memory_usage_pct_zero_capacity() {
        let m = MemoryTracker::new(0, 100);
        assert_eq!(m.usage_pct(), 0.0);
    }

    #[test]
    fn memory_fragmentation_no_alloc() {
        let m = MemoryTracker::new(1000, 100);
        assert_eq!(m.fragmentation_ratio(), 0.0);
    }

    #[test]
    fn memory_fragmentation_after_dealloc() {
        let mut m = MemoryTracker::new(10_000, 100);
        m.allocate(1000);
        m.deallocate(500);
        let frag = m.fragmentation_ratio();
        assert!((frag - 0.5).abs() < 0.01);
    }

    #[test]
    fn memory_set_usage() {
        let mut m = MemoryTracker::new(1000, 100);
        m.set_usage(300, 5);
        assert_eq!(m.current_allocated(), 300);
        assert_eq!(m.allocation_count(), 5);
    }

    #[test]
    fn memory_snapshot_recorded() {
        let mut m = MemoryTracker::new(1000, 100);
        m.allocate(100);
        assert_eq!(m.snapshot_count(), 1);
        let snap = m.latest_snapshot().unwrap();
        assert_eq!(snap.allocated_bytes, 100);
    }

    #[test]
    fn memory_snapshot_eviction() {
        let mut m = MemoryTracker::new(10_000, 3);
        for i in 1..=5 {
            m.allocate(i * 10);
        }
        assert_eq!(m.snapshot_count(), 3);
    }

    #[test]
    fn memory_reset() {
        let mut m = MemoryTracker::new(1000, 100);
        m.allocate(500);
        m.reset();
        assert_eq!(m.current_allocated(), 0);
        assert_eq!(m.peak_allocated(), 0);
        assert_eq!(m.snapshot_count(), 0);
    }

    // ── DeviceUtilization ────────────────────────────────────────────

    #[test]
    fn utilization_new_empty() {
        let u = DeviceUtilization::new("gpu-0", 100);
        assert_eq!(u.device_name(), "gpu-0");
        assert_eq!(u.sample_count(), 0);
        assert_eq!(u.avg_compute(), 0.0);
    }

    #[test]
    fn utilization_record() {
        let mut u = DeviceUtilization::new("gpu-0", 100);
        u.record(80.0, 60.0, 30.0);
        assert_eq!(u.sample_count(), 1);
        let latest = u.latest().unwrap();
        assert!((latest.compute_pct - 80.0).abs() < 0.01);
    }

    #[test]
    fn utilization_averages() {
        let mut u = DeviceUtilization::new("cpu", 100);
        u.record(40.0, 20.0, 10.0);
        u.record(60.0, 40.0, 30.0);
        assert!((u.avg_compute() - 50.0).abs() < 0.01);
        assert!((u.avg_memory() - 30.0).abs() < 0.01);
        assert!((u.avg_transfer() - 20.0).abs() < 0.01);
    }

    #[test]
    fn utilization_clamps_values() {
        let mut u = DeviceUtilization::new("gpu", 100);
        u.record(150.0, -10.0, 200.0);
        let s = u.latest().unwrap();
        assert_eq!(s.compute_pct, 100.0);
        assert_eq!(s.memory_pct, 0.0);
        assert_eq!(s.transfer_pct, 100.0);
    }

    #[test]
    fn utilization_eviction() {
        let mut u = DeviceUtilization::new("gpu", 3);
        for _ in 0..5 {
            u.record(50.0, 50.0, 50.0);
        }
        assert_eq!(u.sample_count(), 3);
    }

    #[test]
    fn utilization_reset() {
        let mut u = DeviceUtilization::new("gpu", 100);
        u.record(50.0, 50.0, 50.0);
        u.reset();
        assert_eq!(u.sample_count(), 0);
    }

    // ── BottleneckDetector ───────────────────────────────────────────

    #[test]
    fn bottleneck_no_data() {
        let mut d = BottleneckDetector::new(10);
        let u = DeviceUtilization::new("gpu", 100);
        let a = d.analyse(&u);
        assert_eq!(a.kind, BottleneckKind::None);
    }

    #[test]
    fn bottleneck_compute_bound() {
        let mut d = BottleneckDetector::new(10);
        let mut u = DeviceUtilization::new("gpu", 100);
        u.record(95.0, 30.0, 10.0);
        let a = d.analyse(&u);
        assert_eq!(a.kind, BottleneckKind::Compute);
    }

    #[test]
    fn bottleneck_memory_bound() {
        let mut d = BottleneckDetector::new(10);
        let mut u = DeviceUtilization::new("gpu", 100);
        u.record(30.0, 95.0, 10.0);
        let a = d.analyse(&u);
        assert_eq!(a.kind, BottleneckKind::MemoryBandwidth);
    }

    #[test]
    fn bottleneck_transfer_bound() {
        let mut d = BottleneckDetector::new(10);
        let mut u = DeviceUtilization::new("gpu", 100);
        u.record(30.0, 30.0, 90.0);
        let a = d.analyse(&u);
        assert_eq!(a.kind, BottleneckKind::Transfer);
    }

    #[test]
    fn bottleneck_balanced() {
        let mut d = BottleneckDetector::new(10);
        let mut u = DeviceUtilization::new("gpu", 100);
        u.record(50.0, 50.0, 50.0);
        let a = d.analyse(&u);
        assert_eq!(a.kind, BottleneckKind::None);
    }

    #[test]
    fn bottleneck_history() {
        let mut d = BottleneckDetector::new(10);
        let mut u = DeviceUtilization::new("gpu", 100);
        u.record(95.0, 30.0, 10.0);
        d.analyse(&u);
        u.record(30.0, 95.0, 10.0);
        d.analyse(&u);
        assert_eq!(d.history().len(), 2);
    }

    #[test]
    fn bottleneck_kind_display() {
        assert_eq!(format!("{}", BottleneckKind::Compute), "compute-bound");
        assert_eq!(format!("{}", BottleneckKind::MemoryBandwidth), "memory-bandwidth-bound");
        assert_eq!(format!("{}", BottleneckKind::Transfer), "transfer-bound");
        assert_eq!(format!("{}", BottleneckKind::None), "none");
    }

    #[test]
    fn bottleneck_reset() {
        let mut d = BottleneckDetector::new(10);
        let mut u = DeviceUtilization::new("gpu", 100);
        u.record(95.0, 30.0, 10.0);
        d.analyse(&u);
        d.reset();
        assert!(d.history().is_empty());
    }

    #[test]
    fn bottleneck_confidence_range() {
        let mut d = BottleneckDetector::new(10);
        let mut u = DeviceUtilization::new("gpu", 100);
        u.record(95.0, 30.0, 10.0);
        let a = d.analyse(&u);
        assert!(a.confidence >= 0.0 && a.confidence <= 1.0);
    }

    // ── AlertManager ─────────────────────────────────────────────────

    #[test]
    fn alert_manager_new_empty() {
        let a = AlertManager::new(100);
        assert_eq!(a.active_count(), 0);
        assert!(a.is_enabled());
    }

    #[test]
    fn alert_fires_above() {
        let mut a = AlertManager::new(100);
        let result = a.evaluate("test_metric", 95.0, 90.0, true, AlertSeverity::Warning);
        assert!(result.is_some());
        assert_eq!(a.active_count(), 1);
    }

    #[test]
    fn alert_no_fire_below_threshold() {
        let mut a = AlertManager::new(100);
        let result = a.evaluate("test_metric", 50.0, 90.0, true, AlertSeverity::Warning);
        assert!(result.is_none());
        assert_eq!(a.active_count(), 0);
    }

    #[test]
    fn alert_fires_below() {
        let mut a = AlertManager::new(100);
        let result = a.evaluate("throughput", 0.5, 1.0, false, AlertSeverity::Critical);
        assert!(result.is_some());
    }

    #[test]
    fn alert_no_fire_when_disabled() {
        let mut a = AlertManager::new(100);
        a.set_enabled(false);
        let result = a.evaluate("test_metric", 95.0, 90.0, true, AlertSeverity::Warning);
        assert!(result.is_none());
    }

    #[test]
    fn alert_acknowledge() {
        let mut a = AlertManager::new(100);
        a.evaluate("m", 95.0, 90.0, true, AlertSeverity::Warning);
        assert!(a.acknowledge(1));
        assert_eq!(a.active_count(), 0);
    }

    #[test]
    fn alert_acknowledge_nonexistent() {
        let mut a = AlertManager::new(100);
        assert!(!a.acknowledge(999));
    }

    #[test]
    fn alert_severity_display() {
        assert_eq!(format!("{}", AlertSeverity::Info), "INFO");
        assert_eq!(format!("{}", AlertSeverity::Warning), "WARNING");
        assert_eq!(format!("{}", AlertSeverity::Critical), "CRITICAL");
    }

    #[test]
    fn alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
    }

    #[test]
    fn alert_ids_monotonic() {
        let mut a = AlertManager::new(100);
        a.evaluate("m1", 95.0, 90.0, true, AlertSeverity::Warning);
        a.evaluate("m2", 95.0, 90.0, true, AlertSeverity::Warning);
        let alerts = a.all_alerts();
        assert_eq!(alerts[0].id, 1);
        assert_eq!(alerts[1].id, 2);
    }

    #[test]
    fn alert_eviction() {
        let mut a = AlertManager::new(3);
        for i in 0..5 {
            a.evaluate(&format!("m{i}"), 95.0, 90.0, true, AlertSeverity::Info);
        }
        assert_eq!(a.all_alerts().len(), 3);
    }

    #[test]
    fn alert_active_vs_acknowledged() {
        let mut a = AlertManager::new(100);
        a.evaluate("m1", 95.0, 90.0, true, AlertSeverity::Warning);
        a.evaluate("m2", 95.0, 90.0, true, AlertSeverity::Warning);
        a.acknowledge(1);
        let active = a.active_alerts();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, 2);
    }

    #[test]
    fn alert_message_content() {
        let mut a = AlertManager::new(100);
        a.evaluate("cpu_temp", 95.0, 90.0, true, AlertSeverity::Critical);
        let alert = &a.all_alerts()[0];
        assert!(alert.message.contains("cpu_temp"));
        assert!(alert.message.contains("above"));
    }

    #[test]
    fn alert_reset() {
        let mut a = AlertManager::new(100);
        a.evaluate("m", 95.0, 90.0, true, AlertSeverity::Warning);
        a.reset();
        assert_eq!(a.active_count(), 0);
        assert!(a.all_alerts().is_empty());
    }

    // ── MetricsExporter ──────────────────────────────────────────────

    #[test]
    fn exporter_new_empty() {
        let e = MetricsExporter::new("test");
        assert_eq!(e.metric_count(), 0);
    }

    #[test]
    fn exporter_register() {
        let mut e = MetricsExporter::new("test");
        e.register("tok_per_sec", 42.0, "tokens/sec", HashMap::new());
        assert_eq!(e.metric_count(), 1);
    }

    #[test]
    fn exporter_prometheus_format() {
        let mut e = MetricsExporter::new("test-instance");
        e.register("my_metric", 3.14, "Test metric", HashMap::new());
        let out = e.export(ExportFormat::Prometheus);
        assert!(out.contains("# HELP my_metric Test metric"));
        assert!(out.contains("# TYPE my_metric gauge"));
        assert!(out.contains("instance=\"test-instance\""));
        assert!(out.contains("3.14"));
    }

    #[test]
    fn exporter_json_format() {
        let mut e = MetricsExporter::new("inst");
        e.register("m1", 1.0, "help1", HashMap::new());
        let out = e.export(ExportFormat::Json);
        assert!(out.starts_with('['));
        assert!(out.ends_with(']'));
        assert!(out.contains("\"name\":\"m1\""));
        assert!(out.contains("\"value\":1"));
    }

    #[test]
    fn exporter_csv_format() {
        let mut e = MetricsExporter::new("inst");
        e.register("m1", 42.0, "my help", HashMap::new());
        let out = e.export(ExportFormat::Csv);
        assert!(out.starts_with("name,value,instance,help\n"));
        assert!(out.contains("m1,42,inst,my help"));
    }

    #[test]
    fn exporter_clear() {
        let mut e = MetricsExporter::new("inst");
        e.register("m1", 1.0, "h", HashMap::new());
        e.clear();
        assert_eq!(e.metric_count(), 0);
    }

    #[test]
    fn exporter_multiple_metrics() {
        let mut e = MetricsExporter::new("inst");
        e.register("a", 1.0, "h1", HashMap::new());
        e.register("b", 2.0, "h2", HashMap::new());
        e.register("c", 3.0, "h3", HashMap::new());
        assert_eq!(e.metric_count(), 3);
        let prom = e.export(ExportFormat::Prometheus);
        assert!(prom.contains("a{"));
        assert!(prom.contains("b{"));
        assert!(prom.contains("c{"));
    }

    #[test]
    fn exporter_with_labels() {
        let mut e = MetricsExporter::new("inst");
        let mut labels = HashMap::new();
        labels.insert("device".into(), "gpu-0".into());
        e.register("utilization", 85.0, "GPU usage", labels);
        let prom = e.export(ExportFormat::Prometheus);
        assert!(prom.contains("device=\"gpu-0\""));
    }

    #[test]
    fn exporter_json_with_labels() {
        let mut e = MetricsExporter::new("inst");
        let mut labels = HashMap::new();
        labels.insert("device".into(), "gpu-0".into());
        e.register("utilization", 85.0, "GPU usage", labels);
        let json = e.export(ExportFormat::Json);
        assert!(json.contains("\"device\":\"gpu-0\""));
    }

    // ── PerformanceReport ────────────────────────────────────────────

    #[test]
    fn report_summary_contains_sections() {
        let report = PerformanceReport {
            tokens_per_sec: 10.5,
            batches_per_sec: 2.1,
            total_tokens: 100,
            p50_latency: Duration::from_millis(5),
            p90_latency: Duration::from_millis(10),
            p95_latency: Duration::from_millis(15),
            p99_latency: Duration::from_millis(50),
            mean_latency: Duration::from_millis(8),
            memory_allocated: 1_048_576,
            memory_peak: 2_097_152,
            memory_usage_pct: 50.0,
            fragmentation: 0.5,
            avg_compute_pct: 75.0,
            avg_memory_util_pct: 40.0,
            bottleneck: BottleneckKind::Compute,
            active_alerts: 1,
            timestamp_ms: 0,
        };
        let s = report.summary();
        assert!(s.contains("Performance Report"));
        assert!(s.contains("tok/s"));
        assert!(s.contains("Latency"));
        assert!(s.contains("Memory"));
        assert!(s.contains("compute-bound"));
    }

    #[test]
    fn report_display_impl() {
        let report = PerformanceReport {
            tokens_per_sec: 0.0,
            batches_per_sec: 0.0,
            total_tokens: 0,
            p50_latency: Duration::ZERO,
            p90_latency: Duration::ZERO,
            p95_latency: Duration::ZERO,
            p99_latency: Duration::ZERO,
            mean_latency: Duration::ZERO,
            memory_allocated: 0,
            memory_peak: 0,
            memory_usage_pct: 0.0,
            fragmentation: 0.0,
            avg_compute_pct: 0.0,
            avg_memory_util_pct: 0.0,
            bottleneck: BottleneckKind::None,
            active_alerts: 0,
            timestamp_ms: 0,
        };
        let display = format!("{report}");
        assert!(display.contains("Performance Report"));
    }

    // ── PerformanceMonitorEngine ─────────────────────────────────────

    #[test]
    fn engine_new_default() {
        let e = PerformanceMonitorEngine::default_engine();
        assert_eq!(e.tick_count(), 0);
    }

    #[test]
    fn engine_record_throughput() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.record_throughput(100, 5);
        assert_eq!(e.throughput().total_tokens(), 100);
    }

    #[test]
    fn engine_record_latency() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.record_latency(Duration::from_millis(10));
        assert_eq!(e.latency().count(), 1);
    }

    #[test]
    fn engine_record_allocation() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.set_memory_capacity(10_000);
        e.record_allocation(500);
        assert_eq!(e.memory().current_allocated(), 500);
    }

    #[test]
    fn engine_record_deallocation() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.set_memory_capacity(10_000);
        e.record_allocation(500);
        e.record_deallocation(200);
        assert_eq!(e.memory().current_allocated(), 300);
    }

    #[test]
    fn engine_record_utilization() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.record_utilization(80.0, 60.0, 30.0);
        assert_eq!(e.utilization().sample_count(), 1);
    }

    #[test]
    fn engine_tick_increments() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.tick();
        e.tick();
        assert_eq!(e.tick_count(), 2);
    }

    #[test]
    fn engine_tick_fires_latency_alert() {
        let mut cfg = MonitorConfig::default();
        cfg.latency_alert_threshold_ms = 10.0;
        let mut e = PerformanceMonitorEngine::new(cfg);
        e.record_latency(Duration::from_millis(50));
        e.tick();
        assert!(e.alerts().active_count() > 0);
    }

    #[test]
    fn engine_tick_fires_memory_alert() {
        let mut cfg = MonitorConfig::default();
        cfg.memory_alert_threshold_pct = 50.0;
        let mut e = PerformanceMonitorEngine::new(cfg);
        e.set_memory_capacity(1000);
        e.record_allocation(800);
        e.tick();
        assert!(e.alerts().active_count() > 0);
    }

    #[test]
    fn engine_tick_no_alert_when_ok() {
        let cfg = MonitorConfig::default();
        let mut e = PerformanceMonitorEngine::new(cfg);
        e.set_memory_capacity(10_000);
        e.record_allocation(100);
        e.record_latency(Duration::from_millis(1));
        e.record_utilization(50.0, 50.0, 50.0);
        e.tick();
        // Memory at 1%, latency 1ms, utilisation 50% — no alerts.
        // (utilisation alert may fire if below threshold but default
        //  threshold is 10% so 50% is OK)
        let mem_alerts: Vec<_> =
            e.alerts().all_alerts().iter().filter(|a| a.metric == "memory_usage_pct").collect();
        assert!(mem_alerts.is_empty());
    }

    #[test]
    fn engine_report_generated() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.record_throughput(10, 1);
        e.record_latency(Duration::from_millis(5));
        let report = e.report();
        assert_eq!(report.total_tokens, 10);
    }

    #[test]
    fn engine_export_prometheus() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.record_throughput(10, 1);
        let out = e.export(ExportFormat::Prometheus);
        assert!(out.contains("bitnet_tokens_per_sec"));
    }

    #[test]
    fn engine_export_json() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.record_throughput(10, 1);
        let out = e.export(ExportFormat::Json);
        assert!(out.contains("bitnet_tokens_per_sec"));
    }

    #[test]
    fn engine_export_csv() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.record_throughput(10, 1);
        let out = e.export(ExportFormat::Csv);
        assert!(out.contains("bitnet_tokens_per_sec"));
    }

    #[test]
    fn engine_set_device_name() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.set_device_name("rtx-4090");
        assert_eq!(e.utilization().device_name(), "rtx-4090");
    }

    #[test]
    fn engine_reset_all() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.record_throughput(100, 5);
        e.record_latency(Duration::from_millis(10));
        e.set_memory_capacity(10_000);
        e.record_allocation(500);
        e.record_utilization(80.0, 60.0, 30.0);
        e.tick();
        e.reset();
        assert_eq!(e.tick_count(), 0);
        assert_eq!(e.throughput().total_tokens(), 0);
        assert_eq!(e.latency().count(), 0);
        assert_eq!(e.memory().current_allocated(), 0);
        assert_eq!(e.utilization().sample_count(), 0);
    }

    #[test]
    fn engine_config_accessible() {
        let cfg = MonitorConfig::default().with_history_size(42);
        let e = PerformanceMonitorEngine::new(cfg);
        assert_eq!(e.config().history_size, 42);
    }

    #[test]
    fn engine_alerts_mut_accessible() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.alerts_mut().set_enabled(false);
        assert!(!e.alerts().is_enabled());
    }

    #[test]
    fn engine_bottleneck_after_tick() {
        let mut e = PerformanceMonitorEngine::default_engine();
        e.record_utilization(95.0, 30.0, 10.0);
        e.tick();
        assert_eq!(e.bottleneck().latest().unwrap().kind, BottleneckKind::Compute);
    }

    #[test]
    fn engine_export_has_nine_metrics() {
        let mut e = PerformanceMonitorEngine::default_engine();
        let out = e.export(ExportFormat::Csv);
        // Header + 9 data rows.
        let lines: Vec<_> = out.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 10, "expected 1 header + 9 metrics");
    }

    #[test]
    fn engine_end_to_end_scenario() {
        let mut cfg = MonitorConfig::default();
        cfg.latency_alert_threshold_ms = 100.0;
        cfg.memory_alert_threshold_pct = 80.0;
        let mut e = PerformanceMonitorEngine::new(cfg);
        e.set_memory_capacity(10_000);
        e.set_device_name("cpu-reference");

        // Simulate some inference work.
        for _ in 0..10 {
            e.record_throughput(5, 1);
            e.record_latency(Duration::from_millis(20));
            e.record_allocation(100);
            e.record_utilization(50.0, 40.0, 10.0);
        }
        e.tick();

        let report = e.report();
        assert!(report.total_tokens > 0);
        assert!(report.memory_allocated > 0);
        assert!(report.avg_compute_pct > 0.0);

        // Export in all formats.
        let prom = e.export(ExportFormat::Prometheus);
        let json = e.export(ExportFormat::Json);
        let csv = e.export(ExportFormat::Csv);
        assert!(!prom.is_empty());
        assert!(!json.is_empty());
        assert!(!csv.is_empty());
    }

    // ── Cross-component integration ──────────────────────────────────

    #[test]
    fn latency_histogram_matches_count() {
        let mut l = LatencyTracker::new(1000);
        for i in 1..=50 {
            l.record(Duration::from_millis(i));
        }
        // The +Inf bucket should equal total count.
        let inf = l.histogram_buckets().get(&u64::MAX).unwrap();
        assert_eq!(*inf, 50);
    }

    #[test]
    fn memory_tracker_peak_never_decreases() {
        let mut m = MemoryTracker::new(100_000, 100);
        m.allocate(1000);
        m.allocate(2000);
        let peak1 = m.peak_allocated();
        m.deallocate(2500);
        assert_eq!(m.peak_allocated(), peak1);
    }

    #[test]
    fn alert_timestamp_nonzero() {
        let mut a = AlertManager::new(10);
        a.evaluate("m", 95.0, 90.0, true, AlertSeverity::Info);
        let alert = &a.all_alerts()[0];
        assert!(alert.timestamp_ms > 0);
    }

    #[test]
    fn report_timestamp_nonzero() {
        let e = PerformanceMonitorEngine::default_engine();
        let r = e.report();
        assert!(r.timestamp_ms > 0);
    }

    #[test]
    fn engine_tick_with_throughput_alert() {
        let mut cfg = MonitorConfig::default();
        cfg.throughput_alert_threshold = 1000.0;
        let mut e = PerformanceMonitorEngine::new(cfg);
        e.record_throughput(1, 1);
        std::thread::sleep(Duration::from_millis(10));
        e.record_throughput(1, 1);
        e.tick();
        // Throughput is well below 1000 tok/s.
        let tp_alerts: Vec<_> =
            e.alerts().all_alerts().iter().filter(|a| a.metric == "tokens_per_sec").collect();
        assert!(!tp_alerts.is_empty(), "expected throughput alert");
    }

    #[test]
    fn engine_tick_utilization_alert() {
        let mut cfg = MonitorConfig::default();
        cfg.utilization_alert_threshold_pct = 50.0;
        let mut e = PerformanceMonitorEngine::new(cfg);
        e.record_utilization(10.0, 10.0, 10.0);
        e.tick();
        let util_alerts: Vec<_> = e
            .alerts()
            .all_alerts()
            .iter()
            .filter(|a| a.metric == "compute_utilization_pct")
            .collect();
        assert!(!util_alerts.is_empty(), "expected utilisation alert");
    }
}
