//! Telemetry and metrics collection/export for OpenCL GPU kernels.
//!
//! Provides counter, gauge, histogram, and timer metrics with thread-safe
//! recording, time-windowed aggregation, Prometheus text-format export, and
//! GPU/kernel-specific metric types.  All implementations are CPU reference
//! code — no OpenCL runtime required.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

// ── MetricKind ─────────────────────────────────────────────────────

/// Discriminant for the four supported metric families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricKind {
    /// Monotonically increasing value (e.g. total dispatches).
    Counter,
    /// Point-in-time value that can go up or down (e.g. memory usage).
    Gauge,
    /// Observation bucketed into configurable ranges.
    Histogram,
    /// Duration measurement (stored as fractional seconds).
    Timer,
}

impl fmt::Display for MetricKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Counter => write!(f, "counter"),
            Self::Gauge => write!(f, "gauge"),
            Self::Histogram => write!(f, "histogram"),
            Self::Timer => write!(f, "timer"),
        }
    }
}

// ── Metric ─────────────────────────────────────────────────────────

/// A single metric observation.
#[derive(Debug, Clone)]
pub struct Metric {
    pub name: String,
    pub kind: MetricKind,
    pub value: f64,
    pub labels: BTreeMap<String, String>,
    pub timestamp: SystemTime,
}

impl Metric {
    /// Create a new metric with the current timestamp.
    pub fn new(name: &str, kind: MetricKind, value: f64) -> Self {
        Self {
            name: name.to_string(),
            kind,
            value,
            labels: BTreeMap::new(),
            timestamp: SystemTime::now(),
        }
    }

    /// Add a label key-value pair.
    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Set an explicit timestamp.
    pub fn with_timestamp(mut self, ts: SystemTime) -> Self {
        self.timestamp = ts;
        self
    }
}

// ── HistogramBucket ────────────────────────────────────────────────

/// Configurable bucket for latency distributions.
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Upper-bound thresholds (must be sorted ascending).
    pub bounds: Vec<f64>,
    /// Count of observations that fell into each bucket.
    counts: Vec<u64>,
    /// Running sum of all observed values.
    sum: f64,
    /// Total observation count.
    total_count: u64,
    min: f64,
    max: f64,
}

impl HistogramBucket {
    /// Create buckets from upper bounds (sorted ascending). An implicit `+Inf`
    /// bucket is always appended.
    pub fn new(bounds: &[f64]) -> Self {
        let mut sorted = bounds.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.dedup();
        // +1 for the implicit +Inf bucket
        let len = sorted.len() + 1;
        Self {
            bounds: sorted,
            counts: vec![0; len],
            sum: 0.0,
            total_count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Standard latency buckets (milliseconds).
    pub fn latency_defaults() -> Self {
        Self::new(&[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
    }

    /// Record one observation.
    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.total_count += 1;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        for (i, &bound) in self.bounds.iter().enumerate() {
            if value <= bound {
                self.counts[i] += 1;
                return;
            }
        }
        // Falls into +Inf bucket
        *self.counts.last_mut().unwrap() += 1;
    }

    /// Total observations.
    pub fn count(&self) -> u64 {
        self.total_count
    }

    /// Sum of all observations.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Average value, or 0.0 if empty.
    pub fn mean(&self) -> f64 {
        if self.total_count == 0 { 0.0 } else { self.sum / self.total_count as f64 }
    }

    pub fn min(&self) -> f64 {
        if self.total_count == 0 { 0.0 } else { self.min }
    }

    pub fn max(&self) -> f64 {
        if self.total_count == 0 { 0.0 } else { self.max }
    }

    /// Cumulative count up to (and including) bucket index `i`.
    pub fn cumulative_count(&self, i: usize) -> u64 {
        self.counts.iter().take(i + 1).sum()
    }

    /// Reset all counts and statistics.
    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.sum = 0.0;
        self.total_count = 0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }

    /// Return bucket bounds and their cumulative counts (including +Inf).
    pub fn buckets(&self) -> Vec<(f64, u64)> {
        let mut result = Vec::with_capacity(self.bounds.len() + 1);
        let mut cumulative = 0u64;
        for (i, &bound) in self.bounds.iter().enumerate() {
            cumulative += self.counts[i];
            result.push((bound, cumulative));
        }
        cumulative += self.counts.last().copied().unwrap_or(0);
        result.push((f64::INFINITY, cumulative));
        result
    }
}

// ── RollingWindow ──────────────────────────────────────────────────

/// Time-stamped sample for rolling-window aggregation.
#[derive(Debug, Clone)]
struct TimedSample {
    value: f64,
    timestamp: Instant,
}

/// Time-windowed metric aggregation.
#[derive(Debug, Clone)]
pub struct RollingWindow {
    window: Duration,
    samples: Vec<TimedSample>,
}

impl RollingWindow {
    /// Create a rolling window with the given duration.
    pub fn new(window: Duration) -> Self {
        Self { window, samples: Vec::new() }
    }

    /// 1-second window.
    pub fn one_second() -> Self {
        Self::new(Duration::from_secs(1))
    }

    /// 10-second window.
    pub fn ten_seconds() -> Self {
        Self::new(Duration::from_secs(10))
    }

    /// 60-second window.
    pub fn sixty_seconds() -> Self {
        Self::new(Duration::from_secs(60))
    }

    /// Add a sample with the current timestamp.
    pub fn record(&mut self, value: f64) {
        self.record_at(value, Instant::now());
    }

    /// Add a sample at a specific instant (for testing).
    pub fn record_at(&mut self, value: f64, timestamp: Instant) {
        self.samples.push(TimedSample { value, timestamp });
        self.evict(timestamp);
    }

    /// Remove samples older than the window relative to `now`.
    fn evict(&mut self, now: Instant) {
        self.samples.retain(|s| now.duration_since(s.timestamp) <= self.window);
    }

    /// Evict stale samples relative to now.
    pub fn evict_stale(&mut self) {
        self.evict(Instant::now());
    }

    /// Number of samples currently in the window.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the window is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Sum of values in the window.
    pub fn sum(&self) -> f64 {
        self.samples.iter().map(|s| s.value).sum()
    }

    /// Average value, or 0.0 if empty.
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() { 0.0 } else { self.sum() / self.samples.len() as f64 }
    }

    /// Minimum value in window.
    pub fn min(&self) -> f64 {
        self.samples.iter().map(|s| s.value).fold(f64::INFINITY, f64::min)
    }

    /// Maximum value in window.
    pub fn max(&self) -> f64 {
        self.samples.iter().map(|s| s.value).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Rate (count / window seconds).
    pub fn rate(&self) -> f64 {
        self.samples.len() as f64 / self.window.as_secs_f64()
    }

    /// Window duration.
    pub fn window_duration(&self) -> Duration {
        self.window
    }
}

// ── MetricsRegistry ────────────────────────────────────────────────

/// Internal storage for a registered metric.
#[derive(Debug)]
struct RegisteredMetric {
    kind: MetricKind,
    value: f64,
    labels: BTreeMap<String, String>,
    histogram: Option<HistogramBucket>,
}

/// Thread-safe registry for recording and querying metrics by name.
#[derive(Debug, Clone)]
pub struct MetricsRegistry {
    inner: Arc<Mutex<HashMap<String, RegisteredMetric>>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// Register a metric. If already registered, the kind must match.
    pub fn register(&self, name: &str, kind: MetricKind) -> Result<(), String> {
        let mut map = self.inner.lock().unwrap();
        if let Some(existing) = map.get(name) {
            if existing.kind != kind {
                return Err(format!(
                    "metric '{}' already registered as {}, cannot re-register as {}",
                    name, existing.kind, kind
                ));
            }
            return Ok(());
        }
        let histogram = if kind == MetricKind::Histogram {
            Some(HistogramBucket::latency_defaults())
        } else {
            None
        };
        map.insert(
            name.to_string(),
            RegisteredMetric { kind, value: 0.0, labels: BTreeMap::new(), histogram },
        );
        Ok(())
    }

    /// Register a histogram with custom bucket bounds.
    pub fn register_histogram(&self, name: &str, bounds: &[f64]) -> Result<(), String> {
        let mut map = self.inner.lock().unwrap();
        if let Some(existing) = map.get(name) {
            if existing.kind != MetricKind::Histogram {
                return Err(format!(
                    "metric '{}' already registered as {}, cannot re-register as histogram",
                    name, existing.kind
                ));
            }
            return Ok(());
        }
        map.insert(
            name.to_string(),
            RegisteredMetric {
                kind: MetricKind::Histogram,
                value: 0.0,
                labels: BTreeMap::new(),
                histogram: Some(HistogramBucket::new(bounds)),
            },
        );
        Ok(())
    }

    /// Increment a counter by `delta` (must be ≥ 0).
    pub fn increment(&self, name: &str, delta: f64) -> Result<(), String> {
        let mut map = self.inner.lock().unwrap();
        let m = map.get_mut(name).ok_or_else(|| format!("metric '{}' not registered", name))?;
        if m.kind != MetricKind::Counter {
            return Err(format!("metric '{}' is not a counter", name));
        }
        if delta < 0.0 {
            return Err("counter increment must be >= 0".to_string());
        }
        m.value += delta;
        Ok(())
    }

    /// Set a gauge to an absolute value.
    pub fn set_gauge(&self, name: &str, value: f64) -> Result<(), String> {
        let mut map = self.inner.lock().unwrap();
        let m = map.get_mut(name).ok_or_else(|| format!("metric '{}' not registered", name))?;
        if m.kind != MetricKind::Gauge {
            return Err(format!("metric '{}' is not a gauge", name));
        }
        m.value = value;
        Ok(())
    }

    /// Record a histogram observation.
    pub fn observe(&self, name: &str, value: f64) -> Result<(), String> {
        let mut map = self.inner.lock().unwrap();
        let m = map.get_mut(name).ok_or_else(|| format!("metric '{}' not registered", name))?;
        if m.kind != MetricKind::Histogram {
            return Err(format!("metric '{}' is not a histogram", name));
        }
        if let Some(h) = m.histogram.as_mut() {
            h.observe(value);
        }
        m.value = value; // last observed
        Ok(())
    }

    /// Record a timer value (fractional seconds).
    pub fn record_timer(&self, name: &str, duration: Duration) -> Result<(), String> {
        let mut map = self.inner.lock().unwrap();
        let m = map.get_mut(name).ok_or_else(|| format!("metric '{}' not registered", name))?;
        if m.kind != MetricKind::Timer {
            return Err(format!("metric '{}' is not a timer", name));
        }
        m.value = duration.as_secs_f64();
        Ok(())
    }

    /// Set labels on a metric.
    pub fn set_labels(&self, name: &str, labels: &[(&str, &str)]) -> Result<(), String> {
        let mut map = self.inner.lock().unwrap();
        let m = map.get_mut(name).ok_or_else(|| format!("metric '{}' not registered", name))?;
        m.labels.clear();
        for &(k, v) in labels {
            m.labels.insert(k.to_string(), v.to_string());
        }
        Ok(())
    }

    /// Get the current value of a metric.
    pub fn get_value(&self, name: &str) -> Option<f64> {
        let map = self.inner.lock().unwrap();
        map.get(name).map(|m| m.value)
    }

    /// Get the kind of a registered metric.
    pub fn get_kind(&self, name: &str) -> Option<MetricKind> {
        let map = self.inner.lock().unwrap();
        map.get(name).map(|m| m.kind)
    }

    /// Get a snapshot of all registered metric names.
    pub fn metric_names(&self) -> Vec<String> {
        let map = self.inner.lock().unwrap();
        map.keys().cloned().collect()
    }

    /// Reset a counter to zero.
    pub fn reset_counter(&self, name: &str) -> Result<(), String> {
        let mut map = self.inner.lock().unwrap();
        let m = map.get_mut(name).ok_or_else(|| format!("metric '{}' not registered", name))?;
        if m.kind != MetricKind::Counter {
            return Err(format!("metric '{}' is not a counter", name));
        }
        m.value = 0.0;
        Ok(())
    }

    /// Reset a histogram.
    pub fn reset_histogram(&self, name: &str) -> Result<(), String> {
        let mut map = self.inner.lock().unwrap();
        let m = map.get_mut(name).ok_or_else(|| format!("metric '{}' not registered", name))?;
        if m.kind != MetricKind::Histogram {
            return Err(format!("metric '{}' is not a histogram", name));
        }
        if let Some(h) = m.histogram.as_mut() {
            h.reset();
        }
        m.value = 0.0;
        Ok(())
    }

    /// Get histogram data for a metric.
    pub fn get_histogram(&self, name: &str) -> Option<HistogramBucket> {
        let map = self.inner.lock().unwrap();
        map.get(name).and_then(|m| m.histogram.clone())
    }

    /// Snapshot all metrics as `Metric` values.
    pub fn snapshot(&self) -> Vec<Metric> {
        let map = self.inner.lock().unwrap();
        map.iter()
            .map(|(name, rm)| {
                let mut m = Metric::new(name, rm.kind, rm.value);
                m.labels = rm.labels.clone();
                m
            })
            .collect()
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── MetricsExporter (Prometheus text format) ───────────────────────

/// Serializes metrics to the Prometheus exposition text format.
pub struct MetricsExporter;

impl MetricsExporter {
    /// Export all metrics in a registry to Prometheus text format.
    pub fn export_prometheus(registry: &MetricsRegistry) -> String {
        let map = registry.inner.lock().unwrap();
        let mut output = String::new();

        // Sort by name for deterministic output
        let mut entries: Vec<_> = map.iter().collect();
        entries.sort_by_key(|(name, _)| (*name).clone());

        for (name, rm) in &entries {
            let prom_type = match rm.kind {
                MetricKind::Counter => "counter",
                MetricKind::Gauge | MetricKind::Timer => "gauge",
                MetricKind::Histogram => "histogram",
            };
            output.push_str(&format!("# HELP {} {}\n", name, name));
            output.push_str(&format!("# TYPE {} {}\n", name, prom_type));

            let label_str = Self::format_labels(&rm.labels);

            match rm.kind {
                MetricKind::Histogram => {
                    if let Some(h) = &rm.histogram {
                        for (bound, cumulative) in h.buckets() {
                            let le = if bound.is_infinite() {
                                "+Inf".to_string()
                            } else {
                                format!("{}", bound)
                            };
                            if rm.labels.is_empty() {
                                output.push_str(&format!(
                                    "{}_bucket{{le=\"{}\"}} {}\n",
                                    name, le, cumulative
                                ));
                            } else {
                                // Strip outer braces from label_str, merge
                                let inner = &label_str[1..label_str.len() - 1];
                                output.push_str(&format!(
                                    "{}_bucket{{{},le=\"{}\"}} {}\n",
                                    name, inner, le, cumulative
                                ));
                            }
                        }
                        output.push_str(&format!("{}_sum{} {}\n", name, label_str, h.sum()));
                        output.push_str(&format!("{}_count{} {}\n", name, label_str, h.count()));
                    }
                }
                _ => {
                    output.push_str(&format!("{}{} {}\n", name, label_str, rm.value));
                }
            }
        }
        output
    }

    fn format_labels(labels: &BTreeMap<String, String>) -> String {
        if labels.is_empty() {
            String::new()
        } else {
            let inner: Vec<String> =
                labels.iter().map(|(k, v)| format!("{}=\"{}\"", k, v)).collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

// ── GpuMetrics ─────────────────────────────────────────────────────

/// GPU-specific metrics (CPU reference — reports simulated values).
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// GPU utilisation percentage (0–100).
    pub utilization: f64,
    /// Used GPU memory in bytes.
    pub memory_used: u64,
    /// Total GPU memory in bytes.
    pub memory_total: u64,
    /// Temperature in degrees Celsius.
    pub temperature: f64,
    /// Core clock speed in MHz.
    pub clock_mhz: u32,
    /// Memory clock speed in MHz.
    pub memory_clock_mhz: u32,
    /// Power draw in watts.
    pub power_watts: f64,
    /// Device name.
    pub device_name: String,
    /// Timestamp of the reading.
    pub timestamp: SystemTime,
}

impl GpuMetrics {
    /// Create idle / zero-initialised GPU metrics.
    pub fn idle(device_name: &str) -> Self {
        Self {
            utilization: 0.0,
            memory_used: 0,
            memory_total: 16 * 1024 * 1024 * 1024, // 16 GiB default (A770)
            temperature: 35.0,
            clock_mhz: 0,
            memory_clock_mhz: 0,
            power_watts: 0.0,
            device_name: device_name.to_string(),
            timestamp: SystemTime::now(),
        }
    }

    /// Simulated busy snapshot for testing.
    pub fn simulated(device_name: &str) -> Self {
        Self {
            utilization: 85.0,
            memory_used: 4 * 1024 * 1024 * 1024,
            memory_total: 16 * 1024 * 1024 * 1024,
            temperature: 72.0,
            clock_mhz: 2100,
            memory_clock_mhz: 8750,
            power_watts: 180.0,
            device_name: device_name.to_string(),
            timestamp: SystemTime::now(),
        }
    }

    /// Memory utilisation fraction (0.0–1.0).
    pub fn memory_utilization(&self) -> f64 {
        if self.memory_total == 0 {
            0.0
        } else {
            self.memory_used as f64 / self.memory_total as f64
        }
    }

    /// Whether temperature exceeds the thermal throttle threshold.
    pub fn is_throttling(&self, threshold: f64) -> bool {
        self.temperature >= threshold
    }

    /// Publish all fields into a `MetricsRegistry`.
    pub fn publish(&self, registry: &MetricsRegistry, prefix: &str) -> Result<(), String> {
        let names = [
            ("utilization", self.utilization),
            ("temperature", self.temperature),
            ("clock_mhz", self.clock_mhz as f64),
            ("memory_clock_mhz", self.memory_clock_mhz as f64),
            ("power_watts", self.power_watts),
            ("memory_used_bytes", self.memory_used as f64),
            ("memory_total_bytes", self.memory_total as f64),
            ("memory_utilization", self.memory_utilization()),
        ];
        for (suffix, val) in &names {
            let name = format!("{}_{}", prefix, suffix);
            registry.register(&name, MetricKind::Gauge)?;
            registry.set_gauge(&name, *val)?;
            registry.set_labels(&name, &[("device", &self.device_name)])?;
        }
        Ok(())
    }
}

impl Default for GpuMetrics {
    fn default() -> Self {
        Self::idle("unknown")
    }
}

// ── KernelMetrics ──────────────────────────────────────────────────

/// Per-kernel execution metrics.
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    pub kernel_name: String,
    pub dispatch_count: u64,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub total_flops: u64,
}

impl KernelMetrics {
    pub fn new(name: &str) -> Self {
        Self {
            kernel_name: name.to_string(),
            dispatch_count: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            total_flops: 0,
        }
    }

    /// Record one kernel dispatch.
    pub fn record_dispatch(&mut self, elapsed: Duration, flops: u64) {
        self.dispatch_count += 1;
        self.total_time += elapsed;
        if elapsed < self.min_time {
            self.min_time = elapsed;
        }
        if elapsed > self.max_time {
            self.max_time = elapsed;
        }
        self.total_flops += flops;
    }

    /// Average dispatch duration, or zero.
    pub fn avg_time(&self) -> Duration {
        if self.dispatch_count == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.dispatch_count as u32
        }
    }

    /// Achieved GFLOP/s over all dispatches.
    pub fn gflops(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs == 0.0 { 0.0 } else { self.total_flops as f64 / secs / 1e9 }
    }

    /// Reset all stats.
    pub fn reset(&mut self) {
        self.dispatch_count = 0;
        self.total_time = Duration::ZERO;
        self.min_time = Duration::MAX;
        self.max_time = Duration::ZERO;
        self.total_flops = 0;
    }

    /// Publish into a `MetricsRegistry`.
    pub fn publish(&self, registry: &MetricsRegistry, prefix: &str) -> Result<(), String> {
        let counter_name = format!("{}_{}_dispatches", prefix, self.kernel_name);
        registry.register(&counter_name, MetricKind::Counter)?;
        // Counters can only increment — reset then increment to desired value
        let _ = registry.reset_counter(&counter_name);
        registry.increment(&counter_name, self.dispatch_count as f64)?;

        let time_name = format!("{}_{}_total_seconds", prefix, self.kernel_name);
        registry.register(&time_name, MetricKind::Gauge)?;
        registry.set_gauge(&time_name, self.total_time.as_secs_f64())?;

        let avg_name = format!("{}_{}_avg_seconds", prefix, self.kernel_name);
        registry.register(&avg_name, MetricKind::Gauge)?;
        registry.set_gauge(&avg_name, self.avg_time().as_secs_f64())?;

        let gflops_name = format!("{}_{}_gflops", prefix, self.kernel_name);
        registry.register(&gflops_name, MetricKind::Gauge)?;
        registry.set_gauge(&gflops_name, self.gflops())?;

        Ok(())
    }
}

/// Aggregator that tracks metrics for multiple kernels.
#[derive(Debug, Clone, Default)]
pub struct KernelMetricsAggregator {
    kernels: HashMap<String, KernelMetrics>,
}

impl KernelMetricsAggregator {
    pub fn new() -> Self {
        Self { kernels: HashMap::new() }
    }

    /// Record a dispatch for the named kernel.
    pub fn record(&mut self, name: &str, elapsed: Duration, flops: u64) {
        self.kernels
            .entry(name.to_string())
            .or_insert_with(|| KernelMetrics::new(name))
            .record_dispatch(elapsed, flops);
    }

    /// Get metrics for a specific kernel.
    pub fn get(&self, name: &str) -> Option<&KernelMetrics> {
        self.kernels.get(name)
    }

    /// Iterate all tracked kernels.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &KernelMetrics)> {
        self.kernels.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Total dispatches across all kernels.
    pub fn total_dispatches(&self) -> u64 {
        self.kernels.values().map(|k| k.dispatch_count).sum()
    }

    /// Total time across all kernels.
    pub fn total_time(&self) -> Duration {
        self.kernels.values().map(|k| k.total_time).sum()
    }

    /// Number of tracked kernels.
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// Reset all kernel metrics.
    pub fn reset_all(&mut self) {
        for km in self.kernels.values_mut() {
            km.reset();
        }
    }

    /// Publish all kernels into a registry.
    pub fn publish_all(&self, registry: &MetricsRegistry, prefix: &str) -> Result<(), String> {
        for km in self.kernels.values() {
            km.publish(registry, prefix)?;
        }
        Ok(())
    }
}

// ── TelemetryConfig ────────────────────────────────────────────────

/// Configuration for the telemetry subsystem.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// How often metrics are exported.
    pub export_interval: Duration,
    /// How long to retain metric history.
    pub retention: Duration,
    /// Which metric kinds are enabled.
    pub enabled_kinds: Vec<MetricKind>,
    /// Maximum number of metrics to track.
    pub max_metrics: usize,
    /// Whether Prometheus export is enabled.
    pub prometheus_enabled: bool,
    /// Optional prefix for all metric names.
    pub metric_prefix: String,
}

impl TelemetryConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.export_interval.is_zero() {
            return Err("export_interval must be > 0".to_string());
        }
        if self.retention.is_zero() {
            return Err("retention must be > 0".to_string());
        }
        if self.retention < self.export_interval {
            return Err("retention must be >= export_interval".to_string());
        }
        if self.enabled_kinds.is_empty() {
            return Err("at least one metric kind must be enabled".to_string());
        }
        if self.max_metrics == 0 {
            return Err("max_metrics must be > 0".to_string());
        }
        Ok(())
    }

    /// Whether a particular metric kind is enabled.
    pub fn is_kind_enabled(&self, kind: MetricKind) -> bool {
        self.enabled_kinds.contains(&kind)
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            export_interval: Duration::from_secs(15),
            retention: Duration::from_secs(3600),
            enabled_kinds: vec![
                MetricKind::Counter,
                MetricKind::Gauge,
                MetricKind::Histogram,
                MetricKind::Timer,
            ],
            max_metrics: 10_000,
            prometheus_enabled: true,
            metric_prefix: "bitnet_opencl".to_string(),
        }
    }
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::{Duration, Instant};

    // ── MetricKind ─────────────────────────────────────────────────

    #[test]
    fn metric_kind_display() {
        assert_eq!(MetricKind::Counter.to_string(), "counter");
        assert_eq!(MetricKind::Gauge.to_string(), "gauge");
        assert_eq!(MetricKind::Histogram.to_string(), "histogram");
        assert_eq!(MetricKind::Timer.to_string(), "timer");
    }

    #[test]
    fn metric_kind_equality() {
        assert_eq!(MetricKind::Counter, MetricKind::Counter);
        assert_ne!(MetricKind::Counter, MetricKind::Gauge);
    }

    #[test]
    fn metric_kind_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MetricKind::Counter);
        set.insert(MetricKind::Counter);
        assert_eq!(set.len(), 1);
        set.insert(MetricKind::Gauge);
        assert_eq!(set.len(), 2);
    }

    // ── Metric ─────────────────────────────────────────────────────

    #[test]
    fn metric_new_defaults() {
        let m = Metric::new("test", MetricKind::Counter, 42.0);
        assert_eq!(m.name, "test");
        assert_eq!(m.kind, MetricKind::Counter);
        assert!((m.value - 42.0).abs() < f64::EPSILON);
        assert!(m.labels.is_empty());
    }

    #[test]
    fn metric_with_labels() {
        let m = Metric::new("x", MetricKind::Gauge, 1.0)
            .with_label("device", "A770")
            .with_label("kernel", "matmul");
        assert_eq!(m.labels.len(), 2);
        assert_eq!(m.labels["device"], "A770");
        assert_eq!(m.labels["kernel"], "matmul");
    }

    #[test]
    fn metric_with_timestamp() {
        let ts = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let m = Metric::new("x", MetricKind::Timer, 0.5).with_timestamp(ts);
        assert_eq!(m.timestamp, ts);
    }

    #[test]
    fn metric_label_ordering() {
        let m = Metric::new("x", MetricKind::Gauge, 0.0)
            .with_label("z", "3")
            .with_label("a", "1")
            .with_label("m", "2");
        let keys: Vec<&String> = m.labels.keys().collect();
        assert_eq!(keys, vec!["a", "m", "z"]); // BTreeMap is sorted
    }

    // ── Counter ────────────────────────────────────────────────────

    #[test]
    fn counter_increment() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();
        reg.increment("c", 1.0).unwrap();
        reg.increment("c", 2.0).unwrap();
        assert!((reg.get_value("c").unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn counter_increment_zero() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();
        reg.increment("c", 0.0).unwrap();
        assert!((reg.get_value("c").unwrap()).abs() < f64::EPSILON);
    }

    #[test]
    fn counter_negative_increment_rejected() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();
        assert!(reg.increment("c", -1.0).is_err());
    }

    #[test]
    fn counter_reset() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();
        reg.increment("c", 10.0).unwrap();
        reg.reset_counter("c").unwrap();
        assert!((reg.get_value("c").unwrap()).abs() < f64::EPSILON);
    }

    #[test]
    fn counter_reset_nonexistent() {
        let reg = MetricsRegistry::new();
        assert!(reg.reset_counter("nope").is_err());
    }

    #[test]
    fn counter_increment_nonexistent() {
        let reg = MetricsRegistry::new();
        assert!(reg.increment("nope", 1.0).is_err());
    }

    #[test]
    fn counter_monotonicity() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();
        let mut prev = 0.0f64;
        for i in 1..=100 {
            reg.increment("c", i as f64).unwrap();
            let cur = reg.get_value("c").unwrap();
            assert!(cur >= prev, "counter must be monotonically non-decreasing");
            prev = cur;
        }
    }

    // ── Gauge ──────────────────────────────────────────────────────

    #[test]
    fn gauge_set_get() {
        let reg = MetricsRegistry::new();
        reg.register("g", MetricKind::Gauge).unwrap();
        reg.set_gauge("g", 42.0).unwrap();
        assert!((reg.get_value("g").unwrap() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gauge_overwrite() {
        let reg = MetricsRegistry::new();
        reg.register("g", MetricKind::Gauge).unwrap();
        reg.set_gauge("g", 1.0).unwrap();
        reg.set_gauge("g", -5.0).unwrap();
        assert!((reg.get_value("g").unwrap() - -5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gauge_nonexistent() {
        let reg = MetricsRegistry::new();
        assert!(reg.set_gauge("nope", 1.0).is_err());
    }

    #[test]
    fn gauge_wrong_kind() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();
        assert!(reg.set_gauge("c", 1.0).is_err());
    }

    // ── Histogram ──────────────────────────────────────────────────

    #[test]
    fn histogram_observe_and_count() {
        let mut h = HistogramBucket::new(&[1.0, 5.0, 10.0]);
        h.observe(0.5);
        h.observe(3.0);
        h.observe(7.0);
        h.observe(15.0);
        assert_eq!(h.count(), 4);
    }

    #[test]
    fn histogram_sum() {
        let mut h = HistogramBucket::new(&[10.0]);
        h.observe(3.0);
        h.observe(7.0);
        assert!((h.sum() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_mean() {
        let mut h = HistogramBucket::new(&[100.0]);
        h.observe(2.0);
        h.observe(4.0);
        h.observe(6.0);
        assert!((h.mean() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_empty_mean() {
        let h = HistogramBucket::new(&[1.0]);
        assert!((h.mean()).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_bucket_distribution() {
        let mut h = HistogramBucket::new(&[1.0, 5.0, 10.0]);
        // 2 in [0,1], 1 in (1,5], 1 in (5,10], 1 in (10,+Inf)
        h.observe(0.5);
        h.observe(0.8);
        h.observe(3.0);
        h.observe(7.0);
        h.observe(20.0);
        let buckets = h.buckets();
        // cumulative: 1.0 → 2, 5.0 → 3, 10.0 → 4, +Inf → 5
        assert_eq!(buckets[0], (1.0, 2));
        assert_eq!(buckets[1], (5.0, 3));
        assert_eq!(buckets[2], (10.0, 4));
        assert!(buckets[3].0.is_infinite());
        assert_eq!(buckets[3].1, 5);
    }

    #[test]
    fn histogram_min_max() {
        let mut h = HistogramBucket::new(&[100.0]);
        h.observe(5.0);
        h.observe(1.0);
        h.observe(9.0);
        assert!((h.min() - 1.0).abs() < f64::EPSILON);
        assert!((h.max() - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_empty_min_max() {
        let h = HistogramBucket::new(&[1.0]);
        assert!((h.min()).abs() < f64::EPSILON);
        assert!((h.max()).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_reset() {
        let mut h = HistogramBucket::new(&[10.0]);
        h.observe(5.0);
        h.observe(15.0);
        h.reset();
        assert_eq!(h.count(), 0);
        assert!((h.sum()).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_cumulative_count() {
        let mut h = HistogramBucket::new(&[1.0, 5.0, 10.0]);
        h.observe(0.5);
        h.observe(3.0);
        h.observe(7.0);
        // bucket 0 (≤1): 1, bucket 1 (≤5): 1, bucket 2 (≤10): 1
        assert_eq!(h.cumulative_count(0), 1);
        assert_eq!(h.cumulative_count(1), 2);
        assert_eq!(h.cumulative_count(2), 3);
    }

    #[test]
    fn histogram_latency_defaults() {
        let h = HistogramBucket::latency_defaults();
        assert_eq!(h.bounds.len(), 12);
        assert!((h.bounds[0] - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_dedup_bounds() {
        let h = HistogramBucket::new(&[1.0, 1.0, 5.0, 5.0, 10.0]);
        assert_eq!(h.bounds.len(), 3);
    }

    #[test]
    fn histogram_registry_observe() {
        let reg = MetricsRegistry::new();
        reg.register("h", MetricKind::Histogram).unwrap();
        reg.observe("h", 1.0).unwrap();
        reg.observe("h", 5.0).unwrap();
        let h = reg.get_histogram("h").unwrap();
        assert_eq!(h.count(), 2);
    }

    #[test]
    fn histogram_registry_custom_bounds() {
        let reg = MetricsRegistry::new();
        reg.register_histogram("h", &[1.0, 10.0, 100.0]).unwrap();
        reg.observe("h", 50.0).unwrap();
        let h = reg.get_histogram("h").unwrap();
        assert_eq!(h.count(), 1);
        assert_eq!(h.bounds.len(), 3);
    }

    #[test]
    fn histogram_registry_reset() {
        let reg = MetricsRegistry::new();
        reg.register("h", MetricKind::Histogram).unwrap();
        reg.observe("h", 5.0).unwrap();
        reg.reset_histogram("h").unwrap();
        let h = reg.get_histogram("h").unwrap();
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn histogram_sum_equals_count_times_avg() {
        let mut h = HistogramBucket::new(&[100.0]);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            h.observe(v);
        }
        let expected = h.count() as f64 * h.mean();
        assert!((h.sum() - expected).abs() < 1e-10);
    }

    // ── Rolling window ─────────────────────────────────────────────

    #[test]
    fn rolling_window_record_and_len() {
        let mut w = RollingWindow::new(Duration::from_secs(60));
        let now = Instant::now();
        w.record_at(1.0, now);
        w.record_at(2.0, now);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn rolling_window_sum() {
        let mut w = RollingWindow::new(Duration::from_secs(60));
        let now = Instant::now();
        w.record_at(3.0, now);
        w.record_at(7.0, now);
        assert!((w.sum() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rolling_window_mean() {
        let mut w = RollingWindow::new(Duration::from_secs(60));
        let now = Instant::now();
        w.record_at(2.0, now);
        w.record_at(4.0, now);
        assert!((w.mean() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rolling_window_empty_mean() {
        let w = RollingWindow::new(Duration::from_secs(1));
        assert!((w.mean()).abs() < f64::EPSILON);
    }

    #[test]
    fn rolling_window_eviction() {
        let mut w = RollingWindow::new(Duration::from_secs(2));
        let base = Instant::now();
        w.record_at(1.0, base);
        w.record_at(2.0, base + Duration::from_secs(1));
        // 3 seconds later — first sample should be evicted
        w.record_at(3.0, base + Duration::from_secs(3));
        // Only the last two should remain
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn rolling_window_all_evicted() {
        let mut w = RollingWindow::new(Duration::from_millis(100));
        let base = Instant::now();
        w.record_at(1.0, base);
        w.record_at(2.0, base);
        // 1 second later — everything evicted
        w.record_at(3.0, base + Duration::from_secs(1));
        assert_eq!(w.len(), 1);
        assert!((w.sum() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rolling_window_min_max() {
        let mut w = RollingWindow::new(Duration::from_secs(60));
        let now = Instant::now();
        w.record_at(5.0, now);
        w.record_at(1.0, now);
        w.record_at(9.0, now);
        assert!((w.min() - 1.0).abs() < f64::EPSILON);
        assert!((w.max() - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rolling_window_rate() {
        let mut w = RollingWindow::new(Duration::from_secs(10));
        let now = Instant::now();
        for i in 0..20 {
            w.record_at(1.0, now + Duration::from_millis(i * 100));
        }
        // 20 samples in a 10-second window → rate = 2.0/s
        assert!((w.rate() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rolling_window_one_second() {
        let w = RollingWindow::one_second();
        assert_eq!(w.window_duration(), Duration::from_secs(1));
    }

    #[test]
    fn rolling_window_ten_seconds() {
        let w = RollingWindow::ten_seconds();
        assert_eq!(w.window_duration(), Duration::from_secs(10));
    }

    #[test]
    fn rolling_window_sixty_seconds() {
        let w = RollingWindow::sixty_seconds();
        assert_eq!(w.window_duration(), Duration::from_secs(60));
    }

    #[test]
    fn rolling_window_is_empty() {
        let w = RollingWindow::new(Duration::from_secs(1));
        assert!(w.is_empty());
    }

    // ── Registry ───────────────────────────────────────────────────

    #[test]
    fn registry_register_duplicate_same_kind() {
        let reg = MetricsRegistry::new();
        reg.register("x", MetricKind::Counter).unwrap();
        reg.register("x", MetricKind::Counter).unwrap(); // idempotent
    }

    #[test]
    fn registry_register_duplicate_different_kind() {
        let reg = MetricsRegistry::new();
        reg.register("x", MetricKind::Counter).unwrap();
        assert!(reg.register("x", MetricKind::Gauge).is_err());
    }

    #[test]
    fn registry_get_nonexistent() {
        let reg = MetricsRegistry::new();
        assert!(reg.get_value("nope").is_none());
    }

    #[test]
    fn registry_get_kind() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();
        assert_eq!(reg.get_kind("c"), Some(MetricKind::Counter));
        assert_eq!(reg.get_kind("nope"), None);
    }

    #[test]
    fn registry_metric_names() {
        let reg = MetricsRegistry::new();
        reg.register("a", MetricKind::Counter).unwrap();
        reg.register("b", MetricKind::Gauge).unwrap();
        let mut names = reg.metric_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn registry_set_labels() {
        let reg = MetricsRegistry::new();
        reg.register("g", MetricKind::Gauge).unwrap();
        reg.set_labels("g", &[("device", "A770"), ("backend", "opencl")]).unwrap();
        let snap = reg.snapshot();
        let m = snap.iter().find(|m| m.name == "g").unwrap();
        assert_eq!(m.labels["device"], "A770");
    }

    #[test]
    fn registry_timer() {
        let reg = MetricsRegistry::new();
        reg.register("t", MetricKind::Timer).unwrap();
        reg.record_timer("t", Duration::from_millis(150)).unwrap();
        let v = reg.get_value("t").unwrap();
        assert!((v - 0.15).abs() < 1e-6);
    }

    #[test]
    fn registry_timer_wrong_kind() {
        let reg = MetricsRegistry::new();
        reg.register("g", MetricKind::Gauge).unwrap();
        assert!(reg.record_timer("g", Duration::from_secs(1)).is_err());
    }

    #[test]
    fn registry_snapshot() {
        let reg = MetricsRegistry::new();
        reg.register("a", MetricKind::Counter).unwrap();
        reg.register("b", MetricKind::Gauge).unwrap();
        reg.increment("a", 5.0).unwrap();
        reg.set_gauge("b", -3.0).unwrap();
        let snap = reg.snapshot();
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn registry_default() {
        let reg = MetricsRegistry::default();
        assert!(reg.metric_names().is_empty());
    }

    #[test]
    fn registry_observe_wrong_kind() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();
        assert!(reg.observe("c", 1.0).is_err());
    }

    #[test]
    fn registry_increment_wrong_kind() {
        let reg = MetricsRegistry::new();
        reg.register("g", MetricKind::Gauge).unwrap();
        assert!(reg.increment("g", 1.0).is_err());
    }

    // ── Thread-safe concurrent recording ───────────────────────────

    #[test]
    fn registry_concurrent_increments() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let r = reg.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        r.increment("c", 1.0).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        let v = reg.get_value("c").unwrap();
        assert!((v - 800.0).abs() < f64::EPSILON);
    }

    #[test]
    fn registry_concurrent_gauges() {
        let reg = MetricsRegistry::new();
        reg.register("g", MetricKind::Gauge).unwrap();

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let r = reg.clone();
                thread::spawn(move || {
                    for _ in 0..50 {
                        r.set_gauge("g", i as f64).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        // Value is one of 0, 1, 2, 3
        let v = reg.get_value("g").unwrap();
        assert!((0.0..=3.0).contains(&v));
    }

    // ── Prometheus export ──────────────────────────────────────────

    #[test]
    fn prometheus_counter_export() {
        let reg = MetricsRegistry::new();
        reg.register("requests_total", MetricKind::Counter).unwrap();
        reg.increment("requests_total", 42.0).unwrap();
        let out = MetricsExporter::export_prometheus(&reg);
        assert!(out.contains("# TYPE requests_total counter"));
        assert!(out.contains("requests_total 42"));
    }

    #[test]
    fn prometheus_gauge_export() {
        let reg = MetricsRegistry::new();
        reg.register("temperature", MetricKind::Gauge).unwrap();
        reg.set_gauge("temperature", 72.5).unwrap();
        let out = MetricsExporter::export_prometheus(&reg);
        assert!(out.contains("# TYPE temperature gauge"));
        assert!(out.contains("temperature 72.5"));
    }

    #[test]
    fn prometheus_histogram_export() {
        let reg = MetricsRegistry::new();
        reg.register_histogram("latency", &[1.0, 5.0, 10.0]).unwrap();
        reg.observe("latency", 3.0).unwrap();
        reg.observe("latency", 7.0).unwrap();
        let out = MetricsExporter::export_prometheus(&reg);
        assert!(out.contains("# TYPE latency histogram"));
        assert!(out.contains("latency_bucket{le=\"1\"} 0"));
        assert!(out.contains("latency_bucket{le=\"5\"} 1"));
        assert!(out.contains("latency_bucket{le=\"10\"} 2"));
        assert!(out.contains("latency_bucket{le=\"+Inf\"} 2"));
        assert!(out.contains("latency_sum 10"));
        assert!(out.contains("latency_count 2"));
    }

    #[test]
    fn prometheus_with_labels() {
        let reg = MetricsRegistry::new();
        reg.register("mem", MetricKind::Gauge).unwrap();
        reg.set_gauge("mem", 1024.0).unwrap();
        reg.set_labels("mem", &[("device", "A770")]).unwrap();
        let out = MetricsExporter::export_prometheus(&reg);
        assert!(out.contains("mem{device=\"A770\"} 1024"));
    }

    #[test]
    fn prometheus_empty_registry() {
        let reg = MetricsRegistry::new();
        let out = MetricsExporter::export_prometheus(&reg);
        assert!(out.is_empty());
    }

    #[test]
    fn prometheus_timer_exported_as_gauge() {
        let reg = MetricsRegistry::new();
        reg.register("dur", MetricKind::Timer).unwrap();
        reg.record_timer("dur", Duration::from_millis(250)).unwrap();
        let out = MetricsExporter::export_prometheus(&reg);
        assert!(out.contains("# TYPE dur gauge"));
        assert!(out.contains("dur 0.25"));
    }

    #[test]
    fn prometheus_deterministic_ordering() {
        let reg = MetricsRegistry::new();
        reg.register("zzz", MetricKind::Counter).unwrap();
        reg.register("aaa", MetricKind::Counter).unwrap();
        let out = MetricsExporter::export_prometheus(&reg);
        let aaa_pos = out.find("aaa").unwrap();
        let zzz_pos = out.find("zzz").unwrap();
        assert!(aaa_pos < zzz_pos, "output must be sorted by name");
    }

    // ── GpuMetrics ─────────────────────────────────────────────────

    #[test]
    fn gpu_metrics_idle() {
        let gm = GpuMetrics::idle("A770");
        assert!((gm.utilization).abs() < f64::EPSILON);
        assert_eq!(gm.memory_used, 0);
        assert_eq!(gm.device_name, "A770");
    }

    #[test]
    fn gpu_metrics_simulated() {
        let gm = GpuMetrics::simulated("A770");
        assert!((gm.utilization - 85.0).abs() < f64::EPSILON);
        assert!(gm.memory_used > 0);
        assert_eq!(gm.clock_mhz, 2100);
    }

    #[test]
    fn gpu_memory_utilization() {
        let mut gm = GpuMetrics::idle("A770");
        gm.memory_used = 8 * 1024 * 1024 * 1024;
        gm.memory_total = 16 * 1024 * 1024 * 1024;
        assert!((gm.memory_utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn gpu_memory_utilization_zero_total() {
        let mut gm = GpuMetrics::idle("test");
        gm.memory_total = 0;
        assert!((gm.memory_utilization()).abs() < f64::EPSILON);
    }

    #[test]
    fn gpu_is_throttling() {
        let mut gm = GpuMetrics::idle("A770");
        gm.temperature = 90.0;
        assert!(gm.is_throttling(85.0));
        assert!(!gm.is_throttling(95.0));
    }

    #[test]
    fn gpu_metrics_publish() {
        let reg = MetricsRegistry::new();
        let gm = GpuMetrics::simulated("A770");
        gm.publish(&reg, "gpu").unwrap();
        assert!(reg.get_value("gpu_utilization").is_some());
        assert!(reg.get_value("gpu_temperature").is_some());
        assert!(reg.get_value("gpu_memory_used_bytes").is_some());
        assert!(reg.get_value("gpu_power_watts").is_some());
    }

    #[test]
    fn gpu_metrics_default() {
        let gm = GpuMetrics::default();
        assert_eq!(gm.device_name, "unknown");
    }

    // ── KernelMetrics ──────────────────────────────────────────────

    #[test]
    fn kernel_metrics_new() {
        let km = KernelMetrics::new("matmul");
        assert_eq!(km.kernel_name, "matmul");
        assert_eq!(km.dispatch_count, 0);
    }

    #[test]
    fn kernel_metrics_record_dispatch() {
        let mut km = KernelMetrics::new("matmul");
        km.record_dispatch(Duration::from_millis(10), 1_000_000);
        km.record_dispatch(Duration::from_millis(20), 2_000_000);
        assert_eq!(km.dispatch_count, 2);
        assert_eq!(km.total_time, Duration::from_millis(30));
        assert_eq!(km.total_flops, 3_000_000);
    }

    #[test]
    fn kernel_metrics_avg_time() {
        let mut km = KernelMetrics::new("softmax");
        km.record_dispatch(Duration::from_millis(10), 0);
        km.record_dispatch(Duration::from_millis(30), 0);
        assert_eq!(km.avg_time(), Duration::from_millis(20));
    }

    #[test]
    fn kernel_metrics_avg_time_empty() {
        let km = KernelMetrics::new("empty");
        assert_eq!(km.avg_time(), Duration::ZERO);
    }

    #[test]
    fn kernel_metrics_min_max_time() {
        let mut km = KernelMetrics::new("rmsnorm");
        km.record_dispatch(Duration::from_millis(5), 0);
        km.record_dispatch(Duration::from_millis(15), 0);
        km.record_dispatch(Duration::from_millis(10), 0);
        assert_eq!(km.min_time, Duration::from_millis(5));
        assert_eq!(km.max_time, Duration::from_millis(15));
    }

    #[test]
    fn kernel_metrics_gflops() {
        let mut km = KernelMetrics::new("matmul");
        // 1 billion flops in 1 second = 1 GFLOP/s
        km.record_dispatch(Duration::from_secs(1), 1_000_000_000);
        assert!((km.gflops() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn kernel_metrics_gflops_empty() {
        let km = KernelMetrics::new("empty");
        assert!((km.gflops()).abs() < f64::EPSILON);
    }

    #[test]
    fn kernel_metrics_reset() {
        let mut km = KernelMetrics::new("matmul");
        km.record_dispatch(Duration::from_millis(10), 100);
        km.reset();
        assert_eq!(km.dispatch_count, 0);
        assert_eq!(km.total_time, Duration::ZERO);
        assert_eq!(km.total_flops, 0);
    }

    #[test]
    fn kernel_metrics_publish() {
        let reg = MetricsRegistry::new();
        let mut km = KernelMetrics::new("matmul");
        km.record_dispatch(Duration::from_millis(10), 1_000_000);
        km.publish(&reg, "kernel").unwrap();
        assert!(reg.get_value("kernel_matmul_dispatches").is_some());
        assert!(reg.get_value("kernel_matmul_total_seconds").is_some());
        assert!(reg.get_value("kernel_matmul_avg_seconds").is_some());
        assert!(reg.get_value("kernel_matmul_gflops").is_some());
    }

    // ── KernelMetricsAggregator ────────────────────────────────────

    #[test]
    fn aggregator_record_and_get() {
        let mut agg = KernelMetricsAggregator::new();
        agg.record("matmul", Duration::from_millis(10), 100);
        agg.record("softmax", Duration::from_millis(5), 50);
        assert_eq!(agg.kernel_count(), 2);
        assert_eq!(agg.get("matmul").unwrap().dispatch_count, 1);
    }

    #[test]
    fn aggregator_total_dispatches() {
        let mut agg = KernelMetricsAggregator::new();
        agg.record("a", Duration::from_millis(1), 0);
        agg.record("a", Duration::from_millis(1), 0);
        agg.record("b", Duration::from_millis(1), 0);
        assert_eq!(agg.total_dispatches(), 3);
    }

    #[test]
    fn aggregator_total_time() {
        let mut agg = KernelMetricsAggregator::new();
        agg.record("a", Duration::from_millis(10), 0);
        agg.record("b", Duration::from_millis(20), 0);
        assert_eq!(agg.total_time(), Duration::from_millis(30));
    }

    #[test]
    fn aggregator_reset_all() {
        let mut agg = KernelMetricsAggregator::new();
        agg.record("a", Duration::from_millis(10), 100);
        agg.reset_all();
        assert_eq!(agg.get("a").unwrap().dispatch_count, 0);
    }

    #[test]
    fn aggregator_publish_all() {
        let reg = MetricsRegistry::new();
        let mut agg = KernelMetricsAggregator::new();
        agg.record("matmul", Duration::from_millis(10), 100);
        agg.record("softmax", Duration::from_millis(5), 50);
        agg.publish_all(&reg, "k").unwrap();
        assert!(reg.get_value("k_matmul_dispatches").is_some());
        assert!(reg.get_value("k_softmax_dispatches").is_some());
    }

    #[test]
    fn aggregator_default() {
        let agg = KernelMetricsAggregator::default();
        assert_eq!(agg.kernel_count(), 0);
    }

    // ── TelemetryConfig ────────────────────────────────────────────

    #[test]
    fn config_default_valid() {
        let cfg = TelemetryConfig::default();
        cfg.validate().unwrap();
    }

    #[test]
    fn config_zero_export_interval() {
        let mut cfg = TelemetryConfig::default();
        cfg.export_interval = Duration::ZERO;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_retention() {
        let mut cfg = TelemetryConfig::default();
        cfg.retention = Duration::ZERO;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_retention_less_than_export() {
        let mut cfg = TelemetryConfig::default();
        cfg.export_interval = Duration::from_secs(60);
        cfg.retention = Duration::from_secs(30);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_no_enabled_kinds() {
        let mut cfg = TelemetryConfig::default();
        cfg.enabled_kinds.clear();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_max_metrics() {
        let mut cfg = TelemetryConfig::default();
        cfg.max_metrics = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_is_kind_enabled() {
        let cfg = TelemetryConfig::default();
        assert!(cfg.is_kind_enabled(MetricKind::Counter));
        assert!(cfg.is_kind_enabled(MetricKind::Histogram));

        let cfg2 = TelemetryConfig {
            enabled_kinds: vec![MetricKind::Counter],
            ..TelemetryConfig::default()
        };
        assert!(!cfg2.is_kind_enabled(MetricKind::Gauge));
    }

    #[test]
    fn config_custom_prefix() {
        let cfg =
            TelemetryConfig { metric_prefix: "custom".to_string(), ..TelemetryConfig::default() };
        assert_eq!(cfg.metric_prefix, "custom");
    }

    // ── Property tests ─────────────────────────────────────────────

    #[test]
    fn property_counter_monotonic_after_increments() {
        let reg = MetricsRegistry::new();
        reg.register("c", MetricKind::Counter).unwrap();
        let mut values = Vec::new();
        for i in 0..50 {
            reg.increment("c", (i % 7 + 1) as f64).unwrap();
            values.push(reg.get_value("c").unwrap());
        }
        for w in values.windows(2) {
            assert!(w[1] >= w[0], "counter must be monotonically non-decreasing");
        }
    }

    #[test]
    fn property_histogram_sum_count_avg() {
        let mut h = HistogramBucket::new(&[1.0, 5.0, 10.0, 50.0, 100.0]);
        let values = [0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 25.0, 50.0, 100.0, 200.0];
        for &v in &values {
            h.observe(v);
        }
        // sum should equal count * mean
        let expected = h.count() as f64 * h.mean();
        assert!(
            (h.sum() - expected).abs() < 1e-9,
            "sum ({}) ≠ count * avg ({})",
            h.sum(),
            expected,
        );
    }

    #[test]
    fn property_histogram_cumulative_nondecreasing() {
        let mut h = HistogramBucket::new(&[1.0, 5.0, 10.0, 50.0]);
        for v in [0.1, 3.0, 7.0, 20.0, 100.0] {
            h.observe(v);
        }
        let buckets = h.buckets();
        for w in buckets.windows(2) {
            assert!(w[1].1 >= w[0].1, "cumulative counts must be non-decreasing");
        }
    }

    #[test]
    fn property_histogram_last_bucket_equals_count() {
        let mut h = HistogramBucket::new(&[1.0, 10.0]);
        for v in [0.5, 5.0, 15.0, 20.0] {
            h.observe(v);
        }
        let buckets = h.buckets();
        assert_eq!(buckets.last().unwrap().1, h.count());
    }
}
