//! Telemetry and metrics collection for GPU inference monitoring.
//!
//! Provides counters, gauges, and histograms with export to
//! Prometheus text format, JSON, and CSV.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Core metric types ────────────────────────────────────────────────────────

/// Monotonically increasing counter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Counter {
    pub name: String,
    pub value: u64,
    pub labels: Vec<(String, String)>,
}

/// Point-in-time gauge value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Gauge {
    pub name: String,
    pub value: f64,
    pub labels: Vec<(String, String)>,
}

/// Distribution of observed values across buckets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Histogram {
    pub name: String,
    pub buckets: Vec<f64>,
    pub counts: Vec<u64>,
    pub sum: f64,
    pub count: u64,
}

impl Histogram {
    fn new(name: impl Into<String>, buckets: Vec<f64>) -> Self {
        let len = buckets.len();
        Self { name: name.into(), buckets, counts: vec![0; len], sum: 0.0, count: 0 }
    }

    fn record(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        for (i, &upper) in self.buckets.iter().enumerate() {
            if value <= upper {
                self.counts[i] = self.counts[i].saturating_add(1);
            }
        }
    }

    /// Approximate percentile (0.0–1.0) from cumulative bucket counts.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let target = (p * self.count as f64).ceil() as u64;
        for (i, &c) in self.counts.iter().enumerate() {
            if c >= target {
                return Some(self.buckets[i]);
            }
        }
        self.buckets.last().copied()
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Controls collector behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub prefix: String,
    pub export_interval_ms: u64,
    pub max_histogram_buckets: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prefix: String::new(),
            export_interval_ms: 10_000,
            max_histogram_buckets: 32,
        }
    }
}

// ── Inference metrics ────────────────────────────────────────────────────────

/// Per-request inference statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub tokens_generated: u64,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_inference_time_ms: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub gpu_memory_used: u64,
    pub gpu_utilization: f64,
    pub batch_size: u32,
    pub model_name: String,
    pub backend: String,
}

// ── Snapshot ─────────────────────────────────────────────────────────────────

/// Immutable point-in-time capture of all metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp_ms: u64,
    pub counters: Vec<Counter>,
    pub gauges: Vec<Gauge>,
    pub histograms: Vec<Histogram>,
}

// ── Export ────────────────────────────────────────────────────────────────────

/// Supported export wire formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Prometheus,
    Json,
    Csv,
}

/// Stateless formatter that converts a [`MetricsSnapshot`] into a string.
pub struct MetricsExporter;

impl MetricsExporter {
    /// Render `snapshot` in the requested `format`.
    pub fn export(snapshot: &MetricsSnapshot, format: ExportFormat) -> String {
        match format {
            ExportFormat::Prometheus => Self::to_prometheus(snapshot),
            ExportFormat::Json => Self::to_json(snapshot),
            ExportFormat::Csv => Self::to_csv(snapshot),
        }
    }

    fn to_prometheus(snap: &MetricsSnapshot) -> String {
        let mut out = String::new();
        for c in &snap.counters {
            let _ = writeln!(out, "# TYPE {} counter", c.name);
            out.push_str(&c.name);
            Self::append_labels(&mut out, &c.labels);
            let _ = writeln!(out, " {}", c.value);
        }
        for g in &snap.gauges {
            let _ = writeln!(out, "# TYPE {} gauge", g.name);
            out.push_str(&g.name);
            Self::append_labels(&mut out, &g.labels);
            let _ = writeln!(out, " {}", format_f64(g.value));
        }
        for h in &snap.histograms {
            let _ = writeln!(out, "# TYPE {} histogram", h.name);
            let cumulative = &h.counts;
            for (i, upper) in h.buckets.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "{}_bucket{{le=\"{}\"}} {}",
                    h.name,
                    format_f64(*upper),
                    cumulative[i],
                );
            }
            let _ = writeln!(out, "{}_bucket{{le=\"+Inf\"}} {}", h.name, h.count);
            let _ = writeln!(out, "{}_sum {}", h.name, format_f64(h.sum));
            let _ = writeln!(out, "{}_count {}", h.name, h.count);
        }
        out
    }

    fn to_json(snap: &MetricsSnapshot) -> String {
        serde_json::to_string(snap).unwrap_or_default()
    }

    fn to_csv(snap: &MetricsSnapshot) -> String {
        let mut out = String::from("type,name,value\n");
        for c in &snap.counters {
            let _ = writeln!(out, "counter,{},{}", c.name, c.value);
        }
        for g in &snap.gauges {
            let _ = writeln!(out, "gauge,{},{}", g.name, format_f64(g.value));
        }
        for h in &snap.histograms {
            let _ = writeln!(out, "histogram,{},{}", h.name, format_f64(h.sum));
        }
        out
    }

    fn append_labels(out: &mut String, labels: &[(String, String)]) {
        if labels.is_empty() {
            return;
        }
        out.push('{');
        for (i, (k, v)) in labels.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "{k}=\"{v}\"");
        }
        out.push('}');
    }
}

/// Format an `f64` without unnecessary trailing zeros but keep at least one
/// decimal for clarity where the value is integer-like.
fn format_f64(v: f64) -> String {
    if v.fract() == 0.0 { format!("{v:.1}") } else { format!("{v}") }
}

// ── Collector ────────────────────────────────────────────────────────────────

/// Central metrics registry.
///
/// When `config.enabled` is `false` every mutating method is a no-op.
pub struct MetricsCollector {
    counters: HashMap<String, Counter>,
    gauges: HashMap<String, Gauge>,
    histograms: HashMap<String, Histogram>,
    config: MetricsConfig,
}

/// Default histogram buckets for latency metrics (seconds).
const DEFAULT_LATENCY_BUCKETS: &[f64] =
    &[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

impl MetricsCollector {
    /// Create a new collector and register the pre-defined inference
    /// metrics.
    pub fn new(config: MetricsConfig) -> Self {
        let mut collector = Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            config,
        };
        collector.register_defaults();
        collector
    }

    // ── Counters ─────────────────────────────────────────────────────

    /// Increment a counter by 1.  Creates the counter if it does not exist.
    pub fn increment(&mut self, name: &str) {
        if !self.config.enabled {
            return;
        }
        self.increment_by(name, 1);
    }

    /// Increment a counter by an arbitrary delta.
    pub fn increment_by(&mut self, name: &str, delta: u64) {
        if !self.config.enabled {
            return;
        }
        let prefixed = self.prefixed(name);
        let counter = self.counters.entry(prefixed.clone()).or_insert_with(|| Counter {
            name: prefixed,
            value: 0,
            labels: Vec::new(),
        });
        counter.value = counter.value.saturating_add(delta);
    }

    /// Read the current value of a counter (0 when absent).
    pub fn counter_value(&self, name: &str) -> u64 {
        let prefixed = self.prefixed(name);
        self.counters.get(&prefixed).map_or(0, |c| c.value)
    }

    // ── Gauges ───────────────────────────────────────────────────────

    /// Set a gauge to an absolute value.
    pub fn set_gauge(&mut self, name: &str, value: f64) {
        if !self.config.enabled {
            return;
        }
        let prefixed = self.prefixed(name);
        let gauge = self.gauges.entry(prefixed.clone()).or_insert_with(|| Gauge {
            name: prefixed,
            value: 0.0,
            labels: Vec::new(),
        });
        gauge.value = value;
    }

    /// Set a gauge with attached labels.
    pub fn set_gauge_with_labels(&mut self, name: &str, value: f64, labels: Vec<(String, String)>) {
        if !self.config.enabled {
            return;
        }
        let prefixed = self.prefixed(name);
        let gauge = self.gauges.entry(prefixed.clone()).or_insert_with(|| Gauge {
            name: prefixed,
            value: 0.0,
            labels: Vec::new(),
        });
        gauge.value = value;
        gauge.labels = labels;
    }

    /// Read the current value of a gauge (0.0 when absent).
    pub fn gauge_value(&self, name: &str) -> f64 {
        let prefixed = self.prefixed(name);
        self.gauges.get(&prefixed).map_or(0.0, |g| g.value)
    }

    // ── Histograms ───────────────────────────────────────────────────

    /// Record a single observation into a histogram.  The histogram is
    /// created with default latency buckets when first referenced.
    pub fn record_histogram(&mut self, name: &str, value: f64) {
        if !self.config.enabled {
            return;
        }
        let prefixed = self.prefixed(name);
        let histogram = self
            .histograms
            .entry(prefixed.clone())
            .or_insert_with(|| Histogram::new(prefixed, DEFAULT_LATENCY_BUCKETS.to_vec()));
        histogram.record(value);
    }

    /// Record a value into a histogram that was created with custom
    /// buckets.
    pub fn record_histogram_with_buckets(&mut self, name: &str, value: f64, buckets: Vec<f64>) {
        if !self.config.enabled {
            return;
        }
        let prefixed = self.prefixed(name);
        let histogram = self
            .histograms
            .entry(prefixed.clone())
            .or_insert_with(|| Histogram::new(prefixed, buckets));
        histogram.record(value);
    }

    // ── Convenience ──────────────────────────────────────────────────

    /// Record a full set of inference metrics in one call.
    #[allow(clippy::cast_precision_loss)]
    pub fn record_inference(&mut self, m: &InferenceMetrics) {
        if !self.config.enabled {
            return;
        }
        self.increment_by("bitnet_tokens_total", m.tokens_generated);
        self.record_histogram(
            "bitnet_inference_duration_seconds",
            m.total_inference_time_ms / 1000.0,
        );
        self.record_histogram(
            "bitnet_time_to_first_token_seconds",
            m.time_to_first_token_ms / 1000.0,
        );
        self.set_gauge("bitnet_gpu_memory_bytes", m.gpu_memory_used as f64);
        self.set_gauge("bitnet_gpu_utilization_ratio", m.gpu_utilization);
        self.set_gauge("bitnet_batch_size", f64::from(m.batch_size));
    }

    /// Increment the pre-defined error counter.
    pub fn record_error(&mut self) {
        self.increment("bitnet_errors_total");
    }

    /// Set the active-request gauge.
    #[allow(clippy::cast_precision_loss)]
    pub fn set_active_requests(&mut self, n: u64) {
        self.set_gauge("bitnet_active_requests", n as f64);
    }

    // ── Snapshot ─────────────────────────────────────────────────────

    /// Capture an immutable snapshot of all current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            timestamp_ms: now_ms(),
            counters: self.counters.values().cloned().collect(),
            gauges: self.gauges.values().cloned().collect(),
            histograms: self.histograms.values().cloned().collect(),
        }
    }

    // ── Internals ────────────────────────────────────────────────────

    fn prefixed(&self, name: &str) -> String {
        if self.config.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}_{}", self.config.prefix, name)
        }
    }

    fn register_defaults(&mut self) {
        // Pre-register counters
        for name in &["bitnet_tokens_total", "bitnet_errors_total"] {
            let prefixed = self.prefixed(name);
            self.counters.entry(prefixed.clone()).or_insert_with(|| Counter {
                name: prefixed,
                value: 0,
                labels: Vec::new(),
            });
        }

        // Pre-register gauges
        for name in &[
            "bitnet_gpu_memory_bytes",
            "bitnet_gpu_utilization_ratio",
            "bitnet_batch_size",
            "bitnet_active_requests",
        ] {
            let prefixed = self.prefixed(name);
            self.gauges.entry(prefixed.clone()).or_insert_with(|| Gauge {
                name: prefixed,
                value: 0.0,
                labels: Vec::new(),
            });
        }

        // Pre-register histograms
        for name in &["bitnet_inference_duration_seconds", "bitnet_time_to_first_token_seconds"] {
            let prefixed = self.prefixed(name);
            self.histograms
                .entry(prefixed.clone())
                .or_insert_with(|| Histogram::new(prefixed, DEFAULT_LATENCY_BUCKETS.to_vec()));
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_collector() -> MetricsCollector {
        MetricsCollector::new(MetricsConfig::default())
    }

    fn sample_inference() -> InferenceMetrics {
        InferenceMetrics {
            tokens_generated: 42,
            tokens_per_second: 21.0,
            time_to_first_token_ms: 150.0,
            total_inference_time_ms: 2000.0,
            prompt_tokens: 10,
            completion_tokens: 42,
            gpu_memory_used: 1_073_741_824,
            gpu_utilization: 0.85,
            batch_size: 4,
            model_name: "bitnet-2B".into(),
            backend: "cuda".into(),
        }
    }

    // ── Counter tests ────────────────────────────────────────────────

    #[test]
    fn counter_increment_once() {
        let mut c = default_collector();
        c.increment("bitnet_tokens_total");
        assert_eq!(c.counter_value("bitnet_tokens_total"), 1);
    }

    #[test]
    fn counter_increment_multiple() {
        let mut c = default_collector();
        c.increment("bitnet_tokens_total");
        c.increment("bitnet_tokens_total");
        c.increment("bitnet_tokens_total");
        assert_eq!(c.counter_value("bitnet_tokens_total"), 3);
    }

    #[test]
    fn counter_increment_by() {
        let mut c = default_collector();
        c.increment_by("bitnet_tokens_total", 100);
        assert_eq!(c.counter_value("bitnet_tokens_total"), 100);
    }

    #[test]
    fn counter_absent_returns_zero() {
        let c = default_collector();
        assert_eq!(c.counter_value("nonexistent"), 0);
    }

    #[test]
    fn counter_created_on_demand() {
        let mut c = default_collector();
        c.increment("new_counter");
        assert_eq!(c.counter_value("new_counter"), 1);
    }

    #[test]
    fn counter_saturating_overflow() {
        let mut c = default_collector();
        c.increment_by("bitnet_tokens_total", u64::MAX);
        c.increment("bitnet_tokens_total");
        assert_eq!(c.counter_value("bitnet_tokens_total"), u64::MAX);
    }

    #[test]
    fn counter_default_registered() {
        let c = default_collector();
        // Pre-registered should exist with value 0.
        assert_eq!(c.counter_value("bitnet_tokens_total"), 0);
        assert_eq!(c.counter_value("bitnet_errors_total"), 0);
    }

    // ── Gauge tests ──────────────────────────────────────────────────

    #[test]
    fn gauge_set_and_get() {
        let mut c = default_collector();
        c.set_gauge("bitnet_gpu_memory_bytes", 1024.0);
        assert!((c.gauge_value("bitnet_gpu_memory_bytes") - 1024.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gauge_overwrite() {
        let mut c = default_collector();
        c.set_gauge("bitnet_gpu_memory_bytes", 100.0);
        c.set_gauge("bitnet_gpu_memory_bytes", 200.0);
        assert!((c.gauge_value("bitnet_gpu_memory_bytes") - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gauge_absent_returns_zero() {
        let c = default_collector();
        assert!((c.gauge_value("nonexistent")).abs() < f64::EPSILON);
    }

    #[test]
    fn gauge_with_labels() {
        let mut c = default_collector();
        c.set_gauge_with_labels(
            "bitnet_gpu_utilization_ratio",
            0.75,
            vec![("gpu".into(), "0".into())],
        );
        assert!((c.gauge_value("bitnet_gpu_utilization_ratio") - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn gauge_default_registered() {
        let c = default_collector();
        assert!((c.gauge_value("bitnet_gpu_memory_bytes")).abs() < f64::EPSILON);
        assert!((c.gauge_value("bitnet_gpu_utilization_ratio")).abs() < f64::EPSILON);
        assert!((c.gauge_value("bitnet_batch_size")).abs() < f64::EPSILON);
        assert!((c.gauge_value("bitnet_active_requests")).abs() < f64::EPSILON);
    }

    #[test]
    fn gauge_negative_value() {
        let mut c = default_collector();
        c.set_gauge("temp", -42.5);
        assert!((c.gauge_value("temp") - (-42.5)).abs() < f64::EPSILON);
    }

    // ── Histogram tests ──────────────────────────────────────────────

    #[test]
    fn histogram_single_record() {
        let mut c = default_collector();
        c.record_histogram("bitnet_inference_duration_seconds", 0.05);
        let snap = c.snapshot();
        let h =
            snap.histograms.iter().find(|h| h.name == "bitnet_inference_duration_seconds").unwrap();
        assert_eq!(h.count, 1);
        assert!((h.sum - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_bucket_assignment() {
        let mut c = default_collector();
        // 0.03 should fall into the 0.05 bucket and all above
        c.record_histogram("bitnet_inference_duration_seconds", 0.03);
        let snap = c.snapshot();
        let h =
            snap.histograms.iter().find(|h| h.name == "bitnet_inference_duration_seconds").unwrap();
        // Cumulative: buckets 0.005, 0.01, 0.025 should be 0; 0.05+ should be 1
        assert_eq!(h.counts[0], 0); // le=0.005
        assert_eq!(h.counts[1], 0); // le=0.01
        assert_eq!(h.counts[2], 0); // le=0.025
        assert_eq!(h.counts[3], 1); // le=0.05
        assert_eq!(h.counts[4], 1); // le=0.1
    }

    #[test]
    fn histogram_multiple_records() {
        let mut c = default_collector();
        c.record_histogram("bitnet_inference_duration_seconds", 0.001);
        c.record_histogram("bitnet_inference_duration_seconds", 0.5);
        c.record_histogram("bitnet_inference_duration_seconds", 5.0);
        let snap = c.snapshot();
        let h =
            snap.histograms.iter().find(|h| h.name == "bitnet_inference_duration_seconds").unwrap();
        assert_eq!(h.count, 3);
        assert!((h.sum - 5.501).abs() < 1e-9);
    }

    #[test]
    fn histogram_percentile_p50() {
        let mut h = Histogram::new("test", vec![1.0, 5.0, 10.0]);
        for _ in 0..50 {
            h.record(0.5); // bucket 1.0
        }
        for _ in 0..50 {
            h.record(7.0); // bucket 10.0
        }
        assert_eq!(h.percentile(0.50), Some(1.0));
    }

    #[test]
    fn histogram_percentile_p95() {
        let mut h = Histogram::new("test", vec![1.0, 5.0, 10.0]);
        for _ in 0..90 {
            h.record(0.5);
        }
        for _ in 0..10 {
            h.record(7.0);
        }
        assert_eq!(h.percentile(0.95), Some(10.0));
    }

    #[test]
    fn histogram_percentile_p99() {
        let mut h = Histogram::new("test", vec![1.0, 5.0, 10.0]);
        for _ in 0..99 {
            h.record(0.5);
        }
        h.record(7.0);
        assert_eq!(h.percentile(0.99), Some(1.0));
    }

    #[test]
    fn histogram_percentile_empty() {
        let h = Histogram::new("test", vec![1.0, 5.0]);
        assert_eq!(h.percentile(0.50), None);
    }

    #[test]
    fn histogram_custom_buckets() {
        let mut c = default_collector();
        c.record_histogram_with_buckets("custom", 7.5, vec![1.0, 5.0, 10.0, 50.0]);
        let snap = c.snapshot();
        let h = snap.histograms.iter().find(|h| h.name == "custom").unwrap();
        assert_eq!(h.buckets.len(), 4);
        assert_eq!(h.count, 1);
    }

    #[test]
    fn histogram_value_exceeds_all_buckets() {
        let mut h = Histogram::new("test", vec![1.0, 5.0]);
        h.record(100.0);
        assert_eq!(h.count, 1);
        assert!((h.sum - 100.0).abs() < f64::EPSILON);
        // No cumulative bucket should match.
        assert_eq!(h.counts[0], 0);
        assert_eq!(h.counts[1], 0);
    }

    #[test]
    fn histogram_default_registered() {
        let c = default_collector();
        let snap = c.snapshot();
        assert!(snap.histograms.iter().any(|h| h.name == "bitnet_inference_duration_seconds"));
        assert!(snap.histograms.iter().any(|h| h.name == "bitnet_time_to_first_token_seconds"));
    }

    // ── InferenceMetrics recording ───────────────────────────────────

    #[test]
    fn record_inference_populates_counters() {
        let mut c = default_collector();
        c.record_inference(&sample_inference());
        assert_eq!(c.counter_value("bitnet_tokens_total"), 42);
    }

    #[test]
    fn record_inference_populates_gauges() {
        let mut c = default_collector();
        c.record_inference(&sample_inference());
        assert!((c.gauge_value("bitnet_gpu_memory_bytes") - 1_073_741_824.0).abs() < f64::EPSILON);
        assert!((c.gauge_value("bitnet_gpu_utilization_ratio") - 0.85).abs() < f64::EPSILON);
        assert!((c.gauge_value("bitnet_batch_size") - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn record_inference_populates_histograms() {
        let mut c = default_collector();
        c.record_inference(&sample_inference());
        let snap = c.snapshot();
        let dur =
            snap.histograms.iter().find(|h| h.name == "bitnet_inference_duration_seconds").unwrap();
        assert_eq!(dur.count, 1);
        // 2000ms -> 2.0s
        assert!((dur.sum - 2.0).abs() < 1e-9);

        let ttft = snap
            .histograms
            .iter()
            .find(|h| h.name == "bitnet_time_to_first_token_seconds")
            .unwrap();
        assert_eq!(ttft.count, 1);
        // 150ms -> 0.15s
        assert!((ttft.sum - 0.15).abs() < 1e-9);
    }

    #[test]
    fn record_inference_accumulates() {
        let mut c = default_collector();
        let m = sample_inference();
        c.record_inference(&m);
        c.record_inference(&m);
        assert_eq!(c.counter_value("bitnet_tokens_total"), 84);
    }

    #[test]
    fn record_error_increments() {
        let mut c = default_collector();
        c.record_error();
        c.record_error();
        assert_eq!(c.counter_value("bitnet_errors_total"), 2);
    }

    #[test]
    fn set_active_requests() {
        let mut c = default_collector();
        c.set_active_requests(5);
        assert!((c.gauge_value("bitnet_active_requests") - 5.0).abs() < f64::EPSILON);
    }

    // ── Snapshot tests ───────────────────────────────────────────────

    #[test]
    fn snapshot_captures_current_state() {
        let mut c = default_collector();
        c.increment("bitnet_tokens_total");
        c.set_gauge("bitnet_gpu_memory_bytes", 512.0);
        let snap = c.snapshot();
        assert!(snap.timestamp_ms > 0);
        assert!(snap.counters.iter().any(|c| c.name == "bitnet_tokens_total" && c.value == 1));
        assert!(
            snap.gauges
                .iter()
                .any(|g| g.name == "bitnet_gpu_memory_bytes"
                    && (g.value - 512.0).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn snapshot_is_independent_of_mutations() {
        let mut c = default_collector();
        c.increment("bitnet_tokens_total");
        let snap1 = c.snapshot();
        c.increment("bitnet_tokens_total");
        let snap2 = c.snapshot();
        let v1 = snap1.counters.iter().find(|c| c.name == "bitnet_tokens_total").unwrap().value;
        let v2 = snap2.counters.iter().find(|c| c.name == "bitnet_tokens_total").unwrap().value;
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);
    }

    #[test]
    fn empty_collector_snapshot() {
        let c = default_collector();
        let snap = c.snapshot();
        // Pre-registered metrics should still appear.
        assert!(!snap.counters.is_empty());
        assert!(!snap.gauges.is_empty());
        assert!(!snap.histograms.is_empty());
    }

    // ── Prometheus export ────────────────────────────────────────────

    #[test]
    fn prometheus_counter_format() {
        let mut c = default_collector();
        c.increment_by("bitnet_tokens_total", 10);
        let snap = c.snapshot();
        let out = MetricsExporter::export(&snap, ExportFormat::Prometheus);
        assert!(out.contains("# TYPE bitnet_tokens_total counter"));
        assert!(out.contains("bitnet_tokens_total 10"));
    }

    #[test]
    fn prometheus_gauge_format() {
        let mut c = default_collector();
        c.set_gauge("bitnet_gpu_memory_bytes", 2048.0);
        let snap = c.snapshot();
        let out = MetricsExporter::export(&snap, ExportFormat::Prometheus);
        assert!(out.contains("# TYPE bitnet_gpu_memory_bytes gauge"));
        assert!(out.contains("bitnet_gpu_memory_bytes 2048.0"));
    }

    #[test]
    fn prometheus_histogram_format() {
        let mut c = default_collector();
        c.record_histogram("bitnet_inference_duration_seconds", 0.05);
        let snap = c.snapshot();
        let out = MetricsExporter::export(&snap, ExportFormat::Prometheus);
        assert!(out.contains("# TYPE bitnet_inference_duration_seconds histogram"));
        assert!(out.contains("bitnet_inference_duration_seconds_bucket{le=\"+Inf\"} 1"));
        assert!(out.contains("bitnet_inference_duration_seconds_sum"));
        assert!(out.contains("bitnet_inference_duration_seconds_count 1"));
    }

    #[test]
    fn prometheus_labels() {
        let mut c = default_collector();
        c.set_gauge_with_labels(
            "bitnet_gpu_utilization_ratio",
            0.9,
            vec![("gpu".into(), "0".into()), ("model".into(), "2B".into())],
        );
        let snap = c.snapshot();
        let out = MetricsExporter::export(&snap, ExportFormat::Prometheus);
        assert!(out.contains("gpu=\"0\""));
        assert!(out.contains("model=\"2B\""));
    }

    #[test]
    fn prometheus_empty_snapshot() {
        let snap = MetricsSnapshot {
            timestamp_ms: 0,
            counters: vec![],
            gauges: vec![],
            histograms: vec![],
        };
        let out = MetricsExporter::export(&snap, ExportFormat::Prometheus);
        assert!(out.is_empty());
    }

    // ── JSON export ──────────────────────────────────────────────────

    #[test]
    fn json_export_roundtrip() {
        let mut c = default_collector();
        c.increment("bitnet_tokens_total");
        let snap = c.snapshot();
        let json = MetricsExporter::export(&snap, ExportFormat::Json);
        let parsed: MetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.counters.len(), snap.counters.len());
        assert_eq!(parsed.gauges.len(), snap.gauges.len());
        assert_eq!(parsed.histograms.len(), snap.histograms.len());
    }

    #[test]
    fn json_export_contains_timestamp() {
        let c = default_collector();
        let snap = c.snapshot();
        let json = MetricsExporter::export(&snap, ExportFormat::Json);
        assert!(json.contains("timestamp_ms"));
    }

    #[test]
    fn json_export_empty_snapshot() {
        let snap = MetricsSnapshot {
            timestamp_ms: 1000,
            counters: vec![],
            gauges: vec![],
            histograms: vec![],
        };
        let json = MetricsExporter::export(&snap, ExportFormat::Json);
        let parsed: MetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert!(parsed.counters.is_empty());
    }

    // ── CSV export ───────────────────────────────────────────────────

    #[test]
    fn csv_export_headers() {
        let c = default_collector();
        let snap = c.snapshot();
        let csv = MetricsExporter::export(&snap, ExportFormat::Csv);
        assert!(csv.starts_with("type,name,value\n"));
    }

    #[test]
    fn csv_export_counter_row() {
        let mut c = default_collector();
        c.increment_by("bitnet_tokens_total", 7);
        let snap = c.snapshot();
        let csv = MetricsExporter::export(&snap, ExportFormat::Csv);
        assert!(csv.contains("counter,bitnet_tokens_total,7"));
    }

    #[test]
    fn csv_export_gauge_row() {
        let mut c = default_collector();
        c.set_gauge("bitnet_gpu_memory_bytes", 256.0);
        let snap = c.snapshot();
        let csv = MetricsExporter::export(&snap, ExportFormat::Csv);
        assert!(csv.contains("gauge,bitnet_gpu_memory_bytes,256.0"));
    }

    #[test]
    fn csv_export_histogram_row() {
        let mut c = default_collector();
        c.record_histogram("bitnet_inference_duration_seconds", 1.5);
        let snap = c.snapshot();
        let csv = MetricsExporter::export(&snap, ExportFormat::Csv);
        assert!(csv.contains("histogram,bitnet_inference_duration_seconds,"));
    }

    #[test]
    fn csv_export_empty() {
        let snap = MetricsSnapshot {
            timestamp_ms: 0,
            counters: vec![],
            gauges: vec![],
            histograms: vec![],
        };
        let csv = MetricsExporter::export(&snap, ExportFormat::Csv);
        assert_eq!(csv, "type,name,value\n");
    }

    // ── Prefix tests ─────────────────────────────────────────────────

    #[test]
    fn prefix_applied_to_counter() {
        let config = MetricsConfig { prefix: "myapp".into(), ..MetricsConfig::default() };
        let mut c = MetricsCollector::new(config);
        c.increment("bitnet_tokens_total");
        assert_eq!(c.counter_value("bitnet_tokens_total"), 1);
    }

    #[test]
    fn prefix_in_snapshot() {
        let config = MetricsConfig { prefix: "myapp".into(), ..MetricsConfig::default() };
        let mut c = MetricsCollector::new(config);
        c.increment("bitnet_tokens_total");
        let snap = c.snapshot();
        assert!(snap.counters.iter().any(|c| c.name.starts_with("myapp_")));
    }

    #[test]
    fn empty_prefix_no_underscore() {
        let c = default_collector();
        let snap = c.snapshot();
        assert!(snap.counters.iter().all(|c| !c.name.starts_with('_')));
    }

    // ── Disabled collector ───────────────────────────────────────────

    #[test]
    fn disabled_collector_increment_noop() {
        let config = MetricsConfig { enabled: false, ..MetricsConfig::default() };
        let mut c = MetricsCollector::new(config);
        c.increment("bitnet_tokens_total");
        // Pre-registered value stays 0.
        assert_eq!(c.counters.values().map(|c| c.value).sum::<u64>(), 0);
    }

    #[test]
    fn disabled_collector_gauge_noop() {
        let config = MetricsConfig { enabled: false, ..MetricsConfig::default() };
        let mut c = MetricsCollector::new(config);
        c.set_gauge("bitnet_gpu_memory_bytes", 999.0);
        assert!(c.gauges.values().all(|g| g.value == 0.0));
    }

    #[test]
    fn disabled_collector_histogram_noop() {
        let config = MetricsConfig { enabled: false, ..MetricsConfig::default() };
        let mut c = MetricsCollector::new(config);
        c.record_histogram("bitnet_inference_duration_seconds", 1.0);
        assert!(c.histograms.values().all(|h| h.count == 0));
    }

    #[test]
    fn disabled_collector_record_inference_noop() {
        let config = MetricsConfig { enabled: false, ..MetricsConfig::default() };
        let mut c = MetricsCollector::new(config);
        c.record_inference(&sample_inference());
        assert_eq!(c.counters.values().map(|c| c.value).sum::<u64>(), 0);
    }

    // ── MetricsConfig defaults ───────────────────────────────────────

    #[test]
    fn config_defaults() {
        let cfg = MetricsConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.prefix.is_empty());
        assert_eq!(cfg.export_interval_ms, 10_000);
        assert_eq!(cfg.max_histogram_buckets, 32);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn histogram_exact_bucket_boundary() {
        let mut h = Histogram::new("test", vec![1.0, 5.0, 10.0]);
        h.record(5.0);
        // 5.0 <= 5.0, so bucket index 1 and above should count.
        assert_eq!(h.counts[0], 0);
        assert_eq!(h.counts[1], 1);
        assert_eq!(h.counts[2], 1);
    }

    #[test]
    fn histogram_zero_value() {
        let mut h = Histogram::new("test", vec![0.0, 1.0]);
        h.record(0.0);
        assert_eq!(h.counts[0], 1);
        assert_eq!(h.count, 1);
    }

    #[test]
    fn gauge_zero_value() {
        let mut c = default_collector();
        c.set_gauge("bitnet_batch_size", 0.0);
        assert!((c.gauge_value("bitnet_batch_size")).abs() < f64::EPSILON);
    }

    #[test]
    fn multiple_histograms_independent() {
        let mut c = default_collector();
        c.record_histogram("bitnet_inference_duration_seconds", 1.0);
        c.record_histogram("bitnet_time_to_first_token_seconds", 0.1);
        let snap = c.snapshot();
        let dur =
            snap.histograms.iter().find(|h| h.name == "bitnet_inference_duration_seconds").unwrap();
        let ttft = snap
            .histograms
            .iter()
            .find(|h| h.name == "bitnet_time_to_first_token_seconds")
            .unwrap();
        assert_eq!(dur.count, 1);
        assert_eq!(ttft.count, 1);
        assert!((dur.sum - 1.0).abs() < f64::EPSILON);
        assert!((ttft.sum - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn format_f64_integer() {
        assert_eq!(format_f64(42.0), "42.0");
    }

    #[test]
    fn format_f64_fractional() {
        assert_eq!(format_f64(1.234), "1.234");
    }

    #[test]
    fn counter_labels_default_empty() {
        let c = default_collector();
        let snap = c.snapshot();
        for counter in &snap.counters {
            assert!(counter.labels.is_empty());
        }
    }

    #[test]
    fn export_format_eq() {
        assert_eq!(ExportFormat::Prometheus, ExportFormat::Prometheus);
        assert_ne!(ExportFormat::Prometheus, ExportFormat::Json);
        assert_ne!(ExportFormat::Json, ExportFormat::Csv);
    }

    #[test]
    fn inference_metrics_serialize() {
        let m = sample_inference();
        let json = serde_json::to_string(&m).unwrap();
        let parsed: InferenceMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tokens_generated, 42);
        assert_eq!(parsed.model_name, "bitnet-2B");
    }

    #[test]
    fn snapshot_serialize_deserialize() {
        let mut c = default_collector();
        c.increment("bitnet_tokens_total");
        let snap = c.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let parsed: MetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.counters.len(), snap.counters.len());
    }

    #[test]
    fn prometheus_multiple_counters() {
        let mut c = default_collector();
        c.increment_by("bitnet_tokens_total", 5);
        c.increment_by("bitnet_errors_total", 2);
        let snap = c.snapshot();
        let out = MetricsExporter::export(&snap, ExportFormat::Prometheus);
        assert!(out.contains("bitnet_tokens_total 5"));
        assert!(out.contains("bitnet_errors_total 2"));
    }

    #[test]
    fn csv_multiple_rows() {
        let mut c = default_collector();
        c.increment_by("bitnet_tokens_total", 3);
        c.set_gauge("bitnet_gpu_memory_bytes", 64.0);
        let snap = c.snapshot();
        let csv = MetricsExporter::export(&snap, ExportFormat::Csv);
        // header + counters + gauges + histograms
        assert!(csv.lines().count() > 3);
    }

    #[test]
    fn histogram_large_count() {
        let mut h = Histogram::new("test", vec![1.0, 10.0]);
        for i in 0..10_000 {
            h.record(f64::from(i % 15));
        }
        assert_eq!(h.count, 10_000);
    }

    #[test]
    fn collector_many_dynamic_metrics() {
        let mut c = default_collector();
        for i in 0..100 {
            c.increment(&format!("dynamic_counter_{i}"));
            c.set_gauge(&format!("dynamic_gauge_{i}"), f64::from(i));
        }
        let snap = c.snapshot();
        // 2 pre-registered + 100 dynamic
        assert!(snap.counters.len() >= 102);
        // 4 pre-registered + 100 dynamic
        assert!(snap.gauges.len() >= 104);
    }
}
