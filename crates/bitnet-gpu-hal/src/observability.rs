//! Comprehensive observability for GPU HAL: metrics, tracing, structured logging.
//!
//! Provides [`ObservabilityEngine`] combining [`MetricsCollector`],
//! [`TraceCollector`], and [`LogCollector`] with Prometheus-format export.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant, SystemTime};

// ─── Configuration ───────────────────────────────────────────────────────

/// Global observability settings.
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    /// Collect counter/gauge/histogram metrics.
    pub enable_metrics: bool,
    /// Collect distributed traces.
    pub enable_tracing: bool,
    /// Collect structured log entries.
    pub enable_logging: bool,
    /// Prefix prepended to every metric name (e.g. `"bitnet_gpu_"`).
    pub metrics_prefix: String,
    /// Sampling rate in `0.0..=1.0` (1.0 = keep everything).
    pub sampling_rate: f64,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            enable_logging: true,
            metrics_prefix: String::new(),
            sampling_rate: 1.0,
        }
    }
}

// ─── Metrics ─────────────────────────────────────────────────────────────

/// Kind of metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Counter => f.write_str("counter"),
            Self::Gauge => f.write_str("gauge"),
            Self::Histogram => f.write_str("histogram"),
            Self::Summary => f.write_str("summary"),
        }
    }
}

/// Schema for a metric (name, type, description, label keys, histogram buckets).
#[derive(Debug, Clone)]
pub struct MetricDefinition {
    pub name: String,
    pub metric_type: MetricType,
    pub description: String,
    pub labels: Vec<String>,
    pub buckets: Vec<f64>,
}

impl MetricDefinition {
    /// Create a counter definition.
    pub fn counter(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            metric_type: MetricType::Counter,
            description: description.into(),
            labels: Vec::new(),
            buckets: Vec::new(),
        }
    }

    /// Create a gauge definition.
    pub fn gauge(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            metric_type: MetricType::Gauge,
            description: description.into(),
            labels: Vec::new(),
            buckets: Vec::new(),
        }
    }

    /// Create a histogram definition with the given bucket boundaries.
    pub fn histogram(
        name: impl Into<String>,
        description: impl Into<String>,
        buckets: Vec<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            metric_type: MetricType::Histogram,
            description: description.into(),
            labels: Vec::new(),
            buckets,
        }
    }

    /// Attach label keys.
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }
}

/// Current value stored for a single metric.
#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram {
        observations: Vec<f64>,
        count: u64,
        sum: f64,
    },
}

impl MetricValue {
    fn new_counter() -> Self {
        Self::Counter(0)
    }
    fn new_gauge() -> Self {
        Self::Gauge(0.0)
    }
    fn new_histogram() -> Self {
        Self::Histogram { observations: Vec::new(), count: 0, sum: 0.0 }
    }
}

/// Collects and stores metric values keyed by `(name, labels)`.
#[derive(Debug)]
pub struct MetricsCollector {
    definitions: HashMap<String, MetricDefinition>,
    /// Key = `(metric_name, sorted_label_pairs)`.
    values: HashMap<(String, Vec<(String, String)>), MetricValue>,
    prefix: String,
}

impl MetricsCollector {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            definitions: HashMap::new(),
            values: HashMap::new(),
            prefix: prefix.into(),
        }
    }

    /// Register a metric definition.
    pub fn register(&mut self, def: MetricDefinition) {
        self.definitions.insert(def.name.clone(), def);
    }

    fn prefixed(&self, name: &str) -> String {
        format!("{}{}", self.prefix, name)
    }

    fn key(name: &str, labels: &[(String, String)]) -> (String, Vec<(String, String)>) {
        let mut sorted = labels.to_vec();
        sorted.sort();
        (name.to_string(), sorted)
    }

    /// Increment a counter by `delta`.
    pub fn increment(&mut self, name: &str, labels: &[(String, String)], delta: u64) {
        let pname = self.prefixed(name);
        let key = Self::key(&pname, labels);
        let entry = self.values.entry(key).or_insert_with(MetricValue::new_counter);
        if let MetricValue::Counter(v) = entry {
            *v = v.saturating_add(delta);
        }
    }

    /// Set a gauge to an absolute value.
    pub fn set_gauge(&mut self, name: &str, labels: &[(String, String)], value: f64) {
        let pname = self.prefixed(name);
        let key = Self::key(&pname, labels);
        let entry = self.values.entry(key).or_insert_with(MetricValue::new_gauge);
        if let MetricValue::Gauge(v) = entry {
            *v = value;
        }
    }

    /// Record a histogram observation.
    pub fn observe(&mut self, name: &str, labels: &[(String, String)], value: f64) {
        let pname = self.prefixed(name);
        let key = Self::key(&pname, labels);
        let entry = self.values.entry(key).or_insert_with(MetricValue::new_histogram);
        if let MetricValue::Histogram { observations, count, sum } = entry {
            observations.push(value);
            *count += 1;
            *sum += value;
        }
    }

    /// Reset a counter back to zero.
    pub fn reset_counter(&mut self, name: &str, labels: &[(String, String)]) {
        let pname = self.prefixed(name);
        let key = Self::key(&pname, labels);
        if let Some(MetricValue::Counter(v)) = self.values.get_mut(&key) {
            *v = 0;
        }
    }

    /// Return a snapshot of all values.
    #[allow(clippy::type_complexity)]
    pub fn get_all(&self) -> &HashMap<(String, Vec<(String, String)>), MetricValue> {
        &self.values
    }

    /// Look up a single metric value.
    pub fn get(&self, name: &str, labels: &[(String, String)]) -> Option<&MetricValue> {
        let pname = self.prefixed(name);
        let key = Self::key(&pname, labels);
        self.values.get(&key)
    }

    /// Return all registered definitions.
    pub fn definitions(&self) -> &HashMap<String, MetricDefinition> {
        &self.definitions
    }
}

// ─── Tracing ─────────────────────────────────────────────────────────────

/// A single span in a distributed trace.
#[derive(Debug, Clone)]
pub struct SpanContext {
    pub trace_id: u64,
    pub span_id: u64,
    pub parent_span_id: Option<u64>,
    pub operation: String,
    pub start_time: Instant,
    pub duration: Option<Duration>,
    pub attributes: HashMap<String, String>,
}

/// Collects distributed trace spans.
#[derive(Debug)]
pub struct TraceCollector {
    active: HashMap<u64, SpanContext>,
    completed: Vec<SpanContext>,
    next_span_id: u64,
}

impl TraceCollector {
    pub fn new() -> Self {
        Self { active: HashMap::new(), completed: Vec::new(), next_span_id: 1 }
    }

    /// Begin a new span and return its `span_id`.
    pub fn start_span(
        &mut self,
        trace_id: u64,
        parent_span_id: Option<u64>,
        operation: impl Into<String>,
    ) -> u64 {
        let span_id = self.next_span_id;
        self.next_span_id += 1;
        let ctx = SpanContext {
            trace_id,
            span_id,
            parent_span_id,
            operation: operation.into(),
            start_time: Instant::now(),
            duration: None,
            attributes: HashMap::new(),
        };
        self.active.insert(span_id, ctx);
        span_id
    }

    /// Add an attribute to an active span.
    pub fn set_attribute(&mut self, span_id: u64, key: impl Into<String>, value: impl Into<String>) {
        if let Some(span) = self.active.get_mut(&span_id) {
            span.attributes.insert(key.into(), value.into());
        }
    }

    /// End an active span, recording its duration.
    pub fn end_span(&mut self, span_id: u64) {
        if let Some(mut span) = self.active.remove(&span_id) {
            span.duration = Some(span.start_time.elapsed());
            self.completed.push(span);
        }
    }

    /// Return all completed traces.
    pub fn get_traces(&self) -> &[SpanContext] {
        &self.completed
    }

    /// Return all active (unfinished) spans.
    pub fn get_active(&self) -> &HashMap<u64, SpanContext> {
        &self.active
    }

    /// Export completed traces, draining the buffer.
    pub fn export(&mut self) -> Vec<SpanContext> {
        std::mem::take(&mut self.completed)
    }
}

impl Default for TraceCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Logging ─────────────────────────────────────────────────────────────

/// Log severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trace => f.write_str("TRACE"),
            Self::Debug => f.write_str("DEBUG"),
            Self::Info => f.write_str("INFO"),
            Self::Warn => f.write_str("WARN"),
            Self::Error => f.write_str("ERROR"),
            Self::Fatal => f.write_str("FATAL"),
        }
    }
}

/// A structured log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: SystemTime,
    pub level: LogLevel,
    pub message: String,
    pub fields: HashMap<String, String>,
    pub trace_id: Option<u64>,
}

/// Collects structured log entries.
#[derive(Debug)]
pub struct LogCollector {
    entries: Vec<LogEntry>,
    min_level: LogLevel,
}

impl LogCollector {
    pub fn new(min_level: LogLevel) -> Self {
        Self { entries: Vec::new(), min_level }
    }

    /// Append a log entry if it meets the minimum level.
    pub fn log(&mut self, entry: LogEntry) {
        if entry.level >= self.min_level {
            self.entries.push(entry);
        }
    }

    /// Convenience: create and store an entry.
    pub fn record(
        &mut self,
        level: LogLevel,
        message: impl Into<String>,
        fields: HashMap<String, String>,
        trace_id: Option<u64>,
    ) {
        self.log(LogEntry {
            timestamp: SystemTime::now(),
            level,
            message: message.into(),
            fields,
            trace_id,
        });
    }

    /// Query entries at or above a given level.
    pub fn query_by_level(&self, min: LogLevel) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.level >= min).collect()
    }

    /// Query entries that have a specific trace_id.
    pub fn query_by_trace(&self, trace_id: u64) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.trace_id == Some(trace_id)).collect()
    }

    /// Query entries whose message contains `substring`.
    pub fn query_by_message(&self, substring: &str) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.message.contains(substring)).collect()
    }

    /// Return all entries.
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Export (drain) all entries.
    pub fn export(&mut self) -> Vec<LogEntry> {
        std::mem::take(&mut self.entries)
    }

    /// Current minimum level.
    pub fn min_level(&self) -> LogLevel {
        self.min_level
    }

    /// Update the minimum level.
    pub fn set_min_level(&mut self, level: LogLevel) {
        self.min_level = level;
    }
}

impl Default for LogCollector {
    fn default() -> Self {
        Self::new(LogLevel::Info)
    }
}

// ─── Prometheus exporter ─────────────────────────────────────────────────

/// Serialises [`MetricsCollector`] contents to Prometheus exposition format.
pub struct PrometheusExporter;

impl PrometheusExporter {
    /// Render all metrics in the collector as a Prometheus text block.
    pub fn export(collector: &MetricsCollector) -> String {
        let mut out = String::new();
        // Gather definitions for TYPE/HELP lines.
        let defs = collector.definitions();

        // Sort keys for deterministic output.
        let mut keys: Vec<_> = collector.get_all().keys().cloned().collect();
        keys.sort();

        // Track which metric names we've already emitted headers for.
        let mut seen_headers: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (name, labels) in &keys {
            // Emit HELP/TYPE once per base name.
            let base_name = name.clone();
            if !seen_headers.contains(&base_name) {
                if let Some(def) = defs.get(&base_name) {
                    out.push_str(&format!("# HELP {} {}\n", base_name, def.description));
                    out.push_str(&format!("# TYPE {} {}\n", base_name, def.metric_type));
                }
                seen_headers.insert(base_name.clone());
            }

            let value = &collector.get_all()[&(name.clone(), labels.clone())];
            let label_str = Self::format_labels(labels);

            match value {
                MetricValue::Counter(v) => {
                    out.push_str(&format!("{}{} {}\n", name, label_str, v));
                }
                MetricValue::Gauge(v) => {
                    out.push_str(&format!("{}{} {}\n", name, label_str, format_f64(*v)));
                }
                MetricValue::Histogram { observations, count, sum } => {
                    // Find bucket boundaries from definition, or synthesise from observations.
                    let buckets = defs
                        .get(name.as_str())
                        .map(|d| d.buckets.clone())
                        .unwrap_or_default();

                    for b in &buckets {
                        let le_count = observations.iter().filter(|&&o| o <= *b).count();
                        out.push_str(&format!(
                            "{}_bucket{{le=\"{}\"{}}} {}\n",
                            name,
                            format_f64(*b),
                            Self::comma_labels(labels),
                            le_count,
                        ));
                    }
                    // +Inf bucket
                    out.push_str(&format!(
                        "{}_bucket{{le=\"+Inf\"{}}} {}\n",
                        name,
                        Self::comma_labels(labels),
                        count,
                    ));
                    out.push_str(&format!("{}_sum{} {}\n", name, label_str, format_f64(*sum)));
                    out.push_str(&format!("{}_count{} {}\n", name, label_str, count));
                }
            }
        }
        out
    }

    fn format_labels(labels: &[(String, String)]) -> String {
        if labels.is_empty() {
            return String::new();
        }
        let inner: Vec<String> =
            labels.iter().map(|(k, v)| format!("{}=\"{}\"", k, v)).collect();
        format!("{{{}}}", inner.join(","))
    }

    fn comma_labels(labels: &[(String, String)]) -> String {
        if labels.is_empty() {
            return String::new();
        }
        let inner: Vec<String> =
            labels.iter().map(|(k, v)| format!("{}=\"{}\"", k, v)).collect();
        format!(",{}", inner.join(","))
    }
}

/// Format an f64 without unnecessary trailing zeros but keep at least one decimal.
fn format_f64(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{v:.0}")
    } else {
        format!("{v}")
    }
}

// ─── Observability engine ────────────────────────────────────────────────

/// Top-level engine combining metrics, traces, and logs.
#[derive(Debug)]
pub struct ObservabilityEngine {
    pub config: ObservabilityConfig,
    pub metrics: MetricsCollector,
    pub traces: TraceCollector,
    pub logs: LogCollector,
    sample_counter: u64,
}

impl ObservabilityEngine {
    pub fn new(config: ObservabilityConfig) -> Self {
        let prefix = config.metrics_prefix.clone();
        Self {
            config,
            metrics: MetricsCollector::new(prefix),
            traces: TraceCollector::new(),
            logs: LogCollector::new(LogLevel::Trace),
            sample_counter: 0,
        }
    }

    /// Whether this event passes the sampling gate.
    pub fn should_sample(&mut self) -> bool {
        if self.config.sampling_rate >= 1.0 {
            return true;
        }
        if self.config.sampling_rate <= 0.0 {
            return false;
        }
        self.sample_counter += 1;
        // Deterministic: sample every `1/rate` events.
        let period = (1.0 / self.config.sampling_rate).ceil() as u64;
        self.sample_counter.is_multiple_of(period)
    }

    /// Register a metric definition in the inner collector.
    pub fn register_metric(&mut self, def: MetricDefinition) {
        self.metrics.register(def);
    }

    /// Increment a counter (respects `enable_metrics`).
    pub fn increment(&mut self, name: &str, labels: &[(String, String)], delta: u64) {
        if self.config.enable_metrics {
            self.metrics.increment(name, labels, delta);
        }
    }

    /// Set a gauge (respects `enable_metrics`).
    pub fn set_gauge(&mut self, name: &str, labels: &[(String, String)], value: f64) {
        if self.config.enable_metrics {
            self.metrics.set_gauge(name, labels, value);
        }
    }

    /// Observe a histogram value (respects `enable_metrics`).
    pub fn observe(&mut self, name: &str, labels: &[(String, String)], value: f64) {
        if self.config.enable_metrics {
            self.metrics.observe(name, labels, value);
        }
    }

    /// Start a trace span (respects `enable_tracing`).
    pub fn start_span(
        &mut self,
        trace_id: u64,
        parent: Option<u64>,
        operation: impl Into<String>,
    ) -> Option<u64> {
        if self.config.enable_tracing {
            Some(self.traces.start_span(trace_id, parent, operation))
        } else {
            None
        }
    }

    /// End a trace span (respects `enable_tracing`).
    pub fn end_span(&mut self, span_id: u64) {
        if self.config.enable_tracing {
            self.traces.end_span(span_id);
        }
    }

    /// Record a log entry (respects `enable_logging`).
    pub fn log(
        &mut self,
        level: LogLevel,
        message: impl Into<String>,
        fields: HashMap<String, String>,
        trace_id: Option<u64>,
    ) {
        if self.config.enable_logging {
            self.logs.record(level, message, fields, trace_id);
        }
    }

    /// Export Prometheus text.
    pub fn export_prometheus(&self) -> String {
        PrometheusExporter::export(&self.metrics)
    }
}

impl Default for ObservabilityEngine {
    fn default() -> Self {
        Self::new(ObservabilityConfig::default())
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    fn no_labels() -> Vec<(String, String)> {
        vec![]
    }

    fn labels(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    // =====================================================================
    // ObservabilityConfig
    // =====================================================================

    #[test]
    fn config_default_enables_all() {
        let c = ObservabilityConfig::default();
        assert!(c.enable_metrics);
        assert!(c.enable_tracing);
        assert!(c.enable_logging);
    }

    #[test]
    fn config_default_sampling_rate_is_one() {
        let c = ObservabilityConfig::default();
        assert!((c.sampling_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn config_default_prefix_empty() {
        assert!(ObservabilityConfig::default().metrics_prefix.is_empty());
    }

    // =====================================================================
    // MetricType
    // =====================================================================

    #[test]
    fn metric_type_display() {
        assert_eq!(MetricType::Counter.to_string(), "counter");
        assert_eq!(MetricType::Gauge.to_string(), "gauge");
        assert_eq!(MetricType::Histogram.to_string(), "histogram");
        assert_eq!(MetricType::Summary.to_string(), "summary");
    }

    #[test]
    fn metric_type_equality() {
        assert_eq!(MetricType::Counter, MetricType::Counter);
        assert_ne!(MetricType::Counter, MetricType::Gauge);
    }

    // =====================================================================
    // MetricDefinition
    // =====================================================================

    #[test]
    fn definition_counter() {
        let d = MetricDefinition::counter("reqs", "Total requests");
        assert_eq!(d.metric_type, MetricType::Counter);
        assert_eq!(d.name, "reqs");
        assert!(d.labels.is_empty());
    }

    #[test]
    fn definition_gauge() {
        let d = MetricDefinition::gauge("temp", "Temperature");
        assert_eq!(d.metric_type, MetricType::Gauge);
    }

    #[test]
    fn definition_histogram() {
        let d = MetricDefinition::histogram("lat", "Latency", vec![0.01, 0.1, 1.0]);
        assert_eq!(d.metric_type, MetricType::Histogram);
        assert_eq!(d.buckets.len(), 3);
    }

    #[test]
    fn definition_with_labels() {
        let d = MetricDefinition::counter("c", "d")
            .with_labels(vec!["method".into(), "path".into()]);
        assert_eq!(d.labels.len(), 2);
    }

    // =====================================================================
    // MetricsCollector — Counters
    // =====================================================================

    #[test]
    fn counter_increment_from_zero() {
        let mut mc = MetricsCollector::new("");
        mc.increment("hits", &no_labels(), 1);
        match mc.get("hits", &no_labels()) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, 1),
            other => panic!("expected Counter, got {other:?}"),
        }
    }

    #[test]
    fn counter_increment_accumulates() {
        let mut mc = MetricsCollector::new("");
        mc.increment("hits", &no_labels(), 3);
        mc.increment("hits", &no_labels(), 7);
        match mc.get("hits", &no_labels()) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, 10),
            other => panic!("expected Counter(10), got {other:?}"),
        }
    }

    #[test]
    fn counter_reset() {
        let mut mc = MetricsCollector::new("");
        mc.increment("c", &no_labels(), 42);
        mc.reset_counter("c", &no_labels());
        match mc.get("c", &no_labels()) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, 0),
            other => panic!("expected Counter(0), got {other:?}"),
        }
    }

    #[test]
    fn counter_reset_nonexistent_is_noop() {
        let mut mc = MetricsCollector::new("");
        mc.reset_counter("nope", &no_labels()); // should not panic
    }

    #[test]
    fn counter_with_labels() {
        let mut mc = MetricsCollector::new("");
        let l1 = labels(&[("method", "GET")]);
        let l2 = labels(&[("method", "POST")]);
        mc.increment("reqs", &l1, 5);
        mc.increment("reqs", &l2, 3);
        match mc.get("reqs", &l1) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, 5),
            other => panic!("unexpected {other:?}"),
        }
        match mc.get("reqs", &l2) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, 3),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn counter_saturating_add() {
        let mut mc = MetricsCollector::new("");
        mc.increment("big", &no_labels(), u64::MAX);
        mc.increment("big", &no_labels(), 1);
        match mc.get("big", &no_labels()) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, u64::MAX),
            other => panic!("expected saturated counter, got {other:?}"),
        }
    }

    // =====================================================================
    // MetricsCollector — Gauges
    // =====================================================================

    #[test]
    fn gauge_set_and_get() {
        let mut mc = MetricsCollector::new("");
        mc.set_gauge("temp", &no_labels(), 36.6);
        match mc.get("temp", &no_labels()) {
            Some(MetricValue::Gauge(v)) => assert!((*v - 36.6).abs() < f64::EPSILON),
            other => panic!("expected Gauge, got {other:?}"),
        }
    }

    #[test]
    fn gauge_overwrite() {
        let mut mc = MetricsCollector::new("");
        mc.set_gauge("g", &no_labels(), 1.0);
        mc.set_gauge("g", &no_labels(), 99.0);
        match mc.get("g", &no_labels()) {
            Some(MetricValue::Gauge(v)) => assert!((*v - 99.0).abs() < f64::EPSILON),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn gauge_negative() {
        let mut mc = MetricsCollector::new("");
        mc.set_gauge("g", &no_labels(), -42.5);
        match mc.get("g", &no_labels()) {
            Some(MetricValue::Gauge(v)) => assert!((*v - (-42.5)).abs() < f64::EPSILON),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn gauge_with_labels() {
        let mut mc = MetricsCollector::new("");
        let l = labels(&[("host", "a")]);
        mc.set_gauge("cpu", &l, 73.0);
        match mc.get("cpu", &l) {
            Some(MetricValue::Gauge(v)) => assert!((*v - 73.0).abs() < f64::EPSILON),
            other => panic!("unexpected {other:?}"),
        }
    }

    // =====================================================================
    // MetricsCollector — Histograms
    // =====================================================================

    #[test]
    fn histogram_single_observation() {
        let mut mc = MetricsCollector::new("");
        mc.observe("lat", &no_labels(), 0.5);
        match mc.get("lat", &no_labels()) {
            Some(MetricValue::Histogram { observations, count, sum }) => {
                assert_eq!(*count, 1);
                assert!((sum - 0.5).abs() < f64::EPSILON);
                assert_eq!(observations.len(), 1);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn histogram_multiple_observations() {
        let mut mc = MetricsCollector::new("");
        for v in [0.1, 0.5, 1.0, 2.5] {
            mc.observe("lat", &no_labels(), v);
        }
        match mc.get("lat", &no_labels()) {
            Some(MetricValue::Histogram { observations, count, sum }) => {
                assert_eq!(*count, 4);
                assert!((sum - 4.1).abs() < 1e-9);
                assert_eq!(observations.len(), 4);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn histogram_bucket_distribution() {
        let mut mc = MetricsCollector::new("");
        mc.register(MetricDefinition::histogram("d", "dur", vec![1.0, 5.0, 10.0]));
        for v in [0.5, 2.0, 3.0, 7.0, 12.0] {
            mc.observe("d", &no_labels(), v);
        }
        // Verify via Prometheus export that buckets are correct.
        let prom = PrometheusExporter::export(&mc);
        assert!(prom.contains("d_bucket{le=\"1\"} 1"));
        assert!(prom.contains("d_bucket{le=\"5\"} 3"));
        assert!(prom.contains("d_bucket{le=\"10\"} 4"));
        assert!(prom.contains("d_bucket{le=\"+Inf\"} 5"));
    }

    // =====================================================================
    // MetricsCollector — Prefix
    // =====================================================================

    #[test]
    fn prefix_applied_to_counter() {
        let mut mc = MetricsCollector::new("gpu_");
        mc.increment("ops", &no_labels(), 1);
        let all = mc.get_all();
        assert!(all.contains_key(&("gpu_ops".to_string(), vec![])));
    }

    #[test]
    fn prefix_applied_to_gauge() {
        let mut mc = MetricsCollector::new("gpu_");
        mc.set_gauge("temp", &no_labels(), 50.0);
        assert!(mc.get_all().contains_key(&("gpu_temp".to_string(), vec![])));
    }

    #[test]
    fn prefix_applied_to_histogram() {
        let mut mc = MetricsCollector::new("gpu_");
        mc.observe("lat", &no_labels(), 1.0);
        assert!(mc.get_all().contains_key(&("gpu_lat".to_string(), vec![])));
    }

    #[test]
    fn empty_prefix_is_identity() {
        let mut mc = MetricsCollector::new("");
        mc.increment("x", &no_labels(), 1);
        assert!(mc.get_all().contains_key(&("x".to_string(), vec![])));
    }

    // =====================================================================
    // MetricsCollector — get_all
    // =====================================================================

    #[test]
    fn get_all_empty_initially() {
        let mc = MetricsCollector::new("");
        assert!(mc.get_all().is_empty());
    }

    #[test]
    fn get_all_reflects_all_metrics() {
        let mut mc = MetricsCollector::new("");
        mc.increment("a", &no_labels(), 1);
        mc.set_gauge("b", &no_labels(), 2.0);
        mc.observe("c", &no_labels(), 3.0);
        assert_eq!(mc.get_all().len(), 3);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let mc = MetricsCollector::new("");
        assert!(mc.get("nope", &no_labels()).is_none());
    }

    // =====================================================================
    // TraceCollector — span lifecycle
    // =====================================================================

    #[test]
    fn start_and_end_span() {
        let mut tc = TraceCollector::new();
        let sid = tc.start_span(1, None, "op");
        assert_eq!(tc.get_active().len(), 1);
        tc.end_span(sid);
        assert!(tc.get_active().is_empty());
        assert_eq!(tc.get_traces().len(), 1);
    }

    #[test]
    fn span_has_duration_after_end() {
        let mut tc = TraceCollector::new();
        let sid = tc.start_span(1, None, "work");
        std::thread::sleep(Duration::from_millis(5));
        tc.end_span(sid);
        let span = &tc.get_traces()[0];
        assert!(span.duration.is_some());
        assert!(span.duration.unwrap() >= Duration::from_millis(1));
    }

    #[test]
    fn span_operation_recorded() {
        let mut tc = TraceCollector::new();
        let sid = tc.start_span(10, None, "kernel_dispatch");
        tc.end_span(sid);
        assert_eq!(tc.get_traces()[0].operation, "kernel_dispatch");
    }

    #[test]
    fn span_trace_id_recorded() {
        let mut tc = TraceCollector::new();
        let sid = tc.start_span(42, None, "x");
        tc.end_span(sid);
        assert_eq!(tc.get_traces()[0].trace_id, 42);
    }

    #[test]
    fn end_nonexistent_span_is_noop() {
        let mut tc = TraceCollector::new();
        tc.end_span(999); // should not panic
        assert!(tc.get_traces().is_empty());
    }

    // ── Nested spans ────────────────────────────────────────────────────

    #[test]
    fn nested_spans_parent_child() {
        let mut tc = TraceCollector::new();
        let parent = tc.start_span(1, None, "parent");
        let child = tc.start_span(1, Some(parent), "child");
        tc.end_span(child);
        tc.end_span(parent);
        let traces = tc.get_traces();
        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].parent_span_id, Some(parent));
        assert_eq!(traces[1].parent_span_id, None);
    }

    #[test]
    fn deeply_nested_spans() {
        let mut tc = TraceCollector::new();
        let s1 = tc.start_span(1, None, "l1");
        let s2 = tc.start_span(1, Some(s1), "l2");
        let s3 = tc.start_span(1, Some(s2), "l3");
        tc.end_span(s3);
        tc.end_span(s2);
        tc.end_span(s1);
        assert_eq!(tc.get_traces().len(), 3);
        assert_eq!(tc.get_traces()[0].parent_span_id, Some(s2));
    }

    // ── Span attributes ─────────────────────────────────────────────────

    #[test]
    fn span_attributes() {
        let mut tc = TraceCollector::new();
        let sid = tc.start_span(1, None, "op");
        tc.set_attribute(sid, "backend", "cuda");
        tc.end_span(sid);
        assert_eq!(tc.get_traces()[0].attributes.get("backend").unwrap(), "cuda");
    }

    #[test]
    fn attribute_on_missing_span_is_noop() {
        let mut tc = TraceCollector::new();
        tc.set_attribute(999, "k", "v"); // should not panic
    }

    // ── Export ───────────────────────────────────────────────────────────

    #[test]
    fn export_drains_completed() {
        let mut tc = TraceCollector::new();
        let sid = tc.start_span(1, None, "op");
        tc.end_span(sid);
        let exported = tc.export();
        assert_eq!(exported.len(), 1);
        assert!(tc.get_traces().is_empty());
    }

    #[test]
    fn export_empty_returns_empty() {
        let mut tc = TraceCollector::new();
        assert!(tc.export().is_empty());
    }

    // ── Span ID uniqueness ──────────────────────────────────────────────

    #[test]
    fn span_ids_are_unique() {
        let mut tc = TraceCollector::new();
        let a = tc.start_span(1, None, "a");
        let b = tc.start_span(1, None, "b");
        let c = tc.start_span(1, None, "c");
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    // =====================================================================
    // LogCollector
    // =====================================================================

    #[test]
    fn log_entry_stored() {
        let mut lc = LogCollector::new(LogLevel::Trace);
        lc.record(LogLevel::Info, "hello", HashMap::new(), None);
        assert_eq!(lc.entries().len(), 1);
        assert_eq!(lc.entries()[0].message, "hello");
    }

    #[test]
    fn log_below_min_level_dropped() {
        let mut lc = LogCollector::new(LogLevel::Warn);
        lc.record(LogLevel::Debug, "dropped", HashMap::new(), None);
        assert!(lc.entries().is_empty());
    }

    #[test]
    fn log_at_min_level_kept() {
        let mut lc = LogCollector::new(LogLevel::Warn);
        lc.record(LogLevel::Warn, "kept", HashMap::new(), None);
        assert_eq!(lc.entries().len(), 1);
    }

    #[test]
    fn log_above_min_level_kept() {
        let mut lc = LogCollector::new(LogLevel::Warn);
        lc.record(LogLevel::Error, "err", HashMap::new(), None);
        assert_eq!(lc.entries().len(), 1);
    }

    #[test]
    fn log_query_by_level() {
        let mut lc = LogCollector::new(LogLevel::Trace);
        lc.record(LogLevel::Debug, "d", HashMap::new(), None);
        lc.record(LogLevel::Info, "i", HashMap::new(), None);
        lc.record(LogLevel::Error, "e", HashMap::new(), None);
        let errors = lc.query_by_level(LogLevel::Error);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "e");
    }

    #[test]
    fn log_query_by_trace_id() {
        let mut lc = LogCollector::new(LogLevel::Trace);
        lc.record(LogLevel::Info, "a", HashMap::new(), Some(42));
        lc.record(LogLevel::Info, "b", HashMap::new(), Some(99));
        lc.record(LogLevel::Info, "c", HashMap::new(), Some(42));
        let traced = lc.query_by_trace(42);
        assert_eq!(traced.len(), 2);
    }

    #[test]
    fn log_query_by_message_substring() {
        let mut lc = LogCollector::new(LogLevel::Trace);
        lc.record(LogLevel::Info, "kernel launch ok", HashMap::new(), None);
        lc.record(LogLevel::Info, "memory alloc", HashMap::new(), None);
        let found = lc.query_by_message("kernel");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn log_query_by_message_no_match() {
        let mut lc = LogCollector::new(LogLevel::Trace);
        lc.record(LogLevel::Info, "hello", HashMap::new(), None);
        assert!(lc.query_by_message("xyz").is_empty());
    }

    #[test]
    fn log_fields_preserved() {
        let mut lc = LogCollector::new(LogLevel::Trace);
        let mut fields = HashMap::new();
        fields.insert("key".to_string(), "value".to_string());
        lc.record(LogLevel::Info, "msg", fields, None);
        assert_eq!(lc.entries()[0].fields.get("key").unwrap(), "value");
    }

    #[test]
    fn log_export_drains() {
        let mut lc = LogCollector::new(LogLevel::Trace);
        lc.record(LogLevel::Info, "a", HashMap::new(), None);
        let exported = lc.export();
        assert_eq!(exported.len(), 1);
        assert!(lc.entries().is_empty());
    }

    #[test]
    fn log_set_min_level() {
        let mut lc = LogCollector::new(LogLevel::Error);
        assert_eq!(lc.min_level(), LogLevel::Error);
        lc.set_min_level(LogLevel::Debug);
        assert_eq!(lc.min_level(), LogLevel::Debug);
    }

    #[test]
    fn log_default_min_level_is_info() {
        let lc = LogCollector::default();
        assert_eq!(lc.min_level(), LogLevel::Info);
    }

    // =====================================================================
    // LogLevel ordering
    // =====================================================================

    #[test]
    fn log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn log_level_display() {
        assert_eq!(LogLevel::Trace.to_string(), "TRACE");
        assert_eq!(LogLevel::Fatal.to_string(), "FATAL");
    }

    // =====================================================================
    // PrometheusExporter
    // =====================================================================

    #[test]
    fn prometheus_counter_format() {
        let mut mc = MetricsCollector::new("");
        mc.register(MetricDefinition::counter("reqs_total", "Total requests"));
        mc.increment("reqs_total", &no_labels(), 42);
        let out = PrometheusExporter::export(&mc);
        assert!(out.contains("# HELP reqs_total Total requests"));
        assert!(out.contains("# TYPE reqs_total counter"));
        assert!(out.contains("reqs_total 42"));
    }

    #[test]
    fn prometheus_gauge_format() {
        let mut mc = MetricsCollector::new("");
        mc.register(MetricDefinition::gauge("temp", "Temperature"));
        mc.set_gauge("temp", &no_labels(), 36.6);
        let out = PrometheusExporter::export(&mc);
        assert!(out.contains("# TYPE temp gauge"));
        assert!(out.contains("temp 36.6"));
    }

    #[test]
    fn prometheus_histogram_format() {
        let mut mc = MetricsCollector::new("");
        mc.register(MetricDefinition::histogram("dur", "Duration", vec![0.5, 1.0]));
        mc.observe("dur", &no_labels(), 0.3);
        mc.observe("dur", &no_labels(), 0.8);
        let out = PrometheusExporter::export(&mc);
        assert!(out.contains("dur_bucket{le=\"0.5\"} 1"));
        assert!(out.contains("dur_bucket{le=\"1\"} 2"));
        assert!(out.contains("dur_bucket{le=\"+Inf\"} 2"));
        assert!(out.contains("dur_count 2"));
    }

    #[test]
    fn prometheus_labels_in_output() {
        let mut mc = MetricsCollector::new("");
        mc.increment("reqs", &labels(&[("method", "GET")]), 5);
        let out = PrometheusExporter::export(&mc);
        assert!(out.contains("reqs{method=\"GET\"} 5"));
    }

    #[test]
    fn prometheus_empty_collector() {
        let mc = MetricsCollector::new("");
        let out = PrometheusExporter::export(&mc);
        assert!(out.is_empty());
    }

    #[test]
    fn prometheus_with_prefix() {
        let mut mc = MetricsCollector::new("gpu_");
        mc.register(MetricDefinition::counter("gpu_ops", "GPU ops"));
        mc.increment("ops", &no_labels(), 7);
        let out = PrometheusExporter::export(&mc);
        assert!(out.contains("gpu_ops 7"));
    }

    // =====================================================================
    // ObservabilityEngine
    // =====================================================================

    #[test]
    fn engine_default_works() {
        let e = ObservabilityEngine::default();
        assert!(e.config.enable_metrics);
    }

    #[test]
    fn engine_increment_counter() {
        let mut e = ObservabilityEngine::default();
        e.increment("ops", &no_labels(), 1);
        match e.metrics.get("ops", &no_labels()) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, 1),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn engine_set_gauge() {
        let mut e = ObservabilityEngine::default();
        e.set_gauge("mem", &no_labels(), 1024.0);
        match e.metrics.get("mem", &no_labels()) {
            Some(MetricValue::Gauge(v)) => assert!((*v - 1024.0).abs() < f64::EPSILON),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn engine_observe_histogram() {
        let mut e = ObservabilityEngine::default();
        e.observe("lat", &no_labels(), 0.5);
        match e.metrics.get("lat", &no_labels()) {
            Some(MetricValue::Histogram { count, .. }) => assert_eq!(*count, 1),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn engine_metrics_disabled_skips_increment() {
        let cfg = ObservabilityConfig { enable_metrics: false, ..Default::default() };
        let mut e = ObservabilityEngine::new(cfg);
        e.increment("ops", &no_labels(), 1);
        assert!(e.metrics.get("ops", &no_labels()).is_none());
    }

    #[test]
    fn engine_tracing_disabled_returns_none() {
        let cfg = ObservabilityConfig { enable_tracing: false, ..Default::default() };
        let mut e = ObservabilityEngine::new(cfg);
        assert!(e.start_span(1, None, "op").is_none());
    }

    #[test]
    fn engine_logging_disabled_skips_log() {
        let cfg = ObservabilityConfig { enable_logging: false, ..Default::default() };
        let mut e = ObservabilityEngine::new(cfg);
        e.log(LogLevel::Error, "boom", HashMap::new(), None);
        assert!(e.logs.entries().is_empty());
    }

    #[test]
    fn engine_start_and_end_span() {
        let mut e = ObservabilityEngine::default();
        let sid = e.start_span(1, None, "test").unwrap();
        e.end_span(sid);
        assert_eq!(e.traces.get_traces().len(), 1);
    }

    #[test]
    fn engine_log_entry() {
        let mut e = ObservabilityEngine::default();
        e.log(LogLevel::Info, "started", HashMap::new(), Some(1));
        assert_eq!(e.logs.entries().len(), 1);
    }

    #[test]
    fn engine_export_prometheus() {
        let mut e = ObservabilityEngine::default();
        e.register_metric(MetricDefinition::counter("x", "X"));
        e.increment("x", &no_labels(), 1);
        let out = e.export_prometheus();
        assert!(out.contains("x 1"));
    }

    #[test]
    fn engine_prefix_propagated() {
        let cfg = ObservabilityConfig {
            metrics_prefix: "hal_".to_string(),
            ..Default::default()
        };
        let mut e = ObservabilityEngine::new(cfg);
        e.increment("ops", &no_labels(), 5);
        assert!(e.metrics.get_all().contains_key(&("hal_ops".to_string(), vec![])));
    }

    // =====================================================================
    // Sampling
    // =====================================================================

    #[test]
    fn sampling_rate_one_always_samples() {
        let mut e = ObservabilityEngine::default();
        for _ in 0..100 {
            assert!(e.should_sample());
        }
    }

    #[test]
    fn sampling_rate_zero_never_samples() {
        let cfg = ObservabilityConfig { sampling_rate: 0.0, ..Default::default() };
        let mut e = ObservabilityEngine::new(cfg);
        for _ in 0..100 {
            assert!(!e.should_sample());
        }
    }

    #[test]
    fn sampling_rate_half_samples_some() {
        let cfg = ObservabilityConfig { sampling_rate: 0.5, ..Default::default() };
        let mut e = ObservabilityEngine::new(cfg);
        let sampled: usize = (0..100).filter(|_| e.should_sample()).count();
        assert!(sampled > 0 && sampled <= 100);
    }

    #[test]
    fn sampling_rate_quarter() {
        let cfg = ObservabilityConfig { sampling_rate: 0.25, ..Default::default() };
        let mut e = ObservabilityEngine::new(cfg);
        let sampled: usize = (0..100).filter(|_| e.should_sample()).count();
        assert!(sampled > 0 && sampled < 100);
    }

    // =====================================================================
    // Edge cases
    // =====================================================================

    #[test]
    fn very_long_label_key() {
        let mut mc = MetricsCollector::new("");
        let long_key = "k".repeat(1000);
        let l = vec![(long_key.clone(), "v".to_string())];
        mc.increment("m", &l, 1);
        match mc.get("m", &l) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, 1),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn very_long_label_value() {
        let mut mc = MetricsCollector::new("");
        let long_val = "v".repeat(10_000);
        let l = vec![("k".to_string(), long_val)];
        mc.increment("m", &l, 1);
        assert_eq!(mc.get_all().len(), 1);
    }

    #[test]
    fn high_cardinality_labels() {
        let mut mc = MetricsCollector::new("");
        for i in 0..500 {
            let l = labels(&[("id", &i.to_string())]);
            mc.increment("reqs", &l, 1);
        }
        assert_eq!(mc.get_all().len(), 500);
    }

    #[test]
    fn label_order_independence() {
        let mut mc = MetricsCollector::new("");
        let l1 = labels(&[("a", "1"), ("b", "2")]);
        let l2 = labels(&[("b", "2"), ("a", "1")]);
        mc.increment("m", &l1, 3);
        mc.increment("m", &l2, 7);
        match mc.get("m", &l1) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, 10),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn empty_metric_name() {
        let mut mc = MetricsCollector::new("");
        mc.increment("", &no_labels(), 1);
        match mc.get("", &no_labels()) {
            Some(MetricValue::Counter(v)) => assert_eq!(*v, 1),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn histogram_zero_observation() {
        let mut mc = MetricsCollector::new("");
        mc.observe("h", &no_labels(), 0.0);
        match mc.get("h", &no_labels()) {
            Some(MetricValue::Histogram { sum, count, .. }) => {
                assert_eq!(*count, 1);
                assert!(sum.abs() < f64::EPSILON);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn histogram_negative_observation() {
        let mut mc = MetricsCollector::new("");
        mc.observe("h", &no_labels(), -1.5);
        match mc.get("h", &no_labels()) {
            Some(MetricValue::Histogram { sum, .. }) => {
                assert!((*sum - (-1.5)).abs() < f64::EPSILON);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn multiple_traces_interleaved() {
        let mut tc = TraceCollector::new();
        let a = tc.start_span(1, None, "a");
        let b = tc.start_span(2, None, "b");
        tc.end_span(a);
        let c = tc.start_span(1, None, "c");
        tc.end_span(c);
        tc.end_span(b);
        assert_eq!(tc.get_traces().len(), 3);
    }

    #[test]
    fn log_with_many_fields() {
        let mut lc = LogCollector::new(LogLevel::Trace);
        let mut fields = HashMap::new();
        for i in 0..100 {
            fields.insert(format!("k{i}"), format!("v{i}"));
        }
        lc.record(LogLevel::Info, "big", fields, None);
        assert_eq!(lc.entries()[0].fields.len(), 100);
    }

    #[test]
    fn trace_collector_default() {
        let tc = TraceCollector::default();
        assert!(tc.get_traces().is_empty());
        assert!(tc.get_active().is_empty());
    }

    // =====================================================================
    // Property tests
    // =====================================================================

    proptest::proptest! {
        #[test]
        fn counter_monotonic(increments in proptest::collection::vec(1u64..100, 1..20)) {
            let mut mc = MetricsCollector::new("");
            let mut expected = 0u64;
            for delta in &increments {
                mc.increment("c", &no_labels(), *delta);
                expected = expected.saturating_add(*delta);
                match mc.get("c", &no_labels()) {
                    Some(MetricValue::Counter(v)) => {
                        proptest::prop_assert_eq!(*v, expected);
                    }
                    other => {
                        proptest::prop_assert!(false, "unexpected value: {other:?}");
                    }
                }
            }
        }

        #[test]
        fn gauge_always_equals_last_set(values in proptest::collection::vec(-1e6f64..1e6, 1..20)) {
            let mut mc = MetricsCollector::new("");
            for v in &values {
                mc.set_gauge("g", &no_labels(), *v);
            }
            let last = values.last().unwrap();
            match mc.get("g", &no_labels()) {
                Some(MetricValue::Gauge(v)) => {
                    proptest::prop_assert!((v - last).abs() < 1e-9);
                }
                other => {
                    proptest::prop_assert!(false, "unexpected: {other:?}");
                }
            }
        }

        #[test]
        fn histogram_count_equals_observations(n in 1usize..50) {
            let mut mc = MetricsCollector::new("");
            for i in 0..n {
                #[allow(clippy::cast_precision_loss)]
                mc.observe("h", &no_labels(), i as f64);
            }
            match mc.get("h", &no_labels()) {
                Some(MetricValue::Histogram { count, .. }) => {
                    proptest::prop_assert_eq!(*count, n as u64);
                }
                other => {
                    proptest::prop_assert!(false, "unexpected: {other:?}");
                }
            }
        }

        #[test]
        fn histogram_sum_is_correct(
            values in proptest::collection::vec(0.0f64..100.0, 1..30)
        ) {
            let mut mc = MetricsCollector::new("");
            let expected_sum: f64 = values.iter().sum();
            for v in &values {
                mc.observe("h", &no_labels(), *v);
            }
            match mc.get("h", &no_labels()) {
                Some(MetricValue::Histogram { sum, .. }) => {
                    proptest::prop_assert!((sum - expected_sum).abs() < 1e-6);
                }
                other => {
                    proptest::prop_assert!(false, "unexpected: {other:?}");
                }
            }
        }

        #[test]
        fn span_ids_always_unique(n in 2usize..50) {
            let mut tc = TraceCollector::new();
            let ids: Vec<u64> = (0..n).map(|_| tc.start_span(1, None, "op")).collect();
            let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
            proptest::prop_assert_eq!(unique.len(), n);
        }

        #[test]
        fn log_level_filtering_consistent(
            min_idx in 0usize..6,
            entry_idx in 0usize..6,
        ) {
            let levels = [
                LogLevel::Trace, LogLevel::Debug, LogLevel::Info,
                LogLevel::Warn, LogLevel::Error, LogLevel::Fatal,
            ];
            let min = levels[min_idx];
            let entry = levels[entry_idx];
            let mut lc = LogCollector::new(min);
            lc.record(entry, "msg", HashMap::new(), None);
            if entry >= min {
                proptest::prop_assert_eq!(lc.entries().len(), 1);
            } else {
                proptest::prop_assert!(lc.entries().is_empty());
            }
        }
    }
}
