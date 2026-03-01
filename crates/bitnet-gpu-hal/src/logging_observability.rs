//! Structured logging and observability for GPU HAL operations.
//!
//! Provides structured JSON logging, metrics collection with Prometheus
//! and JSON export, alert management, request lifecycle tracking,
//! audit logging, and log rotation.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        #[allow(clippy::cast_possible_truncation)]
        let ms = d.as_millis() as u64;
        ms
    })
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Log level for filtering messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trace => write!(f, "TRACE"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Info => write!(f, "INFO"),
            Self::Warn => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// Export format for metrics output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ExportFormat {
    Prometheus,
    Json,
}

/// Top-level observability configuration.
#[derive(Debug, Clone, Serialize)]
pub struct ObservabilityConfig {
    pub log_level: LogLevel,
    pub metrics_enabled: bool,
    pub tracing_enabled: bool,
    pub export_format: ExportFormat,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            metrics_enabled: true,
            tracing_enabled: false,
            export_format: ExportFormat::Prometheus,
        }
    }
}

// ---------------------------------------------------------------------------
// Log context
// ---------------------------------------------------------------------------

/// Per-request context attached to every log entry.
#[derive(Debug, Clone, Serialize)]
pub struct LogContext {
    pub request_id: String,
    pub session_id: String,
    pub model_name: String,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, String>,
}

impl LogContext {
    pub fn new(
        request_id: impl Into<String>,
        session_id: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            session_id: session_id.into(),
            model_name: model_name.into(),
            extra: HashMap::new(),
        }
    }

    /// Add an extra key-value pair to the context.
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Structured log entry
// ---------------------------------------------------------------------------

/// A single structured log entry serialisable to JSON.
#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub timestamp_ms: u64,
    pub level: LogLevel,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<LogContext>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub fields: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Structured logger
// ---------------------------------------------------------------------------

/// Structured JSON logger with level filtering and context support.
pub struct StructuredLogger {
    config: ObservabilityConfig,
    entries: Vec<LogEntry>,
}

impl StructuredLogger {
    pub fn new(config: ObservabilityConfig) -> Self {
        Self { config, entries: Vec::new() }
    }

    /// Log a message at the given level without request context.
    pub fn log(&mut self, level: LogLevel, message: impl Into<String>) {
        if level < self.config.log_level {
            return;
        }
        self.entries.push(LogEntry {
            timestamp_ms: now_ms(),
            level,
            message: message.into(),
            context: None,
            fields: HashMap::new(),
        });
    }

    /// Log a message with attached request context.
    pub fn log_with_context(
        &mut self,
        level: LogLevel,
        message: impl Into<String>,
        context: LogContext,
    ) {
        if level < self.config.log_level {
            return;
        }
        self.entries.push(LogEntry {
            timestamp_ms: now_ms(),
            level,
            message: message.into(),
            context: Some(context),
            fields: HashMap::new(),
        });
    }

    /// Log a message with extra key-value fields.
    pub fn log_with_fields(
        &mut self,
        level: LogLevel,
        message: impl Into<String>,
        fields: HashMap<String, String>,
    ) {
        if level < self.config.log_level {
            return;
        }
        self.entries.push(LogEntry {
            timestamp_ms: now_ms(),
            level,
            message: message.into(),
            context: None,
            fields,
        });
    }

    /// Return all recorded log entries.
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Drain and return all entries, resetting the logger.
    pub fn drain(&mut self) -> Vec<LogEntry> {
        std::mem::take(&mut self.entries)
    }

    /// Serialize all entries to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.entries).unwrap_or_default()
    }

    /// Return current effective log level.
    pub fn log_level(&self) -> LogLevel {
        self.config.log_level
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Classification of a recorded metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Counter => write!(f, "counter"),
            Self::Gauge => write!(f, "gauge"),
            Self::Histogram => write!(f, "histogram"),
            Self::Summary => write!(f, "summary"),
        }
    }
}

/// A single metric data point.
#[derive(Debug, Clone, Serialize)]
pub struct MetricPoint {
    pub name: String,
    pub metric_type: MetricType,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub timestamp_ms: u64,
}

/// Collects and aggregates metrics.
pub struct MetricsCollector {
    enabled: bool,
    counters: HashMap<String, f64>,
    gauges: HashMap<String, f64>,
    histograms: HashMap<String, Vec<f64>>,
    labels: HashMap<String, HashMap<String, String>>,
}

impl MetricsCollector {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            labels: HashMap::new(),
        }
    }

    /// Increment a counter by `delta`.
    pub fn increment_counter(&mut self, name: &str, delta: f64) {
        if !self.enabled {
            return;
        }
        *self.counters.entry(name.to_string()).or_insert(0.0) += delta;
    }

    /// Set a gauge to an absolute value.
    pub fn set_gauge(&mut self, name: &str, value: f64) {
        if !self.enabled {
            return;
        }
        self.gauges.insert(name.to_string(), value);
    }

    /// Record an observation in a histogram.
    pub fn observe_histogram(&mut self, name: &str, value: f64) {
        if !self.enabled {
            return;
        }
        self.histograms.entry(name.to_string()).or_default().push(value);
    }

    /// Attach labels to a named metric.
    pub fn set_labels(&mut self, name: &str, labels: HashMap<String, String>) {
        self.labels.insert(name.to_string(), labels);
    }

    /// Get the current value of a counter.
    pub fn counter_value(&self, name: &str) -> Option<f64> {
        self.counters.get(name).copied()
    }

    /// Get the current value of a gauge.
    pub fn gauge_value(&self, name: &str) -> Option<f64> {
        self.gauges.get(name).copied()
    }

    /// Get all observations for a histogram.
    pub fn histogram_values(&self, name: &str) -> Option<&[f64]> {
        self.histograms.get(name).map(Vec::as_slice)
    }

    /// Collect all metrics as a list of [`MetricPoint`]s.
    pub fn collect(&self) -> Vec<MetricPoint> {
        let ts = now_ms();
        let mut out = Vec::new();
        for (name, &value) in &self.counters {
            out.push(MetricPoint {
                name: name.clone(),
                metric_type: MetricType::Counter,
                value,
                labels: self.labels.get(name).cloned().unwrap_or_default(),
                timestamp_ms: ts,
            });
        }
        for (name, &value) in &self.gauges {
            out.push(MetricPoint {
                name: name.clone(),
                metric_type: MetricType::Gauge,
                value,
                labels: self.labels.get(name).cloned().unwrap_or_default(),
                timestamp_ms: ts,
            });
        }
        for (name, values) in &self.histograms {
            #[allow(clippy::cast_precision_loss)]
            let sum: f64 = values.iter().sum();
            out.push(MetricPoint {
                name: name.clone(),
                metric_type: MetricType::Histogram,
                value: sum,
                labels: self.labels.get(name).cloned().unwrap_or_default(),
                timestamp_ms: ts,
            });
        }
        out
    }

    /// Reset all collected metrics.
    pub fn reset(&mut self) {
        self.counters.clear();
        self.gauges.clear();
        self.histograms.clear();
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// ---------------------------------------------------------------------------
// Metrics exporters
// ---------------------------------------------------------------------------

/// Trait for exporting collected metrics to a wire format.
pub trait MetricsExporter {
    fn export(&self, points: &[MetricPoint]) -> String;
    fn format_name(&self) -> &str;
}

/// Exports metrics in Prometheus text exposition format.
pub struct PrometheusExporter;

impl MetricsExporter for PrometheusExporter {
    fn export(&self, points: &[MetricPoint]) -> String {
        use std::fmt::Write as _;
        let mut buf = String::new();
        for p in points {
            let _ = writeln!(buf, "# TYPE {} {}", p.name, p.metric_type);
            if p.labels.is_empty() {
                let _ = writeln!(buf, "{} {}", p.name, p.value);
            } else {
                let labels: Vec<String> =
                    p.labels.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
                let _ = writeln!(buf, "{}{{{}}} {}", p.name, labels.join(","), p.value);
            }
        }
        buf
    }

    fn format_name(&self) -> &str {
        "prometheus"
    }
}

/// Exports metrics as a JSON array.
pub struct JsonExporter;

impl MetricsExporter for JsonExporter {
    fn export(&self, points: &[MetricPoint]) -> String {
        serde_json::to_string_pretty(points).unwrap_or_default()
    }

    fn format_name(&self) -> &str {
        "json"
    }
}

// ---------------------------------------------------------------------------
// Alert system
// ---------------------------------------------------------------------------

/// Comparison operator for alert thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AlertOp {
    GreaterThan,
    LessThan,
    Equal,
}

/// Severity level of a triggered alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Rule that triggers an alert when a metric crosses a threshold.
#[derive(Debug, Clone, Serialize)]
pub struct AlertRule {
    pub name: String,
    pub metric_name: String,
    pub op: AlertOp,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub message: String,
}

impl AlertRule {
    pub fn new(
        name: impl Into<String>,
        metric_name: impl Into<String>,
        op: AlertOp,
        threshold: f64,
        severity: AlertSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            metric_name: metric_name.into(),
            op,
            threshold,
            severity,
            message: message.into(),
        }
    }

    /// Evaluate the rule against a metric value.
    pub fn evaluate(&self, value: f64) -> bool {
        match self.op {
            AlertOp::GreaterThan => value > self.threshold,
            AlertOp::LessThan => value < self.threshold,
            AlertOp::Equal => (value - self.threshold).abs() < f64::EPSILON,
        }
    }
}

/// A triggered alert with timestamp.
#[derive(Debug, Clone, Serialize)]
pub struct Alert {
    pub rule_name: String,
    pub metric_value: f64,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp_ms: u64,
}

/// Manages alert rules and evaluates them against metrics.
pub struct AlertManager {
    rules: Vec<AlertRule>,
    fired: Vec<Alert>,
}

impl AlertManager {
    pub fn new() -> Self {
        Self { rules: Vec::new(), fired: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    /// Evaluate all rules against collected metrics.
    pub fn evaluate(&mut self, collector: &MetricsCollector) {
        let ts = now_ms();
        for rule in &self.rules {
            let value = match rule.op {
                _ if collector.counter_value(&rule.metric_name).is_some() => {
                    collector.counter_value(&rule.metric_name).unwrap()
                }
                _ if collector.gauge_value(&rule.metric_name).is_some() => {
                    collector.gauge_value(&rule.metric_name).unwrap()
                }
                _ => continue,
            };
            if rule.evaluate(value) {
                self.fired.push(Alert {
                    rule_name: rule.name.clone(),
                    metric_value: value,
                    severity: rule.severity,
                    message: rule.message.clone(),
                    timestamp_ms: ts,
                });
            }
        }
    }

    /// Return all alerts that have fired.
    pub fn fired_alerts(&self) -> &[Alert] {
        &self.fired
    }

    /// Clear all fired alerts.
    pub fn clear_alerts(&mut self) {
        self.fired.clear();
    }

    pub fn rules(&self) -> &[AlertRule] {
        &self.rules
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Request lifecycle logger
// ---------------------------------------------------------------------------

/// Phase of an inference request lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RequestPhase {
    Received,
    Validated,
    Processing,
    Complete,
    Failed,
}

impl fmt::Display for RequestPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Received => write!(f, "received"),
            Self::Validated => write!(f, "validated"),
            Self::Processing => write!(f, "processing"),
            Self::Complete => write!(f, "complete"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Record of a single phase transition for a request.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseRecord {
    pub phase: RequestPhase,
    pub timestamp_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Tracks the full lifecycle of a single inference request.
pub struct RequestLogger {
    request_id: String,
    phases: Vec<PhaseRecord>,
    start_ms: u64,
}

impl RequestLogger {
    pub fn new(request_id: impl Into<String>) -> Self {
        let ts = now_ms();
        let mut logger = Self { request_id: request_id.into(), phases: Vec::new(), start_ms: ts };
        logger.phases.push(PhaseRecord {
            phase: RequestPhase::Received,
            timestamp_ms: ts,
            detail: None,
        });
        logger
    }

    /// Record a phase transition.
    pub fn record_phase(&mut self, phase: RequestPhase) {
        self.phases.push(PhaseRecord { phase, timestamp_ms: now_ms(), detail: None });
    }

    /// Record a phase transition with a detail message.
    pub fn record_phase_with_detail(&mut self, phase: RequestPhase, detail: impl Into<String>) {
        self.phases.push(PhaseRecord {
            phase,
            timestamp_ms: now_ms(),
            detail: Some(detail.into()),
        });
    }

    /// Get elapsed time since the request was received (ms).
    pub fn elapsed_ms(&self) -> u64 {
        now_ms().saturating_sub(self.start_ms)
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn phases(&self) -> &[PhaseRecord] {
        &self.phases
    }

    /// Return the current (latest) phase.
    pub fn current_phase(&self) -> Option<RequestPhase> {
        self.phases.last().map(|r| r.phase)
    }

    /// Serialize the lifecycle to JSON.
    pub fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct Summary<'a> {
            request_id: &'a str,
            phases: &'a [PhaseRecord],
            elapsed_ms: u64,
        }
        serde_json::to_string(&Summary {
            request_id: &self.request_id,
            phases: &self.phases,
            elapsed_ms: self.elapsed_ms(),
        })
        .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Audit log
// ---------------------------------------------------------------------------

/// An immutable audit log entry recording model usage.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    pub timestamp_ms: u64,
    pub action: String,
    pub model_name: String,
    pub user_id: String,
    pub request_id: String,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// Append-only audit trail of model usage.
pub struct AuditLog {
    entries: Vec<AuditEntry>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Record an audit entry.
    pub fn record(
        &mut self,
        action: impl Into<String>,
        model_name: impl Into<String>,
        user_id: impl Into<String>,
        request_id: impl Into<String>,
    ) {
        self.entries.push(AuditEntry {
            timestamp_ms: now_ms(),
            action: action.into(),
            model_name: model_name.into(),
            user_id: user_id.into(),
            request_id: request_id.into(),
            metadata: HashMap::new(),
        });
    }

    /// Record an audit entry with extra metadata.
    pub fn record_with_metadata(
        &mut self,
        action: impl Into<String>,
        model_name: impl Into<String>,
        user_id: impl Into<String>,
        request_id: impl Into<String>,
        metadata: HashMap<String, String>,
    ) {
        self.entries.push(AuditEntry {
            timestamp_ms: now_ms(),
            action: action.into(),
            model_name: model_name.into(),
            user_id: user_id.into(),
            request_id: request_id.into(),
            metadata,
        });
    }

    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Number of audit records.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize the full audit trail to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.entries).unwrap_or_default()
    }

    /// Filter entries by action.
    pub fn filter_by_action(&self, action: &str) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.action == action).collect()
    }

    /// Filter entries by model name.
    pub fn filter_by_model(&self, model: &str) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.model_name == model).collect()
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Log rotation
// ---------------------------------------------------------------------------

/// Policy for when log rotation should occur.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RotationPolicy {
    /// Rotate when accumulated size exceeds `max_bytes`.
    Size { max_bytes: usize },
    /// Rotate when age exceeds `max_age_ms` since last rotation.
    Time { max_age_ms: u64 },
}

/// Manages log rotation based on size or time.
pub struct LogRotation {
    policy: RotationPolicy,
    current_bytes: usize,
    last_rotation_ms: u64,
    rotation_count: u32,
}

impl LogRotation {
    pub fn new(policy: RotationPolicy) -> Self {
        Self { policy, current_bytes: 0, last_rotation_ms: now_ms(), rotation_count: 0 }
    }

    /// Record that `bytes` were written.
    pub fn record_write(&mut self, bytes: usize) {
        self.current_bytes += bytes;
    }

    /// Check if rotation should occur based on the policy.
    pub fn should_rotate(&self) -> bool {
        match self.policy {
            RotationPolicy::Size { max_bytes } => self.current_bytes >= max_bytes,
            RotationPolicy::Time { max_age_ms } => {
                now_ms().saturating_sub(self.last_rotation_ms) >= max_age_ms
            }
        }
    }

    /// Perform rotation, resetting internal counters.
    pub fn rotate(&mut self) {
        self.current_bytes = 0;
        self.last_rotation_ms = now_ms();
        self.rotation_count += 1;
    }

    pub fn rotation_count(&self) -> u32 {
        self.rotation_count
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    pub fn policy(&self) -> RotationPolicy {
        self.policy
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -----------------------------------------------------------

    fn default_config() -> ObservabilityConfig {
        ObservabilityConfig::default()
    }

    fn debug_config() -> ObservabilityConfig {
        ObservabilityConfig {
            log_level: LogLevel::Debug,
            metrics_enabled: true,
            tracing_enabled: true,
            export_format: ExportFormat::Json,
        }
    }

    fn test_context() -> LogContext {
        LogContext::new("req-001", "sess-42", "bitnet-2b")
    }

    // -- ObservabilityConfig -----------------------------------------------

    #[test]
    fn test_default_config_values() {
        let cfg = default_config();
        assert_eq!(cfg.log_level, LogLevel::Info);
        assert!(cfg.metrics_enabled);
        assert!(!cfg.tracing_enabled);
        assert_eq!(cfg.export_format, ExportFormat::Prometheus);
    }

    #[test]
    fn test_config_serialization() {
        let cfg = default_config();
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"log_level\":\"Info\""));
        assert!(json.contains("\"metrics_enabled\":true"));
    }

    #[test]
    fn test_debug_config() {
        let cfg = debug_config();
        assert_eq!(cfg.log_level, LogLevel::Debug);
        assert!(cfg.tracing_enabled);
        assert_eq!(cfg.export_format, ExportFormat::Json);
    }

    // -- LogLevel ----------------------------------------------------------

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }

    #[test]
    fn test_log_level_display() {
        assert_eq!(LogLevel::Trace.to_string(), "TRACE");
        assert_eq!(LogLevel::Info.to_string(), "INFO");
        assert_eq!(LogLevel::Error.to_string(), "ERROR");
    }

    // -- LogContext ---------------------------------------------------------

    #[test]
    fn test_log_context_new() {
        let ctx = test_context();
        assert_eq!(ctx.request_id, "req-001");
        assert_eq!(ctx.session_id, "sess-42");
        assert_eq!(ctx.model_name, "bitnet-2b");
        assert!(ctx.extra.is_empty());
    }

    #[test]
    fn test_log_context_with_field() {
        let ctx = test_context().with_field("device", "cuda:0");
        assert_eq!(ctx.extra.get("device").unwrap(), "cuda:0");
    }

    #[test]
    fn test_log_context_chained_fields() {
        let ctx = test_context().with_field("device", "cuda:0").with_field("batch_size", "32");
        assert_eq!(ctx.extra.len(), 2);
    }

    #[test]
    fn test_log_context_serialization() {
        let ctx = test_context();
        let json = serde_json::to_string(&ctx).unwrap();
        assert!(json.contains("\"request_id\":\"req-001\""));
        assert!(!json.contains("extra")); // empty extra omitted
    }

    #[test]
    fn test_log_context_with_extra_serialization() {
        let ctx = test_context().with_field("gpu", "A100");
        let json = serde_json::to_string(&ctx).unwrap();
        assert!(json.contains("\"gpu\":\"A100\""));
    }

    // -- StructuredLogger --------------------------------------------------

    #[test]
    fn test_logger_empty_initially() {
        let logger = StructuredLogger::new(default_config());
        assert!(logger.entries().is_empty());
    }

    #[test]
    fn test_logger_records_entry() {
        let mut logger = StructuredLogger::new(default_config());
        logger.log(LogLevel::Info, "hello");
        assert_eq!(logger.entries().len(), 1);
        assert_eq!(logger.entries()[0].message, "hello");
        assert_eq!(logger.entries()[0].level, LogLevel::Info);
    }

    #[test]
    fn test_logger_filters_below_level() {
        let mut logger = StructuredLogger::new(default_config()); // Info
        logger.log(LogLevel::Debug, "ignored");
        assert!(logger.entries().is_empty());
    }

    #[test]
    fn test_logger_allows_above_level() {
        let mut logger = StructuredLogger::new(default_config()); // Info
        logger.log(LogLevel::Warn, "warning");
        assert_eq!(logger.entries().len(), 1);
    }

    #[test]
    fn test_logger_with_context() {
        let mut logger = StructuredLogger::new(default_config());
        logger.log_with_context(LogLevel::Info, "start", test_context());
        let entry = &logger.entries()[0];
        assert!(entry.context.is_some());
        assert_eq!(entry.context.as_ref().unwrap().request_id, "req-001");
    }

    #[test]
    fn test_logger_with_fields() {
        let mut logger = StructuredLogger::new(default_config());
        let mut fields = HashMap::new();
        fields.insert("tokens".to_string(), "128".to_string());
        logger.log_with_fields(LogLevel::Info, "gen", fields);
        assert_eq!(logger.entries()[0].fields.get("tokens").unwrap(), "128");
    }

    #[test]
    fn test_logger_drain() {
        let mut logger = StructuredLogger::new(default_config());
        logger.log(LogLevel::Info, "a");
        logger.log(LogLevel::Warn, "b");
        let drained = logger.drain();
        assert_eq!(drained.len(), 2);
        assert!(logger.entries().is_empty());
    }

    #[test]
    fn test_logger_to_json() {
        let mut logger = StructuredLogger::new(default_config());
        logger.log(LogLevel::Error, "fail");
        let json = logger.to_json();
        assert!(json.contains("\"message\":\"fail\""));
        assert!(json.contains("\"level\":\"Error\""));
    }

    #[test]
    fn test_logger_returns_level() {
        let logger = StructuredLogger::new(debug_config());
        assert_eq!(logger.log_level(), LogLevel::Debug);
    }

    #[test]
    fn test_logger_multiple_entries_ordering() {
        let mut logger = StructuredLogger::new(default_config());
        logger.log(LogLevel::Info, "first");
        logger.log(LogLevel::Warn, "second");
        logger.log(LogLevel::Error, "third");
        assert_eq!(logger.entries()[0].message, "first");
        assert_eq!(logger.entries()[2].message, "third");
    }

    #[test]
    fn test_logger_timestamp_non_zero() {
        let mut logger = StructuredLogger::new(default_config());
        logger.log(LogLevel::Info, "ts");
        assert!(logger.entries()[0].timestamp_ms > 0);
    }

    // -- MetricType --------------------------------------------------------

    #[test]
    fn test_metric_type_display() {
        assert_eq!(MetricType::Counter.to_string(), "counter");
        assert_eq!(MetricType::Gauge.to_string(), "gauge");
        assert_eq!(MetricType::Histogram.to_string(), "histogram");
        assert_eq!(MetricType::Summary.to_string(), "summary");
    }

    // -- MetricsCollector --------------------------------------------------

    #[test]
    fn test_collector_counter_increment() {
        let mut c = MetricsCollector::new(true);
        c.increment_counter("reqs", 1.0);
        assert_eq!(c.counter_value("reqs"), Some(1.0));
    }

    #[test]
    fn test_collector_counter_accumulates() {
        let mut c = MetricsCollector::new(true);
        c.increment_counter("reqs", 1.0);
        c.increment_counter("reqs", 2.0);
        assert_eq!(c.counter_value("reqs"), Some(3.0));
    }

    #[test]
    fn test_collector_gauge_set() {
        let mut c = MetricsCollector::new(true);
        c.set_gauge("mem_mb", 512.0);
        assert_eq!(c.gauge_value("mem_mb"), Some(512.0));
    }

    #[test]
    fn test_collector_gauge_overwrite() {
        let mut c = MetricsCollector::new(true);
        c.set_gauge("mem_mb", 512.0);
        c.set_gauge("mem_mb", 768.0);
        assert_eq!(c.gauge_value("mem_mb"), Some(768.0));
    }

    #[test]
    fn test_collector_histogram_observe() {
        let mut c = MetricsCollector::new(true);
        c.observe_histogram("latency", 10.0);
        c.observe_histogram("latency", 20.0);
        let vals = c.histogram_values("latency").unwrap();
        assert_eq!(vals, &[10.0, 20.0]);
    }

    #[test]
    fn test_collector_disabled_ignores() {
        let mut c = MetricsCollector::new(false);
        c.increment_counter("reqs", 1.0);
        c.set_gauge("mem", 42.0);
        c.observe_histogram("lat", 5.0);
        assert_eq!(c.counter_value("reqs"), None);
        assert_eq!(c.gauge_value("mem"), None);
        assert!(c.histogram_values("lat").is_none());
    }

    #[test]
    fn test_collector_collect_all() {
        let mut c = MetricsCollector::new(true);
        c.increment_counter("reqs", 5.0);
        c.set_gauge("mem", 256.0);
        c.observe_histogram("lat", 10.0);
        let points = c.collect();
        assert_eq!(points.len(), 3);
    }

    #[test]
    fn test_collector_reset() {
        let mut c = MetricsCollector::new(true);
        c.increment_counter("a", 1.0);
        c.set_gauge("b", 2.0);
        c.reset();
        assert_eq!(c.counter_value("a"), None);
        assert_eq!(c.gauge_value("b"), None);
    }

    #[test]
    fn test_collector_labels() {
        let mut c = MetricsCollector::new(true);
        c.increment_counter("reqs", 1.0);
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "POST".to_string());
        c.set_labels("reqs", labels);
        let points = c.collect();
        let p = points.iter().find(|p| p.name == "reqs").unwrap();
        assert_eq!(p.labels.get("method").unwrap(), "POST");
    }

    #[test]
    fn test_collector_missing_metric_returns_none() {
        let c = MetricsCollector::new(true);
        assert_eq!(c.counter_value("nope"), None);
        assert_eq!(c.gauge_value("nope"), None);
        assert!(c.histogram_values("nope").is_none());
    }

    #[test]
    fn test_collector_is_enabled() {
        assert!(MetricsCollector::new(true).is_enabled());
        assert!(!MetricsCollector::new(false).is_enabled());
    }

    // -- PrometheusExporter ------------------------------------------------

    #[test]
    fn test_prometheus_export_counter() {
        let mut c = MetricsCollector::new(true);
        c.increment_counter("http_requests_total", 42.0);
        let points = c.collect();
        let output = PrometheusExporter.export(&points);
        assert!(output.contains("# TYPE http_requests_total counter"));
        assert!(output.contains("http_requests_total 42"));
    }

    #[test]
    fn test_prometheus_export_with_labels() {
        let mut c = MetricsCollector::new(true);
        c.increment_counter("reqs", 1.0);
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "GET".to_string());
        c.set_labels("reqs", labels);
        let output = PrometheusExporter.export(&c.collect());
        assert!(output.contains("method=\"GET\""));
    }

    #[test]
    fn test_prometheus_format_name() {
        assert_eq!(PrometheusExporter.format_name(), "prometheus");
    }

    #[test]
    fn test_prometheus_export_empty() {
        let output = PrometheusExporter.export(&[]);
        assert!(output.is_empty());
    }

    // -- JsonExporter ------------------------------------------------------

    #[test]
    fn test_json_export() {
        let mut c = MetricsCollector::new(true);
        c.set_gauge("temp", 72.5);
        let output = JsonExporter.export(&c.collect());
        assert!(output.contains("\"name\": \"temp\""));
        assert!(output.contains("72.5"));
    }

    #[test]
    fn test_json_format_name() {
        assert_eq!(JsonExporter.format_name(), "json");
    }

    #[test]
    fn test_json_export_empty() {
        let output = JsonExporter.export(&[]);
        assert_eq!(output, "[]");
    }

    // -- AlertRule ---------------------------------------------------------

    #[test]
    fn test_alert_rule_greater_than() {
        let rule = AlertRule::new(
            "high_cpu",
            "cpu_pct",
            AlertOp::GreaterThan,
            90.0,
            AlertSeverity::Critical,
            "CPU high",
        );
        assert!(rule.evaluate(95.0));
        assert!(!rule.evaluate(85.0));
    }

    #[test]
    fn test_alert_rule_less_than() {
        let rule = AlertRule::new(
            "low_mem",
            "mem_free",
            AlertOp::LessThan,
            100.0,
            AlertSeverity::Warning,
            "Low memory",
        );
        assert!(rule.evaluate(50.0));
        assert!(!rule.evaluate(200.0));
    }

    #[test]
    fn test_alert_rule_equal() {
        let rule = AlertRule::new(
            "exact",
            "val",
            AlertOp::Equal,
            42.0,
            AlertSeverity::Info,
            "Exact match",
        );
        assert!(rule.evaluate(42.0));
        assert!(!rule.evaluate(42.1));
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
    }

    // -- AlertManager ------------------------------------------------------

    #[test]
    fn test_alert_manager_no_rules_no_alerts() {
        let mut mgr = AlertManager::new();
        let c = MetricsCollector::new(true);
        mgr.evaluate(&c);
        assert!(mgr.fired_alerts().is_empty());
    }

    #[test]
    fn test_alert_manager_fires_on_threshold() {
        let mut mgr = AlertManager::new();
        mgr.add_rule(AlertRule::new(
            "high_err",
            "errors",
            AlertOp::GreaterThan,
            10.0,
            AlertSeverity::Critical,
            "Too many errors",
        ));
        let mut c = MetricsCollector::new(true);
        c.increment_counter("errors", 15.0);
        mgr.evaluate(&c);
        assert_eq!(mgr.fired_alerts().len(), 1);
        assert_eq!(mgr.fired_alerts()[0].rule_name, "high_err");
    }

    #[test]
    fn test_alert_manager_no_fire_below_threshold() {
        let mut mgr = AlertManager::new();
        mgr.add_rule(AlertRule::new(
            "high_err",
            "errors",
            AlertOp::GreaterThan,
            10.0,
            AlertSeverity::Warning,
            "errors",
        ));
        let mut c = MetricsCollector::new(true);
        c.increment_counter("errors", 5.0);
        mgr.evaluate(&c);
        assert!(mgr.fired_alerts().is_empty());
    }

    #[test]
    fn test_alert_manager_clear() {
        let mut mgr = AlertManager::new();
        mgr.add_rule(AlertRule::new("x", "v", AlertOp::GreaterThan, 0.0, AlertSeverity::Info, "m"));
        let mut c = MetricsCollector::new(true);
        c.increment_counter("v", 1.0);
        mgr.evaluate(&c);
        assert_eq!(mgr.fired_alerts().len(), 1);
        mgr.clear_alerts();
        assert!(mgr.fired_alerts().is_empty());
    }

    #[test]
    fn test_alert_manager_rules_accessor() {
        let mut mgr = AlertManager::new();
        mgr.add_rule(AlertRule::new(
            "r1",
            "m1",
            AlertOp::GreaterThan,
            1.0,
            AlertSeverity::Info,
            "msg",
        ));
        assert_eq!(mgr.rules().len(), 1);
    }

    #[test]
    fn test_alert_manager_gauge_trigger() {
        let mut mgr = AlertManager::new();
        mgr.add_rule(AlertRule::new(
            "low_mem",
            "free_mb",
            AlertOp::LessThan,
            256.0,
            AlertSeverity::Critical,
            "Low memory",
        ));
        let mut c = MetricsCollector::new(true);
        c.set_gauge("free_mb", 100.0);
        mgr.evaluate(&c);
        assert_eq!(mgr.fired_alerts().len(), 1);
        assert_eq!(mgr.fired_alerts()[0].severity, AlertSeverity::Critical);
    }

    // -- RequestPhase ------------------------------------------------------

    #[test]
    fn test_request_phase_display() {
        assert_eq!(RequestPhase::Received.to_string(), "received");
        assert_eq!(RequestPhase::Processing.to_string(), "processing");
        assert_eq!(RequestPhase::Complete.to_string(), "complete");
        assert_eq!(RequestPhase::Failed.to_string(), "failed");
    }

    // -- RequestLogger -----------------------------------------------------

    #[test]
    fn test_request_logger_initial_phase() {
        let rl = RequestLogger::new("req-1");
        assert_eq!(rl.request_id(), "req-1");
        assert_eq!(rl.phases().len(), 1);
        assert_eq!(rl.current_phase(), Some(RequestPhase::Received));
    }

    #[test]
    fn test_request_logger_record_phases() {
        let mut rl = RequestLogger::new("req-2");
        rl.record_phase(RequestPhase::Validated);
        rl.record_phase(RequestPhase::Processing);
        rl.record_phase(RequestPhase::Complete);
        assert_eq!(rl.phases().len(), 4);
        assert_eq!(rl.current_phase(), Some(RequestPhase::Complete));
    }

    #[test]
    fn test_request_logger_phase_with_detail() {
        let mut rl = RequestLogger::new("req-3");
        rl.record_phase_with_detail(RequestPhase::Failed, "OOM during attention");
        let last = rl.phases().last().unwrap();
        assert_eq!(last.phase, RequestPhase::Failed);
        assert_eq!(last.detail.as_deref(), Some("OOM during attention"));
    }

    #[test]
    fn test_request_logger_elapsed() {
        let rl = RequestLogger::new("req-4");
        // elapsed should be very small but non-negative
        assert!(rl.elapsed_ms() < 1000);
    }

    #[test]
    fn test_request_logger_to_json() {
        let mut rl = RequestLogger::new("req-5");
        rl.record_phase(RequestPhase::Complete);
        let json = rl.to_json();
        assert!(json.contains("\"request_id\":\"req-5\""));
        assert!(json.contains("\"phase\":\"Complete\""));
    }

    // -- AuditLog ----------------------------------------------------------

    #[test]
    fn test_audit_log_empty() {
        let log = AuditLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_audit_log_record() {
        let mut log = AuditLog::new();
        log.record("inference", "bitnet-2b", "user-1", "req-1");
        assert_eq!(log.len(), 1);
        assert_eq!(log.entries()[0].action, "inference");
    }

    #[test]
    fn test_audit_log_record_with_metadata() {
        let mut log = AuditLog::new();
        let mut meta = HashMap::new();
        meta.insert("tokens".to_string(), "128".to_string());
        log.record_with_metadata("inference", "bitnet-2b", "user-1", "req-1", meta);
        assert_eq!(log.entries()[0].metadata.get("tokens").unwrap(), "128");
    }

    #[test]
    fn test_audit_log_filter_by_action() {
        let mut log = AuditLog::new();
        log.record("inference", "m1", "u1", "r1");
        log.record("validation", "m1", "u1", "r2");
        log.record("inference", "m2", "u2", "r3");
        let inf = log.filter_by_action("inference");
        assert_eq!(inf.len(), 2);
    }

    #[test]
    fn test_audit_log_filter_by_model() {
        let mut log = AuditLog::new();
        log.record("inference", "bitnet-2b", "u1", "r1");
        log.record("inference", "bitnet-7b", "u1", "r2");
        let filtered = log.filter_by_model("bitnet-2b");
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_audit_log_to_json() {
        let mut log = AuditLog::new();
        log.record("load", "m", "u", "r");
        let json = log.to_json();
        assert!(json.contains("\"action\":\"load\""));
    }

    #[test]
    fn test_audit_log_immutability_ordering() {
        let mut log = AuditLog::new();
        log.record("a", "m", "u", "r1");
        log.record("b", "m", "u", "r2");
        log.record("c", "m", "u", "r3");
        assert_eq!(log.entries()[0].action, "a");
        assert_eq!(log.entries()[1].action, "b");
        assert_eq!(log.entries()[2].action, "c");
    }

    // -- LogRotation -------------------------------------------------------

    #[test]
    fn test_rotation_size_policy_no_rotate() {
        let rot = LogRotation::new(RotationPolicy::Size { max_bytes: 1024 });
        assert!(!rot.should_rotate());
        assert_eq!(rot.rotation_count(), 0);
    }

    #[test]
    fn test_rotation_size_policy_triggers() {
        let mut rot = LogRotation::new(RotationPolicy::Size { max_bytes: 100 });
        rot.record_write(101);
        assert!(rot.should_rotate());
    }

    #[test]
    fn test_rotation_resets_on_rotate() {
        let mut rot = LogRotation::new(RotationPolicy::Size { max_bytes: 100 });
        rot.record_write(150);
        assert!(rot.should_rotate());
        rot.rotate();
        assert!(!rot.should_rotate());
        assert_eq!(rot.rotation_count(), 1);
        assert_eq!(rot.current_bytes(), 0);
    }

    #[test]
    fn test_rotation_time_policy_no_rotate_initially() {
        let rot = LogRotation::new(RotationPolicy::Time { max_age_ms: 60_000 });
        assert!(!rot.should_rotate());
    }

    #[test]
    fn test_rotation_current_bytes_accumulates() {
        let mut rot = LogRotation::new(RotationPolicy::Size { max_bytes: 1000 });
        rot.record_write(100);
        rot.record_write(200);
        assert_eq!(rot.current_bytes(), 300);
    }

    #[test]
    fn test_rotation_policy_accessor() {
        let rot = LogRotation::new(RotationPolicy::Size { max_bytes: 512 });
        assert_eq!(rot.policy(), RotationPolicy::Size { max_bytes: 512 });
    }

    #[test]
    fn test_rotation_multiple_cycles() {
        let mut rot = LogRotation::new(RotationPolicy::Size { max_bytes: 50 });
        rot.record_write(60);
        rot.rotate();
        rot.record_write(55);
        rot.rotate();
        assert_eq!(rot.rotation_count(), 2);
    }

    // -- Integration / cross-component ------------------------------------

    #[test]
    fn test_end_to_end_request_with_metrics() {
        let mut logger = StructuredLogger::new(default_config());
        let mut collector = MetricsCollector::new(true);
        let mut req = RequestLogger::new("e2e-1");

        logger.log_with_context(LogLevel::Info, "Request received", test_context());
        collector.increment_counter("requests_total", 1.0);

        req.record_phase(RequestPhase::Validated);
        req.record_phase(RequestPhase::Processing);
        collector.set_gauge("active_requests", 1.0);

        req.record_phase(RequestPhase::Complete);
        collector.set_gauge("active_requests", 0.0);
        collector.observe_histogram("latency_ms", 42.0);

        assert_eq!(req.current_phase(), Some(RequestPhase::Complete));
        assert_eq!(collector.counter_value("requests_total"), Some(1.0));
        assert_eq!(collector.gauge_value("active_requests"), Some(0.0));
        assert_eq!(logger.entries().len(), 1);
    }

    #[test]
    fn test_alert_on_collected_metrics() {
        let mut collector = MetricsCollector::new(true);
        collector.increment_counter("errors", 25.0);

        let mut mgr = AlertManager::new();
        mgr.add_rule(AlertRule::new(
            "high_errors",
            "errors",
            AlertOp::GreaterThan,
            20.0,
            AlertSeverity::Critical,
            "Error rate exceeded",
        ));
        mgr.evaluate(&collector);

        assert_eq!(mgr.fired_alerts().len(), 1);
        let alert = &mgr.fired_alerts()[0];
        assert_eq!(alert.metric_value, 25.0);
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_audit_log_with_request_lifecycle() {
        let mut audit = AuditLog::new();
        let mut req = RequestLogger::new("audit-req");

        audit.record("inference_start", "bitnet-2b", "u1", "audit-req");
        req.record_phase(RequestPhase::Processing);
        req.record_phase(RequestPhase::Complete);
        audit.record("inference_end", "bitnet-2b", "u1", "audit-req");

        assert_eq!(audit.len(), 2);
        assert_eq!(audit.filter_by_action("inference_start").len(), 1);
    }

    #[test]
    fn test_metrics_export_roundtrip() {
        let mut c = MetricsCollector::new(true);
        c.increment_counter("tok_gen", 100.0);
        c.set_gauge("gpu_util", 85.5);

        let prom = PrometheusExporter.export(&c.collect());
        assert!(prom.contains("tok_gen"));
        assert!(prom.contains("gpu_util"));

        let json = JsonExporter.export(&c.collect());
        assert!(json.contains("tok_gen"));
        assert!(json.contains("gpu_util"));
    }

    #[test]
    fn test_log_rotation_with_structured_logger() {
        let mut logger = StructuredLogger::new(default_config());
        let mut rot = LogRotation::new(RotationPolicy::Size { max_bytes: 200 });

        for i in 0..5 {
            let msg = format!("entry {i}");
            logger.log(LogLevel::Info, &msg);
            rot.record_write(msg.len());
        }

        if rot.should_rotate() {
            let _drained = logger.drain();
            rot.rotate();
        }

        // After rotation, logger should be drained
        assert!(logger.entries().is_empty() || !rot.should_rotate());
    }
}
