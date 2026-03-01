//! OpenCL telemetry and metrics collection for A770 GPU inference monitoring.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// TelemetryError
// ---------------------------------------------------------------------------

/// Errors that can occur during telemetry operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TelemetryError {
    /// The collector is disabled and cannot record metrics.
    CollectorDisabled,
    /// An invalid metric value was provided.
    InvalidValue(String),
    /// Export failed.
    ExportFailed(String),
}

impl fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CollectorDisabled => write!(f, "telemetry collector is disabled"),
            Self::InvalidValue(msg) => write!(f, "invalid metric value: {msg}"),
            Self::ExportFailed(msg) => write!(f, "export failed: {msg}"),
        }
    }
}

impl std::error::Error for TelemetryError {}

// ---------------------------------------------------------------------------
// MetricType
// ---------------------------------------------------------------------------

/// The kind of metric being recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    KernelExecution,
    MemoryTransfer,
    BufferAllocation,
    QueueSubmit,
    CompilationTime,
    Throughput,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelExecution => write!(f, "KernelExecution"),
            Self::MemoryTransfer => write!(f, "MemoryTransfer"),
            Self::BufferAllocation => write!(f, "BufferAllocation"),
            Self::QueueSubmit => write!(f, "QueueSubmit"),
            Self::CompilationTime => write!(f, "CompilationTime"),
            Self::Throughput => write!(f, "Throughput"),
        }
    }
}

// ---------------------------------------------------------------------------
// MetricSample
// ---------------------------------------------------------------------------

/// A single recorded metric data point.
#[derive(Debug, Clone)]
pub struct MetricSample {
    pub metric_type: MetricType,
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub timestamp_ns: u64,
}

impl MetricSample {
    pub fn new(
        metric_type: MetricType,
        name: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
    ) -> Self {
        Self { metric_type, name: name.into(), value, unit: unit.into(), timestamp_ns: now_ns() }
    }
}

// ---------------------------------------------------------------------------
// MetricAggregation
// ---------------------------------------------------------------------------

/// Aggregated statistics over a set of metric samples.
#[derive(Debug, Clone)]
pub struct MetricAggregation {
    pub count: u64,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    values: Vec<f64>,
}

impl MetricAggregation {
    fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                sum: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
                values: Vec::new(),
            };
        }
        let count = values.len() as u64;
        let sum: f64 = values.iter().sum();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self { count, sum, min, max, values: sorted }
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }

    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let sum_sq: f64 = self.values.iter().map(|v| (v - mean).powi(2)).sum();
        sum_sq / self.count as f64
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Estimate the p-th percentile (p in 0.0..=1.0) using nearest-rank.
    pub fn percentile_estimate(&self, p: f64) -> f64 {
        if self.values.is_empty() || !(0.0..=1.0).contains(&p) {
            return 0.0;
        }
        if self.values.len() == 1 {
            return self.values[0];
        }
        let idx = ((p * (self.values.len() as f64 - 1.0)).round()) as usize;
        let idx = idx.min(self.values.len() - 1);
        self.values[idx]
    }
}

// ---------------------------------------------------------------------------
// TelemetryCollector
// ---------------------------------------------------------------------------

/// Collects metric samples with ring-buffer eviction.
pub struct TelemetryCollector {
    samples: Vec<MetricSample>,
    enabled: bool,
    max_samples: usize,
}

impl TelemetryCollector {
    pub fn new(max_samples: usize) -> Self {
        Self { samples: Vec::with_capacity(max_samples.min(1024)), enabled: true, max_samples }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record a pre-built sample. When the buffer is full the oldest sample is evicted.
    pub fn record(&mut self, sample: MetricSample) -> Result<(), TelemetryError> {
        if !self.enabled {
            return Err(TelemetryError::CollectorDisabled);
        }
        if sample.value.is_nan() {
            return Err(TelemetryError::InvalidValue("NaN value".into()));
        }
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(sample);
        Ok(())
    }

    /// Convenience: record a kernel-execution timing sample from a duration in nanoseconds.
    pub fn record_timed(
        &mut self,
        name: impl Into<String>,
        duration_ns: u64,
    ) -> Result<(), TelemetryError> {
        let ms = duration_ns as f64 / 1_000_000.0;
        let sample = MetricSample {
            metric_type: MetricType::KernelExecution,
            name: name.into(),
            value: ms,
            unit: "ms".into(),
            timestamp_ns: now_ns(),
        };
        self.record(sample)
    }

    /// Aggregate all samples matching a given `MetricType`.
    pub fn aggregate(&self, metric_type: MetricType) -> MetricAggregation {
        let vals: Vec<f64> =
            self.samples.iter().filter(|s| s.metric_type == metric_type).map(|s| s.value).collect();
        MetricAggregation::from_values(&vals)
    }

    /// Aggregate all samples whose name matches exactly.
    pub fn aggregate_by_name(&self, name: &str) -> MetricAggregation {
        let vals: Vec<f64> =
            self.samples.iter().filter(|s| s.name == name).map(|s| s.value).collect();
        MetricAggregation::from_values(&vals)
    }

    /// Return samples recorded at or after `timestamp_ns`.
    pub fn samples_since(&self, timestamp_ns: u64) -> &[MetricSample] {
        match self.samples.iter().position(|s| s.timestamp_ns >= timestamp_ns) {
            Some(pos) => &self.samples[pos..],
            None => &[],
        }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn is_full(&self) -> bool {
        self.samples.len() >= self.max_samples
    }

    /// Export all samples as a JSON string (hand-rolled, no serde dependency).
    pub fn export_json(&self) -> String {
        let mut out = String::from("{\"samples\":[");
        for (i, s) in self.samples.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"metric_type\":\"{}\",\"name\":\"{}\",\"value\":{},\"unit\":\"{}\",\"timestamp_ns\":{}}}",
                s.metric_type,
                escape_json(&s.name),
                format_f64(s.value),
                escape_json(&s.unit),
                s.timestamp_ns,
            ));
        }
        out.push_str("]}");
        out
    }

    /// Human-readable summary with top metrics per type.
    pub fn summary(&self) -> String {
        if self.samples.is_empty() {
            return "No telemetry samples recorded.".into();
        }
        let mut out = format!("Telemetry Summary ({} samples)\n", self.samples.len());
        out.push_str(&format!("{}\n", "-".repeat(40)));

        let types = [
            MetricType::KernelExecution,
            MetricType::MemoryTransfer,
            MetricType::BufferAllocation,
            MetricType::QueueSubmit,
            MetricType::CompilationTime,
            MetricType::Throughput,
        ];
        for mt in &types {
            let agg = self.aggregate(*mt);
            if agg.count == 0 {
                continue;
            }
            out.push_str(&format!(
                "{}: count={}, mean={:.3}, min={:.3}, max={:.3}, stddev={:.3}\n",
                mt,
                agg.count,
                agg.mean(),
                agg.min,
                agg.max,
                agg.std_dev(),
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// InferenceMetrics
// ---------------------------------------------------------------------------

/// High-level inference-specific metrics.
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    pub tokens_generated: u64,
    pub total_inference_ns: u64,
    pub total_kernel_ns: u64,
    pub total_transfer_ns: u64,
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokens per second (0.0 when no inference time recorded).
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_inference_ns == 0 {
            return 0.0;
        }
        self.tokens_generated as f64 / (self.total_inference_ns as f64 / 1e9)
    }

    /// Fraction of total inference time spent in GPU kernels (0.0–1.0).
    pub fn gpu_utilization(&self) -> f64 {
        if self.total_inference_ns == 0 {
            return 0.0;
        }
        (self.total_kernel_ns + self.total_transfer_ns) as f64 / self.total_inference_ns as f64
    }

    /// Percentage of total inference that is kernel execution only.
    pub fn kernel_overhead_pct(&self) -> f64 {
        if self.total_inference_ns == 0 {
            return 0.0;
        }
        (self.total_kernel_ns as f64 / self.total_inference_ns as f64) * 100.0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

fn format_f64(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 { format!("{v:.1}") } else { format!("{v}") }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- MetricType Display ---------------------------------------------------

    #[test]
    fn metric_type_display() {
        assert_eq!(MetricType::KernelExecution.to_string(), "KernelExecution");
        assert_eq!(MetricType::MemoryTransfer.to_string(), "MemoryTransfer");
        assert_eq!(MetricType::BufferAllocation.to_string(), "BufferAllocation");
        assert_eq!(MetricType::QueueSubmit.to_string(), "QueueSubmit");
        assert_eq!(MetricType::CompilationTime.to_string(), "CompilationTime");
        assert_eq!(MetricType::Throughput.to_string(), "Throughput");
    }

    // -- TelemetryError Display + Error ---------------------------------------

    #[test]
    fn telemetry_error_display() {
        let e = TelemetryError::CollectorDisabled;
        assert!(e.to_string().contains("disabled"));

        let e = TelemetryError::InvalidValue("bad".into());
        assert!(e.to_string().contains("bad"));

        let e = TelemetryError::ExportFailed("io".into());
        assert!(e.to_string().contains("io"));
    }

    #[test]
    fn telemetry_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(TelemetryError::CollectorDisabled);
        assert!(e.to_string().contains("disabled"));
    }

    // -- Record & retrieve ----------------------------------------------------

    #[test]
    fn record_and_retrieve_sample() {
        let mut col = TelemetryCollector::new(100);
        col.record(MetricSample::new(MetricType::KernelExecution, "matmul", 1.5, "ms")).unwrap();
        assert_eq!(col.sample_count(), 1);
        assert_eq!(col.samples[0].name, "matmul");
    }

    #[test]
    fn record_multiple_samples() {
        let mut col = TelemetryCollector::new(100);
        for i in 0..10 {
            col.record(MetricSample::new(MetricType::Throughput, "tok", i as f64, "tokens/s"))
                .unwrap();
        }
        assert_eq!(col.sample_count(), 10);
    }

    #[test]
    fn record_timed_converts_ns_to_ms() {
        let mut col = TelemetryCollector::new(100);
        col.record_timed("kern", 5_000_000).unwrap(); // 5 ms
        let s = &col.samples[0];
        assert!((s.value - 5.0).abs() < 1e-9);
        assert_eq!(s.unit, "ms");
        assert_eq!(s.metric_type, MetricType::KernelExecution);
    }

    // -- Ring buffer eviction -------------------------------------------------

    #[test]
    fn ring_buffer_evicts_oldest() {
        let mut col = TelemetryCollector::new(3);
        for i in 0..5 {
            col.record(MetricSample::new(
                MetricType::KernelExecution,
                format!("k{i}"),
                i as f64,
                "ms",
            ))
            .unwrap();
        }
        assert_eq!(col.sample_count(), 3);
        assert_eq!(col.samples[0].name, "k2");
        assert_eq!(col.samples[2].name, "k4");
    }

    #[test]
    fn is_full_reflects_capacity() {
        let mut col = TelemetryCollector::new(2);
        assert!(!col.is_full());
        col.record(MetricSample::new(MetricType::Throughput, "a", 1.0, "ms")).unwrap();
        assert!(!col.is_full());
        col.record(MetricSample::new(MetricType::Throughput, "b", 2.0, "ms")).unwrap();
        assert!(col.is_full());
    }

    #[test]
    fn ring_buffer_stays_at_max() {
        let mut col = TelemetryCollector::new(5);
        for i in 0..100 {
            col.record(MetricSample::new(MetricType::QueueSubmit, "q", i as f64, "ms")).unwrap();
        }
        assert_eq!(col.sample_count(), 5);
        assert!(col.is_full());
    }

    // -- Aggregation ----------------------------------------------------------

    #[test]
    fn aggregate_mean() {
        let mut col = TelemetryCollector::new(100);
        for v in [10.0, 20.0, 30.0] {
            col.record(MetricSample::new(MetricType::KernelExecution, "k", v, "ms")).unwrap();
        }
        let agg = col.aggregate(MetricType::KernelExecution);
        assert!((agg.mean() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_min_max() {
        let mut col = TelemetryCollector::new(100);
        for v in [5.0, 1.0, 9.0, 3.0] {
            col.record(MetricSample::new(MetricType::MemoryTransfer, "t", v, "GB/s")).unwrap();
        }
        let agg = col.aggregate(MetricType::MemoryTransfer);
        assert!((agg.min - 1.0).abs() < 1e-9);
        assert!((agg.max - 9.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_stddev() {
        let mut col = TelemetryCollector::new(100);
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            col.record(MetricSample::new(MetricType::KernelExecution, "k", v, "ms")).unwrap();
        }
        let agg = col.aggregate(MetricType::KernelExecution);
        // population stddev of that set = 2.0
        assert!((agg.std_dev() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_variance() {
        let mut col = TelemetryCollector::new(100);
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            col.record(MetricSample::new(MetricType::KernelExecution, "k", v, "ms")).unwrap();
        }
        let agg = col.aggregate(MetricType::KernelExecution);
        assert!((agg.variance() - 4.0).abs() < 1e-9);
    }

    // -- Aggregate by type and by name ----------------------------------------

    #[test]
    fn aggregate_by_type_filters_correctly() {
        let mut col = TelemetryCollector::new(100);
        col.record(MetricSample::new(MetricType::KernelExecution, "a", 10.0, "ms")).unwrap();
        col.record(MetricSample::new(MetricType::MemoryTransfer, "b", 20.0, "GB/s")).unwrap();
        col.record(MetricSample::new(MetricType::KernelExecution, "c", 30.0, "ms")).unwrap();

        let agg = col.aggregate(MetricType::KernelExecution);
        assert_eq!(agg.count, 2);
        assert!((agg.mean() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_by_name() {
        let mut col = TelemetryCollector::new(100);
        col.record(MetricSample::new(MetricType::KernelExecution, "matmul", 1.0, "ms")).unwrap();
        col.record(MetricSample::new(MetricType::KernelExecution, "softmax", 2.0, "ms")).unwrap();
        col.record(MetricSample::new(MetricType::KernelExecution, "matmul", 3.0, "ms")).unwrap();

        let agg = col.aggregate_by_name("matmul");
        assert_eq!(agg.count, 2);
        assert!((agg.mean() - 2.0).abs() < 1e-9);
    }

    // -- JSON export ----------------------------------------------------------

    #[test]
    fn export_json_empty() {
        let col = TelemetryCollector::new(100);
        assert_eq!(col.export_json(), "{\"samples\":[]}");
    }

    #[test]
    fn export_json_single_sample() {
        let mut col = TelemetryCollector::new(100);
        let mut s = MetricSample::new(MetricType::Throughput, "tok", 42.5, "tokens/s");
        s.timestamp_ns = 1000;
        col.record(s).unwrap();
        let json = col.export_json();
        assert!(json.contains("\"metric_type\":\"Throughput\""));
        assert!(json.contains("\"name\":\"tok\""));
        assert!(json.contains("\"value\":42.5"));
        assert!(json.contains("\"unit\":\"tokens/s\""));
        assert!(json.contains("\"timestamp_ns\":1000"));
    }

    #[test]
    fn export_json_escapes_special_chars() {
        let mut col = TelemetryCollector::new(100);
        let mut s = MetricSample::new(MetricType::KernelExecution, "kern\"special", 1.0, "ms");
        s.timestamp_ns = 0;
        col.record(s).unwrap();
        let json = col.export_json();
        assert!(json.contains("kern\\\"special"));
    }

    #[test]
    fn export_json_multiple_samples() {
        let mut col = TelemetryCollector::new(100);
        for i in 0..3 {
            let mut s = MetricSample::new(MetricType::QueueSubmit, format!("q{i}"), i as f64, "ms");
            s.timestamp_ns = i as u64;
            col.record(s).unwrap();
        }
        let json = col.export_json();
        // Should have commas between objects
        assert_eq!(json.matches("},{").count(), 2);
    }

    // -- Summary formatting ---------------------------------------------------

    #[test]
    fn summary_empty() {
        let col = TelemetryCollector::new(100);
        assert!(col.summary().contains("No telemetry"));
    }

    #[test]
    fn summary_contains_type_stats() {
        let mut col = TelemetryCollector::new(100);
        col.record(MetricSample::new(MetricType::KernelExecution, "k", 10.0, "ms")).unwrap();
        col.record(MetricSample::new(MetricType::KernelExecution, "k", 20.0, "ms")).unwrap();
        let summary = col.summary();
        assert!(summary.contains("KernelExecution"));
        assert!(summary.contains("count=2"));
    }

    // -- Enabled / disabled toggle --------------------------------------------

    #[test]
    fn disabled_collector_rejects_records() {
        let mut col = TelemetryCollector::new(100);
        col.set_enabled(false);
        let res = col.record(MetricSample::new(MetricType::Throughput, "a", 1.0, "ms"));
        assert_eq!(res, Err(TelemetryError::CollectorDisabled));
        assert_eq!(col.sample_count(), 0);
    }

    #[test]
    fn re_enable_collector() {
        let mut col = TelemetryCollector::new(100);
        col.set_enabled(false);
        assert!(!col.is_enabled());
        col.set_enabled(true);
        assert!(col.is_enabled());
        col.record(MetricSample::new(MetricType::Throughput, "a", 1.0, "ms")).unwrap();
        assert_eq!(col.sample_count(), 1);
    }

    // -- InferenceMetrics calculations ----------------------------------------

    #[test]
    fn tokens_per_second() {
        let m = InferenceMetrics {
            tokens_generated: 100,
            total_inference_ns: 2_000_000_000, // 2 s
            ..Default::default()
        };
        assert!((m.tokens_per_second() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn tokens_per_second_zero_time() {
        let m = InferenceMetrics::new();
        assert!((m.tokens_per_second() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn gpu_utilization() {
        let m = InferenceMetrics {
            tokens_generated: 10,
            total_inference_ns: 1_000_000_000,
            total_kernel_ns: 600_000_000,
            total_transfer_ns: 200_000_000,
        };
        assert!((m.gpu_utilization() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn gpu_utilization_zero_time() {
        let m = InferenceMetrics::new();
        assert!((m.gpu_utilization() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn kernel_overhead_pct() {
        let m = InferenceMetrics {
            tokens_generated: 10,
            total_inference_ns: 1_000_000_000,
            total_kernel_ns: 250_000_000,
            total_transfer_ns: 0,
        };
        assert!((m.kernel_overhead_pct() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn kernel_overhead_pct_zero_time() {
        let m = InferenceMetrics::new();
        assert!((m.kernel_overhead_pct() - 0.0).abs() < 1e-9);
    }

    // -- Empty collector edge cases -------------------------------------------

    #[test]
    fn aggregate_empty_collector() {
        let col = TelemetryCollector::new(100);
        let agg = col.aggregate(MetricType::KernelExecution);
        assert_eq!(agg.count, 0);
        assert!((agg.mean() - 0.0).abs() < 1e-9);
        assert!((agg.std_dev() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_by_name_empty() {
        let col = TelemetryCollector::new(100);
        let agg = col.aggregate_by_name("nonexistent");
        assert_eq!(agg.count, 0);
    }

    #[test]
    fn samples_since_empty() {
        let col = TelemetryCollector::new(100);
        assert!(col.samples_since(0).is_empty());
    }

    // -- Clear resets all state ------------------------------------------------

    #[test]
    fn clear_resets_state() {
        let mut col = TelemetryCollector::new(100);
        for i in 0..5 {
            col.record(MetricSample::new(MetricType::KernelExecution, "k", i as f64, "ms"))
                .unwrap();
        }
        assert_eq!(col.sample_count(), 5);
        col.clear();
        assert_eq!(col.sample_count(), 0);
        assert!(!col.is_full());
        assert!(col.export_json().contains("[]"));
    }

    // -- Concurrent-style recording patterns ----------------------------------

    #[test]
    fn interleaved_metric_types() {
        let mut col = TelemetryCollector::new(100);
        col.record(MetricSample::new(MetricType::KernelExecution, "k1", 1.0, "ms")).unwrap();
        col.record(MetricSample::new(MetricType::MemoryTransfer, "m1", 2.0, "GB/s")).unwrap();
        col.record(MetricSample::new(MetricType::KernelExecution, "k2", 3.0, "ms")).unwrap();
        col.record(MetricSample::new(MetricType::BufferAllocation, "b1", 4.0, "ms")).unwrap();
        col.record(MetricSample::new(MetricType::MemoryTransfer, "m2", 5.0, "GB/s")).unwrap();

        assert_eq!(col.aggregate(MetricType::KernelExecution).count, 2);
        assert_eq!(col.aggregate(MetricType::MemoryTransfer).count, 2);
        assert_eq!(col.aggregate(MetricType::BufferAllocation).count, 1);
    }

    #[test]
    fn rapid_recording_burst() {
        let mut col = TelemetryCollector::new(1000);
        for i in 0..500 {
            col.record(MetricSample::new(MetricType::QueueSubmit, "q", i as f64, "ms")).unwrap();
        }
        assert_eq!(col.sample_count(), 500);
        let agg = col.aggregate(MetricType::QueueSubmit);
        assert_eq!(agg.count, 500);
    }

    // -- Percentile estimation ------------------------------------------------

    #[test]
    fn percentile_p50() {
        let mut col = TelemetryCollector::new(100);
        for v in 1..=100 {
            col.record(MetricSample::new(MetricType::KernelExecution, "k", v as f64, "ms"))
                .unwrap();
        }
        let agg = col.aggregate(MetricType::KernelExecution);
        let p50 = agg.percentile_estimate(0.5);
        // median of 1..=100 ≈ 50 or 51
        assert!((p50 - 50.0).abs() <= 1.0);
    }

    #[test]
    fn percentile_p0_and_p100() {
        let mut col = TelemetryCollector::new(100);
        for v in [10.0, 20.0, 30.0] {
            col.record(MetricSample::new(MetricType::KernelExecution, "k", v, "ms")).unwrap();
        }
        let agg = col.aggregate(MetricType::KernelExecution);
        assert!((agg.percentile_estimate(0.0) - 10.0).abs() < 1e-9);
        assert!((agg.percentile_estimate(1.0) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_empty() {
        let agg = MetricAggregation::from_values(&[]);
        assert!((agg.percentile_estimate(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_single_value() {
        let agg = MetricAggregation::from_values(&[42.0]);
        assert!((agg.percentile_estimate(0.5) - 42.0).abs() < 1e-9);
    }

    // -- Large sample sets ----------------------------------------------------

    #[test]
    fn large_sample_set_aggregation() {
        let mut col = TelemetryCollector::new(10_000);
        for i in 0..10_000 {
            col.record(MetricSample::new(MetricType::Throughput, "tok", i as f64, "tokens/s"))
                .unwrap();
        }
        assert_eq!(col.sample_count(), 10_000);
        let agg = col.aggregate(MetricType::Throughput);
        assert_eq!(agg.count, 10_000);
        assert!((agg.min - 0.0).abs() < 1e-9);
        assert!((agg.max - 9999.0).abs() < 1e-9);
        // mean of 0..9999 = 4999.5
        assert!((agg.mean() - 4999.5).abs() < 1e-9);
    }

    #[test]
    fn large_ring_buffer_eviction() {
        let mut col = TelemetryCollector::new(100);
        for i in 0..1000 {
            col.record(MetricSample::new(MetricType::KernelExecution, "k", i as f64, "ms"))
                .unwrap();
        }
        assert_eq!(col.sample_count(), 100);
        // oldest should be 900
        assert!((col.samples[0].value - 900.0).abs() < 1e-9);
    }

    // -- samples_since --------------------------------------------------------

    #[test]
    fn samples_since_filters_correctly() {
        let mut col = TelemetryCollector::new(100);
        for ts in [100, 200, 300, 400, 500] {
            let mut s = MetricSample::new(MetricType::KernelExecution, "k", 1.0, "ms");
            s.timestamp_ns = ts;
            col.record(s).unwrap();
        }
        let since = col.samples_since(300);
        assert_eq!(since.len(), 3);
        assert_eq!(since[0].timestamp_ns, 300);
    }

    // -- Invalid value rejection -----------------------------------------------

    #[test]
    fn reject_nan_value() {
        let mut col = TelemetryCollector::new(100);
        let res = col.record(MetricSample::new(MetricType::Throughput, "a", f64::NAN, "ms"));
        assert!(matches!(res, Err(TelemetryError::InvalidValue(_))));
    }

    // -- MetricSample::new sets timestamp -------------------------------------

    #[test]
    fn sample_new_sets_timestamp() {
        let before = now_ns();
        let s = MetricSample::new(MetricType::KernelExecution, "k", 1.0, "ms");
        let after = now_ns();
        assert!(s.timestamp_ns >= before);
        assert!(s.timestamp_ns <= after);
    }
}
