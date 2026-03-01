//! GPU hardware performance counters for Intel GPUs.
//!
//! Reads Intel GPU performance metrics via sysfs (`/sys/class/drm/card*/`)
//! with configurable sampling intervals and summary statistics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Default sampling interval for performance counter reads.
const DEFAULT_SAMPLING_INTERVAL: Duration = Duration::from_millis(100);

/// Identifier for a specific performance counter metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerfMetric {
    /// Percentage of execution units actively computing.
    EuActive,
    /// Percentage of execution units stalled.
    EuStall,
    /// L3 cache hit count.
    L3CacheHit,
    /// L3 cache miss count.
    L3CacheMiss,
    /// Memory read bandwidth in bytes/sec.
    MemoryReadBandwidth,
    /// Memory write bandwidth in bytes/sec.
    MemoryWriteBandwidth,
}

impl PerfMetric {
    /// Sysfs filename for this metric (relative to card directory).
    pub fn sysfs_filename(&self) -> &'static str {
        match self {
            PerfMetric::EuActive => "gt/gt0/rcs0/eu_active",
            PerfMetric::EuStall => "gt/gt0/rcs0/eu_stall",
            PerfMetric::L3CacheHit => "gt/gt0/l3_hit",
            PerfMetric::L3CacheMiss => "gt/gt0/l3_miss",
            PerfMetric::MemoryReadBandwidth => "gt/gt0/mem_read_bw",
            PerfMetric::MemoryWriteBandwidth => "gt/gt0/mem_write_bw",
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            PerfMetric::EuActive => "EU Active %",
            PerfMetric::EuStall => "EU Stall %",
            PerfMetric::L3CacheHit => "L3 Cache Hits",
            PerfMetric::L3CacheMiss => "L3 Cache Misses",
            PerfMetric::MemoryReadBandwidth => "Mem Read BW (B/s)",
            PerfMetric::MemoryWriteBandwidth => "Mem Write BW (B/s)",
        }
    }

    /// All known metrics.
    pub fn all() -> &'static [PerfMetric] {
        &[
            PerfMetric::EuActive,
            PerfMetric::EuStall,
            PerfMetric::L3CacheHit,
            PerfMetric::L3CacheMiss,
            PerfMetric::MemoryReadBandwidth,
            PerfMetric::MemoryWriteBandwidth,
        ]
    }
}

/// A single performance counter sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfSample {
    /// Metric that was sampled.
    pub metric: PerfMetric,
    /// Raw counter value.
    pub value: f64,
    /// Time elapsed since measurement window start.
    pub elapsed: Duration,
}

/// Summary statistics over a measurement window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfSummary {
    /// Which metric this summarises.
    pub metric: PerfMetric,
    /// Minimum value observed.
    pub min: f64,
    /// Maximum value observed.
    pub max: f64,
    /// Arithmetic mean.
    pub avg: f64,
    /// 99th-percentile value.
    pub p99: f64,
    /// Number of samples collected.
    pub sample_count: usize,
}

/// Trait for reading sysfs counter values (allows mocking in tests).
pub trait SysfsReader: Send + Sync {
    /// Read a counter value from the given path. Returns `None` if unavailable.
    fn read_counter(&self, path: &Path) -> Option<f64>;
}

/// Real sysfs reader that reads from the filesystem.
#[derive(Debug, Default)]
pub struct RealSysfsReader;

impl SysfsReader for RealSysfsReader {
    fn read_counter(&self, path: &Path) -> Option<f64> {
        let content = std::fs::read_to_string(path).ok()?;
        content.trim().parse::<f64>().ok()
    }
}

/// Configuration for the performance counter collector.
#[derive(Debug, Clone)]
pub struct PerfCounterConfig {
    /// Sampling interval between reads.
    pub sampling_interval: Duration,
    /// Which metrics to collect.
    pub metrics: Vec<PerfMetric>,
    /// Sysfs base path (e.g. `/sys/class/drm/card0`).
    pub sysfs_base: PathBuf,
}

impl Default for PerfCounterConfig {
    fn default() -> Self {
        Self {
            sampling_interval: DEFAULT_SAMPLING_INTERVAL,
            metrics: PerfMetric::all().to_vec(),
            sysfs_base: PathBuf::from("/sys/class/drm/card0"),
        }
    }
}

/// Collects GPU performance counter samples and computes summary statistics.
pub struct PerfCounterCollector<R: SysfsReader = RealSysfsReader> {
    config: PerfCounterConfig,
    reader: R,
    samples: HashMap<PerfMetric, Vec<PerfSample>>,
    start: Option<Instant>,
}

impl PerfCounterCollector<RealSysfsReader> {
    /// Create a collector with real sysfs reads.
    pub fn new(config: PerfCounterConfig) -> Self {
        Self::with_reader(config, RealSysfsReader)
    }
}

impl<R: SysfsReader> PerfCounterCollector<R> {
    /// Create a collector with a custom reader (for testing).
    pub fn with_reader(config: PerfCounterConfig, reader: R) -> Self {
        let samples = config.metrics.iter().map(|m| (*m, Vec::new())).collect();
        Self {
            config,
            reader,
            samples,
            start: None,
        }
    }

    /// Begin a measurement window.
    pub fn start(&mut self) {
        self.samples.values_mut().for_each(|v| v.clear());
        self.start = Some(Instant::now());
    }

    /// Take one sample of all configured metrics.
    pub fn sample(&mut self) {
        let elapsed = self
            .start
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);

        for metric in &self.config.metrics {
            let path = self.config.sysfs_base.join(metric.sysfs_filename());
            if let Some(value) = self.reader.read_counter(&path) {
                self.samples.entry(*metric).or_default().push(PerfSample {
                    metric: *metric,
                    value,
                    elapsed,
                });
            }
        }
    }

    /// Number of samples collected for the given metric.
    pub fn sample_count(&self, metric: PerfMetric) -> usize {
        self.samples.get(&metric).map_or(0, |v| v.len())
    }

    /// Return all raw samples for a metric.
    pub fn raw_samples(&self, metric: PerfMetric) -> &[PerfSample] {
        self.samples
            .get(&metric)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Compute summary statistics for each metric.
    pub fn summarize(&self) -> Vec<PerfSummary> {
        self.config
            .metrics
            .iter()
            .filter_map(|metric| {
                let samples = self.samples.get(metric)?;
                if samples.is_empty() {
                    return None;
                }
                Some(compute_summary(*metric, samples))
            })
            .collect()
    }

    /// Compute summary for a single metric.
    pub fn summarize_metric(&self, metric: PerfMetric) -> Option<PerfSummary> {
        let samples = self.samples.get(&metric)?;
        if samples.is_empty() {
            return None;
        }
        Some(compute_summary(metric, samples))
    }

    /// Export all summaries as JSON.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.summarize())
    }

    /// Current sampling interval.
    pub fn sampling_interval(&self) -> Duration {
        self.config.sampling_interval
    }
}

fn compute_summary(metric: PerfMetric, samples: &[PerfSample]) -> PerfSummary {
    let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
    let n = values.len();

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let avg = values.iter().sum::<f64>() / n as f64;

    let mut sorted = values;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99_idx = ((n as f64) * 0.99).ceil() as usize;
    let p99 = sorted[p99_idx.min(n) - 1];

    PerfSummary {
        metric,
        min,
        max,
        avg,
        p99,
        sample_count: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Mock reader that returns pre-configured values.
    struct MockSysfsReader {
        values: Mutex<HashMap<PathBuf, Vec<f64>>>,
    }

    impl MockSysfsReader {
        fn new() -> Self {
            Self {
                values: Mutex::new(HashMap::new()),
            }
        }

        fn set_values(&self, path: PathBuf, vals: Vec<f64>) {
            self.values.lock().unwrap().insert(path, vals);
        }
    }

    impl SysfsReader for MockSysfsReader {
        fn read_counter(&self, path: &Path) -> Option<f64> {
            let mut map = self.values.lock().unwrap();
            let vals = map.get_mut(path)?;
            if vals.is_empty() {
                None
            } else {
                Some(vals.remove(0))
            }
        }
    }

    fn test_config() -> PerfCounterConfig {
        PerfCounterConfig {
            sampling_interval: Duration::from_millis(10),
            metrics: vec![PerfMetric::EuActive, PerfMetric::EuStall],
            sysfs_base: PathBuf::from("/mock/drm/card0"),
        }
    }

    #[test]
    fn test_perf_metric_all() {
        let all = PerfMetric::all();
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn test_perf_metric_labels() {
        assert_eq!(PerfMetric::EuActive.label(), "EU Active %");
        assert_eq!(PerfMetric::L3CacheMiss.label(), "L3 Cache Misses");
        assert_eq!(
            PerfMetric::MemoryReadBandwidth.label(),
            "Mem Read BW (B/s)"
        );
    }

    #[test]
    fn test_collector_sample_and_summarize() {
        let config = test_config();
        let reader = MockSysfsReader::new();
        let eu_path =
            PathBuf::from("/mock/drm/card0/gt/gt0/rcs0/eu_active");
        let stall_path =
            PathBuf::from("/mock/drm/card0/gt/gt0/rcs0/eu_stall");

        reader.set_values(eu_path, vec![50.0, 60.0, 70.0]);
        reader.set_values(stall_path, vec![10.0, 20.0, 30.0]);

        let mut collector =
            PerfCounterCollector::with_reader(config, reader);
        collector.start();

        for _ in 0..3 {
            collector.sample();
        }

        assert_eq!(collector.sample_count(PerfMetric::EuActive), 3);
        assert_eq!(collector.sample_count(PerfMetric::EuStall), 3);

        let summary = collector.summarize();
        assert_eq!(summary.len(), 2);

        let eu_summary = summary
            .iter()
            .find(|s| s.metric == PerfMetric::EuActive)
            .unwrap();
        assert_eq!(eu_summary.min, 50.0);
        assert_eq!(eu_summary.max, 70.0);
        assert!((eu_summary.avg - 60.0).abs() < 1e-10);
        assert_eq!(eu_summary.sample_count, 3);
    }

    #[test]
    fn test_collector_empty_summary() {
        let config = test_config();
        let reader = MockSysfsReader::new();
        let mut collector =
            PerfCounterCollector::with_reader(config, reader);
        collector.start();

        let summary = collector.summarize();
        assert!(summary.is_empty());
    }

    #[test]
    fn test_collector_single_sample_p99() {
        let config = PerfCounterConfig {
            metrics: vec![PerfMetric::L3CacheHit],
            sysfs_base: PathBuf::from("/mock/drm/card0"),
            ..Default::default()
        };
        let reader = MockSysfsReader::new();
        let path = PathBuf::from("/mock/drm/card0/gt/gt0/l3_hit");
        reader.set_values(path, vec![42.0]);

        let mut collector =
            PerfCounterCollector::with_reader(config, reader);
        collector.start();
        collector.sample();

        let summary =
            collector.summarize_metric(PerfMetric::L3CacheHit).unwrap();
        assert_eq!(summary.min, 42.0);
        assert_eq!(summary.max, 42.0);
        assert_eq!(summary.p99, 42.0);
        assert_eq!(summary.sample_count, 1);
    }

    #[test]
    fn test_collector_missing_metric_returns_none() {
        let config = test_config();
        let reader = MockSysfsReader::new();
        let collector =
            PerfCounterCollector::with_reader(config, reader);

        assert!(
            collector.summarize_metric(PerfMetric::L3CacheMiss).is_none()
        );
    }

    #[test]
    fn test_collector_start_clears_samples() {
        let config = test_config();
        let reader = MockSysfsReader::new();
        let eu_path =
            PathBuf::from("/mock/drm/card0/gt/gt0/rcs0/eu_active");
        reader.set_values(eu_path, vec![10.0, 20.0]);

        let mut collector =
            PerfCounterCollector::with_reader(config, reader);
        collector.start();
        collector.sample();
        assert_eq!(collector.sample_count(PerfMetric::EuActive), 1);

        collector.start(); // clears
        assert_eq!(collector.sample_count(PerfMetric::EuActive), 0);
    }

    #[test]
    fn test_json_export() {
        let config = PerfCounterConfig {
            metrics: vec![PerfMetric::EuActive],
            sysfs_base: PathBuf::from("/mock/drm/card0"),
            ..Default::default()
        };
        let reader = MockSysfsReader::new();
        let path =
            PathBuf::from("/mock/drm/card0/gt/gt0/rcs0/eu_active");
        reader.set_values(path, vec![55.0, 65.0]);

        let mut collector =
            PerfCounterCollector::with_reader(config, reader);
        collector.start();
        collector.sample();
        collector.sample();

        let json = collector.export_json().unwrap();
        let parsed: Vec<PerfSummary> =
            serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].metric, PerfMetric::EuActive);
        assert_eq!(parsed[0].sample_count, 2);
    }

    #[test]
    fn test_default_config() {
        let config = PerfCounterConfig::default();
        assert_eq!(config.sampling_interval, Duration::from_millis(100));
        assert_eq!(config.metrics.len(), 6);
        assert_eq!(
            config.sysfs_base,
            PathBuf::from("/sys/class/drm/card0")
        );
    }

    #[test]
    fn test_raw_samples_access() {
        let config = PerfCounterConfig {
            metrics: vec![PerfMetric::MemoryReadBandwidth],
            sysfs_base: PathBuf::from("/mock/drm/card0"),
            ..Default::default()
        };
        let reader = MockSysfsReader::new();
        let path = PathBuf::from("/mock/drm/card0/gt/gt0/mem_read_bw");
        reader.set_values(path, vec![1000.0, 2000.0]);

        let mut collector =
            PerfCounterCollector::with_reader(config, reader);
        collector.start();
        collector.sample();
        collector.sample();

        let raw =
            collector.raw_samples(PerfMetric::MemoryReadBandwidth);
        assert_eq!(raw.len(), 2);
        assert_eq!(raw[0].value, 1000.0);
        assert_eq!(raw[1].value, 2000.0);
    }
}
