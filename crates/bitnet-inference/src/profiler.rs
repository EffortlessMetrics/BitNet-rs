//! # Model Profiler
//!
//! Per-layer timing, memory tracking, and performance analysis for inference passes.
//! Produces structured reports and chrome://tracing-compatible JSON exports.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Configuration for the model profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Whether profiling is enabled.
    pub enabled: bool,
    /// Whether to record memory snapshots.
    pub record_memory: bool,
    /// Number of warmup iterations to discard before collecting samples.
    pub warmup_iterations: usize,
    /// Number of sample iterations to collect.
    pub sample_size: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self { enabled: true, record_memory: false, warmup_iterations: 0, sample_size: 1 }
    }
}

impl ProfilerConfig {
    /// Create a config with profiling disabled.
    #[must_use]
    pub fn disabled() -> Self {
        Self { enabled: false, ..Default::default() }
    }

    /// Set the number of warmup iterations.
    #[must_use]
    pub fn with_warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set the number of sample iterations.
    #[must_use]
    pub fn with_sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    /// Enable memory recording.
    #[must_use]
    pub fn with_memory(mut self, enabled: bool) -> Self {
        self.record_memory = enabled;
        self
    }
}

/// Timing and resource statistics for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProfile {
    /// Human-readable layer name (e.g. `"layer_0.attention"`).
    pub layer_name: String,
    /// Layer type tag (e.g. `"attention"`, `"ffn"`, `"norm"`).
    pub layer_type: String,
    /// Forward-pass wall time in microseconds.
    pub forward_time_us: f64,
    /// Backward-pass wall time in microseconds (0 for inference-only).
    pub backward_time_us: f64,
    /// Memory usage attributed to this layer in bytes.
    pub memory_bytes: u64,
    /// Estimated floating-point operations.
    pub flops_estimate: u64,
}

/// A single memory snapshot captured during profiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Label describing the point at which the snapshot was taken.
    pub label: String,
    /// Timestamp relative to the session start, in microseconds.
    pub timestamp_us: f64,
    /// Reported memory usage in bytes.
    pub memory_bytes: u64,
}

/// Tracks a single inference pass for profiling purposes.
pub struct ProfileSession {
    config: ProfilerConfig,
    start: Instant,
    layers: Vec<LayerProfile>,
    memory_snapshots: Vec<MemorySnapshot>,
    layer_stack: Vec<(String, String, Instant)>,
    iteration: usize,
}

impl ProfileSession {
    /// Create a new profile session with the given config.
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            start: Instant::now(),
            layers: Vec::new(),
            memory_snapshots: Vec::new(),
            layer_stack: Vec::new(),
            iteration: 0,
        }
    }

    /// Whether the current iteration is still in the warmup phase.
    pub fn is_warmup(&self) -> bool {
        self.iteration < self.config.warmup_iterations
    }

    /// Advance to the next iteration. Returns `true` if profiling is complete.
    pub fn next_iteration(&mut self) -> bool {
        self.iteration += 1;
        self.iteration >= self.config.warmup_iterations + self.config.sample_size
    }

    /// Current iteration index (0-based, includes warmup).
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Begin timing a layer. Call [`end_layer`] to record the result.
    pub fn begin_layer(&mut self, name: &str, layer_type: &str) {
        if !self.config.enabled {
            return;
        }
        self.layer_stack.push((name.to_owned(), layer_type.to_owned(), Instant::now()));
    }

    /// End timing for the most recently started layer and record its profile.
    ///
    /// Warmup iterations are silently discarded.
    pub fn end_layer(&mut self) {
        if !self.config.enabled {
            return;
        }
        if let Some((name, layer_type, start)) = self.layer_stack.pop() {
            if self.is_warmup() {
                return;
            }
            let elapsed = start.elapsed();
            self.layers.push(LayerProfile {
                layer_name: name,
                layer_type,
                forward_time_us: elapsed.as_secs_f64() * 1_000_000.0,
                backward_time_us: 0.0,
                memory_bytes: 0,
                flops_estimate: 0,
            });
        }
    }

    /// Record a memory snapshot with the given label and byte count.
    pub fn record_memory_snapshot(&mut self, label: &str, memory_bytes: u64) {
        if !self.config.enabled || !self.config.record_memory {
            return;
        }
        if self.is_warmup() {
            return;
        }
        let ts = self.start.elapsed().as_secs_f64() * 1_000_000.0;
        self.memory_snapshots.push(MemorySnapshot {
            label: label.to_owned(),
            timestamp_us: ts,
            memory_bytes,
        });
    }

    /// Consume the session and produce a [`ProfileReport`].
    pub fn generate_report(self) -> ProfileReport {
        let total_time_us: f64 = self.layers.iter().map(|l| l.forward_time_us).sum();

        // Aggregate per-layer statistics across samples.
        let mut by_name: HashMap<String, Vec<&LayerProfile>> = HashMap::new();
        for lp in &self.layers {
            by_name.entry(lp.layer_name.clone()).or_default().push(lp);
        }

        let mut per_layer_breakdown: Vec<LayerStats> = by_name
            .iter()
            .map(|(name, profiles)| {
                let times: Vec<f64> = profiles.iter().map(|p| p.forward_time_us).collect();
                let n = times.len() as f64;
                let mean = times.iter().sum::<f64>() / n;
                let min = times.iter().copied().fold(f64::INFINITY, f64::min);
                let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let variance = if times.len() > 1 {
                    times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0)
                } else {
                    0.0
                };
                let std_dev = variance.sqrt();
                let total_memory: u64 = profiles.iter().map(|p| p.memory_bytes).sum();
                let total_flops: u64 = profiles.iter().map(|p| p.flops_estimate).sum();
                LayerStats {
                    layer_name: name.clone(),
                    layer_type: profiles[0].layer_type.clone(),
                    mean_time_us: mean,
                    std_time_us: std_dev,
                    min_time_us: min,
                    max_time_us: max,
                    count: times.len(),
                    total_memory_bytes: total_memory,
                    total_flops,
                }
            })
            .collect();

        // Sort by mean time descending so bottlenecks are first.
        per_layer_breakdown.sort_by(|a, b| b.mean_time_us.partial_cmp(&a.mean_time_us).unwrap());

        let bottleneck_layers: Vec<String> = per_layer_breakdown
            .iter()
            .filter(|s| {
                let threshold = total_time_us * 0.1;
                s.mean_time_us * s.count as f64 > threshold
            })
            .map(|s| s.layer_name.clone())
            .collect();

        let memory_peak = self.memory_snapshots.iter().map(|s| s.memory_bytes).max().unwrap_or(0);

        let estimated_flops: u64 = self.layers.iter().map(|l| l.flops_estimate).sum();

        ProfileReport {
            total_time_us,
            per_layer_breakdown,
            bottleneck_layers,
            memory_peak,
            estimated_flops,
            memory_snapshots: self.memory_snapshots,
            layer_profiles: self.layers,
        }
    }
}

/// Aggregated statistics for a single layer name across samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    pub layer_name: String,
    pub layer_type: String,
    pub mean_time_us: f64,
    pub std_time_us: f64,
    pub min_time_us: f64,
    pub max_time_us: f64,
    pub count: usize,
    pub total_memory_bytes: u64,
    pub total_flops: u64,
}

/// Full profiling report for an inference session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Total wall time across all recorded layers in microseconds.
    pub total_time_us: f64,
    /// Per-layer aggregated statistics, sorted by mean time descending.
    pub per_layer_breakdown: Vec<LayerStats>,
    /// Layers that account for a significant share of total time.
    pub bottleneck_layers: Vec<String>,
    /// Peak memory observed across all snapshots.
    pub memory_peak: u64,
    /// Sum of all layer FLOP estimates.
    pub estimated_flops: u64,
    /// Raw memory snapshots.
    pub memory_snapshots: Vec<MemorySnapshot>,
    /// Raw layer profiles.
    pub layer_profiles: Vec<LayerProfile>,
}

impl ProfileReport {
    /// Serialize the recorded events into a chrome://tracing-compatible JSON string.
    pub fn export_chrome_trace(&self) -> String {
        let mut events: Vec<serde_json::Value> = Vec::new();
        let mut offset_us: f64 = 0.0;

        for lp in &self.layer_profiles {
            let begin = serde_json::json!({
                "name": lp.layer_name,
                "cat": lp.layer_type,
                "ph": "B",
                "ts": offset_us,
                "pid": 1,
                "tid": 1
            });
            let end = serde_json::json!({
                "name": lp.layer_name,
                "cat": lp.layer_type,
                "ph": "E",
                "ts": offset_us + lp.forward_time_us,
                "pid": 1,
                "tid": 1
            });
            events.push(begin);
            events.push(end);
            offset_us += lp.forward_time_us;
        }

        for snap in &self.memory_snapshots {
            let counter = serde_json::json!({
                "name": snap.label,
                "cat": "memory",
                "ph": "C",
                "ts": snap.timestamp_us,
                "pid": 1,
                "tid": 1,
                "args": { "memory_bytes": snap.memory_bytes }
            });
            events.push(counter);
        }

        serde_json::to_string_pretty(&events).unwrap_or_else(|_| "[]".to_owned())
    }
}

/// Convenience wrapper that owns a [`ProfileSession`] and produces reports.
pub struct ModelProfiler {
    config: ProfilerConfig,
}

impl ModelProfiler {
    /// Create a new profiler with the given configuration.
    pub fn new(config: ProfilerConfig) -> Self {
        Self { config }
    }

    /// Start a new profiling session.
    pub fn start_session(&self) -> ProfileSession {
        ProfileSession::new(self.config.clone())
    }

    /// Whether profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Access the profiler's configuration.
    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_layer_timing() {
        let profiler = ModelProfiler::new(ProfilerConfig::default());
        let mut session = profiler.start_session();

        session.begin_layer("layer_0.attention", "attention");
        thread::sleep(Duration::from_millis(5));
        session.end_layer();

        let report = session.generate_report();
        assert_eq!(report.layer_profiles.len(), 1);
        assert!(report.layer_profiles[0].forward_time_us > 0.0);
        assert_eq!(report.layer_profiles[0].layer_name, "layer_0.attention");
    }

    #[test]
    fn test_multiple_layers_in_sequence() {
        let profiler = ModelProfiler::new(ProfilerConfig::default());
        let mut session = profiler.start_session();

        for i in 0..3 {
            session.begin_layer(&format!("layer_{i}"), "transformer");
            thread::sleep(Duration::from_millis(2));
            session.end_layer();
        }

        let report = session.generate_report();
        assert_eq!(report.layer_profiles.len(), 3);
        assert!(report.total_time_us > 0.0);

        for (idx, lp) in report.layer_profiles.iter().enumerate() {
            assert_eq!(lp.layer_name, format!("layer_{idx}"));
        }
    }

    #[test]
    fn test_nested_layer_support() {
        let profiler = ModelProfiler::new(ProfilerConfig::default());
        let mut session = profiler.start_session();

        session.begin_layer("outer", "block");
        thread::sleep(Duration::from_millis(2));
        session.begin_layer("inner", "matmul");
        thread::sleep(Duration::from_millis(2));
        session.end_layer(); // inner
        session.end_layer(); // outer

        let report = session.generate_report();
        assert_eq!(report.layer_profiles.len(), 2);
        // Inner ends first (stack order).
        assert_eq!(report.layer_profiles[0].layer_name, "inner");
        assert_eq!(report.layer_profiles[1].layer_name, "outer");
        assert!(
            report.layer_profiles[1].forward_time_us >= report.layer_profiles[0].forward_time_us
        );
    }

    #[test]
    fn test_memory_snapshot_recording() {
        let config = ProfilerConfig::default().with_memory(true);
        let mut session = ProfileSession::new(config);

        session.record_memory_snapshot("before_attention", 1024);
        session.record_memory_snapshot("after_attention", 2048);
        session.record_memory_snapshot("peak", 4096);

        let report = session.generate_report();
        assert_eq!(report.memory_snapshots.len(), 3);
        assert_eq!(report.memory_peak, 4096);
    }

    #[test]
    fn test_memory_snapshot_disabled() {
        let config = ProfilerConfig::default(); // record_memory = false
        let mut session = ProfileSession::new(config);

        session.record_memory_snapshot("ignored", 999);

        let report = session.generate_report();
        assert!(report.memory_snapshots.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let profiler = ModelProfiler::new(ProfilerConfig::default());
        let mut session = profiler.start_session();

        session.begin_layer("attn", "attention");
        thread::sleep(Duration::from_millis(3));
        session.end_layer();

        session.begin_layer("ffn", "feedforward");
        thread::sleep(Duration::from_millis(3));
        session.end_layer();

        let report = session.generate_report();
        assert_eq!(report.per_layer_breakdown.len(), 2);
        assert!(report.total_time_us > 0.0);
        assert_eq!(report.estimated_flops, 0); // no flops set
    }

    #[test]
    fn test_chrome_trace_export() {
        let profiler = ModelProfiler::new(ProfilerConfig::default());
        let mut session = profiler.start_session();

        session.begin_layer("layer_0", "attention");
        thread::sleep(Duration::from_millis(1));
        session.end_layer();

        let report = session.generate_report();
        let trace = report.export_chrome_trace();

        let parsed: serde_json::Value = serde_json::from_str(&trace).unwrap();
        let arr = parsed.as_array().unwrap();

        // One layer → 2 events (B + E).
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["ph"], "B");
        assert_eq!(arr[1]["ph"], "E");
        assert_eq!(arr[0]["name"], "layer_0");
        assert!(arr[1]["ts"].as_f64().unwrap() > arr[0]["ts"].as_f64().unwrap());
    }

    #[test]
    fn test_chrome_trace_with_memory() {
        let config = ProfilerConfig::default().with_memory(true);
        let mut session = ProfileSession::new(config);

        session.begin_layer("layer_0", "attention");
        session.end_layer();
        session.record_memory_snapshot("peak", 8192);

        let report = session.generate_report();
        let trace = report.export_chrome_trace();
        let parsed: serde_json::Value = serde_json::from_str(&trace).unwrap();
        let arr = parsed.as_array().unwrap();

        // 2 duration events + 1 counter event.
        assert_eq!(arr.len(), 3);

        let counter = &arr[2];
        assert_eq!(counter["ph"], "C");
        assert_eq!(counter["args"]["memory_bytes"], 8192);
    }

    #[test]
    fn test_empty_profile() {
        let profiler = ModelProfiler::new(ProfilerConfig::default());
        let session = profiler.start_session();
        let report = session.generate_report();

        assert_eq!(report.total_time_us, 0.0);
        assert!(report.per_layer_breakdown.is_empty());
        assert!(report.bottleneck_layers.is_empty());
        assert_eq!(report.memory_peak, 0);
        assert_eq!(report.estimated_flops, 0);

        let trace = report.export_chrome_trace();
        let parsed: serde_json::Value = serde_json::from_str(&trace).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_warmup_iterations_discarded() {
        let config = ProfilerConfig::default().with_warmup(2).with_sample_size(1);
        let mut session = ProfileSession::new(config);

        // Warmup iteration 0.
        assert!(session.is_warmup());
        session.begin_layer("layer_0", "attention");
        session.end_layer();
        session.record_memory_snapshot("warmup_snap", 100);
        assert!(!session.next_iteration());

        // Warmup iteration 1.
        assert!(session.is_warmup());
        session.begin_layer("layer_0", "attention");
        session.end_layer();
        assert!(!session.next_iteration());

        // Sample iteration 2 (first real sample).
        assert!(!session.is_warmup());
        session.begin_layer("layer_0", "attention");
        thread::sleep(Duration::from_millis(1));
        session.end_layer();
        assert!(session.next_iteration()); // done

        let report = session.generate_report();
        // Only the non-warmup layer should be recorded.
        assert_eq!(report.layer_profiles.len(), 1);
        assert!(report.memory_snapshots.is_empty()); // warmup snapshot discarded
    }

    #[test]
    fn test_statistics_accuracy() {
        let config = ProfilerConfig::default().with_sample_size(3);
        let mut session = ProfileSession::new(config);

        // Record three samples for the same layer with different timings.
        for ms in [2, 4, 6] {
            session.begin_layer("layer_0", "attention");
            thread::sleep(Duration::from_millis(ms));
            session.end_layer();
        }

        let report = session.generate_report();
        let stats = &report.per_layer_breakdown[0];

        assert_eq!(stats.count, 3);
        assert!(stats.mean_time_us > 0.0);
        assert!(stats.min_time_us <= stats.mean_time_us);
        assert!(stats.max_time_us >= stats.mean_time_us);
        assert!(stats.std_time_us >= 0.0);
        assert!(stats.min_time_us < stats.max_time_us);
    }

    #[test]
    fn test_profiler_disabled() {
        let config = ProfilerConfig::disabled();
        let profiler = ModelProfiler::new(config);

        assert!(!profiler.is_enabled());

        let mut session = profiler.start_session();
        session.begin_layer("layer_0", "attention");
        thread::sleep(Duration::from_millis(2));
        session.end_layer();

        let report = session.generate_report();
        assert!(report.layer_profiles.is_empty());
    }

    #[test]
    fn test_profiler_config_defaults() {
        let config = ProfilerConfig::default();
        assert!(config.enabled);
        assert!(!config.record_memory);
        assert_eq!(config.warmup_iterations, 0);
        assert_eq!(config.sample_size, 1);
    }

    #[test]
    fn test_profiler_config_serialization() {
        let config = ProfilerConfig::default().with_warmup(3).with_sample_size(10);
        let json = serde_json::to_string(&config).unwrap();
        let deser: ProfilerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.warmup_iterations, 3);
        assert_eq!(deser.sample_size, 10);
    }

    #[test]
    fn test_iteration_counting() {
        let config = ProfilerConfig::default().with_warmup(1).with_sample_size(2);
        let mut session = ProfileSession::new(config);

        assert_eq!(session.iteration(), 0);
        assert!(session.is_warmup());
        assert!(!session.next_iteration()); // iteration 1 (warmup done)
        assert!(!session.is_warmup());
        assert!(!session.next_iteration()); // iteration 2 (sample 1)
        assert!(session.next_iteration()); // iteration 3 → complete
    }
}
