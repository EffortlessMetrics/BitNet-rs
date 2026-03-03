//! Per-kernel execution profiling.
//!
//! Tracks execution time, call count, and throughput for
//! individual kernel invocations (matmul, attention, norm, etc.).

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Identifier for a kernel type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelId {
    Matmul,
    Attention,
    LayerNorm,
    RmsNorm,
    SiLU,
    ReLU,
    Softmax,
    RoPE,
    Embedding,
    Quantize,
    Dequantize,
    Custom(u32),
}

impl KernelId {
    pub fn name(&self) -> &'static str {
        match self {
            KernelId::Matmul => "matmul",
            KernelId::Attention => "attention",
            KernelId::LayerNorm => "layer_norm",
            KernelId::RmsNorm => "rms_norm",
            KernelId::SiLU => "silu",
            KernelId::ReLU => "relu",
            KernelId::Softmax => "softmax",
            KernelId::RoPE => "rope",
            KernelId::Embedding => "embedding",
            KernelId::Quantize => "quantize",
            KernelId::Dequantize => "dequantize",
            KernelId::Custom(_) => "custom",
        }
    }
}

/// Statistics for a single kernel.
#[derive(Debug, Clone)]
pub struct KernelStats {
    pub call_count: u64,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub total_flops: u64,
}

impl KernelStats {
    fn new() -> Self {
        Self {
            call_count: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            total_flops: 0,
        }
    }

    fn record(&mut self, elapsed: Duration, flops: u64) {
        self.call_count += 1;
        self.total_time += elapsed;
        self.min_time = self.min_time.min(elapsed);
        self.max_time = self.max_time.max(elapsed);
        self.total_flops += flops;
    }

    pub fn avg_time(&self) -> Duration {
        if self.call_count == 0 {
            return Duration::ZERO;
        }
        self.total_time / self.call_count as u32
    }

    pub fn gflops_per_sec(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_flops as f64 / secs / 1e9
    }

    pub fn calls_per_sec(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.call_count as f64 / secs
    }
}

/// Profiler that collects per-kernel statistics.
#[derive(Debug)]
pub struct KernelProfiler {
    stats: HashMap<KernelId, KernelStats>,
    enabled: bool,
}

impl KernelProfiler {
    pub fn new() -> Self {
        Self { stats: HashMap::new(), enabled: true }
    }

    pub fn disabled() -> Self {
        Self { stats: HashMap::new(), enabled: false }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Start timing a kernel. Returns a guard that records on drop.
    pub fn start(&self, kernel: KernelId) -> KernelTimer {
        KernelTimer {
            kernel,
            start: if self.enabled { Some(Instant::now()) } else { None },
            flops: 0,
        }
    }

    /// Record a completed kernel execution.
    pub fn record(&mut self, kernel: KernelId, elapsed: Duration, flops: u64) {
        if !self.enabled {
            return;
        }
        self.stats.entry(kernel).or_insert_with(KernelStats::new).record(elapsed, flops);
    }

    /// Get stats for a specific kernel.
    pub fn get(&self, kernel: KernelId) -> Option<&KernelStats> {
        self.stats.get(&kernel)
    }

    /// Get all kernel stats.
    pub fn all_stats(&self) -> &HashMap<KernelId, KernelStats> {
        &self.stats
    }

    /// Total time across all kernels.
    pub fn total_time(&self) -> Duration {
        self.stats.values().map(|s| s.total_time).sum()
    }

    /// Get the hottest kernel (most total time).
    pub fn hottest(&self) -> Option<(KernelId, &KernelStats)> {
        self.stats.iter().max_by_key(|(_, s)| s.total_time).map(|(&k, s)| (k, s))
    }

    /// Generate a sorted report (by total time, descending).
    pub fn report(&self) -> Vec<(KernelId, KernelStats)> {
        let mut entries: Vec<_> = self.stats.iter().map(|(&k, s)| (k, s.clone())).collect();
        entries.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));
        entries
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.stats.clear();
    }

    /// Merge another profiler's stats into this one.
    pub fn merge(&mut self, other: &KernelProfiler) {
        for (&k, other_stats) in &other.stats {
            let entry = self.stats.entry(k).or_insert_with(KernelStats::new);
            entry.call_count += other_stats.call_count;
            entry.total_time += other_stats.total_time;
            entry.min_time = entry.min_time.min(other_stats.min_time);
            entry.max_time = entry.max_time.max(other_stats.max_time);
            entry.total_flops += other_stats.total_flops;
        }
    }
}

impl Default for KernelProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer returned by `KernelProfiler::start`.
pub struct KernelTimer {
    pub kernel: KernelId,
    pub start: Option<Instant>,
    pub flops: u64,
}

impl KernelTimer {
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.flops = flops;
        self
    }

    pub fn finish(self) -> Option<Duration> {
        self.start.map(|s| s.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_get() {
        let mut p = KernelProfiler::new();
        p.record(KernelId::Matmul, Duration::from_millis(10), 1000);
        let s = p.get(KernelId::Matmul).unwrap();
        assert_eq!(s.call_count, 1);
    }

    #[test]
    fn test_multiple_records() {
        let mut p = KernelProfiler::new();
        p.record(KernelId::SiLU, Duration::from_millis(5), 100);
        p.record(KernelId::SiLU, Duration::from_millis(15), 200);
        let s = p.get(KernelId::SiLU).unwrap();
        assert_eq!(s.call_count, 2);
        assert_eq!(s.total_time, Duration::from_millis(20));
    }

    #[test]
    fn test_avg_time() {
        let mut s = KernelStats::new();
        s.record(Duration::from_millis(10), 0);
        s.record(Duration::from_millis(30), 0);
        assert_eq!(s.avg_time(), Duration::from_millis(20));
    }

    #[test]
    fn test_min_max() {
        let mut s = KernelStats::new();
        s.record(Duration::from_millis(5), 0);
        s.record(Duration::from_millis(15), 0);
        s.record(Duration::from_millis(10), 0);
        assert_eq!(s.min_time, Duration::from_millis(5));
        assert_eq!(s.max_time, Duration::from_millis(15));
    }

    #[test]
    fn test_disabled_profiler() {
        let mut p = KernelProfiler::disabled();
        p.record(KernelId::Matmul, Duration::from_millis(10), 0);
        assert!(p.get(KernelId::Matmul).is_none());
    }

    #[test]
    fn test_enable_disable() {
        let mut p = KernelProfiler::new();
        assert!(p.is_enabled());
        p.disable();
        assert!(!p.is_enabled());
        p.enable();
        assert!(p.is_enabled());
    }

    #[test]
    fn test_total_time() {
        let mut p = KernelProfiler::new();
        p.record(KernelId::Matmul, Duration::from_millis(10), 0);
        p.record(KernelId::SiLU, Duration::from_millis(5), 0);
        assert_eq!(p.total_time(), Duration::from_millis(15));
    }

    #[test]
    fn test_hottest() {
        let mut p = KernelProfiler::new();
        p.record(KernelId::Matmul, Duration::from_millis(100), 0);
        p.record(KernelId::SiLU, Duration::from_millis(5), 0);
        let (k, _) = p.hottest().unwrap();
        assert_eq!(k, KernelId::Matmul);
    }

    #[test]
    fn test_report_sorted() {
        let mut p = KernelProfiler::new();
        p.record(KernelId::SiLU, Duration::from_millis(5), 0);
        p.record(KernelId::Matmul, Duration::from_millis(100), 0);
        let report = p.report();
        assert_eq!(report[0].0, KernelId::Matmul); // most time first
    }

    #[test]
    fn test_reset() {
        let mut p = KernelProfiler::new();
        p.record(KernelId::Matmul, Duration::from_millis(10), 0);
        p.reset();
        assert!(p.get(KernelId::Matmul).is_none());
    }

    #[test]
    fn test_merge() {
        let mut a = KernelProfiler::new();
        a.record(KernelId::Matmul, Duration::from_millis(10), 100);
        let mut b = KernelProfiler::new();
        b.record(KernelId::Matmul, Duration::from_millis(20), 200);
        a.merge(&b);
        let s = a.get(KernelId::Matmul).unwrap();
        assert_eq!(s.call_count, 2);
        assert_eq!(s.total_flops, 300);
    }

    #[test]
    fn test_kernel_id_name() {
        assert_eq!(KernelId::Matmul.name(), "matmul");
        assert_eq!(KernelId::RmsNorm.name(), "rms_norm");
        assert_eq!(KernelId::Custom(42).name(), "custom");
    }
}
