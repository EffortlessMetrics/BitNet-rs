//! Inference metrics and profiling for BitNet inference pipelines.
//!
//! Provides thread-safe metrics collection, latency histograms, throughput
//! tracking, memory profiling, and serializable reporting.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// InferenceMetrics — snapshot of a single inference run
// ---------------------------------------------------------------------------

/// Summary metrics captured from a single inference invocation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceMetrics {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub time_to_first_token_ms: f64,
    pub total_generation_time_ms: f64,
    pub tokens_per_second: f64,
    pub peak_memory_bytes: u64,
    pub cache_hit_rate: f64,
}

impl InferenceMetrics {
    /// Create metrics from raw timing values. `tokens_per_second` is derived
    /// automatically when `total_generation_time_ms > 0`.
    pub fn new(
        prompt_tokens: u64,
        generated_tokens: u64,
        time_to_first_token_ms: f64,
        total_generation_time_ms: f64,
        peak_memory_bytes: u64,
        cache_hit_rate: f64,
    ) -> Self {
        let tokens_per_second = if total_generation_time_ms > 0.0 {
            (generated_tokens as f64) / (total_generation_time_ms / 1000.0)
        } else {
            0.0
        };
        Self {
            prompt_tokens,
            generated_tokens,
            time_to_first_token_ms,
            total_generation_time_ms,
            tokens_per_second,
            peak_memory_bytes,
            cache_hit_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// MetricsCollector — thread-safe accumulator
// ---------------------------------------------------------------------------

/// Thread-safe metrics collector that accumulates counters using atomics.
#[derive(Debug)]
pub struct MetricsCollector {
    prompt_tokens: AtomicU64,
    generated_tokens: AtomicU64,
    total_requests: AtomicU64,
    total_generation_time_ns: AtomicU64,
    first_token_time_ns: AtomicU64,
    peak_memory_bytes: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            prompt_tokens: AtomicU64::new(self.prompt_tokens.load(Ordering::Relaxed)),
            generated_tokens: AtomicU64::new(self.generated_tokens.load(Ordering::Relaxed)),
            total_requests: AtomicU64::new(self.total_requests.load(Ordering::Relaxed)),
            total_generation_time_ns: AtomicU64::new(
                self.total_generation_time_ns.load(Ordering::Relaxed),
            ),
            first_token_time_ns: AtomicU64::new(self.first_token_time_ns.load(Ordering::Relaxed)),
            peak_memory_bytes: AtomicU64::new(self.peak_memory_bytes.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.cache_misses.load(Ordering::Relaxed)),
        }
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            prompt_tokens: AtomicU64::new(0),
            generated_tokens: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            total_generation_time_ns: AtomicU64::new(0),
            first_token_time_ns: AtomicU64::new(0),
            peak_memory_bytes: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Record a completed inference request.
    pub fn record_request(
        &self,
        prompt_tokens: u64,
        generated_tokens: u64,
        generation_time_ns: u64,
        first_token_ns: u64,
    ) {
        self.prompt_tokens.fetch_add(prompt_tokens, Ordering::Relaxed);
        self.generated_tokens.fetch_add(generated_tokens, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_generation_time_ns.fetch_add(generation_time_ns, Ordering::Relaxed);
        // Store the latest first-token latency (useful for single-request scenarios).
        self.first_token_time_ns.store(first_token_ns, Ordering::Relaxed);
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Update peak memory (max-wins semantics).
    pub fn update_peak_memory(&self, bytes: u64) {
        self.peak_memory_bytes.fetch_max(bytes, Ordering::Relaxed);
    }

    /// Snapshot the current counters into an [`InferenceMetrics`].
    pub fn snapshot(&self) -> InferenceMetrics {
        let gen_ns = self.total_generation_time_ns.load(Ordering::Relaxed);
        let gen_ms = gen_ns as f64 / 1_000_000.0;
        let first_ns = self.first_token_time_ns.load(Ordering::Relaxed);
        let first_ms = first_ns as f64 / 1_000_000.0;
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let cache_hit_rate =
            if hits + misses > 0 { hits as f64 / (hits + misses) as f64 } else { 0.0 };
        InferenceMetrics::new(
            self.prompt_tokens.load(Ordering::Relaxed),
            self.generated_tokens.load(Ordering::Relaxed),
            first_ms,
            gen_ms,
            self.peak_memory_bytes.load(Ordering::Relaxed),
            cache_hit_rate,
        )
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.prompt_tokens.store(0, Ordering::Relaxed);
        self.generated_tokens.store(0, Ordering::Relaxed);
        self.total_requests.store(0, Ordering::Relaxed);
        self.total_generation_time_ns.store(0, Ordering::Relaxed);
        self.first_token_time_ns.store(0, Ordering::Relaxed);
        self.peak_memory_bytes.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
    }

    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LatencyHistogram — bucketed percentile tracking
// ---------------------------------------------------------------------------

/// Bucketed latency histogram that supports percentile queries (p50 … p99).
///
/// Latency values are stored in sorted order so that percentile computation
/// is O(n log n) on `percentile()` only when the internal buffer has been
/// mutated since the last sort.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// Raw latency samples in milliseconds.
    samples: Vec<f64>,
    /// Whether `samples` is currently sorted.
    #[serde(skip)]
    sorted: bool,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self { samples: Vec::new(), sorted: true }
    }

    /// Record a latency sample in milliseconds.
    pub fn record(&mut self, latency_ms: f64) {
        self.samples.push(latency_ms);
        self.sorted = false;
    }

    /// Return the *p*-th percentile (0.0–100.0). Returns `None` when empty.
    pub fn percentile(&mut self, p: f64) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        if !self.sorted {
            self.samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.sorted = true;
        }
        let idx = ((p / 100.0) * (self.samples.len() as f64 - 1.0))
            .round()
            .clamp(0.0, (self.samples.len() - 1) as f64) as usize;
        Some(self.samples[idx])
    }

    /// Convenience accessors.
    pub fn p50(&mut self) -> Option<f64> {
        self.percentile(50.0)
    }
    pub fn p90(&mut self) -> Option<f64> {
        self.percentile(90.0)
    }
    pub fn p95(&mut self) -> Option<f64> {
        self.percentile(95.0)
    }
    pub fn p99(&mut self) -> Option<f64> {
        self.percentile(99.0)
    }

    pub fn count(&self) -> usize {
        self.samples.len()
    }

    pub fn mean(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        Some(self.samples.iter().sum::<f64>() / self.samples.len() as f64)
    }

    pub fn min(&self) -> Option<f64> {
        self.samples.iter().copied().reduce(f64::min)
    }

    pub fn max(&self) -> Option<f64> {
        self.samples.iter().copied().reduce(f64::max)
    }

    /// Reset the histogram, discarding all samples.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.sorted = true;
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ThroughputTracker — sliding-window TPS
// ---------------------------------------------------------------------------

/// Sliding-window throughput tracker.
///
/// Maintains a bounded ring of `(timestamp, token_count)` entries and computes
/// tokens-per-second over the window.
#[derive(Debug, Clone)]
pub struct ThroughputTracker {
    window: Vec<(Instant, u64)>,
    window_size: std::time::Duration,
    max_entries: usize,
}

impl ThroughputTracker {
    /// Create a tracker with the given sliding-window duration.
    pub fn new(window: std::time::Duration) -> Self {
        Self { window: Vec::new(), window_size: window, max_entries: 10_000 }
    }

    /// Record `count` tokens generated at the current instant.
    pub fn record(&mut self, count: u64) {
        self.record_at(Instant::now(), count);
    }

    /// Record with an explicit timestamp (useful for testing).
    pub fn record_at(&mut self, when: Instant, count: u64) {
        self.window.push((when, count));
        self.evict(when);
    }

    /// Tokens per second over the current window.
    pub fn tokens_per_second(&self) -> f64 {
        self.tokens_per_second_at(Instant::now())
    }

    /// TPS at a given reference instant.
    pub fn tokens_per_second_at(&self, now: Instant) -> f64 {
        let cutoff = now.checked_sub(self.window_size).unwrap_or(now);
        let total: u64 = self.window.iter().filter(|(t, _)| *t >= cutoff).map(|(_, c)| c).sum();
        let elapsed = self.window_size.as_secs_f64();
        if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 }
    }

    /// Total tokens recorded (regardless of window).
    pub fn total_tokens(&self) -> u64 {
        self.window.iter().map(|(_, c)| c).sum()
    }

    /// Reset all recorded data.
    pub fn reset(&mut self) {
        self.window.clear();
    }

    fn evict(&mut self, now: Instant) {
        let cutoff = now.checked_sub(self.window_size).unwrap_or(now);
        self.window.retain(|(t, _)| *t >= cutoff);
        // Hard cap to avoid unbounded growth in degenerate cases.
        if self.window.len() > self.max_entries {
            let excess = self.window.len() - self.max_entries;
            self.window.drain(..excess);
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryProfiler
// ---------------------------------------------------------------------------

/// Tracks current and peak memory usage via atomic counters.
///
/// Shareable across threads through `Arc`.
#[derive(Debug)]
pub struct MemoryProfiler {
    current_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
}

impl Clone for MemoryProfiler {
    fn clone(&self) -> Self {
        Self {
            current_bytes: AtomicU64::new(self.current_bytes.load(Ordering::Relaxed)),
            peak_bytes: AtomicU64::new(self.peak_bytes.load(Ordering::Relaxed)),
            allocation_count: AtomicU64::new(self.allocation_count.load(Ordering::Relaxed)),
            deallocation_count: AtomicU64::new(self.deallocation_count.load(Ordering::Relaxed)),
        }
    }
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            current_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
        }
    }

    /// Record an allocation of `bytes`.
    pub fn record_allocation(&self, bytes: u64) {
        let prev = self.current_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.peak_bytes.fetch_max(prev + bytes, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a deallocation of `bytes`.
    pub fn record_deallocation(&self, bytes: u64) {
        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn current_bytes(&self) -> u64 {
        self.current_bytes.load(Ordering::Relaxed)
    }

    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    pub fn allocation_count(&self) -> u64 {
        self.allocation_count.load(Ordering::Relaxed)
    }

    pub fn deallocation_count(&self) -> u64 {
        self.deallocation_count.load(Ordering::Relaxed)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.current_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MetricsReport — serializable summary
// ---------------------------------------------------------------------------

/// Serializable metrics report suitable for JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub inference: InferenceMetrics,
    pub latency_p50_ms: Option<f64>,
    pub latency_p90_ms: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    pub latency_p99_ms: Option<f64>,
    pub latency_mean_ms: Option<f64>,
    pub latency_min_ms: Option<f64>,
    pub latency_max_ms: Option<f64>,
    pub latency_samples: usize,
    pub throughput_tps: f64,
    pub memory_current_bytes: u64,
    pub memory_peak_bytes: u64,
    pub memory_allocation_count: u64,
    pub memory_deallocation_count: u64,
}

impl MetricsReport {
    /// Build a report from the individual profiling components.
    pub fn build(
        collector: &MetricsCollector,
        histogram: &mut LatencyHistogram,
        throughput: &ThroughputTracker,
        memory: &MemoryProfiler,
    ) -> Self {
        Self {
            inference: collector.snapshot(),
            latency_p50_ms: histogram.p50(),
            latency_p90_ms: histogram.p90(),
            latency_p95_ms: histogram.p95(),
            latency_p99_ms: histogram.p99(),
            latency_mean_ms: histogram.mean(),
            latency_min_ms: histogram.min(),
            latency_max_ms: histogram.max(),
            latency_samples: histogram.count(),
            throughput_tps: throughput.tokens_per_second(),
            memory_current_bytes: memory.current_bytes(),
            memory_peak_bytes: memory.peak_bytes(),
            memory_allocation_count: memory.allocation_count(),
            memory_deallocation_count: memory.deallocation_count(),
        }
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    // -- InferenceMetrics ---------------------------------------------------

    #[test]
    fn test_inference_metrics_new_computes_tps() {
        let m = InferenceMetrics::new(10, 20, 50.0, 2000.0, 1024, 0.75);
        assert!((m.tokens_per_second - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_inference_metrics_zero_time_yields_zero_tps() {
        let m = InferenceMetrics::new(0, 5, 0.0, 0.0, 0, 0.0);
        assert_eq!(m.tokens_per_second, 0.0);
    }

    #[test]
    fn test_inference_metrics_serde_roundtrip() {
        let m = InferenceMetrics::new(1, 2, 3.0, 4.0, 5, 0.6);
        let json = serde_json::to_string(&m).unwrap();
        let m2: InferenceMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_inference_metrics_fields() {
        let m = InferenceMetrics::new(8, 16, 12.5, 500.0, 4096, 0.9);
        assert_eq!(m.prompt_tokens, 8);
        assert_eq!(m.generated_tokens, 16);
        assert!((m.time_to_first_token_ms - 12.5).abs() < f64::EPSILON);
        assert_eq!(m.peak_memory_bytes, 4096);
        assert!((m.cache_hit_rate - 0.9).abs() < f64::EPSILON);
    }

    // -- MetricsCollector ---------------------------------------------------

    #[test]
    fn test_collector_initial_snapshot_is_zero() {
        let c = MetricsCollector::new();
        let s = c.snapshot();
        assert_eq!(s.prompt_tokens, 0);
        assert_eq!(s.generated_tokens, 0);
        assert_eq!(s.tokens_per_second, 0.0);
    }

    #[test]
    fn test_collector_record_request() {
        let c = MetricsCollector::new();
        c.record_request(10, 20, 2_000_000_000, 100_000_000);
        let s = c.snapshot();
        assert_eq!(s.prompt_tokens, 10);
        assert_eq!(s.generated_tokens, 20);
        assert!((s.total_generation_time_ms - 2000.0).abs() < 1e-3);
        assert!((s.time_to_first_token_ms - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_collector_multiple_requests_accumulate() {
        let c = MetricsCollector::new();
        c.record_request(5, 10, 1_000_000_000, 50_000_000);
        c.record_request(3, 7, 500_000_000, 30_000_000);
        let s = c.snapshot();
        assert_eq!(s.prompt_tokens, 8);
        assert_eq!(s.generated_tokens, 17);
        assert_eq!(c.total_requests(), 2);
    }

    #[test]
    fn test_collector_cache_hit_rate() {
        let c = MetricsCollector::new();
        c.record_cache_hit();
        c.record_cache_hit();
        c.record_cache_hit();
        c.record_cache_miss();
        let s = c.snapshot();
        assert!((s.cache_hit_rate - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_collector_cache_no_accesses_yields_zero() {
        let c = MetricsCollector::new();
        let s = c.snapshot();
        assert_eq!(s.cache_hit_rate, 0.0);
    }

    #[test]
    fn test_collector_peak_memory() {
        let c = MetricsCollector::new();
        c.update_peak_memory(100);
        c.update_peak_memory(500);
        c.update_peak_memory(200);
        let s = c.snapshot();
        assert_eq!(s.peak_memory_bytes, 500);
    }

    #[test]
    fn test_collector_reset() {
        let c = MetricsCollector::new();
        c.record_request(10, 20, 1_000_000, 500_000);
        c.record_cache_hit();
        c.update_peak_memory(1024);
        c.reset();
        let s = c.snapshot();
        assert_eq!(s.prompt_tokens, 0);
        assert_eq!(s.generated_tokens, 0);
        assert_eq!(s.peak_memory_bytes, 0);
        assert_eq!(s.cache_hit_rate, 0.0);
    }

    #[test]
    fn test_collector_default() {
        let c = MetricsCollector::default();
        assert_eq!(c.total_requests(), 0);
    }

    #[test]
    fn test_collector_thread_safety() {
        let c = Arc::new(MetricsCollector::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = Arc::clone(&c);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        c.record_request(1, 1, 1_000, 500);
                        c.record_cache_hit();
                        c.update_peak_memory(64);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let s = c.snapshot();
        assert_eq!(s.prompt_tokens, 800);
        assert_eq!(s.generated_tokens, 800);
        assert_eq!(c.total_requests(), 800);
    }

    // -- LatencyHistogram ---------------------------------------------------

    #[test]
    fn test_histogram_empty_returns_none() {
        let mut h = LatencyHistogram::new();
        assert!(h.p50().is_none());
        assert!(h.mean().is_none());
        assert!(h.min().is_none());
        assert!(h.max().is_none());
    }

    #[test]
    fn test_histogram_single_sample() {
        let mut h = LatencyHistogram::new();
        h.record(42.0);
        assert_eq!(h.p50(), Some(42.0));
        assert_eq!(h.p99(), Some(42.0));
        assert_eq!(h.count(), 1);
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut h = LatencyHistogram::new();
        for i in 1..=100 {
            h.record(i as f64);
        }
        assert!((h.p50().unwrap() - 50.0).abs() < 1.5);
        assert!((h.p90().unwrap() - 90.0).abs() < 1.5);
        assert!((h.p95().unwrap() - 95.0).abs() < 1.5);
        assert!((h.p99().unwrap() - 99.0).abs() < 1.5);
    }

    #[test]
    fn test_histogram_mean() {
        let mut h = LatencyHistogram::new();
        h.record(10.0);
        h.record(20.0);
        h.record(30.0);
        assert!((h.mean().unwrap() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_min_max() {
        let mut h = LatencyHistogram::new();
        h.record(5.0);
        h.record(1.0);
        h.record(9.0);
        assert_eq!(h.min(), Some(1.0));
        assert_eq!(h.max(), Some(9.0));
    }

    #[test]
    fn test_histogram_reset() {
        let mut h = LatencyHistogram::new();
        h.record(1.0);
        h.record(2.0);
        h.reset();
        assert_eq!(h.count(), 0);
        assert!(h.p50().is_none());
    }

    #[test]
    fn test_histogram_serde_roundtrip() {
        let mut h = LatencyHistogram::new();
        h.record(10.0);
        h.record(20.0);
        let json = serde_json::to_string(&h).unwrap();
        let mut h2: LatencyHistogram = serde_json::from_str(&json).unwrap();
        assert_eq!(h2.count(), 2);
        assert_eq!(h.p50(), h2.p50());
    }

    // -- ThroughputTracker --------------------------------------------------

    #[test]
    fn test_throughput_empty() {
        let t = ThroughputTracker::new(Duration::from_secs(10));
        assert_eq!(t.tokens_per_second(), 0.0);
        assert_eq!(t.total_tokens(), 0);
    }

    #[test]
    fn test_throughput_basic() {
        let mut t = ThroughputTracker::new(Duration::from_secs(1));
        let now = Instant::now();
        t.record_at(now, 100);
        let tps = t.tokens_per_second_at(now);
        assert!((tps - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_throughput_eviction() {
        let mut t = ThroughputTracker::new(Duration::from_millis(100));
        let start = Instant::now();
        t.record_at(start, 50);
        // Record again well past the window.
        let later = start + Duration::from_millis(200);
        t.record_at(later, 30);
        // Only the second entry should survive.
        assert_eq!(t.total_tokens(), 30);
    }

    #[test]
    fn test_throughput_total_tokens() {
        let mut t = ThroughputTracker::new(Duration::from_secs(60));
        let now = Instant::now();
        t.record_at(now, 10);
        t.record_at(now, 20);
        t.record_at(now, 30);
        assert_eq!(t.total_tokens(), 60);
    }

    #[test]
    fn test_throughput_reset() {
        let mut t = ThroughputTracker::new(Duration::from_secs(60));
        t.record(10);
        t.reset();
        assert_eq!(t.total_tokens(), 0);
    }

    // -- MemoryProfiler -----------------------------------------------------

    #[test]
    fn test_memory_profiler_initial_zero() {
        let mp = MemoryProfiler::new();
        assert_eq!(mp.current_bytes(), 0);
        assert_eq!(mp.peak_bytes(), 0);
        assert_eq!(mp.allocation_count(), 0);
    }

    #[test]
    fn test_memory_profiler_allocation_deallocation() {
        let mp = MemoryProfiler::new();
        mp.record_allocation(1024);
        assert_eq!(mp.current_bytes(), 1024);
        assert_eq!(mp.peak_bytes(), 1024);
        assert_eq!(mp.allocation_count(), 1);

        mp.record_deallocation(512);
        assert_eq!(mp.current_bytes(), 512);
        assert_eq!(mp.peak_bytes(), 1024); // peak unchanged
        assert_eq!(mp.deallocation_count(), 1);
    }

    #[test]
    fn test_memory_profiler_peak_tracking() {
        let mp = MemoryProfiler::new();
        mp.record_allocation(100);
        mp.record_allocation(200);
        // peak = 300
        mp.record_deallocation(250);
        mp.record_allocation(10);
        assert_eq!(mp.peak_bytes(), 300);
        assert_eq!(mp.current_bytes(), 60);
    }

    #[test]
    fn test_memory_profiler_reset() {
        let mp = MemoryProfiler::new();
        mp.record_allocation(1024);
        mp.reset();
        assert_eq!(mp.current_bytes(), 0);
        assert_eq!(mp.peak_bytes(), 0);
        assert_eq!(mp.allocation_count(), 0);
        assert_eq!(mp.deallocation_count(), 0);
    }

    #[test]
    fn test_memory_profiler_thread_safety() {
        let mp = Arc::new(MemoryProfiler::new());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let mp = Arc::clone(&mp);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        mp.record_allocation(64);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(mp.allocation_count(), 400);
    }

    #[test]
    fn test_memory_profiler_default() {
        let mp = MemoryProfiler::default();
        assert_eq!(mp.current_bytes(), 0);
    }

    // -- MetricsReport ------------------------------------------------------

    #[test]
    fn test_report_build() {
        let c = MetricsCollector::new();
        c.record_request(5, 10, 1_000_000_000, 100_000_000);
        let mut h = LatencyHistogram::new();
        h.record(50.0);
        h.record(100.0);
        let t = ThroughputTracker::new(Duration::from_secs(60));
        let mp = MemoryProfiler::new();
        mp.record_allocation(2048);

        let report = MetricsReport::build(&c, &mut h, &t, &mp);
        assert_eq!(report.inference.prompt_tokens, 5);
        assert_eq!(report.inference.generated_tokens, 10);
        assert_eq!(report.latency_samples, 2);
        assert_eq!(report.memory_current_bytes, 2048);
        assert_eq!(report.memory_peak_bytes, 2048);
    }

    #[test]
    fn test_report_json_roundtrip() {
        let c = MetricsCollector::new();
        let mut h = LatencyHistogram::new();
        let t = ThroughputTracker::new(Duration::from_secs(1));
        let mp = MemoryProfiler::new();
        let report = MetricsReport::build(&c, &mut h, &t, &mp);
        let json = report.to_json().unwrap();
        let parsed: MetricsReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.inference.prompt_tokens, 0);
    }

    #[test]
    fn test_collector_clone() {
        let c = MetricsCollector::new();
        c.record_request(5, 10, 1_000_000, 500_000);
        let c2 = c.clone();
        assert_eq!(c2.snapshot().prompt_tokens, 5);
    }
}
