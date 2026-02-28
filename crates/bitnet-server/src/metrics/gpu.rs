//! Prometheus metrics for GPU backends.
//!
//! Provides gauges, histograms, and counters that expose GPU utilisation,
//! memory, kernel execution time, error counts, and transfer bandwidth
//! for integration with the `/metrics` Prometheus endpoint.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// ── Metric types (lightweight, no external dependency) ───────────────────────

/// Atomic gauge backed by `AtomicU64` storing `f64` bits.
#[derive(Debug)]
pub struct AtomicGauge(AtomicU64);

impl AtomicGauge {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    pub fn set(&self, val: f64) {
        self.0.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
}

impl Default for AtomicGauge {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe counter.
#[derive(Debug)]
pub struct AtomicCounter(AtomicU64);

impl AtomicCounter {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    pub fn inc(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_by(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

impl Default for AtomicCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple histogram that collects samples in pre-defined buckets.
#[derive(Debug)]
pub struct SimpleHistogram {
    /// Upper bounds (sorted).
    bounds: Vec<f64>,
    /// Counts per bucket (len == bounds.len() + 1 for the +Inf bucket).
    counts: Vec<AtomicU64>,
    sum: AtomicU64,
    count: AtomicU64,
}

impl SimpleHistogram {
    pub fn new(bounds: Vec<f64>) -> Self {
        let n = bounds.len() + 1;
        let counts = (0..n).map(|_| AtomicU64::new(0)).collect();
        Self { bounds, counts, sum: AtomicU64::new(0), count: AtomicU64::new(0) }
    }

    pub fn observe(&self, val: f64) {
        let idx = self.bounds.partition_point(|&b| b < val);
        self.counts[idx].fetch_add(1, Ordering::Relaxed);
        // Update sum via CAS loop.
        loop {
            let old_bits = self.sum.load(Ordering::Relaxed);
            let old = f64::from_bits(old_bits);
            let new = old + val;
            if self
                .sum
                .compare_exchange_weak(
                    old_bits,
                    new.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn sum(&self) -> f64 {
        f64::from_bits(self.sum.load(Ordering::Relaxed))
    }

    /// Return `(upper_bound, cumulative_count)` pairs.
    pub fn buckets(&self) -> Vec<(f64, u64)> {
        let mut cumulative = 0u64;
        let mut out = Vec::with_capacity(self.bounds.len() + 1);
        for (i, &bound) in self.bounds.iter().enumerate() {
            cumulative += self.counts[i].load(Ordering::Relaxed);
            out.push((bound, cumulative));
        }
        cumulative += self.counts[self.bounds.len()].load(Ordering::Relaxed);
        out.push((f64::INFINITY, cumulative));
        out
    }
}

// ── GPU metrics ──────────────────────────────────────────────────────────────

/// Default histogram buckets for kernel execution time (seconds).
fn default_kernel_time_buckets() -> Vec<f64> {
    vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
}

/// Prometheus-style metrics for GPU backends.
#[derive(Debug)]
pub struct GpuMetrics {
    /// GPU utilisation percentage (0–100).
    pub gpu_utilization: AtomicGauge,
    /// GPU memory currently in use (bytes).
    pub memory_used_bytes: AtomicGauge,
    /// GPU total memory (bytes).
    pub memory_total_bytes: AtomicGauge,
    /// Kernel execution time histogram.
    pub kernel_exec_time: SimpleHistogram,
    /// Error count by type.
    pub error_count: AtomicCounter,
    /// Host → GPU transfer bandwidth (bytes/sec).
    pub transfer_host_to_gpu_bps: AtomicGauge,
    /// GPU → Host transfer bandwidth (bytes/sec).
    pub transfer_gpu_to_host_bps: AtomicGauge,
    /// Label for the backend name (e.g. "cuda", "oneapi").
    pub backend_name: String,
}

impl GpuMetrics {
    /// Create a new `GpuMetrics` instance for `backend`.
    pub fn new(backend: impl Into<String>) -> Self {
        Self {
            gpu_utilization: AtomicGauge::new(),
            memory_used_bytes: AtomicGauge::new(),
            memory_total_bytes: AtomicGauge::new(),
            kernel_exec_time: SimpleHistogram::new(default_kernel_time_buckets()),
            error_count: AtomicCounter::new(),
            transfer_host_to_gpu_bps: AtomicGauge::new(),
            transfer_gpu_to_host_bps: AtomicGauge::new(),
            backend_name: backend.into(),
        }
    }

    /// Record a kernel execution duration.
    pub fn record_kernel_execution(&self, duration: std::time::Duration) {
        self.kernel_exec_time.observe(duration.as_secs_f64());
    }

    /// Record a GPU error.
    pub fn record_error(&self) {
        self.error_count.inc();
    }

    /// Update memory gauges.
    pub fn update_memory(&self, used: u64, total: u64) {
        self.memory_used_bytes.set(used as f64);
        self.memory_total_bytes.set(total as f64);
    }

    /// Update transfer bandwidth gauges.
    pub fn update_transfer_bandwidth(&self, host_to_gpu_bps: f64, gpu_to_host_bps: f64) {
        self.transfer_host_to_gpu_bps.set(host_to_gpu_bps);
        self.transfer_gpu_to_host_bps.set(gpu_to_host_bps);
    }

    /// Render metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let b = &self.backend_name;
        let mut out = String::with_capacity(2048);

        // Gauges
        out.push_str(&format!(
            "# HELP bitnet_gpu_utilization_percent GPU utilization percentage\n\
             # TYPE bitnet_gpu_utilization_percent gauge\n\
             bitnet_gpu_utilization_percent{{backend=\"{b}\"}} {:.2}\n",
            self.gpu_utilization.get(),
        ));
        out.push_str(&format!(
            "# HELP bitnet_gpu_memory_used_bytes GPU memory in use\n\
             # TYPE bitnet_gpu_memory_used_bytes gauge\n\
             bitnet_gpu_memory_used_bytes{{backend=\"{b}\"}} {}\n",
            self.memory_used_bytes.get(),
        ));
        out.push_str(&format!(
            "# HELP bitnet_gpu_memory_total_bytes GPU total memory\n\
             # TYPE bitnet_gpu_memory_total_bytes gauge\n\
             bitnet_gpu_memory_total_bytes{{backend=\"{b}\"}} {}\n",
            self.memory_total_bytes.get(),
        ));
        out.push_str(&format!(
            "# HELP bitnet_gpu_transfer_host_to_gpu_bps Host to GPU bandwidth\n\
             # TYPE bitnet_gpu_transfer_host_to_gpu_bps gauge\n\
             bitnet_gpu_transfer_host_to_gpu_bps{{backend=\"{b}\"}} {:.2}\n",
            self.transfer_host_to_gpu_bps.get(),
        ));
        out.push_str(&format!(
            "# HELP bitnet_gpu_transfer_gpu_to_host_bps GPU to host bandwidth\n\
             # TYPE bitnet_gpu_transfer_gpu_to_host_bps gauge\n\
             bitnet_gpu_transfer_gpu_to_host_bps{{backend=\"{b}\"}} {:.2}\n",
            self.transfer_gpu_to_host_bps.get(),
        ));

        // Counter
        out.push_str(&format!(
            "# HELP bitnet_gpu_errors_total GPU error count\n\
             # TYPE bitnet_gpu_errors_total counter\n\
             bitnet_gpu_errors_total{{backend=\"{b}\"}} {}\n",
            self.error_count.get(),
        ));

        // Histogram
        out.push_str(
            "# HELP bitnet_gpu_kernel_exec_seconds Kernel execution time\n\
             # TYPE bitnet_gpu_kernel_exec_seconds histogram\n",
        );
        for (bound, cum) in self.kernel_exec_time.buckets() {
            if bound.is_infinite() {
                out.push_str(&format!(
                    "bitnet_gpu_kernel_exec_seconds_bucket\
                     {{backend=\"{b}\",le=\"+Inf\"}} {cum}\n",
                ));
            } else {
                out.push_str(&format!(
                    "bitnet_gpu_kernel_exec_seconds_bucket\
                     {{backend=\"{b}\",le=\"{bound}\"}} {cum}\n",
                ));
            }
        }
        out.push_str(&format!(
            "bitnet_gpu_kernel_exec_seconds_sum{{backend=\"{b}\"}} {:.6}\n\
             bitnet_gpu_kernel_exec_seconds_count{{backend=\"{b}\"}} {}\n",
            self.kernel_exec_time.sum(),
            self.kernel_exec_time.count(),
        ));

        out
    }
}

/// RAII guard that records kernel execution time on drop.
pub struct KernelTimer {
    start: Instant,
    metrics: Arc<GpuMetrics>,
}

impl KernelTimer {
    pub fn start(metrics: Arc<GpuMetrics>) -> Self {
        Self { start: Instant::now(), metrics }
    }
}

impl Drop for KernelTimer {
    fn drop(&mut self) {
        self.metrics.record_kernel_execution(self.start.elapsed());
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn gauge_set_get() {
        let g = AtomicGauge::new();
        assert!((g.get() - 0.0).abs() < f64::EPSILON);
        g.set(42.5);
        assert!((g.get() - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn counter_inc() {
        let c = AtomicCounter::new();
        assert_eq!(c.get(), 0);
        c.inc();
        c.inc();
        assert_eq!(c.get(), 2);
        c.inc_by(5);
        assert_eq!(c.get(), 7);
    }

    #[test]
    fn histogram_observe_and_buckets() {
        let h = SimpleHistogram::new(vec![1.0, 5.0, 10.0]);
        h.observe(0.5);
        h.observe(3.0);
        h.observe(7.0);
        h.observe(20.0);
        assert_eq!(h.count(), 4);
        let sum = h.sum();
        assert!((sum - 30.5).abs() < 1e-6);
        let buckets = h.buckets();
        // bucket ≤1.0: 1, ≤5.0: 2, ≤10.0: 3, +Inf: 4
        assert_eq!(buckets[0], (1.0, 1));
        assert_eq!(buckets[1], (5.0, 2));
        assert_eq!(buckets[2], (10.0, 3));
        assert_eq!(buckets[3].1, 4);
    }

    #[test]
    fn gpu_metrics_new() {
        let m = GpuMetrics::new("cuda");
        assert_eq!(m.backend_name, "cuda");
        assert!((m.gpu_utilization.get() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gpu_metrics_record_kernel() {
        let m = GpuMetrics::new("test");
        m.record_kernel_execution(Duration::from_millis(5));
        m.record_kernel_execution(Duration::from_millis(50));
        assert_eq!(m.kernel_exec_time.count(), 2);
        assert!(m.kernel_exec_time.sum() > 0.0);
    }

    #[test]
    fn gpu_metrics_record_error() {
        let m = GpuMetrics::new("test");
        m.record_error();
        m.record_error();
        assert_eq!(m.error_count.get(), 2);
    }

    #[test]
    fn gpu_metrics_update_memory() {
        let m = GpuMetrics::new("test");
        m.update_memory(1024, 4096);
        assert!((m.memory_used_bytes.get() - 1024.0).abs() < f64::EPSILON);
        assert!((m.memory_total_bytes.get() - 4096.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gpu_metrics_update_transfer_bandwidth() {
        let m = GpuMetrics::new("test");
        m.update_transfer_bandwidth(1e9, 5e8);
        assert!((m.transfer_host_to_gpu_bps.get() - 1e9).abs() < 1.0);
        assert!((m.transfer_gpu_to_host_bps.get() - 5e8).abs() < 1.0);
    }

    #[test]
    fn render_prometheus_format() {
        let m = GpuMetrics::new("cuda");
        m.gpu_utilization.set(75.5);
        m.update_memory(2_000_000, 8_000_000);
        m.record_kernel_execution(Duration::from_millis(1));
        m.record_error();
        m.update_transfer_bandwidth(1e9, 5e8);

        let output = m.render_prometheus();
        assert!(output.contains("bitnet_gpu_utilization_percent{backend=\"cuda\"}"));
        assert!(output.contains("bitnet_gpu_memory_used_bytes{backend=\"cuda\"}"));
        assert!(output.contains("bitnet_gpu_memory_total_bytes{backend=\"cuda\"}"));
        assert!(output.contains("bitnet_gpu_errors_total{backend=\"cuda\"}"));
        assert!(output.contains("bitnet_gpu_kernel_exec_seconds_bucket"));
        assert!(output.contains("bitnet_gpu_kernel_exec_seconds_sum"));
        assert!(output.contains("bitnet_gpu_kernel_exec_seconds_count"));
        assert!(output.contains("bitnet_gpu_transfer_host_to_gpu_bps"));
        assert!(output.contains("bitnet_gpu_transfer_gpu_to_host_bps"));
        assert!(output.contains("# TYPE bitnet_gpu_kernel_exec_seconds histogram"));
    }

    #[test]
    fn kernel_timer_records_duration() {
        let m = Arc::new(GpuMetrics::new("test"));
        {
            let _timer = KernelTimer::start(m.clone());
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(m.kernel_exec_time.count(), 1);
        assert!(m.kernel_exec_time.sum() >= 0.004);
    }
}
