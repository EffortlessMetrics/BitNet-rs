//! Kernel performance tracking and comparison.

use std::collections::HashMap;
use std::time::Duration;

/// Performance record for a single kernel invocation.
#[derive(Debug, Clone)]
pub struct KernelTiming {
    pub kernel_name: String,
    pub duration: Duration,
    pub input_elements: usize,
    pub flops: Option<u64>,
}

impl KernelTiming {
    pub fn new(name: &str, duration: Duration, elements: usize) -> Self {
        Self { kernel_name: name.to_string(), duration, input_elements: elements, flops: None }
    }

    pub fn with_flops(mut self, flops: u64) -> Self {
        self.flops = Some(flops);
        self
    }

    /// Elements processed per second.
    pub fn throughput(&self) -> f64 {
        if self.duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.input_elements as f64 / self.duration.as_secs_f64()
    }

    /// GFLOPS (if flops were provided).
    pub fn gflops(&self) -> Option<f64> {
        self.flops.map(|f| f as f64 / self.duration.as_secs_f64() / 1e9)
    }
}

/// Tracker that collects kernel timings.
#[derive(Debug, Clone, Default)]
pub struct PerfTracker {
    timings: Vec<KernelTiming>,
}

impl PerfTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, timing: KernelTiming) {
        self.timings.push(timing);
    }

    pub fn timings(&self) -> &[KernelTiming] {
        &self.timings
    }

    pub fn total_time(&self) -> Duration {
        self.timings.iter().map(|t| t.duration).sum()
    }

    pub fn count(&self) -> usize {
        self.timings.len()
    }

    pub fn clear(&mut self) {
        self.timings.clear();
    }

    /// Get timings grouped by kernel name.
    pub fn by_kernel(&self) -> HashMap<String, Vec<&KernelTiming>> {
        let mut map: HashMap<String, Vec<&KernelTiming>> = HashMap::new();
        for t in &self.timings {
            map.entry(t.kernel_name.clone()).or_default().push(t);
        }
        map
    }

    /// Get summary statistics per kernel.
    pub fn kernel_stats(&self) -> Vec<KernelStats> {
        let grouped = self.by_kernel();
        let mut stats: Vec<KernelStats> = grouped
            .into_iter()
            .map(|(name, timings)| {
                let durations: Vec<Duration> = timings.iter().map(|t| t.duration).collect();
                let total: Duration = durations.iter().sum();
                let count = durations.len();
                let avg = total / count as u32;
                let mut sorted = durations.clone();
                sorted.sort();
                let min = sorted.first().copied().unwrap_or(Duration::ZERO);
                let max = sorted.last().copied().unwrap_or(Duration::ZERO);
                let total_elements: usize = timings.iter().map(|t| t.input_elements).sum();
                KernelStats {
                    name,
                    count,
                    total_time: total,
                    avg_time: avg,
                    min_time: min,
                    max_time: max,
                    total_elements,
                }
            })
            .collect();
        stats.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        stats
    }

    /// Find the slowest kernel invocation.
    pub fn slowest(&self) -> Option<&KernelTiming> {
        self.timings.iter().max_by_key(|t| t.duration)
    }

    /// Find the fastest kernel invocation.
    pub fn fastest(&self) -> Option<&KernelTiming> {
        self.timings.iter().min_by_key(|t| t.duration)
    }
}

/// Summary statistics for a kernel.
#[derive(Debug, Clone)]
pub struct KernelStats {
    pub name: String,
    pub count: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub total_elements: usize,
}

impl KernelStats {
    /// Average throughput (elements/sec).
    pub fn avg_throughput(&self) -> f64 {
        if self.total_time.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.total_elements as f64 / self.total_time.as_secs_f64()
    }
}

/// Format tracker results as text.
pub fn format_perf_report(tracker: &PerfTracker) -> String {
    let mut out = "=== Kernel Performance Report ===\n".to_string();
    out.push_str(&format!("Total kernels: {}\n", tracker.count()));
    out.push_str(&format!("Total time:    {:.2?}\n\n", tracker.total_time()));

    let stats = tracker.kernel_stats();
    out.push_str(&format!(
        "{:<20} {:>6} {:>12} {:>12} {:>12}\n",
        "Kernel", "Count", "Total", "Avg", "Throughput"
    ));
    out.push_str(&"-".repeat(70));
    out.push('\n');

    for s in &stats {
        out.push_str(&format!(
            "{:<20} {:>6} {:>12.2?} {:>12.2?} {:>10.0} e/s\n",
            s.name,
            s.count,
            s.total_time,
            s.avg_time,
            s.avg_throughput()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_new() {
        let t = KernelTiming::new("matmul", Duration::from_millis(10), 1024);
        assert_eq!(t.kernel_name, "matmul");
        assert_eq!(t.input_elements, 1024);
    }

    #[test]
    fn test_timing_throughput() {
        let t = KernelTiming::new("test", Duration::from_secs(1), 1000);
        assert!((t.throughput() - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_timing_throughput_zero() {
        let t = KernelTiming::new("test", Duration::ZERO, 1000);
        assert_eq!(t.throughput(), 0.0);
    }

    #[test]
    fn test_timing_with_flops() {
        let t = KernelTiming::new("test", Duration::from_secs(1), 1000).with_flops(1_000_000_000);
        assert!((t.gflops().unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_timing_no_flops() {
        let t = KernelTiming::new("test", Duration::from_millis(1), 100);
        assert!(t.gflops().is_none());
    }

    #[test]
    fn test_tracker_new() {
        let t = PerfTracker::new();
        assert_eq!(t.count(), 0);
        assert_eq!(t.total_time(), Duration::ZERO);
    }

    #[test]
    fn test_tracker_record() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("test", Duration::from_millis(5), 100));
        assert_eq!(t.count(), 1);
    }

    #[test]
    fn test_tracker_total_time() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("a", Duration::from_millis(10), 100));
        t.record(KernelTiming::new("b", Duration::from_millis(20), 200));
        assert_eq!(t.total_time(), Duration::from_millis(30));
    }

    #[test]
    fn test_tracker_clear() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("test", Duration::from_millis(5), 100));
        t.clear();
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn test_tracker_by_kernel() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("matmul", Duration::from_millis(10), 100));
        t.record(KernelTiming::new("softmax", Duration::from_millis(5), 50));
        t.record(KernelTiming::new("matmul", Duration::from_millis(15), 200));
        let grouped = t.by_kernel();
        assert_eq!(grouped["matmul"].len(), 2);
        assert_eq!(grouped["softmax"].len(), 1);
    }

    #[test]
    fn test_kernel_stats() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("matmul", Duration::from_millis(10), 100));
        t.record(KernelTiming::new("matmul", Duration::from_millis(20), 200));
        let stats = t.kernel_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].count, 2);
        assert_eq!(stats[0].total_elements, 300);
    }

    #[test]
    fn test_kernel_stats_sorted_by_time() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("fast", Duration::from_millis(1), 100));
        t.record(KernelTiming::new("slow", Duration::from_millis(100), 100));
        let stats = t.kernel_stats();
        assert_eq!(stats[0].name, "slow");
    }

    #[test]
    fn test_slowest() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("a", Duration::from_millis(5), 100));
        t.record(KernelTiming::new("b", Duration::from_millis(50), 100));
        t.record(KernelTiming::new("c", Duration::from_millis(10), 100));
        let slowest = t.slowest().unwrap();
        assert_eq!(slowest.kernel_name, "b");
    }

    #[test]
    fn test_fastest() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("a", Duration::from_millis(5), 100));
        t.record(KernelTiming::new("b", Duration::from_millis(50), 100));
        let fastest = t.fastest().unwrap();
        assert_eq!(fastest.kernel_name, "a");
    }

    #[test]
    fn test_slowest_empty() {
        let t = PerfTracker::new();
        assert!(t.slowest().is_none());
    }

    #[test]
    fn test_kernel_stats_avg_throughput() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("test", Duration::from_secs(1), 1000));
        let stats = t.kernel_stats();
        assert!((stats[0].avg_throughput() - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_format_report() {
        let mut t = PerfTracker::new();
        t.record(KernelTiming::new("matmul", Duration::from_millis(10), 1024));
        t.record(KernelTiming::new("softmax", Duration::from_millis(5), 512));
        let out = format_perf_report(&t);
        assert!(out.contains("Kernel Performance Report"));
        assert!(out.contains("matmul"));
        assert!(out.contains("softmax"));
    }

    #[test]
    fn test_format_report_empty() {
        let t = PerfTracker::new();
        let out = format_perf_report(&t);
        assert!(out.contains("Total kernels: 0"));
    }

    #[test]
    fn test_default() {
        let t = PerfTracker::default();
        assert!(t.timings().is_empty());
    }
}
