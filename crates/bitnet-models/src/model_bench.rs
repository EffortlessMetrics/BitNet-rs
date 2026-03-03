//! Model benchmarking harness.
//!
//! Standardized performance measurement for model loading,
//! tensor operations, and memory footprint.

use std::time::{Duration, Instant};

/// A single benchmark measurement.
#[derive(Debug, Clone)]
pub struct Measurement {
    pub name: String,
    pub duration: Duration,
    pub iterations: usize,
    pub bytes_processed: usize,
}

impl Measurement {
    pub fn new(name: impl Into<String>, duration: Duration, iterations: usize) -> Self {
        Self { name: name.into(), duration, iterations, bytes_processed: 0 }
    }

    pub fn with_bytes(mut self, bytes: usize) -> Self {
        self.bytes_processed = bytes;
        self
    }

    pub fn avg_duration(&self) -> Duration {
        if self.iterations == 0 {
            return Duration::ZERO;
        }
        self.duration / self.iterations as u32
    }

    pub fn throughput_bytes_per_sec(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (self.bytes_processed * self.iterations) as f64 / secs
    }

    pub fn throughput_mb_per_sec(&self) -> f64 {
        self.throughput_bytes_per_sec() / (1024.0 * 1024.0)
    }

    pub fn ops_per_sec(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.iterations as f64 / secs
    }
}

/// Benchmark harness that collects measurements.
#[derive(Debug, Default)]
pub struct BenchHarness {
    pub measurements: Vec<Measurement>,
    pub warmup_iterations: usize,
}

impl BenchHarness {
    pub fn new() -> Self {
        Self { measurements: Vec::new(), warmup_iterations: 3 }
    }

    pub fn with_warmup(mut self, n: usize) -> Self {
        self.warmup_iterations = n;
        self
    }

    /// Run a benchmark: warmup + measured iterations.
    pub fn bench<F>(&mut self, name: &str, iterations: usize, mut f: F) -> &Measurement
    where
        F: FnMut() -> usize,
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            let _ = f();
        }

        // Measure
        let start = Instant::now();
        let mut total_bytes = 0;
        for _ in 0..iterations {
            total_bytes += f();
        }
        let elapsed = start.elapsed();

        let m =
            Measurement::new(name, elapsed, iterations).with_bytes(total_bytes / iterations.max(1));
        self.measurements.push(m);
        self.measurements.last().unwrap()
    }

    /// Run a benchmark that returns no byte count.
    pub fn bench_simple<F>(&mut self, name: &str, iterations: usize, mut f: F) -> &Measurement
    where
        F: FnMut(),
    {
        self.bench(name, iterations, || {
            f();
            0
        })
    }

    pub fn summary(&self) -> BenchSummary {
        let total_time: Duration = self.measurements.iter().map(|m| m.duration).sum();
        let total_iters: usize = self.measurements.iter().map(|m| m.iterations).sum();
        BenchSummary {
            num_benchmarks: self.measurements.len(),
            total_time,
            total_iterations: total_iters,
        }
    }

    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str("Benchmark Report\n");
        out.push_str(&"=".repeat(60));
        out.push('\n');
        for m in &self.measurements {
            out.push_str(&format!(
                "{:<30} {:>8.2}ms avg ({} iters, {:.1} ops/s)\n",
                m.name,
                m.avg_duration().as_secs_f64() * 1000.0,
                m.iterations,
                m.ops_per_sec(),
            ));
        }
        out
    }

    pub fn fastest(&self) -> Option<&Measurement> {
        self.measurements.iter().min_by_key(|m| m.avg_duration())
    }

    pub fn slowest(&self) -> Option<&Measurement> {
        self.measurements.iter().max_by_key(|m| m.avg_duration())
    }
}

/// Summary of a benchmark suite.
#[derive(Debug)]
pub struct BenchSummary {
    pub num_benchmarks: usize,
    pub total_time: Duration,
    pub total_iterations: usize,
}

/// Time a single operation.
pub fn time_op<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    (result, start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_basic() {
        let m = Measurement::new("test", Duration::from_millis(100), 10);
        assert_eq!(m.avg_duration(), Duration::from_millis(10));
    }

    #[test]
    fn test_measurement_with_bytes() {
        let m = Measurement::new("test", Duration::from_secs(1), 1).with_bytes(1024);
        assert_eq!(m.bytes_processed, 1024);
        assert!((m.throughput_bytes_per_sec() - 1024.0).abs() < 10.0);
    }

    #[test]
    fn test_measurement_zero_iters() {
        let m = Measurement::new("test", Duration::ZERO, 0);
        assert_eq!(m.avg_duration(), Duration::ZERO);
        assert_eq!(m.ops_per_sec(), 0.0);
    }

    #[test]
    fn test_harness_bench_simple() {
        let mut h = BenchHarness::new().with_warmup(1);
        let mut count = 0u64;
        h.bench_simple("inc", 100, || {
            count += 1;
        });
        assert_eq!(h.measurements.len(), 1);
        assert!(count > 100); // warmup + measured
    }

    #[test]
    fn test_harness_bench_with_bytes() {
        let mut h = BenchHarness::new().with_warmup(0);
        h.bench("alloc", 10, || {
            let v: Vec<u8> = vec![0; 1024];
            v.len()
        });
        assert_eq!(h.measurements.len(), 1);
    }

    #[test]
    fn test_summary() {
        let mut h = BenchHarness::new().with_warmup(0);
        h.bench_simple("a", 5, || {});
        h.bench_simple("b", 10, || {});
        let s = h.summary();
        assert_eq!(s.num_benchmarks, 2);
        assert_eq!(s.total_iterations, 15);
    }

    #[test]
    fn test_report() {
        let mut h = BenchHarness::new().with_warmup(0);
        h.bench_simple("test_op", 10, || {});
        let report = h.report();
        assert!(report.contains("test_op"));
        assert!(report.contains("Benchmark Report"));
    }

    #[test]
    fn test_fastest_slowest() {
        let mut h = BenchHarness::new().with_warmup(0);
        h.bench_simple("fast", 100, || {});
        h.bench_simple("slow", 1, || {
            std::thread::sleep(Duration::from_millis(5));
        });
        assert!(h.fastest().is_some());
        assert!(h.slowest().is_some());
    }

    #[test]
    fn test_time_op() {
        let (result, dur) = time_op(|| 42);
        assert_eq!(result, 42);
        assert!(dur < Duration::from_secs(1));
    }

    #[test]
    fn test_empty_harness() {
        let h = BenchHarness::new();
        assert!(h.fastest().is_none());
        assert!(h.slowest().is_none());
        assert_eq!(h.summary().num_benchmarks, 0);
    }

    #[test]
    fn test_throughput_mb() {
        let m = Measurement::new("test", Duration::from_secs(1), 1).with_bytes(1048576);
        assert!((m.throughput_mb_per_sec() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_ops_per_sec() {
        let m = Measurement::new("test", Duration::from_secs(2), 100);
        assert!((m.ops_per_sec() - 50.0).abs() < 1.0);
    }
}
