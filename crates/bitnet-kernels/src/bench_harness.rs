//! Kernel benchmark harness.
//!
//! Lightweight benchmarking for kernel operations without criterion dependency.

use std::time::{Duration, Instant};

/// Benchmark result for a single operation.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub iterations: u64,
    pub total_time: Duration,
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
    pub throughput_ops: f64,
}

impl BenchResult {
    pub fn median_ns(&self) -> f64 {
        self.mean.as_nanos() as f64
    }
    pub fn ops_per_sec(&self) -> f64 {
        self.throughput_ops
    }
}

/// Run a micro-benchmark.
pub fn bench<F>(name: &str, iterations: u64, mut f: F) -> BenchResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..iterations.min(10) {
        f();
    }

    let mut min = Duration::MAX;
    let mut max = Duration::ZERO;
    let mut total = Duration::ZERO;

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        min = min.min(elapsed);
        max = max.max(elapsed);
        total += elapsed;
    }

    let mean = total / iterations as u32;
    let throughput_ops =
        if total.as_secs_f64() > 0.0 { iterations as f64 / total.as_secs_f64() } else { 0.0 };

    BenchResult {
        name: name.to_string(),
        iterations,
        total_time: total,
        min,
        max,
        mean,
        throughput_ops,
    }
}

/// Benchmark with a setup closure that produces input each iteration.
pub fn bench_with_setup<S, F, T>(name: &str, iterations: u64, mut setup: S, mut f: F) -> BenchResult
where
    S: FnMut() -> T,
    F: FnMut(T),
{
    // Warmup
    for _ in 0..iterations.min(10) {
        f(setup());
    }

    let mut min = Duration::MAX;
    let mut max = Duration::ZERO;
    let mut total = Duration::ZERO;

    for _ in 0..iterations {
        let input = setup();
        let start = Instant::now();
        f(input);
        let elapsed = start.elapsed();
        min = min.min(elapsed);
        max = max.max(elapsed);
        total += elapsed;
    }

    let mean = total / iterations as u32;
    let throughput_ops =
        if total.as_secs_f64() > 0.0 { iterations as f64 / total.as_secs_f64() } else { 0.0 };

    BenchResult {
        name: name.to_string(),
        iterations,
        total_time: total,
        min,
        max,
        mean,
        throughput_ops,
    }
}

/// Benchmark suite.
#[derive(Debug)]
pub struct BenchSuite {
    pub name: String,
    pub results: Vec<BenchResult>,
}

impl BenchSuite {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), results: Vec::new() }
    }

    pub fn add(&mut self, result: BenchResult) {
        self.results.push(result);
    }

    pub fn fastest(&self) -> Option<&BenchResult> {
        self.results.iter().min_by_key(|r| r.mean)
    }

    pub fn slowest(&self) -> Option<&BenchResult> {
        self.results.iter().max_by_key(|r| r.mean)
    }

    pub fn total_time(&self) -> Duration {
        self.results.iter().map(|r| r.total_time).sum()
    }

    pub fn count(&self) -> usize {
        self.results.len()
    }
}

/// Compare two benchmark results.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub baseline_name: String,
    pub candidate_name: String,
    pub speedup: f64,
    pub baseline_mean_ns: f64,
    pub candidate_mean_ns: f64,
}

pub fn compare(baseline: &BenchResult, candidate: &BenchResult) -> Comparison {
    let base_ns = baseline.mean.as_nanos() as f64;
    let cand_ns = candidate.mean.as_nanos() as f64;
    let speedup = if cand_ns > 0.0 { base_ns / cand_ns } else { 0.0 };
    Comparison {
        baseline_name: baseline.name.clone(),
        candidate_name: candidate.name.clone(),
        speedup,
        baseline_mean_ns: base_ns,
        candidate_mean_ns: cand_ns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_basic() {
        let r = bench("noop", 100, || {});
        assert_eq!(r.iterations, 100);
        assert!(r.mean < Duration::from_millis(1));
    }

    #[test]
    fn test_bench_with_setup() {
        let r = bench_with_setup(
            "vec_sum",
            50,
            || vec![1.0f32; 100],
            |v| {
                let _sum: f32 = v.iter().sum();
            },
        );
        assert_eq!(r.iterations, 50);
    }

    #[test]
    fn test_throughput() {
        let r = bench("fast", 1000, || {});
        assert!(r.ops_per_sec() > 0.0);
    }

    #[test]
    fn test_min_max() {
        let r = bench("test", 10, || {
            std::hint::black_box(42);
        });
        assert!(r.min <= r.max);
        assert!(r.min <= r.mean);
    }

    #[test]
    fn test_suite() {
        let mut suite = BenchSuite::new("test_suite");
        suite.add(bench("fast", 10, || {}));
        suite.add(bench("slow", 10, || {
            std::thread::sleep(Duration::from_micros(100));
        }));
        assert_eq!(suite.count(), 2);
        assert!(suite.fastest().unwrap().mean < suite.slowest().unwrap().mean);
    }

    #[test]
    fn test_compare() {
        let fast = bench("fast", 100, || {});
        let slow = bench("slow", 100, || {
            std::thread::sleep(Duration::from_micros(10));
        });
        let c = compare(&slow, &fast);
        assert!(c.speedup > 1.0);
    }

    #[test]
    fn test_suite_total_time() {
        let mut suite = BenchSuite::new("s");
        suite.add(bench("a", 5, || {}));
        suite.add(bench("b", 5, || {}));
        assert!(suite.total_time() >= Duration::ZERO);
    }

    #[test]
    fn test_median_ns() {
        let r = bench("test", 10, || {});
        assert!(r.median_ns() >= 0.0);
    }

    #[test]
    fn test_empty_suite() {
        let suite = BenchSuite::new("empty");
        assert!(suite.fastest().is_none());
        assert!(suite.slowest().is_none());
    }

    #[test]
    fn test_comparison_names() {
        let a = bench("baseline", 10, || {});
        let b = bench("candidate", 10, || {});
        let c = compare(&a, &b);
        assert_eq!(c.baseline_name, "baseline");
        assert_eq!(c.candidate_name, "candidate");
    }
}
