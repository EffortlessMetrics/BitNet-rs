//! Automated kernel benchmark runner with A/B comparison.
//!
//! Run benchmarks, collect results, compare against baselines,
//! detect regressions.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Single benchmark measurement.
#[derive(Debug, Clone)]
pub struct BenchMeasurement {
    pub name: String,
    pub iterations: u64,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl BenchMeasurement {
    pub fn avg_ns(&self) -> f64 {
        if self.iterations == 0 {
            return 0.0;
        }
        self.total_time.as_nanos() as f64 / self.iterations as f64
    }

    pub fn throughput_ops_per_sec(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.iterations as f64 / secs
    }
}

/// Run a simple benchmark for a closure.
pub fn bench_fn<F: FnMut()>(name: &str, iterations: u64, mut f: F) -> BenchMeasurement {
    let mut min_time = Duration::MAX;
    let mut max_time = Duration::ZERO;
    let start = Instant::now();
    for _ in 0..iterations {
        let iter_start = Instant::now();
        f();
        let elapsed = iter_start.elapsed();
        min_time = min_time.min(elapsed);
        max_time = max_time.max(elapsed);
    }
    let total_time = start.elapsed();
    BenchMeasurement { name: name.to_string(), iterations, total_time, min_time, max_time }
}

/// Comparison between two measurements (A vs B).
#[derive(Debug)]
pub struct BenchComparison {
    pub name: String,
    pub baseline_avg_ns: f64,
    pub candidate_avg_ns: f64,
    pub speedup: f64,
    pub regression: bool,
}

impl BenchComparison {
    pub fn compare(baseline: &BenchMeasurement, candidate: &BenchMeasurement) -> Self {
        let b = baseline.avg_ns();
        let c = candidate.avg_ns();
        let speedup = if c > 0.0 { b / c } else { 0.0 };
        Self {
            name: baseline.name.clone(),
            baseline_avg_ns: b,
            candidate_avg_ns: c,
            speedup,
            regression: speedup < 0.95,
        }
    }

    pub fn summary(&self) -> String {
        let dir = if self.speedup >= 1.0 { "faster" } else { "slower" };
        format!(
            "{}: {:.1}x {} (baseline={:.0}ns, candidate={:.0}ns)",
            self.name, self.speedup, dir, self.baseline_avg_ns, self.candidate_avg_ns,
        )
    }
}

/// Benchmark suite with multiple named benchmarks.
#[derive(Debug, Default)]
pub struct BenchSuite {
    pub results: HashMap<String, BenchMeasurement>,
}

impl BenchSuite {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, measurement: BenchMeasurement) {
        self.results.insert(measurement.name.clone(), measurement);
    }

    pub fn get(&self, name: &str) -> Option<&BenchMeasurement> {
        self.results.get(name)
    }

    pub fn count(&self) -> usize {
        self.results.len()
    }

    /// Compare all matching benchmarks between two suites.
    pub fn compare_with(&self, other: &BenchSuite) -> Vec<BenchComparison> {
        let mut comparisons = Vec::new();
        for (name, baseline) in &self.results {
            if let Some(candidate) = other.results.get(name) {
                comparisons.push(BenchComparison::compare(baseline, candidate));
            }
        }
        comparisons
            .sort_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap_or(std::cmp::Ordering::Equal));
        comparisons
    }

    /// Detect regressions (threshold = 0.95x).
    pub fn regressions_vs(&self, other: &BenchSuite) -> Vec<BenchComparison> {
        self.compare_with(other).into_iter().filter(|c| c.regression).collect()
    }

    pub fn fastest(&self) -> Option<&BenchMeasurement> {
        self.results
            .values()
            .min_by(|a, b| a.avg_ns().partial_cmp(&b.avg_ns()).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn slowest(&self) -> Option<&BenchMeasurement> {
        self.results
            .values()
            .max_by(|a, b| a.avg_ns().partial_cmp(&b.avg_ns()).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// Regression threshold configuration.
#[derive(Debug, Clone)]
pub struct RegressionPolicy {
    pub threshold: f64,
    pub min_iterations: u64,
}

impl Default for RegressionPolicy {
    fn default() -> Self {
        Self { threshold: 0.95, min_iterations: 100 }
    }
}

impl RegressionPolicy {
    pub fn strict() -> Self {
        Self { threshold: 0.99, min_iterations: 1000 }
    }

    pub fn is_regression(&self, speedup: f64) -> bool {
        speedup < self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_fn() {
        let m = bench_fn("noop", 100, || {});
        assert_eq!(m.name, "noop");
        assert_eq!(m.iterations, 100);
        assert!(m.total_time > Duration::ZERO);
    }

    #[test]
    fn test_avg_ns() {
        let m = BenchMeasurement {
            name: "test".into(),
            iterations: 10,
            total_time: Duration::from_micros(1),
            min_time: Duration::from_nanos(80),
            max_time: Duration::from_nanos(120),
        };
        assert!((m.avg_ns() - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_throughput() {
        let m = BenchMeasurement {
            name: "test".into(),
            iterations: 1000,
            total_time: Duration::from_secs(1),
            min_time: Duration::from_millis(1),
            max_time: Duration::from_millis(1),
        };
        assert!((m.throughput_ops_per_sec() - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_comparison() {
        let baseline = BenchMeasurement {
            name: "kernel".into(),
            iterations: 100,
            total_time: Duration::from_micros(10),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        };
        let candidate = BenchMeasurement {
            name: "kernel".into(),
            iterations: 100,
            total_time: Duration::from_micros(5),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        };
        let cmp = BenchComparison::compare(&baseline, &candidate);
        assert!((cmp.speedup - 2.0).abs() < 0.01);
        assert!(!cmp.regression);
    }

    #[test]
    fn test_regression_detected() {
        let baseline = BenchMeasurement {
            name: "k".into(),
            iterations: 100,
            total_time: Duration::from_micros(5),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        };
        let candidate = BenchMeasurement {
            name: "k".into(),
            iterations: 100,
            total_time: Duration::from_micros(10),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        };
        let cmp = BenchComparison::compare(&baseline, &candidate);
        assert!(cmp.regression);
    }

    #[test]
    fn test_suite_add_and_get() {
        let mut suite = BenchSuite::new();
        suite.add(BenchMeasurement {
            name: "a".into(),
            iterations: 10,
            total_time: Duration::from_nanos(100),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        });
        assert_eq!(suite.count(), 1);
        assert!(suite.get("a").is_some());
    }

    #[test]
    fn test_suite_compare() {
        let mut s1 = BenchSuite::new();
        let mut s2 = BenchSuite::new();
        s1.add(BenchMeasurement {
            name: "k".into(),
            iterations: 100,
            total_time: Duration::from_micros(10),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        });
        s2.add(BenchMeasurement {
            name: "k".into(),
            iterations: 100,
            total_time: Duration::from_micros(5),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        });
        let comps = s1.compare_with(&s2);
        assert_eq!(comps.len(), 1);
    }

    #[test]
    fn test_regressions() {
        let mut s1 = BenchSuite::new();
        let mut s2 = BenchSuite::new();
        s1.add(BenchMeasurement {
            name: "fast".into(),
            iterations: 100,
            total_time: Duration::from_micros(1),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        });
        s2.add(BenchMeasurement {
            name: "fast".into(),
            iterations: 100,
            total_time: Duration::from_micros(5),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        });
        let regs = s1.regressions_vs(&s2);
        assert_eq!(regs.len(), 1);
    }

    #[test]
    fn test_fastest_slowest() {
        let mut suite = BenchSuite::new();
        suite.add(BenchMeasurement {
            name: "slow".into(),
            iterations: 100,
            total_time: Duration::from_micros(10),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        });
        suite.add(BenchMeasurement {
            name: "fast".into(),
            iterations: 100,
            total_time: Duration::from_micros(1),
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
        });
        assert_eq!(suite.fastest().unwrap().name, "fast");
        assert_eq!(suite.slowest().unwrap().name, "slow");
    }

    #[test]
    fn test_regression_policy() {
        let policy = RegressionPolicy::default();
        assert!(policy.is_regression(0.90));
        assert!(!policy.is_regression(1.0));
    }

    #[test]
    fn test_strict_policy() {
        let policy = RegressionPolicy::strict();
        assert!(policy.is_regression(0.98));
        assert!(!policy.is_regression(1.0));
    }

    #[test]
    fn test_comparison_summary() {
        let cmp = BenchComparison {
            name: "kern".into(),
            baseline_avg_ns: 100.0,
            candidate_avg_ns: 50.0,
            speedup: 2.0,
            regression: false,
        };
        let s = cmp.summary();
        assert!(s.contains("2.0x"));
        assert!(s.contains("faster"));
    }
}
