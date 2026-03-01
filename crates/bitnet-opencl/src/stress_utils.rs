//! Stress testing utilities: runner, load generator, result collector, and
//! summary report.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── StressReport ─────────────────────────────────────────────────────

/// Aggregated summary of a stress test run.
#[derive(Debug, Clone)]
pub struct StressReport {
    pub total_requests: u64,
    pub passed: u64,
    pub failed: u64,
    pub error_breakdown: Vec<(String, u64)>,
    pub latency_p50: Duration,
    pub latency_p90: Duration,
    pub latency_p99: Duration,
    pub latency_max: Duration,
    pub wall_time: Duration,
}

impl StressReport {
    /// Pass rate as a fraction in `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pass_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 1.0;
        }
        self.passed as f64 / self.total_requests as f64
    }
}

impl std::fmt::Display for StressReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Stress Test Report ===")?;
        writeln!(
            f,
            "Total: {} | Passed: {} | Failed: {} | Rate: {:.1}%",
            self.total_requests,
            self.passed,
            self.failed,
            self.pass_rate() * 100.0,
        )?;
        writeln!(
            f,
            "Latency  p50={:?}  p90={:?}  p99={:?}  max={:?}",
            self.latency_p50, self.latency_p90, self.latency_p99, self.latency_max,
        )?;
        writeln!(f, "Wall time: {:?}", self.wall_time)?;
        if !self.error_breakdown.is_empty() {
            writeln!(f, "Errors:")?;
            for (kind, count) in &self.error_breakdown {
                writeln!(f, "  {kind}: {count}")?;
            }
        }
        Ok(())
    }
}

// ── ResultCollector ──────────────────────────────────────────────────

/// Thread-safe collector for pass/fail results with latency tracking.
#[derive(Debug, Clone)]
pub struct ResultCollector {
    passed: Arc<AtomicU64>,
    failed: Arc<AtomicU64>,
    latencies: Arc<Mutex<Vec<Duration>>>,
    errors: Arc<Mutex<Vec<String>>>,
    start: Instant,
}

impl ResultCollector {
    #[must_use]
    pub fn new() -> Self {
        Self {
            passed: Arc::new(AtomicU64::new(0)),
            failed: Arc::new(AtomicU64::new(0)),
            latencies: Arc::new(Mutex::new(Vec::new())),
            errors: Arc::new(Mutex::new(Vec::new())),
            start: Instant::now(),
        }
    }

    /// Record a successful operation with its duration.
    pub fn record_pass(&self, duration: Duration) {
        self.passed.fetch_add(1, Ordering::Relaxed);
        self.latencies.lock().expect("poisoned").push(duration);
    }

    /// Record a failed operation with an error description.
    pub fn record_fail(&self, error: &str) {
        self.failed.fetch_add(1, Ordering::Relaxed);
        self.errors.lock().expect("poisoned").push(error.to_string());
    }

    #[must_use]
    pub fn passed_count(&self) -> u64 {
        self.passed.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn failed_count(&self) -> u64 {
        self.failed.load(Ordering::Acquire)
    }

    /// Build a [`StressReport`] from collected data.
    #[must_use]
    pub fn report(&self) -> StressReport {
        let wall_time = self.start.elapsed();
        let passed = self.passed.load(Ordering::Acquire);
        let failed = self.failed.load(Ordering::Acquire);

        let mut latencies = self.latencies.lock().expect("poisoned").clone();
        latencies.sort();

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let percentile = |p: f64| -> Duration {
            if latencies.is_empty() {
                return Duration::ZERO;
            }
            let idx = ((latencies.len() as f64) * p).ceil().min(latencies.len() as f64) as usize;
            latencies[idx.saturating_sub(1)]
        };

        let errors = self.errors.lock().expect("poisoned").clone();
        let mut error_map = std::collections::HashMap::<String, u64>::new();
        for e in &errors {
            // Bucket by first 60 chars to group similar errors.
            let key: String = e.chars().take(60).collect();
            *error_map.entry(key).or_default() += 1;
        }
        let mut error_breakdown: Vec<(String, u64)> = error_map.into_iter().collect();
        error_breakdown.sort_by(|a, b| b.1.cmp(&a.1));

        StressReport {
            total_requests: passed + failed,
            passed,
            failed,
            error_breakdown,
            latency_p50: percentile(0.50),
            latency_p90: percentile(0.90),
            latency_p99: percentile(0.99),
            latency_max: latencies.last().copied().unwrap_or(Duration::ZERO),
            wall_time,
        }
    }
}

impl Default for ResultCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ── LoadGenerator ───────────────────────────────────────────────────

/// Synthetic inference request for load generation.
#[derive(Debug, Clone)]
pub struct SyntheticRequest {
    pub id: u64,
    pub input_size: usize,
    pub output_size: usize,
    pub is_batch: bool,
}

/// Generates synthetic inference requests at a configurable rate.
#[derive(Debug, Clone)]
pub struct LoadGenerator {
    next_id: Arc<AtomicU64>,
    input_size: usize,
    output_size: usize,
    batch_probability: f64,
}

impl LoadGenerator {
    #[must_use]
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            next_id: Arc::new(AtomicU64::new(0)),
            input_size,
            output_size,
            batch_probability: 0.0,
        }
    }

    #[must_use]
    pub const fn with_batch_probability(mut self, prob: f64) -> Self {
        self.batch_probability = prob;
        self
    }

    /// Generate the next synthetic request.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn next_request(&self) -> SyntheticRequest {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        // Deterministic: IDs below threshold percentage are batches.
        let is_batch =
            self.batch_probability > 0.0 && (id % 100) < (self.batch_probability * 100.0) as u64;
        SyntheticRequest {
            id,
            input_size: self.input_size,
            output_size: self.output_size,
            is_batch,
        }
    }

    #[must_use]
    pub fn generated_count(&self) -> u64 {
        self.next_id.load(Ordering::Acquire)
    }
}

impl Default for LoadGenerator {
    fn default() -> Self {
        Self::new(64, 64)
    }
}

// ── StressTestRunner ────────────────────────────────────────────────

/// Configurable concurrent stress test executor.
pub struct StressTestRunner {
    pub threads: usize,
    pub iterations_per_thread: usize,
    pub collector: ResultCollector,
    request_count: Arc<AtomicUsize>,
}

impl StressTestRunner {
    #[must_use]
    pub fn new(threads: usize, iterations_per_thread: usize) -> Self {
        Self {
            threads,
            iterations_per_thread,
            collector: ResultCollector::new(),
            request_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Execute `work` across multiple threads and collect results.
    ///
    /// The closure receives `(thread_index, iteration_index)` and must
    /// return `Ok(())` on success or an error description on failure.
    pub fn run<F>(&self, work: F) -> StressReport
    where
        F: Fn(usize, usize) -> std::result::Result<(), String> + Send + Sync + 'static,
    {
        let work = Arc::new(work);
        let handles: Vec<_> = (0..self.threads)
            .map(|tid| {
                let collector = self.collector.clone();
                let work = Arc::clone(&work);
                let count = Arc::clone(&self.request_count);
                let iters = self.iterations_per_thread;
                std::thread::spawn(move || {
                    for i in 0..iters {
                        let start = Instant::now();
                        match work(tid, i) {
                            Ok(()) => {
                                collector.record_pass(start.elapsed());
                            }
                            Err(e) => {
                                collector.record_fail(&e);
                            }
                        }
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("stress worker panicked");
        }

        self.collector.report()
    }

    /// Total requests completed so far (for progress monitoring).
    #[must_use]
    pub fn completed(&self) -> usize {
        self.request_count.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_collector_counts() {
        let c = ResultCollector::new();
        c.record_pass(Duration::from_millis(1));
        c.record_pass(Duration::from_millis(2));
        c.record_fail("oops");
        let r = c.report();
        assert_eq!(r.passed, 2);
        assert_eq!(r.failed, 1);
        assert!((r.pass_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn load_generator_increments() {
        let load_gen = LoadGenerator::new(32, 32);
        let r1 = load_gen.next_request();
        let r2 = load_gen.next_request();
        assert_eq!(r1.id, 0);
        assert_eq!(r2.id, 1);
        assert_eq!(load_gen.generated_count(), 2);
    }

    #[test]
    fn stress_runner_basic() {
        let runner = StressTestRunner::new(2, 5);
        let report = runner.run(|_tid, _i| Ok(()));
        assert_eq!(report.total_requests, 10);
        assert_eq!(report.passed, 10);
        assert_eq!(report.failed, 0);
    }
}
