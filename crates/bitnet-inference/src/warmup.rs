//! Inference warmup utilities.
//!
//! Pre-warm caches, memory allocators, and kernel dispatchers
//! before serving real inference requests.

use std::time::{Duration, Instant};

/// Configuration for warmup runs.
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Number of warmup iterations.
    pub iterations: usize,
    /// Sequence length to use for warmup tokens.
    pub seq_len: usize,
    /// Whether to warmup KV cache allocation.
    pub warmup_kv_cache: bool,
    /// Whether to warmup kernel dispatch.
    pub warmup_kernels: bool,
    /// Maximum time to spend warming up.
    pub timeout: Duration,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            seq_len: 32,
            warmup_kv_cache: true,
            warmup_kernels: true,
            timeout: Duration::from_secs(30),
        }
    }
}

impl WarmupConfig {
    pub fn fast() -> Self {
        Self {
            iterations: 1,
            seq_len: 8,
            warmup_kv_cache: false,
            warmup_kernels: true,
            timeout: Duration::from_secs(5),
        }
    }

    pub fn thorough() -> Self {
        Self {
            iterations: 5,
            seq_len: 128,
            warmup_kv_cache: true,
            warmup_kernels: true,
            timeout: Duration::from_mins(1),
        }
    }

    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    pub fn with_seq_len(mut self, len: usize) -> Self {
        self.seq_len = len;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Result of a warmup run.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    pub iterations_completed: usize,
    pub total_time: Duration,
    pub iteration_times: Vec<Duration>,
    pub timed_out: bool,
    pub status: WarmupStatus,
}

/// Status of the warmup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupStatus {
    /// All warmup iterations completed.
    Complete,
    /// Warmup timed out before completing all iterations.
    TimedOut,
    /// Warmup was skipped (not configured).
    Skipped,
}

impl WarmupResult {
    /// Average time per warmup iteration.
    pub fn avg_iteration_time(&self) -> Duration {
        if self.iterations_completed == 0 {
            return Duration::ZERO;
        }
        self.total_time / self.iterations_completed as u32
    }

    /// Whether warmup completed successfully.
    pub fn is_success(&self) -> bool {
        self.status == WarmupStatus::Complete
    }

    /// Speedup from first to last iteration (warmup effectiveness).
    pub fn speedup_ratio(&self) -> Option<f64> {
        if self.iteration_times.len() < 2 {
            return None;
        }
        let first = self.iteration_times.first()?.as_secs_f64();
        let last = self.iteration_times.last()?.as_secs_f64();
        if last == 0.0 {
            return None;
        }
        Some(first / last)
    }
}

impl std::fmt::Display for WarmupResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Warmup: {} iterations in {:.2?} (avg {:.2?}, status={:?})",
            self.iterations_completed,
            self.total_time,
            self.avg_iteration_time(),
            self.status,
        )
    }
}

/// Execute warmup with a custom iteration function.
pub fn run_warmup<F>(config: &WarmupConfig, mut iteration_fn: F) -> WarmupResult
where
    F: FnMut(usize) -> Duration,
{
    let start = Instant::now();
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut completed = 0;
    let mut timed_out = false;

    for i in 0..config.iterations {
        if start.elapsed() >= config.timeout {
            timed_out = true;
            break;
        }
        let dur = iteration_fn(i);
        iteration_times.push(dur);
        completed += 1;
    }

    let total = start.elapsed();
    let status = if timed_out { WarmupStatus::TimedOut } else { WarmupStatus::Complete };

    WarmupResult {
        iterations_completed: completed,
        total_time: total,
        iteration_times,
        timed_out,
        status,
    }
}

/// Create a skipped warmup result.
pub fn skip_warmup() -> WarmupResult {
    WarmupResult {
        iterations_completed: 0,
        total_time: Duration::ZERO,
        iteration_times: vec![],
        timed_out: false,
        status: WarmupStatus::Skipped,
    }
}

/// Generate synthetic token IDs for warmup.
pub fn synthetic_tokens(seq_len: usize, vocab_size: u32) -> Vec<u32> {
    (0..seq_len).map(|i| (i as u32) % vocab_size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let c = WarmupConfig::default();
        assert_eq!(c.iterations, 3);
        assert_eq!(c.seq_len, 32);
        assert!(c.warmup_kv_cache);
    }

    #[test]
    fn test_fast_config() {
        let c = WarmupConfig::fast();
        assert_eq!(c.iterations, 1);
        assert_eq!(c.seq_len, 8);
        assert!(!c.warmup_kv_cache);
    }

    #[test]
    fn test_thorough_config() {
        let c = WarmupConfig::thorough();
        assert_eq!(c.iterations, 5);
        assert_eq!(c.seq_len, 128);
    }

    #[test]
    fn test_config_builder() {
        let c = WarmupConfig::default()
            .with_iterations(10)
            .with_seq_len(64)
            .with_timeout(Duration::from_mins(2));
        assert_eq!(c.iterations, 10);
        assert_eq!(c.seq_len, 64);
        assert_eq!(c.timeout, Duration::from_mins(2));
    }

    #[test]
    fn test_run_warmup() {
        let config = WarmupConfig::default().with_iterations(3);
        let result = run_warmup(&config, |_| {
            std::thread::sleep(Duration::from_millis(1));
            Duration::from_millis(1)
        });
        assert_eq!(result.iterations_completed, 3);
        assert!(result.is_success());
        assert_eq!(result.status, WarmupStatus::Complete);
    }

    #[test]
    fn test_warmup_timeout() {
        let config =
            WarmupConfig::default().with_iterations(1000).with_timeout(Duration::from_millis(50));
        let result = run_warmup(&config, |_| {
            std::thread::sleep(Duration::from_millis(20));
            Duration::from_millis(20)
        });
        assert!(result.iterations_completed < 1000);
        assert!(result.timed_out);
        assert_eq!(result.status, WarmupStatus::TimedOut);
    }

    #[test]
    fn test_avg_iteration_time() {
        let result = WarmupResult {
            iterations_completed: 2,
            total_time: Duration::from_millis(100),
            iteration_times: vec![Duration::from_millis(60), Duration::from_millis(40)],
            timed_out: false,
            status: WarmupStatus::Complete,
        };
        assert_eq!(result.avg_iteration_time(), Duration::from_millis(50));
    }

    #[test]
    fn test_avg_iteration_time_zero() {
        let result = skip_warmup();
        assert_eq!(result.avg_iteration_time(), Duration::ZERO);
    }

    #[test]
    fn test_speedup_ratio() {
        let result = WarmupResult {
            iterations_completed: 3,
            total_time: Duration::from_millis(300),
            iteration_times: vec![
                Duration::from_millis(200),
                Duration::from_millis(100),
                Duration::from_millis(100),
            ],
            timed_out: false,
            status: WarmupStatus::Complete,
        };
        let ratio = result.speedup_ratio().unwrap();
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_speedup_ratio_single() {
        let result = WarmupResult {
            iterations_completed: 1,
            total_time: Duration::from_millis(100),
            iteration_times: vec![Duration::from_millis(100)],
            timed_out: false,
            status: WarmupStatus::Complete,
        };
        assert!(result.speedup_ratio().is_none());
    }

    #[test]
    fn test_skip_warmup() {
        let r = skip_warmup();
        assert_eq!(r.status, WarmupStatus::Skipped);
        assert_eq!(r.iterations_completed, 0);
        assert!(!r.is_success());
    }

    #[test]
    fn test_synthetic_tokens() {
        let tokens = synthetic_tokens(10, 100);
        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0], 0);
        assert_eq!(tokens[9], 9);
    }

    #[test]
    fn test_synthetic_tokens_wrapping() {
        let tokens = synthetic_tokens(5, 3);
        assert_eq!(tokens, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn test_warmup_display() {
        let result = WarmupResult {
            iterations_completed: 3,
            total_time: Duration::from_millis(150),
            iteration_times: vec![
                Duration::from_millis(60),
                Duration::from_millis(50),
                Duration::from_millis(40),
            ],
            timed_out: false,
            status: WarmupStatus::Complete,
        };
        let s = format!("{result}");
        assert!(s.contains("3 iterations"));
        assert!(s.contains("Complete"));
    }

    #[test]
    fn test_warmup_zero_iterations() {
        let config = WarmupConfig::default().with_iterations(0);
        let result = run_warmup(&config, |_| Duration::from_millis(1));
        assert_eq!(result.iterations_completed, 0);
        assert!(result.is_success()); // 0 of 0 = complete
    }

    #[test]
    fn test_is_success() {
        assert!(
            WarmupResult {
                iterations_completed: 1,
                total_time: Duration::from_millis(1),
                iteration_times: vec![Duration::from_millis(1)],
                timed_out: false,
                status: WarmupStatus::Complete,
            }
            .is_success()
        );

        assert!(
            !WarmupResult {
                iterations_completed: 0,
                total_time: Duration::ZERO,
                iteration_times: vec![],
                timed_out: true,
                status: WarmupStatus::TimedOut,
            }
            .is_success()
        );
    }
}
