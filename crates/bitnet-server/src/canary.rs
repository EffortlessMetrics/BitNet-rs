//! Canary deployment for GPU backends.
//!
//! `CanaryRouter` routes a configurable percentage of inference traffic to a
//! new ("canary") backend while the rest goes to the baseline. It compares
//! outputs within a tolerance, tracks error rates, and automatically rolls
//! back if the canary exceeds an error threshold.

use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for canary deployment.
#[derive(Debug, Clone)]
pub struct CanaryConfig {
    /// Percentage of traffic routed to the canary backend (0.0–1.0).
    pub canary_ratio: f64,
    /// Maximum tolerable output divergence (L2 norm).
    pub output_tolerance: f64,
    /// Error rate threshold above which automatic rollback triggers (0.0–1.0).
    pub rollback_threshold: f64,
    /// Minimum number of requests before rollback decisions are made.
    pub min_requests_for_decision: u64,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            canary_ratio: 0.05,
            output_tolerance: 1e-3,
            rollback_threshold: 0.10,
            min_requests_for_decision: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Backend identification
// ---------------------------------------------------------------------------

/// Which backend should handle a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Baseline,
    Canary,
}

impl fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Baseline => f.write_str("baseline"),
            Self::Canary => f.write_str("canary"),
        }
    }
}

// ---------------------------------------------------------------------------
// Output comparison
// ---------------------------------------------------------------------------

/// Result of comparing canary output to baseline output.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// L2 divergence between baseline and canary outputs.
    pub divergence: f64,
    /// Whether the divergence is within tolerance.
    pub within_tolerance: bool,
    /// Latency of the baseline backend.
    pub baseline_latency: Duration,
    /// Latency of the canary backend.
    pub canary_latency: Duration,
}

/// Compare two output vectors and return the L2 divergence.
pub fn compute_divergence(baseline: &[f32], canary: &[f32]) -> f64 {
    if baseline.len() != canary.len() {
        return f64::INFINITY;
    }
    let sum_sq: f64 =
        baseline.iter().zip(canary.iter()).map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2)).sum();
    sum_sq.sqrt()
}

// ---------------------------------------------------------------------------
// Canary metrics
// ---------------------------------------------------------------------------

/// Metrics collected during canary deployment.
#[derive(Debug)]
pub struct CanaryMetrics {
    /// Total requests routed to baseline.
    pub baseline_requests: AtomicU64,
    /// Total requests routed to canary.
    pub canary_requests: AtomicU64,
    /// Number of canary responses that exceeded tolerance.
    pub canary_errors: AtomicU64,
    /// Cumulative baseline latency in microseconds.
    pub baseline_latency_us: AtomicU64,
    /// Cumulative canary latency in microseconds.
    pub canary_latency_us: AtomicU64,
    /// Cumulative divergence (scaled by 1e6 for integer storage).
    pub total_divergence_scaled: AtomicU64,
}

impl CanaryMetrics {
    pub fn new() -> Self {
        Self {
            baseline_requests: AtomicU64::new(0),
            canary_requests: AtomicU64::new(0),
            canary_errors: AtomicU64::new(0),
            baseline_latency_us: AtomicU64::new(0),
            canary_latency_us: AtomicU64::new(0),
            total_divergence_scaled: AtomicU64::new(0),
        }
    }

    /// Current canary error rate.
    pub fn error_rate(&self) -> f64 {
        let total = self.canary_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.canary_errors.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Average baseline latency.
    pub fn avg_baseline_latency(&self) -> Duration {
        let reqs = self.baseline_requests.load(Ordering::Relaxed);
        if reqs == 0 {
            return Duration::ZERO;
        }
        Duration::from_micros(self.baseline_latency_us.load(Ordering::Relaxed) / reqs)
    }

    /// Average canary latency.
    pub fn avg_canary_latency(&self) -> Duration {
        let reqs = self.canary_requests.load(Ordering::Relaxed);
        if reqs == 0 {
            return Duration::ZERO;
        }
        Duration::from_micros(self.canary_latency_us.load(Ordering::Relaxed) / reqs)
    }

    /// Average output divergence.
    pub fn avg_divergence(&self) -> f64 {
        let reqs = self.canary_requests.load(Ordering::Relaxed);
        if reqs == 0 {
            return 0.0;
        }
        (self.total_divergence_scaled.load(Ordering::Relaxed) as f64) / 1e6 / reqs as f64
    }
}

impl Default for CanaryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CanaryRouter
// ---------------------------------------------------------------------------

/// Routes traffic between baseline and canary GPU backends.
pub struct CanaryRouter {
    config: CanaryConfig,
    /// Counter used for deterministic routing.
    request_counter: AtomicU64,
    /// Whether the canary has been rolled back.
    rolled_back: AtomicBool,
    /// Reason for rollback, if any.
    rollback_reason: Mutex<Option<String>>,
    /// Runtime metrics.
    pub metrics: CanaryMetrics,
}

impl CanaryRouter {
    /// Create a new canary router.
    pub fn new(config: CanaryConfig) -> Self {
        Self {
            config,
            request_counter: AtomicU64::new(0),
            rolled_back: AtomicBool::new(false),
            rollback_reason: Mutex::new(None),
            metrics: CanaryMetrics::new(),
        }
    }

    /// Route a request to either baseline or canary.
    pub fn route(&self) -> BackendChoice {
        if self.rolled_back.load(Ordering::Relaxed) {
            return BackendChoice::Baseline;
        }

        let seq = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let canary_every_n = if self.config.canary_ratio > 0.0 {
            (1.0 / self.config.canary_ratio).ceil() as u64
        } else {
            return BackendChoice::Baseline;
        };

        if seq % canary_every_n == 0 { BackendChoice::Canary } else { BackendChoice::Baseline }
    }

    /// Record a comparison result and check for automatic rollback.
    pub fn record_comparison(&self, result: &ComparisonResult) {
        self.metrics.canary_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .canary_latency_us
            .fetch_add(result.canary_latency.as_micros() as u64, Ordering::Relaxed);
        self.metrics
            .baseline_latency_us
            .fetch_add(result.baseline_latency.as_micros() as u64, Ordering::Relaxed);
        self.metrics.baseline_requests.fetch_add(1, Ordering::Relaxed);

        let div_scaled = (result.divergence * 1e6) as u64;
        self.metrics.total_divergence_scaled.fetch_add(div_scaled, Ordering::Relaxed);

        if !result.within_tolerance {
            self.metrics.canary_errors.fetch_add(1, Ordering::Relaxed);
        }

        // Check automatic rollback.
        let total = self.metrics.canary_requests.load(Ordering::Relaxed);
        if total >= self.config.min_requests_for_decision {
            let error_rate = self.metrics.error_rate();
            if error_rate > self.config.rollback_threshold {
                self.trigger_rollback(format!(
                    "error rate {error_rate:.2} exceeds threshold {:.2}",
                    self.config.rollback_threshold
                ));
            }
        }
    }

    /// Record a baseline-only request.
    pub fn record_baseline(&self, latency: Duration) {
        self.metrics.baseline_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics.baseline_latency_us.fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }

    /// Manually trigger rollback.
    pub fn trigger_rollback(&self, reason: String) {
        self.rolled_back.store(true, Ordering::Relaxed);
        let mut lock = self.rollback_reason.lock().unwrap();
        if lock.is_none() {
            *lock = Some(reason);
        }
    }

    /// Whether the canary has been rolled back.
    pub fn is_rolled_back(&self) -> bool {
        self.rolled_back.load(Ordering::Relaxed)
    }

    /// Get the rollback reason, if any.
    pub fn rollback_reason(&self) -> Option<String> {
        self.rollback_reason.lock().unwrap().clone()
    }

    /// Compare two output vectors using the configured tolerance.
    pub fn compare_outputs(
        &self,
        baseline: &[f32],
        canary: &[f32],
        baseline_latency: Duration,
        canary_latency: Duration,
    ) -> ComparisonResult {
        let divergence = compute_divergence(baseline, canary);
        let within_tolerance = divergence <= self.config.output_tolerance;
        ComparisonResult { divergence, within_tolerance, baseline_latency, canary_latency }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_router() -> CanaryRouter {
        CanaryRouter::new(CanaryConfig::default())
    }

    #[test]
    fn test_route_respects_canary_ratio() {
        let config = CanaryConfig { canary_ratio: 0.5, ..Default::default() };
        let router = CanaryRouter::new(config);

        let mut canary_count = 0;
        for _ in 0..100 {
            if router.route() == BackendChoice::Canary {
                canary_count += 1;
            }
        }
        assert!(canary_count >= 40 && canary_count <= 60, "got {canary_count}");
    }

    #[test]
    fn test_route_all_baseline_when_ratio_zero() {
        let config = CanaryConfig { canary_ratio: 0.0, ..Default::default() };
        let router = CanaryRouter::new(config);

        for _ in 0..100 {
            assert_eq!(router.route(), BackendChoice::Baseline);
        }
    }

    #[test]
    fn test_automatic_rollback_on_high_error_rate() {
        let config = CanaryConfig {
            canary_ratio: 1.0,
            output_tolerance: 0.01,
            rollback_threshold: 0.10,
            min_requests_for_decision: 5,
        };
        let router = CanaryRouter::new(config);

        for _ in 0..5 {
            let result = ComparisonResult {
                divergence: 1.0,
                within_tolerance: false,
                baseline_latency: Duration::from_millis(10),
                canary_latency: Duration::from_millis(15),
            };
            router.record_comparison(&result);
        }

        assert!(router.is_rolled_back());
        assert!(router.rollback_reason().is_some());
    }

    #[test]
    fn test_no_rollback_below_threshold() {
        let config = CanaryConfig {
            rollback_threshold: 0.50,
            min_requests_for_decision: 4,
            ..Default::default()
        };
        let router = CanaryRouter::new(config);

        for i in 0..4 {
            let result = ComparisonResult {
                divergence: if i == 0 { 1.0 } else { 0.0 },
                within_tolerance: i != 0,
                baseline_latency: Duration::from_millis(10),
                canary_latency: Duration::from_millis(10),
            };
            router.record_comparison(&result);
        }

        assert!(!router.is_rolled_back());
    }

    #[test]
    fn test_route_always_baseline_after_rollback() {
        let router = default_router();
        router.trigger_rollback("test rollback".into());

        for _ in 0..50 {
            assert_eq!(router.route(), BackendChoice::Baseline);
        }
    }

    #[test]
    fn test_compute_divergence_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert!((compute_divergence(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_divergence_different_lengths() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        assert!(compute_divergence(&a, &b).is_infinite());
    }

    #[test]
    fn test_metrics_tracking() {
        let router = default_router();

        let result = ComparisonResult {
            divergence: 0.0005,
            within_tolerance: true,
            baseline_latency: Duration::from_millis(10),
            canary_latency: Duration::from_millis(12),
        };
        router.record_comparison(&result);

        assert_eq!(router.metrics.canary_requests.load(Ordering::Relaxed), 1);
        assert!((router.metrics.error_rate()).abs() < 1e-10);
        assert!(router.metrics.avg_canary_latency() >= Duration::from_millis(10));
    }

    #[test]
    fn test_compare_outputs_within_tolerance() {
        let router =
            CanaryRouter::new(CanaryConfig { output_tolerance: 1.0, ..Default::default() });

        let baseline = vec![1.0f32, 2.0, 3.0];
        let canary = vec![1.0f32, 2.0, 3.1];
        let result = router.compare_outputs(
            &baseline,
            &canary,
            Duration::from_millis(5),
            Duration::from_millis(6),
        );

        assert!(result.within_tolerance);
        assert!(result.divergence < 1.0);
    }

    #[test]
    fn test_compare_outputs_exceeds_tolerance() {
        let router =
            CanaryRouter::new(CanaryConfig { output_tolerance: 0.001, ..Default::default() });

        let baseline = vec![1.0f32, 2.0, 3.0];
        let canary = vec![10.0f32, 20.0, 30.0];
        let result = router.compare_outputs(
            &baseline,
            &canary,
            Duration::from_millis(5),
            Duration::from_millis(50),
        );

        assert!(!result.within_tolerance);
    }

    #[test]
    fn test_default_config_values() {
        let config = CanaryConfig::default();
        assert!((config.canary_ratio - 0.05).abs() < 1e-10);
        assert!((config.output_tolerance - 1e-3).abs() < 1e-10);
        assert!((config.rollback_threshold - 0.10).abs() < 1e-10);
        assert_eq!(config.min_requests_for_decision, 100);
    }
}
