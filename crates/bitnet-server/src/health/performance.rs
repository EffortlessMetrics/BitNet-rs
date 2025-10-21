//! Performance metrics collection for AC05 health checks
//!
//! This module provides real-time performance indicator tracking for:
//! - Average response time (moving average)
//! - Requests per second (sliding time window)
//! - Error rate (errors / total requests)
//! - SLA compliance (percentage of requests meeting SLA target)

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// SLA target for response time (milliseconds)
const SLA_TARGET_MS: f64 = 2000.0;

/// Time window for requests-per-second calculation (seconds)
const REQUESTS_WINDOW_SECS: u64 = 60;

/// Maximum number of samples to keep for moving averages
const MAX_SAMPLES: usize = 1000;

/// Request sample for performance tracking
#[derive(Debug, Clone)]
struct RequestSample {
    timestamp: Instant,
    duration_ms: f64,
    success: bool,
}

/// Performance metrics collector
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    samples: Arc<RwLock<Vec<RequestSample>>>,
    /// Timestamp when metrics collection started (for uptime tracking)
    #[allow(dead_code)] // Reserved for future uptime() method
    start_time: Instant,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMetrics {
    /// Create a new performance metrics collector
    pub fn new() -> Self {
        Self { samples: Arc::new(RwLock::new(Vec::new())), start_time: Instant::now() }
    }

    /// Record a completed request
    pub async fn record_request(&self, duration: Duration, success: bool) {
        let sample = RequestSample {
            timestamp: Instant::now(),
            duration_ms: duration.as_secs_f64() * 1000.0,
            success,
        };

        let mut samples = self.samples.write().await;
        samples.push(sample);

        // Evict old samples to prevent unbounded memory growth
        if samples.len() > MAX_SAMPLES {
            samples.remove(0);
        }
    }

    /// Calculate average response time (milliseconds)
    pub async fn avg_response_time_ms(&self) -> f64 {
        let samples = self.samples.read().await;

        if samples.is_empty() {
            return 0.0;
        }

        let total_ms: f64 = samples.iter().map(|s| s.duration_ms).sum();
        total_ms / samples.len() as f64
    }

    /// Calculate requests per second (sliding window)
    pub async fn requests_per_second(&self) -> f64 {
        let samples = self.samples.read().await;
        let now = Instant::now();
        let window_start = now - Duration::from_secs(REQUESTS_WINDOW_SECS);

        let recent_count = samples.iter().filter(|s| s.timestamp >= window_start).count();

        if recent_count == 0 {
            return 0.0;
        }

        recent_count as f64 / REQUESTS_WINDOW_SECS as f64
    }

    /// Calculate error rate (0.0 to 1.0)
    pub async fn error_rate(&self) -> f64 {
        let samples = self.samples.read().await;

        if samples.is_empty() {
            return 0.0;
        }

        let error_count = samples.iter().filter(|s| !s.success).count();
        error_count as f64 / samples.len() as f64
    }

    /// Calculate SLA compliance (0.0 to 1.0)
    pub async fn sla_compliance(&self) -> f64 {
        let samples = self.samples.read().await;

        if samples.is_empty() {
            return 1.0; // No requests = 100% compliance
        }

        let compliant_count = samples.iter().filter(|s| s.duration_ms <= SLA_TARGET_MS).count();

        compliant_count as f64 / samples.len() as f64
    }

    /// Get all performance indicators at once
    pub async fn get_indicators(&self) -> PerformanceIndicators {
        PerformanceIndicators {
            avg_response_time_ms: self.avg_response_time_ms().await,
            requests_per_second: self.requests_per_second().await,
            error_rate: self.error_rate().await,
            sla_compliance: self.sla_compliance().await,
        }
    }

    /// Get the number of samples currently tracked
    #[allow(dead_code)]
    pub async fn sample_count(&self) -> usize {
        self.samples.read().await.len()
    }

    /// Clear all samples (for testing)
    #[allow(dead_code)]
    pub async fn clear(&self) {
        self.samples.write().await.clear();
    }
}

/// Performance indicators snapshot
#[derive(Debug, Clone)]
pub struct PerformanceIndicators {
    pub avg_response_time_ms: f64,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub sla_compliance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_metrics_return_zero() {
        let metrics = PerformanceMetrics::new();

        assert_eq!(metrics.avg_response_time_ms().await, 0.0);
        assert_eq!(metrics.requests_per_second().await, 0.0);
        assert_eq!(metrics.error_rate().await, 0.0);
        assert_eq!(metrics.sla_compliance().await, 1.0); // No requests = 100% compliance
    }

    #[tokio::test]
    async fn test_avg_response_time_calculation() {
        let metrics = PerformanceMetrics::new();

        metrics.record_request(Duration::from_millis(100), true).await;
        metrics.record_request(Duration::from_millis(200), true).await;
        metrics.record_request(Duration::from_millis(300), true).await;

        let avg = metrics.avg_response_time_ms().await;
        assert!((avg - 200.0).abs() < 1.0, "Expected ~200ms, got {}", avg);
    }

    #[tokio::test]
    async fn test_error_rate_calculation() {
        let metrics = PerformanceMetrics::new();

        // 3 successes, 1 failure = 25% error rate
        metrics.record_request(Duration::from_millis(100), true).await;
        metrics.record_request(Duration::from_millis(100), true).await;
        metrics.record_request(Duration::from_millis(100), true).await;
        metrics.record_request(Duration::from_millis(100), false).await;

        let error_rate = metrics.error_rate().await;
        assert!((error_rate - 0.25).abs() < 0.01, "Expected 0.25, got {}", error_rate);
    }

    #[tokio::test]
    async fn test_sla_compliance_calculation() {
        let metrics = PerformanceMetrics::new();

        // 3 within SLA (≤2000ms), 1 outside SLA = 75% compliance
        metrics.record_request(Duration::from_millis(100), true).await;
        metrics.record_request(Duration::from_millis(500), true).await;
        metrics.record_request(Duration::from_millis(1500), true).await;
        metrics.record_request(Duration::from_millis(3000), true).await;

        let compliance = metrics.sla_compliance().await;
        assert!((compliance - 0.75).abs() < 0.01, "Expected 0.75, got {}", compliance);
    }

    #[tokio::test]
    async fn test_requests_per_second_window() {
        let metrics = PerformanceMetrics::new();

        // Record 10 requests instantly
        for _ in 0..10 {
            metrics.record_request(Duration::from_millis(100), true).await;
        }

        let rps = metrics.requests_per_second().await;
        // All 10 requests are within the 60-second window
        assert!((rps - 10.0 / 60.0).abs() < 0.01, "Expected ~0.167 rps, got {}", rps);
    }

    #[tokio::test]
    async fn test_get_indicators_snapshot() {
        let metrics = PerformanceMetrics::new();

        metrics.record_request(Duration::from_millis(1000), true).await;
        metrics.record_request(Duration::from_millis(1500), true).await;
        metrics.record_request(Duration::from_millis(2500), false).await;

        let indicators = metrics.get_indicators().await;

        assert!((indicators.avg_response_time_ms - 1666.67).abs() < 1.0);
        assert_eq!(indicators.error_rate, 1.0 / 3.0);
        assert_eq!(indicators.sla_compliance, 2.0 / 3.0); // 2 out of 3 within SLA
    }

    #[tokio::test]
    async fn test_sample_eviction() {
        let metrics = PerformanceMetrics::new();

        // Record more than MAX_SAMPLES
        for _ in 0..(MAX_SAMPLES + 100) {
            metrics.record_request(Duration::from_millis(100), true).await;
        }

        let count = metrics.sample_count().await;
        assert_eq!(count, MAX_SAMPLES, "Expected sample eviction to limit to MAX_SAMPLES");
    }

    #[tokio::test]
    async fn test_clear_samples() {
        let metrics = PerformanceMetrics::new();

        metrics.record_request(Duration::from_millis(100), true).await;
        metrics.record_request(Duration::from_millis(200), true).await;

        assert_eq!(metrics.sample_count().await, 2);

        metrics.clear().await;

        assert_eq!(metrics.sample_count().await, 0);
        assert_eq!(metrics.avg_response_time_ms().await, 0.0);
    }
}
