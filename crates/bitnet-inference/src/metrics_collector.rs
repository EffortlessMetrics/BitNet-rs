//! Inference metrics collection.
//!
//! Collect and aggregate inference performance metrics.

use std::time::Duration;

/// Single inference request metrics.
#[derive(Debug, Clone)]
pub struct RequestMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_duration: Duration,
    pub first_token_latency: Duration,
    pub success: bool,
}

impl RequestMetrics {
    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.generated_tokens as f64 / secs
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }
}

/// Aggregated metrics over multiple requests.
#[derive(Debug, Clone)]
pub struct AggregateMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_generated_tokens: u64,
    pub total_duration: Duration,
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub min_first_token: Duration,
    pub max_first_token: Duration,
    latency_sum: Duration,
    first_token_sum: Duration,
}

impl Default for AggregateMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregateMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_prompt_tokens: 0,
            total_generated_tokens: 0,
            total_duration: Duration::ZERO,
            min_latency: Duration::MAX,
            max_latency: Duration::ZERO,
            min_first_token: Duration::MAX,
            max_first_token: Duration::ZERO,
            latency_sum: Duration::ZERO,
            first_token_sum: Duration::ZERO,
        }
    }

    pub fn record(&mut self, m: &RequestMetrics) {
        self.total_requests += 1;
        if m.success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }
        self.total_prompt_tokens += m.prompt_tokens as u64;
        self.total_generated_tokens += m.generated_tokens as u64;
        self.total_duration += m.total_duration;
        if m.total_duration < self.min_latency {
            self.min_latency = m.total_duration;
        }
        if m.total_duration > self.max_latency {
            self.max_latency = m.total_duration;
        }
        if m.first_token_latency < self.min_first_token {
            self.min_first_token = m.first_token_latency;
        }
        if m.first_token_latency > self.max_first_token {
            self.max_first_token = m.first_token_latency;
        }
        self.latency_sum += m.total_duration;
        self.first_token_sum += m.first_token_latency;
    }

    pub fn avg_latency(&self) -> Duration {
        if self.total_requests == 0 {
            return Duration::ZERO;
        }
        self.latency_sum / self.total_requests as u32
    }

    pub fn avg_first_token(&self) -> Duration {
        if self.total_requests == 0 {
            return Duration::ZERO;
        }
        self.first_token_sum / self.total_requests as u32
    }

    pub fn avg_tokens_per_second(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_generated_tokens as f64 / secs
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.successful_requests as f64 / self.total_requests as f64
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_prompt_tokens + self.total_generated_tokens
    }

    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            requests: self.total_requests,
            success_rate: self.success_rate(),
            avg_latency_ms: self.avg_latency().as_secs_f64() * 1000.0,
            avg_first_token_ms: self.avg_first_token().as_secs_f64() * 1000.0,
            avg_tps: self.avg_tokens_per_second(),
            total_tokens: self.total_tokens(),
        }
    }
}

/// Metrics summary for reporting.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub requests: u64,
    pub success_rate: f64,
    pub avg_latency_ms: f64,
    pub avg_first_token_ms: f64,
    pub avg_tps: f64,
    pub total_tokens: u64,
}

/// Merge two aggregate metrics.
pub fn merge_metrics(a: &AggregateMetrics, b: &AggregateMetrics) -> AggregateMetrics {
    AggregateMetrics {
        total_requests: a.total_requests + b.total_requests,
        successful_requests: a.successful_requests + b.successful_requests,
        failed_requests: a.failed_requests + b.failed_requests,
        total_prompt_tokens: a.total_prompt_tokens + b.total_prompt_tokens,
        total_generated_tokens: a.total_generated_tokens + b.total_generated_tokens,
        total_duration: a.total_duration + b.total_duration,
        min_latency: a.min_latency.min(b.min_latency),
        max_latency: a.max_latency.max(b.max_latency),
        min_first_token: a.min_first_token.min(b.min_first_token),
        max_first_token: a.max_first_token.max(b.max_first_token),
        latency_sum: a.latency_sum + b.latency_sum,
        first_token_sum: a.first_token_sum + b.first_token_sum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics(success: bool) -> RequestMetrics {
        RequestMetrics {
            prompt_tokens: 10,
            generated_tokens: 20,
            total_duration: Duration::from_millis(500),
            first_token_latency: Duration::from_millis(50),
            success,
        }
    }

    #[test]
    fn test_request_tps() {
        let m = sample_metrics(true);
        assert!((m.tokens_per_second() - 40.0).abs() < 1.0);
    }

    #[test]
    fn test_request_total() {
        let m = sample_metrics(true);
        assert_eq!(m.total_tokens(), 30);
    }

    #[test]
    fn test_empty_aggregate() {
        let a = AggregateMetrics::new();
        assert_eq!(a.total_requests, 0);
        assert_eq!(a.success_rate(), 0.0);
        assert_eq!(a.avg_latency(), Duration::ZERO);
    }

    #[test]
    fn test_single_record() {
        let mut a = AggregateMetrics::new();
        a.record(&sample_metrics(true));
        assert_eq!(a.total_requests, 1);
        assert_eq!(a.successful_requests, 1);
        assert_eq!(a.total_generated_tokens, 20);
    }

    #[test]
    fn test_multiple_records() {
        let mut a = AggregateMetrics::new();
        a.record(&sample_metrics(true));
        a.record(&sample_metrics(true));
        a.record(&sample_metrics(false));
        assert_eq!(a.total_requests, 3);
        assert_eq!(a.failed_requests, 1);
        assert!((a.success_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_avg_latency() {
        let mut a = AggregateMetrics::new();
        a.record(&sample_metrics(true));
        a.record(&sample_metrics(true));
        assert_eq!(a.avg_latency(), Duration::from_millis(500));
    }

    #[test]
    fn test_summary() {
        let mut a = AggregateMetrics::new();
        a.record(&sample_metrics(true));
        let s = a.summary();
        assert_eq!(s.requests, 1);
        assert!((s.success_rate - 1.0).abs() < 0.01);
        assert!(s.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_merge() {
        let mut a = AggregateMetrics::new();
        a.record(&sample_metrics(true));
        let mut b = AggregateMetrics::new();
        b.record(&sample_metrics(false));
        let merged = merge_metrics(&a, &b);
        assert_eq!(merged.total_requests, 2);
        assert_eq!(merged.successful_requests, 1);
    }

    #[test]
    fn test_min_max_latency() {
        let mut a = AggregateMetrics::new();
        let mut m1 = sample_metrics(true);
        m1.total_duration = Duration::from_millis(100);
        let mut m2 = sample_metrics(true);
        m2.total_duration = Duration::from_millis(900);
        a.record(&m1);
        a.record(&m2);
        assert_eq!(a.min_latency, Duration::from_millis(100));
        assert_eq!(a.max_latency, Duration::from_millis(900));
    }

    #[test]
    fn test_total_tokens() {
        let mut a = AggregateMetrics::new();
        a.record(&sample_metrics(true));
        assert_eq!(a.total_tokens(), 30);
    }

    #[test]
    fn test_avg_first_token() {
        let mut a = AggregateMetrics::new();
        a.record(&sample_metrics(true));
        assert_eq!(a.avg_first_token(), Duration::from_millis(50));
    }

    #[test]
    fn test_zero_duration_tps() {
        let m = RequestMetrics {
            prompt_tokens: 0,
            generated_tokens: 0,
            total_duration: Duration::ZERO,
            first_token_latency: Duration::ZERO,
            success: true,
        };
        assert_eq!(m.tokens_per_second(), 0.0);
    }
}
