//! Aggregate inference performance metrics tracking.
//!
//! [`InferenceMetrics`] accumulates statistics across multiple inference
//! requests and provides throughput, latency percentiles, and a
//! [`MetricsSummary`] snapshot.

/// Aggregate tracker for inference performance across multiple requests.
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    pub total_tokens_generated: u64,
    pub total_prompt_tokens: u64,
    pub total_inference_time_ms: f64,
    pub request_count: u64,
    pub first_token_latencies_ms: Vec<f64>,
    pub inter_token_latencies_ms: Vec<f64>,
}

/// Point-in-time snapshot of all computed metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsSummary {
    pub tokens_per_second: f64,
    pub avg_first_token_ms: f64,
    pub p50_first_token_ms: f64,
    pub p95_first_token_ms: f64,
    pub p99_first_token_ms: f64,
    pub avg_inter_token_ms: f64,
    pub total_requests: u64,
    pub total_tokens: u64,
}

impl InferenceMetrics {
    /// Create an empty metrics tracker.
    pub fn new() -> Self {
        Self {
            total_tokens_generated: 0,
            total_prompt_tokens: 0,
            total_inference_time_ms: 0.0,
            request_count: 0,
            first_token_latencies_ms: Vec::new(),
            inter_token_latencies_ms: Vec::new(),
        }
    }

    /// Record a completed inference request.
    pub fn record_request(
        &mut self,
        prompt_tokens: usize,
        generated_tokens: usize,
        total_ms: f64,
        first_token_ms: f64,
    ) {
        self.total_prompt_tokens += prompt_tokens as u64;
        self.total_tokens_generated += generated_tokens as u64;
        self.total_inference_time_ms += total_ms;
        self.request_count += 1;
        self.first_token_latencies_ms.push(first_token_ms);

        // Derive average inter-token latency for this request.
        if generated_tokens > 1 {
            let remaining_ms = total_ms - first_token_ms;
            let itl = remaining_ms / (generated_tokens as f64 - 1.0);
            self.inter_token_latencies_ms.push(itl);
        }
    }

    /// Overall tokens-per-second throughput across all recorded requests.
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_inference_time_ms <= 0.0 {
            return 0.0;
        }
        (self.total_tokens_generated as f64) / (self.total_inference_time_ms / 1000.0)
    }

    /// Average first-token latency in milliseconds.
    pub fn avg_first_token_latency_ms(&self) -> f64 {
        if self.first_token_latencies_ms.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.first_token_latencies_ms.iter().sum();
        sum / self.first_token_latencies_ms.len() as f64
    }

    /// Percentile of first-token latency (e.g., 50.0 for p50, 99.0 for p99).
    pub fn percentile_first_token_ms(&self, p: f64) -> f64 {
        if self.first_token_latencies_ms.is_empty() {
            return 0.0;
        }
        let mut sorted = self.first_token_latencies_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0))
            .round()
            .clamp(0.0, (sorted.len() - 1) as f64) as usize;
        sorted[idx]
    }

    /// Average inter-token latency in milliseconds.
    pub fn avg_inter_token_latency_ms(&self) -> f64 {
        if self.inter_token_latencies_ms.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.inter_token_latencies_ms.iter().sum();
        sum / self.inter_token_latencies_ms.len() as f64
    }

    /// Clear all accumulated metrics.
    pub fn reset(&mut self) {
        self.total_tokens_generated = 0;
        self.total_prompt_tokens = 0;
        self.total_inference_time_ms = 0.0;
        self.request_count = 0;
        self.first_token_latencies_ms.clear();
        self.inter_token_latencies_ms.clear();
    }

    /// Produce a point-in-time snapshot of all computed metrics.
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            tokens_per_second: self.tokens_per_second(),
            avg_first_token_ms: self.avg_first_token_latency_ms(),
            p50_first_token_ms: self.percentile_first_token_ms(50.0),
            p95_first_token_ms: self.percentile_first_token_ms(95.0),
            p99_first_token_ms: self.percentile_first_token_ms(99.0),
            avg_inter_token_ms: self.avg_inter_token_latency_ms(),
            total_requests: self.request_count,
            total_tokens: self.total_tokens_generated,
        }
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_metrics_zeros() {
        let m = InferenceMetrics::new();
        assert_eq!(m.tokens_per_second(), 0.0);
        assert_eq!(m.avg_first_token_latency_ms(), 0.0);
        assert_eq!(m.avg_inter_token_latency_ms(), 0.0);
        assert_eq!(m.percentile_first_token_ms(50.0), 0.0);
        assert_eq!(m.request_count, 0);
    }

    #[test]
    fn test_empty_summary() {
        let s = InferenceMetrics::new().summary();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.total_tokens, 0);
        assert_eq!(s.tokens_per_second, 0.0);
    }

    #[test]
    fn test_single_request() {
        let mut m = InferenceMetrics::new();
        m.record_request(10, 20, 1000.0, 100.0);
        assert_eq!(m.request_count, 1);
        assert_eq!(m.total_tokens_generated, 20);
        assert_eq!(m.total_prompt_tokens, 10);
        assert!((m.tokens_per_second() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_first_token_latency_single() {
        let mut m = InferenceMetrics::new();
        m.record_request(5, 10, 500.0, 50.0);
        assert!((m.avg_first_token_latency_ms() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_multiple_requests_throughput() {
        let mut m = InferenceMetrics::new();
        m.record_request(10, 50, 1000.0, 100.0);
        m.record_request(10, 50, 1000.0, 100.0);
        // 100 tokens in 2000 ms = 50 tok/s
        assert!((m.tokens_per_second() - 50.0).abs() < 1e-9);
        assert_eq!(m.request_count, 2);
        assert_eq!(m.total_tokens_generated, 100);
    }

    #[test]
    fn test_avg_first_token_latency_multiple() {
        let mut m = InferenceMetrics::new();
        m.record_request(5, 10, 500.0, 40.0);
        m.record_request(5, 10, 500.0, 60.0);
        assert!((m.avg_first_token_latency_ms() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_inter_token_latency() {
        let mut m = InferenceMetrics::new();
        // 10 tokens, 500ms total, 50ms first token → 450ms / 9 = 50ms ITL
        m.record_request(5, 10, 500.0, 50.0);
        assert!((m.avg_inter_token_latency_ms() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_inter_token_latency_single_token() {
        let mut m = InferenceMetrics::new();
        // Only 1 generated token — no inter-token latency recorded.
        m.record_request(5, 1, 100.0, 100.0);
        assert_eq!(m.avg_inter_token_latency_ms(), 0.0);
    }

    #[test]
    fn test_percentile_p50() {
        let mut m = InferenceMetrics::new();
        // Record 10 requests with first-token latencies 10..=100
        for i in 1..=10 {
            m.record_request(5, 5, 200.0, i as f64 * 10.0);
        }
        let p50 = m.percentile_first_token_ms(50.0);
        // Median of [10,20,..,100] → 50 or 60 depending on rounding
        assert!((50.0..=60.0).contains(&p50));
    }

    #[test]
    fn test_percentile_p95_p99() {
        let mut m = InferenceMetrics::new();
        for i in 1..=100 {
            m.record_request(5, 5, 200.0, i as f64);
        }
        let p95 = m.percentile_first_token_ms(95.0);
        let p99 = m.percentile_first_token_ms(99.0);
        assert!((94.0..=96.0).contains(&p95));
        assert!((98.0..=100.0).contains(&p99));
    }

    #[test]
    fn test_reset_clears_everything() {
        let mut m = InferenceMetrics::new();
        m.record_request(10, 20, 1000.0, 100.0);
        m.reset();
        assert_eq!(m.request_count, 0);
        assert_eq!(m.total_tokens_generated, 0);
        assert_eq!(m.total_prompt_tokens, 0);
        assert_eq!(m.total_inference_time_ms, 0.0);
        assert!(m.first_token_latencies_ms.is_empty());
        assert!(m.inter_token_latencies_ms.is_empty());
        assert_eq!(m.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_zero_time_throughput() {
        let mut m = InferenceMetrics::new();
        m.record_request(5, 10, 0.0, 0.0);
        assert_eq!(m.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_large_values() {
        let mut m = InferenceMetrics::new();
        m.record_request(100_000, 1_000_000, 3_600_000.0, 500.0);
        assert_eq!(m.total_tokens_generated, 1_000_000);
        let tps = m.tokens_per_second();
        // 1M tokens / 3600s ≈ 277.78 tok/s
        assert!((tps - 277.778).abs() < 0.1);
    }

    #[test]
    fn test_summary_snapshot() {
        let mut m = InferenceMetrics::new();
        m.record_request(10, 20, 1000.0, 100.0);
        m.record_request(10, 30, 1500.0, 150.0);
        let s = m.summary();
        assert_eq!(s.total_requests, 2);
        assert_eq!(s.total_tokens, 50);
        // 50 tokens / 2.5s = 20 tok/s
        assert!((s.tokens_per_second - 20.0).abs() < 1e-9);
        assert!((s.avg_first_token_ms - 125.0).abs() < 1e-9);
    }
}
