//! Inference metrics collection and reporting.

use std::time::{Duration, Instant};

/// Metrics collected during a single inference run.
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_duration: Duration,
    pub prefill_duration: Duration,
    pub decode_durations: Vec<Duration>,
    pub peak_memory_bytes: Option<u64>,
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self {
            prompt_tokens: 0,
            generated_tokens: 0,
            total_duration: Duration::ZERO,
            prefill_duration: Duration::ZERO,
            decode_durations: Vec::new(),
            peak_memory_bytes: None,
        }
    }

    /// Tokens per second (decode phase).
    pub fn tokens_per_second(&self) -> f64 {
        let decode_total: Duration = self.decode_durations.iter().sum();
        if decode_total.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.generated_tokens as f64 / decode_total.as_secs_f64()
    }

    /// Time to first token (prefill latency).
    pub fn time_to_first_token(&self) -> Duration {
        self.prefill_duration
    }

    /// Average time per token (decode).
    pub fn avg_token_latency(&self) -> Duration {
        if self.decode_durations.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.decode_durations.iter().sum();
        total / self.decode_durations.len() as u32
    }

    /// P50 token latency.
    pub fn p50_latency(&self) -> Duration {
        percentile_duration(&self.decode_durations, 50.0)
    }

    /// P95 token latency.
    pub fn p95_latency(&self) -> Duration {
        percentile_duration(&self.decode_durations, 95.0)
    }

    /// P99 token latency.
    pub fn p99_latency(&self) -> Duration {
        percentile_duration(&self.decode_durations, 99.0)
    }

    /// Total tokens (prompt + generated).
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

fn percentile_duration(durations: &[Duration], pct: f64) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted: Vec<Duration> = durations.to_vec();
    sorted.sort();
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Builder for collecting metrics incrementally.
#[derive(Debug)]
pub struct MetricsCollector {
    start_time: Option<Instant>,
    prefill_end: Option<Instant>,
    prompt_tokens: usize,
    decode_durations: Vec<Duration>,
    last_token_time: Option<Instant>,
    peak_memory: Option<u64>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: None,
            prefill_end: None,
            prompt_tokens: 0,
            decode_durations: Vec::new(),
            last_token_time: None,
            peak_memory: None,
        }
    }

    pub fn start(&mut self, prompt_tokens: usize) {
        self.start_time = Some(Instant::now());
        self.prompt_tokens = prompt_tokens;
    }

    pub fn mark_prefill_done(&mut self) {
        self.prefill_end = Some(Instant::now());
        self.last_token_time = self.prefill_end;
    }

    pub fn record_token(&mut self) {
        let now = Instant::now();
        if let Some(last) = self.last_token_time {
            self.decode_durations.push(now - last);
        }
        self.last_token_time = Some(now);
    }

    pub fn set_peak_memory(&mut self, bytes: u64) {
        self.peak_memory = Some(bytes);
    }

    pub fn finish(self) -> InferenceMetrics {
        let total_duration = self.start_time.map(|s| s.elapsed()).unwrap_or(Duration::ZERO);
        let prefill_duration = match (self.start_time, self.prefill_end) {
            (Some(start), Some(end)) => end.duration_since(start),
            _ => Duration::ZERO,
        };
        InferenceMetrics {
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.decode_durations.len(),
            total_duration,
            prefill_duration,
            decode_durations: self.decode_durations,
            peak_memory_bytes: self.peak_memory,
        }
    }

    pub fn tokens_so_far(&self) -> usize {
        self.decode_durations.len()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Format metrics as a human-readable summary.
pub fn format_metrics(metrics: &InferenceMetrics) -> String {
    let mut out = String::from("=== Inference Metrics ===\n");
    out.push_str(&format!("Prompt tokens:    {}\n", metrics.prompt_tokens));
    out.push_str(&format!("Generated tokens: {}\n", metrics.generated_tokens));
    out.push_str(&format!("Total time:       {:.2?}\n", metrics.total_duration));
    out.push_str(&format!("TTFT:             {:.2?}\n", metrics.time_to_first_token()));
    out.push_str(&format!("Tokens/sec:       {:.2}\n", metrics.tokens_per_second()));
    out.push_str(&format!("Avg latency:      {:.2?}\n", metrics.avg_token_latency()));
    out.push_str(&format!("P50 latency:      {:.2?}\n", metrics.p50_latency()));
    out.push_str(&format!("P95 latency:      {:.2?}\n", metrics.p95_latency()));
    out.push_str(&format!("P99 latency:      {:.2?}\n", metrics.p99_latency()));
    if let Some(mem) = metrics.peak_memory_bytes {
        out.push_str(&format!("Peak memory:      {:.1} MB\n", mem as f64 / 1e6));
    }
    out
}

/// Aggregate metrics from multiple runs.
#[derive(Debug, Clone)]
pub struct AggregateMetrics {
    pub runs: usize,
    pub total_tokens: usize,
    pub total_duration: Duration,
    pub avg_tps: f64,
    pub min_tps: f64,
    pub max_tps: f64,
}

impl AggregateMetrics {
    pub fn from_runs(metrics: &[InferenceMetrics]) -> Self {
        let runs = metrics.len();
        let total_tokens: usize = metrics.iter().map(|m| m.generated_tokens).sum();
        let total_duration: Duration = metrics.iter().map(|m| m.total_duration).sum();
        let tps_values: Vec<f64> = metrics.iter().map(|m| m.tokens_per_second()).collect();
        let avg_tps = if tps_values.is_empty() {
            0.0
        } else {
            tps_values.iter().sum::<f64>() / tps_values.len() as f64
        };
        let min_tps = tps_values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_tps = tps_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self {
            runs,
            total_tokens,
            total_duration,
            avg_tps,
            min_tps: if min_tps.is_infinite() { 0.0 } else { min_tps },
            max_tps: if max_tps.is_infinite() { 0.0 } else { max_tps },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics() -> InferenceMetrics {
        InferenceMetrics {
            prompt_tokens: 10,
            generated_tokens: 5,
            total_duration: Duration::from_millis(500),
            prefill_duration: Duration::from_millis(100),
            decode_durations: vec![
                Duration::from_millis(50),
                Duration::from_millis(60),
                Duration::from_millis(70),
                Duration::from_millis(80),
                Duration::from_millis(90),
            ],
            peak_memory_bytes: Some(1_000_000),
        }
    }

    #[test]
    fn test_metrics_new() {
        let m = InferenceMetrics::new();
        assert_eq!(m.prompt_tokens, 0);
        assert_eq!(m.generated_tokens, 0);
    }

    #[test]
    fn test_tokens_per_second() {
        let m = sample_metrics();
        let tps = m.tokens_per_second();
        // 5 tokens / 0.35s ≈ 14.3
        assert!(tps > 10.0 && tps < 20.0);
    }

    #[test]
    fn test_tokens_per_second_zero() {
        let m = InferenceMetrics::new();
        assert_eq!(m.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_ttft() {
        let m = sample_metrics();
        assert_eq!(m.time_to_first_token(), Duration::from_millis(100));
    }

    #[test]
    fn test_avg_latency() {
        let m = sample_metrics();
        let avg = m.avg_token_latency();
        assert!(avg > Duration::from_millis(60) && avg < Duration::from_millis(80));
    }

    #[test]
    fn test_avg_latency_empty() {
        let m = InferenceMetrics::new();
        assert_eq!(m.avg_token_latency(), Duration::ZERO);
    }

    #[test]
    fn test_p50_latency() {
        let m = sample_metrics();
        assert_eq!(m.p50_latency(), Duration::from_millis(70));
    }

    #[test]
    fn test_p95_latency() {
        let m = sample_metrics();
        let p95 = m.p95_latency();
        assert!(p95 >= Duration::from_millis(80));
    }

    #[test]
    fn test_p99_latency() {
        let m = sample_metrics();
        let p99 = m.p99_latency();
        assert!(p99 >= Duration::from_millis(80));
    }

    #[test]
    fn test_total_tokens() {
        let m = sample_metrics();
        assert_eq!(m.total_tokens(), 15);
    }

    #[test]
    fn test_default() {
        let m = InferenceMetrics::default();
        assert_eq!(m.total_tokens(), 0);
    }

    #[test]
    fn test_collector_new() {
        let c = MetricsCollector::new();
        assert_eq!(c.tokens_so_far(), 0);
    }

    #[test]
    fn test_collector_default() {
        let c = MetricsCollector::default();
        assert_eq!(c.tokens_so_far(), 0);
    }

    #[test]
    fn test_collector_lifecycle() {
        let mut c = MetricsCollector::new();
        c.start(10);
        std::thread::sleep(Duration::from_millis(10));
        c.mark_prefill_done();
        c.record_token();
        c.record_token();
        c.record_token();
        c.set_peak_memory(500_000);
        let m = c.finish();
        assert_eq!(m.prompt_tokens, 10);
        assert_eq!(m.generated_tokens, 3);
        assert!(m.total_duration > Duration::ZERO);
        assert!(m.prefill_duration > Duration::ZERO);
        assert_eq!(m.peak_memory_bytes, Some(500_000));
    }

    #[test]
    fn test_collector_finish_no_start() {
        let c = MetricsCollector::new();
        let m = c.finish();
        assert_eq!(m.total_duration, Duration::ZERO);
    }

    #[test]
    fn test_format_metrics() {
        let m = sample_metrics();
        let out = format_metrics(&m);
        assert!(out.contains("Inference Metrics"));
        assert!(out.contains("Prompt tokens:"));
        assert!(out.contains("Tokens/sec:"));
        assert!(out.contains("Peak memory:"));
    }

    #[test]
    fn test_format_metrics_no_memory() {
        let mut m = sample_metrics();
        m.peak_memory_bytes = None;
        let out = format_metrics(&m);
        assert!(!out.contains("Peak memory:"));
    }

    #[test]
    fn test_aggregate_metrics() {
        let m1 = sample_metrics();
        let m2 = sample_metrics();
        let agg = AggregateMetrics::from_runs(&[m1, m2]);
        assert_eq!(agg.runs, 2);
        assert_eq!(agg.total_tokens, 10);
        assert!(agg.avg_tps > 0.0);
    }

    #[test]
    fn test_aggregate_empty() {
        let agg = AggregateMetrics::from_runs(&[]);
        assert_eq!(agg.runs, 0);
        assert_eq!(agg.avg_tps, 0.0);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile_duration(&[], 50.0), Duration::ZERO);
    }

    #[test]
    fn test_percentile_single() {
        let d = vec![Duration::from_millis(100)];
        assert_eq!(percentile_duration(&d, 50.0), Duration::from_millis(100));
    }
}
