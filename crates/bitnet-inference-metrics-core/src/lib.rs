//! Shared DTOs for inference timing and throughput metrics.

use serde::{Deserialize, Serialize};

/// Enhanced timing metrics for detailed performance tracking.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TimingMetrics {
    /// Prefill latency in milliseconds
    pub prefill_ms: Option<u64>,
    /// Decode latency in milliseconds
    pub decode_ms: Option<u64>,
    /// Tokenization encode time in milliseconds
    pub tokenization_encode_ms: Option<u64>,
    /// Tokenization decode time in milliseconds
    pub tokenization_decode_ms: Option<u64>,
    /// End-to-end total time in milliseconds
    pub total_ms: u64,
}

/// Enhanced throughput metrics for performance analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThroughputMetrics {
    /// Prefill throughput in tokens per second
    pub prefill_tokens_per_sec: Option<f64>,
    /// Decode throughput in tokens per second
    pub decode_tokens_per_sec: Option<f64>,
    /// End-to-end throughput in tokens per second
    pub end_to_end_tokens_per_sec: f64,
    /// Total tokens generated
    pub total_tokens: usize,
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            prefill_tokens_per_sec: None,
            decode_tokens_per_sec: None,
            end_to_end_tokens_per_sec: 0.0,
            total_tokens: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ThroughputMetrics, TimingMetrics};

    #[test]
    fn timing_metrics_default_is_zeroed() {
        assert_eq!(TimingMetrics::default().total_ms, 0);
    }

    #[test]
    fn throughput_metrics_default_has_empty_phase_values() {
        let metrics = ThroughputMetrics::default();
        assert_eq!(metrics.prefill_tokens_per_sec, None);
        assert_eq!(metrics.decode_tokens_per_sec, None);
        assert_eq!(metrics.end_to_end_tokens_per_sec, 0.0);
        assert_eq!(metrics.total_tokens, 0);
    }
}
