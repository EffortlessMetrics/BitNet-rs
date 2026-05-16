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

    #[test]
    fn timing_metrics_clone_and_equality() {
        let original = TimingMetrics {
            prefill_ms: Some(12),
            decode_ms: Some(345),
            tokenization_encode_ms: Some(1),
            tokenization_decode_ms: Some(2),
            total_ms: 360,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
        assert_ne!(cloned, TimingMetrics::default());
    }

    #[test]
    fn throughput_metrics_clone_and_equality() {
        let original = ThroughputMetrics {
            prefill_tokens_per_sec: Some(100.0),
            decode_tokens_per_sec: Some(50.0),
            end_to_end_tokens_per_sec: 75.0,
            total_tokens: 128,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
        assert_ne!(cloned, ThroughputMetrics::default());
    }

    #[test]
    fn timing_metrics_serde_round_trip() {
        let original = TimingMetrics {
            prefill_ms: Some(10),
            decode_ms: None,
            tokenization_encode_ms: Some(3),
            tokenization_decode_ms: None,
            total_ms: 42,
        };
        let json = serde_json::to_string(&original).expect("serialize timing metrics");
        let restored: TimingMetrics =
            serde_json::from_str(&json).expect("deserialize timing metrics");
        assert_eq!(original, restored);
    }

    #[test]
    fn timing_metrics_serializes_none_fields() {
        let metrics = TimingMetrics::default();
        let json = serde_json::to_value(&metrics).expect("serialize default timing metrics");
        // total_ms is non-optional and must always be present.
        assert_eq!(json.get("total_ms"), Some(&serde_json::json!(0)));
    }

    #[test]
    fn throughput_metrics_serde_round_trip() {
        let original = ThroughputMetrics {
            prefill_tokens_per_sec: Some(120.5),
            decode_tokens_per_sec: None,
            end_to_end_tokens_per_sec: 80.0,
            total_tokens: 64,
        };
        let json = serde_json::to_string(&original).expect("serialize throughput metrics");
        let restored: ThroughputMetrics =
            serde_json::from_str(&json).expect("deserialize throughput metrics");
        assert_eq!(original, restored);
    }

    #[test]
    fn throughput_metrics_default_serde_round_trip() {
        let original = ThroughputMetrics::default();
        let json = serde_json::to_string(&original).expect("serialize default throughput metrics");
        let restored: ThroughputMetrics =
            serde_json::from_str(&json).expect("deserialize default throughput metrics");
        assert_eq!(original, restored);
    }
}
