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
    fn timing_metrics_default_has_no_phase_values() {
        let metrics = TimingMetrics::default();
        assert!(metrics.prefill_ms.is_none());
        assert!(metrics.decode_ms.is_none());
        assert!(metrics.tokenization_encode_ms.is_none());
        assert!(metrics.tokenization_decode_ms.is_none());
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
    fn timing_metrics_serde_round_trip() {
        let original = TimingMetrics {
            prefill_ms: Some(11),
            decode_ms: Some(22),
            tokenization_encode_ms: Some(3),
            tokenization_decode_ms: Some(4),
            total_ms: 40,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: TimingMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
    }

    #[test]
    fn throughput_metrics_serde_round_trip() {
        let original = ThroughputMetrics {
            prefill_tokens_per_sec: Some(123.5),
            decode_tokens_per_sec: Some(45.6),
            end_to_end_tokens_per_sec: 78.9,
            total_tokens: 42,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: ThroughputMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
    }

    #[test]
    fn timing_metrics_deserializes_with_missing_optional_fields() {
        let parsed: TimingMetrics =
            serde_json::from_str(r#"{"total_ms": 100}"#).expect("deserialize");
        assert_eq!(parsed.total_ms, 100);
        assert!(parsed.prefill_ms.is_none());
        assert!(parsed.decode_ms.is_none());
    }

    #[test]
    fn metrics_implement_clone() {
        let timing = TimingMetrics { total_ms: 5, ..TimingMetrics::default() };
        let cloned = timing.clone();
        assert_eq!(cloned, timing);

        let throughput = ThroughputMetrics {
            total_tokens: 7,
            end_to_end_tokens_per_sec: 1.0,
            ..ThroughputMetrics::default()
        };
        let cloned_t = throughput.clone();
        assert_eq!(cloned_t, throughput);
    }
}
