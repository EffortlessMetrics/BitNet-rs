//! Reusable streaming generation event contracts for `BitNet` inference.

use bitnet_generation_stop_core::StopReason;
use serde::{Deserialize, Serialize};

/// A token produced during streaming generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    /// Vocabulary index of the token.
    pub id: u32,
    /// Decoded text fragment for this token.
    pub text: String,
}

/// Summary statistics after generation completes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationStats {
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Throughput in tokens/second.
    pub tokens_per_second: f64,
}

/// Events emitted during streaming generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    /// A single token was generated.
    Token(TokenEvent),
    /// Generation is complete.
    Done {
        /// Why generation stopped.
        reason: StopReason,
        /// Performance summary.
        stats: GenerationStats,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_event_done_carries_reason() {
        let ev = StreamEvent::Done {
            reason: StopReason::EosToken,
            stats: GenerationStats { tokens_generated: 10, tokens_per_second: 5.0 },
        };
        match ev {
            StreamEvent::Done { reason, stats } => {
                assert_eq!(reason, StopReason::EosToken);
                assert_eq!(stats.tokens_generated, 10);
            }
            StreamEvent::Token { .. } => panic!("expected Done event"),
        }
    }

    #[test]
    fn stream_event_token_carries_payload() {
        let ev = StreamEvent::Token(TokenEvent { id: 7, text: "hi".to_string() });
        match ev {
            StreamEvent::Token(t) => {
                assert_eq!(t.id, 7);
                assert_eq!(t.text, "hi");
            }
            StreamEvent::Done { .. } => panic!("expected Token event"),
        }
    }

    #[test]
    fn generation_stats_default_is_zeroed() {
        let stats = GenerationStats::default();
        assert_eq!(stats.tokens_generated, 0);
        assert_eq!(stats.tokens_per_second, 0.0);
    }

    #[test]
    fn token_event_serde_round_trip() {
        let original = TokenEvent { id: 42, text: "fox".to_string() };
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: TokenEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.id, original.id);
        assert_eq!(parsed.text, original.text);
    }

    #[test]
    fn stream_event_serde_round_trip_for_done_variant() {
        let original = StreamEvent::Done {
            reason: StopReason::MaxTokens,
            stats: GenerationStats { tokens_generated: 4, tokens_per_second: 12.5 },
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: StreamEvent = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            StreamEvent::Done { reason, stats } => {
                assert_eq!(reason, StopReason::MaxTokens);
                assert_eq!(stats.tokens_generated, 4);
                assert!((stats.tokens_per_second - 12.5).abs() < 1e-9);
            }
            StreamEvent::Token { .. } => panic!("expected Done after round-trip"),
        }
    }

    #[test]
    fn stream_event_serde_round_trip_for_token_variant() {
        let original = StreamEvent::Token(TokenEvent { id: 99, text: "tok".to_string() });
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: StreamEvent = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            StreamEvent::Token(t) => {
                assert_eq!(t.id, 99);
                assert_eq!(t.text, "tok");
            }
            StreamEvent::Done { .. } => panic!("expected Token after round-trip"),
        }
    }
}
