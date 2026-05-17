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
    fn stream_event_done_carries_reason() -> Result<(), Box<dyn std::error::Error>> {
        let ev = StreamEvent::Done {
            reason: StopReason::EosToken,
            stats: GenerationStats { tokens_generated: 10, tokens_per_second: 5.0 },
        };
        let StreamEvent::Done { reason, stats } = ev else {
            return Err(std::io::Error::other("expected Done event").into());
        };
        assert_eq!(reason, StopReason::EosToken);
        assert_eq!(stats.tokens_generated, 10);
        Ok(())
    }

    #[test]
    fn stream_event_token_carries_payload() -> Result<(), Box<dyn std::error::Error>> {
        let ev = StreamEvent::Token(TokenEvent { id: 7, text: "hi".to_string() });
        let StreamEvent::Token(t) = ev else {
            return Err(std::io::Error::other("expected Token event").into());
        };
        assert_eq!(t.id, 7);
        assert_eq!(t.text, "hi");
        Ok(())
    }

    #[test]
    fn generation_stats_default_is_zeroed() {
        let stats = GenerationStats::default();
        assert_eq!(stats.tokens_generated, 0);
        assert_eq!(stats.tokens_per_second, 0.0);
    }

    #[test]
    fn token_event_serde_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let original = TokenEvent { id: 42, text: "fox".to_string() };
        let json = serde_json::to_string(&original)?;
        let parsed: TokenEvent = serde_json::from_str(&json)?;
        assert_eq!(parsed.id, original.id);
        assert_eq!(parsed.text, original.text);
        Ok(())
    }

    #[test]
    fn stream_event_serde_round_trip_for_done_variant() -> Result<(), Box<dyn std::error::Error>> {
        let original = StreamEvent::Done {
            reason: StopReason::MaxTokens,
            stats: GenerationStats { tokens_generated: 4, tokens_per_second: 12.5 },
        };
        let json = serde_json::to_string(&original)?;
        let parsed: StreamEvent = serde_json::from_str(&json)?;
        let StreamEvent::Done { reason, stats } = parsed else {
            return Err(std::io::Error::other("expected Done after round-trip").into());
        };
        assert_eq!(reason, StopReason::MaxTokens);
        assert_eq!(stats.tokens_generated, 4);
        assert!((stats.tokens_per_second - 12.5).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn stream_event_serde_round_trip_for_token_variant() -> Result<(), Box<dyn std::error::Error>> {
        let original = StreamEvent::Token(TokenEvent { id: 99, text: "tok".to_string() });
        let json = serde_json::to_string(&original)?;
        let parsed: StreamEvent = serde_json::from_str(&json)?;
        let StreamEvent::Token(t) = parsed else {
            return Err(std::io::Error::other("expected Token after round-trip").into());
        };
        assert_eq!(t.id, 99);
        assert_eq!(t.text, "tok");
        Ok(())
    }
}
