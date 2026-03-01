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
}
