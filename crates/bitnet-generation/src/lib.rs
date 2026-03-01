//! Decode-loop generation contracts for `BitNet` inference.
//!
//! This crate provides orchestration-facing generation types:
//! - generation config
//! - stream events
//! - generation stats
//!
//! Decode-loop stop criteria and stop-check logic live in
//! `bitnet-generation-stop-core` and are re-exported here.

pub use bitnet_generation_stop_core::{StopCriteria, StopReason, check_stop};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Generation config
// ---------------------------------------------------------------------------

/// Configuration for a single generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Random seed for reproducible generation (None = random).
    pub seed: Option<u64>,
    /// Stopping criteria.
    pub stop_criteria: StopCriteria,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_new_tokens: 128, seed: None, stop_criteria: StopCriteria::default() }
    }
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

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
    fn generation_config_defaults_are_sensible() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_new_tokens, 128);
        assert!(cfg.seed.is_none());
    }

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
