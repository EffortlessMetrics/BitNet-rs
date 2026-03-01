//! Decode-loop generation contracts for `BitNet` inference.
//!
//! This crate provides orchestration-facing generation types:
//! - generation config
//! - stream events (via `bitnet-generation-events-core`)
//! - generation stats (via `bitnet-generation-events-core`)
//!
//! Decode-loop stop criteria and stop-check logic live in
//! `bitnet-generation-stop-core` and are re-exported here.

pub use bitnet_generation_events_core::{GenerationStats, StreamEvent, TokenEvent};
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_config_defaults_are_sensible() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_new_tokens, 128);
        assert!(cfg.seed.is_none());
    }
}
