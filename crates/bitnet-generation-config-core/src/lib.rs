//! Reusable generation request configuration contract for `BitNet` inference.

use bitnet_generation_stop_core::StopCriteria;
use serde::{Deserialize, Serialize};

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
