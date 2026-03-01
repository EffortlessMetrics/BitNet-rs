//! Sampling adapter utilities for text generation.
//!
//! CLI-facing wrapper around the shared `bitnet-sampling` microcrate.

use anyhow::Result;
use bitnet_sampling::{SamplingConfig, SamplingStrategy, softmax_in_place};

/// Sampling strategy for text generation.
///
/// This preserves the existing CLI API while delegating implementation to the
/// reusable `bitnet-sampling` microcrate.
pub struct Sampler {
    strategy: SamplingStrategy,
}

impl Sampler {
    /// Create a new sampler with given parameters.
    pub fn new(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: Option<u64>,
    ) -> Self {
        let config =
            SamplingConfig { temperature, top_k: top_k as u32, top_p, repetition_penalty, seed };

        Self { strategy: SamplingStrategy::new(config) }
    }

    /// Sample next token from logits.
    pub fn sample(&mut self, logits: &[f32], generated_tokens: &[u32]) -> u32 {
        self.strategy.sample(logits, generated_tokens).unwrap_or_else(|_| argmax(logits))
    }
}

/// Argmax helper with deterministic tie-breaking (lowest index) and NaN-safe handling.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;

    for (i, &val) in logits.iter().enumerate() {
        if val.is_nan() {
            continue;
        }

        if val > best_val || (val == best_val && i < best_idx) {
            best_val = val;
            best_idx = i;
        }
    }

    best_idx as u32
}

/// Softmax function.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut out = logits.to_vec();
    softmax_in_place(&mut out);
    out
}

/// Argmax function with deterministic tie-breaking (choose lowest index on tie).
pub fn greedy_tie_break_lowest_id(logits: &[f32]) -> u32 {
    argmax(logits)
}

/// Validate user-provided sampling settings before generation.
pub fn validate_sampling_inputs(temperature: f32, top_p: f32) -> Result<()> {
    if temperature.is_sign_negative() {
        anyhow::bail!("temperature must be >= 0.0, got {temperature}");
    }

    if !(0.0..=1.0).contains(&top_p) {
        anyhow::bail!("top-p must be in [0.0, 1.0], got {top_p}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_argmax_tie_break() {
        let logits = vec![1.0, 2.0, 2.0, 1.5];
        assert_eq!(argmax(&logits), 1);
        assert_eq!(greedy_tie_break_lowest_id(&logits), 1);
    }

    #[test]
    fn test_greedy_sampling() {
        let mut sampler = Sampler::new(0.0, 0, 1.0, 1.0, Some(42));
        let logits = vec![1.0, 3.0, 2.0];
        assert_eq!(sampler.sample(&logits, &[]), 1);
    }

    #[test]
    fn test_sampling_validation() {
        assert!(validate_sampling_inputs(0.0, 1.0).is_ok());
        assert!(validate_sampling_inputs(0.7, 0.95).is_ok());

        let err = validate_sampling_inputs(-0.01, 1.0).unwrap_err();
        assert!(err.to_string().contains("temperature"));

        let err = validate_sampling_inputs(0.7, 1.5).unwrap_err();
        assert!(err.to_string().contains("top-p"));
    }

    #[test]
    fn test_nan_logits_result_is_vocab_bounded() {
        let mut sampler = Sampler::new(0.8, 10, 0.9, 1.0, Some(42));
        let logits = vec![f32::NAN, f32::NAN, 0.5];
        let token = sampler.sample(&logits, &[]);
        assert!(token < logits.len() as u32);
    }
}
