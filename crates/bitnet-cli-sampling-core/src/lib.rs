//! Sampling adapter for CLI generation.
//!
//! The CLI now delegates sampling behavior to the dedicated `bitnet-sampling`
//! microcrate so decode behavior is consistent across CLI and inference.

use anyhow::Result;
use bitnet_sampling::{SamplingConfig, SamplingStrategy};

/// Sampling strategy wrapper used by the CLI decode loop.
pub struct Sampler {
    inner: SamplingStrategy,
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

        Self { inner: SamplingStrategy::new(config) }
    }

    /// Sample next token from logits.
    pub fn sample(&mut self, logits: &[f32], generated_tokens: &[u32]) -> Result<u32> {
        self.inner.sample(logits, generated_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_picks_argmax() {
        let mut sampler = Sampler::new(0.0, 0, 1.0, 1.0, Some(42));
        let logits = vec![1.0, 3.0, 2.0];
        let token = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let mut sampler1 = Sampler::new(0.7, 50, 0.9, 1.0, Some(42));
        let mut sampler2 = Sampler::new(0.7, 50, 0.9, 1.0, Some(42));
        let logits = vec![0.1, 0.4, 0.3, 0.2];

        let t1 = sampler1.sample(&logits, &[]).unwrap();
        let t2 = sampler2.sample(&logits, &[]).unwrap();

        assert_eq!(t1, t2);
    }
}
