//! # Sampling Strategies
//!
//! Comprehensive sampling strategies for text generation including greedy,
//! top-k, top-p (nucleus), temperature, and repetition penalty sampling.

// Re-export pure logits transforms from the dedicated micro-crate.
pub use bitnet_logits::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax, softmax_in_place,
};

use anyhow::{Context, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use tracing::debug;

/// Configuration for sampling strategies
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for sampling (0.0 = deterministic, higher = more random)
    pub temperature: f32,
    /// Top-k sampling limit (0 = disabled)
    pub top_k: u32,
    /// Top-p (nucleus) sampling threshold (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty (1.0 = no penalty, higher = less repetition)
    pub repetition_penalty: f32,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 50, top_p: 0.9, repetition_penalty: 1.0, seed: None }
    }
}

/// Sampling strategy implementation
pub struct SamplingStrategy {
    config: SamplingConfig,
    rng: ChaCha8Rng,
    token_counts: HashMap<u32, usize>,
}

impl SamplingStrategy {
    /// Create a new sampling strategy
    pub fn new(config: SamplingConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_rng(&mut rand::rng())
        };

        Self { config, rng, token_counts: HashMap::new() }
    }

    /// Sample the next token from logits.
    ///
    /// Pipeline (all in-place via `bitnet-logits`):
    /// 1. Count-aware repetition penalty
    /// 2. Greedy short-circuit at temperature == 0.0
    /// 3. Temperature scaling → top-k → softmax → top-p → renormalize → sample
    ///
    /// Note: `apply_top_k` operates in the **logits domain** (writes `NEG_INFINITY`)
    /// and must run *before* `softmax_in_place`.  `apply_top_p` operates on
    /// probabilities and runs *after* softmax.
    pub fn sample(&mut self, logits: &[f32], context_tokens: &[u32]) -> Result<u32> {
        debug!("Sampling from {} logits", logits.len());

        if logits.is_empty() {
            return Err(anyhow::anyhow!("Empty logits slice"));
        }

        let mut buf = logits.to_vec();

        // Count-aware penalty: applies penalty^count per token (distinct from
        // the flat single-occurrence version in bitnet-logits).
        self.penalize_repeated_tokens(&mut buf, context_tokens);

        // Greedy path: temperature == 0.0 → greedy_sample (handles empty input
        // as Err and breaks ties by lowest token ID for llama.cpp compatibility).
        if self.config.temperature == 0.0 {
            let token = greedy_sample(&buf)?;
            *self.token_counts.entry(token).or_insert(0) += 1;
            return Ok(token);
        }

        // Stochastic path:
        //  1. temperature scaling (logit domain)
        //  2. top-k filtering (logit domain — NEG_INFINITY for filtered entries)
        //  3. softmax (NEG_INFINITY → 0.0 probability)
        //  4. top-p filtering (probability domain — zero for filtered entries)
        //  5. renormalize (top-p may leave sum < 1.0)
        apply_temperature(&mut buf, self.config.temperature);

        if self.config.top_k > 0 {
            apply_top_k(&mut buf, self.config.top_k as usize);
        }

        softmax_in_place(&mut buf);

        if self.config.top_p < 1.0 {
            apply_top_p(&mut buf, self.config.top_p);
        }

        // Re-normalize after top-p (top-p zeroes entries without renormalizing).
        let sum: f32 = buf.iter().sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for p in buf.iter_mut() {
                *p *= inv_sum;
            }
        }

        let token = self.sample_from_distribution(&buf)?;
        *self.token_counts.entry(token).or_insert(0) += 1;

        debug!("Sampled token: {}", token);
        Ok(token)
    }

    /// Count-aware repetition penalty applied in-place.
    ///
    /// Applies `penalty ^ occurrence_count` per token, so tokens seen twice are
    /// penalized more than tokens seen once.  This differs from
    /// [`bitnet_logits::apply_repetition_penalty`], which applies a flat single-
    /// occurrence penalty.
    fn penalize_repeated_tokens(&self, logits: &mut [f32], context_tokens: &[u32]) {
        #[allow(clippy::float_cmp)]
        if self.config.repetition_penalty == 1.0 {
            return;
        }

        let mut counts: HashMap<u32, i32> = HashMap::new();
        for &token in context_tokens {
            *counts.entry(token).or_insert(0) += 1;
        }

        for (&token, &count) in &counts {
            let idx = token as usize;
            if idx < logits.len() {
                let penalty = self.config.repetition_penalty.powi(count);
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }

    /// Sample from probability distribution
    fn sample_from_distribution(&mut self, probabilities: &[f32]) -> Result<u32> {
        // Handle edge cases
        if probabilities.is_empty() {
            return Err(anyhow::anyhow!("Empty probability distribution"));
        }

        // Clamp vocabulary size from logits tensor (prevents mismatched vocab issues)
        let vocab_size = probabilities.len();

        // Check if all probabilities are zero
        let sum: f32 = probabilities.iter().sum();
        if sum <= 0.0 {
            // Fallback to uniform distribution within valid vocab range
            let idx = self.rng.random_range(0..vocab_size);
            return Ok(idx as u32);
        }

        // Sample using cumulative distribution
        let random_value: f32 = self.rng.random();
        let mut cumulative = 0.0;

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                // Ensure token ID is within vocabulary bounds
                debug_assert!(
                    i < vocab_size,
                    "Sampled token {} exceeds vocab size {}",
                    i,
                    vocab_size
                );
                return Ok(i as u32);
            }
        }

        // Fallback to last valid token (clamped to vocab size)
        Ok((vocab_size - 1) as u32)
    }

    /// Reset token counts (useful for new sequences)
    pub fn reset(&mut self) {
        self.token_counts.clear();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SamplingConfig) {
        // If seed changed, recreate RNG
        if config.seed != self.config.seed {
            self.rng = if let Some(seed) = config.seed {
                ChaCha8Rng::seed_from_u64(seed)
            } else {
                ChaCha8Rng::from_rng(&mut rand::rng())
            };
        }

        self.config = config;
    }
}

/// Greedy sampling (always pick most likely token)
///
/// On ties (equal logits), chooses the lowest token ID for deterministic behavior
/// matching llama.cpp greedy decode.
pub fn greedy_sample(logits: &[f32]) -> Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(idx_a, a), (idx_b, b)| {
            // First compare logits
            match a.partial_cmp(b).unwrap() {
                std::cmp::Ordering::Equal => {
                    // On tie, prefer lower token ID
                    idx_b.cmp(idx_a) // Reverse: lower idx is "greater" priority
                }
                other => other,
            }
        })
        .map(|(idx, _)| idx as u32)
        .context("Empty logits for greedy sampling")
}

/// Multinomial sampling with temperature
pub fn temperature_sample(logits: &[f32], temperature: f32, _rng: &mut impl Rng) -> Result<u32> {
    if temperature <= 0.0 {
        return greedy_sample(logits);
    }

    let config =
        SamplingConfig { temperature, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: None };

    let mut strategy = SamplingStrategy::new(config);
    strategy.sample(logits, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.repetition_penalty, 1.0);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![0.1, 0.8, 0.1];
        let token = greedy_sample(&logits).unwrap();
        assert_eq!(token, 1); // Index of highest logit
    }

    #[test]
    fn test_temperature_sampling() {
        let logits = vec![0.1, 0.8, 0.1];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Temperature 0 should be greedy
        let token = temperature_sample(&logits, 0.0, &mut rng).unwrap();
        assert_eq!(token, 1);

        // High temperature should allow more randomness
        let token = temperature_sample(&logits, 2.0, &mut rng).unwrap();
        assert!(token < 3);
    }

    #[test]
    fn test_softmax() {
        // Delegate to the bitnet-logits free function (re-exported as `softmax_in_place`)
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        softmax_in_place(&mut logits);

        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher original logit → higher probability after softmax
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_top_k_filtering() {
        // apply_top_k operates in the logits domain: filtered entries become NEG_INFINITY.
        let mut logits = vec![1.0_f32, 4.0, 3.0, 2.0];
        apply_top_k(&mut logits, 2);

        // Top-2 are 4.0 (idx 1) and 3.0 (idx 2) — both must be finite.
        assert!(logits[1].is_finite(), "top logit should survive");
        assert!(logits[2].is_finite(), "second logit should survive");
        assert!(
            logits[0].is_infinite() && logits[0].is_sign_negative(),
            "non-top logit should be NEG_INFINITY"
        );
        assert!(
            logits[3].is_infinite() && logits[3].is_sign_negative(),
            "non-top logit should be NEG_INFINITY"
        );

        // After softmax, NEG_INFINITY entries become 0.0 probability.
        softmax_in_place(&mut logits);
        assert!(logits[1] > 0.0);
        assert!(logits[2] > 0.0);
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[3], 0.0);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "top-k + softmax should produce a valid distribution");
    }

    #[test]
    fn test_top_p_filtering() {
        // Delegate to the bitnet-logits free function (re-exported as `apply_top_p`)
        let mut probs = vec![0.5_f32, 0.3, 0.1, 0.1];
        apply_top_p(&mut probs, 0.8);

        // At least the dominant token should remain; fewer tokens than the original
        let non_zero = probs.iter().filter(|&&x| x > 0.0).count();
        assert!(non_zero >= 1);
        assert!(non_zero <= probs.len());
    }

    #[test]
    fn test_repetition_penalty() {
        // Test the count-aware private implementation via the private accessor.
        let config = SamplingConfig { repetition_penalty: 1.2, ..Default::default() };
        let strategy = SamplingStrategy::new(config);

        let mut logits = vec![1.0_f32, 1.0, 1.0];
        let context = vec![0_u32, 0, 1]; // Token 0 twice, token 1 once

        strategy.penalize_repeated_tokens(&mut logits, &context);

        // Token 0 penalized more (1.2^2) than token 1 (1.2^1); token 2 untouched
        assert!(logits[0] < logits[1]);
        assert!(logits[1] < logits[2]);
    }

    #[test]
    fn test_deterministic_sampling() {
        let config = SamplingConfig { seed: Some(42), ..Default::default() };

        let mut strategy1 = SamplingStrategy::new(config.clone());
        let mut strategy2 = SamplingStrategy::new(config);

        let logits = vec![0.1, 0.4, 0.3, 0.2];

        let token1 = strategy1.sample(&logits, &[]).unwrap();
        let token2 = strategy2.sample(&logits, &[]).unwrap();

        assert_eq!(token1, token2); // Should be deterministic with same seed
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // greedy_sample always returns a valid index into the logit slice.
    proptest! {
        #[test]
        fn greedy_sample_returns_valid_index(
            logits in prop::collection::vec(-1e6f32..=1e6f32, 1..=256),
        ) {
            let result = greedy_sample(&logits).unwrap();
            prop_assert!((result as usize) < logits.len());
        }
    }

    // greedy_sample picks the value at the argmax (returns an index with the highest logit).
    proptest! {
        #[test]
        fn greedy_sample_picks_argmax(
            logits in prop::collection::vec(-100f32..=100f32, 1..=64),
        ) {
            let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let result = greedy_sample(&logits).unwrap();
            prop_assert_eq!(
                logits[result as usize],
                max_val,
                "greedy returned idx {} with value {}, but max is {}",
                result,
                logits[result as usize],
                max_val
            );
        }
    }

    // softmax_in_place produces a valid probability distribution (non-negative, sums to 1).
    proptest! {
        #[test]
        fn softmax_is_valid_distribution(
            logits in prop::collection::vec(-50f32..=50f32, 1..=128),
        ) {
            let mut probs = logits;
            softmax_in_place(&mut probs);
            for &p in &probs {
                prop_assert!(p >= 0.0 && p.is_finite(), "probability {} is not valid", p);
            }
            let sum: f32 = probs.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "softmax sum={} expected ~1.0", sum);
        }
    }

    // apply_top_k leaves at most k finite entries; the rest become NEG_INFINITY.
    proptest! {
        #[test]
        fn top_k_leaves_at_most_k_finite(
            logits in prop::collection::vec(-10f32..=10f32, 2..=64),
            k in 1usize..=32,
        ) {
            let mut filtered = logits.clone();
            let effective_k = k.min(filtered.len());
            apply_top_k(&mut filtered, effective_k);
            let finite_count = filtered.iter().filter(|v| v.is_finite()).count();
            prop_assert!(
                finite_count <= effective_k,
                "finite_count={} > k={}",
                finite_count,
                effective_k
            );
        }
    }

    // SamplingStrategy with temperature=0 behaves like greedy.
    proptest! {
        #[test]
        fn strategy_temp_zero_is_greedy(
            logits in prop::collection::vec(-10f32..=10f32, 2..=32),
            seed in 0u64..=u64::MAX,
        ) {
            let config = SamplingConfig {
                temperature: 0.0,
                seed: Some(seed),
                ..Default::default()
            };
            let mut strategy = SamplingStrategy::new(config);
            let result = strategy.sample(&logits, &[]).unwrap();
            let greedy = greedy_sample(&logits).unwrap();
            prop_assert_eq!(result, greedy, "temperature=0 should be greedy");
        }
    }
}
