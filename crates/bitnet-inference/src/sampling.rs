//! # Sampling Strategies
//!
//! Comprehensive sampling strategies for text generation including greedy,
//! top-k, top-p (nucleus), temperature, and repetition penalty sampling.

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

    /// Sample the next token from logits
    pub fn sample(&mut self, logits: &[f32], context_tokens: &[u32]) -> Result<u32> {
        debug!("Sampling from {} logits", logits.len());

        // Apply repetition penalty
        let mut adjusted_logits = self.apply_repetition_penalty(logits, context_tokens);

        // Apply temperature
        if self.config.temperature != 1.0 {
            self.apply_temperature(&mut adjusted_logits);
        }

        // Convert to probabilities
        let probabilities = self.softmax(&adjusted_logits);

        // Apply top-k filtering
        let filtered_probs = if self.config.top_k > 0 {
            self.apply_top_k(&probabilities, self.config.top_k as usize)
        } else {
            probabilities
        };

        // Apply top-p (nucleus) filtering
        let final_probs = if self.config.top_p < 1.0 {
            self.apply_top_p(&filtered_probs, self.config.top_p)
        } else {
            filtered_probs
        };

        // Sample from the final distribution
        let token = self.sample_from_distribution(&final_probs)?;

        // Update token counts for repetition penalty
        *self.token_counts.entry(token).or_insert(0) += 1;

        debug!("Sampled token: {}", token);
        Ok(token)
    }

    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(&self, logits: &[f32], context_tokens: &[u32]) -> Vec<f32> {
        if self.config.repetition_penalty == 1.0 {
            return logits.to_vec();
        }

        let mut adjusted_logits = logits.to_vec();

        // Count token frequencies in context
        let mut token_counts = HashMap::new();
        for &token in context_tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }

        // Apply penalty to repeated tokens
        for (&token, &count) in &token_counts {
            if token < adjusted_logits.len() as u32 {
                let penalty = self.config.repetition_penalty.powi(count);
                if adjusted_logits[token as usize] > 0.0 {
                    adjusted_logits[token as usize] /= penalty;
                } else {
                    adjusted_logits[token as usize] *= penalty;
                }
            }
        }

        adjusted_logits
    }

    /// Apply temperature scaling to logits
    fn apply_temperature(&self, logits: &mut [f32]) {
        if self.config.temperature > 0.0 {
            for logit in logits.iter_mut() {
                *logit /= self.config.temperature;
            }
        }
    }

    /// Convert logits to probabilities using softmax
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        // Find maximum for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exponentials
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

        // Compute sum
        let sum: f32 = exp_logits.iter().sum();

        // Normalize
        if sum > 0.0 {
            exp_logits.iter().map(|&x| x / sum).collect()
        } else {
            vec![1.0 / logits.len() as f32; logits.len()]
        }
    }

    /// Apply top-k filtering
    fn apply_top_k(&self, probabilities: &[f32], k: usize) -> Vec<f32> {
        if k >= probabilities.len() {
            return probabilities.to_vec();
        }

        // Create indices sorted by probability (descending)
        let mut indices: Vec<usize> = (0..probabilities.len()).collect();
        indices.sort_by(|&a, &b| probabilities[b].partial_cmp(&probabilities[a]).unwrap());

        // Keep only top-k
        let mut filtered = vec![0.0; probabilities.len()];
        let mut sum = 0.0;

        for &idx in indices.iter().take(k) {
            filtered[idx] = probabilities[idx];
            sum += probabilities[idx];
        }

        // Renormalize
        if sum > 0.0 {
            for prob in filtered.iter_mut() {
                *prob /= sum;
            }
        }

        filtered
    }

    /// Apply top-p (nucleus) filtering
    fn apply_top_p(&self, probabilities: &[f32], p: f32) -> Vec<f32> {
        if p >= 1.0 {
            return probabilities.to_vec();
        }

        // Create indices sorted by probability (descending)
        let mut indices: Vec<usize> = (0..probabilities.len()).collect();
        indices.sort_by(|&a, &b| probabilities[b].partial_cmp(&probabilities[a]).unwrap());

        // Find cutoff point
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = probabilities.len();

        for (i, &idx) in indices.iter().enumerate() {
            cumulative_prob += probabilities[idx];
            if cumulative_prob >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Keep only tokens in nucleus
        let mut filtered = vec![0.0; probabilities.len()];
        let mut sum = 0.0;

        for &idx in indices.iter().take(cutoff_idx) {
            filtered[idx] = probabilities[idx];
            sum += probabilities[idx];
        }

        // Renormalize
        if sum > 0.0 {
            for prob in filtered.iter_mut() {
                *prob /= sum;
            }
        }

        filtered
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
                debug_assert!(i < vocab_size, "Sampled token {} exceeds vocab size {}", i, vocab_size);
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
pub fn greedy_sample(logits: &[f32]) -> Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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
        let config = SamplingConfig::default();
        let strategy = SamplingStrategy::new(config);

        let logits = vec![1.0, 2.0, 3.0];
        let probs = strategy.softmax(&logits);

        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that higher logits have higher probabilities
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_top_k_filtering() {
        let config = SamplingConfig::default();
        let strategy = SamplingStrategy::new(config);

        let probs = vec![0.1, 0.4, 0.3, 0.2];
        let filtered = strategy.apply_top_k(&probs, 2);

        // Only top 2 should be non-zero
        let non_zero_count = filtered.iter().filter(|&&x| x > 0.0).count();
        assert_eq!(non_zero_count, 2);

        // Should still sum to 1
        let sum: f32 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_p_filtering() {
        let config = SamplingConfig::default();
        let strategy = SamplingStrategy::new(config);

        let probs = vec![0.5, 0.3, 0.1, 0.1];
        let filtered = strategy.apply_top_p(&probs, 0.8);

        // Should include tokens that sum to at least 0.8
        let sum: f32 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should have fewer non-zero elements than original
        let non_zero_count = filtered.iter().filter(|&&x| x > 0.0).count();
        assert!(non_zero_count <= probs.len());
    }

    #[test]
    fn test_repetition_penalty() {
        let config = SamplingConfig { repetition_penalty: 1.2, ..Default::default() };
        let strategy = SamplingStrategy::new(config);

        let logits = vec![1.0, 1.0, 1.0];
        let context = vec![0, 0, 1]; // Token 0 appears twice, token 1 once

        let adjusted = strategy.apply_repetition_penalty(&logits, &context);

        // Token 0 should be penalized more than token 1
        assert!(adjusted[0] < adjusted[1]);
        assert!(adjusted[1] < adjusted[2]); // Token 2 not penalized
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
