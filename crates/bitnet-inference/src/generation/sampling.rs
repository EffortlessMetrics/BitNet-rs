//! Sampling Strategies for Text Generation
//!
//! Provides various sampling strategies including temperature scaling,
//! top-k sampling, nucleus (top-p) sampling, and repetition penalty.

use anyhow::Result;
use bitnet_common::{BitNetTensor, Tensor};

use rand::{Rng, RngCore};
use std::collections::HashMap;

/// Configuration for sampling strategies
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub do_sample: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            do_sample: true,
        }
    }
}

/// Sampling strategy implementation
#[derive(Debug)]
pub struct SamplingStrategy {
    config: SamplingConfig,
    repetition_counts: HashMap<usize, usize>,
    current_repetition_penalty: f32,
}

impl SamplingStrategy {
    /// Create new sampling strategy
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            current_repetition_penalty: config.repetition_penalty,
            config,
            repetition_counts: HashMap::new(),
        }
    }

    /// Sample next token from logits distribution
    pub async fn sample<R: RngCore>(
        &mut self,
        logits: &BitNetTensor,
        rng: &mut R,
    ) -> Result<(usize, f32)> {
        if !self.config.do_sample {
            return self.greedy_sample(logits).await;
        }

        let logits_candle = logits.to_candle()?;

        // Get the last token's logits (for autoregressive generation)
        let last_logits = if logits_candle.dims().len() == 3 {
            let (batch, seq_len, vocab_size) = logits_candle.dims3()?;
            logits_candle.narrow(1, seq_len - 1, 1)?.reshape(&[batch, vocab_size])?
        } else if logits_candle.dims().len() == 2 {
            logits_candle.clone()
        } else {
            return Err(anyhow::anyhow!("Unexpected logits shape: {:?}", logits_candle.shape()));
        };

        // Optimize: Convert to Vec<f32> once and perform all operations in-place
        // This avoids multiple Tensor allocations and device synchronizations
        let mut logits_vec = last_logits.flatten_all()?.to_vec1::<f32>()?;

        // 1. Temperature
        if self.config.temperature != 1.0 {
            let inv_temp = 1.0 / self.config.temperature;
            for x in &mut logits_vec {
                *x *= inv_temp;
            }
        }

        // 2. Repetition Penalty
        self.apply_repetition_penalty_inplace(&mut logits_vec);

        // 3. Top-K
        if let Some(top_k) = self.config.top_k {
            self.apply_top_k_inplace(&mut logits_vec, top_k);
        }

        // 4. Softmax -> Probs
        self.softmax_inplace(&mut logits_vec);
        let mut probs = logits_vec; // Rename for clarity (now contains probabilities)

        // 5. Top-P (Nucleus)
        if let Some(top_p) = self.config.top_p {
            self.apply_top_p_inplace(&mut probs, top_p);
        }

        // 6. Sample
        self.multinomial_sample(&probs, rng).await
    }

    /// Greedy sampling (argmax)
    async fn greedy_sample(&self, logits: &BitNetTensor) -> Result<(usize, f32)> {
        let logits_candle = logits.to_candle()?;

        // Get the last token's logits
        let last_logits = if logits_candle.dims().len() == 3 {
            let (batch, seq_len, vocab_size) = logits_candle.dims3()?;
            logits_candle.narrow(1, seq_len - 1, 1)?.reshape(&[batch, vocab_size])?
        } else {
            logits_candle.clone()
        };

        // Find argmax
        let probabilities = candle_nn::ops::softmax(&last_logits, candle_core::D::Minus1)?;
        let probs_vec = probabilities.flatten_all()?.to_vec1::<f32>()?;

        let (max_idx, max_prob) = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| anyhow::anyhow!("Empty probability distribution"))?;

        Ok((max_idx, *max_prob))
    }

    /// Apply repetition penalty to logits in-place
    fn apply_repetition_penalty_inplace(&self, logits: &mut [f32]) {
        if self.current_repetition_penalty == 1.0 || self.repetition_counts.is_empty() {
            return;
        }

        for (&token_id, &count) in &self.repetition_counts {
            if token_id < logits.len() && count > 0 {
                let penalty_factor = self.current_repetition_penalty.powi(count as i32);
                if logits[token_id] > 0.0 {
                    logits[token_id] /= penalty_factor;
                } else {
                    logits[token_id] *= penalty_factor;
                }
            }
        }
    }

    /// Apply top-k filtering in-place using efficient selection
    fn apply_top_k_inplace(&self, logits: &mut [f32], k: usize) {
        if k >= logits.len() {
            return;
        }

        // Use indices to track which logits are top-k without fully sorting logits
        let mut indices: Vec<usize> = (0..logits.len()).collect();

        // select_nth_unstable_by is O(N)
        indices.select_nth_unstable_by(k, |&i, &j| {
            logits[j].partial_cmp(&logits[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Indices from k onwards are not in top-k
        for &idx in indices.iter().skip(k) {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    /// Apply nucleus (top-p) sampling in-place
    fn apply_top_p_inplace(&self, probs: &mut [f32], p: f32) {
        if p >= 1.0 {
            return;
        }

        // Get indices sorted by probability (descending)
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&i, &j| {
            probs[j].partial_cmp(&probs[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find cutoff point where cumulative probability exceeds p
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = probs.len();

        for (i, &idx) in indices.iter().enumerate() {
            cumulative_prob += probs[idx];
            if cumulative_prob >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Mask tokens outside nucleus
        for &idx in indices.iter().skip(cutoff_idx) {
            probs[idx] = 0.0;
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for x in probs.iter_mut() {
                *x *= inv_sum;
            }
        }
    }

    /// Compute softmax in-place
    fn softmax_inplace(&self, logits: &mut [f32]) {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;

        for x in logits.iter_mut() {
            *x = (*x - max_logit).exp();
            sum += *x;
        }

        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for x in logits.iter_mut() {
                *x *= inv_sum;
            }
        } else {
             // Fallback for numerical instability (all -inf)
             let n = logits.len() as f32;
             let val = 1.0 / n;
             for x in logits.iter_mut() {
                 *x = val;
             }
        }
    }

    /// Sample from multinomial distribution
    async fn multinomial_sample<R: RngCore>(
        &self,
        probs: &[f32],
        rng: &mut R,
    ) -> Result<(usize, f32)> {
        // Generate random number
        let random_val: f32 = rng.random();

        // Find token by cumulative probability
        let mut cumulative_prob = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative_prob += prob;
            if random_val <= cumulative_prob {
                return Ok((i, prob));
            }
        }

        // Fallback: return last token
        let last_idx = probs.len().saturating_sub(1);
        Ok((last_idx, probs[last_idx]))
    }

    /// Update repetition tracking
    pub fn track_token(&mut self, token_id: usize) {
        *self.repetition_counts.entry(token_id).or_insert(0) += 1;

        // Clean up old entries to prevent unbounded growth
        if self.repetition_counts.len() > 1000 {
            self.repetition_counts.clear();
        }
    }

    /// Increase repetition penalty dynamically
    pub fn increase_repetition_penalty(&mut self) {
        self.current_repetition_penalty = (self.current_repetition_penalty * 1.1).min(2.0);
    }

    /// Reset repetition penalty
    pub fn reset_repetition_penalty(&mut self) {
        self.current_repetition_penalty = self.config.repetition_penalty;
        self.repetition_counts.clear();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SamplingConfig) {
        self.current_repetition_penalty = config.repetition_penalty;
        self.config = config;
    }

    /// Get current effective temperature
    pub fn effective_temperature(&self) -> f32 {
        self.config.temperature
    }

    /// Get current effective repetition penalty
    pub fn effective_repetition_penalty(&self) -> f32 {
        self.current_repetition_penalty
    }
}

/// Specialized sampling strategies
impl SamplingStrategy {
    /// Create strategy for deterministic generation
    pub fn deterministic() -> Self {
        Self::new(SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            do_sample: false,
        })
    }

    /// Create strategy for creative generation
    pub fn creative() -> Self {
        Self::new(SamplingConfig {
            temperature: 1.2,
            top_k: Some(100),
            top_p: Some(0.9),
            repetition_penalty: 1.2,
            do_sample: true,
        })
    }

    /// Create strategy for balanced generation
    pub fn balanced() -> Self {
        Self::new(SamplingConfig {
            temperature: 0.8,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.1,
            do_sample: true,
        })
    }

    /// Create strategy for conservative generation
    pub fn conservative() -> Self {
        Self::new(SamplingConfig {
            temperature: 0.3,
            top_k: Some(20),
            top_p: Some(0.8),
            repetition_penalty: 1.05,
            do_sample: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::{BitNetTensor, Device};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[tokio::test]
    async fn test_sampling_top_k() {
        let config = SamplingConfig {
            top_k: Some(2),
            top_p: None,
            temperature: 1.0,
            repetition_penalty: 1.0,
            do_sample: true,
        };
        let mut strategy = SamplingStrategy::new(config);

        // Logits: [0.1, 0.4, 0.3, 0.2]
        // Sorted: 0.4 (idx 1), 0.3 (idx 2), 0.2 (idx 3), 0.1 (idx 0)
        // Top-K=2 should keep indices 1 and 2.
        // Indices 0 and 3 should be -inf.

        let logits_data = vec![0.1f32, 0.4, 0.3, 0.2];
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&logits_data, &[1, 4], &device).unwrap();

        let mut rng = StdRng::seed_from_u64(42);

        // Run multiple times to ensure we never pick excluded tokens
        for _ in 0..50 {
            let (token, _) = strategy.sample(&tensor, &mut rng).await.unwrap();
            assert!(token == 1 || token == 2, "Token {} sampled, expected 1 or 2", token);
        }
    }

    #[tokio::test]
    async fn test_sampling_top_p() {
        let config = SamplingConfig {
            top_k: None,
            top_p: Some(0.8), // Keep top 80% mass
            temperature: 1.0,
            repetition_penalty: 1.0,
            do_sample: true,
        };
        let mut strategy = SamplingStrategy::new(config);

        // Logits: [2.0, 1.0, 0.1, 0.0]
        // Exp: [7.389, 2.718, 1.105, 1.0] -> Sum = 12.212
        // Probs: [0.605, 0.222, 0.090, 0.081]
        // Cumsum: 0.605, 0.827 (> 0.8), ...
        // So it should include index 0 and 1.

        let logits_data = vec![2.0f32, 1.0, 0.1, 0.0];
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&logits_data, &[1, 4], &device).unwrap();

        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..50 {
            let (token, _) = strategy.sample(&tensor, &mut rng).await.unwrap();
            assert!(token == 0 || token == 1, "Token {} sampled, expected 0 or 1", token);
        }
    }

    #[tokio::test]
    async fn test_repetition_penalty() {
        let config = SamplingConfig {
            top_k: None,
            top_p: None,
            temperature: 1.0,
            repetition_penalty: 2.0, // High penalty
            do_sample: true,
        };
        let mut strategy = SamplingStrategy::new(config);

        // Logits: [1.0, 1.0]
        // Track token 0.
        strategy.track_token(0);

        // Now token 0 should be penalized.
        // If logits > 0, divide by penalty. 1.0 / 2.0 = 0.5.
        // Token 1 stays 1.0.
        // Prob(1) > Prob(0).

        let logits_data = vec![1.0f32, 1.0];
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&logits_data, &[1, 2], &device).unwrap();

        let mut rng = StdRng::seed_from_u64(42);

        // Sampling should prefer token 1 significantly.
        let mut count_0 = 0;
        let mut count_1 = 0;
        for _ in 0..100 {
            let (token, _) = strategy.sample(&tensor, &mut rng).await.unwrap();
            if token == 0 { count_0 += 1; }
            else { count_1 += 1; }
        }

        assert!(count_1 > count_0, "Token 1 should be preferred over penalized token 0. Got 0:{}, 1:{}", count_0, count_1);
    }
}
