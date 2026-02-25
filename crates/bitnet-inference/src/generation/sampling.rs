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

        // Conversion to Vec<f32> happens ONCE here.
        // All subsequent operations are in-place on this vector.
        let mut logits_vec = last_logits.flatten_all()?.to_vec1::<f32>()?;

        // Apply temperature scaling
        if self.config.temperature != 1.0 {
            let temp = self.config.temperature;
            for x in logits_vec.iter_mut() {
                *x /= temp;
            }
        }

        // Apply repetition penalty
        self.apply_repetition_penalty(&mut logits_vec);

        // Apply top-k filtering
        if let Some(top_k) = self.config.top_k {
            self.apply_top_k(&mut logits_vec, top_k);
        }

        // Compute Softmax in-place to get probabilities
        self.softmax_inplace(&mut logits_vec);
        // logits_vec now contains PROBABILITIES.
        let mut probs_vec = logits_vec;

        // Apply nucleus (top-p) sampling on probabilities
        if let Some(top_p) = self.config.top_p {
            self.apply_top_p(&mut probs_vec, top_p);
        }

        // Sample from distribution
        self.multinomial_sample(&probs_vec, rng).await
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

    /// Compute softmax in-place
    fn softmax_inplace(&self, x: &mut [f32]) {
        if x.is_empty() {
            return;
        }
        let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for val in x.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        if sum > 0.0 {
            for val in x.iter_mut() {
                *val /= sum;
            }
        }
    }

    /// Apply repetition penalty to logits (in-place)
    fn apply_repetition_penalty(&self, logits: &mut [f32]) {
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

    /// Apply top-k filtering (in-place)
    fn apply_top_k(&self, logits: &mut [f32], k: usize) {
        let vocab_size = logits.len();

        // k=0 means "all tokens" — skip filtering entirely
        if k == 0 || k >= vocab_size {
            return;
        }

        // O(N) partition: rearrange so that indices[0..k] are the top-k highest logits.
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.select_nth_unstable_by(k - 1, |&i, &j| {
            logits[j].partial_cmp(&logits[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep exactly the k tokens in the top partition; mask all others.
        let mut keep = vec![false; vocab_size];
        for &idx in &indices[..k] {
            keep[idx] = true;
        }
        for (i, val) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *val = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply nucleus (top-p) sampling on probabilities (in-place)
    fn apply_top_p(&self, probs: &mut [f32], p: f32) {
        // Create candidates from non-zero probabilities
        let mut candidates: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();

        // Sort candidates by probability descending
        candidates.sort_unstable_by(|&i, &j| {
            probs[j].partial_cmp(&probs[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find cutoff point where cumulative probability exceeds p
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = candidates.len();

        for (i, &idx) in candidates.iter().enumerate() {
            cumulative_prob += probs[idx];
            if cumulative_prob >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out probabilities for pruned candidates (those after cutoff)
        for &idx in &candidates[cutoff_idx..] {
            probs[idx] = 0.0;
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for val in probs.iter_mut() {
                *val /= sum;
            }
        }
    }

    /// Sample from multinomial distribution
    async fn multinomial_sample<R: RngCore>(
        &self,
        prob_vec: &[f32],
        rng: &mut R,
    ) -> Result<(usize, f32)> {
        // Generate random number
        let random_val: f32 = rng.random();

        // Find token by cumulative probability
        let mut cumulative_prob = 0.0;
        for (i, &prob) in prob_vec.iter().enumerate() {
            cumulative_prob += prob;
            if random_val <= cumulative_prob {
                return Ok((i, prob));
            }
        }

        // Fallback: return last token
        let last_idx = prob_vec.len() - 1;
        Ok((last_idx, prob_vec[last_idx]))
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
    use candle_core::Device;
    use candle_core::Tensor as CandleTensor;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn create_logits(values: &[f32]) -> BitNetTensor {
        let tensor = CandleTensor::from_slice(values, (1, values.len()), &Device::Cpu).unwrap();
        BitNetTensor::new(tensor)
    }

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

        let logits = create_logits(&[10.0, 5.0, 20.0, 3.0]);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..20 {
            let (idx, _) = strategy.sample(&logits, &mut rng).await.unwrap();
            assert!(idx == 0 || idx == 2, "Sampled index {} should be 0 or 2", idx);
        }
    }

    #[tokio::test]
    async fn test_sampling_top_p() {
        let config = SamplingConfig {
            top_k: None,
            top_p: Some(0.8),
            temperature: 1.0,
            repetition_penalty: 1.0,
            do_sample: true,
        };
        let mut strategy = SamplingStrategy::new(config);

        let logits = create_logits(&[10.0, 0.0, 0.0]);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..10 {
            let (idx, _) = strategy.sample(&logits, &mut rng).await.unwrap();
            assert_eq!(idx, 0);
        }
    }

    #[tokio::test]
    async fn test_repetition_penalty() {
        let config = SamplingConfig {
            top_k: None,
            top_p: None,
            temperature: 1.0,
            repetition_penalty: 2.0,
            do_sample: false,
        };
        let mut strategy = SamplingStrategy::new(config);

        let logits = create_logits(&[10.0, 10.0]);
        strategy.track_token(0);

        let (idx, _) = strategy.sample(&logits, &mut StdRng::seed_from_u64(42)).await.unwrap();
        assert_eq!(idx, 1);
    }
}
