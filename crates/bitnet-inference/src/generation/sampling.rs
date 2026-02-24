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
    // Reusable buffers to minimize allocations
    scratch_indices: Vec<usize>,
    scratch_probs: Vec<f32>,
}

impl SamplingStrategy {
    /// Create new sampling strategy
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            current_repetition_penalty: config.repetition_penalty,
            config,
            repetition_counts: HashMap::new(),
            scratch_indices: Vec::new(),
            scratch_probs: Vec::new(),
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

        // Extract logits to Vec<f32> once.
        // This is the main allocation we can't easily avoid without lower-level API access.
        let mut logits_vec = last_logits.flatten_all()?.to_vec1::<f32>()?;

        // Apply temperature scaling in-place
        if self.config.temperature != 1.0 {
            let inv_temp = 1.0 / self.config.temperature;
            for x in logits_vec.iter_mut() {
                *x *= inv_temp;
            }
        }

        // Apply repetition penalty in-place
        self.apply_repetition_penalty_inplace(&mut logits_vec);

        // Apply top-k filtering in-place
        if let Some(top_k) = self.config.top_k {
            self.apply_top_k_inplace(&mut logits_vec, top_k);
        }

        // Apply nucleus (top-p) sampling in-place
        if let Some(top_p) = self.config.top_p {
            self.apply_top_p_inplace(&mut logits_vec, top_p);
        }

        // Compute softmax in-place to get probabilities for sampling
        Self::softmax_inplace(&mut logits_vec);

        // Sample from distribution
        self.multinomial_sample_vec(&logits_vec, rng).await
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

    fn apply_top_k_inplace(&mut self, logits: &mut [f32], k: usize) {
        let vocab_size = logits.len();
        if k == 0 || k >= vocab_size {
            return;
        }

        // Initialize scratch indices
        if self.scratch_indices.len() != vocab_size {
            self.scratch_indices = (0..vocab_size).collect();
        } else {
            // Need to reset because select_nth_unstable permutes
            for i in 0..vocab_size {
                self.scratch_indices[i] = i;
            }
        }

        // O(N) partition
        self.scratch_indices.select_nth_unstable_by(k - 1, |&i, &j| {
            logits[j].partial_cmp(&logits[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Mask indices that are not in the top-k (i.e., those after k-1)
        for &idx in &self.scratch_indices[k..] {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn apply_top_p_inplace(&mut self, logits: &mut [f32], p: f32) {
        // We need probabilities to sort candidates.
        // We reuse scratch_probs to store probabilities of VALID tokens.
        // We avoid full softmax if we can, but logic implies we need full softmax
        // over valid logits to get correct probabilities.

        // Reuse scratch_probs
        self.scratch_probs.resize(logits.len(), 0.0);
        self.scratch_probs.copy_from_slice(logits);

        // Compute softmax on scratch_probs
        Self::softmax_inplace(&mut self.scratch_probs);

        // Reuse scratch_indices for candidates
        // Only consider candidates that are not already masked and have >0 prob
        self.scratch_indices.clear();
        for (i, &prob) in self.scratch_probs.iter().enumerate() {
            if logits[i] > f32::NEG_INFINITY && prob > 0.0 {
                self.scratch_indices.push(i);
            }
        }

        // Sort candidates by probability descending
        let probs = &self.scratch_probs;
        self.scratch_indices.sort_unstable_by(|&i, &j| {
            probs[j].partial_cmp(&probs[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find cutoff
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = self.scratch_indices.len();

        for (i, &idx) in self.scratch_indices.iter().enumerate() {
            cumulative_prob += probs[idx];
            if cumulative_prob >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Mask rejected candidates
        for &idx in self.scratch_indices.iter().skip(cutoff_idx) {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn softmax_inplace(x: &mut [f32]) {
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
        } else {
            // Handle all -inf case (uniform)
            let uniform = 1.0 / x.len() as f32;
            for val in x.iter_mut() {
                *val = uniform;
            }
        }
    }

    async fn multinomial_sample_vec<R: RngCore>(
        &self,
        probs: &[f32],
        rng: &mut R,
    ) -> Result<(usize, f32)> {
        let random_val: f32 = rng.random();
        let mut cumulative_prob = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative_prob += prob;
            if random_val <= cumulative_prob {
                return Ok((i, prob));
            }
        }
        let last_idx = probs.len() - 1;
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

    #[test]
    fn test_top_k_inplace() {
        let config = SamplingConfig::default();
        let mut strategy = SamplingStrategy::new(config);

        let mut logits = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        // top_k = 2. Should keep 0.8 and 0.5. Others -inf.
        strategy.apply_top_k_inplace(&mut logits, 2);

        assert_eq!(logits[3], 0.8);
        assert_eq!(logits[1], 0.5);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn test_softmax_inplace() {
        let mut logits = vec![0.0, 0.0];
        SamplingStrategy::softmax_inplace(&mut logits);
        assert!((logits[0] - 0.5).abs() < 1e-6);
        assert!((logits[1] - 0.5).abs() < 1e-6);

        let mut logits = vec![10.0, 10.0];
        SamplingStrategy::softmax_inplace(&mut logits);
        assert!((logits[0] - 0.5).abs() < 1e-6);
    }
}
