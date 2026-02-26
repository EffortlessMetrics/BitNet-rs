//! Sampling Strategies for Text Generation
//!
//! Provides various sampling strategies including temperature scaling,
//! top-k sampling, nucleus (top-p) sampling, and repetition penalty.

use anyhow::{Context, Result};
use bitnet_common::BitNetTensor;
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
        // Extract logits to Vec<f32> first to avoid repeated tensor allocations
        let mut logits_vec = self.extract_last_logits(logits)?;

        if !self.config.do_sample {
            return self.greedy_sample_vec(&logits_vec);
        }

        // Apply temperature scaling
        if self.config.temperature != 1.0 {
            let inv_temp = 1.0 / self.config.temperature;
            for x in logits_vec.iter_mut() {
                *x *= inv_temp;
            }
        }

        // Apply repetition penalty
        self.apply_repetition_penalty(&mut logits_vec);

        // Apply top-k filtering if specified
        if let Some(top_k) = self.config.top_k {
            self.apply_top_k(&mut logits_vec, top_k);
        }

        // Apply nucleus (top-p) sampling if specified
        if let Some(top_p) = self.config.top_p {
            self.apply_top_p(&mut logits_vec, top_p);
        }

        // Convert to probabilities (softmax in place)
        self.softmax_inplace(&mut logits_vec);

        // Sample from distribution
        self.multinomial_sample(&logits_vec, rng)
    }

    fn extract_last_logits(&self, logits: &BitNetTensor) -> Result<Vec<f32>> {
        let logits_candle = logits.inner();
        let dims = logits_candle.dims();

        // Get the last token's logits (for autoregressive generation)
        // We use narrow on the tensor BEFORE flattening to avoid transferring unnecessary data from GPU
        let last_logits = if dims.len() == 3 {
             let (_, seq_len, _) = logits_candle.dims3()?;
             logits_candle.narrow(1, seq_len - 1, 1)?
        } else {
             logits_candle.clone()
        };

        last_logits.flatten_all()?.to_vec1::<f32>()
            .context("Failed to convert logits to vec")
    }

    /// Greedy sampling (argmax) on Vec
    fn greedy_sample_vec(&self, logits: &[f32]) -> Result<(usize, f32)> {
        let (max_idx, max_logit) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| anyhow::anyhow!("Empty probability distribution"))?;

        // Calculate probability: exp(0) / sum(exp(x - max)) = 1 / sum(exp(x - max))
        let sum_exp: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let max_prob = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

        Ok((max_idx, max_prob))
    }

    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(&self, logits: &mut Vec<f32>) {
        if self.current_repetition_penalty == 1.0 || self.repetition_counts.is_empty() {
            return;
        }

        // Apply penalty to repeated tokens
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

    /// Apply top-k filtering
    fn apply_top_k(&self, logits: &mut Vec<f32>, k: usize) {
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

    /// Apply nucleus (top-p) sampling
    fn apply_top_p(&self, logits: &mut Vec<f32>, p: f32) {
        // Calculate temporary probabilities for sorting
        let mut probs = logits.clone();
        self.softmax_inplace(&mut probs);

        let mut candidates: Vec<usize> = (0..probs.len())
            .filter(|&i| logits[i] > f32::NEG_INFINITY && probs[i] > 0.0)
            .collect();

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

        // Mark kept tokens
        let mut keep = vec![false; logits.len()];
        for &idx in candidates.iter().take(cutoff_idx) {
            keep[idx] = true;
        }

        // Mask the rest
        for (i, val) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *val = f32::NEG_INFINITY;
            }
        }
    }

    fn softmax_inplace(&self, x: &mut Vec<f32>) {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for val in x.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }

        if sum > 0.0 {
            for val in x.iter_mut() {
                *val /= sum;
            }
        } else {
             // Handle edge case of all -inf or NaN?
             // If all are -inf, we can't do anything. But max would be -inf.
             // If max is -inf, exp(0) = 1. sum will be non-zero (count of maxes).
             // If input was empty, loop doesn't run.
        }
    }

    /// Sample from multinomial distribution
    fn multinomial_sample<R: RngCore>(
        &self,
        probabilities: &[f32],
        rng: &mut R,
    ) -> Result<(usize, f32)> {
        // Generate random number
        let random_val: f32 = rng.random();

        // Find token by cumulative probability
        let mut cumulative_prob = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_val <= cumulative_prob {
                return Ok((i, prob));
            }
        }

        // Fallback: return last non-zero token
        for (i, &prob) in probabilities.iter().enumerate().rev() {
            if prob > 0.0 {
                return Ok((i, prob));
            }
        }

        Ok((0, 0.0))
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
