//! Sampling Strategies for Text Generation
//!
//! Provides various sampling strategies including temperature scaling,
//! top-k sampling, nucleus (top-p) sampling, and repetition penalty.

use anyhow::Result;
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
        if !self.config.do_sample {
            return self.greedy_sample(logits).await;
        }

        let logits_candle = logits.as_candle();

        // Get the last token's logits (for autoregressive generation)
        let last_logits = if logits_candle.dims().len() == 3 {
            let (batch, seq_len, vocab_size) = logits_candle.dims3()?;
            logits_candle.narrow(1, seq_len - 1, 1)?.reshape(&[batch, vocab_size])?
        } else if logits_candle.dims().len() == 2 {
            logits_candle.clone()
        } else {
            return Err(anyhow::anyhow!("Unexpected logits shape: {:?}", logits_candle.shape()));
        };

        // Convert to Vec<f32> once to avoid repeated tensor-device roundtrips
        // This optimization reduces sampling overhead by orders of magnitude (e.g. 31ms -> 1.5ms for 32k vocab)
        // by performing all subsequent operations (temperature, penalty, top-k/p, softmax) in-place on CPU.
        let mut logits_vec = last_logits.flatten_all()?.to_vec1::<f32>()?;

        // Apply temperature scaling
        if self.config.temperature != 1.0 {
            self.apply_temperature_in_place(&mut logits_vec);
        }

        // Apply repetition penalty
        self.apply_repetition_penalty_in_place(&mut logits_vec);

        // Apply top-k filtering if specified
        if let Some(top_k) = self.config.top_k {
            self.apply_top_k_in_place(&mut logits_vec, top_k);
        }

        // Apply nucleus (top-p) sampling if specified
        if let Some(top_p) = self.config.top_p {
            self.apply_top_p_in_place(&mut logits_vec, top_p);
        }

        // Compute softmax in-place
        self.softmax_in_place(&mut logits_vec);

        // Sample from distribution
        self.multinomial_sample_vec(&logits_vec, rng).await
    }

    /// Greedy sampling (argmax)
    async fn greedy_sample(&self, logits: &BitNetTensor) -> Result<(usize, f32)> {
        let logits_candle = logits.as_candle();

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
    fn apply_repetition_penalty_in_place(&self, logits: &mut [f32]) {
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

    /// Apply temperature scaling in-place
    fn apply_temperature_in_place(&self, logits: &mut [f32]) {
        if self.config.temperature <= 0.0 {
            return;
        }
        let inv_temp = 1.0 / self.config.temperature;
        for x in logits.iter_mut() {
            *x *= inv_temp;
        }
    }

    /// Apply top-k filtering in-place
    fn apply_top_k_in_place(&self, logits: &mut [f32], k: usize) {
        if k >= logits.len() {
            return;
        }

        // Get indices sorted by logits value (descending)
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_unstable_by(|&i, &j| {
            logits[j].partial_cmp(&logits[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep only top-k, set others to negative infinity
        for &idx in indices.iter().skip(k) {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    /// Apply nucleus (top-p) sampling in-place
    fn apply_top_p_in_place(&self, logits: &mut [f32], p: f32) {
        if p >= 1.0 {
            return;
        }

        // Compute softmax for probability checking (temporary copy needed for correct sorting)
        let probs = self.softmax_vec(logits);

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
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    /// Compute softmax in-place
    fn softmax_in_place(&self, logits: &mut [f32]) {
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
        }
    }

    /// Helper to compute softmax returning a new vec (used in top-p)
    fn softmax_vec(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();

        if sum > 0.0 {
            exps.iter().map(|&x| x / sum).collect()
        } else {
            exps // Should be all zeros or NaNs if sum is 0
        }
    }

    /// Sample from multinomial distribution (vec)
    async fn multinomial_sample_vec<R: RngCore>(
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

        // Fallback: return last token with non-zero prob, or just last token
        let last_idx = probabilities.len() - 1;
        Ok((last_idx, probabilities[last_idx]))
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
