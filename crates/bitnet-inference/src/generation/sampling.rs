//! Sampling Strategies for Text Generation
//!
//! Provides various sampling strategies including temperature scaling,
//! top-k sampling, nucleus (top-p) sampling, and repetition penalty.

use anyhow::{Context, Result};
use bitnet_common::{BitNetTensor, Tensor};
use candle_core::Tensor as CandleTensor;
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

        // Apply temperature scaling
        let scaled_logits = if self.config.temperature != 1.0 {
            last_logits.affine(1.0 / self.config.temperature as f64, 0.0)?
        } else {
            last_logits
        };

        // Apply repetition penalty
        let penalized_logits = self.apply_repetition_penalty(&scaled_logits)?;

        // Apply top-k filtering if specified
        let filtered_logits = if let Some(top_k) = self.config.top_k {
            self.apply_top_k(&penalized_logits, top_k)?
        } else {
            penalized_logits
        };

        // Apply nucleus (top-p) sampling if specified
        let final_logits = if let Some(top_p) = self.config.top_p {
            self.apply_top_p(&filtered_logits, top_p)?
        } else {
            filtered_logits
        };

        // Convert to probabilities
        let probabilities = candle_nn::ops::softmax(&final_logits, candle_core::D::Minus1)?;

        // Sample from distribution
        self.multinomial_sample(&probabilities, rng).await
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

    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(&self, logits: &CandleTensor) -> Result<CandleTensor> {
        if self.current_repetition_penalty == 1.0 || self.repetition_counts.is_empty() {
            return Ok(logits.clone());
        }

        let mut logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;

        // Apply penalty to repeated tokens
        for (&token_id, &count) in &self.repetition_counts {
            if token_id < logits_vec.len() && count > 0 {
                let penalty_factor = self.current_repetition_penalty.powi(count as i32);
                if logits_vec[token_id] > 0.0 {
                    logits_vec[token_id] /= penalty_factor;
                } else {
                    logits_vec[token_id] *= penalty_factor;
                }
            }
        }

        CandleTensor::from_slice(&logits_vec, logits.shape(), logits.device())
            .context("Failed to create tensor from penalized logits")
    }

    /// Apply top-k filtering
    fn apply_top_k(&self, logits: &CandleTensor, k: usize) -> Result<CandleTensor> {
        let mut logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let vocab_size = logits_vec.len();

        // k=0 means "all tokens" — skip filtering entirely
        if k == 0 || k >= vocab_size {
            return Ok(logits.clone());
        }

        // O(N) partition: find the k-th largest value, then mask everything below it.
        // This is significantly faster than a full sort for large vocabularies (32k+).
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        // select_nth_unstable_by rearranges so that element at position k-1 is the
        // (k-1)-th largest (descending), with no guarantees on relative order.
        indices.select_nth_unstable_by(k - 1, |&i, &j| {
            logits_vec[j].partial_cmp(&logits_vec[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Determine the cutoff value (the k-th largest logit)
        let cutoff = logits_vec[indices[k - 1]];

        // Mask all tokens below the cutoff
        for val in logits_vec.iter_mut() {
            if *val < cutoff {
                *val = f32::NEG_INFINITY;
            }
        }

        CandleTensor::from_slice(&logits_vec, logits.shape(), logits.device())
            .context("Failed to create tensor from top-k filtered logits")
    }

    /// Apply nucleus (top-p) sampling
    fn apply_top_p(&self, logits: &CandleTensor, p: f32) -> Result<CandleTensor> {
        let probabilities = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
        let prob_vec = probabilities.flatten_all()?.to_vec1::<f32>()?;
        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;

        // Only consider candidates that are not already masked (NEG_INFINITY)
        // and have non-negligible probability. Sorting a sparse subset is faster
        // than sorting the full vocabulary (important at 32k-128k vocab sizes).
        let mut candidates: Vec<usize> = (0..prob_vec.len())
            .filter(|&i| logits_vec[i] > f32::NEG_INFINITY && prob_vec[i] > 0.0)
            .collect();

        // Sort candidates by probability descending
        candidates.sort_unstable_by(|&i, &j| {
            prob_vec[j].partial_cmp(&prob_vec[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find cutoff point where cumulative probability exceeds p
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = candidates.len();

        for (i, &idx) in candidates.iter().enumerate() {
            cumulative_prob += prob_vec[idx];
            if cumulative_prob >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Build filtered logits: keep nucleus tokens, mask the rest
        let mut filtered_logits = vec![f32::NEG_INFINITY; prob_vec.len()];
        for &idx in candidates.iter().take(cutoff_idx) {
            filtered_logits[idx] = logits_vec[idx];
        }

        CandleTensor::from_slice(&filtered_logits, logits.shape(), logits.device())
            .context("Failed to create tensor from nucleus filtered logits")
    }

    /// Sample from multinomial distribution
    async fn multinomial_sample<R: RngCore>(
        &self,
        probabilities: &CandleTensor,
        rng: &mut R,
    ) -> Result<(usize, f32)> {
        let prob_vec = probabilities.flatten_all()?.to_vec1::<f32>()?;

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
