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
    logits_buffer: Vec<f32>,
}

impl SamplingStrategy {
    /// Create new sampling strategy
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            current_repetition_penalty: config.repetition_penalty,
            config,
            repetition_counts: HashMap::new(),
            logits_buffer: Vec::new(),
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

        let mut buf = std::mem::take(&mut self.logits_buffer);
        buf.clear();

        let logits_candle = logits.to_candle()?;
        let last_logits = if logits_candle.dims().len() == 3 {
            let (batch, seq_len, vocab_size) = logits_candle.dims3()?;
            logits_candle.narrow(1, seq_len - 1, 1)?.reshape(&[batch, vocab_size])?
        } else if logits_candle.dims().len() == 2 {
            logits_candle.clone()
        } else {
            self.logits_buffer = buf;
            return Err(anyhow::anyhow!("Unexpected logits shape: {:?}", logits_candle.shape()));
        };

        let logits_slice = last_logits.flatten_all()?.to_vec1::<f32>()?;
        buf.extend_from_slice(&logits_slice);

        // Apply temperature scaling
        bitnet_logits::apply_temperature(&mut buf, self.config.temperature);

        // Apply repetition penalty
        if self.current_repetition_penalty != 1.0 && !self.repetition_counts.is_empty() {
            let token_ids: Vec<u32> = self
                .repetition_counts
                .iter()
                .flat_map(|(&id, &count)| std::iter::repeat_n(id as u32, count))
                .collect();
            bitnet_logits::apply_repetition_penalty(
                &mut buf,
                &token_ids,
                self.current_repetition_penalty,
            );
        }

        // Apply top-k filtering
        if let Some(k) = self.config.top_k {
            #[allow(clippy::collapsible_if)]
            if k > 0 && k < buf.len() {
                bitnet_logits::apply_top_k(&mut buf, k);
            }
        }

        // Softmax
        bitnet_logits::softmax_in_place(&mut buf);

        // Apply nucleus (top-p)
        if let Some(p) = self.config.top_p {
            #[allow(clippy::collapsible_if)]
            if p < 1.0 {
                bitnet_logits::apply_top_p(&mut buf, p);
            }
        }

        // Re-normalize after top-p
        let _ = bitnet_probability::renormalize_in_place(&mut buf);

        // Multinomial sample
        let random_val: f32 = rng.random();
        let mut cumulative_prob = 0.0;
        let mut max_idx = buf.len() - 1;
        let mut max_prob = *buf.last().unwrap_or(&0.0);

        let mut sampled = false;
        for (i, &prob) in buf.iter().enumerate() {
            cumulative_prob += prob;
            if random_val <= cumulative_prob {
                max_idx = i;
                max_prob = prob;
                sampled = true;
                break;
            }
        }

        if !sampled && !buf.is_empty() {
            max_idx = buf.len() - 1;
            max_prob = buf[max_idx];
        }

        self.logits_buffer = buf;
        Ok((max_idx, max_prob))
    }

    /// Greedy sampling (argmax)
    async fn greedy_sample(&mut self, logits: &BitNetTensor) -> Result<(usize, f32)> {
        let mut buf = std::mem::take(&mut self.logits_buffer);
        buf.clear();

        let logits_candle = logits.to_candle()?;

        // Get the last token's logits
        let last_logits = if logits_candle.dims().len() == 3 {
            let (batch, seq_len, vocab_size) = logits_candle.dims3()?;
            logits_candle.narrow(1, seq_len - 1, 1)?.reshape(&[batch, vocab_size])?
        } else {
            logits_candle.clone()
        };

        let logits_slice = last_logits.flatten_all()?.to_vec1::<f32>()?;
        buf.extend_from_slice(&logits_slice);

        let max_idx = bitnet_logits::argmax(&buf);

        bitnet_logits::softmax_in_place(&mut buf);
        let max_prob = buf.get(max_idx).copied().unwrap_or(0.0);

        self.logits_buffer = buf;
        Ok((max_idx, max_prob))
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
