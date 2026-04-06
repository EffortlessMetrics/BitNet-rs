//! Sampling utilities for text generation
#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::option_if_let_else
)]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::HashMap;

/// Sampling strategy for text generation
pub struct Sampler {
    rng: ChaCha20Rng,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    token_counts: HashMap<u32, usize>,
    buf: Vec<f32>,
}

impl Sampler {
    /// Create a new sampler with given parameters
    pub fn new(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: Option<u64>,
    ) -> Self {
        let rng = if let Some(seed) = seed {
            ChaCha20Rng::seed_from_u64(seed)
        } else {
            ChaCha20Rng::from_rng(&mut rand::rng())
        };

        Self {
            rng,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            token_counts: HashMap::new(),
            buf: Vec::new(),
        }
    }

    /// Sample next token from logits
    pub fn sample(&mut self, logits_in: &[f32], generated_tokens: &[u32]) -> u32 {
        // Update token counts
        for &token in generated_tokens {
            *self.token_counts.entry(token).or_insert(0) += 1;
        }

        // Use pre-allocated buffer
        let mut buf = std::mem::take(&mut self.buf);
        buf.clear();
        buf.extend_from_slice(logits_in);

        // Apply repetition penalty in-place
        self.apply_repetition_penalty_in_place(&mut buf);

        // Replace NaN logits with -inf so they are ignored by later steps
        for logit in &mut buf {
            if logit.is_nan() {
                *logit = f32::NEG_INFINITY;
            }
        }

        // Greedy decoding if temperature is 0
        if self.temperature == 0.0
            || (self.temperature == 1.0 && self.top_k == 0 && self.top_p == 1.0)
        {
            let result = argmax(&buf);
            self.buf = buf;
            return result;
        }

        // Apply temperature
        if self.temperature != 1.0 {
            for logit in &mut buf {
                *logit /= self.temperature;
            }
        }

        // Apply top-k filtering
        if self.top_k > 0 {
            self.top_k_filter_in_place(&mut buf);
        }

        // Apply top-p (nucleus) filtering
        if self.top_p < 1.0 {
            self.top_p_filter_in_place(&mut buf);
        }

        // Convert to probabilities
        let probs = softmax(&buf);

        // Sample from distribution
        let result = self.sample_from_probs(&probs);
        self.buf = buf;
        result
    }

    /// Apply repetition penalty to logits in-place
    fn apply_repetition_penalty_in_place(&self, logits: &mut [f32]) {
        if self.repetition_penalty == 1.0 {
            return;
        }

        for (&token_id, &count) in &self.token_counts {
            let idx = token_id as usize;
            if idx < logits.len() && count > 0 {
                let penalty = self.repetition_penalty.powi(count as i32);
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }

    /// Apply top-k filtering in-place
    fn top_k_filter_in_place(&self, logits: &mut [f32]) {
        if self.top_k == 0 || self.top_k >= logits.len() {
            return;
        }

        let mut indexed: Vec<(usize, f32)> =
            logits.iter().copied().enumerate().filter(|&(_, v)| !v.is_nan()).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let cutoff = self.top_k.min(indexed.len());
        // Track which indices we kept to set others (including NaNs) to -inf
        let mut keep_indices = vec![false; logits.len()];
        for &(idx, _) in indexed.iter().take(cutoff) {
            keep_indices[idx] = true;
        }

        for (i, logit) in logits.iter_mut().enumerate() {
            if !keep_indices[i] {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply top-p (nucleus) filtering in-place
    fn top_p_filter_in_place(&self, logits: &mut [f32]) {
        if self.top_p >= 1.0 {
            return;
        }

        let sanitized: Vec<f32> =
            logits.iter().map(|&v| if v.is_nan() { f32::NEG_INFINITY } else { v }).collect();

        let mut indexed: Vec<(usize, f32)> = sanitized.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let probs = softmax(&sanitized);
        let sorted_probs: Vec<_> = indexed.iter().map(|&(i, _)| probs[i]).collect();

        let mut cumsum = 0.0;
        let mut cutoff_idx = sorted_probs.len();
        for (i, &prob) in sorted_probs.iter().enumerate() {
            cumsum += prob;
            if cumsum > self.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Track which indices we kept to set others (including NaNs) to -inf
        let mut keep_indices = vec![false; logits.len()];
        for &(idx, _) in indexed.iter().take(cutoff_idx) {
            keep_indices[idx] = true;
        }

        for (i, logit) in logits.iter_mut().enumerate() {
            if !keep_indices[i] {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Sample from probability distribution
    fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        let uniform: f32 = self.rng.random();
        let mut cumsum = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum > uniform {
                return i as u32;
            }
        }

        // Fallback to last token
        (probs.len() - 1) as u32
    }
}

/// Softmax function
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0;
    let mut exp_vals = Vec::with_capacity(logits.len());

    for &logit in logits {
        let exp_val = (logit - max).exp();
        exp_vals.push(exp_val);
        exp_sum += exp_val;
    }

    for exp_val in &mut exp_vals {
        *exp_val /= exp_sum;
    }

    exp_vals
}

/// Argmax function with deterministic tie-breaking (choose lowest index on tie)
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;

    for (i, &val) in logits.iter().enumerate() {
        // On tie, prefer lower index for determinism
        if val > best_val || (val == best_val && i < best_idx) {
            best_val = val;
            best_idx = i;
        }
    }

    best_idx as u32
}

/// Greedy selection with deterministic tie-breaking for temperature=0
#[inline]
#[allow(dead_code)]
pub fn greedy_tie_break_lowest_id(logits: &[f32]) -> u32 {
    let mut best = (f32::NEG_INFINITY, u32::MAX);
    for (i, &x) in logits.iter().enumerate() {
        let id = i as u32;
        if x > best.0 || (x == best.0 && id < best.1) {
            best = (x, id);
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_argmax() {
        let logits = vec![1.0, 3.0, 2.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_argmax_tie_break() {
        // On tie, should choose lowest index
        let logits = vec![1.0, 2.0, 2.0, 1.5];
        assert_eq!(argmax(&logits), 1); // index 1, not 2
        assert_eq!(greedy_tie_break_lowest_id(&logits), 1);
    }

    #[test]
    fn test_greedy_sampling() {
        let mut sampler = Sampler::new(0.0, 0, 1.0, 1.0, Some(42));
        let logits = vec![1.0, 3.0, 2.0];
        assert_eq!(sampler.sample(&logits, &[]), 1);
    }

    #[test]
    fn test_top_k_filter() {
        let sampler = Sampler::new(1.0, 2, 1.0, 1.0, Some(42));
        let mut logits = vec![1.0, 3.0, 2.0, 0.5];
        sampler.top_k_filter_in_place(&mut logits);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[1], 3.0);
        assert_eq!(logits[2], 2.0);
    }

    #[test]
    fn test_top_k_filter_with_nan() {
        let sampler = Sampler::new(1.0, 2, 1.0, 1.0, Some(42));
        let mut logits = vec![1.0, f32::NAN, 3.0];
        sampler.top_k_filter_in_place(&mut logits);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0);
    }

    #[test]
    fn test_top_p_filter_with_nan() {
        let sampler = Sampler::new(1.0, 0, 0.9, 1.0, Some(42));
        let mut logits = vec![1.0, f32::NAN, 3.0];
        sampler.top_p_filter_in_place(&mut logits);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0);
    }

    #[test]
    fn test_sample_with_nan_logits() {
        let mut sampler = Sampler::new(1.0, 0, 1.0, 1.0, Some(42));
        let logits = vec![f32::NAN, 0.0, 1.0];
        let token = sampler.sample(&logits, &[]);
        assert_ne!(token, 0);
    }
}
