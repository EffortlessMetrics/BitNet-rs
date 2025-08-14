//! Sampling utilities for text generation

use anyhow::Result;
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
            ChaCha20Rng::from_entropy()
        };

        Self {
            rng,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            token_counts: HashMap::new(),
        }
    }

    /// Sample next token from logits
    pub fn sample(&mut self, logits: &[f32], generated_tokens: &[u32]) -> u32 {
        // Update token counts
        for &token in generated_tokens {
            *self.token_counts.entry(token).or_insert(0) += 1;
        }

        // Apply repetition penalty
        let mut logits = self.apply_repetition_penalty(logits);

        // Greedy decoding if temperature is 0
        if self.temperature == 0.0
            || (self.temperature == 1.0 && self.top_k == 0 && self.top_p == 1.0)
        {
            return argmax(&logits);
        }

        // Apply temperature
        if self.temperature != 1.0 {
            for logit in &mut logits {
                *logit /= self.temperature;
            }
        }

        // Apply top-k filtering
        if self.top_k > 0 {
            logits = self.top_k_filter(logits);
        }

        // Apply top-p (nucleus) filtering
        if self.top_p < 1.0 {
            logits = self.top_p_filter(logits);
        }

        // Convert to probabilities
        let probs = softmax(&logits);

        // Sample from distribution
        self.sample_from_probs(&probs)
    }

    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(&self, logits: &[f32]) -> Vec<f32> {
        if self.repetition_penalty == 1.0 {
            return logits.to_vec();
        }

        let mut penalized = logits.to_vec();
        for (&token_id, &count) in &self.token_counts {
            if (token_id as usize) < penalized.len() && count > 0 {
                let penalty = self.repetition_penalty.powi(count as i32);
                if penalized[token_id as usize] > 0.0 {
                    penalized[token_id as usize] /= penalty;
                } else {
                    penalized[token_id as usize] *= penalty;
                }
            }
        }
        penalized
    }

    /// Apply top-k filtering
    fn top_k_filter(&self, logits: Vec<f32>) -> Vec<f32> {
        if self.top_k == 0 || self.top_k >= logits.len() {
            return logits;
        }

        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut filtered = vec![f32::NEG_INFINITY; logits.len()];
        for i in 0..self.top_k.min(indexed.len()) {
            let (idx, val) = indexed[i];
            filtered[idx] = val;
        }
        filtered
    }

    /// Apply top-p (nucleus) filtering
    fn top_p_filter(&self, mut logits: Vec<f32>) -> Vec<f32> {
        if self.top_p >= 1.0 {
            return logits;
        }

        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let probs = softmax(&logits);
        let mut sorted_probs: Vec<_> = indexed.iter().map(|&(i, _)| probs[i]).collect();

        let mut cumsum = 0.0;
        let mut cutoff_idx = sorted_probs.len();
        for (i, &prob) in sorted_probs.iter().enumerate() {
            cumsum += prob;
            if cumsum > self.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        let mut filtered = vec![f32::NEG_INFINITY; logits.len()];
        for i in 0..cutoff_idx {
            let (idx, val) = indexed[i];
            filtered[idx] = val;
        }
        filtered
    }

    /// Sample from probability distribution
    fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        let uniform: f32 = self.rng.gen();
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

/// Argmax function
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;

    for (i, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }

    best_idx as u32
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
    fn test_greedy_sampling() {
        let mut sampler = Sampler::new(0.0, 0, 1.0, 1.0, Some(42));
        let logits = vec![1.0, 3.0, 2.0];
        assert_eq!(sampler.sample(&logits, &[]), 1);
    }

    #[test]
    fn test_top_k_filter() {
        let sampler = Sampler::new(1.0, 2, 1.0, 1.0, Some(42));
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let filtered = sampler.top_k_filter(logits);
        assert_eq!(filtered[3], f32::NEG_INFINITY);
        assert_eq!(filtered[1], 3.0);
        assert_eq!(filtered[2], 2.0);
    }
}
