//! Sampling strategies for token generation

use crate::SamplingConfig;
use bitnet_common::{BitNetError, BitNetTensor, GenerationConfig, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// Sampling strategy for token generation
pub struct SamplingStrategy {
    config: SamplingConfig,
    rng: ChaCha8Rng,
    token_frequencies: HashMap<u32, usize>,
    token_presence: HashMap<u32, bool>,
}

impl SamplingStrategy {
    /// Create a new sampling strategy
    pub fn new(config: SamplingConfig) -> Result<Self> {
        config.validate()?;
        
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };
        
        Ok(Self {
            config,
            rng,
            token_frequencies: HashMap::new(),
            token_presence: HashMap::new(),
        })
    }
    
    /// Sample next token from logits
    pub fn sample(
        &mut self,
        logits: &BitNetTensor,
        context_tokens: &[u32],
        _step: usize,
        generation_config: &GenerationConfig,
    ) -> Result<u32> {
        // Convert logits to probabilities
        let mut probs = self.logits_to_probs(logits)?;
        
        // Apply temperature scaling
        if self.config.temperature != 1.0 {
            self.apply_temperature(&mut probs, self.config.temperature)?;
        }
        
        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            self.apply_repetition_penalty(&mut probs, context_tokens, self.config.repetition_penalty)?;
        }
        
        // Apply frequency penalty
        if self.config.frequency_penalty != 0.0 {
            self.apply_frequency_penalty(&mut probs, self.config.frequency_penalty)?;
        }
        
        // Apply presence penalty
        if self.config.presence_penalty != 0.0 {
            self.apply_presence_penalty(&mut probs, self.config.presence_penalty)?;
        }
        
        // Apply top-k filtering
        if let Some(top_k) = self.config.top_k {
            self.apply_top_k(&mut probs, top_k)?;
        }
        
        // Apply top-p (nucleus) filtering
        if let Some(top_p) = self.config.top_p {
            self.apply_top_p(&mut probs, top_p)?;
        }
        
        // Sample from the filtered distribution
        let token = if generation_config.do_sample {
            self.sample_from_distribution(&probs)?
        } else {
            self.greedy_sample(&probs)?
        };
        
        // Update token statistics
        self.update_token_stats(token);
        
        Ok(token)
    }
    
    /// Update sampling configuration
    pub fn update_config(&mut self, config: SamplingConfig) -> Result<()> {
        config.validate()?;
        
        // Update RNG if seed changed
        if config.seed != self.config.seed {
            self.rng = match config.seed {
                Some(seed) => ChaCha8Rng::seed_from_u64(seed),
                None => ChaCha8Rng::from_entropy(),
            };
        }
        
        self.config = config;
        Ok(())
    }
    
    /// Reset token statistics
    pub fn reset_stats(&mut self) {
        self.token_frequencies.clear();
        self.token_presence.clear();
    }
    
    /// Get current configuration
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }
    
    /// Convert logits tensor to probability vector
    fn logits_to_probs(&self, _logits: &BitNetTensor) -> Result<Vec<f32>> {
        // This is a simplified implementation
        // In practice, we'd extract the logits from the tensor and apply softmax
        let vocab_size = 32000; // Placeholder
        Ok(vec![1.0 / vocab_size as f32; vocab_size])
    }
    
    /// Apply temperature scaling to probabilities
    fn apply_temperature(&self, probs: &mut [f32], temperature: f32) -> Result<()> {
        if temperature <= 0.0 {
            return Err(BitNetError::Config(
                "Temperature must be positive".to_string()
            ));
        }
        
        // Scale logits by temperature before softmax
        for prob in probs.iter_mut() {
            *prob = (*prob).ln() / temperature;
        }
        
        // Re-normalize with softmax
        self.softmax(probs);
        
        Ok(())
    }
    
    /// Apply repetition penalty
    fn apply_repetition_penalty(
        &self,
        probs: &mut [f32],
        context_tokens: &[u32],
        penalty: f32,
    ) -> Result<()> {
        for &token in context_tokens {
            if let Some(prob) = probs.get_mut(token as usize) {
                if *prob > 0.0 {
                    *prob /= penalty;
                }
            }
        }
        
        // Re-normalize
        self.normalize(probs);
        
        Ok(())
    }
    
    /// Apply frequency penalty
    fn apply_frequency_penalty(&self, probs: &mut [f32], penalty: f32) -> Result<()> {
        for (token, &frequency) in &self.token_frequencies {
            if let Some(prob) = probs.get_mut(*token as usize) {
                *prob -= penalty * frequency as f32;
                *prob = prob.max(0.0); // Ensure non-negative
            }
        }
        
        // Re-normalize
        self.normalize(probs);
        
        Ok(())
    }
    
    /// Apply presence penalty
    fn apply_presence_penalty(&self, probs: &mut [f32], penalty: f32) -> Result<()> {
        for &token in self.token_presence.keys() {
            if let Some(prob) = probs.get_mut(token as usize) {
                *prob -= penalty;
                *prob = prob.max(0.0); // Ensure non-negative
            }
        }
        
        // Re-normalize
        self.normalize(probs);
        
        Ok(())
    }
    
    /// Apply top-k filtering
    fn apply_top_k(&self, probs: &mut [f32], k: usize) -> Result<()> {
        if k == 0 || k >= probs.len() {
            return Ok(());
        }
        
        // Get indices sorted by probability (descending)
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        
        // Zero out probabilities for tokens not in top-k
        for &idx in indices.iter().skip(k) {
            probs[idx] = 0.0;
        }
        
        // Re-normalize
        self.normalize(probs);
        
        Ok(())
    }
    
    /// Apply top-p (nucleus) filtering
    fn apply_top_p(&self, probs: &mut [f32], p: f32) -> Result<()> {
        if p <= 0.0 || p >= 1.0 {
            return Ok(());
        }
        
        // Get indices sorted by probability (descending)
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        
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
        
        // Zero out probabilities for tokens not in nucleus
        for &idx in indices.iter().skip(cutoff_idx) {
            probs[idx] = 0.0;
        }
        
        // Re-normalize
        self.normalize(probs);
        
        Ok(())
    }
    
    /// Sample from probability distribution
    fn sample_from_distribution(&mut self, probs: &[f32]) -> Result<u32> {
        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            return Err(BitNetError::Validation(
                "Invalid probability distribution".to_string()
            ));
        }
        
        let mut random_value: f32 = self.rng.gen_range(0.0..total);
        
        for (i, &prob) in probs.iter().enumerate() {
            random_value -= prob;
            if random_value <= 0.0 {
                return Ok(i as u32);
            }
        }
        
        // Fallback to last token
        Ok((probs.len() - 1) as u32)
    }
    
    /// Greedy sampling (select highest probability token)
    fn greedy_sample(&self, probs: &[f32]) -> Result<u32> {
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| BitNetError::Validation(
                "Empty probability distribution".to_string()
            ))?;
        
        Ok(max_idx as u32)
    }
    
    /// Apply softmax normalization
    fn softmax(&self, logits: &mut [f32]) {
        // Find maximum for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials
        for logit in logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
        }
        
        // Normalize
        self.normalize(logits);
    }
    
    /// Normalize probabilities to sum to 1
    fn normalize(&self, probs: &mut [f32]) {
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }
    }
    
    /// Update token statistics
    fn update_token_stats(&mut self, token: u32) {
        *self.token_frequencies.entry(token).or_insert(0) += 1;
        self.token_presence.insert(token, true);
    }
}

/// Sampling method enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    TopK(usize),
    TopP(f32),
    Temperature(f32),
    Combined,
}

impl SamplingMethod {
    /// Create sampling config for this method
    pub fn to_config(self) -> SamplingConfig {
        match self {
            SamplingMethod::Greedy => SamplingConfig {
                temperature: 1.0,
                top_k: None,
                top_p: None,
                ..Default::default()
            },
            SamplingMethod::TopK(k) => SamplingConfig {
                temperature: 1.0,
                top_k: Some(k),
                top_p: None,
                ..Default::default()
            },
            SamplingMethod::TopP(p) => SamplingConfig {
                temperature: 1.0,
                top_k: None,
                top_p: Some(p),
                ..Default::default()
            },
            SamplingMethod::Temperature(temp) => SamplingConfig {
                temperature: temp,
                top_k: None,
                top_p: None,
                ..Default::default()
            },
            SamplingMethod::Combined => SamplingConfig::default(),
        }
    }
}

/// Sampling statistics
#[derive(Debug, Clone, Default)]
pub struct SamplingStats {
    pub total_tokens: usize,
    pub unique_tokens: usize,
    pub most_frequent_token: Option<u32>,
    pub max_frequency: usize,
    pub entropy: f64,
}

impl SamplingStats {
    /// Calculate from token frequencies
    pub fn from_frequencies(frequencies: &HashMap<u32, usize>) -> Self {
        let total_tokens: usize = frequencies.values().sum();
        let unique_tokens = frequencies.len();
        
        let (most_frequent_token, max_frequency) = frequencies
            .iter()
            .max_by_key(|(_, &freq)| freq)
            .map(|(&token, &freq)| (Some(token), freq))
            .unwrap_or((None, 0));
        
        // Calculate entropy
        let entropy = if total_tokens > 0 {
            frequencies
                .values()
                .map(|&freq| {
                    let p = freq as f64 / total_tokens as f64;
                    -p * p.ln()
                })
                .sum()
        } else {
            0.0
        };
        
        Self {
            total_tokens,
            unique_tokens,
            most_frequent_token,
            max_frequency,
            entropy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sampling_strategy_creation() {
        let config = SamplingConfig::default();
        let strategy = SamplingStrategy::new(config);
        assert!(strategy.is_ok());
    }
    
    #[test]
    fn test_sampling_methods() {
        let greedy = SamplingMethod::Greedy.to_config();
        assert_eq!(greedy.top_k, None);
        assert_eq!(greedy.top_p, None);
        
        let top_k = SamplingMethod::TopK(50).to_config();
        assert_eq!(top_k.top_k, Some(50));
        
        let top_p = SamplingMethod::TopP(0.9).to_config();
        assert_eq!(top_p.top_p, Some(0.9));
    }
    
    #[test]
    fn test_sampling_config_validation() {
        let mut config = SamplingConfig::default();
        assert!(config.validate().is_ok());
        
        config.temperature = 0.0;
        assert!(config.validate().is_err());
        
        config.temperature = 1.0;
        config.top_k = Some(0);
        assert!(config.validate().is_err());
        
        config.top_k = Some(50);
        config.top_p = Some(1.5);
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_sampling_stats() {
        let mut frequencies = HashMap::new();
        frequencies.insert(1, 10);
        frequencies.insert(2, 5);
        frequencies.insert(3, 3);
        
        let stats = SamplingStats::from_frequencies(&frequencies);
        assert_eq!(stats.total_tokens, 18);
        assert_eq!(stats.unique_tokens, 3);
        assert_eq!(stats.most_frequent_token, Some(1));
        assert_eq!(stats.max_frequency, 10);
        assert!(stats.entropy > 0.0);
    }
}