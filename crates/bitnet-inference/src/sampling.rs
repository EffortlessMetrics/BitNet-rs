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
    
    /// Update sampling parameters dynamically during generation
    pub fn update_dynamic_parameters(&mut self, step: usize, total_steps: usize) -> Result<()> {
        // Dynamic temperature adjustment (cool down over time)
        if self.config.temperature > 0.1 {
            let progress = step as f32 / total_steps as f32;
            let base_temp = self.config.temperature;
            self.config.temperature = base_temp * (1.0 - progress * 0.5).max(0.1);
        }
        
        // Dynamic top-k adjustment (become more selective over time)
        if let Some(top_k) = self.config.top_k {
            let progress = step as f32 / total_steps as f32;
            let new_k = (top_k as f32 * (1.0 - progress * 0.3)).max(1.0) as usize;
            self.config.top_k = Some(new_k.max(1));
        }
        
        // Dynamic repetition penalty (increase over time to avoid loops)
        if step > total_steps / 2 {
            let progress = (step - total_steps / 2) as f32 / (total_steps / 2) as f32;
            self.config.repetition_penalty += progress * 0.1;
        }
        
        Ok(())
    }
    
    /// Validate sampling parameters with detailed error messages
    pub fn validate_parameters(&self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();
        
        // Temperature validation
        if self.config.temperature < 0.1 {
            warnings.push("Temperature is very low, may produce repetitive text".to_string());
        } else if self.config.temperature > 2.0 {
            warnings.push("Temperature is very high, may produce incoherent text".to_string());
        }
        
        // Top-k validation
        if let Some(top_k) = self.config.top_k {
            if top_k < 5 {
                warnings.push("Top-k is very low, may produce repetitive text".to_string());
            } else if top_k > 1000 {
                warnings.push("Top-k is very high, may not have much effect".to_string());
            }
        }
        
        // Top-p validation
        if let Some(top_p) = self.config.top_p {
            if top_p < 0.1 {
                warnings.push("Top-p is very low, may produce repetitive text".to_string());
            } else if top_p > 0.99 {
                warnings.push("Top-p is very high, may not have much effect".to_string());
            }
        }
        
        // Repetition penalty validation
        if self.config.repetition_penalty < 1.0 {
            warnings.push("Repetition penalty < 1.0 will encourage repetition".to_string());
        } else if self.config.repetition_penalty > 2.0 {
            warnings.push("Repetition penalty is very high, may produce unnatural text".to_string());
        }
        
        // Conflicting parameters
        if self.config.top_k.is_some() && self.config.top_p.is_some() {
            warnings.push("Both top-k and top-p are set, top-k will be applied first".to_string());
        }
        
        Ok(warnings)
    }
    
    /// Get sampling statistics for analysis
    pub fn get_sampling_stats(&self) -> SamplingStats {
        SamplingStats::from_frequencies(&self.token_frequencies)
    }
    
    /// Adaptive sampling based on context
    pub fn adaptive_sample(
        &mut self,
        logits: &BitNetTensor,
        context_tokens: &[u32],
        step: usize,
        generation_config: &GenerationConfig,
        context_analysis: &ContextAnalysis,
    ) -> Result<u32> {
        // Adjust parameters based on context analysis
        let mut adjusted_config = self.config.clone();
        
        // If context shows repetition, increase penalties
        if context_analysis.repetition_score > 0.7 {
            adjusted_config.repetition_penalty *= 1.2;
            adjusted_config.frequency_penalty += 0.1;
        }
        
        // If context shows low diversity, increase temperature
        if context_analysis.diversity_score < 0.3 {
            adjusted_config.temperature *= 1.1;
        }
        
        // If context shows high uncertainty, be more conservative
        if context_analysis.uncertainty_score > 0.8 {
            adjusted_config.temperature *= 0.9;
            if let Some(top_k) = adjusted_config.top_k {
                adjusted_config.top_k = Some((top_k as f32 * 0.8) as usize);
            }
        }
        
        // Temporarily update config
        let original_config = self.config.clone();
        self.config = adjusted_config;
        
        // Sample with adjusted parameters
        let result = self.sample(logits, context_tokens, step, generation_config);
        
        // Restore original config
        self.config = original_config;
        
        result
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
    
    /// Contrastive search sampling
    pub fn contrastive_sample(
        &mut self,
        logits: &BitNetTensor,
        _context_tokens: &[u32],
        _alpha: f32,
        k: usize,
    ) -> Result<u32> {
        // This is a simplified implementation of contrastive search
        // In practice, would need to compute similarity with context
        
        let mut probs = self.logits_to_probs(logits)?;
        
        // Apply top-k filtering first
        self.apply_top_k(&mut probs, k)?;
        
        // For now, use regular sampling (full contrastive search requires more context)
        self.sample_from_distribution(&probs)
    }
    
    /// Typical sampling (locally typical sampling)
    pub fn typical_sample(
        &mut self,
        logits: &BitNetTensor,
        tau: f32,
    ) -> Result<u32> {
        let probs = self.logits_to_probs(logits)?;
        
        // Calculate entropy of the distribution
        let entropy: f32 = probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        
        // Calculate surprisal for each token
        let mut token_surprisals: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i, -p.ln()))
            .collect();
        
        // Sort by how close surprisal is to entropy (typical tokens)
        token_surprisals.sort_by(|a, b| {
            let diff_a = (a.1 - entropy).abs();
            let diff_b = (b.1 - entropy).abs();
            diff_a.partial_cmp(&diff_b).unwrap()
        });
        
        // Keep tokens within tau of the entropy
        let mut cumulative_prob = 0.0;
        let mut filtered_probs = vec![0.0; probs.len()];
        
        for (idx, surprisal) in token_surprisals {
            if (surprisal - entropy).abs() <= tau {
                filtered_probs[idx] = probs[idx];
                cumulative_prob += probs[idx];
                if cumulative_prob >= 0.95 {
                    break;
                }
            }
        }
        
        // Renormalize
        if cumulative_prob > 0.0 {
            for prob in &mut filtered_probs {
                *prob /= cumulative_prob;
            }
        }
        
        self.sample_from_distribution(&filtered_probs)
    }
    
    /// Mirostat sampling for coherence
    pub fn mirostat_sample(
        &mut self,
        logits: &BitNetTensor,
        target_surprise: f32,
        learning_rate: f32,
        tau: &mut f32,
    ) -> Result<u32> {
        let mut probs = self.logits_to_probs(logits)?;
        
        // Apply current tau threshold
        let mut filtered_probs = vec![0.0; probs.len()];
        let mut cumulative_prob = 0.0;
        
        // Sort by probability (descending)
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Apply tau threshold
        for (idx, prob) in indexed_probs {
            if prob >= *tau {
                filtered_probs[idx] = prob;
                cumulative_prob += prob;
            }
        }
        
        // Renormalize
        if cumulative_prob > 0.0 {
            for prob in &mut filtered_probs {
                *prob /= cumulative_prob;
            }
        }
        
        // Sample token
        let token = self.sample_from_distribution(&filtered_probs)?;
        
        // Calculate actual surprise and update tau
        let token_prob = probs[token as usize];
        let actual_surprise = if token_prob > 0.0 { -token_prob.ln() } else { 10.0 };
        let surprise_error = actual_surprise - target_surprise;
        *tau = (*tau - learning_rate * surprise_error).max(0.001);
        
        Ok(token)
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

/// Context analysis for adaptive sampling
#[derive(Debug, Clone)]
pub struct ContextAnalysis {
    pub repetition_score: f64,
    pub diversity_score: f64,
    pub uncertainty_score: f64,
    pub coherence_score: f64,
}

impl ContextAnalysis {
    /// Analyze context tokens for adaptive sampling
    pub fn analyze(tokens: &[u32]) -> Self {
        let repetition_score = Self::calculate_repetition_score(tokens);
        let diversity_score = Self::calculate_diversity_score(tokens);
        let uncertainty_score = 0.5; // Placeholder - would analyze logits
        let coherence_score = 0.7; // Placeholder - would analyze semantic coherence
        
        Self {
            repetition_score,
            diversity_score,
            uncertainty_score,
            coherence_score,
        }
    }
    
    fn calculate_repetition_score(tokens: &[u32]) -> f64 {
        if tokens.len() < 4 {
            return 0.0;
        }
        
        let mut repeated_sequences = 0;
        let window_size = 3;
        
        for i in 0..tokens.len().saturating_sub(window_size * 2) {
            let window1 = &tokens[i..i + window_size];
            for j in (i + window_size)..tokens.len().saturating_sub(window_size) {
                let window2 = &tokens[j..j + window_size];
                if window1 == window2 {
                    repeated_sequences += 1;
                }
            }
        }
        
        repeated_sequences as f64 / tokens.len().saturating_sub(window_size) as f64
    }
    
    fn calculate_diversity_score(tokens: &[u32]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }
        
        let unique_tokens: std::collections::HashSet<_> = tokens.iter().collect();
        unique_tokens.len() as f64 / tokens.len() as f64
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
    pub repetition_rate: f64,
    pub diversity_score: f64,
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
        
        // Calculate repetition rate
        let repetition_rate = if total_tokens > 0 {
            let repeated_tokens = frequencies.values().filter(|&&freq| freq > 1).sum::<usize>();
            repeated_tokens as f64 / total_tokens as f64
        } else {
            0.0
        };
        
        // Calculate diversity score
        let diversity_score = if total_tokens > 0 {
            unique_tokens as f64 / total_tokens as f64
        } else {
            0.0
        };
        
        Self {
            total_tokens,
            unique_tokens,
            most_frequent_token,
            max_frequency,
            entropy,
            repetition_rate,
            diversity_score,
        }
    }
    
    /// Get quality score (0.0 to 1.0, higher is better)
    pub fn quality_score(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        
        // Combine metrics for overall quality
        let entropy_score = (self.entropy / 10.0).min(1.0); // Normalize entropy
        let diversity_score = self.diversity_score;
        let repetition_penalty = 1.0 - self.repetition_rate;
        
        (entropy_score + diversity_score + repetition_penalty) / 3.0
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
        assert!(stats.repetition_rate > 0.0);
        assert!(stats.diversity_score > 0.0);
        assert!(stats.quality_score() > 0.0);
    }
    
    #[test]
    fn test_context_analysis() {
        let tokens = vec![1, 2, 3, 1, 2, 3, 4, 5]; // Some repetition
        let analysis = ContextAnalysis::analyze(&tokens);
        
        assert!(analysis.repetition_score > 0.0);
        assert!(analysis.diversity_score > 0.0);
        assert!(analysis.diversity_score < 1.0); // Not all unique
    }
    
    #[test]
    fn test_dynamic_parameter_adjustment() {
        let config = SamplingConfig::default();
        let mut strategy = SamplingStrategy::new(config).unwrap();
        
        let original_temp = strategy.config.temperature;
        strategy.update_dynamic_parameters(50, 100).unwrap();
        
        // Temperature should decrease over time
        assert!(strategy.config.temperature <= original_temp);
    }
    
    #[test]
    fn test_parameter_validation() {
        let config = SamplingConfig {
            temperature: 0.05, // Very low
            top_k: Some(2),    // Very low
            top_p: Some(0.05), // Very low
            repetition_penalty: 2.5, // Very high
            ..Default::default()
        };
        
        let strategy = SamplingStrategy::new(config).unwrap();
        let warnings = strategy.validate_parameters().unwrap();
        
        // Should have multiple warnings
        assert!(!warnings.is_empty());
        assert!(warnings.len() >= 3);
    }
    
    #[test]
    fn test_sampling_quality_score() {
        // High quality: diverse, low repetition
        let mut high_quality_freq = HashMap::new();
        for i in 1..=10 {
            high_quality_freq.insert(i, 1); // All unique
        }
        let high_quality_stats = SamplingStats::from_frequencies(&high_quality_freq);
        
        // Low quality: repetitive
        let mut low_quality_freq = HashMap::new();
        low_quality_freq.insert(1, 8);
        low_quality_freq.insert(2, 2);
        let low_quality_stats = SamplingStats::from_frequencies(&low_quality_freq);
        
        assert!(high_quality_stats.quality_score() > low_quality_stats.quality_score());
    }
}