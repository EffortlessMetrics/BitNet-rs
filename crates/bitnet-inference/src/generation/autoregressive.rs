//! Autoregressive Generator Implementation
//!
//! Provides autoregressive text generation with temperature, top-k, and nucleus sampling.

use anyhow::{Context, Result};
use bitnet_common::{BitNetTensor, Device, GenerationConfig as CommonGenConfig};
use candle_core::DType;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;

use super::deterministic::DeterministicGenerator;
use super::sampling::{SamplingConfig, SamplingStrategy};

/// Generation configuration specific to autoregressive generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub seed: Option<u64>,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub min_length: usize,
    pub max_length: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            do_sample: true,
            seed: None,
            eos_token_id: 2, // Common EOS token
            pad_token_id: 0, // Common PAD token
            min_length: 1,
            max_length: 2048,
        }
    }
}

impl From<CommonGenConfig> for GenerationConfig {
    fn from(config: CommonGenConfig) -> Self {
        Self {
            max_new_tokens: config.max_new_tokens,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            do_sample: config.do_sample,
            seed: config.seed,
            eos_token_id: 2,
            pad_token_id: 0,
            min_length: 1,
            max_length: 2048,
        }
    }
}

/// Generation statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub tokens_generated: usize,
    pub total_time_ms: f64,
    pub tokens_per_second: f64,
    pub repetitions_detected: usize,
    pub early_stopping: bool,
}

/// Streaming generation result
#[derive(Debug, Clone)]
pub struct GenerationStep {
    pub token_id: usize,
    pub logits: BitNetTensor,
    pub probability: f32,
    pub is_finished: bool,
    pub cumulative_text: String,
}

/// Autoregressive text generator with various sampling strategies
pub struct AutoregressiveGenerator {
    config: GenerationConfig,
    device: Device,
    rng: ChaCha8Rng,
    sampling_strategy: SamplingStrategy,
    deterministic_gen: Option<DeterministicGenerator>,

    // Generation state
    generated_tokens: VecDeque<usize>,
    repetition_window: VecDeque<usize>,
    current_length: usize,
}

impl AutoregressiveGenerator {
    /// Create new autoregressive generator
    pub fn new(config: GenerationConfig, device: Device) -> Result<Self> {
        let seed = config.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
        });

        let rng = ChaCha8Rng::seed_from_u64(seed);

        let sampling_config = SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            do_sample: config.do_sample,
        };
        let sampling_strategy = SamplingStrategy::new(sampling_config);

        // Check for deterministic generation
        let deterministic_gen = if std::env::var("BITNET_DETERMINISTIC").is_ok() {
            let det_seed =
                std::env::var("BITNET_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42);
            Some(DeterministicGenerator::new(det_seed)?)
        } else {
            None
        };

        Ok(Self {
            config,
            device,
            rng,
            sampling_strategy,
            deterministic_gen,
            generated_tokens: VecDeque::new(),
            repetition_window: VecDeque::new(),
            current_length: 0,
        })
    }

    /// Generate tokens autoregressively
    pub async fn generate<F, Fut>(
        &mut self,
        input_ids: &[usize],
        forward_fn: F,
    ) -> Result<Vec<usize>>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        self.reset_state();

        let mut current_tokens = input_ids.to_vec();
        let mut generated = Vec::new();

        for step in 0..self.config.max_new_tokens {
            // Check length constraints
            if current_tokens.len() >= self.config.max_length {
                break;
            }

            // Create input tensor
            let input_tensor = self.tokens_to_tensor(&current_tokens)?;

            // Forward pass
            let logits =
                forward_fn(input_tensor).await.context("Forward pass failed during generation")?;

            // Sample next token
            let next_token = self.sample_next_token(&logits, step).await?;

            // Check for EOS token
            if next_token == self.config.eos_token_id
                && current_tokens.len() >= self.config.min_length
            {
                break;
            }
            // Continue if we haven't reached min_length

            // Add token to sequence
            current_tokens.push(next_token);
            generated.push(next_token);
            self.update_repetition_tracking(next_token);

            // Check for repetition penalty trigger
            if self.should_apply_repetition_penalty(next_token) {
                self.sampling_strategy.increase_repetition_penalty();
            }
        }

        Ok(generated)
    }

    /// Generate tokens with streaming support
    pub async fn generate_stream<F, Fut>(
        &mut self,
        input_ids: &[usize],
        forward_fn: F,
    ) -> Result<impl futures_util::Stream<Item = Result<GenerationStep>>>
    where
        F: Fn(BitNetTensor) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<BitNetTensor>> + Send,
    {
        self.reset_state();

        let mut current_tokens = input_ids.to_vec();
        let _initial_length = current_tokens.len();

        Ok(async_stream::stream! {
            for step in 0..self.config.max_new_tokens {
                // Check length constraints
                if current_tokens.len() >= self.config.max_length {
                    yield Ok(GenerationStep {
                        token_id: self.config.eos_token_id,
                        logits: BitNetTensor::zeros(&[1], DType::F32, &self.device)?,
                        probability: 1.0,
                        is_finished: true,
                        cumulative_text: String::new(), // Would be populated by tokenizer
                    });
                    break;
                }

                // Create input tensor
                let input_tensor = match self.tokens_to_tensor(&current_tokens) {
                    Ok(tensor) => tensor,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                };

                // Forward pass
                let logits = match forward_fn(input_tensor).await {
                    Ok(logits) => logits,
                    Err(e) => {
                        yield Err(e.context("Forward pass failed during streaming generation"));
                        break;
                    }
                };

                // Sample next token
                let (next_token, probability) = match self.sample_next_token_with_prob(&logits, step).await {
                    Ok(result) => result,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                };

                let is_eos = next_token == self.config.eos_token_id;
                let is_finished = is_eos && current_tokens.len() >= self.config.min_length;

                // Yield generation step
                yield Ok(GenerationStep {
                    token_id: next_token,
                    logits: logits.clone(),
                    probability,
                    is_finished,
                    cumulative_text: String::new(), // Would be populated by tokenizer
                });

                if is_finished {
                    break;
                }

                // Add token to sequence
                current_tokens.push(next_token);
                self.update_repetition_tracking(next_token);

                // Check for repetition penalty trigger
                if self.should_apply_repetition_penalty(next_token) {
                    self.sampling_strategy.increase_repetition_penalty();
                }
            }
        })
    }

    /// Sample next token from logits
    async fn sample_next_token(&mut self, logits: &BitNetTensor, step: usize) -> Result<usize> {
        let (token, _prob) = self.sample_next_token_with_prob(logits, step).await?;
        Ok(token)
    }

    /// Sample next token with probability
    async fn sample_next_token_with_prob(
        &mut self,
        logits: &BitNetTensor,
        step: usize,
    ) -> Result<(usize, f32)> {
        // Use deterministic generation if enabled
        if let Some(ref mut det_gen) = self.deterministic_gen {
            return det_gen.sample_deterministic(logits, step).await;
        }

        // Apply sampling strategy
        self.sampling_strategy.sample(logits, &mut self.rng).await
    }

    /// Convert token sequence to tensor
    fn tokens_to_tensor(&self, tokens: &[usize]) -> Result<BitNetTensor> {
        let token_data: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        Ok(BitNetTensor::from_slice(&token_data, &[1, tokens.len()], &self.device)?)
    }

    /// Reset generation state
    fn reset_state(&mut self) {
        self.generated_tokens.clear();
        self.repetition_window.clear();
        self.current_length = 0;
    }

    /// Update repetition tracking
    fn update_repetition_tracking(&mut self, token: usize) {
        self.generated_tokens.push_back(token);
        self.repetition_window.push_back(token);

        // Keep repetition window to reasonable size
        while self.repetition_window.len() > 50 {
            self.repetition_window.pop_front();
        }

        self.current_length += 1;
    }

    /// Check if repetition penalty should be applied
    fn should_apply_repetition_penalty(&self, token: usize) -> bool {
        // Count occurrences of token in recent window
        let count = self.repetition_window.iter().filter(|&&t| t == token).count();
        count > 2 // Apply penalty if token appears more than twice
    }

    /// Get generation statistics
    pub fn get_stats(&self) -> GenerationStats {
        GenerationStats {
            tokens_generated: self.generated_tokens.len(),
            total_time_ms: 0.0,     // Would be tracked in a real implementation
            tokens_per_second: 0.0, // Would be calculated based on timing
            repetitions_detected: self.count_repetitions(),
            early_stopping: false, // Would be set based on EOS detection
        }
    }

    /// Count repetitions in generated sequence
    fn count_repetitions(&self) -> usize {
        let mut repetitions = 0;
        let tokens: Vec<usize> = self.generated_tokens.iter().cloned().collect();

        // Simple n-gram repetition detection
        for window_size in 2..=4 {
            for i in 0..tokens.len().saturating_sub(window_size * 2) {
                let window1 = &tokens[i..i + window_size];
                let window2 = &tokens[i + window_size..i + window_size * 2];
                if window1 == window2 {
                    repetitions += 1;
                }
            }
        }

        repetitions
    }

    /// Set seed for reproducible generation
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    /// Update generation config
    pub fn update_config(&mut self, config: GenerationConfig) {
        self.config = config.clone();

        // Update sampling strategy
        let sampling_config = SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            do_sample: config.do_sample,
        };
        self.sampling_strategy = SamplingStrategy::new(sampling_config);
    }

    /// Check if generation should stop early
    pub fn should_stop(&self, current_tokens: &[usize]) -> bool {
        // Check max length
        if current_tokens.len() >= self.config.max_length {
            return true;
        }

        // Check for EOS token (if min length is met)
        if let Some(&last_token) = current_tokens.last()
            && last_token == self.config.eos_token_id
            && current_tokens.len() >= self.config.min_length
        {
            return true;
        }

        // Check for excessive repetition
        if self.count_repetitions() > 10 {
            return true;
        }

        false
    }
}
