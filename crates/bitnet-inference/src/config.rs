//! # Inference Configuration
//!
//! Configuration structures for inference engine and text generation.

use serde::{Deserialize, Serialize};

/// Configuration for the inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum context length to maintain
    pub max_context_length: usize,
    /// Number of threads for CPU inference
    pub num_threads: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable mixed precision inference
    pub mixed_precision: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_context_length: 2048,
            num_threads: num_cpus::get(),
            batch_size: 1,
            mixed_precision: false,
            memory_pool_size: 1024 * 1024 * 512, // 512MB
        }
    }
}

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: u32,
    /// Sampling temperature (0.0 = deterministic, higher = more random)
    pub temperature: f32,
    /// Top-k sampling limit (0 = disabled)
    pub top_k: u32,
    /// Top-p (nucleus) sampling threshold (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty (1.0 = no penalty, higher = less repetition)
    pub repetition_penalty: f32,
    /// Stop sequences to end generation
    pub stop_sequences: Vec<String>,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    /// Whether to skip special tokens in output
    pub skip_special_tokens: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            seed: None,
            skip_special_tokens: true,
        }
    }
}

impl GenerationConfig {
    /// Create a greedy generation config (deterministic)
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 1, top_p: 1.0, ..Default::default() }
    }

    /// Create a creative generation config (high randomness)
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_k: 100,
            top_p: 0.95,
            repetition_penalty: 1.1,
            ..Default::default()
        }
    }

    /// Create a balanced generation config
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.05,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.max_new_tokens == 0 {
            return Err("max_new_tokens must be greater than 0".to_string());
        }

        if self.temperature < 0.0 {
            return Err("temperature must be non-negative".to_string());
        }

        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in range (0.0, 1.0]".to_string());
        }

        if self.repetition_penalty <= 0.0 {
            return Err("repetition_penalty must be positive".to_string());
        }

        Ok(())
    }

    /// Set random seed for reproducible generation
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Add stop sequence
    pub fn with_stop_sequence(mut self, stop_seq: String) -> Self {
        self.stop_sequences.push(stop_seq);
        self
    }

    /// Set maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_new_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }
}

impl InferenceConfig {
    /// Create configuration optimized for CPU inference
    pub fn cpu_optimized() -> Self {
        Self {
            num_threads: num_cpus::get(),
            mixed_precision: false,
            batch_size: 1,
            ..Default::default()
        }
    }

    /// Create configuration optimized for GPU inference
    pub fn gpu_optimized() -> Self {
        Self {
            mixed_precision: true,
            batch_size: 4,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            ..Default::default()
        }
    }

    /// Create configuration for memory-constrained environments
    pub fn memory_efficient() -> Self {
        Self {
            max_context_length: 1024,
            batch_size: 1,
            memory_pool_size: 1024 * 1024 * 256, // 256MB
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.max_context_length == 0 {
            return Err("max_context_length must be greater than 0".to_string());
        }

        if self.num_threads == 0 {
            return Err("num_threads must be greater than 0".to_string());
        }

        if self.batch_size == 0 {
            return Err("batch_size must be greater than 0".to_string());
        }

        if self.memory_pool_size == 0 {
            return Err("memory_pool_size must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Set number of threads
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable mixed precision
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }

    /// Set memory pool size
    pub fn with_memory_pool_size(mut self, size: usize) -> Self {
        self.memory_pool_size = size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 100);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.repetition_penalty, 1.0);
        assert!(config.stop_sequences.is_empty());
        assert!(config.seed.is_none());
        assert!(config.skip_special_tokens);
    }

    #[test]
    fn test_generation_config_presets() {
        let greedy = GenerationConfig::greedy();
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(greedy.top_k, 1);

        let creative = GenerationConfig::creative();
        assert_eq!(creative.temperature, 0.9);
        assert_eq!(creative.top_k, 100);

        let balanced = GenerationConfig::balanced();
        assert_eq!(balanced.temperature, 0.7);
        assert_eq!(balanced.repetition_penalty, 1.05);
    }

    #[test]
    fn test_generation_config_validation() {
        let mut config = GenerationConfig::default();
        assert!(config.validate().is_ok());

        config.max_new_tokens = 0;
        assert!(config.validate().is_err());

        config.max_new_tokens = 100;
        config.temperature = -1.0;
        assert!(config.validate().is_err());

        config.temperature = 0.7;
        config.top_p = 0.0;
        assert!(config.validate().is_err());

        config.top_p = 1.5;
        assert!(config.validate().is_err());

        config.top_p = 0.9;
        config.repetition_penalty = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_generation_config_builder() {
        let config = GenerationConfig::default()
            .with_seed(42)
            .with_stop_sequence("</s>".to_string())
            .with_max_tokens(200)
            .with_temperature(0.8)
            .with_top_k(40)
            .with_top_p(0.95);

        assert_eq!(config.seed, Some(42));
        assert_eq!(config.stop_sequences, vec!["</s>"]);
        assert_eq!(config.max_new_tokens, 200);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.top_p, 0.95);
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_context_length, 2048);
        assert_eq!(config.num_threads, num_cpus::get());
        assert_eq!(config.batch_size, 1);
        assert!(!config.mixed_precision);
        assert_eq!(config.memory_pool_size, 1024 * 1024 * 512);
    }

    #[test]
    fn test_inference_config_presets() {
        let cpu_config = InferenceConfig::cpu_optimized();
        assert!(!cpu_config.mixed_precision);
        assert_eq!(cpu_config.batch_size, 1);

        let gpu_config = InferenceConfig::gpu_optimized();
        assert!(gpu_config.mixed_precision);
        assert_eq!(gpu_config.batch_size, 4);

        let memory_config = InferenceConfig::memory_efficient();
        assert_eq!(memory_config.max_context_length, 1024);
        assert_eq!(memory_config.memory_pool_size, 1024 * 1024 * 256);
    }

    #[test]
    fn test_inference_config_validation() {
        let mut config = InferenceConfig::default();
        assert!(config.validate().is_ok());

        config.max_context_length = 0;
        assert!(config.validate().is_err());

        config.max_context_length = 2048;
        config.num_threads = 0;
        assert!(config.validate().is_err());

        config.num_threads = 4;
        config.batch_size = 0;
        assert!(config.validate().is_err());

        config.batch_size = 1;
        config.memory_pool_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_inference_config_builder() {
        let config = InferenceConfig::default()
            .with_threads(8)
            .with_batch_size(4)
            .with_mixed_precision(true)
            .with_memory_pool_size(1024 * 1024 * 1024);

        assert_eq!(config.num_threads, 8);
        assert_eq!(config.batch_size, 4);
        assert!(config.mixed_precision);
        assert_eq!(config.memory_pool_size, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_config_serialization() {
        let gen_config = GenerationConfig::default();
        let serialized = serde_json::to_string(&gen_config).unwrap();
        let deserialized: GenerationConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(gen_config.max_new_tokens, deserialized.max_new_tokens);
        assert_eq!(gen_config.temperature, deserialized.temperature);

        let inf_config = InferenceConfig::default();
        let serialized = serde_json::to_string(&inf_config).unwrap();
        let deserialized: InferenceConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(inf_config.max_context_length, deserialized.max_context_length);
        assert_eq!(inf_config.num_threads, deserialized.num_threads);
    }
}
