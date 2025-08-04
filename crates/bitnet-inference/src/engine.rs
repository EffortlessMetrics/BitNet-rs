//! Core inference engine architecture and abstractions

use crate::{Backend, KVCache, SamplingStrategy, StreamingConfig};
use bitnet_common::{
    BitNetConfig, BitNetError, BitNetTensor, GenerationConfig, InferenceError, 
    PerformanceMetrics, Result, Tensor
};
use bitnet_kernels::KernelProvider;
use bitnet_models::Model;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main inference engine trait
pub trait InferenceEngine: Send + Sync {
    /// Generate text from a prompt
    fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String>;
    
    /// Generate tokens from input tokens
    fn generate_tokens(&mut self, input: &[u32], config: &GenerationConfig) -> Result<Vec<u32>>;
    
    /// Start streaming generation
    fn generate_stream(&mut self, prompt: &str, config: &GenerationConfig) -> Result<Box<dyn crate::GenerationStream>>;
    
    /// Get performance metrics
    fn metrics(&self) -> &PerformanceMetrics;
    
    /// Get model configuration
    fn model_config(&self) -> &BitNetConfig;
    
    /// Reset the engine state
    fn reset(&mut self) -> Result<()>;
}

/// Concrete inference engine implementation
pub struct BitNetInferenceEngine {
    model: Arc<RwLock<Box<dyn Model<Config = BitNetConfig>>>>,
    backend: Box<dyn Backend>,
    cache: KVCache,
    sampling: SamplingStrategy,
    config: InferenceConfig,
    metrics: PerformanceMetrics,
}

impl BitNetInferenceEngine {
    /// Create a new inference engine
    pub fn new(
        model: Box<dyn Model<Config = BitNetConfig>>,
        backend: Box<dyn Backend>,
        config: InferenceConfig,
    ) -> Result<Self> {
        let model_config = model.config().clone();
        let cache = KVCache::new(&model_config, config.max_sequence_length)?;
        let sampling = SamplingStrategy::new(config.sampling.clone())?;
        
        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            backend,
            cache,
            sampling,
            config,
            metrics: PerformanceMetrics::default(),
        })
    }
    
    /// Create inference engine with automatic backend selection
    pub fn with_auto_backend(
        model: Box<dyn Model<Config = BitNetConfig>>,
        config: InferenceConfig,
    ) -> Result<Self> {
        let backend = crate::backend::select_best_backend(&config)?;
        Self::new(model, backend, config)
    }
    
    /// Update configuration at runtime
    pub fn update_config(&mut self, config: InferenceConfig) -> Result<()> {
        // Validate new configuration
        config.validate()?;
        
        // Update sampling strategy if changed
        if self.config.sampling != config.sampling {
            self.sampling = SamplingStrategy::new(config.sampling.clone())?;
        }
        
        // Resize cache if needed
        if self.config.max_sequence_length != config.max_sequence_length {
            let model_config = {
                let model = self.model.try_read()
                    .map_err(|_| InferenceError::GenerationFailed { 
                        reason: "Failed to acquire model lock".to_string() 
                    })?;
                model.config().clone()
            };
            self.cache = KVCache::new(&model_config, config.max_sequence_length)?;
        }
        
        self.config = config;
        Ok(())
    }
    
    /// Get current configuration
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
    
    /// Switch backend at runtime
    pub fn switch_backend(&mut self, backend: Box<dyn Backend>) -> Result<()> {
        // Ensure cache compatibility with new backend
        self.cache.migrate_to_backend(&*backend)?;
        self.backend = backend;
        Ok(())
    }
}

impl InferenceEngine for BitNetInferenceEngine {
    fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let start_time = std::time::Instant::now();
        
        // Tokenize input
        let tokens = self.backend.tokenize(prompt)?;
        
        // Generate tokens
        let generated_tokens = self.generate_tokens(&tokens, config)?;
        
        // Detokenize output
        let output = self.backend.detokenize(&generated_tokens)?;
        
        // Update metrics
        self.metrics.latency_ms = start_time.elapsed().as_millis() as f64;
        self.metrics.tokens_per_second = generated_tokens.len() as f64 / (self.metrics.latency_ms / 1000.0);
        
        Ok(output)
    }
    
    fn generate_tokens(&mut self, input: &[u32], config: &GenerationConfig) -> Result<Vec<u32>> {
        let mut generated = Vec::new();
        let mut current_tokens = input.to_vec();
        
        // Reset cache for new generation
        self.cache.reset();
        
        for step in 0..config.max_new_tokens {
            // Check context length
            if current_tokens.len() >= self.config.max_sequence_length {
                return Err(InferenceError::ContextLengthExceeded { 
                    length: current_tokens.len() 
                }.into());
            }
            
            // Forward pass through model
            let input_tensor = self.backend.tokens_to_tensor(&current_tokens)?;
            let logits = {
                let mut model = self.model.try_write()
                    .map_err(|_| InferenceError::GenerationFailed { 
                        reason: "Failed to acquire model lock".to_string() 
                    })?;
                model.forward(&input_tensor)?
            };
            
            // Sample next token
            let next_token = self.sampling.sample(&logits, &current_tokens, step, config)?;
            
            // Check for EOS token
            if self.backend.is_eos_token(next_token) {
                break;
            }
            
            generated.push(next_token);
            current_tokens.push(next_token);
        }
        
        Ok(generated)
    }
    
    fn generate_stream(&mut self, prompt: &str, config: &GenerationConfig) -> Result<Box<dyn crate::GenerationStream>> {
        let tokens = self.backend.tokenize(prompt)?;
        let stream_config = StreamingConfig {
            buffer_size: self.config.streaming.buffer_size,
            yield_interval: self.config.streaming.yield_interval,
            enable_backpressure: self.config.streaming.enable_backpressure,
        };
        
        Ok(Box::new(crate::streaming::TokenGenerationStream::new(
            self.model.clone(),
            self.backend.clone_backend(),
            tokens,
            config.clone(),
            stream_config,
        )?))
    }
    
    fn metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }
    
    fn model_config(&self) -> &BitNetConfig {
        // This is a simplified implementation - in practice we'd need to handle the async lock
        // For now, we'll return a default config
        &BitNetConfig::default()
    }
    
    fn reset(&mut self) -> Result<()> {
        self.cache.reset();
        self.metrics = PerformanceMetrics::default();
        Ok(())
    }
}

/// Inference engine configuration
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceConfig {
    /// Maximum sequence length
    pub max_sequence_length: usize,
    
    /// Backend selection preference
    pub backend_preference: BackendPreference,
    
    /// Sampling configuration
    pub sampling: SamplingConfig,
    
    /// Streaming configuration
    pub streaming: StreamingConfig,
    
    /// Batch processing configuration
    pub batch: BatchConfig,
    
    /// Performance tuning options
    pub performance: PerformanceConfig,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: 2048,
            backend_preference: BackendPreference::Auto,
            sampling: SamplingConfig::default(),
            streaming: StreamingConfig::default(),
            batch: BatchConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl InferenceConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_sequence_length == 0 {
            return Err(BitNetError::Config(
                "max_sequence_length must be greater than 0".to_string()
            ));
        }
        
        self.sampling.validate()?;
        self.streaming.validate()?;
        self.batch.validate()?;
        self.performance.validate()?;
        
        Ok(())
    }
    
    /// Create a builder for fluent configuration
    pub fn builder() -> InferenceConfigBuilder {
        InferenceConfigBuilder::new()
    }
}

/// Backend selection preference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreference {
    /// Automatically select the best available backend
    Auto,
    /// Prefer CPU backend
    Cpu,
    /// Prefer GPU backend (fallback to CPU if unavailable)
    Gpu,
    /// Force CPU backend only
    CpuOnly,
    /// Force GPU backend only (fail if unavailable)
    GpuOnly,
}

/// Sampling configuration
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }
}

impl SamplingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(BitNetError::Config(
                "temperature must be greater than 0".to_string()
            ));
        }
        
        if let Some(top_k) = self.top_k {
            if top_k == 0 {
                return Err(BitNetError::Config(
                    "top_k must be greater than 0 when specified".to_string()
                ));
            }
        }
        
        if let Some(top_p) = self.top_p {
            if top_p <= 0.0 || top_p > 1.0 {
                return Err(BitNetError::Config(
                    "top_p must be between 0 and 1 when specified".to_string()
                ));
            }
        }
        
        if self.repetition_penalty <= 0.0 {
            return Err(BitNetError::Config(
                "repetition_penalty must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Batch processing configuration
#[derive(Debug, Clone, PartialEq)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub timeout_ms: u64,
    pub enable_dynamic_batching: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            timeout_ms: 100,
            enable_dynamic_batching: true,
        }
    }
}

impl BatchConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_batch_size == 0 {
            return Err(BitNetError::Config(
                "max_batch_size must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Performance configuration
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceConfig {
    pub num_threads: Option<usize>,
    pub enable_memory_pooling: bool,
    pub cache_size_mb: usize,
    pub enable_kernel_fusion: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Auto-detect
            enable_memory_pooling: true,
            cache_size_mb: 512,
            enable_kernel_fusion: true,
        }
    }
}

impl PerformanceConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(num_threads) = self.num_threads {
            if num_threads == 0 {
                return Err(BitNetError::Config(
                    "num_threads must be greater than 0 when specified".to_string()
                ));
            }
        }
        
        if self.cache_size_mb == 0 {
            return Err(BitNetError::Config(
                "cache_size_mb must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Builder for inference configuration
#[derive(Debug, Default)]
pub struct InferenceConfigBuilder {
    config: InferenceConfig,
}

impl InferenceConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn max_sequence_length(mut self, length: usize) -> Self {
        self.config.max_sequence_length = length;
        self
    }
    
    pub fn backend_preference(mut self, preference: BackendPreference) -> Self {
        self.config.backend_preference = preference;
        self
    }
    
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.sampling.temperature = temp;
        self
    }
    
    pub fn top_k(mut self, k: Option<usize>) -> Self {
        self.config.sampling.top_k = k;
        self
    }
    
    pub fn top_p(mut self, p: Option<f32>) -> Self {
        self.config.sampling.top_p = p;
        self
    }
    
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.config.sampling.repetition_penalty = penalty;
        self
    }
    
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.config.sampling.seed = seed;
        self
    }
    
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.batch.max_batch_size = size;
        self
    }
    
    pub fn num_threads(mut self, threads: Option<usize>) -> Self {
        self.config.performance.num_threads = threads;
        self
    }
    
    pub fn enable_memory_pooling(mut self, enable: bool) -> Self {
        self.config.performance.enable_memory_pooling = enable;
        self
    }
    
    pub fn build(self) -> Result<InferenceConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_inference_config_validation() {
        let config = InferenceConfig::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = config.clone();
        invalid_config.max_sequence_length = 0;
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_sampling_config_validation() {
        let config = SamplingConfig::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = config.clone();
        invalid_config.temperature = 0.0;
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_config_builder() {
        let config = InferenceConfig::builder()
            .max_sequence_length(4096)
            .temperature(0.8)
            .top_k(Some(40))
            .build()
            .unwrap();
        
        assert_eq!(config.max_sequence_length, 4096);
        assert_eq!(config.sampling.temperature, 0.8);
        assert_eq!(config.sampling.top_k, Some(40));
    }
}