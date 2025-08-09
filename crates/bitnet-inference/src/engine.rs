//! # Inference Engine Implementation
//!
//! Core inference engine with CPU and GPU backend support, streaming generation,
//! and comprehensive configuration options.

use anyhow::{Result, Context};
use bitnet_common::{BitNetConfig, Device, Tensor, ConcreteTensor, BitNetTensor};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn, instrument};
use candle_core::IndexOp;

use crate::{
    backends::{Backend, CpuBackend, GpuBackend},
    cache::{KVCache, CacheConfig},
    config::{InferenceConfig, GenerationConfig},
    sampling::{SamplingStrategy, SamplingConfig},
    streaming::{GenerationStream, StreamingConfig},
};

/// Result type for inference operations
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub generated_text: String,
    pub tokens_generated: usize,
    pub latency_ms: u64,
    pub tokens_per_second: f64,
}

/// Main inference engine for BitNet models
pub struct InferenceEngine {
    model: Arc<dyn Model>,
    tokenizer: Arc<dyn Tokenizer>,
    backend: Box<dyn Backend>,
    cache: Arc<RwLock<KVCache>>,
    config: InferenceConfig,
    device: Device,
}

impl InferenceEngine {
    /// Create a new inference engine
    #[instrument(skip(model, tokenizer))]
    pub fn new(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
    ) -> Result<Self> {
        info!("Creating inference engine with device: {:?}", device);
        
        let config = InferenceConfig::default();
        let cache_config = CacheConfig::default();
        let cache = Arc::new(RwLock::new(KVCache::new(cache_config)?));
        
        let backend: Box<dyn Backend> = match &device {
            Device::Cpu => {
                debug!("Using CPU backend");
                Box::new(CpuBackend::new(model.clone())?)
            }
            Device::Cuda(_) => {
                debug!("Using GPU backend");
                Box::new(GpuBackend::new(model.clone(), device.clone())?)
            }
            Device::Metal => {
                debug!("Using GPU backend (Metal)");
                Box::new(GpuBackend::new(model.clone(), device.clone())?)
            }
        };
        
        Ok(Self {
            model,
            tokenizer,
            backend,
            cache,
            config,
            device,
        })
    }

    /// Create inference engine with custom configuration
    pub fn with_config(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
        config: InferenceConfig,
    ) -> Result<Self> {
        let mut engine = Self::new(model, tokenizer, device)?;
        engine.config = config;
        Ok(engine)
    }

    /// Generate text from a prompt
    #[instrument(skip(self))]
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let config = GenerationConfig::default();
        self.generate_with_config(prompt, &config).await
    }

    /// Generate text with custom configuration
    #[instrument(skip(self, config))]
    pub async fn generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String> {
        let start_time = std::time::Instant::now();
        
        debug!("Generating text for prompt: {:?}", &prompt[..50.min(prompt.len())]);
        
        // Tokenize input
        let input_tokens = self.tokenizer.encode(prompt, true)
            .context("Failed to tokenize input prompt")?;
        
        debug!("Input tokens: {} tokens", input_tokens.len());
        
        // Generate tokens
        let generated_tokens = self.generate_tokens(&input_tokens, config).await
            .context("Failed to generate tokens")?;
        
        // Decode output
        let generated_text = self.tokenizer.decode(&generated_tokens, true)
            .context("Failed to decode generated tokens")?;
        
        let duration = start_time.elapsed();
        let tokens_per_second = generated_tokens.len() as f64 / duration.as_secs_f64();
        
        info!(
            "Generated {} tokens in {:?} ({:.2} tokens/sec)",
            generated_tokens.len(),
            duration,
            tokens_per_second
        );
        
        Ok(generated_text)
    }

    /// Generate streaming tokens
    pub fn generate_stream(&self, prompt: &str) -> GenerationStream {
        let config = GenerationConfig::default();
        self.generate_stream_with_config(prompt, &config)
    }

    /// Generate streaming tokens with configuration
    pub fn generate_stream_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> GenerationStream {
        let streaming_config = StreamingConfig {
            buffer_size: 10,
            flush_interval_ms: 50,
        };
        
        GenerationStream::new(
            self.model.clone(),
            self.tokenizer.clone(),
            self.backend.clone_backend(),
            self.cache.clone(),
            prompt.to_string(),
            config.clone(),
            streaming_config,
        )
    }

    /// Generate tokens using the configured backend
    async fn generate_tokens(
        &self,
        input_tokens: &[u32],
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let mut generated_tokens = Vec::new();
        let mut current_tokens = input_tokens.to_vec();
        
        let sampling_config = SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            seed: config.seed,
        };
        
        let mut sampling_strategy = SamplingStrategy::new(sampling_config);
        
        for _ in 0..config.max_new_tokens {
            // Forward pass through model
            let logits = self.forward_pass(&current_tokens).await?;
            
            // Sample next token
            let next_token = sampling_strategy.sample(&logits, &current_tokens)?;
            
            // Check for stop conditions
            if self.should_stop(next_token, &generated_tokens, config) {
                break;
            }
            
            generated_tokens.push(next_token);
            current_tokens.push(next_token);
            
            // Limit context length
            if current_tokens.len() > self.config.max_context_length {
                let keep_length = self.config.max_context_length / 2;
                current_tokens = current_tokens[current_tokens.len() - keep_length..].to_vec();
            }
        }
        
        Ok(generated_tokens)
    }

    /// Perform forward pass through the model
    async fn forward_pass(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Convert tokens to tensor
        let input_tensor = self.tokens_to_tensor(tokens)?;
        
        // Forward pass through backend, passing the Arc<RwLock<KVCache>>
        let output_tensor = self.backend.forward(&input_tensor, self.cache.clone()).await?;
        
        // Extract logits from output tensor
        self.tensor_to_logits(&output_tensor)
    }

    /// Convert tokens to input tensor
    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<ConcreteTensor> {
        let shape = [1, tokens.len()];
        let tensor = BitNetTensor::from_slice(tokens, &shape, &self.device)?;
        Ok(ConcreteTensor::BitNet(tensor))
    }

    /// Extract logits from output tensor
    fn tensor_to_logits(&self, tensor: &ConcreteTensor) -> Result<Vec<f32>> {
        let candle_tensor = tensor.to_candle()?;
        let shape = candle_tensor.shape().dims();

        let logits_tensor = if shape.len() == 3 && shape[0] == 1 {
            // Shape [1, seq_len, vocab_size]
            let seq_len = shape[1];
            candle_tensor.i((0, seq_len - 1, ..))
        } else if shape.len() == 2 {
            // Shape [seq_len, vocab_size], assume single sequence
            let seq_len = shape[0];
            candle_tensor.i((seq_len - 1, ..))
        } else {
            return Err(anyhow::anyhow!("Unsupported tensor shape for logits: {:?}", shape));
        }
        .map_err(|e| anyhow::anyhow!("Failed to get logits for last token: {}", e))?;

        let logits_vec = logits_tensor.to_vec1::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert logits to vec: {}", e))?;

        Ok(logits_vec)
    }

    /// Check if generation should stop
    fn should_stop(
        &self,
        token: u32,
        generated_tokens: &[u32],
        config: &GenerationConfig,
    ) -> bool {
        // Check for EOS token
        if let Some(eos_token) = self.tokenizer.eos_token_id() {
            if token == eos_token {
                return true;
            }
        }
        
        // Check for stop sequences
        if !config.stop_sequences.is_empty() {
            let current_text = self.tokenizer.decode(generated_tokens, true).unwrap_or_default();
            for stop_seq in &config.stop_sequences {
                if current_text.ends_with(stop_seq) {
                    return true;
                }
            }
        }
        
        false
    }

    /// Get model configuration
    pub fn model_config(&self) -> &BitNetConfig {
        self.model.config()
    }

    /// Get inference statistics
    pub async fn get_stats(&self) -> InferenceStats {
        let cache = self.cache.read().await;
        InferenceStats {
            cache_size: cache.size(),
            cache_usage: cache.usage_percent(),
            backend_type: self.backend.backend_type(),
        }
    }

    /// Clear the KV cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

/// Statistics about the inference engine
#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub cache_size: usize,
    pub cache_usage: f64,
    pub backend_type: String,
}

// MockTensor is now defined in bitnet_common

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use bitnet_common::{BitNetError, MockTensor};

    struct MockModel {
        config: BitNetConfig,
    }

    impl MockModel {
        fn new() -> Self {
            Self {
                config: BitNetConfig::default(),
            }
        }
    }

    impl Model for MockModel {
        fn config(&self) -> &BitNetConfig {
            &self.config
        }

        fn forward(
            &self,
            _input: &ConcreteTensor,
            _cache: &mut dyn std::any::Any,
        ) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::Mock(MockTensor::new(vec![1, 50257])))
        }
    }

    struct MockTokenizer;

    impl Tokenizer for MockTokenizer {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> bitnet_common::Result<Vec<u32>> {
            Ok(vec![1, 2, 3])
        }

        fn decode(&self, _tokens: &[u32], _skip_special_tokens: bool) -> bitnet_common::Result<String> {
            Ok("mock generated text".to_string())
        }

        fn vocab_size(&self) -> usize {
            50257
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(50256)
        }

        fn pad_token_id(&self) -> Option<u32> {
            None
        }
    }

    #[tokio::test]
    async fn test_inference_engine_creation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_text_generation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer);
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let result = engine.generate("Hello, world!").await;
        
        assert!(result.is_ok());
        let generated_text = result.unwrap();
        assert!(!generated_text.is_empty());
    }
}