//! Unit tests for individual bitnet-inference modules
//!
//! These tests focus on testing individual components in isolation:
//! - Configuration validation and serialization
//! - Sampling strategies and algorithms
//! - Cache management
//! - Backend selection and capabilities
//! - Error handling and edge cases

use bitnet_common::{BitNetConfig, BitNetError, ConcreteTensor, Device, Tensor};
use bitnet_inference::prelude::*;
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;

// Mock implementations for unit testing

struct MockModel {
    config: BitNetConfig,
}

impl MockModel {
    fn new() -> Self {
        Self { config: BitNetConfig::default() }
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
    ) -> Result<ConcreteTensor, BitNetError> {
        Ok(ConcreteTensor::mock(vec![1, 50257]))
    }

    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor, BitNetError> {
        // Create a mock embedding tensor with shape [seq_len, hidden_dim]
        let seq_len = tokens.len();
        let hidden_dim = self.config.model.hidden_size;
        Ok(ConcreteTensor::mock(vec![seq_len, hidden_dim]))
    }

    fn logits(&self, _hidden: &ConcreteTensor) -> Result<ConcreteTensor, BitNetError> {
        // Create a mock logits tensor with shape [batch, vocab_size]
        Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
    }
}

struct MockTokenizer;

impl MockTokenizer {
    fn new() -> Self {
        Self
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, BitNetError> {
        Ok((0..text.len().min(10)).map(|i| i as u32 + 1).collect())
    }

    fn decode(&self, tokens: &[u32], _skip_special_tokens: bool) -> Result<String, BitNetError> {
        Ok(format!("decoded_{}_tokens", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        50257
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(50256)
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(50257)
    }
}

mod config_unit_tests {
    use super::*;

    #[test]
    fn test_inference_config_creation() {
        let config = InferenceConfig::default();
        assert!(config.max_context_length > 0);
        assert!(config.num_threads > 0);
        assert!(config.batch_size > 0);
        assert!(!config.mixed_precision); // Default should be false
        assert!(config.memory_pool_size > 0);
    }

    #[test]
    fn test_inference_config_builder_pattern() {
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
    fn test_inference_config_presets() {
        let cpu_config = InferenceConfig::cpu_optimized();
        assert!(!cpu_config.mixed_precision);
        assert_eq!(cpu_config.batch_size, 1);
        assert_eq!(cpu_config.num_threads, num_cpus::get());

        let gpu_config = InferenceConfig::gpu_optimized();
        assert!(gpu_config.mixed_precision);
        assert_eq!(gpu_config.batch_size, 4);
        assert!(gpu_config.memory_pool_size > cpu_config.memory_pool_size);

        let memory_config = InferenceConfig::memory_efficient();
        assert_eq!(memory_config.max_context_length, 1024);
        assert_eq!(memory_config.batch_size, 1);
        assert_eq!(memory_config.memory_pool_size, 1024 * 1024 * 256);
    }

    #[test]
    fn test_inference_config_validation() {
        let mut config = InferenceConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid max_context_length
        config.max_context_length = 0;
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("max_context_length"));

        // Test invalid num_threads
        config.max_context_length = 2048;
        config.num_threads = 0;
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("num_threads"));

        // Test invalid batch_size
        config.num_threads = 4;
        config.batch_size = 0;
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("batch_size"));

        // Test invalid memory_pool_size
        config.batch_size = 1;
        config.memory_pool_size = 0;
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("memory_pool_size"));
    }

    #[test]
    fn test_generation_config_creation() {
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
        assert_eq!(greedy.top_p, 1.0);

        let creative = GenerationConfig::creative();
        assert_eq!(creative.temperature, 0.9);
        assert_eq!(creative.top_k, 100);
        assert_eq!(creative.top_p, 0.95);
        assert_eq!(creative.repetition_penalty, 1.1);

        let balanced = GenerationConfig::balanced();
        assert_eq!(balanced.temperature, 0.7);
        assert_eq!(balanced.top_k, 50);
        assert_eq!(balanced.top_p, 0.9);
        assert_eq!(balanced.repetition_penalty, 1.05);
    }

    #[test]
    fn test_generation_config_validation() {
        let mut config = GenerationConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid max_new_tokens
        config.max_new_tokens = 0;
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("max_new_tokens"));

        // Test invalid temperature
        config.max_new_tokens = 100;
        config.temperature = -1.0;
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("temperature"));

        // Test invalid top_p (too low)
        config.temperature = 0.7;
        config.top_p = 0.0;
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("top_p"));

        // Test invalid top_p (too high)
        config.top_p = 1.5;
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("top_p"));

        // Test invalid repetition_penalty
        config.top_p = 0.9;
        config.repetition_penalty = 0.0;
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("repetition_penalty"));
    }

    #[test]
    fn test_generation_config_builder_pattern() {
        let config = GenerationConfig::default()
            .with_seed(42)
            .with_stop_sequence("</s>".to_string())
            .with_stop_sequence("\n".to_string())
            .with_max_tokens(200)
            .with_temperature(0.8)
            .with_top_k(40)
            .with_top_p(0.95);

        assert_eq!(config.seed, Some(42));
        assert_eq!(config.stop_sequences, vec!["</s>", "\n"]);
        assert_eq!(config.max_new_tokens, 200);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.top_p, 0.95);
    }

    #[test]
    fn test_config_serialization_deserialization() {
        let original_gen_config =
            GenerationConfig::default().with_seed(42).with_temperature(0.8).with_max_tokens(150);

        let serialized = serde_json::to_string(&original_gen_config).unwrap();
        let deserialized: GenerationConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original_gen_config.seed, deserialized.seed);
        assert_eq!(original_gen_config.temperature, deserialized.temperature);
        assert_eq!(original_gen_config.max_new_tokens, deserialized.max_new_tokens);

        let original_inf_config = InferenceConfig::default().with_threads(8).with_batch_size(4);

        let serialized = serde_json::to_string(&original_inf_config).unwrap();
        let deserialized: InferenceConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original_inf_config.num_threads, deserialized.num_threads);
        assert_eq!(original_inf_config.batch_size, deserialized.batch_size);
    }
}

mod sampling_unit_tests {
    use bitnet_inference::sampling::{SamplingConfig, SamplingStrategy};

    #[test]
    fn test_sampling_config_creation() {
        let config = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: Some(42),
        };

        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_sampling_strategy_creation() {
        let config = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: Some(42),
        };

        let _strategy = SamplingStrategy::new(config);
        // Strategy should be created successfully
        // We can't test much more without exposing internal state
    }

    #[test]
    fn test_sampling_with_different_temperatures() {
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let context = vec![1, 2, 3];

        // Test with low temperature (more deterministic)
        let low_temp_config = SamplingConfig {
            temperature: 0.1,
            top_k: 10,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        };
        let mut low_temp_strategy = SamplingStrategy::new(low_temp_config);
        let low_temp_token = low_temp_strategy.sample(&logits, &context);
        assert!(low_temp_token.is_ok());

        // Test with high temperature (more random)
        let high_temp_config = SamplingConfig {
            temperature: 2.0,
            top_k: 10,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        };
        let mut high_temp_strategy = SamplingStrategy::new(high_temp_config);
        let high_temp_token = high_temp_strategy.sample(&logits, &context);
        assert!(high_temp_token.is_ok());
    }

    #[test]
    fn test_sampling_with_top_k() {
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let context = vec![1, 2, 3];

        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 3, // Only consider top 3 tokens
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        };

        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &context);
        assert!(token.is_ok());

        // Token should be one of the top 3 (indices 7, 8, 9)
        let token_id = token.unwrap();
        assert!(token_id >= 7 && token_id <= 9);
    }

    #[test]
    fn test_sampling_with_top_p() {
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let context = vec![1, 2, 3];

        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,   // Disable top-k
            top_p: 0.5, // Only consider tokens that make up 50% of probability mass
            repetition_penalty: 1.0,
            seed: Some(42),
        };

        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &context);
        assert!(token.is_ok());
    }

    #[test]
    fn test_sampling_with_repetition_penalty() {
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let context = vec![9, 8, 7]; // Recent tokens that should be penalized

        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.5, // Penalize repetition
            seed: Some(42),
        };

        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &context);
        assert!(token.is_ok());

        // The sampled token should be less likely to be one of the recent tokens
        // This is probabilistic, so we can't guarantee it, but we can test the mechanism
    }

    #[test]
    fn test_sampling_reproducibility() {
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let context = vec![1, 2, 3];

        let config = SamplingConfig {
            temperature: 0.8,
            top_k: 5,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: Some(42),
        };

        let mut strategy1 = SamplingStrategy::new(config.clone());
        let mut strategy2 = SamplingStrategy::new(config);

        let token1 = strategy1.sample(&logits, &context).unwrap();
        let token2 = strategy2.sample(&logits, &context).unwrap();

        // With the same seed, results should be reproducible
        assert_eq!(token1, token2);
    }

    #[test]
    fn test_sampling_edge_cases() {
        let context = vec![1, 2, 3];

        // Test with empty logits
        let empty_logits = vec![];
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        };
        let mut strategy = SamplingStrategy::new(config);
        let result = strategy.sample(&empty_logits, &context);
        assert!(result.is_err());

        // Test with single logit
        let single_logit = vec![1.0];
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        };
        let mut strategy = SamplingStrategy::new(config);
        let result = strategy.sample(&single_logit, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);

        // Test with zero temperature
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        };
        let mut strategy = SamplingStrategy::new(config);
        let result = strategy.sample(&logits, &context);
        assert!(result.is_ok());
        // Should always pick the highest logit (index 4)
        assert_eq!(result.unwrap(), 4);
    }
}

mod cache_unit_tests {
    use bitnet_inference::cache::{CacheConfig, KVCache};

    #[test]
    fn test_cache_config_creation() {
        let config = CacheConfig::default();
        assert!(config.max_size_bytes > 0);
        assert!(config.max_sequence_length > 0);
    }

    #[test]
    fn test_cache_creation() {
        let config = CacheConfig::default();
        let cache = KVCache::new(config);
        assert!(cache.is_ok());

        let cache = cache.unwrap();
        assert_eq!(cache.size(), 0);
        assert_eq!(cache.usage_percent(), 0.0);
    }

    #[test]
    fn test_cache_operations() {
        let config = CacheConfig::default();
        let mut cache = KVCache::new(config).unwrap();

        // Initially empty
        assert_eq!(cache.size(), 0);
        assert_eq!(cache.usage_percent(), 0.0);

        // Clear should work on empty cache
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_config_validation() {
        let mut config = CacheConfig::default();

        // Valid config should work
        let cache = KVCache::new(config.clone());
        assert!(cache.is_ok());

        // Test with zero max_size_bytes (should still work, might use default)
        config.max_size_bytes = 0;
        let cache = KVCache::new(config.clone());
        // Implementation might handle this gracefully
        assert!(cache.is_ok());

        // Test with zero max_sequence_length
        config.max_size_bytes = 1024 * 1024;
        config.max_sequence_length = 0;
        let cache = KVCache::new(config);
        // Implementation might handle this gracefully
        assert!(cache.is_ok());
    }

    #[test]
    fn test_cache_size_limits() {
        let config = CacheConfig {
            max_size_bytes: 1024, // Small cache
            max_sequence_length: 10,
            ..Default::default()
        };

        let cache = KVCache::new(config);
        assert!(cache.is_ok());

        let cache = cache.unwrap();
        assert_eq!(cache.size(), 0);

        // Usage should be within bounds
        let usage = cache.usage_percent();
        assert!(usage >= 0.0 && usage <= 100.0);
    }
}

mod backend_unit_tests {
    use super::*;
    use bitnet_inference::backends::{
        select_backend, Backend, BackendCapabilities, CpuBackend, GpuBackend,
    };

    #[tokio::test]
    async fn test_cpu_backend_creation() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model);
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        assert_eq!(backend.backend_type(), "cpu");
    }

    #[tokio::test]
    async fn test_cpu_backend_with_threads() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::with_threads(model, 4);
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        assert_eq!(backend.backend_type(), "cpu");
    }

    #[tokio::test]
    async fn test_cpu_backend_capabilities() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model).unwrap();
        let capabilities = backend.capabilities();

        assert!(!capabilities.supports_mixed_precision);
        assert!(capabilities.supports_batching);
        assert!(capabilities.max_batch_size > 0);
        assert!(capabilities.memory_efficient);
    }

    #[tokio::test]
    async fn test_cpu_backend_forward() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model).unwrap();
        let input = ConcreteTensor::mock(vec![1, 512]);
        let mut cache = KVCache::new(CacheConfig::default()).unwrap();

        let output = backend.forward(&input, &mut cache).await;
        assert!(output.is_ok());

        let output_tensor = output.unwrap();
        assert_eq!(output_tensor.shape(), &[1, 50257]);
    }

    #[tokio::test]
    async fn test_cpu_backend_warmup() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model).unwrap();

        let result = backend.warmup().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cpu_backend_clone() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model).unwrap();
        let cloned = backend.clone_backend();

        assert_eq!(backend.backend_type(), cloned.backend_type());
    }

    #[test]
    fn test_gpu_backend_availability() {
        // Test GPU availability check
        let is_available = GpuBackend::is_available();
        // This depends on compile-time features, so we just check it doesn't panic
        assert!(is_available == true || is_available == false);
    }

    #[tokio::test]
    async fn test_gpu_backend_creation() {
        let model = Arc::new(MockModel::new());
        let device = Device::Cuda(0);

        let backend = GpuBackend::new(model, device);

        if GpuBackend::is_available() {
            assert!(backend.is_ok());
            let backend = backend.unwrap();
            assert!(backend.backend_type().starts_with("gpu_"));
        } else {
            // Should fail if GPU not available
            assert!(backend.is_err());
        }
    }

    #[tokio::test]
    async fn test_gpu_backend_with_mixed_precision() {
        let model = Arc::new(MockModel::new());
        let device = Device::Cuda(0);

        let backend = GpuBackend::with_mixed_precision(model, device, true);

        if GpuBackend::is_available() {
            assert!(backend.is_ok());
        } else {
            assert!(backend.is_err());
        }
    }

    #[tokio::test]
    async fn test_gpu_backend_capabilities() {
        if !GpuBackend::is_available() {
            return; // Skip if GPU not available
        }

        let model = Arc::new(MockModel::new());
        let device = Device::Cuda(0);
        let backend = GpuBackend::new(model, device).unwrap();
        let capabilities = backend.capabilities();

        assert!(capabilities.supports_mixed_precision);
        assert!(capabilities.supports_batching);
        assert!(capabilities.max_batch_size > 1);
        assert!(!capabilities.memory_efficient); // GPU uses more memory
    }

    #[test]
    fn test_backend_selection_cpu() {
        let model = Arc::new(MockModel::new());
        let backend = select_backend(model, Some(Device::Cpu));
        assert!(backend.is_ok());
        assert_eq!(backend.unwrap().backend_type(), "cpu");
    }

    #[test]
    fn test_backend_selection_auto() {
        let model = Arc::new(MockModel::new());
        let backend = select_backend(model, None);
        assert!(backend.is_ok());

        let backend_type = backend.unwrap().backend_type();
        assert!(backend_type == "cpu" || backend_type.starts_with("gpu_"));
    }

    #[test]
    fn test_backend_selection_gpu_fallback() {
        let model = Arc::new(MockModel::new());
        let backend = select_backend(model, Some(Device::Cuda(0)));
        assert!(backend.is_ok());

        // Should either be GPU or fallback to CPU
        let backend_type = backend.unwrap().backend_type();
        assert!(backend_type == "cpu" || backend_type.starts_with("gpu_"));
    }

    #[test]
    fn test_backend_capabilities_default() {
        let capabilities = BackendCapabilities::default();
        assert!(!capabilities.supports_mixed_precision);
        assert!(capabilities.supports_batching);
        assert_eq!(capabilities.max_batch_size, 1);
        assert!(capabilities.memory_efficient);
    }
}

mod streaming_unit_tests {
    use bitnet_inference::streaming::StreamingConfig;

    #[test]
    fn test_streaming_config_creation() {
        let config = StreamingConfig { buffer_size: 10, flush_interval_ms: 50 };

        assert_eq!(config.buffer_size, 10);
        assert_eq!(config.flush_interval_ms, 50);
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert!(config.buffer_size > 0);
        assert!(config.flush_interval_ms > 0);
    }

    #[test]
    fn test_streaming_config_validation() {
        // Test various buffer sizes
        let config = StreamingConfig { buffer_size: 1, flush_interval_ms: 1 };
        assert_eq!(config.buffer_size, 1);
        assert_eq!(config.flush_interval_ms, 1);

        let config = StreamingConfig { buffer_size: 1000, flush_interval_ms: 1000 };
        assert_eq!(config.buffer_size, 1000);
        assert_eq!(config.flush_interval_ms, 1000);
    }
}

mod error_handling_unit_tests {
    use super::*;

    #[test]
    fn test_config_validation_errors() {
        let mut config = InferenceConfig::default();

        config.max_context_length = 0;
        let error = config.validate().unwrap_err();
        assert!(error.contains("max_context_length"));

        config.max_context_length = 2048;
        config.num_threads = 0;
        let error = config.validate().unwrap_err();
        assert!(error.contains("num_threads"));

        config.num_threads = 4;
        config.batch_size = 0;
        let error = config.validate().unwrap_err();
        assert!(error.contains("batch_size"));

        config.batch_size = 1;
        config.memory_pool_size = 0;
        let error = config.validate().unwrap_err();
        assert!(error.contains("memory_pool_size"));
    }

    #[test]
    fn test_generation_config_validation_errors() {
        let mut config = GenerationConfig::default();

        config.max_new_tokens = 0;
        let error = config.validate().unwrap_err();
        assert!(error.contains("max_new_tokens"));

        config.max_new_tokens = 100;
        config.temperature = -1.0;
        let error = config.validate().unwrap_err();
        assert!(error.contains("temperature"));

        config.temperature = 0.7;
        config.top_p = 0.0;
        let error = config.validate().unwrap_err();
        assert!(error.contains("top_p"));

        config.top_p = 1.5;
        let error = config.validate().unwrap_err();
        assert!(error.contains("top_p"));

        config.top_p = 0.9;
        config.repetition_penalty = 0.0;
        let error = config.validate().unwrap_err();
        assert!(error.contains("repetition_penalty"));
    }

    #[tokio::test]
    async fn test_backend_error_handling() {
        // Test invalid device for GPU backend
        let model = Arc::new(MockModel::new());
        let invalid_device = Device::Cpu; // CPU device for GPU backend

        // This should work since we're using CPU device
        let _backend = GpuBackend::new(model.clone(), invalid_device);
        // Actually, this might fail because GPU backend expects CUDA device
        // The behavior depends on implementation
    }

    #[test]
    fn test_error_message_quality() {
        let mut config = InferenceConfig::default();
        config.max_context_length = 0;

        let error = config.validate().unwrap_err();
        assert!(!error.is_empty());
        assert!(error.contains("max_context_length"));
        assert!(error.contains("greater than 0"));
    }
}

mod integration_unit_tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_with_different_configs() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());

        // Test with CPU optimized config
        let cpu_config = InferenceConfig::cpu_optimized();
        let engine =
            InferenceEngine::with_config(model.clone(), tokenizer.clone(), Device::Cpu, cpu_config);
        assert!(engine.is_ok());

        // Test with memory efficient config
        let memory_config = InferenceConfig::memory_efficient();
        let engine = InferenceEngine::with_config(
            model.clone(),
            tokenizer.clone(),
            Device::Cpu,
            memory_config,
        );
        assert!(engine.is_ok());

        // Test with GPU optimized config (should fallback to CPU if GPU not available)
        let gpu_config = InferenceConfig::gpu_optimized();
        let engine = InferenceEngine::with_config(model, tokenizer, Device::Cuda(0), gpu_config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_generation_with_different_configs() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        // Test greedy generation
        let greedy_config = GenerationConfig::greedy();
        let result = engine.generate_with_config("Test", &greedy_config).await;
        assert!(result.is_ok());

        // Test creative generation
        let creative_config = GenerationConfig::creative();
        let result = engine.generate_with_config("Test", &creative_config).await;
        assert!(result.is_ok());

        // Test balanced generation
        let balanced_config = GenerationConfig::balanced();
        let result = engine.generate_with_config("Test", &balanced_config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cache_integration() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        // Generate text to populate cache
        let _ = engine.generate("First prompt").await;
        let stats_after_first = engine.get_stats().await;

        // Generate more text
        let _ = engine.generate("Second prompt").await;
        let stats_after_second = engine.get_stats().await;

        // Cache usage should be tracked
        assert!(stats_after_first.cache_usage >= 0.0);
        assert!(stats_after_second.cache_usage >= 0.0);

        // Clear cache
        engine.clear_cache().await;
        let stats_after_clear = engine.get_stats().await;

        // Cache should be cleared
        assert!(stats_after_clear.cache_size <= stats_after_second.cache_size);
    }
}
