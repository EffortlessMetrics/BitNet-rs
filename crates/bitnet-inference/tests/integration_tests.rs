//! Integration tests for bitnet-inference crate
//!
//! These tests validate the complete inference pipeline including:
//! - Engine initialization and configuration
//! - Text generation and streaming
//! - Batch processing
//! - Performance and resource management
//! - Error handling and edge cases

#![cfg(feature = "integration-tests")]

use bitnet_common::{BitNetConfig, BitNetError, ConcreteTensor, Device, InferenceError};
use bitnet_inference::prelude::*;
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use futures_util::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

// Mock implementations for testing

struct MockModel {
    config: BitNetConfig,
    should_fail: bool,
}

impl MockModel {
    fn new() -> Self {
        Self { config: BitNetConfig::default(), should_fail: false }
    }

    fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
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
        if self.should_fail {
            return Err(BitNetError::Inference(InferenceError::GenerationFailed {
                reason: "Mock model failure".to_string(),
            }));
        }
        Ok(ConcreteTensor::mock(vec![1, 50257]))
    }

    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor, BitNetError> {
        if self.should_fail {
            return Err(BitNetError::Inference(InferenceError::GenerationFailed {
                reason: "Mock embed failure".to_string(),
            }));
        }
        // Create a mock embedding tensor with shape [seq_len, hidden_dim]
        let seq_len = tokens.len();
        let hidden_dim = self.config.model.hidden_size;
        Ok(ConcreteTensor::mock(vec![seq_len, hidden_dim]))
    }

    fn logits(&self, _hidden: &ConcreteTensor) -> Result<ConcreteTensor, BitNetError> {
        if self.should_fail {
            return Err(BitNetError::Inference(InferenceError::GenerationFailed {
                reason: "Mock logits failure".to_string(),
            }));
        }
        // Create a mock logits tensor with shape [batch, vocab_size]
        Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
    }
}

struct MockTokenizer {
    vocab_size: usize,
    should_fail: bool,
}

impl MockTokenizer {
    fn new() -> Self {
        Self { vocab_size: 50257, should_fail: false }
    }

    fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> Result<Vec<u32>, BitNetError> {
        if self.should_fail {
            return Err(BitNetError::Inference(
                bitnet_common::InferenceError::TokenizationFailed {
                    reason: "Mock tokenizer failure".to_string(),
                },
            ));
        }
        // Simple mock encoding: convert text length to tokens
        Ok((0..text.len().min(10)).map(|i| i as u32 + 1).collect())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, BitNetError> {
        if self.should_fail {
            return Err(BitNetError::Inference(
                bitnet_common::InferenceError::TokenizationFailed {
                    reason: "Mock decode failure".to_string(),
                },
            ));
        }
        Ok(format!("decoded_{}_tokens", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(50256)
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(50257)
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        Some(format!("<token_{}>", token))
    }
}

// Test modules

mod engine_tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation_cpu() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device);
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        let stats = engine.get_stats().await;
        assert_eq!(stats.backend_type, "cpu");
    }

    #[tokio::test]
    async fn test_engine_creation_with_config() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let config = InferenceConfig::default()
            .with_threads(4)
            .with_batch_size(2)
            .with_mixed_precision(false);

        let engine = InferenceEngine::with_config(model, tokenizer, device, config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_engine_creation_gpu_fallback() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cuda(0);

        // Should fallback to CPU if GPU not available
        let engine = InferenceEngine::new(model, tokenizer, device);
        // Engine creation should succeed even if GPU is not available (fallback to CPU)
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_basic_text_generation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let result = engine.generate("Hello, world!").await;

        assert!(result.is_ok());
        let generated_text = result.unwrap();
        assert!(!generated_text.is_empty());
        assert!(generated_text.starts_with("decoded_"));
    }

    #[tokio::test]
    async fn test_text_generation_with_config() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let config = GenerationConfig::default()
            .with_max_tokens(50)
            .with_temperature(0.8)
            .with_top_k(40)
            .with_seed(42);

        let result = engine.generate_with_config("Test prompt", &config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_generation_with_stop_sequences() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let config =
            GenerationConfig::default().with_stop_sequence("</s>".to_string()).with_max_tokens(100);

        let result = engine.generate_with_config("Test prompt", &config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_model_config_access() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let model_config = engine.model_config();

        // Should be able to access model configuration
        assert!(model_config.model.vocab_size > 0);
    }

    #[tokio::test]
    async fn test_cache_management() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        // Generate some text to populate cache
        let _ = engine.generate("Test prompt").await;

        let stats_before = engine.get_stats().await;

        // Clear cache
        engine.clear_cache().await;

        let stats_after = engine.get_stats().await;

        // Cache should be cleared (size might be 0 or reset)
        assert!(stats_after.cache_size <= stats_before.cache_size);
    }

    #[tokio::test]
    async fn test_engine_error_handling() {
        let model = Arc::new(MockModel::new().with_failure());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let result = engine.generate("Test prompt").await;

        // Should handle model failures gracefully
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tokenizer_error_handling() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new().with_failure());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let result = engine.generate("Test prompt").await;

        // Should handle tokenizer failures gracefully
        assert!(result.is_err());
    }
}

mod streaming_tests {
    use super::*;

    #[tokio::test]
    async fn test_streaming_generation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let mut stream = engine.generate_stream("Hello, world!").unwrap();

        let mut token_count = 0;
        let mut total_text = String::new();

        // Collect tokens with timeout to prevent hanging
        while let Ok(Some(result)) = timeout(Duration::from_secs(5), stream.next()).await {
            match result {
                Ok(text_chunk) => {
                    assert!(!text_chunk.text.is_empty());
                    total_text.push_str(&text_chunk.text);
                    token_count += 1;
                }
                Err(e) => {
                    panic!("Stream error: {}", e);
                }
            }

            // Prevent infinite loop in test
            if token_count >= 5 {
                break;
            }
        }

        assert!(token_count > 0);
        assert!(!total_text.is_empty());
    }

    #[tokio::test]
    async fn test_streaming_with_config() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let config = GenerationConfig::default().with_max_tokens(10).with_temperature(0.5);

        let mut stream = engine.generate_stream_with_config("Test prompt", &config).unwrap();

        let mut received_tokens = 0;
        while let Ok(Some(result)) = timeout(Duration::from_secs(3), stream.next()).await {
            match result {
                Ok(_text_chunk) => {
                    received_tokens += 1;
                }
                Err(_) => break,
            }

            if received_tokens >= 5 {
                break;
            }
        }

        assert!(received_tokens > 0);
    }

    #[tokio::test]
    async fn test_streaming_early_termination() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let mut stream = engine.generate_stream("Test prompt").unwrap();

        // Take only first few tokens then drop stream
        let first_result = timeout(Duration::from_secs(2), stream.next()).await;
        assert!(first_result.is_ok());

        // Drop the stream (simulating client disconnect)
        drop(stream);

        // Test should complete without hanging
    }

    #[tokio::test]
    async fn test_streaming_config_validation() {
        let config = StreamingConfig {
            buffer_size: 5,
            flush_interval_ms: 100,
            cancellable: true,
            max_retries: 3,
            token_timeout_ms: 1000,
        };

        assert_eq!(config.buffer_size, 5);
        assert_eq!(config.flush_interval_ms, 100);

        let default_config = StreamingConfig::default();
        assert!(default_config.buffer_size > 0);
        assert!(default_config.flush_interval_ms > 0);
    }
}

mod config_tests {
    use super::*;

    #[test]
    fn test_inference_config_defaults() {
        let config = InferenceConfig::default();
        assert!(config.max_context_length > 0);
        assert!(config.num_threads > 0);
        assert!(config.batch_size > 0);
        assert!(config.memory_pool_size > 0);
    }

    #[test]
    fn test_inference_config_presets() {
        let cpu_config = InferenceConfig::cpu_optimized();
        assert!(!cpu_config.mixed_precision);
        assert_eq!(cpu_config.batch_size, 1);

        let gpu_config = InferenceConfig::gpu_optimized();
        assert!(gpu_config.mixed_precision);
        assert!(gpu_config.batch_size > 1);

        let memory_config = InferenceConfig::memory_efficient();
        assert!(memory_config.max_context_length <= InferenceConfig::default().max_context_length);
        assert!(memory_config.memory_pool_size <= InferenceConfig::default().memory_pool_size);
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
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert!(config.max_new_tokens > 0);
        assert!(config.temperature >= 0.0);
        assert!(config.top_p > 0.0 && config.top_p <= 1.0);
        assert!(config.repetition_penalty > 0.0);
    }

    #[test]
    fn test_generation_config_presets() {
        let greedy = GenerationConfig::greedy();
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(greedy.top_k, 1);

        let creative = GenerationConfig::creative();
        assert!(creative.temperature > GenerationConfig::default().temperature);
        assert!(creative.top_k > GenerationConfig::default().top_k);

        let balanced = GenerationConfig::balanced();
        assert!(balanced.repetition_penalty > 1.0);
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

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_inference_performance_metrics() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let start_time = Instant::now();
        let result = engine.generate("Performance test prompt").await;
        let duration = start_time.elapsed();

        assert!(result.is_ok());
        assert!(duration.as_millis() < 5000); // Should complete within 5 seconds

        let stats = engine.get_stats().await;
        
        // Validate meaningful stat invariants
        assert!(stats.cache_usage.is_finite(), "cache_usage should be finite");
        assert!(
            (0.0..=100.0).contains(&stats.cache_usage),
            "cache_usage should be a percentage [0,100] (got {})",
            stats.cache_usage
        );
        assert!(!stats.backend_type.is_empty(), "backend_type should be specified");
    }

    #[tokio::test]
    async fn test_concurrent_inference() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = Arc::new(InferenceEngine::new(model, tokenizer, device).unwrap());

        let mut handles = Vec::new();

        // Launch multiple concurrent inference tasks
        for i in 0..5 {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                let prompt = format!("Concurrent test prompt {}", i);
                engine_clone.generate(&prompt).await
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut success_count = 0;
        for handle in handles {
            if let Ok(result) = handle.await
                && result.is_ok() {
                    success_count += 1;
                }
        }

        // At least some should succeed
        assert!(success_count > 0);
    }

    #[tokio::test]
    async fn test_memory_usage_monitoring() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let _stats_before = engine.get_stats().await;

        // Generate text to use memory
        let _ = engine.generate("Memory usage test prompt").await;

        let stats_after = engine.get_stats().await;

        // Memory usage should be tracked
        assert!(stats_after.cache_usage >= 0.0);
        assert!(stats_after.cache_usage <= 100.0);
    }

    #[tokio::test]
    async fn test_resource_cleanup() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        // Generate text to allocate resources
        let _ = engine.generate("Resource cleanup test").await;

        let stats_before = engine.get_stats().await;

        // Clear cache to free resources
        engine.clear_cache().await;

        let stats_after = engine.get_stats().await;

        // Resources should be cleaned up
        assert!(stats_after.cache_size <= stats_before.cache_size);
    }

    #[tokio::test]
    async fn test_throughput_measurement() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = Arc::new(InferenceEngine::new(model, tokenizer, device).unwrap());

        let start_time = Instant::now();
        let num_requests = 10;

        let mut handles = Vec::new();
        for i in 0..num_requests {
            let engine_clone = Arc::clone(&engine);
            let handle = tokio::spawn(async move {
                let prompt = format!("Throughput test {}", i);
                engine_clone.generate(&prompt).await
            });
            handles.push(handle);
        }

        let mut completed = 0;
        for handle in handles {
            if let Ok(result) = handle.await
                && result.is_ok() {
                    completed += 1;
                }
        }

        let total_time = start_time.elapsed();
        let throughput = completed as f64 / total_time.as_secs_f64();

        assert!(completed > 0);
        assert!(throughput > 0.0);
        println!("Throughput: {:.2} requests/second", throughput);
    }
}

mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_model_failure_handling() {
        let model = Arc::new(MockModel::new().with_failure());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let result = engine.generate("Test prompt").await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Mock model failure"));
    }

    #[tokio::test]
    async fn test_tokenizer_failure_handling() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new().with_failure());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let result = engine.generate("Test prompt").await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Mock tokenizer failure"));
    }

    #[tokio::test]
    async fn test_empty_prompt_handling() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let result = engine.generate("").await;

        // Should handle empty prompts gracefully
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_very_long_prompt_handling() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        // Create a very long prompt
        let long_prompt = "a".repeat(10000);
        let result = engine.generate(&long_prompt).await;

        // Should handle long prompts gracefully
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_invalid_generation_config() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let _engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let invalid_config = GenerationConfig {
            max_new_tokens: 0,
            ..GenerationConfig::default()
        };

        // Should validate config before generation
        assert!(invalid_config.validate().is_err());
    }

    #[tokio::test]
    async fn test_streaming_error_propagation() {
        let model = Arc::new(MockModel::new().with_failure());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();
        let mut stream = engine.generate_stream("Test prompt")
            .expect("generation stream should be created");

        // Should propagate model errors through stream
        let maybe = timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("stream next() should not timeout");
        
        if let Some(result) = maybe {
            // The result should be an error due to mock failure
            assert!(result.is_err(), "stream should yield error from mock model failure");
        } else {
            panic!("stream should yield at least one item (even if it's an error)");
        }
    }
}

mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_unicode_prompt_handling() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let unicode_prompt = "Hello 世界! 🌍 Здравствуй мир!";
        let result = engine.generate(unicode_prompt).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_special_characters_prompt() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let special_prompt = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~";
        let result = engine.generate(special_prompt).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_zero_temperature_generation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let config = GenerationConfig::default().with_temperature(0.0);
        let result = engine.generate_with_config("Test prompt", &config).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_high_temperature_generation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let config = GenerationConfig::default().with_temperature(2.0);
        let result = engine.generate_with_config("Test prompt", &config).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_single_token_generation() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let config = GenerationConfig::default().with_max_tokens(1);
        let result = engine.generate_with_config("Test prompt", &config).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_max_context_length_handling() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let config = InferenceConfig::default().with_memory_pool_size(1024); // Very small
        let engine = InferenceEngine::with_config(model, tokenizer, device, config).unwrap();

        // Generate with a config that might exceed context length
        let gen_config = GenerationConfig::default().with_max_tokens(1000);
        let result = engine.generate_with_config("Test prompt", &gen_config).await;

        // Should handle context length limits gracefully
        assert!(result.is_ok());
    }
}
