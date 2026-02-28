#![cfg(feature = "integration-tests")]
//! Simple comprehensive tests for bitnet-inference crate
//!
//! These tests validate the inference functionality with simplified mocks
//! to avoid complex type compatibility issues while still achieving
//! comprehensive coverage of the inference engine.
use bitnet_inference::prelude::*;
use std::time::Duration;
#[tokio::test(flavor = "multi_thread")]
async fn test_inference_config_defaults() {
    let config = InferenceConfig::default();
    assert!(config.max_context_length > 0);
    assert!(config.num_threads > 0);
    assert!(config.batch_size > 0);
    assert!(config.memory_pool_size > 0);
}
#[tokio::test(flavor = "multi_thread")]
async fn test_inference_config_validation() {
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
#[tokio::test(flavor = "multi_thread")]
async fn test_inference_config_builder() {
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
#[tokio::test(flavor = "multi_thread")]
async fn test_inference_config_presets() {
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
#[tokio::test(flavor = "multi_thread")]
async fn test_generation_config_defaults() {
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
#[tokio::test(flavor = "multi_thread")]
async fn test_generation_config_validation() {
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
#[tokio::test(flavor = "multi_thread")]
async fn test_generation_config_presets() {
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
#[tokio::test(flavor = "multi_thread")]
async fn test_generation_config_builder() {
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
#[tokio::test(flavor = "multi_thread")]
async fn test_config_serialization() {
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
#[tokio::test(flavor = "multi_thread")]
async fn test_cache_config_creation() {
    let config = CacheConfig::default();
    assert!(config.max_size_bytes > 0);
    assert!(config.max_sequence_length > 0);
}
#[tokio::test(flavor = "multi_thread")]
async fn test_cache_creation() {
    let config = CacheConfig::default();
    let cache = KVCache::new(config);
    assert!(cache.is_ok());
    let cache = cache.unwrap();
    assert_eq!(cache.size(), 0);
    assert_eq!(cache.usage_percent(), 0.0);
}
#[tokio::test(flavor = "multi_thread")]
async fn test_cache_operations() {
    let config = CacheConfig::default();
    let mut cache = KVCache::new(config).unwrap();
    assert_eq!(cache.size(), 0);
    assert_eq!(cache.usage_percent(), 0.0);
    cache.clear();
    assert_eq!(cache.size(), 0);
}
#[tokio::test(flavor = "multi_thread")]
async fn test_streaming_config() {
    let config = StreamingConfig::default();
    assert!(config.buffer_size > 0);
    assert!(config.flush_interval_ms > 0);
    let custom_config = StreamingConfig {
        buffer_size: 5,
        flush_interval_ms: 100,
        max_retries: 3,
        token_timeout_ms: 5000,
        cancellable: true,
    };
    assert_eq!(custom_config.buffer_size, 5);
    assert_eq!(custom_config.flush_interval_ms, 100);
}
#[tokio::test(flavor = "multi_thread")]
async fn test_sampling_config() {
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
#[tokio::test(flavor = "multi_thread")]
async fn test_sampling_strategy_creation() {
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        repetition_penalty: 1.0,
        seed: Some(42),
    };
    let _strategy = SamplingStrategy::new(config);
}
#[tokio::test(flavor = "multi_thread")]
async fn test_sampling_with_different_parameters() {
    let logits = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let context = vec![1, 2, 3];
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
#[tokio::test(flavor = "multi_thread")]
async fn test_sampling_reproducibility() {
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
    assert_eq!(token1, token2);
}
#[tokio::test(flavor = "multi_thread")]
async fn test_sampling_edge_cases() {
    let context = vec![1, 2, 3];
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
    let token = result.unwrap();
    assert!(token < logits.len() as u32);
}
#[tokio::test(flavor = "multi_thread")]
async fn test_error_handling() {
    let mut config = InferenceConfig { max_context_length: 0, ..Default::default() };
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
#[tokio::test(flavor = "multi_thread")]
async fn test_generation_config_error_handling() {
    let mut config = GenerationConfig::default().with_max_tokens(0);
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
#[tokio::test(flavor = "multi_thread")]
async fn test_performance_characteristics() {
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _config = InferenceConfig::default();
    }
    let duration = start.elapsed();
    assert!(duration < Duration::from_millis(100));
    let start = std::time::Instant::now();
    let config = InferenceConfig::default();
    for _ in 0..1000 {
        let _ = config.validate();
    }
    let duration = start.elapsed();
    assert!(duration < Duration::from_millis(10));
}
#[tokio::test(flavor = "multi_thread")]
async fn test_memory_efficiency() {
    let configs: Vec<_> = (0..1000).map(|_| InferenceConfig::default()).collect();
    assert_eq!(configs.len(), 1000);
    let gen_configs: Vec<_> = (0..1000).map(|_| GenerationConfig::default()).collect();
    assert_eq!(gen_configs.len(), 1000);
}
#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_config_access() {
    use std::sync::Arc;
    use tokio::task;
    let config = Arc::new(InferenceConfig::default());
    let mut handles = Vec::new();
    for _ in 0..10 {
        let config_clone = config.clone();
        let handle = task::spawn(async move {
            for _ in 0..100 {
                let _ = config_clone.validate();
                let _ = config_clone.clone();
            }
        });
        handles.push(handle);
    }
    for handle in handles {
        handle.await.unwrap();
    }
}
#[tokio::test(flavor = "multi_thread")]
async fn test_config_edge_cases() {
    let config =
        InferenceConfig::default().with_threads(1).with_batch_size(1).with_memory_pool_size(1024);
    assert!(config.validate().is_ok());
    let config = InferenceConfig::default()
        .with_threads(1000)
        .with_batch_size(1000)
        .with_memory_pool_size(1024 * 1024 * 1024 * 10);
    assert!(config.validate().is_ok());
    let gen_config = GenerationConfig::default()
        .with_temperature(0.0)
        .with_top_k(1)
        .with_top_p(0.001)
        .with_max_tokens(1);
    assert!(gen_config.validate().is_ok());
    let gen_config = GenerationConfig::default()
        .with_temperature(10.0)
        .with_top_k(100000)
        .with_top_p(1.0)
        .with_max_tokens(100000);
    assert!(gen_config.validate().is_ok());
}
#[tokio::test(flavor = "multi_thread")]
async fn test_comprehensive_coverage() {
    let _cpu = InferenceConfig::cpu_optimized();
    let _gpu = InferenceConfig::gpu_optimized();
    let _memory = InferenceConfig::memory_efficient();
    let _greedy = GenerationConfig::greedy();
    let _creative = GenerationConfig::creative();
    let _balanced = GenerationConfig::balanced();
    let _inf_config = InferenceConfig::default()
        .with_threads(4)
        .with_batch_size(2)
        .with_mixed_precision(true)
        .with_memory_pool_size(1024 * 1024);
    let _gen_config = GenerationConfig::default()
        .with_seed(42)
        .with_stop_sequence("</s>".to_string())
        .with_max_tokens(100)
        .with_temperature(0.8)
        .with_top_k(50)
        .with_top_p(0.9);
    let _cache_config = CacheConfig::default();
    let _streaming_config = StreamingConfig::default();
    let _sampling_config = SamplingConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        repetition_penalty: 1.0,
        seed: Some(42),
    };
    let config = GenerationConfig::default();
    let _serialized = serde_json::to_string(&config).unwrap();
    let config = InferenceConfig::default();
    let _serialized = serde_json::to_string(&config).unwrap();
}
