//! Performance and benchmarking tests for bitnet-inference
//!
//! These tests validate performance characteristics and resource usage:
//! - Inference latency and throughput
//! - Memory usage and resource management
//! - Concurrent processing performance
//! - Cache efficiency and optimization
//! - Backend performance comparison

#![cfg(feature = "integration-tests")]
#![allow(dead_code)] // Test utilities may not be used in all test configurations

use bitnet_common::{BitNetConfig, BitNetError, ConcreteTensor, Device};
use bitnet_inference::prelude::*;
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;

// Mock implementations optimized for performance testing

struct MockModel {
    config: BitNetConfig,
    processing_delay: Duration,
    memory_usage: usize,
}

impl MockModel {
    fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            processing_delay: Duration::from_millis(1),
            memory_usage: 1024 * 1024, // 1MB
        }
    }

    fn with_delay(mut self, delay: Duration) -> Self {
        self.processing_delay = delay;
        self
    }

    fn with_memory_usage(mut self, memory: usize) -> Self {
        self.memory_usage = memory;
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
        // Simulate processing time
        std::thread::sleep(self.processing_delay);

        // Simulate memory allocation
        let _memory_simulation = vec![0u8; self.memory_usage];

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

struct MockTokenizer {
    processing_delay: Duration,
}

impl MockTokenizer {
    fn new() -> Self {
        Self { processing_delay: Duration::from_micros(100) }
    }

    fn with_delay(mut self, delay: Duration) -> Self {
        self.processing_delay = delay;
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
        std::thread::sleep(self.processing_delay);
        Ok((0..text.len().min(100)).map(|i| i as u32 + 1).collect())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, BitNetError> {
        std::thread::sleep(self.processing_delay);
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

    fn token_to_piece(&self, token: u32) -> Option<String> {
        Some(format!("<token_{}>", token))
    }
}

// Performance measurement utilities

struct PerformanceMetrics {
    latency: Duration,
    throughput: f64,
    memory_peak: usize,
    cpu_usage: f64,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self { latency: Duration::from_millis(0), throughput: 0.0, memory_peak: 0, cpu_usage: 0.0 }
    }
}

struct PerformanceMeasurer {
    start_time: Instant,
    request_count: usize,
}

impl PerformanceMeasurer {
    fn new() -> Self {
        Self { start_time: Instant::now(), request_count: 0 }
    }

    fn record_request(&mut self) {
        self.request_count += 1;
    }

    fn get_metrics(&self) -> PerformanceMetrics {
        let elapsed = self.start_time.elapsed();
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            self.request_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        PerformanceMetrics {
            latency: elapsed,
            throughput,
            memory_peak: 0, // Would need actual memory monitoring
            cpu_usage: 0.0, // Would need actual CPU monitoring
        }
    }
}

mod latency_tests {
    use super::*;

    async fn create_test_engine() -> InferenceEngine {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;
        InferenceEngine::new(model, tokenizer, device).unwrap()
    }

    #[tokio::test]
    async fn test_single_inference_latency() {
        let engine = create_test_engine().await;

        let start_time = Instant::now();
        let result = engine.generate("Test prompt for latency measurement").await;
        let latency = start_time.elapsed();

        assert!(result.is_ok());
        assert!(latency < Duration::from_millis(100)); // Should be fast with mock

        println!("Single inference latency: {:?}", latency);
    }

    #[tokio::test]
    async fn test_inference_latency_with_different_prompt_lengths() {
        let engine = create_test_engine().await;

        let prompt_lengths = [10, 50, 100, 500, 1000];

        for length in prompt_lengths {
            let prompt = "a".repeat(length);

            let start_time = Instant::now();
            let result = engine.generate(&prompt).await;
            let latency = start_time.elapsed();

            assert!(result.is_ok());
            println!("Prompt length {}: latency {:?}", length, latency);

            // Latency should be reasonable regardless of prompt length
            assert!(latency < Duration::from_millis(200));
        }
    }

    #[tokio::test]
    async fn test_inference_latency_with_different_generation_lengths() {
        let engine = create_test_engine().await;

        let generation_lengths = [1, 10, 50, 100];

        for max_tokens in generation_lengths {
            let config = GenerationConfig::default().with_max_tokens(max_tokens);

            let start_time = Instant::now();
            let result = engine.generate_with_config("Test prompt", &config).await;
            let latency = start_time.elapsed();

            assert!(result.is_ok());
            println!("Max tokens {}: latency {:?}", max_tokens, latency);

            // Latency should scale reasonably with generation length
            let expected_max_latency = Duration::from_millis(10 + max_tokens as u64 * 2);
            assert!(latency < expected_max_latency);
        }
    }

    #[tokio::test]
    async fn test_first_token_latency() {
        let engine = create_test_engine().await;

        // Measure time to first token in streaming
        let start_time = Instant::now();
        let mut stream = engine.generate_stream("Test prompt for first token latency").unwrap();

        if let Ok(Some(result)) = timeout(Duration::from_secs(1), stream.next()).await {
            let first_token_latency = start_time.elapsed();

            match result {
                Ok(_) => {
                    println!("First token latency: {:?}", first_token_latency);
                    assert!(first_token_latency < Duration::from_millis(50));
                }
                Err(e) => panic!("First token generation failed: {}", e),
            }
        } else {
            panic!("First token generation timed out");
        }
    }

    #[tokio::test]
    async fn test_latency_consistency() {
        let engine = create_test_engine().await;
        let num_runs = 10;
        let mut latencies = Vec::new();

        for i in 0..num_runs {
            let prompt = format!("Consistency test run {}", i);

            let start_time = Instant::now();
            let result = engine.generate(&prompt).await;
            let latency = start_time.elapsed();

            assert!(result.is_ok());
            latencies.push(latency);
        }

        // Calculate statistics
        let avg_latency = latencies.iter().sum::<Duration>() / num_runs as u32;
        let min_latency = latencies.iter().min().unwrap();
        let max_latency = latencies.iter().max().unwrap();

        println!(
            "Latency stats - Avg: {:?}, Min: {:?}, Max: {:?}",
            avg_latency, min_latency, max_latency
        );

        // Latency should be reasonably consistent
        let latency_variance = max_latency.as_millis() - min_latency.as_millis();
        assert!(latency_variance < 50); // Less than 50ms variance
    }
}

mod throughput_tests {
    use super::*;
    use futures_util::StreamExt;

    async fn create_test_engine() -> InferenceEngine {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;
        InferenceEngine::new(model, tokenizer, device).unwrap()
    }

    #[tokio::test]
    async fn test_sequential_throughput() {
        let engine = create_test_engine().await;
        let num_requests = 20;

        let start_time = Instant::now();

        for i in 0..num_requests {
            let prompt = format!("Sequential throughput test {}", i);
            let result = engine.generate(&prompt).await;
            assert!(result.is_ok());
        }

        let total_time = start_time.elapsed();
        let throughput = num_requests as f64 / total_time.as_secs_f64();

        println!("Sequential throughput: {:.2} requests/second", throughput);
        assert!(throughput > 0.0);
        assert!(throughput < 1000.0); // Sanity check
    }

    #[tokio::test]
    async fn test_concurrent_throughput() {
        let engine = Arc::new(create_test_engine().await);
        let num_requests = 20;
        let concurrency = 5;

        let start_time = Instant::now();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut handles = Vec::new();

        for i in 0..num_requests {
            let engine_clone = engine.clone();
            let permit = semaphore.clone().acquire_owned().await.unwrap();

            let handle = tokio::spawn(async move {
                let _permit = permit;
                let prompt = format!("Concurrent throughput test {}", i);
                engine_clone.generate(&prompt).await
            });

            handles.push(handle);
        }

        let mut successful_requests = 0;
        for handle in handles {
            if let Ok(result) = handle.await {
                if result.is_ok() {
                    successful_requests += 1;
                }
            }
        }

        let total_time = start_time.elapsed();
        let throughput = successful_requests as f64 / total_time.as_secs_f64();

        println!("Concurrent throughput: {:.2} requests/second", throughput);
        assert_eq!(successful_requests, num_requests);
        assert!(throughput > 0.0);
    }

    #[tokio::test]
    async fn test_streaming_throughput() {
        let engine = create_test_engine().await;
        let config = GenerationConfig::default().with_max_tokens(20);

        let start_time = Instant::now();
        let mut stream =
            engine.generate_stream_with_config("Streaming throughput test", &config).unwrap();

        let mut token_count = 0;
        while let Ok(Some(result)) = timeout(Duration::from_secs(5), stream.next()).await {
            match result {
                Ok(_token_text) => {
                    token_count += 1;
                }
                Err(_) => break,
            }

            if token_count >= 15 {
                break;
            }
        }

        let total_time = start_time.elapsed();
        let tokens_per_second = token_count as f64 / total_time.as_secs_f64();

        println!("Streaming throughput: {:.2} tokens/second", tokens_per_second);
        assert!(token_count > 0);
        assert!(tokens_per_second > 0.0);
    }

    #[tokio::test]
    async fn test_throughput_scaling() {
        let engine = Arc::new(create_test_engine().await);
        let base_requests = 10;

        for concurrency in [1, 2, 4, 8] {
            let start_time = Instant::now();
            let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
            let mut handles = Vec::new();

            for i in 0..base_requests {
                let engine_clone = engine.clone();
                let permit = semaphore.clone().acquire_owned().await.unwrap();

                let handle = tokio::spawn(async move {
                    let _permit = permit;
                    let prompt = format!("Scaling test {} with concurrency {}", i, concurrency);
                    engine_clone.generate(&prompt).await
                });

                handles.push(handle);
            }

            let mut successful_requests = 0;
            for handle in handles {
                if let Ok(result) = handle.await {
                    if result.is_ok() {
                        successful_requests += 1;
                    }
                }
            }

            let total_time = start_time.elapsed();
            let throughput = successful_requests as f64 / total_time.as_secs_f64();

            println!("Concurrency {}: {:.2} requests/second", concurrency, throughput);
            assert_eq!(successful_requests, base_requests);
        }
    }
}

mod memory_tests {
    use super::*;

    async fn create_test_engine() -> InferenceEngine {
        let model = Arc::new(MockModel::new().with_memory_usage(1024 * 1024)); // 1MB per forward pass
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;
        InferenceEngine::new(model, tokenizer, device).unwrap()
    }

    #[tokio::test]
    async fn test_memory_usage_single_inference() {
        let engine = create_test_engine().await;

        // Get initial stats
        let stats_before = engine.get_stats().await;

        // Perform inference
        let result = engine.generate("Memory usage test").await;
        assert!(result.is_ok());

        // Get stats after inference
        let stats_after = engine.get_stats().await;

        // Memory usage should be tracked
        assert!(stats_after.cache_usage >= 0.0);
        assert!(stats_after.cache_usage <= 100.0);

        println!(
            "Memory usage - Before: {:.2}%, After: {:.2}%",
            stats_before.cache_usage, stats_after.cache_usage
        );
    }

    #[tokio::test]
    async fn test_memory_usage_multiple_inferences() {
        let engine = create_test_engine().await;
        let num_inferences = 5;

        let mut memory_usage_history = Vec::new();

        for i in 0..num_inferences {
            let prompt = format!("Memory test inference {}", i);
            let result = engine.generate(&prompt).await;
            assert!(result.is_ok());

            let stats = engine.get_stats().await;
            memory_usage_history.push(stats.cache_usage);

            println!("Inference {}: Memory usage {:.2}%", i, stats.cache_usage);
        }

        // Memory usage should be reasonable and not grow unboundedly
        for usage in &memory_usage_history {
            assert!(*usage >= 0.0);
            assert!(*usage <= 100.0);
        }
    }

    #[tokio::test]
    async fn test_memory_cleanup_after_cache_clear() {
        let engine = create_test_engine().await;

        // Perform several inferences to build up cache
        for i in 0..5 {
            let prompt = format!("Cache buildup test {}", i);
            let _ = engine.generate(&prompt).await;
        }

        let stats_before_clear = engine.get_stats().await;

        // Clear cache
        engine.clear_cache().await;

        let stats_after_clear = engine.get_stats().await;

        // Memory usage should decrease after cache clear
        assert!(stats_after_clear.cache_size <= stats_before_clear.cache_size);

        println!(
            "Memory before clear: {:.2}%, after clear: {:.2}%",
            stats_before_clear.cache_usage, stats_after_clear.cache_usage
        );
    }

    #[tokio::test]
    async fn test_memory_usage_with_different_context_lengths() {
        let context_lengths = [512, 1024, 2048];

        for context_length in context_lengths {
            let config = InferenceConfig::default().with_memory_pool_size(context_length * 1024); // Scale memory pool

            let model = Arc::new(MockModel::new());
            let tokenizer = Arc::new(MockTokenizer::new());
            let device = Device::Cpu;

            let engine = InferenceEngine::with_config(model, tokenizer, device, config).unwrap();

            let prompt = "a".repeat(context_length / 4); // Use part of context length
            let result = engine.generate(&prompt).await;
            assert!(result.is_ok());

            let stats = engine.get_stats().await;
            println!("Context length {}: Memory usage {:.2}%", context_length, stats.cache_usage);

            assert!(stats.cache_usage >= 0.0);
            assert!(stats.cache_usage <= 100.0);
        }
    }

    #[tokio::test]
    async fn test_memory_efficiency_with_batching() {
        let engine = Arc::new(create_test_engine().await);
        let num_concurrent = 4;

        // Test concurrent requests (simulating batching)
        let start_time = Instant::now();
        let mut handles = Vec::new();

        for i in 0..num_concurrent {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                let prompt = format!("Batch memory test {}", i);
                let result = engine_clone.generate(&prompt).await;
                let stats = engine_clone.get_stats().await;
                (result, stats)
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            if let Ok((result, stats)) = handle.await {
                assert!(result.is_ok());
                results.push(stats);
            }
        }

        let total_time = start_time.elapsed();

        // All requests should complete with reasonable memory usage
        assert_eq!(results.len(), num_concurrent);
        for stats in &results {
            assert!(stats.cache_usage >= 0.0);
            assert!(stats.cache_usage <= 100.0);
        }

        println!(
            "Batch processing completed in {:?} with {} concurrent requests",
            total_time, num_concurrent
        );
    }
}

mod cache_performance_tests {
    use super::*;

    async fn create_test_engine() -> InferenceEngine {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;
        InferenceEngine::new(model, tokenizer, device).unwrap()
    }

    #[tokio::test]
    async fn test_cache_hit_performance() {
        let engine = create_test_engine().await;
        let prompt = "Cache performance test prompt";

        // First inference (cache miss)
        let start_time = Instant::now();
        let result1 = engine.generate(prompt).await;
        let first_inference_time = start_time.elapsed();
        assert!(result1.is_ok());

        // Second inference with same prompt (potential cache hit)
        let start_time = Instant::now();
        let result2 = engine.generate(prompt).await;
        let second_inference_time = start_time.elapsed();
        assert!(result2.is_ok());

        println!(
            "First inference: {:?}, Second inference: {:?}",
            first_inference_time, second_inference_time
        );

        // Both should complete successfully
        // In a real implementation, second might be faster due to caching
        assert!(first_inference_time > Duration::from_millis(0));
        assert!(second_inference_time > Duration::from_millis(0));
    }

    #[tokio::test]
    async fn test_cache_efficiency_with_similar_prompts() {
        let engine = create_test_engine().await;
        let base_prompt = "This is a test prompt for cache efficiency";

        let mut inference_times = Vec::new();

        // Test with variations of the same prompt
        for i in 0..5 {
            let prompt = format!("{} - variation {}", base_prompt, i);

            let start_time = Instant::now();
            let result = engine.generate(&prompt).await;
            let inference_time = start_time.elapsed();

            assert!(result.is_ok());
            inference_times.push(inference_time);

            println!("Variation {}: {:?}", i, inference_time);
        }

        // All inferences should complete in reasonable time
        for time in &inference_times {
            assert!(*time < Duration::from_millis(100));
        }
    }

    #[tokio::test]
    async fn test_cache_size_impact_on_performance() {
        let cache_sizes = [1024, 4096, 16384]; // Different cache sizes

        for cache_size in cache_sizes {
            let config = InferenceConfig::default().with_memory_pool_size(cache_size);

            let model = Arc::new(MockModel::new());
            let tokenizer = Arc::new(MockTokenizer::new());
            let device = Device::Cpu;

            let engine = InferenceEngine::with_config(model, tokenizer, device, config).unwrap();

            // Perform multiple inferences to test cache behavior
            let start_time = Instant::now();
            for i in 0..10 {
                let prompt = format!("Cache size test {} with size {}", i, cache_size);
                let result = engine.generate(&prompt).await;
                assert!(result.is_ok());
            }
            let total_time = start_time.elapsed();

            let throughput = 10.0 / total_time.as_secs_f64();
            println!("Cache size {}: {:.2} requests/second", cache_size, throughput);

            assert!(throughput > 0.0);
        }
    }

    #[tokio::test]
    async fn test_cache_clear_performance_impact() {
        let engine = create_test_engine().await;

        // Build up cache with multiple inferences
        for i in 0..5 {
            let prompt = format!("Cache buildup {}", i);
            let _ = engine.generate(&prompt).await;
        }

        // Measure performance before cache clear
        let start_time = Instant::now();
        let result_before = engine.generate("Performance test before clear").await;
        let time_before_clear = start_time.elapsed();
        assert!(result_before.is_ok());

        // Clear cache
        let clear_start = Instant::now();
        engine.clear_cache().await;
        let clear_time = clear_start.elapsed();

        // Measure performance after cache clear
        let start_time = Instant::now();
        let result_after = engine.generate("Performance test after clear").await;
        let time_after_clear = start_time.elapsed();
        assert!(result_after.is_ok());

        println!(
            "Before clear: {:?}, Clear operation: {:?}, After clear: {:?}",
            time_before_clear, clear_time, time_after_clear
        );

        // Cache clear should be fast
        assert!(clear_time < Duration::from_millis(10));
    }
}

mod backend_performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_cpu_backend_performance() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        let num_requests = 10;
        let start_time = Instant::now();

        for i in 0..num_requests {
            let prompt = format!("CPU backend test {}", i);
            let result = engine.generate(&prompt).await;
            assert!(result.is_ok());
        }

        let total_time = start_time.elapsed();
        let throughput = num_requests as f64 / total_time.as_secs_f64();

        println!("CPU backend throughput: {:.2} requests/second", throughput);
        assert!(throughput > 0.0);

        let stats = engine.get_stats().await;
        assert_eq!(stats.backend_type, "cpu");
    }

    #[tokio::test]
    async fn test_backend_selection_performance() {
        // Test automatic backend selection
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());

        // Test CPU backend
        let cpu_engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu).unwrap();

        let start_time = Instant::now();
        let result = cpu_engine.generate("Backend selection test").await;
        let cpu_time = start_time.elapsed();
        assert!(result.is_ok());

        // Test GPU backend (will fallback to CPU if not available)
        let gpu_engine = InferenceEngine::new(model, tokenizer, Device::Cuda(0)).unwrap();

        let start_time = Instant::now();
        let result = gpu_engine.generate("Backend selection test").await;
        let gpu_time = start_time.elapsed();
        assert!(result.is_ok());

        println!("CPU backend time: {:?}, GPU backend time: {:?}", cpu_time, gpu_time);

        // Both should complete successfully
        assert!(cpu_time > Duration::from_millis(0));
        assert!(gpu_time > Duration::from_millis(0));
    }

    #[tokio::test]
    async fn test_backend_warmup_performance() {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model, tokenizer, device).unwrap();

        // First inference (cold start)
        let start_time = Instant::now();
        let result1 = engine.generate("Cold start test").await;
        let cold_start_time = start_time.elapsed();
        assert!(result1.is_ok());

        // Second inference (warmed up)
        let start_time = Instant::now();
        let result2 = engine.generate("Warmed up test").await;
        let warmed_up_time = start_time.elapsed();
        assert!(result2.is_ok());

        println!("Cold start: {:?}, Warmed up: {:?}", cold_start_time, warmed_up_time);

        // Both should complete, warmed up might be faster
        assert!(cold_start_time > Duration::from_millis(0));
        assert!(warmed_up_time > Duration::from_millis(0));
    }
}

mod stress_tests {
    use super::*;

    async fn create_test_engine() -> InferenceEngine {
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;
        InferenceEngine::new(model, tokenizer, device).unwrap()
    }

    #[tokio::test]
    async fn test_high_concurrency_stress() {
        let engine = Arc::new(create_test_engine().await);
        let num_requests = 50;
        let concurrency = 10;

        let start_time = Instant::now();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut handles = Vec::new();

        for i in 0..num_requests {
            let engine_clone = engine.clone();
            let permit = semaphore.clone().acquire_owned().await.unwrap();

            let handle = tokio::spawn(async move {
                let _permit = permit;
                let prompt = format!("Stress test request {}", i);
                engine_clone.generate(&prompt).await
            });

            handles.push(handle);
        }

        let mut successful_requests = 0;
        let mut failed_requests = 0;

        for handle in handles {
            match handle.await {
                Ok(result) => {
                    if result.is_ok() {
                        successful_requests += 1;
                    } else {
                        failed_requests += 1;
                    }
                }
                Err(_) => failed_requests += 1,
            }
        }

        let total_time = start_time.elapsed();
        let throughput = successful_requests as f64 / total_time.as_secs_f64();

        println!(
            "Stress test results: {} successful, {} failed, {:.2} req/sec",
            successful_requests, failed_requests, throughput
        );

        // Most requests should succeed under stress
        assert!(successful_requests > num_requests * 8 / 10); // At least 80% success rate
        assert!(throughput > 0.0);
    }

    #[tokio::test]
    async fn test_long_running_performance() {
        let engine = create_test_engine().await;
        let duration = Duration::from_secs(10); // Run for 10 seconds
        let start_time = Instant::now();

        let mut request_count = 0;
        let mut total_latency = Duration::from_millis(0);

        while start_time.elapsed() < duration {
            let request_start = Instant::now();
            let prompt = format!("Long running test {}", request_count);

            match timeout(Duration::from_secs(1), engine.generate(&prompt)).await {
                Ok(Ok(_)) => {
                    let request_latency = request_start.elapsed();
                    total_latency += request_latency;
                    request_count += 1;
                }
                Ok(Err(_)) => {
                    // Request failed
                    break;
                }
                Err(_) => {
                    // Request timed out
                    break;
                }
            }
        }

        let actual_duration = start_time.elapsed();
        let throughput = request_count as f64 / actual_duration.as_secs_f64();
        let avg_latency = if request_count > 0 {
            total_latency / request_count as u32
        } else {
            Duration::from_millis(0)
        };

        println!(
            "Long running test: {} requests in {:?}, {:.2} req/sec, avg latency {:?}",
            request_count, actual_duration, throughput, avg_latency
        );

        assert!(request_count > 0);
        assert!(throughput > 0.0);
        assert!(avg_latency < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_memory_stress() {
        let engine = create_test_engine().await;
        let num_requests = 100;

        // Generate many requests to stress memory usage
        for i in 0..num_requests {
            let prompt = format!("Memory stress test {} with longer prompt to use more memory", i);
            let result = engine.generate(&prompt).await;
            assert!(result.is_ok());

            // Check memory usage periodically
            if i % 10 == 0 {
                let stats = engine.get_stats().await;
                println!("Request {}: Memory usage {:.2}%", i, stats.cache_usage);
                assert!(stats.cache_usage >= 0.0);
                assert!(stats.cache_usage <= 100.0);
            }
        }

        // Final memory check
        let final_stats = engine.get_stats().await;
        println!("Final memory usage: {:.2}%", final_stats.cache_usage);
        assert!(final_stats.cache_usage >= 0.0);
        assert!(final_stats.cache_usage <= 100.0);
    }
}
