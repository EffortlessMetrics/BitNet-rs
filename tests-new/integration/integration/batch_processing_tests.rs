//! # Batch Processing Integration Tests
//!
//! Tests batch processing workflows, parallel execution, and resource management
//! for multiple simultaneous inference requests.

use super::*;
#[cfg(feature = "fixtures")]
use crate::common::FixtureManager;
use crate::common::harness::FixtureCtx;
use crate::{BYTES_PER_MB, TestCase, TestError, TestMetrics, TestResult};
use anyhow::Result;
use async_trait::async_trait;
use bitnet_inference::config::GenerationConfig;
use futures_util::future::join_all;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Test suite for batch processing workflows
pub struct BatchProcessingTestSuite;

impl crate::TestSuite for BatchProcessingTestSuite {
    fn name(&self) -> &str {
        "Batch Processing Integration Tests"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(BasicBatchProcessingTest),
            Box::new(ParallelBatchProcessingTest),
            Box::new(BatchSizeOptimizationTest),
            Box::new(BatchResourceManagementTest),
            Box::new(BatchErrorHandlingTest),
        ]
    }
}

/// Test basic batch processing workflow
struct BasicBatchProcessingTest;

#[async_trait]
impl TestCase for BasicBatchProcessingTest {
    fn name(&self) -> &str {
        "basic_batch_processing"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up basic batch processing test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        debug!("Creating components for batch processing test");
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                TestError::execution(format!("Failed to create inference engine: {}", e))
            })?;

        // Test sequential batch processing
        debug!("Testing sequential batch processing");
        let batch_prompts = vec![
            "First batch item",
            "Second batch item",
            "Third batch item",
            "Fourth batch item",
            "Fifth batch item",
        ];

        let mut sequential_results = Vec::new();
        let mut sequential_times = Vec::new();
        let sequential_start = Instant::now();

        for (i, prompt) in batch_prompts.iter().enumerate() {
            debug!("Processing batch item {}: '{}'", i + 1, prompt);

            let item_start = Instant::now();
            let result = engine
                .generate(prompt)
                .await
                .map_err(|e| TestError::execution(format!("Batch item {} failed: {}", i + 1, e)))?;
            let item_time = item_start.elapsed();

            if result.is_empty() {
                return Err(TestError::assertion(format!(
                    "Batch item {} should produce output",
                    i + 1
                )));
            }

            sequential_results.push(result);
            sequential_times.push(item_time);

            debug!(
                "Batch item {} completed in {:?}: '{}'",
                i + 1,
                item_time,
                sequential_results[i]
            );
        }

        let sequential_total_time = sequential_start.elapsed();

        if sequential_results.len() != batch_prompts.len() {
            return Err(TestError::assertion("Not all batch items were processed"));
        }

        // Test batch processing with different configurations
        debug!("Testing batch with different configurations");
        let batch_configs = vec![
            GenerationConfig { max_new_tokens: 3, temperature: 0.0, ..Default::default() },
            GenerationConfig { max_new_tokens: 5, temperature: 0.5, ..Default::default() },
            GenerationConfig { max_new_tokens: 7, temperature: 1.0, ..Default::default() },
        ];

        let config_prompt = "Configuration batch test";
        let mut config_results = Vec::new();
        let mut config_times = Vec::new();

        for (i, config) in batch_configs.iter().enumerate() {
            debug!(
                "Processing config batch item {}: max_tokens={}, temp={}",
                i + 1,
                config.max_new_tokens,
                config.temperature
            );

            let config_start = Instant::now();
            let result = engine.generate_with_config(config_prompt, config).await.map_err(|e| {
                TestError::execution(format!("Config batch item {} failed: {}", i + 1, e))
            })?;
            let config_time = config_start.elapsed();

            config_results.push(result);
            config_times.push(config_time);

            debug!("Config batch item {} completed in {:?}", i + 1, config_time);
        }

        // Test mixed prompt lengths in batch
        debug!("Testing mixed prompt lengths");
        let mixed_prompts = vec![
            "Short",
            "Medium length prompt for testing",
            "This is a much longer prompt that contains multiple sentences and should test how the system handles varying input lengths in a batch processing scenario.",
            "Another short one",
            "Final medium length prompt to complete the batch",
        ];

        let mut mixed_results = Vec::new();
        let mut mixed_times = Vec::new();
        let mut mixed_token_counts = Vec::new();

        for (i, prompt) in mixed_prompts.iter().enumerate() {
            debug!("Processing mixed batch item {}: {} chars", i + 1, prompt.len());

            let mixed_start = Instant::now();

            // Get token count for analysis
            let tokens = tokenizer.encode(prompt, true, false).map_err(|e| {
                TestError::execution(format!("Tokenization failed for mixed item {}: {}", i + 1, e))
            })?;
            mixed_token_counts.push(tokens.len());

            let result = engine.generate(prompt).await.map_err(|e| {
                TestError::execution(format!("Mixed batch item {} failed: {}", i + 1, e))
            })?;
            let mixed_time = mixed_start.elapsed();

            mixed_results.push(result);
            mixed_times.push(mixed_time);

            debug!(
                "Mixed batch item {} ({} tokens) completed in {:?}",
                i + 1,
                tokens.len(),
                mixed_time
            );
        }

        // Verify component interactions
        let model_calls = model.forward_call_count();
        let encode_calls = tokenizer.encode_call_count();
        let decode_calls = tokenizer.decode_call_count();

        debug!(
            "Component calls - Model: {}, Encode: {}, Decode: {}",
            model_calls, encode_calls, decode_calls
        );

        // Calculate statistics
        let avg_sequential_time =
            sequential_times.iter().sum::<std::time::Duration>() / sequential_times.len() as u32;
        let max_sequential_time = sequential_times.iter().max().unwrap();
        let min_sequential_time = sequential_times.iter().min().unwrap();

        let avg_config_time =
            config_times.iter().sum::<std::time::Duration>() / config_times.len() as u32;
        let avg_mixed_time =
            mixed_times.iter().sum::<std::time::Duration>() / mixed_times.len() as u32;

        let avg_token_count =
            mixed_token_counts.iter().sum::<usize>() as f64 / mixed_token_counts.len() as f64;
        let max_token_count = mixed_token_counts.iter().max().unwrap_or(&0);
        let min_token_count = mixed_token_counts.iter().min().unwrap_or(&0);

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("sequential_batch_size".to_string(), batch_prompts.len() as f64),
                ("sequential_total_time_ms".to_string(), sequential_total_time.as_millis() as f64),
                ("avg_sequential_time_ms".to_string(), avg_sequential_time.as_millis() as f64),
                ("max_sequential_time_ms".to_string(), max_sequential_time.as_millis() as f64),
                ("min_sequential_time_ms".to_string(), min_sequential_time.as_millis() as f64),
                ("config_batch_size".to_string(), batch_configs.len() as f64),
                ("avg_config_time_ms".to_string(), avg_config_time.as_millis() as f64),
                ("mixed_batch_size".to_string(), mixed_prompts.len() as f64),
                ("avg_mixed_time_ms".to_string(), avg_mixed_time.as_millis() as f64),
                ("avg_token_count".to_string(), avg_token_count),
                ("max_token_count".to_string(), *max_token_count as f64),
                ("min_token_count".to_string(), *min_token_count as f64),
                ("model_forward_calls".to_string(), model_calls as f64),
                ("tokenizer_encode_calls".to_string(), encode_calls as f64),
                ("tokenizer_decode_calls".to_string(), decode_calls as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up basic batch processing test");
        Ok(())
    }
}

/// Test parallel batch processing
struct ParallelBatchProcessingTest;

#[async_trait]
impl TestCase for ParallelBatchProcessingTest {
    fn name(&self) -> &str {
        "parallel_batch_processing"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up parallel batch processing test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        // Test concurrent processing with multiple engines
        debug!("Testing concurrent batch processing");
        let concurrent_prompts = vec![
            "Concurrent task 1",
            "Concurrent task 2",
            "Concurrent task 3",
            "Concurrent task 4",
            "Concurrent task 5",
            "Concurrent task 6",
        ];

        let mut concurrent_handles = Vec::new();
        let concurrent_start = Instant::now();

        for (i, prompt) in concurrent_prompts.iter().enumerate() {
            let model_clone = model.clone();
            let tokenizer_clone = tokenizer.clone();
            let prompt_clone = prompt.to_string();

            let handle = tokio::spawn(async move {
                let engine = InferenceEngine::new(model_clone, tokenizer_clone, device)?;
                let start_time = Instant::now();
                let result = engine.generate(&prompt_clone).await?;
                let duration = start_time.elapsed();
                Ok::<(String, std::time::Duration), anyhow::Error>((result, duration))
            });

            concurrent_handles.push((i, handle));
        }

        let mut concurrent_results = Vec::new();
        let mut concurrent_times = Vec::new();
        let mut successful_concurrent = 0;
        let mut failed_concurrent = 0;

        for (i, handle) in concurrent_handles {
            match handle.await {
                Ok(Ok((result, duration))) => {
                    debug!("Concurrent task {} completed in {:?}: '{}'", i + 1, duration, result);
                    concurrent_results.push(result);
                    concurrent_times.push(duration);
                    successful_concurrent += 1;
                }
                Ok(Err(e)) => {
                    warn!("Concurrent task {} failed: {}", i + 1, e);
                    failed_concurrent += 1;
                }
                Err(e) => {
                    warn!("Concurrent task {} join failed: {}", i + 1, e);
                    failed_concurrent += 1;
                }
            }
        }

        let concurrent_total_time = concurrent_start.elapsed();

        debug!(
            "Concurrent processing: {} successful, {} failed, total time {:?}",
            successful_concurrent, failed_concurrent, concurrent_total_time
        );

        // Test parallel processing with shared engine
        debug!("Testing parallel processing with shared engine");
        let shared_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create shared engine: {}", e)))?;

        let shared_engine = Arc::new(shared_engine);
        let shared_prompts = vec![
            "Shared engine task 1",
            "Shared engine task 2",
            "Shared engine task 3",
            "Shared engine task 4",
        ];

        let mut shared_handles = Vec::new();
        let shared_start = Instant::now();

        for (i, prompt) in shared_prompts.iter().enumerate() {
            let engine_clone = shared_engine.clone();
            let prompt_clone = prompt.to_string();

            let handle = tokio::spawn(async move {
                let start_time = Instant::now();
                let result = engine_clone.generate(&prompt_clone).await?;
                let duration = start_time.elapsed();
                Ok::<(String, std::time::Duration), anyhow::Error>((result, duration))
            });

            shared_handles.push((i, handle));
        }

        let mut shared_results = Vec::new();
        let mut shared_times = Vec::new();
        let mut successful_shared = 0;

        for (i, handle) in shared_handles {
            match handle.await {
                Ok(Ok((result, duration))) => {
                    debug!(
                        "Shared engine task {} completed in {:?}: '{}'",
                        i + 1,
                        duration,
                        result
                    );
                    shared_results.push(result);
                    shared_times.push(duration);
                    successful_shared += 1;
                }
                Ok(Err(e)) => {
                    warn!("Shared engine task {} failed: {}", i + 1, e);
                }
                Err(e) => {
                    warn!("Shared engine task {} join failed: {}", i + 1, e);
                }
            }
        }

        let shared_total_time = shared_start.elapsed();

        // Test batch processing with futures::join_all
        debug!("Testing batch processing with join_all");
        let batch_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create batch engine: {}", e)))?;

        let batch_prompts = vec!["Batch future 1", "Batch future 2", "Batch future 3"];

        let batch_futures: Vec<_> =
            batch_prompts.iter().map(|prompt| batch_engine.generate(prompt)).collect();

        let batch_start = Instant::now();
        let batch_results = join_all(batch_futures).await;
        let batch_total_time = batch_start.elapsed();

        let mut successful_batch = 0;
        let mut failed_batch = 0;

        for (i, result) in batch_results.into_iter().enumerate() {
            match result {
                Ok(text) => {
                    debug!("Batch future {} succeeded: '{}'", i + 1, text);
                    successful_batch += 1;
                }
                Err(e) => {
                    warn!("Batch future {} failed: {}", i + 1, e);
                    failed_batch += 1;
                }
            }
        }

        // Test resource contention
        debug!("Testing resource contention");
        let contention_count = 10;
        let mut contention_handles = Vec::new();
        let contention_start = Instant::now();

        for i in 0..contention_count {
            let model_clone = model.clone();
            let tokenizer_clone = tokenizer.clone();

            let handle = tokio::spawn(async move {
                let engine = InferenceEngine::new(model_clone, tokenizer_clone, device)?;
                let result = engine.generate(&format!("Contention test {}", i + 1)).await?;
                Ok::<String, anyhow::Error>(result)
            });

            contention_handles.push(handle);
        }

        let mut contention_successes = 0;
        let mut contention_failures = 0;

        for handle in contention_handles {
            match handle.await {
                Ok(Ok(_)) => contention_successes += 1,
                Ok(Err(_)) => contention_failures += 1,
                Err(_) => contention_failures += 1,
            }
        }

        let contention_total_time = contention_start.elapsed();

        debug!(
            "Resource contention: {} successes, {} failures in {:?}",
            contention_successes, contention_failures, contention_total_time
        );

        // Calculate performance metrics
        let avg_concurrent_time = if !concurrent_times.is_empty() {
            concurrent_times.iter().sum::<std::time::Duration>() / concurrent_times.len() as u32
        } else {
            std::time::Duration::ZERO
        };

        let avg_shared_time = if !shared_times.is_empty() {
            shared_times.iter().sum::<std::time::Duration>() / shared_times.len() as u32
        } else {
            std::time::Duration::ZERO
        };

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("concurrent_tasks".to_string(), concurrent_prompts.len() as f64),
                ("successful_concurrent".to_string(), successful_concurrent as f64),
                ("failed_concurrent".to_string(), failed_concurrent as f64),
                ("concurrent_total_time_ms".to_string(), concurrent_total_time.as_millis() as f64),
                ("avg_concurrent_time_ms".to_string(), avg_concurrent_time.as_millis() as f64),
                ("shared_tasks".to_string(), shared_prompts.len() as f64),
                ("successful_shared".to_string(), successful_shared as f64),
                ("shared_total_time_ms".to_string(), shared_total_time.as_millis() as f64),
                ("avg_shared_time_ms".to_string(), avg_shared_time.as_millis() as f64),
                ("batch_tasks".to_string(), batch_prompts.len() as f64),
                ("successful_batch".to_string(), successful_batch as f64),
                ("failed_batch".to_string(), failed_batch as f64),
                ("batch_total_time_ms".to_string(), batch_total_time.as_millis() as f64),
                ("contention_tasks".to_string(), contention_count as f64),
                ("contention_successes".to_string(), contention_successes as f64),
                ("contention_failures".to_string(), contention_failures as f64),
                ("contention_total_time_ms".to_string(), contention_total_time.as_millis() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up parallel batch processing test");
        Ok(())
    }
}

/// Test batch size optimization
struct BatchSizeOptimizationTest;

#[async_trait]
impl TestCase for BatchSizeOptimizationTest {
    fn name(&self) -> &str {
        "batch_size_optimization"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up batch size optimization test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        // Test different batch sizes
        let batch_sizes = vec![1, 2, 4, 8];
        let mut batch_size_results = Vec::new();

        for batch_size in batch_sizes {
            debug!("Testing batch size: {}", batch_size);

            // Create inference config with specific batch size
            let config = InferenceConfig { batch_size, ..Default::default() };

            let engine =
                InferenceEngine::with_config(model.clone(), tokenizer.clone(), device, config)
                    .map_err(|e| {
                        TestError::execution(format!(
                            "Failed to create engine with batch size {}: {}",
                            batch_size, e
                        ))
                    })?;

            // Generate test prompts for this batch size
            let prompts: Vec<String> = (0..batch_size)
                .map(|i| format!("Batch size {} item {}", batch_size, i + 1))
                .collect();

            let batch_start = Instant::now();
            let mut batch_results = Vec::new();

            // Process all prompts in this batch
            for prompt in &prompts {
                match engine.generate(prompt).await {
                    Ok(result) => {
                        batch_results.push(result);
                    }
                    Err(e) => {
                        warn!("Batch size {} item failed: {}", batch_size, e);
                    }
                }
            }

            let batch_duration = batch_start.elapsed();
            let throughput = batch_results.len() as f64 / batch_duration.as_secs_f64();

            batch_size_results.push((batch_size, batch_results.len(), batch_duration, throughput));

            debug!(
                "Batch size {} completed: {} items in {:?} ({:.2} items/sec)",
                batch_size,
                batch_results.len(),
                batch_duration,
                throughput
            );
        }

        // Test optimal batch size detection
        debug!("Analyzing optimal batch size");
        let optimal_batch_size = batch_size_results
            .iter()
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(size, _, _, _)| *size)
            .unwrap_or(1);

        debug!("Optimal batch size detected: {}", optimal_batch_size);

        // Test memory efficiency with different batch sizes
        debug!("Testing memory efficiency across batch sizes");
        let mut memory_measurements = Vec::new();

        for batch_size in [1, 4, 8] {
            let memory_config =
                InferenceConfig { batch_size, ..InferenceConfig::memory_efficient() };

            let memory_engine = InferenceEngine::with_config(
                model.clone(),
                tokenizer.clone(),
                device,
                memory_config,
            )
            .map_err(|e| {
                TestError::execution(format!(
                    "Failed to create memory engine with batch size {}: {}",
                    batch_size, e
                ))
            })?;

            let initial_stats = memory_engine.get_stats().await;

            // Process a few items
            for i in 0..batch_size.min(3) {
                let _ = memory_engine
                    .generate(&format!("Memory test {} item {}", batch_size, i + 1))
                    .await;
            }

            let final_stats = memory_engine.get_stats().await;
            memory_measurements.push((
                batch_size,
                initial_stats.cache_usage,
                final_stats.cache_usage,
            ));

            debug!(
                "Batch size {} memory: {:.2}% -> {:.2}%",
                batch_size, initial_stats.cache_usage, final_stats.cache_usage
            );
        }

        // Test adaptive batch sizing
        debug!("Testing adaptive batch sizing behavior");
        let adaptive_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| {
                TestError::execution(format!("Failed to create adaptive engine: {}", e))
            })?;

        // Simulate varying load
        let load_patterns =
            vec![("Low load", 2), ("Medium load", 5), ("High load", 10), ("Peak load", 15)];

        let mut adaptive_results = Vec::new();

        for &(load_name, concurrent_requests) in &load_patterns {
            debug!(
                "Testing adaptive behavior under {}: {} concurrent requests",
                load_name, concurrent_requests
            );

            let mut adaptive_handles = Vec::new();
            let adaptive_start = Instant::now();

            for i in 0..concurrent_requests {
                let engine_clone = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
                    .map_err(|e| {
                    TestError::execution(format!("Failed to create adaptive engine clone: {}", e))
                })?;

                let handle = tokio::spawn(async move {
                    engine_clone.generate(&format!("{} request {}", load_name, i + 1)).await
                });

                adaptive_handles.push(handle);
            }

            let mut successful_adaptive = 0;
            let mut failed_adaptive = 0;

            for handle in adaptive_handles {
                match handle.await {
                    Ok(Ok(_)) => successful_adaptive += 1,
                    Ok(Err(_)) => failed_adaptive += 1,
                    Err(_) => failed_adaptive += 1,
                }
            }

            let adaptive_duration = adaptive_start.elapsed();
            let adaptive_throughput = successful_adaptive as f64 / adaptive_duration.as_secs_f64();

            adaptive_results.push((
                concurrent_requests,
                successful_adaptive,
                adaptive_duration,
                adaptive_throughput,
            ));

            debug!(
                "{}: {}/{} successful in {:?} ({:.2} req/sec)",
                load_name,
                successful_adaptive,
                concurrent_requests,
                adaptive_duration,
                adaptive_throughput
            );
        }

        // Calculate optimization metrics
        let max_throughput = batch_size_results
            .iter()
            .map(|(_, _, _, throughput)| *throughput)
            .fold(0.0f64, |a, b| a.max(b));

        let avg_throughput =
            batch_size_results.iter().map(|(_, _, _, throughput)| *throughput).sum::<f64>()
                / batch_size_results.len() as f64;

        let total_adaptive_requests: usize =
            adaptive_results.iter().map(|(reqs, _, _, _)| *reqs).sum();
        let total_adaptive_successful: usize =
            adaptive_results.iter().map(|(_, succ, _, _)| *succ).sum();
        let adaptive_success_rate =
            total_adaptive_successful as f64 / total_adaptive_requests as f64;

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("batch_sizes_tested".to_string(), batch_size_results.len() as f64),
                ("optimal_batch_size".to_string(), optimal_batch_size as f64),
                ("max_throughput".to_string(), max_throughput),
                ("avg_throughput".to_string(), avg_throughput),
                ("memory_configs_tested".to_string(), memory_measurements.len() as f64),
                ("load_patterns_tested".to_string(), load_patterns.len() as f64),
                ("total_adaptive_requests".to_string(), total_adaptive_requests as f64),
                ("total_adaptive_successful".to_string(), total_adaptive_successful as f64),
                ("adaptive_success_rate".to_string(), adaptive_success_rate),
                ("model_forward_calls".to_string(), model.forward_call_count() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up batch size optimization test");
        Ok(())
    }
}

/// Test batch resource management
struct BatchResourceManagementTest;

#[async_trait]
impl TestCase for BatchResourceManagementTest {
    fn name(&self) -> &str {
        "batch_resource_management"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up batch resource management test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        // Test memory management during batch processing
        debug!("Testing memory management during batch processing");
        let memory_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create memory engine: {}", e)))?;

        let initial_stats = memory_engine.get_stats().await;
        debug!(
            "Initial memory stats: cache_size={}, usage={:.2}%",
            initial_stats.cache_size, initial_stats.cache_usage
        );

        // Process multiple batches and monitor memory
        let batch_count = 5;
        let items_per_batch = 3;
        let mut memory_progression = Vec::new();

        for batch_i in 0..batch_count {
            debug!("Processing memory batch {}", batch_i + 1);

            for item_i in 0..items_per_batch {
                let prompt = format!("Memory batch {} item {}", batch_i + 1, item_i + 1);
                let _ = memory_engine.generate(&prompt).await.map_err(|e| {
                    TestError::execution(format!("Memory batch item failed: {}", e))
                })?;
            }

            let batch_stats = memory_engine.get_stats().await;
            memory_progression.push(batch_stats.cache_usage);

            debug!("After batch {}: cache_usage={:.2}%", batch_i + 1, batch_stats.cache_usage);
        }

        // Test cache cleanup during batch processing
        debug!("Testing cache cleanup");
        memory_engine.clear_cache().await;

        let after_cleanup_stats = memory_engine.get_stats().await;
        debug!("After cleanup: cache_usage={:.2}%", after_cleanup_stats.cache_usage);

        if after_cleanup_stats.cache_usage > initial_stats.cache_usage + 10.0 {
            warn!("Cache usage didn't decrease significantly after cleanup");
        }

        // Test resource limits
        debug!("Testing resource limits");
        let limited_config = InferenceConfig {
            memory_pool_size: (BYTES_PER_MB * 100) as usize, // 100MB limit
            ..Default::default()
        };

        let limited_engine =
            InferenceEngine::with_config(model.clone(), tokenizer.clone(), device, limited_config)
                .map_err(|e| {
                    TestError::execution(format!("Failed to create limited engine: {}", e))
                })?;

        // Try to exceed resource limits
        let stress_prompts =
            (0..20).map(|i| format!("Resource stress test {}", i + 1)).collect::<Vec<_>>();

        let mut stress_successes = 0;
        let mut stress_failures = 0;

        for prompt in stress_prompts {
            match limited_engine.generate(&prompt).await {
                Ok(_) => stress_successes += 1,
                Err(e) => {
                    debug!("Resource limit hit (expected): {}", e);
                    stress_failures += 1;
                }
            }
        }

        debug!(
            "Resource stress test: {} successes, {} failures",
            stress_successes, stress_failures
        );

        // Test concurrent resource management
        debug!("Testing concurrent resource management");
        let concurrent_engines = 4;
        let mut resource_handles = Vec::new();

        for i in 0..concurrent_engines {
            let model_clone = model.clone();
            let tokenizer_clone = tokenizer.clone();

            let handle = tokio::spawn(async move {
                let engine = InferenceEngine::new(model_clone, tokenizer_clone, device)?;

                let mut results = Vec::new();
                for j in 0..3 {
                    let prompt = format!("Concurrent resource test {} item {}", i + 1, j + 1);
                    match engine.generate(&prompt).await {
                        Ok(result) => results.push(result),
                        Err(e) => return Err(e),
                    }
                }

                let stats = engine.get_stats().await;
                Ok::<(Vec<String>, f64), anyhow::Error>((results, stats.cache_usage))
            });

            resource_handles.push(handle);
        }

        let mut concurrent_resource_results = Vec::new();
        let mut concurrent_cache_usages = Vec::new();

        for handle in resource_handles {
            match handle.await {
                Ok(Ok((results, cache_usage))) => {
                    concurrent_resource_results.push(results.len());
                    concurrent_cache_usages.push(cache_usage);
                }
                Ok(Err(e)) => {
                    warn!("Concurrent resource test failed: {}", e);
                }
                Err(e) => {
                    warn!("Concurrent resource test join failed: {}", e);
                }
            }
        }

        debug!(
            "Concurrent resource management: {} engines completed",
            concurrent_resource_results.len()
        );

        // Test resource cleanup after batch completion
        debug!("Testing resource cleanup after batch completion");
        let cleanup_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create cleanup engine: {}", e)))?;

        // Process a batch
        for i in 0..5 {
            let _ = cleanup_engine.generate(&format!("Cleanup test {}", i + 1)).await;
        }

        let before_cleanup = cleanup_engine.get_stats().await;

        // Simulate batch completion cleanup
        cleanup_engine.clear_cache().await;

        let after_batch_cleanup = cleanup_engine.get_stats().await;

        debug!(
            "Batch cleanup: {:.2}% -> {:.2}%",
            before_cleanup.cache_usage, after_batch_cleanup.cache_usage
        );

        // Calculate resource management metrics
        let max_memory_usage = memory_progression.iter().fold(0.0f64, |a, &b| a.max(b));
        let memory_growth = if !memory_progression.is_empty() {
            memory_progression.last().unwrap() - memory_progression.first().unwrap()
        } else {
            0.0
        };

        let avg_concurrent_cache_usage = if !concurrent_cache_usages.is_empty() {
            concurrent_cache_usages.iter().sum::<f64>() / concurrent_cache_usages.len() as f64
        } else {
            0.0
        };

        let successful_concurrent_engines = concurrent_resource_results.len();

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("memory_batches_processed".to_string(), batch_count as f64),
                ("items_per_batch".to_string(), items_per_batch as f64),
                ("max_memory_usage_percent".to_string(), max_memory_usage),
                ("memory_growth_percent".to_string(), memory_growth),
                ("initial_cache_usage".to_string(), initial_stats.cache_usage),
                ("after_cleanup_cache_usage".to_string(), after_cleanup_stats.cache_usage),
                ("stress_successes".to_string(), stress_successes as f64),
                ("stress_failures".to_string(), stress_failures as f64),
                ("concurrent_engines_tested".to_string(), concurrent_engines as f64),
                ("successful_concurrent_engines".to_string(), successful_concurrent_engines as f64),
                ("avg_concurrent_cache_usage".to_string(), avg_concurrent_cache_usage),
                ("before_batch_cleanup_usage".to_string(), before_cleanup.cache_usage),
                ("after_batch_cleanup_usage".to_string(), after_batch_cleanup.cache_usage),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up batch resource management test");
        Ok(())
    }
}

/// Test batch error handling
struct BatchErrorHandlingTest;

#[async_trait]
impl TestCase for BatchErrorHandlingTest {
    fn name(&self) -> &str {
        "batch_error_handling"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up batch error handling test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                TestError::execution(format!("Failed to create inference engine: {}", e))
            })?;

        // Test partial batch failure
        debug!("Testing partial batch failure handling");
        let mixed_prompts = vec![
            "Valid prompt 1",
            "", // Empty prompt (might fail)
            "Valid prompt 2",
            "\0", // Null character (might fail)
            "Valid prompt 3",
        ];

        let mut partial_results = Vec::new();
        let mut partial_successes = 0;
        let mut partial_failures = 0;

        for (i, prompt) in mixed_prompts.iter().enumerate() {
            debug!("Processing mixed prompt {}: {:?}", i + 1, prompt);

            match engine.generate(prompt).await {
                Ok(result) => {
                    debug!("Mixed prompt {} succeeded: '{}'", i + 1, result);
                    partial_results.push(Some(result));
                    partial_successes += 1;
                }
                Err(e) => {
                    debug!("Mixed prompt {} failed (expected for some): {}", i + 1, e);
                    partial_results.push(None);
                    partial_failures += 1;
                }
            }
        }

        debug!(
            "Partial batch results: {} successes, {} failures",
            partial_successes, partial_failures
        );

        // Test error recovery in batch processing
        debug!("Testing error recovery in batch processing");
        let recovery_prompts = vec![
            "Before error",
            "", // Potential error
            "After error 1",
            "After error 2",
        ];

        let mut recovery_results = Vec::new();
        let mut recovery_successes = 0;

        for (i, prompt) in recovery_prompts.iter().enumerate() {
            match engine.generate(prompt).await {
                Ok(result) => {
                    debug!("Recovery prompt {} succeeded: '{}'", i + 1, result);
                    recovery_results.push(result);
                    recovery_successes += 1;
                }
                Err(e) => {
                    debug!("Recovery prompt {} failed: {}", i + 1, e);
                    // Continue processing despite error
                }
            }
        }

        if recovery_successes == 0 {
            return Err(TestError::assertion("Should have at least some successful recovery"));
        }

        // Test concurrent error handling
        debug!("Testing concurrent error handling");
        let concurrent_error_prompts = vec![
            "Concurrent valid 1",
            "", // Empty
            "Concurrent valid 2",
            "\0", // Null
            "Concurrent valid 3",
            "🚀", // Unicode
        ];

        let mut concurrent_error_handles = Vec::new();

        for (i, prompt) in concurrent_error_prompts.iter().enumerate() {
            let model_clone = model.clone();
            let tokenizer_clone = tokenizer.clone();
            let prompt_clone = prompt.to_string();

            let handle = tokio::spawn(async move {
                let engine = InferenceEngine::new(model_clone, tokenizer_clone, device)?;
                let result = engine.generate(&prompt_clone).await;
                Ok::<Result<String, anyhow::Error>, anyhow::Error>(result)
            });

            concurrent_error_handles.push((i, handle));
        }

        let mut concurrent_error_successes = 0;
        let mut concurrent_error_failures = 0;

        for (i, handle) in concurrent_error_handles {
            match handle.await {
                Ok(Ok(Ok(result))) => {
                    debug!("Concurrent error test {} succeeded: '{}'", i + 1, result);
                    concurrent_error_successes += 1;
                }
                Ok(Ok(Err(e))) => {
                    debug!("Concurrent error test {} failed (expected): {}", i + 1, e);
                    concurrent_error_failures += 1;
                }
                Ok(Err(e)) => {
                    debug!("Concurrent error test {} engine creation failed: {}", i + 1, e);
                    concurrent_error_failures += 1;
                }
                Err(e) => {
                    debug!("Concurrent error test {} join failed: {}", i + 1, e);
                    concurrent_error_failures += 1;
                }
            }
        }

        // Test batch timeout handling
        debug!("Testing batch timeout handling");
        let timeout_duration = std::time::Duration::from_millis(1000);
        let timeout_prompts = vec!["Timeout test 1", "Timeout test 2", "Timeout test 3"];

        let mut timeout_results = Vec::new();

        for (i, prompt) in timeout_prompts.iter().enumerate() {
            debug!("Testing timeout for prompt {}", i + 1);

            let timeout_result =
                tokio::time::timeout(timeout_duration, engine.generate(prompt)).await;

            match timeout_result {
                Ok(Ok(result)) => {
                    debug!("Timeout test {} completed: '{}'", i + 1, result);
                    timeout_results.push(true);
                }
                Ok(Err(e)) => {
                    debug!("Timeout test {} failed: {}", i + 1, e);
                    timeout_results.push(false);
                }
                Err(_) => {
                    debug!("Timeout test {} timed out", i + 1);
                    timeout_results.push(false);
                }
            }
        }

        let timeout_successes = timeout_results.iter().filter(|&&x| x).count();

        // Test error isolation
        debug!("Testing error isolation between batch items");
        let isolation_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| {
                TestError::execution(format!("Failed to create isolation engine: {}", e))
            })?;

        // Cause an error
        let _ = isolation_engine.generate("").await; // Might fail

        // Test that subsequent operations work
        let isolation_result = isolation_engine
            .generate("Isolation test")
            .await
            .map_err(|e| TestError::execution(format!("Isolation test failed: {}", e)))?;

        if isolation_result.is_empty() {
            return Err(TestError::assertion(
                "Error isolation failed - subsequent operations should work",
            ));
        }

        debug!("Error isolation successful: '{}'", isolation_result);

        // Calculate error handling metrics
        let partial_success_rate = partial_successes as f64 / mixed_prompts.len() as f64;
        let recovery_success_rate = recovery_successes as f64 / recovery_prompts.len() as f64;
        let concurrent_success_rate =
            concurrent_error_successes as f64 / concurrent_error_prompts.len() as f64;
        let timeout_success_rate = timeout_successes as f64 / timeout_prompts.len() as f64;

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("mixed_prompts_tested".to_string(), mixed_prompts.len() as f64),
                ("partial_successes".to_string(), partial_successes as f64),
                ("partial_failures".to_string(), partial_failures as f64),
                ("partial_success_rate".to_string(), partial_success_rate),
                ("recovery_prompts_tested".to_string(), recovery_prompts.len() as f64),
                ("recovery_successes".to_string(), recovery_successes as f64),
                ("recovery_success_rate".to_string(), recovery_success_rate),
                ("concurrent_error_prompts".to_string(), concurrent_error_prompts.len() as f64),
                ("concurrent_error_successes".to_string(), concurrent_error_successes as f64),
                ("concurrent_error_failures".to_string(), concurrent_error_failures as f64),
                ("concurrent_success_rate".to_string(), concurrent_success_rate),
                ("timeout_prompts_tested".to_string(), timeout_prompts.len() as f64),
                ("timeout_successes".to_string(), timeout_successes as f64),
                ("timeout_success_rate".to_string(), timeout_success_rate),
                ("error_isolation_successful".to_string(), 1.0),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up batch error handling test");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestConfig, TestHarness};

    #[tokio::test]
    async fn test_batch_processing_suite() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let suite = BatchProcessingTestSuite;

        let result = harness.run_test_suite(&suite).await;
        assert!(result.is_ok());

        let suite_result = result.unwrap();
        assert!(suite_result.summary.total_tests > 0);
    }
}
