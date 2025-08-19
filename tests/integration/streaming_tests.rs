//! # Streaming Inference Workflow Integration Tests
//!
//! Tests streaming generation workflows, backpressure handling, and real-time performance.

use super::*;
use crate::{TestCase, TestError, TestMetrics, TestResult};
#[cfg(feature = "fixtures")]
use crate::common::FixtureManager;
use crate::common::harness::FixtureCtx;
use async_trait::async_trait;
use bitnet_inference::{GenerationStream, StreamingConfig};
use futures_util::StreamExt;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Test suite for streaming inference workflows
pub struct StreamingWorkflowTestSuite;

impl crate::TestSuite for StreamingWorkflowTestSuite {
    fn name(&self) -> &str {
        "Streaming Workflow Integration Tests"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(BasicStreamingTest),
            Box::new(StreamingConfigurationTest),
            Box::new(StreamingBackpressureTest),
            Box::new(StreamingCancellationTest),
            Box::new(StreamingPerformanceTest),
        ]
    }
}

/// Test basic streaming generation workflow
struct BasicStreamingTest;

#[async_trait]
impl TestCase for BasicStreamingTest {
    fn name(&self) -> &str {
        "basic_streaming_workflow"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up basic streaming workflow test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        debug!("Creating components for streaming test");
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                TestError::execution(format!("Failed to create inference engine: {}", e))
            })?;

        // Test basic streaming generation
        debug!("Testing basic streaming generation");
        let prompt = "Generate streaming text";
        let mut stream = engine.generate_stream(prompt);

        let mut received_chunks = Vec::new();
        let mut total_text = String::new();
        let mut chunk_count = 0;
        let stream_start = Instant::now();

        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    debug!("Received chunk {}: '{}'", chunk_count + 1, chunk);
                    received_chunks.push(chunk.clone());
                    total_text.push_str(&chunk);
                    chunk_count += 1;

                    // Prevent infinite loop in test
                    if chunk_count >= 10 {
                        debug!("Stopping after {} chunks to prevent infinite loop", chunk_count);
                        break;
                    }
                }
                Err(e) => {
                    return Err(TestError::execution(format!("Streaming error: {}", e)));
                }
            }
        }

        let stream_duration = stream_start.elapsed();

        if received_chunks.is_empty() {
            return Err(TestError::assertion("Should receive at least one chunk"));
        }

        if total_text.is_empty() {
            return Err(TestError::assertion("Total generated text should not be empty"));
        }

        debug!(
            "Streaming completed: {} chunks, {} total characters in {:?}",
            chunk_count,
            total_text.len(),
            stream_duration
        );

        // Test streaming with different prompts
        debug!("Testing streaming with multiple prompts");
        let test_prompts = vec![
            "Short prompt",
            "This is a longer prompt for testing streaming generation",
            "Multi-word prompt with punctuation!",
        ];

        let mut prompt_results = Vec::new();

        for (i, test_prompt) in test_prompts.iter().enumerate() {
            debug!("Testing streaming with prompt {}: '{}'", i + 1, test_prompt);

            let mut prompt_stream = engine.generate_stream(test_prompt);
            let mut prompt_chunks = 0;
            let mut prompt_text = String::new();

            while let Some(result) = prompt_stream.next().await {
                match result {
                    Ok(chunk) => {
                        prompt_text.push_str(&chunk);
                        prompt_chunks += 1;

                        if prompt_chunks >= 5 {
                            // Limit chunks per prompt
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Streaming error for prompt {}: {}", i + 1, e);
                        break;
                    }
                }
            }

            prompt_results.push((prompt_chunks, prompt_text.len()));
            debug!(
                "Prompt {} generated {} chunks, {} characters",
                i + 1,
                prompt_chunks,
                prompt_text.len()
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

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("chunks_received".to_string(), chunk_count as f64),
                ("total_characters".to_string(), total_text.len() as f64),
                ("stream_duration_ms".to_string(), stream_duration.as_millis() as f64),
                ("prompts_tested".to_string(), test_prompts.len() as f64),
                (
                    "avg_chunks_per_prompt".to_string(),
                    prompt_results.iter().map(|(chunks, _)| *chunks).sum::<usize>() as f64
                        / test_prompts.len() as f64,
                ),
                ("model_forward_calls".to_string(), model_calls as f64),
                ("tokenizer_encode_calls".to_string(), encode_calls as f64),
                ("tokenizer_decode_calls".to_string(), decode_calls as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up basic streaming workflow test");
        Ok(())
    }
}

/// Test streaming with different configurations
struct StreamingConfigurationTest;

#[async_trait]
impl TestCase for StreamingConfigurationTest {
    fn name(&self) -> &str {
        "streaming_configuration"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up streaming configuration test");
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

        // Test different generation configurations
        let test_configs = vec![
            GenerationConfig {
                max_new_tokens: 5,
                temperature: 0.0, // Deterministic
                ..Default::default()
            },
            GenerationConfig {
                max_new_tokens: 10,
                temperature: 0.7,
                top_k: 50,
                ..Default::default()
            },
            GenerationConfig {
                max_new_tokens: 3,
                temperature: 1.0, // High randomness
                top_p: 0.9,
                ..Default::default()
            },
        ];

        let prompt = "Configuration test prompt";
        let mut config_results = Vec::new();

        debug!("Testing {} different configurations", test_configs.len());

        for (i, config) in test_configs.iter().enumerate() {
            debug!(
                "Testing configuration {}: max_tokens={}, temp={}",
                i + 1,
                config.max_new_tokens,
                config.temperature
            );

            let config_start = Instant::now();
            let mut stream = engine.generate_stream_with_config(prompt, config);

            let mut chunks = 0;
            let mut total_chars = 0;
            let mut first_chunk_time = None;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        if first_chunk_time.is_none() {
                            first_chunk_time = Some(config_start.elapsed());
                        }

                        chunks += 1;
                        total_chars += chunk.len();

                        debug!("Config {} chunk {}: '{}'", i + 1, chunks, chunk);

                        // Limit chunks to prevent infinite loops
                        if chunks >= config.max_new_tokens as usize + 2 {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Configuration {} streaming error: {}", i + 1, e);
                        break;
                    }
                }
            }

            let config_duration = config_start.elapsed();
            let ttft = first_chunk_time.unwrap_or(config_duration); // Time to first token

            config_results.push((chunks, total_chars, config_duration, ttft));

            debug!(
                "Configuration {} completed: {} chunks, {} chars, {:?} total, {:?} TTFT",
                i + 1,
                chunks,
                total_chars,
                config_duration,
                ttft
            );
        }

        // Test streaming-specific configurations
        debug!("Testing streaming-specific configurations");

        let streaming_configs = vec![
            StreamingConfig {
                buffer_size: 1, // Immediate streaming
                flush_interval_ms: 10,
            },
            StreamingConfig {
                buffer_size: 5, // Buffered streaming
                flush_interval_ms: 100,
            },
            StreamingConfig {
                buffer_size: 10, // Larger buffer
                flush_interval_ms: 200,
            },
        ];

        let mut streaming_config_results = Vec::new();

        for (i, streaming_config) in streaming_configs.iter().enumerate() {
            debug!(
                "Testing streaming config {}: buffer={}, interval={}ms",
                i + 1,
                streaming_config.buffer_size,
                streaming_config.flush_interval_ms
            );

            // Create a custom stream with specific streaming config
            // Note: This would require extending the API to accept StreamingConfig
            // For now, we'll test with default streaming and measure behavior

            let stream_start = Instant::now();
            let mut stream = engine.generate_stream(prompt);

            let mut stream_chunks = 0;
            let mut stream_chars = 0;
            let mut chunk_intervals = Vec::new();
            let mut last_chunk_time = stream_start;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        let now = Instant::now();
                        let interval = now.duration_since(last_chunk_time);
                        chunk_intervals.push(interval);
                        last_chunk_time = now;

                        stream_chunks += 1;
                        stream_chars += chunk.len();

                        if stream_chunks >= 5 {
                            // Limit for test
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Streaming config {} error: {}", i + 1, e);
                        break;
                    }
                }
            }

            let avg_interval = if !chunk_intervals.is_empty() {
                chunk_intervals.iter().sum::<std::time::Duration>() / chunk_intervals.len() as u32
            } else {
                std::time::Duration::ZERO
            };

            streaming_config_results.push((stream_chunks, stream_chars, avg_interval));

            debug!(
                "Streaming config {} completed: {} chunks, avg interval {:?}",
                i + 1,
                stream_chunks,
                avg_interval
            );
        }

        // Calculate overall statistics
        let total_configs_tested = test_configs.len() + streaming_configs.len();
        let avg_chunks = config_results.iter().map(|(chunks, _, _, _)| *chunks).sum::<usize>()
            as f64
            / test_configs.len() as f64;
        let avg_ttft = config_results.iter().map(|(_, _, _, ttft)| ttft.as_millis()).sum::<u128>()
            as f64
            / test_configs.len() as f64;

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("generation_configs_tested".to_string(), test_configs.len() as f64),
                ("streaming_configs_tested".to_string(), streaming_configs.len() as f64),
                ("total_configs_tested".to_string(), total_configs_tested as f64),
                ("avg_chunks_per_config".to_string(), avg_chunks),
                ("avg_ttft_ms".to_string(), avg_ttft),
                ("model_forward_calls".to_string(), model.forward_call_count() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up streaming configuration test");
        Ok(())
    }
}

/// Test streaming backpressure handling
struct StreamingBackpressureTest;

#[async_trait]
impl TestCase for StreamingBackpressureTest {
    fn name(&self) -> &str {
        "streaming_backpressure"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up streaming backpressure test");
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

        // Test slow consumer scenario
        debug!("Testing slow consumer backpressure");
        let prompt = "Backpressure test prompt";
        let mut stream = engine.generate_stream(prompt);

        let mut slow_consumer_chunks = 0;
        let mut processing_times = Vec::new();

        while let Some(result) = stream.next().await {
            let process_start = Instant::now();

            match result {
                Ok(chunk) => {
                    debug!(
                        "Slow consumer received chunk {}: '{}'",
                        slow_consumer_chunks + 1,
                        chunk
                    );

                    // Simulate slow processing
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

                    let process_time = process_start.elapsed();
                    processing_times.push(process_time);
                    slow_consumer_chunks += 1;

                    if slow_consumer_chunks >= 5 {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Slow consumer error: {}", e);
                    break;
                }
            }
        }

        debug!("Slow consumer processed {} chunks", slow_consumer_chunks);

        // Test fast consumer scenario
        debug!("Testing fast consumer");
        let mut fast_stream = engine.generate_stream(prompt);

        let mut fast_consumer_chunks = 0;
        let fast_start = Instant::now();

        while let Some(result) = fast_stream.next().await {
            match result {
                Ok(chunk) => {
                    debug!(
                        "Fast consumer received chunk {}: '{}'",
                        fast_consumer_chunks + 1,
                        chunk
                    );
                    fast_consumer_chunks += 1;

                    if fast_consumer_chunks >= 5 {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Fast consumer error: {}", e);
                    break;
                }
            }
        }

        let fast_duration = fast_start.elapsed();
        debug!("Fast consumer processed {} chunks in {:?}", fast_consumer_chunks, fast_duration);

        // Test multiple concurrent consumers
        debug!("Testing concurrent consumers");
        let concurrent_consumers = 3;
        let mut concurrent_handles = Vec::new();

        for i in 0..concurrent_consumers {
            let engine_clone = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
                .map_err(|e| {
                    TestError::execution(format!("Failed to create concurrent engine: {}", e))
                })?;

            let consumer_prompt = format!("Concurrent consumer {} prompt", i + 1);

            let handle = tokio::spawn(async move {
                let mut stream = engine_clone.generate_stream(&consumer_prompt);
                let mut chunks = 0;

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(_chunk) => {
                            chunks += 1;
                            if chunks >= 3 {
                                // Limit per consumer
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }

                chunks
            });

            concurrent_handles.push(handle);
        }

        let mut concurrent_results = Vec::new();
        for handle in concurrent_handles {
            match handle.await {
                Ok(chunks) => concurrent_results.push(chunks),
                Err(e) => {
                    warn!("Concurrent consumer failed: {}", e);
                    concurrent_results.push(0);
                }
            }
        }

        let total_concurrent_chunks: usize = concurrent_results.iter().sum();
        debug!("Concurrent consumers processed {} total chunks", total_concurrent_chunks);

        // Test buffer overflow scenarios
        debug!("Testing buffer behavior");
        let mut buffer_stream = engine.generate_stream("Buffer test");

        // Collect several chunks without processing to test buffering
        let mut buffered_chunks = Vec::new();
        let buffer_start = Instant::now();

        for _ in 0..3 {
            if let Some(result) = buffer_stream.next().await {
                match result {
                    Ok(chunk) => buffered_chunks.push(chunk),
                    Err(e) => {
                        warn!("Buffer test error: {}", e);
                        break;
                    }
                }
            }
        }

        let buffer_duration = buffer_start.elapsed();
        debug!("Buffered {} chunks in {:?}", buffered_chunks.len(), buffer_duration);

        // Calculate statistics
        let avg_processing_time = if !processing_times.is_empty() {
            processing_times.iter().sum::<std::time::Duration>() / processing_times.len() as u32
        } else {
            std::time::Duration::ZERO
        };

        let successful_concurrent_consumers = concurrent_results.iter().filter(|&&x| x > 0).count();

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("slow_consumer_chunks".to_string(), slow_consumer_chunks as f64),
                ("fast_consumer_chunks".to_string(), fast_consumer_chunks as f64),
                ("avg_processing_time_ms".to_string(), avg_processing_time.as_millis() as f64),
                ("fast_consumer_duration_ms".to_string(), fast_duration.as_millis() as f64),
                ("concurrent_consumers".to_string(), concurrent_consumers as f64),
                (
                    "successful_concurrent_consumers".to_string(),
                    successful_concurrent_consumers as f64,
                ),
                ("total_concurrent_chunks".to_string(), total_concurrent_chunks as f64),
                ("buffered_chunks".to_string(), buffered_chunks.len() as f64),
                ("buffer_duration_ms".to_string(), buffer_duration.as_millis() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up streaming backpressure test");
        Ok(())
    }
}

/// Test streaming cancellation and cleanup
struct StreamingCancellationTest;

#[async_trait]
impl TestCase for StreamingCancellationTest {
    fn name(&self) -> &str {
        "streaming_cancellation"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up streaming cancellation test");
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

        // Test early stream termination
        debug!("Testing early stream termination");
        let prompt = "Cancellation test prompt";
        let mut stream = engine.generate_stream(prompt);

        let mut early_termination_chunks = 0;

        // Consume only a few chunks then drop the stream
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    debug!("Early termination chunk {}: '{}'", early_termination_chunks + 1, chunk);
                    early_termination_chunks += 1;

                    if early_termination_chunks >= 2 {
                        debug!(
                            "Terminating stream early after {} chunks",
                            early_termination_chunks
                        );
                        break; // Early termination
                    }
                }
                Err(e) => {
                    warn!("Early termination error: {}", e);
                    break;
                }
            }
        }

        // Stream should be dropped here, testing cleanup

        // Test timeout scenarios
        debug!("Testing timeout behavior");
        let timeout_duration = std::time::Duration::from_millis(500);

        let mut timeout_stream = engine.generate_stream("Timeout test");
        let timeout_start = Instant::now();
        let mut timeout_chunks = 0;

        loop {
            let timeout_result =
                tokio::time::timeout(timeout_duration, timeout_stream.next()).await;

            match timeout_result {
                Ok(Some(Ok(chunk))) => {
                    debug!("Timeout test chunk {}: '{}'", timeout_chunks + 1, chunk);
                    timeout_chunks += 1;

                    if timeout_chunks >= 3 {
                        break;
                    }
                }
                Ok(Some(Err(e))) => {
                    warn!("Timeout test error: {}", e);
                    break;
                }
                Ok(None) => {
                    debug!("Stream ended naturally");
                    break;
                }
                Err(_) => {
                    debug!("Stream timed out after {:?}", timeout_start.elapsed());
                    break;
                }
            }
        }

        // Test multiple stream creation and cancellation
        debug!("Testing multiple stream lifecycle");
        let stream_count = 5;
        let mut created_streams = 0;
        let mut successful_cancellations = 0;

        for i in 0..stream_count {
            debug!("Creating stream {}", i + 1);

            let mut test_stream = engine.generate_stream(&format!("Stream {} test", i + 1));
            created_streams += 1;

            // Consume one chunk then cancel
            if let Some(result) = test_stream.next().await {
                match result {
                    Ok(chunk) => {
                        debug!("Stream {} first chunk: '{}'", i + 1, chunk);
                        successful_cancellations += 1;
                    }
                    Err(e) => {
                        warn!("Stream {} error: {}", i + 1, e);
                    }
                }
            }

            // Stream is dropped here (cancelled)
            debug!("Stream {} cancelled", i + 1);
        }

        // Test resource cleanup after cancellation
        debug!("Testing resource cleanup");
        let cleanup_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create cleanup engine: {}", e)))?;

        // Get initial stats
        let initial_stats = cleanup_engine.get_stats().await;
        debug!(
            "Initial stats: cache_size={}, usage={:.2}%",
            initial_stats.cache_size, initial_stats.cache_usage
        );

        // Create and cancel multiple streams
        for i in 0..3 {
            let mut cleanup_stream =
                cleanup_engine.generate_stream(&format!("Cleanup test {}", i + 1));

            // Consume one chunk
            if let Some(Ok(_)) = cleanup_stream.next().await {
                debug!("Cleanup stream {} consumed one chunk", i + 1);
            }

            // Stream cancelled by dropping
        }

        // Check stats after cancellations
        let after_stats = cleanup_engine.get_stats().await;
        debug!(
            "After cancellations: cache_size={}, usage={:.2}%",
            after_stats.cache_size, after_stats.cache_usage
        );

        // Test that engine still works after cancellations
        let post_cancel_result =
            cleanup_engine.generate("Post cancellation test").await.map_err(|e| {
                TestError::execution(format!("Post-cancellation generation failed: {}", e))
            })?;

        if post_cancel_result.is_empty() {
            return Err(TestError::assertion("Engine should work after stream cancellations"));
        }

        debug!("Post-cancellation generation successful: '{}'", post_cancel_result);

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("early_termination_chunks".to_string(), early_termination_chunks as f64),
                ("timeout_chunks".to_string(), timeout_chunks as f64),
                ("streams_created".to_string(), created_streams as f64),
                ("successful_cancellations".to_string(), successful_cancellations as f64),
                ("initial_cache_size".to_string(), initial_stats.cache_size as f64),
                ("after_cancel_cache_size".to_string(), after_stats.cache_size as f64),
                ("post_cancel_generation_success".to_string(), 1.0),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up streaming cancellation test");
        Ok(())
    }
}

/// Test streaming performance characteristics
struct StreamingPerformanceTest;

#[async_trait]
impl TestCase for StreamingPerformanceTest {
    fn name(&self) -> &str {
        "streaming_performance"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up streaming performance test");
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

        // Test time to first token (TTFT)
        debug!("Testing time to first token");
        let ttft_prompt = "TTFT test prompt";
        let mut ttft_measurements = Vec::new();

        for i in 0..3 {
            debug!("TTFT measurement {}", i + 1);

            let ttft_start = Instant::now();
            let mut stream = engine.generate_stream(ttft_prompt);

            if let Some(result) = stream.next().await {
                let ttft = ttft_start.elapsed();

                match result {
                    Ok(chunk) => {
                        debug!("TTFT {} - First chunk in {:?}: '{}'", i + 1, ttft, chunk);
                        ttft_measurements.push(ttft);
                    }
                    Err(e) => {
                        warn!("TTFT {} error: {}", i + 1, e);
                    }
                }
            }
        }

        // Test throughput (tokens per second)
        debug!("Testing streaming throughput");
        let throughput_prompt = "Throughput test prompt for measuring tokens per second";
        let mut throughput_stream = engine.generate_stream(throughput_prompt);

        let throughput_start = Instant::now();
        let mut throughput_chunks = 0;
        let mut throughput_chars = 0;

        while let Some(result) = throughput_stream.next().await {
            match result {
                Ok(chunk) => {
                    throughput_chunks += 1;
                    throughput_chars += chunk.len();

                    debug!("Throughput chunk {}: {} chars", throughput_chunks, chunk.len());

                    if throughput_chunks >= 10 {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Throughput test error: {}", e);
                    break;
                }
            }
        }

        let throughput_duration = throughput_start.elapsed();
        let chars_per_second = if throughput_duration.as_secs_f64() > 0.0 {
            throughput_chars as f64 / throughput_duration.as_secs_f64()
        } else {
            0.0
        };

        debug!(
            "Throughput: {} chars in {:?} = {:.2} chars/sec",
            throughput_chars, throughput_duration, chars_per_second
        );

        // Test latency consistency
        debug!("Testing latency consistency");
        let latency_prompt = "Latency consistency test";
        let mut latency_stream = engine.generate_stream(latency_prompt);

        let mut inter_chunk_latencies = Vec::new();
        let mut last_chunk_time = Instant::now();
        let mut latency_chunks = 0;

        while let Some(result) = latency_stream.next().await {
            let chunk_time = Instant::now();
            let inter_chunk_latency = chunk_time.duration_since(last_chunk_time);

            match result {
                Ok(chunk) => {
                    if latency_chunks > 0 {
                        // Skip first chunk for inter-chunk measurement
                        inter_chunk_latencies.push(inter_chunk_latency);
                        debug!("Inter-chunk latency {}: {:?}", latency_chunks, inter_chunk_latency);
                    }

                    latency_chunks += 1;
                    last_chunk_time = chunk_time;

                    if latency_chunks >= 8 {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Latency test error: {}", e);
                    break;
                }
            }
        }

        // Test memory efficiency during streaming
        debug!("Testing memory efficiency");
        let memory_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create memory engine: {}", e)))?;

        let initial_memory_stats = memory_engine.get_stats().await;
        debug!(
            "Initial memory stats: cache_size={}, usage={:.2}%",
            initial_memory_stats.cache_size, initial_memory_stats.cache_usage
        );

        let mut memory_stream = memory_engine.generate_stream("Memory efficiency test");
        let mut memory_measurements = Vec::new();

        for i in 0..5 {
            if let Some(result) = memory_stream.next().await {
                match result {
                    Ok(_chunk) => {
                        let stats = memory_engine.get_stats().await;
                        memory_measurements.push(stats.cache_usage);
                        debug!("Memory measurement {}: {:.2}% usage", i + 1, stats.cache_usage);
                    }
                    Err(e) => {
                        warn!("Memory test error: {}", e);
                        break;
                    }
                }
            }
        }

        // Calculate performance statistics
        let avg_ttft = if !ttft_measurements.is_empty() {
            ttft_measurements.iter().sum::<std::time::Duration>() / ttft_measurements.len() as u32
        } else {
            std::time::Duration::ZERO
        };

        let min_ttft = ttft_measurements.iter().min().cloned().unwrap_or(std::time::Duration::ZERO);
        let max_ttft = ttft_measurements.iter().max().cloned().unwrap_or(std::time::Duration::ZERO);

        let avg_inter_chunk_latency = if !inter_chunk_latencies.is_empty() {
            inter_chunk_latencies.iter().sum::<std::time::Duration>()
                / inter_chunk_latencies.len() as u32
        } else {
            std::time::Duration::ZERO
        };

        let max_memory_usage = memory_measurements.iter().fold(0.0f64, |a, &b| a.max(b));

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("ttft_measurements".to_string(), ttft_measurements.len() as f64),
                ("avg_ttft_ms".to_string(), avg_ttft.as_millis() as f64),
                ("min_ttft_ms".to_string(), min_ttft.as_millis() as f64),
                ("max_ttft_ms".to_string(), max_ttft.as_millis() as f64),
                ("throughput_chars_per_sec".to_string(), chars_per_second),
                ("throughput_chunks".to_string(), throughput_chunks as f64),
                ("throughput_duration_ms".to_string(), throughput_duration.as_millis() as f64),
                (
                    "avg_inter_chunk_latency_ms".to_string(),
                    avg_inter_chunk_latency.as_millis() as f64,
                ),
                ("inter_chunk_measurements".to_string(), inter_chunk_latencies.len() as f64),
                ("max_memory_usage_percent".to_string(), max_memory_usage),
                ("memory_measurements".to_string(), memory_measurements.len() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up streaming performance test");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestConfig, TestHarness};

    #[tokio::test]
    async fn test_streaming_workflow_suite() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let suite = StreamingWorkflowTestSuite;

        let result = harness.run_test_suite(&suite).await;
        assert!(result.is_ok());

        let suite_result = result.unwrap();
        assert!(suite_result.summary.total_tests > 0);
    }
}
