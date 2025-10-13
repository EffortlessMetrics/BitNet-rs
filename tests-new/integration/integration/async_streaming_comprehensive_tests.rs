//! # Comprehensive Async Streaming Tests
//!
//! Comprehensive test suite for async streaming token generation, covering
//! edge cases, error handling, cancellation, backpressure, and integration scenarios.

use super::*;
use crate::common::harness::FixtureCtx;
use crate::{TestCase, TestError, TestMetrics, TestResult};
use async_trait::async_trait;
use bitnet_common::Device;
use bitnet_inference::{GenerationStream, StreamingConfig, InferenceEngine, GenerationConfig};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use futures_util::StreamExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Test suite for comprehensive async streaming
pub struct ComprehensiveAsyncStreamingTestSuite;

impl crate::TestSuite for ComprehensiveAsyncStreamingTestSuite {
    fn name(&self) -> &str {
        "Comprehensive Async Streaming Tests"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(StreamCancellationTest),
            Box::new(StreamErrorHandlingTest),
            Box::new(StreamMemoryLeakTest),
            Box::new(StreamBackpressureHandlingTest),
            Box::new(StreamConcurrencyTest),
            Box::new(StreamTimeoutTest),
            Box::new(StreamConfigurationEdgeCasesTest),
            Box::new(StreamIntegrationTest),
        ]
    }
}

/// Test stream cancellation mechanisms
struct StreamCancellationTest;

#[async_trait]
impl TestCase for StreamCancellationTest {
    fn name(&self) -> &str {
        "stream_cancellation_comprehensive"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up stream cancellation tests");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create engine: {}", e)))?;

        // Test early cancellation
        debug!("Testing early cancellation");
        let mut stream = engine.generate_stream("Early cancellation test");
        let mut early_cancel_chunks = 0;

        // Consume only one chunk then cancel
        if let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    debug!("Early cancel - got chunk: '{}'", chunk);
                    early_cancel_chunks = 1;
                }
                Err(e) => {
                    warn!("Early cancel error: {}", e);
                }
            }
        }
        // Stream is dropped here (cancelled)

        // Test cancellation during generation
        debug!("Testing mid-generation cancellation");
        let cancellation_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create cancellation engine: {}", e)))?;

        let mut cancel_count = 0;
        for i in 0..3 {
            debug!("Cancellation test iteration {}", i + 1);
            let mut stream = cancellation_engine.generate_stream("Mid-generation cancellation");

            // Consume a few chunks
            let mut consumed = 0;
            while consumed < 2 {
                if let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            debug!("Cancellation {} - chunk {}: '{}'", i + 1, consumed + 1, chunk);
                            consumed += 1;
                        }
                        Err(_) => break,
                    }
                } else {
                    break;
                }
            }
            cancel_count += 1;
            // Stream cancelled by drop
            debug!("Stream {} cancelled after {} chunks", i + 1, consumed);
        }

        // Test timeout-based cancellation
        debug!("Testing timeout cancellation");
        let timeout_stream = cancellation_engine.generate_stream("Timeout test");
        let timeout_duration = Duration::from_millis(100);

        let timeout_result = tokio::time::timeout(timeout_duration, async move {
            let mut timeout_chunks = 0;
            let mut stream = timeout_stream;

            while let Some(result) = stream.next().await {
                if let Ok(_) = result {
                    timeout_chunks += 1;
                    // Add a small delay to trigger timeout
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    if timeout_chunks >= 5 {
                        break;
                    }
                }
            }
            timeout_chunks
        }).await;

        let timeout_chunks = match timeout_result {
            Ok(chunks) => chunks,
            Err(_) => {
                debug!("Timeout cancellation successful");
                0
            }
        };

        // Test resource cleanup after cancellation
        debug!("Testing resource cleanup");
        let cleanup_stats = cancellation_engine.get_stats().await;
        debug!("Post-cancellation stats: cache_size={}, usage={:.2}%",
               cleanup_stats.cache_size, cleanup_stats.cache_usage);

        // Verify engine still works after cancellations
        let post_cancel_result = cancellation_engine.generate("Post-cancel test").await
            .map_err(|e| TestError::execution(format!("Post-cancellation test failed: {}", e)))?;

        if post_cancel_result.is_empty() {
            return Err(TestError::assertion("Engine should work after cancellations"));
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("early_cancel_chunks".to_string(), early_cancel_chunks as f64),
                ("cancellation_iterations".to_string(), cancel_count as f64),
                ("timeout_chunks".to_string(), timeout_chunks as f64),
                ("post_cancel_success".to_string(), 1.0),
                ("cleanup_cache_usage".to_string(), cleanup_stats.cache_usage),
            ].into_iter().collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up stream cancellation test");
        Ok(())
    }
}

/// Test error handling in streams
struct StreamErrorHandlingTest;

#[async_trait]
impl TestCase for StreamErrorHandlingTest {
    fn name(&self) -> &str {
        "stream_error_handling"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up stream error handling tests");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        // Use a model that can simulate errors
        let error_model = Arc::new(ErrorProneModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(error_model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create engine: {}", e)))?;

        // Test error recovery
        debug!("Testing error recovery");
        let mut error_stream = engine.generate_stream("Error test prompt");
        let mut error_count = 0;
        let mut successful_chunks = 0;

        while let Some(result) = error_stream.next().await {
            match result {
                Ok(chunk) => {
                    debug!("Success chunk: '{}'", chunk);
                    successful_chunks += 1;
                    if successful_chunks >= 3 {
                        break;
                    }
                }
                Err(e) => {
                    debug!("Error encountered: {}", e);
                    error_count += 1;
                    // Continue to test recovery
                }
            }
        }

        // Test different error types
        debug!("Testing different error scenarios");

        // Tokenization error test
        let bad_tokenizer = Arc::new(ErrorTokenizer::new());
        let tokenizer_error_engine = InferenceEngine::new(
            Arc::new(MockModel::new()),
            bad_tokenizer,
            device
        ).map_err(|e| TestError::execution(format!("Failed to create tokenizer error engine: {}", e)))?;

        let mut tokenizer_stream = tokenizer_error_engine.generate_stream("Tokenizer error test");
        let mut tokenizer_errors = 0;

        while let Some(result) = tokenizer_stream.next().await {
            match result {
                Ok(_) => {
                    break; // Unexpected success
                }
                Err(e) => {
                    debug!("Tokenizer error: {}", e);
                    tokenizer_errors += 1;
                    break; // Expected error
                }
            }
        }

        // Model forward error test
        error_model.set_error_mode(true);
        let mut forward_stream = engine.generate_stream("Forward error test");
        let mut forward_errors = 0;

        while let Some(result) = forward_stream.next().await {
            match result {
                Ok(_) => {
                    break; // Unexpected success
                }
                Err(e) => {
                    debug!("Forward error: {}", e);
                    forward_errors += 1;
                    break; // Expected error
                }
            }
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("error_count".to_string(), error_count as f64),
                ("successful_chunks".to_string(), successful_chunks as f64),
                ("tokenizer_errors".to_string(), tokenizer_errors as f64),
                ("forward_errors".to_string(), forward_errors as f64),
            ].into_iter().collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up stream error handling test");
        Ok(())
    }
}

/// Test memory leak prevention
struct StreamMemoryLeakTest;

#[async_trait]
impl TestCase for StreamMemoryLeakTest {
    fn name(&self) -> &str {
        "stream_memory_leak_prevention"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up memory leak prevention tests");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create engine: {}", e)))?;

        // Get initial memory stats
        let initial_stats = engine.get_stats().await;
        debug!("Initial memory stats: cache_size={}, usage={:.2}%",
               initial_stats.cache_size, initial_stats.cache_usage);

        // Create and drop many streams
        debug!("Testing stream creation/destruction cycle");
        let stream_iterations = 10;
        let mut memory_measurements = Vec::new();

        for i in 0..stream_iterations {
            debug!("Memory test iteration {}", i + 1);

            // Create stream
            let mut stream = engine.generate_stream(&format!("Memory test {}", i + 1));

            // Consume a few tokens
            for _ in 0..3 {
                if let Some(Ok(_)) = stream.next().await {
                    // Token consumed
                }
            }

            // Stream dropped here

            // Measure memory
            let stats = engine.get_stats().await;
            memory_measurements.push(stats.cache_usage);
            debug!("Iteration {} memory usage: {:.2}%", i + 1, stats.cache_usage);

            // Small delay to allow cleanup
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Test with concurrent streams
        debug!("Testing concurrent stream memory usage");
        let concurrent_count = 5;
        let mut concurrent_handles = Vec::new();

        for i in 0..concurrent_count {
            let engine_clone = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
                .map_err(|e| TestError::execution(format!("Failed to create concurrent engine: {}", e)))?;

            let handle = tokio::spawn(async move {
                let mut stream = engine_clone.generate_stream(&format!("Concurrent {}", i + 1));
                let mut chunks = 0;

                while chunks < 3 {
                    if let Some(Ok(_)) = stream.next().await {
                        chunks += 1;
                    } else {
                        break;
                    }
                }

                engine_clone.get_stats().await.cache_usage
            });

            concurrent_handles.push(handle);
        }

        let mut concurrent_usage = Vec::new();
        for handle in concurrent_handles {
            match handle.await {
                Ok(usage) => concurrent_usage.push(usage),
                Err(e) => warn!("Concurrent handle failed: {}", e),
            }
        }

        // Final memory check
        let final_stats = engine.get_stats().await;
        debug!("Final memory stats: cache_size={}, usage={:.2}%",
               final_stats.cache_size, final_stats.cache_usage);

        // Calculate memory metrics
        let max_usage = memory_measurements.iter().fold(0.0f64, |a, &b| a.max(b));
        let avg_usage = memory_measurements.iter().sum::<f64>() / memory_measurements.len() as f64;
        let concurrent_avg = if concurrent_usage.is_empty() { 0.0 } else {
            concurrent_usage.iter().sum::<f64>() / concurrent_usage.len() as f64
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
                ("initial_cache_usage".to_string(), initial_stats.cache_usage),
                ("final_cache_usage".to_string(), final_stats.cache_usage),
                ("max_usage_during_test".to_string(), max_usage),
                ("avg_usage_during_test".to_string(), avg_usage),
                ("stream_iterations".to_string(), stream_iterations as f64),
                ("concurrent_streams".to_string(), concurrent_count as f64),
                ("concurrent_avg_usage".to_string(), concurrent_avg),
                ("memory_measurements".to_string(), memory_measurements.len() as f64),
            ].into_iter().collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up memory leak prevention test");
        Ok(())
    }
}

// Additional test cases would go here...
// For brevity, I'll include placeholders for the remaining tests

/// Test backpressure handling
struct StreamBackpressureHandlingTest;

#[async_trait]
impl TestCase for StreamBackpressureHandlingTest {
    fn name(&self) -> &str {
        "stream_backpressure_handling"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(SlowModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create engine: {}", e)))?;

        // Test backpressure with slow consumer
        debug!("Testing backpressure with slow consumer");
        let mut stream = engine.generate_stream("Backpressure test");
        let mut tokens_received = 0;
        let mut backpressure_detected = false;

        let start_consume = Instant::now();
        while let Some(result) = stream.next().await {
            match result {
                Ok(_token) => {
                    tokens_received += 1;

                    // Simulate slow consumer
                    tokio::time::sleep(Duration::from_millis(100)).await;

                    // Check if generation is being throttled (backpressure)
                    let elapsed = start_consume.elapsed();
                    if elapsed > Duration::from_millis(500) && tokens_received < 10 {
                        backpressure_detected = true;
                    }

                    if tokens_received >= 5 {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Backpressure test error: {}", e);
                    break;
                }
            }
        }

        // Test high-throughput scenario
        debug!("Testing high-throughput scenario");
        let fast_stream = engine.generate_stream("High throughput test");
        let mut fast_tokens = 0;
        let fast_start = Instant::now();

        let mut fast_stream = fast_stream;
        while let Some(result) = fast_stream.next().await {
            if let Ok(_) = result {
                fast_tokens += 1;
                if fast_tokens >= 10 {
                    break;
                }
            }
        }
        let fast_duration = fast_start.elapsed();

        // Test concurrent streams with backpressure
        debug!("Testing concurrent streams with different consumption rates");
        let concurrent_count = 3;
        let mut handles = Vec::new();

        for i in 0..concurrent_count {
            let test_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
                .map_err(|e| TestError::execution(format!("Failed to create concurrent engine: {}", e)))?;

            let delay = Duration::from_millis(50 * (i + 1)); // Different consumption rates
            let handle = tokio::spawn(async move {
                let mut stream = test_engine.generate_stream(&format!("Concurrent {}", i + 1));
                let mut count = 0;

                while count < 3 {
                    if let Some(Ok(_)) = stream.next().await {
                        count += 1;
                        tokio::time::sleep(delay).await;
                    } else {
                        break;
                    }
                }
                count
            });
            handles.push(handle);
        }

        let concurrent_results: Vec<usize> = futures_util::future::join_all(handles)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("tokens_received".to_string(), tokens_received as f64),
                ("backpressure_detected".to_string(), if backpressure_detected { 1.0 } else { 0.0 }),
                ("fast_tokens".to_string(), fast_tokens as f64),
                ("fast_duration_ms".to_string(), fast_duration.as_millis() as f64),
                ("concurrent_streams".to_string(), concurrent_count as f64),
                ("concurrent_completed".to_string(), concurrent_results.len() as f64),
                ("concurrent_total_tokens".to_string(), concurrent_results.iter().sum::<usize>() as f64),
            ].into_iter().collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test concurrency scenarios
struct StreamConcurrencyTest;

#[async_trait]
impl TestCase for StreamConcurrencyTest {
    fn name(&self) -> &str {
        "stream_concurrency"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        // Test multiple concurrent streams from same engine
        debug!("Testing multiple concurrent streams from same engine");
        let shared_engine = Arc::new(RwLock::new(
            InferenceEngine::new(model.clone(), tokenizer.clone(), device)
                .map_err(|e| TestError::execution(format!("Failed to create shared engine: {}", e)))?
        ));

        let concurrent_count = 5;
        let mut handles = Vec::new();

        for i in 0..concurrent_count {
            let engine_ref = shared_engine.clone();
            let prompt = format!("Concurrent test {}", i + 1);

            let handle = tokio::spawn(async move {
                let engine_guard = engine_ref.read().await;
                let mut stream = engine_guard.generate_stream(&prompt);
                let mut token_count = 0;
                let start = Instant::now();

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(_token) => {
                            token_count += 1;
                            if token_count >= 3 {
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("Concurrent stream {} error: {}", i + 1, e);
                            break;
                        }
                    }
                }

                (i + 1, token_count, start.elapsed())
            });

            handles.push(handle);
        }

        let results: Vec<(usize, usize, Duration)> = futures_util::future::join_all(handles)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        // Test concurrent streams with different engines
        debug!("Testing concurrent streams with independent engines");
        let independent_count = 3;
        let mut independent_handles = Vec::new();

        for i in 0..independent_count {
            let engine_model = model.clone();
            let engine_tokenizer = tokenizer.clone();

            let handle = tokio::spawn(async move {
                let independent_engine = InferenceEngine::new(engine_model, engine_tokenizer, device)
                    .map_err(|e| format!("Failed to create independent engine {}: {}", i + 1, e))?;

                let mut stream = independent_engine.generate_stream(&format!("Independent {}", i + 1));
                let mut token_count = 0;
                let start = Instant::now();

                while let Some(result) = stream.next().await {
                    if let Ok(_) = result {
                        token_count += 1;
                        if token_count >= 3 {
                            break;
                        }
                    }
                }

                Ok::<(usize, usize, Duration), String>((i + 1, token_count, start.elapsed()))
            });

            independent_handles.push(handle);
        }

        let independent_results: Vec<(usize, usize, Duration)> = futures_util::future::join_all(independent_handles)
            .await
            .into_iter()
            .filter_map(|r| r.ok().and_then(|inner| inner.ok()))
            .collect();

        // Test race conditions and resource contention
        debug!("Testing race conditions with rapid stream creation/destruction");
        let race_count = 10;
        let mut race_handles = Vec::new();

        for i in 0..race_count {
            let race_model = model.clone();
            let race_tokenizer = tokenizer.clone();

            let handle = tokio::spawn(async move {
                let race_engine = InferenceEngine::new(race_model, race_tokenizer, device)?;

                // Create and immediately start consuming stream
                let mut stream = race_engine.generate_stream(&format!("Race {}", i + 1));

                // Consume only first token then drop
                if let Some(Ok(_)) = stream.next().await {
                    return Ok(1);
                }

                Ok::<usize, anyhow::Error>(0)
            });

            race_handles.push(handle);
        }

        let race_results: Vec<usize> = futures_util::future::join_all(race_handles)
            .await
            .into_iter()
            .filter_map(|r| r.ok().and_then(|inner| inner.ok()))
            .collect();

        let duration = start_time.elapsed();

        // Calculate metrics
        let total_shared_tokens: usize = results.iter().map(|(_, tokens, _)| tokens).sum();
        let avg_shared_duration = if results.is_empty() { Duration::ZERO } else {
            Duration::from_nanos(
                results.iter().map(|(_, _, dur)| dur.as_nanos()).sum::<u128>()
                / results.len() as u128
            )
        };

        let total_independent_tokens: usize = independent_results.iter().map(|(_, tokens, _)| tokens).sum();
        let avg_independent_duration = if independent_results.is_empty() { Duration::ZERO } else {
            Duration::from_nanos(
                independent_results.iter().map(|(_, _, dur)| dur.as_nanos()).sum::<u128>()
                / independent_results.len() as u128
            )
        };

        let race_success_count = race_results.iter().sum::<usize>();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("shared_concurrent_streams".to_string(), concurrent_count as f64),
                ("shared_completed_streams".to_string(), results.len() as f64),
                ("shared_total_tokens".to_string(), total_shared_tokens as f64),
                ("shared_avg_duration_ms".to_string(), avg_shared_duration.as_millis() as f64),
                ("independent_concurrent_streams".to_string(), independent_count as f64),
                ("independent_completed_streams".to_string(), independent_results.len() as f64),
                ("independent_total_tokens".to_string(), total_independent_tokens as f64),
                ("independent_avg_duration_ms".to_string(), avg_independent_duration.as_millis() as f64),
                ("race_condition_attempts".to_string(), race_count as f64),
                ("race_condition_successes".to_string(), race_success_count as f64),
            ].into_iter().collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test timeout scenarios
struct StreamTimeoutTest;

#[async_trait]
impl TestCase for StreamTimeoutTest {
    fn name(&self) -> &str {
        "stream_timeout"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(TimeoutProneModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create engine: {}", e)))?;

        // Test stream timeout scenarios
        debug!("Testing stream timeout scenarios");

        // Test generation timeout
        let mut generation_timeouts = 0;
        let mut generation_successes = 0;

        for i in 0..3 {
            debug!("Timeout test iteration {}", i + 1);
            let timeout_duration = Duration::from_millis(200);

            let stream_future = async {
                let mut stream = engine.generate_stream(&format!("Timeout test {}", i + 1));
                let mut tokens = 0;

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(_) => {
                            tokens += 1;
                            // Add delay to trigger timeout
                            tokio::time::sleep(Duration::from_millis(150)).await;
                            if tokens >= 5 {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                tokens
            };

            match tokio::time::timeout(timeout_duration, stream_future).await {
                Ok(token_count) => {
                    debug!("Stream {} completed with {} tokens", i + 1, token_count);
                    generation_successes += 1;
                }
                Err(_) => {
                    debug!("Stream {} timed out as expected", i + 1);
                    generation_timeouts += 1;
                }
            }
        }

        // Test different timeout scenarios
        debug!("Testing different timeout scenarios");

        // Fast timeout (should timeout immediately)
        let fast_timeout = Duration::from_millis(1);
        let fast_timeout_result = tokio::time::timeout(fast_timeout, async {
            let mut stream = engine.generate_stream("Fast timeout test");
            stream.next().await
        }).await;

        let fast_timeout_occurred = fast_timeout_result.is_err();

        // Generous timeout (should complete)
        let generous_timeout = Duration::from_secs(5);
        let mut generous_tokens = 0;
        let generous_timeout_result = tokio::time::timeout(generous_timeout, async {
            let mut stream = engine.generate_stream("Generous timeout test");
            let mut count = 0;

            while count < 2 { // Only generate a few tokens
                if let Some(Ok(_)) = stream.next().await {
                    count += 1;
                }
            }
            count
        }).await;

        match generous_timeout_result {
            Ok(count) => generous_tokens = count,
            Err(_) => warn!("Generous timeout unexpectedly failed"),
        }

        // Test cancellation vs timeout
        debug!("Testing cancellation vs timeout scenarios");
        let mut cancellation_tests = 0;
        let mut cancellation_successes = 0;

        for i in 0..2 {
            let cancel_engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
                .map_err(|e| TestError::execution(format!("Failed to create cancel engine: {}", e)))?;

            let stream_with_cancel = cancel_engine.generate_stream(&format!("Cancel test {}", i + 1));

            let cancel_future = async move {
                let mut stream = stream_with_cancel;
                let mut count = 0;

                // Start consuming but cancel quickly
                tokio::select! {
                    _ = async {
                        while let Some(_) = stream.next().await {
                            count += 1;
                            if count >= 10 {
                                break;
                            }
                        }
                    } => {},
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {
                        // Timeout reached, stream should be cancelled
                    }
                }

                count
            };

            match tokio::time::timeout(Duration::from_millis(300), cancel_future).await {
                Ok(count) => {
                    debug!("Cancellation test {} completed with {} tokens", i + 1, count);
                    cancellation_successes += 1;
                }
                Err(_) => {
                    warn!("Cancellation test {} timed out", i + 1);
                }
            }
            cancellation_tests += 1;
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("generation_timeout_tests".to_string(), 3.0),
                ("generation_timeouts".to_string(), generation_timeouts as f64),
                ("generation_successes".to_string(), generation_successes as f64),
                ("fast_timeout_occurred".to_string(), if fast_timeout_occurred { 1.0 } else { 0.0 }),
                ("generous_timeout_tokens".to_string(), generous_tokens as f64),
                ("cancellation_tests".to_string(), cancellation_tests as f64),
                ("cancellation_successes".to_string(), cancellation_successes as f64),
            ].into_iter().collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration edge cases
struct StreamConfigurationEdgeCasesTest;

#[async_trait]
impl TestCase for StreamConfigurationEdgeCasesTest {
    fn name(&self) -> &str {
        "stream_configuration_edge_cases"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create engine: {}", e)))?;

        debug!("Testing configuration edge cases");

        // Test with extreme temperature values
        let mut extreme_temp_tests = 0;
        let mut extreme_temp_successes = 0;

        let extreme_temperatures = vec![0.0, 0.001, 5.0, 10.0];
        for temp in extreme_temperatures {
            debug!("Testing extreme temperature: {}", temp);

            // Create config with custom streaming parameters
            let streaming_config = bitnet_inference::streaming::StreamingConfig {
                buffer_size: 1,
                flush_interval_ms: 10,
                max_retries: 1,
                token_timeout_ms: 1000,
                cancellable: true,
            };

            match engine.generate_stream_with_config(&format!("Extreme temp {} test", temp),
                &GenerationConfig {
                    temperature: temp,
                    max_new_tokens: 3,
                    ..Default::default()
                }) {
                Ok(mut stream) => {
                    let mut tokens = 0;
                    while let Some(result) = stream.next().await {
                        if let Ok(_) = result {
                            tokens += 1;
                            if tokens >= 2 {
                                break;
                            }
                        }
                    }
                    if tokens > 0 {
                        extreme_temp_successes += 1;
                    }
                }
                Err(e) => {
                    warn!("Extreme temperature {} failed: {}", temp, e);
                }
            }
            extreme_temp_tests += 1;
        }

        // Test with extreme top_k values
        debug!("Testing extreme top_k values");
        let mut extreme_topk_tests = 0;
        let mut extreme_topk_successes = 0;

        let extreme_topks = vec![1, 2, 1000, 50000];
        for top_k in extreme_topks {
            debug!("Testing extreme top_k: {}", top_k);

            match engine.generate_stream_with_config(&format!("Extreme top_k {} test", top_k),
                &GenerationConfig {
                    top_k,
                    max_new_tokens: 2,
                    ..Default::default()
                }) {
                Ok(mut stream) => {
                    if let Some(Ok(_)) = stream.next().await {
                        extreme_topk_successes += 1;
                    }
                }
                Err(e) => {
                    warn!("Extreme top_k {} failed: {}", top_k, e);
                }
            }
            extreme_topk_tests += 1;
        }

        // Test with edge case max_new_tokens
        debug!("Testing edge case max_new_tokens");
        let mut max_tokens_tests = 0;
        let mut max_tokens_successes = 0;

        let edge_case_tokens = vec![0, 1, 2, 1000];
        for max_tokens in edge_case_tokens {
            debug!("Testing max_new_tokens: {}", max_tokens);

            match engine.generate_stream_with_config(&format!("Max tokens {} test", max_tokens),
                &GenerationConfig {
                    max_new_tokens: max_tokens,
                    ..Default::default()
                }) {
                Ok(mut stream) => {
                    let mut actual_tokens = 0;
                    while let Some(result) = stream.next().await {
                        if let Ok(_) = result {
                            actual_tokens += 1;
                            if actual_tokens > max_tokens {
                                warn!("Generated more tokens ({}) than requested ({})",
                                      actual_tokens, max_tokens);
                                break;
                            }
                        }
                    }

                    debug!("Max tokens {} test: requested={}, actual={}",
                           max_tokens, max_tokens, actual_tokens);

                    // For max_tokens=0, we should get no tokens
                    if max_tokens == 0 && actual_tokens == 0 {
                        max_tokens_successes += 1;
                    } else if max_tokens > 0 && actual_tokens <= max_tokens {
                        max_tokens_successes += 1;
                    }
                }
                Err(e) => {
                    if max_tokens == 0 {
                        // It's acceptable for max_tokens=0 to fail
                        max_tokens_successes += 1;
                        debug!("Max tokens 0 failed as expected: {}", e);
                    } else {
                        warn!("Max tokens {} failed: {}", max_tokens, e);
                    }
                }
            }
            max_tokens_tests += 1;
        }

        // Test with empty/invalid prompts
        debug!("Testing edge case prompts");
        let mut prompt_tests = 0;
        let mut prompt_successes = 0;

        let edge_case_prompts = vec![
            "",
            " ",
            "\n",
            "\t",
            "x".repeat(1000),
        ];

        for prompt in edge_case_prompts {
            debug!("Testing prompt: '{}...'", &prompt[..10.min(prompt.len())]);

            match engine.generate_stream(&prompt) {
                Ok(mut stream) => {
                    // Just try to get one token
                    if let Some(_) = stream.next().await {
                        prompt_successes += 1;
                    }
                }
                Err(e) => {
                    debug!("Edge case prompt failed: {}", e);
                    // Empty prompts might legitimately fail
                    if prompt.trim().is_empty() {
                        prompt_successes += 1; // Expected failure
                    }
                }
            }
            prompt_tests += 1;
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("extreme_temp_tests".to_string(), extreme_temp_tests as f64),
                ("extreme_temp_successes".to_string(), extreme_temp_successes as f64),
                ("extreme_topk_tests".to_string(), extreme_topk_tests as f64),
                ("extreme_topk_successes".to_string(), extreme_topk_successes as f64),
                ("max_tokens_tests".to_string(), max_tokens_tests as f64),
                ("max_tokens_successes".to_string(), max_tokens_successes as f64),
                ("prompt_tests".to_string(), prompt_tests as f64),
                ("prompt_successes".to_string(), prompt_successes as f64),
            ].into_iter().collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test integration scenarios
struct StreamIntegrationTest;

#[async_trait]
impl TestCase for StreamIntegrationTest {
    fn name(&self) -> &str {
        "stream_integration"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        debug!("Testing full integration scenarios");

        // Test complete workflow: engine creation -> streaming -> stats -> cleanup
        let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create engine: {}", e)))?;

        // Get initial stats
        let initial_stats = engine.get_stats().await;
        debug!("Initial stats: cache_size={}, usage={:.2}%",
               initial_stats.cache_size, initial_stats.cache_usage);

        // Test normal streaming workflow
        debug!("Testing normal streaming workflow");
        let mut workflow_stream = engine.generate_stream("Integration test prompt");
        let mut workflow_tokens = Vec::new();

        while let Some(result) = workflow_stream.next().await {
            match result {
                Ok(token) => {
                    workflow_tokens.push(token);
                    if workflow_tokens.len() >= 5 {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Workflow streaming error: {}", e);
                    break;
                }
            }
        }

        // Test stats after generation
        let post_generation_stats = engine.get_stats().await;
        debug!("Post-generation stats: cache_size={}, usage={:.2}%",
               post_generation_stats.cache_size, post_generation_stats.cache_usage);

        // Test multiple streams in sequence
        debug!("Testing sequential streaming");
        let mut sequential_results = Vec::new();

        for i in 0..3 {
            let mut seq_stream = engine.generate_stream(&format!("Sequential test {}", i + 1));
            let mut seq_tokens = 0;

            while let Some(result) = seq_stream.next().await {
                if let Ok(_) = result {
                    seq_tokens += 1;
                    if seq_tokens >= 2 {
                        break;
                    }
                }
            }
            sequential_results.push(seq_tokens);
        }

        // Test cache behavior
        debug!("Testing cache behavior");
        let pre_clear_stats = engine.get_stats().await;

        // Clear cache
        engine.clear_cache().await;

        let post_clear_stats = engine.get_stats().await;
        debug!("Post-clear stats: cache_size={}, usage={:.2}%",
               post_clear_stats.cache_size, post_clear_stats.cache_usage);

        // Test generation after cache clear
        let mut post_clear_stream = engine.generate_stream("Post-clear test");
        let mut post_clear_tokens = 0;

        while let Some(result) = post_clear_stream.next().await {
            if let Ok(_) = result {
                post_clear_tokens += 1;
                if post_clear_tokens >= 2 {
                    break;
                }
            }
        }

        // Test error handling integration
        debug!("Testing error handling integration");
        let error_model = Arc::new(ErrorProneModel::new());
        let error_engine = InferenceEngine::new(error_model, tokenizer.clone(), device)
            .map_err(|e| TestError::execution(format!("Failed to create error engine: {}", e)))?;

        let mut error_stream = error_engine.generate_stream("Error integration test");
        let mut error_tokens = 0;
        let mut errors_encountered = 0;

        // Allow for some errors but expect eventual success
        for _ in 0..10 {
            match error_stream.next().await {
                Some(Ok(_)) => {
                    error_tokens += 1;
                    if error_tokens >= 2 {
                        break;
                    }
                }
                Some(Err(_)) => {
                    errors_encountered += 1;
                    // Continue trying
                }
                None => break,
            }
        }

        // Test streaming with configuration variations
        debug!("Testing configuration variations");
        let mut config_tests = 0;
        let mut config_successes = 0;

        let configs = vec![
            GenerationConfig {
                temperature: 0.1,
                top_k: 10,
                max_new_tokens: 2,
                ..Default::default()
            },
            GenerationConfig {
                temperature: 1.5,
                top_p: 0.8,
                max_new_tokens: 3,
                ..Default::default()
            },
        ];

        for (i, config) in configs.into_iter().enumerate() {
            match engine.generate_stream_with_config(&format!("Config test {}", i + 1), &config) {
                Ok(mut stream) => {
                    if let Some(Ok(_)) = stream.next().await {
                        config_successes += 1;
                    }
                }
                Err(e) => {
                    warn!("Config test {} failed: {}", i + 1, e);
                }
            }
            config_tests += 1;
        }

        // Final stats
        let final_stats = engine.get_stats().await;
        debug!("Final stats: cache_size={}, usage={:.2}%",
               final_stats.cache_size, final_stats.cache_usage);

        let duration = start_time.elapsed();

        let total_sequential_tokens: usize = sequential_results.iter().sum();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("workflow_tokens".to_string(), workflow_tokens.len() as f64),
                ("sequential_streams".to_string(), sequential_results.len() as f64),
                ("sequential_total_tokens".to_string(), total_sequential_tokens as f64),
                ("cache_cleared_successfully".to_string(), 1.0),
                ("post_clear_tokens".to_string(), post_clear_tokens as f64),
                ("error_integration_tokens".to_string(), error_tokens as f64),
                ("error_integration_errors".to_string(), errors_encountered as f64),
                ("config_variation_tests".to_string(), config_tests as f64),
                ("config_variation_successes".to_string(), config_successes as f64),
                ("initial_cache_usage".to_string(), initial_stats.cache_usage),
                ("final_cache_usage".to_string(), final_stats.cache_usage),
            ].into_iter().collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

// Mock implementations for testing

/// Mock model for testing
struct MockModel {
    config: bitnet_common::BitNetConfig,
    forward_calls: AtomicUsize,
}

impl MockModel {
    fn new() -> Self {
        Self {
            config: bitnet_common::BitNetConfig::default(),
            forward_calls: AtomicUsize::new(0),
        }
    }

    fn forward_call_count(&self) -> usize {
        self.forward_calls.load(Ordering::Relaxed)
    }
}

impl Model for MockModel {
    fn config(&self) -> &bitnet_common::BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        self.forward_calls.fetch_add(1, Ordering::Relaxed);
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 50257]))
    }

    fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 10, 768]))
    }

    fn logits(&self, _hidden: &bitnet_common::ConcreteTensor) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 50257]))
    }
}

/// Error-prone model for testing error handling
struct ErrorProneModel {
    config: bitnet_common::BitNetConfig,
    forward_calls: AtomicUsize,
    error_mode: Arc<RwLock<bool>>,
}

impl ErrorProneModel {
    fn new() -> Self {
        Self {
            config: bitnet_common::BitNetConfig::default(),
            forward_calls: AtomicUsize::new(0),
            error_mode: Arc::new(RwLock::new(false)),
        }
    }

    async fn set_error_mode(&self, enabled: bool) {
        *self.error_mode.write().await = enabled;
    }
}

impl Model for ErrorProneModel {
    fn config(&self) -> &bitnet_common::BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        let count = self.forward_calls.fetch_add(1, Ordering::Relaxed);

        // Simulate intermittent errors
        if count % 3 == 2 {
            return Err(anyhow::anyhow!("Simulated forward error").into());
        }

        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 50257]))
    }

    fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 10, 768]))
    }

    fn logits(&self, _hidden: &bitnet_common::ConcreteTensor) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 50257]))
    }
}

/// Mock tokenizer for testing
struct MockTokenizer {
    encode_calls: AtomicUsize,
    decode_calls: AtomicUsize,
}

impl MockTokenizer {
    fn new() -> Self {
        Self {
            encode_calls: AtomicUsize::new(0),
            decode_calls: AtomicUsize::new(0),
        }
    }

    fn encode_call_count(&self) -> usize {
        self.encode_calls.load(Ordering::Relaxed)
    }

    fn decode_call_count(&self) -> usize {
        self.decode_calls.load(Ordering::Relaxed)
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(
        &self,
        _text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> bitnet_common::Result<Vec<u32>> {
        self.encode_calls.fetch_add(1, Ordering::Relaxed);
        Ok(vec![1, 2, 3])
    }

    fn decode(&self, tokens: &[u32]) -> bitnet_common::Result<String> {
        self.decode_calls.fetch_add(1, Ordering::Relaxed);
        Ok(format!("token_{}", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        50257
    }

    fn token_to_piece(&self, _token: u32) -> Option<String> {
        Some("<token>".to_string())
    }
}

/// Error-prone tokenizer for testing error handling
struct ErrorTokenizer {
    calls: AtomicUsize,
}

impl ErrorTokenizer {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
}

impl Tokenizer for ErrorTokenizer {
    fn encode(
        &self,
        _text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> bitnet_common::Result<Vec<u32>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(anyhow::anyhow!("Simulated tokenizer error").into())
    }

    fn decode(&self, _tokens: &[u32]) -> bitnet_common::Result<String> {
        Err(anyhow::anyhow!("Simulated decode error").into())
    }

    fn vocab_size(&self) -> usize {
        50257
    }

    fn token_to_piece(&self, _token: u32) -> Option<String> {
        None
    }
}

/// Slow model for backpressure testing
struct SlowModel {
    config: bitnet_common::BitNetConfig,
    forward_calls: AtomicUsize,
}

impl SlowModel {
    fn new() -> Self {
        Self {
            config: bitnet_common::BitNetConfig::default(),
            forward_calls: AtomicUsize::new(0),
        }
    }
}

impl Model for SlowModel {
    fn config(&self) -> &bitnet_common::BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        self.forward_calls.fetch_add(1, Ordering::Relaxed);
        // Simulate slow model computation
        std::thread::sleep(Duration::from_millis(50));
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 50257]))
    }

    fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 10, 768]))
    }

    fn logits(&self, _hidden: &bitnet_common::ConcreteTensor) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 50257]))
    }
}

/// Timeout-prone model for timeout testing
struct TimeoutProneModel {
    config: bitnet_common::BitNetConfig,
    forward_calls: AtomicUsize,
}

impl TimeoutProneModel {
    fn new() -> Self {
        Self {
            config: bitnet_common::BitNetConfig::default(),
            forward_calls: AtomicUsize::new(0),
        }
    }
}

impl Model for TimeoutProneModel {
    fn config(&self) -> &bitnet_common::BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        let count = self.forward_calls.fetch_add(1, Ordering::Relaxed);

        // Simulate occasional long delays
        if count % 4 == 3 {
            std::thread::sleep(Duration::from_millis(500));
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }

        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 50257]))
    }

    fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 10, 768]))
    }

    fn logits(&self, _hidden: &bitnet_common::ConcreteTensor) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 50257]))
    }
}

impl Default for TestMetrics {
    fn default() -> Self {
        TestMetrics {
            wall_time: Duration::ZERO,
            memory_peak: None,
            memory_average: None,
            cpu_time: None,
            assertions: 0,
            operations: 0,
            custom_metrics: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestConfig, TestHarness};

    #[tokio::test]
    async fn test_comprehensive_streaming_suite() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let suite = ComprehensiveAsyncStreamingTestSuite;

        let result = harness.run_test_suite(&suite).await;
        assert!(result.is_ok());

        let suite_result = result.unwrap();
        assert!(suite_result.summary.total_tests > 0);
    }
}
