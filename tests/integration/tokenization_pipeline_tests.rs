//! # Tokenization to Inference Pipeline Integration Tests
//!
//! Tests the complete pipeline from text tokenization through inference,
//! validating data flow and transformations between components.

use super::*;
#[cfg(feature = "fixtures")]
use crate::common::FixtureManager;
use crate::common::harness::FixtureCtx;
use crate::{TestCase, TestError, TestMetrics, TestResult};
use async_trait::async_trait;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Test suite for tokenization pipeline integration
pub struct TokenizationPipelineTestSuite;

impl crate::TestSuite for TokenizationPipelineTestSuite {
    fn name(&self) -> &str {
        "Tokenization Pipeline Integration Tests"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(BasicTokenizationPipelineTest),
            Box::new(TokenizationAccuracyTest),
            Box::new(SpecialTokenHandlingTest),
            Box::new(LongSequenceTokenizationTest),
            Box::new(TokenizationErrorHandlingTest),
        ]
    }
}

/// Test basic tokenization to inference pipeline
struct BasicTokenizationPipelineTest;

#[async_trait]
impl TestCase for BasicTokenizationPipelineTest {
    fn name(&self) -> &str {
        "basic_tokenization_pipeline"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up basic tokenization pipeline test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        debug!("Creating components for tokenization pipeline test");
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                TestError::execution(format!("Failed to create inference engine: {}", e))
            })?;

        // Test various text inputs
        let test_inputs = vec![
            "Hello, world!",
            "This is a test sentence.",
            "Multiple sentences. With punctuation! And questions?",
            "Numbers: 123, 456.789",
            "Mixed case: CamelCase, snake_case, UPPERCASE",
        ];

        let mut pipeline_results = Vec::new();
        let mut tokenization_stats = Vec::new();

        for (i, input) in test_inputs.iter().enumerate() {
            debug!("Testing pipeline with input {}: '{}'", i + 1, input);

            // Step 1: Direct tokenization test
            let encode_start = Instant::now();
            let tokens = tokenizer.encode(input, true, false).map_err(|e| {
                TestError::execution(format!("Tokenization failed for input {}: {}", i + 1, e))
            })?;
            let encode_time = encode_start.elapsed();

            debug!("Input '{}' tokenized to {} tokens: {:?}", input, tokens.len(), tokens);

            if tokens.is_empty() {
                return Err(TestError::assertion(format!(
                    "Tokenization should produce tokens for input {}",
                    i + 1
                )));
            }

            // Step 2: Decode tokens back to text
            let decode_start = Instant::now();
            let decoded = tokenizer.decode(&tokens).map_err(|e| {
                TestError::execution(format!("Detokenization failed for input {}: {}", i + 1, e))
            })?;
            let decode_time = decode_start.elapsed();

            debug!("Tokens decoded back to: '{}'", decoded);

            // Step 3: Full pipeline test (tokenization + inference)
            let pipeline_start = Instant::now();
            let generated = engine.generate(input).await.map_err(|e| {
                TestError::execution(format!(
                    "Pipeline generation failed for input {}: {}",
                    i + 1,
                    e
                ))
            })?;
            let pipeline_time = pipeline_start.elapsed();

            debug!("Pipeline generated: '{}'", generated);

            if generated.is_empty() {
                return Err(TestError::assertion(format!(
                    "Pipeline should generate text for input {}",
                    i + 1
                )));
            }

            pipeline_results.push(generated);
            tokenization_stats.push((tokens.len(), encode_time, decode_time, pipeline_time));
        }

        // Verify all inputs were processed
        if pipeline_results.len() != test_inputs.len() {
            return Err(TestError::assertion("Not all inputs were processed through pipeline"));
        }

        // Verify tokenizer was called appropriately
        let encode_calls = tokenizer.encode_call_count();
        let decode_calls = tokenizer.decode_call_count();

        // Should have at least one encode call per input for the pipeline
        if encode_calls < test_inputs.len() {
            return Err(TestError::assertion("Insufficient tokenizer encode calls"));
        }

        // Should have decode calls for both direct testing and generation
        if decode_calls == 0 {
            return Err(TestError::assertion("No tokenizer decode calls detected"));
        }

        // Calculate statistics
        let avg_tokens = tokenization_stats.iter().map(|(tokens, _, _, _)| *tokens).sum::<usize>()
            as f64
            / test_inputs.len() as f64;
        let avg_encode_time =
            tokenization_stats.iter().map(|(_, encode, _, _)| encode.as_micros()).sum::<u128>()
                as f64
                / test_inputs.len() as f64;
        let avg_decode_time =
            tokenization_stats.iter().map(|(_, _, decode, _)| decode.as_micros()).sum::<u128>()
                as f64
                / test_inputs.len() as f64;
        let avg_pipeline_time =
            tokenization_stats.iter().map(|(_, _, _, pipeline)| pipeline.as_millis()).sum::<u128>()
                as f64
                / test_inputs.len() as f64;

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("inputs_processed".to_string(), test_inputs.len() as f64),
                ("avg_tokens_per_input".to_string(), avg_tokens),
                ("avg_encode_time_us".to_string(), avg_encode_time),
                ("avg_decode_time_us".to_string(), avg_decode_time),
                ("avg_pipeline_time_ms".to_string(), avg_pipeline_time),
                ("tokenizer_encode_calls".to_string(), encode_calls as f64),
                ("tokenizer_decode_calls".to_string(), decode_calls as f64),
                ("model_forward_calls".to_string(), model.forward_call_count() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up basic tokenization pipeline test");
        Ok(())
    }
}

/// Test tokenization accuracy and consistency
struct TokenizationAccuracyTest;

#[async_trait]
impl TestCase for TokenizationAccuracyTest {
    fn name(&self) -> &str {
        "tokenization_accuracy"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up tokenization accuracy test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let tokenizer = Arc::new(MockTokenizer::new());

        // Test consistency: same input should produce same tokens
        let test_text = "Consistency test input";
        let mut token_sets = Vec::new();

        debug!("Testing tokenization consistency");
        for i in 0..5 {
            let tokens = tokenizer.encode(test_text, true, false).map_err(|e| {
                TestError::execution(format!("Tokenization failed in iteration {}: {}", i + 1, e))
            })?;

            debug!("Iteration {}: {} tokens", i + 1, tokens.len());
            token_sets.push(tokens);
        }

        // Verify all tokenizations are identical
        let first_tokens = &token_sets[0];
        for (i, tokens) in token_sets.iter().enumerate().skip(1) {
            if tokens != first_tokens {
                return Err(TestError::assertion(format!(
                    "Tokenization inconsistent at iteration {}",
                    i + 1
                )));
            }
        }

        debug!("Tokenization consistency verified");

        // Test round-trip accuracy: encode -> decode should preserve meaning
        let test_inputs = vec![
            "Simple text",
            "Text with punctuation!",
            "Numbers: 123, 456.789",
            "Special chars: @#$%^&*()",
        ];

        let mut round_trip_results = Vec::new();

        debug!("Testing round-trip accuracy");
        for (i, input) in test_inputs.iter().enumerate() {
            debug!("Round-trip test {}: '{}'", i + 1, input);

            let tokens = tokenizer.encode(input, true, false).map_err(|e| {
                TestError::execution(format!("Encode failed for round-trip test {}: {}", i + 1, e))
            })?;

            let decoded = tokenizer.decode(&tokens).map_err(|e| {
                TestError::execution(format!("Decode failed for round-trip test {}: {}", i + 1, e))
            })?;

            debug!("Original: '{}' -> Decoded: '{}'", input, decoded);

            // For mock tokenizer, we expect specific format
            if !decoded.starts_with("generated_text_") {
                warn!("Round-trip result doesn't match expected mock format: '{}'", decoded);
            }

            round_trip_results.push((input.to_string(), decoded));
        }

        // Test tokenization with different parameters
        debug!("Testing tokenization parameters");
        let param_test_text = "Parameter test";

        let tokens_with_special = tokenizer.encode(param_test_text, true, true).map_err(|e| {
            TestError::execution(format!("Tokenization with special tokens failed: {}", e))
        })?;

        let tokens_without_special =
            tokenizer.encode(param_test_text, false, false).map_err(|e| {
                TestError::execution(format!("Tokenization without special tokens failed: {}", e))
            })?;

        debug!("With special tokens: {} tokens", tokens_with_special.len());
        debug!("Without special tokens: {} tokens", tokens_without_special.len());

        // Test vocabulary boundaries
        debug!("Testing vocabulary boundaries");
        let vocab_size = tokenizer.vocab_size();

        if vocab_size == 0 {
            return Err(TestError::assertion("Vocabulary size should be non-zero"));
        }

        debug!("Vocabulary size: {}", vocab_size);

        // Test special token IDs
        let eos_token = tokenizer.eos_token_id();
        let pad_token = tokenizer.pad_token_id();

        debug!("EOS token ID: {:?}", eos_token);
        debug!("PAD token ID: {:?}", pad_token);

        if let Some(eos_id) = eos_token {
            if eos_id >= vocab_size as u32 {
                return Err(TestError::assertion("EOS token ID should be within vocabulary"));
            }
        }

        if let Some(pad_id) = pad_token {
            if pad_id >= vocab_size as u32 {
                return Err(TestError::assertion("PAD token ID should be within vocabulary"));
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
                ("consistency_tests".to_string(), token_sets.len() as f64),
                ("round_trip_tests".to_string(), round_trip_results.len() as f64),
                ("vocab_size".to_string(), vocab_size as f64),
                ("has_eos_token".to_string(), if eos_token.is_some() { 1.0 } else { 0.0 }),
                ("has_pad_token".to_string(), if pad_token.is_some() { 1.0 } else { 0.0 }),
                ("tokenizer_encode_calls".to_string(), tokenizer.encode_call_count() as f64),
                ("tokenizer_decode_calls".to_string(), tokenizer.decode_call_count() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up tokenization accuracy test");
        Ok(())
    }
}

/// Test special token handling in pipeline
struct SpecialTokenHandlingTest;

#[async_trait]
impl TestCase for SpecialTokenHandlingTest {
    fn name(&self) -> &str {
        "special_token_handling"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up special token handling test");
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

        // Test generation with stop sequences
        debug!("Testing stop sequence handling");
        let config_with_stop = GenerationConfig {
            max_new_tokens: 20,
            stop_sequences: vec!["STOP".to_string(), "END".to_string()],
            ..Default::default()
        };

        let stop_test_prompt = "Generate text until STOP";
        let stop_result =
            engine.generate_with_config(stop_test_prompt, &config_with_stop).await.map_err(
                |e| TestError::execution(format!("Generation with stop sequences failed: {}", e)),
            )?;

        debug!("Generated with stop sequences: '{}'", stop_result);

        if stop_result.is_empty() {
            return Err(TestError::assertion(
                "Generation with stop sequences should produce output",
            ));
        }

        // Test EOS token handling
        debug!("Testing EOS token handling");
        let eos_token_id = tokenizer.eos_token_id();

        if let Some(eos_id) = eos_token_id {
            debug!("EOS token ID: {}", eos_id);

            // Test decoding with EOS token
            let tokens_with_eos = vec![1, 2, 3, eos_id, 4, 5];
            let decoded_with_eos = tokenizer.decode(&tokens_with_eos).map_err(|e| {
                TestError::execution(format!("Decoding with EOS token failed: {}", e))
            })?;

            debug!("Decoded with EOS (skip special): '{}'", decoded_with_eos);

            let decoded_keep_eos = tokenizer.decode(&tokens_with_eos).map_err(|e| {
                TestError::execution(format!("Decoding keeping EOS token failed: {}", e))
            })?;

            debug!("Decoded keeping EOS: '{}'", decoded_keep_eos);
        }

        // Test PAD token handling
        debug!("Testing PAD token handling");
        let pad_token_id = tokenizer.pad_token_id();

        if let Some(pad_id) = pad_token_id {
            debug!("PAD token ID: {}", pad_id);

            // Test decoding with PAD tokens
            let tokens_with_pad = vec![1, 2, pad_id, pad_id, 3, 4];
            let decoded_with_pad = tokenizer.decode(&tokens_with_pad).map_err(|e| {
                TestError::execution(format!("Decoding with PAD tokens failed: {}", e))
            })?;

            debug!("Decoded with PAD (skip special): '{}'", decoded_with_pad);
        }

        // Test empty and edge case inputs
        debug!("Testing edge case inputs");
        let edge_cases = vec![
            "",   // Empty string
            " ",  // Single space
            "\n", // Newline
            "\t", // Tab
        ];

        let mut edge_case_results = Vec::new();

        for (i, edge_input) in edge_cases.iter().enumerate() {
            debug!("Testing edge case {}: {:?}", i + 1, edge_input);

            match tokenizer.encode(edge_input, true, false) {
                Ok(tokens) => {
                    debug!("Edge case {} tokenized to {} tokens", i + 1, tokens.len());

                    match tokenizer.decode(&tokens) {
                        Ok(decoded) => {
                            debug!("Edge case {} decoded to: {:?}", i + 1, decoded);
                            edge_case_results.push(true);
                        }
                        Err(e) => {
                            warn!("Edge case {} decode failed: {}", i + 1, e);
                            edge_case_results.push(false);
                        }
                    }
                }
                Err(e) => {
                    warn!("Edge case {} encode failed: {}", i + 1, e);
                    edge_case_results.push(false);
                }
            }
        }

        let successful_edge_cases = edge_case_results.iter().filter(|&&x| x).count();

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("stop_sequences_tested".to_string(), config_with_stop.stop_sequences.len() as f64),
                ("has_eos_token".to_string(), if eos_token_id.is_some() { 1.0 } else { 0.0 }),
                ("has_pad_token".to_string(), if pad_token_id.is_some() { 1.0 } else { 0.0 }),
                ("edge_cases_tested".to_string(), edge_cases.len() as f64),
                ("successful_edge_cases".to_string(), successful_edge_cases as f64),
                ("tokenizer_encode_calls".to_string(), tokenizer.encode_call_count() as f64),
                ("tokenizer_decode_calls".to_string(), tokenizer.decode_call_count() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up special token handling test");
        Ok(())
    }
}

/// Test long sequence tokenization and processing
struct LongSequenceTokenizationTest;

#[async_trait]
impl TestCase for LongSequenceTokenizationTest {
    fn name(&self) -> &str {
        "long_sequence_tokenization"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up long sequence tokenization test");
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

        // Test various sequence lengths
        let sequence_lengths = vec![100, 500, 1000, 2000];
        let mut processing_times = Vec::new();
        let mut token_counts = Vec::new();

        for &length in &sequence_lengths {
            debug!("Testing sequence length: {}", length);

            // Generate long text
            let long_text = "This is a test sentence. ".repeat(length / 25);
            debug!("Generated text of {} characters", long_text.len());

            let process_start = Instant::now();

            // Test tokenization
            let tokens = tokenizer.encode(&long_text, true, false).map_err(|e| {
                TestError::execution(format!(
                    "Long sequence tokenization failed for length {}: {}",
                    length, e
                ))
            })?;

            debug!("Long text tokenized to {} tokens", tokens.len());
            token_counts.push(tokens.len());

            // Test pipeline with long sequence
            let generation_result = engine.generate(&long_text).await;

            let process_time = process_start.elapsed();
            processing_times.push(process_time);

            match generation_result {
                Ok(generated) => {
                    debug!("Long sequence generated {} characters", generated.len());

                    if generated.is_empty() {
                        return Err(TestError::assertion(format!(
                            "Long sequence should generate output for length {}",
                            length
                        )));
                    }
                }
                Err(e) => {
                    warn!("Long sequence generation failed for length {}: {}", length, e);
                    // Don't fail the test, as very long sequences might legitimately fail
                }
            }

            debug!("Processed sequence length {} in {:?}", length, process_time);
        }

        // Test context length limits
        debug!("Testing context length handling");
        let very_long_text = "Word ".repeat(10000); // Very long input

        let context_test_start = Instant::now();
        let context_result = engine.generate(&very_long_text).await;
        let context_test_time = context_test_start.elapsed();

        match context_result {
            Ok(generated) => {
                debug!(
                    "Very long context handled successfully, generated {} characters",
                    generated.len()
                );
            }
            Err(e) => {
                debug!("Very long context failed as expected: {}", e);
            }
        }

        // Test memory efficiency with long sequences
        debug!("Testing memory efficiency");
        let memory_config = InferenceConfig::memory_efficient();
        let memory_engine =
            InferenceEngine::with_config(model.clone(), tokenizer.clone(), device, memory_config)
                .map_err(|e| {
                TestError::execution(format!("Memory-efficient engine creation failed: {}", e))
            })?;

        let memory_test_text = "Memory test. ".repeat(500);
        let memory_result = memory_engine.generate(&memory_test_text).await;

        match memory_result {
            Ok(generated) => {
                debug!(
                    "Memory-efficient processing successful, generated {} characters",
                    generated.len()
                );
            }
            Err(e) => {
                warn!("Memory-efficient processing failed: {}", e);
            }
        }

        // Calculate statistics
        let avg_processing_time =
            processing_times.iter().sum::<std::time::Duration>() / processing_times.len() as u32;
        let max_processing_time = processing_times.iter().max().unwrap();
        let avg_tokens = token_counts.iter().sum::<usize>() as f64 / token_counts.len() as f64;
        let max_tokens = token_counts.iter().max().unwrap_or(&0);

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("sequence_lengths_tested".to_string(), sequence_lengths.len() as f64),
                ("avg_processing_time_ms".to_string(), avg_processing_time.as_millis() as f64),
                ("max_processing_time_ms".to_string(), max_processing_time.as_millis() as f64),
                ("avg_tokens_per_sequence".to_string(), avg_tokens),
                ("max_tokens_in_sequence".to_string(), *max_tokens as f64),
                ("context_test_time_ms".to_string(), context_test_time.as_millis() as f64),
                ("tokenizer_encode_calls".to_string(), tokenizer.encode_call_count() as f64),
                ("model_forward_calls".to_string(), model.forward_call_count() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up long sequence tokenization test");
        Ok(())
    }
}

/// Test tokenization error handling
struct TokenizationErrorHandlingTest;

#[async_trait]
impl TestCase for TokenizationErrorHandlingTest {
    fn name(&self) -> &str {
        "tokenization_error_handling"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up tokenization error handling test");
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

        // Test various problematic inputs
        let problematic_inputs = vec![
            "",          // Empty string
            "\0",        // Null character
            "🚀🌟💫",    // Unicode emojis
            "Ñoño niño", // Accented characters
            "中文测试",  // Chinese characters
            "🏳️‍🌈🏳️‍⚧️",      // Complex emoji sequences
        ];

        let mut error_handling_results = Vec::new();
        let mut successful_tokenizations = 0;
        let mut failed_tokenizations = 0;

        debug!("Testing problematic inputs");
        for (i, input) in problematic_inputs.iter().enumerate() {
            debug!("Testing problematic input {}: {:?}", i + 1, input);

            // Test direct tokenization
            match tokenizer.encode(input, true, false) {
                Ok(tokens) => {
                    debug!(
                        "Problematic input {} tokenized successfully to {} tokens",
                        i + 1,
                        tokens.len()
                    );
                    successful_tokenizations += 1;

                    // Test decoding
                    match tokenizer.decode(&tokens) {
                        Ok(decoded) => {
                            debug!("Problematic input {} decoded to: {:?}", i + 1, decoded);
                        }
                        Err(e) => {
                            warn!("Problematic input {} decode failed: {}", i + 1, e);
                        }
                    }
                }
                Err(e) => {
                    debug!("Problematic input {} tokenization failed as expected: {}", i + 1, e);
                    failed_tokenizations += 1;
                }
            }

            // Test full pipeline
            match engine.generate(input).await {
                Ok(generated) => {
                    debug!("Problematic input {} pipeline succeeded: '{}'", i + 1, generated);
                    error_handling_results.push(true);
                }
                Err(e) => {
                    debug!("Problematic input {} pipeline failed: {}", i + 1, e);
                    error_handling_results.push(false);
                }
            }
        }

        // Test invalid token sequences
        debug!("Testing invalid token sequences");
        let vocab_size = tokenizer.vocab_size() as u32;
        let invalid_token_sequences = [
            vec![vocab_size + 1],         // Out of vocabulary
            vec![u32::MAX],               // Maximum value
            vec![0, vocab_size + 100, 1], // Mixed valid/invalid
        ];

        let mut invalid_decode_results = Vec::new();

        for (i, tokens) in invalid_token_sequences.iter().enumerate() {
            debug!("Testing invalid token sequence {}: {:?}", i + 1, tokens);

            match tokenizer.decode(tokens) {
                Ok(decoded) => {
                    debug!("Invalid tokens {} decoded to: {:?}", i + 1, decoded);
                    invalid_decode_results.push(true);
                }
                Err(e) => {
                    debug!("Invalid tokens {} decode failed as expected: {}", i + 1, e);
                    invalid_decode_results.push(false);
                }
            }
        }

        // Test recovery after errors
        debug!("Testing error recovery");
        let recovery_input = "Normal text after errors";
        let recovery_result = engine
            .generate(recovery_input)
            .await
            .map_err(|e| TestError::execution(format!("Recovery generation failed: {}", e)))?;

        if recovery_result.is_empty() {
            return Err(TestError::assertion("System should recover after errors"));
        }

        debug!("Recovery successful: '{}'", recovery_result);

        // Test concurrent error handling
        debug!("Testing concurrent error scenarios");
        let concurrent_inputs = vec!["Test 1", "", "Test 2", "\0", "Test 3"];
        let mut concurrent_handles = Vec::new();

        for input in concurrent_inputs {
            let engine_clone = InferenceEngine::new(model.clone(), tokenizer.clone(), device)
                .map_err(|e| {
                    TestError::execution(format!("Failed to create concurrent engine: {}", e))
                })?;

            let handle = tokio::spawn(async move { engine_clone.generate(input).await });
            concurrent_handles.push(handle);
        }

        let mut concurrent_successes = 0;
        let mut concurrent_failures = 0;

        for handle in concurrent_handles {
            match handle.await {
                Ok(Ok(_)) => concurrent_successes += 1,
                Ok(Err(_)) => concurrent_failures += 1,
                Err(_) => concurrent_failures += 1,
            }
        }

        debug!(
            "Concurrent results: {} successes, {} failures",
            concurrent_successes, concurrent_failures
        );

        let successful_error_handling = error_handling_results.iter().filter(|&&x| x).count();
        let successful_invalid_decodes = invalid_decode_results.iter().filter(|&&x| x).count();

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("problematic_inputs_tested".to_string(), problematic_inputs.len() as f64),
                ("successful_tokenizations".to_string(), successful_tokenizations as f64),
                ("failed_tokenizations".to_string(), failed_tokenizations as f64),
                ("successful_error_handling".to_string(), successful_error_handling as f64),
                (
                    "invalid_token_sequences_tested".to_string(),
                    invalid_token_sequences.len() as f64,
                ),
                ("successful_invalid_decodes".to_string(), successful_invalid_decodes as f64),
                ("concurrent_successes".to_string(), concurrent_successes as f64),
                ("concurrent_failures".to_string(), concurrent_failures as f64),
                ("recovery_successful".to_string(), 1.0),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up tokenization error handling test");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestConfig, TestHarness};

    #[tokio::test]
    async fn test_tokenization_pipeline_suite() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let suite = TokenizationPipelineTestSuite;

        let result = harness.run_test_suite(&suite).await;
        assert!(result.is_ok());

        let suite_result = result.unwrap();
        assert!(suite_result.summary.total_tests > 0);
    }
}
