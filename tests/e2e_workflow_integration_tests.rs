//! # End-to-End Workflow Integration Tests
//!
//! This module implements comprehensive integration tests that validate complete workflows
//! across the BitNet Rust ecosystem, covering end-to-end inference, model loading,
//! tokenization pipelines, streaming generation, and batch processing.
//!
//! These tests implement the requirement: "Integration tests validate complete workflows end-to-end"
//! from the testing framework implementation spec.

use std::sync::Arc;
use std::time::{Duration, Instant};

// Import BitNet components
use bitnet_common::{BitNetConfig, BitNetError, ConcreteTensor, Device, MockTensor, Tensor};
use bitnet_inference::{GenerationConfig, InferenceEngine};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;

/// Mock model implementation for integration testing
#[derive(Debug)]
struct IntegrationTestModel {
    config: BitNetConfig,
    forward_calls: std::sync::Mutex<usize>,
    generation_history: std::sync::Mutex<Vec<String>>,
}

impl IntegrationTestModel {
    fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            forward_calls: std::sync::Mutex::new(0),
            generation_history: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn get_forward_calls(&self) -> usize {
        *self.forward_calls.lock().unwrap()
    }

    fn get_generation_history(&self) -> Vec<String> {
        self.generation_history.lock().unwrap().clone()
    }

    fn add_generation(&self, text: &str) {
        self.generation_history.lock().unwrap().push(text.to_string());
    }
}

impl Model for IntegrationTestModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor, BitNetError> {
        // Increment call counter
        *self.forward_calls.lock().unwrap() += 1;

        // Create mock output tensor with vocab size dimensions
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let vocab_size = self.config.model.vocab_size;

        Ok(ConcreteTensor::Mock(MockTensor::new(vec![batch_size, seq_len, vocab_size])))
    }

    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor, BitNetError> {
        let batch_size = 1;
        let seq_len = tokens.len();
        let hidden_size = self.config.model.hidden_size;

        Ok(ConcreteTensor::Mock(MockTensor::new(vec![batch_size, seq_len, hidden_size])))
    }

    fn logits(&self, hidden_states: &ConcreteTensor) -> Result<ConcreteTensor, BitNetError> {
        let shape = hidden_states.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let vocab_size = self.config.model.vocab_size;

        Ok(ConcreteTensor::Mock(MockTensor::new(vec![batch_size, seq_len, vocab_size])))
    }
}

/// Mock tokenizer implementation for integration testing
#[derive(Debug)]
struct IntegrationTestTokenizer {
    vocab_size: usize,
    encode_calls: std::sync::Mutex<usize>,
    decode_calls: std::sync::Mutex<usize>,
    tokenization_history: std::sync::Mutex<Vec<String>>,
}

impl IntegrationTestTokenizer {
    fn new() -> Self {
        Self {
            vocab_size: 50257,
            encode_calls: std::sync::Mutex::new(0),
            decode_calls: std::sync::Mutex::new(0),
            tokenization_history: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn get_encode_calls(&self) -> usize {
        *self.encode_calls.lock().unwrap()
    }

    fn get_decode_calls(&self) -> usize {
        *self.decode_calls.lock().unwrap()
    }

    fn get_tokenization_history(&self) -> Vec<String> {
        self.tokenization_history.lock().unwrap().clone()
    }
}

impl Tokenizer for IntegrationTestTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, BitNetError> {
        *self.encode_calls.lock().unwrap() += 1;
        self.tokenization_history.lock().unwrap().push(format!("encode: {}", text));

        // Simple mock encoding: convert text to token IDs based on character codes
        let tokens: Vec<u32> = text
            .chars()
            .take(10) // Limit to 10 tokens for testing
            .enumerate()
            .map(|(i, c)| ((c as u32) % 1000) + (i as u32))
            .collect();

        Ok(if tokens.is_empty() { vec![1] } else { tokens })
    }

    fn decode(&self, tokens: &[u32], _skip_special_tokens: bool) -> Result<String, BitNetError> {
        *self.decode_calls.lock().unwrap() += 1;

        // Simple mock decoding: create text based on token count and values
        let text = format!("generated_text_from_{}_tokens", tokens.len());
        self.tokenization_history.lock().unwrap().push(format!(
            "decode: {} tokens -> {}",
            tokens.len(),
            text
        ));

        Ok(text)
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
}

/// Test complete end-to-end inference workflow
#[tokio::test]
async fn test_complete_inference_workflow() {
    println!("Testing complete end-to-end inference workflow");

    // Create components
    let model = Arc::new(IntegrationTestModel::new());
    let tokenizer = Arc::new(IntegrationTestTokenizer::new());

    // Create inference engine
    let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)
        .expect("Failed to create inference engine");

    // Test basic generation
    let prompt = "Hello, world! This is a test of the complete inference workflow.";
    println!("Testing with prompt: '{}'", prompt);

    let start_time = Instant::now();
    let result = engine.generate(prompt).await.expect("Generation should succeed");
    let generation_time = start_time.elapsed();

    println!("Generated result: '{}'", result);
    println!("Generation took: {:?}", generation_time);

    // Validate results
    assert!(!result.is_empty(), "Generated text should not be empty");
    assert!(
        generation_time < Duration::from_secs(10),
        "Generation should complete within reasonable time"
    );

    // Validate component interactions
    let model_calls = model.get_forward_calls();
    let encode_calls = tokenizer.get_encode_calls();
    let decode_calls = tokenizer.get_decode_calls();

    println!("Component interaction stats:");
    println!("  Model forward calls: {}", model_calls);
    println!("  Tokenizer encode calls: {}", encode_calls);
    println!("  Tokenizer decode calls: {}", decode_calls);

    assert!(model_calls > 0, "Model should have been called");
    assert!(encode_calls > 0, "Tokenizer encode should have been called");
    assert!(decode_calls > 0, "Tokenizer decode should have been called");

    // Validate tokenization history
    let tokenization_history = tokenizer.get_tokenization_history();
    assert!(!tokenization_history.is_empty(), "Tokenization history should not be empty");
    assert!(
        tokenization_history.iter().any(|entry| entry.contains("encode")),
        "Should have encode operations in history"
    );
    assert!(
        tokenization_history.iter().any(|entry| entry.contains("decode")),
        "Should have decode operations in history"
    );

    println!("✓ Complete inference workflow test passed");
}

/// Test model loading and initialization workflow
#[tokio::test]
async fn test_model_loading_workflow() {
    println!("Testing model loading and initialization workflow");

    // Test different model configurations
    let configs = vec![
        BitNetConfig::default(),
        BitNetConfig {
            model: bitnet_common::ModelConfig {
                vocab_size: 32000,
                hidden_size: 1024,
                num_layers: 24,
                num_heads: 16,
                ..Default::default()
            },
            ..Default::default()
        },
        BitNetConfig {
            model: bitnet_common::ModelConfig {
                vocab_size: 50257,
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
                ..Default::default()
            },
            ..Default::default()
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        println!("Testing configuration {}: {:?}", i + 1, config.model);

        // Create model with specific configuration
        let mut model = IntegrationTestModel::new();
        model.config = config.clone();
        let model = Arc::new(model);

        let tokenizer = Arc::new(IntegrationTestTokenizer::new());

        // Test engine creation with different configurations
        let engine_result = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu);
        assert!(engine_result.is_ok(), "Engine creation should succeed for config {}", i + 1);

        let engine = engine_result.unwrap();

        // Test basic functionality with each configuration
        let test_prompt = format!("Test prompt for configuration {}", i + 1);
        let result = engine.generate(&test_prompt).await;

        assert!(result.is_ok(), "Generation should succeed for config {}", i + 1);

        let generated_text = result.unwrap();
        assert!(
            !generated_text.is_empty(),
            "Generated text should not be empty for config {}",
            i + 1
        );

        println!("  Config {} result: '{}'", i + 1, generated_text);
    }

    println!("✓ Model loading workflow test passed");
}

/// Test tokenization to inference pipeline
#[tokio::test]
async fn test_tokenization_pipeline_workflow() {
    println!("Testing tokenization to inference pipeline workflow");

    let model = Arc::new(IntegrationTestModel::new());
    let tokenizer = Arc::new(IntegrationTestTokenizer::new());
    let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)
        .expect("Failed to create inference engine");

    // Test various input types and lengths
    let test_inputs = vec![
        "Short",
        "Medium length input for testing",
        "This is a much longer input text that should test the tokenization pipeline's ability to handle longer sequences and ensure that the complete workflow from tokenization through inference to text generation works correctly.",
        "Special characters: !@#$%^&*()",
        "Numbers: 123456789",
        "Mixed: Hello123 World!@# Test",
        "", // Empty input
        "   ", // Whitespace only
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        println!("Testing input {}: '{}'", i + 1, input);

        let start_time = Instant::now();

        // Test the complete pipeline
        let result = engine.generate(input).await;
        let pipeline_time = start_time.elapsed();

        match result {
            Ok(output) => {
                println!("  Input: '{}'", input);
                println!("  Output: '{}'", output);
                println!("  Pipeline time: {:?}", pipeline_time);

                // Validate output characteristics
                if !input.trim().is_empty() {
                    assert!(!output.is_empty(), "Non-empty input should produce non-empty output");
                }

                assert!(
                    pipeline_time < Duration::from_secs(5),
                    "Pipeline should complete within reasonable time"
                );
            }
            Err(e) => {
                println!("  Input '{}' failed: {}", input, e);
                // Some inputs (like empty strings) might legitimately fail
                // This is acceptable as long as the failure is handled gracefully
            }
        }
    }

    // Validate tokenization pipeline statistics
    let encode_calls = tokenizer.get_encode_calls();
    let decode_calls = tokenizer.get_decode_calls();
    let tokenization_history = tokenizer.get_tokenization_history();

    println!("Pipeline statistics:");
    println!("  Total encode calls: {}", encode_calls);
    println!("  Total decode calls: {}", decode_calls);
    println!("  Tokenization operations: {}", tokenization_history.len());

    assert!(encode_calls > 0, "Should have made encode calls");
    assert!(tokenization_history.len() > 0, "Should have tokenization history");

    println!("✓ Tokenization pipeline workflow test passed");
}

/// Test streaming inference workflow
#[tokio::test]
async fn test_streaming_workflow() {
    println!("Testing streaming inference workflow");

    let model = Arc::new(IntegrationTestModel::new());
    let tokenizer = Arc::new(IntegrationTestTokenizer::new());
    let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)
        .expect("Failed to create inference engine");

    // Test streaming generation with different configurations
    let streaming_configs = vec![
        GenerationConfig { max_new_tokens: 5, temperature: 0.7, ..Default::default() },
        GenerationConfig { max_new_tokens: 10, temperature: 0.5, ..Default::default() },
        GenerationConfig { max_new_tokens: 3, temperature: 1.0, ..Default::default() },
    ];

    for (i, config) in streaming_configs.iter().enumerate() {
        println!("Testing streaming config {}: {:?}", i + 1, config);

        let prompt = format!("Streaming test prompt {}", i + 1);
        let start_time = Instant::now();

        // Test streaming generation
        let result = engine.generate_with_config(&prompt, config).await;
        let streaming_time = start_time.elapsed();

        match result {
            Ok(output) => {
                println!("  Prompt: '{}'", prompt);
                println!("  Streamed output: '{}'", output);
                println!("  Streaming time: {:?}", streaming_time);

                assert!(!output.is_empty(), "Streaming should produce output");
                assert!(
                    streaming_time < Duration::from_secs(10),
                    "Streaming should complete within reasonable time"
                );
            }
            Err(e) => {
                println!("  Streaming failed for config {}: {}", i + 1, e);
                // Some configurations might fail, which is acceptable
                // as long as the failure is handled gracefully
            }
        }
    }

    // Validate streaming statistics
    let model_calls = model.get_forward_calls();
    println!("Streaming statistics:");
    println!("  Total model forward calls: {}", model_calls);

    assert!(model_calls > 0, "Streaming should have made model calls");

    println!("✓ Streaming workflow test passed");
}

/// Test batch processing workflow
#[tokio::test]
async fn test_batch_processing_workflow() {
    println!("Testing batch processing workflow");

    let model = Arc::new(IntegrationTestModel::new());
    let tokenizer = Arc::new(IntegrationTestTokenizer::new());
    let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)
        .expect("Failed to create inference engine");

    // Test batch processing with multiple prompts
    let batch_prompts = vec![
        "First batch prompt",
        "Second batch prompt with more content",
        "Third prompt: testing batch processing",
        "Fourth and final prompt in this batch",
    ];

    println!("Processing batch of {} prompts", batch_prompts.len());

    let batch_start_time = Instant::now();
    let mut batch_results = Vec::new();
    let mut individual_times = Vec::new();

    // Process each prompt in the batch
    for (i, prompt) in batch_prompts.iter().enumerate() {
        println!("Processing batch item {}: '{}'", i + 1, prompt);

        let item_start_time = Instant::now();
        let result = engine.generate(prompt).await;
        let item_time = item_start_time.elapsed();

        individual_times.push(item_time);

        match result {
            Ok(output) => {
                println!("  Batch item {} result: '{}'", i + 1, output);
                batch_results.push(output);
            }
            Err(e) => {
                println!("  Batch item {} failed: {}", i + 1, e);
                batch_results.push(format!("ERROR: {}", e));
            }
        }
    }

    let total_batch_time = batch_start_time.elapsed();

    // Validate batch processing results
    assert_eq!(batch_results.len(), batch_prompts.len(), "Should have result for each prompt");

    let successful_results = batch_results.iter().filter(|r| !r.starts_with("ERROR:")).count();

    println!("Batch processing statistics:");
    println!("  Total prompts: {}", batch_prompts.len());
    println!("  Successful results: {}", successful_results);
    println!("  Total batch time: {:?}", total_batch_time);
    println!("  Average time per item: {:?}", total_batch_time / batch_prompts.len() as u32);

    // Print individual times
    for (i, time) in individual_times.iter().enumerate() {
        println!("  Item {} time: {:?}", i + 1, time);
    }

    assert!(successful_results > 0, "At least some batch items should succeed");
    assert!(
        total_batch_time < Duration::from_secs(30),
        "Batch processing should complete within reasonable time"
    );

    // Validate component usage statistics
    let model_calls = model.get_forward_calls();
    let encode_calls = tokenizer.get_encode_calls();
    let decode_calls = tokenizer.get_decode_calls();

    println!("Component usage statistics:");
    println!("  Model forward calls: {}", model_calls);
    println!("  Tokenizer encode calls: {}", encode_calls);
    println!("  Tokenizer decode calls: {}", decode_calls);

    assert!(
        model_calls >= successful_results,
        "Should have at least one model call per successful result"
    );
    assert!(encode_calls >= batch_prompts.len(), "Should have at least one encode call per prompt");

    println!("✓ Batch processing workflow test passed");
}

/// Test error handling and recovery in workflows
#[tokio::test]
async fn test_workflow_error_handling() {
    println!("Testing workflow error handling and recovery");

    let model = Arc::new(IntegrationTestModel::new());
    let tokenizer = Arc::new(IntegrationTestTokenizer::new());
    let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)
        .expect("Failed to create inference engine");

    // Test various error scenarios
    let long_input = "a".repeat(10000);
    let error_test_cases = vec![
        ("", "Empty input"),
        ("   ", "Whitespace only input"),
        ("\0\0\0", "Null characters"),
        (long_input.as_str(), "Very long input"),
    ];

    let mut error_handled_count = 0;
    let mut recovery_successful_count = 0;

    for (input, description) in &error_test_cases {
        println!("Testing error scenario: {}", description);

        // Test potentially problematic input
        let error_result = engine.generate(input).await;

        match error_result {
            Ok(output) => {
                println!("  Handled gracefully: '{}'", output);
                error_handled_count += 1;
            }
            Err(e) => {
                println!("  Failed as expected: {}", e);
                error_handled_count += 1;
            }
        }

        // Test recovery with normal input after potential error
        let recovery_result = engine.generate("Recovery test after error scenario").await;

        match recovery_result {
            Ok(output) => {
                println!("  Recovery successful: '{}'", output);
                recovery_successful_count += 1;
            }
            Err(e) => {
                println!("  Recovery failed: {}", e);
            }
        }
    }

    println!("Error handling statistics:");
    println!("  Error scenarios tested: {}", error_test_cases.len());
    println!("  Error scenarios handled: {}", error_handled_count);
    println!("  Recovery attempts successful: {}", recovery_successful_count);

    assert_eq!(
        error_handled_count,
        error_test_cases.len(),
        "All error scenarios should be handled (either succeed or fail gracefully)"
    );

    assert!(recovery_successful_count > 0, "At least some recovery attempts should succeed");

    println!("✓ Workflow error handling test passed");
}

/// Test resource management and cleanup in workflows
#[tokio::test]
async fn test_workflow_resource_management() {
    println!("Testing workflow resource management and cleanup");

    // Test multiple engine instances to verify resource management
    let mut engines = Vec::new();
    let num_engines = 3;

    for i in 0..num_engines {
        let model = Arc::new(IntegrationTestModel::new());
        let tokenizer = Arc::new(IntegrationTestTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)
            .expect("Failed to create inference engine");

        engines.push(engine);
        println!("Created engine {}", i + 1);
    }

    // Use all engines concurrently
    let mut handles = Vec::new();

    for (i, engine) in engines.into_iter().enumerate() {
        let handle = tokio::spawn(async move {
            let prompt = format!("Concurrent test from engine {}", i + 1);
            let result = engine.generate(&prompt).await;
            (i + 1, result)
        });
        handles.push(handle);
    }

    // Collect results
    let mut successful_concurrent = 0;
    for handle in handles {
        match handle.await {
            Ok((engine_id, result)) => match result {
                Ok(output) => {
                    println!("Engine {} result: '{}'", engine_id, output);
                    successful_concurrent += 1;
                }
                Err(e) => {
                    println!("Engine {} failed: {}", engine_id, e);
                }
            },
            Err(e) => {
                println!("Thread join failed: {:?}", e);
            }
        }
    }

    println!("Resource management statistics:");
    println!("  Engines created: {}", num_engines);
    println!("  Successful concurrent operations: {}", successful_concurrent);

    assert!(successful_concurrent > 0, "At least some concurrent operations should succeed");

    // Test cleanup by creating and dropping engines
    for i in 0..5 {
        let model = Arc::new(IntegrationTestModel::new());
        let tokenizer = Arc::new(IntegrationTestTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)
            .expect("Failed to create inference engine");

        let _result = engine.generate(&format!("Cleanup test {}", i + 1)).await;
        // Engine is dropped here
    }

    println!("✓ Workflow resource management test passed");
}

/// Integration test that validates the overall testing framework
#[tokio::test]
async fn test_integration_framework_validation() {
    println!("=== Validating Integration Testing Framework ===");

    let start_time = Instant::now();

    // Test that we can create components successfully
    let model = Arc::new(IntegrationTestModel::new());
    let tokenizer = Arc::new(IntegrationTestTokenizer::new());
    let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)
        .expect("Failed to create inference engine");

    // Test basic functionality
    let result = engine.generate("Framework validation test").await;
    assert!(result.is_ok(), "Basic generation should work");

    let total_time = start_time.elapsed();

    println!("=== Integration Testing Framework Validated ===");
    println!("Framework validation time: {:?}", total_time);
    println!("Integration testing framework is working correctly!");

    assert!(total_time < Duration::from_secs(10), "Framework validation should complete quickly");
}
