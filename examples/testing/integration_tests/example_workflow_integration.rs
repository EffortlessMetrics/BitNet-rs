//! Example integration tests for complete bitnet-rs workflows
//!
//! Demonstrates end-to-end testing patterns for inference pipelines,
//! component interactions, and system-level validation

use bitnet_common::{ModelConfig, QuantizationConfig};
use bitnet_inference::{InferenceConfig, InferenceEngine};
use bitnet_models::BitNetModel;
use bitnet_tokenizers::BitNetTokenizer;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

#[cfg(test)]
mod workflow_integration_examples {
    use super::*;

    /// Example: Complete inference workflow test
    #[tokio::test]
    async fn test_complete_inference_workflow() {
        // Setup: Create test environment
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir).await;
        let tokenizer_path = setup_test_tokenizer(&temp_dir).await;

        // Step 1: Load model
        let model = BitNetModel::from_file(&model_path)
            .await
            .expect("Failed to load test model");

        // Step 2: Initialize tokenizer
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path)
            .await
            .expect("Failed to load test tokenizer");

        // Step 3: Create inference engine
        let config = InferenceConfig::builder()
            .max_tokens(100)
            .temperature(0.7)
            .top_p(0.9)
            .build()
            .unwrap();

        let mut engine = InferenceEngine::new(model, tokenizer, config)
            .await
            .expect("Failed to create inference engine");

        // Step 4: Run inference
        let input_text = "The future of AI is";
        let result = engine.generate(input_text).await.expect("Inference failed");

        // Verify: Check results
        assert!(!result.text.is_empty());
        assert!(result.tokens.len() > 0);
        assert!(result.tokens.len() <= 100); // Respects max_tokens
        assert!(result.generation_time.as_millis() > 0);

        // Verify: Check that output is coherent
        assert!(result.text.starts_with(input_text));
        assert!(result.text.len() > input_text.len());

        println!("Generated text: {}", result.text);
        println!("Generation time: {:?}", result.generation_time);
        println!(
            "Tokens per second: {:.2}",
            result.tokens.len() as f64 / result.generation_time.as_secs_f64()
        );
    }

    /// Example: Streaming inference workflow test
    #[tokio::test]
    async fn test_streaming_inference_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir).await;
        let tokenizer_path = setup_test_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(50)
            .temperature(0.8)
            .streaming(true)
            .build()
            .unwrap();

        let mut engine = InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap();

        // Test streaming generation
        let input_text = "Once upon a time";
        let mut stream = engine.generate_stream(input_text).await.unwrap();

        let mut generated_tokens = Vec::new();
        let mut generated_text = String::new();
        let mut chunk_count = 0;

        // Collect streaming results
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("Stream error");
            generated_tokens.extend(chunk.tokens);
            generated_text.push_str(&chunk.text);
            chunk_count += 1;

            // Verify each chunk is valid
            assert!(!chunk.text.is_empty());
            assert!(chunk.tokens.len() > 0);
        }

        // Verify streaming results
        assert!(
            chunk_count > 1,
            "Expected multiple chunks, got {}",
            chunk_count
        );
        assert!(generated_tokens.len() <= 50);
        assert!(generated_text.starts_with(input_text));

        println!("Streaming generated {} chunks", chunk_count);
        println!("Final text: {}", generated_text);
    }

    /// Example: Batch processing workflow test
    #[tokio::test]
    async fn test_batch_processing_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir).await;
        let tokenizer_path = setup_test_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(20)
            .temperature(0.5)
            .batch_size(4)
            .build()
            .unwrap();

        let mut engine = InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap();

        // Prepare batch inputs
        let inputs = vec![
            "The weather today is",
            "Machine learning is",
            "The best programming language is",
            "In the future, we will",
        ];

        // Process batch
        let start_time = std::time::Instant::now();
        let results = engine.generate_batch(&inputs).await.unwrap();
        let batch_time = start_time.elapsed();

        // Verify batch results
        assert_eq!(results.len(), inputs.len());

        for (i, result) in results.iter().enumerate() {
            assert!(result.text.starts_with(inputs[i]));
            assert!(result.tokens.len() <= 20);
            assert!(!result.text.is_empty());
        }

        // Verify batch processing efficiency
        let avg_time_per_item = batch_time.as_millis() / inputs.len() as u128;
        println!("Batch processing time: {:?}", batch_time);
        println!("Average time per item: {}ms", avg_time_per_item);

        // Compare with sequential processing time (if available)
        // This would help verify batch processing benefits
    }

    /// Example: Model quantization workflow test
    #[tokio::test]
    async fn test_quantization_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir).await;

        // Step 1: Load original model
        let original_model = BitNetModel::from_file(&model_path).await.unwrap();
        let original_size = fs::metadata(&model_path).await.unwrap().len();

        // Step 2: Configure quantization
        let quant_config = QuantizationConfig::new(4, 128, true); // 4-bit, group_size=128, symmetric

        // Step 3: Quantize model
        let quantized_path = temp_dir.path().join("quantized_model.gguf");
        let quantized_model = original_model
            .quantize(&quant_config, &quantized_path)
            .await
            .expect("Quantization failed");

        // Step 4: Verify quantization results
        let quantized_size = fs::metadata(&quantized_path).await.unwrap().len();
        let compression_ratio = original_size as f64 / quantized_size as f64;

        assert!(
            compression_ratio > 1.5,
            "Expected compression ratio > 1.5, got {:.2}",
            compression_ratio
        );
        assert_eq!(quantized_model.quantization_config().unwrap().bits(), 4);

        // Step 5: Test quantized model inference
        let tokenizer_path = setup_test_tokenizer(&temp_dir).await;
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(10)
            .temperature(0.0) // Deterministic for comparison
            .build()
            .unwrap();

        let mut original_engine =
            InferenceEngine::new(original_model, tokenizer.clone(), config.clone())
                .await
                .unwrap();
        let mut quantized_engine = InferenceEngine::new(quantized_model, tokenizer, config)
            .await
            .unwrap();

        // Compare outputs
        let test_input = "Hello world";
        let original_result = original_engine.generate(test_input).await.unwrap();
        let quantized_result = quantized_engine.generate(test_input).await.unwrap();

        // Verify quantized model produces reasonable output
        assert!(!quantized_result.text.is_empty());
        assert!(quantized_result.tokens.len() > 0);

        // Check performance difference
        let performance_ratio = original_result.generation_time.as_secs_f64()
            / quantized_result.generation_time.as_secs_f64();

        println!("Original model size: {} bytes", original_size);
        println!("Quantized model size: {} bytes", quantized_size);
        println!("Compression ratio: {:.2}x", compression_ratio);
        println!("Performance ratio: {:.2}x", performance_ratio);
    }

    /// Example: Error recovery workflow test
    #[tokio::test]
    async fn test_error_recovery_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir).await;
        let tokenizer_path = setup_test_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(100)
            .temperature(0.7)
            .build()
            .unwrap();

        let mut engine = InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap();

        // Test 1: Recovery from invalid input
        let invalid_input = "\x00\x01\x02"; // Invalid UTF-8
        let result = engine.generate(invalid_input).await;

        // Should handle gracefully
        match result {
            Ok(_) => {
                // If it succeeds, verify output is reasonable
            }
            Err(e) => {
                // Should be a specific error type
                assert!(matches!(
                    e,
                    bitnet_inference::InferenceError::InvalidInput { .. }
                ));
            }
        }

        // Test 2: Recovery after error - engine should still work
        let valid_input = "This is a valid input";
        let result = engine.generate(valid_input).await;
        assert!(result.is_ok(), "Engine should recover after error");

        // Test 3: Memory exhaustion recovery
        let very_long_input = "word ".repeat(10000); // Very long input
        let result = engine.generate(&very_long_input).await;

        // Should either succeed or fail gracefully
        match result {
            Ok(output) => {
                assert!(!output.text.is_empty());
            }
            Err(e) => {
                assert!(matches!(
                    e,
                    bitnet_inference::InferenceError::ResourceExhausted { .. }
                        | bitnet_inference::InferenceError::InputTooLong { .. }
                ));
            }
        }

        // Test 4: Engine should still work after resource exhaustion
        let normal_input = "Normal input";
        let result = engine.generate(normal_input).await;
        assert!(
            result.is_ok(),
            "Engine should recover after resource exhaustion"
        );
    }

    /// Example: Configuration validation workflow test
    #[tokio::test]
    async fn test_configuration_validation_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir).await;
        let tokenizer_path = setup_test_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        // Test various configuration combinations
        let test_configs = vec![
            // Valid configurations
            InferenceConfig::builder()
                .max_tokens(10)
                .temperature(0.5)
                .build(),
            InferenceConfig::builder()
                .max_tokens(100)
                .temperature(1.0)
                .top_p(0.9)
                .build(),
            InferenceConfig::builder()
                .max_tokens(50)
                .temperature(0.0)
                .build(), // Deterministic
            // Edge case configurations
            InferenceConfig::builder()
                .max_tokens(1)
                .temperature(0.1)
                .build(),
            InferenceConfig::builder()
                .max_tokens(1000)
                .temperature(2.0)
                .build(),
        ];

        for (i, config_result) in test_configs.into_iter().enumerate() {
            match config_result {
                Ok(config) => {
                    // Test that valid config works
                    let engine_result =
                        InferenceEngine::new(model.clone(), tokenizer.clone(), config).await;

                    assert!(engine_result.is_ok(), "Config {} should be valid", i);

                    let mut engine = engine_result.unwrap();
                    let result = engine.generate("Test").await;
                    assert!(result.is_ok(), "Inference with config {} should work", i);
                }
                Err(e) => {
                    // Test that invalid config is properly rejected
                    println!("Config {} properly rejected: {}", i, e);
                }
            }
        }

        // Test invalid configurations
        let invalid_configs = vec![
            InferenceConfig::builder()
                .max_tokens(0)
                .temperature(0.5)
                .build(),
            InferenceConfig::builder()
                .max_tokens(10)
                .temperature(-1.0)
                .build(),
            InferenceConfig::builder()
                .max_tokens(10)
                .temperature(0.5)
                .top_p(1.5)
                .build(),
        ];

        for (i, config_result) in invalid_configs.into_iter().enumerate() {
            assert!(
                config_result.is_err(),
                "Invalid config {} should be rejected",
                i
            );
        }
    }
}

/// Integration test utilities
pub mod integration_test_utils {
    use super::*;

    /// Setup a test model file
    pub async fn setup_test_model(temp_dir: &TempDir) -> PathBuf {
        let model_path = temp_dir.path().join("test_model.gguf");

        // Create mock model data with proper structure
        let model_data = create_integration_test_model_data();
        fs::write(&model_path, model_data).await.unwrap();

        model_path
    }

    /// Setup a test tokenizer file
    pub async fn setup_test_tokenizer(temp_dir: &TempDir) -> PathBuf {
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        // Create mock tokenizer data
        let tokenizer_data = create_integration_test_tokenizer_data();
        fs::write(&tokenizer_path, tokenizer_data).await.unwrap();

        tokenizer_path
    }

    /// Create mock model data suitable for integration testing
    fn create_integration_test_model_data() -> Vec<u8> {
        // More comprehensive mock data than unit tests
        let mut data = Vec::new();

        // GGUF header
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&20u64.to_le_bytes()); // tensor count
        data.extend_from_slice(&10u64.to_le_bytes()); // metadata count

        // Add mock tensors and metadata for realistic testing
        data.extend_from_slice(&vec![0u8; 50 * 1024]); // 50KB mock model

        data
    }

    /// Create mock tokenizer data
    fn create_integration_test_tokenizer_data() -> String {
        // Simplified tokenizer.json format
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": {
                "type": "Sequence",
                "normalizers": []
            },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "vocab": {
                    "hello": 0,
                    "world": 1,
                    "test": 2,
                    "the": 3,
                    "is": 4,
                    "a": 5,
                    "to": 6,
                    "and": 7,
                    "of": 8,
                    "in": 9
                },
                "merges": []
            }
        })
        .to_string()
    }

    /// Verify inference result quality
    pub fn verify_inference_result(result: &bitnet_inference::InferenceResult, input: &str) {
        assert!(
            !result.text.is_empty(),
            "Generated text should not be empty"
        );
        assert!(
            result.tokens.len() > 0,
            "Should generate at least one token"
        );
        assert!(
            result.text.starts_with(input),
            "Output should start with input"
        );
        assert!(
            result.generation_time.as_millis() > 0,
            "Should have measurable generation time"
        );

        // Check for reasonable output length
        assert!(
            result.text.len() > input.len(),
            "Should generate additional content"
        );
        assert!(
            result.text.len() < input.len() + 10000,
            "Should not generate excessively long output"
        );
    }

    /// Performance assertion helper
    pub fn assert_performance_acceptable(
        result: &bitnet_inference::InferenceResult,
        max_time_per_token_ms: u128,
    ) {
        let time_per_token = result.generation_time.as_millis() / result.tokens.len() as u128;
        assert!(
            time_per_token <= max_time_per_token_ms,
            "Time per token {}ms exceeds limit {}ms",
            time_per_token,
            max_time_per_token_ms
        );
    }

    /// Memory usage assertion helper
    pub fn assert_memory_usage_reasonable(peak_memory_mb: u64, max_memory_mb: u64) {
        assert!(
            peak_memory_mb <= max_memory_mb,
            "Peak memory usage {}MB exceeds limit {}MB",
            peak_memory_mb,
            max_memory_mb
        );
    }
}
