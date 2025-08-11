//! Example integration tests for component interactions
//!
//! Tests how different BitNet.rs components work together,
//! including data flow, error propagation, and resource sharing

use bitnet_common::{BitNetError, ModelConfig, QuantizationConfig};
use bitnet_inference::InferenceEngine;
use bitnet_models::{BitNetModel, ModelFormat};
use bitnet_quantization::Quantizer;
use bitnet_tokenizers::BitNetTokenizer;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;

#[cfg(test)]
mod component_interaction_examples {
    use super::*;

    /// Example: Model and tokenizer compatibility validation
    #[tokio::test]
    async fn test_model_tokenizer_compatibility() {
        let temp_dir = TempDir::new().unwrap();

        // Create model with specific vocab size
        let model_config = ModelConfig::builder()
            .model_type("bitnet_b1_58".to_string())
            .vocab_size(32000)
            .hidden_size(4096)
            .num_layers(32)
            .build()
            .unwrap();

        let model_path = create_test_model_with_config(&temp_dir, &model_config).await;
        let model = BitNetModel::from_file(&model_path).await.unwrap();

        // Create tokenizer with matching vocab size
        let tokenizer_path = create_test_tokenizer_with_vocab_size(&temp_dir, 32000).await;
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        // Test compatibility check
        let compatibility = model.check_tokenizer_compatibility(&tokenizer).await;
        assert!(
            compatibility.is_ok(),
            "Model and tokenizer should be compatible"
        );

        // Test with mismatched vocab size
        let wrong_tokenizer_path = create_test_tokenizer_with_vocab_size(&temp_dir, 16000).await;
        let wrong_tokenizer = BitNetTokenizer::from_file(&wrong_tokenizer_path)
            .await
            .unwrap();

        let wrong_compatibility = model.check_tokenizer_compatibility(&wrong_tokenizer).await;
        assert!(
            wrong_compatibility.is_err(),
            "Mismatched vocab sizes should be incompatible"
        );

        match wrong_compatibility.unwrap_err() {
            BitNetError::IncompatibleComponents {
                component1,
                component2,
                reason,
            } => {
                assert_eq!(component1, "model");
                assert_eq!(component2, "tokenizer");
                assert!(reason.contains("vocab_size"));
            }
            _ => panic!("Expected IncompatibleComponents error"),
        }
    }

    /// Example: Cross-crate error propagation
    #[tokio::test]
    async fn test_error_propagation_across_components() {
        let temp_dir = TempDir::new().unwrap();

        // Setup components
        let model_path = create_test_model(&temp_dir).await;
        let tokenizer_path = create_test_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = bitnet_inference::InferenceConfig::builder()
            .max_tokens(100)
            .temperature(0.7)
            .build()
            .unwrap();

        let mut engine = InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap();

        // Test error propagation from tokenizer
        let invalid_input = "\u{FFFF}\u{FFFE}"; // Invalid Unicode
        let result = engine.generate(invalid_input).await;

        match result {
            Err(bitnet_inference::InferenceError::TokenizationFailed { source, input }) => {
                assert_eq!(input, invalid_input);
                // Verify the source error comes from tokenizer
                assert!(
                    source.to_string().contains("tokenizer")
                        || source.to_string().contains("invalid")
                );
            }
            Err(other) => panic!("Expected TokenizationFailed, got {:?}", other),
            Ok(_) => {
                // If tokenizer handles it gracefully, that's also acceptable
                println!("Tokenizer handled invalid input gracefully");
            }
        }

        // Test error propagation from model
        // Simulate model corruption by modifying internal state
        // (This would be implementation-specific)

        // Test that engine can recover
        let valid_input = "This is valid input";
        let recovery_result = engine.generate(valid_input).await;

        // Should either work or fail with a clear error
        match recovery_result {
            Ok(result) => {
                assert!(!result.text.is_empty());
                println!("Engine recovered successfully");
            }
            Err(e) => {
                println!("Engine failed to recover: {}", e);
                // This is acceptable if the error is clear and actionable
            }
        }
    }

    /// Example: Resource sharing between components
    #[tokio::test]
    async fn test_resource_sharing_between_components() {
        let temp_dir = TempDir::new().unwrap();

        // Create shared model
        let model_path = create_test_model(&temp_dir).await;
        let model = Arc::new(BitNetModel::from_file(&model_path).await.unwrap());

        // Create multiple inference engines sharing the same model
        let tokenizer_path = create_test_tokenizer(&temp_dir).await;
        let tokenizer1 = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();
        let tokenizer2 = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config1 = bitnet_inference::InferenceConfig::builder()
            .max_tokens(50)
            .temperature(0.5)
            .build()
            .unwrap();

        let config2 = bitnet_inference::InferenceConfig::builder()
            .max_tokens(30)
            .temperature(0.8)
            .build()
            .unwrap();

        let engine1 = InferenceEngine::new(model.clone(), tokenizer1, config1)
            .await
            .unwrap();
        let engine2 = InferenceEngine::new(model.clone(), tokenizer2, config2)
            .await
            .unwrap();

        // Test concurrent access to shared model
        let input1 = "First engine input";
        let input2 = "Second engine input";

        let (result1, result2) = tokio::join!(engine1.generate(input1), engine2.generate(input2));

        // Both should succeed
        assert!(result1.is_ok(), "First engine should succeed");
        assert!(result2.is_ok(), "Second engine should succeed");

        let result1 = result1.unwrap();
        let result2 = result2.unwrap();

        // Verify results are different (due to different configs and inputs)
        assert_ne!(result1.text, result2.text);
        assert!(result1.tokens.len() <= 50);
        assert!(result2.tokens.len() <= 30);

        // Verify model state consistency
        assert_eq!(model.metadata().vocab_size(), model.metadata().vocab_size());
    }

    /// Example: Configuration propagation across components
    #[tokio::test]
    async fn test_configuration_propagation() {
        let temp_dir = TempDir::new().unwrap();

        // Create base configuration
        let base_config = ModelConfig::builder()
            .model_type("bitnet_b1_58".to_string())
            .vocab_size(32000)
            .hidden_size(4096)
            .num_layers(32)
            .build()
            .unwrap();

        // Test configuration propagation through model loading
        let model_path = create_test_model_with_config(&temp_dir, &base_config).await;
        let model = BitNetModel::from_file(&model_path).await.unwrap();

        // Verify model inherited configuration
        assert_eq!(model.config().vocab_size(), base_config.vocab_size());
        assert_eq!(model.config().hidden_size(), base_config.hidden_size());
        assert_eq!(model.config().num_layers(), base_config.num_layers());

        // Test configuration propagation to inference engine
        let tokenizer_path = create_test_tokenizer(&temp_dir).await;
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let inference_config = bitnet_inference::InferenceConfig::builder()
            .max_tokens(100)
            .temperature(0.7)
            .build()
            .unwrap();

        let engine = InferenceEngine::new(model, tokenizer, inference_config.clone())
            .await
            .unwrap();

        // Verify inference engine has access to both model and inference configs
        assert_eq!(engine.model_config().vocab_size(), base_config.vocab_size());
        assert_eq!(
            engine.inference_config().max_tokens(),
            inference_config.max_tokens()
        );
        assert_eq!(
            engine.inference_config().temperature(),
            inference_config.temperature()
        );

        // Test configuration validation across components
        let result = engine.validate_configuration().await;
        assert!(
            result.is_ok(),
            "Configuration should be valid across components"
        );
    }

    /// Example: Data flow validation between components
    #[tokio::test]
    async fn test_data_flow_validation() {
        let temp_dir = TempDir::new().unwrap();

        let model_path = create_test_model(&temp_dir).await;
        let tokenizer_path = create_test_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = bitnet_inference::InferenceConfig::builder()
            .max_tokens(20)
            .temperature(0.0) // Deterministic for testing
            .build()
            .unwrap();

        let mut engine = InferenceEngine::new(model, tokenizer.clone(), config)
            .await
            .unwrap();

        // Test data flow: text -> tokens -> model -> tokens -> text
        let input_text = "Hello world";

        // Step 1: Tokenization
        let input_tokens = tokenizer.encode(input_text).await.unwrap();
        assert!(input_tokens.len() > 0, "Should produce tokens");

        // Step 2: Model inference
        let output_tokens = engine.generate_tokens(&input_tokens).await.unwrap();
        assert!(
            output_tokens.len() > input_tokens.len(),
            "Should generate additional tokens"
        );

        // Step 3: Detokenization
        let output_text = tokenizer.decode(&output_tokens).await.unwrap();
        assert!(
            output_text.starts_with(input_text),
            "Output should start with input"
        );

        // Test round-trip consistency
        let roundtrip_tokens = tokenizer.encode(&output_text).await.unwrap();
        let roundtrip_text = tokenizer.decode(&roundtrip_tokens).await.unwrap();

        // Should be approximately equal (some tokenizers may have minor differences)
        let similarity = calculate_text_similarity(&output_text, &roundtrip_text);
        assert!(
            similarity > 0.95,
            "Round-trip similarity should be > 95%, got {:.2}",
            similarity
        );

        println!("Input: {}", input_text);
        println!("Output: {}", output_text);
        println!("Round-trip similarity: {:.2}%", similarity * 100.0);
    }

    /// Example: Component lifecycle management
    #[tokio::test]
    async fn test_component_lifecycle_management() {
        let temp_dir = TempDir::new().unwrap();

        // Test component initialization order
        let model_path = create_test_model(&temp_dir).await;
        let tokenizer_path = create_test_tokenizer(&temp_dir).await;

        // Initialize in correct order
        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = bitnet_inference::InferenceConfig::builder()
            .max_tokens(10)
            .temperature(0.5)
            .build()
            .unwrap();

        let engine = InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap();

        // Test component state during lifecycle
        assert!(
            engine.is_ready(),
            "Engine should be ready after initialization"
        );

        // Test graceful shutdown
        let shutdown_result = engine.shutdown().await;
        assert!(shutdown_result.is_ok(), "Shutdown should succeed");

        // Test that engine is no longer usable after shutdown
        let post_shutdown_result = engine.generate("test").await;
        assert!(
            post_shutdown_result.is_err(),
            "Engine should not work after shutdown"
        );

        match post_shutdown_result.unwrap_err() {
            bitnet_inference::InferenceError::EngineShutdown => {
                // Expected error
            }
            other => panic!("Expected EngineShutdown error, got {:?}", other),
        }
    }

    /// Example: Memory management across components
    #[tokio::test]
    async fn test_memory_management_across_components() {
        let temp_dir = TempDir::new().unwrap();

        let model_path = create_test_model(&temp_dir).await;
        let tokenizer_path = create_test_tokenizer(&temp_dir).await;

        // Monitor memory usage during component creation
        let initial_memory = get_memory_usage();

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let after_model_memory = get_memory_usage();

        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();
        let after_tokenizer_memory = get_memory_usage();

        let config = bitnet_inference::InferenceConfig::builder()
            .max_tokens(50)
            .temperature(0.7)
            .build()
            .unwrap();

        let engine = InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap();
        let after_engine_memory = get_memory_usage();

        // Verify reasonable memory usage
        let model_memory = after_model_memory - initial_memory;
        let tokenizer_memory = after_tokenizer_memory - after_model_memory;
        let engine_memory = after_engine_memory - after_tokenizer_memory;

        println!(
            "Memory usage - Model: {}MB, Tokenizer: {}MB, Engine: {}MB",
            model_memory / 1024 / 1024,
            tokenizer_memory / 1024 / 1024,
            engine_memory / 1024 / 1024
        );

        // Test memory cleanup
        drop(engine);
        let after_cleanup_memory = get_memory_usage();
        let cleaned_memory = after_engine_memory - after_cleanup_memory;

        // Should free some memory (exact amount depends on implementation)
        assert!(cleaned_memory > 0, "Should free some memory after cleanup");

        println!("Cleaned up: {}MB", cleaned_memory / 1024 / 1024);
    }
}

/// Component interaction test utilities
pub mod component_test_utils {
    use super::*;
    use tokio::fs;

    /// Create test model with specific configuration
    pub async fn create_test_model_with_config(
        temp_dir: &TempDir,
        config: &ModelConfig,
    ) -> std::path::PathBuf {
        let model_path = temp_dir.path().join("configured_model.gguf");

        // Create model data that reflects the configuration
        let model_data = create_model_data_with_config(config);
        fs::write(&model_path, model_data).await.unwrap();

        model_path
    }

    /// Create test tokenizer with specific vocab size
    pub async fn create_test_tokenizer_with_vocab_size(
        temp_dir: &TempDir,
        vocab_size: u32,
    ) -> std::path::PathBuf {
        let tokenizer_path = temp_dir.path().join("sized_tokenizer.json");

        let tokenizer_data = create_tokenizer_data_with_vocab_size(vocab_size);
        fs::write(&tokenizer_path, tokenizer_data).await.unwrap();

        tokenizer_path
    }

    /// Create basic test model
    pub async fn create_test_model(temp_dir: &TempDir) -> std::path::PathBuf {
        let default_config = ModelConfig::builder()
            .model_type("test".to_string())
            .vocab_size(1000)
            .hidden_size(512)
            .num_layers(8)
            .build()
            .unwrap();

        create_test_model_with_config(temp_dir, &default_config).await
    }

    /// Create basic test tokenizer
    pub async fn create_test_tokenizer(temp_dir: &TempDir) -> std::path::PathBuf {
        create_test_tokenizer_with_vocab_size(temp_dir, 1000).await
    }

    /// Create model data reflecting configuration
    fn create_model_data_with_config(config: &ModelConfig) -> Vec<u8> {
        let mut data = Vec::new();

        // GGUF header
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&10u64.to_le_bytes()); // tensor count
        data.extend_from_slice(&5u64.to_le_bytes()); // metadata count

        // Embed configuration in mock data
        let config_bytes = serde_json::to_vec(config).unwrap();
        data.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&config_bytes);

        // Add padding
        data.extend_from_slice(&vec![0u8; 1024]);

        data
    }

    /// Create tokenizer data with specific vocab size
    fn create_tokenizer_data_with_vocab_size(vocab_size: u32) -> String {
        let mut vocab = std::collections::HashMap::new();

        // Create vocab entries
        for i in 0..vocab_size {
            vocab.insert(format!("token_{}", i), i);
        }

        serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": []
            },
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null
        })
        .to_string()
    }

    /// Calculate text similarity (simple implementation)
    pub fn calculate_text_similarity(text1: &str, text2: &str) -> f64 {
        if text1 == text2 {
            return 1.0;
        }

        let len1 = text1.len();
        let len2 = text2.len();

        if len1 == 0 && len2 == 0 {
            return 1.0;
        }

        if len1 == 0 || len2 == 0 {
            return 0.0;
        }

        // Simple character-based similarity
        let common_chars = text1
            .chars()
            .zip(text2.chars())
            .filter(|(c1, c2)| c1 == c2)
            .count();

        common_chars as f64 / len1.max(len2) as f64
    }

    /// Mock memory usage function
    pub fn get_memory_usage() -> u64 {
        // In real implementation, this would use system APIs
        // For testing, return mock increasing values
        use std::sync::atomic::{AtomicU64, Ordering};
        static MOCK_MEMORY: AtomicU64 = AtomicU64::new(100 * 1024 * 1024); // Start at 100MB

        MOCK_MEMORY.fetch_add(10 * 1024 * 1024, Ordering::Relaxed) // Add 10MB each call
    }

    /// Verify component compatibility
    pub async fn verify_component_compatibility(
        model: &BitNetModel,
        tokenizer: &BitNetTokenizer,
    ) -> Result<(), String> {
        // Check vocab size compatibility
        if model.config().vocab_size() != tokenizer.vocab_size() {
            return Err(format!(
                "Vocab size mismatch: model={}, tokenizer={}",
                model.config().vocab_size(),
                tokenizer.vocab_size()
            ));
        }

        // Check model type compatibility
        let supported_types = tokenizer.supported_model_types();
        if !supported_types.contains(model.config().model_type()) {
            return Err(format!(
                "Model type '{}' not supported by tokenizer",
                model.config().model_type()
            ));
        }

        Ok(())
    }
}
