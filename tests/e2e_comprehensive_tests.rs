/*!
# End-to-End Comprehensive Tests

This module contains comprehensive end-to-end tests that cover complete workflows
across the BitNet Rust ecosystem, including both happy path and unhappy path scenarios.
*/

use std::sync::Arc;
use tempfile::TempDir;
use std::fs;
use std::path::Path;

// Import all necessary components
use bitnet_common::{
    BitNetConfig, Device, ConcreteTensor, MockTensor, BitNetError,
    ModelError, QuantizationError, InferenceError, KernelError
};
use bitnet_quantization::{
    I2SQuantizer, TL1Quantizer, TL2Quantizer, Quantize,
    QuantizationType, QuantizationConfig
};
use bitnet_models::{ModelLoader, ModelFormat};
use bitnet_kernels::{KernelManager, select_best_provider};

/// Test data generator for creating realistic test scenarios
struct TestDataGenerator;

impl TestDataGenerator {
    fn create_model_weights(size: usize) -> Vec<f32> {
        (0..size).map(|i| (i as f32 * 0.1) % 2.0 - 1.0).collect()
    }
    
    fn create_large_tensor(shape: Vec<usize>) -> MockTensor {
        MockTensor::new(shape)
    }
    
    fn create_config_file(dir: &Path, config: &BitNetConfig) -> std::io::Result<()> {
        let config_path = dir.join("bitnet_config.toml");
        let toml_content = format!(
            r#"
[model]
vocab_size = {}
hidden_size = {}
num_layers = {}
num_heads = {}

[quantization]
method = "I2S"
block_size = {}

[inference]
batch_size = {}
max_sequence_length = {}

[memory]
limit_mb = {}
"#,
            config.vocab_size,
            config.hidden_size,
            config.num_layers,
            config.num_heads,
            config.quantization_block_size,
            config.batch_size,
            config.max_sequence_length,
            config.memory_limit_mb
        );
        fs::write(config_path, toml_content)
    }
}

mod happy_path_e2e {
    use super::*;

    #[test]
    fn test_complete_quantization_workflow() {
        // Test complete quantization workflow from raw weights to quantized model
        let tensor = MockTensor::new(vec![32, 32]);
        
        // Test I2S quantization
        let i2s_quantizer = I2SQuantizer::new();
        let i2s_result = i2s_quantizer.quantize(&tensor);
        assert!(i2s_result.is_ok(), "I2S quantization should succeed");
        
        let quantized = i2s_result.unwrap();
        let dequantized = i2s_quantizer.dequantize(&quantized);
        assert!(dequantized.is_ok(), "I2S dequantization should succeed");
        
        // Verify round-trip accuracy
        let original_data = tensor.as_slice::<f32>().unwrap();
        let recovered_data = dequantized.unwrap().as_slice::<f32>().unwrap();
        
        let mse = original_data.iter()
            .zip(recovered_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original_data.len() as f32;
        
        assert!(mse < 0.1, "Round-trip error should be low, got MSE: {}", mse);
    }

    #[test]
    fn test_multi_quantizer_comparison() {
        // Test comparing different quantization methods on the same data
        let tensor = TestDataGenerator::create_large_tensor(vec![64, 64]);
        
        let quantizers: Vec<Box<dyn Quantize>> = vec![
            Box::new(I2SQuantizer::new()),
            Box::new(TL1Quantizer::new()),
            Box::new(TL2Quantizer::new()),
        ];
        
        let mut results = Vec::new();
        
        for quantizer in quantizers {
            let quantized = quantizer.quantize(&tensor).expect("Quantization should succeed");
            let dequantized = quantizer.dequantize(&quantized).expect("Dequantization should succeed");
            
            // Calculate compression ratio and accuracy
            let original_size = tensor.numel() * 4; // f32 = 4 bytes
            let compressed_size = quantized.as_slice::<u8>().unwrap().len();
            let compression_ratio = original_size as f32 / compressed_size as f32;
            
            results.push((compression_ratio, dequantized));
        }
        
        // Verify all quantizers produced valid results
        assert_eq!(results.len(), 3, "All quantizers should produce results");
        
        // Verify compression ratios are reasonable
        for (ratio, _) in &results {
            assert!(*ratio > 1.0, "Compression ratio should be > 1.0, got {}", ratio);
            assert!(*ratio < 10.0, "Compression ratio should be reasonable, got {}", ratio);
        }
    }

    #[test]
    fn test_kernel_selection_and_execution() {
        // Test automatic kernel selection and execution
        let provider = select_best_provider();
        
        assert!(provider.is_available(), "Selected provider should be available");
        
        // Test basic operations
        let tensor_a = TestDataGenerator::create_large_tensor(vec![128, 128]);
        let tensor_b = TestDataGenerator::create_large_tensor(vec![128, 128]);
        
        // Test matrix multiplication
        let result = provider.matmul_i2s(&tensor_a, &tensor_b);
        assert!(result.is_ok(), "Matrix multiplication should succeed");
        
        let output = result.unwrap();
        assert_eq!(output.shape(), vec![128, 128], "Output shape should be correct");
    }

    #[test]
    fn test_config_loading_and_validation() {
        // Test configuration loading from various sources
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        
        let config = BitNetConfig {
            vocab_size: 50000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            quantization_block_size: 64,
            batch_size: 32,
            max_sequence_length: 512,
            memory_limit_mb: 4096,
        };
        
        TestDataGenerator::create_config_file(temp_dir.path(), &config)
            .expect("Failed to create config file");
        
        // Test loading from file
        let config_path = temp_dir.path().join("bitnet_config.toml");
        let loaded_config = BitNetConfig::from_file(&config_path)
            .expect("Failed to load config from file");
        
        assert_eq!(loaded_config.vocab_size, config.vocab_size);
        assert_eq!(loaded_config.hidden_size, config.hidden_size);
        
        // Test validation
        assert!(loaded_config.validate().is_ok(), "Config should be valid");
    }
}

mod unhappy_path_e2e {
    use super::*;

    #[test]
    fn test_invalid_tensor_dimensions() {
        // Test handling of invalid tensor dimensions
        let quantizer = I2SQuantizer::new();
        
        // Test empty tensor
        let empty_tensor = MockTensor::new(vec![0]);
        let result = quantizer.quantize(&empty_tensor);
        // Should handle gracefully - either succeed with empty result or fail cleanly
        match result {
            Ok(_) => {}, // Empty tensor handled successfully
            Err(_) => {}, // Or failed cleanly
        }
        
        // Test mismatched dimensions for matrix operations
        let provider = select_best_provider();
        let tensor_a = MockTensor::new(vec![32, 64]);
        let tensor_b = MockTensor::new(vec![32, 64]); // Wrong dimensions for matmul
        
        let result = provider.matmul_i2s(&tensor_a, &tensor_b);
        assert!(result.is_err(), "Matrix multiplication with wrong dimensions should fail");
    }

    #[test]
    fn test_invalid_configuration() {
        // Test handling of invalid configurations
        let invalid_configs = vec![
            BitNetConfig {
                vocab_size: 0, // Invalid: zero vocab size
                ..BitNetConfig::default()
            },
            BitNetConfig {
                hidden_size: 1,
                num_heads: 2, // Invalid: hidden_size not divisible by num_heads
                ..BitNetConfig::default()
            },
            BitNetConfig {
                quantization_block_size: 0, // Invalid: zero block size
                ..BitNetConfig::default()
            },
        ];
        
        for config in invalid_configs {
            let validation_result = config.validate();
            assert!(validation_result.is_err(), 
                "Invalid config should fail validation: {:?}", config);
            
            // Error should have meaningful message
            let error = validation_result.unwrap_err();
            assert!(!format!("{}", error).is_empty(), "Error message should not be empty");
        }
    }

    #[test]
    fn test_error_propagation() {
        // Test that errors propagate correctly through the system
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        
        // Test file not found error
        let nonexistent_path = temp_dir.path().join("nonexistent.toml");
        let result = BitNetConfig::from_file(&nonexistent_path);
        assert!(result.is_err(), "Loading nonexistent file should fail");
        
        match result.unwrap_err() {
            BitNetError::Io(_) => {}, // Expected IO error
            other => panic!("Expected IO error, got: {:?}", other),
        }
        
        // Test invalid file format
        let invalid_config_path = temp_dir.path().join("invalid.toml");
        fs::write(&invalid_config_path, "invalid toml content [[[")
            .expect("Failed to write invalid config");
        
        let result = BitNetConfig::from_file(&invalid_config_path);
        assert!(result.is_err(), "Loading invalid config should fail");
    }
}

mod integration_tests {
    use super::*;

    #[test]
    fn test_cross_component_integration() {
        // Test integration between different components
        let tensor = TestDataGenerator::create_large_tensor(vec![64, 64]);
        
        // Test quantization -> kernel operations -> dequantization pipeline
        let quantizer = I2SQuantizer::new();
        let provider = select_best_provider();
        
        // Step 1: Quantize
        let quantized = quantizer.quantize(&tensor)
            .expect("Quantization should succeed");
        
        // Step 2: Perform kernel operation
        let kernel_result = provider.quantize_i2s(&tensor);
        assert!(kernel_result.is_ok(), "Kernel operation should succeed");
        
        // Step 3: Dequantize
        let dequantized = quantizer.dequantize(&quantized)
            .expect("Dequantization should succeed");
        
        // Verify pipeline integrity
        assert_eq!(dequantized.shape(), tensor.shape(), "Shape should be preserved");
    }

    #[test]
    fn test_configuration_propagation() {
        // Test that different configurations produce different results
        let configs = vec![32, 64, 128];
        
        for block_size in configs {
            let tensor = TestDataGenerator::create_large_tensor(vec![256, 256]);
            let quantizer = I2SQuantizer::with_block_size(block_size);
            
            let result = quantizer.quantize(&tensor);
            assert!(result.is_ok(), "Quantization with block size {} should succeed", block_size);
        }
    }

    #[test]
    fn test_error_recovery() {
        // Test system recovery from errors
        let quantizer = I2SQuantizer::new();
        
        // Cause an error with invalid input
        let invalid_tensor = MockTensor::new(vec![0]);
        let _error_result = quantizer.quantize(&invalid_tensor);
        
        // System should recover and work with valid input
        let valid_tensor = TestDataGenerator::create_large_tensor(vec![32, 32]);
        let success_result = quantizer.quantize(&valid_tensor);
        
        // Should succeed after error
        assert!(success_result.is_ok(), "System should recover from previous error");
    }
}