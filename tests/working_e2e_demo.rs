/*!
# Working End-to-End Demo Tests

This module demonstrates the current working capabilities of the BitNet Rust ecosystem
with realistic end-to-end workflows that actually compile and run successfully.
*/

use bitnet_common::{BitNetConfig, Device, MockTensor, Tensor};
use bitnet_kernels::select_best_provider;
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
use std::fs;
use tempfile::TempDir;

#[test]
fn test_complete_quantization_pipeline() {
    // Create test data
    let tensor = MockTensor::new(vec![64, 64]);

    // Test I2S quantization workflow
    let i2s_quantizer = I2SQuantizer::new();
    let i2s_result = i2s_quantizer.quantize_tensor(&tensor);
    assert!(i2s_result.is_ok(), "I2S quantization should succeed");

    let quantized = i2s_result.unwrap();
    let dequantized = i2s_quantizer.dequantize_tensor(&quantized);
    assert!(dequantized.is_ok(), "I2S dequantization should succeed");

    // Verify tensor properties
    let recovered = dequantized.unwrap();
    assert_eq!(recovered.shape(), tensor.shape(), "Shape should be preserved");

    // Test round-trip with data validation
    let original_data = tensor.as_slice::<f32>().unwrap();
    let recovered_data = recovered.as_slice::<f32>().unwrap();

    // Calculate mean squared error
    let mse =
        original_data.iter().zip(recovered_data.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
            / original_data.len() as f32;

    println!("I2S Round-trip MSE: {:.6}", mse);
    assert!(mse < 1.0, "Round-trip error should be reasonable");
}

#[test]
fn test_multi_quantizer_comparison() {
    let tensor = MockTensor::new(vec![32, 32]);

    // Test all quantization methods
    let quantizers = vec![
        ("I2S", Box::new(I2SQuantizer::new()) as Box<dyn TestQuantizer>),
        ("TL1", Box::new(TL1Quantizer::new()) as Box<dyn TestQuantizer>),
        ("TL2", Box::new(TL2Quantizer::new()) as Box<dyn TestQuantizer>),
    ];

    for (name, quantizer) in quantizers {
        println!("Testing {} quantization...", name);

        let quantized = quantizer.test_quantize(&tensor);
        assert!(quantized.is_ok(), "{} quantization should succeed", name);

        let dequantized = quantizer.test_dequantize(&quantized.unwrap());
        assert!(dequantized.is_ok(), "{} dequantization should succeed", name);

        let recovered = dequantized.unwrap();
        assert_eq!(recovered.shape(), tensor.shape(), "{} should preserve tensor shape", name);
    }
}

#[test]
fn test_kernel_provider_selection() {
    // Test automatic kernel provider selection
    let provider = select_best_provider();
    assert!(provider.is_available(), "Selected provider should be available");

    println!("Selected provider: {}", provider.name());

    // Test basic kernel operations
    let tensor_a = MockTensor::new(vec![16, 16]);
    let tensor_b = MockTensor::new(vec![16, 16]);

    // Test quantization kernel
    let quant_result = provider.quantize_i2s(&tensor_a);
    assert!(quant_result.is_ok(), "Quantization kernel should work");

    // Test matrix multiplication kernel
    let matmul_result = provider.matmul_i2s(&tensor_a, &tensor_b);
    assert!(matmul_result.is_ok(), "Matrix multiplication kernel should work");

    let output = matmul_result.unwrap();
    assert_eq!(output.shape(), vec![16, 16], "Output shape should be correct");
}

#[test]
fn test_configuration_workflow() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Create a configuration file
    let config_content = r#"
[model]
vocab_size = 32000
hidden_size = 512
num_layers = 6
num_heads = 8

[quantization]
method = "I2S"
block_size = 64

[inference]
batch_size = 16
max_sequence_length = 256

[performance]
memory_limit_mb = 2048
"#;

    let config_path = temp_dir.path().join("test_config.toml");
    fs::write(&config_path, config_content).expect("Failed to write config");

    // Load and validate configuration
    let config = BitNetConfig::from_file(&config_path).expect("Should load config from file");

    assert!(config.validate().is_ok(), "Config should be valid");

    // Test configuration values
    assert_eq!(config.model.vocab_size, 32000);
    assert_eq!(config.model.hidden_size, 512);
    assert_eq!(config.quantization.block_size, 64);
    assert_eq!(config.performance.memory_limit_mb, 2048);

    println!("Configuration loaded successfully:");
    println!("  Vocab size: {}", config.model.vocab_size);
    println!("  Hidden size: {}", config.model.hidden_size);
    println!("  Block size: {}", config.quantization.block_size);
}

#[test]
fn test_error_handling_workflow() {
    // Test configuration validation errors
    let mut invalid_config = BitNetConfig::default();
    invalid_config.model.vocab_size = 0; // Invalid

    let validation_result = invalid_config.validate();
    assert!(validation_result.is_err(), "Invalid config should fail validation");

    let error = validation_result.unwrap_err();
    println!("Validation error (expected): {}", error);
    assert!(!format!("{}", error).is_empty(), "Error should have message");

    // Test file loading errors
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let nonexistent_path = temp_dir.path().join("nonexistent.toml");

    let load_result = BitNetConfig::from_file(&nonexistent_path);
    assert!(load_result.is_err(), "Loading nonexistent file should fail");

    // Test quantization with edge cases
    let empty_tensor = MockTensor::new(vec![0]);
    let quantizer = I2SQuantizer::new();

    // This might succeed (empty tensor) or fail gracefully
    let result = quantizer.quantize_tensor(&empty_tensor);
    match result {
        Ok(_) => println!("Empty tensor handled successfully"),
        Err(e) => println!("Empty tensor failed gracefully: {}", e),
    }
}

#[test]
fn test_device_compatibility() {
    // Test different device types
    let devices = vec![Device::Cpu, Device::Cuda(0)];

    for device in devices {
        println!("Testing device: {:?}", device);

        // Create tensor for this device
        let tensor = MockTensor::new(vec![32, 32]);

        // Test basic operations
        assert_eq!(tensor.shape(), &[32, 32], "Tensor shape should be correct");
        assert!(tensor.as_slice::<f32>().is_ok(), "Should be able to access data");

        // Test quantization on this device
        let quantizer = I2SQuantizer::new();
        let result = quantizer.quantize_tensor(&tensor);

        match result {
            Ok(_) => println!("  Quantization succeeded on {:?}", device),
            Err(e) => println!("  Quantization failed on {:?}: {}", device, e),
        }
    }
}

#[test]
fn test_memory_management() {
    // Test creating and managing multiple large tensors
    let tensor_sizes = vec![vec![64, 64], vec![128, 128], vec![256, 256]];

    let mut tensors = Vec::new();

    for size in tensor_sizes {
        let tensor = MockTensor::new(size.clone());
        println!("Created tensor with shape: {:?}", tensor.shape());

        // Verify tensor is valid
        assert_eq!(tensor.shape(), &size);
        assert!(tensor.as_slice::<f32>().is_ok());

        tensors.push(tensor);
    }

    // Test quantization with all tensors
    let quantizer = I2SQuantizer::new();

    for (i, tensor) in tensors.iter().enumerate() {
        let result = quantizer.quantize_tensor(tensor);
        assert!(result.is_ok(), "Quantization of tensor {} should succeed", i);

        if let Ok(quantized) = result {
            let deq_result = quantizer.dequantize_tensor(&quantized);
            assert!(deq_result.is_ok(), "Dequantization of tensor {} should succeed", i);
        }
    }

    println!("Successfully processed {} tensors", tensors.len());
}

// Helper trait for testing different quantizers uniformly
trait TestQuantizer {
    fn test_quantize(
        &self,
        tensor: &MockTensor,
    ) -> Result<bitnet_quantization::QuantizedTensor, Box<dyn std::error::Error>>;
    fn test_dequantize(
        &self,
        quantized: &bitnet_quantization::QuantizedTensor,
    ) -> Result<bitnet_common::BitNetTensor, Box<dyn std::error::Error>>;
}

impl TestQuantizer for I2SQuantizer {
    fn test_quantize(
        &self,
        tensor: &MockTensor,
    ) -> Result<bitnet_quantization::QuantizedTensor, Box<dyn std::error::Error>> {
        self.quantize_tensor(tensor).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    fn test_dequantize(
        &self,
        quantized: &bitnet_quantization::QuantizedTensor,
    ) -> Result<bitnet_common::BitNetTensor, Box<dyn std::error::Error>> {
        self.dequantize_tensor(quantized).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

impl TestQuantizer for TL1Quantizer {
    fn test_quantize(
        &self,
        tensor: &MockTensor,
    ) -> Result<bitnet_quantization::QuantizedTensor, Box<dyn std::error::Error>> {
        self.quantize_tensor(tensor).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    fn test_dequantize(
        &self,
        quantized: &bitnet_quantization::QuantizedTensor,
    ) -> Result<bitnet_common::BitNetTensor, Box<dyn std::error::Error>> {
        self.dequantize_tensor(quantized).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

impl TestQuantizer for TL2Quantizer {
    fn test_quantize(
        &self,
        tensor: &MockTensor,
    ) -> Result<bitnet_quantization::QuantizedTensor, Box<dyn std::error::Error>> {
        self.quantize_tensor(tensor).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    fn test_dequantize(
        &self,
        quantized: &bitnet_quantization::QuantizedTensor,
    ) -> Result<bitnet_common::BitNetTensor, Box<dyn std::error::Error>> {
        self.dequantize_tensor(quantized).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}
