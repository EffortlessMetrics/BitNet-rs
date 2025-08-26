#![cfg(feature = "integration-tests")]
/*!
# Working End-to-End Demo Tests

This module demonstrates the current working capabilities of the BitNet Rust ecosystem
with realistic end-to-end workflows that actually compile and run successfully.
*/

use bitnet_common::{
    BitNetConfig, BitNetTensor, Device, MockTensor, ModelFormat, QuantizationConfig, Tensor,
};
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
use std::fs;
use tempfile::TempDir;

#[test]
fn test_complete_quantization_pipeline() {
    // Create test data
    let mock_tensor = MockTensor::new(vec![64, 64]);
    // Convert MockTensor data to BitNetTensor
    let data = mock_tensor.as_slice::<f32>().unwrap();
    let tensor = BitNetTensor::from_slice(data, mock_tensor.shape(), &Device::Cpu).unwrap();

    // Test I2S quantization workflow
    let i2s_quantizer = I2SQuantizer::new();
    let i2s_result = i2s_quantizer.quantize_tensor(&tensor);
    assert!(i2s_result.is_ok(), "I2S quantization should succeed");

    let quantized = i2s_result.unwrap();
    let dequantized = i2s_quantizer.dequantize_tensor(&quantized, &Device::Cpu);
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
    let mock_tensor = MockTensor::new(vec![32, 32]);
    let data = mock_tensor.as_slice::<f32>().unwrap();
    let tensor = BitNetTensor::from_slice(data, mock_tensor.shape(), &Device::Cpu).unwrap();

    // Test all quantization methods
    let quantizers = vec![
        ("I2S", Box::new(I2SQuantizer::new()) as Box<dyn QuantizerWrapper>),
        ("TL1", Box::new(TL1Wrapper::new())),
        ("TL2", Box::new(TL2Wrapper::new())),
    ];

    println!("\nQuantizer Comparison:");
    println!("{}", "=".repeat(40));

    for (name, quantizer) in quantizers {
        let result = quantizer.quantize(&tensor);
        match result {
            Ok(quantized) => {
                let size_bytes = quantized.data.len();
                let compression_ratio =
                    (tensor.shape().iter().product::<usize>() * 4) as f32 / size_bytes as f32;
                println!(
                    "{:10} | Size: {:6} bytes | Compression: {:.2}x",
                    name, size_bytes, compression_ratio
                );
            }
            Err(e) => {
                println!("{:10} | Error: {}", name, e);
            }
        }
    }
}

#[test]
fn test_provider_selection() {
    // Test automatic provider selection
    println!("Testing quantization with CPU provider");

    // Create test tensor
    let mock_tensor = MockTensor::new(vec![16, 16]);
    let data = mock_tensor.as_slice::<f32>().unwrap();
    let tensor = BitNetTensor::from_slice(data, mock_tensor.shape(), &Device::Cpu).unwrap();

    // Test with selected provider
    let quantizer = I2SQuantizer::new();
    let result = quantizer.quantize_tensor(&tensor);
    assert!(result.is_ok(), "Quantization should work with CPU provider");
}

#[test]
fn test_configuration_management() {
    // Create a test configuration
    let config = BitNetConfig {
        model: bitnet_common::ModelConfig {
            path: Some(std::path::PathBuf::from("test-model.gguf")),
            format: ModelFormat::Gguf,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 50257,
            intermediate_size: 3072,
            max_position_embeddings: 2048,
            rope_theta: None,
            rope_scaling: None,
        },
        inference: bitnet_common::InferenceConfig {
            max_length: 2048,
            max_new_tokens: 512,
            temperature: 0.7,
            top_k: Some(40),
            top_p: Some(0.9),
            repetition_penalty: 1.0,
            seed: None,
        },
        performance: bitnet_common::PerformanceConfig {
            num_threads: Some(4),
            use_gpu: false,
            batch_size: 8,
            memory_limit: Some(2048 * 1024 * 1024),
        },
        quantization: QuantizationConfig::default(),
    };

    // Test configuration validation
    assert_eq!(config.model.path.as_ref().unwrap().to_str().unwrap(), "test-model.gguf");
    assert_eq!(config.inference.max_new_tokens, 512);
    assert_eq!(config.performance.memory_limit, Some(2048 * 1024 * 1024));

    // Test serialization to temp file
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = temp_dir.path().join("test_config.toml");

    // Write configuration
    let toml_string = toml::to_string(&config).expect("Failed to serialize config");
    fs::write(&config_path, toml_string).expect("Failed to write config file");

    // Read it back
    let loaded_config = BitNetConfig::from_file(&config_path).expect("Failed to load config");

    // Verify loaded configuration
    assert_eq!(loaded_config.model.path, config.model.path);
    assert_eq!(loaded_config.inference.max_new_tokens, config.inference.max_new_tokens);
    assert_eq!(loaded_config.performance.memory_limit, config.performance.memory_limit);
}

#[test]
fn test_error_handling() {
    // Test file loading errors
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let nonexistent_path = temp_dir.path().join("nonexistent.toml");

    let load_result = BitNetConfig::from_file(&nonexistent_path);
    assert!(load_result.is_err(), "Loading nonexistent file should fail");

    // Test quantization with edge cases
    let empty_mock = MockTensor::new(vec![0]);
    let data = empty_mock.as_slice::<f32>().unwrap();
    let empty_tensor = BitNetTensor::from_slice(data, empty_mock.shape(), &Device::Cpu).unwrap();
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
        let mock_tensor = MockTensor::new(vec![32, 32]);
        let data = mock_tensor.as_slice::<f32>().unwrap();
        let tensor =
            BitNetTensor::from_slice(data, mock_tensor.shape(), &device).unwrap_or_else(|_| {
                // If device is not available, fall back to CPU
                BitNetTensor::from_slice(data, mock_tensor.shape(), &Device::Cpu).unwrap()
            });

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
        let mock_tensor = MockTensor::new(size.clone());
        println!("Created tensor with shape: {:?}", mock_tensor.shape());

        // Verify tensor is valid
        assert_eq!(mock_tensor.shape(), &size);
        assert!(mock_tensor.as_slice::<f32>().is_ok());

        // Convert to BitNetTensor for quantization
        let data = mock_tensor.as_slice::<f32>().unwrap();
        let tensor = BitNetTensor::from_slice(data, mock_tensor.shape(), &Device::Cpu).unwrap();
        tensors.push(tensor);
    }

    // Test quantization on all tensors
    let quantizer = I2SQuantizer::new();
    for tensor in &tensors {
        let result = quantizer.quantize_tensor(tensor);
        assert!(result.is_ok(), "Should be able to quantize tensor");
    }

    println!("All {} tensors quantized successfully", tensors.len());
}

// Helper trait for abstracting quantizers
trait QuantizerWrapper: Send + Sync {
    fn quantize(
        &self,
        tensor: &BitNetTensor,
    ) -> Result<bitnet_quantization::QuantizedTensor, Box<dyn std::error::Error>>;
}

impl QuantizerWrapper for I2SQuantizer {
    fn quantize(
        &self,
        tensor: &BitNetTensor,
    ) -> Result<bitnet_quantization::QuantizedTensor, Box<dyn std::error::Error>> {
        self.quantize_tensor(tensor).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

// Wrapper for TL1 quantizer
struct TL1Wrapper {
    quantizer: TL1Quantizer,
}

impl TL1Wrapper {
    fn new() -> Self {
        Self { quantizer: TL1Quantizer::new() }
    }
}

impl QuantizerWrapper for TL1Wrapper {
    fn quantize(
        &self,
        tensor: &BitNetTensor,
    ) -> Result<bitnet_quantization::QuantizedTensor, Box<dyn std::error::Error>> {
        self.quantizer
            .quantize_tensor(tensor)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

// Wrapper for TL2 quantizer
struct TL2Wrapper {
    quantizer: TL2Quantizer,
}

impl TL2Wrapper {
    fn new() -> Self {
        Self { quantizer: TL2Quantizer::new() }
    }
}

impl QuantizerWrapper for TL2Wrapper {
    fn quantize(
        &self,
        tensor: &BitNetTensor,
    ) -> Result<bitnet_quantization::QuantizedTensor, Box<dyn std::error::Error>> {
        self.quantizer
            .quantize_tensor(tensor)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}
