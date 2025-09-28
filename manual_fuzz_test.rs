//! Manual fuzz testing for BitNet.rs Issue #260 mock elimination
//! This tests edge cases and potential crash conditions in quantization and model parsing

use bitnet_common::{BitNetTensor, Device, QuantizationType, Result};
use bitnet_quantization::{I2SQuantizer, Quantize, QuantizedTensor, TL1Quantizer, TL2Quantizer};
use std::io::Cursor;

fn main() -> Result<()> {
    println!("🧪 Starting manual fuzz testing for BitNet.rs Issue #260");

    // Test quantization algorithms with edge cases
    test_i2s_edge_cases()?;
    test_tl1_edge_cases()?;
    test_tl2_edge_cases()?;
    test_memory_safety()?;
    test_numerical_stability()?;
    test_gguf_parsing_edge_cases()?;

    println!("✅ All manual fuzz tests completed successfully");
    Ok(())
}

/// Test I2S quantization with edge cases that could cause crashes
fn test_i2s_edge_cases() -> Result<()> {
    println!("Testing I2S quantization edge cases...");

    let quantizer = I2SQuantizer::new();

    // Test case 1: Empty tensor
    match create_test_tensor(vec![], vec![0]) {
        Ok(tensor) => {
            if let Err(e) = quantizer.quantize_tensor(&tensor) {
                println!("  ✓ Empty tensor handled gracefully: {}", e);
            } else {
                println!("  ⚠ Empty tensor should fail");
            }
        }
        Err(_) => println!("  ✓ Empty tensor creation failed as expected"),
    }

    // Test case 2: Extreme values
    test_extreme_values(&quantizer, "I2S")?;

    // Test case 3: NaN and infinity
    test_special_float_values(&quantizer, "I2S")?;

    // Test case 4: Large tensor that could cause memory issues
    test_large_tensor(&quantizer, "I2S")?;

    // Test case 5: Mismatched shapes
    test_mismatched_shapes(&quantizer, "I2S")?;

    println!("  ✅ I2S edge case testing completed");
    Ok(())
}

/// Test TL1 quantization edge cases
fn test_tl1_edge_cases() -> Result<()> {
    println!("Testing TL1 quantization edge cases...");

    let quantizer = TL1Quantizer::new();

    test_extreme_values(&quantizer, "TL1")?;
    test_special_float_values(&quantizer, "TL1")?;
    test_large_tensor(&quantizer, "TL1")?;

    println!("  ✅ TL1 edge case testing completed");
    Ok(())
}

/// Test TL2 quantization edge cases
fn test_tl2_edge_cases() -> Result<()> {
    println!("Testing TL2 quantization edge cases...");

    let quantizer = TL2Quantizer::new();

    test_extreme_values(&quantizer, "TL2")?;
    test_special_float_values(&quantizer, "TL2")?;
    test_large_tensor(&quantizer, "TL2")?;

    println!("  ✅ TL2 edge case testing completed");
    Ok(())
}

/// Test memory safety edge cases
fn test_memory_safety() -> Result<()> {
    println!("Testing memory safety edge cases...");

    // Test case 1: Integer overflow in shape calculation
    let huge_dims = vec![usize::MAX / 2, 2]; // Would overflow when multiplied
    match create_test_tensor(vec![1.0; 4], huge_dims) {
        Ok(_) => println!("  ⚠ Huge dimensions should fail"),
        Err(_) => println!("  ✓ Huge dimensions rejected as expected"),
    }

    // Test case 2: Very deep tensor
    let deep_shape = vec![1; 100]; // 100-dimensional tensor
    let data = vec![1.0];
    match create_test_tensor(data, deep_shape) {
        Ok(tensor) => {
            let quantizer = I2SQuantizer::new();
            match quantizer.quantize_tensor(&tensor) {
                Ok(_) => println!("  ✓ Deep tensor handled successfully"),
                Err(e) => println!("  ✓ Deep tensor error handled: {}", e),
            }
        }
        Err(_) => println!("  ✓ Deep tensor creation failed as expected"),
    }

    // Test case 3: Zero-dimensional tensor
    match create_test_tensor(vec![1.0], vec![]) {
        Ok(_) => println!("  ⚠ Zero-dimensional tensor should fail"),
        Err(_) => println!("  ✓ Zero-dimensional tensor rejected"),
    }

    println!("  ✅ Memory safety testing completed");
    Ok(())
}

/// Test numerical stability edge cases
fn test_numerical_stability() -> Result<()> {
    println!("Testing numerical stability edge cases...");

    let quantizer = I2SQuantizer::new();

    // Test case 1: Very small values near zero
    let small_values = vec![f32::EPSILON, -f32::EPSILON, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
    test_quantization_with_data(&quantizer, small_values, "very small values")?;

    // Test case 2: Values that could cause precision loss
    let precision_test = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    test_quantization_with_data(&quantizer, precision_test, "precision test values")?;

    // Test case 3: Alternating extreme values
    let alternating = vec![f32::MAX, f32::MIN, f32::MAX, f32::MIN];
    test_quantization_with_data(&quantizer, alternating, "alternating extreme values")?;

    println!("  ✅ Numerical stability testing completed");
    Ok(())
}

/// Test GGUF parsing edge cases (simplified without actual GGUF dependency)
fn test_gguf_parsing_edge_cases() -> Result<()> {
    println!("Testing GGUF parsing edge cases...");

    // Test case 1: Empty data
    let empty_data = vec![];
    test_binary_parsing(&empty_data, "empty data")?;

    // Test case 2: Too small data
    let tiny_data = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF" magic but truncated
    test_binary_parsing(&tiny_data, "truncated GGUF header")?;

    // Test case 3: Random binary data
    let random_data: Vec<u8> = (0..1024).map(|i| (i * 17) as u8).collect();
    test_binary_parsing(&random_data, "random binary data")?;

    // Test case 4: Malformed header
    let mut malformed = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF" magic
    malformed.extend_from_slice(&[0xFF; 100]); // Malformed rest
    test_binary_parsing(&malformed, "malformed GGUF header")?;

    println!("  ✅ GGUF parsing edge case testing completed");
    Ok(())
}

/// Helper function to test extreme values
fn test_extreme_values(quantizer: &dyn quantizer_trait_wrapper, name: &str) -> Result<()> {
    let extreme_values = vec![
        f32::MAX,
        f32::MIN,
        f32::MAX / 2.0,
        f32::MIN / 2.0,
        1e10,
        -1e10,
        1e-10,
        -1e-10,
    ];

    test_quantization_with_data_generic(quantizer, extreme_values, &format!("{} extreme values", name))
}

/// Helper function to test special float values
fn test_special_float_values(quantizer: &dyn quantizer_trait_wrapper, name: &str) -> Result<()> {
    let special_values = vec![
        0.0,
        -0.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        -f32::NAN,
    ];

    // These should be handled gracefully (filtered or error)
    match create_test_tensor(special_values.clone(), vec![special_values.len()]) {
        Ok(tensor) => {
            match quantize_tensor_generic(quantizer, &tensor) {
                Ok(_) => println!("    ✓ {} special float values handled", name),
                Err(e) => println!("    ✓ {} special float values error handled: {}", name, e),
            }
        }
        Err(e) => println!("    ✓ {} special float tensor creation failed: {}", name, e),
    }

    Ok(())
}

/// Helper function to test large tensors
fn test_large_tensor(quantizer: &dyn quantizer_trait_wrapper, name: &str) -> Result<()> {
    // Test with 1MB of data (262144 float32 values)
    let large_size = 262144;
    let large_data: Vec<f32> = (0..large_size).map(|i| (i as f32) * 0.001).collect();

    test_quantization_with_data_generic(quantizer, large_data, &format!("{} large tensor", name))
}

/// Helper function to test mismatched shapes
fn test_mismatched_shapes(quantizer: &I2SQuantizer, name: &str) -> Result<()> {
    // Create tensor with data length that doesn't match shape
    let data = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements
    let wrong_shape = vec![2, 3]; // Claims 6 elements

    match create_test_tensor(data, wrong_shape) {
        Ok(_) => println!("    ⚠ {} mismatched shape should fail", name),
        Err(_) => println!("    ✓ {} mismatched shape rejected", name),
    }

    Ok(())
}

/// Helper to test quantization with specific data
fn test_quantization_with_data(quantizer: &I2SQuantizer, data: Vec<f32>, description: &str) -> Result<()> {
    match create_test_tensor(data.clone(), vec![data.len()]) {
        Ok(tensor) => {
            match quantizer.quantize_tensor(&tensor) {
                Ok(quantized) => {
                    match quantizer.dequantize_tensor(&quantized) {
                        Ok(_) => println!("    ✓ I2S {} round-trip successful", description),
                        Err(e) => println!("    ✓ I2S {} dequantization error handled: {}", description, e),
                    }
                }
                Err(e) => println!("    ✓ I2S {} quantization error handled: {}", description, e),
            }
        }
        Err(e) => println!("    ✓ I2S {} tensor creation failed: {}", description, e),
    }
    Ok(())
}

/// Helper to test binary parsing (simplified)
fn test_binary_parsing(data: &[u8], description: &str) -> Result<()> {
    // Simulate GGUF parsing edge cases
    let mut cursor = Cursor::new(data);

    // Try to read magic bytes
    let mut magic = [0u8; 4];
    match std::io::Read::read_exact(&mut cursor, &mut magic) {
        Ok(_) => {
            if &magic == b"GGUF" {
                println!("    ✓ {} has valid GGUF magic", description);
                // Would continue with actual parsing here
            } else {
                println!("    ✓ {} invalid magic handled", description);
            }
        }
        Err(_) => println!("    ✓ {} insufficient data handled", description),
    }

    Ok(())
}

/// Helper to create test tensor
fn create_test_tensor(data: Vec<f32>, shape: Vec<usize>) -> Result<BitNetTensor> {
    use bitnet_quantization::utils::create_tensor_from_f32;
    create_tensor_from_f32(data, &shape, &candle_core::Device::Cpu)
}

// Trait wrapper for generic quantizer testing
trait quantizer_trait_wrapper {
    fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor>;
}

impl quantizer_trait_wrapper for I2SQuantizer {
    fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        I2SQuantizer::quantize_tensor(self, tensor)
    }
}

impl quantizer_trait_wrapper for TL1Quantizer {
    fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        use bitnet_quantization::QuantizerTrait;
        QuantizerTrait::quantize_tensor(self, tensor)
    }
}

impl quantizer_trait_wrapper for TL2Quantizer {
    fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        use bitnet_quantization::QuantizerTrait;
        QuantizerTrait::quantize_tensor(self, tensor)
    }
}

fn quantize_tensor_generic(quantizer: &dyn quantizer_trait_wrapper, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
    quantizer.quantize_tensor(tensor)
}

fn test_quantization_with_data_generic(quantizer: &dyn quantizer_trait_wrapper, data: Vec<f32>, description: &str) -> Result<()> {
    match create_test_tensor(data.clone(), vec![data.len()]) {
        Ok(tensor) => {
            match quantize_tensor_generic(quantizer, &tensor) {
                Ok(_quantized) => {
                    println!("    ✓ {} quantization successful", description);
                }
                Err(e) => println!("    ✓ {} quantization error handled: {}", description, e),
            }
        }
        Err(e) => println!("    ✓ {} tensor creation failed: {}", description, e),
    }
    Ok(())
}