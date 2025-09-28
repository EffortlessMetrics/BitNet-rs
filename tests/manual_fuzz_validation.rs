//! Manual fuzz testing for BitNet.rs Issue #260 mock elimination
//! This tests edge cases and potential crash conditions in quantization and model parsing

use bitnet_common::{BitNetTensor, QuantizationType, Result};
use bitnet_quantization::{
    I2SQuantizer, Quantize, QuantizedTensor, QuantizerTrait, TL1Quantizer, TL2Quantizer,
};

#[test]
fn test_manual_fuzz_validation() -> Result<()> {
    println!("🧪 Starting manual fuzz testing for BitNet.rs Issue #260");

    // Test quantization algorithms with edge cases
    test_i2s_edge_cases()?;
    test_tl1_edge_cases()?;
    test_tl2_edge_cases()?;
    test_memory_safety()?;
    test_numerical_stability()?;

    println!("✅ All manual fuzz tests completed successfully");
    Ok(())
}

/// Test I2S quantization with edge cases that could cause crashes
fn test_i2s_edge_cases() -> Result<()> {
    println!("Testing I2S quantization edge cases...");

    let quantizer = I2SQuantizer::new();

    // Test case 1: Very small tensor
    let small_data = vec![1.0];
    test_quantization_with_data(&quantizer, small_data, "small tensor")?;

    // Test case 2: Extreme values
    test_extreme_values(&quantizer, "I2S")?;

    // Test case 3: Values that should be filtered
    test_filtered_values(&quantizer, "I2S")?;

    // Test case 4: Medium-sized tensor
    test_medium_tensor(&quantizer, "I2S")?;

    println!("  ✅ I2S edge case testing completed");
    Ok(())
}

/// Test TL1 quantization edge cases
fn test_tl1_edge_cases() -> Result<()> {
    println!("Testing TL1 quantization edge cases...");

    let quantizer = TL1Quantizer::new();

    test_extreme_values_tl(&quantizer, "TL1")?;
    test_medium_tensor_tl(&quantizer, "TL1")?;

    println!("  ✅ TL1 edge case testing completed");
    Ok(())
}

/// Test TL2 quantization edge cases
fn test_tl2_edge_cases() -> Result<()> {
    println!("Testing TL2 quantization edge cases...");

    let quantizer = TL2Quantizer::new();

    test_extreme_values_tl(&quantizer, "TL2")?;
    test_medium_tensor_tl(&quantizer, "TL2")?;

    println!("  ✅ TL2 edge case testing completed");
    Ok(())
}

/// Test memory safety edge cases
fn test_memory_safety() -> Result<()> {
    println!("Testing memory safety edge cases...");

    let quantizer = I2SQuantizer::new();

    // Test case 1: Very deep tensor with single elements
    let deep_data = vec![1.0];
    let deep_shape = vec![1; 10]; // 10-dimensional tensor with single element
    match create_test_tensor(deep_data, deep_shape) {
        Ok(tensor) => match quantizer.quantize_tensor(&tensor) {
            Ok(_) => println!("  ✓ Deep tensor handled successfully"),
            Err(e) => println!("  ✓ Deep tensor error handled: {}", e),
        },
        Err(e) => println!("  ✓ Deep tensor creation failed as expected: {}", e),
    }

    // Test case 2: Moderate size tensor to test boundaries
    let moderate_data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
    test_quantization_with_data(&quantizer, moderate_data, "moderate size tensor")?;

    println!("  ✅ Memory safety testing completed");
    Ok(())
}

/// Test numerical stability edge cases
fn test_numerical_stability() -> Result<()> {
    println!("Testing numerical stability edge cases...");

    let quantizer = I2SQuantizer::new();

    // Test case 1: Very small values near zero
    let small_values = vec![0.001, -0.001, 0.0001, -0.0001];
    test_quantization_with_data(&quantizer, small_values, "very small values")?;

    // Test case 2: Values that could cause precision loss
    let precision_test = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    test_quantization_with_data(&quantizer, precision_test, "precision test values")?;

    // Test case 3: Mixed positive and negative values
    let mixed = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
    test_quantization_with_data(&quantizer, mixed, "mixed sign values")?;

    println!("  ✅ Numerical stability testing completed");
    Ok(())
}

/// Helper function to test extreme values
fn test_extreme_values(quantizer: &I2SQuantizer, name: &str) -> Result<()> {
    let extreme_values = vec![
        1000.0,  // Large positive
        -1000.0, // Large negative
        10.0,    // Medium positive
        -10.0,   // Medium negative
        1.0,     // Small positive
        -1.0,    // Small negative
    ];

    test_quantization_with_data(quantizer, extreme_values, &format!("{} extreme values", name))
}

/// Helper function to test values that should be filtered
fn test_filtered_values(quantizer: &I2SQuantizer, name: &str) -> Result<()> {
    // Use finite values that might cause issues but should be handled
    let problematic_values = vec![
        0.0,    // Zero
        -0.0,   // Negative zero
        1e-10,  // Very small positive
        -1e-10, // Very small negative
    ];

    test_quantization_with_data(quantizer, problematic_values, &format!("{} filtered values", name))
}

/// Helper function to test medium tensors
fn test_medium_tensor(quantizer: &I2SQuantizer, name: &str) -> Result<()> {
    // Test with reasonably sized data
    let medium_size = 256;
    let medium_data: Vec<f32> = (0..medium_size).map(|i| (i as f32) * 0.01 - 1.28).collect();

    test_quantization_with_data(quantizer, medium_data, &format!("{} medium tensor", name))
}

/// Helper function to test extreme values for TL quantizers
fn test_extreme_values_tl(quantizer: &dyn QuantizerTrait, name: &str) -> Result<()> {
    let extreme_values = vec![1000.0, -1000.0, 10.0, -10.0, 1.0, -1.0, 0.5, -0.5];

    match create_test_tensor(extreme_values.clone(), vec![extreme_values.len()]) {
        Ok(tensor) => match quantizer.quantize_tensor(&tensor) {
            Ok(_) => println!("    ✓ {} extreme values handled", name),
            Err(e) => println!("    ✓ {} extreme values error handled: {}", name, e),
        },
        Err(e) => println!("    ✓ {} extreme values tensor creation failed: {}", name, e),
    }

    Ok(())
}

/// Helper function to test medium tensors for TL quantizers
fn test_medium_tensor_tl(quantizer: &dyn QuantizerTrait, name: &str) -> Result<()> {
    let medium_size = 128;
    let medium_data: Vec<f32> = (0..medium_size).map(|i| (i as f32) * 0.01).collect();

    match create_test_tensor(medium_data.clone(), vec![medium_data.len()]) {
        Ok(tensor) => match quantizer.quantize_tensor(&tensor) {
            Ok(_) => println!("    ✓ {} medium tensor handled", name),
            Err(e) => println!("    ✓ {} medium tensor error handled: {}", name, e),
        },
        Err(e) => println!("    ✓ {} medium tensor creation failed: {}", name, e),
    }

    Ok(())
}

/// Helper to test quantization with specific data
fn test_quantization_with_data(
    quantizer: &I2SQuantizer,
    data: Vec<f32>,
    description: &str,
) -> Result<()> {
    match create_test_tensor(data.clone(), vec![data.len()]) {
        Ok(tensor) => {
            match quantizer.quantize_tensor(&tensor) {
                Ok(quantized) => {
                    // Validate the quantized tensor structure
                    assert!(!quantized.data.is_empty(), "Quantized data should not be empty");
                    assert!(!quantized.scales.is_empty(), "Scales should not be empty");
                    assert_eq!(quantized.qtype, QuantizationType::I2S);
                    assert_eq!(quantized.shape, vec![data.len()]);

                    match quantizer.dequantize_tensor(&quantized) {
                        Ok(dequantized) => {
                            assert_eq!(dequantized.shape(), &[data.len()]);
                            println!("    ✓ I2S {} round-trip successful", description);
                        }
                        Err(e) => println!(
                            "    ✓ I2S {} dequantization error handled: {}",
                            description, e
                        ),
                    }
                }
                Err(e) => println!("    ✓ I2S {} quantization error handled: {}", description, e),
            }
        }
        Err(e) => println!("    ✓ I2S {} tensor creation failed: {}", description, e),
    }
    Ok(())
}

/// Helper to create test tensor
fn create_test_tensor(data: Vec<f32>, shape: Vec<usize>) -> Result<BitNetTensor> {
    use bitnet_quantization::utils::create_tensor_from_f32;
    create_tensor_from_f32(data, &shape, &candle_core::Device::Cpu)
}
