//! Unit tests for QK256 dispatch in QuantizedLinear
//!
//! This test verifies that:
//! 1. QK256 data can be set on a QuantizedLinear layer
//! 2. Forward pass automatically detects and uses QK256 kernel
//! 3. Output shape and numerical correctness are maintained

use anyhow::Result;
use bitnet_common::{BitNetTensor, Device, Tensor};
use bitnet_inference::layers::quantized_linear::QuantizedLinear;
use bitnet_quantization::I2SQuantizer;

#[tokio::test]
async fn test_qk256_dispatch_basic() -> Result<()> {
    // Test dimensions
    let in_features = 256; // Single QK256 block
    let out_features = 64;
    let batch_size = 2;

    // Create dummy weight tensor (shape: [in_features, out_features])
    let weight_data: Vec<f32> =
        (0..in_features * out_features).map(|i| (i as f32 * 0.001).sin()).collect();
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[in_features, out_features], &Device::Cpu)?;

    // Quantize weights using I2S quantizer (for initialization)
    let quantizer = I2SQuantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;

    // Create linear layer (in_features from shape[0], out_features from shape[1])
    let mut linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;

    // Create QK256 data (all codes = 2 → +1.0 with default LUT)
    // QK256 format: rows = out_features, cols = in_features
    let qk256_rows = out_features;
    let qk256_cols = in_features;
    let blocks_per_row = (qk256_cols + 255) / 256; // ceil(cols/256)
    let row_stride_bytes = blocks_per_row * 64; // 64 bytes per block
    let total_bytes = qk256_rows * row_stride_bytes;
    let qk256_bytes = vec![0xAAu8; total_bytes]; // 0xAA = 0b_10_10_10_10 (all codes = 2)

    // Set QK256 data
    linear.set_qk256_data(qk256_bytes, qk256_rows, qk256_cols)?;

    // Create input tensor
    let input_data: Vec<f32> = (0..batch_size * in_features).map(|i| i as f32 * 0.01).collect();
    let input = BitNetTensor::from_slice(&input_data, &[batch_size, in_features], &Device::Cpu)?;

    // Forward pass (should use QK256 kernel)
    let output = linear.forward(&input).await?;

    // Verify output shape
    assert_eq!(output.shape(), &[batch_size, out_features]);

    // Verify output is non-zero (computation occurred)
    let output_candle = output.to_candle()?;
    let output_vec = output_candle.flatten_all()?.to_vec1::<f32>()?;
    let all_zero = output_vec.iter().all(|&x| x == 0.0);
    assert!(!all_zero, "Output should be non-zero after QK256 GEMV");

    // Verify numerical properties (code 2 → +1.0, so output ≈ sum of input)
    for batch_idx in 0..batch_size {
        let input_start = batch_idx * in_features;
        let input_sum: f32 = input_data[input_start..input_start + in_features].iter().sum();

        let output_start = batch_idx * out_features;
        for out_idx in 0..out_features {
            let output_val = output_vec[output_start + out_idx];
            // Each output feature is dot product of input with row of all +1.0 weights
            // So output ≈ sum(input) = input_sum
            assert!(
                (output_val - input_sum).abs() < 1.0,
                "Output value {} should be close to input sum {}",
                output_val,
                input_sum
            );
        }
    }

    println!("✓ QK256 dispatch test passed: output shape and numerical correctness verified");
    Ok(())
}

#[tokio::test]
async fn test_qk256_with_non_aligned_dims() -> Result<()> {
    // Test with dimensions not multiple of 256
    let in_features = 300; // Requires 2 QK256 blocks (256 + 44)
    let out_features = 32;
    let batch_size = 1;

    // Create dummy weight tensor (shape: [in_features, out_features])
    let weight_data: Vec<f32> =
        (0..in_features * out_features).map(|i| (i as f32 * 0.002).cos()).collect();
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[in_features, out_features], &Device::Cpu)?;

    // Quantize weights
    let quantizer = I2SQuantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;

    // Create linear layer
    let mut linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;

    // Create QK256 data
    // QK256 format: rows = out_features, cols = in_features
    let qk256_rows = out_features;
    let qk256_cols = in_features;
    let blocks_per_row = (qk256_cols + 255) / 256; // = 2
    let row_stride_bytes = blocks_per_row * 64; // = 128 bytes
    let total_bytes = qk256_rows * row_stride_bytes;
    let qk256_bytes = vec![0x55u8; total_bytes]; // 0x55 = 0b_01_01_01_01 (all codes = 1 → -1.0)

    // Set QK256 data
    linear.set_qk256_data(qk256_bytes, qk256_rows, qk256_cols)?;

    // Create input tensor
    let input_data: Vec<f32> = vec![1.0; in_features];
    let input = BitNetTensor::from_slice(&input_data, &[batch_size, in_features], &Device::Cpu)?;

    // Forward pass
    let output = linear.forward(&input).await?;

    // Verify output shape
    assert_eq!(output.shape(), &[batch_size, out_features]);

    // Verify output is negative (code 1 → -1.0, input all +1.0, so output ≈ -in_features)
    let output_candle = output.to_candle()?;
    let output_vec = output_candle.flatten_all()?.to_vec1::<f32>()?;

    for &val in &output_vec {
        assert!(val < 0.0, "Output {} should be negative (weights all -1.0, input all +1.0)", val);
        assert!(
            (val + in_features as f32).abs() < 10.0,
            "Output {} should be close to -{} (sum of negative weights)",
            val,
            in_features
        );
    }

    println!("✓ QK256 non-aligned dimensions test passed");
    Ok(())
}

#[tokio::test]
async fn test_qk256_dimension_validation() -> Result<()> {
    // Test that dimension mismatches are caught
    let in_features = 256;
    let out_features = 64;

    // Create dummy weight tensor (shape: [in_features, out_features])
    let weight_data: Vec<f32> = vec![0.0; in_features * out_features];
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[in_features, out_features], &Device::Cpu)?;

    // Quantize weights
    let quantizer = I2SQuantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;

    // Create linear layer
    let mut linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;

    // Try to set QK256 data with wrong rows
    let qk256_bytes = vec![0xAAu8; 32 * 64]; // 32 rows instead of 64
    let result = linear.set_qk256_data(qk256_bytes, 32, in_features);
    assert!(result.is_err(), "Should reject QK256 data with wrong rows");

    // Try to set QK256 data with wrong cols
    let qk256_bytes = vec![0xAAu8; out_features * 64];
    let result = linear.set_qk256_data(qk256_bytes, out_features, 128);
    assert!(result.is_err(), "Should reject QK256 data with wrong cols");

    // Try to set QK256 data with wrong byte count
    let qk256_bytes = vec![0xAAu8; 100]; // Wrong size
    let result = linear.set_qk256_data(qk256_bytes, out_features, in_features);
    assert!(result.is_err(), "Should reject QK256 data with wrong byte count");

    println!("✓ QK256 dimension validation test passed");
    Ok(())
}
