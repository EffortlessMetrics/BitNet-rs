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
#[tokio::test(flavor = "multi_thread")]
async fn test_qk256_dispatch_basic() -> Result<()> {
    let in_features = 256;
    let out_features = 64;
    let batch_size = 2;
    let weight_data: Vec<f32> =
        (0..in_features * out_features).map(|i| (i as f32 * 0.001).sin()).collect();
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[in_features, out_features], &Device::Cpu)?;
    let quantizer = I2SQuantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
    let mut linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;
    let qk256_rows = out_features;
    let qk256_cols = in_features;
    let blocks_per_row = qk256_cols.div_ceil(256);
    let row_stride_bytes = blocks_per_row * 64;
    let total_bytes = qk256_rows * row_stride_bytes;
    let qk256_bytes = vec![0xAAu8; total_bytes];
    linear.set_qk256_data(qk256_bytes, qk256_rows, qk256_cols)?;
    let input_data: Vec<f32> = (0..batch_size * in_features).map(|i| i as f32 * 0.01).collect();
    let input = BitNetTensor::from_slice(&input_data, &[batch_size, in_features], &Device::Cpu)?;
    let output = linear.forward(&input).await?;
    assert_eq!(output.shape(), &[batch_size, out_features]);
    let output_candle = output.to_candle()?;
    let output_vec = output_candle.flatten_all()?.to_vec1::<f32>()?;
    let all_zero = output_vec.iter().all(|&x| x == 0.0);
    assert!(!all_zero, "Output should be non-zero after QK256 GEMV");
    for batch_idx in 0..batch_size {
        let input_start = batch_idx * in_features;
        let input_sum: f32 = input_data[input_start..input_start + in_features].iter().sum();
        let output_start = batch_idx * out_features;
        for out_idx in 0..out_features {
            let output_val = output_vec[output_start + out_idx];
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
#[tokio::test(flavor = "multi_thread")]
async fn test_qk256_with_non_aligned_dims() -> Result<()> {
    let in_features = 300;
    let out_features = 32;
    let batch_size = 1;
    let weight_data: Vec<f32> =
        (0..in_features * out_features).map(|i| (i as f32 * 0.002).cos()).collect();
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[in_features, out_features], &Device::Cpu)?;
    let quantizer = I2SQuantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
    let mut linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;
    let qk256_rows = out_features;
    let qk256_cols = in_features;
    let blocks_per_row = qk256_cols.div_ceil(256);
    let row_stride_bytes = blocks_per_row * 64;
    let total_bytes = qk256_rows * row_stride_bytes;
    let qk256_bytes = vec![0x55u8; total_bytes];
    linear.set_qk256_data(qk256_bytes, qk256_rows, qk256_cols)?;
    let input_data: Vec<f32> = vec![1.0; in_features];
    let input = BitNetTensor::from_slice(&input_data, &[batch_size, in_features], &Device::Cpu)?;
    let output = linear.forward(&input).await?;
    assert_eq!(output.shape(), &[batch_size, out_features]);
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
#[tokio::test(flavor = "multi_thread")]
async fn test_qk256_dimension_validation() -> Result<()> {
    let in_features = 256;
    let out_features = 64;
    let weight_data: Vec<f32> = vec![0.0; in_features * out_features];
    let weight_tensor =
        BitNetTensor::from_slice(&weight_data, &[in_features, out_features], &Device::Cpu)?;
    let quantizer = I2SQuantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
    let mut linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;
    let qk256_bytes = vec![0xAAu8; 32 * 64];
    let result = linear.set_qk256_data(qk256_bytes, 32, in_features);
    assert!(result.is_err(), "Should reject QK256 data with wrong rows");
    let qk256_bytes = vec![0xAAu8; out_features * 64];
    let result = linear.set_qk256_data(qk256_bytes, out_features, 128);
    assert!(result.is_err(), "Should reject QK256 data with wrong cols");
    let qk256_bytes = vec![0xAAu8; 100];
    let result = linear.set_qk256_data(qk256_bytes, out_features, in_features);
    assert!(result.is_err(), "Should reject QK256 data with wrong byte count");
    println!("✓ QK256 dimension validation test passed");
    Ok(())
}
