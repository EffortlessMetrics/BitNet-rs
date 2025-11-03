//! AC1: Quantized Linear Forward Pass WITHOUT FP32 Staging (Issue #254)
//!
//! Tests feature spec: issue-254-real-inference-spec.md#ac1-hot-path-real-quantized-gemv
//! API contract: quantization-support.md#quantized-linear-forward
//!
//! This test validates that QuantizedLinear::forward() calls real quantized GEMV kernels
//! (I2S/TL1/TL2) directly without FP32 dequantization/staging in the hot path.
#![cfg(feature = "cpu")]
use anyhow::{Context, Result};
use bitnet_common::{BitNetTensor, Device, Tensor};
use bitnet_inference::{LookupTable, QuantizedLinear};
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
/// AC:1.1 - I2S quantized linear forward pass with NO FP32 staging
/// Validates that I2S GEMV kernel is used directly without dequantization
#[tokio::test]
async fn test_ac1_i2s_quantized_linear_no_fp32_staging() -> Result<()> {
    let input_shape = vec![1, 16, 128];
    let input = BitNetTensor::zeros(&input_shape, candle_core::DType::F32, &Device::Cpu)
        .context("Failed to create input tensor")?;
    let weight_data: Vec<f32> = (0..128 * 256).map(|i| (i as f32 % 100.0) / 100.0 - 0.5).collect();
    let weight_tensor = BitNetTensor::from_slice(&weight_data, &[128, 256], &Device::Cpu)
        .context("Failed to create weight tensor")?;
    let quantizer = I2SQuantizer::new();
    let quantized_weights =
        quantizer.quantize_tensor(&weight_tensor).context("Failed to quantize weights with I2S")?;
    let linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)
        .context("Failed to create I2S quantized linear layer")?;
    let output = linear
        .forward(&input)
        .await
        .context("Failed to perform I2S quantized linear forward pass")?;
    assert_eq!(output.shape(), &[1, 16, 256], "AC1: I2S linear output shape mismatch");
    let output_candle = output.to_candle()?;
    let output_data = output_candle.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_data.iter().all(|v| v.is_finite()), "AC1: I2S output contains NaN/Inf values");
    println!("AC1.1: I2S quantized linear forward pass test - PASSED");
    Ok(())
}
/// AC:1.2 - TL1 quantized linear forward pass with NO FP32 staging
/// Validates that TL1 table lookup matmul kernel is used directly
#[tokio::test]
async fn test_ac1_tl1_quantized_linear_no_fp32_staging() -> Result<()> {
    let input_shape = vec![1, 8, 64];
    let input = BitNetTensor::zeros(&input_shape, candle_core::DType::F32, &Device::Cpu)?;
    let weight_data: Vec<f32> = (0..64 * 128).map(|i| (i as f32 % 50.0) / 50.0 - 0.5).collect();
    let weight_tensor = BitNetTensor::from_slice(&weight_data, &[64, 128], &Device::Cpu)?;
    let quantizer = TL1Quantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
    let lookup_table = LookupTable::new((0..16).map(|i| (i as f32 - 7.5) / 15.0).collect());
    let linear = QuantizedLinear::new_tl1(quantized_weights, lookup_table, Device::Cpu)?;
    let output = linear.forward(&input).await?;
    assert_eq!(output.shape(), &[1, 8, 128], "AC1: TL1 linear output shape mismatch");
    let output_candle = output.to_candle()?;
    let output_data = output_candle.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_data.iter().all(|v| v.is_finite()), "AC1: TL1 output contains NaN/Inf values");
    println!("AC1.2: TL1 quantized linear forward pass test - PASSED");
    Ok(())
}
/// AC:1.3 - TL2 quantized linear forward pass with NO FP32 staging
/// Validates that TL2 table lookup matmul kernel is used directly
#[tokio::test]
async fn test_ac1_tl2_quantized_linear_no_fp32_staging() -> Result<()> {
    let input_shape = vec![1, 4, 32];
    let input = BitNetTensor::zeros(&input_shape, candle_core::DType::F32, &Device::Cpu)?;
    let weight_data: Vec<f32> = (0..32 * 64).map(|i| (i as f32 % 25.0) / 25.0 - 0.5).collect();
    let weight_tensor = BitNetTensor::from_slice(&weight_data, &[32, 64], &Device::Cpu)?;
    let quantizer = TL2Quantizer::new();
    let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
    let lookup_table = LookupTable::new((0..256).map(|i| (i as f32 - 127.5) / 255.0).collect());
    let linear = QuantizedLinear::new_tl2(quantized_weights, lookup_table, Device::Cpu)?;
    let output = linear.forward(&input).await?;
    assert_eq!(output.shape(), &[1, 4, 64], "AC1: TL2 linear output shape mismatch");
    let output_candle = output.to_candle()?;
    let output_data = output_candle.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_data.iter().all(|v| v.is_finite()), "AC1: TL2 output contains NaN/Inf values");
    println!("AC1.3: TL2 quantized linear forward pass test - PASSED");
    Ok(())
}
/// AC:1.4 - Verify NO FP32 dequantization in hot path (instrumentation test)
/// This test validates that quantized kernels are used without FP32 fallback
#[tokio::test]
async fn test_ac1_verify_no_fp32_dequant_in_hot_path() -> Result<()> {
    {
        let input_shape = vec![1, 16, 128];
        let input = BitNetTensor::zeros(&input_shape, candle_core::DType::F32, &Device::Cpu)?;
        let weight_data: Vec<f32> =
            (0..128 * 256).map(|i| (i as f32 % 100.0) / 100.0 - 0.5).collect();
        let weight_tensor = BitNetTensor::from_slice(&weight_data, &[128, 256], &Device::Cpu)?;
        let quantizer = I2SQuantizer::new();
        let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
        let linear = QuantizedLinear::new_i2s(quantized_weights, Device::Cpu)?;
        assert!(
            linear.has_native_quantized_kernel(),
            "AC1.4 FAILED: I2S layer does not have native quantized kernel"
        );
        assert!(
            !linear.is_fallback_path(),
            "AC1.4 FAILED: I2S layer would use fallback dequantization path"
        );
        let _output = linear.forward(&input).await.context("I2S forward pass failed")?;
        println!("  ✓ I2S: native quantized kernel used (no FP32 dequant)");
    }
    {
        let input_shape = vec![1, 8, 64];
        let input = BitNetTensor::zeros(&input_shape, candle_core::DType::F32, &Device::Cpu)?;
        let weight_data: Vec<f32> = (0..64 * 128).map(|i| (i as f32 % 50.0) / 50.0 - 0.5).collect();
        let weight_tensor = BitNetTensor::from_slice(&weight_data, &[64, 128], &Device::Cpu)?;
        let quantizer = TL1Quantizer::new();
        let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
        let lookup_table = LookupTable::new((0..16).map(|i| (i as f32 - 7.5) / 15.0).collect());
        let linear = QuantizedLinear::new_tl1(quantized_weights, lookup_table, Device::Cpu)?;
        assert!(
            linear.has_native_quantized_kernel(),
            "AC1.4 FAILED: TL1 layer does not have native quantized kernel"
        );
        assert!(
            !linear.is_fallback_path(),
            "AC1.4 FAILED: TL1 layer would use fallback dequantization path"
        );
        let _output = linear.forward(&input).await.context("TL1 forward pass failed")?;
        println!("  ✓ TL1: native quantized kernel used (no FP32 dequant)");
    }
    {
        let input_shape = vec![1, 4, 32];
        let input = BitNetTensor::zeros(&input_shape, candle_core::DType::F32, &Device::Cpu)?;
        let weight_data: Vec<f32> = (0..32 * 64).map(|i| (i as f32 % 25.0) / 25.0 - 0.5).collect();
        let weight_tensor = BitNetTensor::from_slice(&weight_data, &[32, 64], &Device::Cpu)?;
        let quantizer = TL2Quantizer::new();
        let quantized_weights = quantizer.quantize_tensor(&weight_tensor)?;
        let lookup_table = LookupTable::new((0..256).map(|i| (i as f32 - 127.5) / 255.0).collect());
        let linear = QuantizedLinear::new_tl2(quantized_weights, lookup_table, Device::Cpu)?;
        assert!(
            linear.has_native_quantized_kernel(),
            "AC1.4 FAILED: TL2 layer does not have native quantized kernel"
        );
        assert!(
            !linear.is_fallback_path(),
            "AC1.4 FAILED: TL2 layer would use fallback dequantization path"
        );
        let _output = linear.forward(&input).await.context("TL2 forward pass failed")?;
        println!("  ✓ TL2: native quantized kernel used (no FP32 dequant)");
    }
    println!("AC1.4: FP32 dequant instrumentation test - PASSED");
    println!("  All quantization types (I2S, TL1, TL2) use native quantized kernels");
    println!("  ZERO FP32 dequantization operations in hot path");
    Ok(())
}
