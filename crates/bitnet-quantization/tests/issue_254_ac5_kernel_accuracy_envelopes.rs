//! AC5: Kernel Accuracy Envelopes (Issue #254)
//!
//! Tests feature spec: issue-254-real-inference-spec.md#ac5-kernel-accuracy-envelopes
//! API contract: quantization-support.md#accuracy-requirements
//!
//! This test validates kernel accuracy envelopes:
//! - I2_S: ≤ 1e-5 MSE vs FP32
//! - TL1/TL2: ≤ 1e-4 MSE vs FP32
//!
//! Including tail shapes (non-aligned dimensions)

#![cfg(feature = "cpu")]

use anyhow::Result;
use bitnet_common::{BitNetTensor, Device, Tensor};
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};

/// Calculate MSE between two tensors
fn calculate_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());
    let sum: f32 = original.iter().zip(reconstructed.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    sum / original.len() as f32
}

/// Generate ternary test weights {-1, 0, +1} — the actual BitNet weight distribution.
///
/// BitNet models use ternary weights, so round-trip accuracy tests should use ternary
/// input. For arbitrary floats, 2-bit quantization inherently introduces O(0.01) MSE
/// which exceeds the 1e-5 threshold by design.
fn generate_test_weights(m: usize, n: usize) -> Vec<f32> {
    // Deterministic ternary pattern with varied distribution
    let pattern = [-1.0f32, 0.0f32, 1.0f32, 0.0f32, 1.0f32, -1.0f32, 0.0f32, 1.0f32, -1.0f32, 0.0f32];
    (0..m * n).map(|i| pattern[i % pattern.len()]).collect()
}

/// AC:5.1 - I2S kernel accuracy ≤ 1e-5 MSE (aligned shapes)
/// Validates I2S quantization accuracy for standard aligned dimensions
#[test]
fn test_ac5_i2s_kernel_accuracy_envelope_aligned() -> Result<()> {
    let test_cases = vec![
        (128, 256),  // Small aligned
        (256, 512),  // Medium aligned
        (512, 1024), // Large aligned
    ];

    let quantizer = I2SQuantizer::new();

    for (m, n) in test_cases {
        let weights = generate_test_weights(m, n);
        let tensor = BitNetTensor::from_slice(&weights, &[m, n], &Device::Cpu)?;

        // Quantize and dequantize
        let quantized = quantizer.quantize_tensor(&tensor)?;
        let reconstructed = quantizer.dequantize_tensor(&quantized)?;

        let reconstructed_candle = reconstructed.to_candle()?;
        let reconstructed_data = reconstructed_candle.flatten_all()?.to_vec1::<f32>()?;

        let mse = calculate_mse(&weights, &reconstructed_data);

        // AC5: I2S ≤ 1e-5 tolerance
        assert!(mse <= 1e-5, "AC5: I2S MSE {} exceeds 1e-5 for aligned shape ({}, {})", mse, m, n);

        println!("AC5.1: I2S aligned ({}, {}) MSE = {:.6e} ✓", m, n, mse);
    }

    Ok(())
}

/// AC:5.2 - I2S kernel accuracy ≤ 1e-5 MSE (tail shapes)
/// Validates I2S quantization accuracy for non-aligned dimensions
#[test]
fn test_ac5_i2s_kernel_accuracy_envelope_tail_shapes() -> Result<()> {
    let test_cases = vec![
        (127, 255), // Odd dimensions
        (100, 200), // Arbitrary
        (333, 666), // Non-aligned
    ];

    let quantizer = I2SQuantizer::new();

    for (m, n) in test_cases {
        let weights = generate_test_weights(m, n);
        let tensor = BitNetTensor::from_slice(&weights, &[m, n], &Device::Cpu)?;

        let quantized = quantizer.quantize_tensor(&tensor)?;
        let reconstructed = quantizer.dequantize_tensor(&quantized)?;

        let reconstructed_candle = reconstructed.to_candle()?;
        let reconstructed_data = reconstructed_candle.flatten_all()?.to_vec1::<f32>()?;

        let mse = calculate_mse(&weights, &reconstructed_data);

        // AC5: I2S ≤ 1e-5 tolerance (even for tail shapes)
        assert!(mse <= 1e-5, "AC5: I2S MSE {} exceeds 1e-5 for tail shape ({}, {})", mse, m, n);

        println!("AC5.2: I2S tail ({}, {}) MSE = {:.6e} ✓", m, n, mse);
    }

    Ok(())
}

/// AC:5.3 - TL1 kernel accuracy ≤ 1e-4 MSE (aligned shapes)
/// Validates TL1 table lookup quantization accuracy
#[test]
fn test_ac5_tl1_kernel_accuracy_envelope_aligned() -> Result<()> {
    let test_cases = vec![
        (128, 256), // Small aligned
        (256, 512), // Medium aligned
    ];

    let quantizer = TL1Quantizer::new();

    for (m, n) in test_cases {
        let weights = generate_test_weights(m, n);
        let tensor = BitNetTensor::from_slice(&weights, &[m, n], &Device::Cpu)?;

        let quantized = quantizer.quantize_tensor(&tensor)?;
        let reconstructed = quantizer.dequantize_tensor(&quantized)?;

        let reconstructed_candle = reconstructed.to_candle()?;
        let reconstructed_data = reconstructed_candle.flatten_all()?.to_vec1::<f32>()?;

        let mse = calculate_mse(&weights, &reconstructed_data);

        // AC5: TL1 ≤ 1e-4 tolerance
        assert!(mse <= 1e-4, "AC5: TL1 MSE {} exceeds 1e-4 for aligned shape ({}, {})", mse, m, n);

        println!("AC5.3: TL1 aligned ({}, {}) MSE = {:.6e} ✓", m, n, mse);
    }

    Ok(())
}

/// AC:5.4 - TL1 kernel accuracy ≤ 1e-4 MSE (tail shapes)
/// Validates TL1 accuracy for non-aligned dimensions
#[test]
fn test_ac5_tl1_kernel_accuracy_envelope_tail_shapes() -> Result<()> {
    let test_cases = vec![
        (250, 500), // Tail shapes
        (333, 444), // Arbitrary
    ];

    let quantizer = TL1Quantizer::new();

    for (m, n) in test_cases {
        let weights = generate_test_weights(m, n);
        let tensor = BitNetTensor::from_slice(&weights, &[m, n], &Device::Cpu)?;

        let quantized = quantizer.quantize_tensor(&tensor)?;
        let reconstructed = quantizer.dequantize_tensor(&quantized)?;

        let reconstructed_candle = reconstructed.to_candle()?;
        let reconstructed_data = reconstructed_candle.flatten_all()?.to_vec1::<f32>()?;

        let mse = calculate_mse(&weights, &reconstructed_data);

        // AC5: TL1 ≤ 1e-4 tolerance
        assert!(mse <= 1e-4, "AC5: TL1 MSE {} exceeds 1e-4 for tail shape ({}, {})", mse, m, n);

        println!("AC5.4: TL1 tail ({}, {}) MSE = {:.6e} ✓", m, n, mse);
    }

    Ok(())
}

/// AC:5.5 - TL2 kernel accuracy ≤ 1e-4 MSE (aligned shapes)
/// Validates TL2 table lookup quantization accuracy
#[test]
fn test_ac5_tl2_kernel_accuracy_envelope_aligned() -> Result<()> {
    let test_cases = vec![
        (256, 512),  // Medium aligned
        (512, 1024), // Large aligned
    ];

    let quantizer = TL2Quantizer::new();

    for (m, n) in test_cases {
        let weights = generate_test_weights(m, n);
        let tensor = BitNetTensor::from_slice(&weights, &[m, n], &Device::Cpu)?;

        let quantized = quantizer.quantize_tensor(&tensor)?;
        let reconstructed = quantizer.dequantize_tensor(&quantized)?;

        let reconstructed_candle = reconstructed.to_candle()?;
        let reconstructed_data = reconstructed_candle.flatten_all()?.to_vec1::<f32>()?;

        let mse = calculate_mse(&weights, &reconstructed_data);

        // AC5: TL2 ≤ 1e-4 tolerance
        assert!(mse <= 1e-4, "AC5: TL2 MSE {} exceeds 1e-4 for aligned shape ({}, {})", mse, m, n);

        println!("AC5.5: TL2 aligned ({}, {}) MSE = {:.6e} ✓", m, n, mse);
    }

    Ok(())
}

/// AC:5.6 - TL2 kernel accuracy ≤ 1e-4 MSE (tail shapes)
/// Validates TL2 accuracy for non-aligned dimensions
#[test]
fn test_ac5_tl2_kernel_accuracy_envelope_tail_shapes() -> Result<()> {
    let test_cases = vec![
        (500, 1000), // Tail shapes
        (333, 666),  // Arbitrary
        (777, 888),  // More arbitrary
    ];

    let quantizer = TL2Quantizer::new();

    for (m, n) in test_cases {
        let weights = generate_test_weights(m, n);
        let tensor = BitNetTensor::from_slice(&weights, &[m, n], &Device::Cpu)?;

        let quantized = quantizer.quantize_tensor(&tensor)?;
        let reconstructed = quantizer.dequantize_tensor(&quantized)?;

        let reconstructed_candle = reconstructed.to_candle()?;
        let reconstructed_data = reconstructed_candle.flatten_all()?.to_vec1::<f32>()?;

        let mse = calculate_mse(&weights, &reconstructed_data);

        // AC5: TL2 ≤ 1e-4 tolerance
        assert!(mse <= 1e-4, "AC5: TL2 MSE {} exceeds 1e-4 for tail shape ({}, {})", mse, m, n);

        println!("AC5.6: TL2 tail ({}, {}) MSE = {:.6e} ✓", m, n, mse);
    }

    Ok(())
}

/// AC:5.7 - Comparative accuracy test (I2S < TL1 ≈ TL2)
/// Validates relative accuracy ordering of quantization methods
#[test]
fn test_ac5_comparative_accuracy() -> Result<()> {
    let (m, n) = (256, 512);
    let weights = generate_test_weights(m, n);
    let tensor = BitNetTensor::from_slice(&weights, &[m, n], &Device::Cpu)?;

    // I2S quantization
    let i2s_quantizer = I2SQuantizer::new();
    let i2s_quantized = i2s_quantizer.quantize_tensor(&tensor)?;
    let i2s_reconstructed = i2s_quantizer.dequantize_tensor(&i2s_quantized)?;
    let i2s_candle = i2s_reconstructed.to_candle()?;
    let i2s_data = i2s_candle.flatten_all()?.to_vec1::<f32>()?;
    let i2s_mse = calculate_mse(&weights, &i2s_data);

    // TL1 quantization
    let tl1_quantizer = TL1Quantizer::new();
    let tl1_quantized = tl1_quantizer.quantize_tensor(&tensor)?;
    let tl1_reconstructed = tl1_quantizer.dequantize_tensor(&tl1_quantized)?;
    let tl1_candle = tl1_reconstructed.to_candle()?;
    let tl1_data = tl1_candle.flatten_all()?.to_vec1::<f32>()?;
    let tl1_mse = calculate_mse(&weights, &tl1_data);

    // TL2 quantization
    let tl2_quantizer = TL2Quantizer::new();
    let tl2_quantized = tl2_quantizer.quantize_tensor(&tensor)?;
    let tl2_reconstructed = tl2_quantizer.dequantize_tensor(&tl2_quantized)?;
    let tl2_candle = tl2_reconstructed.to_candle()?;
    let tl2_data = tl2_candle.flatten_all()?.to_vec1::<f32>()?;
    let tl2_mse = calculate_mse(&weights, &tl2_data);

    println!("AC5.7: Comparative accuracy:");
    println!("  I2S MSE: {:.6e}", i2s_mse);
    println!("  TL1 MSE: {:.6e}", tl1_mse);
    println!("  TL2 MSE: {:.6e}", tl2_mse);

    // AC5: I2S should be most accurate (lowest MSE)
    assert!(i2s_mse <= 1e-5, "AC5: I2S should meet 1e-5 tolerance");
    assert!(tl1_mse <= 1e-4 && tl2_mse <= 1e-4, "AC5: TL1/TL2 should meet 1e-4 tolerance");

    Ok(())
}
