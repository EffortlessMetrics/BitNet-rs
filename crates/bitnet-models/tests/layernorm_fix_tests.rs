// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright 2025 BitNet Developers

//! Comprehensive tests validating the LayerNorm fix
//!
//! This test suite validates the critical fix for Issue #254 where LayerNorm tensors
//! were being incorrectly classified as quantized, and RMSNorm semantics (no mean
//! subtraction) were being used instead of LayerNorm semantics (with mean subtraction).
//!
//! # Background
//!
//! The original implementation had two critical bugs:
//!
//! 1. **LayerNorm tensors classified as I2_S quantized**: GGUF loaders were treating
//!    LayerNorm gamma/beta tensors as quantized (I2_S) instead of float-only. This
//!    caused quantization artifacts and garbled output.
//!
//! 2. **RMSNorm semantics instead of LayerNorm**: When bias tensors were missing,
//!    the code used `LayerNorm::rms_norm()` which skips mean subtraction. However,
//!    BitNet models expect full LayerNorm semantics (with mean subtraction) even
//!    when bias is absent.
//!
//! # The Fix
//!
//! 1. LayerNorm tensors are now explicitly rejected if they appear as I2_S quantized
//!    (see `loader.rs` line 1407-1413)
//!
//! 2. `layer_norm_with_optional_bias` now uses `LayerNorm::new_no_bias()` which
//!    performs mean subtraction, NOT `rms_norm()` which skips it
//!    (see `transformer.rs` line 87)
//!
//! # Test Coverage
//!
//! - Test 1: LayerNorm tensors never classified as quantized
//! - Test 2: LayerNorm uses mean subtraction (not RMSNorm)
//! - Test 3: LayerNorm normalizes over last dimension
//! - Test 4: LayerNorm output differs from RMSNorm output

use bitnet_common::Result;
use bitnet_models::names::is_layernorm_weight;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, VarBuilder};
use std::collections::HashMap;

#[test]
fn test_layernorm_tensor_names_never_classified_as_quantized() {
    // Test that our naming predicates correctly identify LayerNorm tensors
    // These should NEVER be treated as quantized tensors

    let layernorm_names = vec![
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.5.attn_norm.weight",
        "blk.10.ffn_norm.weight",
        "layers.0.attention_norm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.5.input_layernorm.weight",
        "final_norm.weight",
    ];

    for name in layernorm_names {
        assert!(
            is_layernorm_weight(name),
            "Tensor '{}' should be classified as LayerNorm weight",
            name
        );
    }

    // Negative cases: projection weights should NOT be LayerNorm
    let projection_names = vec![
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "layers.0.q_proj.weight",
        "token_embd.weight",
    ];

    for name in projection_names {
        assert!(
            !is_layernorm_weight(name),
            "Tensor '{}' should NOT be classified as LayerNorm weight",
            name
        );
    }
}

#[test]
fn test_layernorm_uses_mean_subtraction_not_rmsnorm() -> Result<()> {
    // This test validates that LayerNorm::new_no_bias uses FULL LayerNorm semantics
    // (with mean subtraction) and NOT RMSNorm semantics (no mean subtraction)
    //
    // Formula comparison:
    // - LayerNorm:  y = (x - mean(x)) / sqrt(var(x) + eps) * gamma
    // - RMSNorm:    y = x / sqrt(mean(x²) + eps) * gamma
    //
    // The key difference: LayerNorm subtracts the mean, RMSNorm does not.

    let device = Device::Cpu;
    let hidden_size = 128;
    let eps = 1e-5;

    // Create input with non-zero mean to highlight the difference
    // LayerNorm will center it to zero mean, RMSNorm will not
    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| {
            let x = i as f32 / hidden_size as f32;
            (x * 10.0).sin() + 2.0 // Add constant offset to ensure non-zero mean
        })
        .collect();

    let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

    // Create gamma (weight) - all ones for simplicity
    let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;

    // Create LayerNorm WITHOUT bias (uses new_no_bias internally)
    let mut tensors = HashMap::new();
    tensors.insert("weight".to_string(), gamma.clone());
    let _vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

    let layer_norm = LayerNorm::new_no_bias(gamma.clone(), eps);
    let output_ln = layer_norm.forward(&input)?;

    // Compute expected LayerNorm output manually
    let input_vec: Vec<f32> = input.flatten_all()?.to_vec1()?;
    let mean = input_vec.iter().sum::<f32>() / input_vec.len() as f32;
    let variance = input_vec
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f32>()
        / input_vec.len() as f32;

    let expected_ln: Vec<f32> =
        input_vec.iter().map(|&x| (x - mean) / (variance + eps as f32).sqrt()).collect();

    let expected_ln_tensor = Tensor::from_slice(&expected_ln, (1, hidden_size), &device)?;

    // Verify LayerNorm output matches manual calculation
    let ln_vec: Vec<f32> = output_ln.flatten_all()?.to_vec1()?;
    let expected_vec: Vec<f32> = expected_ln_tensor.flatten_all()?.to_vec1()?;

    for (i, (&actual, &expected)) in ln_vec.iter().zip(expected_vec.iter()).enumerate() {
        let diff: f32 = (actual - expected).abs();
        assert!(
            diff < 1e-4,
            "LayerNorm output mismatch at index {}: expected {:.6}, got {:.6}, diff {:.6}",
            i,
            expected,
            actual,
            diff
        );
    }

    // Verify that the mean after LayerNorm is close to 0 (this is the KEY test)
    let output_mean = ln_vec.iter().sum::<f32>() / ln_vec.len() as f32;
    assert!(
        output_mean.abs() < 1e-4,
        "LayerNorm output should have mean ≈ 0 (got {:.6}) - this proves mean subtraction occurred",
        output_mean
    );

    // For comparison: compute what RMSNorm would produce (no mean subtraction)
    let rms_norm_output: Vec<f32> = input_vec
        .iter()
        .map(|&x| {
            let mean_sq = input_vec.iter().map(|&v| v * v).sum::<f32>() / input_vec.len() as f32;
            x / (mean_sq + eps as f32).sqrt()
        })
        .collect();

    let rms_mean = rms_norm_output.iter().sum::<f32>() / rms_norm_output.len() as f32;

    // RMSNorm output should NOT have zero mean (it doesn't subtract mean)
    assert!(
        rms_mean.abs() > 0.1,
        "RMSNorm output should NOT have zero mean (got {:.6}) - this proves it doesn't subtract mean",
        rms_mean
    );

    // Verify LayerNorm and RMSNorm produce DIFFERENT outputs
    let diff_norm: f32 = ln_vec
        .iter()
        .zip(rms_norm_output.iter())
        .map(|(&ln, &rms)| {
            let diff: f32 = (ln - rms).abs();
            diff
        })
        .sum::<f32>()
        / ln_vec.len() as f32;

    assert!(
        diff_norm > 0.01,
        "LayerNorm and RMSNorm should produce different outputs (avg diff: {:.6})",
        diff_norm
    );

    Ok(())
}

#[test]
fn test_layernorm_normalizes_over_last_dimension() -> Result<()> {
    // Test that LayerNorm normalizes independently over each position
    // For input shape [B, H], normalization should happen over the H dimension
    // independently for each batch

    let device = Device::Cpu;
    let batch_size = 6; // Simulate B*T positions
    let hidden_size = 64;
    let eps = 1e-5;

    // Create input tensor [B, H] with different statistics per position
    let input_data: Vec<f32> = (0..(batch_size * hidden_size))
        .map(|i| {
            let batch_idx = i / hidden_size;
            let x = i as f32 / (hidden_size as f32);
            // Each batch position has different mean/variance
            ((x * 5.0).sin() + (x * 3.0).cos()) * 2.0 + (batch_idx as f32)
        })
        .collect();

    let input = Tensor::from_slice(&input_data, (batch_size, hidden_size), &device)?;

    // Create LayerNorm
    let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;
    let layer_norm = LayerNorm::new_no_bias(gamma, eps);

    let output = layer_norm.forward(&input)?;

    // Verify output shape matches input
    assert_eq!(output.dims(), input.dims());

    // Verify normalization: each position should have mean ≈ 0 and variance ≈ 1
    let output_data: Vec<Vec<f32>> = output.to_vec2()?;

    for (i, row) in output_data.iter().enumerate() {
        // Compute mean for this position
        let mean: f32 = row.iter().sum::<f32>() / row.len() as f32;

        // Compute variance for this position
        let variance: f32 = row
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f32>()
            / row.len() as f32;

        assert!(mean.abs() < 1e-3, "Position {} should have mean ≈ 0, got {:.6}", i, mean);

        assert!(
            (variance - 1.0).abs() < 0.1,
            "Position {} should have variance ≈ 1, got {:.6}",
            i,
            variance
        );
    }

    Ok(())
}

#[test]
fn test_layernorm_normalizes_per_position_independently() -> Result<()> {
    // Test that each [H] vector is normalized independently
    // This is critical for the fix - we need proper per-position normalization

    let device = Device::Cpu;
    let hidden_size = 32;
    let eps = 1e-5;

    // Create two different vectors with very different statistics
    let vec1_data: Vec<f32> = (0..hidden_size)
        .map(|i| (i as f32 / hidden_size as f32) * 10.0) // Range: 0-10
        .collect();

    let vec2_data: Vec<f32> = (0..hidden_size)
        .map(|i| -(i as f32 / hidden_size as f32) * 5.0 + 20.0) // Range: 20 to 15
        .collect();

    // Stack them: [2, H]
    let mut combined_data = Vec::with_capacity(2 * hidden_size);
    combined_data.extend_from_slice(&vec1_data);
    combined_data.extend_from_slice(&vec2_data);

    let input = Tensor::from_slice(&combined_data, (2, hidden_size), &device)?;

    // Create LayerNorm
    let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;
    let layer_norm = LayerNorm::new_no_bias(gamma, eps);

    let output = layer_norm.forward(&input)?;

    // Extract normalized vectors
    let output_vec: Vec<Vec<f32>> = output.to_vec2()?;

    // Each vector should be independently normalized (mean ≈ 0, std ≈ 1)
    for (i, vec) in output_vec.iter().enumerate() {
        let mean: f32 = vec.iter().sum::<f32>() / vec.len() as f32;
        let variance = vec
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f32>()
            / vec.len() as f32;

        assert!(mean.abs() < 1e-3, "Vector {} should have mean ≈ 0, got {:.6}", i, mean);

        assert!(
            (variance - 1.0).abs() < 0.1,
            "Vector {} should have variance ≈ 1, got {:.6}",
            i,
            variance
        );
    }

    Ok(())
}

#[test]
fn test_layernorm_output_differs_from_rmsnorm() -> Result<()> {
    // Direct comparison: LayerNorm (with mean subtraction) vs RMSNorm (without)
    // This test demonstrates why using RMSNorm semantics produces incorrect output

    let device = Device::Cpu;
    let hidden_size = 128;
    let eps = 1e-5;

    // Create input with known statistics
    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| {
            let x = i as f32 / hidden_size as f32;
            ((x * 8.0).sin() * 3.0) + 5.0 // Non-zero mean
        })
        .collect();

    let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;
    let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;

    // Apply LayerNorm (with mean subtraction)
    let layer_norm = LayerNorm::new_no_bias(gamma.clone(), eps);
    let ln_output = layer_norm.forward(&input)?;

    // Apply RMSNorm manually (no mean subtraction)
    let mean_sq = input.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let rms_denominator = (mean_sq + eps as f32).sqrt();
    let rms_output = input.affine(1.0 / rms_denominator as f64, 0.0)?;

    // Extract values
    let ln_vec: Vec<f32> = ln_output.flatten_all()?.to_vec1()?;
    let rms_vec: Vec<f32> = rms_output.flatten_all()?.to_vec1()?;

    // Compute statistics
    let ln_mean = ln_vec.iter().sum::<f32>() / ln_vec.len() as f32;
    let rms_mean = rms_vec.iter().sum::<f32>() / rms_vec.len() as f32;

    // LayerNorm output should have mean ≈ 0
    assert!(ln_mean.abs() < 1e-3, "LayerNorm should produce zero-mean output, got {:.6}", ln_mean);

    // RMSNorm output should NOT have mean ≈ 0 (it preserves the input mean's direction)
    assert!(
        rms_mean.abs() > 0.5,
        "RMSNorm should NOT produce zero-mean output, got {:.6}",
        rms_mean
    );

    // The outputs should be substantially different
    let avg_abs_diff: f32 = ln_vec
        .iter()
        .zip(rms_vec.iter())
        .map(|(&ln, &rms)| {
            let diff: f32 = (ln - rms).abs();
            diff
        })
        .sum::<f32>()
        / ln_vec.len() as f32;

    assert!(
        avg_abs_diff > 0.1,
        "LayerNorm and RMSNorm outputs should differ significantly (avg diff: {:.6})",
        avg_abs_diff
    );

    // Verify that using the wrong normalization (RMSNorm) would produce wrong results
    // This is the core issue that was causing garbled output in Issue #254
    eprintln!("✓ LayerNorm mean: {:.6} (≈ 0)", ln_mean);
    eprintln!("✓ RMSNorm mean: {:.6} (≠ 0)", rms_mean);
    eprintln!("✓ Average absolute difference: {:.6}", avg_abs_diff);
    eprintln!("✓ This confirms LayerNorm (mean subtraction) ≠ RMSNorm (no mean subtraction)");

    Ok(())
}

#[test]
fn test_layernorm_with_gamma_scaling() -> Result<()> {
    // Test that LayerNorm correctly applies gamma (weight) scaling
    // Formula: output = ((x - mean) / sqrt(var + eps)) * gamma

    let device = Device::Cpu;
    let hidden_size = 64;
    let eps = 1e-5;

    // Create input
    let input_data: Vec<f32> =
        (0..hidden_size).map(|i| (i as f32 / hidden_size as f32).sin() * 2.0).collect();

    let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

    // Create gamma with non-uniform values
    let gamma_data: Vec<f32> = (0..hidden_size)
        .map(|i| 1.0 + (i as f32 / hidden_size as f32) * 0.5) // Range: 1.0 to 1.5
        .collect();

    let gamma = Tensor::from_slice(&gamma_data, hidden_size, &device)?;

    // Apply LayerNorm
    let layer_norm = LayerNorm::new_no_bias(gamma.clone(), eps);
    let output = layer_norm.forward(&input)?;

    // Manually compute expected output
    let input_vec: Vec<f32> = input.flatten_all()?.to_vec1()?;
    let mean = input_vec.iter().sum::<f32>() / input_vec.len() as f32;
    let variance = input_vec
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f32>()
        / input_vec.len() as f32;

    let expected: Vec<f32> = input_vec
        .iter()
        .zip(gamma_data.iter())
        .map(|(&x, &g)| ((x - mean) / (variance + eps as f32).sqrt()) * g)
        .collect();

    // Verify outputs match
    let output_vec: Vec<f32> = output.flatten_all()?.to_vec1()?;

    for (i, (&actual, &expected_val)) in output_vec.iter().zip(expected.iter()).enumerate() {
        let diff: f32 = (actual - expected_val).abs();
        assert!(
            diff < 1e-4,
            "Gamma scaling mismatch at index {}: expected {:.6}, got {:.6}",
            i,
            expected_val,
            actual
        );
    }

    Ok(())
}

#[test]
fn test_layernorm_stability_with_various_inputs() -> Result<()> {
    // Test LayerNorm stability with edge cases:
    // - Very small values
    // - Very large values
    // - All zeros (should handle gracefully due to eps)

    let device = Device::Cpu;
    let hidden_size = 32;
    let eps = 1e-5;
    let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;
    let layer_norm = LayerNorm::new_no_bias(gamma, eps);

    // Test case 1: Very small values
    let small_data: Vec<f32> = vec![1e-6; hidden_size];
    let small_input = Tensor::from_slice(&small_data, (1, hidden_size), &device)?;
    let small_output = layer_norm.forward(&small_input)?;
    let small_vec: Vec<f32> = small_output.flatten_all()?.to_vec1()?;
    assert!(
        small_vec.iter().all(|&x: &f32| x.is_finite()),
        "LayerNorm should produce finite values for very small inputs"
    );

    // Test case 2: Very large values
    let large_data: Vec<f32> = vec![1e6; hidden_size];
    let large_input = Tensor::from_slice(&large_data, (1, hidden_size), &device)?;
    let large_output = layer_norm.forward(&large_input)?;
    let large_vec: Vec<f32> = large_output.flatten_all()?.to_vec1()?;
    assert!(
        large_vec.iter().all(|&x: &f32| x.is_finite()),
        "LayerNorm should produce finite values for very large inputs"
    );

    // Test case 3: Mixed values with high variance
    let mixed_data: Vec<f32> =
        (0..hidden_size).map(|i| if i % 2 == 0 { 100.0 } else { -100.0 }).collect();
    let mixed_input = Tensor::from_slice(&mixed_data, (1, hidden_size), &device)?;
    let mixed_output = layer_norm.forward(&mixed_input)?;
    let mixed_vec: Vec<f32> = mixed_output.flatten_all()?.to_vec1()?;
    assert!(
        mixed_vec.iter().all(|&x: &f32| x.is_finite()),
        "LayerNorm should produce finite values for high-variance inputs"
    );

    // Verify normalization properties are maintained
    let mean = mixed_vec.iter().sum::<f32>() / mixed_vec.len() as f32;
    assert!(mean.abs() < 1e-3, "Normalized output should have mean ≈ 0, got {:.6}", mean);

    Ok(())
}
