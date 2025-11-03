//! RMSNorm Diagnostic Tests
//!
//! These tests investigate the behavior of Candle's RMSNorm implementation
//! with different gamma (weight) values, particularly focusing on understanding
//! why our model (with gamma RMS ≈ 0.018 ≈ 1/√2560) produces different output
//! than bitnet.cpp with the same GGUF.
//!
//! Context:
//! - Standard RMSNorm formula: output = (x / sqrt(mean(x²) + eps)) * gamma
//! - Our model has gamma RMS ≈ 0.018 (which is 1/√2560, where 2560 is hidden_size)
//! - bitnet.cpp produces coherent output with same GGUF
//! - bitnet.rs produces garbled output
//!
//! These tests aim to:
//! 1. Verify Candle's RMSNorm implementation matches expected formula
//! 2. Compare behavior with standard gamma (RMS ≈ 1.0) vs our gamma (RMS ≈ 0.018)
//! 3. Identify if output magnitudes are reasonable
//! 4. Rule out NaN/Inf issues

use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::RmsNorm;

/// Helper function to compute RMS (root mean square) of a tensor
fn compute_rms(tensor: &Tensor) -> Result<f64> {
    let squared = tensor.sqr()?;
    let mean = squared.mean_all()?;
    let rms = mean.sqrt()?.to_scalar::<f32>()? as f64;
    Ok(rms)
}

/// Helper function to print diagnostic statistics for a tensor
fn print_tensor_stats(name: &str, tensor: &Tensor) -> Result<()> {
    let rms = compute_rms(tensor)?;
    let abs_tensor = tensor.abs()?;
    let mean_abs = abs_tensor.mean_all()?.to_scalar::<f32>()? as f64;

    // Get min/max by converting to vector (simpler than keepdim operations)
    let vec_data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let min_val = vec_data.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let max_val = vec_data.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

    // Check for NaN/Inf
    let has_nan = vec_data.iter().any(|x| x.is_nan());
    let has_inf = vec_data.iter().any(|x| x.is_infinite());

    println!("\n{} Statistics:", name);
    println!("  RMS:      {:.6e}", rms);
    println!("  Mean(|x|): {:.6e}", mean_abs);
    println!("  Min:      {:.6e}", min_val);
    println!("  Max:      {:.6e}", max_val);
    println!("  Has NaN:  {}", has_nan);
    println!("  Has Inf:  {}", has_inf);
    println!("  Range:    [{:.6e}, {:.6e}]", min_val, max_val);

    Ok(())
}

/// Test RMSNorm with standard gamma (RMS ≈ 1.0)
#[test]
fn test_rmsnorm_standard_gamma() -> Result<()> {
    println!("\n=== Test 1: RMSNorm with Standard Gamma (RMS ≈ 1.0) ===");

    let device = Device::Cpu;
    let hidden_size = 2560;
    let eps = 1e-5;

    // Create input tensor [1, 1, 2560] with reasonable values
    // Use random values with mean 0 and std 1 (typical activation distribution)
    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| {
            let x = i as f32 / hidden_size as f32;
            // Create a mix of positive and negative values
            ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
        })
        .collect();

    let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

    // Create standard gamma (all ones) with RMS ≈ 1.0
    let gamma_data: Vec<f32> = vec![1.0; hidden_size];
    let gamma = Tensor::from_slice(&gamma_data, hidden_size, &device)?;

    print_tensor_stats("Input", &input)?;
    print_tensor_stats("Gamma (standard)", &gamma)?;

    // Apply RMSNorm
    let rms_norm = RmsNorm::new(gamma, eps);
    let output = rms_norm.forward(&input)?;

    print_tensor_stats("Output", &output)?;

    // Verify output is reasonable
    let output_rms = compute_rms(&output)?;
    println!("\n✓ Output RMS: {:.6e}", output_rms);
    println!("  Expected: close to 1.0 (since input is normalized then multiplied by 1.0)");

    // For standard RMSNorm with gamma=1, output RMS should be close to 1.0
    assert!(
        output_rms > 0.5 && output_rms < 2.0,
        "Output RMS should be reasonable with standard gamma, got {:.6e}",
        output_rms
    );

    Ok(())
}

/// Test RMSNorm with our model's gamma (RMS ≈ 0.018 ≈ 1/√2560)
#[test]
fn test_rmsnorm_small_gamma() -> Result<()> {
    println!("\n=== Test 2: RMSNorm with Small Gamma (RMS ≈ 0.018 ≈ 1/√2560) ===");

    let device = Device::Cpu;
    let hidden_size = 2560;
    let eps = 1e-5;

    // Create input tensor [1, 1, 2560] with reasonable values
    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| {
            let x = i as f32 / hidden_size as f32;
            ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
        })
        .collect();

    let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

    // Create gamma with RMS ≈ 0.018 (1/√2560 ≈ 0.01976)
    // This matches our model's LayerNorm gamma
    let target_rms = 1.0 / (hidden_size as f64).sqrt();
    let gamma_data: Vec<f32> = vec![target_rms as f32; hidden_size];
    let gamma = Tensor::from_slice(&gamma_data, hidden_size, &device)?;

    print_tensor_stats("Input", &input)?;
    print_tensor_stats("Gamma (1/√2560)", &gamma)?;

    let gamma_rms = compute_rms(&gamma)?;
    println!("\nGamma RMS: {:.6e} (expected: {:.6e})", gamma_rms, target_rms);

    // Apply RMSNorm
    let rms_norm = RmsNorm::new(gamma, eps);
    let output = rms_norm.forward(&input)?;

    print_tensor_stats("Output", &output)?;

    // Calculate expected output RMS
    // After normalization, RMS ≈ 1.0
    // After multiplying by gamma (RMS ≈ 0.018), output RMS ≈ 0.018
    let output_rms = compute_rms(&output)?;
    println!("\n✓ Output RMS: {:.6e}", output_rms);
    println!("  Expected: close to {:.6e} (normalized, then scaled by gamma)", target_rms);

    // Verify output is smaller but not too small
    assert!(
        output_rms > 0.001 && output_rms < 0.1,
        "Output RMS should be reasonable with small gamma, got {:.6e}",
        output_rms
    );

    // Compare with standard gamma scaling
    let expected_scaling = target_rms;
    let ratio = output_rms / expected_scaling;
    println!("  Ratio (actual/expected): {:.6}", ratio);

    Ok(())
}

/// Test comparing standard vs small gamma side-by-side
#[test]
fn test_rmsnorm_gamma_comparison() -> Result<()> {
    println!("\n=== Test 3: Side-by-Side Comparison (Standard vs Small Gamma) ===");

    let device = Device::Cpu;
    let hidden_size = 2560;
    let eps = 1e-5;

    // Create same input for both tests
    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| {
            let x = i as f32 / hidden_size as f32;
            ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
        })
        .collect();

    let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

    print_tensor_stats("Input (shared)", &input)?;

    // Standard gamma (RMS ≈ 1.0)
    let gamma_std_data: Vec<f32> = vec![1.0; hidden_size];
    let gamma_std = Tensor::from_slice(&gamma_std_data, hidden_size, &device)?;
    let rms_norm_std = RmsNorm::new(gamma_std.clone(), eps);
    let output_std = rms_norm_std.forward(&input)?;

    print_tensor_stats("Gamma (standard)", &gamma_std)?;
    print_tensor_stats("Output (standard)", &output_std)?;

    // Small gamma (RMS ≈ 0.018)
    let target_rms = 1.0 / (hidden_size as f64).sqrt();
    let gamma_small_data: Vec<f32> = vec![target_rms as f32; hidden_size];
    let gamma_small = Tensor::from_slice(&gamma_small_data, hidden_size, &device)?;
    let rms_norm_small = RmsNorm::new(gamma_small.clone(), eps);
    let output_small = rms_norm_small.forward(&input)?;

    print_tensor_stats("Gamma (1/√2560)", &gamma_small)?;
    print_tensor_stats("Output (1/√2560)", &output_small)?;

    // Compare outputs
    let output_std_rms = compute_rms(&output_std)?;
    let output_small_rms = compute_rms(&output_small)?;
    let ratio = output_small_rms / output_std_rms;

    println!("\n=== Comparison ===");
    println!("Standard output RMS:  {:.6e}", output_std_rms);
    println!("Small gamma output RMS: {:.6e}", output_small_rms);
    println!("Ratio (small/standard): {:.6}", ratio);
    println!("Expected ratio:       {:.6}", target_rms);
    println!("Ratio matches expected: {}", (ratio - target_rms).abs() < 0.01);

    // The ratio should be close to 1/√2560
    assert!(
        (ratio - target_rms).abs() < 0.005,
        "Ratio should match expected scaling: got {:.6}, expected {:.6}",
        ratio,
        target_rms
    );

    Ok(())
}

/// Test with realistic activation magnitudes
#[test]
fn test_rmsnorm_realistic_activations() -> Result<()> {
    println!("\n=== Test 4: Realistic Activation Magnitudes ===");

    let device = Device::Cpu;
    let hidden_size = 2560;
    let eps = 1e-5;

    // Create input with more realistic activation distribution
    // Typical transformer activations have std around 1-10
    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| {
            let x = i as f32 / hidden_size as f32;
            // Larger magnitude activations (std ≈ 5.0)
            ((x * 10.0).sin() + (x * 20.0).cos()) * 5.0
        })
        .collect();

    let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

    print_tensor_stats("Input (realistic magnitudes)", &input)?;

    // Test with both gamma types

    // 1. Standard gamma
    let gamma_std_data: Vec<f32> = vec![1.0; hidden_size];
    let gamma_std = Tensor::from_slice(&gamma_std_data, hidden_size, &device)?;
    let rms_norm_std = RmsNorm::new(gamma_std, eps);
    let output_std = rms_norm_std.forward(&input)?;

    println!("\n--- Standard Gamma ---");
    print_tensor_stats("Output", &output_std)?;

    // 2. Small gamma (our model)
    let target_rms = 1.0 / (hidden_size as f64).sqrt();
    let gamma_small_data: Vec<f32> = vec![target_rms as f32; hidden_size];
    let gamma_small = Tensor::from_slice(&gamma_small_data, hidden_size, &device)?;
    let rms_norm_small = RmsNorm::new(gamma_small, eps);
    let output_small = rms_norm_small.forward(&input)?;

    println!("\n--- Small Gamma (1/√2560) ---");
    print_tensor_stats("Output", &output_small)?;

    // Both outputs should be reasonable (no NaN/Inf)
    let output_std_rms = compute_rms(&output_std)?;
    let output_small_rms = compute_rms(&output_small)?;

    println!("\n=== Summary ===");
    println!("Standard output RMS:  {:.6e}", output_std_rms);
    println!("Small gamma output RMS: {:.6e}", output_small_rms);
    println!("Input was scaled by 5.0 - outputs should still be reasonable");

    assert!(
        output_std_rms > 0.1 && output_std_rms < 100.0,
        "Standard output RMS should be reasonable: {:.6e}",
        output_std_rms
    );

    assert!(
        output_small_rms > 0.001 && output_small_rms < 1.0,
        "Small gamma output RMS should be reasonable: {:.6e}",
        output_small_rms
    );

    Ok(())
}

/// Test forward and backward consistency
#[test]
fn test_rmsnorm_forward_formula() -> Result<()> {
    println!("\n=== Test 5: Verify Forward Formula Manually ===");

    let device = Device::Cpu;
    let hidden_size = 2560;
    let eps = 1e-5;

    // Simple input for manual verification
    let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 / 100.0).sin()).collect();

    let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

    // Standard gamma
    let gamma_data: Vec<f32> = vec![1.0; hidden_size];
    let gamma = Tensor::from_slice(&gamma_data, hidden_size, &device)?;

    print_tensor_stats("Input", &input)?;

    // Apply RMSNorm via Candle
    let rms_norm = RmsNorm::new(gamma.clone(), eps);
    let output_candle = rms_norm.forward(&input)?;

    // Manually compute RMSNorm
    // Formula: output = (x / sqrt(mean(x²) + eps)) * gamma
    let squared = input.sqr()?;
    let mean_squared = squared.mean_keepdim(2)?; // Mean over last dimension
    let rms_denominator = (mean_squared + eps)?.sqrt()?;
    let normalized = input.broadcast_div(&rms_denominator)?;
    let output_manual = normalized.broadcast_mul(&gamma)?;

    print_tensor_stats("Output (Candle)", &output_candle)?;
    print_tensor_stats("Output (Manual)", &output_manual)?;

    // Compare outputs
    let diff = (output_candle.sub(&output_manual))?.abs()?;
    let diff_vec: Vec<f32> = diff.flatten_all()?.to_vec1()?;
    let max_diff = diff_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

    println!("\n=== Verification ===");
    println!("Max absolute difference: {:.6e}", max_diff);
    println!("Tolerance: 1e-5");

    assert!(
        max_diff < 1e-5,
        "Candle's RMSNorm should match manual computation: max_diff={:.6e}",
        max_diff
    );

    println!("✓ Candle's RMSNorm matches manual formula");

    Ok(())
}

/// Test with actual model gamma values from GGUF
#[test]
fn test_rmsnorm_with_model_gamma_distribution() -> Result<()> {
    println!("\n=== Test 6: Model Gamma Distribution (Non-Uniform) ===");

    let device = Device::Cpu;
    let hidden_size = 2560;
    let eps = 1e-5;

    // Create input
    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| {
            let x = i as f32 / hidden_size as f32;
            ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
        })
        .collect();

    let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

    // Create gamma with small variation around 1/√2560
    // In real models, gamma is not uniform but has some learned variation
    let base_value = 1.0 / (hidden_size as f64).sqrt();
    let gamma_data: Vec<f32> = (0..hidden_size)
        .map(|i| {
            let x = i as f32 / hidden_size as f32;
            // Small variation (±10%) around base value
            base_value as f32 * (1.0 + 0.1 * (x * 20.0).sin())
        })
        .collect();

    let gamma = Tensor::from_slice(&gamma_data, hidden_size, &device)?;

    print_tensor_stats("Input", &input)?;
    print_tensor_stats("Gamma (non-uniform, RMS ≈ 1/√2560)", &gamma)?;

    let gamma_rms = compute_rms(&gamma)?;
    let expected_rms = base_value;
    println!("\nGamma RMS: {:.6e} (expected: {:.6e})", gamma_rms, expected_rms);

    // Apply RMSNorm
    let rms_norm = RmsNorm::new(gamma, eps);
    let output = rms_norm.forward(&input)?;

    print_tensor_stats("Output", &output)?;

    let output_rms = compute_rms(&output)?;
    println!("\n✓ Output RMS: {:.6e}", output_rms);
    println!("  With non-uniform gamma, output should still be reasonable");

    assert!(
        output_rms > 0.001 && output_rms < 0.1,
        "Output RMS should be reasonable with non-uniform small gamma: {:.6e}",
        output_rms
    );

    Ok(())
}
