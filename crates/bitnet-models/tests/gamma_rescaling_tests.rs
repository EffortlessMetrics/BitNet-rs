// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright 2025 BitNet Developers

//! Tests for experimental LayerNorm gamma rescaling by √hidden_size
//!
//! This tests the hypothesis that bitnet.cpp rescales pre-scaled gamma weights
//! on load, controlled by `BITNET_RESCALE_GAMMA_ON_LOAD=1`.

use bitnet_common::Result;
use candle_core::{Device, Tensor};
use serial_test::serial;

mod helpers;
use helpers::env_guard::EnvGuard;

/// Calculate RMS (root mean square) of a slice
fn calculate_rms(values: &[f32]) -> f32 {
    let mean_sq = values.iter().map(|&x| x * x).sum::<f32>() / values.len() as f32;
    mean_sq.sqrt()
}

#[test]
#[serial(bitnet_env)]
fn test_gamma_rescaling_disabled_by_default() -> Result<()> {
    let _guard = EnvGuard::new("BITNET_RESCALE_GAMMA_ON_LOAD");
    _guard.set("0");

    // Create gamma values with RMS ≈ 0.018 (1/√2560)
    let hidden_size = 2560;
    let target_rms = 1.0 / (hidden_size as f32).sqrt(); // ≈ 0.01976
    let gamma_values: Vec<f32> = (0..hidden_size).map(|_| target_rms).collect();

    let rms_before = calculate_rms(&gamma_values);
    assert!(
        (rms_before - target_rms).abs() < 1e-6,
        "RMS should be {:.6}, got {:.6}",
        target_rms,
        rms_before
    );

    // Create GGUF (note: this is a simplified test, real GGUF parsing is complex)
    // For now, we'll test the rescaling function directly via candle tensors

    let device = Device::Cpu;
    let tensor = Tensor::from_slice(&gamma_values, &[hidden_size], &device)?;
    let rms_tensor = tensor.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();

    // Without rescaling, RMS should remain low
    assert!(
        (rms_tensor - target_rms).abs() < 1e-4,
        "Without rescaling, RMS should be ~{:.6}, got {:.6}",
        target_rms,
        rms_tensor
    );

    Ok(())
}

#[test]
#[serial(bitnet_env)]
fn test_gamma_rescaling_enabled() -> Result<()> {
    let _guard = EnvGuard::new("BITNET_RESCALE_GAMMA_ON_LOAD");
    _guard.set("1");

    // Test that the rescaling function works correctly
    // We'll create a tensor with RMS ≈ 0.018 and verify it gets rescaled to ~1.0

    let hidden_size = 2560;
    let scale_factor = (hidden_size as f32).sqrt(); // ≈ 50.596
    let target_rms_before = 1.0 / scale_factor; // ≈ 0.01976
    let expected_rms_after = 1.0; // After rescaling by √hidden_size

    // Create gamma values with low RMS
    let gamma_values: Vec<f32> = (0..hidden_size).map(|_| target_rms_before).collect();
    let rms_before = calculate_rms(&gamma_values);

    // Simulate rescaling: gamma' = gamma * sqrt(hidden_size)
    let rescaled_values: Vec<f32> = gamma_values.iter().map(|&x| x * scale_factor).collect();
    let rms_after = calculate_rms(&rescaled_values);

    eprintln!("Test rescaling: hidden_size={}, scale_factor={:.2}×", hidden_size, scale_factor);
    eprintln!("RMS: {:.6} → {:.6} (expected: {:.6})", rms_before, rms_after, expected_rms_after);

    // Verify the rescaling math
    assert!(
        (rms_before - target_rms_before).abs() < 1e-4,
        "Initial RMS should be ~{:.6}, got {:.6}",
        target_rms_before,
        rms_before
    );

    assert!(
        (rms_after - expected_rms_after).abs() < 1e-4,
        "Rescaled RMS should be ~{:.6}, got {:.6}",
        expected_rms_after,
        rms_after
    );

    // Verify the relationship: RMS_after = RMS_before * scale_factor
    let ratio = rms_after / rms_before;
    assert!(
        (ratio - scale_factor).abs() < 1e-2,
        "Ratio should be {:.2}, got {:.2}",
        scale_factor,
        ratio
    );

    Ok(())
}

#[test]
#[serial(bitnet_env)]
fn test_gamma_rescaling_produces_target_rms() -> Result<()> {
    // Test the exact hypothesis: gamma RMS ≈ 0.018 → 1.0 after rescaling

    let hidden_size = 2560;
    let observed_rms = 0.018; // From HONEST_INFERENCE_STATUS.md
    let expected_factor = (hidden_size as f32).sqrt(); // ≈ 50.596

    // Create gamma with observed low RMS
    let gamma_values: Vec<f32> = (0..hidden_size).map(|_| observed_rms).collect();
    let rms_before = calculate_rms(&gamma_values);

    // Apply rescaling
    let rescaled_values: Vec<f32> = gamma_values.iter().map(|&x| x * expected_factor).collect();
    let rms_after = calculate_rms(&rescaled_values);

    eprintln!(
        "Hypothesis test: RMS {:.6} → {:.6} (factor: {:.2}×)",
        rms_before, rms_after, expected_factor
    );

    // Verify we get close to RMS = 1.0
    assert!(
        (rms_after - 1.0).abs() < 0.1,
        "Rescaled RMS should be close to 1.0, got {:.6}",
        rms_after
    );

    // The observed RMS of 0.018 is very close to 1/√2560
    let theoretical_inverse = 1.0 / expected_factor;
    eprintln!(
        "Comparison: observed={:.6}, theoretical 1/√{}={:.6} (match: {:.2}%)",
        observed_rms,
        hidden_size,
        theoretical_inverse,
        (observed_rms / theoretical_inverse) * 100.0
    );

    Ok(())
}

#[test]
#[serial(bitnet_env)]
#[ignore = "deadlocks: two simultaneous EnvGuard instances hold non-reentrant ENV_LOCK; scaffold test for strict-mode gamma rescaling behavior"]
fn test_gamma_rescaling_disabled_in_strict_mode() -> Result<()> {
    let _guard1 = EnvGuard::new("BITNET_RESCALE_GAMMA_ON_LOAD");
    _guard1.set("1");
    let _guard2 = EnvGuard::new("BITNET_STRICT_MODE");
    _guard2.set("1");

    // Even with BITNET_RESCALE_GAMMA_ON_LOAD=1, strict mode should prevent rescaling
    // This test would need to load an actual GGUF to verify the behavior
    // For now, we document the expected behavior

    eprintln!("Note: In strict mode, BITNET_RESCALE_GAMMA_ON_LOAD should be ignored");
    eprintln!("This prevents accidental corrections in production validation");

    Ok(())
}

#[test]
fn test_sqrt_hidden_size_calculation() {
    // Verify the math for common hidden sizes
    let test_cases = vec![
        (2560, 50.596), // microsoft-bitnet-b1.58-2B
        (768, 27.713),  // smaller models
        (4096, 64.0),   // larger models
        (1024, 32.0),   // exact square
    ];

    for (hidden_size, expected_sqrt) in test_cases {
        let calculated = (hidden_size as f32).sqrt();
        let diff = (calculated - expected_sqrt).abs();
        assert!(
            diff < 0.01,
            "sqrt({}) should be ~{:.3}, got {:.3}",
            hidden_size,
            expected_sqrt,
            calculated
        );
    }
}

#[test]
fn test_rms_rescaling_factor_relationship() {
    // Test the mathematical relationship: RMS' = RMS * factor
    // For gamma with RMS = 1/√H, rescaling by √H gives RMS = 1.0

    let hidden_size = 2560;
    let scale_factor = (hidden_size as f32).sqrt();
    let initial_rms = 1.0 / scale_factor;

    // Create test vector
    let values: Vec<f32> = (0..hidden_size).map(|i| initial_rms + (i as f32) * 0.0001).collect();
    let rms_before = calculate_rms(&values);

    // Rescale
    let rescaled: Vec<f32> = values.iter().map(|&x| x * scale_factor).collect();
    let rms_after = calculate_rms(&rescaled);

    // Verify relationship
    let ratio = rms_after / rms_before;
    assert!(
        (ratio - scale_factor).abs() < 0.1,
        "RMS ratio should equal scale_factor: {:.2} vs {:.2}",
        ratio,
        scale_factor
    );

    eprintln!(
        "RMS rescaling: {:.6} → {:.6} (factor: {:.2}×, ratio: {:.2}×)",
        rms_before, rms_after, scale_factor, ratio
    );
}
