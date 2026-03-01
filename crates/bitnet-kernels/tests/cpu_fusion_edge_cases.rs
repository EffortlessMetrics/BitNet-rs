//! Edge-case tests for CPU kernel fusion operations.
//!
//! Tests cover fused RMSNorm+linear, GELU+linear, softmax+mask,
//! add+normalize, and scale+add operations.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::fusion::{
    FusionConfig, fused_add_normalize, fused_gelu_linear, fused_rmsnorm_linear, fused_scale_add,
    fused_softmax_mask,
};

// ── FusionConfig ─────────────────────────────────────────────────────

#[test]
fn fusion_config_disabled() {
    let config = FusionConfig::disabled();
    assert!(config.validate().is_ok());
}

// ── fused_rmsnorm_linear ─────────────────────────────────────────────

#[test]
fn fused_rmsnorm_linear_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight =
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]; // 4x4 identity
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let result = fused_rmsnorm_linear(&input, &weight, &gamma, 1e-5).unwrap();
    assert_eq!(result.len(), 4);
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
fn fused_rmsnorm_linear_uniform() {
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let weight =
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let result = fused_rmsnorm_linear(&input, &weight, &gamma, 1e-5).unwrap();
    // Uniform input → RMS = 1.0 → normalized = [1,1,1,1] → identity linear = [1,1,1,1]
    for val in &result {
        assert!((val - 1.0).abs() < 0.01, "Expected ~1.0, got {val}");
    }
}

// ── fused_gelu_linear ────────────────────────────────────────────────

#[test]
fn fused_gelu_linear_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight =
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let bias = vec![0.0, 0.0, 0.0, 0.0];
    let result = fused_gelu_linear(&input, &weight, &bias).unwrap();
    assert_eq!(result.len(), 4);
    // GELU of positive values ≈ x for large x
    assert!(result[3] > 3.0, "GELU(4) should be close to 4");
}

#[test]
fn fused_gelu_linear_zeros() {
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let weight =
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let bias = vec![0.0, 0.0, 0.0, 0.0];
    let result = fused_gelu_linear(&input, &weight, &bias).unwrap();
    for val in &result {
        assert!(val.abs() < 0.01, "GELU(0) should be ~0, got {val}");
    }
}

// ── fused_softmax_mask ───────────────────────────────────────────────

#[test]
fn fused_softmax_mask_basic() {
    let scores = vec![1.0, 2.0, 3.0, 4.0];
    let mask = vec![1.0, 1.0, 1.0, 1.0]; // no masking
    let result = fused_softmax_mask(&scores, &mask, 1.0).unwrap();
    assert_eq!(result.len(), 4);
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "Softmax should sum to 1.0: {sum}");
}

#[test]
fn fused_softmax_mask_with_zeros() {
    let scores = vec![1.0, 2.0, 3.0, 4.0];
    let mask = vec![1.0, 1.0, 0.0, 0.0]; // partial mask
    let result = fused_softmax_mask(&scores, &mask, 1.0).unwrap();
    assert_eq!(result.len(), 4);
    for val in &result {
        assert!(*val >= 0.0 && val.is_finite(), "Invalid probability: {val}");
    }
}

#[test]
fn fused_softmax_mask_scale() {
    let scores = vec![1.0, 1.0, 1.0, 1.0]; // uniform
    let mask = vec![1.0, 1.0, 1.0, 1.0];
    let result = fused_softmax_mask(&scores, &mask, 0.5).unwrap();
    // Uniform scores → uniform probabilities regardless of scale
    for val in &result {
        assert!((val - 0.25).abs() < 0.01);
    }
}

// ── fused_add_normalize ──────────────────────────────────────────────

#[test]
fn fused_add_normalize_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let result = fused_add_normalize(&a, &b, &gamma, 1e-5).unwrap();
    assert_eq!(result.len(), 4);
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
fn fused_add_normalize_zeros() {
    let a = vec![0.0, 0.0, 0.0, 0.0];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let result = fused_add_normalize(&a, &b, &gamma, 1e-5).unwrap();
    // [1,1,1,1] → RMS=1 → normalized=[1,1,1,1]
    for val in &result {
        assert!((val - 1.0).abs() < 0.01, "Expected ~1.0, got {val}");
    }
}

// ── fused_scale_add ──────────────────────────────────────────────────

#[test]
fn fused_scale_add_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![10.0, 20.0, 30.0];
    let result = fused_scale_add(&a, &b, 0.1).unwrap();
    // a + b * scale = [1+1, 2+2, 3+3] = [2, 4, 6]
    assert_eq!(result.len(), 3);
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
fn fused_scale_add_zero_scale() {
    let a = vec![5.0, 10.0];
    let b = vec![100.0, 200.0];
    let result = fused_scale_add(&a, &b, 0.0).unwrap();
    // a + b * 0 = a
    assert!((result[0] - 5.0).abs() < 1e-6);
    assert!((result[1] - 10.0).abs() < 1e-6);
}
