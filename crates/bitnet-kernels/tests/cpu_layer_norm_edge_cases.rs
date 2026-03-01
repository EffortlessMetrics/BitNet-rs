//! Edge-case tests for CPU layer normalization kernels.
//!
//! Tests cover LayerNorm, RMSNorm, GroupNorm, InstanceNorm with
//! boundary conditions, special inputs, and batch variants.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::layer_norm::{
    GroupNormConfig, LayerNormConfig, batch_layer_norm, batch_rms_norm, group_norm, instance_norm,
    layer_norm, rms_norm,
};

fn approx_eq(a: f32, b: f32, tol: f32) {
    assert!((a - b).abs() <= tol, "expected {a} ≈ {b} (tol={tol}, diff={})", (a - b).abs());
}

// ── LayerNormConfig ──────────────────────────────────────────────────

#[test]
fn layer_norm_config_basic() {
    let config = LayerNormConfig { normalized_shape: vec![4], eps: 1e-5, elementwise_affine: true };
    assert_eq!(config.normalized_shape, vec![4]);
    assert!((config.eps - 1e-5).abs() < 1e-10);
}

// ── layer_norm ───────────────────────────────────────────────────────

#[test]
fn layer_norm_uniform_input() {
    let config = LayerNormConfig { normalized_shape: vec![4], eps: 1e-5, elementwise_affine: true };
    let input = vec![1.0f32; 4];
    let gamma = vec![1.0f32; 4];
    let result = layer_norm(&input, &gamma, None, &config).unwrap();
    // Uniform input → zero variance → output depends on eps
    assert_eq!(result.len(), 4);
    for v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn layer_norm_with_known_values() {
    let config = LayerNormConfig { normalized_shape: vec![4], eps: 1e-5, elementwise_affine: true };
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let result = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
    assert_eq!(result.len(), 4);
    // Mean = 2.5, Var = 1.25, Std = ~1.118
    // Normalized: [-1.342, -0.447, 0.447, 1.342] approximately
    assert!(result[0] < 0.0);
    assert!(result[3] > 0.0);
    // Sum of normalized values should be ~0
    let sum: f32 = result.iter().sum();
    assert!(sum.abs() < 0.01);
}

#[test]
fn layer_norm_with_beta_offset() {
    let config = LayerNormConfig { normalized_shape: vec![2], eps: 1e-5, elementwise_affine: true };
    let input = vec![1.0, 3.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![10.0, 10.0];
    let result = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
    // Result should be shifted by +10
    for v in &result {
        assert!(*v > 5.0, "expected values > 5 with beta=10, got {v}");
    }
}

#[test]
fn layer_norm_gamma_scaling() {
    let config = LayerNormConfig { normalized_shape: vec![3], eps: 1e-5, elementwise_affine: true };
    let input = vec![1.0, 2.0, 3.0];
    let gamma_one = vec![1.0; 3];
    let gamma_two = vec![2.0; 3];
    let r1 = layer_norm(&input, &gamma_one, None, &config).unwrap();
    let r2 = layer_norm(&input, &gamma_two, None, &config).unwrap();
    // Doubling gamma should approximately double the output
    for (a, b) in r1.iter().zip(r2.iter()) {
        approx_eq(*b, *a * 2.0, 0.01);
    }
}

// ── rms_norm ─────────────────────────────────────────────────────────

#[test]
fn rms_norm_uniform_input() {
    let config = LayerNormConfig { normalized_shape: vec![4], eps: 1e-5, elementwise_affine: true };
    let input = vec![2.0f32; 4];
    let gamma = vec![1.0; 4];
    let result = rms_norm(&input, &gamma, &config).unwrap();
    assert_eq!(result.len(), 4);
    // RMS of [2,2,2,2] = 2, so normalized = 2/2 = 1.0
    for v in &result {
        approx_eq(*v, 1.0, 0.001);
    }
}

#[test]
fn rms_norm_known_values() {
    let config = LayerNormConfig { normalized_shape: vec![3], eps: 1e-5, elementwise_affine: true };
    let input = vec![1.0, 2.0, 3.0];
    let gamma = vec![1.0; 3];
    let result = rms_norm(&input, &gamma, &config).unwrap();
    // RMS = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
    let rms = (14.0f32 / 3.0).sqrt();
    for (x, y) in input.iter().zip(result.iter()) {
        approx_eq(*y, *x / rms, 0.01);
    }
}

#[test]
fn rms_norm_zero_input() {
    let config = LayerNormConfig { normalized_shape: vec![3], eps: 1e-5, elementwise_affine: true };
    let input = vec![0.0; 3];
    let gamma = vec![1.0; 3];
    let result = rms_norm(&input, &gamma, &config).unwrap();
    // RMS ≈ eps, result should be 0/(eps) ≈ 0
    for v in &result {
        assert!(v.is_finite());
    }
}

// ── GroupNormConfig ──────────────────────────────────────────────────

#[test]
fn group_norm_config_basic() {
    let config = GroupNormConfig {
        num_groups: 2,
        num_channels: 4,
        spatial_size: 1,
        eps: 1e-5,
        elementwise_affine: true,
    };
    assert_eq!(config.num_groups, 2);
    assert_eq!(config.num_channels, 4);
}

// ── group_norm ───────────────────────────────────────────────────────

#[test]
fn group_norm_basic() {
    let config = GroupNormConfig {
        num_groups: 2,
        num_channels: 4,
        spatial_size: 1,
        eps: 1e-5,
        elementwise_affine: true,
    };
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 4 channels, 1 spatial
    let gamma = vec![1.0; 4];
    let result = group_norm(&input, &gamma, None, &config).unwrap();
    assert_eq!(result.len(), 4);
    for v in &result {
        assert!(v.is_finite());
    }
}

// ── instance_norm ────────────────────────────────────────────────────

#[test]
fn instance_norm_basic() {
    let config = GroupNormConfig {
        num_groups: 4, // instance norm: groups == channels
        num_channels: 4,
        spatial_size: 1,
        eps: 1e-5,
        elementwise_affine: true,
    };
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let result = instance_norm(&input, &gamma, None, &config).unwrap();
    assert_eq!(result.len(), 4);
    for v in &result {
        assert!(v.is_finite());
    }
}

// ── Batch variants ───────────────────────────────────────────────────

#[test]
fn batch_layer_norm_multiple_inputs() {
    let config = LayerNormConfig { normalized_shape: vec![3], eps: 1e-5, elementwise_affine: true };
    let input1 = vec![1.0f32, 2.0, 3.0];
    let input2 = vec![4.0f32, 5.0, 6.0];
    let inputs: Vec<&[f32]> = vec![&input1, &input2];
    let gamma = vec![1.0; 3];
    let results = batch_layer_norm(&inputs, &gamma, None, &config).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 3);
    assert_eq!(results[1].len(), 3);
}

#[test]
fn batch_layer_norm_single_input_matches_non_batch() {
    let config = LayerNormConfig { normalized_shape: vec![4], eps: 1e-5, elementwise_affine: true };
    let input = vec![1.0f32, 3.0, 5.0, 7.0];
    let gamma = vec![1.0; 4];
    let single = layer_norm(&input, &gamma, None, &config).unwrap();
    let batch = batch_layer_norm(&[input.as_slice()], &gamma, None, &config).unwrap();
    assert_eq!(batch.len(), 1);
    for (a, b) in single.iter().zip(batch[0].iter()) {
        approx_eq(*a, *b, 1e-6);
    }
}

#[test]
fn batch_rms_norm_multiple_inputs() {
    let config = LayerNormConfig { normalized_shape: vec![3], eps: 1e-5, elementwise_affine: true };
    let input1 = vec![1.0f32, 2.0, 3.0];
    let input2 = vec![4.0f32, 5.0, 6.0];
    let inputs: Vec<&[f32]> = vec![&input1, &input2];
    let gamma = vec![1.0; 3];
    let results = batch_rms_norm(&inputs, &gamma, &config).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn batch_rms_norm_single_matches_non_batch() {
    let config = LayerNormConfig { normalized_shape: vec![3], eps: 1e-5, elementwise_affine: true };
    let input = vec![2.0f32, 4.0, 6.0];
    let gamma = vec![1.0; 3];
    let single = rms_norm(&input, &gamma, &config).unwrap();
    let batch = batch_rms_norm(&[input.as_slice()], &gamma, &config).unwrap();
    assert_eq!(batch.len(), 1);
    for (a, b) in single.iter().zip(batch[0].iter()) {
        approx_eq(*a, *b, 1e-6);
    }
}

// ── Numerical stability ──────────────────────────────────────────────

#[test]
fn layer_norm_large_values_stable() {
    let config = LayerNormConfig { normalized_shape: vec![4], eps: 1e-5, elementwise_affine: true };
    let input = vec![1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
    let gamma = vec![1.0; 4];
    let result = layer_norm(&input, &gamma, None, &config).unwrap();
    for v in &result {
        assert!(v.is_finite(), "non-finite output for large input: {v}");
    }
}

#[test]
fn rms_norm_negative_values() {
    let config = LayerNormConfig { normalized_shape: vec![3], eps: 1e-5, elementwise_affine: true };
    let input = vec![-1.0, -2.0, -3.0];
    let gamma = vec![1.0; 3];
    let result = rms_norm(&input, &gamma, &config).unwrap();
    // RMS is computed from squares, so negatives don't affect RMS
    for v in &result {
        assert!(v.is_finite());
        assert!(*v < 0.0); // scaled negatives remain negative
    }
}
