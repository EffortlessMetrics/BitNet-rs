//! Edge-case tests for CPU batch normalization.
//!
//! Tests cover batch norm forward and inference modes,
//! with various configurations and edge cases.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};

// ── BatchNormConfig ──────────────────────────────────────────────────

#[test]
fn config_new() {
    let config = BatchNormConfig::new(64);
    assert_eq!(config.num_features, 64);
}

// ── batch_norm_inference ─────────────────────────────────────────────

#[test]
fn inference_identity_params() {
    // gamma=1, beta=0, mean=0, var=1 → output ≈ input
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 4 features, 1 sample
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0, 0.0];
    let mean = vec![0.0, 0.0, 0.0, 0.0];
    let var = vec![1.0, 1.0, 1.0, 1.0];
    let result = batch_norm_inference(&input, &gamma, &beta, &mean, &var, 1e-5).unwrap();
    assert_eq!(result.len(), 4);
    for (r, i) in result.iter().zip(input.iter()) {
        assert!((r - i).abs() < 0.01, "Expected {i}, got {r}");
    }
}

#[test]
fn inference_with_scale() {
    // gamma=2 → doubles the normalized output
    let input = vec![0.0, 0.0, 0.0, 0.0]; // mean-centered
    let gamma = vec![2.0, 2.0, 2.0, 2.0];
    let beta = vec![0.0, 0.0, 0.0, 0.0];
    let mean = vec![0.0, 0.0, 0.0, 0.0];
    let var = vec![1.0, 1.0, 1.0, 1.0];
    let result = batch_norm_inference(&input, &gamma, &beta, &mean, &var, 1e-5).unwrap();
    for val in &result {
        assert!(val.abs() < 0.01, "Expected ~0, got {val}");
    }
}

#[test]
fn inference_with_shift() {
    // beta=5 → shifts output by 5
    let input = vec![0.0, 0.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![5.0, 5.0];
    let mean = vec![0.0, 0.0];
    let var = vec![1.0, 1.0];
    let result = batch_norm_inference(&input, &gamma, &beta, &mean, &var, 1e-5).unwrap();
    for val in &result {
        assert!((val - 5.0).abs() < 0.01, "Expected ~5, got {val}");
    }
}

#[test]
fn inference_with_nonzero_mean() {
    // input=5, mean=5 → normalized=0 → output=beta=0
    let input = vec![5.0, 5.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let mean = vec![5.0, 5.0];
    let var = vec![1.0, 1.0];
    let result = batch_norm_inference(&input, &gamma, &beta, &mean, &var, 1e-5).unwrap();
    for val in &result {
        assert!(val.abs() < 0.01, "Expected ~0, got {val}");
    }
}

// ── batch_norm_forward ───────────────────────────────────────────────

#[test]
fn forward_basic() {
    let config = BatchNormConfig::new(2);
    // 2 features, batch of 2: [[1, 2], [3, 4]] flattened
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let running_mean = vec![0.0, 0.0];
    let running_var = vec![1.0, 1.0];
    let (output, _new_mean, _new_var) =
        batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();
    assert!(!output.is_empty());
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn forward_returns_statistics() {
    let config = BatchNormConfig::new(2);
    let input = vec![1.0, 10.0, 3.0, 20.0]; // 2 features, batch 2
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let running_mean = vec![0.0, 0.0];
    let running_var = vec![1.0, 1.0];
    let (_output, new_mean, new_var) =
        batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &config).unwrap();
    assert!(!new_mean.is_empty());
    assert!(!new_var.is_empty());
    for val in &new_var {
        assert!(*val >= 0.0, "Variance should be non-negative");
    }
}
