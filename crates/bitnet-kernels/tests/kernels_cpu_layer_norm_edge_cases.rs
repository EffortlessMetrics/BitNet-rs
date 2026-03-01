//! Edge-case tests for `bitnet_kernels::cpu::layer_norm` module.
//!
//! Covers:
//! - LayerNormConfig: creation, defaults, validation
//! - layer_norm: affine/no-affine, with/without beta, batch, errors
//! - rms_norm: basic, batch, errors, compared to known math
//! - GroupNormConfig / group_norm: basic, batch, errors
//! - instance_norm: correctness and error cases
//! - batch_* wrappers for all norm types
//! - Numerical properties: zero-mean output, near-unit variance

use bitnet_kernels::cpu::layer_norm::{
    GroupNormConfig, LayerNormConfig, batch_group_norm, batch_instance_norm, batch_layer_norm,
    batch_rms_norm, group_norm, instance_norm, layer_norm, rms_norm,
};

// =========================================================================
// LayerNormConfig
// =========================================================================

#[test]
fn layer_norm_config_new() {
    let cfg = LayerNormConfig::new(vec![128]);
    assert_eq!(cfg.normalized_shape, vec![128]);
    assert!((cfg.eps - 1e-5).abs() < 1e-10);
    assert!(cfg.elementwise_affine);
}

#[test]
fn layer_norm_config_default() {
    let cfg = LayerNormConfig::default();
    assert_eq!(cfg.normalized_shape, vec![1]);
    assert!(cfg.elementwise_affine);
}

#[test]
fn layer_norm_config_multi_dim() {
    let cfg = LayerNormConfig::new(vec![4, 8]);
    // norm_size = 4 * 8 = 32 (private, but we test via layer_norm)
    let gamma = vec![1.0; 32];
    let input = vec![1.0; 32];
    layer_norm(&input, &gamma, None, &cfg).unwrap();
}

// =========================================================================
// layer_norm: basic correctness
// =========================================================================

#[test]
fn layer_norm_constant_input() {
    // Constant input → zero variance → output ≈ 0 (centered)
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![5.0; 4];
    let gamma = vec![1.0; 4];
    let out = layer_norm(&input, &gamma, None, &cfg).unwrap();
    for &v in &out {
        assert!(v.abs() < 1e-3, "constant input should normalize to ~0, got {v}");
    }
}

#[test]
fn layer_norm_with_beta() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![5.0; 4];
    let gamma = vec![1.0; 4];
    let beta = vec![10.0; 4];
    let out = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
    for &v in &out {
        assert!((v - 10.0).abs() < 1e-3, "constant input + beta should shift to ~10, got {v}");
    }
}

#[test]
fn layer_norm_zero_mean_output() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let out = layer_norm(&input, &gamma, None, &cfg).unwrap();
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    assert!(mean.abs() < 1e-5, "output should be zero-mean, got {mean}");
}

#[test]
fn layer_norm_unit_variance_output() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let out = layer_norm(&input, &gamma, None, &cfg).unwrap();
    let mean: f32 = out.iter().sum::<f32>() / 4.0;
    let var: f32 = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
    assert!((var - 1.0).abs() < 0.01, "output variance should be ~1.0, got {var}");
}

#[test]
fn layer_norm_no_affine() {
    let cfg = LayerNormConfig { normalized_shape: vec![3], eps: 1e-5, elementwise_affine: false };
    let input = vec![1.0, 2.0, 3.0];
    let gamma = vec![]; // Ignored
    let out = layer_norm(&input, &gamma, None, &cfg).unwrap();
    let mean: f32 = out.iter().sum::<f32>() / 3.0;
    assert!(mean.abs() < 1e-5);
}

#[test]
fn layer_norm_gamma_scaling() {
    let cfg = LayerNormConfig::new(vec![2]);
    let input = vec![0.0, 2.0]; // mean=1, will normalize to [-1, 1] approx
    let gamma = vec![3.0, 3.0]; // Scale by 3
    let out = layer_norm(&input, &gamma, None, &cfg).unwrap();
    // Without gamma: values are ~[-1, 1]
    // With gamma=3: values are ~[-3, 3]
    assert!(out[0] < -1.0, "gamma should amplify");
    assert!(out[1] > 1.0, "gamma should amplify");
}

// =========================================================================
// layer_norm: batch processing
// =========================================================================

#[test]
fn layer_norm_batched() {
    let cfg = LayerNormConfig::new(vec![3]);
    let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]; // 2 sequences of length 3
    let gamma = vec![1.0; 3];
    let out = layer_norm(&input, &gamma, None, &cfg).unwrap();
    assert_eq!(out.len(), 6);

    // Each 3-element subsequence should be independently normalized
    let mean1: f32 = out[0..3].iter().sum::<f32>() / 3.0;
    let mean2: f32 = out[3..6].iter().sum::<f32>() / 3.0;
    assert!(mean1.abs() < 1e-5);
    assert!(mean2.abs() < 1e-5);
}

// =========================================================================
// layer_norm: error cases
// =========================================================================

#[test]
fn layer_norm_empty_input_error() {
    let cfg = LayerNormConfig::new(vec![4]);
    let gamma = vec![1.0; 4];
    assert!(layer_norm(&[], &gamma, None, &cfg).is_err());
}

#[test]
fn layer_norm_wrong_gamma_length_error() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1.0; 4];
    let gamma = vec![1.0; 3]; // Wrong length
    assert!(layer_norm(&input, &gamma, None, &cfg).is_err());
}

#[test]
fn layer_norm_wrong_beta_length_error() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1.0; 4];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 3]; // Wrong length
    assert!(layer_norm(&input, &gamma, Some(&beta), &cfg).is_err());
}

#[test]
fn layer_norm_non_divisible_input_error() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1.0; 5]; // 5 not divisible by 4
    let gamma = vec![1.0; 4];
    assert!(layer_norm(&input, &gamma, None, &cfg).is_err());
}

#[test]
fn layer_norm_zero_norm_shape_error() {
    let cfg = LayerNormConfig::new(vec![0]);
    let gamma = vec![];
    assert!(layer_norm(&[1.0], &gamma, None, &cfg).is_err());
}

#[test]
fn layer_norm_zero_eps_error() {
    let cfg = LayerNormConfig { normalized_shape: vec![2], eps: 0.0, elementwise_affine: false };
    assert!(layer_norm(&[1.0, 2.0], &[], None, &cfg).is_err());
}

#[test]
fn layer_norm_negative_eps_error() {
    let cfg = LayerNormConfig { normalized_shape: vec![2], eps: -1e-5, elementwise_affine: false };
    assert!(layer_norm(&[1.0, 2.0], &[], None, &cfg).is_err());
}

// =========================================================================
// rms_norm: basic correctness
// =========================================================================

#[test]
fn rms_norm_basic() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let out = rms_norm(&input, &gamma, &cfg).unwrap();
    assert_eq!(out.len(), 4);
    // RMS = sqrt(mean(x²)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // Each output: x / sqrt(rms + eps) * gamma
    // rms_norm does NOT subtract mean, so output is not zero-mean
    for &v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn rms_norm_unit_input() {
    // All ones → rms = 1.0 → output = 1 / sqrt(1+eps) ≈ 1.0
    let cfg = LayerNormConfig::new(vec![3]);
    let input = vec![1.0; 3];
    let gamma = vec![1.0; 3];
    let out = rms_norm(&input, &gamma, &cfg).unwrap();
    for &v in &out {
        assert!((v - 1.0).abs() < 0.01, "unit input rms_norm ≈ 1, got {v}");
    }
}

#[test]
fn rms_norm_scaling_by_gamma() {
    let cfg = LayerNormConfig::new(vec![2]);
    let input = vec![1.0, 1.0];
    let gamma_1 = vec![1.0; 2];
    let gamma_2 = vec![2.0; 2];
    let out1 = rms_norm(&input, &gamma_1, &cfg).unwrap();
    let out2 = rms_norm(&input, &gamma_2, &cfg).unwrap();
    assert!((out2[0] / out1[0] - 2.0).abs() < 1e-4, "gamma=2 should double output");
}

#[test]
fn rms_norm_batched() {
    let cfg = LayerNormConfig::new(vec![2]);
    let input = vec![1.0, 3.0, 10.0, 30.0]; // 2 batches of size 2
    let gamma = vec![1.0; 2];
    let out = rms_norm(&input, &gamma, &cfg).unwrap();
    assert_eq!(out.len(), 4);
    // Batch 1 and batch 2 normalized independently
    // Ratios should be preserved within each batch
    let ratio1 = out[1] / out[0];
    let ratio2 = out[3] / out[2];
    assert!((ratio1 - 3.0).abs() < 0.01, "rms preserves ratios, got {ratio1}");
    assert!((ratio2 - 3.0).abs() < 0.01, "rms preserves ratios, got {ratio2}");
}

// =========================================================================
// rms_norm: error cases
// =========================================================================

#[test]
fn rms_norm_empty_input_error() {
    let cfg = LayerNormConfig::new(vec![4]);
    let gamma = vec![1.0; 4];
    assert!(rms_norm(&[], &gamma, &cfg).is_err());
}

#[test]
fn rms_norm_wrong_gamma_length_error() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1.0; 4];
    let gamma = vec![1.0; 3];
    assert!(rms_norm(&input, &gamma, &cfg).is_err());
}

// =========================================================================
// GroupNormConfig
// =========================================================================

#[test]
fn group_norm_config_new() {
    let cfg = GroupNormConfig::new(4, 8, 16);
    assert_eq!(cfg.num_groups, 4);
    assert_eq!(cfg.num_channels, 8);
    assert_eq!(cfg.spatial_size, 16);
    assert!((cfg.eps - 1e-5).abs() < 1e-10);
    assert!(cfg.elementwise_affine);
}

// =========================================================================
// group_norm: basic correctness
// =========================================================================

#[test]
fn group_norm_basic() {
    // 2 groups, 4 channels, spatial=2
    // batch_size = 1
    let cfg = GroupNormConfig::new(2, 4, 2);
    let input = vec![
        1.0, 2.0, // ch0
        3.0, 4.0, // ch1
        5.0, 6.0, // ch2
        7.0, 8.0, // ch3
    ];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let out = group_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
    assert_eq!(out.len(), 8);
    // Group 0 (ch0,ch1) should be zero-mean within the group
    let g0_mean: f32 = out[0..4].iter().sum::<f32>() / 4.0;
    assert!(g0_mean.abs() < 1e-4, "group 0 mean should be ~0, got {g0_mean}");
}

#[test]
fn group_norm_no_affine() {
    let cfg = GroupNormConfig {
        num_groups: 1,
        num_channels: 2,
        spatial_size: 2,
        eps: 1e-5,
        elementwise_affine: false,
    };
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![];
    let out = group_norm(&input, &gamma, None, &cfg).unwrap();
    assert_eq!(out.len(), 4);
    let mean: f32 = out.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5);
}

// =========================================================================
// group_norm: error cases
// =========================================================================

#[test]
fn group_norm_not_divisible_error() {
    // 3 channels, 2 groups → not divisible
    let cfg = GroupNormConfig::new(2, 3, 1);
    let input = vec![1.0; 3];
    let gamma = vec![1.0; 3];
    assert!(group_norm(&input, &gamma, None, &cfg).is_err());
}

#[test]
fn group_norm_empty_input_error() {
    let cfg = GroupNormConfig::new(1, 2, 2);
    let gamma = vec![1.0; 2];
    assert!(group_norm(&[], &gamma, None, &cfg).is_err());
}

#[test]
fn group_norm_wrong_gamma_length_error() {
    let cfg = GroupNormConfig::new(1, 2, 2);
    let input = vec![1.0; 4];
    let gamma = vec![1.0; 3]; // Should be 2
    assert!(group_norm(&input, &gamma, None, &cfg).is_err());
}

// =========================================================================
// instance_norm
// =========================================================================

#[test]
fn instance_norm_basic() {
    // instance norm = group norm with num_groups == num_channels
    let cfg = GroupNormConfig::new(3, 3, 4);
    let input = vec![
        1.0, 2.0, 3.0, 4.0, // ch0
        5.0, 6.0, 7.0, 8.0, // ch1
        9.0, 10.0, 11.0, 12.0, // ch2
    ];
    let gamma = vec![1.0; 3];
    let out = instance_norm(&input, &gamma, None, &cfg).unwrap();
    assert_eq!(out.len(), 12);
    // Each channel independently normalized
    for ch in 0..3 {
        let ch_out = &out[ch * 4..(ch + 1) * 4];
        let mean: f32 = ch_out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "ch{ch} mean should be ~0, got {mean}");
    }
}

#[test]
fn instance_norm_groups_ne_channels_error() {
    let cfg = GroupNormConfig::new(2, 4, 1); // groups != channels
    let input = vec![1.0; 4];
    let gamma = vec![1.0; 4];
    assert!(instance_norm(&input, &gamma, None, &cfg).is_err());
}

// =========================================================================
// batch_layer_norm
// =========================================================================

#[test]
fn batch_layer_norm_two_inputs() {
    let cfg = LayerNormConfig::new(vec![3]);
    let gamma = vec![1.0; 3];
    let inp1: Vec<f32> = vec![1.0, 2.0, 3.0];
    let inp2: Vec<f32> = vec![10.0, 20.0, 30.0];
    let results = batch_layer_norm(&[&inp1, &inp2], &gamma, None, &cfg).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 3);
    assert_eq!(results[1].len(), 3);
}

#[test]
fn batch_layer_norm_empty() {
    let cfg = LayerNormConfig::new(vec![3]);
    let gamma = vec![1.0; 3];
    let results = batch_layer_norm(&[], &gamma, None, &cfg).unwrap();
    assert!(results.is_empty());
}

// =========================================================================
// batch_rms_norm
// =========================================================================

#[test]
fn batch_rms_norm_two_inputs() {
    let cfg = LayerNormConfig::new(vec![2]);
    let gamma = vec![1.0; 2];
    let inp1: Vec<f32> = vec![1.0, 2.0];
    let inp2: Vec<f32> = vec![3.0, 4.0];
    let results = batch_rms_norm(&[&inp1, &inp2], &gamma, &cfg).unwrap();
    assert_eq!(results.len(), 2);
}

// =========================================================================
// batch_group_norm
// =========================================================================

#[test]
fn batch_group_norm_two_inputs() {
    let cfg = GroupNormConfig::new(1, 2, 2);
    let gamma = vec![1.0; 2];
    let inp1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let inp2: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let results = batch_group_norm(&[&inp1, &inp2], &gamma, None, &cfg).unwrap();
    assert_eq!(results.len(), 2);
}

// =========================================================================
// batch_instance_norm
// =========================================================================

#[test]
fn batch_instance_norm_basic() {
    let cfg = GroupNormConfig::new(2, 2, 3);
    let gamma = vec![1.0; 2];
    let inp: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let results = batch_instance_norm(&[&inp], &gamma, None, &cfg).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 6);
}

// =========================================================================
// Numerical properties
// =========================================================================

#[test]
fn layer_norm_large_values_stable() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
    let gamma = vec![1.0; 4];
    let out = layer_norm(&input, &gamma, None, &cfg).unwrap();
    for &v in &out {
        assert!(v.is_finite(), "output should be finite for large inputs");
    }
    let mean: f32 = out.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 0.01, "should be approximately zero-mean");
}

#[test]
fn rms_norm_large_values_stable() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1e6, 1e6, 1e6, 1e6];
    let gamma = vec![1.0; 4];
    let out = rms_norm(&input, &gamma, &cfg).unwrap();
    for &v in &out {
        assert!(v.is_finite(), "output should be finite for large inputs");
    }
}

#[test]
fn layer_norm_small_values_stable() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1e-7, 2e-7, 3e-7, 4e-7];
    let gamma = vec![1.0; 4];
    let out = layer_norm(&input, &gamma, None, &cfg).unwrap();
    for &v in &out {
        assert!(v.is_finite(), "output should be finite for small inputs");
    }
}

// =========================================================================
// rms_norm vs layer_norm: rms preserves sign proportions
// =========================================================================

#[test]
fn rms_norm_preserves_sign() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![-2.0, -1.0, 1.0, 2.0];
    let gamma = vec![1.0; 4];
    let out = rms_norm(&input, &gamma, &cfg).unwrap();
    assert!(out[0] < 0.0, "negative input should give negative output");
    assert!(out[1] < 0.0);
    assert!(out[2] > 0.0, "positive input should give positive output");
    assert!(out[3] > 0.0);
}

#[test]
fn layer_norm_preserves_order() {
    let cfg = LayerNormConfig::new(vec![4]);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let out = layer_norm(&input, &gamma, None, &cfg).unwrap();
    // Output should preserve ordering
    assert!(out[0] < out[1]);
    assert!(out[1] < out[2]);
    assert!(out[2] < out[3]);
}
