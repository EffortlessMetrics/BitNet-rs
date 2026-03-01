//! Edge-case integration tests for `bitnet_kernels::cpu::conv2d` module.
//!
//! Covers:
//! - Conv2dConfig: creation, defaults, validation
//! - compute_output_size: formula correctness, padding, dilation, stride
//! - conv2d: basic, with bias, padding, stride, groups, batch
//! - depthwise_conv2d: basic, with bias, error paths
//! - im2col: basic transform, padding, group index
//! - Error paths: zero dimensions, size mismatches, invalid groups

use bitnet_kernels::cpu::conv2d::{
    Conv2dConfig, compute_output_size, conv2d, depthwise_conv2d, im2col,
};

const TOL: f32 = 1e-5;

// =========================================================================
// Conv2dConfig
// =========================================================================

#[test]
fn config_new_defaults() {
    let cfg = Conv2dConfig::new(3, 16, 3);
    assert_eq!(cfg.in_channels, 3);
    assert_eq!(cfg.out_channels, 16);
    assert_eq!(cfg.kernel_h, 3);
    assert_eq!(cfg.kernel_w, 3);
    assert_eq!(cfg.stride_h, 1);
    assert_eq!(cfg.stride_w, 1);
    assert_eq!(cfg.padding_h, 0);
    assert_eq!(cfg.padding_w, 0);
    assert_eq!(cfg.dilation_h, 1);
    assert_eq!(cfg.dilation_w, 1);
    assert_eq!(cfg.groups, 1);
}

#[test]
fn config_default() {
    let cfg = Conv2dConfig::default();
    assert_eq!(cfg.in_channels, 1);
    assert_eq!(cfg.out_channels, 1);
    assert_eq!(cfg.kernel_h, 1);
}

// =========================================================================
// compute_output_size
// =========================================================================

#[test]
fn output_size_no_padding() {
    // (5 + 0 - 1*(3-1) - 1) / 1 + 1 = (5 - 3) / 1 + 1 = 3
    assert_eq!(compute_output_size(5, 3, 1, 0, 1), 3);
}

#[test]
fn output_size_with_padding() {
    // (5 + 2*1 - 1*(3-1) - 1) / 1 + 1 = (7 - 3) / 1 + 1 = 5
    assert_eq!(compute_output_size(5, 3, 1, 1, 1), 5);
}

#[test]
fn output_size_with_stride() {
    // (6 + 0 - 3) / 2 + 1 = 3/2 + 1 = 2
    assert_eq!(compute_output_size(6, 3, 2, 0, 1), 2);
}

#[test]
fn output_size_with_dilation() {
    // effective_kernel = 2*(3-1) + 1 = 5
    // (7 + 0 - 5) / 1 + 1 = 3
    assert_eq!(compute_output_size(7, 3, 1, 0, 2), 3);
}

#[test]
fn output_size_kernel_larger_than_input() {
    // 3 + 0 < 5 → 0
    assert_eq!(compute_output_size(3, 5, 1, 0, 1), 0);
}

#[test]
fn output_size_1x1_kernel() {
    // (N + 0 - 1) / 1 + 1 = N
    assert_eq!(compute_output_size(10, 1, 1, 0, 1), 10);
}

#[test]
fn output_size_same_padding() {
    // 5 + 2*2 - 1*(5-1) - 1 = 9 - 5 = 4, /1 + 1 = 5
    assert_eq!(compute_output_size(5, 5, 1, 2, 1), 5);
}

// =========================================================================
// conv2d: basic 1×1 convolution (scaling)
// =========================================================================

#[test]
fn conv2d_1x1_basic() {
    // 1 in_ch, 1 out_ch, 1x1 kernel, 2x2 input
    let cfg = Conv2dConfig::new(1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1×1×2×2
    let weight = vec![2.0]; // 1×1×1×1
    let out = conv2d(&input, &weight, None, &cfg, 1, 2, 2).unwrap();
    assert_eq!(out.len(), 4);
    assert!((out[0] - 2.0).abs() < TOL);
    assert!((out[1] - 4.0).abs() < TOL);
    assert!((out[2] - 6.0).abs() < TOL);
    assert!((out[3] - 8.0).abs() < TOL);
}

#[test]
fn conv2d_1x1_with_bias() {
    let cfg = Conv2dConfig::new(1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0];
    let bias = vec![10.0];
    let out = conv2d(&input, &weight, Some(&bias), &cfg, 1, 2, 2).unwrap();
    assert!((out[0] - 11.0).abs() < TOL);
    assert!((out[3] - 14.0).abs() < TOL);
}

// =========================================================================
// conv2d: 3x3 kernel
// =========================================================================

#[test]
fn conv2d_3x3_identity_like() {
    // 1 in, 1 out, 3x3 kernel on 3x3 input → 1x1 output
    let cfg = Conv2dConfig::new(1, 1, 3);
    #[rustfmt::skip]
    let input = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    // Identity-like kernel (all ones)
    let weight = vec![1.0; 9];
    let out = conv2d(&input, &weight, None, &cfg, 1, 3, 3).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 3.0).abs() < TOL); // sum of diagonal = 3
}

#[test]
fn conv2d_3x3_with_padding() {
    let mut cfg = Conv2dConfig::new(1, 1, 3);
    cfg.padding_h = 1;
    cfg.padding_w = 1;
    let input = vec![1.0; 4]; // 1×1×2×2
    let weight = vec![1.0; 9]; // All ones
    let out = conv2d(&input, &weight, None, &cfg, 1, 2, 2).unwrap();
    // With padding=1 on 2x2 input, output is 2x2
    assert_eq!(out.len(), 4);
    // Corner: 1 real element * 1 weight = 1 (only the 1 non-padded position in the 3x3 window)
    assert!(out[0] > 0.0);
}

// =========================================================================
// conv2d: multi-channel
// =========================================================================

#[test]
fn conv2d_multi_in_channel() {
    // 2 in_ch, 1 out_ch, 1x1 kernel, 1x1 input
    let cfg = Conv2dConfig::new(2, 1, 1);
    let input = vec![3.0, 5.0]; // 1×2×1×1
    let weight = vec![2.0, 1.0]; // 1×2×1×1
    let out = conv2d(&input, &weight, None, &cfg, 1, 1, 1).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 11.0).abs() < TOL); // 3*2 + 5*1
}

#[test]
fn conv2d_multi_out_channel() {
    // 1 in, 2 out, 1x1 kernel
    let cfg = Conv2dConfig::new(1, 2, 1);
    let input = vec![5.0]; // 1×1×1×1
    let weight = vec![1.0, 3.0]; // 2×1×1×1
    let out = conv2d(&input, &weight, None, &cfg, 1, 1, 1).unwrap();
    assert_eq!(out.len(), 2);
    assert!((out[0] - 5.0).abs() < TOL);
    assert!((out[1] - 15.0).abs() < TOL);
}

// =========================================================================
// conv2d: stride
// =========================================================================

#[test]
fn conv2d_stride_2() {
    let mut cfg = Conv2dConfig::new(1, 1, 1);
    cfg.stride_h = 2;
    cfg.stride_w = 2;
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let weight = vec![1.0];
    let out = conv2d(&input, &weight, None, &cfg, 1, 4, 4).unwrap();
    // Stride 2 on 4x4 → 2x2
    assert_eq!(out.len(), 4);
    assert!((out[0] - 1.0).abs() < TOL);
    assert!((out[1] - 3.0).abs() < TOL);
    assert!((out[2] - 9.0).abs() < TOL);
    assert!((out[3] - 11.0).abs() < TOL);
}

// =========================================================================
// conv2d: grouped convolution
// =========================================================================

#[test]
fn conv2d_groups_2() {
    let mut cfg = Conv2dConfig::new(4, 4, 1);
    cfg.groups = 2;
    // 4 in_ch (2 groups of 2), 4 out_ch (2 groups of 2), 1x1 kernel
    // input: 1×4×1×1
    let input = vec![1.0, 2.0, 3.0, 4.0];
    // weight: 4×2×1×1 (each group: 2 out_ch × 2 in_ch)
    let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let out = conv2d(&input, &weight, None, &cfg, 1, 1, 1).unwrap();
    assert_eq!(out.len(), 4);
}

// =========================================================================
// conv2d: batch processing
// =========================================================================

#[test]
fn conv2d_batch_2() {
    let cfg = Conv2dConfig::new(1, 1, 1);
    let input = vec![2.0, 3.0]; // 2×1×1×1
    let weight = vec![10.0];
    let out = conv2d(&input, &weight, None, &cfg, 2, 1, 1).unwrap();
    assert_eq!(out.len(), 2);
    assert!((out[0] - 20.0).abs() < TOL);
    assert!((out[1] - 30.0).abs() < TOL);
}

// =========================================================================
// conv2d: error paths
// =========================================================================

#[test]
fn conv2d_zero_in_channels_error() {
    let cfg = Conv2dConfig::new(0, 1, 1);
    assert!(conv2d(&[1.0], &[1.0], None, &cfg, 1, 1, 1).is_err());
}

#[test]
fn conv2d_zero_out_channels_error() {
    let cfg = Conv2dConfig::new(1, 0, 1);
    assert!(conv2d(&[1.0], &[], None, &cfg, 1, 1, 1).is_err());
}

#[test]
fn conv2d_input_size_mismatch_error() {
    let cfg = Conv2dConfig::new(1, 1, 1);
    assert!(conv2d(&[1.0, 2.0], &[1.0], None, &cfg, 1, 1, 1).is_err());
}

#[test]
fn conv2d_weight_size_mismatch_error() {
    let cfg = Conv2dConfig::new(1, 1, 3);
    let input = vec![0.0; 9]; // 1×1×3×3
    let weight = vec![1.0; 5]; // Wrong
    assert!(conv2d(&input, &weight, None, &cfg, 1, 3, 3).is_err());
}

#[test]
fn conv2d_bias_length_mismatch_error() {
    let cfg = Conv2dConfig::new(1, 2, 1);
    let input = vec![1.0]; // 1×1×1×1
    let weight = vec![1.0, 1.0]; // 2×1×1×1
    let bias = vec![0.0]; // Should be length 2
    assert!(conv2d(&input, &weight, Some(&bias), &cfg, 1, 1, 1).is_err());
}

#[test]
fn conv2d_kernel_larger_than_input_error() {
    let cfg = Conv2dConfig::new(1, 1, 5);
    let input = vec![0.0; 4]; // 1×1×2×2
    let weight = vec![0.0; 25]; // 1×1×5×5
    assert!(conv2d(&input, &weight, None, &cfg, 1, 2, 2).is_err());
}

#[test]
fn conv2d_groups_not_divisible_error() {
    let mut cfg = Conv2dConfig::new(3, 3, 1);
    cfg.groups = 2; // 3 not divisible by 2
    assert!(conv2d(&[0.0; 3], &[0.0; 3], None, &cfg, 1, 1, 1).is_err());
}

// =========================================================================
// depthwise_conv2d
// =========================================================================

#[test]
fn depthwise_basic() {
    let mut cfg = Conv2dConfig::new(2, 2, 1);
    cfg.groups = 2;
    // 2 channels, 1x1 kernel, 2x2 input
    let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // 1×2×2×2
    let weight = vec![2.0, 3.0]; // 2×1×1×1
    let out = depthwise_conv2d(&input, &weight, None, &cfg, 1, 2, 2).unwrap();
    assert_eq!(out.len(), 8);
    // Channel 0: scaled by 2
    assert!((out[0] - 2.0).abs() < TOL);
    assert!((out[3] - 8.0).abs() < TOL);
    // Channel 1: scaled by 3
    assert!((out[4] - 30.0).abs() < TOL);
    assert!((out[7] - 120.0).abs() < TOL);
}

#[test]
fn depthwise_with_bias() {
    let mut cfg = Conv2dConfig::new(1, 1, 1);
    cfg.groups = 1;
    let input = vec![5.0]; // 1×1×1×1
    let weight = vec![2.0];
    let bias = vec![100.0];
    let out = depthwise_conv2d(&input, &weight, Some(&bias), &cfg, 1, 1, 1).unwrap();
    assert!((out[0] - 110.0).abs() < TOL);
}

#[test]
fn depthwise_groups_ne_channels_error() {
    let mut cfg = Conv2dConfig::new(4, 4, 1);
    cfg.groups = 2; // groups != in_channels
    assert!(depthwise_conv2d(&[0.0; 4], &[0.0; 4], None, &cfg, 1, 1, 1).is_err());
}

#[test]
fn depthwise_in_ne_out_channels_error() {
    let mut cfg = Conv2dConfig::new(2, 4, 1);
    cfg.groups = 2;
    assert!(depthwise_conv2d(&[0.0; 2], &[0.0; 2], None, &cfg, 1, 1, 1).is_err());
}

// =========================================================================
// im2col
// =========================================================================

#[test]
fn im2col_basic_1x1() {
    let cfg = Conv2dConfig::new(1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1×1×2×2
    let cols = im2col(&input, &cfg, 2, 2, 0).unwrap();
    // col_h = 1*1*1 = 1, col_w = 2*2 = 4
    assert_eq!(cols.len(), 4);
    assert!((cols[0] - 1.0).abs() < TOL);
    assert!((cols[3] - 4.0).abs() < TOL);
}

#[test]
fn im2col_3x3_kernel() {
    let cfg = Conv2dConfig::new(1, 1, 3);
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let cols = im2col(&input, &cfg, 3, 3, 0).unwrap();
    // out_h=1, out_w=1, col_h=9, col_w=1
    assert_eq!(cols.len(), 9);
    // Should be the entire input flattened
    for (i, v) in cols.iter().enumerate() {
        assert!((v - (i as f32 + 1.0)).abs() < TOL, "im2col[{i}] = {v}, expected {}", i + 1);
    }
}

#[test]
fn im2col_group_index_error() {
    let cfg = Conv2dConfig::new(1, 1, 1);
    let input = vec![1.0]; // 1×1×1×1
    assert!(im2col(&input, &cfg, 1, 1, 1).is_err()); // group 1 >= groups(1)
}

#[test]
fn im2col_input_size_mismatch_error() {
    let cfg = Conv2dConfig::new(2, 1, 1);
    let input = vec![1.0]; // Should be 2
    assert!(im2col(&input, &cfg, 1, 1, 0).is_err());
}

// =========================================================================
// conv2d: zero-weight (all zeros)
// =========================================================================

#[test]
fn conv2d_zero_weights_no_bias() {
    let cfg = Conv2dConfig::new(1, 1, 3);
    let input = vec![1.0; 9]; // 1×1×3×3
    let weight = vec![0.0; 9];
    let out = conv2d(&input, &weight, None, &cfg, 1, 3, 3).unwrap();
    assert!((out[0]).abs() < TOL, "zero weights should give zero output");
}

#[test]
fn conv2d_zero_weights_with_bias() {
    let cfg = Conv2dConfig::new(1, 1, 3);
    let input = vec![1.0; 9];
    let weight = vec![0.0; 9];
    let bias = vec![42.0];
    let out = conv2d(&input, &weight, Some(&bias), &cfg, 1, 3, 3).unwrap();
    assert!((out[0] - 42.0).abs() < TOL, "should output only bias");
}
