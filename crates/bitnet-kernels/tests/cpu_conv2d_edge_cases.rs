//! Edge-case tests for CPU conv2d operations.
//!
//! Tests cover 2D convolution, depthwise convolution,
//! output size computation, and configuration.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::conv2d::{Conv2dConfig, compute_output_size, conv2d, depthwise_conv2d};

// ── Output size computation ──────────────────────────────────────────

#[test]
fn output_size_no_padding() {
    // in=4, kernel=3, stride=1, pad=0, dilation=1 → out=2
    let out = compute_output_size(4, 3, 1, 0, 1);
    assert_eq!(out, 2);
}

#[test]
fn output_size_with_padding() {
    // in=4, kernel=3, stride=1, pad=1, dilation=1 → out=4
    let out = compute_output_size(4, 3, 1, 1, 1);
    assert_eq!(out, 4);
}

#[test]
fn output_size_stride_2() {
    // in=6, kernel=3, stride=2, pad=0, dilation=1 → out=2
    let out = compute_output_size(6, 3, 2, 0, 1);
    assert_eq!(out, 2);
}

#[test]
fn output_size_dilation() {
    // in=7, kernel=3, stride=1, pad=0, dilation=2 → effective_k=5, out=3
    let out = compute_output_size(7, 3, 1, 0, 2);
    assert_eq!(out, 3);
}

#[test]
fn output_size_1x1() {
    // 1x1 conv with stride=1
    let out = compute_output_size(8, 1, 1, 0, 1);
    assert_eq!(out, 8);
}

// ── Conv2dConfig ─────────────────────────────────────────────────────

#[test]
fn config_defaults() {
    let config = Conv2dConfig::new(3, 16, 3);
    assert_eq!(config.in_channels, 3);
    assert_eq!(config.out_channels, 16);
    assert_eq!(config.kernel_h, 3);
    assert_eq!(config.kernel_w, 3);
}

// ── conv2d ───────────────────────────────────────────────────────────

#[test]
fn conv2d_1x1_identity_like() {
    let config = Conv2dConfig::new(1, 1, 1); // 1 in, 1 out, 1x1 kernel
    // Single channel 2x2 input
    let input = vec![1.0, 2.0, 3.0, 4.0];
    // 1x1 kernel weight = 1.0
    let weight = vec![1.0];
    let result = conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
    assert_eq!(result.len(), 4);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[3] - 4.0).abs() < 1e-6);
}

#[test]
fn conv2d_with_bias() {
    let config = Conv2dConfig::new(1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 batch, 1 chan, 2x2
    let weight = vec![1.0];
    let bias = vec![10.0];
    let result = conv2d(&input, &weight, Some(&bias), &config, 1, 2, 2).unwrap();
    assert!((result[0] - 11.0).abs() < 1e-6);
    assert!((result[1] - 12.0).abs() < 1e-6);
}

#[test]
fn conv2d_3x3_on_4x4() {
    let mut config = Conv2dConfig::new(1, 1, 3);
    config.stride_h = 1;
    config.stride_w = 1;
    config.padding_h = 0;
    config.padding_w = 0;
    // 4x4 input with all 1s
    let input = vec![1.0; 16];
    // 3x3 kernel with all 1s → each output = 9.0
    let weight = vec![1.0; 9];
    let result = conv2d(&input, &weight, None, &config, 1, 4, 4).unwrap();
    // Output: 2x2
    assert_eq!(result.len(), 4);
    for val in &result {
        assert!((val - 9.0).abs() < 1e-4, "Expected 9.0, got {val}");
    }
}

// ── depthwise_conv2d ─────────────────────────────────────────────────

#[test]
fn depthwise_conv2d_single_channel() {
    let config = Conv2dConfig::new(1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![2.0]; // scale by 2
    let result = depthwise_conv2d(&input, &weight, None, &config, 1, 2, 2).unwrap();
    assert_eq!(result.len(), 4);
    assert!((result[0] - 2.0).abs() < 1e-6);
    assert!((result[3] - 8.0).abs() < 1e-6);
}

#[test]
fn depthwise_conv2d_with_bias() {
    let config = Conv2dConfig::new(1, 1, 1);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0];
    let bias = vec![5.0];
    let result = depthwise_conv2d(&input, &weight, Some(&bias), &config, 1, 2, 2).unwrap();
    assert!((result[0] - 6.0).abs() < 1e-6);
}
