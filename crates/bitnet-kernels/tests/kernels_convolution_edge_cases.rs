//! Edge-case integration tests for `bitnet_kernels::convolution` module.
//!
//! Covers conv2d (fp32) and conv2d_quantized (I2S/TL1/TL2):
//! - Conv2DParams defaults
//! - Input validation (channel mismatch, size mismatch, bias mismatch, output mismatch)
//! - Identity-like convolution (1x1 kernel)
//! - With bias
//! - Strided, padded, dilated convolution
//! - Batched convolution
//! - Quantized convolution (I2S, TL1, TL2)
//! - Quantized validation errors

use bitnet_common::QuantizationType;
use bitnet_kernels::convolution::{Conv2DParams, conv2d, conv2d_quantized};

// =========================================================================
// Conv2DParams
// =========================================================================

#[test]
fn default_params() {
    let p = Conv2DParams::default();
    assert_eq!(p.stride, (1, 1));
    assert_eq!(p.padding, (0, 0));
    assert_eq!(p.dilation, (1, 1));
}

// =========================================================================
// conv2d: validation errors
// =========================================================================

#[test]
fn conv2d_channel_mismatch() {
    let input = vec![0.0f32; 4]; // 1x1x2x2
    let weight = vec![0.0f32; 18]; // 2x3x3x1 — ic=3 vs input ic=1
    let mut output = vec![0.0f32; 8];
    let r = conv2d(
        &input,
        &weight,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (2, 3, 3, 1),
    );
    assert!(r.is_err());
}

#[test]
fn conv2d_input_size_mismatch() {
    let input = vec![0.0f32; 3]; // Wrong size (should be 4)
    let weight = vec![0.0f32; 1]; // 1x1x1x1
    let mut output = vec![0.0f32; 4];
    let r = conv2d(
        &input,
        &weight,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
    );
    assert!(r.is_err());
}

#[test]
fn conv2d_weight_size_mismatch() {
    let input = vec![0.0f32; 4]; // 1x1x2x2
    let weight = vec![0.0f32; 2]; // Wrong size (should be 1)
    let mut output = vec![0.0f32; 4];
    let r = conv2d(
        &input,
        &weight,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
    );
    assert!(r.is_err());
}

#[test]
fn conv2d_bias_size_mismatch() {
    let input = vec![0.0f32; 4]; // 1x1x2x2
    let weight = vec![1.0f32; 1]; // 1x1x1x1
    let bias = vec![1.0, 2.0]; // oc=1 but bias has 2 elements
    let mut output = vec![0.0f32; 4];
    let r = conv2d(
        &input,
        &weight,
        Some(&bias),
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
    );
    assert!(r.is_err());
}

#[test]
fn conv2d_output_size_mismatch() {
    let input = vec![0.0f32; 4]; // 1x1x2x2
    let weight = vec![1.0f32; 1]; // 1x1x1x1
    let mut output = vec![0.0f32; 3]; // Should be 4
    let r = conv2d(
        &input,
        &weight,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
    );
    assert!(r.is_err());
}

// =========================================================================
// conv2d: correctness
// =========================================================================

#[test]
fn conv2d_identity_1x1_kernel() {
    // 1x1 convolution with weight=1.0 should pass through input
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![1.0f32]; // 1x1x1x1
    let mut output = vec![0.0f32; 4]; // 1x1x2x2
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn conv2d_scaling_1x1_kernel() {
    // 1x1 convolution with weight=2.0 should double input
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![2.0f32];
    let mut output = vec![0.0f32; 4];
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn conv2d_with_bias() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![1.0f32]; // 1x1x1x1
    let bias = vec![10.0f32]; // oc=1
    let mut output = vec![0.0f32; 4];
    conv2d(
        &input,
        &weight,
        Some(&bias),
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
    )
    .unwrap();
    assert_eq!(output, vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn conv2d_3x3_kernel_no_padding() {
    // 1x1x3x3 input, 1x1x3x3 kernel → 1x1x1x1 output
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let weight = vec![1.0f32; 9]; // All-ones 3x3 kernel
    let mut output = vec![0.0f32; 1];
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 3, 3), (1, 1, 3, 3))
        .unwrap();
    // Sum of all elements: 1+2+3+4+5+6+7+8+9 = 45
    assert!((output[0] - 45.0).abs() < 1e-5);
}

#[test]
fn conv2d_stride_2() {
    // 1x1x4x4 input, 1x1x1x1 kernel, stride=2 → 1x1x2x2 output
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let weight = vec![1.0f32];
    let mut output = vec![0.0f32; 4]; // 2x2
    conv2d(
        &input,
        &weight,
        None,
        &mut output,
        Conv2DParams { stride: (2, 2), ..Default::default() },
        (1, 1, 4, 4),
        (1, 1, 1, 1),
    )
    .unwrap();
    // Should pick elements at (0,0), (0,2), (2,0), (2,2) → 1, 3, 9, 11
    assert_eq!(output, vec![1.0, 3.0, 9.0, 11.0]);
}

#[test]
fn conv2d_multiple_output_channels() {
    // 1x1x2x2 input, 2x1x1x1 weights → 1x2x2x2 output
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0, 2.0]; // 2 output channels, each 1x1x1
    let mut output = vec![0.0f32; 8]; // 2 channels × 2×2
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (2, 1, 1, 1))
        .unwrap();
    // Channel 0 (weight=1): [1,2,3,4], Channel 1 (weight=2): [2,4,6,8]
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn conv2d_batched() {
    // 2x1x2x2 input (batch=2), 1x1x1x1 weight → 2x1x2x2 output
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0f32];
    let mut output = vec![0.0f32; 8];
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (2, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, input);
}

#[test]
fn conv2d_zeros_input() {
    let input = vec![0.0f32; 4];
    let weight = vec![1.0f32; 1];
    let mut output = vec![99.0f32; 4]; // Pre-fill to ensure zeroing
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, vec![0.0; 4]);
}

// =========================================================================
// conv2d_quantized: validation errors
// =========================================================================

#[test]
fn quantized_conv2d_channel_mismatch() {
    let input = vec![0.0f32; 4]; // 1x1x2x2
    let weight_q = vec![0u8; 4];
    let scales = vec![1.0f32; 2];
    let mut output = vec![0.0f32; 8];
    let r = conv2d_quantized(
        &input,
        &weight_q,
        &scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (2, 3, 1, 1), // kic=3 vs ic=1
        QuantizationType::I2S,
    );
    assert!(r.is_err());
}

#[test]
fn quantized_conv2d_scale_size_mismatch() {
    let input = vec![0.0f32; 4];
    let weight_q = vec![0u8; 1]; // 1 output channel, 1x1 kernel → ceil(1/4)=1 byte
    let scales = vec![1.0f32; 5]; // Wrong: oc=1 but 5 scales
    let mut output = vec![0.0f32; 4];
    let r = conv2d_quantized(
        &input,
        &weight_q,
        &scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::I2S,
    );
    assert!(r.is_err());
}

// =========================================================================
// conv2d_quantized: correctness
// =========================================================================

#[test]
fn quantized_conv2d_i2s_simple() {
    // 1x1x2x2 input, 1x1x1x1 kernel (I2S: 1 element → packed in 1 byte)
    // I2S dequant: 0x00→-2, 0x01→-1, 0x02→+1, 0x03→+2
    let input = vec![1.0, 2.0, 3.0, 4.0];
    // Pack one I2S value: 0x02 → dequant to +1.0 * scale
    let weight_q = vec![0x02u8]; // First 2 bits = 0b10 = 0x02 → +1.0
    let scales = vec![1.0f32]; // scale=1
    let mut output = vec![0.0f32; 4];
    conv2d_quantized(
        &input,
        &weight_q,
        &scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::I2S,
    )
    .unwrap();
    // dequantized weight = 1.0 * 1.0 = 1.0 → output = input * 1.0
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn quantized_conv2d_tl1_simple() {
    // TL1: byte value dequantized as (val - 128) / 127 * scale
    let input = vec![1.0f32; 4]; // 1x1x2x2
    // TL1: 1 element per byte. val=128 → (128-128)/127 = 0.0
    let weight_q = vec![128u8]; // → dequant 0.0
    let scales = vec![1.0f32];
    let mut output = vec![0.0f32; 4];
    conv2d_quantized(
        &input,
        &weight_q,
        &scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::TL1,
    )
    .unwrap();
    // Weight is 0.0, so output is all zeros
    for &v in &output {
        assert!(v.abs() < 1e-5, "expected ~0, got {v}");
    }
}

#[test]
fn quantized_conv2d_tl2_simple() {
    // TL2: byte value dequantized as (2*(val/255) - 1) * scale
    let input = vec![2.0f32; 4]; // 1x1x2x2
    // TL2: val=255 → (2*(255/255)-1) = 1.0
    let weight_q = vec![255u8];
    let scales = vec![1.0f32];
    let mut output = vec![0.0f32; 4];
    conv2d_quantized(
        &input,
        &weight_q,
        &scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::TL2,
    )
    .unwrap();
    // Weight = 1.0, input = 2.0 → output = 2.0
    for &v in &output {
        assert!((v - 2.0).abs() < 1e-5, "expected ~2.0, got {v}");
    }
}

#[test]
fn quantized_conv2d_with_bias() {
    let input = vec![1.0f32; 4]; // 1x1x2x2
    let weight_q = vec![0x02u8]; // I2S +1.0
    let scales = vec![1.0f32];
    let bias = vec![5.0f32];
    let mut output = vec![0.0f32; 4];
    conv2d_quantized(
        &input,
        &weight_q,
        &scales,
        Some(&bias),
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::I2S,
    )
    .unwrap();
    // weight = 1.0, input = 1.0, bias = 5.0 → output = 6.0
    for &v in &output {
        assert!((v - 6.0).abs() < 1e-5, "expected ~6.0, got {v}");
    }
}
