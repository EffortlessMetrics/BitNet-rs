//! Edge-case tests for the convolution module.
//!
//! Tests cover `conv2d`, `conv2d_quantized`, `Conv2DParams`, dimension validation,
//! padding, stride, dilation, bias, batch processing, and quantization types (I2S, TL1, TL2).

use bitnet_common::QuantizationType;
use bitnet_kernels::convolution::{Conv2DParams, conv2d, conv2d_quantized};

// ── Conv2DParams ─────────────────────────────────────────────────────────────

#[test]
fn conv2d_params_default() {
    let p = Conv2DParams::default();
    assert_eq!(p.stride, (1, 1));
    assert_eq!(p.padding, (0, 0));
    assert_eq!(p.dilation, (1, 1));
}

#[test]
fn conv2d_params_clone_debug() {
    let p = Conv2DParams { stride: (2, 2), padding: (1, 1), dilation: (1, 1) };
    let p2 = p;
    assert_eq!(p2.stride, (2, 2));
    let dbg = format!("{:?}", p);
    assert!(dbg.contains("Conv2DParams"));
}

// ── conv2d: basic correctness ────────────────────────────────────────────────

#[test]
fn conv2d_identity_1x1_kernel() {
    // 1x1 kernel with weight=1 should copy input values
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![1.0]; // 1x1x1x1
    let mut output = vec![0.0; 4]; // 1x1x2x2
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn conv2d_identity_1x1_with_scaling() {
    // 1x1 kernel with weight=2 should double input
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![2.0]; // 1x1x1x1
    let mut output = vec![0.0; 4]; // 1x1x2x2
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn conv2d_3x3_single_pixel_output() {
    // 3x3 input with 3x3 kernel → 1x1 output (sum of element-wise products)
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect(); // 1x1x3x3
    let weight = vec![1.0; 9]; // 1x1x3x3 (all ones)
    let mut output = vec![0.0; 1]; // 1x1x1x1
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 3, 3), (1, 1, 3, 3))
        .unwrap();
    // Sum of 1..=9 = 45
    assert_eq!(output[0], 45.0);
}

// ── conv2d: bias ─────────────────────────────────────────────────────────────

#[test]
fn conv2d_with_bias() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![1.0]; // 1x1x1x1
    let bias = vec![10.0]; // bias for 1 output channel
    let mut output = vec![0.0; 4]; // 1x1x2x2
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
fn conv2d_with_multi_channel_bias() {
    // 2 output channels, each with different bias
    let input = vec![1.0; 4]; // 1x1x2x2
    let weight = vec![1.0, 1.0]; // 2x1x1x1
    let bias = vec![10.0, 20.0]; // bias for 2 channels
    let mut output = vec![0.0; 8]; // 1x2x2x2
    conv2d(
        &input,
        &weight,
        Some(&bias),
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (2, 1, 1, 1),
    )
    .unwrap();
    // Channel 0: 1.0 + 10.0 = 11.0 for each pixel
    // Channel 1: 1.0 + 20.0 = 21.0 for each pixel
    assert_eq!(&output[..4], &[11.0, 11.0, 11.0, 11.0]);
    assert_eq!(&output[4..], &[21.0, 21.0, 21.0, 21.0]);
}

// ── conv2d: stride ───────────────────────────────────────────────────────────

#[test]
fn conv2d_stride_2() {
    // 4x4 input with 1x1 kernel, stride 2 → 2x2 output
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect(); // 1x1x4x4
    let weight = vec![1.0]; // 1x1x1x1
    let mut output = vec![0.0; 4]; // 1x1x2x2
    let params = Conv2DParams { stride: (2, 2), ..Default::default() };
    conv2d(&input, &weight, None, &mut output, params, (1, 1, 4, 4), (1, 1, 1, 1)).unwrap();
    // Picks positions (0,0), (0,2), (2,0), (2,2) = 1, 3, 9, 11
    assert_eq!(output, vec![1.0, 3.0, 9.0, 11.0]);
}

// ── conv2d: padding ──────────────────────────────────────────────────────────

#[test]
fn conv2d_padding_same_3x3() {
    // 3x3 input, 3x3 kernel, padding=1 → 3x3 output (same size)
    let input = vec![1.0; 9]; // 1x1x3x3 all ones
    let weight = vec![1.0; 9]; // 1x1x3x3 all ones
    let mut output = vec![0.0; 9]; // 1x1x3x3
    let params = Conv2DParams { padding: (1, 1), ..Default::default() };
    conv2d(&input, &weight, None, &mut output, params, (1, 1, 3, 3), (1, 1, 3, 3)).unwrap();
    // Center pixel sees all 9 ones → 9.0
    // Edge pixels see fewer → check center
    assert_eq!(output[4], 9.0); // center (1,1) sees full 3x3
}

// ── conv2d: dilation ─────────────────────────────────────────────────────────

#[test]
fn conv2d_dilation_2() {
    // 5x5 input, 3x3 kernel, dilation=2 → effective 5x5 receptive field → 1x1 output
    let input = vec![1.0; 25]; // 1x1x5x5
    let weight = vec![1.0; 9]; // 1x1x3x3
    let mut output = vec![0.0; 1]; // 1x1x1x1
    let params = Conv2DParams { dilation: (2, 2), ..Default::default() };
    conv2d(&input, &weight, None, &mut output, params, (1, 1, 5, 5), (1, 1, 3, 3)).unwrap();
    // With dilation=2 on 5x5 all-ones, the 3x3 kernel picks 9 positions → 9.0
    assert_eq!(output[0], 9.0);
}

// ── conv2d: multiple channels ────────────────────────────────────────────────

#[test]
fn conv2d_multi_input_channels() {
    // 2 input channels, 1 output channel, 1x1 kernel
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 1x2x2x2
    let weight = vec![1.0, 1.0]; // 1x2x1x1 (sum of both channels)
    let mut output = vec![0.0; 4]; // 1x1x2x2
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 2, 2, 2), (1, 2, 1, 1))
        .unwrap();
    // output[0] = input[0]*1 + input[4]*1 = 1+5 = 6
    // output[1] = input[1]*1 + input[5]*1 = 2+6 = 8
    assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
}

// ── conv2d: batch processing ─────────────────────────────────────────────────

#[test]
fn conv2d_batch_size_2() {
    // Batch of 2, each 1x2x2 input with 1x1x1x1 kernel
    let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // 2x1x2x2
    let weight = vec![1.0]; // 1x1x1x1
    let mut output = vec![0.0; 8]; // 2x1x2x2
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (2, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(&output[..4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&output[4..], &[10.0, 20.0, 30.0, 40.0]);
}

// ── conv2d: zero weight ──────────────────────────────────────────────────────

#[test]
fn conv2d_zero_kernel() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![0.0]; // 1x1x1x1
    let mut output = vec![0.0; 4]; // 1x1x2x2
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
}

// ── conv2d: validation errors ────────────────────────────────────────────────

#[test]
fn conv2d_channel_mismatch() {
    let input = vec![1.0; 4]; // 1x1x2x2
    let weight = vec![1.0; 4]; // 1x2x1x2 (2 input channels in weight, but input has 1)
    let mut output = vec![0.0; 4];
    let result = conv2d(
        &input,
        &weight,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 2, 1, 2),
    );
    assert!(result.is_err());
}

#[test]
fn conv2d_input_size_mismatch() {
    let input = vec![1.0; 3]; // wrong size
    let weight = vec![1.0]; // 1x1x1x1
    let mut output = vec![0.0; 4];
    let result = conv2d(
        &input,
        &weight,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
    );
    assert!(result.is_err());
}

#[test]
fn conv2d_weight_size_mismatch() {
    let input = vec![1.0; 4]; // 1x1x2x2
    let weight = vec![1.0; 3]; // wrong size
    let mut output = vec![0.0; 4];
    let result = conv2d(
        &input,
        &weight,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
    );
    assert!(result.is_err());
}

#[test]
fn conv2d_output_size_mismatch() {
    let input = vec![1.0; 4]; // 1x1x2x2
    let weight = vec![1.0]; // 1x1x1x1
    let mut output = vec![0.0; 3]; // wrong size (should be 4)
    let result = conv2d(
        &input,
        &weight,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
    );
    assert!(result.is_err());
}

#[test]
fn conv2d_bias_size_mismatch() {
    let input = vec![1.0; 4]; // 1x1x2x2
    let weight = vec![1.0]; // 1x1x1x1
    let bias = vec![1.0, 2.0]; // 2 bias values for 1 output channel
    let mut output = vec![0.0; 4];
    let result = conv2d(
        &input,
        &weight,
        Some(&bias),
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
    );
    assert!(result.is_err());
}

// ── conv2d_quantized: I2S ────────────────────────────────────────────────────

#[test]
fn conv2d_quantized_i2s_basic() {
    // 1x1 quantized kernel, I2S format
    // Packed: 4 values per byte. Single value at bit offset 0.
    // 0x02 = binary 10 → dequantized to 1.0, times scale
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight_quantized = vec![0x02u8]; // 1 element packed (value = 1.0 * scale)
    let weight_scales = vec![1.0];
    let mut output = vec![0.0; 4]; // 1x1x2x2
    conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::I2S,
    )
    .unwrap();
    // weight = 1.0 * 1.0 = 1.0, so output = input
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn conv2d_quantized_i2s_negative() {
    // 0x01 = binary 01 → dequantized to -1.0
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight_quantized = vec![0x01u8]; // value = -1.0 * scale
    let weight_scales = vec![1.0];
    let mut output = vec![0.0; 4]; // 1x1x2x2
    conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::I2S,
    )
    .unwrap();
    assert_eq!(output, vec![-1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn conv2d_quantized_i2s_with_scale() {
    // 0x02 → 1.0, scale=0.5 → effective weight 0.5
    let input = vec![2.0, 4.0, 6.0, 8.0]; // 1x1x2x2
    let weight_quantized = vec![0x02u8];
    let weight_scales = vec![0.5];
    let mut output = vec![0.0; 4];
    conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::I2S,
    )
    .unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

// ── conv2d_quantized: TL1 ───────────────────────────────────────────────────

#[test]
fn conv2d_quantized_tl1_midpoint() {
    // TL1: byte 128 → (128-128)/127 = 0.0
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight_quantized = vec![128u8]; // → 0.0 * scale
    let weight_scales = vec![1.0];
    let mut output = vec![0.0; 4];
    conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::TL1,
    )
    .unwrap();
    // All zeros since weight is 0
    assert!(output.iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn conv2d_quantized_tl1_max() {
    // TL1: byte 255 → (255-128)/127 = 1.0
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight_quantized = vec![255u8]; // → 1.0 * scale
    let weight_scales = vec![1.0];
    let mut output = vec![0.0; 4];
    conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::TL1,
    )
    .unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

// ── conv2d_quantized: TL2 ───────────────────────────────────────────────────

#[test]
fn conv2d_quantized_tl2_midpoint() {
    // TL2: byte 127 → (127/255)*2 - 1 ≈ -0.00392
    // byte 128 → (128/255)*2 - 1 ≈ 0.00392
    // Both are near zero
    let input = vec![1.0; 4]; // 1x1x2x2
    let weight_quantized = vec![128u8];
    let weight_scales = vec![1.0];
    let mut output = vec![0.0; 4];
    conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::TL2,
    )
    .unwrap();
    // Close to zero but not exactly zero
    for v in &output {
        assert!(v.abs() < 0.01);
    }
}

#[test]
fn conv2d_quantized_tl2_max() {
    // TL2: byte 255 → (255/255)*2 - 1 = 1.0
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight_quantized = vec![255u8];
    let weight_scales = vec![1.0];
    let mut output = vec![0.0; 4];
    conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::TL2,
    )
    .unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

// ── conv2d_quantized: validation ─────────────────────────────────────────────

#[test]
fn conv2d_quantized_channel_mismatch() {
    let input = vec![1.0; 4]; // 1x1x2x2
    let weight_quantized = vec![0u8; 4]; // 1x2x... (2 input channels)
    let weight_scales = vec![1.0];
    let mut output = vec![0.0; 4];
    let result = conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 2, 1, 2), // 2 input channels in weight != 1 in input
        QuantizationType::TL1,
    );
    assert!(result.is_err());
}

#[test]
fn conv2d_quantized_scales_mismatch() {
    let input = vec![1.0; 4]; // 1x1x2x2
    let weight_quantized = vec![128u8];
    let weight_scales = vec![1.0, 2.0]; // 2 scales for 1 output channel
    let mut output = vec![0.0; 4];
    let result = conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::TL1,
    );
    assert!(result.is_err());
}

#[test]
fn conv2d_quantized_i2s_weight_size_mismatch() {
    // I2S packs 4 elements per byte. For 1x1x1x1 kernel: 1 element → div_ceil(1,4) = 1 byte
    let input = vec![1.0; 4]; // 1x1x2x2
    let weight_quantized = vec![0u8; 5]; // too many bytes
    let weight_scales = vec![1.0];
    let mut output = vec![0.0; 4];
    let result = conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        None,
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::I2S,
    );
    assert!(result.is_err());
}

// ── conv2d_quantized: with bias ──────────────────────────────────────────────

#[test]
fn conv2d_quantized_with_bias() {
    // TL1: byte 255 → 1.0, + bias 5.0
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight_quantized = vec![255u8]; // → 1.0
    let weight_scales = vec![1.0];
    let bias = vec![5.0];
    let mut output = vec![0.0; 4];
    conv2d_quantized(
        &input,
        &weight_quantized,
        &weight_scales,
        Some(&bias),
        &mut output,
        Conv2DParams::default(),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        QuantizationType::TL1,
    )
    .unwrap();
    assert_eq!(output, vec![6.0, 7.0, 8.0, 9.0]);
}

// ── conv2d: negative values ──────────────────────────────────────────────────

#[test]
fn conv2d_negative_input() {
    let input = vec![-1.0, -2.0, -3.0, -4.0]; // 1x1x2x2
    let weight = vec![1.0]; // 1x1x1x1
    let mut output = vec![0.0; 4];
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, vec![-1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn conv2d_negative_weight() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![-1.0]; // 1x1x1x1
    let mut output = vec![0.0; 4];
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, vec![-1.0, -2.0, -3.0, -4.0]);
}

// ── conv2d: 1x1 input ───────────────────────────────────────────────────────

#[test]
fn conv2d_1x1_input_1x1_kernel() {
    let input = vec![5.0]; // 1x1x1x1
    let weight = vec![3.0]; // 1x1x1x1
    let mut output = vec![0.0; 1];
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 1, 1), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output[0], 15.0);
}

// ── conv2d: multiple output channels ─────────────────────────────────────────

#[test]
fn conv2d_multiple_output_channels() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
    let weight = vec![1.0, 2.0]; // 2x1x1x1
    let mut output = vec![0.0; 8]; // 1x2x2x2
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (2, 1, 1, 1))
        .unwrap();
    // Channel 0: weight=1 → identity
    assert_eq!(&output[..4], &[1.0, 2.0, 3.0, 4.0]);
    // Channel 1: weight=2 → doubled
    assert_eq!(&output[4..], &[2.0, 4.0, 6.0, 8.0]);
}

// ── conv2d: all zeros ────────────────────────────────────────────────────────

#[test]
fn conv2d_all_zeros_input() {
    let input = vec![0.0; 4]; // 1x1x2x2
    let weight = vec![5.0]; // 1x1x1x1
    let mut output = vec![0.0; 4];
    conv2d(&input, &weight, None, &mut output, Conv2DParams::default(), (1, 1, 2, 2), (1, 1, 1, 1))
        .unwrap();
    assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
}
