use bitnet_common::QuantizationType;
use bitnet_scalar::{matmul_i2s, quantize};

#[test]
fn matmul_i2s_identity_preserves_input() {
    let a = vec![1i8, 2, 3, 4];
    let b = vec![1u8, 0, 0, 1];
    let mut c = vec![0.0f32; 4];
    matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();
    assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn quantize_clears_existing_output_bits() {
    let input = vec![0.0f32; 8];
    let mut output = vec![0xFFu8; 2];
    let mut scales = vec![0.0f32; 1];
    quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    assert_eq!(output, vec![0u8; 2]);
}

#[test]
fn quantize_tl1_handles_non_multiple_of_four() {
    let input = vec![1.0f32; 5];
    let mut output = vec![0u8; input.len().div_ceil(4)];
    let mut scales = vec![0.0f32; input.len().div_ceil(64)];
    quantize(&input, &mut output, &mut scales, QuantizationType::TL1).unwrap();
    assert!(scales[0] > 0.0);
}
