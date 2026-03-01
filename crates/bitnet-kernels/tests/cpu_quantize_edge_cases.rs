//! Edge-case tests for CPU quantization operations.
//!
//! Tests cover symmetric i8 quantize/dequantize, asymmetric u8,
//! ternary, binary quantization, and error measurement.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_asymmetric_u8, dequantize_symmetric_i8,
    quantize_asymmetric_u8, quantize_binary, quantize_symmetric_i8, quantize_ternary,
};

// ── Symmetric i8 quantization ────────────────────────────────────────

#[test]
fn symmetric_i8_roundtrip_8bit() {
    let input = vec![1.0, -1.0, 0.5, -0.5, 0.0];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);
    assert_eq!(dequantized.len(), input.len());
    for (orig, deq) in input.iter().zip(dequantized.iter()) {
        assert!((orig - deq).abs() < 0.02, "Roundtrip error too large: {orig} → {deq}");
    }
}

#[test]
fn symmetric_i8_zeros() {
    let input = vec![0.0, 0.0, 0.0];
    let (quantized, _scale) = quantize_symmetric_i8(&input, 8);
    for q in &quantized {
        assert_eq!(*q, 0);
    }
}

#[test]
fn symmetric_i8_scale_reflects_max() {
    let input = vec![0.0, 2.0, -2.0]; // max abs = 2.0
    let (_quantized, scale) = quantize_symmetric_i8(&input, 8);
    assert!(scale > 0.0, "Scale should be positive");
}

#[test]
fn symmetric_i8_4bit() {
    let input = vec![1.0, -1.0, 0.0, 0.5];
    let (quantized, scale) = quantize_symmetric_i8(&input, 4);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);
    // 4-bit has more quantization error but should still be close
    for (orig, deq) in input.iter().zip(dequantized.iter()) {
        assert!((orig - deq).abs() < 0.2, "4-bit roundtrip error: {orig} → {deq}");
    }
}

// ── Asymmetric u8 quantization ───────────────────────────────────────

#[test]
fn asymmetric_u8_roundtrip() {
    let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
    let dequantized = dequantize_asymmetric_u8(&quantized, scale, zero_point);
    for (orig, deq) in input.iter().zip(dequantized.iter()) {
        assert!((orig - deq).abs() < 0.02, "Asymmetric roundtrip error: {orig} → {deq}");
    }
}

#[test]
fn asymmetric_u8_negative_values() {
    let input = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
    let dequantized = dequantize_asymmetric_u8(&quantized, scale, zero_point);
    for (orig, deq) in input.iter().zip(dequantized.iter()) {
        assert!((orig - deq).abs() < 0.02, "Roundtrip error: {orig} → {deq}");
    }
    assert!(scale > 0.0);
}

// ── Ternary quantization ─────────────────────────────────────────────

#[test]
fn ternary_basic() {
    let input = vec![1.0, -1.0, 0.01, -0.01, 5.0];
    let result = quantize_ternary(&input, 0.5);
    assert_eq!(result.len(), 5);
    assert_eq!(result[0], 1); // > threshold
    assert_eq!(result[1], -1); // < -threshold
    assert_eq!(result[2], 0); // within threshold
    assert_eq!(result[3], 0); // within threshold
    assert_eq!(result[4], 1); // > threshold
}

#[test]
fn ternary_all_zero() {
    let input = vec![0.0, 0.0, 0.0];
    let result = quantize_ternary(&input, 0.1);
    assert!(result.iter().all(|&v| v == 0));
}

#[test]
fn ternary_high_threshold() {
    let input = vec![0.5, -0.5, 1.0, -1.0];
    let result = quantize_ternary(&input, 100.0);
    // Everything within threshold → all zeros
    assert!(result.iter().all(|&v| v == 0));
}

// ── Binary quantization ──────────────────────────────────────────────

#[test]
fn binary_basic() {
    let input = vec![1.0, -1.0, 0.5, -0.5, 0.0];
    let result = quantize_binary(&input);
    assert_eq!(result[0], 1);
    assert_eq!(result[1], -1);
    assert_eq!(result[2], 1);
    assert_eq!(result[3], -1);
}

#[test]
fn binary_all_positive() {
    let input = vec![0.1, 1.0, 100.0];
    let result = quantize_binary(&input);
    assert!(result.iter().all(|&v| v == 1));
}

#[test]
fn binary_all_negative() {
    let input = vec![-0.1, -1.0, -100.0];
    let result = quantize_binary(&input);
    assert!(result.iter().all(|&v| v == -1));
}

// ── Quantization error ───────────────────────────────────────────────

#[test]
fn quantization_error_zero_for_identical() {
    let a = vec![1.0, 2.0, 3.0];
    let error = compute_quantization_error(&a, &a);
    assert!((error.mse - 0.0).abs() < 1e-10);
    assert!((error.max_abs_error - 0.0).abs() < 1e-10);
}

#[test]
fn quantization_error_nonzero() {
    let original = vec![1.0, 2.0, 3.0];
    let quantized = vec![1.1, 1.9, 3.2];
    let error = compute_quantization_error(&original, &quantized);
    assert!(error.mse > 0.0);
    assert!(error.max_abs_error > 0.0);
    assert!((error.max_abs_error - 0.2).abs() < 0.01);
}

#[test]
fn quantization_error_snr_finite() {
    let original = vec![1.0, 2.0, 3.0, 4.0];
    let quantized = vec![1.1, 2.1, 2.9, 3.9];
    let error = compute_quantization_error(&original, &quantized);
    assert!(error.snr.is_finite());
    assert!(error.snr > 0.0, "SNR should be positive for small errors");
}
