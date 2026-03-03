//! BDD Wave 19 — Quantization operation tests.
//!
//! Covers I2_S pack/unpack, symmetric/asymmetric quantization,
//! ternary encoding, QK256-style block dequantization, scale extraction,
//! and round-trip error measurement.

use bitnet_kernels::cpu::dequant::{
    dequant_i2s_block, dequant_i2s_row, dequant_ternary, pack_ternary,
};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_asymmetric_u8, dequantize_symmetric_i8,
    quantize_asymmetric_u8, quantize_binary, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::quantized_matmul::pack_i2s;

const TOL: f32 = 1e-5;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
    }
}

// ── I2_S Pack / Unpack ─────────────────────────────────────────────

#[test]
fn given_all_zeros_when_pack_i2s_then_byte_is_zero() {
    let byte = pack_i2s([0, 0, 0, 0]);
    assert_eq!(byte, 0x00);
}

#[test]
fn given_all_plus_ones_when_pack_i2s_then_correct_encoding() {
    // +1 → 0b01 at each 2-bit slot: 0b01_01_01_01 = 0x55
    let byte = pack_i2s([1, 1, 1, 1]);
    assert_eq!(byte, 0x55);
}

#[test]
fn given_all_minus_ones_when_pack_i2s_then_correct_encoding() {
    // -1 → 0b11 at each slot: 0b11_11_11_11 = 0xFF
    let byte = pack_i2s([-1, -1, -1, -1]);
    assert_eq!(byte, 0xFF);
}

#[test]
fn given_mixed_ternary_when_pack_i2s_then_round_trips_via_dequant() {
    let vals: [i8; 4] = [1, -1, 0, 1];
    let byte = pack_i2s(vals);
    // Dequant with scale 1.0 should recover the same pattern.
    let recovered = dequant_i2s_block(&[byte], 1.0, 4).unwrap();
    let expected: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
    approx_eq(&recovered, &expected, TOL);
}

#[test]
fn given_alternating_pattern_when_pack_i2s_then_bits_correct() {
    let byte = pack_i2s([1, 0, -1, 0]);
    // slot0=01, slot1=00, slot2=11, slot3=00
    // byte = 0b00_11_00_01 = 0x31 (bit order: slot3|slot2|slot1|slot0)
    // slot0 at bits[1:0]=01, slot1 at bits[3:2]=00, slot2 at bits[5:4]=11, slot3 at bits[7:6]=00
    let expected = 0b00_11_00_01u8;
    assert_eq!(byte, expected, "got {byte:#010b}, expected {expected:#010b}");
}

// ── I2_S Block Dequantization ──────────────────────────────────────

#[test]
fn given_single_block_when_dequant_i2s_then_scale_applied() {
    let packed = [pack_i2s([1, -1, 1, 0])];
    let scale = 2.5;
    let result = dequant_i2s_block(&packed, scale, 4).unwrap();
    approx_eq(&result, &[2.5, -2.5, 2.5, 0.0], TOL);
}

#[test]
fn given_zero_scale_when_dequant_i2s_then_all_zeros() {
    let packed = [pack_i2s([1, -1, 1, -1])];
    let result = dequant_i2s_block(&packed, 0.0, 4).unwrap();
    approx_eq(&result, &[0.0, 0.0, 0.0, 0.0], TOL);
}

#[test]
fn given_partial_block_when_dequant_i2s_then_only_requested_elements() {
    let packed = [pack_i2s([1, -1, 0, 1])];
    let result = dequant_i2s_block(&packed, 1.0, 3).unwrap();
    assert_eq!(result.len(), 3);
    approx_eq(&result, &[1.0, -1.0, 0.0], TOL);
}

#[test]
fn given_insufficient_bytes_when_dequant_i2s_then_error() {
    // block_size=8 needs 2 bytes, provide only 1
    let result = dequant_i2s_block(&[0x55], 1.0, 8);
    assert!(result.is_err());
}

// ── I2_S Row Dequantization with Per-Block Scales ──────────────────

#[test]
fn given_two_blocks_when_dequant_row_then_each_block_gets_own_scale() {
    let b0 = pack_i2s([1, 1, 1, 1]);
    let b1 = pack_i2s([-1, -1, -1, -1]);
    let packed = [b0, b1];
    let scales = [2.0, 3.0];
    let result = dequant_i2s_row(&packed, &scales, 4).unwrap();
    approx_eq(&result, &[2.0, 2.0, 2.0, 2.0, -3.0, -3.0, -3.0, -3.0], TOL);
}

#[test]
fn given_insufficient_scales_when_dequant_row_then_error() {
    let packed = [0x55, 0x55]; // 8 values, block_size=4 → needs 2 scales
    let result = dequant_i2s_row(&packed, &[1.0], 4);
    assert!(result.is_err());
}

// ── Ternary Pack / Dequant ─────────────────────────────────────────

#[test]
fn given_ternary_values_when_pack_then_dequant_recovers_signs() {
    let values = vec![0.5, -0.3, 0.01, 0.8];
    let threshold = 0.1;
    let (packed, scale) = pack_ternary(&values, threshold);
    let recovered = dequant_ternary(&packed, scale);
    // 0.5 → +1, -0.3 → -1, 0.01 → 0, 0.8 → +1
    assert!(recovered[0] > 0.0);
    assert!(recovered[1] < 0.0);
    assert_eq!(recovered[2], 0.0);
    assert!(recovered[3] > 0.0);
}

#[test]
fn given_all_below_threshold_when_pack_ternary_then_all_zero_codes() {
    let values = vec![0.01, -0.02, 0.005, -0.001];
    let (packed, _scale) = pack_ternary(&values, 0.1);
    let recovered = dequant_ternary(&packed, 0.0);
    for v in &recovered[..4] {
        assert_eq!(*v, 0.0);
    }
}

// ── Symmetric i8 Quantization ──────────────────────────────────────

#[test]
fn given_uniform_input_when_quantize_symmetric_then_round_trip_within_tolerance() {
    let input = vec![1.0, -1.0, 0.5, -0.5, 0.0];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let recovered = dequantize_symmetric_i8(&quantized, scale);
    approx_eq(&input, &recovered, 0.01);
}

#[test]
fn given_all_zeros_when_quantize_symmetric_then_scale_is_zero() {
    let input = vec![0.0; 8];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    assert_eq!(scale, 0.0);
    assert!(quantized.iter().all(|&v| v == 0));
}

#[test]
fn given_2bit_when_quantize_symmetric_then_values_in_range() {
    let input = vec![1.0, -1.0, 0.3, -0.7];
    let (quantized, _scale) = quantize_symmetric_i8(&input, 2);
    // 2-bit range: [-1, 1]
    for &v in &quantized {
        assert!((-1..=1).contains(&v), "2-bit value {v} out of range");
    }
}

#[test]
fn given_large_dynamic_range_when_quantize_8bit_then_error_bounded() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let recovered = dequantize_symmetric_i8(&quantized, scale);
    let err = compute_quantization_error(&input, &recovered);
    assert!(err.max_abs_error < 0.01, "max error too large: {}", err.max_abs_error);
}

// ── Asymmetric u8 Quantization ─────────────────────────────────────

#[test]
fn given_positive_range_when_quantize_asymmetric_then_round_trip_accurate() {
    let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let (quantized, scale, zp) = quantize_asymmetric_u8(&input);
    let recovered = dequantize_asymmetric_u8(&quantized, scale, zp);
    approx_eq(&input, &recovered, 0.005);
}

#[test]
fn given_constant_input_when_quantize_asymmetric_then_scale_zero() {
    let input = vec![3.14; 4];
    let (_quantized, scale, _zp) = quantize_asymmetric_u8(&input);
    assert_eq!(scale, 0.0);
}

#[test]
fn given_negative_range_when_quantize_asymmetric_then_zero_point_positive() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let (_quantized, _scale, zp) = quantize_asymmetric_u8(&input);
    assert!(zp >= 0, "zero point should compensate for negative min");
}

// ── Ternary / Binary Quantization ──────────────────────────────────

#[test]
fn given_mixed_input_when_quantize_ternary_then_correct_mapping() {
    let input = vec![0.5, -0.3, 0.02, -0.8, 0.0];
    let result = quantize_ternary(&input, 0.1);
    assert_eq!(result, vec![1, -1, 0, -1, 0]);
}

#[test]
fn given_positive_input_when_quantize_binary_then_all_plus_one() {
    let input = vec![0.1, 0.5, 1.0, 0.001];
    let result = quantize_binary(&input);
    assert!(result.iter().all(|&v| v == 1));
}

#[test]
fn given_negative_input_when_quantize_binary_then_all_minus_one() {
    let input = vec![-0.1, -0.5, -1.0, -0.001];
    let result = quantize_binary(&input);
    assert!(result.iter().all(|&v| v == -1));
}

#[test]
fn given_zero_when_quantize_binary_then_maps_to_plus_one() {
    let result = quantize_binary(&[0.0]);
    assert_eq!(result, vec![1]);
}

// ── Quantization Error Metrics ─────────────────────────────────────

#[test]
fn given_identical_signals_when_compute_error_then_zero_mse_and_infinite_snr() {
    let signal = vec![1.0, 2.0, 3.0, 4.0];
    let err = compute_quantization_error(&signal, &signal);
    assert_eq!(err.mse, 0.0);
    assert_eq!(err.max_abs_error, 0.0);
    assert!(err.snr.is_infinite() && err.snr > 0.0);
}

#[test]
fn given_known_offset_when_compute_error_then_mse_correct() {
    let original = vec![1.0, 2.0, 3.0, 4.0];
    let quantized = vec![1.1, 2.1, 3.1, 4.1];
    let err = compute_quantization_error(&original, &quantized);
    assert!((err.mse - 0.01).abs() < 1e-4, "expected MSE ≈ 0.01, got {}", err.mse);
    assert!((err.max_abs_error - 0.1).abs() < 1e-4);
}

#[test]
fn given_8bit_round_trip_when_compute_error_then_snr_above_threshold() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
    let (q, scale) = quantize_symmetric_i8(&input, 8);
    let recovered = dequantize_symmetric_i8(&q, scale);
    let err = compute_quantization_error(&input, &recovered);
    // 8-bit quantization should yield reasonable SNR
    assert!(err.snr > 30.0, "SNR too low: {} dB", err.snr);
}
