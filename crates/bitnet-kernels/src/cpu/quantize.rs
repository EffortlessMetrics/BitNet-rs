//! CPU quantization utility functions.
//!
//! Provides symmetric/asymmetric integer quantization, ternary and
//! binary quantization, plus error-measurement helpers for quality
//! assessment.  All routines operate on contiguous `f32` slices and
//! are suitable for pre/post-processing around the low-bit kernels.

// ── Types ──────────────────────────────────────────────────────────

/// Quantization error metrics between an original and reconstructed signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationError {
    /// Mean squared error.
    pub mse: f32,
    /// Maximum absolute element-wise error.
    pub max_abs_error: f32,
    /// Signal-to-noise ratio in dB (`10 · log10(signal_power / noise_power)`).
    /// Returns [`f32::INFINITY`] when the noise is zero and [`f32::NEG_INFINITY`]
    /// when the signal power is zero.
    pub snr: f32,
}

// ── Symmetric i8 quantization ──────────────────────────────────────

/// Symmetric signed quantization to `bits`-bit range.
///
/// Maps `[-abs_max, abs_max]` linearly to `[-(2^(bits-1)-1), 2^(bits-1)-1]`.
/// Returns `(quantized, scale)` where `scale = abs_max / qmax`.
///
/// An all-zero input produces scale = 0 and an all-zero output.
#[inline]
pub fn quantize_symmetric_i8(input: &[f32], bits: u8) -> (Vec<i8>, f32) {
    assert!((2..=8).contains(&bits), "bits must be in [2, 8]");

    let qmax = ((1i32 << (bits - 1)) - 1) as f32; // e.g. 127 for 8-bit
    let abs_max = input.iter().copied().fold(0.0_f32, |m, v| m.max(v.abs()));

    if abs_max == 0.0 {
        return (vec![0i8; input.len()], 0.0);
    }

    let scale = abs_max / qmax;
    let inv_scale = 1.0 / scale;

    let quantized =
        input.iter().map(|&v| (v * inv_scale).round().clamp(-qmax, qmax) as i8).collect();

    (quantized, scale)
}

/// Dequantize symmetric i8 values: `output[i] = input[i] as f32 * scale`.
#[inline]
pub fn dequantize_symmetric_i8(input: &[i8], scale: f32) -> Vec<f32> {
    input.iter().map(|&v| v as f32 * scale).collect()
}

// ── Asymmetric u8 quantization ─────────────────────────────────────

/// Asymmetric unsigned 8-bit quantization.
///
/// Maps `[min, max]` to `[0, 255]`.  Returns `(quantized, scale, zero_point)`
/// so that `real ≈ (quantized as i32 - zero_point) as f32 * scale`.
///
/// A constant input produces scale = 0, zero_point = 0, and all-zero output.
#[inline]
pub fn quantize_asymmetric_u8(input: &[f32]) -> (Vec<u8>, f32, i32) {
    let (mut min_val, mut max_val) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in input {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    if input.is_empty() || max_val <= min_val {
        return (vec![0u8; input.len()], 0.0, 0);
    }

    let scale = (max_val - min_val) / 255.0;
    let inv_scale = 1.0 / scale;
    let zero_point = (-min_val * inv_scale).round() as i32;

    let quantized = input
        .iter()
        .map(|&v| (v * inv_scale + zero_point as f32).round().clamp(0.0, 255.0) as u8)
        .collect();

    (quantized, scale, zero_point)
}

/// Dequantize asymmetric u8 values:
/// `output[i] = (input[i] as i32 - zero_point) as f32 * scale`.
#[inline]
pub fn dequantize_asymmetric_u8(input: &[u8], scale: f32, zero_point: i32) -> Vec<f32> {
    input.iter().map(|&v| (v as i32 - zero_point) as f32 * scale).collect()
}

// ── Ternary quantization ───────────────────────────────────────────

/// Ternary quantization: maps each element to −1, 0, or +1.
///
/// Values with `|v| ≤ threshold` become 0; positive above threshold
/// become +1, negative below −threshold become −1.
#[inline]
pub fn quantize_ternary(input: &[f32], threshold: f32) -> Vec<i8> {
    assert!(threshold >= 0.0, "threshold must be non-negative");
    input
        .iter()
        .map(|&v| {
            if v > threshold {
                1
            } else if v < -threshold {
                -1
            } else {
                0
            }
        })
        .collect()
}

// ── Binary quantization ────────────────────────────────────────────

/// Binary quantization: maps each element to −1 or +1.
///
/// Non-negative values become +1, negative values become −1.
#[inline]
pub fn quantize_binary(input: &[f32]) -> Vec<i8> {
    input.iter().map(|&v| if v >= 0.0 { 1 } else { -1 }).collect()
}

// ── Error measurement ──────────────────────────────────────────────

/// Compute quantization error between `original` and `quantized` signals.
///
/// # Panics
///
/// Panics if the two slices have different lengths or are empty.
pub fn compute_quantization_error(original: &[f32], quantized: &[f32]) -> QuantizationError {
    assert_eq!(original.len(), quantized.len(), "length mismatch");
    assert!(!original.is_empty(), "slices must not be empty");

    let n = original.len() as f32;
    let mut sum_sq_err = 0.0_f64;
    let mut max_abs = 0.0_f32;
    let mut sum_sq_signal = 0.0_f64;

    for (&o, &q) in original.iter().zip(quantized.iter()) {
        let err = (o - q) as f64;
        sum_sq_err += err * err;
        let ae = (o - q).abs();
        if ae > max_abs {
            max_abs = ae;
        }
        sum_sq_signal += (o as f64) * (o as f64);
    }

    let mse = (sum_sq_err / n as f64) as f32;

    let snr = if sum_sq_err == 0.0 {
        f32::INFINITY
    } else if sum_sq_signal == 0.0 {
        f32::NEG_INFINITY
    } else {
        (10.0 * (sum_sq_signal / sum_sq_err).log10()) as f32
    };

    QuantizationError { mse, max_abs_error: max_abs, snr }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    // ── Symmetric i8 roundtrip ─────────────────────────────────

    #[test]
    fn symmetric_i8_roundtrip_8bit() {
        let input = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let (q, scale) = quantize_symmetric_i8(&input, 8);
        let output = dequantize_symmetric_i8(&q, scale);
        for (&orig, &deq) in input.iter().zip(output.iter()) {
            assert!((orig - deq).abs() < 0.01, "orig={orig} deq={deq}");
        }
    }

    #[test]
    fn symmetric_i8_roundtrip_4bit() {
        let input = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let (q, scale) = quantize_symmetric_i8(&input, 4);
        let output = dequantize_symmetric_i8(&q, scale);
        for (&orig, &deq) in input.iter().zip(output.iter()) {
            assert!((orig - deq).abs() < 0.15, "4-bit: orig={orig} deq={deq}");
        }
    }

    #[test]
    fn symmetric_i8_roundtrip_2bit() {
        let input = vec![-1.0, 0.0, 1.0];
        let (q, scale) = quantize_symmetric_i8(&input, 2);
        let output = dequantize_symmetric_i8(&q, scale);
        assert!(approx(output[0], -1.0));
        assert!(approx(output[1], 0.0));
        assert!(approx(output[2], 1.0));
    }

    #[test]
    fn symmetric_i8_all_zeros() {
        let input = vec![0.0; 8];
        let (q, scale) = quantize_symmetric_i8(&input, 8);
        assert_eq!(scale, 0.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn symmetric_i8_single_element() {
        let (q, scale) = quantize_symmetric_i8(&[3.0], 8);
        assert_eq!(q.len(), 1);
        assert_eq!(q[0], 127); // 3.0 maps to qmax
        assert!(approx(scale, 3.0 / 127.0));
    }

    #[test]
    fn symmetric_i8_negative_only() {
        let input = vec![-4.0, -2.0, -1.0];
        let (q, scale) = quantize_symmetric_i8(&input, 8);
        let output = dequantize_symmetric_i8(&q, scale);
        for (&orig, &deq) in input.iter().zip(output.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig} deq={deq}");
        }
    }

    #[test]
    fn symmetric_i8_scale_correctness() {
        let input = vec![-10.0, 5.0, 10.0];
        let (_, scale) = quantize_symmetric_i8(&input, 8);
        assert!(approx(scale, 10.0 / 127.0));
    }

    #[test]
    #[should_panic(expected = "bits must be in [2, 8]")]
    fn symmetric_i8_bits_too_low() {
        quantize_symmetric_i8(&[1.0], 1);
    }

    #[test]
    #[should_panic(expected = "bits must be in [2, 8]")]
    fn symmetric_i8_bits_too_high() {
        quantize_symmetric_i8(&[1.0], 9);
    }

    // ── Asymmetric u8 roundtrip ────────────────────────────────

    #[test]
    fn asymmetric_u8_roundtrip() {
        let input = vec![-1.0, 0.0, 0.5, 1.0, 2.0];
        let (q, scale, zp) = quantize_asymmetric_u8(&input);
        let output = dequantize_asymmetric_u8(&q, scale, zp);
        for (&orig, &deq) in input.iter().zip(output.iter()) {
            assert!((orig - deq).abs() < 0.02, "orig={orig} deq={deq}");
        }
    }

    #[test]
    fn asymmetric_u8_all_positive() {
        let input = vec![10.0, 20.0, 30.0];
        let (q, scale, zp) = quantize_asymmetric_u8(&input);
        let output = dequantize_asymmetric_u8(&q, scale, zp);
        for (&orig, &deq) in input.iter().zip(output.iter()) {
            assert!((orig - deq).abs() < 0.15, "orig={orig} deq={deq}");
        }
    }

    #[test]
    fn asymmetric_u8_constant_input() {
        let input = vec![5.0; 4];
        let (q, scale, zp) = quantize_asymmetric_u8(&input);
        assert_eq!(scale, 0.0);
        assert_eq!(zp, 0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn asymmetric_u8_range_coverage() {
        let input = vec![0.0, 255.0];
        let (q, _scale, _zp) = quantize_asymmetric_u8(&input);
        // min should map to 0, max to 255
        assert_eq!(*q.iter().min().unwrap(), 0);
        assert_eq!(*q.iter().max().unwrap(), 255);
    }

    #[test]
    fn asymmetric_u8_single_element() {
        let input = vec![42.0];
        let (q, scale, zp) = quantize_asymmetric_u8(&input);
        assert_eq!(scale, 0.0);
        assert_eq!(zp, 0);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn asymmetric_u8_negative_range() {
        let input = vec![-10.0, -5.0, -1.0];
        let (q, scale, zp) = quantize_asymmetric_u8(&input);
        let output = dequantize_asymmetric_u8(&q, scale, zp);
        for (&orig, &deq) in input.iter().zip(output.iter()) {
            assert!((orig - deq).abs() < 0.05, "orig={orig} deq={deq}");
        }
    }

    // ── Ternary quantization ───────────────────────────────────

    #[test]
    fn ternary_basic() {
        let input = vec![-2.0, -0.1, 0.0, 0.1, 2.0];
        let q = quantize_ternary(&input, 0.5);
        assert_eq!(q, vec![-1, 0, 0, 0, 1]);
    }

    #[test]
    fn ternary_zero_threshold() {
        let input = vec![-1.0, 0.0, 1.0];
        let q = quantize_ternary(&input, 0.0);
        assert_eq!(q, vec![-1, 0, 1]);
    }

    #[test]
    fn ternary_high_threshold() {
        let input = vec![-0.5, 0.0, 0.5];
        let q = quantize_ternary(&input, 1.0);
        assert_eq!(q, vec![0, 0, 0]);
    }

    #[test]
    fn ternary_all_positive_above_threshold() {
        let input = vec![1.0, 2.0, 3.0];
        let q = quantize_ternary(&input, 0.5);
        assert!(q.iter().all(|&v| v == 1));
    }

    #[test]
    fn ternary_all_negative_below_threshold() {
        let input = vec![-1.0, -2.0, -3.0];
        let q = quantize_ternary(&input, 0.5);
        assert!(q.iter().all(|&v| v == -1));
    }

    #[test]
    #[should_panic(expected = "threshold must be non-negative")]
    fn ternary_negative_threshold() {
        quantize_ternary(&[1.0], -0.1);
    }

    // ── Binary quantization ────────────────────────────────────

    #[test]
    fn binary_basic() {
        let input = vec![-2.0, -0.1, 0.0, 0.1, 2.0];
        let q = quantize_binary(&input);
        assert_eq!(q, vec![-1, -1, 1, 1, 1]);
    }

    #[test]
    fn binary_all_negative() {
        let q = quantize_binary(&[-3.0, -2.0, -1.0]);
        assert!(q.iter().all(|&v| v == -1));
    }

    #[test]
    fn binary_all_non_negative() {
        let q = quantize_binary(&[0.0, 1.0, 2.0]);
        assert!(q.iter().all(|&v| v == 1));
    }

    #[test]
    fn binary_single_zero() {
        assert_eq!(quantize_binary(&[0.0]), vec![1]);
    }

    // ── Quantization error ─────────────────────────────────────

    #[test]
    fn error_perfect_reconstruction() {
        let a = vec![1.0, 2.0, 3.0];
        let err = compute_quantization_error(&a, &a);
        assert!(approx(err.mse, 0.0));
        assert!(approx(err.max_abs_error, 0.0));
        assert_eq!(err.snr, f32::INFINITY);
    }

    #[test]
    fn error_known_values() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let quantized = vec![1.1, 2.0, 2.9, 4.2];
        let err = compute_quantization_error(&original, &quantized);
        // MSE = (0.01 + 0 + 0.01 + 0.04) / 4 = 0.015
        assert!((err.mse - 0.015).abs() < 1e-4);
        assert!((err.max_abs_error - 0.2).abs() < 1e-5);
        assert!(err.snr > 0.0);
    }

    #[test]
    fn error_zero_signal() {
        let original = vec![0.0, 0.0];
        let quantized = vec![0.1, -0.1];
        let err = compute_quantization_error(&original, &quantized);
        assert_eq!(err.snr, f32::NEG_INFINITY);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn error_length_mismatch() {
        compute_quantization_error(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "slices must not be empty")]
    fn error_empty_slices() {
        compute_quantization_error(&[], &[]);
    }

    // ── Roundtrip error measurement ────────────────────────────

    #[test]
    fn symmetric_roundtrip_error_decreases_with_bits() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();

        let (q4, s4) = quantize_symmetric_i8(&input, 4);
        let deq4 = dequantize_symmetric_i8(&q4, s4);
        let err4 = compute_quantization_error(&input, &deq4);

        let (q8, s8) = quantize_symmetric_i8(&input, 8);
        let deq8 = dequantize_symmetric_i8(&q8, s8);
        let err8 = compute_quantization_error(&input, &deq8);

        assert!(
            err8.mse < err4.mse,
            "8-bit MSE ({}) should be less than 4-bit MSE ({})",
            err8.mse,
            err4.mse
        );
    }

    #[test]
    fn asymmetric_roundtrip_error_bounded() {
        let input: Vec<f32> = (0..128).map(|i| i as f32 / 64.0).collect();
        let (q, s, zp) = quantize_asymmetric_u8(&input);
        let deq = dequantize_asymmetric_u8(&q, s, zp);
        let err = compute_quantization_error(&input, &deq);
        // u8 quantization over a 2.0 range → step ≈ 2/255 ≈ 0.008
        assert!(err.max_abs_error < 0.01, "max_abs_error={}", err.max_abs_error);
    }

    // ── Large input ────────────────────────────────────────────

    #[test]
    fn symmetric_large_input() {
        let input: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.01).sin()).collect();
        let (q, scale) = quantize_symmetric_i8(&input, 8);
        assert_eq!(q.len(), 4096);
        assert!(scale > 0.0);
        let deq = dequantize_symmetric_i8(&q, scale);
        let err = compute_quantization_error(&input, &deq);
        assert!(err.mse < 0.001);
    }

    #[test]
    fn asymmetric_large_input() {
        let input: Vec<f32> = (0..4096).map(|i| i as f32 * 0.1).collect();
        let (q, scale, zp) = quantize_asymmetric_u8(&input);
        assert_eq!(q.len(), 4096);
        let deq = dequantize_asymmetric_u8(&q, scale, zp);
        let err = compute_quantization_error(&input, &deq);
        assert!(err.snr > 0.0, "SNR should be positive for non-trivial signal");
    }

    // ── Mixed / additional edge cases ──────────────────────────

    #[test]
    fn ternary_empty_input() {
        let q = quantize_ternary(&[], 0.5);
        assert!(q.is_empty());
    }

    #[test]
    fn binary_empty_input() {
        let q = quantize_binary(&[]);
        assert!(q.is_empty());
    }

    #[test]
    fn symmetric_i8_empty_input() {
        let (q, scale) = quantize_symmetric_i8(&[], 8);
        assert!(q.is_empty());
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn asymmetric_u8_empty_input() {
        let (q, scale, zp) = quantize_asymmetric_u8(&[]);
        assert!(q.is_empty());
        assert_eq!(scale, 0.0);
        assert_eq!(zp, 0);
    }

    #[test]
    fn error_snr_improves_with_bits() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0) - 1.0).collect();

        let deq2 = {
            let (q, s) = quantize_symmetric_i8(&input, 2);
            dequantize_symmetric_i8(&q, s)
        };
        let deq8 = {
            let (q, s) = quantize_symmetric_i8(&input, 8);
            dequantize_symmetric_i8(&q, s)
        };

        let snr2 = compute_quantization_error(&input, &deq2).snr;
        let snr8 = compute_quantization_error(&input, &deq8).snr;
        assert!(snr8 > snr2, "8-bit SNR ({snr8}) should exceed 2-bit SNR ({snr2})");
    }
}
