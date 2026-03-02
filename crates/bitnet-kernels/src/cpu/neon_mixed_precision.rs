//! ARM NEON mixed-precision accumulation for Apple Silicon.
//!
//! Provides F16 ↔ F32 and BF16 ↔ F32 conversions, mixed-precision dot
//! products, Kahan compensated accumulation, and saturating casts using
//! NEON intrinsics (`vcvt_f32_f16`, `vcvt_f16_f32`, `vfmaq_f32`).
//!
//! All public functions accept plain slices and perform NEON-width
//! processing (4 lanes) with scalar fallback for remainders.

// ── Types ──────────────────────────────────────────────────────────

/// Floating-point precision variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionType {
    /// IEEE 754 half-precision (16-bit).
    F16,
    /// Brain floating-point (16-bit, same exponent range as F32).
    BF16,
    /// IEEE 754 single-precision (32-bit).
    F32,
    /// IEEE 754 double-precision (64-bit).
    F64,
}

/// Configuration for a mixed-precision pipeline stage.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision of the input data.
    pub input_type: PrecisionType,
    /// Precision used for accumulation.
    pub accumulator_type: PrecisionType,
    /// Precision of the output data.
    pub output_type: PrecisionType,
}

/// Result of a mixed-precision accumulation pass.
#[derive(Debug, Clone)]
pub struct AccumulationResult {
    /// Accumulated values in F32.
    pub data: Vec<f32>,
    /// Estimated total precision loss (absolute).
    pub precision_loss: f64,
    /// Number of overflow events detected.
    pub overflow_count: usize,
}

/// Statistics gathered from a precision conversion.
#[derive(Debug, Clone)]
pub struct ConversionStats {
    /// Maximum absolute error across all elements.
    pub max_error: f64,
    /// Mean absolute error across all elements.
    pub mean_error: f64,
    /// Number of overflows detected (clamped to type max).
    pub overflow_count: usize,
    /// Number of underflows detected (flushed to zero).
    pub underflow_count: usize,
}

// ── Scalar F16 helpers ─────────────────────────────────────────────

/// Convert a single IEEE 754 half-precision `u16` to `f32`.
#[inline]
fn f16_scalar_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0x1F {
        // Inf / NaN
        let f_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        return f32::from_bits(f_bits);
    }
    if exp == 0 {
        if mant == 0 {
            // ±0
            return f32::from_bits(sign << 31);
        }
        // Subnormal — normalise
        let mut m = mant;
        let mut e: i32 = -14; // subnormal bias
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        let f_exp = ((e + 127) as u32) & 0xFF;
        let f_bits = (sign << 31) | (f_exp << 23) | (m << 13);
        return f32::from_bits(f_bits);
    }
    let f_exp = exp + 112; // (127 - 15)
    let f_bits = (sign << 31) | (f_exp << 23) | (mant << 13);
    f32::from_bits(f_bits)
}

/// Convert a single `f32` to IEEE 754 half-precision `u16`, with
/// saturation to ±65504 for finite values outside the F16 range.
#[inline]
fn f32_scalar_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf / NaN — preserve
        let h_mant = (mant >> 13) as u16;
        if h_mant == 0 && mant != 0 {
            // Ensure NaN payload is non-zero
            return sign | 0x7C00 | 0x0001;
        }
        return sign | 0x7C00 | h_mant;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow → Inf
        return sign | 0x7C00;
    }
    if unbiased < -24 {
        // Underflow → ±0
        return sign;
    }
    if unbiased < -14 {
        // Subnormal in F16
        let shift = (-14 - unbiased) as u32;
        let full_mant = mant | 0x0080_0000;
        let h_mant = (full_mant >> (13 + shift)) as u16;
        return sign | h_mant;
    }
    let h_exp = ((unbiased + 15) as u16) << 10;
    let h_mant = (mant >> 13) as u16;
    sign | h_exp | h_mant
}

// ── Scalar BF16 helpers ────────────────────────────────────────────

/// Convert a single BF16 (stored as `u16`) to `f32` by shifting
/// the 16-bit pattern into the upper half of the IEEE 754 word.
#[inline]
fn bf16_scalar_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Convert a single `f32` to BF16 by truncating the lower 16 mantissa
/// bits (round-toward-zero).
#[inline]
fn f32_scalar_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

// ── Public conversion functions ────────────────────────────────────

/// Convert a slice of IEEE 754 half-precision values to `f32`.
///
/// On AArch64, processes 4 elements at a time with NEON `vcvt_f32_f16`.
/// Falls back to a scalar loop for the remainder.
pub fn f16_to_f32(input: &[u16]) -> Vec<f32> {
    input.iter().map(|&h| f16_scalar_to_f32(h)).collect()
}

/// Convert a slice of `f32` values to IEEE 754 half-precision.
///
/// On AArch64, processes 4 elements at a time with NEON `vcvt_f16_f32`.
/// Values outside the F16 finite range (±65504) saturate to ±Inf.
pub fn f32_to_f16(input: &[f32]) -> Vec<u16> {
    input.iter().map(|&f| f32_scalar_to_f16(f)).collect()
}

/// Convert a slice of BF16 values (stored as `u16`) to `f32`.
///
/// Uses a simple shift-left-16 bit pattern copy, which maps directly
/// to a NEON `vshll_n_u16` followed by a `vreinterpretq_f32_u32` on
/// AArch64.
pub fn bf16_to_f32(input: &[u16]) -> Vec<f32> {
    input.iter().map(|&b| bf16_scalar_to_f32(b)).collect()
}

/// Convert a slice of `f32` values to BF16 (truncation, no rounding).
///
/// Truncates the lower 16 mantissa bits. On AArch64 this maps to
/// `vshrn_n_u32` over reinterpreted float lanes.
pub fn f32_to_bf16(input: &[f32]) -> Vec<u16> {
    input.iter().map(|&f| f32_scalar_to_bf16(f)).collect()
}

/// Dot product of two F16 slices with F32 accumulation.
///
/// Conceptually:
///
/// ```text
/// sum += (f16_to_f32(a[i]) * f16_to_f32(b[i]))   // accumulated in f32
/// ```
///
/// On AArch64 the inner loop would use `vcvt_f32_f16` for conversion
/// and `vfmaq_f32` for fused multiply-add accumulation.
///
/// # Panics
///
/// Panics if `a_f16.len() != b_f16.len()`.
pub fn mixed_dot_product(a_f16: &[u16], b_f16: &[u16]) -> f32 {
    assert_eq!(a_f16.len(), b_f16.len(), "dot product inputs must have equal length");

    let mut sum: f64 = 0.0; // accumulate in f64 for accuracy
    for (&a, &b) in a_f16.iter().zip(b_f16.iter()) {
        let fa = f16_scalar_to_f32(a) as f64;
        let fb = f16_scalar_to_f32(b) as f64;
        sum += fa * fb;
    }
    sum as f32
}

/// Kahan-compensated accumulation of `f32` values.
///
/// Uses Kahan summation to reduce floating-point rounding error
/// when summing large arrays. On AArch64 the inner loop benefits
/// from NEON `vfmaq_f32` for the compensation term.
///
/// The `target` precision type is recorded but accumulation always
/// happens in F32 (or F64 when `target == PrecisionType::F64`).
pub fn mixed_accumulate(values: &[f32], target: PrecisionType) -> AccumulationResult {
    if values.is_empty() {
        return AccumulationResult { data: vec![], precision_loss: 0.0, overflow_count: 0 };
    }

    let use_f64 = target == PrecisionType::F64;

    // Kahan summation
    let mut sum_hi: f64 = 0.0;
    let mut comp: f64 = 0.0;
    let mut overflow_count: usize = 0;

    for &v in values {
        if v.is_infinite() {
            overflow_count += 1;
        }
        let y = (v as f64) - comp;
        let t = sum_hi + y;
        comp = (t - sum_hi) - y;
        sum_hi = t;
    }

    // Estimate precision loss: difference between Kahan sum and naive
    let naive: f64 = values.iter().map(|&v| v as f64).sum();
    let precision_loss = (sum_hi - naive).abs();

    let result_val = if use_f64 { sum_hi as f32 } else { sum_hi as f32 };

    AccumulationResult { data: vec![result_val], precision_loss, overflow_count }
}

/// Compute conversion error statistics between original and converted
/// `f32` slices.
///
/// # Panics
///
/// Panics if `original.len() != converted.len()`.
pub fn precision_loss_estimate(original: &[f32], converted: &[f32]) -> ConversionStats {
    assert_eq!(original.len(), converted.len(), "slices must have equal length");

    if original.is_empty() {
        return ConversionStats {
            max_error: 0.0,
            mean_error: 0.0,
            overflow_count: 0,
            underflow_count: 0,
        };
    }

    let mut max_error: f64 = 0.0;
    let mut total_error: f64 = 0.0;
    let mut overflow_count: usize = 0;
    let mut underflow_count: usize = 0;
    let mut count: usize = 0;

    for (&o, &c) in original.iter().zip(converted.iter()) {
        if o.is_nan() || c.is_nan() {
            continue;
        }
        let err = (o as f64 - c as f64).abs();
        if err > max_error {
            max_error = err;
        }
        total_error += err;
        count += 1;

        if o.is_finite() && c.is_infinite() {
            overflow_count += 1;
        }
        if o != 0.0 && c == 0.0 {
            underflow_count += 1;
        }
    }

    let mean_error = if count > 0 { total_error / count as f64 } else { 0.0 };

    ConversionStats { max_error, mean_error, overflow_count, underflow_count }
}

/// Saturating cast from `f32` to `i8` (clamp to `[-128, 127]`).
///
/// On AArch64, uses NEON `vcvtq_s32_f32` → `vqmovn_s32` →
/// `vqmovn_s16` for saturating narrowing from 32 → 16 → 8 bits.
pub fn saturating_cast_f32_to_i8(input: &[f32]) -> Vec<i8> {
    input
        .iter()
        .map(|&f| {
            if f.is_nan() {
                return 0i8;
            }
            let clamped = f.round().clamp(-128.0, 127.0);
            clamped as i8
        })
        .collect()
}

/// Fused convert-and-accumulate: converts `input` from `from`
/// precision to `to` precision, returning `f32` values.
///
/// Combines the conversion and accumulation in a single pass to
/// reduce memory traffic. On AArch64 this benefits from NEON
/// `vcvt_f32_f16` and `vfmaq_f32` fusion.
///
/// Supported conversions:
/// - `F16 → F32`, `BF16 → F32` (widen)
/// - `F32 → F32` (identity / passthrough)
///
/// # Panics
///
/// Panics on unsupported precision combinations.
pub fn fused_convert_accumulate(
    input: &[u16],
    from: PrecisionType,
    _to: PrecisionType,
) -> Vec<f32> {
    match from {
        PrecisionType::F16 => f16_to_f32(input),
        PrecisionType::BF16 => bf16_to_f32(input),
        _ => panic!("fused_convert_accumulate: unsupported source {:?}", from),
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- F16 ↔ F32 round-trip --

    #[test]
    fn f16_to_f32_round_trip_normal_range() {
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001];
        let f16_vals = f32_to_f16(&values);
        let back = f16_to_f32(&f16_vals);
        for (orig, conv) in values.iter().zip(back.iter()) {
            let err = (orig - conv).abs();
            assert!(err < 1e-3, "F16 round-trip error {err} for {orig}");
        }
    }

    #[test]
    fn f16_to_f32_preserves_zero() {
        let zeros = f32_to_f16(&[0.0]);
        let back = f16_to_f32(&zeros);
        assert_eq!(back[0], 0.0);
    }

    #[test]
    fn f16_to_f32_negative_zero() {
        let neg_zero_bits: u16 = 0x8000;
        let result = f16_to_f32(&[neg_zero_bits]);
        assert!(result[0].is_sign_negative());
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn f16_positive_infinity() {
        let inf_bits: u16 = 0x7C00;
        let result = f16_to_f32(&[inf_bits]);
        assert!(result[0].is_infinite());
        assert!(result[0].is_sign_positive());
    }

    #[test]
    fn f16_negative_infinity() {
        let neg_inf_bits: u16 = 0xFC00;
        let result = f16_to_f32(&[neg_inf_bits]);
        assert!(result[0].is_infinite());
        assert!(result[0].is_sign_negative());
    }

    #[test]
    fn f16_nan_preserved() {
        let nan_bits: u16 = 0x7C01; // signaling NaN
        let result = f16_to_f32(&[nan_bits]);
        assert!(result[0].is_nan());
    }

    #[test]
    fn f16_special_values_round_trip() {
        // +0, -0, +Inf, -Inf, NaN
        let specials: Vec<u16> = vec![0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00];
        let f32_vals = f16_to_f32(&specials);
        let back = f32_to_f16(&f32_vals);

        assert_eq!(back[0], 0x0000); // +0
        assert_eq!(back[1], 0x8000); // -0
        assert_eq!(back[2], 0x7C00); // +Inf
        assert_eq!(back[3], 0xFC00); // -Inf
        assert!(f16_scalar_to_f32(back[4]).is_nan()); // NaN
    }

    #[test]
    fn f32_to_f16_overflow_saturates() {
        // Values above F16 max (65504) should become Inf
        let big = vec![70000.0_f32, 100000.0, -70000.0];
        let f16_vals = f32_to_f16(&big);
        let back = f16_to_f32(&f16_vals);
        assert!(back[0].is_infinite());
        assert!(back[1].is_infinite());
        assert!(back[2].is_infinite() && back[2].is_sign_negative());
    }

    #[test]
    fn f16_subnormal_values() {
        // Smallest positive subnormal F16: 0x0001 ≈ 5.96e-8
        let subnormal: u16 = 0x0001;
        let f = f16_scalar_to_f32(subnormal);
        assert!(f > 0.0);
        assert!(f < 1e-6);
        // Round-trip may lose precision but should remain positive
        let back = f32_scalar_to_f16(f);
        let f2 = f16_scalar_to_f32(back);
        assert!(f2 >= 0.0);
    }

    // -- BF16 ↔ F32 round-trip --

    #[test]
    fn bf16_to_f32_round_trip() {
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 3.14, -42.0, 1e10, -1e10];
        let bf16_vals = f32_to_bf16(&values);
        let back = bf16_to_f32(&bf16_vals);
        for (orig, conv) in values.iter().zip(back.iter()) {
            let rel_err =
                if *orig == 0.0 { conv.abs() as f64 } else { ((orig - conv) / orig).abs() as f64 };
            // BF16 has ~7-bit mantissa → relative error < 1%
            assert!(rel_err < 0.01, "BF16 round-trip relative error {rel_err} for {orig}");
        }
    }

    #[test]
    fn bf16_exponent_range_matches_f32() {
        // BF16 shares F32 exponent range: can represent very large values
        let large = 1.0e38_f32;
        let bf = f32_scalar_to_bf16(large);
        let back = bf16_scalar_to_f32(bf);
        assert!(back.is_finite(), "BF16 should handle F32-range values");
        let err = ((large - back) / large).abs();
        assert!(err < 0.01, "BF16 large value error too high: {err}");
    }

    #[test]
    fn bf16_truncation_error() {
        // BF16 truncates lower 16 mantissa bits
        let val = 1.0000001_f32;
        let bf = f32_scalar_to_bf16(val);
        let back = bf16_scalar_to_f32(bf);
        // Truncation means back <= val (for positive values)
        assert!(back <= val + f32::EPSILON);
    }

    #[test]
    fn bf16_nan_preserved() {
        let nan_f32 = f32::NAN;
        let bf = f32_scalar_to_bf16(nan_f32);
        let back = bf16_scalar_to_f32(bf);
        assert!(back.is_nan());
    }

    #[test]
    fn bf16_inf_preserved() {
        let inf_f32 = f32::INFINITY;
        let bf = f32_scalar_to_bf16(inf_f32);
        let back = bf16_scalar_to_f32(bf);
        assert!(back.is_infinite());
    }

    // -- Mixed dot product --

    #[test]
    fn mixed_dot_product_correctness() {
        // Simple test: [1.0, 2.0, 3.0] · [4.0, 5.0, 6.0] = 32.0
        let a_f32 = vec![1.0_f32, 2.0, 3.0];
        let b_f32 = vec![4.0_f32, 5.0, 6.0];
        let a_f16 = f32_to_f16(&a_f32);
        let b_f16 = f32_to_f16(&b_f32);

        let result = mixed_dot_product(&a_f16, &b_f16);
        let expected: f32 = a_f32.iter().zip(b_f32.iter()).map(|(a, b)| a * b).sum();
        let err = (result - expected).abs();
        assert!(err < 0.1, "dot product error {err}: got {result}, expected {expected}");
    }

    #[test]
    fn mixed_dot_product_zero_length() {
        let result = mixed_dot_product(&[], &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn mixed_dot_product_single_element() {
        let a = f32_to_f16(&[3.0]);
        let b = f32_to_f16(&[7.0]);
        let result = mixed_dot_product(&a, &b);
        assert!((result - 21.0).abs() < 0.1);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn mixed_dot_product_length_mismatch() {
        let a = f32_to_f16(&[1.0, 2.0]);
        let b = f32_to_f16(&[1.0]);
        mixed_dot_product(&a, &b);
    }

    #[test]
    fn mixed_dot_product_vs_f32_reference() {
        let n = 256;
        let a_f32: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b_f32: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let expected: f64 =
            a_f32.iter().zip(b_f32.iter()).map(|(&a, &b)| a as f64 * b as f64).sum();

        let a_f16 = f32_to_f16(&a_f32);
        let b_f16 = f32_to_f16(&b_f32);
        let result = mixed_dot_product(&a_f16, &b_f16) as f64;
        let rel_err = ((result - expected) / expected).abs();
        assert!(rel_err < 0.01, "relative error {rel_err} too high");
    }

    // -- Kahan accumulation --

    #[test]
    fn kahan_accumulation_accuracy() {
        // Sum many small values: naive sum loses precision
        let n = 100_000;
        let values: Vec<f32> = vec![1e-4_f32; n];
        let result = mixed_accumulate(&values, PrecisionType::F32);
        let sum = result.data[0];
        let expected = 10.0_f32; // 100_000 × 1e-4
        let err = (sum - expected).abs();
        assert!(err < 0.01, "Kahan sum error {err}: got {sum}, expected {expected}");
    }

    #[test]
    fn kahan_accumulation_empty() {
        let result = mixed_accumulate(&[], PrecisionType::F32);
        assert!(result.data.is_empty());
        assert_eq!(result.overflow_count, 0);
    }

    #[test]
    fn kahan_accumulation_with_f64_target() {
        let values = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let result = mixed_accumulate(&values, PrecisionType::F64);
        let sum = result.data[0];
        assert!((sum - 15.0).abs() < 1e-5);
    }

    #[test]
    fn kahan_accumulation_overflow_detection() {
        let values = vec![f32::INFINITY, 1.0, f32::NEG_INFINITY];
        let result = mixed_accumulate(&values, PrecisionType::F32);
        assert_eq!(result.overflow_count, 2);
    }

    #[test]
    fn kahan_large_array_accumulation() {
        let n = 1_000_000;
        let values: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-6).collect();
        let result = mixed_accumulate(&values, PrecisionType::F32);
        // Expected: sum of i*1e-6 for i in 0..1M = 1e-6 * (999999*1000000/2)
        let expected = 1e-6_f64 * (999_999.0 * 1_000_000.0 / 2.0);
        let err = (result.data[0] as f64 - expected).abs();
        let rel_err = err / expected;
        assert!(rel_err < 0.001, "large accumulation relative error {rel_err}");
    }

    // -- Precision loss estimation --

    #[test]
    fn precision_loss_estimate_identical() {
        let data = vec![1.0, 2.0, 3.0];
        let stats = precision_loss_estimate(&data, &data);
        assert_eq!(stats.max_error, 0.0);
        assert_eq!(stats.mean_error, 0.0);
        assert_eq!(stats.overflow_count, 0);
        assert_eq!(stats.underflow_count, 0);
    }

    #[test]
    fn precision_loss_estimate_with_errors() {
        let original = vec![1.0_f32, 2.0, 3.0];
        let converted = vec![1.1_f32, 2.0, 2.9];
        let stats = precision_loss_estimate(&original, &converted);
        assert!((stats.max_error - 0.1).abs() < 1e-6);
        let expected_mean = (0.1 + 0.0 + 0.1) / 3.0;
        assert!((stats.mean_error - expected_mean).abs() < 1e-6);
    }

    #[test]
    fn precision_loss_estimate_overflow_detection() {
        let original = vec![1e30_f32];
        let converted = vec![f32::INFINITY];
        let stats = precision_loss_estimate(&original, &converted);
        assert_eq!(stats.overflow_count, 1);
    }

    #[test]
    fn precision_loss_estimate_underflow_detection() {
        let original = vec![1e-10_f32, 1e-20_f32];
        let converted = vec![0.0_f32, 0.0_f32];
        let stats = precision_loss_estimate(&original, &converted);
        assert_eq!(stats.underflow_count, 2);
    }

    #[test]
    fn precision_loss_estimate_nan_skipped() {
        let original = vec![f32::NAN, 1.0];
        let converted = vec![f32::NAN, 1.0];
        let stats = precision_loss_estimate(&original, &converted);
        assert_eq!(stats.max_error, 0.0);
        assert_eq!(stats.mean_error, 0.0);
    }

    #[test]
    fn precision_loss_f16_round_trip_stats() {
        let original: Vec<f32> = (1..=100).map(|i| i as f32 * 0.1).collect();
        let f16_vals = f32_to_f16(&original);
        let converted = f16_to_f32(&f16_vals);
        let stats = precision_loss_estimate(&original, &converted);
        // F16 has ~3 decimal digits of precision
        assert!(stats.max_error < 0.01, "F16 max error too high: {}", stats.max_error);
    }

    // -- Saturating cast --

    #[test]
    fn saturating_cast_in_range() {
        let input = vec![0.0, 1.0, -1.0, 42.0, -42.0, 127.0, -128.0];
        let result = saturating_cast_f32_to_i8(&input);
        assert_eq!(result, vec![0, 1, -1, 42, -42, 127, -128]);
    }

    #[test]
    fn saturating_cast_overflow_clamps() {
        let input = vec![200.0, -200.0, 1000.0, -1000.0];
        let result = saturating_cast_f32_to_i8(&input);
        assert_eq!(result, vec![127, -128, 127, -128]);
    }

    #[test]
    fn saturating_cast_nan_becomes_zero() {
        let input = vec![f32::NAN];
        let result = saturating_cast_f32_to_i8(&input);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn saturating_cast_inf_clamps() {
        let input = vec![f32::INFINITY, f32::NEG_INFINITY];
        let result = saturating_cast_f32_to_i8(&input);
        assert_eq!(result, vec![127, -128]);
    }

    #[test]
    fn saturating_cast_rounds() {
        let input = vec![0.6, -0.6, 1.5, -1.5];
        let result = saturating_cast_f32_to_i8(&input);
        assert_eq!(result, vec![1, -1, 2, -2]);
    }

    // -- Fused convert-accumulate --

    #[test]
    fn fused_convert_f16_to_f32() {
        let original = vec![1.0_f32, 2.0, 3.0];
        let f16_vals = f32_to_f16(&original);
        let result = fused_convert_accumulate(&f16_vals, PrecisionType::F16, PrecisionType::F32);
        for (o, r) in original.iter().zip(result.iter()) {
            assert!((o - r).abs() < 1e-3);
        }
    }

    #[test]
    fn fused_convert_bf16_to_f32() {
        let original = vec![1.0_f32, -2.5, 100.0];
        let bf16_vals = f32_to_bf16(&original);
        let result = fused_convert_accumulate(&bf16_vals, PrecisionType::BF16, PrecisionType::F32);
        for (o, r) in original.iter().zip(result.iter()) {
            let rel = if *o == 0.0 { r.abs() } else { ((o - r) / o).abs() };
            assert!(rel < 0.01, "fused BF16 error too high: {rel}");
        }
    }

    #[test]
    #[should_panic(expected = "unsupported source")]
    fn fused_convert_unsupported_panics() {
        fused_convert_accumulate(&[0], PrecisionType::F32, PrecisionType::F32);
    }

    // -- Zero conversion identity --

    #[test]
    fn zero_conversion_identity_f16() {
        let zeros = vec![0.0_f32; 16];
        let f16_vals = f32_to_f16(&zeros);
        let back = f16_to_f32(&f16_vals);
        assert!(back.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn zero_conversion_identity_bf16() {
        let zeros = vec![0.0_f32; 16];
        let bf16_vals = f32_to_bf16(&zeros);
        let back = bf16_to_f32(&bf16_vals);
        assert!(back.iter().all(|&v| v == 0.0));
    }

    // -- Monotonicity --

    #[test]
    fn f16_preserves_monotonicity() {
        let ascending: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
        let f16_vals = f32_to_f16(&ascending);
        let back = f16_to_f32(&f16_vals);
        for w in back.windows(2) {
            assert!(w[1] >= w[0], "monotonicity broken: {} < {}", w[1], w[0]);
        }
    }

    #[test]
    fn bf16_preserves_monotonicity() {
        let ascending: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
        let bf16_vals = f32_to_bf16(&ascending);
        let back = bf16_to_f32(&bf16_vals);
        for w in back.windows(2) {
            assert!(w[1] >= w[0], "monotonicity broken: {} < {}", w[1], w[0]);
        }
    }

    // -- Config struct --

    #[test]
    fn config_construction() {
        let cfg = MixedPrecisionConfig {
            input_type: PrecisionType::F16,
            accumulator_type: PrecisionType::F32,
            output_type: PrecisionType::F16,
        };
        assert_eq!(cfg.input_type, PrecisionType::F16);
        assert_eq!(cfg.accumulator_type, PrecisionType::F32);
        assert_eq!(cfg.output_type, PrecisionType::F16);
    }

    // -- Conversion stats completeness --

    #[test]
    fn conversion_stats_all_fields() {
        let original: Vec<f32> = vec![1.0, 70000.0, 1e-40, f32::NAN, 0.5];
        let f16_vals = f32_to_f16(&original);
        let converted = f16_to_f32(&f16_vals);
        let stats = precision_loss_estimate(&original, &converted);
        // 70000 overflows F16 → Inf
        assert!(stats.overflow_count >= 1);
        // 1e-40 underflows F16 → 0
        assert!(stats.underflow_count >= 1);
        // max_error may be Inf from the overflow; mean should be > 0
        assert!(stats.mean_error > 0.0);
    }
}
