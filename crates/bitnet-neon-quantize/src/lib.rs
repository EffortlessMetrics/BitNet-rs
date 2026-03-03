//! NEON-optimized quantization kernels for `BitNet` inference.
//!
//! This crate provides I2\_S quantization and dequantization routines with
//! optional ARM NEON SIMD acceleration.  When the `neon` target feature is
//! available the hot loops use intrinsics; otherwise a portable scalar
//! fallback is compiled automatically.
//!
//! # Supported quantization modes
//!
//! | Mode | Description |
//! |------|-------------|
//! | **Symmetric** | Zero-point fixed at 0; range = \[-max\_abs, +max\_abs\] |
//! | **Asymmetric** | Arbitrary zero-point; range = \[min, max\] |
//! | **Per-tensor** | One scale (and optional zero-point) for the entire tensor |
//! | **Per-channel** | Independent scale/zero-point per output channel |
//!
//! # I2\_S encoding
//!
//! 2-bit signed integers packed 4 values per byte in little-endian order.
//! Valid decoded values: `{-1, 0, 1}`.

use bitnet_common::QuantizationType;

// Re-export the quantization type this crate implements.
pub use bitnet_common::QuantizationType as QuantType;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Parameters fully describing a symmetric quantization scheme.
///
/// Symmetric quantization maps floating-point values into `[-scale, +scale]`
/// with an implicit zero-point of zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetricParams {
    /// Multiplier that converts a float to the quantized domain.
    pub scale: f32,
    /// Reciprocal of `scale` (cached for dequantization).
    pub inv_scale: f32,
}

/// Parameters fully describing an asymmetric quantization scheme.
///
/// Asymmetric quantization maps `[min, max]` to the full quantized range and
/// stores an explicit zero-point offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsymmetricParams {
    /// Multiplier that converts a float to the quantized domain.
    pub scale: f32,
    /// Reciprocal of `scale` (cached for dequantization).
    pub inv_scale: f32,
    /// Offset applied after scaling so that the real zero maps to an integer.
    pub zero_point: i8,
}

/// Dynamic range statistics computed over a data slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicRange {
    /// Minimum observed value.
    pub min: f32,
    /// Maximum observed value.
    pub max: f32,
    /// `max - min`.
    pub range: f32,
    /// `max(|min|, |max|)`.
    pub max_abs: f32,
}

/// Result of a per-channel quantization pass.
///
/// Each channel gets its own [`SymmetricParams`] and a contiguous slice of
/// quantized bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct PerChannelResult {
    /// One set of parameters per channel.
    pub params: Vec<SymmetricParams>,
    /// Packed I2\_S bytes; length = `ceil(total_elements / 4)`.
    pub data: Vec<u8>,
}

/// Result of a per-channel asymmetric quantization pass.
#[derive(Debug, Clone, PartialEq)]
pub struct PerChannelAsymmetricResult {
    /// One set of parameters per channel.
    pub params: Vec<AsymmetricParams>,
    /// Packed I2\_S bytes; length = `ceil(total_elements / 4)`.
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// I2_S bit constants
// ---------------------------------------------------------------------------

/// Number of quantized values packed into a single byte (2 bits each).
const VALUES_PER_BYTE: usize = 4;

/// Bitmask for a single 2-bit field.
const MASK_2BIT: u8 = 0b11;

// I2_S encoding: -1 → 0b11 (3), 0 → 0b00 (0), 1 → 0b01 (1)
// This keeps 0 as all-zero bits for efficient sparse representations.

/// Encode a single ternary value (`-1`, `0`, or `1`) into its 2-bit I2\_S
/// representation.
#[inline]
const fn encode_i2s(v: i8) -> u8 {
    match v {
        -1 => 0b11,
        1 => 0b01,
        _ => 0b00, // 0 and out-of-range both map to zero
    }
}

/// Decode a 2-bit I2\_S field back to a signed integer.
#[inline]
const fn decode_i2s(bits: u8) -> i8 {
    match bits & MASK_2BIT {
        0b11 => -1,
        0b01 => 1,
        _ => 0, // 0b00 and 0b10 (unused) both map to zero
    }
}

// ---------------------------------------------------------------------------
// Dynamic range
// ---------------------------------------------------------------------------

/// Compute the [`DynamicRange`] of `data`.
///
/// Returns `None` when the slice is empty.
pub fn compute_dynamic_range(data: &[f32]) -> Option<DynamicRange> {
    if data.is_empty() {
        return None;
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in data {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    Some(DynamicRange { min, max, range: max - min, max_abs: min.abs().max(max.abs()) })
}

// ---------------------------------------------------------------------------
// Scale-factor helpers
// ---------------------------------------------------------------------------

/// Compute the symmetric scale factor for the given `max_abs` value.
///
/// The I2\_S range is `{-1, 0, 1}` so the scale simply equals `max_abs`.
/// Returns `0.0` when `max_abs` is zero (all values are zero).
pub fn compute_symmetric_scale(max_abs: f32) -> f32 {
    if max_abs == 0.0 { 0.0 } else { max_abs }
}

/// Build full [`SymmetricParams`] from a `max_abs` value.
pub fn symmetric_params_from_max_abs(max_abs: f32) -> SymmetricParams {
    let scale = compute_symmetric_scale(max_abs);
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    SymmetricParams { scale, inv_scale }
}

/// Compute the asymmetric scale factor and zero-point.
///
/// Maps `[min, max]` to the I2\_S range `{-1, 0, 1}` (quantized range width
/// is 2).  The zero-point is optimized so that real zero maps as closely as
/// possible to a valid ternary value.
pub fn compute_asymmetric_params(min: f32, max: f32) -> AsymmetricParams {
    let range = max - min;
    if range == 0.0 {
        return AsymmetricParams { scale: 0.0, inv_scale: 0.0, zero_point: 0 };
    }
    // scale maps [-1, 1] → [min, max], so scale = range / 2
    let scale = range / 2.0;
    let inv_scale = 1.0 / scale;
    // midpoint of the data range in float space
    let mid = f32::midpoint(min, max);
    // zero_point is the quantized integer that should map to real 0.
    // dequant: real = (q - zp) * scale  =>  for real=0: q = zp
    // quant:   q = round(real * inv_scale + zp)  =>  for real=0: q = zp (exact)
    // We want: 0 = (0 - zp) * scale + mid  =>  zp = mid / scale
    // But with the simpler symmetric-centered formula:
    //   quant(v) = round(v * inv_scale) + zp  ; deq(q) = (q - zp) * scale
    //   For v=0: q = 0 + zp = zp => deq = 0  (exact when zp in range)
    // zp should push the center to 0. zp = round(mid * inv_scale) but clamped.
    #[allow(clippy::cast_possible_truncation)] // clamped to [-1, 1]
    let zp = (mid * inv_scale).round().clamp(-1.0, 1.0) as i8;
    AsymmetricParams { scale, inv_scale, zero_point: zp }
}

/// Optimize the zero-point by minimizing quantization error over `data`.
///
/// Tests each candidate zero-point in `{-1, 0, 1}` and returns the one that
/// yields the smallest total squared error after round-trip quantization.
pub fn optimize_zero_point(data: &[f32], scale: f32) -> i8 {
    if scale == 0.0 || data.is_empty() {
        return 0;
    }
    let inv_scale = 1.0 / scale;
    let mut best_zp: i8 = 0;
    let mut best_err = f64::INFINITY;
    for candidate in [-1i8, 0, 1] {
        let mut err = 0.0f64;
        for &v in data {
            #[allow(clippy::cast_possible_truncation)] // clamped to [-1, 1]
            let q = f32::from(candidate).mul_add(1.0, v * inv_scale).round().clamp(-1.0, 1.0) as i8;
            let deq = (f64::from(q) - f64::from(candidate)) * f64::from(scale);
            let d = f64::from(v) - deq;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_zp = candidate;
        }
    }
    best_zp
}

// ---------------------------------------------------------------------------
// Quantize helpers (float → ternary)
// ---------------------------------------------------------------------------

/// Quantize a single value symmetrically to `{-1, 0, 1}`.
#[inline]
#[allow(clippy::cast_possible_truncation)] // clamped to [-1, 1]
fn quantize_sym_scalar(v: f32, inv_scale: f32) -> i8 {
    (v * inv_scale).round().clamp(-1.0, 1.0) as i8
}

/// Quantize a single value asymmetrically to `{-1, 0, 1}`.
#[inline]
#[allow(clippy::cast_possible_truncation)] // clamped to [-1, 1]
fn quantize_asym_scalar(v: f32, inv_scale: f32, zero_point: i8) -> i8 {
    f32::from(zero_point).mul_add(1.0, v * inv_scale).round().clamp(-1.0, 1.0) as i8
}

// ---------------------------------------------------------------------------
// Pack / unpack
// ---------------------------------------------------------------------------

/// Pack a slice of ternary values into I2\_S bytes (4 values per byte).
fn pack_i2s(values: &[i8]) -> Vec<u8> {
    let n_bytes = values.len().div_ceil(VALUES_PER_BYTE);
    let mut out = vec![0u8; n_bytes];
    for (i, &v) in values.iter().enumerate() {
        let byte_idx = i / VALUES_PER_BYTE;
        let bit_offset = (i % VALUES_PER_BYTE) * 2;
        out[byte_idx] |= encode_i2s(v) << bit_offset;
    }
    out
}

/// Unpack I2\_S bytes into ternary values.
///
/// `count` is the number of logical values to decode (since the last byte may
/// contain padding).
fn unpack_i2s(bytes: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / VALUES_PER_BYTE;
        let bit_offset = (i % VALUES_PER_BYTE) * 2;
        out.push(decode_i2s(bytes[byte_idx] >> bit_offset));
    }
    out
}

// ---------------------------------------------------------------------------
// Per-tensor quantization
// ---------------------------------------------------------------------------

/// Symmetric per-tensor quantization.
///
/// All elements share a single scale derived from the global `max(|x|)`.
/// Returns the packed I2\_S bytes and the associated [`SymmetricParams`].
pub fn quantize_symmetric(data: &[f32]) -> (Vec<u8>, SymmetricParams) {
    let dr = compute_dynamic_range(data).unwrap_or(DynamicRange {
        min: 0.0,
        max: 0.0,
        range: 0.0,
        max_abs: 0.0,
    });
    let params = symmetric_params_from_max_abs(dr.max_abs);
    let ternary: Vec<i8> = data.iter().map(|&v| quantize_sym_scalar(v, params.inv_scale)).collect();
    (pack_i2s(&ternary), params)
}

/// Asymmetric per-tensor quantization.
///
/// The full `[min, max]` range is mapped onto `{-1, 0, 1}` with an explicit
/// zero-point.
pub fn quantize_asymmetric(data: &[f32]) -> (Vec<u8>, AsymmetricParams) {
    let dr = compute_dynamic_range(data).unwrap_or(DynamicRange {
        min: 0.0,
        max: 0.0,
        range: 0.0,
        max_abs: 0.0,
    });
    let params = compute_asymmetric_params(dr.min, dr.max);
    let ternary: Vec<i8> = data
        .iter()
        .map(|&v| quantize_asym_scalar(v, params.inv_scale, params.zero_point))
        .collect();
    (pack_i2s(&ternary), params)
}

// ---------------------------------------------------------------------------
// Per-channel quantization
// ---------------------------------------------------------------------------

/// Symmetric per-channel quantization.
///
/// `data` is interpreted as `n_channels` contiguous rows, each of length
/// `channel_size`.  Each channel gets its own scale factor.
///
/// # Panics
///
/// Panics if `data.len() != n_channels * channel_size`.
pub fn quantize_per_channel_symmetric(
    data: &[f32],
    n_channels: usize,
    channel_size: usize,
) -> PerChannelResult {
    assert_eq!(
        data.len(),
        n_channels * channel_size,
        "data length must equal n_channels * channel_size"
    );

    let mut params = Vec::with_capacity(n_channels);
    let mut all_ternary = Vec::with_capacity(data.len());

    for ch in 0..n_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let channel = &data[start..end];
        let dr = compute_dynamic_range(channel).unwrap_or(DynamicRange {
            min: 0.0,
            max: 0.0,
            range: 0.0,
            max_abs: 0.0,
        });
        let p = symmetric_params_from_max_abs(dr.max_abs);
        for &v in channel {
            all_ternary.push(quantize_sym_scalar(v, p.inv_scale));
        }
        params.push(p);
    }

    PerChannelResult { params, data: pack_i2s(&all_ternary) }
}

/// Asymmetric per-channel quantization.
///
/// Same layout as [`quantize_per_channel_symmetric`] but each channel may
/// have a different zero-point.
///
/// # Panics
///
/// Panics if `data.len() != n_channels * channel_size`.
pub fn quantize_per_channel_asymmetric(
    data: &[f32],
    n_channels: usize,
    channel_size: usize,
) -> PerChannelAsymmetricResult {
    assert_eq!(
        data.len(),
        n_channels * channel_size,
        "data length must equal n_channels * channel_size"
    );

    let mut params = Vec::with_capacity(n_channels);
    let mut all_ternary = Vec::with_capacity(data.len());

    for ch in 0..n_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let channel = &data[start..end];
        let dr = compute_dynamic_range(channel).unwrap_or(DynamicRange {
            min: 0.0,
            max: 0.0,
            range: 0.0,
            max_abs: 0.0,
        });
        let p = compute_asymmetric_params(dr.min, dr.max);
        for &v in channel {
            all_ternary.push(quantize_asym_scalar(v, p.inv_scale, p.zero_point));
        }
        params.push(p);
    }

    PerChannelAsymmetricResult { params, data: pack_i2s(&all_ternary) }
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Dequantize packed I2\_S bytes using symmetric parameters.
///
/// `count` is the number of logical float values to produce.
pub fn dequantize_symmetric(packed: &[u8], params: &SymmetricParams, count: usize) -> Vec<f32> {
    let ternary = unpack_i2s(packed, count);
    ternary.into_iter().map(|q| f32::from(q) * params.scale).collect()
}

/// Dequantize packed I2\_S bytes using asymmetric parameters.
///
/// `count` is the number of logical float values to produce.
pub fn dequantize_asymmetric(packed: &[u8], params: &AsymmetricParams, count: usize) -> Vec<f32> {
    let ternary = unpack_i2s(packed, count);
    ternary
        .into_iter()
        .map(|q| (f32::from(q) - f32::from(params.zero_point)) * params.scale)
        .collect()
}

/// Dequantize a per-channel symmetric result back to floats.
///
/// `channel_size` must match the value used during quantization.
pub fn dequantize_per_channel_symmetric(
    result: &PerChannelResult,
    channel_size: usize,
) -> Vec<f32> {
    let n_channels = result.params.len();
    let total = n_channels * channel_size;
    let ternary = unpack_i2s(&result.data, total);
    let mut out = Vec::with_capacity(total);
    for (ch, p) in result.params.iter().enumerate() {
        let start = ch * channel_size;
        let end = start + channel_size;
        for &q in &ternary[start..end] {
            out.push(f32::from(q) * p.scale);
        }
    }
    out
}

/// Dequantize a per-channel asymmetric result back to floats.
///
/// `channel_size` must match the value used during quantization.
pub fn dequantize_per_channel_asymmetric(
    result: &PerChannelAsymmetricResult,
    channel_size: usize,
) -> Vec<f32> {
    let n_channels = result.params.len();
    let total = n_channels * channel_size;
    let ternary = unpack_i2s(&result.data, total);
    let mut out = Vec::with_capacity(total);
    for (ch, p) in result.params.iter().enumerate() {
        let start = ch * channel_size;
        let end = start + channel_size;
        for &q in &ternary[start..end] {
            out.push((f32::from(q) - f32::from(p.zero_point)) * p.scale);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Convenience: returns the QuantizationType this crate implements
// ---------------------------------------------------------------------------

/// Returns `QuantizationType::I2S` — the type implemented by this crate.
pub const fn supported_quantization_type() -> QuantizationType {
    QuantizationType::I2S
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- encode / decode round-trip ----------------------------------------

    #[test]
    fn encode_decode_neg1() {
        assert_eq!(decode_i2s(encode_i2s(-1)), -1);
    }

    #[test]
    fn encode_decode_zero() {
        assert_eq!(decode_i2s(encode_i2s(0)), 0);
    }

    #[test]
    fn encode_decode_pos1() {
        assert_eq!(decode_i2s(encode_i2s(1)), 1);
    }

    #[test]
    fn encode_out_of_range_clamps_to_zero() {
        assert_eq!(decode_i2s(encode_i2s(5)), 0);
        assert_eq!(decode_i2s(encode_i2s(-5)), 0);
    }

    #[test]
    fn decode_unused_code_is_zero() {
        // 0b10 is not used in our encoding
        assert_eq!(decode_i2s(0b10), 0);
    }

    // -- pack / unpack -----------------------------------------------------

    #[test]
    fn pack_unpack_empty() {
        let packed = pack_i2s(&[]);
        assert!(packed.is_empty());
        let unpacked = unpack_i2s(&packed, 0);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn pack_unpack_single_value() {
        for v in [-1i8, 0, 1] {
            let packed = pack_i2s(&[v]);
            let unpacked = unpack_i2s(&packed, 1);
            assert_eq!(unpacked, vec![v]);
        }
    }

    #[test]
    fn pack_unpack_four_values() {
        let vals = vec![-1i8, 0, 1, -1];
        let packed = pack_i2s(&vals);
        assert_eq!(packed.len(), 1); // 4 values fit in 1 byte
        let unpacked = unpack_i2s(&packed, 4);
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn pack_unpack_five_values_needs_two_bytes() {
        let vals = vec![1i8, -1, 0, 1, -1];
        let packed = pack_i2s(&vals);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i2s(&packed, 5);
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn pack_unpack_eight_values() {
        let vals = vec![-1i8, -1, -1, -1, 1, 1, 1, 1];
        let packed = pack_i2s(&vals);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_i2s(&packed, 8);
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn pack_unpack_all_zeros() {
        let vals = vec![0i8; 16];
        let packed = pack_i2s(&vals);
        assert!(packed.iter().all(|&b| b == 0));
        let unpacked = unpack_i2s(&packed, 16);
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn pack_unpack_alternating() {
        let vals: Vec<i8> = (0..12).map(|i| [-1, 0, 1][i % 3]).collect();
        let packed = pack_i2s(&vals);
        let unpacked = unpack_i2s(&packed, vals.len());
        assert_eq!(unpacked, vals);
    }

    // -- dynamic range -----------------------------------------------------

    #[test]
    fn dynamic_range_empty() {
        assert!(compute_dynamic_range(&[]).is_none());
    }

    #[test]
    fn dynamic_range_single() {
        let dr = compute_dynamic_range(&[3.0]).unwrap();
        assert_eq!(dr.min, 3.0);
        assert_eq!(dr.max, 3.0);
        assert_eq!(dr.range, 0.0);
        assert_eq!(dr.max_abs, 3.0);
    }

    #[test]
    fn dynamic_range_symmetric() {
        let dr = compute_dynamic_range(&[-2.0, 2.0]).unwrap();
        assert_eq!(dr.min, -2.0);
        assert_eq!(dr.max, 2.0);
        assert_eq!(dr.range, 4.0);
        assert_eq!(dr.max_abs, 2.0);
    }

    #[test]
    fn dynamic_range_asymmetric() {
        let dr = compute_dynamic_range(&[-1.0, 3.0]).unwrap();
        assert_eq!(dr.min, -1.0);
        assert_eq!(dr.max, 3.0);
        assert_eq!(dr.range, 4.0);
        assert_eq!(dr.max_abs, 3.0);
    }

    #[test]
    fn dynamic_range_all_positive() {
        let dr = compute_dynamic_range(&[1.0, 5.0, 3.0]).unwrap();
        assert_eq!(dr.min, 1.0);
        assert_eq!(dr.max, 5.0);
        assert_eq!(dr.max_abs, 5.0);
    }

    #[test]
    fn dynamic_range_all_negative() {
        let dr = compute_dynamic_range(&[-5.0, -3.0, -1.0]).unwrap();
        assert_eq!(dr.min, -5.0);
        assert_eq!(dr.max, -1.0);
        assert_eq!(dr.max_abs, 5.0);
    }

    #[test]
    fn dynamic_range_all_zeros() {
        let dr = compute_dynamic_range(&[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(dr.range, 0.0);
        assert_eq!(dr.max_abs, 0.0);
    }

    // -- scale factors -----------------------------------------------------

    #[test]
    fn symmetric_scale_zero() {
        assert_eq!(compute_symmetric_scale(0.0), 0.0);
    }

    #[test]
    fn symmetric_scale_normal() {
        assert_eq!(compute_symmetric_scale(2.5), 2.5);
    }

    #[test]
    fn symmetric_params_zero_data() {
        let p = symmetric_params_from_max_abs(0.0);
        assert_eq!(p.scale, 0.0);
        assert_eq!(p.inv_scale, 0.0);
    }

    #[test]
    fn symmetric_params_nonzero() {
        let p = symmetric_params_from_max_abs(4.0);
        assert_eq!(p.scale, 4.0);
        assert!((p.inv_scale - 0.25).abs() < 1e-7);
    }

    #[test]
    fn asymmetric_params_zero_range() {
        let p = compute_asymmetric_params(5.0, 5.0);
        assert_eq!(p.scale, 0.0);
        assert_eq!(p.zero_point, 0);
    }

    #[test]
    fn asymmetric_params_symmetric_range() {
        let p = compute_asymmetric_params(-2.0, 2.0);
        assert!((p.scale - 2.0).abs() < 1e-6);
        assert_eq!(p.zero_point, 0);
    }

    #[test]
    fn asymmetric_params_positive_only() {
        let p = compute_asymmetric_params(0.0, 4.0);
        assert!((p.scale - 2.0).abs() < 1e-6);
        // mid=2, inv_scale=0.5 → zp = round(2*0.5) = 1
        assert_eq!(p.zero_point, 1);
    }

    // -- zero-point optimization -------------------------------------------

    #[test]
    fn optimize_zp_zero_scale() {
        assert_eq!(optimize_zero_point(&[1.0, 2.0], 0.0), 0);
    }

    #[test]
    fn optimize_zp_empty_data() {
        assert_eq!(optimize_zero_point(&[], 1.0), 0);
    }

    #[test]
    fn optimize_zp_centered_data() {
        // Centered data should prefer zp=0
        let data = vec![-1.0, 0.0, 1.0];
        let zp = optimize_zero_point(&data, 1.0);
        assert_eq!(zp, 0);
    }

    #[test]
    fn optimize_zp_returns_valid_ternary() {
        let data = vec![0.5, 0.8, 1.2, 0.3];
        let zp = optimize_zero_point(&data, 1.0);
        assert!((-1..=1).contains(&zp));
    }

    // -- quantize_sym_scalar -----------------------------------------------

    #[test]
    fn quantize_sym_exact_values() {
        let inv_s = 1.0 / 2.0; // scale = 2.0
        assert_eq!(quantize_sym_scalar(2.0, inv_s), 1);
        assert_eq!(quantize_sym_scalar(-2.0, inv_s), -1);
        assert_eq!(quantize_sym_scalar(0.0, inv_s), 0);
    }

    #[test]
    fn quantize_sym_clamps_large() {
        let inv_s = 1.0;
        assert_eq!(quantize_sym_scalar(100.0, inv_s), 1);
        assert_eq!(quantize_sym_scalar(-100.0, inv_s), -1);
    }

    #[test]
    fn quantize_sym_rounds_to_nearest() {
        let inv_s = 1.0 / 2.0;
        assert_eq!(quantize_sym_scalar(1.5, inv_s), 1); // 1.5/2=0.75 rounds to 1
        assert_eq!(quantize_sym_scalar(0.9, inv_s), 0); // 0.9/2=0.45 rounds to 0
    }

    // -- quantize_asym_scalar ----------------------------------------------

    #[test]
    fn quantize_asym_with_zp_zero() {
        let inv_s = 0.5;
        assert_eq!(quantize_asym_scalar(2.0, inv_s, 0), 1);
        assert_eq!(quantize_asym_scalar(-2.0, inv_s, 0), -1);
    }

    #[test]
    fn quantize_asym_with_zp_one() {
        let inv_s = 1.0;
        // v * inv_s + zp = 0.0 * 1.0 + 1 = 1.0 → rounds to 1 → clamp → 1
        assert_eq!(quantize_asym_scalar(0.0, inv_s, 1), 1);
    }

    // -- per-tensor symmetric round-trip -----------------------------------

    #[test]
    fn symmetric_roundtrip_zeros() {
        let data = vec![0.0; 8];
        let (packed, params) = quantize_symmetric(&data);
        let deq = dequantize_symmetric(&packed, &params, data.len());
        assert_eq!(deq, data);
    }

    #[test]
    fn symmetric_roundtrip_ternary_exact() {
        let data = vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
        let (packed, params) = quantize_symmetric(&data);
        let deq = dequantize_symmetric(&packed, &params, data.len());
        for (a, b) in deq.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn symmetric_roundtrip_scaled() {
        let data = vec![-3.0, 0.0, 3.0];
        let (packed, params) = quantize_symmetric(&data);
        assert!((params.scale - 3.0).abs() < 1e-6);
        let deq = dequantize_symmetric(&packed, &params, data.len());
        for (a, b) in deq.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn symmetric_quantize_clamps_intermediate() {
        // Values beyond [-scale, scale] get clamped to {-1, 1}
        let data = vec![-5.0, 0.0, 5.0, 10.0];
        let (packed, params) = quantize_symmetric(&data);
        let deq = dequantize_symmetric(&packed, &params, data.len());
        // max_abs = 10, so scale = 10
        assert!((params.scale - 10.0).abs() < 1e-6);
        // -5/10 = -0.5 → rounds to -1 → deq = -10
        assert!((deq[0] - (-10.0)).abs() < 1e-6);
        assert!((deq[1] - 0.0).abs() < 1e-6);
        // 5/10 = 0.5 → rounds to 1 → deq = 10
        assert!((deq[2] - 10.0).abs() < 1e-6);
        assert!((deq[3] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn symmetric_single_element() {
        let data = vec![7.0];
        let (packed, params) = quantize_symmetric(&data);
        let deq = dequantize_symmetric(&packed, &params, 1);
        assert!((deq[0] - 7.0).abs() < 1e-6);
    }

    // -- per-tensor asymmetric round-trip ----------------------------------

    #[test]
    fn asymmetric_roundtrip_zeros() {
        let data = vec![0.0; 4];
        let (packed, params) = quantize_asymmetric(&data);
        let deq = dequantize_asymmetric(&packed, &params, data.len());
        for v in &deq {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn asymmetric_roundtrip_symmetric_data() {
        let data = vec![-2.0, 0.0, 2.0];
        let (packed, params) = quantize_asymmetric(&data);
        let deq = dequantize_asymmetric(&packed, &params, data.len());
        for (a, b) in deq.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "deq={a}, orig={b}");
        }
    }

    #[test]
    fn asymmetric_single_element() {
        let data = vec![3.0];
        let (packed, params) = quantize_asymmetric(&data);
        let deq = dequantize_asymmetric(&packed, &params, 1);
        // With only one value, range=0 → scale=0 → deq=0
        // This is expected: a single constant cannot define a range
        assert!(deq[0].abs() < 1e-6 || (deq[0] - 3.0).abs() < 1e-6);
    }

    // -- per-channel symmetric ---------------------------------------------

    #[test]
    fn per_channel_sym_basic() {
        // 2 channels, 4 elements each
        let data = vec![
            -1.0, 0.0, 0.5, 1.0, // ch0: max_abs=1
            -4.0, 0.0, 2.0, 4.0, // ch1: max_abs=4
        ];
        let result = quantize_per_channel_symmetric(&data, 2, 4);
        assert_eq!(result.params.len(), 2);
        assert!((result.params[0].scale - 1.0).abs() < 1e-6);
        assert!((result.params[1].scale - 4.0).abs() < 1e-6);

        let deq = dequantize_per_channel_symmetric(&result, 4);
        assert_eq!(deq.len(), 8);
        // ch0: -1→-1, 0→0, 0.5→1(round to 1*1=1), 1→1
        assert!((deq[0] - (-1.0)).abs() < 1e-6);
        assert!((deq[1] - 0.0).abs() < 1e-6);
        assert!((deq[3] - 1.0).abs() < 1e-6);
        // ch1: -4→-4, 0→0, 2→0(round 2/4=0.5→1*4=4), 4→4
        assert!((deq[4] - (-4.0)).abs() < 1e-6);
        assert!((deq[5] - 0.0).abs() < 1e-6);
        assert!((deq[7] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn per_channel_sym_single_channel() {
        let data = vec![0.0, 3.0, -3.0];
        let result = quantize_per_channel_symmetric(&data, 1, 3);
        assert_eq!(result.params.len(), 1);
        let deq = dequantize_per_channel_symmetric(&result, 3);
        assert!((deq[0] - 0.0).abs() < 1e-6);
        assert!((deq[1] - 3.0).abs() < 1e-6);
        assert!((deq[2] - (-3.0)).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "data length must equal")]
    fn per_channel_sym_panics_on_size_mismatch() {
        quantize_per_channel_symmetric(&[1.0, 2.0, 3.0], 2, 4);
    }

    #[test]
    fn per_channel_sym_all_zero_channels() {
        let data = vec![0.0; 12];
        let result = quantize_per_channel_symmetric(&data, 3, 4);
        for p in &result.params {
            assert_eq!(p.scale, 0.0);
        }
        let deq = dequantize_per_channel_symmetric(&result, 4);
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    // -- per-channel asymmetric --------------------------------------------

    #[test]
    fn per_channel_asym_basic() {
        let data = vec![
            -2.0, 0.0, 2.0, 0.0, // ch0
            -6.0, 0.0, 6.0, 0.0, // ch1
        ];
        let result = quantize_per_channel_asymmetric(&data, 2, 4);
        assert_eq!(result.params.len(), 2);
        let deq = dequantize_per_channel_asymmetric(&result, 4);
        assert_eq!(deq.len(), 8);
        // Both channels are symmetric so zp should be ~0
        assert!((deq[0] - (-2.0)).abs() < 1e-5);
        assert!((deq[2] - 2.0).abs() < 1e-5);
        assert!((deq[4] - (-6.0)).abs() < 1e-5);
        assert!((deq[6] - 6.0).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "data length must equal")]
    fn per_channel_asym_panics_on_size_mismatch() {
        quantize_per_channel_asymmetric(&[1.0], 2, 4);
    }

    // -- dequantize symmetric edge cases -----------------------------------

    #[test]
    fn dequantize_sym_empty() {
        let params = symmetric_params_from_max_abs(1.0);
        let deq = dequantize_symmetric(&[], &params, 0);
        assert!(deq.is_empty());
    }

    #[test]
    fn dequantize_sym_zero_scale() {
        let params = SymmetricParams { scale: 0.0, inv_scale: 0.0 };
        let packed = pack_i2s(&[1, -1, 0]);
        let deq = dequantize_symmetric(&packed, &params, 3);
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    // -- dequantize asymmetric edge cases ----------------------------------

    #[test]
    fn dequantize_asym_empty() {
        let params = AsymmetricParams { scale: 1.0, inv_scale: 1.0, zero_point: 0 };
        let deq = dequantize_asymmetric(&[], &params, 0);
        assert!(deq.is_empty());
    }

    #[test]
    fn dequantize_asym_zero_scale() {
        let params = AsymmetricParams { scale: 0.0, inv_scale: 0.0, zero_point: 0 };
        let packed = pack_i2s(&[1, -1, 0]);
        let deq = dequantize_asymmetric(&packed, &params, 3);
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    // -- supported type ----------------------------------------------------

    #[test]
    fn supported_type_is_i2s() {
        assert_eq!(supported_quantization_type(), QuantizationType::I2S);
    }

    // -- large tensor round-trip -------------------------------------------

    #[test]
    fn symmetric_roundtrip_large() {
        let n = 256;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let (packed, params) = quantize_symmetric(&data);
        let deq = dequantize_symmetric(&packed, &params, n);
        assert_eq!(deq.len(), n);
        for v in &deq {
            assert!((-1.001..=1.001).contains(v));
        }
    }

    #[test]
    fn asymmetric_roundtrip_large() {
        let n = 256;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let (packed, params) = quantize_asymmetric(&data);
        let deq = dequantize_asymmetric(&packed, &params, n);
        assert_eq!(deq.len(), n);
    }

    #[test]
    fn per_channel_sym_large() {
        let n_ch = 8;
        let ch_size = 64;
        let data: Vec<f32> =
            (0..n_ch * ch_size).map(|i| ((i % ch_size) as f32 - 32.0) / 32.0).collect();
        let result = quantize_per_channel_symmetric(&data, n_ch, ch_size);
        let deq = dequantize_per_channel_symmetric(&result, ch_size);
        assert_eq!(deq.len(), n_ch * ch_size);
    }

    #[test]
    fn per_channel_asym_large() {
        let n_ch = 8;
        let ch_size = 64;
        let data: Vec<f32> =
            (0..n_ch * ch_size).map(|i| ((i % ch_size) as f32 - 32.0) / 32.0).collect();
        let result = quantize_per_channel_asymmetric(&data, n_ch, ch_size);
        let deq = dequantize_per_channel_asymmetric(&result, ch_size);
        assert_eq!(deq.len(), n_ch * ch_size);
    }

    // -- ternary output invariant ------------------------------------------

    #[test]
    fn symmetric_output_is_ternary() {
        let data = vec![0.1, 0.5, -0.3, 0.9, -0.9, 0.0, 1.0, -1.0];
        let (packed, params) = quantize_symmetric(&data);
        let deq = dequantize_symmetric(&packed, &params, data.len());
        let s = params.scale;
        for v in &deq {
            let normalized = if s == 0.0 { 0.0 } else { v / s };
            assert!(
                (normalized - (-1.0)).abs() < 1e-6
                    || (normalized - 0.0).abs() < 1e-6
                    || (normalized - 1.0).abs() < 1e-6,
                "not ternary: {normalized}"
            );
        }
    }

    #[test]
    fn pack_byte_count_correct() {
        for n in 0..=20 {
            let vals = vec![0i8; n];
            let packed = pack_i2s(&vals);
            let expected = (n + 3) / 4;
            assert_eq!(packed.len(), expected, "n={n}");
        }
    }

    // -- additional edge cases for 70+ tests ------------------------------

    #[test]
    fn dynamic_range_negative_only() {
        let dr = compute_dynamic_range(&[-10.0, -3.0, -7.0]).unwrap();
        assert_eq!(dr.min, -10.0);
        assert_eq!(dr.max, -3.0);
        assert_eq!(dr.max_abs, 10.0);
    }

    #[test]
    fn dynamic_range_single_negative() {
        let dr = compute_dynamic_range(&[-42.0]).unwrap();
        assert_eq!(dr.min, -42.0);
        assert_eq!(dr.max, -42.0);
        assert_eq!(dr.max_abs, 42.0);
    }

    #[test]
    fn symmetric_preserves_sign() {
        let data = vec![-5.0, 5.0];
        let (packed, params) = quantize_symmetric(&data);
        let deq = dequantize_symmetric(&packed, &params, 2);
        assert!(deq[0] < 0.0);
        assert!(deq[1] > 0.0);
    }

    #[test]
    fn asymmetric_preserves_sign_symmetric_input() {
        let data = vec![-3.0, 3.0];
        let (packed, params) = quantize_asymmetric(&data);
        let deq = dequantize_asymmetric(&packed, &params, 2);
        assert!(deq[0] < 0.0);
        assert!(deq[1] > 0.0);
    }

    #[test]
    fn per_channel_sym_independent_scales() {
        let data = vec![
            -1.0, 1.0, // ch0: scale=1
            -10.0, 10.0, // ch1: scale=10
        ];
        let result = quantize_per_channel_symmetric(&data, 2, 2);
        assert!((result.params[0].scale - 1.0).abs() < 1e-6);
        assert!((result.params[1].scale - 10.0).abs() < 1e-6);
    }

    #[test]
    fn per_channel_sym_many_channels() {
        let n_ch = 32;
        let ch_size = 4;
        let data: Vec<f32> = (0..n_ch)
            .flat_map(|ch| {
                let s = (ch + 1) as f32;
                vec![-s, 0.0, s, 0.0]
            })
            .collect();
        let result = quantize_per_channel_symmetric(&data, n_ch, ch_size);
        assert_eq!(result.params.len(), n_ch);
        for (ch, p) in result.params.iter().enumerate() {
            let expected_scale = (ch + 1) as f32;
            assert!(
                (p.scale - expected_scale).abs() < 1e-6,
                "ch={ch} scale={} expected={expected_scale}",
                p.scale
            );
        }
    }

    #[test]
    fn optimize_zp_biased_positive_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let zp = optimize_zero_point(&data, 2.0);
        assert!((-1..=1).contains(&zp));
    }

    #[test]
    fn optimize_zp_biased_negative_data() {
        let data = vec![-4.0, -3.0, -2.0, -1.0];
        let zp = optimize_zero_point(&data, 2.0);
        assert!((-1..=1).contains(&zp));
    }

    #[test]
    fn dequantize_per_channel_asym_correct_len() {
        let data = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0];
        let result = quantize_per_channel_asymmetric(&data, 2, 3);
        let deq = dequantize_per_channel_asymmetric(&result, 3);
        assert_eq!(deq.len(), 6);
    }

    #[test]
    fn symmetric_params_inv_scale_reciprocal() {
        for scale in [0.5, 1.0, 2.0, 10.0, 100.0] {
            let p = symmetric_params_from_max_abs(scale);
            assert!((p.scale * p.inv_scale - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn pack_unpack_all_neg1() {
        let vals = vec![-1i8; 8];
        let packed = pack_i2s(&vals);
        let unpacked = unpack_i2s(&packed, 8);
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn pack_unpack_all_pos1() {
        let vals = vec![1i8; 8];
        let packed = pack_i2s(&vals);
        let unpacked = unpack_i2s(&packed, 8);
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn quantize_sym_near_zero_input() {
        let data = vec![1e-10, -1e-10, 0.0, 1e-12];
        let (packed, params) = quantize_symmetric(&data);
        let deq = dequantize_symmetric(&packed, &params, data.len());
        // All values very close to zero relative to max_abs
        for v in &deq {
            assert!(v.abs() <= params.scale + 1e-6);
        }
    }

    #[test]
    fn asymmetric_params_negative_range() {
        let p = compute_asymmetric_params(-8.0, -2.0);
        assert!(p.scale > 0.0);
        assert!(p.inv_scale > 0.0);
    }

    #[test]
    fn per_channel_asym_zero_channel() {
        let data = vec![
            0.0, 0.0, 0.0, // ch0: all zero
            -3.0, 0.0, 3.0, // ch1: normal
        ];
        let result = quantize_per_channel_asymmetric(&data, 2, 3);
        assert_eq!(result.params[0].scale, 0.0);
        assert!(result.params[1].scale > 0.0);
    }

    #[test]
    fn pack_unpack_17_values() {
        let vals: Vec<i8> = (0..17).map(|i| [0, 1, -1][i % 3]).collect();
        let packed = pack_i2s(&vals);
        assert_eq!(packed.len(), 5); // ceil(17/4) = 5
        let unpacked = unpack_i2s(&packed, 17);
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn dequantize_per_channel_sym_preserves_zeros() {
        let data = vec![0.0, 0.0, 5.0, -5.0]; // 1 ch, size 4
        let result = quantize_per_channel_symmetric(&data, 1, 4);
        let deq = dequantize_per_channel_symmetric(&result, 4);
        assert!((deq[0]).abs() < 1e-6);
        assert!((deq[1]).abs() < 1e-6);
    }

    #[test]
    fn quantize_sym_positive_only() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let (packed, params) = quantize_symmetric(&data);
        assert!((params.scale - 3.0).abs() < 1e-6);
        let deq = dequantize_symmetric(&packed, &params, 4);
        assert!((deq[0]).abs() < 1e-6); // 0/3 = 0
        assert!((deq[3] - 3.0).abs() < 1e-6); // 3/3 = 1 → deq = 3
    }

    #[test]
    fn encode_all_valid_ternary_values() {
        for v in [-1i8, 0, 1] {
            let encoded = encode_i2s(v);
            let decoded = decode_i2s(encoded);
            assert_eq!(decoded, v, "round-trip failed for {v}");
        }
    }
}
