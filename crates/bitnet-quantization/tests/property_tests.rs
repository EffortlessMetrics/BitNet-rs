//! Property-based tests for bitnet-quantization public API.
//!
//! Tests mathematical invariants that must hold for all valid inputs:
//! - `qk256_tolerance_bytes`: monotonicity, minimum floor, proportionality
//! - `utils::calculate_scale`: always positive finite for finite inputs
//! - `utils::pack_2bit_values` / `unpack_2bit_values`: round-trip identity
//! - `utils::quantize_value` / `dequantize_value`: order preservation, finite output

#![cfg(feature = "cpu")]

use bitnet_quantization::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};
use bitnet_quantization::utils::{
    calculate_scale, pack_2bit_values, unpack_2bit_values, quantize_value, dequantize_value,
};
use proptest::prelude::*;

// ── qk256_tolerance_bytes ───────────────────────────────────────────────────

proptest! {
    /// Tolerance output is always at least 8 bytes (alignment floor).
    #[test]
    fn tolerance_minimum_floor(n in 0usize..usize::MAX) {
        prop_assert!(qk256_tolerance_bytes(n) >= 8);
    }

    /// Tolerance never exceeds the input size.
    #[test]
    fn tolerance_does_not_exceed_input(n in 1usize..1_000_000_000usize) {
        prop_assert!(qk256_tolerance_bytes(n) <= n);
    }

    /// Tolerance is monotone: larger tensors ≥ tolerance of smaller tensors.
    #[test]
    fn tolerance_monotone(a in 0usize..500_000_000usize, b in 0usize..500_000_000usize) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(qk256_tolerance_bytes(lo) <= qk256_tolerance_bytes(hi));
    }

    /// Tolerance is approximately QK256_SIZE_TOLERANCE_PERCENT of the input
    /// (within 1 byte of ceiling, for inputs large enough to exceed the 8-byte floor).
    #[test]
    fn tolerance_proportional_for_large_inputs(n in 10_000usize..1_000_000_000usize) {
        let tol = qk256_tolerance_bytes(n) as f64;
        let expected = (n as f64) * QK256_SIZE_TOLERANCE_PERCENT;
        // Ceiling gives at most 1 more than floor
        prop_assert!(tol >= expected, "tolerance {} < expected {}", tol, expected);
        prop_assert!(tol <= expected + 1.0, "tolerance {} too far above expected {}", tol, expected);
    }
}

// ── utils::calculate_scale ──────────────────────────────────────────────────

proptest! {
    /// Scale is always positive and finite for non-empty finite-valued slices.
    #[test]
    fn scale_positive_finite_for_finite_data(
        data in prop::collection::vec(-1000.0f32..1000.0f32, 1..256),
        bits in 2u8..8u8,
    ) {
        let scale = calculate_scale(&data, bits);
        prop_assert!(scale.is_finite(), "scale must be finite, got {}", scale);
        prop_assert!(scale > 0.0, "scale must be positive, got {}", scale);
    }

    /// Scale fallback for all-zero data is exactly 1.0.
    #[test]
    fn scale_all_zeros_returns_one(len in 1usize..256usize) {
        let zeros = vec![0.0f32; len];
        prop_assert_eq!(calculate_scale(&zeros, 2), 1.0);
    }

    /// Scale is always 1.0 (safe fallback) when all values are non-finite.
    #[test]
    fn scale_non_finite_data_fallback(len in 1usize..64usize) {
        let nans = vec![f32::NAN; len];
        prop_assert_eq!(calculate_scale(&nans, 2), 1.0);
    }
}

// ── pack/unpack round-trip ──────────────────────────────────────────────────

proptest! {
    /// Packing then unpacking 2-bit values recovers the original (for in-range i8 values).
    /// 2-bit signed range: -1, 0, 1 (plus -2 for full 2-bit signed).
    #[test]
    fn pack_unpack_roundtrip(
        values in prop::collection::vec(-1i8..=1i8, 4..128),
    ) {
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, values.len());
        prop_assert_eq!(unpacked, values, "pack→unpack must be identity");
    }

    /// Packed length is always ceil(n / 4) bytes (4 2-bit values per byte).
    #[test]
    fn pack_length_is_ceil_n_over_4(
        values in prop::collection::vec(-1i8..=1i8, 1..256),
    ) {
        let packed = pack_2bit_values(&values);
        let expected_len = values.len().div_ceil(4);
        prop_assert_eq!(packed.len(), expected_len);
    }
}

// ── quantize_value / dequantize_value ───────────────────────────────────────

proptest! {
    /// Quantized value is always in [-2^(bits-1), 2^(bits-1)-1].
    #[test]
    fn quantized_value_in_range(
        value in -100.0f32..100.0f32,
        scale in 0.001f32..10.0f32,
        bits in 2u8..8u8,
    ) {
        let q = quantize_value(value, scale, bits);
        let lo = -(1i8 << (bits - 1));
        let hi = (1i8 << (bits - 1)) - 1;
        prop_assert!(q >= lo && q <= hi,
            "quantized {} not in [{}, {}]", q, lo, hi);
    }

    /// Dequantized value is always finite for finite inputs.
    #[test]
    fn dequantize_always_finite(
        q in -4i8..=4i8,
        scale in 0.001f32..10.0f32,
    ) {
        let v = dequantize_value(q, scale);
        prop_assert!(v.is_finite(), "dequantized {} * {} = {} must be finite", q, scale, v);
    }

    /// Order preservation: larger quantized values dequantize to larger floats
    /// (same positive scale).
    #[test]
    fn dequantize_order_preserving(
        q1 in -4i8..=4i8,
        q2 in -4i8..=4i8,
        scale in 0.001f32..10.0f32,
    ) {
        let v1 = dequantize_value(q1, scale);
        let v2 = dequantize_value(q2, scale);
        if q1 < q2 {
            prop_assert!(v1 < v2, "order not preserved: q1={} q2={} v1={} v2={}", q1, q2, v1, v2);
        } else if q1 > q2 {
            prop_assert!(v1 > v2, "order not preserved: q1={} q2={} v1={} v2={}", q1, q2, v1, v2);
        } else {
            prop_assert_eq!(v1, v2);
        }
    }
}
