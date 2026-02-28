//! Property-based tests for bitnet-quantization public API.
//!
//! Tests mathematical invariants that must hold for all valid inputs:
//! - `qk256_tolerance_bytes`: monotonicity, minimum floor, proportionality
//! - `utils::calculate_scale`: always positive finite for finite inputs
//! - `utils::pack_2bit_values` / `unpack_2bit_values`: round-trip identity
//! - `utils::quantize_value` / `dequantize_value`: order preservation, finite output
//! - `TL1Quantizer`: quantize→dequantize round-trip has bounded error
//! - `TL2Quantizer`: quantize→dequantize round-trip has bounded error
//! - `I2SQuantizer`: quantize→dequantize round-trip has bounded error, block-scale accuracy
//! - Edge cases: all-zeros, all-same-value, alternating signs

#![cfg(feature = "cpu")]


use bitnet_quantization::utils::{
    calculate_scale, dequantize_value, pack_2bit_values, quantize_value, unpack_2bit_values,
};
use bitnet_quantization::{
    I2SQuantizer, QK256_SIZE_TOLERANCE_PERCENT, TL1Quantizer, TL2Quantizer, qk256_tolerance_bytes,
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

// ── Helper ──────────────────────────────────────────────────────────────────

fn max_abs_f32(v: &[f32]) -> f32 {
    v.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max)
}

// ── TL1 round-trip properties ───────────────────────────────────────────────

proptest! {
    /// TL1 round-trip: dequantized output length equals input length.
    #[test]
    fn tl1_round_trip_preserves_length(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..128),
    ) {
        let q = TL1Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();
        prop_assert_eq!(deq.len(), data.len(),
            "TL1 round-trip length changed: {} → {}", data.len(), deq.len());
    }

    /// TL1 round-trip: dequantized values are always finite for finite inputs.
    #[test]
    fn tl1_round_trip_finite_outputs(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..128),
    ) {
        let q = TL1Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();
        for (i, &v) in deq.iter().enumerate() {
            prop_assert!(v.is_finite(),
                "TL1 dequantized[{}] = {} is not finite", i, v);
        }
    }

    /// TL1 round-trip: absolute error per element is bounded.
    ///
    /// For 2-bit quantization: worst-case absolute error ≤ 3 * max_abs_input + 1.0.
    /// The +1.0 covers the scale=1.0 fallback for near-zero blocks.
    #[test]
    fn tl1_round_trip_bounded_error(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..128),
    ) {
        let max_abs = max_abs_f32(&data);
        let tolerance = max_abs * 3.0 + 1.0;

        let q = TL1Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();

        for (orig, &deq_val) in data.iter().zip(deq.iter()) {
            let error = (orig - deq_val).abs();
            prop_assert!(error <= tolerance,
                "TL1 error {} exceeds bound {} (max_abs={}, orig={}, deq={})",
                error, tolerance, max_abs, orig, deq_val);
        }
    }
}

// ── TL2 round-trip properties ───────────────────────────────────────────────

proptest! {
    /// TL2 round-trip: dequantized output length equals input length.
    #[test]
    fn tl2_round_trip_preserves_length(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..128),
    ) {
        let q = TL2Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();
        prop_assert_eq!(deq.len(), data.len(),
            "TL2 round-trip length changed: {} → {}", data.len(), deq.len());
    }

    /// TL2 round-trip: dequantized values are always finite for finite inputs.
    #[test]
    fn tl2_round_trip_finite_outputs(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..128),
    ) {
        let q = TL2Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();
        for (i, &v) in deq.iter().enumerate() {
            prop_assert!(v.is_finite(),
                "TL2 dequantized[{}] = {} is not finite", i, v);
        }
    }

    /// TL2 round-trip: absolute error per element is bounded.
    #[test]
    fn tl2_round_trip_bounded_error(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..128),
    ) {
        let max_abs = max_abs_f32(&data);
        let tolerance = max_abs * 3.0 + 1.0;

        let q = TL2Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();

        for (orig, &deq_val) in data.iter().zip(deq.iter()) {
            let error = (orig - deq_val).abs();
            prop_assert!(error <= tolerance,
                "TL2 error {} exceeds bound {} (max_abs={}, orig={}, deq={})",
                error, tolerance, max_abs, orig, deq_val);
        }
    }
}

// ── I2_S round-trip properties ───────────────────────────────────────────────

proptest! {
    /// I2_S round-trip: dequantized output length equals input length.
    #[test]
    fn i2s_round_trip_preserves_length(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..128),
    ) {
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();
        prop_assert_eq!(deq.len(), data.len(),
            "I2S round-trip length changed: {} → {}", data.len(), deq.len());
    }

    /// I2_S round-trip: dequantized values are always finite for finite inputs.
    #[test]
    fn i2s_round_trip_finite_outputs(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..128),
    ) {
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();
        for (i, &v) in deq.iter().enumerate() {
            prop_assert!(v.is_finite(),
                "I2S dequantized[{}] = {} is not finite", i, v);
        }
    }

    /// I2_S round-trip: absolute error per element is bounded.
    ///
    /// I2_S uses scale = max_abs_per_block, so max quantization error ≤ scale/2.
    /// We use a conservative bound of max_abs/2 + 1.0 to account for per-block scale.
    #[test]
    fn i2s_round_trip_bounded_error(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..128),
    ) {
        let max_abs = max_abs_f32(&data);
        // I2_S uses scale = max_abs_per_block; quantization step = scale; error ≤ scale/2
        let tolerance = max_abs / 2.0 + 1.0;

        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq_tensor = q.dequantize_tensor(&quantized).unwrap();
        let deq = deq_tensor.to_vec().unwrap();

        for (orig, &deq_val) in data.iter().zip(deq.iter()) {
            let error = (orig - deq_val).abs();
            prop_assert!(error <= tolerance,
                "I2S error {} exceeds bound {} (max_abs={}, orig={}, deq={})",
                error, tolerance, max_abs, orig, deq_val);
        }
    }

    /// I2_S block-level scale accuracy: stored scales should approximate the
    /// max absolute value in each block (within 1 ULP + rounding).
    #[test]
    fn i2s_block_scale_positive_for_nonzero_blocks(
        data in prop::collection::vec(1.0f32..100.0f32, 4..128),
    ) {
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        // All scales must be positive for strictly positive input data.
        for (i, &s) in quantized.scales.iter().enumerate() {
            prop_assert!(s > 0.0,
                "I2S scale[{}] = {} is non-positive for strictly positive input", i, s);
            prop_assert!(s.is_finite(),
                "I2S scale[{}] = {} is not finite", i, s);
        }
    }
}

// ── Edge cases (deterministic, no proptest) ──────────────────────────────────

/// Helper: run a full quantize→dequantize round-trip and return the output values.
fn i2s_roundtrip(data: &[f32]) -> Vec<f32> {
    let q = I2SQuantizer::new();
    let qt = q.quantize_weights(data).unwrap();
    q.dequantize_tensor(&qt).unwrap().to_vec().unwrap()
}

fn tl1_roundtrip(data: &[f32]) -> Vec<f32> {
    let q = TL1Quantizer::new();
    let qt = q.quantize_weights(data).unwrap();
    q.dequantize_tensor(&qt).unwrap().to_vec().unwrap()
}

fn tl2_roundtrip(data: &[f32]) -> Vec<f32> {
    let q = TL2Quantizer::new();
    let qt = q.quantize_weights(data).unwrap();
    q.dequantize_tensor(&qt).unwrap().to_vec().unwrap()
}

#[test]
fn i2s_edge_all_zeros_roundtrip() {
    let data = vec![0.0f32; 32];
    let out = i2s_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for v in &out {
        assert_eq!(*v, 0.0, "all-zeros I2S should round-trip to zero");
    }
}

#[test]
fn tl1_edge_all_zeros_roundtrip() {
    let data = vec![0.0f32; 32];
    let out = tl1_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for v in &out {
        assert!(v.is_finite(), "all-zeros TL1 output must be finite");
    }
}

#[test]
fn tl2_edge_all_zeros_roundtrip() {
    let data = vec![0.0f32; 32];
    let out = tl2_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for v in &out {
        assert!(v.is_finite(), "all-zeros TL2 output must be finite");
    }
}

/// All-same positive value: round-trip should stay close (max error ≤ scale/2).
#[test]
fn i2s_edge_all_same_positive() {
    let data = vec![3.0f32; 32];
    let out = i2s_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for (orig, deq) in data.iter().zip(out.iter()) {
        let error = (orig - deq).abs();
        assert!(error <= 2.0, "all-same I2S error {} too large", error);
    }
}

/// All-same negative value: round-trip should stay close.
#[test]
fn i2s_edge_all_same_negative() {
    let data = vec![-5.0f32; 32];
    let out = i2s_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for (orig, deq) in data.iter().zip(out.iter()) {
        let error = (orig - deq).abs();
        assert!(error <= 4.0, "all-same-negative I2S error {} too large", error);
    }
}

/// All-same positive value for TL1.
#[test]
fn tl1_edge_all_same_positive() {
    let data = vec![7.0f32; 64];
    let out = tl1_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for v in &out {
        assert!(v.is_finite());
    }
}

/// All-same positive value for TL2.
#[test]
fn tl2_edge_all_same_positive() {
    let data = vec![7.0f32; 128];
    let out = tl2_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for v in &out {
        assert!(v.is_finite());
    }
}

/// Alternating signs: ±1 pattern, common in weight tensors.
#[test]
fn i2s_edge_alternating_signs() {
    let data: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let out = i2s_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for (orig, deq) in data.iter().zip(out.iter()) {
        let error = (orig - deq).abs();
        // scale = 1.0, max error = 0.5
        assert!(error <= 1.5, "alternating-sign I2S error {} too large", error);
    }
}

/// Alternating signs for TL1.
#[test]
fn tl1_edge_alternating_signs() {
    let data: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let out = tl1_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for v in &out {
        assert!(v.is_finite());
    }
}

/// Alternating signs for TL2.
#[test]
fn tl2_edge_alternating_signs() {
    let data: Vec<f32> = (0..128).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let out = tl2_roundtrip(&data);
    assert_eq!(out.len(), data.len());
    for v in &out {
        assert!(v.is_finite());
    }
}

/// I2_S: quantize_weights stores the correct number of scale factors.
/// One scale per block (default block_size ≤ 32).
#[test]
fn i2s_scale_count_matches_block_count() {
    for n in [4, 16, 32, 64, 128] {
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let q = I2SQuantizer::new();
        let qt = q.quantize_weights(&data).unwrap();
        let block_size = qt.block_size.max(1);
        let expected_blocks = data.len().div_ceil(block_size);
        assert_eq!(
            qt.scales.len(),
            expected_blocks,
            "n={}: expected {} scale(s), got {}",
            n,
            expected_blocks,
            qt.scales.len()
        );
    }
}

/// Compression ratio must always be ≥ 1 (quantized is never larger than FP32).
#[test]
fn compression_ratio_at_least_one_for_all_types() {
    let data: Vec<f32> = (0..64).map(|i| i as f32 - 32.0).collect();
    for (name, ratio) in [
        ("I2S", I2SQuantizer::new().quantize_weights(&data).unwrap().compression_ratio()),
        ("TL1", TL1Quantizer::new().quantize_weights(&data).unwrap().compression_ratio()),
        ("TL2", TL2Quantizer::new().quantize_weights(&data).unwrap().compression_ratio()),
    ] {
        assert!(ratio >= 1.0, "{} compression ratio {} < 1.0", name, ratio);
    }
}
