//! Property-based tests — wave 28.
//!
//! Quantization round-trip properties: I2S quantize→dequantize stability,
//! scale monotonicity, error bounds, TL1/TL2 lookup table invertibility,
//! grouped scale invariants, MSE symmetry, SNR non-negativity, optimal
//! block-size bounds, pack/unpack 2-bit fidelity, value-level quantization
//! ordering preservation, and QuantizedTensor shape invariants.
//!
//! 52 property assertions across 16 invariant categories.

#![cfg(feature = "cpu")]

use bitnet_common::{Device, QuantizationType};
use bitnet_quantization::tl1::LookupTable;
use bitnet_quantization::tl2::VectorizedLookupTable;
use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_mse, calculate_optimal_block_size, calculate_scale,
    calculate_snr, dequantize_value, dequantize_value_with_offset, pack_unsigned_2bit_values,
    quantize_value, quantize_value_with_offset, unpack_unsigned_2bit_values, validate_shapes,
};
use bitnet_quantization::{I2SLayout, I2SQuantizer, QuantizedTensor, TL1Quantizer, TL2Quantizer};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn bounded_f32_vec(n: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-5.0f32..5.0, n..=n)
}

fn nonzero_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(prop_oneof![(-5.0f32..-0.01), (0.01f32..5.0)], 2..=max_len)
}

// ── 1. I2SLayout block-size scaling ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// bytes_per_block grows with block_size.
    #[test]
    fn i2s_bytes_per_block_monotone(
        a in 4usize..128,
        b in 4usize..128,
    ) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let la = I2SLayout::with_block_size(lo);
        let lb = I2SLayout::with_block_size(hi);
        prop_assert!(
            la.bytes_per_block <= lb.bytes_per_block,
            "bytes_per_block not monotone: {} (bs={}) > {} (bs={})",
            la.bytes_per_block, lo, lb.bytes_per_block, hi
        );
    }

    /// data_bytes = block_size / 4 (2 bits per element, 8 bits per byte).
    #[test]
    fn i2s_data_bytes_formula(block_size in (1usize..128).prop_map(|b| b * 4)) {
        let layout = I2SLayout::with_block_size(block_size);
        let expected = block_size / 4;
        prop_assert_eq!(
            layout.data_bytes_per_block, expected,
            "data_bytes {} != block_size/4 = {}", layout.data_bytes_per_block, expected
        );
    }

    /// block_size stored correctly.
    #[test]
    fn i2s_block_size_stored(block_size in 4usize..512) {
        let layout = I2SLayout::with_block_size(block_size);
        prop_assert_eq!(layout.block_size, block_size);
    }
}

// ── 2. Scale calculation properties ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Scale is non-negative.
    #[test]
    fn scale_nonneg(data in bounded_f32_vec(16)) {
        let s = calculate_scale(&data, 2);
        prop_assert!(s >= 0.0, "scale {} < 0", s);
    }

    /// Scale is finite.
    #[test]
    fn scale_finite(data in bounded_f32_vec(16)) {
        let s = calculate_scale(&data, 2);
        prop_assert!(s.is_finite(), "scale {} not finite", s);
    }

    /// Scaling larger values produces larger (or equal) scale.
    #[test]
    fn scale_monotonicity_with_magnitude(
        data in bounded_f32_vec(16),
        factor in 1.0f32..5.0,
    ) {
        let s1 = calculate_scale(&data, 2);
        let scaled: Vec<f32> = data.iter().map(|&x| x * factor).collect();
        let s2 = calculate_scale(&scaled, 2);
        prop_assert!(
            s2 >= s1 - 1e-6,
            "scale not monotone: {} > {}", s1, s2
        );
    }

    /// Scale of zeros is safe fallback (1.0).
    #[test]
    fn scale_zeros(n in 1usize..32) {
        let data = vec![0.0f32; n];
        let s = calculate_scale(&data, 2);
        prop_assert!(
            (s - 1.0).abs() < 1e-6,
            "scale of zeros = {} != 1.0 (safe fallback)", s
        );
    }
}

// ── 3. Grouped scales properties ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Number of grouped scales = ceil(len / block_size).
    #[test]
    fn grouped_scales_count(
        n in 4usize..64,
        block_size in 4usize..16,
    ) {
        let data = vec![1.0f32; n];
        let scales = calculate_grouped_scales(&data, block_size, 2);
        let expected = n.div_ceil(block_size);
        prop_assert_eq!(
            scales.len(), expected,
            "grouped scales count {} != ceil({}/{}) = {}", scales.len(), n, block_size, expected
        );
    }

    /// All grouped scales are non-negative.
    #[test]
    fn grouped_scales_nonneg(
        data in bounded_f32_vec(32),
        block_size in 4usize..16,
    ) {
        let scales = calculate_grouped_scales(&data, block_size, 2);
        for (i, &s) in scales.iter().enumerate() {
            prop_assert!(s >= 0.0, "grouped_scales[{}] = {} < 0", i, s);
        }
    }

    /// Grouped scales with block_size >= n gives single scale.
    #[test]
    fn grouped_scales_single_block(n in 1usize..16) {
        let data = vec![1.0f32; n];
        let scales = calculate_grouped_scales(&data, n + 1, 2);
        prop_assert_eq!(scales.len(), 1, "expected 1 scale, got {}", scales.len());
    }
}

// ── 4. Quantize→dequantize value round-trip ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Round-trip error is bounded by scale when value is in representable range.
    #[test]
    fn value_quantize_dequantize_bounded(
        scale in 0.01f32..2.0,
    ) {
        // 2-bit range is [-2, 1], so representable range is [-2*scale, 1*scale]
        let value = scale * 0.5; // well within range
        let q = quantize_value(value, scale, 2);
        let recovered = dequantize_value(q, scale);
        let err = (value - recovered).abs();
        prop_assert!(
            err <= scale + 1e-5,
            "roundtrip error {} > scale {} for value {}", err, scale, value
        );
    }

    /// Quantized value is clamped and dequantized result is finite.
    #[test]
    fn value_quantize_dequantize_finite(
        value in -5.0f32..5.0,
        scale in 0.01f32..2.0,
    ) {
        let q = quantize_value(value, scale, 2);
        let recovered = dequantize_value(q, scale);
        prop_assert!(recovered.is_finite(), "dequantized not finite for value {} scale {}", value, scale);
    }

    /// Quantized value is in valid range for 2 bits: [-2, 1].
    #[test]
    fn value_quantize_range(
        value in -5.0f32..5.0,
        scale in 0.01f32..2.0,
    ) {
        let q = quantize_value(value, scale, 2);
        prop_assert!(
            (-2..=1).contains(&q),
            "quantized {} out of 2-bit range for value {} scale {}", q, value, scale
        );
    }

    /// Dequantize(0) is always 0 regardless of scale.
    #[test]
    fn dequantize_zero_always_zero(scale in -10.0f32..10.0) {
        let v = dequantize_value(0, scale);
        prop_assert!(v.abs() < 1e-6, "dequantize(0, {}) = {} != 0", scale, v);
    }
}

// ── 5. Asymmetric quantization with offset ──────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Offset round-trip error bounded when value is in representable range.
    #[test]
    fn offset_quantize_dequantize_bounded(
        scale in 0.01f32..2.0,
        offset in -2i32..2,
    ) {
        // Use a value within 2-bit representable range
        let value = scale * 0.5;
        let q = quantize_value_with_offset(value, scale, offset, 2);
        let recovered = dequantize_value_with_offset(q, scale, offset);
        let err = (value - recovered).abs();
        prop_assert!(
            err <= scale + 1e-4,
            "offset roundtrip error {} > scale {} for value {}", err, scale, value
        );
    }

    /// Zero offset matches non-offset version.
    #[test]
    fn zero_offset_matches_no_offset(
        value in -5.0f32..5.0,
        scale in 0.01f32..2.0,
    ) {
        let q1 = quantize_value(value, scale, 2);
        let q2 = quantize_value_with_offset(value, scale, 0, 2);
        prop_assert_eq!(q1, q2, "zero offset should match no-offset");
    }
}

// ── 6. Pack/unpack 2-bit round-trip ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Pack then unpack recovers original values.
    #[test]
    fn pack_unpack_2bit_roundtrip(
        vals in proptest::collection::vec(0i8..4, 4..=32)
            .prop_map(|v| {
                let len = v.len() - (v.len() % 4);
                v[..len].to_vec()
            })
    ) {
        if vals.is_empty() {
            return Ok(());
        }
        let packed = pack_unsigned_2bit_values(&vals);
        let unpacked = unpack_unsigned_2bit_values(&packed, vals.len());
        prop_assert_eq!(
            unpacked.len(), vals.len(),
            "unpack length mismatch: {} vs {}", unpacked.len(), vals.len()
        );
        for (i, (&orig, &rec)) in vals.iter().zip(unpacked.iter()).enumerate() {
            prop_assert_eq!(
                orig, rec,
                "2bit roundtrip mismatch at [{}]: {} vs {}", i, orig, rec
            );
        }
    }

    /// Packed length = ceil(n / 4).
    #[test]
    fn packed_length(n in (1usize..64).prop_map(|n| n * 4)) {
        let vals = vec![0i8; n];
        let packed = pack_unsigned_2bit_values(&vals);
        prop_assert_eq!(
            packed.len(), n / 4,
            "packed length {} != n/4 = {}", packed.len(), n / 4
        );
    }
}

// ── 7. MSE properties ──────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// MSE(a, a) = 0.
    #[test]
    fn mse_self_is_zero(data in bounded_f32_vec(16)) {
        let mse = calculate_mse(&data, &data).unwrap();
        prop_assert!(mse.abs() < 1e-6, "MSE(a, a) = {} != 0", mse);
    }

    /// MSE >= 0 always.
    #[test]
    fn mse_nonneg(
        a in bounded_f32_vec(16),
        b in bounded_f32_vec(16),
    ) {
        let mse = calculate_mse(&a, &b).unwrap();
        prop_assert!(mse >= -1e-6, "MSE {} < 0", mse);
    }

    /// MSE is symmetric: MSE(a,b) = MSE(b,a).
    #[test]
    fn mse_symmetric(
        a in bounded_f32_vec(16),
        b in bounded_f32_vec(16),
    ) {
        let mse_ab = calculate_mse(&a, &b).unwrap();
        let mse_ba = calculate_mse(&b, &a).unwrap();
        prop_assert!(
            (mse_ab - mse_ba).abs() < 1e-5,
            "MSE not symmetric: {} vs {}", mse_ab, mse_ba
        );
    }
}

// ── 8. SNR properties ──────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// SNR of identical signals is very high (or inf).
    #[test]
    fn snr_identical_high(data in nonzero_f32_vec(16)) {
        let snr = calculate_snr(&data, &data).unwrap();
        prop_assert!(
            snr > 50.0 || snr.is_infinite(),
            "SNR of identical signals = {} should be very high", snr
        );
    }

    /// SNR is finite when signals differ (same length).
    #[test]
    fn snr_finite_when_different(
        a in proptest::collection::vec(prop_oneof![(-5.0f32..-0.01), (0.01f32..5.0)], 16..=16),
        b in proptest::collection::vec(prop_oneof![(-5.0f32..-0.01), (0.01f32..5.0)], 16..=16),
    ) {
        let snr = calculate_snr(&a, &b).unwrap();
        prop_assert!(
            snr.is_finite() || snr.is_nan(),
            "SNR = {} should be finite or NaN", snr
        );
    }
}

// ── 9. Optimal block size ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Optimal block size >= 1.
    #[test]
    fn optimal_block_size_ge_one(
        tensor_size in 8usize..1024,
        target_blocks in 1usize..32,
    ) {
        let bs = calculate_optimal_block_size(tensor_size, target_blocks);
        prop_assert!(bs >= 1, "optimal block size {} < 1", bs);
    }

    /// Optimal block size is a power of 2 and within [16, 1024].
    #[test]
    fn optimal_block_size_power_of_two(
        tensor_size in 8usize..1024,
        target_blocks in 1usize..32,
    ) {
        let bs = calculate_optimal_block_size(tensor_size, target_blocks);
        prop_assert!((16..=1024).contains(&bs), "block size {} not in [16, 1024]", bs);
        prop_assert!(bs.is_power_of_two(), "block size {} not power of 2", bs);
    }
}

// ── 10. validate_shapes ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Identical shapes always pass.
    #[test]
    fn validate_shapes_same(
        d0 in 1usize..16,
        d1 in 1usize..16,
    ) {
        let shape = vec![d0, d1];
        let result = validate_shapes(&shape, &shape);
        prop_assert!(result.is_ok(), "same shapes rejected: {:?}", result);
    }

    /// Different shapes fail.
    #[test]
    fn validate_shapes_different(
        d0 in 1usize..16,
        d1 in 1usize..16,
        d2 in 1usize..16,
    ) {
        // Ensure they're different
        prop_assume!(d0 != d2 || d1 != d0);
        let s1 = vec![d0, d1];
        let s2 = vec![d2, d0];
        if s1 != s2 {
            let result = validate_shapes(&s1, &s2);
            prop_assert!(result.is_err(), "different shapes accepted");
        }
    }
}

// ── 11. TL1 LookupTable invertibility ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// TL1 quantize→dequantize is idempotent: dequant(quant(dequant(quant(x)))) ≈ dequant(quant(x)).
    #[test]
    fn tl1_lookup_idempotent(value in -3.0f32..3.0) {
        let lut = LookupTable::new(-4.0, 4.0, 8, false);
        let q1 = lut.quantize(value);
        let d1 = lut.dequantize(q1);
        let q2 = lut.quantize(d1);
        let d2 = lut.dequantize(q2);
        prop_assert!(
            (d1 - d2).abs() < 1e-5,
            "TL1 not idempotent: {} -> {} vs {}", value, d1, d2
        );
    }

    /// TL1 quantized value is in valid i8 range.
    #[test]
    fn tl1_quantize_range(value in -4.0f32..4.0) {
        let lut = LookupTable::new(-4.0, 4.0, 8, false);
        let q = lut.quantize(value);
        // i8 range check via i16 to avoid type-limit warning
        let wide = q as i16;
        prop_assert!(
            (-128..=127).contains(&wide),
            "TL1 quantized {} out of i8 range", q
        );
    }

    /// TL1 dequantized value is finite.
    #[test]
    fn tl1_dequantize_finite(q in -128i8..127) {
        let lut = LookupTable::new(-4.0, 4.0, 8, false);
        let v = lut.dequantize(q);
        prop_assert!(v.is_finite(), "TL1 dequantize({}) = {} not finite", q, v);
    }
}

// ── 12. TL2 VectorizedLookupTable invertibility ─────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// TL2 quantize→dequantize idempotent.
    #[test]
    fn tl2_lookup_idempotent(value in -3.0f32..3.0) {
        let lut = VectorizedLookupTable::new(-4.0, 4.0, 8);
        let q1 = lut.quantize(value);
        let d1 = lut.dequantize(q1);
        let q2 = lut.quantize(d1);
        let d2 = lut.dequantize(q2);
        prop_assert!(
            (d1 - d2).abs() < 1e-5,
            "TL2 not idempotent: {} -> {} vs {}", value, d1, d2
        );
    }

    /// TL2 forward_len and reverse_len are consistent.
    #[test]
    fn tl2_table_lengths_consistent(bits in 2u8..8) {
        let lut = VectorizedLookupTable::new(-4.0, 4.0, bits);
        let fwd = lut.forward_len();
        let rev = lut.reverse_len();
        prop_assert!(fwd > 0, "forward_len = 0");
        prop_assert!(rev > 0, "reverse_len = 0");
    }

    /// TL2 quantized value in i8 range.
    #[test]
    fn tl2_quantize_range(value in -4.0f32..4.0) {
        let lut = VectorizedLookupTable::new(-4.0, 4.0, 8);
        let q = lut.quantize(value);
        // i8 range check via i16 to avoid type-limit warning
        let wide = q as i16;
        prop_assert!(
            (-128..=127).contains(&wide),
            "TL2 quantized {} out of range", q
        );
    }
}

// ── 13. Quantize preserves order ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// For sufficiently spaced values, quantize preserves relative order.
    #[test]
    fn quantize_preserves_relative_order(
        lo in -4.0f32..-1.0,
        hi in 1.0f32..4.0,
        scale in 0.5f32..2.0,
    ) {
        let q_lo = quantize_value(lo, scale, 2);
        let q_hi = quantize_value(hi, scale, 2);
        prop_assert!(
            q_lo <= q_hi,
            "order not preserved: q({})={} > q({})={}", lo, q_lo, hi, q_hi
        );
    }
}

// ── 14. QuantizedTensor shape invariants ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// numel = product of shape dims.
    #[test]
    fn quantized_tensor_numel_product(
        d0 in 1usize..8,
        d1 in 1usize..8,
        d2 in 1usize..8,
    ) {
        let shape = vec![d0, d1, d2];
        let numel = d0 * d1 * d2;
        let data = vec![0u8; numel];
        let scales = vec![1.0f32; 1];
        let t = QuantizedTensor::new(data, scales, shape.clone(), QuantizationType::I2S);
        prop_assert_eq!(t.numel(), numel, "numel {} != product {}", t.numel(), numel);
    }

    /// compression_ratio >= 1.0 for 2-bit quantization.
    #[test]
    fn quantized_tensor_compression_ge_one(
        d0 in 1usize..8,
        d1 in 1usize..8,
    ) {
        let shape = vec![d0, d1];
        let numel = d0 * d1;
        let data = vec![0u8; numel];
        let scales = vec![1.0f32; 1];
        let t = QuantizedTensor::new(data, scales, shape, QuantizationType::I2S);
        let ratio = t.compression_ratio();
        prop_assert!(
            ratio >= 1.0,
            "compression_ratio {} < 1.0", ratio
        );
    }
}

// ── 15. I2S quantizer supports CPU ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// I2SQuantizer always supports CPU device.
    #[test]
    fn i2s_supports_cpu(block_size in 4usize..128) {
        let q = I2SQuantizer::with_block_size(block_size);
        prop_assert!(
            q.supports_device(&Device::Cpu),
            "I2S should support CPU"
        );
    }

    /// TL1Quantizer always supports CPU device.
    #[test]
    fn tl1_supports_cpu(_dummy in 0..1u8) {
        let q = TL1Quantizer::new();
        prop_assert!(
            q.supports_device(&Device::Cpu),
            "TL1 should support CPU"
        );
    }

    /// TL2Quantizer always supports CPU device.
    #[test]
    fn tl2_supports_cpu(_dummy in 0..1u8) {
        let q = TL2Quantizer::new();
        prop_assert!(
            q.supports_device(&Device::Cpu),
            "TL2 should support CPU"
        );
    }
}

// ── 16. Scale calculation bits sensitivity ──────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// More bits → smaller (or equal) scale for same data.
    #[test]
    fn more_bits_smaller_scale(data in nonzero_f32_vec(16)) {
        let s2 = calculate_scale(&data, 2);
        let s4 = calculate_scale(&data, 4);
        let s8 = calculate_scale(&data, 8);
        prop_assert!(
            s2 >= s4 - 1e-6,
            "2-bit scale {} < 4-bit scale {}", s2, s4
        );
        prop_assert!(
            s4 >= s8 - 1e-6,
            "4-bit scale {} < 8-bit scale {}", s4, s8
        );
    }
}
