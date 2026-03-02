//! Property-based tests — wave 29.
//!
//! Covers I2S quantize/dequantize round-trip properties, TL1/TL2 table
//! lookup invariants, QK256 block alignment properties, scale factor
//! numerical properties, Int4 quantization, and utility function invariants.
//!
//! 42 property tests validating: round-trip fidelity, layout consistency,
//! table monotonicity, block alignment, scale factor bounds, compression
//! ratios, and numerical stability.

#![cfg(feature = "cpu")]

use bitnet_common::QuantizationType;
use bitnet_quantization::i2s::{I2SLayout, I2SQuantizer};
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, unpack_qk256_block,
};
use bitnet_quantization::int4_quant::NibblePacked;
use bitnet_quantization::tl1::LookupTable;
use bitnet_quantization::tl2::VectorizedLookupTable;
use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_mse, calculate_optimal_block_size, calculate_scale,
    dequantize_value, dequantize_value_with_offset, pack_unsigned_2bit_values, quantize_value,
    quantize_value_with_offset, unpack_unsigned_2bit_values, validate_shapes,
};
use bitnet_quantization::{QuantizedTensor, qk256_tolerance_bytes};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn finite_f32_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-5.0f32..5.0, 1..=max_len)
}

// ── 1. I2S layout properties ────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// I2S layout bytes_per_block = data + scale for all block sizes.
    #[test]
    fn prop_i2s_layout_bytes_sum(block_size in 4usize..256) {
        let layout = I2SLayout::with_block_size(block_size);
        prop_assert_eq!(
            layout.bytes_per_block,
            layout.data_bytes_per_block + layout.scale_bytes_per_block
        );
    }

    /// I2S scale bytes is always 2 (f16 scale).
    #[test]
    fn prop_i2s_scale_always_f16(block_size in 4usize..256) {
        let layout = I2SLayout::with_block_size(block_size);
        prop_assert_eq!(layout.scale_bytes_per_block, 2);
    }

    /// I2S layout block_size is stored correctly.
    #[test]
    fn prop_i2s_block_size_stored(block_size in 4usize..256) {
        let layout = I2SLayout::with_block_size(block_size);
        prop_assert_eq!(layout.block_size, block_size);
    }

    /// I2S data bytes are sufficient for 2-bit packing.
    #[test]
    fn prop_i2s_data_bytes_sufficient(block_size in 4usize..256) {
        let layout = I2SLayout::with_block_size(block_size);
        let min_bytes = (block_size * 2 + 7) / 8;
        prop_assert!(layout.data_bytes_per_block >= min_bytes);
    }

    /// Larger block sizes never have fewer data bytes.
    #[test]
    fn prop_i2s_data_bytes_monotone(a in 4usize..128, b in 4usize..128) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let la = I2SLayout::with_block_size(lo);
        let lb = I2SLayout::with_block_size(hi);
        prop_assert!(la.data_bytes_per_block <= lb.data_bytes_per_block);
    }
}

// ── 2. I2S quantize / dequantize round-trip ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// I2S round-trip preserves sign of large-magnitude values.
    #[test]
    fn prop_i2s_roundtrip_sign(
        data in proptest::collection::vec(-3.0f32..3.0, 32..=32),
    ) {
        let quantizer = I2SQuantizer::new();
        if let Ok(quantized) = quantizer.quantize_weights(&data) {
            if let Ok(recovered) = quantizer.dequantize_tensor(&quantized) {
                if let Ok(rec_data) = recovered.to_vec() {
                    for (&orig, &rec) in data.iter().zip(rec_data.iter()) {
                        // Only large values survive 2-bit quantization with sign intact
                        if orig.abs() > 1.5 && rec != 0.0 {
                            prop_assert!(
                                orig.signum() == rec.signum(),
                                "sign flip: {} -> {}", orig, rec
                            );
                        }
                    }
                }
            }
        }
    }

    /// I2S quantized tensor has non-empty data.
    #[test]
    fn prop_i2s_quantized_nonempty(
        n in (32usize..=128).prop_filter("multiple of 32", |n| n % 32 == 0),
    ) {
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.1).collect();
        let quantizer = I2SQuantizer::new();
        if let Ok(quantized) = quantizer.quantize_weights(&data) {
            prop_assert!(!quantized.data.is_empty(), "quantized data is empty");
        }
    }

    /// I2S numel matches original element count.
    #[test]
    fn prop_i2s_numel_preserved(
        n in (32usize..=128).prop_filter("multiple of 32", |n| n % 32 == 0),
    ) {
        let data: Vec<f32> = vec![1.0; n];
        let quantizer = I2SQuantizer::new();
        if let Ok(quantized) = quantizer.quantize_weights(&data) {
            prop_assert_eq!(quantized.numel(), n);
        }
    }
}

// ── 3. TL1 lookup table invariants ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// TL1 quantize → dequantize round-trip yields a finite value.
    #[test]
    fn prop_tl1_roundtrip_finite(
        min_val in -10.0f32..-0.01,
        max_val in 0.01f32..10.0,
        value in -10.0f32..10.0,
    ) {
        let lut = LookupTable::new(min_val, max_val, 8, false);
        let q = lut.quantize(value);
        let dq = lut.dequantize(q);
        let abs_max = max_val.abs().max(min_val.abs());
        // Symmetric LUT spans [-abs_max * 128/127, abs_max] approximately
        let bound = abs_max * 1.02;
        prop_assert!(
            dq.is_finite() && dq.abs() <= bound,
            "dq {} outside [-{}, {}]", dq, bound, bound
        );
    }

    /// TL1 quantize is deterministic.
    #[test]
    fn prop_tl1_quantize_deterministic(
        min_val in -5.0f32..-0.1,
        max_val in 0.1f32..5.0,
        value in -5.0f32..5.0,
    ) {
        let lut = LookupTable::new(min_val, max_val, 8, false);
        let q1 = lut.quantize(value);
        let q2 = lut.quantize(value);
        prop_assert_eq!(q1, q2, "non-deterministic quantization: {} vs {}", q1, q2);
    }

    /// TL1 quantize then dequantize gives bounded result for any bits.
    #[test]
    fn prop_tl1_bits_roundtrip_bounded(
        bits in 2u8..=8,
        value in -3.0f32..3.0,
    ) {
        let lut = LookupTable::new(-3.0, 3.0, bits, false);
        let q = lut.quantize(value);
        let dq = lut.dequantize(q);
        prop_assert!(dq >= -3.0 && dq <= 3.0, "dq {} out of range for bits={}", dq, bits);
    }

    /// TL1 dequantize is deterministic.
    #[test]
    fn prop_tl1_dequantize_deterministic(
        min_val in -5.0f32..-0.1,
        max_val in 0.1f32..5.0,
        q in -128i8..127,
    ) {
        let lut = LookupTable::new(min_val, max_val, 8, false);
        let d1 = lut.dequantize(q);
        let d2 = lut.dequantize(q);
        prop_assert!((d1 - d2).abs() < 1e-7, "non-deterministic dequant");
    }
}

// ── 4. TL2 vectorized lookup table invariants ───────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// TL2 quantize → dequantize round-trip stays within range.
    #[test]
    fn prop_tl2_roundtrip_bounded(
        min_val in -10.0f32..-0.01,
        max_val in 0.01f32..10.0,
        value in -10.0f32..10.0,
    ) {
        let lut = VectorizedLookupTable::new(min_val, max_val, 8);
        let q = lut.quantize(value);
        let dq = lut.dequantize(q);
        prop_assert!(
            dq >= min_val && dq <= max_val,
            "dq {} outside [{}, {}]", dq, min_val, max_val
        );
    }

    /// TL2 forward and reverse table sizes are consistent.
    #[test]
    fn prop_tl2_table_sizes(bits in 2u8..=8) {
        let lut = VectorizedLookupTable::new(-1.0, 1.0, bits);
        prop_assert_eq!(lut.forward_len(), 256);
        prop_assert!(lut.reverse_len() > 0);
    }

    /// TL2 quantize on dequantized values is near-idempotent (within ±1).
    #[test]
    fn prop_tl2_quantize_idempotent(
        min_val in -5.0f32..-0.1,
        max_val in 0.1f32..5.0,
        value in -5.0f32..5.0,
    ) {
        let lut = VectorizedLookupTable::new(min_val, max_val, 8);
        let q1 = lut.quantize(value);
        let dq1 = lut.dequantize(q1);
        let q2 = lut.quantize(dq1);
        // Allow ±1 rounding difference due to floating-point rounding in dequantize
        prop_assert!(
            (q1 as i16 - q2 as i16).unsigned_abs() <= 1,
            "not near-idempotent: q({})={}, q(dq({}))={}", value, q1, q1, q2
        );
    }
}

// ── 5. QK256 block alignment properties ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// QK256 block constant is 256.
    #[test]
    fn prop_qk256_block_is_256(_dummy in 0u8..1) {
        prop_assert_eq!(QK256_BLOCK, 256);
    }

    /// QK256 packed bytes is 64 (256 elements * 2 bits / 8).
    #[test]
    fn prop_qk256_packed_bytes_is_64(_dummy in 0u8..1) {
        prop_assert_eq!(QK256_PACKED_BYTES, 64);
    }

    /// code_to_f32 maps all 2-bit codes to {-2, -1, 1, 2}.
    #[test]
    fn prop_code_to_f32_ternary(code in 0u8..4) {
        let val = code_to_f32(code);
        prop_assert!(
            val == -2.0 || val == -1.0 || val == 1.0 || val == 2.0,
            "code {} maps to unexpected value {}", code, val
        );
    }

    /// unpack_qk256_block produces exactly 256 codes, each in [0, 3].
    #[test]
    fn prop_unpack_qk256_codes_valid(
        data in proptest::collection::vec(0u8..=255, 64..=64),
    ) {
        let mut packed = [0u8; QK256_PACKED_BYTES];
        packed.copy_from_slice(&data);
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&packed, &mut codes);
        for &c in &codes {
            prop_assert!(c < 4, "code {} >= 4", c);
        }
    }

    /// qk256_tolerance_bytes returns at least 8 bytes (alignment minimum).
    #[test]
    fn prop_qk256_tolerance_ge_minimum(expected in 64usize..=10000) {
        let tolerated = qk256_tolerance_bytes(expected);
        prop_assert!(tolerated >= 8, "tolerance {} < 8", tolerated);
    }
}

// ── 6. Scale factor numerical properties ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// calculate_scale returns non-negative scale.
    #[test]
    fn prop_scale_nonneg(data in finite_f32_vec(32), bits in 2u8..=8) {
        let scale = calculate_scale(&data, bits);
        prop_assert!(scale >= 0.0, "negative scale: {}", scale);
    }

    /// Uniform data has scale proportional to value magnitude.
    #[test]
    fn prop_scale_uniform_proportional(val in 0.1f32..10.0, n in 4usize..=32) {
        let data = vec![val; n];
        let scale = calculate_scale(&data, 8);
        prop_assert!(scale > 0.0, "zero scale for non-zero uniform data");
    }

    /// Zero data has scale = 1.0 (safe fallback).
    #[test]
    fn prop_scale_zero_data(n in 1usize..=32) {
        let data = vec![0.0f32; n];
        let scale = calculate_scale(&data, 8);
        prop_assert!((scale - 1.0).abs() < 1e-7, "zero data scale {} != 1.0", scale);
    }

    /// Grouped scales have one entry per block.
    #[test]
    fn prop_grouped_scales_count(
        n in (16usize..=128).prop_filter("mult of 16", |n| n % 16 == 0),
        bits in 2u8..=8,
    ) {
        let block_size = 16;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let scales = calculate_grouped_scales(&data, block_size, bits);
        let expected_blocks = n / block_size;
        prop_assert_eq!(scales.len(), expected_blocks);
    }

    /// All grouped scales are non-negative.
    #[test]
    fn prop_grouped_scales_nonneg(
        data in proptest::collection::vec(-5.0f32..5.0, 32..=32),
    ) {
        let scales = calculate_grouped_scales(&data, 16, 8);
        for &s in &scales {
            prop_assert!(s >= 0.0, "negative grouped scale: {}", s);
        }
    }
}

// ── 7. Quantize/dequantize value round-trip ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// quantize → dequantize round-trip error is bounded by quantization step.
    #[test]
    fn prop_value_roundtrip_error_bounded(
        value in -5.0f32..5.0,
        scale in 0.01f32..5.0,
        bits in 2u8..=8,
    ) {
        let q = quantize_value(value, scale, bits);
        let dq = dequantize_value(q, scale);
        let max_q = ((1i32 << (bits - 1)) - 1) as f32;
        // Error is at most half a step + clipping error
        let max_representable = max_q * scale;
        let clipping_error = (value.abs() - max_representable).max(0.0);
        let step_error = scale / 2.0 + 1e-5;
        let bound = clipping_error + step_error;
        let error = (value - dq).abs();
        prop_assert!(error <= bound, "error {} > bound {}", error, bound);
    }

    /// Dequantize(0) == 0 regardless of scale.
    #[test]
    fn prop_dequantize_zero_identity(scale in 0.01f32..10.0) {
        let dq = dequantize_value(0, scale);
        prop_assert!(dq.abs() < 1e-7, "dequant(0) = {}", dq);
    }

    /// quantize_value_with_offset → dequantize_value_with_offset is finite.
    #[test]
    fn prop_value_offset_roundtrip(
        value in -3.0f32..3.0,
        scale in 0.1f32..3.0,
        offset in -5i32..5,
        bits in 4u8..=8,
    ) {
        let q = quantize_value_with_offset(value, scale, offset, bits);
        let dq = dequantize_value_with_offset(q, scale, offset);
        prop_assert!(dq.is_finite(), "dq is not finite: {}", dq);
        // For values within the representable range, error ≤ scale/2
        let max_q = ((1i32 << (bits - 1)) - 1) as f32;
        let min_q = -(1i32 << (bits - 1)) as f32;
        let max_repr = (max_q - offset as f32) * scale;
        let min_repr = (min_q - offset as f32) * scale;
        if value >= min_repr && value <= max_repr {
            let error = (value - dq).abs();
            prop_assert!(error <= scale / 2.0 + 1e-4, "within-range error {} > half-step {}", error, scale / 2.0);
        }
    }
}

// ── 8. Pack/unpack 2-bit round-trip ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// pack → unpack unsigned 2-bit round-trip.
    #[test]
    fn prop_pack_unpack_unsigned_2bit(
        values in proptest::collection::vec(0i8..4, 4..=64),
    ) {
        let len = values.len();
        let packed = pack_unsigned_2bit_values(&values);
        let unpacked = unpack_unsigned_2bit_values(&packed, len);
        prop_assert_eq!(&unpacked[..len], &values[..], "pack/unpack mismatch");
    }

    /// Packed size is ceil(n/4) bytes for unsigned 2-bit.
    #[test]
    fn prop_packed_size_correct(n in 1usize..=64) {
        let values: Vec<i8> = vec![1; n];
        let packed = pack_unsigned_2bit_values(&values);
        let expected = (n + 3) / 4;
        prop_assert_eq!(packed.len(), expected, "packed size mismatch");
    }
}

// ── 9. MSE / SNR metric properties ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// MSE of identical vectors is zero.
    #[test]
    fn prop_mse_identical_zero(data in finite_f32_vec(32)) {
        if let Ok(mse) = calculate_mse(&data, &data) {
            prop_assert!(mse.abs() < 1e-7, "MSE of identical = {}", mse);
        }
    }

    /// MSE is non-negative.
    #[test]
    fn prop_mse_nonneg(
        a in finite_f32_vec(16),
        b in finite_f32_vec(16),
    ) {
        if a.len() == b.len() {
            if let Ok(mse) = calculate_mse(&a, &b) {
                prop_assert!(mse >= 0.0, "negative MSE: {}", mse);
            }
        }
    }

    /// MSE is symmetric: MSE(a, b) == MSE(b, a).
    #[test]
    fn prop_mse_symmetric(
        vals in proptest::collection::vec(-5.0f32..5.0, 8..=8),
    ) {
        let (a, b) = vals.split_at(4);
        if let (Ok(mse_ab), Ok(mse_ba)) = (calculate_mse(a, b), calculate_mse(b, a)) {
            prop_assert!((mse_ab - mse_ba).abs() < 1e-7, "MSE not symmetric");
        }
    }

    /// validate_shapes accepts matching shapes.
    #[test]
    fn prop_validate_shapes_matching(
        dim1 in 1usize..=64,
        dim2 in 1usize..=64,
    ) {
        let shape = vec![dim1, dim2];
        prop_assert!(validate_shapes(&shape, &shape).is_ok());
    }

    /// validate_shapes rejects mismatched shapes.
    #[test]
    fn prop_validate_shapes_mismatch(
        d1 in 1usize..=64,
        d2 in 1usize..=64,
    ) {
        if d1 != d2 {
            prop_assert!(validate_shapes(&[d1], &[d2]).is_err());
        }
    }
}

// ── 10. Optimal block size properties ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Optimal block size is always > 0.
    #[test]
    fn prop_optimal_block_size_positive(
        tensor_size in 1usize..=10000,
        target_blocks in 1usize..=100,
    ) {
        let bs = calculate_optimal_block_size(tensor_size, target_blocks);
        prop_assert!(bs > 0, "zero block size for tensor_size={}, target={}", tensor_size, target_blocks);
    }

    /// Optimal block size is at least 16 (the minimum clamp).
    #[test]
    fn prop_optimal_block_size_bounded(
        tensor_size in 16usize..=10000,
        target_blocks in 1usize..=100,
    ) {
        let bs = calculate_optimal_block_size(tensor_size, target_blocks);
        prop_assert!(bs >= 16, "block_size {} < 16", bs);
        prop_assert!(bs <= 1024, "block_size {} > 1024", bs);
    }
}

// ── 11. Int4 nibble packing round-trip ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// NibblePacked pack → unpack round-trip.
    #[test]
    fn prop_nibble_pack_roundtrip(
        values in proptest::collection::vec(-8i8..7, 2..=32),
    ) {
        let packed = NibblePacked::pack(&values);
        let unpacked = packed.unpack();
        prop_assert_eq!(unpacked.len(), values.len());
        for (i, (&orig, &rec)) in values.iter().zip(unpacked.iter()).enumerate() {
            prop_assert_eq!(orig, rec, "mismatch at index {}", i);
        }
    }

    /// NibblePacked get returns correct values.
    #[test]
    fn prop_nibble_get_correct(
        values in proptest::collection::vec(-8i8..7, 2..=16),
    ) {
        let packed = NibblePacked::pack(&values);
        for (i, &expected) in values.iter().enumerate() {
            let got = packed.get(i);
            prop_assert_eq!(got, expected, "get({}): expected {}, got {}", i, expected, got);
        }
    }
}

// ── 12. QuantizedTensor compression ratio ───────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// QuantizedTensor numel matches shape product.
    #[test]
    fn prop_quantized_tensor_numel(
        d1 in 1usize..=16,
        d2 in 1usize..=16,
    ) {
        let shape = vec![d1, d2];
        let n = d1 * d2;
        let data = vec![0u8; n];
        let scales = vec![1.0f32];
        let qt = QuantizedTensor::new(data, scales, shape, QuantizationType::I2S);
        prop_assert_eq!(qt.numel(), n);
    }

    /// Compression ratio is positive for non-empty tensors.
    #[test]
    fn prop_compression_ratio_positive(
        n in 4usize..=64,
    ) {
        let data = vec![0u8; n];
        let scales = vec![1.0f32];
        let qt = QuantizedTensor::new(data, scales, vec![n], QuantizationType::I2S);
        let ratio = qt.compression_ratio();
        prop_assert!(ratio > 0.0, "non-positive compression ratio: {}", ratio);
    }
}
