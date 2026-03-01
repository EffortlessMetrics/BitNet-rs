//! Property-based tests — wave 16.
//!
//! New invariants for quantization utilities, I2S layout, QuantizedTensor,
//! grouped scales, MSE/SNR metrics, and optimal block size calculation.

#![cfg(feature = "cpu")]

use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_mse, calculate_optimal_block_size, calculate_scale,
    calculate_snr, dequantize_value, dequantize_value_with_offset, pack_unsigned_2bit_values,
    quantize_value, quantize_value_with_offset, unpack_unsigned_2bit_values, validate_shapes,
};
use bitnet_quantization::{I2SLayout, QuantizedTensor};

use bitnet_common::QuantizationType;
use proptest::prelude::*;

// ── I2SLayout invariants ────────────────────────────────────────────────────

proptest! {
    /// bytes_per_block = data_bytes + scale_bytes for any block size.
    #[test]
    fn i2s_layout_bytes_sum(block_size in 4usize..512) {
        let layout = I2SLayout::with_block_size(block_size);
        prop_assert_eq!(
            layout.bytes_per_block,
            layout.data_bytes_per_block + layout.scale_bytes_per_block,
            "bytes_per_block must equal data + scale bytes"
        );
    }

    /// Scale bytes is always 2 (f16).
    #[test]
    fn i2s_layout_scale_always_2(block_size in 4usize..512) {
        let layout = I2SLayout::with_block_size(block_size);
        prop_assert_eq!(layout.scale_bytes_per_block, 2);
    }

    /// data_bytes >= ceil(block_size * 2 / 8) — enough bits for 2-bit values.
    #[test]
    fn i2s_layout_data_bytes_sufficient(block_size in 4usize..512) {
        let layout = I2SLayout::with_block_size(block_size);
        let min_bytes = (block_size * 2).div_ceil(8);
        prop_assert!(
            layout.data_bytes_per_block >= min_bytes,
            "data_bytes {} < min {} for block_size {}",
            layout.data_bytes_per_block, min_bytes, block_size
        );
    }

    /// Monotonicity: larger block_size → more data bytes.
    #[test]
    fn i2s_layout_data_bytes_monotone(
        a in 4usize..256,
        b in 4usize..256,
    ) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let la = I2SLayout::with_block_size(lo);
        let lb = I2SLayout::with_block_size(hi);
        prop_assert!(
            la.data_bytes_per_block <= lb.data_bytes_per_block,
            "data bytes not monotone: {} (bs={}) > {} (bs={})",
            la.data_bytes_per_block, lo, lb.data_bytes_per_block, hi
        );
    }

    /// Default layout has block_size 32.
    #[test]
    fn i2s_layout_default_block_size(_dummy in 0u8..1) {
        let layout = I2SLayout::default();
        prop_assert_eq!(layout.block_size, 32);
        prop_assert_eq!(layout.bytes_per_block, 10);
        prop_assert_eq!(layout.data_bytes_per_block, 8);
    }
}

// ── QuantizedTensor properties ──────────────────────────────────────────────

proptest! {
    /// numel equals product of shape dimensions.
    #[test]
    fn quantized_tensor_numel(
        d0 in 1usize..16,
        d1 in 1usize..16,
    ) {
        let shape = vec![d0, d1];
        let numel = d0 * d1;
        let qt = QuantizedTensor::new(
            vec![0u8; numel / 4 + 1],
            vec![1.0f32; 1],
            shape.clone(),
            QuantizationType::I2S,
        );
        prop_assert_eq!(qt.numel(), numel);
    }

    /// compression_ratio >= 1.0 (never expands).
    #[test]
    fn quantized_tensor_compression_ratio_ge_one(
        n in 32usize..1024,
    ) {
        let qt = QuantizedTensor::new(
            vec![0u8; n / 4],
            vec![1.0f32; n / 32],
            vec![n],
            QuantizationType::I2S,
        );
        prop_assert!(
            qt.compression_ratio() >= 1.0,
            "compression_ratio {} < 1.0", qt.compression_ratio()
        );
    }

    /// new_with_params preserves all fields.
    #[test]
    fn quantized_tensor_new_with_params(
        block_size in 4usize..128,
    ) {
        let data = vec![0xABu8; 16];
        let scales = vec![0.5f32; 4];
        let zp = vec![1i32; 4];
        let shape = vec![64];
        let qt = QuantizedTensor::new_with_params(
            data.clone(),
            scales.clone(),
            Some(zp.clone()),
            shape.clone(),
            QuantizationType::TL2,
            block_size,
        );
        prop_assert_eq!(&qt.data, &data);
        prop_assert_eq!(&qt.scales, &scales);
        prop_assert_eq!(qt.zero_points.as_ref().unwrap(), &zp);
        prop_assert_eq!(&qt.shape, &shape);
        prop_assert_eq!(qt.qtype, QuantizationType::TL2);
        prop_assert_eq!(qt.block_size, block_size);
    }
}

// ── Grouped scales properties ───────────────────────────────────────────────

proptest! {
    /// Number of grouped scales = ceil(data.len() / block_size).
    #[test]
    fn grouped_scales_count(
        data in prop::collection::vec(-10.0f32..10.0, 1..256),
        block_size in 1usize..64,
    ) {
        let scales = calculate_grouped_scales(&data, block_size, 2);
        let expected = data.len().div_ceil(block_size);
        prop_assert_eq!(
            scales.len(), expected,
            "expected {} scales, got {}", expected, scales.len()
        );
    }

    /// All grouped scales are positive and finite for finite data.
    #[test]
    fn grouped_scales_positive_finite(
        data in prop::collection::vec(-100.0f32..100.0, 1..128),
        block_size in 1usize..32,
    ) {
        let scales = calculate_grouped_scales(&data, block_size, 2);
        for (i, &s) in scales.iter().enumerate() {
            prop_assert!(s.is_finite(), "scale[{}] is not finite: {}", i, s);
            prop_assert!(s > 0.0, "scale[{}] is not positive: {}", i, s);
        }
    }

    /// Grouped scales with block_size == data.len() equals single-block scale.
    #[test]
    fn grouped_scales_single_block(
        data in prop::collection::vec(-10.0f32..10.0, 1..64),
    ) {
        let scales = calculate_grouped_scales(&data, data.len(), 2);
        prop_assert_eq!(scales.len(), 1);
        let single = calculate_scale(&data, 2);
        prop_assert!(
            (scales[0] - single).abs() < 1e-10,
            "single-block scale mismatch: {} vs {}", scales[0], single
        );
    }
}

// ── quantize_value / dequantize_value roundtrip ─────────────────────────────

proptest! {
    /// Quantize→dequantize produces a result within ±scale/2 of the clamped
    /// representable range. For values that saturate the quantized range, the
    /// error may be large, so we only check within the representable domain.
    #[test]
    fn quantize_dequantize_bounded_error(
        value in -5.0f32..5.0,
        scale in 0.1f32..5.0,
        bits in 2u8..8,
    ) {
        let max_q = (1i32 << (bits - 1)) - 1;
        let min_q = -(1i32 << (bits - 1));
        let max_representable = max_q as f32 * scale;
        let min_representable = min_q as f32 * scale;
        // Only check error bound if value is within representable range
        if value >= min_representable && value <= max_representable {
            let q = quantize_value(value, scale, bits);
            let dq = dequantize_value(q, scale);
            let err = (value - dq).abs();
            prop_assert!(
                err <= scale / 2.0 + 1e-6,
                "error {} > scale/2 {} for value={}, q={}, dq={}",
                err, scale / 2.0, value, q, dq
            );
        }
    }

    /// Quantized values stay within the signed bit range.
    #[test]
    fn quantize_value_range(
        value in -100.0f32..100.0,
        scale in 0.01f32..10.0,
        bits in 2u8..8,
    ) {
        let q = quantize_value(value, scale, bits);
        let max_val = (1i8 << (bits - 1)) - 1;
        let min_val = -(1i8 << (bits - 1));
        prop_assert!(q >= min_val && q <= max_val,
            "q={} outside [{}, {}] for bits={}", q, min_val, max_val, bits);
    }

    /// Order preservation: if a > b then quantize(a) >= quantize(b) with same scale.
    #[test]
    fn quantize_preserves_order(
        a in -10.0f32..10.0,
        b in -10.0f32..10.0,
        scale in 0.01f32..5.0,
    ) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let qlo = quantize_value(lo, scale, 8);
        let qhi = quantize_value(hi, scale, 8);
        prop_assert!(qlo <= qhi,
            "order not preserved: q({})={} > q({})={}", lo, qlo, hi, qhi);
    }

    /// dequantize of zero is always zero.
    #[test]
    fn dequantize_zero_is_zero(scale in -100.0f32..100.0) {
        let dq = dequantize_value(0, scale);
        prop_assert_eq!(dq, 0.0);
    }
}

// ── Asymmetric quantize/dequantize ──────────────────────────────────────────

proptest! {
    /// With offset=0, asymmetric functions match symmetric ones.
    #[test]
    fn asymmetric_zero_offset_matches_symmetric(
        value in -10.0f32..10.0,
        scale in 0.01f32..5.0,
        bits in 2u8..8,
    ) {
        let q_sym = quantize_value(value, scale, bits);
        let q_asym = quantize_value_with_offset(value, scale, 0, bits);
        prop_assert_eq!(q_sym, q_asym);

        let dq_sym = dequantize_value(q_sym, scale);
        let dq_asym = dequantize_value_with_offset(q_asym, scale, 0);
        prop_assert!(
            (dq_sym - dq_asym).abs() < 1e-10,
            "dequantize mismatch: {} vs {}", dq_sym, dq_asym
        );
    }
}

// ── Unsigned 2-bit pack/unpack roundtrip ────────────────────────────────────

proptest! {
    /// Unsigned pack→unpack is identity for values in [0, 3].
    #[test]
    fn unsigned_pack_unpack_roundtrip(
        values in prop::collection::vec(0i8..=3i8, 4..64),
    ) {
        let packed = pack_unsigned_2bit_values(&values);
        let unpacked = unpack_unsigned_2bit_values(&packed, values.len());
        prop_assert_eq!(&unpacked, &values);
    }

    /// Unsigned packed length is ceil(n / 4).
    #[test]
    fn unsigned_packed_length(n in 1usize..256) {
        let values = vec![0i8; n];
        let packed = pack_unsigned_2bit_values(&values);
        prop_assert_eq!(packed.len(), n.div_ceil(4));
    }
}

// ── MSE / SNR properties ────────────────────────────────────────────────────

proptest! {
    /// MSE of identical vectors is zero.
    #[test]
    fn mse_identical_is_zero(
        data in prop::collection::vec(-10.0f32..10.0, 1..64),
    ) {
        let mse = calculate_mse(&data, &data).unwrap();
        prop_assert!(mse.abs() < 1e-10, "MSE of identical data should be ~0, got {}", mse);
    }

    /// MSE is always non-negative.
    #[test]
    fn mse_non_negative(
        a in prop::collection::vec(-10.0f32..10.0, 1..64),
        b in prop::collection::vec(-10.0f32..10.0, 1..64),
    ) {
        if a.len() == b.len() {
            let mse = calculate_mse(&a, &b).unwrap();
            prop_assert!(mse >= 0.0, "MSE must be non-negative, got {}", mse);
        }
    }

    /// MSE is symmetric: MSE(a, b) == MSE(b, a).
    #[test]
    fn mse_symmetric(
        a in prop::collection::vec(-10.0f32..10.0, 8..32),
    ) {
        let b: Vec<f32> = a.iter().map(|x| x + 0.1).collect();
        let mse_ab = calculate_mse(&a, &b).unwrap();
        let mse_ba = calculate_mse(&b, &a).unwrap();
        prop_assert!(
            (mse_ab - mse_ba).abs() < 1e-6,
            "MSE not symmetric: {} vs {}", mse_ab, mse_ba
        );
    }

    /// MSE returns error for mismatched lengths.
    #[test]
    fn mse_mismatched_lengths_error(
        a_len in 1usize..32,
        b_len in 1usize..32,
    ) {
        prop_assume!(a_len != b_len);
        let a = vec![0.0f32; a_len];
        let b = vec![0.0f32; b_len];
        prop_assert!(calculate_mse(&a, &b).is_err());
    }

    /// SNR of identical vectors is infinite.
    #[test]
    fn snr_identical_is_infinite(
        data in prop::collection::vec(1.0f32..10.0, 2..32),
    ) {
        let snr = calculate_snr(&data, &data).unwrap();
        prop_assert!(snr.is_infinite() && snr > 0.0,
            "SNR of identical data should be +inf, got {}", snr);
    }
}

// ── validate_shapes ─────────────────────────────────────────────────────────

proptest! {
    /// Identical shapes always validate.
    #[test]
    fn validate_shapes_identical_ok(
        shape in prop::collection::vec(1usize..16, 1..4),
    ) {
        prop_assert!(validate_shapes(&shape, &shape).is_ok());
    }

    /// Different shapes always fail.
    #[test]
    fn validate_shapes_different_err(
        a in prop::collection::vec(1usize..16, 1..4),
        b in prop::collection::vec(1usize..16, 1..4),
    ) {
        prop_assume!(a != b);
        prop_assert!(validate_shapes(&a, &b).is_err());
    }
}

// ── calculate_optimal_block_size ────────────────────────────────────────────

proptest! {
    /// Result is always a power of two.
    #[test]
    fn optimal_block_size_power_of_two(
        tensor_size in 16usize..10_000,
        target_blocks in 1usize..100,
    ) {
        let bs = calculate_optimal_block_size(tensor_size, target_blocks);
        prop_assert!(bs.is_power_of_two(),
            "block_size {} is not a power of two", bs);
    }

    /// Result is clamped to [16, 1024].
    #[test]
    fn optimal_block_size_clamped(
        tensor_size in 1usize..100_000,
        target_blocks in 1usize..1000,
    ) {
        let bs = calculate_optimal_block_size(tensor_size, target_blocks);
        prop_assert!((16..=1024).contains(&bs),
            "block_size {} outside [16, 1024]", bs);
    }
}
