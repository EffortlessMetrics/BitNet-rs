//! Property-based round-trip tests for quantization pipelines.
//!
//! Invariants verified:
//! - Quantize -> dequantize preserves values within known error bounds
//! - Quantized representation byte count is consistent with bit width
//! - Scale factors are always positive and finite for non-zero inputs
//! - Block alignment: number of scales matches expected block count
//! - Shape is preserved through the quantize -> dequantize cycle

#![cfg(feature = "cpu")]

use bitnet_quantization::utils::calculate_scale;
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
use proptest::prelude::*;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn max_abs_f32(v: &[f32]) -> f32 {
    v.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max)
}

// ── I2_S quantize -> dequantize round-trip depth ────────────────────────────

proptest! {
    /// I2_S round-trip error is bounded relative to per-block max absolute value.
    #[test]
    fn i2s_per_block_error_bounded(
        data in prop::collection::vec(-50.0f32..50.0f32, 32..256),
    ) {
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let deq = q.dequantize_tensor(&quantized).unwrap().to_vec().unwrap();
        let block_size = quantized.block_size.max(1);

        for (block_idx, chunk) in data.chunks(block_size).enumerate() {
            let block_max = max_abs_f32(chunk);
            let tolerance = block_max + 1.0;
            let offset = block_idx * block_size;
            for (j, &orig) in chunk.iter().enumerate() {
                let error = (orig - deq[offset + j]).abs();
                prop_assert!(
                    error <= tolerance,
                    "block {} elem {}: error {} > tolerance {} \
                     (block_max={}, orig={}, deq={})",
                    block_idx, j, error, tolerance, block_max, orig, deq[offset + j]
                );
            }
        }
    }

    /// Quantized byte count is consistent with 2-bit packing: ceil(n / 4) bytes.
    #[test]
    fn i2s_data_byte_count_matches_packing(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..512),
    ) {
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let expected_bytes = data.len().div_ceil(4);
        prop_assert_eq!(
            quantized.data.len(),
            expected_bytes,
            "n={}: expected {} packed bytes, got {}",
            data.len(), expected_bytes, quantized.data.len()
        );
    }

    /// All scale factors are positive and finite for strictly positive input.
    #[test]
    fn i2s_scales_positive_finite(
        data in prop::collection::vec(0.01f32..100.0f32, 4..256),
    ) {
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        for (i, &s) in quantized.scales.iter().enumerate() {
            prop_assert!(s > 0.0, "scale[{}] = {} is non-positive", i, s);
            prop_assert!(s.is_finite(), "scale[{}] = {} is not finite", i, s);
        }
    }

    /// Block alignment: number of scales equals ceil(n / block_size).
    #[test]
    fn i2s_block_alignment(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..512),
    ) {
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let block_size = quantized.block_size.max(1);
        let expected_blocks = data.len().div_ceil(block_size);
        prop_assert_eq!(quantized.scales.len(), expected_blocks);
    }

    /// Shape is preserved: quantized.numel() matches original length.
    #[test]
    fn i2s_shape_preserved(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..256),
    ) {
        let q = I2SQuantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        prop_assert_eq!(quantized.numel(), data.len());
    }
}

// ── TL1 quantize -> dequantize round-trip depth ─────────────────────────────

proptest! {
    /// TL1 quantized byte count is consistent with 2-bit packing.
    #[test]
    fn tl1_data_byte_count_matches_packing(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..512),
    ) {
        let q = TL1Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let expected_bytes = data.len().div_ceil(4);
        prop_assert_eq!(
            quantized.data.len(),
            expected_bytes,
            "n={}: expected {} packed bytes, got {}",
            data.len(), expected_bytes, quantized.data.len()
        );
    }

    /// TL1 scale count matches block count.
    #[test]
    fn tl1_block_alignment(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..512),
    ) {
        let q = TL1Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let block_size = quantized.block_size.max(1);
        let expected_blocks = data.len().div_ceil(block_size);
        prop_assert_eq!(quantized.scales.len(), expected_blocks);
    }

    /// TL1 scales are always positive and finite for non-zero input.
    #[test]
    fn tl1_scales_positive_finite(
        data in prop::collection::vec(0.01f32..100.0f32, 4..256),
    ) {
        let q = TL1Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        for (i, &s) in quantized.scales.iter().enumerate() {
            prop_assert!(s > 0.0, "TL1 scale[{}] = {} is non-positive", i, s);
            prop_assert!(s.is_finite(), "TL1 scale[{}] = {} is not finite", i, s);
        }
    }
}

// ── TL2 quantize -> dequantize round-trip depth ─────────────────────────────

proptest! {
    /// TL2 quantized byte count is consistent with 2-bit packing.
    #[test]
    fn tl2_data_byte_count_matches_packing(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..512),
    ) {
        let q = TL2Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let expected_bytes = data.len().div_ceil(4);
        prop_assert_eq!(
            quantized.data.len(),
            expected_bytes,
            "n={}: expected {} packed bytes, got {}",
            data.len(), expected_bytes, quantized.data.len()
        );
    }

    /// TL2 scale count matches block count.
    #[test]
    fn tl2_block_alignment(
        data in prop::collection::vec(-100.0f32..100.0f32, 4..512),
    ) {
        let q = TL2Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        let block_size = quantized.block_size.max(1);
        let expected_blocks = data.len().div_ceil(block_size);
        prop_assert_eq!(quantized.scales.len(), expected_blocks);
    }

    /// TL2 scales are always positive and finite for non-zero input.
    #[test]
    fn tl2_scales_positive_finite(
        data in prop::collection::vec(0.01f32..100.0f32, 4..256),
    ) {
        let q = TL2Quantizer::new();
        let quantized = q.quantize_weights(&data).unwrap();
        for (i, &s) in quantized.scales.iter().enumerate() {
            prop_assert!(s > 0.0, "TL2 scale[{}] = {} is non-positive", i, s);
            prop_assert!(s.is_finite(), "TL2 scale[{}] = {} is not finite", i, s);
        }
    }
}

// ── Cross-quantizer consistency ─────────────────────────────────────────────

proptest! {
    /// All three quantizers produce the same output length as input length
    /// after a full round-trip.
    #[test]
    fn all_quantizers_preserve_length(
        data in prop::collection::vec(-50.0f32..50.0f32, 4..128),
    ) {
        let i2s = I2SQuantizer::new();
        let tl1 = TL1Quantizer::new();
        let tl2 = TL2Quantizer::new();

        let deq_i2s = i2s
            .dequantize_tensor(&i2s.quantize_weights(&data).unwrap())
            .unwrap()
            .to_vec()
            .unwrap();
        let deq_tl1 = tl1
            .dequantize_tensor(&tl1.quantize_weights(&data).unwrap())
            .unwrap()
            .to_vec()
            .unwrap();
        let deq_tl2 = tl2
            .dequantize_tensor(&tl2.quantize_weights(&data).unwrap())
            .unwrap()
            .to_vec()
            .unwrap();

        prop_assert_eq!(deq_i2s.len(), data.len(), "I2S length mismatch");
        prop_assert_eq!(deq_tl1.len(), data.len(), "TL1 length mismatch");
        prop_assert_eq!(deq_tl2.len(), data.len(), "TL2 length mismatch");
    }

    /// Compression ratio is always >= 1 for all quantizers (2-bit < 32-bit).
    #[test]
    fn all_quantizers_compression_ratio_ge_one(
        data in prop::collection::vec(-100.0f32..100.0f32, 32..256),
    ) {
        let i2s = I2SQuantizer::new().quantize_weights(&data).unwrap();
        let tl1 = TL1Quantizer::new().quantize_weights(&data).unwrap();
        let tl2 = TL2Quantizer::new().quantize_weights(&data).unwrap();

        prop_assert!(
            i2s.compression_ratio() >= 1.0,
            "I2S compression ratio {} < 1.0",
            i2s.compression_ratio()
        );
        prop_assert!(
            tl1.compression_ratio() >= 1.0,
            "TL1 compression ratio {} < 1.0",
            tl1.compression_ratio()
        );
        prop_assert!(
            tl2.compression_ratio() >= 1.0,
            "TL2 compression ratio {} < 1.0",
            tl2.compression_ratio()
        );
    }
}

// ── Scale factor invariants across bit widths ───────────────────────────────

proptest! {
    /// Higher bit width produces a smaller or equal scale for the same data,
    /// because the representable range per level is finer.
    #[test]
    fn scale_monotone_with_bit_width(
        data in prop::collection::vec(1.0f32..100.0f32, 4..128),
    ) {
        let s2 = calculate_scale(&data, 2);
        let s4 = calculate_scale(&data, 4);
        prop_assert!(
            s4 <= s2 + f32::EPSILON,
            "4-bit scale ({}) should be <= 2-bit scale ({})",
            s4,
            s2
        );
    }
}
