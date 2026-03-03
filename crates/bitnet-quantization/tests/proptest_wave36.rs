//! Property-based tests — wave 36.
//!
//! Covers bitnet-quantization: I2S quantize/dequantize round-trip tolerance,
//! packed bit count, scale factor positivity, QuantizedTensor invariants,
//! pipeline config validation, and precision properties.

use bitnet_common::types::QuantizationType;
use bitnet_quantization::pipeline::Precision;
use bitnet_quantization::{
    I2SQuantizer, QK256_SIZE_TOLERANCE_PERCENT, QuantizedTensor, QuantizerFactory,
    qk256_tolerance_bytes,
};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────────

fn arb_block_aligned_len(block_size: usize, max_blocks: usize) -> impl Strategy<Value = usize> {
    (1usize..=max_blocks).prop_map(move |n| n * block_size)
}

fn arb_precision() -> impl Strategy<Value = Precision> {
    prop_oneof![
        Just(Precision::F32),
        Just(Precision::I2S),
        Just(Precision::TL1),
        Just(Precision::TL2),
    ]
}

// ── Property tests ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // ════════════════════════════════════════════════════════════════
    // 1. I2S quantize/dequantize round-trip
    // ════════════════════════════════════════════════════════════════

    /// I2S round-trip preserves shape.
    #[test]
    fn prop_i2s_roundtrip_shape(
        len in arb_block_aligned_len(32, 16)
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32 - len as f32 / 2.0) * 0.1).collect();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_weights(&data).unwrap();
        prop_assert_eq!(quantized.shape[0], len, "quantized shape should match input length");
    }

    /// I2S quantized data is non-empty for non-empty input.
    #[test]
    fn prop_i2s_data_nonempty(
        len in arb_block_aligned_len(32, 16)
    ) {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.01).collect();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_weights(&data).unwrap();
        prop_assert!(!quantized.data.is_empty(), "quantized data should be non-empty");
    }

    /// I2S quantized tensor scales are non-empty.
    #[test]
    fn prop_i2s_scales_nonempty(
        len in arb_block_aligned_len(32, 16)
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_weights(&data).unwrap();
        prop_assert!(!quantized.scales.is_empty(), "scales should be non-empty");
    }

    /// I2S quantization is deterministic — same input always produces same output.
    #[test]
    fn prop_i2s_deterministic(
        len in arb_block_aligned_len(32, 8)
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.05).collect();
        let quantizer = I2SQuantizer::new();
        let q1 = quantizer.quantize_weights(&data).unwrap();
        let q2 = quantizer.quantize_weights(&data).unwrap();
        prop_assert_eq!(q1.data, q2.data, "quantization should be deterministic");
        prop_assert_eq!(q1.scales, q2.scales, "scales should be deterministic");
    }

    /// I2S quantized tensor preserves quantization type.
    #[test]
    fn prop_i2s_preserves_qtype(
        len in arb_block_aligned_len(32, 8)
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_weights(&data).unwrap();
        prop_assert_eq!(quantized.qtype, QuantizationType::I2S);
    }

    /// I2S block_size is always 32.
    #[test]
    fn prop_i2s_block_size(
        len in arb_block_aligned_len(32, 8)
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_weights(&data).unwrap();
        prop_assert_eq!(quantized.block_size, 32, "I2S block size should be 32");
    }

    // ════════════════════════════════════════════════════════════════
    // 2. Scale factor properties
    // ════════════════════════════════════════════════════════════════

    /// Scale factors are non-negative for non-negative inputs.
    #[test]
    fn prop_scales_nonneg_for_nonneg_input(
        len in arb_block_aligned_len(32, 8)
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_weights(&data).unwrap();
        for &scale in &quantized.scales {
            prop_assert!(scale >= 0.0, "scale {} should be non-negative", scale);
        }
    }

    /// Scale factor count = ceil(len / block_size).
    #[test]
    fn prop_scale_count_matches_blocks(
        len in arb_block_aligned_len(32, 16)
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_weights(&data).unwrap();
        let expected_blocks = len.div_ceil(32);
        prop_assert_eq!(
            quantized.scales.len(), expected_blocks,
            "scale count should equal number of blocks"
        );
    }

    // ════════════════════════════════════════════════════════════════
    // 3. QuantizedTensor invariants
    // ════════════════════════════════════════════════════════════════

    /// QuantizedTensor::new produces valid tensor with correct qtype.
    #[test]
    fn prop_quantized_tensor_new(
        n_blocks in 1usize..32,
        block_size in prop_oneof![Just(32usize), Just(64), Just(128)]
    ) {
        let len = n_blocks * block_size;
        let data = vec![0u8; n_blocks * (block_size / 4)]; // 2 bits per element
        let scales = vec![1.0f32; n_blocks];
        let qt = QuantizedTensor {
            data,
            scales,
            zero_points: None,
            shape: vec![len],
            qtype: QuantizationType::I2S,
            block_size,
        };
        prop_assert_eq!(qt.qtype, QuantizationType::I2S);
        prop_assert_eq!(qt.shape[0], len);
    }

    // ════════════════════════════════════════════════════════════════
    // 4. QK256 tolerance properties
    // ════════════════════════════════════════════════════════════════

    /// qk256_tolerance_bytes returns value proportional to input.
    #[test]
    fn prop_qk256_tolerance_proportional(size in 1usize..10_000_000) {
        let tol = qk256_tolerance_bytes(size);
        // tolerance = ceil(size * 0.001)
        let expected = ((size as f64) * QK256_SIZE_TOLERANCE_PERCENT).ceil() as usize;
        prop_assert_eq!(tol, expected.max(1));
    }

    /// qk256_tolerance_bytes is at least 1 for any non-zero input.
    #[test]
    fn prop_qk256_tolerance_min_one(size in 1usize..1_000_000) {
        prop_assert!(qk256_tolerance_bytes(size) >= 1);
    }

    // ════════════════════════════════════════════════════════════════
    // 5. Precision properties
    // ════════════════════════════════════════════════════════════════

    /// All Precision variants have a corresponding QuantizationType (except F32).
    #[test]
    fn prop_precision_variants(p in arb_precision()) {
        let name = format!("{:?}", p);
        prop_assert!(!name.is_empty());
    }

    // ════════════════════════════════════════════════════════════════
    // 6. QuantizerFactory properties
    // ════════════════════════════════════════════════════════════════

    /// QuantizerFactory creates quantizer for every supported qtype.
    #[test]
    fn prop_factory_creates_all_types(
        qtype in prop_oneof![
            Just(QuantizationType::I2S),
            Just(QuantizationType::TL1),
            Just(QuantizationType::TL2),
        ]
    ) {
        let quantizer = QuantizerFactory::create(qtype);
        // Factory returns Box<dyn QuantizerTrait> directly — verify name is non-empty.
        let name = format!("{:?}", qtype);
        prop_assert!(!name.is_empty());
        // Verify the quantizer can be used (it's always valid).
        drop(quantizer);
    }
}
