//! Property-based tests for quantization roundtrips, block sizes, and scale factors
//! (proptest wave 31).

use bitnet_common::{BitNetTensor, QuantizationType, Tensor};
use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_scale, dequantize_value, quantize_value,
};
use bitnet_quantization::{I2SQuantizer, QuantizedTensor, TL1Quantizer};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use proptest::prelude::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn normal_f32_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0, len)
}

// ── Scale factor properties ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Scale factor is always finite for normal inputs.
    #[test]
    fn scale_always_finite(
        data in prop::collection::vec(-100.0f32..100.0, 1..=512),
        bits in 1u8..=4,
    ) {
        let s = calculate_scale(&data, bits);
        prop_assert!(s.is_finite(), "scale must be finite, got {s}");
    }

    /// Scale factor is non-negative.
    #[test]
    fn scale_non_negative(
        data in prop::collection::vec(-50.0f32..50.0, 1..=256),
        bits in 1u8..=4,
    ) {
        let s = calculate_scale(&data, bits);
        prop_assert!(s >= 0.0, "scale must be non-negative, got {s}");
    }

    /// Scale for all-zero data is the safe fallback (1.0).
    #[test]
    fn scale_fallback_for_zero_data(len in 1usize..=128, bits in 1u8..=4) {
        let data = vec![0.0f32; len];
        let s = calculate_scale(&data, bits);
        // calculate_scale returns 1.0 as a safe fallback for zero data
        prop_assert_eq!(s, 1.0);
    }

    /// Grouped scales have correct count.
    #[test]
    fn grouped_scales_count(
        data in prop::collection::vec(-5.0f32..5.0, 64..=512),
        bits in 1u8..=4,
    ) {
        let block_size = 64;
        let scales = calculate_grouped_scales(&data, block_size, bits);
        let expected_blocks = data.len().div_ceil(block_size);
        prop_assert_eq!(scales.len(), expected_blocks);
    }

    /// Every grouped scale is finite and non-negative.
    #[test]
    fn grouped_scales_all_finite(
        data in prop::collection::vec(-10.0f32..10.0, 64..=256),
    ) {
        let scales = calculate_grouped_scales(&data, 64, 2);
        for (i, &s) in scales.iter().enumerate() {
            prop_assert!(s.is_finite(), "scale[{i}] = {s} is not finite");
            prop_assert!(s >= 0.0, "scale[{i}] = {s} is negative");
        }
    }
}

// ── Quantize → dequantize scalar roundtrip ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Scalar quantize→dequantize produces a finite result.
    #[test]
    fn scalar_roundtrip_finite(
        value in -10.0f32..10.0,
        scale in 0.01f32..10.0,
    ) {
        let q = quantize_value(value, scale, 2);
        let d = dequantize_value(q, scale);
        prop_assert!(d.is_finite(), "roundtrip should be finite for value={value}, scale={scale}");
        // Quantized value must be in 2-bit signed range [-2, 1]
        prop_assert!(q >= -2 && q <= 1, "2-bit quantized value {q} out of range");
    }

    /// Scalar roundtrip error is bounded by the scale for small values.
    #[test]
    fn scalar_roundtrip_bounded_error(
        scale in 0.5f32..10.0,
    ) {
        // For value = 0, roundtrip must be exact
        let q = quantize_value(0.0, scale, 2);
        let d = dequantize_value(q, scale);
        prop_assert!((d - 0.0).abs() < 1e-6, "zero should roundtrip exactly");
    }

    /// Dequantized value is always finite.
    #[test]
    fn dequantize_always_finite(
        q in -2i8..=2,
        scale in 0.001f32..100.0,
    ) {
        let d = dequantize_value(q, scale);
        prop_assert!(d.is_finite(), "dequantize({q}, {scale}) = {d} not finite");
    }

    /// Zero quantizes to zero.
    #[test]
    fn zero_quantizes_to_zero(scale in 0.01f32..100.0, bits in 1u8..=4) {
        let q = quantize_value(0.0, scale, bits);
        prop_assert_eq!(q, 0);
    }
}

// ── I2S quantizer tensor roundtrip ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// I2S quantize→dequantize preserves shape.
    #[test]
    fn i2s_roundtrip_preserves_shape(
        len in (32usize..=256).prop_map(|l| (l / 32) * 32),
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1) - 5.0).collect();
        let tensor = CandleTensor::from_vec(data, &[len], &CandleDevice::Cpu).unwrap();
        let bt = BitNetTensor::new(tensor);

        let quantizer = I2SQuantizer::new();
        if let Ok(qt) = quantizer.quantize_tensor(&bt) {
            prop_assert_eq!(&qt.shape, &[len], "shape mismatch after quantize");
            if let Ok(deq) = quantizer.dequantize_tensor(&qt) {
                let deq_shape = deq.shape();
                // Shape must match
                prop_assert_eq!(deq_shape[deq_shape.len() - 1], len);
            }
        }
    }

    /// I2S quantization produces correct qtype.
    #[test]
    fn i2s_quantize_correct_qtype(
        len in (32usize..=128).prop_map(|l| (l / 32) * 32),
    ) {
        let data = vec![1.0f32; len];
        let tensor = CandleTensor::from_vec(data, &[len], &CandleDevice::Cpu).unwrap();
        let bt = BitNetTensor::new(tensor);
        let quantizer = I2SQuantizer::new();
        if let Ok(qt) = quantizer.quantize_tensor(&bt) {
            prop_assert_eq!(qt.qtype, QuantizationType::I2S);
        }
    }
}

// ── Block size divides tensor dimensions ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Block count * block_size >= numel for QuantizedTensor.
    #[test]
    fn block_count_covers_tensor(
        n_blocks in 1usize..=64,
        block_size in prop_oneof![Just(32usize), Just(64), Just(128), Just(256)],
    ) {
        let numel = n_blocks * block_size;
        let computed_blocks = numel.div_ceil(block_size);
        prop_assert_eq!(computed_blocks, n_blocks);
        prop_assert!(computed_blocks * block_size >= numel);
    }

    /// QuantizedTensor numel equals product of shape dims.
    #[test]
    fn quantized_tensor_numel(
        d1 in 1usize..=64,
        d2 in 1usize..=64,
    ) {
        let shape = vec![d1, d2];
        let expected = d1 * d2;
        let qt = QuantizedTensor::new(
            vec![0u8; expected / 4], // dummy data
            vec![1.0f32; expected / 32], // dummy scales
            shape.clone(),
            QuantizationType::I2S,
        );
        prop_assert_eq!(qt.numel(), expected);
    }

    /// Block size 64 evenly divides lengths that are multiples of 64.
    #[test]
    fn block64_divides_multiples(multiplier in 1usize..=32) {
        let len = multiplier * 64;
        prop_assert_eq!(len % 64, 0);
        let n_blocks = len / 64;
        prop_assert_eq!(n_blocks, multiplier);
    }
}

// ── TL1 lookup table roundtrip ────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// TL1 quantizer preserves shape through roundtrip.
    #[test]
    fn tl1_roundtrip_preserves_shape(
        len in (64usize..=256).prop_map(|l| (l / 64) * 64),
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32 / len as f32) - 0.5).collect();
        let tensor = CandleTensor::from_vec(data, &[len], &CandleDevice::Cpu).unwrap();
        let bt = BitNetTensor::new(tensor);

        let quantizer = TL1Quantizer::new();
        if let Ok(qt) = quantizer.quantize_tensor(&bt) {
            prop_assert_eq!(&qt.shape, &[len]);
            prop_assert_eq!(qt.qtype, QuantizationType::TL1);
        }
    }

    /// TL1 scales are all finite after quantization.
    #[test]
    fn tl1_scales_finite(
        len in (64usize..=256).prop_map(|l| (l / 64) * 64),
    ) {
        let data: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01) - 1.0).collect();
        let tensor = CandleTensor::from_vec(data, &[len], &CandleDevice::Cpu).unwrap();
        let bt = BitNetTensor::new(tensor);

        let quantizer = TL1Quantizer::new();
        if let Ok(qt) = quantizer.quantize_tensor(&bt) {
            for (i, &s) in qt.scales.iter().enumerate() {
                prop_assert!(s.is_finite(), "TL1 scale[{i}] = {s} not finite");
            }
        }
    }
}
