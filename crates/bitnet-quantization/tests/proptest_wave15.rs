//! Wave 15 property tests: quantization invariants for I2_S round-trip,
//! TL1/TL2 table lookup, QK256 block alignment, error bounds, scale factors,
//! and zero-input identity.
//!
//! Key invariants tested (12 properties):
//! - I2_S round-trip preserves sign of non-zero elements
//! - I2_S round-trip preserves tensor length
//! - TL1 quantize-dequantize output is always in a valid finite range
//! - TL2 quantize-dequantize output is always in a valid finite range
//! - QK256 block size is always 256
//! - QK256 unpack produces codes in {0,1,2,3}
//! - Quantization error is bounded by scale factor
//! - Scale factor is always positive and finite
//! - Scale factor with all-zero input returns fallback (1.0)
//! - Zero input quantizes to zero (quantize_value)
//! - Pack/unpack 2-bit round-trip is lossless
//! - Grouped scales have correct count

use bitnet_common::{BitNetTensor, Device, QuantizationType};
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, unpack_qk256_block,
};
use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_scale, pack_2bit_values, quantize_value, unpack_2bit_values,
};
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Non-empty f32 vector with finite values in [-5, 5], length a multiple of 32
/// (I2S default block size).
fn quantizable_vec(max_blocks: usize) -> impl Strategy<Value = Vec<f32>> {
    (1..=max_blocks).prop_flat_map(|blocks| {
        let len = blocks * 32;
        prop::collection::vec(-5.0f32..5.0f32, len..=len)
    })
}

/// Random 2-bit signed values in {-1, 0, 1} (I2S range).
fn i2s_values(len: usize) -> impl Strategy<Value = Vec<i8>> {
    prop::collection::vec(prop::sample::select(vec![-1i8, 0, 1]), len..=len)
}

// ===================================================================
// 1. I2_S round-trip: quantize then dequantize preserves sign
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// I2_S quantize→dequantize preserves sign of non-zero elements.
    #[test]
    fn prop_i2s_roundtrip_preserves_sign(data in quantizable_vec(4)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        let deq_data = dequantized.to_vec().unwrap();

        prop_assert_eq!(
            deq_data.len(), data.len(),
            "round-trip changed tensor length"
        );

        for (i, (&orig, &deq)) in data.iter().zip(deq_data.iter()).enumerate() {
            if orig.abs() > 0.5 {
                // For values with significant magnitude, sign should be preserved
                prop_assert!(
                    orig.signum() == deq.signum() || deq == 0.0,
                    "index {}: sign changed: orig={}, deq={}", i, orig, deq
                );
            }
        }
    }

    /// I2_S quantized tensor always has I2S quantization type.
    #[test]
    fn prop_i2s_quantizer_type(data in quantizable_vec(2)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        prop_assert_eq!(
            quantized.qtype,
            QuantizationType::I2S,
            "expected I2S quantization type"
        );
    }
}

// ===================================================================
// 2. TL1/TL2 table lookup: output is always in valid range
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// TL1 quantize→dequantize output values are all finite.
    #[test]
    fn prop_tl1_output_finite(data in quantizable_vec(4)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = TL1Quantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        let deq_data = dequantized.to_vec().unwrap();

        for (i, &v) in deq_data.iter().enumerate() {
            prop_assert!(
                v.is_finite(),
                "TL1 dequantized[{}] = {} is not finite", i, v
            );
        }
    }

    /// TL2 quantize→dequantize output values are all finite.
    #[test]
    fn prop_tl2_output_finite(data in quantizable_vec(4)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = TL2Quantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        let deq_data = dequantized.to_vec().unwrap();

        for (i, &v) in deq_data.iter().enumerate() {
            prop_assert!(
                v.is_finite(),
                "TL2 dequantized[{}] = {} is not finite", i, v
            );
        }
    }
}

// ===================================================================
// 3. QK256 block alignment: block_size is always 256
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// QK256 block constants are self-consistent: 256 elems × 2 bits / 8 = 64 bytes.
    #[test]
    fn prop_qk256_block_constants_consistent(_dummy in 0..1i32) {
        prop_assert_eq!(QK256_BLOCK, 256);
        prop_assert_eq!(QK256_PACKED_BYTES, 64);
        prop_assert_eq!(QK256_BLOCK * 2 / 8, QK256_PACKED_BYTES);
    }

    /// QK256 unpack always produces codes in {0, 1, 2, 3}.
    #[test]
    fn prop_qk256_unpack_codes_in_range(
        packed in prop::collection::vec(any::<u8>(), 64..=64),
    ) {
        let packed_arr: [u8; QK256_PACKED_BYTES] = packed.try_into().unwrap();
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&packed_arr, &mut codes);

        for (i, &c) in codes.iter().enumerate() {
            prop_assert!(
                c <= 3,
                "code[{}] = {}, expected 0..=3", i, c
            );
        }
    }

    /// QK256 code_to_f32 maps every valid code to {-2, -1, 1, 2}.
    #[test]
    fn prop_qk256_code_to_f32_valid(code in 0u8..4) {
        let val = code_to_f32(code);
        prop_assert!(
            val == -2.0 || val == -1.0 || val == 1.0 || val == 2.0,
            "code_to_f32({}) = {}, expected {{-2,-1,1,2}}", code, val
        );
    }
}

// ===================================================================
// 4. Quantization error bounds: error < threshold for random inputs
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// quantize_value→dequantize round-trip error is bounded by the scale.
    #[test]
    fn prop_quantize_value_error_bounded(
        value in -5.0f32..5.0f32,
        bits in 2u8..=8u8,
    ) {
        let scale = calculate_scale(&[value], bits);
        let quantized = quantize_value(value, scale, bits);
        let dequantized = quantized as f32 * scale;
        let error = (value - dequantized).abs();
        prop_assert!(
            error <= scale + 1e-5,
            "error={} > scale={} for value={}, bits={}", error, scale, value, bits
        );
    }
}

// ===================================================================
// 5. Scale factor is always positive and finite
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// calculate_scale always returns a positive, finite value.
    #[test]
    fn prop_scale_positive_finite(
        data in prop::collection::vec(-10.0f32..10.0f32, 1..=64),
        bits in 2u8..=8u8,
    ) {
        let scale = calculate_scale(&data, bits);
        prop_assert!(
            scale > 0.0 && scale.is_finite(),
            "scale={} should be positive and finite for bits={}", scale, bits
        );
    }

    /// Grouped scales have the correct count (ceil(len / block_size)).
    #[test]
    fn prop_grouped_scales_count(
        data in prop::collection::vec(-5.0f32..5.0f32, 1..=256),
        block_size in prop::sample::select(vec![32usize, 64, 128]),
        bits in 2u8..=4u8,
    ) {
        let scales = calculate_grouped_scales(&data, block_size, bits);
        let expected = data.len().div_ceil(block_size);
        prop_assert_eq!(
            scales.len(), expected,
            "expected {} scales for {} elements with block_size={}",
            expected, data.len(), block_size
        );
    }
}

// ===================================================================
// 6. Zero input quantizes to zero
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// quantize_value(0.0, ...) always returns 0.
    #[test]
    fn prop_zero_input_quantizes_to_zero(
        scale in 0.01f32..10.0f32,
        bits in 2u8..=8u8,
    ) {
        let q = quantize_value(0.0, scale, bits);
        prop_assert_eq!(q, 0i8, "quantize_value(0.0, scale={}, bits={}) = {}", scale, bits, q);
    }

    /// All-zero data produces fallback scale (1.0).
    #[test]
    fn prop_all_zero_scale_fallback(
        len in 1usize..=64,
        bits in 2u8..=8u8,
    ) {
        let data = vec![0.0f32; len];
        let scale = calculate_scale(&data, bits);
        prop_assert!(
            (scale - 1.0).abs() < 1e-6,
            "all-zero scale={}, expected 1.0", scale
        );
    }
}

// ===================================================================
// 7. Pack/unpack 2-bit round-trip is lossless
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// pack_2bit_values then unpack_2bit_values recovers original values.
    #[test]
    fn prop_pack_unpack_2bit_roundtrip(values in i2s_values(128)) {
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, values.len());
        prop_assert_eq!(
            unpacked.len(), values.len(),
            "unpack length mismatch"
        );
        for (i, (&orig, &recovered)) in values.iter().zip(unpacked.iter()).enumerate() {
            prop_assert_eq!(
                orig, recovered,
                "index {}: pack/unpack mismatch: orig={}, recovered={}", i, orig, recovered
            );
        }
    }
}
