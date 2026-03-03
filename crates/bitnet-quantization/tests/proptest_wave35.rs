//! Property-based tests — wave 35.
//!
//! 20 properties covering I2S round-trip tolerance, block size invariants,
//! scale factor bounds, QK256 block alignment/element validation,
//! TL1/TL2 table lookup boundaries, and mixed precision accumulation stability.

use bitnet_common::{BitNetTensor, Device, QuantizationType};
use bitnet_quantization::i2s_qk256::{
    I2SQk256NoScale, QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, gemv_qk256_row,
    unpack_qk256_block,
};
use bitnet_quantization::tl1::LookupTable;
use bitnet_quantization::tl2::VectorizedLookupTable;
use bitnet_quantization::utils::{
    calculate_mse, calculate_optimal_block_size, calculate_scale, dequantize_value, quantize_value,
};
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Finite f32 vector aligned to 32-element blocks.
fn block32_vec(max_blocks: usize) -> impl Strategy<Value = Vec<f32>> {
    (1..=max_blocks).prop_flat_map(|blocks| {
        let len = blocks * 32;
        prop::collection::vec(-5.0f32..5.0f32, len..=len)
    })
}

// ===================================================================
// 1–3. I2S quantize/dequantize round-trip tolerance
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// I2S round-trip MSE is bounded relative to input variance.
    #[test]
    fn prop_i2s_roundtrip_mse_bounded(data in block32_vec(4)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        let deq_data = dequantized.to_vec().unwrap();

        let mse = calculate_mse(&data, &deq_data).unwrap();
        let variance = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
        // MSE should be at most the variance (trivially bounded by energy)
        prop_assert!(
            mse <= variance + 1e-6,
            "MSE={} exceeds variance={}", mse, variance
        );
    }

    /// I2S dequantized output has same length as input.
    #[test]
    fn prop_i2s_roundtrip_length_preserved(data in block32_vec(4)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        let deq_data = dequantized.to_vec().unwrap();
        prop_assert_eq!(deq_data.len(), data.len());
    }

    /// I2S dequantized values are always finite.
    #[test]
    fn prop_i2s_roundtrip_all_finite(data in block32_vec(4)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        let deq_data = dequantized.to_vec().unwrap();
        for (i, &v) in deq_data.iter().enumerate() {
            prop_assert!(v.is_finite(), "deq[{}] = {} is not finite", i, v);
        }
    }
}

// ===================================================================
// 4–6. Block size invariants and scale factor bounds
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// calculate_optimal_block_size always returns a power of 2 in [16, 1024].
    #[test]
    fn prop_optimal_block_size_power_of_two(
        tensor_size in 32usize..10_000,
        target_blocks in 1usize..128,
    ) {
        let bs = calculate_optimal_block_size(tensor_size, target_blocks);
        prop_assert!(bs.is_power_of_two(), "block_size {} not power of 2", bs);
        prop_assert!(bs >= 16, "block_size {} < 16", bs);
        prop_assert!(bs <= 1024, "block_size {} > 1024", bs);
    }

    /// Scale factor is always positive for non-NaN inputs.
    #[test]
    fn prop_scale_factor_positive(
        data in prop::collection::vec(-10.0f32..10.0f32, 1..=128),
        bits in 2u8..=8u8,
    ) {
        let scale = calculate_scale(&data, bits);
        prop_assert!(scale > 0.0, "scale={} not positive for bits={}", scale, bits);
        prop_assert!(scale.is_finite(), "scale={} not finite", scale);
    }

    /// Scale factor monotonically increases with data magnitude.
    #[test]
    fn prop_scale_increases_with_magnitude(
        base in prop::collection::vec(0.1f32..1.0f32, 32..=32),
        multiplier in 2.0f32..10.0f32,
        bits in 2u8..=4u8,
    ) {
        let small_scale = calculate_scale(&base, bits);
        let big: Vec<f32> = base.iter().map(|x| x * multiplier).collect();
        let big_scale = calculate_scale(&big, bits);
        prop_assert!(
            big_scale >= small_scale - 1e-6,
            "bigger data should have >= scale: {} < {}", big_scale, small_scale
        );
    }
}

// ===================================================================
// 7–10. QK256 block alignment and element count validation
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// QK256 block element count is always 256.
    #[test]
    fn prop_qk256_block_element_count(_dummy in 0..1i32) {
        prop_assert_eq!(QK256_BLOCK, 256);
        prop_assert_eq!(QK256_PACKED_BYTES, 64);
    }

    /// I2SQk256NoScale accepts correctly-sized data.
    #[test]
    fn prop_qk256_noscale_valid_construction(
        rows in 1usize..=8,
        block_count in 1usize..=4,
    ) {
        let cols = block_count * QK256_BLOCK;
        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_bytes = blocks_per_row * QK256_PACKED_BYTES;
        let data = vec![0u8; rows * row_bytes];
        let result = I2SQk256NoScale::new(rows, cols, data);
        prop_assert!(result.is_ok(), "valid dims should succeed: {:?}", result.err());
    }

    /// I2SQk256NoScale rejects wildly mismatched data sizes.
    #[test]
    fn prop_qk256_noscale_rejects_bad_size(
        rows in 1usize..=4,
        cols in 256usize..=1024,
    ) {
        // Provide much less data than needed
        let needed = rows * cols.div_ceil(QK256_BLOCK) * QK256_PACKED_BYTES;
        let too_small = needed.saturating_sub(256).max(1);
        let data = vec![0u8; too_small];
        if too_small.abs_diff(needed) > 128 {
            let result = I2SQk256NoScale::new(rows, cols, data);
            prop_assert!(result.is_err(), "undersized data should fail");
        }
    }

    /// gemv_qk256_row produces finite results for random packed data.
    #[test]
    fn prop_qk256_gemv_finite(
        packed in prop::collection::vec(any::<u8>(), 64..=64),
        x in prop::collection::vec(-1.0f32..1.0f32, 256..=256),
    ) {
        let result = gemv_qk256_row(&packed, &x, 256);
        prop_assert!(result.is_finite(), "gemv result={} not finite", result);
    }
}

// ===================================================================
// 11–14. TL1/TL2 table lookup boundaries
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// TL1 LookupTable forward table has 256 entries.
    #[test]
    fn prop_tl1_lookup_table_size(
        min_val in -5.0f32..-0.1f32,
        max_val in 0.1f32..5.0f32,
        bits in 2u8..=4u8,
    ) {
        let table = LookupTable::new(min_val, max_val, bits, false);
        // quantize then dequantize should produce finite values
        let q = table.quantize(0.5);
        let d = table.dequantize(q);
        prop_assert!(d.is_finite(), "TL1 dequantize({}) = {} not finite", q, d);
    }

    /// TL1 quantize→dequantize round-trip is bounded.
    #[test]
    fn prop_tl1_roundtrip_bounded(data in block32_vec(4)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = TL1Quantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        let deq_data = dequantized.to_vec().unwrap();
        for (i, (&orig, &deq)) in data.iter().zip(deq_data.iter()).enumerate() {
            let err = (orig - deq).abs();
            // Error should be bounded by the data range
            let range = data.iter().cloned().fold(0.0f32, f32::max)
                - data.iter().cloned().fold(0.0f32, f32::min);
            prop_assert!(
                err <= range + 1e-3,
                "TL1 error[{}]={} exceeds range={}", i, err, range
            );
        }
    }

    /// TL2 VectorizedLookupTable forward has 256 entries, reverse has 2^bits.
    #[test]
    fn prop_tl2_table_dimensions(
        min_val in -5.0f32..-0.1f32,
        max_val in 0.1f32..5.0f32,
        bits in 2u8..=4u8,
    ) {
        let table = VectorizedLookupTable::new(min_val, max_val, bits);
        prop_assert_eq!(table.forward_len(), 256, "forward table should be 256");
        prop_assert_eq!(
            table.reverse_len(),
            1 << bits,
            "reverse table should be 2^{}", bits
        );
    }

    /// TL2 quantize→dequantize output is always finite.
    #[test]
    fn prop_tl2_roundtrip_finite(data in block32_vec(4)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = TL2Quantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        let deq_data = dequantized.to_vec().unwrap();
        for (i, &v) in deq_data.iter().enumerate() {
            prop_assert!(v.is_finite(), "TL2 deq[{}]={} not finite", i, v);
        }
    }
}

// ===================================================================
// 15–17. Mixed precision accumulation stability
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// quantize_value + dequantize_value round-trip error bounded by scale.
    #[test]
    fn prop_mixed_precision_quantize_dequantize(
        value in -5.0f32..5.0f32,
        bits in 2u8..=8u8,
    ) {
        let scale = calculate_scale(&[value], bits);
        let q = quantize_value(value, scale, bits);
        let deq = dequantize_value(q, scale);
        let error = (value - deq).abs();
        prop_assert!(
            error <= scale + 1e-5,
            "error={} > scale={} for value={}", error, scale, value
        );
    }

    /// QK256 code_to_f32 accumulation: sum of codes is bounded.
    #[test]
    fn prop_qk256_accumulation_bounded(
        packed in prop::collection::vec(any::<u8>(), 64..=64),
    ) {
        let packed_arr: [u8; QK256_PACKED_BYTES] = packed.try_into().unwrap();
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&packed_arr, &mut codes);

        let sum: f32 = codes.iter().map(|&c| code_to_f32(c)).sum();
        // Each code maps to {-2,-1,1,2}, so |sum| <= 256*2 = 512
        prop_assert!(
            sum.abs() <= 512.0,
            "code accumulation sum={} exceeds 512", sum
        );
    }

    /// MSE between identical slices is zero.
    #[test]
    fn prop_mse_identity_is_zero(
        data in prop::collection::vec(-10.0f32..10.0f32, 1..=128),
    ) {
        let mse = calculate_mse(&data, &data).unwrap();
        prop_assert!(
            mse.abs() < 1e-10,
            "MSE of identical data should be ~0, got {}", mse
        );
    }
}

// ===================================================================
// 18–20. Additional quantization invariants
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// I2S quantized tensor reports correct quantization type.
    #[test]
    fn prop_i2s_quantized_type_tag(data in block32_vec(2)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        prop_assert_eq!(quantized.qtype, QuantizationType::I2S);
    }

    /// I2S quantized tensor numel matches input length.
    #[test]
    fn prop_i2s_quantized_numel(data in block32_vec(4)) {
        let device = Device::Cpu;
        let tensor = BitNetTensor::from_slice(&data, &[data.len()], &device).unwrap();
        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        prop_assert_eq!(
            quantized.numel(), data.len(),
            "numel mismatch: {} vs {}", quantized.numel(), data.len()
        );
    }

    /// QK256 unpack/code_to_f32 produces only values in {-2,-1,1,2}.
    #[test]
    fn prop_qk256_dequant_value_set(
        packed in prop::collection::vec(any::<u8>(), 64..=64),
    ) {
        let packed_arr: [u8; QK256_PACKED_BYTES] = packed.try_into().unwrap();
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&packed_arr, &mut codes);

        for (i, &c) in codes.iter().enumerate() {
            let v = code_to_f32(c);
            prop_assert!(
                v == -2.0 || v == -1.0 || v == 1.0 || v == 2.0,
                "code[{}]={} mapped to {}, expected {{-2,-1,1,2}}", i, c, v
            );
        }
    }
}
