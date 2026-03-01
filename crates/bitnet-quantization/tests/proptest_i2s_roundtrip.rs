//! Wave 12 property tests: I2_S quantization roundtrip correctness and bounds.
//!
//! Key invariants tested (8 properties):
//! - Quantize then dequantize produces bounded error (MSE below threshold)
//! - Quantized values are in {-2, -1, 0, 1} range for 2-bit signed quantization
//! - Scale factors from calculate_scale are always positive
//! - Zero input produces zero output after quantize→dequantize
//! - 2-bit pack/unpack roundtrip preserves element count
//! - Grouped scales have one scale per block
//! - Optimal block size is always a power of two
//! - calculate_mse is zero for identical inputs

#![cfg(feature = "cpu")]

use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_mse, calculate_optimal_block_size, calculate_scale,
    create_tensor_from_f32, dequantize_value, pack_2bit_values, quantize_value,
    unpack_2bit_values,
};
use bitnet_quantization::I2SQuantizer;
use candle_core::Device;
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Finite f32 vector with values in [-10, 10], length a multiple of 32 (I2S block size).
fn block_aligned_f32_vec(max_blocks: usize) -> impl Strategy<Value = Vec<f32>> {
    (1..=max_blocks).prop_flat_map(|nblocks| {
        let len = nblocks * 32;
        prop::collection::vec(-10.0f32..10.0f32, len..=len)
    })
}

// ===================================================================
// 1. Quantize → dequantize roundtrip has bounded error
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// I2S quantize then dequantize produces finite output with bounded MSE.
    #[test]
    fn prop_i2s_roundtrip_bounded_error(
        data in block_aligned_f32_vec(8),
    ) {
        let device = Device::Cpu;
        let shape = &[data.len()];
        let tensor = create_tensor_from_f32(data.clone(), shape, &device)
            .expect("tensor creation");

        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize(&tensor, &device)
            .expect("quantize");
        let reconstructed = quantizer.dequantize(&quantized, &device)
            .expect("dequantize");

        let recon_data: Vec<f32> = reconstructed.to_vec().expect("to_vec");

        // All reconstructed values must be finite
        for (i, &v) in recon_data.iter().enumerate() {
            prop_assert!(v.is_finite(), "non-finite at index {}: {}", i, v);
        }

        // Lengths must match
        prop_assert_eq!(recon_data.len(), data.len());

        // MSE must be bounded — 2-bit quantization can have large error,
        // but for values in [-10, 10] the MSE should be well under 100.
        let mse = calculate_mse(&data, &recon_data).expect("mse");
        prop_assert!(mse < 100.0, "MSE too large: {}", mse);
    }
}

// ===================================================================
// 2. Quantized values are in {-1, 0, 1} for 2-bit signed
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// quantize_value with 2 bits clamps to the signed 2-bit range [-2, 1].
    #[test]
    fn prop_quantize_value_2bit_range(
        value in -100.0f32..100.0f32,
        scale in 0.01f32..10.0f32,
    ) {
        let q = quantize_value(value, scale, 2);
        // 2-bit signed: min = -(1<<1) = -2, max = (1<<1)-1 = 1
        prop_assert!(
            (-2..=1).contains(&q),
            "quantized value {} out of [-2, 1] for input={}, scale={}",
            q, value, scale
        );
    }

    /// dequantize_value always produces a finite result for finite inputs.
    #[test]
    fn prop_dequantize_value_finite(
        q in -1i8..=1i8,
        scale in 0.01f32..100.0f32,
    ) {
        let v = dequantize_value(q, scale);
        prop_assert!(v.is_finite(), "non-finite dequantized value: {}", v);
    }
}

// ===================================================================
// 3. Scale factors are always positive
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// calculate_scale returns a positive finite value for any finite non-empty input.
    #[test]
    fn prop_scale_always_positive(
        data in prop::collection::vec(-1000.0f32..1000.0f32, 1..128),
    ) {
        let scale = calculate_scale(&data, 2);
        prop_assert!(scale > 0.0, "scale must be positive, got {}", scale);
        prop_assert!(scale.is_finite(), "scale must be finite, got {}", scale);
    }

    /// Grouped scales produce one scale per block, all positive.
    #[test]
    fn prop_grouped_scales_per_block(
        nblocks in 1usize..=16,
    ) {
        let block_size = 32usize;
        let data: Vec<f32> = (0..nblocks * block_size)
            .map(|i| (i as f32) * 0.1 - 5.0)
            .collect();
        let scales = calculate_grouped_scales(&data, block_size, 2);
        prop_assert_eq!(
            scales.len(), nblocks,
            "expected {} scales, got {}", nblocks, scales.len()
        );
        for (i, &s) in scales.iter().enumerate() {
            prop_assert!(s > 0.0, "scale[{}] must be positive, got {}", i, s);
            prop_assert!(s.is_finite(), "scale[{}] must be finite, got {}", i, s);
        }
    }
}

// ===================================================================
// 4. Zero input produces zero output
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// All-zeros input quantizes and dequantizes back to all zeros.
    #[test]
    fn prop_zero_input_zero_output(
        nblocks in 1usize..=8,
    ) {
        let len = nblocks * 32;
        let zeros = vec![0.0f32; len];
        let device = Device::Cpu;
        let tensor = create_tensor_from_f32(zeros.clone(), &[len], &device)
            .expect("tensor creation");

        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize(&tensor, &device).expect("quantize");
        let reconstructed = quantizer.dequantize(&quantized, &device).expect("dequantize");

        let recon_data: Vec<f32> = reconstructed.to_vec().expect("to_vec");
        for (i, v) in recon_data.iter().enumerate() {
            prop_assert!(
                v.abs() < 1e-6,
                "expected ~0 at index {}, got {}", i, v
            );
        }
    }
}

// ===================================================================
// 5. Pack/unpack roundtrip preserves element count
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// pack then unpack 2-bit values preserves the original values.
    #[test]
    fn prop_pack_unpack_roundtrip_len(
        values in prop::collection::vec(-2i8..=1i8, 4..128),
    ) {
        let packed = pack_2bit_values(&values);
        let unpacked = unpack_2bit_values(&packed, values.len());
        prop_assert_eq!(
            unpacked.len(), values.len(),
            "length mismatch after roundtrip"
        );
        prop_assert_eq!(unpacked, values, "value mismatch after roundtrip");
    }
}

// ===================================================================
// 6. Optimal block size is always a power of two
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// calculate_optimal_block_size returns a power-of-two in [16, 1024].
    #[test]
    fn prop_optimal_block_size_power_of_two(
        tensor_size in 32usize..100_000,
        target_blocks in 1usize..1_000,
    ) {
        let bs = calculate_optimal_block_size(tensor_size, target_blocks);
        prop_assert!(bs.is_power_of_two(), "block size {} is not power of two", bs);
        prop_assert!(bs >= 16, "block size {} < 16", bs);
        prop_assert!(bs <= 1024, "block size {} > 1024", bs);
    }
}

// ===================================================================
// 7. MSE is zero for identical inputs
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// calculate_mse of a vector with itself is exactly zero.
    #[test]
    fn prop_mse_zero_for_identical(
        data in prop::collection::vec(-100.0f32..100.0f32, 1..256),
    ) {
        let mse = calculate_mse(&data, &data).expect("mse");
        prop_assert!(
            mse.abs() < 1e-10,
            "MSE of identical vectors should be ~0, got {}", mse
        );
    }
}

// ===================================================================
// 8. QuantizedTensor numel matches shape product
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// QuantizedTensor::numel() equals the product of its shape dimensions.
    #[test]
    fn prop_quantized_tensor_numel_shape_product(
        d0 in 1usize..=64,
        d1 in 1usize..=64,
    ) {
        let qt = bitnet_quantization::QuantizedTensor::new(
            vec![0u8; 8],
            vec![1.0f32],
            vec![d0, d1],
            bitnet_common::types::QuantizationType::I2S,
        );
        prop_assert_eq!(qt.numel(), d0 * d1);
    }
}
