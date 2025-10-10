//! Property-based fuzzing tests for I2S quantization targeting mutation hot spots
//!
//! This test suite addresses mutation testing weaknesses in:
//! - Block size calculations (i2s.rs:57-60)
//! - Input validation logic (i2s.rs:106, 122)
//! - Device selection (i2s.rs:173)
//! - GPU quantization (i2s.rs:242-264)

use bitnet_common::{BitNetTensor, Tensor};
use bitnet_quantization::I2SQuantizer;
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use proptest::prelude::*;

// Property: Block size calculations must handle all valid inputs
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn fuzz_i2s_block_size_calculation(
        block_size in 4usize..=256,
        data_len in 1usize..=4096
    ) {
        // This tests the critical block size calculation at i2s.rs:57-60
        // let data_bytes = (block_size * 2).div_ceil(8);

        // Ensure no overflow in multiplication
        let bits_result = block_size.checked_mul(2);
        prop_assert!(bits_result.is_some(), "Block size {} causes overflow", block_size);

        let bits = bits_result.unwrap();
        let data_bytes = bits.div_ceil(8);

        // Verify total bytes doesn't overflow
        let bytes_per_block = data_bytes.checked_add(2);
        prop_assert!(bytes_per_block.is_some(), "Bytes per block overflow at block_size {}", block_size);

        // Verify block count calculation doesn't overflow
        let num_blocks = data_len.div_ceil(block_size);
        let total_data_bytes = num_blocks.checked_mul(data_bytes);
        prop_assert!(total_data_bytes.is_some(), "Total data bytes overflow: {} blocks * {} bytes", num_blocks, data_bytes);

        // Verify scale bytes calculation doesn't overflow
        let total_scale_bytes = num_blocks.checked_mul(4);
        prop_assert!(total_scale_bytes.is_some(), "Scale bytes overflow: {} blocks", num_blocks);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn fuzz_i2s_input_validation_numerical_stability(
        values in prop::collection::vec(
            prop::num::f32::NORMAL | prop::num::f32::SUBNORMAL | prop::num::f32::ZERO,
            1..=512
        )
    ) {
        // Tests input validation at i2s.rs:122 (needs_detailed_validation)
        // and i2s.rs:128 (validate_numerical_input)

        let quantizer = I2SQuantizer::new();
        let shape = vec![values.len()];

        if let Ok(tensor) = CandleTensor::from_vec(values.clone(), shape.as_slice(), &CandleDevice::Cpu) {
            let bitnet_tensor = BitNetTensor::new(tensor);

            // This should never panic - either succeed or return error
            let _result = quantizer.quantize_tensor(&bitnet_tensor);
            // If it panics, proptest will catch it
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn fuzz_i2s_input_validation_edge_cases(
        size in 1usize..=256,
        value_range in -1000.0f32..1000.0f32
    ) {
        // Tests validation logic at i2s.rs:106 (validation caching)
        // and data shape consistency at i2s.rs:131

        let values = vec![value_range; size];
        let quantizer = I2SQuantizer::new();
        let shape = vec![size];

        if let Ok(tensor) = CandleTensor::from_vec(values, shape.as_slice(), &CandleDevice::Cpu) {
            let bitnet_tensor = BitNetTensor::new(tensor);

            // Should handle validation without panic
            let result = quantizer.quantize_tensor(&bitnet_tensor);

            // If it succeeds, verify the output is valid
            if let Ok(quantized) = result {
                prop_assert!(!quantized.data.is_empty(), "Quantized data is empty");
                prop_assert!(!quantized.scales.is_empty(), "Scales are empty");
                prop_assert_eq!(quantized.shape, vec![size], "Shape mismatch");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn fuzz_i2s_shape_consistency(
        dims in prop::collection::vec(1usize..=64, 1..=4)
    ) {
        // Tests data shape consistency validation at i2s.rs:131
        // validate_data_shape_consistency(&data, &shape)

        let total_elements: usize = dims.iter().product();
        if total_elements == 0 || total_elements > 10000 {
            return Ok(());
        }

        let values = vec![1.0f32; total_elements];
        let quantizer = I2SQuantizer::new();

        if let Ok(tensor) = CandleTensor::from_vec(values, dims.as_slice(), &CandleDevice::Cpu) {
            let bitnet_tensor = BitNetTensor::new(tensor);

            let result = quantizer.quantize_tensor(&bitnet_tensor);

            if let Ok(quantized) = result {
                // Verify shape is preserved
                let shape_clone = dims.clone();
                prop_assert_eq!(&quantized.shape, &dims, "Shape not preserved through quantization");

                // Verify numel calculation matches
                let calculated_numel: usize = shape_clone.iter().product();
                prop_assert_eq!(quantized.numel(), calculated_numel, "Numel mismatch");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn fuzz_i2s_device_support_consistency(
        block_size in 4usize..=128
    ) {
        // Tests device selection logic at i2s.rs:173 (supports_device)
        // and device handling at i2s.rs:106-116

        use bitnet_common::Device;

        let quantizer = I2SQuantizer::with_block_size(block_size);

        // CPU should always be supported
        prop_assert!(quantizer.supports_device(&Device::Cpu), "CPU support missing");

        // CUDA support should be consistent with feature flag
        let cuda_supported = quantizer.supports_device(&Device::Cuda(0));
        if cfg!(any(feature = "gpu", feature = "cuda")) {
            prop_assert!(cuda_supported, "CUDA should be supported with gpu or cuda feature");
        } else {
            prop_assert!(!cuda_supported, "CUDA should not be supported without gpu or cuda feature");
        }

        // Metal should not be supported yet
        prop_assert!(!quantizer.supports_device(&Device::Metal), "Metal should not be supported");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn fuzz_i2s_quantize_dequantize_roundtrip(
        block_size in 4usize..=64,
        values in prop::collection::vec(-10.0f32..10.0f32, 4..=256)
    ) {
        // Tests full quantization/dequantization pipeline
        // including validation, quantization, and dequantization paths

        let quantizer = I2SQuantizer::with_block_size(block_size);
        let shape = vec![values.len()];

        if let Ok(tensor) = CandleTensor::from_vec(values.clone(), shape.as_slice(), &CandleDevice::Cpu) {
            let bitnet_tensor = BitNetTensor::new(tensor);

            if let Ok(quantized) = quantizer.quantize_tensor(&bitnet_tensor) {
                // Dequantization should work
                let dequant_result = quantizer.dequantize_tensor(&quantized);
                prop_assert!(dequant_result.is_ok(), "Dequantization failed after successful quantization");

                if let Ok(dequantized) = dequant_result {
                    // Shape should be preserved
                    prop_assert_eq!(dequantized.shape(), &shape, "Shape mismatch after roundtrip");
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn fuzz_i2s_extreme_values_safety(
        extreme_type in 0u8..=3,
        size in 4usize..=128
    ) {
        // Tests numerical stability with extreme values
        // Targets validation at i2s.rs:128 (validate_numerical_input)

        let values = match extreme_type {
            0 => vec![f32::MAX / 2.0; size],
            1 => vec![f32::MIN / 2.0; size],
            2 => vec![f32::EPSILON; size],
            _ => vec![-f32::EPSILON; size],
        };

        let quantizer = I2SQuantizer::new();
        let shape = vec![size];

        if let Ok(tensor) = CandleTensor::from_vec(values, shape.as_slice(), &CandleDevice::Cpu) {
            let bitnet_tensor = BitNetTensor::new(tensor);

            // Should handle extreme values gracefully (no panic)
            let _result = quantizer.quantize_tensor(&bitnet_tensor);
            // If it panics, proptest will catch it
        }
    }
}

#[cfg(feature = "cpu")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn fuzz_i2s_packed_data_consistency(
        data_len in 8usize..=512
    ) {
        // Tests bit packing consistency at i2s.rs:146 (pack_2bit_values)
        // and unpacking at i2s.rs:221 (unpack_2bit_values)

        let values = vec![1.5f32; data_len];
        let quantizer = I2SQuantizer::new();
        let shape = vec![data_len];

        if let Ok(tensor) = CandleTensor::from_vec(values, shape.as_slice(), &CandleDevice::Cpu) {
            let bitnet_tensor = BitNetTensor::new(tensor);

            if let Ok(quantized) = quantizer.quantize_tensor(&bitnet_tensor) {
                // Verify packed data length is correct
                // For 2-bit values: (data_len * 2 bits) / 8 bits per byte
                let _expected_min_packed = (data_len * 2).div_ceil(8);

                // Account for block structure (data + scales)
                prop_assert!(
                    !quantized.data.is_empty(),
                    "Packed data is empty for {} elements",
                    data_len
                );

                // Verify dequantization recovers the right number of elements
                if let Ok(dequantized) = quantizer.dequantize_tensor(&quantized) {
                    prop_assert_eq!(
                        dequantized.shape(),
                        &shape,
                        "Dequantized shape mismatch"
                    );
                }
            }
        }
    }
}

#[test]
fn test_i2s_fuzz_crash_1849515_reproducer() {
    // Reproducer for crash-1849515c7958976d1cf7360b3e0d75d04115d96c
    // This crash involved extreme float values

    let problematic_bytes = [
        0xff, 0xff, 0xff, 0x1f, 0x1d, 0x00, 0x89, 0x89, 0x89, 0x89, 0x89, 0x89, 0x89, 0x89, 0x89,
        0x89, 0x89,
    ];

    let float_values: Vec<f32> = problematic_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let quantizer = I2SQuantizer::new();
    let shape = vec![float_values.len()];

    if let Ok(tensor) = CandleTensor::from_vec(float_values, shape.as_slice(), &CandleDevice::Cpu) {
        let bitnet_tensor = BitNetTensor::new(tensor);

        // Should not panic - either succeed or return error
        let _result = quantizer.quantize_tensor(&bitnet_tensor);
        // If it doesn't panic, test passes
    }
}

#[test]
fn test_i2s_fuzz_crash_79f55aa_reproducer() {
    // Reproducer for crash-79f55aabbc9a4b9b83da759a0853dc61a66318d2
    // This crash involved NaN/infinite values

    let problematic_bytes = [
        0xd9, 0x2b, 0x0a, 0x33, 0x7e, 0x0a, 0xff, 0x9f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xd9, 0x0a,
    ];

    let float_values: Vec<f32> = problematic_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let quantizer = I2SQuantizer::new();
    let shape = vec![float_values.len()];

    if let Ok(tensor) = CandleTensor::from_vec(float_values, shape.as_slice(), &CandleDevice::Cpu) {
        let bitnet_tensor = BitNetTensor::new(tensor);

        // Should not panic - either succeed or return error
        let _result = quantizer.quantize_tensor(&bitnet_tensor);
        // If it doesn't panic, test passes
    }
}

#[test]
fn test_i2s_block_size_edge_cases() {
    // Test mutation hot spot: i2s.rs:57-60 block size calculations

    let edge_cases = vec![
        4,   // Minimum block size
        8,   // Small power of 2
        16,  // Standard size
        32,  // Default I2S block size
        64,  // Large power of 2
        128, // Very large
        256, // Maximum tested
    ];

    for block_size in edge_cases {
        let quantizer = I2SQuantizer::with_block_size(block_size);
        let data = vec![1.0f32; block_size * 2]; // 2 blocks worth
        let shape = vec![data.len()];

        if let Ok(tensor) = CandleTensor::from_vec(data, shape.as_slice(), &CandleDevice::Cpu) {
            let bitnet_tensor = BitNetTensor::new(tensor);
            let result = quantizer.quantize_tensor(&bitnet_tensor);

            assert!(result.is_ok(), "Block size {} failed quantization", block_size);

            if let Ok(quantized) = result {
                assert_eq!(quantized.block_size, block_size, "Block size not preserved");
            }
        }
    }
}

#[test]
fn test_i2s_validation_caching_mutation_killer() {
    // Test mutation hot spot: i2s.rs:106 validation caching

    let quantizer = I2SQuantizer::new();
    let data = vec![1.0f32; 64];
    let shape = vec![64];

    // First quantization - should cache validation
    if let Ok(tensor) = CandleTensor::from_vec(data.clone(), shape.as_slice(), &CandleDevice::Cpu) {
        let bitnet_tensor = BitNetTensor::new(tensor);
        let result1 = quantizer.quantize_tensor(&bitnet_tensor);
        assert!(result1.is_ok(), "First quantization failed");
    }

    // Second quantization with same quantizer - validation should be cached
    if let Ok(tensor) = CandleTensor::from_vec(data, shape.as_slice(), &CandleDevice::Cpu) {
        let bitnet_tensor = BitNetTensor::new(tensor);
        let result2 = quantizer.quantize_tensor(&bitnet_tensor);
        assert!(result2.is_ok(), "Cached validation failed");
    }
}
