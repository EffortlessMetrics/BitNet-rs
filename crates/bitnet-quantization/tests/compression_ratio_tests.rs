//! Compression ratio tests targeting surviving mutants in lib.rs:95
//!
//! This test suite specifically targets the arithmetic mutations that survived
//! in the compression_ratio() method to ensure correct calculations.

use bitnet_common::QuantizationType;
use bitnet_quantization::QuantizedTensor;
use proptest::prelude::*;

/// Test compression ratio calculations with known values
#[cfg(test)]
mod compression_ratio_validation {
    use super::*;

    #[test]
    fn test_compression_ratio_basic_calculation() {
        // Create a quantized tensor with known dimensions
        let data = vec![1, 2, 3, 4]; // 4 bytes of quantized data
        let scales = vec![1.0, 2.0]; // 2 scales = 8 bytes
        let shape = vec![16]; // 16 elements originally = 64 bytes FP32

        let tensor =
            QuantizedTensor::new_with_params(data, scales, None, shape, QuantizationType::I2S, 32);

        // Expected: 64 bytes / (4 + 8) = 64/12 = 5.333...
        let ratio = tensor.compression_ratio();
        assert!((ratio - (64.0 / 12.0)).abs() < 1e-6, "Expected ratio ~5.33, got {}", ratio);
    }

    #[test]
    fn test_compression_ratio_prevents_division_mutations() {
        // Test that + mutation is caught: original_bytes + compressed_bytes != ratio
        let data = vec![0; 100]; // 100 bytes quantized
        let scales = vec![1.0; 25]; // 25 scales = 100 bytes
        let shape = vec![800]; // 800 elements = 3200 bytes FP32

        let tensor =
            QuantizedTensor::new_with_params(data, scales, None, shape, QuantizationType::I2S, 32);

        let ratio = tensor.compression_ratio();

        // If + mutation survived: ratio would be (3200 + 200) / 200 = 17
        // Correct calculation: 3200 / 200 = 16
        assert!(
            (ratio - 16.0).abs() < 1e-6,
            "Division mutation detected: expected 16.0, got {}",
            ratio
        );

        // If - mutation survived: ratio would be (3200 - 200) / 200 = 15
        assert!(ratio > 15.5, "Subtraction mutation detected: got {}", ratio);

        // If * mutation survived: ratio would be 3200 * 200 = very large
        assert!(ratio < 100.0, "Multiplication mutation detected: got {}", ratio);
    }

    #[test]
    fn test_compression_ratio_scales_multiplication() {
        // Target the scales.len() * 4 calculation specifically
        let data = vec![0; 50]; // 50 bytes
        let scales = vec![1.0; 10]; // Should be 10 * 4 = 40 bytes
        let shape = vec![200]; // 200 * 4 = 800 bytes FP32

        let tensor =
            QuantizedTensor::new_with_params(data, scales, None, shape, QuantizationType::I2S, 32);

        let ratio = tensor.compression_ratio();
        let expected = 800.0 / (50.0 + 40.0); // 800 / 90 = 8.888...

        assert!(
            (ratio - expected).abs() < 1e-6,
            "Scales multiplication error: expected {}, got {}",
            expected,
            ratio
        );

        // If + mutation in scales: 10 + 4 = 14 bytes instead of 40
        // Would give 800 / (50 + 14) = 12.5
        assert!(ratio < 10.0, "Scales addition mutation detected: got {}", ratio);

        // If / mutation in scales: 10 / 4 = 2.5 bytes instead of 40
        // Would give 800 / (50 + 2.5) = 15.24
        assert!(ratio < 12.0, "Scales division mutation detected: got {}", ratio);
    }

    #[test]
    fn test_compression_ratio_zero_protection() {
        // Test the zero division protection
        let empty_data = vec![];
        let empty_scales = vec![];
        let shape = vec![0];

        let tensor = QuantizedTensor::new_with_params(
            empty_data,
            empty_scales,
            None,
            shape,
            QuantizationType::I2S,
            32,
        );

        let ratio = tensor.compression_ratio();
        assert_eq!(ratio, 1.0, "Zero bytes should return 1.0 ratio");

        // Edge case: non-zero elements but zero compressed bytes
        let shape = vec![100]; // 400 bytes FP32
        let tensor = QuantizedTensor::new_with_params(
            vec![],
            vec![],
            None,
            shape,
            QuantizationType::I2S,
            32,
        );

        let ratio = tensor.compression_ratio();
        assert_eq!(ratio, 1.0, "Zero compressed bytes should return 1.0");
    }

    #[test]
    fn test_compression_ratio_numel_calculation() {
        // Test that numel() calculation affects ratio correctly
        let data = vec![0; 64]; // 64 bytes
        let scales = vec![0.0; 16]; // 64 bytes

        // 2D tensor: 8x8 = 64 elements = 256 bytes FP32
        let shape_2d = vec![8, 8];
        let tensor_2d = QuantizedTensor::new_with_params(
            data.clone(),
            scales.clone(),
            None,
            shape_2d,
            QuantizationType::I2S,
            32,
        );

        // 1D tensor: 64 elements = 256 bytes FP32
        let shape_1d = vec![64];
        let tensor_1d = QuantizedTensor::new_with_params(
            data,
            scales,
            None,
            shape_1d,
            QuantizationType::I2S,
            32,
        );

        let ratio_2d = tensor_2d.compression_ratio();
        let ratio_1d = tensor_1d.compression_ratio();

        // Should be equal: 256 / 128 = 2.0
        assert!((ratio_2d - 2.0).abs() < 1e-6, "2D ratio should be 2.0, got {}", ratio_2d);
        assert!((ratio_1d - 2.0).abs() < 1e-6, "1D ratio should be 2.0, got {}", ratio_1d);
        assert!((ratio_2d - ratio_1d).abs() < 1e-6, "Ratios should be equal");
    }

    #[test]
    fn test_compression_ratio_minimum_bound() {
        // Test the .max(1.0) bound on ratio
        let data = vec![0; 1000]; // Very large compressed data
        let scales = vec![0.0; 1000]; // Very large scales
        let shape = vec![10]; // Small original: 40 bytes

        let tensor =
            QuantizedTensor::new_with_params(data, scales, None, shape, QuantizationType::I2S, 32);

        let ratio = tensor.compression_ratio();

        // Even if compressed > original, ratio should be at least 1.0
        assert!(ratio >= 1.0, "Ratio should be at least 1.0, got {}", ratio);

        // Should actually be exactly 1.0 due to max() bound
        assert_eq!(ratio, 1.0, "Ratio should be clamped to 1.0 when compressed > original");
    }

    proptest! {
        #[test]
        fn test_compression_ratio_property_based(
            data_len in 0usize..1000,
            scales_count in 0usize..100,
            element_count in 1usize..10000
        ) {
            let data = vec![0; data_len];
            let scales = vec![0.0; scales_count];
            let shape = vec![element_count];

            let tensor = QuantizedTensor::new_with_params(
                data, scales, None, shape, QuantizationType::I2S, 32
            );

            let ratio = tensor.compression_ratio();

            // Basic sanity checks
            prop_assert!(ratio >= 1.0, "Ratio must be at least 1.0");
            prop_assert!(ratio.is_finite(), "Ratio must be finite");

            // Mathematical consistency check
            let original_bytes = element_count * 4;
            let compressed_bytes = data_len + scales_count * 4;

            if compressed_bytes > 0 {
                let expected = (original_bytes as f32 / compressed_bytes as f32).max(1.0);
                prop_assert!((ratio - expected).abs() < 1e-6,
                    "Ratio calculation inconsistent: expected {}, got {}", expected, ratio);
            } else {
                prop_assert_eq!(ratio, 1.0, "Zero compressed bytes should give ratio 1.0");
            }
        }

        #[test]
        fn test_compression_ratio_arithmetic_invariants(
            element_count in 1usize..1000,
            data_scale in 1usize..10
        ) {
            let data_len = element_count / data_scale;
            let scales_count = element_count / (data_scale * 4);

            let data = vec![0; data_len];
            let scales = vec![0.0; scales_count];
            let shape = vec![element_count];

            let tensor = QuantizedTensor::new_with_params(
                data, scales, None, shape, QuantizationType::I2S, 32
            );

            let ratio = tensor.compression_ratio();

            // Verify arithmetic operations haven't been mutated
            let original_bytes = element_count * 4;
            let compressed_bytes = data_len + scales_count * 4;

            if compressed_bytes > 0 {
                // Test division specifically
                let manual_ratio = original_bytes as f32 / compressed_bytes as f32;
                let bounded_ratio = manual_ratio.max(1.0);

                prop_assert!((ratio - bounded_ratio).abs() < 1e-6,
                    "Arithmetic mutation detected: expected {}, got {}", bounded_ratio, ratio);

                // Ensure it's not using wrong operations
                let wrong_add = (original_bytes + compressed_bytes) as f32 / compressed_bytes as f32;
                let wrong_sub = ((original_bytes as i64 - compressed_bytes as i64).abs()) as f32 / compressed_bytes as f32;
                let wrong_mul = (original_bytes * compressed_bytes) as f32 / compressed_bytes as f32;

                if wrong_add != bounded_ratio {
                    prop_assert!((ratio - wrong_add).abs() > 1e-6, "Addition mutation detected");
                }
                if wrong_sub != bounded_ratio {
                    prop_assert!((ratio - wrong_sub).abs() > 1e-6, "Subtraction mutation detected");
                }
                if wrong_mul != bounded_ratio {
                    prop_assert!((ratio - wrong_mul).abs() > 1e-6, "Multiplication mutation detected");
                }
            }
        }
    }
}
