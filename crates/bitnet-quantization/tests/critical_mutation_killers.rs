//! Critical mutation killer tests targeting specific surviving mutants from analysis
//!
//! Based on mutation analysis, this specifically targets:
//! 1. lib.rs:95 - Compression ratio arithmetic (*, +, -, / mutations)
//! 2. Mathematical operator validation
//! 3. Boundary condition testing

use bitnet_common::QuantizationType;
use bitnet_quantization::QuantizedTensor;

#[cfg(test)]
mod compression_ratio_mutation_killers {
    use super::*;

    #[test]
    fn test_kill_compression_ratio_plus_mutation() {
        // Kill "replace * with +" mutation in scales.len() * 4 calculation
        let data = vec![0; 50];
        let scales = vec![1.0; 10]; // 10 scales should be 10 * 4 = 40 bytes
        let shape = vec![200]; // 200 * 4 = 800 bytes FP32

        let tensor =
            QuantizedTensor::new_with_params(data, scales, None, shape, QuantizationType::I2S, 32);

        let ratio = tensor.compression_ratio();
        let expected = 800.0 / (50.0 + 40.0); // 800 / 90 = 8.888...

        // If + mutation survived: would be 10 + 4 = 14 bytes instead of 40
        // Giving 800 / (50 + 14) = 12.5, which would fail this assertion
        assert!(
            ratio < 10.0,
            "Compression ratio {} too high - possible + mutation in scales calculation",
            ratio
        );
        assert!(
            (ratio - expected).abs() < 0.001,
            "Expected {}, got {} - possible arithmetic mutation",
            expected,
            ratio
        );
    }

    #[test]
    fn test_kill_compression_ratio_subtraction_mutation() {
        // Kill "replace + with -" mutation in original_bytes + compressed_bytes calculation
        let data = vec![0; 100]; // 100 bytes quantized
        let scales = vec![1.0; 25]; // 25 scales = 100 bytes
        let shape = vec![800]; // 800 elements = 3200 bytes FP32

        let tensor =
            QuantizedTensor::new_with_params(data, scales, None, shape, QuantizationType::I2S, 32);

        let ratio = tensor.compression_ratio();

        // Correct calculation: 3200 / (100 + 100) = 16.0
        // If - mutation: (3200 - 200) / 200 = 15.0
        assert!(
            (ratio - 16.0).abs() < 0.001,
            "Expected 16.0, got {} - possible subtraction mutation",
            ratio
        );

        // This would catch the subtraction mutation specifically
        assert!(ratio > 15.5, "Ratio {} too low - possible subtraction mutation", ratio);
    }

    #[test]
    fn test_kill_compression_ratio_multiplication_mutation() {
        // Kill "replace / with *" mutation in division calculation
        let data = vec![0; 64]; // 64 bytes
        let scales = vec![0.0; 16]; // 64 bytes
        let shape = vec![64]; // 256 bytes FP32

        let tensor =
            QuantizedTensor::new_with_params(data, scales, None, shape, QuantizationType::I2S, 32);

        let ratio = tensor.compression_ratio();

        // Correct: 256 / 128 = 2.0
        // If * mutation: 256 * 128 = massive number
        assert!(ratio < 100.0, "Ratio {} way too high - possible multiplication mutation", ratio);
        assert!((ratio - 2.0).abs() < 0.001, "Expected 2.0, got {}", ratio);
    }

    #[test]
    fn test_kill_division_mutation() {
        // Kill "replace * with /" mutation in numel calculation
        let data = vec![0; 10];
        let scales = vec![0.0; 2];

        // 2D shape: 4x4 = 16 elements
        let shape_2d = vec![4, 4];
        let tensor = QuantizedTensor::new_with_params(
            data.clone(),
            scales.clone(),
            None,
            shape_2d,
            QuantizationType::I2S,
            32,
        );

        let numel = tensor.numel();
        assert_eq!(numel, 16, "2D numel should be 16, got {} - possible division mutation", numel);

        // 3D shape: 2x2x4 = 16 elements
        let shape_3d = vec![2, 2, 4];
        let tensor_3d = QuantizedTensor::new_with_params(
            data,
            scales,
            None,
            shape_3d,
            QuantizationType::I2S,
            32,
        );

        let numel_3d = tensor_3d.numel();
        assert_eq!(
            numel_3d, 16,
            "3D numel should be 16, got {} - possible division mutation",
            numel_3d
        );
    }

    #[test]
    fn test_kill_comparison_mutations() {
        // Kill comparison mutations in zero check
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

        // Should return 1.0 for zero case, not fail
        assert_eq!(ratio, 1.0, "Zero case should return 1.0, got {}", ratio);

        // Test boundary at exactly 1.0 ratio
        let equal_data = vec![0; 100]; // 100 bytes
        let equal_scales = vec![0.0; 75]; // 300 bytes (75 * 4)
        let equal_shape = vec![100]; // 400 bytes FP32 (100 * 4)

        let equal_tensor = QuantizedTensor::new_with_params(
            equal_data,
            equal_scales,
            None,
            equal_shape,
            QuantizationType::I2S,
            32,
        );

        let equal_ratio = equal_tensor.compression_ratio();

        // 400 / 400 = 1.0, but with .max(1.0) should still be 1.0
        assert!(equal_ratio >= 1.0, "Ratio should be at least 1.0, got {}", equal_ratio);
        assert_eq!(equal_ratio, 1.0, "Equal size should give exactly 1.0, got {}", equal_ratio);
    }

    #[test]
    fn test_kill_boolean_logic_mutations() {
        // Test to kill any boolean logic mutations in conditional checks
        let test_cases = vec![
            // (data_len, scales_count, elements, expected_passes)
            (0, 0, 0, true),      // All zero case
            (1, 1, 1, true),      // Minimal case
            (100, 25, 200, true), // Normal case
        ];

        for (data_len, scales_count, elements, should_pass) in test_cases {
            let data = vec![0; data_len];
            let scales = vec![0.0; scales_count];
            let shape = vec![elements];

            let tensor = QuantizedTensor::new_with_params(
                data,
                scales,
                None,
                shape,
                QuantizationType::I2S,
                32,
            );

            let ratio = tensor.compression_ratio();

            if should_pass {
                assert!(ratio >= 1.0, "Valid case should have ratio >= 1.0, got {}", ratio);
                assert!(ratio.is_finite(), "Ratio should be finite, got {}", ratio);
            }
        }
    }

    #[test]
    fn test_kill_boundary_mutations() {
        // Kill boundary condition mutations around zero
        let cases = vec![
            (vec![0; 1], vec![0.0; 0], vec![1]), // Minimal data, no scales
            (vec![0; 0], vec![0.0; 1], vec![1]), // No data, minimal scales
            (vec![0; 1], vec![0.0; 1], vec![4]), // Normal minimal case
        ];

        for (data, scales, shape) in cases {
            let tensor = QuantizedTensor::new_with_params(
                data.clone(),
                scales.clone(),
                None,
                shape.clone(),
                QuantizationType::I2S,
                32,
            );

            let ratio = tensor.compression_ratio();

            // Should not panic and should give valid result
            assert!(
                ratio.is_finite(),
                "Ratio should be finite for data={}, scales={}, shape={:?}",
                data.len(),
                scales.len(),
                shape
            );
            assert!(
                ratio >= 1.0,
                "Ratio should be >= 1.0 for data={}, scales={}, shape={:?}",
                data.len(),
                scales.len(),
                shape
            );
        }
    }

    /// Property-based test to kill systematic arithmetic mutations
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn kill_arithmetic_mutations_property(
            data_len in 0usize..200,
            scales_count in 0usize..50,
            element_count in 1usize..1000
        ) {
            let data = vec![0; data_len];
            let scales = vec![0.0; scales_count];
            let shape = vec![element_count];

            let tensor = QuantizedTensor::new_with_params(
                data, scales, None, shape, QuantizationType::I2S, 32
            );

            let ratio = tensor.compression_ratio();

            // Kill all arithmetic mutations by checking mathematical consistency
            let original_bytes = element_count * 4;
            let compressed_bytes = data_len + scales_count * 4;

            if compressed_bytes > 0 {
                let expected = (original_bytes as f32 / compressed_bytes as f32).max(1.0);
                prop_assert!((ratio - expected).abs() < 1e-5,
                    "Arithmetic inconsistency: expected {}, got {}", expected, ratio);

                // Kill specific mutations:

                // + mutation: original + compressed instead of original / compressed
                let wrong_add = (original_bytes + compressed_bytes) as f32 / compressed_bytes as f32;
                if (wrong_add - expected).abs() > 1e-5 {
                    prop_assert!((ratio - wrong_add).abs() > 1e-5, "Addition mutation detected");
                }

                // - mutation: abs(original - compressed) instead of original / compressed
                let wrong_sub = ((original_bytes as i64 - compressed_bytes as i64).abs()) as f32 / compressed_bytes as f32;
                if (wrong_sub - expected).abs() > 1e-5 {
                    prop_assert!((ratio - wrong_sub).abs() > 1e-5, "Subtraction mutation detected");
                }

                // * mutation: original * compressed instead of original / compressed
                let wrong_mul = (original_bytes * compressed_bytes) as f32 / compressed_bytes as f32;
                if wrong_mul != expected {
                    prop_assert!((ratio - wrong_mul).abs() > 1e-5, "Multiplication mutation detected");
                }
            } else {
                prop_assert_eq!(ratio, 1.0, "Zero bytes should give ratio 1.0");
            }

            // General invariants
            prop_assert!(ratio >= 1.0, "Ratio must be at least 1.0");
            prop_assert!(ratio.is_finite(), "Ratio must be finite");
        }
    }
}
