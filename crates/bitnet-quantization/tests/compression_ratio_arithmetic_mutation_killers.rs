//! Compression ratio arithmetic mutation killers targeting specific surviving mutants
//!
//! This test file specifically targets the 3 surviving mutants in lib.rs:95
//! in the QuantizedTensor::compression_ratio() method:
//!
//! 1. + -> - mutation: self.data.len() + self.scales.len() * 4  => self.data.len() - self.scales.len() * 4
//! 2. * -> + mutation: self.data.len() + self.scales.len() * 4  => self.data.len() + self.scales.len() + 4
//! 3. * -> / mutation: self.data.len() + self.scales.len() * 4  => self.data.len() + self.scales.len() / 4

use bitnet_common::QuantizationType;
use bitnet_quantization::QuantizedTensor;

#[cfg(test)]
mod arithmetic_mutation_killers {
    use super::*;

    #[test]
    fn test_kill_addition_to_subtraction_mutation() {
        // Test case specifically designed to kill + -> - mutation
        // Original: data.len() + scales.len() * 4
        // Mutated:  data.len() - scales.len() * 4

        let data = vec![0u8; 50];  // 50 bytes of data
        let scales = vec![1.0f32; 20];  // 20 scales = 80 bytes (20 * 4)
        let shape = vec![400];  // 400 elements = 1600 bytes FP32 (400 * 4)

        let tensor = QuantizedTensor::new_with_params(
            data, scales, None, shape, QuantizationType::I2S, 32
        );

        let ratio = tensor.compression_ratio();

        // Correct calculation: 1600 / (50 + 80) = 1600 / 130 = 12.307...
        let expected_correct = 1600.0 / 130.0;

        // Mutated calculation: 1600 / (50 - 80) = 1600 / (-30) = negative (clamped to 1.0)
        let expected_mutated = 1.0;  // Division by negative number, clamped

        assert!(
            (ratio - expected_correct).abs() < 1e-6,
            "Expected correct ratio {}, got {}. Mutation NOT killed!",
            expected_correct, ratio
        );

        assert!(
            (ratio - expected_mutated).abs() > 1e-6,
            "Addition to subtraction mutation detected! Expected {}, got {}",
            expected_correct, ratio
        );
    }

    #[test]
    fn test_kill_multiplication_to_addition_mutation() {
        // Test case specifically designed to kill * -> + mutation
        // Original: data.len() + scales.len() * 4
        // Mutated:  data.len() + scales.len() + 4

        let data = vec![0u8; 100];  // 100 bytes of data
        let scales = vec![1.0f32; 50];  // 50 scales normally = 200 bytes (50 * 4)
        let shape = vec![800];  // 800 elements = 3200 bytes FP32 (800 * 4)

        let tensor = QuantizedTensor::new_with_params(
            data, scales, None, shape, QuantizationType::I2S, 32
        );

        let ratio = tensor.compression_ratio();

        // Correct calculation: 3200 / (100 + 200) = 3200 / 300 = 10.666...
        let expected_correct = 3200.0 / 300.0;

        // Mutated calculation: 3200 / (100 + 50 + 4) = 3200 / 154 = 20.779...
        let expected_mutated = 3200.0 / 154.0;

        assert!(
            (ratio - expected_correct).abs() < 1e-6,
            "Expected correct ratio {}, got {}. Mutation NOT killed!",
            expected_correct, ratio
        );

        assert!(
            (ratio - expected_mutated).abs() > 1e-6,
            "Multiplication to addition mutation detected! Expected {}, got {}. Difference: {}",
            expected_correct, ratio, (ratio - expected_mutated).abs()
        );
    }

    #[test]
    fn test_kill_multiplication_to_division_mutation() {
        // Test case specifically designed to kill * -> / mutation
        // Original: data.len() + scales.len() * 4
        // Mutated:  data.len() + scales.len() / 4

        let data = vec![0u8; 60];  // 60 bytes of data
        let scales = vec![1.0f32; 32];  // 32 scales normally = 128 bytes (32 * 4)
        let shape = vec![400];  // 400 elements = 1600 bytes FP32 (400 * 4)

        let tensor = QuantizedTensor::new_with_params(
            data, scales, None, shape, QuantizationType::I2S, 32
        );

        let ratio = tensor.compression_ratio();

        // Correct calculation: 1600 / (60 + 128) = 1600 / 188 = 8.510...
        let expected_correct = 1600.0 / 188.0;

        // Mutated calculation: 1600 / (60 + 8) = 1600 / 68 = 23.529...
        let expected_mutated = 1600.0 / 68.0;

        assert!(
            (ratio - expected_correct).abs() < 1e-6,
            "Expected correct ratio {}, got {}. Mutation NOT killed!",
            expected_correct, ratio
        );

        assert!(
            (ratio - expected_mutated).abs() > 1e-6,
            "Multiplication to division mutation detected! Expected {}, got {}. Difference: {}",
            expected_correct, ratio, (ratio - expected_mutated).abs()
        );
    }

    #[test]
    fn test_kill_all_arithmetic_mutations_comprehensive() {
        // Comprehensive test case that should kill all three mutations

        let test_cases = vec![
            // (data_len, scales_count, element_count, test_name)
            (40, 10, 320, "small_tensor"),
            (100, 25, 800, "medium_tensor"),
            (200, 60, 1600, "large_tensor"),
        ];

        for (data_len, scales_count, element_count, test_name) in test_cases {
            let data = vec![0u8; data_len];
            let scales = vec![1.0f32; scales_count];
            let shape = vec![element_count];

            let tensor = QuantizedTensor::new_with_params(
                data, scales, None, shape, QuantizationType::I2S, 32
            );

            let ratio = tensor.compression_ratio();

            // Calculate all possible mutations
            let original_bytes = element_count * 4;
            let compressed_bytes_correct = data_len + scales_count * 4;
            let compressed_bytes_sub_mutation = data_len.saturating_sub(scales_count * 4);
            let compressed_bytes_add_mutation = data_len + scales_count + 4;
            let compressed_bytes_div_mutation = data_len + scales_count / 4;

            let expected_correct = if compressed_bytes_correct > 0 {
                (original_bytes as f32 / compressed_bytes_correct as f32).max(1.0)
            } else {
                1.0
            };

            let expected_sub_mutation = if compressed_bytes_sub_mutation > 0 {
                (original_bytes as f32 / compressed_bytes_sub_mutation as f32).max(1.0)
            } else {
                1.0
            };

            let expected_add_mutation = if compressed_bytes_add_mutation > 0 {
                (original_bytes as f32 / compressed_bytes_add_mutation as f32).max(1.0)
            } else {
                1.0
            };

            let expected_div_mutation = if compressed_bytes_div_mutation > 0 {
                (original_bytes as f32 / compressed_bytes_div_mutation as f32).max(1.0)
            } else {
                1.0
            };

            // Verify correct calculation
            assert!(
                (ratio - expected_correct).abs() < 1e-6,
                "{}: Expected correct ratio {}, got {}",
                test_name, expected_correct, ratio
            );

            // Kill + -> - mutation
            if expected_correct != expected_sub_mutation {
                assert!(
                    (ratio - expected_sub_mutation).abs() > 1e-6,
                    "{}: Addition to subtraction mutation not killed! Expected {}, got {}, sub_mutation={}",
                    test_name, expected_correct, ratio, expected_sub_mutation
                );
            }

            // Kill * -> + mutation
            if expected_correct != expected_add_mutation {
                assert!(
                    (ratio - expected_add_mutation).abs() > 1e-6,
                    "{}: Multiplication to addition mutation not killed! Expected {}, got {}, add_mutation={}",
                    test_name, expected_correct, ratio, expected_add_mutation
                );
            }

            // Kill * -> / mutation
            if expected_correct != expected_div_mutation {
                assert!(
                    (ratio - expected_div_mutation).abs() > 1e-6,
                    "{}: Multiplication to division mutation not killed! Expected {}, got {}, div_mutation={}",
                    test_name, expected_correct, ratio, expected_div_mutation
                );
            }
        }
    }

    #[test]
    fn test_edge_case_arithmetic_mutations() {
        // Edge cases that specifically target the arithmetic operations

        // Case 1: Zero scales (should trigger specific mutation patterns)
        let tensor_no_scales = QuantizedTensor::new_with_params(
            vec![0u8; 64], vec![], None, vec![256], QuantizationType::I2S, 32
        );

        let ratio_no_scales = tensor_no_scales.compression_ratio();
        let expected_no_scales = 1024.0 / 64.0; // 256*4 / 64 = 16.0

        assert!(
            (ratio_no_scales - expected_no_scales).abs() < 1e-6,
            "Zero scales case failed: expected {}, got {}",
            expected_no_scales, ratio_no_scales
        );

        // Case 2: Large scales count to amplify multiplication mutations
        let tensor_many_scales = QuantizedTensor::new_with_params(
            vec![0u8; 10], vec![1.0f32; 100], None, vec![200], QuantizationType::I2S, 32
        );

        let ratio_many_scales = tensor_many_scales.compression_ratio();

        // Correct: 800 / (10 + 400) = 800 / 410 = 1.95...
        let expected_many_scales = 800.0 / 410.0;

        // Mutated * -> +: 800 / (10 + 100 + 4) = 800 / 114 = 7.01...
        let mutated_add = 800.0 / 114.0;

        // Mutated * -> /: 800 / (10 + 25) = 800 / 35 = 22.85...
        let mutated_div = 800.0 / 35.0;

        assert!(
            (ratio_many_scales - expected_many_scales).abs() < 1e-6,
            "Many scales case failed: expected {}, got {}",
            expected_many_scales, ratio_many_scales
        );

        assert!(
            (ratio_many_scales - mutated_add).abs() > 0.1,
            "Multiplication to addition mutation not detected: {} vs {}",
            ratio_many_scales, mutated_add
        );

        assert!(
            (ratio_many_scales - mutated_div).abs() > 0.1,
            "Multiplication to division mutation not detected: {} vs {}",
            ratio_many_scales, mutated_div
        );
    }

    #[test]
    fn test_precise_arithmetic_validation() {
        // Very precise test with carefully chosen numbers to maximize mutation detection

        let data = vec![0u8; 17];  // Prime number
        let scales = vec![1.0f32; 13];  // Another prime, 13 * 4 = 52
        let shape = vec![89];  // Prime number, 89 * 4 = 356

        let tensor = QuantizedTensor::new_with_params(
            data, scales, None, shape, QuantizationType::I2S, 32
        );

        let ratio = tensor.compression_ratio();

        // Correct: 356 / (17 + 52) = 356 / 69 = 5.159...
        let expected_correct = 356.0 / 69.0;

        // + -> - mutation: 356 / (17 - 52) = 356 / (-35) = negative, clamped to 1.0
        let expected_sub = 1.0;

        // * -> + mutation: 356 / (17 + 13 + 4) = 356 / 34 = 10.470...
        let expected_add = 356.0 / 34.0;

        // * -> / mutation: 356 / (17 + 3) = 356 / 20 = 17.8
        let expected_div = 356.0 / 20.0;

        // Verify correct calculation
        assert!(
            (ratio - expected_correct).abs() < 1e-6,
            "Precise test failed: expected {}, got {}",
            expected_correct, ratio
        );

        // All mutations should produce different results
        assert!(
            (expected_correct - expected_sub).abs() > 1.0,
            "Subtraction mutation would not be detectable"
        );

        assert!(
            (expected_correct - expected_add).abs() > 1.0,
            "Addition mutation would not be detectable"
        );

        assert!(
            (expected_correct - expected_div).abs() > 1.0,
            "Division mutation would not be detectable"
        );

        // Ensure the actual ratio matches correct and not any mutation
        assert!(
            (ratio - expected_sub).abs() > 1.0,
            "Subtraction mutation detected in result!"
        );

        assert!(
            (ratio - expected_add).abs() > 1.0,
            "Addition mutation detected in result!"
        );

        assert!(
            (ratio - expected_div).abs() > 1.0,
            "Division mutation detected in result!"
        );
    }
}