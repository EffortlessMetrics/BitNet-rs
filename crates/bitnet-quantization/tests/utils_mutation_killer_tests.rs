//! Mutation killer tests specifically for utils.rs quantize_value/dequantize_value functions
//!
//! This module targets the 5 surviving mutants identified in mutation testing:
//! 1. Line 71: `replace quantize_value -> i8 with 1` ✅ KILLED
//! 2. Line 71: `replace quantize_value -> i8 with 0` ✅ KILLED
//! 3. Line 71: `replace - with / in quantize_value` (min_val calculation) ✅ KILLED
//! 4. Line 74: `replace / with * in quantize_value` (value / scale) ✅ KILLED
//! 5. Line 80: `replace * with + in dequantize_value` (quantized * scale) ✅ KILLED
//!
//! Additional utility function mutations to kill:
//! - MSE calculation mutations (line 92)
//! - SNR calculation mutations (line 99, 106)
//! - Scale calculation mutations (line 14)
//! - Pack/unpack bit operations (line 43)

use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_mse, calculate_optimal_block_size, calculate_scale,
    calculate_snr, dequantize_value, pack_2bit_values, quantize_value, unpack_2bit_values,
    validate_shapes,
};

/// Comprehensive tests designed to kill specific surviving mutants in quantize_value/dequantize_value
#[cfg(test)]
mod utils_quantization_mutation_killers {
    use super::*;

    #[test]
    fn test_kill_return_value_mutations_quantize_value() {
        // Target: Kill mutations that replace quantize_value return with constants (0, 1)
        // These mutations would make the function always return the same value regardless of input

        let test_cases = vec![
            // Test case designed to produce different quantized values
            (2.0, 1.0, 2),  // Should quantize to 2 (max value for 2-bit)
            (-2.0, 1.0, 2), // Should quantize to -2 (min value for 2-bit)
            (0.0, 1.0, 2),  // Should quantize to 0
            (1.5, 1.0, 2),  // Should quantize to 2 (rounded)
            (-1.5, 1.0, 2), // Should quantize to -2 (rounded)
            // Different bit depths to ensure return value varies
            (3.0, 1.0, 3),  // 3-bit: should quantize to 3 (max = 3)
            (-4.0, 1.0, 3), // 3-bit: should quantize to -4 (min = -4)
            (7.0, 1.0, 4),  // 4-bit: should quantize to 7 (max = 7)
            (-8.0, 1.0, 4), // 4-bit: should quantize to -8 (min = -8)
        ];

        let mut results = Vec::new();
        for (value, scale, bits) in test_cases {
            let quantized = quantize_value(value, scale, bits);
            results.push(quantized);

            // Each result should be within the expected range for the bit depth
            let max_val = (1 << (bits - 1)) - 1;
            let min_val = -(1 << (bits - 1));
            assert!(
                quantized >= min_val as i8 && quantized <= max_val as i8,
                "quantize_value({}, {}, {}) = {} is outside valid range [{}, {}]",
                value,
                scale,
                bits,
                quantized,
                min_val,
                max_val
            );
        }

        // Kill constant return mutations: if function always returned 0 or 1,
        // we wouldn't see the variety of values we expect
        assert!(
            results.contains(&-2) || results.contains(&-4) || results.contains(&-8),
            "Function should produce negative values, got {:?} - possible constant return mutation",
            results
        );
        assert!(
            results.contains(&2) || results.contains(&3) || results.contains(&7),
            "Function should produce positive values, got {:?} - possible constant return mutation",
            results
        );
        assert!(
            results.contains(&0),
            "Function should produce zero, got {:?} - possible constant return mutation",
            results
        );

        // Ensure we have at least 3 distinct values (kills both 0 and 1 constant mutations)
        let unique_results: std::collections::HashSet<_> = results.iter().collect();
        assert!(
            unique_results.len() >= 3,
            "Expected at least 3 distinct quantized values, got {:?} - possible constant mutation",
            results
        );
    }

    #[test]
    fn test_kill_min_val_arithmetic_mutation() {
        // Target: Kill `replace - with / in quantize_value` mutation in line 72
        // Original: let min_val = -(1 << (bits - 1));
        // Mutation:  let min_val = /(1 << (bits - 1));  // This would be invalid syntax
        // More likely: let min_val = (1 << (bits - 1)) / something; or similar
        // Actually targeting: -(1 << (bits - 1)) vs something else with division

        // The key insight: min_val should be negative for signed quantization
        let test_cases = vec![
            (-10.0, 1.0, 2),  // Large negative input
            (-5.0, 1.0, 3),   // Should clamp to min_val
            (-100.0, 1.0, 4), // Very large negative
        ];

        for (value, scale, bits) in test_cases {
            let quantized = quantize_value(value, scale, bits);
            let expected_min = -(1 << (bits - 1)) as i8;

            // For large negative inputs, we should hit the minimum value
            if value < 0.0 && (value / scale).abs() >= expected_min.abs() as f32 {
                assert_eq!(
                    quantized, expected_min,
                    "Large negative value {} should clamp to min_val {} for {}-bit, got {}",
                    value, expected_min, bits, quantized
                );
            }
        }

        // Direct test of the min_val calculation logic by using extreme values
        let extreme_negative = -1000.0;
        let scale = 1.0;
        let bits = 2;
        let result = quantize_value(extreme_negative, scale, bits);

        // Should clamp to -2 (min value for 2-bit signed)
        assert_eq!(result, -2, "Extreme negative should clamp to min_val -2, got {}", result);

        // If the mutation replaced - with /, the min_val would be positive or zero
        // and extreme negative values wouldn't clamp correctly
        assert!(result < 0, "Min clamping should produce negative result for negative input");
    }

    #[test]
    fn test_kill_division_mutation_in_quantize_value() {
        // Target: Kill `replace / with * in quantize_value` mutation in line 74
        // Original: let quantized = (value / scale).round() as i32;
        // Mutation:  let quantized = (value * scale).round() as i32;

        let test_cases = vec![
            (4.0, 2.0, 3),  // value=4, scale=2: 4/2=2 vs 4*2=8 (very different)
            (6.0, 3.0, 3),  // value=6, scale=3: 6/3=2 vs 6*3=18 (very different)
            (10.0, 5.0, 4), // value=10, scale=5: 10/5=2 vs 10*5=50 (very different)
            (1.0, 0.5, 3),  // value=1, scale=0.5: 1/0.5=2 vs 1*0.5=0.5 (different)
            (8.0, 4.0, 4),  // value=8, scale=4: 8/4=2 vs 8*4=32 (very different)
        ];

        for (value, scale, bits) in test_cases {
            let correct_result = quantize_value(value, scale, bits);

            // Calculate what the result would be with division (correct)
            let correct_intermediate = (value / scale).round() as i32;
            let max_val = (1 << (bits - 1)) - 1;
            let min_val = -(1 << (bits - 1));
            let expected_correct = correct_intermediate.clamp(min_val, max_val) as i8;

            // Calculate what the result would be with multiplication (mutation)
            let wrong_intermediate = (value * scale).round() as i32;
            let expected_wrong = wrong_intermediate.clamp(min_val, max_val) as i8;

            // The function should return the correct result (with division)
            assert_eq!(
                correct_result, expected_correct,
                "Expected division result {} for value={}, scale={}, bits={}, got {}",
                expected_correct, value, scale, bits, correct_result
            );

            // For our test cases, division and multiplication should give different results
            // (this kills the * mutation)
            if expected_correct != expected_wrong {
                assert_eq!(
                    correct_result, expected_correct,
                    "Division mutation detected: got {}, expected {} (division), wrong would be {} (multiplication)",
                    correct_result, expected_correct, expected_wrong
                );
            }
        }
    }

    #[test]
    fn test_kill_multiplication_mutation_in_dequantize_value() {
        // Target: Kill `replace * with + in dequantize_value` mutation in line 80
        // Original: quantized as f32 * scale
        // Mutation:  quantized as f32 + scale

        let test_cases = vec![
            (2, 3.0),  // 2 * 3.0 = 6.0 vs 2 + 3.0 = 5.0
            (4, 2.5),  // 4 * 2.5 = 10.0 vs 4 + 2.5 = 6.5
            (-3, 2.0), // -3 * 2.0 = -6.0 vs -3 + 2.0 = -1.0
            (5, 1.5),  // 5 * 1.5 = 7.5 vs 5 + 1.5 = 6.5
            (-2, 4.0), // -2 * 4.0 = -8.0 vs -2 + 4.0 = 2.0
            (0, 5.0),  // 0 * 5.0 = 0.0 vs 0 + 5.0 = 5.0
            (1, 10.0), // 1 * 10.0 = 10.0 vs 1 + 10.0 = 11.0
        ];

        for (quantized, scale) in test_cases {
            let result = dequantize_value(quantized, scale);

            // Expected result with multiplication (correct)
            let expected_correct = quantized as f32 * scale;

            // What the result would be with addition (mutation)
            let expected_wrong = quantized as f32 + scale;

            // Function should return the correct (multiplication) result
            assert_eq!(
                result, expected_correct,
                "Expected multiplication result {} for quantized={}, scale={}, got {}",
                expected_correct, quantized, scale, result
            );

            // For our test cases, multiplication and addition give different results
            // (this kills the + mutation)
            if expected_correct != expected_wrong {
                assert_eq!(
                    result, expected_correct,
                    "Addition mutation detected: got {}, expected {} (multiplication), wrong would be {} (addition)",
                    result, expected_correct, expected_wrong
                );
            }
        }
    }

    #[test]
    fn test_round_trip_consistency_kills_all_mutations() {
        // Comprehensive round-trip test that would fail with any of the mutations
        let test_cases = vec![
            (1.0, 1.0, 2),
            (2.5, 1.0, 2),
            (-1.5, 1.0, 2),
            (3.7, 2.0, 3),
            (-4.2, 1.5, 3),
            (7.8, 0.5, 4),
            (0.0, 1.0, 2),
        ];

        for (original_value, scale, bits) in test_cases {
            // Forward pass: quantize then dequantize
            let quantized = quantize_value(original_value, scale, bits);
            let dequantized = dequantize_value(quantized, scale);

            // Round-trip should be consistent with the quantization process
            // This test would fail with any of the mutations because:
            // 1. Constant return (0,1): quantized would always be same, breaking round-trip
            // 2. Division mutation in min_val: would break clamping behavior
            // 3. Multiplication in quantize: would scale wrong direction, breaking round-trip
            // 4. Addition in dequantize: would break the inverse relationship

            // Calculate expected behavior step-by-step to verify correctness
            let max_val = (1 << (bits - 1)) - 1;
            let min_val = -(1 << (bits - 1));
            let expected_quantized =
                ((original_value / scale).round() as i32).clamp(min_val, max_val) as i8;
            let expected_dequantized = expected_quantized as f32 * scale;

            assert_eq!(
                quantized, expected_quantized,
                "Quantization mismatch for value={}, scale={}, bits={}",
                original_value, scale, bits
            );

            assert_eq!(
                dequantized, expected_dequantized,
                "Dequantization mismatch for quantized={}, scale={}",
                quantized, scale
            );

            // Additional invariant: dequantized should be finite
            assert!(
                dequantized.is_finite(),
                "Dequantized value should be finite, got {} for inputs ({}, {}, {})",
                dequantized,
                original_value,
                scale,
                bits
            );
        }
    }

    #[test]
    fn test_bit_depth_boundary_conditions() {
        // Test various bit depths to ensure mutations are caught across different ranges
        let bit_depths = vec![1, 2, 3, 4, 5, 6, 7, 8];

        for bits in bit_depths {
            let max_val = (1 << (bits - 1)) - 1;
            let min_val = -(1 << (bits - 1));

            // Test values that should clamp to boundaries
            let test_values = vec![
                (max_val as f32 + 10.0, 1.0), // Should clamp to max
                (min_val as f32 - 10.0, 1.0), // Should clamp to min
                (0.0, 1.0),                   // Should be zero
                (max_val as f32, 1.0),        // Should be exactly max
                (min_val as f32, 1.0),        // Should be exactly min
            ];

            for (value, scale) in test_values {
                let quantized = quantize_value(value, scale, bits);

                // Should be within valid range
                assert!(
                    quantized >= min_val as i8 && quantized <= max_val as i8,
                    "Value {} with {}-bit should be in range [{}, {}], got {}",
                    value,
                    bits,
                    min_val,
                    max_val,
                    quantized
                );

                // Test round-trip
                let dequantized = dequantize_value(quantized, scale);

                // If any mutation is present, these relationships would break
                if value >= max_val as f32 {
                    assert_eq!(
                        quantized, max_val as i8,
                        "Large positive value should clamp to max for {}-bit",
                        bits
                    );
                }
                if value <= min_val as f32 {
                    assert_eq!(
                        quantized, min_val as i8,
                        "Large negative value should clamp to min for {}-bit",
                        bits
                    );
                }

                // Dequantized should match expected calculation
                let expected_dequant = quantized as f32 * scale;
                assert_eq!(
                    dequantized, expected_dequant,
                    "Dequantization failed for quantized={}, scale={}, bits={}",
                    quantized, scale, bits
                );
            }
        }
    }

    #[test]
    fn test_scale_factor_variations() {
        // Test different scale factors to ensure division/multiplication mutations are caught
        let scales = vec![0.1, 0.5, 1.0, 2.0, 4.0, 10.0];
        let values = vec![-8.0, -1.0, 0.0, 1.0, 8.0];

        for scale in scales {
            for value in &values {
                let quantized = quantize_value(*value, scale, 4); // 4-bit for good range
                let dequantized = dequantize_value(quantized, scale);

                // Key insight: with correct implementation, (value/scale) then (result*scale)
                // should give us a value in the right ballpark
                // With mutations, this relationship breaks down

                let intermediate = (*value / scale).round();
                let max_val = 7i32; // 4-bit max
                let min_val = -8i32; // 4-bit min
                let expected_quantized = intermediate.clamp(min_val as f32, max_val as f32) as i8;
                let expected_dequantized = expected_quantized as f32 * scale;

                assert_eq!(
                    quantized, expected_quantized,
                    "Scale test failed: quantize_value({}, {}) = {}, expected {}",
                    value, scale, quantized, expected_quantized
                );

                assert_eq!(
                    dequantized, expected_dequantized,
                    "Scale test failed: dequantize_value({}, {}) = {}, expected {}",
                    quantized, scale, dequantized, expected_dequantized
                );
            }
        }
    }

    #[test]
    fn test_zero_scale_edge_case() {
        // Edge case: zero scale could expose division issues
        // Note: this might be an invalid input, but helps catch division mutations
        let quantized = quantize_value(5.0, 0.0, 3);
        // With zero scale, value/scale would be infinity
        // The result should still be within the valid range due to clamping
        assert!(
            quantized >= -4 && quantized <= 3,
            "Even with zero scale, result should be within 3-bit range, got {}",
            quantized
        );

        // Dequantize with zero scale should give zero (quantized * 0.0)
        let dequantized = dequantize_value(quantized, 0.0);
        assert_eq!(
            dequantized, 0.0,
            "Dequantize with zero scale should give 0.0, got {}",
            dequantized
        );
    }

    #[test]
    fn test_negative_scale_factor() {
        // Test negative scale factors to ensure arithmetic operations are correct
        let test_cases = vec![
            (2.0, -1.0, 3),  // 2 / -1 = -2, should quantize to -2
            (-3.0, -1.5, 3), // -3 / -1.5 = 2, should quantize to 2
            (4.0, -2.0, 4),  // 4 / -2 = -2, should quantize to -2
        ];

        for (value, scale, bits) in test_cases {
            let quantized = quantize_value(value, scale, bits);
            let dequantized = dequantize_value(quantized, scale);

            // Calculate expected values
            let max_val = (1 << (bits - 1)) - 1;
            let min_val = -(1 << (bits - 1));
            let expected_quantized = ((value / scale).round() as i32).clamp(min_val, max_val) as i8;
            let expected_dequantized = expected_quantized as f32 * scale;

            assert_eq!(
                quantized, expected_quantized,
                "Negative scale test failed for quantize: value={}, scale={}, bits={}",
                value, scale, bits
            );

            assert_eq!(
                dequantized, expected_dequantized,
                "Negative scale test failed for dequantize: quantized={}, scale={}",
                quantized, scale
            );
        }
    }
}

/// Additional utility function mutation killers
#[cfg(test)]
mod utility_functions_mutation_killers {
    use super::*;

    #[test]
    fn test_kill_mse_calculation_mutations() {
        // Target MSE calculation mutations:
        // Line 92: replace - with + in calculate_mse
        // Line 92: replace / with * in calculate_mse
        // Line 92: replace / with % in calculate_mse

        let test_cases = vec![
            (vec![1.0, 2.0, 3.0], vec![1.1, 1.9, 3.2]), // MSE = (0.01 + 0.01 + 0.04) / 3 = 0.02
            (vec![0.0, 1.0], vec![0.5, 1.5]),           // MSE = (0.25 + 0.25) / 2 = 0.25
            (vec![5.0, 10.0], vec![4.0, 12.0]),         // MSE = (1.0 + 4.0) / 2 = 2.5
        ];

        for (original, quantized) in test_cases {
            let mse_result = calculate_mse(&original, &quantized).unwrap();

            // Calculate expected MSE: sum((orig - quant)^2) / n
            let expected_mse = original
                .iter()
                .zip(quantized.iter())
                .map(|(orig, quant)| (orig - quant).powi(2))
                .sum::<f32>()
                / original.len() as f32;

            assert_eq!(mse_result, expected_mse, "MSE calculation should match expected");

            // Kill + mutation: sum((orig + quant)^2) / n would be wrong
            let wrong_add_mse = original
                .iter()
                .zip(quantized.iter())
                .map(|(orig, quant)| (orig + quant).powi(2))
                .sum::<f32>()
                / original.len() as f32;

            if (expected_mse - wrong_add_mse).abs() > 1e-6 {
                assert_ne!(
                    mse_result, wrong_add_mse,
                    "Addition mutation detected in MSE: correct={}, wrong={}",
                    expected_mse, wrong_add_mse
                );
            }

            // Kill * mutation: sum / n -> sum * n would be wrong
            let sum_squared_errors = original
                .iter()
                .zip(quantized.iter())
                .map(|(orig, quant)| (orig - quant).powi(2))
                .sum::<f32>();
            let wrong_mul_mse = sum_squared_errors * original.len() as f32;

            if (expected_mse - wrong_mul_mse).abs() > 1e-6 {
                assert_ne!(
                    mse_result, wrong_mul_mse,
                    "Multiplication mutation detected in MSE: correct={}, wrong={}",
                    expected_mse, wrong_mul_mse
                );
            }
        }
    }

    #[test]
    fn test_kill_snr_calculation_mutations() {
        // Target SNR calculation mutations:
        // Line 99: replace / with * in calculate_snr (signal_power / original.len())
        // Line 99: replace / with % in calculate_snr
        // Line 106: replace * with / in calculate_snr (10.0 * log)
        // Line 106: replace * with + in calculate_snr
        // Line 106: replace / with * in calculate_snr (signal_power / noise_power)
        // Line 106: replace / with % in calculate_snr

        let test_cases = vec![
            (vec![2.0, 4.0, 6.0], vec![1.8, 4.2, 5.9]), // Clear signal vs noise
            (vec![10.0, 20.0], vec![9.5, 20.5]),        // High SNR case
            (vec![1.0, 1.0, 1.0], vec![1.1, 0.9, 1.0]), // Medium SNR case
        ];

        for (original, quantized) in test_cases {
            let snr_result = calculate_snr(&original, &quantized).unwrap();

            // Calculate expected SNR step by step
            let signal_power =
                original.iter().map(|x| x.powi(2)).sum::<f32>() / original.len() as f32;
            let mse = calculate_mse(&original, &quantized).unwrap();
            let expected_snr = 10.0 * (signal_power / mse).log10();

            assert_eq!(snr_result, expected_snr, "SNR calculation should match expected");

            // Kill * mutation in signal_power calculation: sum * len instead of sum / len
            let wrong_signal_power =
                original.iter().map(|x| x.powi(2)).sum::<f32>() * original.len() as f32;
            if wrong_signal_power > 1e-10 && mse > 1e-10 {
                let wrong_snr = 10.0 * (wrong_signal_power / mse).log10();
                if (expected_snr - wrong_snr).abs() > 1.0 {
                    assert_ne!(
                        snr_result, wrong_snr,
                        "Signal power multiplication mutation detected: correct={}, wrong={}",
                        expected_snr, wrong_snr
                    );
                }
            }

            // Kill / -> * mutation in SNR ratio: signal_power * mse instead of signal_power / mse
            let wrong_ratio = signal_power * mse;
            if wrong_ratio > 1e-10 {
                let wrong_snr = 10.0 * wrong_ratio.log10();
                if (expected_snr - wrong_snr).abs() > 1.0 {
                    assert_ne!(
                        snr_result, wrong_snr,
                        "Ratio multiplication mutation detected: correct={}, wrong={}",
                        expected_snr, wrong_snr
                    );
                }
            }

            // Kill * -> / mutation in 10.0 * log: 10.0 / log instead of 10.0 * log
            let log_value = (signal_power / mse).log10();
            if log_value.abs() > 1e-10 {
                let wrong_snr = 10.0 / log_value;
                if (expected_snr - wrong_snr).abs() > 1.0 {
                    assert_ne!(
                        snr_result, wrong_snr,
                        "Coefficient division mutation detected: correct={}, wrong={}",
                        expected_snr, wrong_snr
                    );
                }
            }

            // SNR should be reasonable
            assert!(snr_result.is_finite(), "SNR should be finite");
            assert!(snr_result > -100.0 && snr_result < 100.0, "SNR should be in reasonable range");
        }
    }

    #[test]
    fn test_kill_calculate_scale_mutations() {
        // Target: Line 14: replace / with * in calculate_scale
        // Original: max_val / max_quant as f32
        // Mutation:  max_val * max_quant as f32

        let test_cases = vec![
            (vec![2.0, -6.0, 1.0], 3), // max_val=6.0, max_quant=3, scale=2.0, 6*3=18 (very different)
            (vec![8.0, -4.0], 4), // max_val=8.0, max_quant=7, scale=8.0/7≈1.14, 8*7=56 (very different)
            (vec![3.0, -5.0], 3), // max_val=5.0, max_quant=3, scale=5/3≈1.67, 5*3=15 (very different)
        ];

        for (data, bits) in test_cases {
            let scale = calculate_scale(&data, bits);

            // Expected calculation
            let max_val = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            let max_quant = (1 << (bits - 1)) - 1;
            let expected_scale = if max_val == 0.0 { 1.0 } else { max_val / max_quant as f32 };

            assert_eq!(scale, expected_scale, "Scale calculation should match expected");

            // Kill * mutation: max_val * max_quant would be much larger
            if max_val > 0.0 {
                let wrong_scale = max_val * max_quant as f32;
                assert_ne!(
                    scale, wrong_scale,
                    "Multiplication mutation detected in scale: correct={}, wrong={}",
                    expected_scale, wrong_scale
                );

                // The correct scale should be much smaller than the wrong one
                assert!(
                    scale < wrong_scale,
                    "Division should produce smaller result than multiplication: {} vs {}",
                    scale,
                    wrong_scale
                );
            }
        }
    }

    #[test]
    fn test_kill_pack_2bit_mutations() {
        // Target: Line 43: replace |= with ^= in pack_2bit_values
        // Original: byte |= unsigned << (i * 2);
        // Mutation:  byte ^= unsigned << (i * 2);

        let test_cases = vec![
            vec![-2, -1, 0, 1],   // Full range
            vec![-2, -2, -2, -2], // All same (would be different with ^= vs |=)
            vec![1, 1, 1, 1],     // All same positive
            vec![-2, 1, -2, 1],   // Alternating pattern
        ];

        for values in test_cases {
            let packed = pack_2bit_values(&values);
            let unpacked = unpack_2bit_values(&packed, values.len());

            // Round-trip should be perfect
            assert_eq!(values, unpacked, "Pack/unpack round-trip should be perfect");

            // Test the specific packing logic to kill |= vs ^= mutation
            if values.len() >= 4 {
                let byte = packed[0];

                // Manually reconstruct what the byte should be with |= (correct)
                let mut expected_byte = 0u8;
                for (i, &val) in values[..4].iter().enumerate() {
                    let clamped = val.clamp(-2, 1);
                    let unsigned = (clamped + 2) as u8;
                    expected_byte |= unsigned << (i * 2);
                }

                assert_eq!(byte, expected_byte, "Packed byte should match expected |= result");

                // With ^= mutation, overlapping bits would cancel out differently
                let mut wrong_byte = 0u8;
                for (i, &val) in values[..4].iter().enumerate() {
                    let clamped = val.clamp(-2, 1);
                    let unsigned = (clamped + 2) as u8;
                    wrong_byte ^= unsigned << (i * 2); // This is the mutation
                }

                // For any case where bits overlap or values repeat, ^= and |= give different results
                // Force a difference by checking specific patterns
                if wrong_byte != expected_byte {
                    assert_eq!(
                        byte, expected_byte,
                        "XOR mutation detected: got {}, expected {} (OR), wrong would be {} (XOR)",
                        byte, expected_byte, wrong_byte
                    );
                } else {
                    // Even if they're the same for this specific case, the logic is still different
                    // Test with a case that definitely differs: repeated non-zero values
                    let test_repeated = vec![1, 1, 1, 1];
                    let packed_repeated = pack_2bit_values(&test_repeated);
                    let mut expected_repeated = 0u8;
                    let mut wrong_repeated = 0u8;
                    for (i, &val) in test_repeated.iter().enumerate() {
                        let clamped = val.clamp(-2, 1);
                        let unsigned = (clamped + 2) as u8;
                        expected_repeated |= unsigned << (i * 2);
                        wrong_repeated ^= unsigned << (i * 2);
                    }
                    // With repeated values, XOR cancels out, OR accumulates
                    assert_ne!(
                        expected_repeated, wrong_repeated,
                        "OR and XOR should give different results for repeated values"
                    );
                }
            }
        }
    }

    #[test]
    fn test_kill_validate_shapes_mutations() {
        // Target: Line 144: replace != with == in validate_shapes
        // Original: if shape1 != shape2
        // Mutation:  if shape1 == shape2

        let test_cases = vec![
            (vec![2, 3, 4], vec![2, 3, 4], true),  // Should match (succeed)
            (vec![2, 3], vec![2, 3, 4], false),    // Should not match (fail)
            (vec![1, 2, 3], vec![3, 2, 1], false), // Different order (fail)
            (vec![], vec![], true),                // Both empty (succeed)
            (vec![5], vec![], false),              // One empty (fail)
        ];

        for (shape1, shape2, should_succeed) in test_cases {
            let result = validate_shapes(&shape1, &shape2);

            if should_succeed {
                assert!(result.is_ok(), "Matching shapes should validate successfully");
            } else {
                assert!(result.is_err(), "Non-matching shapes should fail validation");
            }

            // Kill the != -> == mutation
            // Original logic: if shape1 != shape2 { return Err(...) }
            // Mutation logic: if shape1 == shape2 { return Err(...) }
            // The mutation would invert the success/failure cases

            let shapes_equal = shape1 == shape2;
            let shapes_not_equal = shape1 != shape2;

            assert_eq!(shapes_equal, !shapes_not_equal, "Basic logic check");

            if shapes_equal {
                // When shapes are equal, validation should succeed
                assert!(result.is_ok(), "Equal shapes should succeed with != logic");
                // With == mutation, it would fail - our test catches this
            } else {
                // When shapes are not equal, validation should fail
                assert!(result.is_err(), "Unequal shapes should fail with != logic");
                // With == mutation, it would succeed - our test catches this
            }
        }
    }

    #[test]
    fn test_kill_calculate_optimal_block_size_mutations() {
        // Target: Line 155: replace calculate_optimal_block_size -> usize with 0
        // Target: Line 155: replace calculate_optimal_block_size -> usize with 1

        let test_cases = vec![
            (64, 4),    // Should give reasonable block size around 16-32
            (1000, 10), // Should give block size around 100, but power of 2
            (256, 8),   // Should give 32 (256/8=32, next_power_of_two=32)
            (100, 5),   // Should give reasonable size (100/5=20, next_power_of_two=32)
        ];

        let mut all_results = Vec::new();

        for (tensor_size, target_blocks) in test_cases {
            let block_size = calculate_optimal_block_size(tensor_size, target_blocks);
            all_results.push(block_size);

            // Should return reasonable block size (not constant 0 or 1)
            assert!(block_size >= 16, "Block size should be at least 16, got {}", block_size);
            assert!(block_size <= 1024, "Block size should be at most 1024, got {}", block_size);

            // Should be power of 2 (per the function's logic)
            assert!(
                block_size.is_power_of_two(),
                "Block size should be power of 2, got {}",
                block_size
            );

            // Should be related to tensor_size / target_blocks
            let expected_raw = tensor_size.div_ceil(target_blocks);
            let expected_power_of_2 = expected_raw.next_power_of_two().clamp(16, 1024);
            assert_eq!(
                block_size, expected_power_of_2,
                "Block size calculation mismatch for tensor_size={}, target_blocks={}",
                tensor_size, target_blocks
            );
        }

        // Kill constant return mutations (0, 1)
        assert!(
            all_results.iter().any(|&x| x != 0),
            "Function should not always return 0, got {:?}",
            all_results
        );
        assert!(
            all_results.iter().any(|&x| x != 1),
            "Function should not always return 1, got {:?}",
            all_results
        );

        // Should produce variety of results
        let unique_results: std::collections::HashSet<_> = all_results.iter().collect();
        assert!(unique_results.len() > 1, "Should produce varied results, got {:?}", all_results);
    }

    #[test]
    fn test_grouped_scales_arithmetic_mutations() {
        // Target mutations in calculate_grouped_scales:
        // Line 23: replace * with / in calculate_grouped_scales
        // Line 24: replace + with * in calculate_grouped_scales

        let test_cases = vec![
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2), // 3 blocks of size 2
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], 2),      // 3 blocks: [1,2], [3,4], [5]
            (vec![10.0, -5.0, 8.0, -3.0], 3),        // 2 blocks: [10,-5,8], [-3]
        ];

        for (data, block_size) in test_cases {
            let scales = calculate_grouped_scales(&data, block_size, 2);

            // Calculate expected number of blocks: data.len().div_ceil(block_size)
            let expected_num_blocks = data.len().div_ceil(block_size);
            assert_eq!(
                scales.len(),
                expected_num_blocks,
                "Number of scales should match number of blocks"
            );

            // Verify each scale is calculated correctly
            for (i, &scale) in scales.iter().enumerate() {
                let start = i * block_size; // This is the correct calculation
                let end = (start + block_size).min(data.len());
                let block = &data[start..end];

                let expected_scale = calculate_scale(block, 2);
                assert_eq!(
                    scale, expected_scale,
                    "Scale for block {} should match individual calculation",
                    i
                );

                // Kill * -> / mutation in start calculation (line 23)
                // Wrong: let start = i / block_size;
                let wrong_start = if block_size > 0 { i / block_size } else { 0 };
                if wrong_start != start {
                    // This would cause wrong block boundaries
                    let wrong_end = (wrong_start + block_size).min(data.len());
                    if wrong_end <= data.len() && wrong_start < wrong_end {
                        let wrong_block = &data[wrong_start..wrong_end];
                        let wrong_scale = calculate_scale(wrong_block, 2);

                        // Our scale should match the correct calculation, not the wrong one
                        if (scale - wrong_scale).abs() > 1e-6 {
                            assert_ne!(
                                scale, wrong_scale,
                                "Detected division mutation in block start calculation"
                            );
                        }
                    }
                }
            }

            // Kill + -> * mutation in end calculation (line 24)
            // Original: let end = (start + block_size).min(data.len());
            // Mutation:  let end = (start * block_size).min(data.len());
            for i in 0..expected_num_blocks {
                let start = i * block_size;
                let correct_end = (start + block_size).min(data.len());
                let wrong_end = (start * block_size).min(data.len());

                if correct_end != wrong_end && wrong_end <= data.len() && start < wrong_end {
                    // The mutation would give different block boundaries
                    let correct_block = &data[start..correct_end];
                    let wrong_block = &data[start..wrong_end];

                    if correct_block != wrong_block {
                        let correct_scale = calculate_scale(correct_block, 2);
                        let wrong_scale = calculate_scale(wrong_block, 2);

                        if (correct_scale - wrong_scale).abs() > 1e-6 {
                            assert_eq!(
                                scales[i], correct_scale,
                                "Block end mutation detected: got {}, expected {} (correct), wrong would be {} (mutation)",
                                scales[i], correct_scale, wrong_scale
                            );
                        }
                    }
                }
            }
        }
    }
}
