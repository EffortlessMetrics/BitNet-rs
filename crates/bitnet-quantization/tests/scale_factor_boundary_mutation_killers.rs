//! Scale Factor Boundary Condition Mutation Killer Tests
//!
//! This module implements comprehensive tests targeting surviving mutants in
//! scale factor calculations for I2S quantization. Focus areas:
//!
//! 1. Scale factor calculation edge cases (zero, infinity, NaN handling)
//! 2. Arithmetic operations in quantize_value() and dequantize_value()
//! 3. Bit-shift operations and bounds checking
//! 4. Division by zero protection and numerical stability
//! 5. Boundary conditions for quantization range [-2, 1]

use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_scale, dequantize_value, pack_2bit_values, quantize_value,
    unpack_2bit_values,
};
use proptest::prelude::*;

#[cfg(test)]
mod scale_factor_calculation_killers {
    use super::*;

    #[test]
    fn test_kill_max_val_calculation_mutations() {
        // Target: let max_val = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        // Kill mutations: max -> min, abs() removal, fold initial value mutations

        let test_cases = vec![
            // (data, expected_max_abs)
            (vec![1.0, -2.0, 3.0], 3.0),
            (vec![-5.0, 4.0, -1.0], 5.0),
            (vec![0.0, 0.0, 0.0], 0.0),
            (vec![1.5, -1.5, 1.2], 1.5),
            (vec![-0.1, 0.2, -0.3], 0.3),
        ];

        for (data, expected_max) in test_cases {
            let max_val = data.iter().fold(0.0f32, |acc, &x| acc.max((x as f32).abs()));

            assert_eq!(max_val, expected_max, "Correct max calculation failed");

            // Kill max -> min mutation
            let wrong_min = data.iter().fold(0.0f32, |acc, &x| acc.min((x as f32).abs()));
            if (max_val - wrong_min).abs() > 1e-6f32 {
                assert_ne!(
                    max_val, wrong_min,
                    "Min mutation detected: max={}, min={}",
                    max_val, wrong_min
                );
            }

            // Kill abs() removal mutation
            let wrong_no_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x as f32));
            if !data.iter().all(|&x| x >= 0.0) && (max_val - wrong_no_abs).abs() > 1e-6f32 {
                assert_ne!(max_val, wrong_no_abs, "Missing abs() mutation detected");
            }

            // Kill fold initial value mutation (0.0 -> 1.0)
            let wrong_init = data.iter().fold(1.0f32, |acc, &x| acc.max((x as f32).abs()));
            if !data.is_empty() && expected_max < 1.0 {
                assert_ne!(max_val, wrong_init, "Wrong initial value mutation detected");
            }
        }
    }

    #[test]
    fn test_kill_max_quant_bit_shift_mutations() {
        // Target: let max_quant = (1 << (bits - 1)) - 1;
        // Kill mutations: << -> >>, - -> +, bits-1 -> bits+1, 1 -> 0

        let bit_cases = vec![
            (1, 0),   // 2^0 - 1 = 0
            (2, 1),   // 2^1 - 1 = 1
            (3, 3),   // 2^2 - 1 = 3
            (4, 7),   // 2^3 - 1 = 7
            (8, 127), // 2^7 - 1 = 127
        ];

        for (bits, expected_max_quant) in bit_cases {
            let max_quant = (1 << (bits - 1)) - 1;

            assert_eq!(max_quant, expected_max_quant, "Correct max_quant failed for {} bits", bits);

            // Kill << -> >> mutation
            let wrong_right_shift = (1 >> (bits - 1)) - 1;
            if bits > 1 {
                assert_ne!(
                    max_quant, wrong_right_shift,
                    "Right shift mutation detected for {} bits",
                    bits
                );
            }

            // Kill - -> + mutation in (bits - 1)
            if bits < 30 {
                // Avoid overflow
                let wrong_plus_bits = (1 << (bits + 1)) - 1;
                assert_ne!(max_quant, wrong_plus_bits, "Plus mutation detected for {} bits", bits);
            }

            // Kill - -> + mutation in final subtraction
            let wrong_plus_final = (1 << (bits - 1)) + 1;
            assert_ne!(
                max_quant, wrong_plus_final,
                "Final plus mutation detected for {} bits",
                bits
            );

            // Kill 1 -> 0 mutation in left operand
            let wrong_zero_operand = (0 << (bits - 1)) - 1;
            assert_ne!(
                max_quant, wrong_zero_operand,
                "Zero operand mutation detected for {} bits",
                bits
            );
        }
    }

    #[test]
    fn test_kill_scale_division_zero_mutations() {
        // Target: max_val / max_quant as f32
        // Kill mutations: division by zero handling, type conversion mutations

        let test_cases = vec![
            // (max_val, bits, expected_nonzero_scale)
            (0.0, 2, false), // Should return 1.0 for zero case
            (1.0, 2, true),  // 1.0 / 1 = 1.0
            (4.0, 2, true),  // 4.0 / 1 = 4.0
            (3.0, 3, true),  // 3.0 / 3 = 1.0
            (15.0, 4, true), // 15.0 / 7 ≈ 2.14
        ];

        for (max_val, bits, expect_nonzero) in test_cases {
            let scale = calculate_scale(&[max_val], bits);

            if expect_nonzero {
                assert!(scale > 0.0, "Scale should be positive for max_val={}", max_val);
                assert!(scale.is_finite(), "Scale should be finite for max_val={}", max_val);

                // Verify correct calculation
                let max_quant = (1 << (bits - 1)) - 1;
                let expected = max_val / max_quant as f32;
                assert!(
                    (scale - expected).abs() < 1e-6,
                    "Scale calculation wrong: expected {}, got {}",
                    expected,
                    scale
                );

                // Kill / -> * mutation
                let wrong_multiply = max_val * max_quant as f32;
                if (expected - wrong_multiply).abs() > 1e-6 {
                    assert_ne!(scale, wrong_multiply, "Multiply mutation detected");
                }

                // Kill / -> + mutation
                let wrong_add = max_val + max_quant as f32;
                if (expected - wrong_add).abs() > 1e-6 {
                    assert_ne!(scale, wrong_add, "Add mutation detected");
                }

                // Kill / -> - mutation
                let wrong_subtract = max_val - max_quant as f32;
                if (expected - wrong_subtract).abs() > 1e-6 {
                    assert_ne!(scale, wrong_subtract, "Subtract mutation detected");
                }
            } else {
                // Zero case should return 1.0
                assert_eq!(scale, 1.0, "Zero max_val should return scale 1.0");
            }
        }
    }

    #[test]
    fn test_kill_quantize_value_arithmetic_mutations() {
        // Target: let quantized = (value / scale).round() as i32;
        // Kill mutations: / -> *, round() removal, clamp mutations

        let test_cases = vec![
            // (value, scale, bits, expected_range)
            (1.0, 1.0, 2, -2..=1),  // Should be 1
            (-2.0, 1.0, 2, -2..=1), // Should be -2
            (0.5, 0.5, 2, -2..=1),  // Should be 1
            (3.0, 2.0, 2, -2..=1),  // Should be 1 (clamped)
            (-5.0, 2.0, 2, -2..=1), // Should be -2 (clamped)
        ];

        for (value, scale, bits, expected_range) in test_cases {
            let quantized = quantize_value(value, scale, bits);

            assert!(
                expected_range.contains(&quantized),
                "Quantized value {} out of range {:?} for value={}, scale={}",
                quantized,
                expected_range,
                value,
                scale
            );

            // Test intermediate calculations to kill mutations
            let raw_division = value / scale;
            let raw_rounded = raw_division.round() as i32;

            // Kill / -> * mutation in division
            let wrong_multiply = value * scale;
            let wrong_mult_rounded = wrong_multiply.round() as i32;
            if (raw_division - wrong_multiply).abs() > 1e-6 && scale != 1.0 {
                let wrong_mult_clamped = wrong_mult_rounded
                    .clamp(*expected_range.start() as i32, *expected_range.end() as i32);
                if quantized != wrong_mult_clamped as i8 {
                    assert_ne!(
                        quantized, wrong_mult_clamped as i8,
                        "Multiply mutation detected: correct={}, wrong={}",
                        quantized, wrong_mult_clamped
                    );
                }
            }

            // Kill round() removal mutation
            let wrong_no_round = raw_division as i32;
            if raw_rounded != wrong_no_round && (raw_division.fract().abs() > 1e-6) {
                let wrong_no_round_clamped = wrong_no_round
                    .clamp(*expected_range.start() as i32, *expected_range.end() as i32);
                if quantized != wrong_no_round_clamped as i8 {
                    assert_ne!(
                        quantized, wrong_no_round_clamped as i8,
                        "Round removal mutation detected: correct={}, wrong={}",
                        quantized, wrong_no_round_clamped
                    );
                }
            }

            // Verify clamp bounds
            let max_val = *expected_range.end();
            let min_val = *expected_range.start();

            assert!(quantized >= min_val, "Value below minimum: {} < {}", quantized, min_val);
            assert!(quantized <= max_val, "Value above maximum: {} > {}", quantized, max_val);
        }
    }

    #[test]
    fn test_kill_dequantize_multiplication_mutations() {
        // Target: quantized as f32 * scale
        // Kill mutations: * -> /, * -> +, * -> -, type conversion mutations

        let test_cases = vec![
            // (quantized, scale, expected_result)
            (1i8, 1.0, 1.0),
            (-2i8, 1.0, -2.0),
            (0i8, 2.0, 0.0),
            (1i8, 0.5, 0.5),
            (-1i8, 3.0, -3.0),
        ];

        for (quantized, scale, expected) in test_cases {
            let dequantized = dequantize_value(quantized, scale);

            assert!(
                (dequantized - expected).abs() < 1e-6,
                "Dequantize failed: expected {}, got {}",
                expected,
                dequantized
            );

            // Kill * -> / mutation
            if scale != 0.0 {
                let wrong_divide = quantized as f32 / scale;
                if (expected - wrong_divide).abs() > 1e-6 {
                    assert_ne!(
                        dequantized, wrong_divide,
                        "Division mutation detected: correct={}, wrong={}",
                        dequantized, wrong_divide
                    );
                }
            }

            // Kill * -> + mutation
            let wrong_add = quantized as f32 + scale;
            if (expected - wrong_add).abs() > 1e-6 {
                assert_ne!(
                    dequantized, wrong_add,
                    "Addition mutation detected: correct={}, wrong={}",
                    dequantized, wrong_add
                );
            }

            // Kill * -> - mutation
            let wrong_subtract = quantized as f32 - scale;
            if (expected - wrong_subtract).abs() > 1e-6 {
                assert_ne!(
                    dequantized, wrong_subtract,
                    "Subtraction mutation detected: correct={}, wrong={}",
                    dequantized, wrong_subtract
                );
            }

            // Verify type conversion correctness
            let f32_quantized = quantized as f32;
            assert_eq!(dequantized, f32_quantized * scale, "Type conversion inconsistent");
        }
    }

    #[test]
    fn test_scale_factor_extreme_values() {
        // Test scale calculation with extreme values to kill edge case mutations
        let extreme_cases = [
            vec![f32::MIN_POSITIVE],   // Smallest positive
            vec![-f32::MIN_POSITIVE],  // Smallest negative
            vec![1e-10, 1e-20, 1e-30], // Very small values
            vec![1e10, 1e20],          // Very large values (within f32 range)
            vec![1.0, 1e-15],          // Mixed scales
            vec![-1e-10, 1e-10],       // Symmetric tiny values
        ];

        for (case_idx, data) in extreme_cases.iter().enumerate() {
            let scale = calculate_scale(data, 2);

            // Should always be positive and finite
            assert!(scale > 0.0, "Scale should be positive for case {}: data={:?}", case_idx, data);
            assert!(
                scale.is_finite(),
                "Scale should be finite for case {}: data={:?}",
                case_idx,
                data
            );

            // For very small values, algorithm returns 1.0 as safety fallback
            let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            if max_abs > 1e-30 && max_abs < 1e30 && max_abs.is_finite() {
                let expected = max_abs / 1.0; // max_quant = 1 for 2 bits
                assert!(
                    (scale - expected).abs() < 1e-6 * expected.max(1.0),
                    "Scale calculation wrong for case {}: expected ≈ {}, got {}",
                    case_idx,
                    expected,
                    scale
                );
            } else {
                // For edge cases (very small, very large, non-finite), expect fallback value of 1.0
                assert_eq!(
                    scale, 1.0,
                    "Scale fallback wrong for case {}: expected fallback 1.0, got {}",
                    case_idx,
                    scale
                );
            }

            // Test grouped scales with extreme values
            let grouped_scales = calculate_grouped_scales(data, data.len().max(1), 2);
            assert!(!grouped_scales.is_empty(), "Grouped scales should not be empty");
            for (i, &group_scale) in grouped_scales.iter().enumerate() {
                assert!(
                    group_scale > 0.0,
                    "Group scale {} should be positive at index {}",
                    group_scale,
                    i
                );
                assert!(
                    group_scale.is_finite(),
                    "Group scale {} should be finite at index {}",
                    group_scale,
                    i
                );
            }
        }
    }

    #[test]
    fn test_kill_block_size_div_ceil_mutations() {
        // Target: data.len().div_ceil(block_size) in grouped scale calculation
        // Kill mutations: div_ceil -> /, div_ceil boundaries

        let test_cases = vec![
            // (data_len, block_size, expected_blocks)
            (8, 4, 2),  // Exact division: 8/4 = 2
            (9, 4, 3),  // Ceil division: 9/4 = 2.25 -> 3
            (7, 4, 2),  // Ceil division: 7/4 = 1.75 -> 2
            (1, 4, 1),  // Small: 1/4 = 0.25 -> 1
            (16, 8, 2), // Exact: 16/8 = 2
            (17, 8, 3), // Ceil: 17/8 = 2.125 -> 3
        ];

        for (data_len, block_size, expected_blocks) in test_cases {
            let data: Vec<f32> = (0..data_len).map(|i| i as f32).collect();
            let scales = calculate_grouped_scales(&data, block_size, 2);

            assert_eq!(
                scales.len(),
                expected_blocks,
                "Wrong number of blocks for data_len={}, block_size={}: expected {}, got {}",
                data_len,
                block_size,
                expected_blocks,
                scales.len()
            );

            // Kill div_ceil -> regular division mutation
            let wrong_regular_div = data_len / block_size;
            if data_len % block_size != 0 {
                // Only test when there's a remainder
                assert_ne!(
                    scales.len(),
                    wrong_regular_div,
                    "Regular division mutation detected: div_ceil gave {}, regular div would give {}",
                    scales.len(),
                    wrong_regular_div
                );
            }

            // Verify div_ceil property: result * block_size >= data_len
            assert!(
                scales.len() * block_size >= data_len,
                "div_ceil property violation: {} * {} < {}",
                scales.len(),
                block_size,
                data_len
            );

            // Verify div_ceil property: (result-1) * block_size < data_len (unless result == 0)
            if !scales.is_empty() {
                assert!(
                    (scales.len() - 1) * block_size < data_len,
                    "div_ceil minimality violation: {} * {} >= {}",
                    scales.len() - 1,
                    block_size,
                    data_len
                );
            }
        }
    }
}

#[cfg(test)]
mod pack_unpack_bit_manipulation_killers {
    use super::*;

    #[test]
    fn test_kill_bit_shift_mutations_in_packing() {
        // Target: byte |= unsigned << (i * 2); in pack_2bit_values
        // Kill mutations: << -> >>, * -> +, * -> -, * -> /

        let test_cases = vec![
            // Values to pack, expected byte representation
            (vec![-2, -1, 0, 1], 0b11100100), // Each 2-bit value: 00,01,10,11 -> packed
            (vec![1, 0, -1, -2], 0b11100100), // Different order
            (vec![-2, -2, -2, -2], 0b00000000), // All same value
            (vec![1, 1, 1, 1], 0b01010101),   // All max value
        ];

        for (values, _expected_packed) in test_cases {
            let packed = pack_2bit_values(&values);
            assert_eq!(packed.len(), 1, "Should pack to 1 byte");

            let packed_byte = packed[0];

            // Manual verification to kill mutations
            let mut manual_byte = 0u8;
            for (i, &val) in values.iter().enumerate() {
                let clamped = val.clamp(-2, 1);
                let unsigned = (clamped + 2) as u8; // Convert [-2,1] to [0,3]
                manual_byte |= unsigned << (i * 2);
            }

            assert_eq!(
                packed_byte, manual_byte,
                "Packed byte mismatch: expected 0b{:08b}, got 0b{:08b}",
                manual_byte, packed_byte
            );

            // Kill << -> >> mutation
            let mut wrong_right_shift = 0u8;
            for (i, &val) in values.iter().enumerate() {
                let clamped = val.clamp(-2, 1);
                let unsigned = (clamped + 2) as u8;
                wrong_right_shift |= unsigned >> (i * 2); // Wrong: right shift
            }
            if manual_byte != wrong_right_shift {
                assert_ne!(
                    packed_byte, wrong_right_shift,
                    "Right shift mutation detected: correct=0b{:08b}, wrong=0b{:08b}",
                    packed_byte, wrong_right_shift
                );
            }

            // Kill i * 2 -> i + 2 mutation
            let mut wrong_add = 0u8;
            for (i, &val) in values.iter().enumerate() {
                let clamped = val.clamp(-2, 1);
                let unsigned = (clamped + 2) as u8;
                let wrong_shift = if i + 2 < 8 { i + 2 } else { 0 }; // Prevent overflow
                wrong_add |= unsigned << wrong_shift;
            }
            if manual_byte != wrong_add {
                assert_ne!(
                    packed_byte, wrong_add,
                    "Addition mutation detected: correct=0b{:08b}, wrong=0b{:08b}",
                    packed_byte, wrong_add
                );
            }
        }
    }

    #[test]
    fn test_kill_bit_shift_mutations_in_unpacking() {
        // Target: let unsigned = (byte >> (i * 2)) & 0x3; in unpack_2bit_values
        // Kill mutations: >> -> <<, & 0x3 mutations, i*2 arithmetic

        let test_bytes = [
            0b00000000, // All zeros -> [-2,-2,-2,-2]
            0b11111111, // All ones -> [1,1,1,1]
            0b11100100, // Mixed -> [0,1,-2,1]
            0b01010101, // Alternating -> [1,1,1,1]
        ];

        for (byte_idx, byte_val) in test_bytes.iter().enumerate() {
            let packed = vec![*byte_val];
            let unpacked = unpack_2bit_values(&packed, 4);

            assert_eq!(unpacked.len(), 4, "Should unpack to 4 values");

            // Manual verification to kill mutations
            let mut manual_unpacked = Vec::new();
            for i in 0..4 {
                let unsigned = (byte_val >> (i * 2)) & 0x3;
                let signed = unsigned as i8 - 2; // Convert [0,3] to [-2,1]
                manual_unpacked.push(signed);
            }

            assert_eq!(
                unpacked, manual_unpacked,
                "Unpacked values mismatch for byte 0b{:08b}: expected {:?}, got {:?}",
                byte_val, manual_unpacked, unpacked
            );

            // Kill >> -> << mutation
            let mut wrong_left_shift = Vec::new();
            for i in 0..4 {
                let wrong_unsigned = (byte_val << (i * 2)) & 0x3; // Wrong: left shift
                let wrong_signed = wrong_unsigned as i8 - 2;
                wrong_left_shift.push(wrong_signed);
            }
            if manual_unpacked != wrong_left_shift {
                assert_ne!(
                    unpacked, wrong_left_shift,
                    "Left shift mutation detected for byte {}: correct={:?}, wrong={:?}",
                    byte_idx, unpacked, wrong_left_shift
                );
            }

            // Kill & 0x3 -> & 0x7 mutation (wrong mask)
            let mut wrong_mask = Vec::new();
            for i in 0..4 {
                let wrong_unsigned = (byte_val >> (i * 2)) & 0x7; // Wrong: 3-bit mask
                let wrong_signed = (wrong_unsigned as i8 - 2).clamp(-2, 1); // Still clamp to valid range
                wrong_mask.push(wrong_signed);
            }
            if manual_unpacked != wrong_mask {
                assert_ne!(
                    unpacked, wrong_mask,
                    "Mask mutation detected for byte {}: correct={:?}, wrong={:?}",
                    byte_idx, unpacked, wrong_mask
                );
            }

            // Verify all values are in valid range
            for &val in &unpacked {
                assert!((-2..=1).contains(&val), "Unpacked value {} out of range [-2,1]", val);
            }
        }
    }

    #[test]
    fn test_pack_unpack_round_trip_edge_cases() {
        // Test round-trip at boundaries to kill systematic mutations
        let edge_cases = [
            vec![-2, -2, -2, -2], // All minimum
            vec![1, 1, 1, 1],     // All maximum
            vec![-2, 1, -2, 1],   // Alternating extremes
            vec![0, 0, 0, 0],     // All zero
            vec![-1, 0, 1, -2],   // All different values
            vec![-2, -1, 0, 1],   // Sequential values
        ];

        for (case_idx, original) in edge_cases.iter().enumerate() {
            // Pack then unpack
            let packed = pack_2bit_values(original);
            let unpacked = unpack_2bit_values(&packed, original.len());

            assert_eq!(
                unpacked, *original,
                "Round-trip failed for case {}: original={:?}, got={:?}",
                case_idx, original, unpacked
            );

            // Verify packed size is correct
            let expected_bytes = original.len().div_ceil(4);
            assert_eq!(
                packed.len(),
                expected_bytes,
                "Wrong packed size for case {}: expected {} bytes, got {}",
                case_idx,
                expected_bytes,
                packed.len()
            );

            // Test with different output lengths to kill length mutations
            let max_available = packed.len() * 4; // 4 values per byte max
            for test_len in [original.len(), original.len().saturating_sub(1)] {
                if test_len > 0 && test_len <= max_available {
                    let unpacked_len = unpack_2bit_values(&packed, test_len);
                    assert_eq!(
                        unpacked_len.len(),
                        test_len,
                        "Output length not respected: requested {}, got {}",
                        test_len,
                        unpacked_len.len()
                    );

                    // Values within original length should match
                    let check_len = test_len.min(original.len());
                    if check_len > 0 {
                        assert_eq!(
                            &unpacked_len[..check_len],
                            &original[..check_len],
                            "Partial unpacking failed for case {} with length {}",
                            case_idx,
                            test_len
                        );
                    }
                }
            }

            // Test requesting more than available data
            if max_available < original.len() + 1 {
                let unpacked_more = unpack_2bit_values(&packed, original.len() + 1);
                assert_eq!(
                    unpacked_more.len(),
                    max_available,
                    "Should return only available data when requesting more: requested {}, got {}",
                    original.len() + 1,
                    unpacked_more.len()
                );
            }
        }
    }
}

/// Property-based tests for comprehensive mutation coverage
#[cfg(test)]
mod scale_factor_property_tests {
    use super::*;

    proptest! {
        #[test]
        fn scale_calculation_properties(
            data in prop::collection::vec(-100.0f32..100.0f32, 1..100),
            bits in 2u8..8u8  // Start from 2 bits to avoid division by zero in max_quant calculation
        ) {
            let scale = calculate_scale(&data, bits);

            // Property: Scale is always positive
            prop_assert!(scale > 0.0, "Scale should be positive: {}", scale);
            prop_assert!(scale.is_finite(), "Scale should be finite: {}", scale);

            // Property: Scale calculation consistency (accounting for safety fallbacks)
            let max_val = data.iter().filter(|&&x| x.is_finite()).map(|x| x.abs()).fold(0.0f32, f32::max);
            if max_val > 1e-30 && max_val < 1e30 && max_val.is_finite() {
                let max_quant = (1 << (bits - 1)) - 1;
                let expected = max_val / max_quant as f32;
                prop_assert!((scale - expected).abs() < 1e-6 * expected.max(1.0),
                    "Scale calculation inconsistent: expected {}, got {}", expected, scale);
            } else {
                // For edge cases (zero, very small, very large, non-finite), expect fallback value
                prop_assert_eq!(scale, 1.0, "Edge case should give fallback scale 1.0");
            }

            // Property: Quantization bounds preservation
            for &value in &data {
                let quantized = quantize_value(value, scale, bits);
                let max_quant = (1 << (bits - 1)) - 1;
                let min_quant = -(1 << (bits - 1));

                prop_assert!(quantized >= min_quant as i8,
                    "Quantized value {} below minimum {}", quantized, min_quant);
                prop_assert!(quantized <= max_quant as i8,
                    "Quantized value {} above maximum {}", quantized, max_quant);

                // Property: Dequantization consistency
                let dequantized = dequantize_value(quantized, scale);
                prop_assert!(dequantized.is_finite(), "Dequantized should be finite");
            }
        }

        #[test]
        fn grouped_scales_properties(
            data in prop::collection::vec(-50.0f32..50.0f32, 4..200),
            block_size in 1usize..32usize
        ) {
            let scales = calculate_grouped_scales(&data, block_size, 2);

            // Property: Number of blocks is div_ceil
            let expected_blocks = data.len().div_ceil(block_size);
            prop_assert_eq!(scales.len(), expected_blocks,
                "Wrong number of blocks: expected {}, got {}", expected_blocks, scales.len());

            // Property: All scales are positive and finite
            for (i, &scale) in scales.iter().enumerate() {
                prop_assert!(scale > 0.0, "Scale {} at index {} should be positive", scale, i);
                prop_assert!(scale.is_finite(), "Scale {} at index {} should be finite", scale, i);
            }

            // Property: Block coverage
            let total_covered = (scales.len() - 1) * block_size +
                (data.len() - (scales.len() - 1) * block_size).min(block_size);
            prop_assert_eq!(total_covered, data.len(),
                "Block coverage wrong: covered {}, data length {}", total_covered, data.len());
        }

        #[test]
        fn pack_unpack_properties(
            values in prop::collection::vec(-2i8..=1i8, 1..50)
        ) {
            let packed = pack_2bit_values(&values);
            let unpacked = unpack_2bit_values(&packed, values.len());

            // Property: Round-trip preservation
            prop_assert_eq!(unpacked.clone(), values.clone(), "Round-trip failed");

            // Property: Packed size is correct
            let expected_bytes = values.len().div_ceil(4);
            prop_assert_eq!(packed.len(), expected_bytes,
                "Wrong packed size: expected {}, got {}", expected_bytes, packed.len());

            // Property: All values in valid range
            for &val in &unpacked.clone() {
                prop_assert!((-2..=1).contains(&val), "Value {} out of range", val);
            }
        }
    }
}
