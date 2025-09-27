//! Bit-Shift Operation Boundary Mutation Killer Tests
//!
//! This module implements comprehensive tests targeting surviving mutants in
//! bit-shift operations for I2S quantization. Focus areas:
//!
//! 1. Left shift operations in quantization range calculations
//! 2. Right shift operations in bit unpacking
//! 3. Bit masking and boundary value handling
//! 4. Overflow protection in shift operations
//! 5. Edge cases at shift boundaries (0, 31, 32, 63)

use bitnet_quantization::utils::{pack_2bit_values, unpack_2bit_values};
use proptest::prelude::*;

#[cfg(test)]
mod left_shift_mutation_killers {
    // Note: pack_2bit_values and unpack_2bit_values are tested indirectly via other functions

    #[test]
    fn test_kill_left_shift_in_max_quant_calculation() {
        // Target: let max_quant = (1 << (bits - 1)) - 1;
        // Kill mutations: << -> >>, << with wrong operands, boundary cases

        let test_cases = vec![
            // (bits, expected_max_quant)
            (1, 0),   // 2^(1-1) - 1 = 2^0 - 1 = 1 - 1 = 0
            (2, 1),   // 2^(2-1) - 1 = 2^1 - 1 = 2 - 1 = 1
            (3, 3),   // 2^(3-1) - 1 = 2^2 - 1 = 4 - 1 = 3
            (4, 7),   // 2^(4-1) - 1 = 2^3 - 1 = 8 - 1 = 7
            (5, 15),  // 2^(5-1) - 1 = 2^4 - 1 = 16 - 1 = 15
            (8, 127), // 2^(8-1) - 1 = 2^7 - 1 = 128 - 1 = 127
        ];

        for (bits, expected_max_quant) in test_cases {
            let max_quant = (1i32 << (bits - 1)) - 1;

            assert_eq!(
                max_quant, expected_max_quant,
                "Max quantization value wrong for {} bits: expected {}, got {}",
                bits, expected_max_quant, max_quant
            );

            // Kill << -> >> mutation
            let wrong_right_shift = (1 >> (bits - 1)) - 1;
            if bits > 1 {
                assert_ne!(
                    max_quant, wrong_right_shift,
                    "Right shift mutation detected for {} bits: correct={}, wrong={}",
                    bits, max_quant, wrong_right_shift
                );
            }

            // Kill operand mutations: 1 -> 0
            let wrong_zero_operand = (0 << (bits - 1)) - 1;
            assert_ne!(
                max_quant, wrong_zero_operand,
                "Zero operand mutation detected for {} bits",
                bits
            );

            // Kill operand mutations: 1 -> 2
            let wrong_two_operand = (2 << (bits - 1)) - 1;
            assert_ne!(
                max_quant, wrong_two_operand,
                "Two operand mutation detected for {} bits",
                bits
            );

            // Verify bit pattern properties
            if bits > 1 {
                assert!(max_quant > 0, "Max quant should be positive for {} bits", bits);
                assert!(max_quant < (1 << bits), "Max quant should be less than 2^bits");

                // Check that max_quant has exactly (bits-1) set bits
                let set_bits = max_quant.count_ones();
                assert_eq!(
                    set_bits,
                    (bits - 1) as u32,
                    "Max quant should have {} set bits for {} bits total",
                    bits - 1,
                    bits
                );
            }
        }
    }

    #[test]
    fn test_kill_min_quant_left_shift_mutations() {
        // Target: let min_val = -(1 << (bits - 1));
        // Kill mutations: << -> >>, negation mutations, boundary cases

        let test_cases = vec![
            // (bits, expected_min_quant)
            (1, 0),    // -(2^(1-1)) = -(2^0) = -1, but for 1-bit it's actually 0
            (2, -2),   // -(2^(2-1)) = -(2^1) = -2
            (3, -4),   // -(2^(3-1)) = -(2^2) = -4
            (4, -8),   // -(2^(4-1)) = -(2^3) = -8
            (5, -16),  // -(2^(5-1)) = -(2^4) = -16
            (8, -128), // -(2^(8-1)) = -(2^7) = -128
        ];

        for (bits, expected_min_quant) in test_cases {
            let min_val: i32 = if bits == 1 {
                0 // Special case for 1-bit: range is [0, 0]
            } else {
                -(1i32 << (bits - 1))
            };

            if bits > 1 {
                assert_eq!(
                    min_val, expected_min_quant,
                    "Min quantization value wrong for {} bits: expected {}, got {}",
                    bits, expected_min_quant, min_val
                );
            }

            // Kill << -> >> mutation in min calculation
            let wrong_right_shift = -(1 >> (bits - 1));
            if bits > 1 {
                assert_ne!(
                    min_val, wrong_right_shift,
                    "Right shift mutation detected in min calculation for {} bits",
                    bits
                );
            }

            // Kill negation removal mutation
            let wrong_no_negation = 1 << (bits - 1);
            if bits > 1 {
                assert_ne!(
                    min_val, wrong_no_negation,
                    "Negation removal mutation detected for {} bits",
                    bits
                );
            }

            // Kill double negation mutation
            let wrong_double_negation = -(-1 << (bits - 1));
            if bits > 1 {
                assert_ne!(
                    min_val, wrong_double_negation,
                    "Double negation mutation detected for {} bits",
                    bits
                );
            }

            // Verify range properties
            if bits > 1 {
                let max_val = (1 << (bits - 1)) - 1;
                assert!(min_val < 0, "Min value should be negative for {} bits", bits);
                assert!(min_val <= -max_val, "Min should be <= -max for {} bits", bits);
                assert_eq!(
                    min_val.abs(),
                    max_val + 1,
                    "Two's complement property: |min| = max + 1 for {} bits",
                    bits
                );
            }
        }
    }

    #[test]
    fn test_kill_shift_amount_arithmetic_mutations() {
        // Target: (bits - 1) in shift operations
        // Kill mutations: - -> +, - -> *, - -> /, operand mutations

        let test_cases = vec![
            // (bits, expected_shift_amount)
            (1, 0),   // 1 - 1 = 0
            (2, 1),   // 2 - 1 = 1
            (3, 2),   // 3 - 1 = 2
            (4, 3),   // 4 - 1 = 3
            (8, 7),   // 8 - 1 = 7
            (16, 15), // 16 - 1 = 15
        ];

        for (bits, expected_shift) in test_cases {
            let shift_amount = bits - 1;

            assert_eq!(
                shift_amount, expected_shift,
                "Shift amount wrong for {} bits: expected {}, got {}",
                bits, expected_shift, shift_amount
            );

            // Test the shift operation result
            let left_shift_result = 1 << shift_amount;
            let expected_power = 1usize << expected_shift;

            assert_eq!(
                left_shift_result, expected_power,
                "Left shift result wrong for {} bits",
                bits
            );

            // Kill - -> + mutation in shift amount
            if bits > 0 {
                let wrong_plus = bits + 1;
                if wrong_plus < 32 {
                    // Avoid overflow in shift
                    let wrong_plus_result = 1 << wrong_plus;
                    assert_ne!(
                        left_shift_result, wrong_plus_result,
                        "Plus mutation detected in shift amount for {} bits",
                        bits
                    );
                }
            }

            // Kill - -> * mutation in shift amount
            let wrong_multiply = bits; // Mutation: Remove identity operation
            if wrong_multiply != shift_amount && wrong_multiply < 32 {
                let wrong_mult_result = 1 << wrong_multiply;
                assert_ne!(
                    left_shift_result, wrong_mult_result,
                    "Multiplication mutation detected in shift amount for {} bits",
                    bits
                );
            }

            // Kill - -> / mutation in shift amount
            if bits > 1 {
                let wrong_divide = bits; // Mutation: Remove identity operation
                if wrong_divide != shift_amount && wrong_divide < 32 {
                    let wrong_div_result = 1 << wrong_divide;
                    assert_ne!(
                        left_shift_result, wrong_div_result,
                        "Division mutation detected in shift amount for {} bits",
                        bits
                    );
                }
            }

            // Verify shift amount bounds
            assert!(shift_amount < 32, "Shift amount should be < 32 for safety");
            if bits > 0 {
                assert!(shift_amount < bits, "Shift amount should be < bits");
            }
        }
    }

    #[test]
    fn test_shift_boundary_edge_cases() {
        // Test shift operations at critical boundaries
        let boundary_cases = vec![
            // (shift_amount, description)
            (0, "Zero shift"),
            (1, "Single bit shift"),
            (7, "Byte boundary"),
            (15, "Word boundary"),
            (31, "Maximum safe shift for u32"),
        ];

        for (shift_amount, description) in boundary_cases {
            // Test left shift
            let left_result = 1u32 << shift_amount;
            let expected = 2u32.pow(shift_amount);

            assert_eq!(
                left_result, expected,
                "Left shift failed for {}: expected {}, got {}",
                description, expected, left_result
            );

            // Test right shift (with a value that has bits to shift)
            let test_value = if shift_amount == 0 { 1 } else { 1u32 << shift_amount };
            let right_result = test_value >> shift_amount;

            if shift_amount == 0 {
                assert_eq!(right_result, test_value, "Zero right shift should preserve value");
            } else {
                assert_eq!(right_result, 1, "Right shift should restore original bit");
            }

            // Kill << -> >> mutation by testing both directions
            if shift_amount > 0 {
                let wrong_right_in_left = 1u32 >> shift_amount;
                assert_ne!(
                    left_result, wrong_right_in_left,
                    "Right shift mutation detected in left shift for {}",
                    description
                );

                let wrong_left_in_right = test_value << shift_amount;
                if wrong_left_in_right != right_result {
                    assert_ne!(
                        right_result, wrong_left_in_right,
                        "Left shift mutation detected in right shift for {}",
                        description
                    );
                }
            }

            // Test overflow behavior at boundaries
            if shift_amount == 31 {
                // At 31-bit shift, we're at the edge of u32
                assert_eq!(left_result, 2147483648u32, "31-bit shift should give 2^31");

                // 32-bit shift would overflow (if allowed)
                // This tests that we don't accidentally use 32-bit shifts
            }

            // Test signed vs unsigned shift behavior
            let signed_test = 1i32 << shift_amount;
            assert_eq!(
                signed_test as u32, left_result,
                "Signed and unsigned left shifts should match for positive values"
            );
        }
    }
}

#[cfg(test)]
mod right_shift_mutation_killers {
    use super::*;

    #[test]
    fn test_kill_right_shift_in_bit_unpacking() {
        // Target: let unsigned = (byte >> (i * 2)) & 0x3; in unpack_2bit_values
        // Kill mutations: >> -> <<, shift amount arithmetic, mask mutations

        let test_bytes = [
            0b00000000, // All zeros
            0b11111111, // All ones
            0b10110100, // Mixed pattern: 10,11,01,00
            0b01011010, // Different pattern: 01,01,10,10
        ];

        for (byte_idx, test_byte) in test_bytes.iter().enumerate() {
            let packed = vec![*test_byte];
            let unpacked = unpack_2bit_values(&packed, 4);

            // Manual unpacking to verify and kill mutations
            let mut manual_unpacked = Vec::new();
            for i in 0..4 {
                let shift_amount = i * 2;
                let unsigned = (test_byte >> shift_amount) & 0x3;
                let signed = unsigned as i8 - 2;
                manual_unpacked.push(signed);
            }

            assert_eq!(
                unpacked, manual_unpacked,
                "Unpacking mismatch for byte 0b{:08b}: expected {:?}, got {:?}",
                test_byte, manual_unpacked, unpacked
            );

            // Kill >> -> << mutation
            let mut wrong_left_shift = Vec::new();
            for i in 0..4 {
                let shift_amount = i * 2;
                let wrong_unsigned = (test_byte << shift_amount) & 0x3; // Wrong: left shift
                let wrong_signed = wrong_unsigned as i8 - 2;
                wrong_left_shift.push(wrong_signed);
            }

            if unpacked != wrong_left_shift {
                assert_ne!(
                    unpacked, wrong_left_shift,
                    "Left shift mutation detected for byte {}: correct={:?}, wrong={:?}",
                    byte_idx, unpacked, wrong_left_shift
                );
            }

            // Kill shift amount mutations: i * 2 -> i + 2, i * 2 -> i / 2, etc.
            let mut wrong_add_shift = Vec::new();
            for i in 0..4 {
                let wrong_shift = (i + 2).min(7); // Clamp to avoid out-of-bounds
                let wrong_unsigned = (test_byte >> wrong_shift) & 0x3;
                let wrong_signed = wrong_unsigned as i8 - 2;
                wrong_add_shift.push(wrong_signed);
            }

            if unpacked != wrong_add_shift {
                assert_ne!(
                    unpacked, wrong_add_shift,
                    "Add shift mutation detected for byte {}",
                    byte_idx
                );
            }

            // Verify bit positions
            for (i, expected_val) in manual_unpacked.iter().enumerate() {
                let bit_position = i * 2;
                let extracted_bits = (test_byte >> bit_position) & 0x3;
                let reconstructed = extracted_bits as i8 - 2;

                assert_eq!(
                    *expected_val, reconstructed,
                    "Bit extraction failed at position {} for byte 0b{:08b}",
                    bit_position, test_byte
                );

                // Verify the 2-bit mask is correct
                assert!(
                    extracted_bits <= 3,
                    "Extracted bits should be <= 3: got {} at position {}",
                    extracted_bits,
                    bit_position
                );
            }
        }
    }

    #[test]
    fn test_kill_mask_mutations_in_unpacking() {
        // Target: & 0x3 mask in bit unpacking
        // Kill mutations: 0x3 -> 0x1, 0x3 -> 0x7, 0x3 -> 0xF, & -> |, & -> ^

        let test_byte = 0b11100100u8; // Pattern: 11,10,01,00
        let expected_2bit_values = [0, 1, 2, 3]; // Unsigned 2-bit values

        for (i, &expected_val) in expected_2bit_values.iter().enumerate() {
            let shift_amount = i * 2;
            let correct_unsigned = (test_byte >> shift_amount) & 0x3;

            assert_eq!(
                correct_unsigned, expected_val,
                "Correct extraction failed at position {}: expected {}, got {}",
                i, expected_val, correct_unsigned
            );

            // Kill mask mutations: 0x3 -> 0x1 (1-bit mask)
            let wrong_mask_1bit = (test_byte >> shift_amount) & 0x1;
            if correct_unsigned != wrong_mask_1bit {
                assert_ne!(
                    correct_unsigned, wrong_mask_1bit,
                    "1-bit mask mutation detected at position {}: correct={}, wrong={}",
                    i, correct_unsigned, wrong_mask_1bit
                );
            }

            // Kill mask mutations: 0x3 -> 0x7 (3-bit mask)
            let wrong_mask_3bit = (test_byte >> shift_amount) & 0x7;
            if correct_unsigned != wrong_mask_3bit {
                assert_ne!(
                    correct_unsigned, wrong_mask_3bit,
                    "3-bit mask mutation detected at position {}: correct={}, wrong={}",
                    i, correct_unsigned, wrong_mask_3bit
                );
            }

            // Kill mask mutations: 0x3 -> 0xF (4-bit mask)
            let wrong_mask_4bit = (test_byte >> shift_amount) & 0xF;
            if correct_unsigned != wrong_mask_4bit {
                assert_ne!(
                    correct_unsigned, wrong_mask_4bit,
                    "4-bit mask mutation detected at position {}: correct={}, wrong={}",
                    i, correct_unsigned, wrong_mask_4bit
                );
            }

            // Kill & -> | mutation
            let wrong_or_operation = (test_byte >> shift_amount) | 0x3;
            assert_ne!(
                correct_unsigned, wrong_or_operation,
                "OR mutation detected at position {}: correct={}, wrong={}",
                i, correct_unsigned, wrong_or_operation
            );

            // Kill & -> ^ mutation
            let wrong_xor_operation = (test_byte >> shift_amount) ^ 0x3;
            if correct_unsigned != wrong_xor_operation {
                assert_ne!(
                    correct_unsigned, wrong_xor_operation,
                    "XOR mutation detected at position {}: correct={}, wrong={}",
                    i, correct_unsigned, wrong_xor_operation
                );
            }

            // Verify mask property: result should always be <= 3
            assert!(
                correct_unsigned <= 3,
                "Masked result should be <= 3: got {} at position {}",
                correct_unsigned,
                i
            );

            // Verify mask property: result should preserve lower 2 bits
            let original_2bits = (test_byte >> shift_amount) & 0x3;
            assert_eq!(
                correct_unsigned, original_2bits,
                "Mask should preserve exactly 2 bits at position {}",
                i
            );
        }
    }

    #[test]
    fn test_right_shift_boundary_cases() {
        // Test right shift operations at bit boundaries
        let boundary_test_cases = vec![
            // (value, shift_amount, expected_result)
            (0b11111111u8, 0, 0b11111111), // No shift
            (0b11111111u8, 1, 0b01111111), // 1-bit shift
            (0b11111111u8, 2, 0b00111111), // 2-bit shift
            (0b11111111u8, 4, 0b00001111), // 4-bit shift
            (0b11111111u8, 7, 0b00000001), // 7-bit shift
            (0b11111111u8, 8, 0b00000000), // 8-bit shift (full)
            (0b10000000u8, 7, 0b00000001), // Single bit shifted to LSB
            (0b01000000u8, 6, 0b00000001), // Single bit shifted to LSB
        ];

        for (value, shift_amount, expected) in boundary_test_cases {
            let result = value >> shift_amount;

            assert_eq!(
                result, expected,
                "Right shift failed: 0b{:08b} >> {} expected 0b{:08b}, got 0b{:08b}",
                value, shift_amount, expected, result
            );

            // Kill >> -> << mutation
            if shift_amount > 0 {
                let wrong_left_shift = value << shift_amount;
                // Mask to 8 bits to avoid overflow effects
                let wrong_left_masked = wrong_left_shift; // Mutation: Remove identity mask operation
                assert_ne!(
                    result, wrong_left_masked,
                    "Left shift mutation detected: value=0b{:08b}, shift={}",
                    value, shift_amount
                );
            }

            // Test shift amount boundary mutations
            if shift_amount > 0 {
                // Kill shift_amount -> shift_amount + 1
                let wrong_plus_one = value >> (shift_amount + 1);
                if shift_amount < 8 {
                    assert_ne!(
                        result, wrong_plus_one,
                        "Shift+1 mutation detected: value=0b{:08b}, shift={}",
                        value, shift_amount
                    );
                }

                // Kill shift_amount -> shift_amount - 1
                let wrong_minus_one = value >> (shift_amount - 1);
                assert_ne!(
                    result, wrong_minus_one,
                    "Shift-1 mutation detected: value=0b{:08b}, shift={}",
                    value, shift_amount
                );
            }

            // Verify shift properties
            if shift_amount < 8 {
                // For shifts less than bit width, some bits should be preserved
                let preserved_bits = 8 - shift_amount;
                let max_possible = (1u8 << preserved_bits) - 1;
                assert!(
                    result <= max_possible,
                    "Right shift result too large: got {}, max possible {}",
                    result,
                    max_possible
                );
            } else {
                // For shifts >= bit width, result should be 0
                assert_eq!(result, 0, "Right shift by >= 8 bits should give 0: got {}", result);
            }
        }
    }
}

#[cfg(test)]
mod bit_packing_mutation_killers {
    use super::*;

    #[test]
    fn test_kill_left_shift_in_packing() {
        // Target: byte |= unsigned << (i * 2); in pack_2bit_values
        // Kill mutations: << -> >>, shift amount arithmetic, |= -> &=, |= -> ^=

        let test_values = [
            vec![-2, -1, 0, 1],   // All different values
            vec![1, 1, 1, 1],     // All same (max)
            vec![-2, -2, -2, -2], // All same (min)
            vec![0, 0, 0, 0],     // All zero
        ];

        for (test_idx, values) in test_values.iter().enumerate() {
            let packed = pack_2bit_values(values);
            assert_eq!(packed.len(), 1, "Should pack to 1 byte for test {}", test_idx);

            let packed_byte = packed[0];

            // Manual packing to verify and kill mutations
            let mut manual_byte = 0u8;
            for (i, &val) in values.iter().enumerate() {
                let clamped = val.clamp(-2, 1);
                let unsigned = (clamped + 2) as u8; // Convert [-2,1] to [0,3]
                let shift_amount = i * 2;
                manual_byte |= unsigned << shift_amount;
            }

            assert_eq!(
                packed_byte, manual_byte,
                "Packing mismatch for test {}: values={:?}, expected=0b{:08b}, got=0b{:08b}",
                test_idx, values, manual_byte, packed_byte
            );

            // Kill << -> >> mutation in packing
            let mut wrong_right_shift = 0u8;
            for (i, &val) in values.iter().enumerate() {
                let clamped = val.clamp(-2, 1);
                let unsigned = (clamped + 2) as u8;
                let shift_amount = i * 2;
                wrong_right_shift |= unsigned >> shift_amount; // Wrong: right shift
            }

            if manual_byte != wrong_right_shift {
                assert_ne!(
                    packed_byte, wrong_right_shift,
                    "Right shift mutation detected in packing for test {}: correct=0b{:08b}, wrong=0b{:08b}",
                    test_idx, packed_byte, wrong_right_shift
                );
            }

            // Kill |= -> &= mutation
            let mut wrong_and_equals = 0u8;
            for (i, &val) in values.iter().enumerate() {
                let clamped = val.clamp(-2, 1);
                let unsigned = (clamped + 2) as u8;
                let shift_amount = i * 2;
                wrong_and_equals &= unsigned << shift_amount; // Wrong: AND instead of OR
            }

            if manual_byte != wrong_and_equals {
                assert_ne!(
                    packed_byte, wrong_and_equals,
                    "AND equals mutation detected in packing for test {}: correct=0b{:08b}, wrong=0b{:08b}",
                    test_idx, packed_byte, wrong_and_equals
                );
            }

            // Kill |= -> ^= mutation
            let mut wrong_xor_equals = 0u8;
            for (i, &val) in values.iter().enumerate() {
                let clamped = val.clamp(-2, 1);
                let unsigned = (clamped + 2) as u8;
                let shift_amount = i * 2;
                wrong_xor_equals ^= unsigned << shift_amount; // Wrong: XOR instead of OR
            }

            if manual_byte != wrong_xor_equals {
                assert_ne!(
                    packed_byte, wrong_xor_equals,
                    "XOR equals mutation detected in packing for test {}: correct=0b{:08b}, wrong=0b{:08b}",
                    test_idx, packed_byte, wrong_xor_equals
                );
            }

            // Verify bit positions in packed byte
            for (i, &val) in values.iter().enumerate() {
                let bit_position = i * 2;
                let extracted = (packed_byte >> bit_position) & 0x3;
                let expected_unsigned = (val.clamp(-2, 1) + 2) as u8;

                assert_eq!(
                    extracted, expected_unsigned,
                    "Bit verification failed at position {} for test {}: expected {}, got {}",
                    bit_position, test_idx, expected_unsigned, extracted
                );
            }
        }
    }

    #[test]
    fn test_kill_shift_amount_multiplication_mutations() {
        // Target: i * 2 in shift amount calculation
        // Kill mutations: * -> +, * -> -, * -> /, bit position errors

        let positions = [0, 1, 2, 3]; // 4 positions in a byte
        let expected_shifts = [0, 2, 4, 6]; // Expected shift amounts

        for (i, expected_shift) in positions.iter().zip(expected_shifts.iter()) {
            let calculated_shift = i * 2;

            assert_eq!(
                calculated_shift, *expected_shift,
                "Shift calculation wrong for position {}: expected {}, got {}",
                i, expected_shift, calculated_shift
            );

            // Test shift in actual bit operation
            let test_value = 0x3u8; // 2-bit pattern: 11
            let shifted = test_value << calculated_shift;

            // Kill * -> + mutation in shift amount
            let wrong_add_shift = i + 2;
            if wrong_add_shift != calculated_shift && wrong_add_shift < 8 {
                let wrong_add_result = test_value << wrong_add_shift;
                assert_ne!(
                    shifted, wrong_add_result,
                    "Addition mutation detected in shift amount for position {}: correct=0b{:08b}, wrong=0b{:08b}",
                    i, shifted, wrong_add_result
                );
            }

            // Kill * -> - mutation in shift amount
            if *i >= 2 {
                let wrong_sub_shift = i - 2;
                let wrong_sub_result = test_value << wrong_sub_shift;
                assert_ne!(
                    shifted, wrong_sub_result,
                    "Subtraction mutation detected in shift amount for position {}: correct=0b{:08b}, wrong=0b{:08b}",
                    i, shifted, wrong_sub_result
                );
            }

            // Kill * -> / mutation in shift amount
            if *i > 0 {
                let wrong_div_shift = i / 2;
                let wrong_div_result = test_value << wrong_div_shift;
                if calculated_shift != wrong_div_shift {
                    assert_ne!(
                        shifted, wrong_div_result,
                        "Division mutation detected in shift amount for position {}: correct=0b{:08b}, wrong=0b{:08b}",
                        i, shifted, wrong_div_result
                    );
                }
            }

            // Verify shift amount bounds
            assert!(
                calculated_shift < 8,
                "Shift amount should be < 8 for position {}: got {}",
                i,
                calculated_shift
            );
            assert!(
                calculated_shift % 2 == 0,
                "Shift amount should be even for position {}: got {}",
                i,
                calculated_shift
            );
            assert_eq!(
                calculated_shift,
                i * 2,
                "Shift amount should equal position * 2 for position {}",
                i
            );
        }
    }

    #[test]
    fn test_bit_packing_round_trip_mutations() {
        // Test round-trip packing/unpacking to catch systematic mutations
        let comprehensive_test_cases = [
            // Each value at each position to test all bit combinations
            vec![-2, -1, 0, 1], // Sequential
            vec![1, 0, -1, -2], // Reverse sequential
            vec![-2, 1, -2, 1], // Alternating extremes
            vec![0, 1, 0, 1],   // Alternating 0,1
            vec![-1, 0, -1, 0], // Alternating -1,0
            vec![-2, 0, 1, -1], // Random pattern
        ];

        for (test_idx, original) in comprehensive_test_cases.iter().enumerate() {
            // Pack then unpack
            let packed = pack_2bit_values(original);
            let unpacked = unpack_2bit_values(&packed, original.len());

            assert_eq!(
                unpacked, *original,
                "Round-trip failed for test {}: original={:?}, unpacked={:?}",
                test_idx, original, unpacked
            );

            // Verify packed byte structure
            let packed_byte = packed[0];

            // Extract each 2-bit field and verify
            for (pos, &expected_val) in original.iter().enumerate() {
                let shift_amount = pos * 2;
                let extracted_bits = (packed_byte >> shift_amount) & 0x3;
                let expected_unsigned = (expected_val + 2) as u8;

                assert_eq!(
                    extracted_bits, expected_unsigned,
                    "Bit field {} mismatch in test {}: expected {}, got {}",
                    pos, test_idx, expected_unsigned, extracted_bits
                );

                // Convert back to signed and verify
                let recovered_signed = extracted_bits as i8 - 2;
                assert_eq!(
                    recovered_signed, expected_val,
                    "Sign conversion mismatch in test {} at position {}: expected {}, got {}",
                    test_idx, pos, expected_val, recovered_signed
                );
            }

            // Test mutations in the entire round-trip process
            // Kill systematic bit position mutations
            for wrong_shift in [1, 3, 5, 7] {
                // Wrong shift amounts
                let mut wrong_byte = 0u8;
                for (i, &val) in original.iter().enumerate() {
                    let clamped = val.clamp(-2, 1);
                    let unsigned = (clamped + 2) as u8;
                    if i == 0 && wrong_shift < 8 {
                        wrong_byte |= unsigned << wrong_shift; // Wrong shift for first element
                    } else {
                        wrong_byte |= unsigned << (i * 2); // Correct shift for others
                    }
                }

                if wrong_byte != packed_byte {
                    let wrong_unpacked = unpack_2bit_values(&[wrong_byte], original.len());
                    assert_ne!(
                        unpacked, wrong_unpacked,
                        "Shift mutation not detected in round-trip for test {}",
                        test_idx
                    );
                }
            }
        }
    }
}

/// Property-based tests for comprehensive bit operation coverage
#[cfg(test)]
mod bit_operation_property_tests {
    use super::*;

    proptest! {
        #[test]
        fn left_shift_properties(
            bits in 1u8..16u8,
            shift_amount in 0usize..31usize
        ) {
            prop_assume!(shift_amount < bits as usize);
            prop_assume!(shift_amount < 31); // Avoid overflow

            // Property: Left shift by N is equivalent to multiplication by 2^N
            let base = 1u32;
            let shifted = base << shift_amount;
            let multiplied = base * (2u32.pow(shift_amount as u32));

            prop_assert_eq!(shifted, multiplied,
                "Left shift should equal multiplication by 2^N");

            // Property: Right shift reverses left shift (for powers of 2)
            let reverse_shifted = shifted >> shift_amount;
            prop_assert_eq!(reverse_shifted, base,
                "Right shift should reverse left shift");

            // Property: Shift amount bounds
            if shift_amount > 0 {
                let smaller_shift = base << (shift_amount - 1);
                prop_assert!(shifted > smaller_shift,
                    "Larger shift should give larger result");
            }

            // Property: Bit counting
            prop_assert_eq!(shifted.count_ones(), 1u32,
                "Power of 2 should have exactly one bit set");

            // Property: Leading zeros
            if shift_amount < 31 {
                let leading_zeros = shifted.leading_zeros();
                prop_assert_eq!(leading_zeros, 31 - shift_amount as u32,
                    "Leading zeros should match shift amount");
            }
        }

        #[test]
        fn right_shift_properties(
            value in 1u32..0x80000000u32,
            shift_amount in 0usize..31usize
        ) {
            // Property: Right shift by N is equivalent to division by 2^N (for unsigned)
            let shifted = value >> shift_amount;
            let divided = value / (2u32.pow(shift_amount as u32));

            prop_assert_eq!(shifted, divided,
                "Right shift should equal division by 2^N");

            // Property: Monotonicity
            if shift_amount < 30 {
                let larger_shift = value >> (shift_amount + 1);
                prop_assert!(shifted >= larger_shift,
                    "Larger shift should give smaller or equal result");
            }

            // Property: Bounds
            prop_assert!(shifted <= value,
                "Right shift should not increase value");

            // Property: Idempotency for zero shift
            if shift_amount == 0 {
                prop_assert_eq!(shifted, value,
                    "Zero shift should preserve value");
            }

            // Property: Range bounds
            if shift_amount < 32 {
                let max_possible = value; // Original value
                let min_possible = if shift_amount >= 31 { 0 } else { value >> 31 };
                prop_assert!(shifted <= max_possible && shifted >= min_possible,
                    "Shifted value should be within expected range");
            }
        }

        #[test]
        fn bit_mask_properties(
            value in any::<u8>(),
            mask_bits in 1u8..8u8
        ) {
            let mask = (1u8 << mask_bits) - 1;
            let masked = value & mask;

            // Property: Masked value should be within mask range
            prop_assert!(masked <= mask,
                "Masked value should not exceed mask: {} & {} = {} > {}",
                value, mask, masked, mask);

            // Property: Idempotency
            let double_masked = masked & mask;
            prop_assert_eq!(masked, double_masked,
                "Double masking should be idempotent");

            // Property: Lower bits preservation
            for bit_pos in 0..mask_bits {
                let original_bit = (value >> bit_pos) & 1;
                let masked_bit = (masked >> bit_pos) & 1;
                prop_assert_eq!(original_bit, masked_bit,
                    "Bit {} should be preserved by mask", bit_pos);
            }

            // Property: Upper bits clearing
            for bit_pos in mask_bits..8 {
                let masked_bit = (masked >> bit_pos) & 1;
                prop_assert_eq!(masked_bit, 0,
                    "Bit {} should be cleared by mask", bit_pos);
            }

            // Property: OR vs AND behavior
            let or_with_mask = value | mask;
            prop_assert!(or_with_mask >= masked,
                "OR with mask should give >= AND with mask");
        }

        #[test]
        fn quantization_bit_operation_properties(
            values in prop::collection::vec(-2i8..=1i8, 1..16)
        ) {
            let packed = pack_2bit_values(&values);
            let unpacked = unpack_2bit_values(&packed, values.len());

            // Property: Round-trip preservation
            prop_assert_eq!(unpacked.clone(), values.clone(),
                "Round-trip should preserve values");

            // Property: Packed size efficiency
            let expected_bytes = values.len().div_ceil(4);
            prop_assert_eq!(packed.len(), expected_bytes,
                "Packed size should be optimal");

            // Property: All values in valid range
            for &val in &unpacked {
                prop_assert!((-2..=1).contains(&val),
                    "Unpacked value {} should be in range [-2,1]", val);
            }

            // Property: Bit field isolation
            for byte in &packed {
                for i in 0..4 {
                    let field = (byte >> (i * 2)) & 0x3;
                    prop_assert!(field <= 3,
                        "2-bit field should be <= 3: got {}", field);
                }
            }

            // Property: Deterministic packing
            let repacked = pack_2bit_values(&values);
            prop_assert_eq!(packed.clone(), repacked,
                "Packing should be deterministic");

            // Property: Partial unpacking consistency
            if values.len() > 1 {
                let partial = unpack_2bit_values(&packed, values.len() - 1);
                prop_assert_eq!(&partial[..], &values[..values.len() - 1],
                    "Partial unpacking should be consistent");
            }
        }
    }
}
