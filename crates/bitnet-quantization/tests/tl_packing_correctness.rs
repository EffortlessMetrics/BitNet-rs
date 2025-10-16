//! Comprehensive TL Packing Correctness Tests
//!
//! This test file validates the correctness of TL1/TL2 quantization packing formats:
//! - TL1/TL2: 2-bit packing with 4 elements per byte (precision_bits=2)
//! - Nibble extraction and byte boundary handling
//! - Format-aware indexing for packed element access
//!
//! **Packing Format**: Both TL1 and TL2 use 2-bit quantization:
//! - 4 elements fit in 1 byte
//! - Element 0: bits [1:0]
//! - Element 1: bits [3:2]
//! - Element 2: bits [5:4]
//! - Element 3: bits [7:6]
//!
//! Test Plan Reference: Issue #465 CPU path followup
//! Mutation Testing Target: Kill survivors in packing/unpacking arithmetic

#![cfg(feature = "cpu")]

use bitnet_quantization::utils::{pack_2bit_values, unpack_2bit_values};

// ============================================================================
// Test 1: TL1/TL2 2-bit Packing Correctness (4 elements per byte)
// ============================================================================

/// Test 1.1: Basic 2-bit packing - 4 elements fit in 1 byte
///
/// Validates that 4 elements are correctly packed into a single byte with
/// proper bit positioning: elem[i] at bits [2*i+1:2*i]
///
/// # Expected Behavior
/// - pack_2bit_values([0, 1, -1, -2]) produces 1 byte
/// - Each element occupies 2 bits in correct position
/// - Values are clamped to [-2, 1] range and offset to [0, 3]
#[test]
#[cfg(feature = "cpu")]
fn test_tl_2bit_packing_basic_correctness() {
    // Test packing 4 elements (exactly 1 byte)
    let values = vec![0i8, 1, -1, -2]; // Valid 2-bit signed range [-2, 1]
    let packed = pack_2bit_values(&values);

    assert_eq!(packed.len(), 1, "4 elements should pack into 1 byte");

    // Validate packed byte format
    // Values after offset: [2, 3, 1, 0] in unsigned [0, 3]
    // Byte layout: elem[0]=2 (bits 1-0), elem[1]=3 (bits 3-2), elem[2]=1 (bits 5-4), elem[3]=0 (bits 7-6)
    // Expected byte: 0b00_01_11_10 = 0x1E
    let expected_byte = 0x1E;
    assert_eq!(
        packed[0], expected_byte,
        "Packed byte should be 0x{:02X}, got 0x{:02X}",
        expected_byte, packed[0]
    );

    // Verify round-trip
    let unpacked = unpack_2bit_values(&packed, 4);
    assert_eq!(unpacked, values, "Round-trip should reconstruct original values");
}

/// Test 1.2: 2-bit packing - verify each element position
///
/// Tests that each element occupies the correct 2-bit field within the byte
///
/// # Note
/// Default value 0 → unsigned 2 (0b10), so non-set elements contribute 0b10 at their positions
#[test]
#[cfg(feature = "cpu")]
fn test_tl_2bit_packing_element_positions() {
    // Pack single elements at a time to verify bit positions
    // Each test sets one element to 1 (unsigned 3), rest are 0 (unsigned 2)
    let test_cases = [
        // elem[0] = 1 (uns 3 = 0b11), others = 0 (uns 2 = 0b10)
        // Byte: [elem3=10][elem2=10][elem1=10][elem0=11] = 0b10_10_10_11 = 0xAB
        (vec![1i8, 0, 0, 0], 0xAB),
        // elem[1] = 1 (uns 3 = 0b11), others = 0 (uns 2 = 0b10)
        // Byte: [elem3=10][elem2=10][elem1=11][elem0=10] = 0b10_10_11_10 = 0xAE
        (vec![0, 1, 0, 0], 0xAE),
        // elem[2] = 1 (uns 3 = 0b11), others = 0 (uns 2 = 0b10)
        // Byte: [elem3=10][elem2=11][elem1=10][elem0=10] = 0b10_11_10_10 = 0xBA
        (vec![0, 0, 1, 0], 0xBA),
        // elem[3] = 1 (uns 3 = 0b11), others = 0 (uns 2 = 0b10)
        // Byte: [elem3=11][elem2=10][elem1=10][elem0=10] = 0b11_10_10_10 = 0xEA
        (vec![0, 0, 0, 1], 0xEA),
    ];

    for (values, expected_byte) in test_cases {
        let packed = pack_2bit_values(&values);
        assert_eq!(packed.len(), 1, "Should pack to 1 byte");
        assert_eq!(
            packed[0], expected_byte,
            "Element position mismatch: values={:?}, expected=0x{:02X}, got=0x{:02X}",
            values, expected_byte, packed[0]
        );
    }
}

/// Test 1.3: 2-bit value clamping to [-2, 1] range
///
/// Validates that out-of-range values are clamped to 2-bit signed range
#[test]
#[cfg(feature = "cpu")]
fn test_tl_2bit_packing_value_clamping() {
    // Test values outside [-2, 1] are clamped
    let values = vec![
        -128, // Should clamp to -2
        -3,   // Should clamp to -2
        2,    // Should clamp to 1
        127,  // Should clamp to 1
    ];
    let packed = pack_2bit_values(&values);

    // After clamping: [-2, -2, 1, 1]
    // Unsigned: [0, 0, 3, 3]
    // Byte: 0b11_11_00_00 = 0xF0
    assert_eq!(packed[0], 0xF0, "Values should be clamped to [-2, 1] range");

    let unpacked = unpack_2bit_values(&packed, 4);
    assert_eq!(unpacked, vec![-2, -2, 1, 1], "Unpacked should be clamped values");
}

/// Test 1.4: 2-bit packing boundary conditions
///
/// Tests byte boundaries with multiple packed bytes
#[test]
#[cfg(feature = "cpu")]
fn test_tl_2bit_packing_byte_boundaries() {
    // Pack 8 elements (2 bytes)
    let values = vec![
        -2, -1, 0, 1, // First byte: [0, 1, 2, 3] unsigned
        1, 0, -1, -2, // Second byte: [3, 2, 1, 0] unsigned
    ];
    let packed = pack_2bit_values(&values);

    assert_eq!(packed.len(), 2, "8 elements should pack into 2 bytes");

    // First byte: 0b11_10_01_00 = 0xE4
    assert_eq!(packed[0], 0xE4, "First byte packing incorrect");

    // Second byte: 0b00_01_10_11 = 0x1B
    assert_eq!(packed[1], 0x1B, "Second byte packing incorrect");

    // Verify round-trip
    let unpacked = unpack_2bit_values(&packed, 8);
    assert_eq!(unpacked, values, "Byte boundary round-trip failed");
}

/// Test 1.5: 2-bit packing with partial last byte
///
/// Tests handling of non-multiple-of-4 element counts
#[test]
#[cfg(feature = "cpu")]
fn test_tl_2bit_packing_partial_byte() {
    // Pack 6 elements (1.5 bytes, should allocate 2 bytes)
    let values = vec![-2, -1, 0, 1, 0, -1];
    let packed = pack_2bit_values(&values);

    assert_eq!(packed.len(), 2, "6 elements should pack into 2 bytes");

    // First byte: [-2, -1, 0, 1] → [0, 1, 2, 3] unsigned → 0xE4
    assert_eq!(packed[0], 0xE4, "First byte incorrect");

    // Second byte: [0, -1] → [2, 1] unsigned → 0b00_00_01_10 = 0x06
    // Note: Remaining bits should be 0 (only 2 elements used)
    assert_eq!(packed[1], 0x06, "Partial byte incorrect");

    // Unpack should only return 6 elements
    let unpacked = unpack_2bit_values(&packed, 6);
    assert_eq!(unpacked.len(), 6, "Should unpack exactly 6 elements");
    assert_eq!(unpacked, values, "Partial byte round-trip failed");
}

// ============================================================================
// Test 2: TL Format-Aware Indexing (4 elements per byte)
// ============================================================================

/// Test 2.1: Format-aware indexing - calculate byte and bit offset
///
/// Validates that packed element access uses correct byte index and bit offset
/// for 2-bit packing format (4 elements per byte)
///
/// # Formula
/// - byte_index = elem_index / 4  (integer division)
/// - bit_offset = (elem_index % 4) * 2
#[test]
#[cfg(feature = "cpu")]
fn test_tl_format_aware_indexing_calculations() {
    // Test index to byte/bit offset mapping for 2-bit packing
    let test_cases = [
        (0, 0, 0),  // elem[0] → byte 0, bits [1:0]
        (1, 0, 2),  // elem[1] → byte 0, bits [3:2]
        (2, 0, 4),  // elem[2] → byte 0, bits [5:4]
        (3, 0, 6),  // elem[3] → byte 0, bits [7:6]
        (4, 1, 0),  // elem[4] → byte 1, bits [1:0]
        (5, 1, 2),  // elem[5] → byte 1, bits [3:2]
        (8, 2, 0),  // elem[8] → byte 2, bits [1:0]
        (15, 3, 6), // elem[15] → byte 3, bits [7:6]
    ];

    for (elem_index, expected_byte_index, expected_bit_offset) in test_cases {
        let byte_index = elem_index / 4; // Elements per byte for 2-bit packing
        let bit_offset = (elem_index % 4) * 2; // Bits per element

        assert_eq!(byte_index, expected_byte_index, "Byte index mismatch for elem[{}]", elem_index);
        assert_eq!(bit_offset, expected_bit_offset, "Bit offset mismatch for elem[{}]", elem_index);
    }
}

/// Test 2.2: Verify indexing via pack/unpack operations
///
/// Tests that element indexing correctly accesses packed values
#[test]
#[cfg(feature = "cpu")]
fn test_tl_format_aware_indexing_via_unpack() {
    // Create a packed array with known pattern
    let values: Vec<i8> = (0..16).map(|i| (i % 4) as i8 - 2).collect(); // [-2, -1, 0, 1] repeating
    let packed = pack_2bit_values(&values);

    assert_eq!(packed.len(), 4, "16 elements should pack into 4 bytes");

    // Verify each element can be accessed correctly via unpacking
    for (i, &expected_value) in values.iter().enumerate() {
        let byte_index = i / 4;
        let bit_offset = (i % 4) * 2;

        // Extract 2-bit value manually
        let byte = packed[byte_index];
        let unsigned_value = (byte >> bit_offset) & 0x3;
        let signed_value = unsigned_value as i8 - 2;

        assert_eq!(
            signed_value, expected_value,
            "Element {} mismatch: expected {}, got {} (byte[{}]=0x{:02X}, offset={})",
            i, expected_value, signed_value, byte_index, byte, bit_offset
        );
    }

    // Verify full unpack
    let unpacked = unpack_2bit_values(&packed, 16);
    assert_eq!(unpacked, values, "Full unpack should match original values");
}

// ============================================================================
// Test 3: TL Packing Edge Cases and Stress Tests
// ============================================================================

/// Test 3.1: Empty input
///
/// Validates handling of empty input arrays
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_empty_input() {
    let values: Vec<i8> = vec![];
    let packed = pack_2bit_values(&values);

    assert_eq!(packed.len(), 0, "Empty input should produce empty packed array");

    let unpacked = unpack_2bit_values(&packed, 0);
    assert_eq!(unpacked.len(), 0, "Unpacking empty array should produce empty output");
}

/// Test 3.2: Single element (partial byte)
///
/// Tests handling of single element requiring only 2 bits
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_single_element() {
    let values = vec![1i8];
    let packed = pack_2bit_values(&values);

    assert_eq!(packed.len(), 1, "1 element should pack into 1 byte");

    // Value 1 → unsigned 3 → 0b00_00_00_11 = 0x03
    assert_eq!(packed[0], 0x03, "Single element packing incorrect");

    let unpacked = unpack_2bit_values(&packed, 1);
    assert_eq!(unpacked, values, "Single element round-trip failed");
}

/// Test 3.3: Large array stress test
///
/// Tests packing/unpacking of large arrays to verify no off-by-one errors
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_large_array() {
    // Create 1024 elements (256 bytes)
    let values: Vec<i8> = (0..1024).map(|i| ((i % 4) as i8) - 2).collect();
    let packed = pack_2bit_values(&values);

    assert_eq!(packed.len(), 256, "1024 elements should pack into 256 bytes");

    let unpacked = unpack_2bit_values(&packed, 1024);
    assert_eq!(unpacked.len(), 1024, "Should unpack all 1024 elements");
    assert_eq!(unpacked, values, "Large array round-trip failed");
}

/// Test 3.4: All possible 2-bit values
///
/// Tests each of the 4 possible 2-bit signed values [-2, -1, 0, 1]
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_all_2bit_values() {
    let values = vec![-2i8, -1, 0, 1];
    let packed = pack_2bit_values(&values);

    assert_eq!(packed.len(), 1, "4 values should pack into 1 byte");

    // Values → unsigned: [0, 1, 2, 3]
    // Byte: 0b11_10_01_00 = 0xE4
    assert_eq!(packed[0], 0xE4, "All 2-bit values packing incorrect");

    let unpacked = unpack_2bit_values(&packed, 4);
    assert_eq!(unpacked, values, "All values round-trip failed");
}

/// Test 3.5: Repeated values pattern
///
/// Tests packing of repeated value patterns
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_repeated_patterns() {
    // Test all same value
    let values_all_zero = vec![0i8; 8];
    let packed_zero = pack_2bit_values(&values_all_zero);

    // 0 → unsigned 2 → 0b10_10_10_10 = 0xAA per byte
    assert_eq!(packed_zero[0], 0xAA, "All zeros packing incorrect (byte 0)");
    assert_eq!(packed_zero[1], 0xAA, "All zeros packing incorrect (byte 1)");

    // Test all -2 (minimum value)
    let values_all_min = vec![-2i8; 4];
    let packed_min = pack_2bit_values(&values_all_min);

    // -2 → unsigned 0 → 0b00_00_00_00 = 0x00
    assert_eq!(packed_min[0], 0x00, "All -2 values packing incorrect");

    // Test all 1 (maximum value)
    let values_all_max = vec![1i8; 4];
    let packed_max = pack_2bit_values(&values_all_max);

    // 1 → unsigned 3 → 0b11_11_11_11 = 0xFF
    assert_eq!(packed_max[0], 0xFF, "All 1 values packing incorrect");
}

// ============================================================================
// Test 4: TL Packing Mutation Killers
// ============================================================================

/// Test 4.1: Mutation killer - bit shift amounts
///
/// Validates that bit shifts use correct amounts (2 bits per element)
/// Catches mutations like: `i * 2` → `i * 1`, `i * 3`, `i + 2`, etc.
///
/// # Note
/// Default value 0 → unsigned 2, so non-set elements contribute to the byte value
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_bit_shift_correctness() {
    // Test specific values that expose incorrect shift amounts
    // Set one element to -2 (unsigned 0), rest to 1 (unsigned 3) to make shifts obvious
    let test_cases = [
        // elem[0] = -2 (uns 0), others = 1 (uns 3)
        // Byte: [elem3=11][elem2=11][elem1=11][elem0=00] = 0b11_11_11_00 = 0xFC
        (vec![-2, 1, 1, 1], 0xFC), // elem[0] shift 0
        // elem[1] = -2 (uns 0), others = 1 (uns 3)
        // Byte: [elem3=11][elem2=11][elem1=00][elem0=11] = 0b11_11_00_11 = 0xF3
        (vec![1, -2, 1, 1], 0xF3), // elem[1] shift 2 (not 1 or 3)
        // elem[2] = -2 (uns 0), others = 1 (uns 3)
        // Byte: [elem3=11][elem2=00][elem1=11][elem0=11] = 0b11_00_11_11 = 0xCF
        (vec![1, 1, -2, 1], 0xCF), // elem[2] shift 4 (not 2 or 6)
        // elem[3] = -2 (uns 0), others = 1 (uns 3)
        // Byte: [elem3=00][elem2=11][elem1=11][elem0=11] = 0b00_11_11_11 = 0x3F
        (vec![1, 1, 1, -2], 0x3F), // elem[3] shift 6 (not 3 or 9)
    ];

    for (values, expected_byte) in test_cases {
        let packed = pack_2bit_values(&values);
        assert_eq!(
            packed[0], expected_byte,
            "Bit shift mutation detected: values={:?}, expected=0x{:02X}, got=0x{:02X}",
            values, expected_byte, packed[0]
        );
    }
}

/// Test 4.2: Mutation killer - bit mask correctness
///
/// Validates that bit extraction uses correct mask (0x3 for 2 bits)
/// Catches mutations like: `& 0x3` → `& 0x1`, `& 0x7`, `& 0x2`, etc.
#[test]
#[cfg(feature = "cpu")]
fn test_tl_unpacking_bit_mask_correctness() {
    // Pack values and verify unpacking uses correct mask
    let values = vec![-2, -1, 0, 1]; // Full 2-bit range
    let packed = pack_2bit_values(&values);

    let unpacked = unpack_2bit_values(&packed, 4);

    // If mask is wrong (e.g., 0x1 instead of 0x3), values will be truncated
    assert_eq!(unpacked[0], -2, "Bit mask mutation: elem[0] should be -2");
    assert_eq!(unpacked[1], -1, "Bit mask mutation: elem[1] should be -1");
    assert_eq!(unpacked[2], 0, "Bit mask mutation: elem[2] should be 0");
    assert_eq!(unpacked[3], 1, "Bit mask mutation: elem[3] should be 1");
}

/// Test 4.3: Mutation killer - offset arithmetic
///
/// Validates that signed/unsigned conversion uses correct offset (±2)
/// Catches mutations like: `+ 2` → `+ 1`, `+ 3`, `- 2`, etc.
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_offset_arithmetic() {
    // Pack boundary values
    let values = vec![-2, 1]; // Min and max 2-bit signed values
    let packed = pack_2bit_values(&values);

    // -2 should map to unsigned 0, 1 should map to unsigned 3
    // If offset is wrong, these mappings will be incorrect

    let unpacked = unpack_2bit_values(&packed, 2);
    assert_eq!(unpacked[0], -2, "Offset mutation: -2 should round-trip correctly");
    assert_eq!(unpacked[1], 1, "Offset mutation: 1 should round-trip correctly");

    // Verify packed representation
    // [-2, 1] → unsigned [0, 3] → 0b00_00_11_00 = 0x0C
    assert_eq!(
        packed[0], 0x0C,
        "Offset arithmetic mutation: expected 0x0C, got 0x{:02X}",
        packed[0]
    );
}

/// Test 4.4: Mutation killer - division rounding
///
/// Validates that byte index calculation uses integer division
/// Catches mutations like: `/ 4` → `/ 3`, `/ 5`, `% 4`, etc.
#[test]
#[cfg(feature = "cpu")]
fn test_tl_indexing_division_correctness() {
    // Test that elements 0-3 map to byte 0, 4-7 to byte 1, etc.
    let values: Vec<i8> = (0..12).map(|i| (i % 4) as i8 - 2).collect();
    let packed = pack_2bit_values(&values);

    assert_eq!(packed.len(), 3, "12 elements should pack into 3 bytes (12/4)");

    // If division is wrong (e.g., /3 or /5), packed length will be incorrect
    // Verify unpacking correctly maps indices
    let unpacked = unpack_2bit_values(&packed, 12);
    assert_eq!(unpacked, values, "Division rounding mutation detected");
}

/// Test 4.5: Mutation killer - boundary off-by-one
///
/// Validates that byte boundaries are exact (no ±1 errors)
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_boundary_exact() {
    // Test exactly 4, 8, 12 elements (byte boundaries)
    for num_elems in [4, 8, 12] {
        let values: Vec<i8> = (0..num_elems).map(|i| (i % 4) as i8 - 2).collect();
        let packed = pack_2bit_values(&values);

        let expected_bytes = num_elems / 4;
        assert_eq!(
            packed.len(),
            expected_bytes,
            "Boundary off-by-one: {} elements should pack into {} bytes, got {}",
            num_elems,
            expected_bytes,
            packed.len()
        );

        let unpacked = unpack_2bit_values(&packed, num_elems);
        assert_eq!(
            unpacked, values,
            "Boundary off-by-one: round-trip failed for {} elements",
            num_elems
        );
    }
}

// ============================================================================
// Test 5: TL Integration with Quantization Block Structure
// ============================================================================

/// Test 5.1: Block-aligned packing
///
/// Tests that packing works correctly with quantization block sizes
/// (typical block sizes: 64, 128, 256 elements)
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_block_aligned() {
    for block_size in [64, 128, 256] {
        let values: Vec<i8> = (0..block_size).map(|i| (i % 4) as i8 - 2).collect();
        let packed = pack_2bit_values(&values);

        let expected_bytes = block_size / 4;
        assert_eq!(
            packed.len(),
            expected_bytes,
            "Block size {} should pack into {} bytes",
            block_size,
            expected_bytes
        );

        let unpacked = unpack_2bit_values(&packed, block_size);
        assert_eq!(
            unpacked, values,
            "Block-aligned round-trip failed for block_size={}",
            block_size
        );
    }
}

/// Test 5.2: Multiple blocks packing
///
/// Tests packing multiple quantization blocks sequentially
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_multiple_blocks() {
    const BLOCK_SIZE: usize = 64;
    const NUM_BLOCKS: usize = 4;

    let values: Vec<i8> = (0..BLOCK_SIZE * NUM_BLOCKS).map(|i| (i % 4) as i8 - 2).collect();
    let packed = pack_2bit_values(&values);

    let expected_bytes = (BLOCK_SIZE * NUM_BLOCKS) / 4;
    assert_eq!(packed.len(), expected_bytes, "Multiple blocks packing incorrect");

    let unpacked = unpack_2bit_values(&packed, BLOCK_SIZE * NUM_BLOCKS);
    assert_eq!(unpacked, values, "Multiple blocks round-trip failed");
}

// ============================================================================
// Test 6: Performance and Determinism
// ============================================================================

/// Test 6.1: Deterministic packing
///
/// Validates that packing produces consistent results
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_deterministic() {
    let values: Vec<i8> = (0..100).map(|i| (i % 4) as i8 - 2).collect();

    // Pack multiple times and verify identical results
    let packed1 = pack_2bit_values(&values);
    let packed2 = pack_2bit_values(&values);
    let packed3 = pack_2bit_values(&values);

    assert_eq!(packed1, packed2, "Packing is not deterministic (run 1 vs 2)");
    assert_eq!(packed2, packed3, "Packing is not deterministic (run 2 vs 3)");
}

/// Test 6.2: Round-trip fidelity stress test
///
/// Tests that pack/unpack is a perfect inverse operation for all valid inputs
#[test]
#[cfg(feature = "cpu")]
fn test_tl_packing_round_trip_fidelity() {
    // Test all combinations of 4 elements with all possible 2-bit values
    for v0 in -2..=1 {
        for v1 in -2..=1 {
            for v2 in -2..=1 {
                for v3 in -2..=1 {
                    let values = vec![v0, v1, v2, v3];
                    let packed = pack_2bit_values(&values);
                    let unpacked = unpack_2bit_values(&packed, 4);

                    assert_eq!(
                        unpacked, values,
                        "Round-trip fidelity failed for [{}, {}, {}, {}]",
                        v0, v1, v2, v3
                    );
                }
            }
        }
    }
}
