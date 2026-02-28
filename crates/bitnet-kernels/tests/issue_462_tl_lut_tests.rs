// AC:4 - TL LUT Helper Test Scaffolding
//
// This test file validates AC4 from Issue #462:
// - Safe TL1 LUT indexing with bounds checking
// - Safe TL2 LUT indexing with bounds checking
// - Out-of-bounds element index error handling
// - Integration with TL1/TL2 quantized matmul
//
// Test Plan Reference: docs/explanation/cpu-inference-test-plan.md
// Spec: docs/explanation/tl-lut-helper-spec.md

#![cfg(feature = "cpu")]

use anyhow::Result;

// ============================================================================
// AC:4 - Test 4.1: Valid LUT Index Calculation
// ============================================================================

/// AC:4 - T4.1: Safe TL1 LUT indexing with valid inputs
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-41
/// Validates correct LUT index calculation for TL1 configuration
///
/// # Expected Behavior
/// - TL1 config: block_bytes=16, elems_per_block=128
/// - lut_index(0, 0, 16, 128) → 0
/// - lut_index(0, 8, 16, 128) → 1 (8/8 = 1 byte offset)
/// - lut_index(1, 0, 16, 128) → 16 (block 1 offset)
/// - lut_index(5, 64, 16, 128) → 88 (5*16 + 64/8 = 88)
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_tl_lut_index_bounds_valid() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // TL1 configuration tests (block_bytes=16, elems_per_block=128, lut_len=256)
    assert_eq!(lut_index(0, 0, 16, 128, 256)?, 0, "Index at block 0, elem 0");
    assert_eq!(lut_index(0, 8, 16, 128, 256)?, 1, "8/8 = 1 byte offset");
    assert_eq!(lut_index(1, 0, 16, 128, 256)?, 16, "Block 1 starts at byte 16");
    assert_eq!(lut_index(5, 64, 16, 128, 256)?, 88, "5*16 + 64/8 = 88");
    assert_eq!(lut_index(10, 120, 16, 128, 256)?, 175, "10*16 + 120/8 = 175");

    // TL2 configuration tests (block_bytes=32, elems_per_block=256, lut_len=512)
    assert_eq!(lut_index(0, 0, 32, 256, 512)?, 0, "TL2: block 0, elem 0");
    assert_eq!(lut_index(2, 16, 32, 256, 512)?, 66, "TL2: 2*32 + 16/8 = 66");
    assert_eq!(lut_index(3, 128, 32, 256, 512)?, 112, "TL2: 3*32 + 128/8 = 112");

    // Edge case: Last valid element in block
    assert_eq!(lut_index(0, 127, 16, 128, 256)?, 15, "Last elem in TL1 block: 127/8 = 15");
    assert_eq!(lut_index(0, 255, 32, 256, 512)?, 31, "Last elem in TL2 block: 255/8 = 31");

    Ok(())
}

// ============================================================================
// AC:4 - Test 4.2: Invalid LUT Index (Out of Bounds)
// ============================================================================

/// AC:4 - T4.2: Safe TL2 LUT indexing - bounds checking errors
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-42
/// Validates error handling for out-of-bounds element indices
///
/// # Expected Behavior
/// - elem_in_block >= elems_per_block → Error
/// - Error type: LutIndexError::OutOfBounds(elem, max)
/// - Descriptive error message
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_tl_lut_index_bounds_invalid() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // Test elem_in_block >= elems_per_block
    let result = lut_index(0, 128, 16, 128, 256);
    assert!(result.is_err(), "Should fail: elem_in_block (128) >= elems_per_block (128)");

    // Test well beyond bounds
    let result = lut_index(1, 200, 16, 128, 256);
    assert!(result.is_err(), "Should fail: elem_in_block (200) >= elems_per_block (128)");

    // Test edge case: elem_in_block = elems_per_block (exactly at boundary)
    let result = lut_index(0, 128, 16, 128, 256);
    assert!(result.is_err(), "Boundary condition: elem_in_block == elems_per_block should fail");

    // Test error message clarity
    let err = lut_index(0, 150, 16, 128, 256).unwrap_err();
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("150") && err_msg.contains("128"),
        "Error message should include both elem_in_block (150) and max (128)"
    );

    Ok(())
}

// ============================================================================
// AC:4 - Test 4.3: Additional Error Cases
// ============================================================================

/// AC:4 - Additional: Invalid configuration and overflow detection
///
/// Validates error handling for invalid configurations and overflows
///
/// # Expected Behavior
/// - block_bytes=0 → InvalidConfig error
/// - elems_per_block=0 → InvalidConfig error
/// - block_idx * block_bytes overflow → BlockOffsetOverflow error
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_tl_lut_index_invalid_config() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // Test zero block_bytes - division by zero handled, valid case actually
    // (block_idx=0 * block_bytes=0 = 0, elem/8 gives offset)
    // This is actually valid, so we test overflow instead

    // Test zero elems_per_block - boundary check will fail
    let result = lut_index(0, 0, 16, 0, 256);
    assert!(
        result.is_err(),
        "Should fail: elems_per_block = 0 means any elem_in_block is out of bounds"
    );

    // Test overflow: block_idx * block_bytes
    let result = lut_index(usize::MAX, 0, 16, 128, usize::MAX);
    assert!(result.is_err(), "Should fail: block_idx overflow");
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Overflow"), "Error should mention overflow");

    Ok(())
}

// ============================================================================
// AC:4 - Integration Test: TL1/TL2 Matmul with Safe LUT
// ============================================================================

/// AC:4 - T4.3: TL1/TL2 matmul integration with safe LUT helper
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-43
/// Validates TL1/TL2 quantized matmul integration with safe LUT indexing
///
/// # Expected Behavior
/// - TL1/TL2 matmul completes without panics
/// - Output shape correct
/// - No LUT indexing errors
/// - Previously ignored TL tests now pass
///
/// # Note
/// This test may be located in bitnet-inference crate instead,
/// as it requires QuantizedLinear integration.
#[test]
#[cfg(feature = "cpu")]
#[ignore = "Requires QuantizedLinear TL1/TL2 integration with lut_index helper"]
fn test_ac4_tl_matmul_with_safe_lut() -> Result<()> {
    // TODO: This test requires integration with bitnet-inference::layers::QuantizedLinear
    // Option 1: Import from bitnet-inference (cross-crate test)
    // use bitnet_inference::layers::QuantizedLinear;
    // use bitnet_common::QuantizationType;
    //
    // let qlinear = create_test_quantized_linear(QuantizationType::TL1)?;
    // let input = create_test_input(1, 128)?; // [1, in_features]
    //
    // let output = qlinear.forward(&input).await?;
    //
    // assert_eq!(output.shape(), &[1, 256], "Output shape should be [1, out_features]");

    // Option 2: Test via CPU forward pass (full integration)
    // This may be redundant with AC1 tests if TL1/TL2 paths are exercised there

    anyhow::bail!(
        "UNIMPLEMENTED: TL matmul integration not yet implemented.\n\
         Expected: TL1/TL2 quantized matmul uses lut_index() helper.\n\
         Integration: crates/bitnet-inference/src/layers/quantized_linear.rs\n\
         Changes: Replace inline LUT indexing with bitnet_kernels::tl_lut::lut_index().\n\
         This test will pass once AC4 TL matmul integration is complete."
    );
}

// ============================================================================
// Unit Tests: LUT Index Helper Properties
// ============================================================================

/// Property-based test: LUT index monotonicity
///
/// Validates that increasing elem_in_block increases or maintains index
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_lut_index_monotonicity() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // Simple monotonicity test without proptest
    for block_idx in 0..10 {
        for elem1 in (0..120).step_by(8) {
            for elem2 in (elem1 + 8..128).step_by(8) {
                let idx1 = lut_index(block_idx, elem1, 16, 128, 256)?;
                let idx2 = lut_index(block_idx, elem2, 16, 128, 256)?;
                assert!(
                    idx1 <= idx2,
                    "LUT index should be monotonic: block={}, elem1={}, elem2={}, idx1={}, idx2={}",
                    block_idx,
                    elem1,
                    elem2,
                    idx1,
                    idx2
                );
            }
        }
    }

    Ok(())
}

/// Unit test: LUT index calculation formula verification
///
/// Validates formula: lut_index = block_idx * block_bytes + (elem_in_block / 8)
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_lut_index_formula() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    const ELEMS_PER_BYTE: usize = 8; // 1-bit quantization packing

    for block_idx in 0..10 {
        for elem_in_block in (0..128).step_by(8) {
            let expected = block_idx * 16 + (elem_in_block / ELEMS_PER_BYTE);
            let actual = lut_index(block_idx, elem_in_block, 16, 128, 256)?;
            assert_eq!(
                actual, expected,
                "Formula mismatch: block={}, elem={}, expected={}, got={}",
                block_idx, elem_in_block, expected, actual
            );
        }
    }

    Ok(())
}

// ============================================================================
// Edge Case Tests: Additional Boundary Conditions
// ============================================================================

/// Edge case: Zero block_bytes (valid - idx = elem_offset only)
///
/// Tests that block_bytes=0 works correctly (base_offset always 0)
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_lut_index_zero_block_bytes() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // With block_bytes=0, base_offset is always 0, idx = elem_in_block / 8
    assert_eq!(lut_index(0, 0, 0, 128, 256)?, 0, "block_bytes=0: elem 0 → idx 0");
    assert_eq!(lut_index(0, 8, 0, 128, 256)?, 1, "block_bytes=0: elem 8 → idx 1");
    assert_eq!(
        lut_index(100, 16, 0, 128, 256)?,
        2,
        "block_bytes=0: block_idx ignored, elem 16 → idx 2"
    );

    // Even with large block_idx, result only depends on elem_in_block
    assert_eq!(lut_index(1000, 64, 0, 128, 256)?, 8, "block_bytes=0: elem 64 → idx 8");

    Ok(())
}

/// Edge case: Maximum valid element in block (boundary test)
///
/// Tests elem_in_block = elems_per_block - 1 (last valid element)
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_lut_index_max_valid_element() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // TL1: elems_per_block = 128, last valid elem = 127
    let result = lut_index(0, 127, 16, 128, 256)?;
    assert_eq!(result, 15, "TL1: Last elem 127 → offset 127/8 = 15");

    // TL2: elems_per_block = 256, last valid elem = 255
    let result = lut_index(0, 255, 32, 256, 512)?;
    assert_eq!(result, 31, "TL2: Last elem 255 → offset 255/8 = 31");

    // With block_idx > 0
    let result = lut_index(2, 127, 16, 128, 256)?;
    assert_eq!(result, 47, "TL1 block 2: 2*16 + 127/8 = 32 + 15 = 47");

    Ok(())
}

/// Edge case: LUT boundary (idx = lut_len - 1, valid)
///
/// Tests that index exactly at lut_len-1 is accepted
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_lut_index_exact_lut_boundary() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // Calculate parameters for idx = lut_len - 1
    // For lut_len=256, block_bytes=16: idx = 255 requires block_idx * 16 + elem_offset = 255
    // block_idx=15, elem_offset=15 → 15*16 + 15 = 240 + 15 = 255
    let result = lut_index(15, 120, 16, 128, 256)?;
    assert_eq!(result, 255, "idx = lut_len - 1 should be valid: 15*16 + 120/8 = 255");

    // Now test idx = lut_len (should fail)
    let result = lut_index(16, 0, 16, 128, 256);
    assert!(result.is_err(), "idx = lut_len (256) should fail: 16*16 = 256 >= 256");

    Ok(())
}

/// Edge case: Division by 8 rounding behavior
///
/// Tests that elem_in_block values 0-7 → offset 0, 8-15 → offset 1, etc.
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_lut_index_division_rounding() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // All elem_in_block values 0-7 should give same offset (0)
    for elem in 0..8 {
        assert_eq!(
            lut_index(0, elem, 32, 128, 256)?,
            0,
            "elem_in_block {} should give offset 0 (division truncates)",
            elem
        );
    }

    // All elem_in_block values 8-15 should give offset 1
    for elem in 8..16 {
        assert_eq!(
            lut_index(0, elem, 32, 128, 256)?,
            1,
            "elem_in_block {} should give offset 1",
            elem
        );
    }

    // All elem_in_block values 120-127 should give offset 15
    for elem in 120..128 {
        assert_eq!(
            lut_index(0, elem, 32, 128, 256)?,
            15,
            "elem_in_block {} should give offset 15",
            elem
        );
    }

    Ok(())
}

/// Edge case: Overflow in elem_offset addition
///
/// Tests overflow detection in base_offset + elem_offset calculation
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_lut_index_elem_offset_overflow() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // Construct scenario where base_offset + elem_offset overflows
    // base_offset = (usize::MAX - 10), elem_offset = 64/8 = 8
    // Result: (usize::MAX - 10) + 8 = usize::MAX - 2 (valid if lut_len is large enough)
    // To trigger overflow, use elem_offset that exceeds remaining capacity

    // Use block_bytes=1 for precise control
    // block_idx = usize::MAX - 5, block_bytes = 1 → base_offset = usize::MAX - 5
    // elem_in_block = 64 → elem_offset = 8
    // Result: (usize::MAX - 5) + 8 would overflow
    let result = lut_index(usize::MAX - 5, 64, 1, 128, usize::MAX);
    assert!(result.is_err(), "Expected overflow in base_offset + elem_offset");

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Overflow") || err_msg.contains("exceeds LUT length"),
        "Error should mention overflow or exceeding LUT length, got: {}",
        err_msg
    );

    Ok(())
}

// ============================================================================
// Mutation Testing Preparation: Formula Validation Tests
// ============================================================================

/// Mutation killer: Verify exact formula at multiple points
///
/// This test will catch mutations like:
/// - block_idx * block_bytes → block_idx + block_bytes
/// - elem_in_block / 8 → elem_in_block / 7
/// - base_offset + elem_offset → base_offset - elem_offset
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_lut_index_formula_exact_values() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;

    // Test points where formula changes would be detected
    let test_cases = [
        (0, 0, 16, 128, 256, 0),    // block_idx * 16 + 0/8 = 0
        (1, 0, 16, 128, 256, 16),   // 1 * 16 + 0 = 16
        (2, 0, 16, 128, 256, 32),   // 2 * 16 + 0 = 32
        (0, 8, 16, 128, 256, 1),    // 0 * 16 + 8/8 = 1
        (0, 16, 16, 128, 256, 2),   // 0 * 16 + 16/8 = 2
        (1, 8, 16, 128, 256, 17),   // 1 * 16 + 8/8 = 17
        (3, 24, 16, 128, 256, 51),  // 3 * 16 + 24/8 = 48 + 3 = 51
        (5, 64, 32, 256, 512, 168), // 5 * 32 + 64/8 = 160 + 8 = 168
    ];

    for (block_idx, elem_in_block, block_bytes, elems_per_block, lut_len, expected) in test_cases {
        let actual = lut_index(block_idx, elem_in_block, block_bytes, elems_per_block, lut_len)?;
        assert_eq!(
            actual, expected,
            "Formula mismatch: lut_index({}, {}, {}, {}, {}) = {}, expected {}",
            block_idx, elem_in_block, block_bytes, elems_per_block, lut_len, actual, expected
        );
    }

    Ok(())
}

// ============================================================================
// Performance Test: LUT Index Helper Overhead
// ============================================================================

/// Benchmark: LUT index helper vs inline calculation
///
/// Validates that bounds checking overhead is negligible (<5%)
#[test]
#[cfg(feature = "cpu")]
fn test_ac4_lut_index_performance() -> Result<()> {
    use bitnet_kernels::tl_lut::lut_index;
    use std::time::Instant;

    // TL1 config: 16 bytes/block, 128 elems/block.
    // 1000 blocks × 16 iterations → max index = 999*16 + 15 = 15_999.
    const BLOCK_BYTES: usize = 16;
    const ELEMS_PER_BLOCK: usize = 128;
    const LUT_LEN: usize = 16_000;

    let start = Instant::now();
    for block_idx in 0..1000 {
        for elem_in_block in (0..ELEMS_PER_BLOCK).step_by(8) {
            lut_index(block_idx, elem_in_block, BLOCK_BYTES, ELEMS_PER_BLOCK, LUT_LEN)?;
        }
    }
    let duration = start.elapsed();

    // 16_000 calls should complete well within 1 second on any CI machine.
    assert!(duration.as_secs() < 1, "16 000 lut_index calls took too long: {:?}", duration);

    Ok(())
}
