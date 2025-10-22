//! QK256 Error Handling and Edge Case Tests
//!
//! Tests feature spec: i2s-dual-flavor.md#qk256-error-handling
//!
//! This test suite validates comprehensive error handling and edge cases for QK256:
//!
//! - Dimension mismatches (tensor size validation)
//! - Invalid code values (should never occur but test boundary)
//! - Zero/negative dimensions
//! - Extremely large matrices (memory limits)
//! - Concurrent access patterns (thread safety)

use bitnet_models::quant::i2s_qk256::{
    I2SQk256NoScale, QK256_BLOCK, QK256_PACKED_BYTES, gemv_qk256, gemv_qk256_row,
    unpack_qk256_block,
};

// ==================== Dimension Validation Tests ====================

/// Test spec: i2s-dual-flavor.md#dimension-mismatch-detection
///
/// Verify gemv_qk256 detects and rejects mismatched dimensions
#[test]
fn test_gemv_qk256_mismatched_output_dimension() {
    let rows = 4;
    let cols = 256;
    let qs_data = vec![0xAAu8; rows * QK256_PACKED_BYTES];
    let input = vec![1.0f32; cols];

    // Test: output vector too small
    let mut output_too_small = vec![0.0f32; rows - 1];
    let result =
        gemv_qk256(&qs_data, &input, &mut output_too_small, rows, cols, QK256_PACKED_BYTES);
    assert!(result.is_err(), "Should reject output too small");
    assert!(result.unwrap_err().to_string().contains("y_out length"));

    // Test: output vector too large
    let mut output_too_large = vec![0.0f32; rows + 1];
    let result =
        gemv_qk256(&qs_data, &input, &mut output_too_large, rows, cols, QK256_PACKED_BYTES);
    assert!(result.is_err(), "Should reject output too large");
    assert!(result.unwrap_err().to_string().contains("y_out length"));
}

/// Test spec: i2s-dual-flavor.md#insufficient-input-detection
///
/// Verify gemv_qk256 detects insufficient input data
#[test]
fn test_gemv_qk256_insufficient_input() {
    let rows = 4;
    let cols = 256;
    let qs_data = vec![0xAAu8; rows * QK256_PACKED_BYTES];

    // Test: input vector too short
    let short_input = vec![1.0f32; cols - 10];
    let mut output = vec![0.0f32; rows];

    let result = gemv_qk256(&qs_data, &short_input, &mut output, rows, cols, QK256_PACKED_BYTES);
    assert!(result.is_err(), "Should reject insufficient input");
    assert!(result.unwrap_err().to_string().contains("x length"));
}

/// Test spec: i2s-dual-flavor.md#insufficient-quantized-data-detection
///
/// Verify gemv_qk256 detects insufficient quantized data
#[test]
fn test_gemv_qk256_insufficient_quantized_data() {
    let rows = 4;
    let cols = 256;

    // Test: quantized data too short (only 1 row instead of 4)
    let short_qs = vec![0xAAu8; QK256_PACKED_BYTES];
    let input = vec![1.0f32; cols];
    let mut output = vec![0.0f32; rows];

    let result = gemv_qk256(&short_qs, &input, &mut output, rows, cols, QK256_PACKED_BYTES);
    assert!(result.is_err(), "Should reject insufficient quantized data");
    assert!(result.unwrap_err().to_string().contains("data too short"));
}

/// Test spec: i2s-dual-flavor.md#i2s-qk256-struct-size-validation
///
/// Verify I2SQk256NoScale::new validates data size
#[test]
fn test_i2s_qk256_no_scale_size_validation() {
    let rows = 8;
    let cols: usize = 512; // 2 blocks
    let blocks_per_row = cols.div_ceil(QK256_BLOCK); // = 2
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES; // = 128
    let expected_bytes = rows * row_stride_bytes; // = 1024

    // Note: I2SQk256NoScale::new allows a tolerance of 128 bytes for alignment padding
    const TOLERANCE: usize = 128;

    // Test 1: Correct size (should succeed)
    let qs_correct = vec![0u8; expected_bytes];
    let result = I2SQk256NoScale::new(rows, cols, qs_correct);
    assert!(result.is_ok(), "Correct size should succeed");

    // Test 2: Within tolerance (should succeed)
    let qs_within_tolerance = vec![0u8; expected_bytes + TOLERANCE];
    let result = I2SQk256NoScale::new(rows, cols, qs_within_tolerance);
    assert!(result.is_ok(), "Size within tolerance should succeed");

    // Test 3: Too small beyond tolerance (should fail)
    let qs_too_small = vec![0u8; expected_bytes - TOLERANCE - 1];
    let result = I2SQk256NoScale::new(rows, cols, qs_too_small);
    assert!(result.is_err(), "Too small beyond tolerance should fail");
    assert!(result.unwrap_err().to_string().contains("data size mismatch"));

    // Test 4: Too large beyond tolerance (should fail)
    let qs_too_large = vec![0u8; expected_bytes + TOLERANCE + 1];
    let result = I2SQk256NoScale::new(rows, cols, qs_too_large);
    assert!(result.is_err(), "Too large beyond tolerance should fail");
    assert!(result.unwrap_err().to_string().contains("data size mismatch"));
}

// ==================== Edge Case Tests ====================

/// Test spec: i2s-dual-flavor.md#zero-input-handling
///
/// Verify QK256 handles zero input correctly (should produce zero output)
#[test]
fn test_qk256_zero_input_vector() {
    let rows = 8;
    let cols = 256;
    let qs_data = vec![0xAAu8; rows * QK256_PACKED_BYTES]; // Code 2 → +1.0

    let zero_input = vec![0.0f32; cols];
    let mut output = vec![0.0f32; rows];

    gemv_qk256(&qs_data, &zero_input, &mut output, rows, cols, QK256_PACKED_BYTES)
        .expect("Zero input should succeed");

    // All outputs should be zero (weight * 0 = 0)
    for (i, &val) in output.iter().enumerate() {
        assert!(val.abs() < 1e-9, "Row {}: expected 0.0, got {} (zero input failed)", i, val);
    }
}

/// Test spec: i2s-dual-flavor.md#zero-weights-handling
///
/// Verify QK256 handles zero weights correctly (code 0 and 2 average to zero)
#[test]
fn test_qk256_zero_weights_effect() {
    let rows = 2;
    let cols = 256;

    // Alternating codes: 0 (-2.0) and 3 (+2.0) should cancel out
    // Pattern: 0b_11_00_11_00 = 0xCC
    let qs_data = vec![0xCCu8; rows * QK256_PACKED_BYTES];

    let input = vec![1.0f32; cols];
    let mut output = vec![0.0f32; rows];

    gemv_qk256(&qs_data, &input, &mut output, rows, cols, QK256_PACKED_BYTES)
        .expect("Alternating weights should succeed");

    // Outputs should be near zero (half -2.0, half +2.0)
    for (i, &val) in output.iter().enumerate() {
        assert!(val.abs() < 1e-3, "Row {}: expected ~0.0 (alternating weights), got {}", i, val);
    }
}

/// Test spec: i2s-dual-flavor.md#single-element-matrix
///
/// Verify QK256 handles minimal matrix (1x1)
#[test]
fn test_qk256_single_element_matrix() {
    let rows = 1;
    let cols = 1; // Single element (still needs 64-byte block)
    let qs_data = vec![0xAAu8; QK256_PACKED_BYTES]; // Code 2 → +1.0

    let input = vec![3.0f32];
    let mut output = vec![0.0f32; rows];

    gemv_qk256(&qs_data, &input, &mut output, rows, cols, QK256_PACKED_BYTES)
        .expect("Single element should succeed");

    // Output should be 1.0 * 3.0 = 3.0
    assert!((output[0] - 3.0).abs() < 1e-5, "Expected 3.0, got {}", output[0]);
}

/// Test spec: i2s-dual-flavor.md#large-matrix-handling
///
/// Verify QK256 handles large production-size matrices
#[test]
fn test_qk256_large_production_matrix() {
    // Large production-size matrix: 4096×4096 (16 blocks per row)
    let rows = 4096;
    let cols: usize = 4096;
    let blocks_per_row = cols.div_ceil(QK256_BLOCK); // = 16
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES; // = 1024

    // Use pattern: code 2 → +1.0
    let qs_data = vec![0xAAu8; rows * row_stride_bytes];

    // Input: all ones
    let input = vec![1.0f32; cols];

    // Only compute first 8 rows (to keep test fast)
    let test_rows = 8;
    let mut output = vec![0.0f32; test_rows];

    // Extract first 8 rows of quantized data
    let qs_slice = &qs_data[..test_rows * row_stride_bytes];

    gemv_qk256(qs_slice, &input, &mut output, test_rows, cols, row_stride_bytes)
        .expect("Large matrix should succeed");

    // Each row: dot product of all +1.0 weights with all +1.0 inputs = cols
    let expected = cols as f32;
    for (i, &val) in output.iter().enumerate() {
        assert!((val - expected).abs() < 1e-2, "Row {}: expected {}, got {}", i, expected, val);
    }
}

/// Test spec: i2s-dual-flavor.md#tail-block-edge-cases
///
/// Verify QK256 handles various tail sizes correctly
#[test]
fn test_qk256_tail_block_edge_cases() {
    // Test various tail sizes: 1, 44, 128, 255
    let tail_sizes: Vec<usize> = vec![1, 44, 128, 255];

    for tail in tail_sizes {
        let rows = 2;
        let cols = 256 + tail; // 1 full block + tail
        let blocks_per_row = cols.div_ceil(QK256_BLOCK); // = 2
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

        let qs_data = vec![0xAAu8; rows * row_stride_bytes]; // Code 2 → +1.0
        let input = vec![1.0f32; cols];
        let mut output = vec![0.0f32; rows];

        gemv_qk256(&qs_data, &input, &mut output, rows, cols, row_stride_bytes)
            .unwrap_or_else(|_| panic!("Tail size {} should succeed", tail));

        // Expected: sum of all +1.0 weights * +1.0 inputs = cols
        let expected = cols as f32;
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - expected).abs() < 1e-3,
                "Tail {}, Row {}: expected {}, got {}",
                tail,
                i,
                expected,
                val
            );
        }
    }
}

// ==================== Numerical Stability Tests ====================

/// Test spec: i2s-dual-flavor.md#numerical-stability
///
/// Verify QK256 maintains numerical stability with extreme values
#[test]
fn test_qk256_numerical_stability_large_inputs() {
    let rows = 4;
    let cols = 256;
    let qs_data = vec![0xAAu8; rows * QK256_PACKED_BYTES]; // Code 2 → +1.0

    // Test with large input values
    let large_input = vec![1e6f32; cols];
    let mut output = vec![0.0f32; rows];

    gemv_qk256(&qs_data, &large_input, &mut output, rows, cols, QK256_PACKED_BYTES)
        .expect("Large input should succeed");

    // Expected: sum of 256 weights (+1.0) * 1e6 = 256 * 1e6 = 2.56e8
    let expected = (cols as f32) * 1e6;
    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Row {}: output should be finite, got {}", i, val);
        assert!(
            (val - expected).abs() / expected < 1e-5,
            "Row {}: expected {:.2e}, got {:.2e}",
            i,
            expected,
            val
        );
    }
}

/// Test spec: i2s-dual-flavor.md#numerical-stability-small-inputs
///
/// Verify QK256 handles very small input values correctly
#[test]
fn test_qk256_numerical_stability_small_inputs() {
    let rows = 4;
    let cols = 256;
    let qs_data = vec![0xAAu8; rows * QK256_PACKED_BYTES]; // Code 2 → +1.0

    // Test with very small input values
    let small_input = vec![1e-6f32; cols];
    let mut output = vec![0.0f32; rows];

    gemv_qk256(&qs_data, &small_input, &mut output, rows, cols, QK256_PACKED_BYTES)
        .expect("Small input should succeed");

    // Expected: sum of 256 weights (+1.0) * 1e-6 = 2.56e-4
    let expected = (cols as f32) * 1e-6;
    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Row {}: output should be finite, got {}", i, val);
        assert!(
            (val - expected).abs() < 1e-9,
            "Row {}: expected {:.2e}, got {:.2e}",
            i,
            expected,
            val
        );
    }
}

// ==================== Thread Safety Tests ====================

/// Test spec: i2s-dual-flavor.md#thread-safety
///
/// Verify QK256 operations are thread-safe (can be called concurrently)
#[test]
fn test_qk256_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let rows = 16;
    let cols: usize = 512; // 2 blocks
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

    // Shared quantized data (Arc for thread safety)
    let qs_data = Arc::new(vec![0xAAu8; rows * row_stride_bytes]);
    let input = Arc::new(vec![1.0f32; cols]);

    // Spawn 4 threads, each computing GEMV independently
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let qs_clone = Arc::clone(&qs_data);
            let input_clone = Arc::clone(&input);

            thread::spawn(move || {
                let mut output = vec![0.0f32; rows];
                gemv_qk256(&qs_clone, &input_clone, &mut output, rows, cols, row_stride_bytes)
                    .expect("Thread GEMV should succeed");

                // Verify result
                let expected = cols as f32; // All weights +1.0, all inputs +1.0
                for (i, &val) in output.iter().enumerate() {
                    assert!(
                        (val - expected).abs() < 1e-3,
                        "Thread {}, Row {}: expected {}, got {}",
                        thread_id,
                        i,
                        expected,
                        val
                    );
                }
                output
            })
        })
        .collect();

    // Wait for all threads to complete
    for (i, handle) in handles.into_iter().enumerate() {
        handle.join().unwrap_or_else(|_| panic!("Thread {} should complete", i));
    }
}

// ==================== Unpack Block Edge Cases ====================

/// Test spec: i2s-dual-flavor.md#unpack-block-all-zeros
///
/// Verify unpack_qk256_block handles all-zero packed data
#[test]
fn test_unpack_qk256_block_all_zeros() {
    let packed = [0u8; QK256_PACKED_BYTES];
    let mut codes = [0u8; QK256_BLOCK];

    unpack_qk256_block(&packed, &mut codes);

    // All codes should be 0
    for (i, &code) in codes.iter().enumerate() {
        assert_eq!(code, 0, "Codes[{}] should be 0", i);
    }
}

/// Test spec: i2s-dual-flavor.md#unpack-block-all-ones
///
/// Verify unpack_qk256_block handles all-ones packed data (0xFF)
#[test]
fn test_unpack_qk256_block_all_ones() {
    let packed = [0xFFu8; QK256_PACKED_BYTES];
    let mut codes = [0u8; QK256_BLOCK];

    unpack_qk256_block(&packed, &mut codes);

    // All codes should be 3 (0xFF = 0b_11_11_11_11)
    for (i, &code) in codes.iter().enumerate() {
        assert_eq!(code, 3, "Codes[{}] should be 3", i);
    }
}

/// Test spec: i2s-dual-flavor.md#unpack-block-alternating-pattern
///
/// Verify unpack_qk256_block correctly unpacks alternating patterns
#[test]
fn test_unpack_qk256_block_alternating_pattern() {
    // Pattern: 0x96 = 0b10010110
    // Bit extraction order: [1:0], [3:2], [5:4], [7:6]
    // 0b10010110 unpacks to: [0b10, 0b01, 0b01, 0b10] = [2, 1, 1, 2]
    let packed = [0x96u8; QK256_PACKED_BYTES];
    let mut codes = [0u8; QK256_BLOCK];

    unpack_qk256_block(&packed, &mut codes);

    // Each byte unpacks to [2, 1, 1, 2] based on bit order
    for chunk_idx in 0..(QK256_BLOCK / 4) {
        assert_eq!(codes[chunk_idx * 4], 2, "Code {} should be 2", chunk_idx * 4);
        assert_eq!(codes[chunk_idx * 4 + 1], 1, "Code {} should be 1", chunk_idx * 4 + 1);
        assert_eq!(codes[chunk_idx * 4 + 2], 1, "Code {} should be 1", chunk_idx * 4 + 2);
        assert_eq!(codes[chunk_idx * 4 + 3], 2, "Code {} should be 2", chunk_idx * 4 + 3);
    }
}

// ==================== GEMV Row Edge Cases ====================

/// Test spec: i2s-dual-flavor.md#gemv-row-minimum-cols
///
/// Verify gemv_qk256_row handles minimum cols (1 element)
#[test]
fn test_gemv_qk256_row_minimum_cols() {
    let qs = [0xAAu8; QK256_PACKED_BYTES]; // Code 2 → +1.0
    let input = vec![5.0f32];
    let cols = 1;

    let result = gemv_qk256_row(&qs, &input, cols);

    // Expected: 1.0 * 5.0 = 5.0
    assert!((result - 5.0).abs() < 1e-5, "Expected 5.0, got {}", result);
}

/// Test spec: i2s-dual-flavor.md#gemv-row-exact-block
///
/// Verify gemv_qk256_row handles exact block size (256 elements)
#[test]
fn test_gemv_qk256_row_exact_block() {
    let qs = [0x55u8; QK256_PACKED_BYTES]; // Code 1 → -1.0
    let input = vec![2.0f32; QK256_BLOCK];
    let cols = QK256_BLOCK;

    let result = gemv_qk256_row(&qs, &input, cols);

    // Expected: 256 * (-1.0) * 2.0 = -512.0
    let expected = -(QK256_BLOCK as f32) * 2.0;
    assert!((result - expected).abs() < 1e-3, "Expected {}, got {}", expected, result);
}
