//! Property-based tests for QK256 kernel correctness
//!
//! Tests feature spec: i2s-dual-flavor.md#qk256-kernel-correctness
//!
//! This test suite uses property-based testing (proptest) to validate QK256 kernel
//! implementation against reference FP32 computation across random inputs.
//!
//! ## Coverage
//!
//! - Random matrix dimensions (rows, cols with various QK256 block alignments)
//! - Random code patterns (0..=3 mapping to -2.0, -1.0, 1.0, 2.0)
//! - Random input vectors with various distributions
//! - Tail handling (cols not multiple of 256)
//! - Numerical accuracy validation against FP32 reference

use bitnet_models::quant::i2s_qk256::{
    I2SQk256NoScale, QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, gemv_qk256, gemv_qk256_row,
    unpack_qk256_block,
};
use proptest::prelude::*;

// Import tolerance helpers from test helpers
mod helpers;
use helpers::qk256_tolerance::approx_eq_with_len;

// ==================== Property Test Strategies ====================

/// Strategy for generating valid QK256 dimensions
fn qk256_dimensions() -> impl Strategy<Value = (usize, usize)> {
    prop::sample::select(vec![
        (1, 256),    // Single block, single row
        (4, 256),    // Single block, multiple rows
        (8, 512),    // Multiple blocks aligned
        (16, 300),   // Multiple blocks with tail
        (32, 768),   // Large matrix aligned
        (64, 1000),  // Large matrix with tail
        (128, 2048), // Production-size matrix
        (2048, 256), // Wide matrix
        (256, 2048), // Tall matrix
    ])
}

/// Strategy for generating random QK256 codes (0..=3)
fn qk256_codes(len: usize) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(0u8..=3, len)
}

/// Strategy for generating random input vectors
fn random_input_vector(len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0, len)
}

// ==================== Code Mapping Property Tests ====================

/// Test spec: i2s-dual-flavor.md#code-to-f32-lut-verification
///
/// Verify code_to_f32 LUT values match GGML reference: {-2, -1, 1, 2}
#[test]
fn test_code_to_f32_lut_values() {
    // Verified against GGML ggml-quants.c:62
    assert_eq!(code_to_f32(0), -2.0);
    assert_eq!(code_to_f32(1), -1.0);
    assert_eq!(code_to_f32(2), 1.0);
    assert_eq!(code_to_f32(3), 2.0);
}

proptest! {
    /// Test spec: i2s-dual-flavor.md#code-mapping-consistency
    ///
    /// Property: All codes 0..=3 map to valid float values
    #[test]
    fn prop_code_to_f32_range(code in 0u8..=3) {
        let value = code_to_f32(code);
        assert!(value.is_finite(), "Code {} maps to non-finite value", code);
        assert!(value.abs() <= 2.0, "Code {} maps to out-of-range value {}", code, value);
    }
}

// ==================== Unpack Block Property Tests ====================

proptest! {
    /// Test spec: i2s-dual-flavor.md#unpack-block-correctness
    ///
    /// Property: Unpacking 64 bytes yields exactly 256 valid codes (0..=3)
    #[test]
    fn prop_unpack_qk256_block_valid_codes(packed_bytes in prop::collection::vec(any::<u8>(), QK256_PACKED_BYTES)) {
        let mut codes = [0u8; QK256_BLOCK];
        let packed_arr: [u8; QK256_PACKED_BYTES] = packed_bytes.try_into().unwrap();

        unpack_qk256_block(&packed_arr, &mut codes);

        // Verify all codes are valid (0..=3)
        for (i, &code) in codes.iter().enumerate() {
            assert!(code <= 3, "Unpacked code[{}] = {} exceeds valid range 0..=3", i, code);
        }

        // Verify exactly 256 codes
        assert_eq!(codes.len(), QK256_BLOCK);
    }

    /// Test spec: i2s-dual-flavor.md#unpack-block-determinism
    ///
    /// Property: Unpacking is deterministic (same input yields same output)
    #[test]
    fn prop_unpack_qk256_block_deterministic(packed_bytes in prop::collection::vec(any::<u8>(), QK256_PACKED_BYTES)) {
        let packed_arr: [u8; QK256_PACKED_BYTES] = packed_bytes.try_into().unwrap();

        let mut codes1 = [0u8; QK256_BLOCK];
        let mut codes2 = [0u8; QK256_BLOCK];

        unpack_qk256_block(&packed_arr, &mut codes1);
        unpack_qk256_block(&packed_arr, &mut codes2);

        assert_eq!(codes1, codes2, "Unpacking should be deterministic");
    }
}

// ==================== GEMV Row Property Tests ====================

proptest! {
    /// Test spec: i2s-dual-flavor.md#gemv-row-vs-reference
    ///
    /// Property: gemv_qk256_row matches FP32 reference within quantization tolerance
    #[test]
    fn prop_gemv_qk256_row_matches_reference(
        codes in qk256_codes(QK256_BLOCK),
        input in random_input_vector(QK256_BLOCK),
    ) {
        // Pack codes into bytes (4 codes per byte)
        let mut packed = [0u8; QK256_PACKED_BYTES];
        for (i, chunk) in codes.chunks(4).enumerate() {
            let mut byte = 0u8;
            for (j, &code) in chunk.iter().enumerate() {
                byte |= code << (j * 2);
            }
            packed[i] = byte;
        }

        // Compute QK256 result
        let qk256_result = gemv_qk256_row(&packed, &input, QK256_BLOCK);

        // Compute FP32 reference: dot product of decoded weights and input
        let fp32_result: f32 = codes.iter()
            .zip(input.iter())
            .map(|(&code, &x)| code_to_f32(code) * x)
            .sum();

        // Verify within tolerance using adaptive tolerance helper
        assert!(
            approx_eq_with_len(qk256_result, fp32_result, QK256_BLOCK),
            "QK256 result {} differs from FP32 reference {} by {} (exceeds tolerance for len={})",
            qk256_result, fp32_result, (qk256_result - fp32_result).abs(), QK256_BLOCK
        );
    }

    /// Test spec: i2s-dual-flavor.md#gemv-row-tail-handling
    ///
    /// Property: gemv_qk256_row handles cols < 256 correctly
    #[test]
    fn prop_gemv_qk256_row_tail_handling(
        codes in qk256_codes(QK256_BLOCK),
        input in random_input_vector(QK256_BLOCK),
        cols in 1usize..=255, // Tail only (< QK256_BLOCK)
    ) {
        // Pack codes
        let mut packed = [0u8; QK256_PACKED_BYTES];
        for (i, chunk) in codes.chunks(4).enumerate() {
            let mut byte = 0u8;
            for (j, &code) in chunk.iter().enumerate() {
                byte |= code << (j * 2);
            }
            packed[i] = byte;
        }

        // Compute QK256 result with limited cols
        let qk256_result = gemv_qk256_row(&packed, &input, cols);

        // Compute FP32 reference (only first `cols` elements)
        let fp32_result: f32 = codes.iter()
            .take(cols)
            .zip(input.iter().take(cols))
            .map(|(&code, &x)| code_to_f32(code) * x)
            .sum();

        // Verify within tolerance for tail handling
        assert!(
            approx_eq_with_len(qk256_result, fp32_result, cols),
            "QK256 tail result {} differs from FP32 reference {} by {} (exceeds tolerance for len={})",
            qk256_result, fp32_result, (qk256_result - fp32_result).abs(), cols
        );
    }
}

// ==================== GEMV Multi-Row Property Tests ====================

proptest! {
    /// Test spec: i2s-dual-flavor.md#gemv-multi-row-correctness
    ///
    /// Property: gemv_qk256 matches FP32 reference for multi-row matrices
    #[test]
    fn prop_gemv_qk256_matches_fp32_reference(
        (rows, cols) in qk256_dimensions(),
        seed in any::<u64>(),
    ) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate random codes for each row
        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;
        let total_codes = rows * cols;

        let codes: Vec<u8> = (0..total_codes).map(|_| rng.random_range(0..=3)).collect();

        // Pack codes into bytes
        let mut packed_data = vec![0u8; rows * row_stride_bytes];
        for row_idx in 0..rows {
            let row_start_code = row_idx * cols;
            let row_start_byte = row_idx * row_stride_bytes;

            for block_idx in 0..blocks_per_row {
                let block_start_code = row_start_code + block_idx * QK256_BLOCK;
                let block_start_byte = row_start_byte + block_idx * QK256_PACKED_BYTES;

                let codes_in_block = QK256_BLOCK.min(cols - block_idx * QK256_BLOCK);

                for byte_idx in 0..QK256_PACKED_BYTES {
                    let mut byte = 0u8;
                    for j in 0..4 {
                        let code_idx = block_start_code + byte_idx * 4 + j;
                        if byte_idx * 4 + j < codes_in_block {
                            let code = codes[code_idx];
                            byte |= code << (j * 2);
                        }
                    }
                    packed_data[block_start_byte + byte_idx] = byte;
                }
            }
        }

        // Generate random input vector
        let input: Vec<f32> = (0..cols).map(|_| rng.random_range(-10.0..10.0)).collect();

        // Compute QK256 result
        let mut qk256_output = vec![0.0f32; rows];
        gemv_qk256(&packed_data, &input, &mut qk256_output, rows, cols, row_stride_bytes)
            .expect("gemv_qk256 should succeed");

        // Compute FP32 reference
        let mut fp32_output = vec![0.0f32; rows];
        for row_idx in 0..rows {
            let mut sum = 0.0f32;
            for col_idx in 0..cols {
                let code = codes[row_idx * cols + col_idx];
                sum += code_to_f32(code) * input[col_idx];
            }
            fp32_output[row_idx] = sum;
        }

        // Verify results match within adaptive tolerance
        for row_idx in 0..rows {
            let qk256_val = qk256_output[row_idx];
            let fp32_val = fp32_output[row_idx];
            let abs_diff = (qk256_val - fp32_val).abs();

            assert!(
                approx_eq_with_len(qk256_val, fp32_val, cols),
                "Row {}: QK256={}, FP32={}, abs_diff={} (exceeds tolerance for cols={})\n\
                 Relative diff: {:.2e}",
                row_idx, qk256_val, fp32_val, abs_diff, cols,
                if fp32_val.abs() > 1e-6 { abs_diff / fp32_val.abs() } else { f32::NAN }
            );
        }
    }
}

// ==================== I2SQk256NoScale Property Tests ====================

proptest! {
    /// Test spec: i2s-dual-flavor.md#qk256-struct-validation
    ///
    /// Property: I2SQk256NoScale::new validates dimensions correctly
    #[test]
    fn prop_i2s_qk256_no_scale_dimension_validation(
        rows in 1usize..=256,
        cols in 1usize..=2048,
    ) {
        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;
        let expected_bytes = rows * row_stride_bytes;

        // Test 1: Valid creation with correct size
        let qs_valid = vec![0u8; expected_bytes];
        let result = I2SQk256NoScale::new(rows, cols, qs_valid);
        assert!(result.is_ok(), "Valid dimensions should succeed");

        let qk256 = result.unwrap();
        assert_eq!(qk256.rows, rows);
        assert_eq!(qk256.cols, cols);
        assert_eq!(qk256.row_stride_bytes, row_stride_bytes);

        // Test 2: Invalid size (too small)
        if expected_bytes > 0 {
            let qs_invalid = vec![0u8; expected_bytes - 1];
            let result = I2SQk256NoScale::new(rows, cols, qs_invalid);
            assert!(result.is_err(), "Invalid size should fail");
        }

        // Test 3: Invalid size (too large)
        let qs_invalid = vec![0u8; expected_bytes + 1];
        let result = I2SQk256NoScale::new(rows, cols, qs_invalid);
        assert!(result.is_err(), "Invalid size should fail");
    }

    /// Test spec: i2s-dual-flavor.md#qk256-row-bytes-access
    ///
    /// Property: row_bytes returns correct slice for each row
    #[test]
    fn prop_i2s_qk256_no_scale_row_bytes(
        rows in 1usize..=64,
        cols in 256usize..=1024,
    ) {
        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

        // Create QK256 struct with distinct bytes per row
        let mut qs = Vec::new();
        for row_idx in 0..rows {
            let row_pattern = (row_idx % 256) as u8;
            qs.extend(vec![row_pattern; row_stride_bytes]);
        }

        let qk256 = I2SQk256NoScale::new(rows, cols, qs).expect("Valid creation");

        // Verify each row returns correct slice
        for row_idx in 0..rows {
            let row_bytes = qk256.row_bytes(row_idx);
            assert_eq!(row_bytes.len(), row_stride_bytes);

            let expected_pattern = (row_idx % 256) as u8;
            assert!(
                row_bytes.iter().all(|&b| b == expected_pattern),
                "Row {} should have pattern {}", row_idx, expected_pattern
            );
        }
    }
}

// ==================== Error Handling Property Tests ====================

#[test]
fn test_gemv_qk256_error_handling() {
    // Test spec: i2s-dual-flavor.md#gemv-error-cases

    let rows = 4;
    let cols = 256;
    let row_stride_bytes = QK256_PACKED_BYTES;
    let qs_data = vec![0xAAu8; rows * row_stride_bytes];

    // Test 1: Mismatched output size
    let input = vec![1.0f32; cols];
    let mut output = vec![0.0f32; rows + 1]; // Wrong size

    let result = gemv_qk256(&qs_data, &input, &mut output, rows, cols, row_stride_bytes);
    assert!(result.is_err(), "Should reject mismatched output size");
    assert!(result.unwrap_err().to_string().contains("y_out length"));

    // Test 2: Insufficient input data
    let short_input = vec![1.0f32; cols - 10];
    let mut output = vec![0.0f32; rows];

    let result = gemv_qk256(&qs_data, &short_input, &mut output, rows, cols, row_stride_bytes);
    assert!(result.is_err(), "Should reject insufficient input");
    assert!(result.unwrap_err().to_string().contains("x length"));

    // Test 3: Insufficient quantized data
    let short_qs = vec![0xAAu8; row_stride_bytes]; // Only 1 row
    let input = vec![1.0f32; cols];
    let mut output = vec![0.0f32; rows];

    let result = gemv_qk256(&short_qs, &input, &mut output, rows, cols, row_stride_bytes);
    assert!(result.is_err(), "Should reject insufficient quantized data");
    assert!(result.unwrap_err().to_string().contains("data too short"));
}

// ==================== Numerical Accuracy Property Tests ====================

proptest! {
    /// Test spec: i2s-dual-flavor.md#qk256-accuracy-requirements
    ///
    /// Property: QK256 maintains >99.8% accuracy vs FP32 reference
    #[test]
    fn prop_qk256_accuracy_target(
        (rows, cols) in qk256_dimensions(),
        seed in any::<u64>(),
    ) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate random matrix and input
        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

        // Create random codes
        let codes: Vec<u8> = (0..rows * cols).map(|_| rng.random_range(0..=3)).collect();

        // Pack codes
        let mut packed_data = vec![0u8; rows * row_stride_bytes];
        for row_idx in 0..rows {
            for block_idx in 0..blocks_per_row {
                let block_start_code = row_idx * cols + block_idx * QK256_BLOCK;
                let block_start_byte = row_idx * row_stride_bytes + block_idx * QK256_PACKED_BYTES;

                let codes_in_block = QK256_BLOCK.min(cols - block_idx * QK256_BLOCK);

                for byte_idx in 0..QK256_PACKED_BYTES {
                    let mut byte = 0u8;
                    for j in 0..4 {
                        if byte_idx * 4 + j < codes_in_block {
                            let code_idx = block_start_code + byte_idx * 4 + j;
                            byte |= codes[code_idx] << (j * 2);
                        }
                    }
                    packed_data[block_start_byte + byte_idx] = byte;
                }
            }
        }

        // Generate random input
        let input: Vec<f32> = (0..cols).map(|_| rng.random_range(-5.0..5.0)).collect();

        // Compute both versions
        let mut qk256_output = vec![0.0f32; rows];
        gemv_qk256(&packed_data, &input, &mut qk256_output, rows, cols, row_stride_bytes)
            .expect("gemv_qk256 should succeed");

        let mut fp32_output = vec![0.0f32; rows];
        for (row_idx, output_val) in fp32_output.iter_mut().enumerate().take(rows) {
            *output_val = codes.iter()
                .skip(row_idx * cols)
                .take(cols)
                .zip(&input)
                .map(|(&code, &x)| code_to_f32(code) * x)
                .sum();
        }

        // Compute correlation (accuracy metric)
        let qk_mean: f32 = qk256_output.iter().sum::<f32>() / rows as f32;
        let fp_mean: f32 = fp32_output.iter().sum::<f32>() / rows as f32;

        let mut cov = 0.0f32;
        let mut qk_var = 0.0f32;
        let mut fp_var = 0.0f32;

        for i in 0..rows {
            let qk_diff = qk256_output[i] - qk_mean;
            let fp_diff = fp32_output[i] - fp_mean;
            cov += qk_diff * fp_diff;
            qk_var += qk_diff * qk_diff;
            fp_var += fp_diff * fp_diff;
        }

        let correlation = if qk_var > 1e-9 && fp_var > 1e-9 {
            cov / (qk_var.sqrt() * fp_var.sqrt())
        } else {
            1.0 // Both outputs are constant (perfect match)
        };

        // Verify >99.8% accuracy (correlation >= 0.998)
        assert!(
            correlation >= 0.998,
            "QK256 accuracy {:.4}% < 99.8% requirement (rows={}, cols={})",
            correlation * 100.0, rows, cols
        );
    }
}
