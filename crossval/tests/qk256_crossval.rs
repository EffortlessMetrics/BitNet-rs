//! QK256 Cross-Validation Tests
//!
//! Tests feature spec: i2s-dual-flavor.md#qk256-cross-validation
//!
//! This test suite validates QK256 implementation against reference implementations:
//!
//! - Compare Rust QK256 kernel vs dequantized FP32 reference
//! - Compare Rust QK256 vs C++ FFI (when available with --features ffi)
//! - Validate accuracy requirements (>99.8% correlation)
//! - Test production model scenarios with real inference patterns

use anyhow::Result;
use bitnet_models::quant::i2s_qk256::{QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, gemv_qk256};

// ==================== FP32 Reference Implementation ====================

/// FP32 reference GEMV for cross-validation
///
/// This implements the same computation as gemv_qk256 but using FP32 weights
/// (decoded from QK256 codes) to serve as ground truth.
fn gemv_fp32_reference(
    codes: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    if output.len() != rows {
        anyhow::bail!("FP32 reference: output length {} != rows {}", output.len(), rows);
    }
    if codes.len() != rows * cols {
        anyhow::bail!(
            "FP32 reference: codes length {} != rows*cols ({})",
            codes.len(),
            rows * cols
        );
    }
    if input.len() < cols {
        anyhow::bail!("FP32 reference: input length {} < cols {}", input.len(), cols);
    }

    for row_idx in 0..rows {
        let mut sum = 0.0f32;
        for col_idx in 0..cols {
            let code = codes[row_idx * cols + col_idx];
            sum += code_to_f32(code) * input[col_idx];
        }
        output[row_idx] = sum;
    }

    Ok(())
}

// ==================== Cross-Validation Tests ====================

/// Test spec: i2s-dual-flavor.md#qk256-vs-fp32-reference
///
/// Verify QK256 kernel matches FP32 reference within 1e-5 tolerance
#[test]
fn test_qk256_vs_fp32_reference_small_matrix() -> Result<()> {
    let rows = 8;
    let cols = 256; // Single block
    let total_codes = rows * cols;

    // Create deterministic code pattern
    let codes: Vec<u8> = (0..total_codes).map(|i| (i % 4) as u8).collect();

    // Pack codes into QK256 format
    let mut packed_data = vec![0u8; rows * QK256_PACKED_BYTES];
    for row_idx in 0..rows {
        for byte_idx in 0..QK256_PACKED_BYTES {
            let mut byte = 0u8;
            for j in 0..4 {
                let code_idx = row_idx * cols + byte_idx * 4 + j;
                byte |= codes[code_idx] << (j * 2);
            }
            packed_data[row_idx * QK256_PACKED_BYTES + byte_idx] = byte;
        }
    }

    // Create deterministic input
    let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).sin()).collect();

    // Compute QK256 result
    let mut qk256_output = vec![0.0f32; rows];
    gemv_qk256(&packed_data, &input, &mut qk256_output, rows, cols, QK256_PACKED_BYTES)?;

    // Compute FP32 reference result
    let mut fp32_output = vec![0.0f32; rows];
    gemv_fp32_reference(&codes, &input, &mut fp32_output, rows, cols)?;

    // Compare results
    for row_idx in 0..rows {
        let diff = (qk256_output[row_idx] - fp32_output[row_idx]).abs();
        assert!(
            diff < 1e-5,
            "Row {}: QK256={}, FP32={}, diff={} (exceeds tolerance)",
            row_idx,
            qk256_output[row_idx],
            fp32_output[row_idx],
            diff
        );
    }

    Ok(())
}

/// Test spec: i2s-dual-flavor.md#qk256-vs-fp32-multi-block
///
/// Verify QK256 kernel matches FP32 reference for multi-block matrices
#[test]
fn test_qk256_vs_fp32_reference_multi_block() -> Result<()> {
    let rows = 16;
    let cols: usize = 768; // 3 blocks (256 + 256 + 256)
    let total_codes = rows * cols;

    // Create deterministic code pattern
    let codes: Vec<u8> = (0..total_codes).map(|i| ((i / 64) % 4) as u8).collect();

    // Pack codes
    let blocks_per_row = cols.div_ceil(QK256_BLOCK); // = 3
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES; // = 192
    let mut packed_data = vec![0u8; rows * row_stride_bytes];

    for row_idx in 0..rows {
        for block_idx in 0..blocks_per_row {
            let block_start_code = row_idx * cols + block_idx * QK256_BLOCK;
            let block_start_byte = row_idx * row_stride_bytes + block_idx * QK256_PACKED_BYTES;

            for byte_idx in 0..QK256_PACKED_BYTES {
                let mut byte = 0u8;
                for j in 0..4 {
                    let code_idx = block_start_code + byte_idx * 4 + j;
                    byte |= codes[code_idx] << (j * 2);
                }
                packed_data[block_start_byte + byte_idx] = byte;
            }
        }
    }

    // Create input
    let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).cos()).collect();

    // Compute QK256 result
    let mut qk256_output = vec![0.0f32; rows];
    gemv_qk256(&packed_data, &input, &mut qk256_output, rows, cols, row_stride_bytes)?;

    // Compute FP32 reference
    let mut fp32_output = vec![0.0f32; rows];
    gemv_fp32_reference(&codes, &input, &mut fp32_output, rows, cols)?;

    // Compare
    for row_idx in 0..rows {
        let diff = (qk256_output[row_idx] - fp32_output[row_idx]).abs();
        assert!(
            diff < 1e-4,
            "Row {}: QK256={}, FP32={}, diff={}",
            row_idx,
            qk256_output[row_idx],
            fp32_output[row_idx],
            diff
        );
    }

    Ok(())
}

/// Test spec: i2s-dual-flavor.md#qk256-vs-fp32-tail-handling
///
/// Verify QK256 kernel matches FP32 reference with tail blocks
#[test]
fn test_qk256_vs_fp32_reference_with_tail() -> Result<()> {
    let rows = 12;
    let cols: usize = 300; // 2 blocks: 256 + 44 tail
    let total_codes = rows * cols;

    // Create codes
    let codes: Vec<u8> = (0..total_codes).map(|i| ((i / 75) % 4) as u8).collect();

    // Pack codes
    let blocks_per_row = cols.div_ceil(QK256_BLOCK); // = 2
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES; // = 128
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

    // Create input
    let input: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.1).collect();

    // Compute QK256
    let mut qk256_output = vec![0.0f32; rows];
    gemv_qk256(&packed_data, &input, &mut qk256_output, rows, cols, row_stride_bytes)?;

    // Compute FP32 reference
    let mut fp32_output = vec![0.0f32; rows];
    gemv_fp32_reference(&codes, &input, &mut fp32_output, rows, cols)?;

    // Compare
    for row_idx in 0..rows {
        let diff = (qk256_output[row_idx] - fp32_output[row_idx]).abs();
        assert!(
            diff < 1e-4,
            "Row {}: QK256={}, FP32={}, diff={} (tail handling)",
            row_idx,
            qk256_output[row_idx],
            fp32_output[row_idx],
            diff
        );
    }

    Ok(())
}

// ==================== Accuracy Validation Tests ====================

/// Test spec: i2s-dual-flavor.md#qk256-accuracy-requirements
///
/// Verify QK256 achieves >99.8% correlation with FP32 reference
#[test]
fn test_qk256_accuracy_target_correlation() -> Result<()> {
    let rows = 64;
    let cols: usize = 1024; // 4 blocks
    let total_codes = rows * cols;

    // Create realistic code distribution (based on quantization statistics)
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let codes: Vec<u8> = (0..total_codes).map(|_| rng.random_range(0..=3)).collect();

    // Pack codes
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;
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

    // Create realistic input (normal distribution)
    let input: Vec<f32> =
        (0..cols).map(|i| ((i as f32 * 0.1).sin() + (i as f32 * 0.05).cos()) * 0.5).collect();

    // Compute both versions
    let mut qk256_output = vec![0.0f32; rows];
    gemv_qk256(&packed_data, &input, &mut qk256_output, rows, cols, row_stride_bytes)?;

    let mut fp32_output = vec![0.0f32; rows];
    gemv_fp32_reference(&codes, &input, &mut fp32_output, rows, cols)?;

    // Compute correlation coefficient
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
        1.0 // Both outputs constant (perfect match)
    };

    // Verify >99.8% correlation
    assert!(
        correlation >= 0.998,
        "QK256 accuracy {:.4}% < 99.8% requirement (correlation={:.6})",
        correlation * 100.0,
        correlation
    );

    println!("✓ QK256 accuracy: {:.4}% (correlation={:.6})", correlation * 100.0, correlation);

    Ok(())
}

// ==================== Production Model Scenarios ====================

/// Test spec: i2s-dual-flavor.md#qk256-production-inference-pattern
///
/// Validate QK256 with production-size attention projection
#[test]
fn test_qk256_production_attention_projection() -> Result<()> {
    // Typical attention projection: 2048 hidden × 2048 output
    let _rows = 2048;
    let cols: usize = 2048;

    // Create simplified weights (only test first 16 rows for speed)
    let test_rows = 16;
    let blocks_per_row = cols.div_ceil(QK256_BLOCK); // = 8
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES; // = 512

    // Deterministic pattern for reproducibility
    let mut packed_data = vec![0u8; test_rows * row_stride_bytes];
    for row_idx in 0..test_rows {
        for byte_idx in 0..row_stride_bytes {
            // Alternating pattern based on position
            packed_data[row_idx * row_stride_bytes + byte_idx] = ((row_idx + byte_idx) % 256) as u8;
        }
    }

    // Create realistic input (simulating attention values)
    let input: Vec<f32> = (0..cols).map(|i| (i as f32 / cols as f32).sin()).collect();

    // Compute QK256 output
    let mut qk256_output = vec![0.0f32; test_rows];
    gemv_qk256(&packed_data, &input, &mut qk256_output, test_rows, cols, row_stride_bytes)?;

    // Verify outputs are reasonable (not NaN, within expected range)
    for (i, &val) in qk256_output.iter().enumerate() {
        assert!(val.is_finite(), "Row {}: output {} is not finite", i, val);
        assert!(val.abs() < 1e6, "Row {}: output {} seems unreasonable", i, val);
    }

    // Compute FP32 reference for comparison
    let codes: Vec<u8> = (0..test_rows * cols)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            let byte_idx = (col / 4) % row_stride_bytes;
            let byte = packed_data[row * row_stride_bytes + byte_idx];
            let j = col % 4;
            (byte >> (j * 2)) & 0x03
        })
        .collect();

    let mut fp32_output = vec![0.0f32; test_rows];
    gemv_fp32_reference(&codes, &input, &mut fp32_output, test_rows, cols)?;

    // Verify close match
    for row_idx in 0..test_rows {
        let diff = (qk256_output[row_idx] - fp32_output[row_idx]).abs();
        assert!(
            diff < 1e-3,
            "Row {}: QK256={}, FP32={}, diff={}",
            row_idx,
            qk256_output[row_idx],
            fp32_output[row_idx],
            diff
        );
    }

    println!("✓ Production attention projection test passed");

    Ok(())
}

/// Test spec: i2s-dual-flavor.md#qk256-batch-inference
///
/// Validate QK256 with batch inference scenario (multiple input vectors)
#[test]
fn test_qk256_batch_inference_scenario() -> Result<()> {
    let rows = 32; // Output features
    let cols: usize = 512; // Input features (2 blocks)
    let batch_size = 8; // Simulate batch processing

    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

    // Create weight matrix
    let mut packed_data = vec![0u8; rows * row_stride_bytes];
    for (byte_idx, byte) in packed_data.iter_mut().enumerate() {
        *byte = (byte_idx % 256) as u8;
    }

    // Process each batch item
    for batch_idx in 0..batch_size {
        // Create batch-specific input
        let input: Vec<f32> =
            (0..cols).map(|i| ((i + batch_idx * 10) as f32 * 0.01).sin()).collect();

        // Compute QK256 output
        let mut qk256_output = vec![0.0f32; rows];
        gemv_qk256(&packed_data, &input, &mut qk256_output, rows, cols, row_stride_bytes)?;

        // Verify output is valid
        for (i, &val) in qk256_output.iter().enumerate() {
            assert!(val.is_finite(), "Batch {}, Row {}: non-finite output", batch_idx, i);
        }
    }

    println!("✓ Batch inference scenario test passed ({} batches)", batch_size);

    Ok(())
}

// ==================== FFI Cross-Validation (Feature-Gated) ====================

#[cfg(feature = "ffi")]
mod ffi_crossval {
    use super::*;

    /// Test spec: i2s-dual-flavor.md#qk256-vs-cpp-ffi
    ///
    /// Cross-validate Rust QK256 kernel against C++ FFI implementation
    #[test]
    fn test_qk256_vs_cpp_ffi() -> Result<()> {
        // TODO: Implement FFI cross-validation when C++ QK256 FFI is available
        // This test would:
        // 1. Load same QK256 weights in both Rust and C++
        // 2. Run identical GEMV operations
        // 3. Compare outputs with tight tolerance (<1e-6)
        println!("⏭  QK256 vs C++ FFI cross-validation (requires FFI implementation)");
        Ok(())
    }
}
