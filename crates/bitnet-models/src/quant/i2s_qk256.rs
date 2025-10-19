//! GGML I2_S (QK=256) scalar reference kernels
//!
//! This module implements pure-Rust dequantization and GEMV for GGML's I2_S format:
//! - Block size: 256 elements
//! - Packed format: 64 bytes per block (2 bits/element, no embedded scales)
//! - Code mapping: **VERIFIED** against GGML reference (ggml-quants.c:62)
//!
//! ## Memory Layout
//!
//! Each block contains 256 elements packed into 64 bytes:
//! ```text
//! [byte 0: elem 0..3] [byte 1: elem 4..7] ... [byte 63: elem 252..255]
//! ```
//!
//! Each byte packs 4 elements (2 bits each):
//! ```text
//! byte = elem0 | (elem1 << 2) | (elem2 << 4) | (elem3 << 6)
//! ```
//!
//! ## Code Mapping (VERIFIED)
//!
//! The 2-bit codes map to signed weights according to GGML's IQ2_S specification
//! (verified in `crates/bitnet-ggml-ffi/csrc/ggml/src/ggml-quants.c:62`):
//!
//! - Code 0 → -2.0
//! - Code 1 → -1.0
//! - Code 2 → +1.0
//! - Code 3 → +2.0
//!
//! **Format variants:**
//! - **GgmlQk256NoScale** (MS BitNet): No per-block scale, use LUT values directly
//! - **Full GGML IQ2_S** (82B/block): Multiply LUT values by per-block FP16 scale `d`
//!
//! This implementation supports the "no-scale" variant used by MS BitNet GGUF models.

use anyhow::{Result, bail};

/// Block size for GGML I2_S format
pub const QK256_BLOCK: usize = 256;

/// Packed bytes per block (2 bits/elem * 256 elem / 8 bits/byte)
pub const QK256_PACKED_BYTES: usize = 64;

/// Storage for GGML I2_S (QK=256) quantized weights without per-block scales
///
/// This structure holds raw packed 2-bit codes for a weight tensor in the
/// "GgmlQk256NoScale" format used by MS BitNet GGUF models. The data is stored
/// in row-major order without dequantization.
///
/// # Memory Layout
///
/// - `rows`: Number of rows in the weight matrix
/// - `cols`: Number of columns in the weight matrix
/// - `row_stride_bytes`: Bytes per row = ceil(cols/256) * 64
/// - `qs`: Contiguous packed bytes (rows * row_stride_bytes total)
///
/// # Example
///
/// For a 512×1024 weight matrix:
/// - `rows = 512`
/// - `cols = 1024`
/// - `blocks_per_row = ceil(1024/256) = 4`
/// - `row_stride_bytes = 4 * 64 = 256 bytes`
/// - `qs.len() = 512 * 256 = 131,072 bytes`
#[derive(Clone, Debug)]
pub struct I2SQk256NoScale {
    pub rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub qs: Vec<u8>,
}

impl I2SQk256NoScale {
    /// Create a new QK256 quantized tensor
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `qs` - Packed quantized data (must be exactly rows * row_stride_bytes)
    ///
    /// # Returns
    ///
    /// `Result<Self>` - The quantized tensor or error if dimensions don't match
    pub fn new(rows: usize, cols: usize, qs: Vec<u8>) -> Result<Self> {
        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;
        let expected_bytes = rows * row_stride_bytes;

        // Allow for alignment padding (e.g., 32 bytes for cache line alignment)
        const TOLERANCE: usize = 128;
        let size_diff = qs.len().abs_diff(expected_bytes);

        if size_diff > TOLERANCE {
            bail!(
                "I2SQk256NoScale: data size mismatch: got {} bytes, expected {} for {}×{} matrix. \
                 Check tensor orientation: QK256 requires [out_dim, in_dim] layout.",
                qs.len(),
                expected_bytes,
                rows,
                cols
            );
        }

        Ok(Self { rows, cols, row_stride_bytes, qs })
    }

    /// Get a slice of bytes for a specific row
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0..rows)
    ///
    /// # Returns
    ///
    /// Slice of packed bytes for the row
    ///
    /// # Panics
    ///
    /// Panics if row index is out of bounds (debug builds only).
    #[inline]
    pub fn row_bytes(&self, row: usize) -> &[u8] {
        debug_assert!(row < self.rows, "I2SQk256NoScale: row {} >= rows {}", row, self.rows);
        let start = row * self.row_stride_bytes;
        let end = start + self.row_stride_bytes;
        &self.qs[start..end]
    }
}

/// Code-to-float lookup table
///
/// **VERIFIED**: This mapping matches GGML's IQ2_S dequantization (ggml-quants.c:62).
/// Reference: `const float qmap[4] = { -2.f, -1.f, 1.f, 2.f };`
///
/// For MS BitNet "GgmlQk256NoScale" format, these values are used directly
/// (no per-block scale). For full GGML IQ2_S format (82B/block with FP16 scale),
/// these would be multiplied by the scale factor.
#[inline]
pub fn code_to_f32(code: u8) -> f32 {
    // SAFETY: code is masked to 0..=3 by caller
    debug_assert!(code < 4, "I2S_QK256: code must be 0..=3, got {}", code);

    // Verified against GGML reference (crates/bitnet-ggml-ffi/csrc/ggml/src/ggml-quants.c:62)
    const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];
    LUT[code as usize]
}

/// Unpack one 64-byte block of 2-bit codes (QK=256) into 256 u8 codes (0..=3)
///
/// # Arguments
///
/// * `qs64` - Input packed block (64 bytes)
/// * `out_codes256` - Output codes array (256 elements)
///
/// # Panics
///
/// Panics if slice lengths don't match expected sizes (debug builds only).
#[inline]
pub fn unpack_qk256_block(qs64: &[u8; QK256_PACKED_BYTES], out_codes256: &mut [u8; QK256_BLOCK]) {
    // Each byte contains 4 codes: bits [1:0], [3:2], [5:4], [7:6]
    for (i, &b) in qs64.iter().enumerate() {
        let base = i * 4;
        out_codes256[base] = b & 0x03;
        out_codes256[base + 1] = (b >> 2) & 0x03;
        out_codes256[base + 2] = (b >> 4) & 0x03;
        out_codes256[base + 3] = (b >> 6) & 0x03;
    }
}

/// Compute RMS (root mean square) of a slice
#[inline]
fn compute_rms(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = xs.iter().map(|x| x * x).sum();
    (sum_sq / (xs.len() as f32)).sqrt()
}

/// Compute dot product between one quantized QK256 row and a dense input vector
///
/// # Arguments
///
/// * `qs_row` - Row-major packed bytes (N * 64 bytes, where N = ceil(cols/256))
/// * `x` - Dense input vector (length = cols)
/// * `cols` - Number of columns (may not be multiple of 256)
///
/// # Returns
///
/// Scalar dot product result
///
/// # Panics
///
/// Panics if `qs_row` length doesn't match expected packing or if `x` is shorter than `cols`.
#[inline]
pub fn gemv_qk256_row(qs_row: &[u8], x: &[f32], cols: usize) -> f32 {
    let blocks_needed = cols.div_ceil(QK256_BLOCK);
    let expected_bytes = blocks_needed * QK256_PACKED_BYTES;

    debug_assert_eq!(
        qs_row.len(),
        expected_bytes,
        "I2S_QK256: row bytes mismatch: got {}, expected {} for {} cols",
        qs_row.len(),
        expected_bytes,
        cols
    );
    debug_assert!(x.len() >= cols, "I2S_QK256: x too short: {} < {}", x.len(), cols);

    let mut acc = 0.0f32;

    // Scratch buffer for unpacking codes (stack-allocated for scalar path)
    let mut codes = [0u8; QK256_BLOCK];

    // Debug: check if BITNET_QUANT_SANITY is enabled once
    let sanity_check = std::env::var("BITNET_QUANT_SANITY").as_deref() == Ok("1");

    let mut col = 0usize;
    for (block_idx, blk) in qs_row.chunks_exact(QK256_PACKED_BYTES).enumerate() {
        // Unpack 64B → 256 2-bit codes
        let blk_arr: &[u8; QK256_PACKED_BYTES] =
            blk.try_into().expect("QK256: block must be 64 bytes");
        unpack_qk256_block(blk_arr, &mut codes);

        // Number of valid columns left in this block
        let take = QK256_BLOCK.min(cols - col);

        // Probe B: QK256 block-level histogram and sanity check (only if enabled)
        if sanity_check {
            // Histogram of 2-bit codes
            let mut hist = [0usize; 4];
            for &code in codes.iter().take(take) {
                hist[(code & 0b11) as usize] += 1;
            }

            // Dequantize block
            let mut weights = [0.0f32; QK256_BLOCK];
            for (j, &code) in codes.iter().enumerate().take(take) {
                weights[j] = code_to_f32(code);
            }

            let rms = compute_rms(&weights[..take]);

            // Report first block diagnostics
            if block_idx == 0 {
                let sample_len = take.min(16);
                eprintln!(
                    "qk256: hist={:?} rms_first={:.3} sample={:?}",
                    hist,
                    rms,
                    &weights[..sample_len]
                );
            }

            // Warn on suspicious RMS
            if rms > 10.0 {
                eprintln!("qk256: block={} rms={:.3} (suspicious scale/unpack)", block_idx, rms);
            }
        }

        // Decode codes and accumulate dot product
        for j in 0..take {
            let w = code_to_f32(codes[j]);
            acc += w * x[col + j];
        }

        col += take;
        if col >= cols {
            break;
        }
    }

    acc
}

/// Multi-row GEMV: y = Ax where A is quantized QK256, x is dense
///
/// # Arguments
///
/// * `qs_data` - Contiguous row-major quantized data (rows * row_stride_bytes)
/// * `x` - Dense input vector (length = cols)
/// * `y_out` - Output vector (length = rows)
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `row_stride_bytes` - Bytes per row (ceil(cols/256) * 64)
///
/// # Errors
///
/// Returns error if dimensions don't match or data is insufficient.
pub fn gemv_qk256(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    if y_out.len() != rows {
        bail!("I2S_QK256: y_out length {} != rows {}", y_out.len(), rows);
    }
    if x.len() < cols {
        bail!("I2S_QK256: x length {} < cols {}", x.len(), cols);
    }

    let expected_total = rows * row_stride_bytes;
    if qs_data.len() < expected_total {
        bail!("I2S_QK256: data too short: {} < {}", qs_data.len(), expected_total);
    }

    for (row, output) in y_out.iter_mut().enumerate().take(rows) {
        let start = row * row_stride_bytes;
        let end = start + row_stride_bytes;
        let row_bytes = &qs_data[start..end];
        *output = gemv_qk256_row(row_bytes, x, cols);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_block_smoke() {
        // Pattern: 0b_11_10_01_00 repeated
        let mut qs = [0u8; QK256_PACKED_BYTES];
        for (i, b) in qs.iter_mut().enumerate() {
            *b = 0b_11_10_01_00u8.wrapping_add(i as u8 & 0x03);
        }
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&qs, &mut codes);

        // Verify codes are in 0..=3
        assert!(codes.iter().all(|&c| c < 4), "All codes must be 0..=3");

        // Verify first few codes match pattern
        assert_eq!(codes[0], 0);
        assert_eq!(codes[1], 1);
        assert_eq!(codes[2], 2);
        assert_eq!(codes[3], 3);
    }

    #[test]
    fn gemv_row_smoke() {
        // All codes = 2 (→ +1.0 with default LUT), so dot == sum(x)
        let mut qs = [0u8; QK256_PACKED_BYTES];
        // Code 2 everywhere → 0b_10_10_10_10 = 0xAA
        qs.fill(0xAA);

        let cols = 512usize; // 2 blocks
        let mut row = Vec::new();
        row.extend_from_slice(&qs);
        row.extend_from_slice(&qs); // 2 blocks packed

        let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.01).collect();
        let expected: f32 = x.iter().sum(); // because weight=+1.0 everywhere
        let got = gemv_qk256_row(&row, &x, cols);

        // Allow small floating-point error
        assert!((got - expected).abs() < 1e-3, "Expected ~{}, got {}", expected, got);
    }

    #[test]
    fn gemv_row_with_tail() {
        // Test with cols=300 (not multiple of 256)
        // Block 1: 256 elements, Block 2: 44 elements (tail)
        let cols = 300usize;
        let blocks_needed = cols.div_ceil(QK256_BLOCK); // = 2
        let qs_row = vec![0xAAu8; blocks_needed * QK256_PACKED_BYTES];

        let x: Vec<f32> = (0..cols).map(|i| (i % 7) as f32).collect();
        let got = gemv_qk256_row(&qs_row, &x, cols);

        // Code 2 → +1.0, so result should be sum of x[0..300]
        let expected: f32 = x.iter().sum();
        assert!(
            (got - expected).abs() < 1e-3,
            "Tail handling: expected ~{}, got {}",
            expected,
            got
        );
    }

    #[test]
    fn gemv_multi_row() {
        let rows = 3usize;
        let cols = 256usize;
        let row_stride_bytes = QK256_PACKED_BYTES;

        // All codes = 1 (→ -1.0)
        let qs_data = vec![0x55u8; rows * row_stride_bytes]; // 0b_01_01_01_01

        let x: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let mut y_out = vec![0.0f32; rows];

        gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, row_stride_bytes)
            .expect("gemv_qk256 should succeed");

        // Code 1 → -1.0, so each row = -sum(x)
        let expected: f32 = -x.iter().sum::<f32>();
        for (i, &val) in y_out.iter().enumerate() {
            assert!(
                (val - expected).abs() < 1e-3,
                "Row {}: expected ~{}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn code_to_f32_lut() {
        // Verify LUT values (verified against GGML ggml-quants.c:62)
        assert_eq!(code_to_f32(0), -2.0);
        assert_eq!(code_to_f32(1), -1.0);
        assert_eq!(code_to_f32(2), 1.0);
        assert_eq!(code_to_f32(3), 2.0);
    }

    #[test]
    #[should_panic(expected = "y_out length")]
    fn gemv_mismatched_y() {
        let qs_data = vec![0u8; 64];
        let x = vec![0.0f32; 256];
        let mut y_out = vec![0.0f32; 2]; // Wrong size!

        gemv_qk256(&qs_data, &x, &mut y_out, 1, 256, 64).unwrap();
    }

    /// Regression test for QK256 size tolerance (prevents enhanced→minimal fallback)
    ///
    /// This test verifies that the `I2SQk256NoScale::new` constructor accepts
    /// data sizes with alignment padding up to TOLERANCE=128 bytes. This is critical
    /// for keeping the enhanced loader active instead of falling back to the minimal
    /// loader with its 32/0 default dimensions.
    ///
    /// Test cases:
    /// 1. Exact size: should succeed
    /// 2. Exact + 32B (common padding): should succeed
    /// 3. Exact + 128B (at tolerance boundary): should succeed
    /// 4. Exact + 129B (beyond tolerance): should fail
    #[test]
    fn test_qk256_size_tolerance() {
        let rows = 512usize;
        let cols = 1024usize;
        let blocks_per_row = cols.div_ceil(QK256_BLOCK); // 4 blocks
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES; // 4 * 64 = 256 bytes
        let exact_size = rows * row_stride_bytes; // 512 * 256 = 131,072 bytes

        // Test 1: Exact size - should succeed
        let qs_exact = vec![0u8; exact_size];
        let result = I2SQk256NoScale::new(rows, cols, qs_exact);
        assert!(result.is_ok(), "Exact size should be accepted");

        // Test 2: Exact + 32 bytes (common alignment padding) - should succeed
        let qs_plus_32 = vec![0u8; exact_size + 32];
        let result = I2SQk256NoScale::new(rows, cols, qs_plus_32);
        assert!(result.is_ok(), "Size with +32B padding should be accepted (within TOLERANCE=128)");

        // Test 3: Exact + 128 bytes (at tolerance boundary) - should succeed
        let qs_plus_128 = vec![0u8; exact_size + 128];
        let result = I2SQk256NoScale::new(rows, cols, qs_plus_128);
        assert!(
            result.is_ok(),
            "Size with +128B padding should be accepted (at TOLERANCE boundary)"
        );

        // Test 4: Exact + 129 bytes (beyond tolerance) - should fail
        let qs_plus_129 = vec![0u8; exact_size + 129];
        let result = I2SQk256NoScale::new(rows, cols, qs_plus_129);
        assert!(
            result.is_err(),
            "Size with +129B padding should be rejected (beyond TOLERANCE=128)"
        );

        // Test 5: Way too small - should fail
        let qs_too_small = vec![0u8; exact_size / 2];
        let result = I2SQk256NoScale::new(rows, cols, qs_too_small);
        assert!(result.is_err(), "Size too small should be rejected");

        println!(
            "✅ QK256 tolerance regression test passed: exact={}, tolerance=±128B",
            exact_size
        );
    }

    // ========================================================================
    // QK256 Test Scaffolding: Tests A-D (Core Correctness)
    // ========================================================================
    // These tests lock in QK256 correctness per the specification.
    // Tests feature spec: docs/explanation/i2s-dual-flavor.md#qk256-format
    // Tests API contract: docs/reference/quantization-support.md#qk256-kernels
    // ========================================================================

    /// Test (A): LUT Sanity (NoScale)
    ///
    /// Tests feature spec: i2s-dual-flavor.md#code-mapping
    /// Verifies that the code-to-float lookup table matches GGML reference:
    /// - Code 0 → -2.0
    /// - Code 1 → -1.0
    /// - Code 2 → +1.0
    /// - Code 3 → +2.0
    #[test]
    fn qk256_lut_basic() {
        assert_eq!(code_to_f32(0), -2.0, "Code 0 should map to -2.0");
        assert_eq!(code_to_f32(1), -1.0, "Code 1 should map to -1.0");
        assert_eq!(code_to_f32(2), 1.0, "Code 2 should map to +1.0");
        assert_eq!(code_to_f32(3), 2.0, "Code 3 should map to +2.0");
    }

    /// Test (B): Block Decode Golden (64B → 256 f32)
    ///
    /// Tests feature spec: i2s-dual-flavor.md#memory-layout
    /// Pack 256 two-bit codes (LSB-first) cycling 0..3 into 64 bytes.
    /// Decode using the unpack path and verify:
    /// - RMS in range [0.1, 5.0]
    /// - First 16 values contain the set {-2, -1, 1, 2}
    #[test]
    fn qk256_block_decode_golden() {
        // Pack pattern: 0,1,2,3,0,1,2,3,... (cycling through all codes)
        let mut qs64 = [0u8; QK256_PACKED_BYTES];
        for (i, byte) in qs64.iter_mut().enumerate() {
            // Each byte packs 4 codes: elem0 | (elem1 << 2) | (elem2 << 4) | (elem3 << 6)
            let base = i * 4;
            let code0 = (base % 4) as u8;
            let code1 = ((base + 1) % 4) as u8;
            let code2 = ((base + 2) % 4) as u8;
            let code3 = ((base + 3) % 4) as u8;
            *byte = code0 | (code1 << 2) | (code2 << 4) | (code3 << 6);
        }

        // Unpack block
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&qs64, &mut codes);

        // Verify codes cycle 0..3
        for (i, &code) in codes.iter().enumerate() {
            let expected = (i % 4) as u8;
            assert_eq!(
                code, expected,
                "Code at position {} should be {}, got {}",
                i, expected, code
            );
        }

        // Dequantize codes to f32 using LUT
        let mut weights = [0.0f32; QK256_BLOCK];
        for (i, &code) in codes.iter().enumerate() {
            weights[i] = code_to_f32(code);
        }

        // Compute RMS: sqrt(mean(x^2))
        let sum_sq: f32 = weights.iter().map(|x| x * x).sum();
        let rms = (sum_sq / QK256_BLOCK as f32).sqrt();

        // Verify RMS is reasonable (should be ~1.58 for uniform {-2,-1,1,2})
        assert!((0.1..=5.0).contains(&rms), "RMS {} should be in range [0.1, 5.0]", rms);

        // Verify first 16 values contain all expected codes
        let first_16: Vec<f32> = weights[..16].to_vec();
        assert!(first_16.contains(&-2.0), "First 16 values should contain -2.0");
        assert!(first_16.contains(&-1.0), "First 16 values should contain -1.0");
        assert!(first_16.contains(&1.0), "First 16 values should contain 1.0");
        assert!(first_16.contains(&2.0), "First 16 values should contain 2.0");
    }

    /// Test (C): Tiny GEMV E2E (1×256 × 256×256)
    ///
    /// Tests feature spec: i2s-dual-flavor.md#gemv-operation
    /// Input: ones vector (256 elements)
    /// Packed weight: 1 row of 256 elements (64 bytes packed)
    /// Reference: dequantize packed → f32 matmul
    #[test]
    fn qk256_tiny_gemv_e2e() -> Result<()> {
        let rows = 1usize;
        let cols = 256usize;
        let row_stride_bytes = QK256_PACKED_BYTES;

        // Create packed data: all codes = 2 (→ +1.0)
        // Pattern: 0b_10_10_10_10 = 0xAA
        let qs_data = vec![0xAAu8; row_stride_bytes];

        // Input: ones vector
        let x = vec![1.0f32; cols];

        // Expected output: dot product of [1.0; 256] with [1.0; 256] = 256.0
        // (since code 2 → +1.0, and we have 256 elements)
        let expected = 256.0f32;

        // Call QK256 kernel
        let mut y_out = vec![0.0f32; rows];
        gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, row_stride_bytes)?;

        // Verify result (allow small floating-point error)
        let abs_diff = (y_out[0] - expected).abs();
        assert!(abs_diff < 1e-4, "Expected ~{}, got {}, diff={}", expected, y_out[0], abs_diff);

        // Reference path: dequantize and compute dot product manually
        let mut codes = [0u8; QK256_BLOCK];
        let qs_arr: &[u8; QK256_PACKED_BYTES] =
            qs_data[..QK256_PACKED_BYTES].try_into().expect("Should be 64 bytes");
        unpack_qk256_block(qs_arr, &mut codes);

        let mut ref_result = 0.0f32;
        for (i, &code) in codes.iter().enumerate() {
            let w = code_to_f32(code);
            ref_result += w * x[i];
        }

        // Verify kernel matches reference
        let ref_diff = (y_out[0] - ref_result).abs();
        assert!(
            ref_diff < 1e-6,
            "Kernel result {} should match reference {}, diff={}",
            y_out[0],
            ref_result,
            ref_diff
        );

        Ok(())
    }

    /// Test (D): Negatives - Dimension/Size Checks
    ///
    /// Tests feature spec: i2s-dual-flavor.md#error-handling
    /// Tests API contract: docs/reference/quantization-support.md#validation
    /// Multiple test cases that should fail with clear error messages:
    /// 1. Input vector shorter than cols
    /// 2. Packed buffer too small for dimensions
    /// 3. Output vector wrong size
    ///
    /// Note: Mismatched row_stride_bytes is caught by debug_assert in gemv_qk256_row
    /// and tested separately in qk256_stride_mismatch_panics.
    #[test]
    fn qk256_negatives_dimension_checks() {
        // Test 1: Input vector shorter than cols
        {
            let rows = 1usize;
            let cols = 256usize;
            let row_stride_bytes = QK256_PACKED_BYTES;
            let qs_data = vec![0u8; row_stride_bytes];
            let x = vec![1.0f32; cols - 10]; // Too short!
            let mut y_out = vec![0.0f32; rows];

            let result = gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, row_stride_bytes);
            assert!(result.is_err(), "Should fail with short input vector");
            assert!(
                result.unwrap_err().to_string().contains("x length"),
                "Error should mention input length mismatch"
            );
        }

        // Test 2: Packed buffer too small for dimensions
        {
            let rows = 2usize;
            let cols = 256usize;
            let row_stride_bytes = QK256_PACKED_BYTES;
            let qs_data = vec![0u8; row_stride_bytes]; // Only 1 row worth!
            let x = vec![1.0f32; cols];
            let mut y_out = vec![0.0f32; rows];

            let result = gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, row_stride_bytes);
            assert!(result.is_err(), "Should fail with buffer too small");
            assert!(
                result.unwrap_err().to_string().contains("too short"),
                "Error should mention data size mismatch"
            );
        }

        // Test 3: Output vector wrong size
        {
            let rows = 2usize;
            let cols = 256usize;
            let row_stride_bytes = QK256_PACKED_BYTES;
            let qs_data = vec![0u8; rows * row_stride_bytes];
            let x = vec![1.0f32; cols];
            let mut y_out = vec![0.0f32; 1]; // Wrong size!

            let result = gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, row_stride_bytes);
            assert!(result.is_err(), "Should fail with wrong output size");
            assert!(
                result.unwrap_err().to_string().contains("y_out length"),
                "Error should mention output length mismatch"
            );
        }
    }

    /// Test for stride mismatch (panics in debug mode via debug_assert)
    ///
    /// This test verifies that mismatched row_stride_bytes vs cols is caught
    /// by the debug_assert in gemv_qk256_row. In release builds, this test
    /// is skipped since debug_assert is disabled.
    #[test]
    #[should_panic(expected = "row bytes mismatch")]
    #[cfg(debug_assertions)]
    fn qk256_stride_mismatch_panics() {
        let rows = 1usize;
        let cols = 256usize;
        let wrong_stride = 128usize; // Should be 64 for 256 cols
        let qs_data = vec![0u8; rows * wrong_stride];
        let x = vec![1.0f32; cols];
        let mut y_out = vec![0.0f32; rows];

        // This will panic in debug mode due to debug_assert in gemv_qk256_row
        let _ = gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, wrong_stride);
    }
}
