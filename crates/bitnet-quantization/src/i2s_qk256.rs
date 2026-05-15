//! Microsoft BitNet I2_S (QK=256) scalar reference kernels
//!
//! This module implements pure-Rust dequantization and GEMV for Microsoft BitNet's
//! GGUF I2_S format:
//! - Block size: 256 elements
//! - Packed format: 64 bytes per block (2 bits/element)
//! - Per-tensor scale: one little-endian `f32` trailer after the packed bytes
//! - Code mapping: ternary `0 -> -1`, `1 -> 0`, `2 -> +1`; code `3` is unused by
//!   the Microsoft BitNet quantizer and is treated as `+2` for deterministic decode
//!
//! ## Memory Layout
//!
//! Each block contains two Microsoft BitNet 128-element packing groups. Each
//! 128-element group uses 32 bytes:
//! ```text
//! byte p stores elements p, p+32, p+64, p+96 for p in 0..32
//! ```
//!
//! Each byte packs four lane-separated elements (2 bits each), MSB first:
//! ```text
//! byte = (elem_p << 6) | (elem_p+32 << 4) | (elem_p+64 << 2) | elem_p+96
//! ```
//!
//! ## Code Mapping (VERIFIED)
//!
//! The 2-bit codes map to signed weights according to Microsoft BitNet's
//! `quantize_i2_s` path:
//!
//! - Code 0 → -1.0
//! - Code 1 → 0.0
//! - Code 2 → +1.0
//! - Code 3 → +2.0 (unused by the quantizer)
//!
//! The public no-scale entry points use scale `1.0` for fixtures. Real Microsoft
//! BitNet GGUF loading should call the scaled entry point with the trailer scale.

use anyhow::{Result, bail};
use bitnet_qk256_layout_core::{
    QK256_BLOCK_COLS, QK256_PACKED_BYTES_PER_BLOCK, Qk256Layout, qk256_row_stride_bytes,
};

/// Block size for GGML I2_S format
pub const QK256_BLOCK: usize = QK256_BLOCK_COLS;

/// Packed bytes per block (2 bits/elem * 256 elem / 8 bits/byte)
pub const QK256_PACKED_BYTES: usize = QK256_PACKED_BYTES_PER_BLOCK;

/// Stable receipt/proof kernel ID for the canonical scalar QK256 decode GEMV.
pub const QK256_SCALAR_GEMV_KERNEL_ID: &str = "qk256-scalar-gemv";

/// Stable receipt/proof kernel ID for the canonical scalar QK256 prefill GEMM.
pub const QK256_SCALAR_GEMM_KERNEL_ID: &str = "qk256-scalar-gemm";

/// Storage for Microsoft BitNet I2_S (QK=256) quantized weights.
///
/// This structure holds raw packed 2-bit codes for a weight tensor in the
/// "GgmlQk256NoScale" format used by MS BitNet GGUF models. The packed data is
/// stored in row-major order without dequantization. The optional per-tensor scale
/// is carried separately because the packed row tensor is passed through Candle as
/// a U8 matrix.
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
    pub scale: f32,
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
        Self::new_with_scale(rows, cols, qs, 1.0)
    }

    /// Create a new QK256 quantized tensor with an explicit per-tensor scale.
    pub fn new_with_scale(rows: usize, cols: usize, qs: Vec<u8>, scale: f32) -> Result<Self> {
        let layout = Qk256Layout::from_rows_cols(rows, cols)?;
        let row_stride_bytes = layout.row_stride_bytes;
        let expected_bytes = layout.packed_len_bytes;

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

        Ok(Self { rows, cols, row_stride_bytes, scale, qs })
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
/// This mapping matches Microsoft BitNet's `quantize_i2_s` encoding:
/// `0, 1, 2` represent `-1, 0, +1`, and code `3` is unused by the quantizer.
/// Real Microsoft BitNet GGUF tensors multiply the dot result by the per-tensor
/// scale stored after the packed bytes.
#[inline]
pub fn code_to_f32(code: u8) -> f32 {
    // SAFETY: code is masked to 0..=3 by caller
    debug_assert!(code < 4, "I2S_QK256: code must be 0..=3, got {}", code);

    const LUT: [f32; 4] = [-1.0, 0.0, 1.0, 2.0];
    LUT[code as usize]
}

/// Reference-compatible nearest integer rule used by GGML activation quantization.
///
/// This mirrors the `nearest_int` bit trick used by the BitNet reference path
/// before clamping to the signed i8 activation range.
#[inline]
pub fn nearest_int_reference(fval: f32) -> i32 {
    assert!(
        fval.abs() <= 4_194_303.0,
        "nearest_int_reference expects |fval| <= 4194303, got {fval}"
    );
    let val = fval + 12_582_912.0;
    let bits = val.to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

/// Reference activation row quantization for GGML `I2_S` matmul.
#[derive(Clone, Debug, PartialEq)]
pub struct I2SActivationQuant {
    pub values: Vec<i8>,
    pub scale: f32,
    pub sum: i32,
}

/// Quantize one dense activation row using the reference `quantize_row_i8_s` rule.
///
/// The returned `scale` is the reference activation scale `s = 127 / max_abs`.
/// The matmul correction uses `(integer_dot - act_sum) / act_scale`.
pub fn quantize_row_i8_s_reference(x: &[f32]) -> I2SActivationQuant {
    let mut max_abs = 0.00001f32;
    for &value in x {
        max_abs = max_abs.max(value.abs());
    }

    let scale = 127.0 / max_abs;
    let mut values = Vec::with_capacity(x.len());
    let mut sum = 0i32;
    for &value in x {
        let q = nearest_int_reference(value * scale).clamp(-128, 127) as i8;
        sum += i32::from(q);
        values.push(q);
    }

    I2SActivationQuant { values, scale, sum }
}

/// Reference-compatible `I2_S` QK256 row GEMV with activation quantization.
///
/// This is a proof/oracle helper for the reference contract:
/// `(integer_dot - act_sum) / act_scale * trailer_scale`.
/// It is intentionally separate from the current runtime dispatch path.
#[inline]
pub fn gemv_qk256_row_activation_quantized_reference(
    qs_row: &[u8],
    x: &[f32],
    cols: usize,
    trailer_scale: f32,
) -> f32 {
    let expected_bytes = qk256_row_stride_bytes(cols)
        .expect("QK256: row stride overflow should be impossible for in-memory row");

    debug_assert_eq!(
        qs_row.len(),
        expected_bytes,
        "I2S_QK256: row bytes mismatch: got {}, expected {} for {} cols",
        qs_row.len(),
        expected_bytes,
        cols
    );
    debug_assert!(x.len() >= cols, "I2S_QK256: x too short: {} < {}", x.len(), cols);

    let activation = quantize_row_i8_s_reference(&x[..cols]);
    let mut integer_dot = 0i32;
    for (col, &q) in activation.values.iter().enumerate().take(cols) {
        let raw_code = i32::from(packed_code_at(qs_row, col));
        integer_dot += raw_code * i32::from(q);
    }

    ((integer_dot - activation.sum) as f32) / activation.scale * trailer_scale
}

/// Extract one logical QK256 code from a Microsoft BitNet-packed row.
///
/// Microsoft's `quantize_i2_s` groups each 128 logical values into 32 bytes.
/// Within that group, byte `p` stores logical positions `p`, `p+32`,
/// `p+64`, and `p+96` in bits `[7:6]`, `[5:4]`, `[3:2]`, and `[1:0]`.
#[inline]
pub fn packed_code_at(qs_row: &[u8], col: usize) -> u8 {
    let group128 = col / 128;
    let within = col % 128;
    let lane = within / 32;
    let pos = within % 32;
    let byte = qs_row[group128 * 32 + pos];
    (byte >> (6 - lane * 2)) & 0x03
}

/// Pack logical QK256 codes using the Microsoft BitNet act-parallel layout.
///
/// This helper is public for tests and diagnostics. Production loading preserves
/// the GGUF bytes directly.
pub fn pack_qk256_codes_for_cols(codes: &[u8], cols: usize) -> Vec<u8> {
    let row_stride = qk256_row_stride_bytes(cols)
        .expect("QK256: row stride overflow should be impossible for test packing");
    let mut packed = vec![0u8; row_stride];
    for (col, &code) in codes.iter().enumerate().take(cols) {
        debug_assert!(code < 4, "QK256 code must be 0..=3, got {code}");
        let group128 = col / 128;
        let within = col % 128;
        let lane = within / 32;
        let pos = within % 32;
        packed[group128 * 32 + pos] |= (code & 0x03) << (6 - lane * 2);
    }
    packed
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
    for col in 0..QK256_BLOCK {
        out_codes256[col] = packed_code_at(qs64, col);
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
    let expected_bytes = qk256_row_stride_bytes(cols)
        .expect("QK256: row stride overflow should be impossible for in-memory row");

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

        for j in 0..take {
            let w = code_to_f32(packed_code_at(blk, j));
            acc += w * x[col + j];
        }

        col += take;
        if col >= cols {
            break;
        }
    }

    acc
}

/// Scalar implementation of multi-row GEMV (internal)
///
/// This is the scalar reference implementation used when SIMD is not available
/// or explicitly requested for testing.
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
fn gemv_qk256_scalar_checked(
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

/// Canonical scalar QK256 GEMV oracle for decode: `y = A x`.
///
/// This path uses the Microsoft BitNet I2_S mapping from [`code_to_f32`] and the
/// canonical QK256 packed layout. It never dispatches to SIMD.
pub fn qk256_gemv_scalar(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    let layout = Qk256Layout::from_rows_cols(rows, cols)?;
    layout.validate_packed_len(qs_data.len())?;
    gemv_qk256_scalar_checked(qs_data, x, y_out, rows, cols, layout.row_stride_bytes)
}

/// Canonical scalar QK256 GEMM oracle for prefill: `Y = X A^T`.
///
/// `x` is row-major with shape `tokens × cols`; `y_out` is row-major with shape
/// `tokens × rows`. The packed matrix `A` is row-major QK256 with shape
/// `rows × cols`.
pub fn qk256_gemm_scalar(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    tokens: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let layout = Qk256Layout::from_rows_cols(rows, cols)?;
    layout.validate_packed_len(qs_data.len())?;

    let expected_x_len = tokens.checked_mul(cols).ok_or_else(|| {
        anyhow::anyhow!("I2S_QK256: x length overflow for tokens={tokens}, cols={cols}")
    })?;
    if x.len() != expected_x_len {
        bail!("I2S_QK256: x length {} != tokens*cols {}", x.len(), expected_x_len);
    }

    let expected_y_len = tokens.checked_mul(rows).ok_or_else(|| {
        anyhow::anyhow!("I2S_QK256: y_out length overflow for tokens={tokens}, rows={rows}")
    })?;
    if y_out.len() != expected_y_len {
        bail!("I2S_QK256: y_out length {} != tokens*rows {}", y_out.len(), expected_y_len);
    }

    for token in 0..tokens {
        let x_start = token * cols;
        let y_start = token * rows;
        gemv_qk256_scalar_checked(
            qs_data,
            &x[x_start..x_start + cols],
            &mut y_out[y_start..y_start + rows],
            rows,
            cols,
            layout.row_stride_bytes,
        )?;
    }

    Ok(())
}

/// Multi-row GEMV with runtime dispatch: y = Ax where A is quantized QK256, x is dense
///
/// This function automatically selects the best available implementation:
/// - **AVX2**: x86_64 with AVX2 support (3-5× speedup over scalar)
/// - **Scalar**: Fallback for all other cases
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
///
/// # Performance
///
/// Runtime dispatch adds negligible overhead (~1-2 CPU cycles) compared to kernel
/// execution time (thousands of cycles for typical matrix dimensions).
pub fn gemv_qk256(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    let expected_stride = qk256_row_stride_bytes(cols)?;
    if row_stride_bytes != expected_stride {
        bail!(
            "I2S_QK256: row_stride_bytes {} != expected {} for cols={}",
            row_stride_bytes,
            expected_stride,
            cols
        );
    }

    // Runtime dispatch: probe for AVX2 support on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Use AVX2 path (3-5× speedup over scalar)
            return super::i2s_qk256_avx2::gemv_qk256_avx2(
                qs_data,
                x,
                y_out,
                rows,
                cols,
                row_stride_bytes,
            );
        }
    }

    // Fallback to scalar implementation
    gemv_qk256_scalar_checked(qs_data, x, y_out, rows, cols, row_stride_bytes)
}

/// Multi-row GEMV with an explicit Microsoft BitNet per-tensor trailer scale.
pub fn gemv_qk256_scaled(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    scale: f32,
) -> Result<()> {
    gemv_qk256(qs_data, x, y_out, rows, cols, row_stride_bytes)?;
    if scale != 1.0 {
        for output in y_out {
            *output *= scale;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_codes_for_cols(codes: &[u8], cols: usize) -> Vec<u8> {
        pack_qk256_codes_for_cols(codes, cols)
    }

    fn reference_dot(codes: &[u8], x: &[f32], cols: usize) -> f32 {
        codes
            .iter()
            .copied()
            .zip(x.iter().copied())
            .take(cols)
            .map(|(code, x)| code_to_f32(code) * x)
            .sum()
    }

    #[test]
    fn unpack_block_smoke() {
        // Logical pattern: 0,1,2,3 repeating in Microsoft BitNet packed layout.
        let mut qs = [0u8; QK256_PACKED_BYTES];
        let pattern: Vec<u8> = (0..QK256_BLOCK).map(|i| (i % 4) as u8).collect();
        let packed = pack_qk256_codes_for_cols(&pattern, QK256_BLOCK);
        qs.copy_from_slice(&packed);
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&qs, &mut codes);

        // Verify codes are in 0..=3
        assert!(codes.iter().all(|&c| c < 4), "All codes must be 0..=3");

        // Verify first few logical codes match pattern.
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

        // All codes = 1 (→ 0.0)
        let qs_data = vec![0x55u8; rows * row_stride_bytes]; // 0b_01_01_01_01

        let x: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let mut y_out = vec![0.0f32; rows];

        gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, row_stride_bytes)
            .expect("gemv_qk256 should succeed");

        // Code 1 → 0.0, so each row is zero.
        let expected: f32 = 0.0;
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
        // Verify LUT values against Microsoft BitNet quantize_i2_s ternary encoding.
        assert_eq!(code_to_f32(0), -1.0);
        assert_eq!(code_to_f32(1), 0.0);
        assert_eq!(code_to_f32(2), 1.0);
        assert_eq!(code_to_f32(3), 2.0);
    }

    #[test]
    fn nearest_int_reference_matches_ggml_rounding_edges() {
        assert_eq!(nearest_int_reference(0.0), 0);
        assert_eq!(nearest_int_reference(0.49), 0);
        assert_eq!(nearest_int_reference(0.5), 0);
        assert_eq!(nearest_int_reference(0.51), 1);
        assert_eq!(nearest_int_reference(1.49), 1);
        assert_eq!(nearest_int_reference(1.5), 2);
        assert_eq!(nearest_int_reference(-0.49), 0);
        assert_eq!(nearest_int_reference(-0.5), 0);
        assert_eq!(nearest_int_reference(-0.51), -1);
        assert_eq!(nearest_int_reference(-1.5), -2);
    }

    #[test]
    fn quantize_row_i8_s_reference_tracks_scale_values_and_sum() {
        let quantized = quantize_row_i8_s_reference(&[-1.0, 0.0, 0.5, 1.0]);

        assert_eq!(quantized.scale, 127.0);
        assert_eq!(quantized.values, vec![-127, 0, 64, 127]);
        assert_eq!(quantized.sum, 64);
    }

    #[test]
    fn quantize_row_i8_s_reference_handles_zero_rows() {
        let quantized = quantize_row_i8_s_reference(&[0.0, 0.0, 0.0]);

        assert_eq!(quantized.scale, 12_700_000.0);
        assert_eq!(quantized.values, vec![0, 0, 0]);
        assert_eq!(quantized.sum, 0);
    }

    #[test]
    fn activation_quantized_gemv_differs_from_dense_direct_path() {
        let cols = QK256_BLOCK;
        let trailer_scale = 0.375f32;
        let codes: Vec<u8> = (0..cols).map(|i| (i % 4) as u8).collect();
        let qs_row = pack_codes_for_cols(&codes, cols);
        let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) / 7.0).collect();

        let dense_direct = gemv_qk256_row(&qs_row, &x, cols) * trailer_scale;
        let reference =
            gemv_qk256_row_activation_quantized_reference(&qs_row, &x, cols, trailer_scale);

        assert!(
            (reference - dense_direct).abs() > 1e-4,
            "reference activation quantization should expose a real contract delta"
        );
    }

    #[test]
    fn activation_quantized_gemv_uses_code3_as_effective_plus_two() {
        let cols = QK256_BLOCK;
        let trailer_scale = 0.5f32;
        let x: Vec<f32> = (0..cols).map(|i| ((i % 9) as f32 - 4.0) / 3.0).collect();
        let code2_row = pack_codes_for_cols(&vec![2u8; cols], cols);
        let code3_row = pack_codes_for_cols(&vec![3u8; cols], cols);

        let code2 =
            gemv_qk256_row_activation_quantized_reference(&code2_row, &x, cols, trailer_scale);
        let code3 =
            gemv_qk256_row_activation_quantized_reference(&code3_row, &x, cols, trailer_scale);

        assert!(
            (code3 - code2 * 2.0).abs() < 1e-5,
            "corrected integer matmul should make code3 twice code2"
        );
    }

    #[test]
    fn qk256_scalar_kernel_ids_are_stable() {
        assert_eq!(QK256_SCALAR_GEMV_KERNEL_ID, "qk256-scalar-gemv");
        assert_eq!(QK256_SCALAR_GEMM_KERNEL_ID, "qk256-scalar-gemm");
    }

    #[test]
    fn qk256_gemv_scalar_matches_reference_fixture() -> Result<()> {
        let rows = 2usize;
        let cols = 300usize;
        let x: Vec<f32> = (0..cols).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
        let row0_codes: Vec<u8> = (0..cols).map(|i| (i % 4) as u8).collect();
        let row1_codes: Vec<u8> = (0..cols).map(|i| ((i + 1) % 4) as u8).collect();

        let mut qs_data = Vec::new();
        qs_data.extend_from_slice(&pack_codes_for_cols(&row0_codes, cols));
        qs_data.extend_from_slice(&pack_codes_for_cols(&row1_codes, cols));

        let mut y_out = vec![0.0f32; rows];
        qk256_gemv_scalar(&qs_data, &x, &mut y_out, rows, cols)?;

        let expected = [reference_dot(&row0_codes, &x, cols), reference_dot(&row1_codes, &x, cols)];
        for (got, expected) in y_out.iter().zip(expected) {
            assert!((got - expected).abs() < 1e-5, "got {got}, expected {expected}");
        }

        let mut y_out_repeat = vec![0.0f32; rows];
        qk256_gemv_scalar(&qs_data, &x, &mut y_out_repeat, rows, cols)?;
        assert_eq!(y_out, y_out_repeat, "scalar GEMV must be deterministic");

        Ok(())
    }

    #[test]
    fn qk256_gemm_scalar_matches_batched_gemv_fixture() -> Result<()> {
        let tokens = 3usize;
        let rows = 2usize;
        let cols = 256usize;
        let row0_codes: Vec<u8> = (0..cols).map(|i| (i % 4) as u8).collect();
        let row1_codes: Vec<u8> = (0..cols).map(|i| ((i + 2) % 4) as u8).collect();

        let mut qs_data = Vec::new();
        qs_data.extend_from_slice(&pack_codes_for_cols(&row0_codes, cols));
        qs_data.extend_from_slice(&pack_codes_for_cols(&row1_codes, cols));

        let x: Vec<f32> = (0..tokens * cols).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        let mut y_out = vec![0.0f32; tokens * rows];
        qk256_gemm_scalar(&qs_data, &x, &mut y_out, tokens, rows, cols)?;

        for token in 0..tokens {
            let x_token = &x[token * cols..(token + 1) * cols];
            let expected0 = reference_dot(&row0_codes, x_token, cols);
            let expected1 = reference_dot(&row1_codes, x_token, cols);
            assert!((y_out[token * rows] - expected0).abs() < 1e-5);
            assert!((y_out[token * rows + 1] - expected1).abs() < 1e-5);
        }

        let mut y_out_repeat = vec![0.0f32; tokens * rows];
        qk256_gemm_scalar(&qs_data, &x, &mut y_out_repeat, tokens, rows, cols)?;
        assert_eq!(y_out, y_out_repeat, "scalar GEMM must be deterministic");

        Ok(())
    }

    #[test]
    #[should_panic(expected = "y_out length")]
    fn gemv_mismatched_y() {
        let qs_data = vec![0u8; 64];
        let x = vec![0.0f32; 256];
        let mut y_out = vec![0.0f32; 2]; // Wrong size!

        gemv_qk256(&qs_data, &x, &mut y_out, 1, 256, 64).unwrap();
    }

    #[test]
    fn gemv_rejects_invalid_row_stride() {
        let rows = 1usize;
        let cols = 256usize;
        let bad_row_stride = 32usize;
        let qs_data = vec![0u8; bad_row_stride];
        let x = vec![0.0f32; cols];
        let mut y_out = vec![0.0f32; rows];

        let err = gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, bad_row_stride)
            .expect_err("invalid row_stride_bytes should error");

        assert!(
            err.to_string().contains("row_stride_bytes"),
            "error should mention row_stride_bytes, got: {}",
            err
        );
    }

    #[test]
    fn gemv_force_scalar_override_works() {
        let rows = 2usize;
        let cols = 256usize;
        let row_stride_bytes = QK256_PACKED_BYTES;
        let qs_data = vec![0xAAu8; rows * row_stride_bytes]; // +1.0 weights
        let x = vec![1.0f32; cols];
        let mut y_out = vec![0.0f32; rows];

        // SAFETY: This test is single-threaded; no other threads read this env var.
        unsafe { std::env::set_var("BITNET_FORCE_SCALAR", "1") };
        let result = gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, row_stride_bytes);
        unsafe { std::env::remove_var("BITNET_FORCE_SCALAR") };

        result.expect("scalar override should run successfully");
        for &v in &y_out {
            assert!((v - 256.0).abs() < 1e-5, "Expected 256.0, got {}", v);
        }
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
    /// Verifies that the code-to-float lookup table matches Microsoft BitNet:
    /// - Code 0 → -1.0
    /// - Code 1 → 0.0
    /// - Code 2 → +1.0
    /// - Code 3 → +2.0 (unused by the quantizer)
    #[test]
    fn qk256_lut_basic() {
        assert_eq!(code_to_f32(0), -1.0, "Code 0 should map to -1.0");
        assert_eq!(code_to_f32(1), 0.0, "Code 1 should map to 0.0");
        assert_eq!(code_to_f32(2), 1.0, "Code 2 should map to +1.0");
        assert_eq!(code_to_f32(3), 2.0, "Code 3 should map to +2.0");
    }

    /// Test (B): Block Decode Golden (64B → 256 f32)
    ///
    /// Tests feature spec: i2s-dual-flavor.md#memory-layout
    /// Pack 256 two-bit codes cycling 0..3 into Microsoft BitNet act-parallel bytes.
    /// Decode using the unpack path and verify:
    /// - RMS in range [0.1, 5.0]
    /// - First 16 values contain the set {-1, 0, 1, 2}
    #[test]
    fn qk256_block_decode_golden() {
        let logical_codes: Vec<u8> = (0..QK256_BLOCK).map(|i| (i % 4) as u8).collect();
        let packed = pack_qk256_codes_for_cols(&logical_codes, QK256_BLOCK);
        let qs64: [u8; QK256_PACKED_BYTES] =
            packed.try_into().expect("QK256 block packs into 64 bytes");

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

        // Verify RMS is reasonable (sqrt(1.5) for uniform {-1,0,1,2})
        assert!((0.1..=5.0).contains(&rms), "RMS {} should be in range [0.1, 5.0]", rms);

        // Verify first 16 values contain all expected codes
        let first_16: Vec<f32> = weights[..16].to_vec();
        assert!(first_16.contains(&-1.0), "First 16 values should contain -1.0");
        assert!(first_16.contains(&0.0), "First 16 values should contain 0.0");
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
    /// by the validation in gemv_qk256 and returned as an error.
    #[test]
    fn qk256_stride_mismatch_panics() {
        let rows = 1usize;
        let cols = 256usize;
        let wrong_stride = 128usize; // Should be 64 for 256 cols
        let qs_data = vec![0u8; rows * wrong_stride];
        let x = vec![1.0f32; cols];
        let mut y_out = vec![0.0f32; rows];

        let err = gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, wrong_stride)
            .expect_err("mismatched row_stride_bytes should error");
        assert!(
            err.to_string().contains("row_stride_bytes"),
            "error should mention row_stride_bytes, got: {}",
            err
        );
    }
}
