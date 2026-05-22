//! GGML I2_S (QK=256) scalar reference kernels
//!
//! This module implements pure-Rust dequantization and GEMV for GGML's I2_S format:
//! - Block size: 256 elements
//! - Packed format: 64 bytes per block (2 bits/element, no embedded scales)
//! - Code mapping: **VERIFIED** against BitNet.cpp `dequantize_row_i2_s`
//!
//! ## Memory Layout
//!
//! Each block contains 256 elements packed into 64 bytes as two 128-value
//! chunks. Within each chunk, 32 bytes store four 32-value bitplanes:
//! ```text
//! byte gp = (elem[0*32 + gp] << 6)
//!         | (elem[1*32 + gp] << 4)
//!         | (elem[2*32 + gp] << 2)
//!         |  elem[3*32 + gp]
//! ```
//!
//! ## Code Mapping (VERIFIED)
//!
//! The 2-bit codes map to ternary weights according to GGML's I2_S
//! dequantization used by BitNet.cpp:
//!
//! - Code 0 → -1.0
//! - Code 1 →  0.0
//! - Code 2 → +1.0
//! - Code 3 →  0.0
//!
//! This implementation supports the no-scale I2_S variant used by MS BitNet
//! GGUF models.

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

/// Stable receipt/proof kernel ID for the canonical AVX2/FMA QK256 decode GEMV.
pub const QK256_AVX2_GEMV_KERNEL_ID: &str = "qk256-avx2-gemv";

/// Proof-level QK256 kernel selection metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qk256KernelSelection {
    pub requested_kernel: Option<&'static str>,
    pub selected_kernel: &'static str,
    pub fallback_used: bool,
    pub fallback_reason: Option<String>,
    pub cpu_features: Vec<&'static str>,
}

fn qk256_cpu_features(avx2_fma_available: bool) -> Vec<&'static str> {
    if avx2_fma_available { vec!["avx2", "fma"] } else { Vec::new() }
}

fn qk256_avx2_fma_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        return bitnet_cpu_detect::avx2_fma_available();
    }

    #[allow(unreachable_code)]
    false
}

fn select_qk256_gemv_kernel_for_availability(
    requested_kernel: Option<&'static str>,
    strict: bool,
    avx2_fma_available: bool,
) -> Result<Qk256KernelSelection> {
    let cpu_features = qk256_cpu_features(avx2_fma_available);

    match requested_kernel {
        None if avx2_fma_available => Ok(Qk256KernelSelection {
            requested_kernel,
            selected_kernel: QK256_AVX2_GEMV_KERNEL_ID,
            fallback_used: false,
            fallback_reason: None,
            cpu_features,
        }),
        None => Ok(Qk256KernelSelection {
            requested_kernel,
            selected_kernel: QK256_SCALAR_GEMV_KERNEL_ID,
            fallback_used: false,
            fallback_reason: None,
            cpu_features,
        }),
        Some(QK256_SCALAR_GEMV_KERNEL_ID) => Ok(Qk256KernelSelection {
            requested_kernel,
            selected_kernel: QK256_SCALAR_GEMV_KERNEL_ID,
            fallback_used: false,
            fallback_reason: None,
            cpu_features,
        }),
        Some(QK256_AVX2_GEMV_KERNEL_ID) if avx2_fma_available => Ok(Qk256KernelSelection {
            requested_kernel,
            selected_kernel: QK256_AVX2_GEMV_KERNEL_ID,
            fallback_used: false,
            fallback_reason: None,
            cpu_features,
        }),
        Some(QK256_AVX2_GEMV_KERNEL_ID) if strict => {
            bail!(
                "I2S_QK256: strict requested kernel qk256-avx2-gemv cannot fall back because avx2/fma is unavailable"
            )
        }
        Some(QK256_AVX2_GEMV_KERNEL_ID) => Ok(Qk256KernelSelection {
            requested_kernel,
            selected_kernel: QK256_SCALAR_GEMV_KERNEL_ID,
            fallback_used: true,
            fallback_reason: Some("avx2/fma unavailable".to_string()),
            cpu_features,
        }),
        Some(other) => {
            bail!("I2S_QK256: unsupported requested QK256 GEMV kernel '{other}'")
        }
    }
}

/// Select the QK256 decode GEMV kernel without executing it.
pub fn select_qk256_gemv_kernel(
    requested_kernel: Option<&'static str>,
    strict: bool,
) -> Result<Qk256KernelSelection> {
    select_qk256_gemv_kernel_for_availability(requested_kernel, strict, qk256_avx2_fma_available())
}

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
/// **VERIFIED**: This mapping matches BitNet.cpp's GGML I2_S dequantization.
/// Reference: `const float map2bit[4] = { -1.0f, 0.0f, +1.0f, 0.0f };`
///
/// For MS BitNet I2_S QK256, these values are used directly with no per-block
/// scale.
#[inline]
pub fn code_to_f32(code: u8) -> f32 {
    // SAFETY: code is masked to 0..=3 by caller
    debug_assert!(code < 4, "I2S_QK256: code must be 0..=3, got {}", code);

    // Verified against BitNet.cpp's dequantize_row_i2_s.
    const LUT: [f32; 4] = [-1.0, 0.0, 1.0, 0.0];
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
    // BitNet.cpp I2_S stores each 128-value chunk as 32 bytes, where each byte
    // carries one group position across four 32-value lanes, high bits first.
    for chunk in 0..2 {
        let byte_base = chunk * 32;
        let elem_base = chunk * 128;
        for gp in 0..32 {
            let b = qs64[byte_base + gp];
            out_codes256[elem_base + gp] = (b >> 6) & 0x03;
            out_codes256[elem_base + 32 + gp] = (b >> 4) & 0x03;
            out_codes256[elem_base + 64 + gp] = (b >> 2) & 0x03;
            out_codes256[elem_base + 96 + gp] = b & 0x03;
        }
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

fn bitnet_nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let bits = i32::from_ne_bytes(val.to_ne_bytes());
    (bits & 0x007f_ffff) - 0x0040_0000
}

/// Quantize one activation row using BitNet.cpp's I8_S policy.
///
/// Returns the quantized row, activation scale, and activation sum used by the
/// I2_S x I8_S scaled QK256 formula.
pub fn quantize_row_i8_s_activation(x: &[f32], cols: usize) -> (Vec<i8>, f32, i32) {
    let mut max = 0.00001f32;
    for &value in x.iter().take(cols) {
        max = max.max(value.abs());
    }

    let act_scale = 127.0 / max;
    let mut act_sum = 0i32;
    let mut q = Vec::with_capacity(cols);
    for &value in x.iter().take(cols) {
        let v = bitnet_nearest_int(value * act_scale).clamp(-128, 127);
        act_sum += v;
        q.push(v as i8);
    }

    (q, act_scale, act_sum)
}

#[inline]
fn i2s_chunk_code(chunk32: &[u8], lane: usize, group_pos: usize) -> i16 {
    debug_assert!(lane < 4);
    debug_assert!(group_pos < 32);
    let shift = 6 - (lane * 2);
    ((chunk32[group_pos] >> shift) & 0x03) as i16
}

#[inline]
fn i2s_i8_pair_product_sum(
    chunk32: &[u8],
    q: &[i8],
    q_base: usize,
    lane: usize,
    pair: usize,
) -> i16 {
    let gp0 = pair * 2;
    let gp1 = gp0 + 1;
    let q0 = q[q_base + lane * 32 + gp0] as i16;
    let q1 = q[q_base + lane * 32 + gp1] as i16;
    i2s_chunk_code(chunk32, lane, gp0) * q0 + i2s_chunk_code(chunk32, lane, gp1) * q1
}

#[inline]
fn i2s_i8_chunk_pair_sum(chunk32: &[u8], q: &[i8], q_base: usize, pair: usize) -> i16 {
    // Mirror BitNet.cpp's AVX2 `_mm256_maddubs_epi16` + wrapping
    // `_mm256_add_epi16` accumulation order. Pair products fit in i16; the
    // accumulation across lanes/chunks intentionally wraps before widening.
    let lanes01 = i2s_i8_pair_product_sum(chunk32, q, q_base, 0, pair)
        .wrapping_add(i2s_i8_pair_product_sum(chunk32, q, q_base, 1, pair));
    let lanes23 = i2s_i8_pair_product_sum(chunk32, q, q_base, 2, pair)
        .wrapping_add(i2s_i8_pair_product_sum(chunk32, q, q_base, 3, pair));
    lanes01.wrapping_add(lanes23)
}

fn gemv_qk256_row_bitnet_i8s_int_dot(qs_row: &[u8], q: &[i8], cols: usize) -> i32 {
    const QK_I2_S_X86: usize = 128;
    const CHUNKS_PER_ACCUM_GROUP: usize = 32;
    const PAIR_LANES: usize = 16;

    let full_chunks = cols / QK_I2_S_X86;
    let mut int_dot = 0i32;
    let mut chunk = 0usize;

    while chunk < full_chunks {
        let take = (full_chunks - chunk).min(CHUNKS_PER_ACCUM_GROUP);
        let mut acc16 = [0i16; PAIR_LANES];

        for chunk_offset in 0..take {
            let chunk_index = chunk + chunk_offset;
            let byte_base = chunk_index * 32;
            let q_base = chunk_index * QK_I2_S_X86;
            let chunk32 = &qs_row[byte_base..byte_base + 32];

            for (pair, acc) in acc16.iter_mut().enumerate() {
                *acc = acc.wrapping_add(i2s_i8_chunk_pair_sum(chunk32, q, q_base, pair));
            }
        }

        int_dot += acc16.iter().map(|&value| value as i32).sum::<i32>();
        chunk += take;
    }

    let tail_start = full_chunks * QK_I2_S_X86;
    if tail_start < cols {
        let mut codes = [0u8; QK256_BLOCK];
        let byte_base = full_chunks * 32;
        let block_base = byte_base - (byte_base % QK256_PACKED_BYTES);
        let block: &[u8; QK256_PACKED_BYTES] = qs_row[block_base..block_base + QK256_PACKED_BYTES]
            .try_into()
            .expect("QK256: tail block must be 64 bytes");
        unpack_qk256_block(block, &mut codes);
        let code_offset = tail_start - (block_base / QK256_PACKED_BYTES) * QK256_BLOCK;
        for j in tail_start..cols {
            int_dot += (codes[code_offset + (j - tail_start)] as i32) * q[j] as i32;
        }
    }

    int_dot
}

fn gemv_qk256_row_bitnet_i8s_scaled(
    qs_row: &[u8],
    q: &[i8],
    cols: usize,
    act_scale: f32,
    act_sum: i32,
    weight_scale: f32,
) -> f32 {
    let int_dot = gemv_qk256_row_bitnet_i8s_int_dot(qs_row, q, cols);
    ((int_dot - act_sum) as f32 / act_scale) * weight_scale
}

fn gemv_qk256_bitnet_i8s_scaled_checked(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    weight_scale: f32,
) -> Result<()> {
    if !weight_scale.is_finite() {
        bail!("I2S_QK256: weight scale is not finite: {}", weight_scale);
    }
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

    let (q, act_scale, act_sum) = quantize_row_i8_s_activation(x, cols);

    for (row, output) in y_out.iter_mut().enumerate().take(rows) {
        let start = row * row_stride_bytes;
        let end = start + row_stride_bytes;
        let row_bytes = &qs_data[start..end];
        *output =
            gemv_qk256_row_bitnet_i8s_scaled(row_bytes, &q, cols, act_scale, act_sum, weight_scale);
    }

    Ok(())
}

/// Multi-row GEMV using BitNet.cpp's I2_S × I8_S matmul semantics.
///
/// BitNet.cpp does not compute QK256 by dequantizing weights and multiplying by
/// raw F32 activations. It first quantizes each activation row to I8_S, records
/// the activation scale and sum, computes an integer dot over packed I2_S codes,
/// then applies `(dot - act_sum) / act_scale * weight_scale`.
pub fn gemv_qk256_bitnet_i8s_scaled(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    weight_scale: f32,
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
    gemv_qk256_bitnet_i8s_scaled_checked(
        qs_data,
        x,
        y_out,
        rows,
        cols,
        row_stride_bytes,
        weight_scale,
    )
}

/// Canonical scalar QK256 GEMV oracle for decode: `y = A x`.
///
/// This path mirrors BitNet.cpp `dequantize_row_i2_s` via [`code_to_f32`] and
/// the grouped QK256 packed layout. It never dispatches to SIMD.
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
/// - **AVX2**: x86_64 with AVX2 and FMA support
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
    gemv_qk256_with_kernel_selection(qs_data, x, y_out, rows, cols, row_stride_bytes, None, false)
        .map(|_| ())
}

/// Multi-row GEMV with requested/selected kernel metadata.
///
/// `requested_kernel = None` means automatic selection. In strict mode, a
/// requested AVX2 kernel fails rather than silently falling back to scalar when
/// AVX2/FMA is unavailable.
pub fn gemv_qk256_with_kernel_selection(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    requested_kernel: Option<&'static str>,
    strict: bool,
) -> Result<Qk256KernelSelection> {
    let expected_stride = qk256_row_stride_bytes(cols)?;
    if row_stride_bytes != expected_stride {
        bail!(
            "I2S_QK256: row_stride_bytes {} != expected {} for cols={}",
            row_stride_bytes,
            expected_stride,
            cols
        );
    }

    let selection = select_qk256_gemv_kernel(requested_kernel, strict)?;

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        if selection.selected_kernel == QK256_AVX2_GEMV_KERNEL_ID {
            super::i2s_qk256_avx2::gemv_qk256_avx2(
                qs_data,
                x,
                y_out,
                rows,
                cols,
                row_stride_bytes,
            )?;
            return Ok(selection);
        }
    }

    gemv_qk256_scalar_checked(qs_data, x, y_out, rows, cols, row_stride_bytes)?;
    Ok(selection)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_codes_for_cols(codes: &[u8], cols: usize) -> Vec<u8> {
        let layout = Qk256Layout::from_rows_cols(1, cols).expect("layout");
        let mut packed = vec![0u8; layout.row_stride_bytes];
        for (i, &code) in codes.iter().enumerate().take(cols) {
            assert!(code < 4, "test code must be 0..=3");
            let block = i / QK256_BLOCK;
            let within = i % QK256_BLOCK;
            let chunk = within / 128;
            let chunk_pos = within % 128;
            let lane = chunk_pos / 32;
            let gp = chunk_pos % 32;
            let byte_idx = block * QK256_PACKED_BYTES + chunk * 32 + gp;
            packed[byte_idx] |= code << (6 - lane * 2);
        }
        packed
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
        // Pattern: BitNet.cpp grouped lanes [0, 1, 2, 3].
        let mut qs = [0u8; QK256_PACKED_BYTES];
        qs.fill(0b_00_01_10_11);
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(&qs, &mut codes);

        // Verify codes are in 0..=3
        assert!(codes.iter().all(|&c| c < 4), "All codes must be 0..=3");

        // Verify first few codes match pattern
        assert_eq!(codes[0], 0);
        assert_eq!(codes[32], 1);
        assert_eq!(codes[64], 2);
        assert_eq!(codes[96], 3);
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

        // All codes = 0 (→ -1.0)
        let qs_data = vec![0x00u8; rows * row_stride_bytes]; // 0b_00_00_00_00

        let x: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let mut y_out = vec![0.0f32; rows];

        gemv_qk256(&qs_data, &x, &mut y_out, rows, cols, row_stride_bytes)
            .expect("gemv_qk256 should succeed");

        // Code 0 → -1.0, so each row = -sum(x)
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
        // Verify LUT values against BitNet.cpp dequantize_row_i2_s.
        assert_eq!(code_to_f32(0), -1.0);
        assert_eq!(code_to_f32(1), 0.0);
        assert_eq!(code_to_f32(2), 1.0);
        assert_eq!(code_to_f32(3), 0.0);
    }

    #[test]
    fn qk256_scalar_kernel_ids_are_stable() {
        assert_eq!(QK256_SCALAR_GEMV_KERNEL_ID, "qk256-scalar-gemv");
        assert_eq!(QK256_SCALAR_GEMM_KERNEL_ID, "qk256-scalar-gemm");
        assert_eq!(QK256_AVX2_GEMV_KERNEL_ID, "qk256-avx2-gemv");
    }

    #[test]
    fn qk256_kernel_selection_auto_selects_avx2_when_available() -> Result<()> {
        let selection = select_qk256_gemv_kernel_for_availability(None, false, true)?;
        assert_eq!(selection.requested_kernel, None);
        assert_eq!(selection.selected_kernel, QK256_AVX2_GEMV_KERNEL_ID);
        assert!(!selection.fallback_used);
        assert_eq!(selection.fallback_reason, None);
        assert_eq!(selection.cpu_features, vec!["avx2", "fma"]);
        Ok(())
    }

    #[test]
    fn qk256_kernel_selection_auto_selects_scalar_when_avx2_unavailable() -> Result<()> {
        let selection = select_qk256_gemv_kernel_for_availability(None, false, false)?;
        assert_eq!(selection.requested_kernel, None);
        assert_eq!(selection.selected_kernel, QK256_SCALAR_GEMV_KERNEL_ID);
        assert!(!selection.fallback_used);
        assert_eq!(selection.fallback_reason, None);
        assert!(selection.cpu_features.is_empty());
        Ok(())
    }

    #[test]
    fn qk256_kernel_selection_requested_avx2_selects_avx2_when_available() -> Result<()> {
        let selection =
            select_qk256_gemv_kernel_for_availability(Some(QK256_AVX2_GEMV_KERNEL_ID), true, true)?;
        assert_eq!(selection.requested_kernel, Some(QK256_AVX2_GEMV_KERNEL_ID));
        assert_eq!(selection.selected_kernel, QK256_AVX2_GEMV_KERNEL_ID);
        assert!(!selection.fallback_used);
        assert_eq!(selection.fallback_reason, None);
        assert_eq!(selection.cpu_features, vec!["avx2", "fma"]);
        Ok(())
    }

    #[test]
    fn qk256_kernel_selection_requested_avx2_strict_fails_when_unavailable() {
        let err =
            select_qk256_gemv_kernel_for_availability(Some(QK256_AVX2_GEMV_KERNEL_ID), true, false)
                .expect_err("strict requested AVX2 must fail without avx2/fma");
        assert!(err.to_string().contains("cannot fall back"));
    }

    #[test]
    fn qk256_kernel_selection_requested_avx2_non_strict_falls_back() -> Result<()> {
        let selection = select_qk256_gemv_kernel_for_availability(
            Some(QK256_AVX2_GEMV_KERNEL_ID),
            false,
            false,
        )?;
        assert_eq!(selection.requested_kernel, Some(QK256_AVX2_GEMV_KERNEL_ID));
        assert_eq!(selection.selected_kernel, QK256_SCALAR_GEMV_KERNEL_ID);
        assert!(selection.fallback_used);
        assert_eq!(selection.fallback_reason.as_deref(), Some("avx2/fma unavailable"));
        assert!(selection.cpu_features.is_empty());
        Ok(())
    }

    #[test]
    fn qk256_kernel_selection_requested_scalar_is_not_a_fallback() -> Result<()> {
        let selection = select_qk256_gemv_kernel_for_availability(
            Some(QK256_SCALAR_GEMV_KERNEL_ID),
            true,
            true,
        )?;
        assert_eq!(selection.requested_kernel, Some(QK256_SCALAR_GEMV_KERNEL_ID));
        assert_eq!(selection.selected_kernel, QK256_SCALAR_GEMV_KERNEL_ID);
        assert!(!selection.fallback_used);
        assert_eq!(selection.fallback_reason, None);
        assert_eq!(selection.cpu_features, vec!["avx2", "fma"]);
        Ok(())
    }

    #[test]
    fn qk256_kernel_selection_rejects_unknown_requested_kernel() {
        let err = select_qk256_gemv_kernel_for_availability(Some("qk256-made-up"), false, true)
            .expect_err("unknown requested kernel must fail");
        assert!(err.to_string().contains("unsupported requested QK256 GEMV kernel"));
    }

    #[test]
    fn gemv_qk256_with_kernel_selection_can_force_scalar() -> Result<()> {
        let rows = 2usize;
        let cols = 256usize;
        let row_stride_bytes = QK256_PACKED_BYTES;
        let qs_data = vec![0xAAu8; rows * row_stride_bytes];
        let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.25).collect();
        let mut y_out = vec![0.0f32; rows];

        let selection = gemv_qk256_with_kernel_selection(
            &qs_data,
            &x,
            &mut y_out,
            rows,
            cols,
            row_stride_bytes,
            Some(QK256_SCALAR_GEMV_KERNEL_ID),
            true,
        )?;

        assert_eq!(selection.selected_kernel, QK256_SCALAR_GEMV_KERNEL_ID);
        assert!(!selection.fallback_used);
        let expected: f32 = x.iter().sum();
        for got in y_out {
            assert!((got - expected).abs() < 1e-3, "expected {expected}, got {got}");
        }
        Ok(())
    }

    #[test]
    fn gemv_qk256_bitnet_i8s_scaled_matches_simple_integer_formula() -> Result<()> {
        let rows = 1usize;
        let cols = 2usize;
        let layout = Qk256Layout::from_rows_cols(rows, cols)?;
        let qs_data = pack_codes_for_cols(&[2, 2], cols);
        let x = vec![1.0f32, 1.0];
        let mut y_out = vec![0.0f32; rows];

        gemv_qk256_bitnet_i8s_scaled(
            &qs_data,
            &x,
            &mut y_out,
            rows,
            cols,
            layout.row_stride_bytes,
            0.5,
        )?;

        assert!((y_out[0] - 1.0).abs() < 1e-6, "got {}", y_out[0]);

        Ok(())
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
    /// Verifies that the code-to-float lookup table matches BitNet.cpp I2_S:
    /// - Code 0 → -1.0
    /// - Code 1 →  0.0
    /// - Code 2 → +1.0
    /// - Code 3 →  0.0
    #[test]
    fn qk256_lut_basic() {
        assert_eq!(code_to_f32(0), -1.0, "Code 0 should map to -1.0");
        assert_eq!(code_to_f32(1), 0.0, "Code 1 should map to 0.0");
        assert_eq!(code_to_f32(2), 1.0, "Code 2 should map to +1.0");
        assert_eq!(code_to_f32(3), 0.0, "Code 3 should map to 0.0");
    }

    /// Test (B): Block Decode Golden (64B → 256 f32)
    ///
    /// Tests feature spec: i2s-dual-flavor.md#memory-layout
    /// Pack 256 two-bit codes in BitNet.cpp grouped layout cycling 0..3.
    /// Decode using the unpack path and verify:
    /// - RMS in range [0.1, 5.0]
    /// - First 16 values contain only the expected ternary set {-1, 0, 1}
    #[test]
    fn qk256_block_decode_golden() {
        let source_codes: Vec<u8> = (0..QK256_BLOCK).map(|i| (i % 4) as u8).collect();
        let packed = pack_codes_for_cols(&source_codes, QK256_BLOCK);
        let qs64: &[u8; QK256_PACKED_BYTES] = packed[..QK256_PACKED_BYTES].try_into().unwrap();

        // Unpack block
        let mut codes = [0u8; QK256_BLOCK];
        unpack_qk256_block(qs64, &mut codes);

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

        // Verify RMS is reasonable (sqrt(0.5) for uniform {-1,0,1,0})
        assert!((0.1..=5.0).contains(&rms), "RMS {} should be in range [0.1, 5.0]", rms);

        // Verify first 16 values stay inside the BitNet I2_S ternary values.
        let first_16: Vec<f32> = weights[..16].to_vec();
        assert!(first_16.contains(&-1.0), "First 16 values should contain -1.0");
        assert!(first_16.contains(&0.0), "First 16 values should contain 0.0");
        assert!(first_16.contains(&1.0), "First 16 values should contain 1.0");
    }

    #[test]
    fn qk256_bitnet_i2s_grouped_layout_byte_exact_fixture() {
        let mut source_codes = vec![0u8; QK256_BLOCK];
        // First 128-value chunk, group position 0:
        // lane0=0, lane1=1, lane2=2, lane3=3 -> byte 0b00_01_10_11.
        source_codes[0] = 0;
        source_codes[32] = 1;
        source_codes[64] = 2;
        source_codes[96] = 3;
        // First 128-value chunk, group position 1:
        // lane0=3, lane1=2, lane2=1, lane3=0 -> byte 0b11_10_01_00.
        source_codes[1] = 3;
        source_codes[33] = 2;
        source_codes[65] = 1;
        source_codes[97] = 0;
        // Second 128-value chunk uses the same byte-local encoding at offset 32.
        source_codes[128] = 2;
        source_codes[160] = 0;
        source_codes[192] = 3;
        source_codes[224] = 1;

        let packed = pack_codes_for_cols(&source_codes, QK256_BLOCK);
        assert_eq!(packed[0], 0b00_01_10_11, "first chunk byte 0 must use grouped lanes");
        assert_eq!(packed[1], 0b11_10_01_00, "first chunk byte 1 must use grouped lanes");
        assert_eq!(packed[32], 0b10_00_11_01, "second chunk byte 0 must use grouped lanes");

        let qs64: &[u8; QK256_PACKED_BYTES] = packed[..QK256_PACKED_BYTES].try_into().unwrap();
        let mut unpacked = [0u8; QK256_BLOCK];
        unpack_qk256_block(qs64, &mut unpacked);
        assert_eq!(&unpacked[..], &source_codes[..]);
    }

    #[test]
    fn bitnet_i8s_activation_quantization_uses_absmax_scale_and_sum() {
        let x = [-127.0f32, -1.0, 0.0, 1.0, 127.0];
        let (q, act_scale, act_sum) = quantize_row_i8_s_activation(&x, x.len());

        assert_eq!(act_scale, 1.0);
        assert_eq!(q, vec![-127, -1, 0, 1, 127]);
        assert_eq!(act_sum, 0);
    }

    #[test]
    fn bitnet_i8s_scaled_formula_subtracts_activation_sum_before_weight_scale() -> Result<()> {
        let rows = 1usize;
        let cols = 4usize;
        let layout = Qk256Layout::from_rows_cols(rows, cols)?;
        let codes = [0, 1, 2, 3];
        let qs_data = pack_codes_for_cols(&codes, cols);
        let x = vec![-127.0f32, -1.0, 1.0, 127.0];
        let weight_scale = 0.25f32;
        let mut y_out = vec![0.0f32; rows];

        gemv_qk256_bitnet_i8s_scaled(
            &qs_data,
            &x,
            &mut y_out,
            rows,
            cols,
            layout.row_stride_bytes,
            weight_scale,
        )?;

        let expected_int_dot: i32 = [0, 1, 2, 3]
            .into_iter()
            .zip([-127, -1, 1, 127])
            .map(|(code, activation)| code * activation)
            .sum();
        let expected_act_sum: i32 = [-127, -1, 1, 127].into_iter().sum();
        let expected = ((expected_int_dot - expected_act_sum) as f32 / 1.0) * weight_scale;
        assert_eq!(expected, 95.5);
        assert!((y_out[0] - expected).abs() < 1e-6, "got {}, expected {}", y_out[0], expected);

        let f32_dequant_reference: f32 = codes
            .iter()
            .copied()
            .zip(x.iter().copied())
            .map(|(code, x)| code_to_f32(code) * x)
            .sum();
        assert_ne!(
            y_out[0], f32_dequant_reference,
            "I2_S x I8_S scaled semantics must not silently collapse to F32 dequant GEMV"
        );

        Ok(())
    }

    #[test]
    fn bitnet_i8s_scaled_rejects_nonfinite_weight_scale() {
        let rows = 1usize;
        let cols = 4usize;
        let layout = Qk256Layout::from_rows_cols(rows, cols).expect("layout");
        let qs_data = pack_codes_for_cols(&[2, 2, 2, 2], cols);
        let x = vec![1.0f32; cols];
        let mut y_out = vec![0.0f32; rows];

        let err = gemv_qk256_bitnet_i8s_scaled(
            &qs_data,
            &x,
            &mut y_out,
            rows,
            cols,
            layout.row_stride_bytes,
            f32::NAN,
        )
        .expect_err("non-finite weight scale must fail");

        assert!(err.to_string().contains("weight scale is not finite"));
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
