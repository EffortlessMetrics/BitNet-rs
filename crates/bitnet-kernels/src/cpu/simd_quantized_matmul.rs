//! SIMD-accelerated quantized matrix multiplication kernels.
//!
//! Provides INT2×INT8, INT4×INT8, block-quantized, fused dequant-matmul,
//! tiled, mixed-precision, batched, and sparse quantized matmul routines
//! with runtime AVX2 detection and portable scalar fallback.
//!
//! # Layout conventions
//!
//! * Activations are **row-major** `[m, k]`.
//! * INT2 weights use I2_S column-major packing (4 values/byte, 2 bits
//!   each, LSB-first) identical to [`super::quantized_matmul`].
//! * INT4 weights pack 2 values per byte (low nibble first), column-major.
//! * Scales have one entry per quantization block per output column:
//!   `n * ceil(k / block_size)` entries.
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};

// ── INT2 helpers ───────────────────────────────────────────────────────

/// Decode a 2-bit I2_S code to its signed value.
#[inline(always)]
fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

/// Pack four ternary values into one byte, LSB-first.
pub fn pack_i2s(vals: [i8; 4]) -> u8 {
    let mut byte = 0u8;
    for (i, &v) in vals.iter().enumerate() {
        let code: u8 = match v {
            1 => 0b01,
            -1 => 0b11,
            _ => 0b00,
        };
        byte |= code << (i * 2);
    }
    byte
}

// ── INT4 helpers ───────────────────────────────────────────────────────

/// Decode the low nibble of a byte to a signed 4-bit value (range -8..7).
#[inline(always)]
fn decode_int4_lo(byte: u8) -> i8 {
    let raw = (byte & 0x0F) as i8;
    if raw >= 8 { raw - 16 } else { raw }
}

/// Decode the high nibble of a byte to a signed 4-bit value (range -8..7).
#[inline(always)]
fn decode_int4_hi(byte: u8) -> i8 {
    let raw = ((byte >> 4) & 0x0F) as i8;
    if raw >= 8 { raw - 16 } else { raw }
}

/// Pack two signed 4-bit values into one byte (low nibble first).
pub fn pack_int4(lo: i8, hi: i8) -> u8 {
    let lo_u = (lo as u8) & 0x0F;
    let hi_u = (hi as u8) & 0x0F;
    lo_u | (hi_u << 4)
}

// ── Sparse representation ──────────────────────────────────────────────

/// CSR (Compressed Sparse Row) representation for quantized weights.
#[derive(Debug, Clone)]
pub struct SparseQuantizedWeights {
    /// Non-zero weight values (ternary: -1, 0, +1 for INT2).
    pub values: Vec<i8>,
    /// Column indices for each non-zero value.
    pub col_indices: Vec<usize>,
    /// Row pointers: `row_ptrs[i]..row_ptrs[i+1]` spans row `i`.
    pub row_ptrs: Vec<usize>,
    /// One scale per block per column: `n * ceil(k / block_size)`.
    pub scales: Vec<f32>,
    /// Number of rows (k dimension).
    pub num_rows: usize,
    /// Number of columns (n dimension).
    pub num_cols: usize,
    /// Block size for quantization.
    pub block_size: usize,
}

// ── Validation ─────────────────────────────────────────────────────────

fn validate_dims(m: usize, n: usize, k: usize) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("dimensions must be > 0: m={m}, n={n}, k={k}"),
        }));
    }
    Ok(())
}

fn validate_block_size(block_size: usize) -> Result<()> {
    if block_size == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "block_size must be > 0".into(),
        }));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_i2s_args(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &[f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_dims(m, n, k)?;
    validate_block_size(block_size)?;

    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    if activations.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("activations too small: need {}, got {}", m * k, activations.len()),
        }));
    }
    if weights_packed.len() < packed_k * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "weights_packed too small: need {}, got {}",
                packed_k * n,
                weights_packed.len()
            ),
        }));
    }
    if scales.len() < n * num_blocks_k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("scales too small: need {}, got {}", n * num_blocks_k, scales.len()),
        }));
    }
    if out.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output too small: need {}, got {}", m * n, out.len()),
        }));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_int4_args(
    activations: &[i8],
    weights_packed: &[u8],
    scales: &[f32],
    out: &[f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_dims(m, n, k)?;
    validate_block_size(block_size)?;

    let packed_k = k.div_ceil(2);
    let num_blocks_k = k.div_ceil(block_size);

    if activations.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("activations too small: need {}, got {}", m * k, activations.len()),
        }));
    }
    if weights_packed.len() < packed_k * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "weights_packed too small: need {}, got {}",
                packed_k * n,
                weights_packed.len()
            ),
        }));
    }
    if scales.len() < n * num_blocks_k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("scales too small: need {}, got {}", n * num_blocks_k, scales.len()),
        }));
    }
    if out.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output too small: need {}, got {}", m * n, out.len()),
        }));
    }
    Ok(())
}

// ── Runtime dispatch ───────────────────────────────────────────────────

#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ── 1. INT2 × INT8 matmul with SIMD accumulation ──────────────────────

/// INT2 weights × INT8 activations with SIMD accumulation.
///
/// Computes `out[m×n] = activations_i8[m×k] · dequant(weights_i2[k×n], scales)`.
///
/// Activations are signed 8-bit integers. Weights are I2_S packed (4 per
/// byte). The inner dot-product uses integer accumulation before a final
/// f32 scale multiply, preserving precision for integer inputs.
#[allow(clippy::too_many_arguments)]
pub fn int2_int8_matmul(
    activations: &[i8],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_dims(m, n, k)?;
    validate_block_size(block_size)?;

    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    if activations.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("activations too small: need {}, got {}", m * k, activations.len()),
        }));
    }
    if weights_packed.len() < packed_k * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "weights_packed too small: need {}, got {}",
                packed_k * n,
                weights_packed.len()
            ),
        }));
    }
    if scales.len() < n * num_blocks_k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("scales too small: need {}, got {}", n * num_blocks_k, scales.len()),
        }));
    }
    if out.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output too small: need {}, got {}", m * n, out.len()),
        }));
    }

    out[..m * n].fill(0.0);

    for row in 0..m {
        let a_row = &activations[row * k..(row + 1) * k];
        for col in 0..n {
            let mut acc_f32 = 0.0f32;
            for blk in 0..num_blocks_k {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks_k + blk];

                let mut acc_i32 = 0i32;
                for (idx, &a_val) in a_row.iter().enumerate().take(blk_end).skip(blk_start) {
                    let byte_idx = col * packed_k + idx / 4;
                    let bit_off = (idx % 4) * 2;
                    let w = decode_i2s((weights_packed[byte_idx] >> bit_off) & 0x03);
                    acc_i32 += a_val as i32 * w as i32;
                }
                acc_f32 += acc_i32 as f32 * scale;
            }
            out[row * n + col] = acc_f32;
        }
    }
    Ok(())
}

// ── 2. INT4 × INT8 matmul with nibble unpacking ───────────────────────

/// INT4 weights × INT8 activations with nibble unpacking.
///
/// Weights are packed 2 per byte (low nibble first, signed 4-bit range
/// -8..7). Activations are signed 8-bit integers. Scales are per-block.
#[allow(clippy::too_many_arguments)]
pub fn int4_int8_matmul(
    activations: &[i8],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_int4_args(activations, weights_packed, scales, out, m, n, k, block_size)?;

    let packed_k = k.div_ceil(2);
    let num_blocks_k = k.div_ceil(block_size);

    out[..m * n].fill(0.0);

    for row in 0..m {
        let a_row = &activations[row * k..(row + 1) * k];
        for col in 0..n {
            let mut acc_f32 = 0.0f32;
            for blk in 0..num_blocks_k {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks_k + blk];

                let mut acc_i32 = 0i32;
                for (idx, &a_val) in a_row.iter().enumerate().take(blk_end).skip(blk_start) {
                    let byte_idx = col * packed_k + idx / 2;
                    let w = if idx % 2 == 0 {
                        decode_int4_lo(weights_packed[byte_idx])
                    } else {
                        decode_int4_hi(weights_packed[byte_idx])
                    };
                    acc_i32 += a_val as i32 * w as i32;
                }
                acc_f32 += acc_i32 as f32 * scale;
            }
            out[row * n + col] = acc_f32;
        }
    }
    Ok(())
}

// ── 3. Block-quantized matmul ──────────────────────────────────────────

/// Block-quantized I2_S matmul that processes quantization blocks together.
///
/// Structures the outer loop over blocks so that all rows accumulate
/// contributions from the same block before moving to the next. This
/// layout is friendlier to SIMD tiling since the weight block is decoded
/// once and reused across all rows.
#[allow(clippy::too_many_arguments)]
pub fn block_quantized_matmul(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_i2s_args(activations, weights_packed, scales, out, m, n, k, block_size)?;

    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    out[..m * n].fill(0.0);

    for blk in 0..num_blocks_k {
        let blk_start = blk * block_size;
        let blk_end = (blk_start + block_size).min(k);
        let blk_len = blk_end - blk_start;

        for col in 0..n {
            let scale = scales[col * num_blocks_k + blk];

            // Decode block weights once.
            let mut w_blk = vec![0i8; blk_len];
            for (i, idx) in (blk_start..blk_end).enumerate() {
                let byte_idx = col * packed_k + idx / 4;
                let bit_off = (idx % 4) * 2;
                w_blk[i] = decode_i2s((weights_packed[byte_idx] >> bit_off) & 0x03);
            }

            if has_avx2() {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    block_accum_avx2(
                        activations,
                        &w_blk,
                        out,
                        scale,
                        m,
                        n,
                        k,
                        col,
                        blk_start,
                        blk_len,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                block_accum_scalar(
                    activations,
                    &w_blk,
                    out,
                    scale,
                    m,
                    n,
                    k,
                    col,
                    blk_start,
                    blk_len,
                );
            } else {
                block_accum_scalar(
                    activations,
                    &w_blk,
                    out,
                    scale,
                    m,
                    n,
                    k,
                    col,
                    blk_start,
                    blk_len,
                );
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn block_accum_scalar(
    activations: &[f32],
    w_blk: &[i8],
    out: &mut [f32],
    scale: f32,
    m: usize,
    n: usize,
    k: usize,
    col: usize,
    blk_start: usize,
    blk_len: usize,
) {
    for row in 0..m {
        let a_base = row * k + blk_start;
        let mut acc = 0.0f32;
        for i in 0..blk_len {
            acc += activations[a_base + i] * w_blk[i] as f32;
        }
        out[row * n + col] += acc * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn block_accum_avx2(
    activations: &[f32],
    w_blk: &[i8],
    out: &mut [f32],
    scale: f32,
    m: usize,
    n: usize,
    k: usize,
    col: usize,
    blk_start: usize,
    blk_len: usize,
) {
    use std::arch::x86_64::*;

    let scale_v = _mm256_set1_ps(scale);
    for row in 0..m {
        let a_base = row * k + blk_start;
        let mut acc = _mm256_setzero_ps();
        let mut r = 0usize;
        while r + 8 <= blk_len {
            let av = _mm256_loadu_ps(activations.as_ptr().add(a_base + r));
            let wi = _mm256_set_epi32(
                *w_blk.get_unchecked(r + 7) as i32,
                *w_blk.get_unchecked(r + 6) as i32,
                *w_blk.get_unchecked(r + 5) as i32,
                *w_blk.get_unchecked(r + 4) as i32,
                *w_blk.get_unchecked(r + 3) as i32,
                *w_blk.get_unchecked(r + 2) as i32,
                *w_blk.get_unchecked(r + 1) as i32,
                *w_blk.get_unchecked(r) as i32,
            );
            let wf = _mm256_cvtepi32_ps(wi);
            acc = _mm256_fmadd_ps(av, wf, acc);
            r += 8;
        }
        acc = _mm256_mul_ps(acc, scale_v);
        let mut sum = hsum_avx2(acc);
        for r2 in r..blk_len {
            sum += activations[a_base + r2] * w_blk[r2] as f32 * scale;
        }
        *out.get_unchecked_mut(row * n + col) += sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

// ── 4. Fused dequant-matmul ────────────────────────────────────────────

/// Fused dequantization and matrix multiplication.
///
/// Combines I2_S weight dequantization with the dot product in a single
/// pass, avoiding a separate dequantized weight buffer.
#[allow(clippy::too_many_arguments)]
pub fn fused_dequant_matmul(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_i2s_args(activations, weights_packed, scales, out, m, n, k, block_size)?;

    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    out[..m * n].fill(0.0);

    for row in 0..m {
        let a_row = &activations[row * k..(row + 1) * k];
        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..num_blocks_k {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks_k + blk];

                let mut blk_acc = 0.0f32;
                for (idx, &a_val) in a_row.iter().enumerate().take(blk_end).skip(blk_start) {
                    let byte_idx = col * packed_k + idx / 4;
                    let bit_off = (idx % 4) * 2;
                    let w = decode_i2s((weights_packed[byte_idx] >> bit_off) & 0x03) as f32;
                    blk_acc += a_val * w;
                }
                acc += blk_acc * scale;
            }
            out[row * n + col] = acc;
        }
    }
    Ok(())
}

// ── 5. Tiled matmul for cache efficiency ───────────────────────────────

/// Tile configuration for quantized matmul.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedTileConfig {
    /// Tile rows (M dimension). Must be > 0.
    pub tile_m: usize,
    /// Tile columns (N dimension). Must be > 0.
    pub tile_n: usize,
}

impl QuantizedTileConfig {
    /// Sensible defaults for L1d cache efficiency.
    pub const DEFAULT: Self = Self { tile_m: 32, tile_n: 32 };

    /// Smaller tiles for constrained caches.
    pub const SMALL: Self = Self { tile_m: 16, tile_n: 16 };

    /// Create a custom tile configuration.
    pub fn new(tile_m: usize, tile_n: usize) -> Self {
        Self { tile_m, tile_n }
    }
}

impl Default for QuantizedTileConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Cache-tiled I2_S quantized matmul.
///
/// Semantics identical to [`fused_dequant_matmul`]; tiles the M and N
/// dimensions for improved L1/L2 cache utilisation on large matrices.
#[allow(clippy::too_many_arguments)]
pub fn tiled_quantized_matmul(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
    tiles: &QuantizedTileConfig,
) -> Result<()> {
    validate_i2s_args(activations, weights_packed, scales, out, m, n, k, block_size)?;

    if tiles.tile_m == 0 || tiles.tile_n == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "tile dimensions must be > 0".into(),
        }));
    }

    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    out[..m * n].fill(0.0);

    for ii in (0..m).step_by(tiles.tile_m) {
        let i_end = (ii + tiles.tile_m).min(m);
        for jj in (0..n).step_by(tiles.tile_n) {
            let j_end = (jj + tiles.tile_n).min(n);

            for row in ii..i_end {
                let a_row = &activations[row * k..(row + 1) * k];
                for col in jj..j_end {
                    let mut acc = 0.0f32;
                    for blk in 0..num_blocks_k {
                        let blk_start = blk * block_size;
                        let blk_end = (blk_start + block_size).min(k);
                        let scale = scales[col * num_blocks_k + blk];

                        let mut blk_acc = 0.0f32;
                        for (idx, &a_val) in a_row.iter().enumerate().take(blk_end).skip(blk_start)
                        {
                            let byte_idx = col * packed_k + idx / 4;
                            let bit_off = (idx % 4) * 2;
                            let w = decode_i2s((weights_packed[byte_idx] >> bit_off) & 0x03) as f32;
                            blk_acc += a_val * w;
                        }
                        acc += blk_acc * scale;
                    }
                    out[row * n + col] = acc;
                }
            }
        }
    }
    Ok(())
}

// ── 6. Mixed precision matmul ──────────────────────────────────────────

/// Mixed-precision matmul: INT2 weights × FP32 activations with bias.
///
/// Weights are I2_S packed. Activations are f32. This is the most common
/// inference path where the model weights are quantised but the running
/// activations stay in full precision. An optional per-column bias is
/// added after the matmul.
#[allow(clippy::too_many_arguments)]
pub fn mixed_precision_matmul(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_i2s_args(activations, weights_packed, scales, out, m, n, k, block_size)?;

    if let Some(b) = bias
        && b.len() < n
    {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("bias too small: need {n}, got {}", b.len()),
        }));
    }

    fused_dequant_matmul(activations, weights_packed, scales, out, m, n, k, block_size)?;

    if let Some(b) = bias {
        for row in 0..m {
            for col in 0..n {
                out[row * n + col] += b[col];
            }
        }
    }
    Ok(())
}

// ── 7. Batch matmul for multi-head attention ───────────────────────────

/// Batched quantized matmul for multi-head attention.
///
/// Runs `fused_dequant_matmul` independently for each head, where all
/// heads share the same `weights_packed` and `scales` but have separate
/// activation slices of shape `[m, k]` and output slices `[m, n]`.
///
/// `activations` is `[num_heads, m, k]` row-major.
/// `out` is `[num_heads, m, n]` row-major.
#[allow(clippy::too_many_arguments)]
pub fn batched_quantized_matmul(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    num_heads: usize,
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<()> {
    validate_dims(m, n, k)?;
    validate_block_size(block_size)?;

    if num_heads == 0 {
        return Ok(());
    }

    let act_stride = m * k;
    let out_stride = m * n;

    if activations.len() < num_heads * act_stride {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "activations too small: need {}, got {}",
                num_heads * act_stride,
                activations.len()
            ),
        }));
    }
    if out.len() < num_heads * out_stride {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output too small: need {}, got {}", num_heads * out_stride, out.len()),
        }));
    }

    for head in 0..num_heads {
        let act_slice = &activations[head * act_stride..(head + 1) * act_stride];
        let out_slice = &mut out[head * out_stride..(head + 1) * out_stride];
        fused_dequant_matmul(act_slice, weights_packed, scales, out_slice, m, n, k, block_size)?;
    }
    Ok(())
}

// ── 8. Sparse quantized matmul ─────────────────────────────────────────

/// Sparse quantized matmul using CSR weights.
///
/// For pruned weight matrices where a significant fraction of weights
/// are zero, this avoids the multiply-by-zero overhead by iterating
/// only non-zero entries.
///
/// `activations` is row-major `[m, k]`, `out` is row-major `[m, n]`.
pub fn sparse_quantized_matmul(
    activations: &[f32],
    weights: &SparseQuantizedWeights,
    out: &mut [f32],
    m: usize,
) -> Result<()> {
    let n = weights.num_cols;
    let k = weights.num_rows;
    validate_dims(m, n, k)?;

    if activations.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("activations too small: need {}, got {}", m * k, activations.len()),
        }));
    }
    if out.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output too small: need {}, got {}", m * n, out.len()),
        }));
    }
    if weights.row_ptrs.len() != k + 1 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "row_ptrs length mismatch: expected {}, got {}",
                k + 1,
                weights.row_ptrs.len()
            ),
        }));
    }

    let num_blocks_k = k.div_ceil(weights.block_size);

    out[..m * n].fill(0.0);

    for row in 0..m {
        let a_row = &activations[row * k..row * k + k];
        for (w_row, &a_val) in a_row.iter().enumerate().take(k) {
            let blk = w_row / weights.block_size;
            if a_val == 0.0 {
                continue;
            }
            let start = weights.row_ptrs[w_row];
            let end = weights.row_ptrs[w_row + 1];
            for nz in start..end {
                let col = weights.col_indices[nz];
                let w_val = weights.values[nz] as f32;
                let scale = weights.scales[col * num_blocks_k + blk];
                out[row * n + col] += a_val * w_val * scale;
            }
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    fn naive_matmul(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * w[l * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn pack_i2s_weights(
        weights: &[i8],
        k: usize,
        n: usize,
        block_size: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        let packed_k = k.div_ceil(4);
        let num_blocks_k = k.div_ceil(block_size);
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let val = weights[row * n + col];
                let code: u8 = match val {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                let byte_idx = col * packed_k + row / 4;
                let bit_off = (row % 4) * 2;
                packed[byte_idx] |= code << bit_off;
            }
        }
        let scales = vec![1.0f32; n * num_blocks_k];
        (packed, scales)
    }

    fn pack_int4_weights(
        weights: &[i8],
        k: usize,
        n: usize,
        block_size: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        let packed_k = k.div_ceil(2);
        let num_blocks_k = k.div_ceil(block_size);
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let val = weights[row * n + col];
                let byte_idx = col * packed_k + row / 2;
                if row % 2 == 0 {
                    packed[byte_idx] |= (val as u8) & 0x0F;
                } else {
                    packed[byte_idx] |= ((val as u8) & 0x0F) << 4;
                }
            }
        }
        let scales = vec![1.0f32; n * num_blocks_k];
        (packed, scales)
    }

    fn make_sparse(
        weights: &[i8],
        k: usize,
        n: usize,
        block_size: usize,
    ) -> SparseQuantizedWeights {
        let num_blocks_k = k.div_ceil(block_size);
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = Vec::with_capacity(k + 1);
        for row in 0..k {
            row_ptrs.push(values.len());
            for col in 0..n {
                let v = weights[row * n + col];
                if v != 0 {
                    values.push(v);
                    col_indices.push(col);
                }
            }
        }
        row_ptrs.push(values.len());
        let scales = vec![1.0f32; n * num_blocks_k];
        SparseQuantizedWeights {
            values,
            col_indices,
            row_ptrs,
            scales,
            num_rows: k,
            num_cols: n,
            block_size,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 1. INT2 × INT8 matmul
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_int2_int8_identity() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_i2s_weights(&w, 2, 2, 32);
        let act: Vec<i8> = vec![3, -2, 5, 7];
        let mut out = vec![0.0f32; 4];
        int2_int8_matmul(&act, &packed, &scales, &mut out, 2, 2, 2, 32).unwrap();
        assert_close(&out, &[3.0, -2.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_int2_int8_all_ones() {
        let (m, n, k, bs) = (4, 4, 4, 32);
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<i8> = (0..m * k).map(|i| (i % 10) as i8).collect();
        let mut out = vec![0.0f32; m * n];
        int2_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        let expected: Vec<f32> = (0..m)
            .flat_map(|row| {
                let sum: i32 = act[row * k..(row + 1) * k].iter().map(|&v| v as i32).sum();
                vec![sum as f32; n]
            })
            .collect();
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn test_int2_int8_all_neg_ones() {
        let (m, n, k, bs) = (3, 3, 4, 32);
        let w = vec![-1i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act = vec![1i8; m * k];
        let mut out = vec![0.0f32; m * n];
        int2_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &vec![-4.0f32; m * n], 1e-6);
    }

    #[test]
    fn test_int2_int8_zero_weights() {
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w = vec![0i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act = vec![42i8; m * k];
        let mut out = vec![0.0f32; m * n];
        int2_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &vec![0.0f32; m * n], 1e-6);
    }

    #[test]
    fn test_int2_int8_1x1() {
        let (packed, scales) = pack_i2s_weights(&[1], 1, 1, 32);
        let mut out = vec![0.0f32; 1];
        int2_int8_matmul(&[7], &packed, &scales, &mut out, 1, 1, 1, 32).unwrap();
        assert_close(&out, &[7.0], 1e-6);
    }

    #[test]
    fn test_int2_int8_non_aligned_k() {
        let (m, n, k, bs) = (2, 2, 5, 32);
        let w: Vec<i8> = vec![1, -1, 0, 1, -1, -1, 0, 1, 1, 0];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut out = vec![0.0f32; m * n];
        int2_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        // col0: [1,0,-1,0,1], col1: [-1,1,-1,1,0]
        // row0=[1,2,3,4,5]: col0=1-3+5=3, col1=-1+2-3+4=2
        // row1=[6,7,8,9,10]: col0=6-8+10=8, col1=-6+7-8+9=2
        assert_close(&out, &[3.0, 2.0, 8.0, 2.0], 1e-6);
    }

    #[test]
    fn test_int2_int8_with_scales() {
        let (m, n, k, bs) = (1usize, 2usize, 4usize, 32usize);
        let _w = vec![1i8; k * n];
        let packed_k = k.div_ceil(4);
        let num_blocks_k = k.div_ceil(bs);
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let byte_idx = col * packed_k + row / 4;
                let bit_off = (row % 4) * 2;
                packed[byte_idx] |= 0b01 << bit_off;
            }
        }
        let mut scales = vec![0.0f32; n * num_blocks_k];
        scales[0] = 2.0;
        scales[1] = 0.5;
        let act = vec![1i8; m * k];
        let mut out = vec![0.0f32; m * n];
        int2_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &[8.0, 2.0], 1e-5);
    }

    #[test]
    fn test_int2_int8_medium() {
        let (m, n, k, bs) = (8, 6, 16, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1][i % 4]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<i8> = (0..m * k).map(|i| ((i % 7) as i8) - 3).collect();
        let mut out = vec![0.0f32; m * n];
        int2_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        let act_f32: Vec<f32> = act.iter().map(|&v| v as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act_f32, &w_f32, m, n, k);
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_int2_int8_dim_zero_rejected() {
        let mut out = vec![0.0f32; 4];
        assert!(int2_int8_matmul(&[1], &[0], &[1.0], &mut out, 0, 1, 1, 32).is_err());
    }

    #[test]
    fn test_int2_int8_block_size_zero_rejected() {
        let mut out = vec![0.0f32; 1];
        assert!(int2_int8_matmul(&[1], &[0], &[1.0], &mut out, 1, 1, 1, 0).is_err());
    }

    #[test]
    fn test_int2_int8_output_too_small() {
        let (packed, scales) = pack_i2s_weights(&[1, 0, 0, 1], 2, 2, 32);
        let mut out = vec![0.0f32; 1];
        assert!(int2_int8_matmul(&[1, 2, 3, 4], &packed, &scales, &mut out, 2, 2, 2, 32).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2. INT4 × INT8 matmul
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_int4_int8_identity() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_int4_weights(&w, 2, 2, 32);
        let act: Vec<i8> = vec![3, -2, 5, 7];
        let mut out = vec![0.0f32; 4];
        int4_int8_matmul(&act, &packed, &scales, &mut out, 2, 2, 2, 32).unwrap();
        assert_close(&out, &[3.0, -2.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_int4_int8_all_ones() {
        let (m, n, k, bs) = (3, 3, 4, 32);
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_int4_weights(&w, k, n, bs);
        let act = vec![2i8; m * k];
        let mut out = vec![0.0f32; m * n];
        int4_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &vec![8.0f32; m * n], 1e-6);
    }

    #[test]
    fn test_int4_int8_negative_weights() {
        let (m, n, k, bs) = (2, 2, 4, 32);
        let w: Vec<i8> = vec![-3, 2, 1, -4, 5, -1, -2, 3];
        let (packed, scales) = pack_int4_weights(&w, k, n, bs);
        let act: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut out = vec![0.0f32; m * n];
        int4_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        let act_f32: Vec<f32> = act.iter().map(|&v| v as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act_f32, &w_f32, m, n, k);
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_int4_int8_zero_weights() {
        let (m, n, k, bs) = (2, 3, 6, 32);
        let w = vec![0i8; k * n];
        let (packed, scales) = pack_int4_weights(&w, k, n, bs);
        let act = vec![5i8; m * k];
        let mut out = vec![0.0f32; m * n];
        int4_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &vec![0.0f32; m * n], 1e-6);
    }

    #[test]
    fn test_int4_int8_1x1() {
        let (packed, scales) = pack_int4_weights(&[3], 1, 1, 32);
        let mut out = vec![0.0f32; 1];
        int4_int8_matmul(&[4], &packed, &scales, &mut out, 1, 1, 1, 32).unwrap();
        assert_close(&out, &[12.0], 1e-6);
    }

    #[test]
    fn test_int4_int8_odd_k() {
        let (m, n, k, bs) = (2, 2, 3, 32);
        let w: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let (packed, scales) = pack_int4_weights(&w, k, n, bs);
        let act: Vec<i8> = vec![1, 1, 1, 2, 2, 2];
        let mut out = vec![0.0f32; m * n];
        int4_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        let act_f32: Vec<f32> = act.iter().map(|&v| v as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act_f32, &w_f32, m, n, k);
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_int4_int8_with_scales() {
        let (m, n, k, bs) = (1, 2, 4, 32);
        let w = vec![1i8; k * n];
        let (packed, _) = pack_int4_weights(&w, k, n, bs);
        let num_blocks_k = k.div_ceil(bs);
        let mut scales = vec![0.0f32; n * num_blocks_k];
        scales[0] = 3.0;
        scales[1] = 0.25;
        let act = vec![2i8; m * k];
        let mut out = vec![0.0f32; m * n];
        int4_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &[24.0, 2.0], 1e-5);
    }

    #[test]
    fn test_int4_int8_medium() {
        let (m, n, k, bs) = (8, 4, 16, 32);
        let w: Vec<i8> = (0..k * n).map(|i| ((i % 7) as i8) - 3).collect();
        let (packed, scales) = pack_int4_weights(&w, k, n, bs);
        let act: Vec<i8> = (0..m * k).map(|i| ((i % 5) as i8) - 2).collect();
        let mut out = vec![0.0f32; m * n];
        int4_int8_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        let act_f32: Vec<f32> = act.iter().map(|&v| v as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act_f32, &w_f32, m, n, k);
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_int4_int8_dim_zero_rejected() {
        let mut out = vec![0.0f32; 4];
        assert!(int4_int8_matmul(&[1], &[0], &[1.0], &mut out, 0, 1, 1, 32).is_err());
    }

    #[test]
    fn test_int4_int8_block_size_zero_rejected() {
        let mut out = vec![0.0f32; 1];
        assert!(int4_int8_matmul(&[1], &[0], &[1.0], &mut out, 1, 1, 1, 0).is_err());
    }

    #[test]
    fn test_int4_nibble_roundtrip() {
        for v in -8i8..8 {
            let byte = pack_int4(v, 0);
            assert_eq!(decode_int4_lo(byte), v, "lo roundtrip failed for {v}");
            let byte2 = pack_int4(0, v);
            assert_eq!(decode_int4_hi(byte2), v, "hi roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_int4_pack_both_nibbles() {
        let byte = pack_int4(3, -4);
        assert_eq!(decode_int4_lo(byte), 3);
        assert_eq!(decode_int4_hi(byte), -4);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3. Block-quantized matmul
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_block_quantized_identity() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_i2s_weights(&w, 2, 2, 32);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let mut out = vec![0.0f32; 4];
        block_quantized_matmul(&act, &packed, &scales, &mut out, 2, 2, 2, 32).unwrap();
        assert_close(&out, &[3.0, -2.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_block_quantized_all_ones() {
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        block_quantized_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_block_quantized_neg_ones() {
        let (m, n, k, bs) = (3, 3, 4, 32);
        let w = vec![-1i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act = vec![1.0f32; m * k];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        block_quantized_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_block_quantized_zeros() {
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w = vec![0i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act = vec![99.0f32; m * k];
        let mut out = vec![0.0f32; m * n];
        block_quantized_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &vec![0.0f32; m * n], 1e-6);
    }

    #[test]
    fn test_block_quantized_1x1() {
        let (packed, scales) = pack_i2s_weights(&[-1], 1, 1, 32);
        let mut out = vec![0.0f32; 1];
        block_quantized_matmul(&[5.0], &packed, &scales, &mut out, 1, 1, 1, 32).unwrap();
        assert_close(&out, &[-5.0], 1e-6);
    }

    #[test]
    fn test_block_quantized_non_aligned_k() {
        let (m, n, k, bs) = (3, 2, 7, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1][i % 3]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        block_quantized_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_block_quantized_block256() {
        let (m, n, k, bs) = (4, 4, 32, 256);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        block_quantized_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_block_quantized_medium() {
        let (m, n, k, bs) = (16, 8, 48, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1, -1][i % 5]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        block_quantized_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_block_quantized_agrees_with_fused() {
        let (m, n, k, bs) = (8, 6, 24, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let mut out_block = vec![0.0f32; m * n];
        let mut out_fused = vec![0.0f32; m * n];
        block_quantized_matmul(&act, &packed, &scales, &mut out_block, m, n, k, bs).unwrap();
        fused_dequant_matmul(&act, &packed, &scales, &mut out_fused, m, n, k, bs).unwrap();
        assert_close(&out_block, &out_fused, 1e-4);
    }

    #[test]
    fn test_block_quantized_dim_zero_rejected() {
        let mut out = vec![0.0f32; 4];
        assert!(block_quantized_matmul(&[1.0], &[0], &[1.0], &mut out, 0, 1, 1, 32).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 4. Fused dequant-matmul
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_fused_identity() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_i2s_weights(&w, 2, 2, 32);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let mut out = vec![0.0f32; 4];
        fused_dequant_matmul(&act, &packed, &scales, &mut out, 2, 2, 2, 32).unwrap();
        assert_close(&out, &[3.0, -2.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_fused_all_ones() {
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        fused_dequant_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_fused_neg_ones() {
        let (m, n, k, bs) = (3, 3, 4, 32);
        let w = vec![-1i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act = vec![1.0f32; m * k];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        fused_dequant_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_fused_zero_weights() {
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w = vec![0i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act = vec![99.0f32; m * k];
        let mut out = vec![0.0f32; m * n];
        fused_dequant_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &vec![0.0f32; m * n], 1e-6);
    }

    #[test]
    fn test_fused_1x1() {
        let (packed, scales) = pack_i2s_weights(&[1], 1, 1, 32);
        let mut out = vec![0.0f32; 1];
        fused_dequant_matmul(&[7.5], &packed, &scales, &mut out, 1, 1, 1, 32).unwrap();
        assert_close(&out, &[7.5], 1e-6);
    }

    #[test]
    fn test_fused_with_scales() {
        let (m, n, k, bs) = (1usize, 2usize, 4usize, 32usize);
        let packed_k = k.div_ceil(4);
        let num_blocks_k = k.div_ceil(bs);
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let byte_idx = col * packed_k + row / 4;
                let bit_off = (row % 4) * 2;
                packed[byte_idx] |= 0b01 << bit_off;
            }
        }
        let mut scales = vec![0.0f32; n * num_blocks_k];
        scales[0] = 2.0;
        scales[1] = 0.5;
        let act = vec![1.0f32; m * k];
        let mut out = vec![0.0f32; m * n];
        fused_dequant_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &[8.0, 2.0], 1e-5);
    }

    #[test]
    fn test_fused_large_mixed() {
        let (m, n, k, bs) = (16, 8, 48, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1, -1][i % 5]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        fused_dequant_matmul(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_fused_dim_zero_rejected() {
        let mut out = vec![0.0f32; 4];
        assert!(fused_dequant_matmul(&[1.0], &[0], &[1.0], &mut out, 0, 1, 1, 32).is_err());
    }

    #[test]
    fn test_fused_block_size_zero_rejected() {
        let mut out = vec![0.0f32; 1];
        assert!(fused_dequant_matmul(&[1.0], &[0], &[1.0], &mut out, 1, 1, 1, 0).is_err());
    }

    #[test]
    fn test_fused_output_too_small() {
        let (packed, scales) = pack_i2s_weights(&[1, 0, 0, 1], 2, 2, 32);
        let mut out = vec![0.0f32; 1];
        assert!(fused_dequant_matmul(&[1.0; 4], &packed, &scales, &mut out, 2, 2, 2, 32).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 5. Tiled quantized matmul
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_tiled_identity() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_i2s_weights(&w, 2, 2, 32);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let mut out = vec![0.0f32; 4];
        let tiles = QuantizedTileConfig::DEFAULT;
        tiled_quantized_matmul(&act, &packed, &scales, &mut out, 2, 2, 2, 32, &tiles).unwrap();
        assert_close(&out, &[3.0, -2.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_tiled_agrees_with_fused() {
        let (m, n, k, bs) = (16, 8, 48, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1, -1][i % 5]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();
        let mut out_tiled = vec![0.0f32; m * n];
        let mut out_fused = vec![0.0f32; m * n];
        let tiles = QuantizedTileConfig::SMALL;
        tiled_quantized_matmul(&act, &packed, &scales, &mut out_tiled, m, n, k, bs, &tiles)
            .unwrap();
        fused_dequant_matmul(&act, &packed, &scales, &mut out_fused, m, n, k, bs).unwrap();
        assert_close(&out_tiled, &out_fused, 1e-4);
    }

    #[test]
    fn test_tiled_small_tiles() {
        let (m, n, k, bs) = (8, 6, 12, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        let tiles = QuantizedTileConfig::new(4, 3);
        tiled_quantized_matmul(&act, &packed, &scales, &mut out, m, n, k, bs, &tiles).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_tiled_tile_larger_than_matrix() {
        let (m, n, k, bs) = (2, 2, 4, 32);
        let w: Vec<i8> = vec![1, -1, 0, 1, 1, 0, -1, -1];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        let tiles = QuantizedTileConfig::new(128, 128);
        tiled_quantized_matmul(&act, &packed, &scales, &mut out, m, n, k, bs, &tiles).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_tiled_single_element_tile() {
        let (m, n, k, bs) = (3, 3, 4, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        let tiles = QuantizedTileConfig::new(1, 1);
        tiled_quantized_matmul(&act, &packed, &scales, &mut out, m, n, k, bs, &tiles).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_tiled_zero_tile_rejected() {
        let (packed, scales) = pack_i2s_weights(&[1], 1, 1, 32);
        let mut out = vec![0.0f32; 1];
        let tiles = QuantizedTileConfig::new(0, 1);
        assert!(
            tiled_quantized_matmul(&[1.0], &packed, &scales, &mut out, 1, 1, 1, 32, &tiles)
                .is_err()
        );
    }

    #[test]
    fn test_tiled_dim_zero_rejected() {
        let mut out = vec![0.0f32; 4];
        let tiles = QuantizedTileConfig::DEFAULT;
        assert!(
            tiled_quantized_matmul(&[1.0], &[0], &[1.0], &mut out, 0, 1, 1, 32, &tiles).is_err()
        );
    }

    #[test]
    fn test_tiled_default_config() {
        let cfg = QuantizedTileConfig::DEFAULT;
        assert_eq!(cfg.tile_m, 32);
        assert_eq!(cfg.tile_n, 32);
    }

    #[test]
    fn test_tiled_small_config() {
        let cfg = QuantizedTileConfig::SMALL;
        assert_eq!(cfg.tile_m, 16);
        assert_eq!(cfg.tile_n, 16);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 6. Mixed precision matmul
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_mixed_precision_identity() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_i2s_weights(&w, 2, 2, 32);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let mut out = vec![0.0f32; 4];
        mixed_precision_matmul(&act, &packed, &scales, None, &mut out, 2, 2, 2, 32).unwrap();
        assert_close(&out, &[3.0, -2.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_mixed_precision_with_bias() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_i2s_weights(&w, 2, 2, 32);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let bias = vec![10.0f32, 20.0];
        let mut out = vec![0.0f32; 4];
        mixed_precision_matmul(&act, &packed, &scales, Some(&bias), &mut out, 2, 2, 2, 32).unwrap();
        assert_close(&out, &[13.0, 18.0, 15.0, 27.0], 1e-6);
    }

    #[test]
    fn test_mixed_precision_no_bias_matches_fused() {
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1][i % 3]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let mut out_mixed = vec![0.0f32; m * n];
        let mut out_fused = vec![0.0f32; m * n];
        mixed_precision_matmul(&act, &packed, &scales, None, &mut out_mixed, m, n, k, bs).unwrap();
        fused_dequant_matmul(&act, &packed, &scales, &mut out_fused, m, n, k, bs).unwrap();
        assert_close(&out_mixed, &out_fused, 1e-4);
    }

    #[test]
    fn test_mixed_precision_bias_broadcast() {
        let (m, n, k, bs) = (3, 2, 4, 32);
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act = vec![1.0f32; m * k];
        let bias = vec![100.0f32, 200.0];
        let mut out = vec![0.0f32; m * n];
        mixed_precision_matmul(&act, &packed, &scales, Some(&bias), &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &[104.0, 204.0, 104.0, 204.0, 104.0, 204.0], 1e-5);
    }

    #[test]
    fn test_mixed_precision_bias_too_small() {
        let (packed, scales) = pack_i2s_weights(&[1, 0, 0, 1], 2, 2, 32);
        let bias = vec![1.0f32];
        let mut out = vec![0.0f32; 4];
        assert!(
            mixed_precision_matmul(&[1.0; 4], &packed, &scales, Some(&bias), &mut out, 2, 2, 2, 32)
                .is_err()
        );
    }

    #[test]
    fn test_mixed_precision_large() {
        let (m, n, k, bs) = (16, 8, 48, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.02).cos()).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let mut expected = naive_matmul(&act, &w_f32, m, n, k);
        for row in 0..m {
            for col in 0..n {
                expected[row * n + col] += bias[col];
            }
        }
        let mut out = vec![0.0f32; m * n];
        mixed_precision_matmul(&act, &packed, &scales, Some(&bias), &mut out, m, n, k, bs).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_mixed_precision_dim_zero_rejected() {
        let mut out = vec![0.0f32; 4];
        assert!(mixed_precision_matmul(&[1.0], &[0], &[1.0], None, &mut out, 0, 1, 1, 32).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 7. Batched quantized matmul
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_batched_single_head() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_i2s_weights(&w, 2, 2, 32);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let mut out = vec![0.0f32; 4];
        batched_quantized_matmul(&act, &packed, &scales, &mut out, 1, 2, 2, 2, 32).unwrap();
        assert_close(&out, &[3.0, -2.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_batched_two_heads() {
        let (m, n, k, bs) = (2, 2, 4, 32);
        let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, 1, -1];
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let num_heads = 2;
        let act: Vec<f32> = (0..(num_heads * m * k)).map(|i| (i as f32) * 0.5).collect();
        let mut out = vec![0.0f32; num_heads * m * n];
        batched_quantized_matmul(&act, &packed, &scales, &mut out, num_heads, m, n, k, bs).unwrap();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        for h in 0..num_heads {
            let act_h = &act[h * m * k..(h + 1) * m * k];
            let expected = naive_matmul(act_h, &w_f32, m, n, k);
            let out_h = &out[h * m * n..(h + 1) * m * n];
            assert_close(out_h, &expected, 1e-4);
        }
    }

    #[test]
    fn test_batched_four_heads() {
        let (m, n, k, bs) = (4, 3, 8, 32);
        let num_heads = 4;
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..(num_heads * m * k)).map(|i| (i as f32 * 0.07).sin()).collect();
        let mut out = vec![0.0f32; num_heads * m * n];
        batched_quantized_matmul(&act, &packed, &scales, &mut out, num_heads, m, n, k, bs).unwrap();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        for h in 0..num_heads {
            let act_h = &act[h * m * k..(h + 1) * m * k];
            let expected = naive_matmul(act_h, &w_f32, m, n, k);
            let out_h = &out[h * m * n..(h + 1) * m * n];
            assert_close(out_h, &expected, 1e-3);
        }
    }

    #[test]
    fn test_batched_zero_heads() {
        let (packed, scales) = pack_i2s_weights(&[1], 1, 1, 32);
        let mut out: Vec<f32> = vec![];
        batched_quantized_matmul(&[], &packed, &scales, &mut out, 0, 1, 1, 1, 32).unwrap();
    }

    #[test]
    fn test_batched_activations_too_small() {
        let (packed, scales) = pack_i2s_weights(&[1, 0, 0, 1], 2, 2, 32);
        let act = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 8];
        assert!(
            batched_quantized_matmul(&act, &packed, &scales, &mut out, 2, 2, 2, 2, 32).is_err()
        );
    }

    #[test]
    fn test_batched_output_too_small() {
        let (packed, scales) = pack_i2s_weights(&[1, 0, 0, 1], 2, 2, 32);
        let act = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 4];
        assert!(
            batched_quantized_matmul(&act, &packed, &scales, &mut out, 2, 2, 2, 2, 32).is_err()
        );
    }

    #[test]
    fn test_batched_dim_zero_rejected() {
        let mut out = vec![0.0f32; 4];
        assert!(batched_quantized_matmul(&[1.0], &[0], &[1.0], &mut out, 1, 0, 1, 1, 32).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════
    // 8. Sparse quantized matmul
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_sparse_identity() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let sparse = make_sparse(&w, 2, 2, 32);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let mut out = vec![0.0f32; 4];
        sparse_quantized_matmul(&act, &sparse, &mut out, 2).unwrap();
        assert_close(&out, &[3.0, -2.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_sparse_all_ones() {
        let (m, n, k, bs) = (4, 4, 4, 32);
        let w = vec![1i8; k * n];
        let sparse = make_sparse(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        sparse_quantized_matmul(&act, &sparse, &mut out, m).unwrap();
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_sparse_all_zeros() {
        let (m, n, k, bs) = (3, 3, 4, 32);
        let w = vec![0i8; k * n];
        let sparse = make_sparse(&w, k, n, bs);
        let act = vec![99.0f32; m * k];
        let mut out = vec![0.0f32; m * n];
        sparse_quantized_matmul(&act, &sparse, &mut out, m).unwrap();
        assert_close(&out, &vec![0.0f32; m * n], 1e-6);
    }

    #[test]
    fn test_sparse_mixed_ternary() {
        let (m, n, k, bs) = (4, 3, 6, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1][i % 3]).collect();
        let sparse = make_sparse(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        sparse_quantized_matmul(&act, &sparse, &mut out, m).unwrap();
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_sparse_1x1() {
        let sparse = make_sparse(&[-1], 1, 1, 32);
        let mut out = vec![0.0f32; 1];
        sparse_quantized_matmul(&[5.0], &sparse, &mut out, 1).unwrap();
        assert_close(&out, &[-5.0], 1e-6);
    }

    #[test]
    fn test_sparse_highly_sparse() {
        let (m, n, k, bs) = (4, 4, 20, 32);
        let w: Vec<i8> = (0..k * n).map(|i| if i % 10 == 0 { 1 } else { 0 }).collect();
        let sparse = make_sparse(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let mut out = vec![0.0f32; m * n];
        sparse_quantized_matmul(&act, &sparse, &mut out, m).unwrap();
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_sparse_agrees_with_dense() {
        let (m, n, k, bs) = (8, 6, 12, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1, 0][i % 5]).collect();
        let sparse = make_sparse(&w, k, n, bs);
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut out_sparse = vec![0.0f32; m * n];
        let mut out_dense = vec![0.0f32; m * n];
        sparse_quantized_matmul(&act, &sparse, &mut out_sparse, m).unwrap();
        fused_dequant_matmul(&act, &packed, &scales, &mut out_dense, m, n, k, bs).unwrap();
        assert_close(&out_sparse, &out_dense, 1e-4);
    }

    #[test]
    fn test_sparse_dim_zero_rejected() {
        let sparse = make_sparse(&[1], 1, 1, 32);
        let mut out = vec![0.0f32; 1];
        assert!(sparse_quantized_matmul(&[], &sparse, &mut out, 0).is_err());
    }

    #[test]
    fn test_sparse_activations_too_small() {
        let sparse = make_sparse(&[1, 0, 0, 1], 2, 2, 32);
        let mut out = vec![0.0f32; 4];
        assert!(sparse_quantized_matmul(&[1.0], &sparse, &mut out, 2).is_err());
    }

    #[test]
    fn test_sparse_output_too_small() {
        let sparse = make_sparse(&[1, 0, 0, 1], 2, 2, 32);
        let mut out = vec![0.0f32; 1];
        assert!(sparse_quantized_matmul(&[1.0; 4], &sparse, &mut out, 2).is_err());
    }

    #[test]
    fn test_sparse_bad_row_ptrs() {
        let mut sparse = make_sparse(&[1], 1, 1, 32);
        sparse.row_ptrs = vec![0];
        let mut out = vec![0.0f32; 1];
        assert!(sparse_quantized_matmul(&[1.0], &sparse, &mut out, 1).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════
    // Cross-kernel consistency
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_all_kernels_agree_medium() {
        let (m, n, k, bs) = (8, 6, 24, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1, -1][i % 5]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut out_block = vec![0.0f32; m * n];
        let mut out_fused = vec![0.0f32; m * n];
        let mut out_tiled = vec![0.0f32; m * n];
        let mut out_mixed = vec![0.0f32; m * n];
        block_quantized_matmul(&act, &packed, &scales, &mut out_block, m, n, k, bs).unwrap();
        fused_dequant_matmul(&act, &packed, &scales, &mut out_fused, m, n, k, bs).unwrap();
        tiled_quantized_matmul(
            &act,
            &packed,
            &scales,
            &mut out_tiled,
            m,
            n,
            k,
            bs,
            &QuantizedTileConfig::SMALL,
        )
        .unwrap();
        mixed_precision_matmul(&act, &packed, &scales, None, &mut out_mixed, m, n, k, bs).unwrap();
        assert_close(&out_block, &out_fused, 1e-4);
        assert_close(&out_fused, &out_tiled, 1e-4);
        assert_close(&out_tiled, &out_mixed, 1e-4);
    }

    #[test]
    fn test_all_kernels_agree_large() {
        let (m, n, k, bs) = (32, 16, 64, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).cos()).collect();
        let mut out_block = vec![0.0f32; m * n];
        let mut out_fused = vec![0.0f32; m * n];
        let mut out_tiled = vec![0.0f32; m * n];
        block_quantized_matmul(&act, &packed, &scales, &mut out_block, m, n, k, bs).unwrap();
        fused_dequant_matmul(&act, &packed, &scales, &mut out_fused, m, n, k, bs).unwrap();
        tiled_quantized_matmul(
            &act,
            &packed,
            &scales,
            &mut out_tiled,
            m,
            n,
            k,
            bs,
            &QuantizedTileConfig::DEFAULT,
        )
        .unwrap();
        assert_close(&out_block, &out_fused, 1e-3);
        assert_close(&out_fused, &out_tiled, 1e-3);
    }

    #[test]
    fn test_dense_sparse_consistency() {
        let (m, n, k, bs) = (6, 4, 16, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, 0, -1, 0][i % 5]).collect();
        let sparse = make_sparse(&w, k, n, bs);
        let (packed, scales) = pack_i2s_weights(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let mut out_fused = vec![0.0f32; m * n];
        let mut out_sparse = vec![0.0f32; m * n];
        fused_dequant_matmul(&act, &packed, &scales, &mut out_fused, m, n, k, bs).unwrap();
        sparse_quantized_matmul(&act, &sparse, &mut out_sparse, m).unwrap();
        assert_close(&out_fused, &out_sparse, 1e-4);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pack helper round-trips
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pack_i2s_roundtrip() {
        let vals: [i8; 4] = [1, -1, 0, 1];
        let byte = pack_i2s(vals);
        for (i, &expected) in vals.iter().enumerate() {
            let bits = (byte >> (i * 2)) & 0x03;
            assert_eq!(decode_i2s(bits), expected, "mismatch at {i}");
        }
    }

    #[test]
    fn test_pack_i2s_all_zero() {
        assert_eq!(pack_i2s([0, 0, 0, 0]), 0x00);
    }

    #[test]
    fn test_pack_i2s_all_plus_one() {
        assert_eq!(pack_i2s([1, 1, 1, 1]), 0b01_01_01_01);
    }

    #[test]
    fn test_pack_i2s_all_minus_one() {
        assert_eq!(pack_i2s([-1, -1, -1, -1]), 0b11_11_11_11);
    }
}
