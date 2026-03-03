#![allow(
    unsafe_op_in_unsafe_fn,
    unused_unsafe,
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::manual_abs_diff,
    clippy::manual_contains,
    clippy::manual_is_multiple_of,
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_cast
)]
//! ARM NEON quantized GEMV (General Matrix-Vector multiply) kernels for Apple Silicon.
//!
//! Provides NEON-accelerated GEMV operations for ternary and I2_S 2-bit
//! quantized weights against f32 input vectors. Includes tiled, multi-row,
//! scale-fused, batched, Kahan-compensated, and sparse variants.
//!
//! I2_S encoding (2 bits per value, 4 values per byte, LSB-first):
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── I2_S decode helpers ─────────────────────────────────────────────

/// I2_S f32 LUT: index by 2-bit code → f32 value.
const I2S_LUT: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

/// Decode a single 2-bit I2_S code to its signed float value.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    I2S_LUT[(bits & 0x03) as usize]
}

/// Unpack one packed byte into 4 f32 values via the LUT.
#[inline(always)]
fn unpack_byte_f32(byte: u8) -> [f32; 4] {
    [
        I2S_LUT[(byte & 0x03) as usize],
        I2S_LUT[((byte >> 2) & 0x03) as usize],
        I2S_LUT[((byte >> 4) & 0x03) as usize],
        I2S_LUT[((byte >> 6) & 0x03) as usize],
    ]
}

// ── Scalar reference (used as fallback on non-aarch64) ──────────────

/// Scalar reference GEMV: `output[r] = scale * Σ_c dequant(W)[r,c] * input[c]`.
///
/// Used as the non-aarch64 fallback and as the correctness oracle in tests.
pub fn scalar_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    let packed_cols = cols.div_ceil(4);
    for row in 0..rows {
        let mut sum = 0.0f32;
        let row_off = row * packed_cols;
        for c in 0..cols {
            let byte_idx = c / 4;
            let bit_off = (c % 4) * 2;
            let byte = weights_packed[row_off + byte_idx];
            let w = decode_i2s((byte >> bit_off) & 0x03);
            sum += w * input[c];
        }
        output[row] = sum * scale;
    }
}

/// Scalar reference ternary GEMV: explicit {-1,0,+1} weights × f32 input.
pub fn scalar_ternary_gemv(
    weights: &[i8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    for row in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += (weights[row * cols + c] as f32) * input[c];
        }
        output[row] = sum * scale;
    }
}

// ── 1. Ternary weight GEMV ──────────────────────────────────────────

/// NEON ternary weight GEMV: {-1, 0, +1} weights × f32 input vector.
///
/// Weights are stored as `i8` in row-major order `[rows × cols]`.
/// Uses NEON `vfmaq_f32` for the hot loop with scalar tail.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_ternary_gemv(
    weights: &[i8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    debug_assert!(weights.len() >= rows * cols);
    debug_assert!(input.len() >= cols);
    debug_assert!(output.len() >= rows);

    let chunks = cols / 4;
    let tail = cols % 4;

    for row in 0..rows {
        let row_off = row * cols;
        let mut acc = unsafe { vdupq_n_f32(0.0) };

        for i in 0..chunks {
            let off = row_off + i * 4;
            // Convert i8 weights to f32 one lane at a time
            let w_arr = [
                weights[off] as f32,
                weights[off + 1] as f32,
                weights[off + 2] as f32,
                weights[off + 3] as f32,
            ];
            unsafe {
                let vw = vld1q_f32(w_arr.as_ptr());
                let va = vld1q_f32(input.as_ptr().add(i * 4));
                acc = vfmaq_f32(acc, vw, va);
            }
        }

        let mut sum = unsafe { vaddvq_f32(acc) };

        // Scalar tail
        let tail_start = chunks * 4;
        for j in 0..tail {
            sum += (weights[row_off + tail_start + j] as f32) * input[tail_start + j];
        }

        output[row] = sum * scale;
    }
}

/// Scalar fallback for ternary GEMV on non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_ternary_gemv(
    weights: &[i8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    scalar_ternary_gemv(weights, input, output, rows, cols, scale);
}

// ── 2. I2_S packed weight GEMV ──────────────────────────────────────

/// NEON I2_S packed weight GEMV with NEON unpacking.
///
/// `weights_packed`: row-major I2_S, each row `ceil(cols/4)` bytes.
/// Uses LUT-based byte unpacking and `vfmaq_f32` accumulation.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    let packed_cols = cols.div_ceil(4);
    debug_assert!(weights_packed.len() >= rows * packed_cols);
    debug_assert!(input.len() >= cols);
    debug_assert!(output.len() >= rows);

    let full_bytes = cols / 4;
    let remainder = cols % 4;

    for row in 0..rows {
        let row_start = row * packed_cols;
        let mut acc = unsafe { vdupq_n_f32(0.0) };

        for b in 0..full_bytes {
            let byte = weights_packed[row_start + b];
            let w_arr = unpack_byte_f32(byte);
            unsafe {
                let vw = vld1q_f32(w_arr.as_ptr());
                let va = vld1q_f32(input.as_ptr().add(b * 4));
                acc = vfmaq_f32(acc, vw, va);
            }
        }

        let mut sum = unsafe { vaddvq_f32(acc) };

        // Scalar tail
        if remainder > 0 && full_bytes < packed_cols {
            let byte = weights_packed[row_start + full_bytes];
            for j in 0..remainder {
                let w = decode_i2s((byte >> (j * 2)) & 0x03);
                sum += w * input[full_bytes * 4 + j];
            }
        }

        output[row] = sum * scale;
    }
}

/// Scalar fallback for I2_S GEMV on non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_i2s_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    scalar_gemv(weights_packed, input, output, rows, cols, scale);
}

// ── 3. Tiled GEMV ───────────────────────────────────────────────────

/// Tile height for tiled GEMV (rows processed per tile).
const TILE_ROWS: usize = 4;
/// Tile width in packed bytes (elements = TILE_K_BYTES * 4).
const TILE_K_BYTES: usize = 8;

/// Tiled I2_S GEMV: process in `[TILE_ROWS × TILE_K_BYTES*4]` tiles
/// for cache efficiency.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_tiled_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    let packed_cols = cols.div_ceil(4);
    debug_assert!(weights_packed.len() >= rows * packed_cols);
    debug_assert!(input.len() >= cols);
    debug_assert!(output.len() >= rows);

    let row_tiles = rows / TILE_ROWS;
    let full_bytes = cols / 4;
    let k_tiles = full_bytes / TILE_K_BYTES;
    let remainder = cols % 4;

    // Process full row-tiles
    for rt in 0..row_tiles {
        let row0 = rt * TILE_ROWS;
        let mut accs: [[f32; 4]; TILE_ROWS] = [[0.0; 4]; TILE_ROWS];

        // Process full k-tiles with NEON
        for kt in 0..k_tiles {
            let b0 = kt * TILE_K_BYTES;
            for ti in 0..TILE_ROWS {
                let row_off = (row0 + ti) * packed_cols;
                let mut acc = unsafe { vdupq_n_f32(0.0) };
                for b in 0..TILE_K_BYTES {
                    let byte = weights_packed[row_off + b0 + b];
                    let w_arr = unpack_byte_f32(byte);
                    unsafe {
                        let vw = vld1q_f32(w_arr.as_ptr());
                        let va = vld1q_f32(input.as_ptr().add((b0 + b) * 4));
                        acc = vfmaq_f32(acc, vw, va);
                    }
                }
                // Store partial sums back
                unsafe { vst1q_f32(accs[ti].as_mut_ptr(), acc) };
            }
        }

        // Remaining k-columns (full bytes past last k-tile)
        let k_rem_start = k_tiles * TILE_K_BYTES;
        for ti in 0..TILE_ROWS {
            let row_off = (row0 + ti) * packed_cols;
            let mut partial = 0.0f32;
            for b in k_rem_start..full_bytes {
                let byte = weights_packed[row_off + b];
                let w = unpack_byte_f32(byte);
                for j in 0..4 {
                    partial += w[j] * input[b * 4 + j];
                }
            }
            // Scalar tail
            if remainder > 0 && full_bytes < packed_cols {
                let byte = weights_packed[row_off + full_bytes];
                for j in 0..remainder {
                    let w = decode_i2s((byte >> (j * 2)) & 0x03);
                    partial += w * input[full_bytes * 4 + j];
                }
            }
            let acc_sum: f32 = accs[ti].iter().sum();
            output[row0 + ti] = (acc_sum + partial) * scale;
        }
    }

    // Remaining rows
    let row_rem = row_tiles * TILE_ROWS;
    for row in row_rem..rows {
        let row_off = row * packed_cols;
        let mut sum = 0.0f32;
        for b in 0..full_bytes {
            let byte = weights_packed[row_off + b];
            let w = unpack_byte_f32(byte);
            for j in 0..4 {
                sum += w[j] * input[b * 4 + j];
            }
        }
        if remainder > 0 && full_bytes < packed_cols {
            let byte = weights_packed[row_off + full_bytes];
            for j in 0..remainder {
                let w = decode_i2s((byte >> (j * 2)) & 0x03);
                sum += w * input[full_bytes * 4 + j];
            }
        }
        output[row] = sum * scale;
    }
}

/// Scalar fallback for tiled GEMV on non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_i2s_tiled_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    scalar_gemv(weights_packed, input, output, rows, cols, scale);
}

// ── 4. Multi-row GEMV ───────────────────────────────────────────────

/// Number of rows processed simultaneously in multi-row GEMV.
const MULTI_ROWS: usize = 4;

/// Multi-row NEON GEMV: process 4 output rows simultaneously.
///
/// Each iteration accumulates 4 independent dot products sharing the
/// same input vector, maximising NEON register utilisation.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_multirow_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    let packed_cols = cols.div_ceil(4);
    debug_assert!(weights_packed.len() >= rows * packed_cols);
    debug_assert!(input.len() >= cols);
    debug_assert!(output.len() >= rows);

    let full_bytes = cols / 4;
    let remainder = cols % 4;
    let row_groups = rows / MULTI_ROWS;

    for g in 0..row_groups {
        let row0 = g * MULTI_ROWS;
        let mut acc0 = unsafe { vdupq_n_f32(0.0) };
        let mut acc1 = unsafe { vdupq_n_f32(0.0) };
        let mut acc2 = unsafe { vdupq_n_f32(0.0) };
        let mut acc3 = unsafe { vdupq_n_f32(0.0) };

        for b in 0..full_bytes {
            unsafe {
                let va = vld1q_f32(input.as_ptr().add(b * 4));

                let w0 = unpack_byte_f32(weights_packed[(row0) * packed_cols + b]);
                let vw0 = vld1q_f32(w0.as_ptr());
                acc0 = vfmaq_f32(acc0, vw0, va);

                let w1 = unpack_byte_f32(weights_packed[(row0 + 1) * packed_cols + b]);
                let vw1 = vld1q_f32(w1.as_ptr());
                acc1 = vfmaq_f32(acc1, vw1, va);

                let w2 = unpack_byte_f32(weights_packed[(row0 + 2) * packed_cols + b]);
                let vw2 = vld1q_f32(w2.as_ptr());
                acc2 = vfmaq_f32(acc2, vw2, va);

                let w3 = unpack_byte_f32(weights_packed[(row0 + 3) * packed_cols + b]);
                let vw3 = vld1q_f32(w3.as_ptr());
                acc3 = vfmaq_f32(acc3, vw3, va);
            }
        }

        let mut sums =
            unsafe { [vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)] };

        // Scalar tail
        if remainder > 0 && full_bytes < packed_cols {
            for ri in 0..MULTI_ROWS {
                let byte = weights_packed[(row0 + ri) * packed_cols + full_bytes];
                for j in 0..remainder {
                    let w = decode_i2s((byte >> (j * 2)) & 0x03);
                    sums[ri] += w * input[full_bytes * 4 + j];
                }
            }
        }

        for ri in 0..MULTI_ROWS {
            output[row0 + ri] = sums[ri] * scale;
        }
    }

    // Remaining rows (< MULTI_ROWS)
    let row_rem = row_groups * MULTI_ROWS;
    for row in row_rem..rows {
        let row_off = row * packed_cols;
        let mut acc = unsafe { vdupq_n_f32(0.0) };

        for b in 0..full_bytes {
            let w_arr = unpack_byte_f32(weights_packed[row_off + b]);
            unsafe {
                let vw = vld1q_f32(w_arr.as_ptr());
                let va = vld1q_f32(input.as_ptr().add(b * 4));
                acc = vfmaq_f32(acc, vw, va);
            }
        }

        let mut sum = unsafe { vaddvq_f32(acc) };
        if remainder > 0 && full_bytes < packed_cols {
            let byte = weights_packed[row_off + full_bytes];
            for j in 0..remainder {
                let w = decode_i2s((byte >> (j * 2)) & 0x03);
                sum += w * input[full_bytes * 4 + j];
            }
        }

        output[row] = sum * scale;
    }
}

/// Scalar fallback for multi-row GEMV on non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_i2s_multirow_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    scalar_gemv(weights_packed, input, output, rows, cols, scale);
}

// ── 5. Scale-fused GEMV ─────────────────────────────────────────────

/// Scale-fused NEON GEMV: per-block scales applied during computation.
///
/// Each block of `block_size` columns has its own scale factor in
/// `block_scales`. The final output also has `global_scale` applied.
///
/// `block_scales.len()` must be `ceil(cols / block_size)`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_scale_fused_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    block_size: usize,
    block_scales: &[f32],
    global_scale: f32,
) {
    let packed_cols = cols.div_ceil(4);
    let num_blocks = cols.div_ceil(block_size);
    debug_assert!(weights_packed.len() >= rows * packed_cols);
    debug_assert!(input.len() >= cols);
    debug_assert!(output.len() >= rows);
    debug_assert!(block_scales.len() >= num_blocks);
    debug_assert!(block_size > 0);

    for row in 0..rows {
        let row_off = row * packed_cols;
        let mut total = 0.0f32;

        for blk in 0..num_blocks {
            let col_start = blk * block_size;
            let col_end = (col_start + block_size).min(cols);
            let blk_scale = block_scales[blk];
            let mut blk_sum = 0.0f32;

            // NEON portion: process full 4-element groups within block
            let byte_start = col_start / 4;
            let byte_end = col_end / 4;
            let mut acc = unsafe { vdupq_n_f32(0.0) };

            // Only use NEON for aligned 4-element groups
            let aligned_start = col_start.div_ceil(4);
            let aligned_end = col_end / 4;

            // Leading scalar elements
            for c in col_start..(aligned_start * 4).min(col_end) {
                let bi = c / 4;
                let bo = (c % 4) * 2;
                let byte = weights_packed[row_off + bi];
                let w = decode_i2s((byte >> bo) & 0x03);
                blk_sum += w * input[c];
            }

            // NEON body
            for b in aligned_start..aligned_end {
                let byte = weights_packed[row_off + b];
                let w_arr = unpack_byte_f32(byte);
                unsafe {
                    let vw = vld1q_f32(w_arr.as_ptr());
                    let va = vld1q_f32(input.as_ptr().add(b * 4));
                    acc = vfmaq_f32(acc, vw, va);
                }
            }

            blk_sum += unsafe { vaddvq_f32(acc) };

            // Trailing scalar elements
            let trail_start = aligned_end * 4;
            for c in trail_start.max(col_start)..col_end {
                let bi = c / 4;
                let bo = (c % 4) * 2;
                let byte = weights_packed[row_off + bi];
                let w = decode_i2s((byte >> bo) & 0x03);
                blk_sum += w * input[c];
            }

            total += blk_sum * blk_scale;
        }

        output[row] = total * global_scale;
    }
}

/// Scalar fallback for scale-fused GEMV on non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_i2s_scale_fused_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    block_size: usize,
    block_scales: &[f32],
    global_scale: f32,
) {
    scalar_scale_fused_gemv(
        weights_packed,
        input,
        output,
        rows,
        cols,
        block_size,
        block_scales,
        global_scale,
    );
}

/// Scalar reference for scale-fused GEMV.
pub fn scalar_scale_fused_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    block_size: usize,
    block_scales: &[f32],
    global_scale: f32,
) {
    let packed_cols = cols.div_ceil(4);
    let num_blocks = cols.div_ceil(block_size);
    for row in 0..rows {
        let row_off = row * packed_cols;
        let mut total = 0.0f32;
        for blk in 0..num_blocks {
            let col_start = blk * block_size;
            let col_end = (col_start + block_size).min(cols);
            let blk_scale = block_scales[blk];
            let mut blk_sum = 0.0f32;
            for c in col_start..col_end {
                let bi = c / 4;
                let bo = (c % 4) * 2;
                let byte = weights_packed[row_off + bi];
                let w = decode_i2s((byte >> bo) & 0x03);
                blk_sum += w * input[c];
            }
            total += blk_sum * blk_scale;
        }
        output[row] = total * global_scale;
    }
}

// ── 6. Kahan-compensated GEMV ───────────────────────────────────────

/// Kahan-compensated I2_S GEMV for improved numerical accuracy.
///
/// Uses Kahan summation to reduce floating-point rounding errors in
/// the inner dot product, important for long vectors where naive
/// accumulation loses precision.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_kahan_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    let packed_cols = cols.div_ceil(4);
    debug_assert!(weights_packed.len() >= rows * packed_cols);
    debug_assert!(input.len() >= cols);
    debug_assert!(output.len() >= rows);

    let full_bytes = cols / 4;
    let remainder = cols % 4;

    for row in 0..rows {
        let row_off = row * packed_cols;
        // Kahan state: sum and compensation per lane
        let mut sum = unsafe { vdupq_n_f32(0.0) };
        let mut comp = unsafe { vdupq_n_f32(0.0) };

        for b in 0..full_bytes {
            let byte = weights_packed[row_off + b];
            let w_arr = unpack_byte_f32(byte);
            unsafe {
                let vw = vld1q_f32(w_arr.as_ptr());
                let va = vld1q_f32(input.as_ptr().add(b * 4));
                let product = vmulq_f32(vw, va);
                // Kahan: y = product - comp
                let y = vsubq_f32(product, comp);
                // t = sum + y
                let t = vaddq_f32(sum, y);
                // comp = (t - sum) - y
                comp = vsubq_f32(vsubq_f32(t, sum), y);
                sum = t;
            }
        }

        // Horizontal sum with scalar Kahan for final reduction
        let mut s = 0.0f32;
        let mut c = 0.0f32;
        let mut lane_vals = [0.0f32; 4];
        unsafe { vst1q_f32(lane_vals.as_mut_ptr(), sum) };
        for &v in &lane_vals {
            let y = v - c;
            let t = s + y;
            c = (t - s) - y;
            s = t;
        }

        // Scalar tail with Kahan
        if remainder > 0 && full_bytes < packed_cols {
            let byte = weights_packed[row_off + full_bytes];
            for j in 0..remainder {
                let w = decode_i2s((byte >> (j * 2)) & 0x03);
                let product = w * input[full_bytes * 4 + j];
                let y = product - c;
                let t = s + y;
                c = (t - s) - y;
                s = t;
            }
        }

        output[row] = s * scale;
    }
}

/// Scalar Kahan-compensated GEMV reference.
pub fn scalar_kahan_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    let packed_cols = cols.div_ceil(4);
    for row in 0..rows {
        let row_off = row * packed_cols;
        let mut sum = 0.0f32;
        let mut comp = 0.0f32;
        for c in 0..cols {
            let bi = c / 4;
            let bo = (c % 4) * 2;
            let byte = weights_packed[row_off + bi];
            let w = decode_i2s((byte >> bo) & 0x03);
            let product = w * input[c];
            let y = product - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        output[row] = sum * scale;
    }
}

/// Scalar fallback for Kahan GEMV on non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_i2s_kahan_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    scalar_kahan_gemv(weights_packed, input, output, rows, cols, scale);
}

// ── 7. Batched GEMV ─────────────────────────────────────────────────

/// Batched I2_S GEMV with shared weights.
///
/// Computes `output_b[r] = scale * Σ_c dequant(W)[r,c] * input_b[c]`
/// for each input vector in the batch. All vectors share the same
/// weight matrix.
///
/// `inputs_batch`: `[batch_size × cols]` row-major.
/// `outputs_batch`: `[batch_size × rows]` row-major.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_batched_gemv(
    weights_packed: &[u8],
    inputs_batch: &[f32],
    outputs_batch: &mut [f32],
    rows: usize,
    cols: usize,
    batch_size: usize,
    scale: f32,
) {
    debug_assert!(inputs_batch.len() >= batch_size * cols);
    debug_assert!(outputs_batch.len() >= batch_size * rows);

    for b in 0..batch_size {
        let in_off = b * cols;
        let out_off = b * rows;
        unsafe {
            neon_i2s_gemv(
                weights_packed,
                &inputs_batch[in_off..in_off + cols],
                &mut outputs_batch[out_off..out_off + rows],
                rows,
                cols,
                scale,
            );
        }
    }
}

/// Scalar fallback for batched GEMV on non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_i2s_batched_gemv(
    weights_packed: &[u8],
    inputs_batch: &[f32],
    outputs_batch: &mut [f32],
    rows: usize,
    cols: usize,
    batch_size: usize,
    scale: f32,
) {
    for b in 0..batch_size {
        let in_off = b * cols;
        let out_off = b * rows;
        scalar_gemv(
            weights_packed,
            &inputs_batch[in_off..in_off + cols],
            &mut outputs_batch[out_off..out_off + rows],
            rows,
            cols,
            scale,
        );
    }
}

// ── 8. Sparse GEMV ──────────────────────────────────────────────────

/// Sparse I2_S GEMV: skip zero-only packed bytes for efficiency.
///
/// For each packed byte, checks whether all 4 encoded weights are zero
/// (byte == 0x00) and skips the multiply-add entirely. Beneficial
/// when the weight matrix is very sparse (many zero blocks).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_sparse_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    let packed_cols = cols.div_ceil(4);
    debug_assert!(weights_packed.len() >= rows * packed_cols);
    debug_assert!(input.len() >= cols);
    debug_assert!(output.len() >= rows);

    let full_bytes = cols / 4;
    let remainder = cols % 4;

    for row in 0..rows {
        let row_off = row * packed_cols;
        let mut acc = unsafe { vdupq_n_f32(0.0) };

        for b in 0..full_bytes {
            let byte = weights_packed[row_off + b];
            // Skip zero blocks entirely
            if byte == 0x00 {
                continue;
            }
            let w_arr = unpack_byte_f32(byte);
            unsafe {
                let vw = vld1q_f32(w_arr.as_ptr());
                let va = vld1q_f32(input.as_ptr().add(b * 4));
                acc = vfmaq_f32(acc, vw, va);
            }
        }

        let mut sum = unsafe { vaddvq_f32(acc) };

        if remainder > 0 && full_bytes < packed_cols {
            let byte = weights_packed[row_off + full_bytes];
            if byte != 0x00 {
                for j in 0..remainder {
                    let w = decode_i2s((byte >> (j * 2)) & 0x03);
                    sum += w * input[full_bytes * 4 + j];
                }
            }
        }

        output[row] = sum * scale;
    }
}

/// Scalar fallback for sparse GEMV on non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_i2s_sparse_gemv(
    weights_packed: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    // Scalar with zero-skip
    let packed_cols = cols.div_ceil(4);
    for row in 0..rows {
        let row_off = row * packed_cols;
        let mut sum = 0.0f32;
        for c in 0..cols {
            let bi = c / 4;
            let bo = (c % 4) * 2;
            let byte = weights_packed[row_off + bi];
            let w = decode_i2s((byte >> bo) & 0x03);
            sum += w * input[c];
        }
        output[row] = sum * scale;
    }
}

// ── Utility: pack ternary weights ───────────────────────────────────

/// Pack an `i8` ternary weight row into I2_S byte format.
///
/// Each value must be in `{-1, 0, +1}`. Four values are packed per byte
/// (LSB-first): +1 → 0b01, −1 → 0b11, 0 → 0b00.
pub fn pack_ternary_row(weights: &[i8]) -> Vec<u8> {
    let packed_len = weights.len().div_ceil(4);
    let mut packed = vec![0u8; packed_len];
    for (i, &w) in weights.iter().enumerate() {
        let code: u8 = match w {
            1 => 0b01,
            -1 => 0b11,
            _ => 0b00,
        };
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        packed[byte_idx] |= code << bit_off;
    }
    packed
}

/// Pack a full ternary weight matrix `[rows × cols]` into I2_S.
pub fn pack_ternary_matrix(weights: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    let packed_cols = cols.div_ceil(4);
    let mut packed = vec![0u8; rows * packed_cols];
    for row in 0..rows {
        let row_packed = pack_ternary_row(&weights[row * cols..(row + 1) * cols]);
        packed[row * packed_cols..(row * packed_cols + packed_cols)].copy_from_slice(&row_packed);
    }
    packed
}

// ── Utility: sparsity measurement ───────────────────────────────────

/// Compute the fraction of zero-only packed bytes in the weight matrix.
pub fn sparsity_ratio(weights_packed: &[u8]) -> f64 {
    if weights_packed.is_empty() {
        return 0.0;
    }
    let zeros = weights_packed.iter().filter(|&&b| b == 0x00).count();
    zeros as f64 / weights_packed.len() as f64
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test helpers ────────────────────────────────────────────────

    /// Assert two f32 slices are within tolerance.
    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "{msg}: index {i}: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    /// Create a simple input vector [1.0, 2.0, 3.0, ...].
    fn ramp_input(n: usize) -> Vec<f32> {
        (1..=n).map(|i| i as f32).collect()
    }

    /// Create an all-ones input vector.
    fn ones_input(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    // ── 1. decode / pack helpers ────────────────────────────────────

    #[test]
    fn test_decode_i2s_values() {
        assert_eq!(decode_i2s(0b00), 0.0);
        assert_eq!(decode_i2s(0b01), 1.0);
        assert_eq!(decode_i2s(0b10), 0.0); // unused
        assert_eq!(decode_i2s(0b11), -1.0);
    }

    #[test]
    fn test_unpack_byte_f32() {
        // byte = 0b11_00_01_01 => [+1, +1, 0, -1]
        let vals = unpack_byte_f32(0b11_00_01_01);
        assert_eq!(vals, [1.0, 1.0, 0.0, -1.0]);
    }

    #[test]
    fn test_unpack_byte_all_zeros() {
        assert_eq!(unpack_byte_f32(0x00), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unpack_byte_all_plus_ones() {
        // 0b01_01_01_01 = 0x55
        assert_eq!(unpack_byte_f32(0x55), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_unpack_byte_all_minus_ones() {
        // 0b11_11_11_11 = 0xFF
        assert_eq!(unpack_byte_f32(0xFF), [-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_pack_ternary_row_basic() {
        let weights: Vec<i8> = vec![1, -1, 0, 1];
        let packed = pack_ternary_row(&weights);
        assert_eq!(packed.len(), 1);
        // +1=01, -1=11, 0=00, +1=01 → 0b01_00_11_01
        assert_eq!(packed[0], 0b01_00_11_01);
    }

    #[test]
    fn test_pack_ternary_row_remainder() {
        let weights: Vec<i8> = vec![1, -1, 0]; // 3 values → 1 byte
        let packed = pack_ternary_row(&weights);
        assert_eq!(packed.len(), 1);
        // +1=01, -1=11, 0=00, pad=00 → 0b00_00_11_01
        assert_eq!(packed[0], 0b00_00_11_01);
    }

    #[test]
    fn test_pack_ternary_row_empty() {
        let packed = pack_ternary_row(&[]);
        assert!(packed.is_empty());
    }

    #[test]
    fn test_pack_ternary_row_8_values() {
        let weights: Vec<i8> = vec![1, 0, -1, 0, -1, 1, 0, 0];
        let packed = pack_ternary_row(&weights);
        assert_eq!(packed.len(), 2);
    }

    #[test]
    fn test_pack_ternary_matrix() {
        let weights: Vec<i8> = vec![
            1, -1, 0, 1, // row 0
            0, 0, 1, -1, // row 1
        ];
        let packed = pack_ternary_matrix(&weights, 2, 4);
        assert_eq!(packed.len(), 2); // 2 rows × 1 byte each
    }

    #[test]
    fn test_pack_roundtrip() {
        let weights: Vec<i8> = vec![1, -1, 0, 1];
        let packed = pack_ternary_row(&weights);
        let mut out = vec![0.0f32; 4];
        for (i, o) in out.iter_mut().enumerate() {
            let bi = i / 4;
            let bo = (i % 4) * 2;
            *o = decode_i2s((packed[bi] >> bo) & 0x03);
        }
        let expected: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_sparsity_ratio_all_zeros() {
        assert_eq!(sparsity_ratio(&[0, 0, 0, 0]), 1.0);
    }

    #[test]
    fn test_sparsity_ratio_no_zeros() {
        assert_eq!(sparsity_ratio(&[0x55, 0xFF, 0x55, 0xFF]), 0.0);
    }

    #[test]
    fn test_sparsity_ratio_half() {
        assert_eq!(sparsity_ratio(&[0x00, 0x55]), 0.5);
    }

    #[test]
    fn test_sparsity_ratio_empty() {
        assert_eq!(sparsity_ratio(&[]), 0.0);
    }

    // ── 2. scalar_gemv reference ────────────────────────────────────

    #[test]
    fn test_scalar_gemv_identity_row() {
        // Single row: [+1, +1, +1, +1] × [1,2,3,4] = 10
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        assert!((output[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_gemv_with_scale() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut output, 1, 4, 2.0);
        assert!((output[0] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_gemv_negative_weights() {
        let weights: Vec<i8> = vec![-1, -1, -1, -1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        assert!((output[0] - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_gemv_mixed_weights() {
        // [+1, -1, 0, +1] × [1,2,3,4] = 1 - 2 + 0 + 4 = 3
        let weights: Vec<i8> = vec![1, -1, 0, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        assert!((output[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_gemv_multi_row() {
        let weights: Vec<i8> = vec![
            1, 0, 0, 0, // row 0: dot = 1
            0, 1, 0, 0, // row 1: dot = 2
            0, 0, 1, 0, // row 2: dot = 3
            0, 0, 0, 1, // row 3: dot = 4
        ];
        let packed = pack_ternary_matrix(&weights, 4, 4);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 4];
        scalar_gemv(&packed, &input, &mut output, 4, 4, 1.0);
        assert_close(&output, &[1.0, 2.0, 3.0, 4.0], 1e-6, "identity rows");
    }

    #[test]
    fn test_scalar_gemv_non_multiple_of_4() {
        // 5 columns: tests remainder handling
        let weights: Vec<i8> = vec![1, -1, 1, -1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(5); // [1,2,3,4,5]
        let mut output = vec![0.0f32; 1];
        // 1 - 2 + 3 - 4 + 5 = 3
        scalar_gemv(&packed, &input, &mut output, 1, 5, 1.0);
        assert!((output[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_gemv_all_zeros() {
        let weights: Vec<i8> = vec![0, 0, 0, 0];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        assert!((output[0]).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_ternary_gemv_basic() {
        let weights: Vec<i8> = vec![1, -1, 0, 1];
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        scalar_ternary_gemv(&weights, &input, &mut output, 1, 4, 1.0);
        assert!((output[0] - 3.0).abs() < 1e-6);
    }

    // ── 3. neon_ternary_gemv ────────────────────────────────────────

    #[test]
    fn test_ternary_gemv_basic() {
        let weights: Vec<i8> = vec![1, -1, 0, 1];
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_ternary_gemv(&weights, &input, &mut output, 1, 4, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_ternary_gemv(&weights, &input, &mut output, 1, 4, 1.0);
        assert!((output[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_gemv_multi_row() {
        let weights: Vec<i8> = vec![
            1, 1, 1, 1, // row 0: 10
            -1, -1, -1, -1, // row 1: -10
        ];
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 2];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_ternary_gemv(&weights, &input, &mut output, 2, 4, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_ternary_gemv(&weights, &input, &mut output, 2, 4, 1.0);
        assert_close(&output, &[10.0, -10.0], 1e-6, "ternary multi-row");
    }

    #[test]
    fn test_ternary_gemv_with_scale() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_ternary_gemv(&weights, &input, &mut output, 1, 4, 0.5);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_ternary_gemv(&weights, &input, &mut output, 1, 4, 0.5);
        assert!((output[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_gemv_odd_cols() {
        // 5 cols → tests scalar tail
        let weights: Vec<i8> = vec![1, -1, 1, -1, 1];
        let input = ramp_input(5);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_ternary_gemv(&weights, &input, &mut output, 1, 5, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_ternary_gemv(&weights, &input, &mut output, 1, 5, 1.0);
        assert!((output[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_gemv_all_zeros() {
        let weights: Vec<i8> = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let input = ramp_input(8);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_ternary_gemv(&weights, &input, &mut output, 1, 8, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_ternary_gemv(&weights, &input, &mut output, 1, 8, 1.0);
        assert!((output[0]).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_gemv_vs_scalar() {
        let weights: Vec<i8> = vec![
            1, -1, 0, 1, 1, 0, -1, 1, // row 0
            0, 1, -1, 1, 0, -1, 1, 0, // row 1
            -1, 0, 1, 0, 1, -1, 0, 1, // row 2
        ];
        let input = ramp_input(8);
        let mut neon_out = vec![0.0f32; 3];
        let mut scalar_out = vec![0.0f32; 3];
        scalar_ternary_gemv(&weights, &input, &mut scalar_out, 3, 8, 1.5);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_ternary_gemv(&weights, &input, &mut neon_out, 3, 8, 1.5);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_ternary_gemv(&weights, &input, &mut neon_out, 3, 8, 1.5);
        assert_close(&neon_out, &scalar_out, 1e-5, "ternary vs scalar");
    }

    // ── 4. neon_i2s_gemv ────────────────────────────────────────────

    #[test]
    fn test_i2s_gemv_basic() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        assert!((output[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_i2s_gemv_vs_scalar() {
        let weights: Vec<i8> = vec![
            1, -1, 0, 1, -1, 0, 1, -1, // row 0
            0, 1, 1, -1, 0, -1, 0, 1, // row 1
        ];
        let packed = pack_ternary_matrix(&weights, 2, 8);
        let input = ramp_input(8);
        let mut neon_out = vec![0.0f32; 2];
        let mut scalar_out = vec![0.0f32; 2];
        scalar_gemv(&packed, &input, &mut scalar_out, 2, 8, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut neon_out, 2, 8, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut neon_out, 2, 8, 1.0);
        assert_close(&neon_out, &scalar_out, 1e-5, "i2s gemv vs scalar");
    }

    #[test]
    fn test_i2s_gemv_non_aligned() {
        // 7 cols: tests scalar tail
        let weights: Vec<i8> = vec![1, -1, 0, 1, 1, -1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(7);
        let mut neon_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut scalar_out, 1, 7, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut neon_out, 1, 7, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut neon_out, 1, 7, 1.0);
        assert_close(&neon_out, &scalar_out, 1e-5, "i2s gemv non-aligned");
    }

    #[test]
    fn test_i2s_gemv_single_col() {
        let weights: Vec<i8> = vec![1];
        let packed = pack_ternary_row(&weights);
        let input = vec![5.0f32];
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut output, 1, 1, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut output, 1, 1, 1.0);
        assert!((output[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_i2s_gemv_with_scale() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ones_input(4);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut output, 1, 4, 3.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut output, 1, 4, 3.0);
        assert!((output[0] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_i2s_gemv_large_matrix() {
        let rows = 16;
        let cols = 64;
        let weights: Vec<i8> = (0..rows * cols)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut neon_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        assert_close(&neon_out, &scalar_out, 1e-4, "i2s large matrix");
    }

    // ── 5. neon_i2s_tiled_gemv ──────────────────────────────────────

    #[test]
    fn test_tiled_gemv_basic() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_tiled_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_tiled_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        assert!((output[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_tiled_gemv_vs_scalar() {
        let rows = 8;
        let cols = 32;
        let weights: Vec<i8> = (0..rows * cols).map(|i| [1i8, -1, 0, 1][i % 4]).collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut tiled_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 1.0);
        assert_close(&tiled_out, &scalar_out, 1e-4, "tiled vs scalar");
    }

    #[test]
    fn test_tiled_gemv_odd_rows() {
        let rows = 5; // not divisible by TILE_ROWS=4
        let cols = 8;
        let weights: Vec<i8> = vec![1; rows * cols];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ones_input(cols);
        let mut tiled_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 1.0);
        assert_close(&tiled_out, &scalar_out, 1e-5, "tiled odd rows");
    }

    #[test]
    fn test_tiled_gemv_odd_cols() {
        let rows = 4;
        let cols = 13; // not divisible by 4
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut tiled_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 1.0);
        assert_close(&tiled_out, &scalar_out, 1e-4, "tiled odd cols");
    }

    #[test]
    fn test_tiled_gemv_large() {
        let rows = 17;
        let cols = 65;
        let weights: Vec<i8> = (0..rows * cols)
            .map(|i| match i % 5 {
                0 | 1 => 1,
                2 | 3 => -1,
                _ => 0,
            })
            .collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut tiled_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 2.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 2.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 2.0);
        assert_close(&tiled_out, &scalar_out, 1e-3, "tiled large");
    }

    #[test]
    fn test_tiled_gemv_single_row() {
        let weights: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(8);
        let mut output = vec![0.0f32; 1];
        let mut expected = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut expected, 1, 8, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_tiled_gemv(&packed, &input, &mut output, 1, 8, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_tiled_gemv(&packed, &input, &mut output, 1, 8, 1.0);
        assert_close(&output, &expected, 1e-5, "tiled single row");
    }

    // ── 6. neon_i2s_multirow_gemv ───────────────────────────────────

    #[test]
    fn test_multirow_gemv_basic() {
        let rows = 4;
        let cols = 8;
        let weights: Vec<i8> = vec![
            1, 0, -1, 0, 1, 0, -1, 0, // row 0
            0, 1, 0, -1, 0, 1, 0, -1, // row 1
            1, 1, 1, 1, 0, 0, 0, 0, // row 2
            0, 0, 0, 0, -1, -1, -1, -1, // row 3
        ];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut neon_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        assert_close(&neon_out, &scalar_out, 1e-5, "multirow basic");
    }

    #[test]
    fn test_multirow_gemv_remainder_rows() {
        let rows = 7; // 4 + 3 remainder rows
        let cols = 8;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut neon_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        assert_close(&neon_out, &scalar_out, 1e-4, "multirow remainder rows");
    }

    #[test]
    fn test_multirow_gemv_odd_cols() {
        let rows = 4;
        let cols = 11;
        let weights: Vec<i8> = vec![1; rows * cols];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ones_input(cols);
        let mut neon_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        assert_close(&neon_out, &scalar_out, 1e-5, "multirow odd cols");
    }

    #[test]
    fn test_multirow_gemv_with_scale() {
        let rows = 8;
        let cols = 16;
        let weights: Vec<i8> = (0..rows * cols).map(|i| [1i8, -1, 0][i % 3]).collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut neon_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 0.25);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 0.25);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 0.25);
        assert_close(&neon_out, &scalar_out, 1e-4, "multirow with scale");
    }

    #[test]
    fn test_multirow_gemv_single_row() {
        let weights: Vec<i8> = vec![1, -1, 1, -1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut neon_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut scalar_out, 1, 4, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, 1, 4, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, 1, 4, 1.0);
        assert_close(&neon_out, &scalar_out, 1e-5, "multirow single row");
    }

    // ── 7. neon_i2s_scale_fused_gemv ────────────────────────────────

    #[test]
    fn test_scale_fused_uniform_blocks() {
        // All blocks have scale 1.0 → equivalent to regular GEMV
        let rows = 2;
        let cols = 8;
        let block_size = 4;
        let weights: Vec<i8> = vec![
            1, 1, 1, 1, -1, -1, -1, -1, // row 0
            1, -1, 1, -1, 1, -1, 1, -1, // row 1
        ];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let block_scales = vec![1.0f32; 2];
        let mut neon_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_scale_fused_gemv(
            &packed,
            &input,
            &mut scalar_out,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_scale_fused_gemv(
                &packed,
                &input,
                &mut neon_out,
                rows,
                cols,
                block_size,
                &block_scales,
                1.0,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_scale_fused_gemv(
            &packed,
            &input,
            &mut neon_out,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        assert_close(&neon_out, &scalar_out, 1e-5, "scale fused uniform");
    }

    #[test]
    fn test_scale_fused_different_scales() {
        let rows = 1;
        let cols = 8;
        let block_size = 4;
        let weights: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ones_input(cols);
        let block_scales = vec![2.0f32, 0.5];
        let mut neon_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        // block0: 4*2.0=8, block1: 4*0.5=2 → total=10
        scalar_scale_fused_gemv(
            &packed,
            &input,
            &mut scalar_out,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_scale_fused_gemv(
                &packed,
                &input,
                &mut neon_out,
                rows,
                cols,
                block_size,
                &block_scales,
                1.0,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_scale_fused_gemv(
            &packed,
            &input,
            &mut neon_out,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        assert!((neon_out[0] - 10.0).abs() < 1e-5);
        assert_close(&neon_out, &scalar_out, 1e-5, "scale fused diff");
    }

    #[test]
    fn test_scale_fused_with_global_scale() {
        let rows = 1;
        let cols = 4;
        let block_size = 4;
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ones_input(4);
        let block_scales = vec![1.0f32];
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_scale_fused_gemv(
                &packed,
                &input,
                &mut output,
                rows,
                cols,
                block_size,
                &block_scales,
                3.0,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_scale_fused_gemv(
            &packed,
            &input,
            &mut output,
            rows,
            cols,
            block_size,
            &block_scales,
            3.0,
        );
        assert!((output[0] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale_fused_partial_block() {
        // 6 cols, block_size=4 → 2 blocks (4 + 2)
        let rows = 1;
        let cols = 6;
        let block_size = 4;
        let weights: Vec<i8> = vec![1, 1, 1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ones_input(cols);
        let block_scales = vec![2.0f32, 3.0];
        let mut neon_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        // block0: 4*2=8, block1: 2*3=6 → 14
        scalar_scale_fused_gemv(
            &packed,
            &input,
            &mut scalar_out,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_scale_fused_gemv(
                &packed,
                &input,
                &mut neon_out,
                rows,
                cols,
                block_size,
                &block_scales,
                1.0,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_scale_fused_gemv(
            &packed,
            &input,
            &mut neon_out,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        assert!((neon_out[0] - 14.0).abs() < 1e-5);
        assert_close(&neon_out, &scalar_out, 1e-5, "scale partial block");
    }

    #[test]
    fn test_scale_fused_multi_row() {
        let rows = 3;
        let cols = 8;
        let block_size = 4;
        let weights: Vec<i8> = (0..rows * cols).map(|i| [1i8, -1, 0][i % 3]).collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let block_scales = vec![1.5f32, 0.5];
        let mut neon_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_scale_fused_gemv(
            &packed,
            &input,
            &mut scalar_out,
            rows,
            cols,
            block_size,
            &block_scales,
            2.0,
        );
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_scale_fused_gemv(
                &packed,
                &input,
                &mut neon_out,
                rows,
                cols,
                block_size,
                &block_scales,
                2.0,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_scale_fused_gemv(
            &packed,
            &input,
            &mut neon_out,
            rows,
            cols,
            block_size,
            &block_scales,
            2.0,
        );
        assert_close(&neon_out, &scalar_out, 1e-4, "scale fused multi-row");
    }

    #[test]
    fn test_scalar_scale_fused_reference() {
        let rows = 1;
        let cols = 4;
        let block_size = 2;
        let weights: Vec<i8> = vec![1, 1, -1, -1];
        let packed = pack_ternary_row(&weights);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let block_scales = vec![1.0f32, 2.0];
        let mut output = vec![0.0f32; 1];
        // block0: (1+2)*1.0=3, block1: (-3-4)*2.0=-14 → total=-11
        scalar_scale_fused_gemv(
            &packed,
            &input,
            &mut output,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        assert!((output[0] - (-11.0)).abs() < 1e-5);
    }

    // ── 8. neon_i2s_kahan_gemv ──────────────────────────────────────

    #[test]
    fn test_kahan_gemv_basic() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_kahan_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_kahan_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        assert!((output[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_kahan_gemv_vs_scalar() {
        let rows = 4;
        let cols = 16;
        let weights: Vec<i8> = (0..rows * cols).map(|i| [1i8, -1, 0][i % 3]).collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut kahan_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_kahan_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_kahan_gemv(&packed, &input, &mut kahan_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_kahan_gemv(&packed, &input, &mut kahan_out, rows, cols, 1.0);
        assert_close(&kahan_out, &scalar_out, 1e-4, "kahan vs scalar");
    }

    #[test]
    fn test_kahan_gemv_vs_naive() {
        // Kahan and naive should agree for small vectors
        let weights: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(8);
        let mut kahan_out = vec![0.0f32; 1];
        let mut naive_out = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut naive_out, 1, 8, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_kahan_gemv(&packed, &input, &mut kahan_out, 1, 8, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_kahan_gemv(&packed, &input, &mut kahan_out, 1, 8, 1.0);
        assert_close(&kahan_out, &naive_out, 1e-5, "kahan vs naive");
    }

    #[test]
    fn test_kahan_gemv_with_scale() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ones_input(4);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_kahan_gemv(&packed, &input, &mut output, 1, 4, 2.5);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_kahan_gemv(&packed, &input, &mut output, 1, 4, 2.5);
        assert!((output[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_kahan_gemv_tail() {
        // 5 cols → tests scalar Kahan tail
        let weights: Vec<i8> = vec![1, 1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(5);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_kahan_gemv(&packed, &input, &mut output, 1, 5, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_kahan_gemv(&packed, &input, &mut output, 1, 5, 1.0);
        assert!((output[0] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_kahan_gemv_reference() {
        let weights: Vec<i8> = vec![1, -1, 1, -1];
        let packed = pack_ternary_row(&weights);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 1];
        // 1 - 2 + 3 - 4 = -2
        scalar_kahan_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        assert!((output[0] - (-2.0)).abs() < 1e-6);
    }

    // ── 9. neon_i2s_batched_gemv ────────────────────────────────────

    #[test]
    fn test_batched_gemv_single_batch() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_batched_gemv(&packed, &input, &mut output, 1, 4, 1, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_batched_gemv(&packed, &input, &mut output, 1, 4, 1, 1.0);
        assert!((output[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_batched_gemv_multi_batch() {
        let rows = 2;
        let cols = 4;
        let batch = 3;
        let weights: Vec<i8> = vec![
            1, 1, 1, 1, // row 0
            -1, -1, -1, -1, // row 1
        ];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let inputs: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            2.0, 2.0, 2.0, 2.0, // batch 1
            0.0, 0.0, 0.0, 1.0, // batch 2
        ];
        let mut output = vec![0.0f32; batch * rows];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_batched_gemv(&packed, &inputs, &mut output, rows, cols, batch, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_batched_gemv(&packed, &inputs, &mut output, rows, cols, batch, 1.0);
        // batch 0: [10, -10], batch 1: [8, -8], batch 2: [1, -1]
        let expected = [10.0, -10.0, 8.0, -8.0, 1.0, -1.0];
        assert_close(&output, &expected, 1e-5, "batched multi");
    }

    #[test]
    fn test_batched_gemv_with_scale() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let inputs: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let mut output = vec![0.0f32; 2];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_batched_gemv(&packed, &inputs, &mut output, 1, 4, 2, 0.5);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_batched_gemv(&packed, &inputs, &mut output, 1, 4, 2, 0.5);
        assert!((output[0] - 2.0).abs() < 1e-5);
        assert!((output[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_batched_gemv_vs_loop() {
        let rows = 3;
        let cols = 8;
        let batch = 4;
        let weights: Vec<i8> = (0..rows * cols).map(|i| [1i8, -1, 0][i % 3]).collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let inputs: Vec<f32> = (0..batch * cols).map(|i| (i as f32) * 0.1).collect();
        let mut batched_out = vec![0.0f32; batch * rows];
        let mut loop_out = vec![0.0f32; batch * rows];

        // Compute with loop of scalar_gemv
        for b in 0..batch {
            scalar_gemv(
                &packed,
                &inputs[b * cols..(b + 1) * cols],
                &mut loop_out[b * rows..(b + 1) * rows],
                rows,
                cols,
                1.0,
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_batched_gemv(&packed, &inputs, &mut batched_out, rows, cols, batch, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_batched_gemv(&packed, &inputs, &mut batched_out, rows, cols, batch, 1.0);
        assert_close(&batched_out, &loop_out, 1e-4, "batched vs loop");
    }

    // ── 10. neon_i2s_sparse_gemv ────────────────────────────────────

    #[test]
    fn test_sparse_gemv_all_zeros() {
        let packed = vec![0u8; 4]; // 4 bytes = 16 zero weights
        let input = ramp_input(16);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_sparse_gemv(&packed, &input, &mut output, 1, 16, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_sparse_gemv(&packed, &input, &mut output, 1, 16, 1.0);
        assert!((output[0]).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_gemv_vs_dense() {
        // Sparse and dense should produce identical results
        let weights: Vec<i8> = vec![0, 0, 0, 0, 1, -1, 0, 0]; // mostly zeros
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(8);
        let mut sparse_out = vec![0.0f32; 1];
        let mut dense_out = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut dense_out, 1, 8, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 8, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 8, 1.0);
        assert_close(&sparse_out, &dense_out, 1e-5, "sparse vs dense");
    }

    #[test]
    fn test_sparse_gemv_fully_dense() {
        // No zero bytes → same as regular GEMV
        let weights: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(8);
        let mut sparse_out = vec![0.0f32; 1];
        let mut dense_out = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut dense_out, 1, 8, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 8, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 8, 1.0);
        assert_close(&sparse_out, &dense_out, 1e-5, "sparse fully dense");
    }

    #[test]
    fn test_sparse_gemv_multi_row() {
        let rows = 4;
        let cols = 16;
        // Make rows 0 and 2 all-zero
        let mut weights: Vec<i8> = vec![0; rows * cols];
        for c in 0..cols {
            weights[1 * cols + c] = if c % 2 == 0 { 1 } else { -1 };
            weights[3 * cols + c] = 1;
        }
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut sparse_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, rows, cols, 1.0);
        assert_close(&sparse_out, &scalar_out, 1e-4, "sparse multi-row");
    }

    #[test]
    fn test_sparse_gemv_tail_only() {
        // 5 cols → first byte non-zero, tail byte zero
        let weights: Vec<i8> = vec![1, 1, 1, 1, 0];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(5);
        let mut sparse_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut scalar_out, 1, 5, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 5, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 5, 1.0);
        assert_close(&sparse_out, &scalar_out, 1e-5, "sparse tail");
    }

    #[test]
    fn test_sparse_gemv_with_scale() {
        let weights: Vec<i8> = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ones_input(8);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_sparse_gemv(&packed, &input, &mut output, 1, 8, 2.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_sparse_gemv(&packed, &input, &mut output, 1, 8, 2.0);
        assert!((output[0] - 8.0).abs() < 1e-5);
    }

    // ── 11. Cross-kernel consistency ────────────────────────────────

    #[test]
    fn test_all_kernels_agree_4x8() {
        let rows = 4;
        let cols = 8;
        let weights: Vec<i8> = (0..rows * cols).map(|i| [1i8, -1, 0][i % 3]).collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let scale = 1.0;

        let mut ref_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut ref_out, rows, cols, scale);

        let mut i2s_out = vec![0.0f32; rows];
        let mut tiled_out = vec![0.0f32; rows];
        let mut multi_out = vec![0.0f32; rows];
        let mut kahan_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut i2s_out, rows, cols, scale);
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, scale);
            neon_i2s_multirow_gemv(&packed, &input, &mut multi_out, rows, cols, scale);
            neon_i2s_kahan_gemv(&packed, &input, &mut kahan_out, rows, cols, scale);
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, rows, cols, scale);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            neon_i2s_gemv(&packed, &input, &mut i2s_out, rows, cols, scale);
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, scale);
            neon_i2s_multirow_gemv(&packed, &input, &mut multi_out, rows, cols, scale);
            neon_i2s_kahan_gemv(&packed, &input, &mut kahan_out, rows, cols, scale);
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, rows, cols, scale);
        }

        assert_close(&i2s_out, &ref_out, 1e-5, "i2s vs ref");
        assert_close(&tiled_out, &ref_out, 1e-4, "tiled vs ref");
        assert_close(&multi_out, &ref_out, 1e-5, "multi vs ref");
        assert_close(&kahan_out, &ref_out, 1e-4, "kahan vs ref");
        assert_close(&sparse_out, &ref_out, 1e-5, "sparse vs ref");
    }

    #[test]
    fn test_all_kernels_agree_odd() {
        let rows = 5;
        let cols = 13;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let scale = 0.5;

        let mut ref_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut ref_out, rows, cols, scale);

        let mut i2s_out = vec![0.0f32; rows];
        let mut tiled_out = vec![0.0f32; rows];
        let mut multi_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut i2s_out, rows, cols, scale);
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, scale);
            neon_i2s_multirow_gemv(&packed, &input, &mut multi_out, rows, cols, scale);
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, rows, cols, scale);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            neon_i2s_gemv(&packed, &input, &mut i2s_out, rows, cols, scale);
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, scale);
            neon_i2s_multirow_gemv(&packed, &input, &mut multi_out, rows, cols, scale);
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, rows, cols, scale);
        }

        assert_close(&i2s_out, &ref_out, 1e-4, "i2s vs ref odd");
        assert_close(&tiled_out, &ref_out, 1e-3, "tiled vs ref odd");
        assert_close(&multi_out, &ref_out, 1e-4, "multi vs ref odd");
        assert_close(&sparse_out, &ref_out, 1e-4, "sparse vs ref odd");
    }

    #[test]
    fn test_all_kernels_zero_scale() {
        let rows = 4;
        let cols = 8;
        let weights: Vec<i8> = vec![1; rows * cols];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let zeros = vec![0.0f32; rows];

        let mut i2s_out = vec![0.0f32; rows];
        let mut tiled_out = vec![0.0f32; rows];
        let mut multi_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut i2s_out, rows, cols, 0.0);
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 0.0);
            neon_i2s_multirow_gemv(&packed, &input, &mut multi_out, rows, cols, 0.0);
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, rows, cols, 0.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            neon_i2s_gemv(&packed, &input, &mut i2s_out, rows, cols, 0.0);
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 0.0);
            neon_i2s_multirow_gemv(&packed, &input, &mut multi_out, rows, cols, 0.0);
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, rows, cols, 0.0);
        }

        assert_close(&i2s_out, &zeros, 1e-6, "i2s zero scale");
        assert_close(&tiled_out, &zeros, 1e-6, "tiled zero scale");
        assert_close(&multi_out, &zeros, 1e-6, "multi zero scale");
        assert_close(&sparse_out, &zeros, 1e-6, "sparse zero scale");
    }

    #[test]
    fn test_all_kernels_single_element() {
        let packed = pack_ternary_row(&[1i8]);
        let input = vec![7.0f32];
        let mut i2s_out = vec![0.0f32; 1];
        let mut tiled_out = vec![0.0f32; 1];
        let mut multi_out = vec![0.0f32; 1];
        let mut sparse_out = vec![0.0f32; 1];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut i2s_out, 1, 1, 1.0);
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, 1, 1, 1.0);
            neon_i2s_multirow_gemv(&packed, &input, &mut multi_out, 1, 1, 1.0);
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 1, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            neon_i2s_gemv(&packed, &input, &mut i2s_out, 1, 1, 1.0);
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, 1, 1, 1.0);
            neon_i2s_multirow_gemv(&packed, &input, &mut multi_out, 1, 1, 1.0);
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 1, 1.0);
        }

        let expected = [7.0f32];
        assert_close(&i2s_out, &expected, 1e-6, "single i2s");
        assert_close(&tiled_out, &expected, 1e-6, "single tiled");
        assert_close(&multi_out, &expected, 1e-6, "single multi");
        assert_close(&sparse_out, &expected, 1e-6, "single sparse");
    }

    // ── 12. Edge cases ──────────────────────────────────────────────

    #[test]
    fn test_i2s_gemv_exactly_4_cols() {
        let weights: Vec<i8> = vec![1, -1, 1, -1];
        let packed = pack_ternary_row(&weights);
        let input = vec![2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut output, 1, 4, 1.0);
        // 2 - 3 + 4 - 5 = -2
        assert!((output[0] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_i2s_gemv_2_cols() {
        let weights: Vec<i8> = vec![1, -1];
        let packed = pack_ternary_row(&weights);
        let input = vec![10.0, 3.0];
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut output, 1, 2, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut output, 1, 2, 1.0);
        assert!((output[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_i2s_gemv_3_cols() {
        let weights: Vec<i8> = vec![1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut output, 1, 3, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut output, 1, 3, 1.0);
        assert!((output[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_scale() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(4);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut output, 1, 4, -1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut output, 1, 4, -1.0);
        assert!((output[0] - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_large_rows_multirow() {
        let rows = 33; // 8 groups of 4 + 1 remainder
        let cols = 4;
        let weights: Vec<i8> = vec![1; rows * cols];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ones_input(cols);
        let mut neon_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_multirow_gemv(&packed, &input, &mut neon_out, rows, cols, 1.0);
        assert_close(&neon_out, &scalar_out, 1e-5, "large rows multirow");
    }

    #[test]
    fn test_alternating_pattern() {
        // Alternating +1/-1 weights with uniform input → sum should be 0
        let cols = 16;
        let weights: Vec<i8> = (0..cols).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let packed = pack_ternary_row(&weights);
        let input = ones_input(cols);
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut output, 1, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut output, 1, cols, 1.0);
        assert!((output[0]).abs() < 1e-6);
    }

    #[test]
    fn test_batched_gemv_zero_batch() {
        let packed = pack_ternary_row(&[1, 1, 1, 1]);
        let inputs: Vec<f32> = vec![];
        let mut outputs: Vec<f32> = vec![];
        // Zero batch should be a no-op
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_batched_gemv(&packed, &inputs, &mut outputs, 1, 4, 0, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_batched_gemv(&packed, &inputs, &mut outputs, 1, 4, 0, 1.0);
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_scale_fused_block_size_1() {
        // Each column is its own block
        let rows = 1;
        let cols = 4;
        let block_size = 1;
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let block_scales = vec![1.0, 2.0, 3.0, 4.0];
        let mut neon_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        // 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
        scalar_scale_fused_gemv(
            &packed,
            &input,
            &mut scalar_out,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_scale_fused_gemv(
                &packed,
                &input,
                &mut neon_out,
                rows,
                cols,
                block_size,
                &block_scales,
                1.0,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_scale_fused_gemv(
            &packed,
            &input,
            &mut neon_out,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        assert!((neon_out[0] - 30.0).abs() < 1e-4);
        assert_close(&neon_out, &scalar_out, 1e-4, "block_size=1");
    }

    #[test]
    fn test_scale_fused_block_larger_than_cols() {
        let rows = 1;
        let cols = 4;
        let block_size = 16; // block covers entire row
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_ternary_row(&weights);
        let input = ones_input(4);
        let block_scales = vec![2.0f32];
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_scale_fused_gemv(
                &packed,
                &input,
                &mut output,
                rows,
                cols,
                block_size,
                &block_scales,
                1.0,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_scale_fused_gemv(
            &packed,
            &input,
            &mut output,
            rows,
            cols,
            block_size,
            &block_scales,
            1.0,
        );
        assert!((output[0] - 8.0).abs() < 1e-5);
    }

    // ── 13. Additional edge-case and stress tests ───────────────────

    #[test]
    fn test_i2s_gemv_16_cols() {
        let weights: Vec<i8> = vec![1; 16];
        let packed = pack_ternary_row(&weights);
        let input = ones_input(16);
        let mut output = vec![0.0f32; 1];
        let mut expected = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut expected, 1, 16, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_gemv(&packed, &input, &mut output, 1, 16, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_gemv(&packed, &input, &mut output, 1, 16, 1.0);
        assert_close(&output, &expected, 1e-6, "16 cols");
    }

    #[test]
    fn test_ternary_gemv_large_scale() {
        let weights: Vec<i8> = vec![1, -1, 1, -1];
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0f32; 1];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_ternary_gemv(&weights, &input, &mut output, 1, 4, 1000.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_ternary_gemv(&weights, &input, &mut output, 1, 4, 1000.0);
        assert!((output[0]).abs() < 1e-3);
    }

    #[test]
    fn test_tiled_gemv_exact_tile_fit() {
        // TILE_ROWS=4 rows, TILE_K_BYTES=8 → 32 cols exactly
        let rows = 4;
        let cols = 32;
        let weights: Vec<i8> = vec![1; rows * cols];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ones_input(cols);
        let mut tiled_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_tiled_gemv(&packed, &input, &mut tiled_out, rows, cols, 1.0);
        assert_close(&tiled_out, &scalar_out, 1e-5, "exact tile fit");
    }

    #[test]
    fn test_multirow_gemv_exactly_4_rows() {
        let rows = 4;
        let cols = 4;
        let weights: Vec<i8> = vec![
            1, 0, 0, 0, // row 0
            0, 1, 0, 0, // row 1
            0, 0, 1, 0, // row 2
            0, 0, 0, 1, // row 3
        ];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0f32; rows];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_multirow_gemv(&packed, &input, &mut output, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_multirow_gemv(&packed, &input, &mut output, rows, cols, 1.0);
        assert_close(&output, &[10.0, 20.0, 30.0, 40.0], 1e-5, "identity 4x4");
    }

    #[test]
    fn test_kahan_gemv_multi_row() {
        let rows = 3;
        let cols = 8;
        let weights: Vec<i8> = vec![1; rows * cols];
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ones_input(cols);
        let mut kahan_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_kahan_gemv(&packed, &input, &mut scalar_out, rows, cols, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_kahan_gemv(&packed, &input, &mut kahan_out, rows, cols, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_kahan_gemv(&packed, &input, &mut kahan_out, rows, cols, 1.0);
        assert_close(&kahan_out, &scalar_out, 1e-5, "kahan multi-row");
    }

    #[test]
    fn test_sparse_gemv_single_nonzero_byte() {
        // Only byte 2 (cols 8-11) is non-zero
        let mut weights: Vec<i8> = vec![0; 16];
        weights[8] = 1;
        weights[9] = 1;
        weights[10] = 1;
        weights[11] = 1;
        let packed = pack_ternary_row(&weights);
        let input = ramp_input(16);
        let mut sparse_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        scalar_gemv(&packed, &input, &mut scalar_out, 1, 16, 1.0);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 16, 1.0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        neon_i2s_sparse_gemv(&packed, &input, &mut sparse_out, 1, 16, 1.0);
        // 9+10+11+12 = 42
        assert!((sparse_out[0] - 42.0).abs() < 1e-5);
        assert_close(&sparse_out, &scalar_out, 1e-5, "sparse single nonzero");
    }

    #[test]
    fn test_pack_ternary_matrix_roundtrip() {
        let weights: Vec<i8> = vec![
            1, -1, 0, 1, 0, -1, 1, 0, // row 0
            0, 1, -1, 0, 1, 0, -1, 1, // row 1
        ];
        let rows = 2;
        let cols = 8;
        let packed = pack_ternary_matrix(&weights, rows, cols);
        let input = ramp_input(cols);
        let mut packed_out = vec![0.0f32; rows];
        let mut scalar_out = vec![0.0f32; rows];
        scalar_gemv(&packed, &input, &mut packed_out, rows, cols, 1.0);
        scalar_ternary_gemv(&weights, &input, &mut scalar_out, rows, cols, 1.0);
        assert_close(&packed_out, &scalar_out, 1e-6, "pack roundtrip");
    }
}
