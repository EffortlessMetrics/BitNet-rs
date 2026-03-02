//! ARM NEON quantized GEMM kernels for Apple Silicon.
//!
//! Provides NEON-accelerated general matrix multiply (GEMM) operations for
//! I2_S 2-bit quantized weights against f32 and f16 activations. Includes
//! tiled, batched, transposed, and fused variants for cache efficiency
//! and reduced memory traffic.
//!
//! I2_S encoding (2 bits per value, 4 values per byte, LSB-first):
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── I2_S decode helpers ────────────────────────────────────────────

/// Decode a single 2-bit I2_S code to its signed float value.
#[cfg(test)]
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0, // 0b00 = 0, 0b10 = unused → 0
    }
}

/// I2_S f32 LUT: index by 2-bit code → f32 value.
const I2S_LUT: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

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

/// Scalar f16→f32 conversion for f16 GEMM support.
#[inline(always)]
fn f16_to_f32_scalar(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((127 - 15 + 1) as i32 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 0x1F {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }
    let f32_exp = exp + (127 - 15);
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

// ── 1. I2_S × f32 GEMM ────────────────────────────────────────────

/// NEON I2_S × f32 GEMM: `C[m×n] += scale · dequant(W)[m×k] · A[k×n]`.
///
/// Weight matrix `W` is row-major I2_S packed (`m` rows, each row
/// `ceil(k/4)` bytes). `A` is row-major f32 `[k×n]`, `C` is
/// row-major f32 `[m×n]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_f32_gemm(
    weights_packed: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);
    debug_assert!(weights_packed.len() >= m * packed_k);
    debug_assert!(activations.len() >= k * n);
    debug_assert!(output.len() >= m * n);

    let a_ptr = activations.as_ptr();

    for row in 0..m {
        let w_row = row * packed_k;
        for col in 0..n {
            let full_bytes = k / 4;

            let mut acc = vdupq_n_f32(0.0);

            for b in 0..full_bytes {
                let vals = unpack_byte_f32(weights_packed[w_row + b]);
                let vw = unsafe { vld1q_f32(vals.as_ptr()) };
                let a_arr = unsafe {
                    [
                        *a_ptr.add((b * 4) * n + col),
                        *a_ptr.add((b * 4 + 1) * n + col),
                        *a_ptr.add((b * 4 + 2) * n + col),
                        *a_ptr.add((b * 4 + 3) * n + col),
                    ]
                };
                let va = unsafe { vld1q_f32(a_arr.as_ptr()) };
                acc = vfmaq_f32(acc, vw, va);
            }

            let mut sum = vaddvq_f32(acc);

            let tail = full_bytes * 4;
            if tail < k {
                let byte = weights_packed[w_row + full_bytes];
                for j in 0..(k - tail) {
                    let w_val = I2S_LUT[((byte >> (j * 2)) & 0x03) as usize];
                    sum += w_val * unsafe { *a_ptr.add((tail + j) * n + col) };
                }
            }

            output[row * n + col] += sum * scale;
        }
    }
}

// ── 2. I2_S × f16 GEMM ────────────────────────────────────────────

/// NEON I2_S × f16 GEMM using scalar f16↔f32 conversion.
///
/// Activations are stored as raw `u16` f16 bit-patterns. Internally
/// converts to f32 via [`f16_to_f32_scalar`], computes in f32, then
/// stores the result as f32 in `output`.
///
/// Layout matches [`neon_i2s_f32_gemm`]: `W[m×k]` packed, `A[k×n]`
/// row-major f16, `C[m×n]` row-major f32.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_f16_gemm(
    weights_packed: &[u8],
    activations_f16: &[u16],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);
    debug_assert!(weights_packed.len() >= m * packed_k);
    debug_assert!(activations_f16.len() >= k * n);
    debug_assert!(output.len() >= m * n);

    for row in 0..m {
        let w_row = row * packed_k;
        for col in 0..n {
            let full_bytes = k / 4;

            let mut acc = vdupq_n_f32(0.0);

            for b in 0..full_bytes {
                let vals = unpack_byte_f32(weights_packed[w_row + b]);
                let vw = unsafe { vld1q_f32(vals.as_ptr()) };
                // Convert 4 f16→f32 via scalar, then NEON dot
                let a_arr = [
                    f16_to_f32_scalar(activations_f16[(b * 4) * n + col]),
                    f16_to_f32_scalar(activations_f16[(b * 4 + 1) * n + col]),
                    f16_to_f32_scalar(activations_f16[(b * 4 + 2) * n + col]),
                    f16_to_f32_scalar(activations_f16[(b * 4 + 3) * n + col]),
                ];
                let va = unsafe { vld1q_f32(a_arr.as_ptr()) };
                acc = vfmaq_f32(acc, vw, va);
            }

            let mut sum = vaddvq_f32(acc);

            let tail = full_bytes * 4;
            if tail < k {
                let byte = weights_packed[w_row + full_bytes];
                for j in 0..(k - tail) {
                    let w_val = I2S_LUT[((byte >> (j * 2)) & 0x03) as usize];
                    let a_f32 = f16_to_f32_scalar(activations_f16[(tail + j) * n + col]);
                    sum += w_val * a_f32;
                }
            }

            output[row * n + col] += sum * scale;
        }
    }
}

// ── 3. Tiled GEMM ─────────────────────────────────────────────────

/// Tile size for the row dimension.
const TILE_M: usize = 4;
/// Tile size for the column dimension.
const TILE_N: usize = 4;

/// NEON tiled I2_S × f32 GEMM with `TILE_M × TILE_N` register tiles.
///
/// Tiles the output to improve register reuse and cache locality.
/// Falls back to scalar for edge tiles.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_tiled_gemm(
    weights_packed: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);
    debug_assert!(weights_packed.len() >= m * packed_k);
    debug_assert!(activations.len() >= k * n);
    debug_assert!(output.len() >= m * n);

    let m_tiles = m / TILE_M;
    let n_tiles = n / TILE_N;

    for mt in 0..m_tiles {
        for nt in 0..n_tiles {
            let row0 = mt * TILE_M;
            let col0 = nt * TILE_N;

            let mut acc = [[0.0f32; TILE_N]; TILE_M];

            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;

                for ti in 0..TILE_M {
                    let r = row0 + ti;
                    let byte = weights_packed[r * packed_k + byte_idx];
                    let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];

                    for tj in 0..TILE_N {
                        let c = col0 + tj;
                        acc[ti][tj] += w_val * activations[kk * n + c];
                    }
                }
            }

            for ti in 0..TILE_M {
                for tj in 0..TILE_N {
                    output[(row0 + ti) * n + (col0 + tj)] += acc[ti][tj] * scale;
                }
            }
        }
    }

    // Edge rows
    let row_rem = m_tiles * TILE_M;
    for row in row_rem..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let byte = weights_packed[row * packed_k + byte_idx];
                let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];
                sum += w_val * activations[kk * n + col];
            }
            output[row * n + col] += sum * scale;
        }
    }

    // Edge cols (for full row-tiles only)
    let col_rem = n_tiles * TILE_N;
    for row in 0..row_rem {
        for col in col_rem..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let byte = weights_packed[row * packed_k + byte_idx];
                let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];
                sum += w_val * activations[kk * n + col];
            }
            output[row * n + col] += sum * scale;
        }
    }
}

// ── 4. Batched GEMM ───────────────────────────────────────────────

/// Batched I2_S × f32 GEMM with shared weights.
///
/// Computes `C_b[m×n] += scale · dequant(W)[m×k] · A_b[k×n]` for
/// `batch_size` independent activation matrices.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_batched_gemm(
    weights_packed: &[u8],
    activations_batch: &[f32],
    output_batch: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    batch_size: usize,
    scale: f32,
) {
    let a_stride = k * n;
    let c_stride = m * n;
    debug_assert!(activations_batch.len() >= batch_size * a_stride);
    debug_assert!(output_batch.len() >= batch_size * c_stride);

    for b in 0..batch_size {
        let a_off = b * a_stride;
        let c_off = b * c_stride;
        unsafe {
            neon_i2s_f32_gemm(
                weights_packed,
                &activations_batch[a_off..a_off + a_stride],
                &mut output_batch[c_off..c_off + c_stride],
                m,
                k,
                n,
                scale,
            );
        }
    }
}

// ── 5. Transposed GEMM ────────────────────────────────────────────

/// Transposed-weight GEMM: `C[m×n] += scale · dequant(W)^T · A`.
///
/// `W` is packed as `[k × m]` row-major I2_S; the effective weight
/// matrix is `W^T[m × k]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_gemm_wt(
    weights_packed: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_m = m.div_ceil(4);
    debug_assert!(weights_packed.len() >= k * packed_m);
    debug_assert!(activations.len() >= k * n);
    debug_assert!(output.len() >= m * n);

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk * packed_m + row / 4;
                let bit_off = (row % 4) * 2;
                let byte = weights_packed[byte_idx];
                let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];
                sum += w_val * activations[kk * n + col];
            }
            output[row * n + col] += sum * scale;
        }
    }
}

/// Transposed-activation GEMM: `C[m×n] += scale · dequant(W) · A^T`.
///
/// `A` is stored as `[n × k]` row-major (transposed from `[k × n]`).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_gemm_at(
    weights_packed: &[u8],
    activations_t: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);
    debug_assert!(weights_packed.len() >= m * packed_k);
    debug_assert!(activations_t.len() >= n * k);
    debug_assert!(output.len() >= m * n);

    for row in 0..m {
        let w_row = row * packed_k;
        for col in 0..n {
            let full_bytes = k / 4;

            let mut acc = vdupq_n_f32(0.0);

            for b in 0..full_bytes {
                let vals = unpack_byte_f32(weights_packed[w_row + b]);
                let vw = unsafe { vld1q_f32(vals.as_ptr()) };
                let va = unsafe { vld1q_f32(activations_t.as_ptr().add(col * k + b * 4)) };
                acc = vfmaq_f32(acc, vw, va);
            }

            let mut sum = vaddvq_f32(acc);

            let tail = full_bytes * 4;
            if tail < k {
                let byte = weights_packed[w_row + full_bytes];
                for j in 0..(k - tail) {
                    let w_val = I2S_LUT[((byte >> (j * 2)) & 0x03) as usize];
                    sum += w_val * activations_t[col * k + tail + j];
                }
            }

            output[row * n + col] += sum * scale;
        }
    }
}

// ── 6. Fused GEMM + bias ──────────────────────────────────────────

/// Fused I2_S × f32 GEMM + bias: `C[m×n] = scale · dequant(W)[m×k]
/// · A[k×n] + bias[n]`.
///
/// Bias is broadcast across all rows. Output is *written* (not
/// accumulated).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_gemm_bias(
    weights_packed: &[u8],
    activations: &[f32],
    bias: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    debug_assert!(bias.len() >= n);

    output[..m * n].fill(0.0);
    unsafe {
        neon_i2s_f32_gemm(weights_packed, activations, output, m, k, n, scale);
    }

    // Fuse bias addition via NEON
    for row in 0..m {
        let row_off = row * n;
        let full = n / 4;
        for i in 0..full {
            let off = row_off + i * 4;
            unsafe {
                let vc = vld1q_f32(output.as_ptr().add(off));
                let vb = vld1q_f32(bias.as_ptr().add(i * 4));
                vst1q_f32(output.as_mut_ptr().add(off), vaddq_f32(vc, vb));
            }
        }
        for j in (full * 4)..n {
            output[row_off + j] += bias[j];
        }
    }
}

// ── 7. Fused GEMM + ReLU ──────────────────────────────────────────

/// Fused I2_S × f32 GEMM + ReLU: `C[m×n] = max(0, scale ·
/// dequant(W)[m×k] · A[k×n])`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_gemm_relu(
    weights_packed: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    output[..m * n].fill(0.0);
    unsafe {
        neon_i2s_f32_gemm(weights_packed, activations, output, m, k, n, scale);
    }

    let zero = vdupq_n_f32(0.0);
    let total = m * n;
    let full = total / 4;
    for i in 0..full {
        let off = i * 4;
        unsafe {
            let v = vld1q_f32(output.as_ptr().add(off));
            vst1q_f32(output.as_mut_ptr().add(off), vmaxq_f32(v, zero));
        }
    }
    for j in (full * 4)..total {
        if output[j] < 0.0 {
            output[j] = 0.0;
        }
    }
}

// ── 8. QK256 GEMM ─────────────────────────────────────────────────

/// QK256 block size.
const QK256_BLOCK: usize = 256;

/// QK256 I2_S GEMM with per-block scale factors.
///
/// Weights are packed in 256-element blocks along the `k` dimension.
/// Each block has an independent scale factor per row. Layout:
///
/// - `weights_packed`: I2_S packed, `m * ceil(k/4)` bytes, row-major
/// - `block_scales`: `m * num_blocks_k` f32 values
/// - `activations`: row-major `[k × n]` f32
/// - `output`: row-major `[m × n]` f32
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_qk256_gemm(
    weights_packed: &[u8],
    block_scales: &[f32],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let packed_k = k.div_ceil(4);
    let num_blocks = k.div_ceil(QK256_BLOCK);
    debug_assert!(weights_packed.len() >= m * packed_k);
    debug_assert!(block_scales.len() >= m * num_blocks);
    debug_assert!(activations.len() >= k * n);
    debug_assert!(output.len() >= m * n);

    let a_ptr = activations.as_ptr();

    for row in 0..m {
        let w_row = row * packed_k;
        let s_row = row * num_blocks;

        for col in 0..n {
            let mut total = 0.0f32;

            for blk in 0..num_blocks {
                let blk_start = blk * QK256_BLOCK;
                let blk_end = (blk_start + QK256_BLOCK).min(k);
                let blk_len = blk_end - blk_start;
                let blk_scale = block_scales[s_row + blk];

                let mut acc = vdupq_n_f32(0.0);
                let full_iters = blk_len / 4;
                let blk_byte_start = blk_start / 4;

                for b in 0..full_iters {
                    let byte = weights_packed[w_row + blk_byte_start + b];
                    let vals = unpack_byte_f32(byte);
                    let vw = unsafe { vld1q_f32(vals.as_ptr()) };

                    let kk = blk_start + b * 4;
                    let a_arr = unsafe {
                        [
                            *a_ptr.add(kk * n + col),
                            *a_ptr.add((kk + 1) * n + col),
                            *a_ptr.add((kk + 2) * n + col),
                            *a_ptr.add((kk + 3) * n + col),
                        ]
                    };
                    let va = unsafe { vld1q_f32(a_arr.as_ptr()) };
                    acc = vfmaq_f32(acc, vw, va);
                }

                let mut blk_sum = vaddvq_f32(acc);

                let tail = full_iters * 4;
                if tail < blk_len {
                    let byte = weights_packed[w_row + blk_byte_start + full_iters];
                    for j in 0..(blk_len - tail) {
                        let w_val = I2S_LUT[((byte >> (j * 2)) & 0x03) as usize];
                        let kk = blk_start + tail + j;
                        blk_sum += w_val * activations[kk * n + col];
                    }
                }

                total += blk_sum * blk_scale;
            }

            output[row * n + col] += total;
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── scalar reference helpers ───────────────────────────────────

    /// Pack a row-major i8 matrix `[rows × cols]` into row-major I2_S.
    fn pack_row_major(vals: &[i8], rows: usize, cols: usize) -> Vec<u8> {
        let packed_cols = cols.div_ceil(4);
        let mut packed = vec![0u8; rows * packed_cols];
        for row in 0..rows {
            for col in 0..cols {
                let v = vals[row * cols + col];
                let code: u8 = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                let byte_idx = row * packed_cols + col / 4;
                let bit_off = (col % 4) * 2;
                packed[byte_idx] |= code << bit_off;
            }
        }
        packed
    }

    /// Scalar GEMM reference: `C[m×n] = scale · W[m×k] · A[k×n]`.
    fn scalar_gemm(w: &[i8], a: &[f32], m: usize, k: usize, n: usize, scale: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += (w[row * k + kk] as f32) * a[kk * n + col];
                }
                out[row * n + col] = sum * scale;
            }
        }
        out
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol {tol})");
        }
    }

    // ── 1. I2_S × f32 GEMM tests ─────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f32_gemm_identity() {
        let w: [i8; 4] = [1, 0, 0, 1];
        let packed = pack_row_major(&w, 2, 2);
        let a = [3.0f32, 5.0, 7.0, 11.0];
        let mut out = [0.0f32; 4];
        unsafe {
            neon_i2s_f32_gemm(&packed, &a, &mut out, 2, 2, 2, 1.0);
        }
        assert_close(&out, &[3.0, 5.0, 7.0, 11.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f32_gemm_scale() {
        let w = [1i8; 4];
        let packed = pack_row_major(&w, 2, 2);
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe {
            neon_i2s_f32_gemm(&packed, &a, &mut out, 2, 2, 2, 2.0);
        }
        assert_close(&out, &[8.0, 12.0, 8.0, 12.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f32_gemm_vs_scalar() {
        let m = 3;
        let k = 7;
        let n = 5;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let expected = scalar_gemm(&w, &a, m, k, n, 0.5);
        let packed = pack_row_major(&w, m, k);
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_f32_gemm(&packed, &a, &mut out, m, k, n, 0.5);
        }
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f32_gemm_zero_weights() {
        let w = [0i8; 6];
        let packed = pack_row_major(&w, 2, 3);
        let a = [1.0f32; 6];
        let mut out = [0.0f32; 4];
        unsafe {
            neon_i2s_f32_gemm(&packed, &a, &mut out, 2, 3, 2, 1.0);
        }
        assert_close(&out, &[0.0; 4], 1e-6);
    }

    // ── 2. I2_S × f16 GEMM tests ─────────────────────────────────

    #[test]
    fn test_f16_scalar_conversion() {
        assert!((f16_to_f32_scalar(0x3C00) - 1.0).abs() < 1e-4);
        assert!((f16_to_f32_scalar(0x0000)).abs() < 1e-8);
        assert!((f16_to_f32_scalar(0xBC00) + 1.0).abs() < 1e-4);
        assert!((f16_to_f32_scalar(0x3800) - 0.5).abs() < 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f16_gemm_identity() {
        let w: [i8; 4] = [1, 0, 0, 1];
        let packed = pack_row_major(&w, 2, 2);
        // f16: 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200, 4.0=0x4400
        let a_f16: [u16; 4] = [0x3C00, 0x4000, 0x4200, 0x4400];
        let mut out = [0.0f32; 4];
        unsafe {
            neon_i2s_f16_gemm(&packed, &a_f16, &mut out, 2, 2, 2, 1.0);
        }
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 0.01);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f16_gemm_vs_f32() {
        let m = 2;
        let k = 4;
        let n = 2;
        let w: Vec<i8> = vec![1, -1, 0, 1, 0, 1, -1, 0];
        let packed = pack_row_major(&w, m, k);
        // f32 activations
        let a_f32 = [1.0f32, 2.0, 0.5, 1.5, 3.0, 0.25, 2.0, 1.0];
        // Convert to f16 bit-patterns
        let a_f16: Vec<u16> = a_f32.iter().map(|&v| f32_to_f16_scalar(v)).collect();
        let mut out_f32 = vec![0.0f32; m * n];
        let mut out_f16 = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_f32_gemm(&packed, &a_f32, &mut out_f32, m, k, n, 1.0);
            neon_i2s_f16_gemm(&packed, &a_f16, &mut out_f16, m, k, n, 1.0);
        }
        // f16 has limited precision, so use larger tolerance
        assert_close(&out_f16, &out_f32, 0.05);
    }

    /// Helper: scalar f32→f16 for test data generation.
    fn f32_to_f16_scalar(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;
        if exp == 0 {
            return (sign << 15) as u16;
        }
        if exp == 0xFF {
            return ((sign << 15) | (0x1F << 10) | (mant >> 13)) as u16;
        }
        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            return (sign << 15) as u16;
        }
        if new_exp >= 0x1F {
            return ((sign << 15) | (0x1F << 10)) as u16;
        }
        ((sign << 15) | ((new_exp as u32) << 10) | (mant >> 13)) as u16
    }

    // ── 3. Tiled GEMM tests ───────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_tiled_gemm_exact_tile() {
        let m = 4;
        let k = 8;
        let n = 4;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let expected = scalar_gemm(&w, &a, m, k, n, 1.0);
        let packed = pack_row_major(&w, m, k);
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_tiled_gemm(&packed, &a, &mut out, m, k, n, 1.0);
        }
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_tiled_gemm_remainder() {
        let m = 5;
        let k = 7;
        let n = 3;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let expected = scalar_gemm(&w, &a, m, k, n, 0.75);
        let packed = pack_row_major(&w, m, k);
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_tiled_gemm(&packed, &a, &mut out, m, k, n, 0.75);
        }
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_tiled_gemm_vs_plain() {
        let m = 9;
        let k = 13;
        let n = 7;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1, -1][i % 5]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let packed = pack_row_major(&w, m, k);
        let mut out_plain = vec![0.0f32; m * n];
        let mut out_tiled = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_f32_gemm(&packed, &a, &mut out_plain, m, k, n, 1.0);
            neon_i2s_tiled_gemm(&packed, &a, &mut out_tiled, m, k, n, 1.0);
        }
        assert_close(&out_tiled, &out_plain, 1e-5);
    }

    // ── 4. Batched GEMM tests ─────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batched_gemm_single() {
        let m = 2;
        let k = 4;
        let n = 2;
        let w: Vec<i8> = vec![1, -1, 0, 1, 0, 1, -1, 0];
        let packed = pack_row_major(&w, m, k);
        let a = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_batched_gemm(&packed, &a, &mut out, m, k, n, 1, 1.0);
        }
        let expected = scalar_gemm(&w, &a, m, k, n, 1.0);
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batched_gemm_multi() {
        let m = 2;
        let k = 4;
        let n = 2;
        let batch = 3;
        let w: Vec<i8> = vec![1, -1, 0, 1, 0, 1, -1, 0];
        let packed = pack_row_major(&w, m, k);
        let mut a_batch = vec![0.0f32; batch * k * n];
        for b in 0..batch {
            for i in 0..(k * n) {
                a_batch[b * k * n + i] = (b as f32 + 1.0) * (i as f32 + 1.0) * 0.1;
            }
        }
        let mut out = vec![0.0f32; batch * m * n];
        unsafe {
            neon_i2s_batched_gemm(&packed, &a_batch, &mut out, m, k, n, batch, 1.0);
        }
        for b in 0..batch {
            let a_slice = &a_batch[b * k * n..(b + 1) * k * n];
            let expected = scalar_gemm(&w, a_slice, m, k, n, 1.0);
            let out_slice = &out[b * m * n..(b + 1) * m * n];
            assert_close(out_slice, &expected, 1e-4);
        }
    }

    // ── 5. Transposed GEMM tests ──────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_wt_identity() {
        let w_stored: [i8; 4] = [1, 0, 0, 1];
        let packed = pack_row_major(&w_stored, 2, 2);
        let a = [3.0f32, 5.0, 7.0, 11.0];
        let mut out = [0.0f32; 4];
        unsafe {
            neon_i2s_gemm_wt(&packed, &a, &mut out, 2, 2, 2, 1.0);
        }
        assert_close(&out, &[3.0, 5.0, 7.0, 11.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_wt_vs_explicit_transpose() {
        let m = 3;
        let k = 5;
        let n = 2;
        let w_stored: Vec<i8> = (0..k * m).map(|i| [1, -1, 0][i % 3]).collect();
        let packed_stored = pack_row_major(&w_stored, k, m);
        let mut w_eff = vec![0i8; m * k];
        for r in 0..k {
            for c in 0..m {
                w_eff[c * k + r] = w_stored[r * m + c];
            }
        }
        let a: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();
        let expected = scalar_gemm(&w_eff, &a, m, k, n, 1.0);
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_gemm_wt(&packed_stored, &a, &mut out, m, k, n, 1.0);
        }
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_at_vs_plain() {
        let m = 3;
        let k = 5;
        let n = 4;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let packed = pack_row_major(&w, m, k);
        let a: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();
        let mut a_t = vec![0.0f32; n * k];
        for r in 0..k {
            for c in 0..n {
                a_t[c * k + r] = a[r * n + c];
            }
        }
        let expected = scalar_gemm(&w, &a, m, k, n, 1.0);
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_gemm_at(&packed, &a_t, &mut out, m, k, n, 1.0);
        }
        assert_close(&out, &expected, 1e-4);
    }

    // ── 6. Fused GEMM + bias tests ────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_bias_basic() {
        let m = 2;
        let k = 4;
        let n = 3;
        let w: Vec<i8> = vec![1; m * k];
        let packed = pack_row_major(&w, m, k);
        let a = vec![1.0f32; k * n];
        let bias = [10.0f32, 20.0, 30.0];
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_gemm_bias(&packed, &a, &bias, &mut out, m, k, n, 1.0);
        }
        assert_close(&out, &[14.0, 24.0, 34.0, 14.0, 24.0, 34.0], 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_bias_zero_bias() {
        let m = 2;
        let k = 4;
        let n = 2;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1][i % 2]).collect();
        let packed = pack_row_major(&w, m, k);
        let a: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let bias = [0.0f32; 2];
        let mut out_bias = vec![0.0f32; m * n];
        let mut out_plain = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_gemm_bias(&packed, &a, &bias, &mut out_bias, m, k, n, 1.0);
            neon_i2s_f32_gemm(&packed, &a, &mut out_plain, m, k, n, 1.0);
        }
        assert_close(&out_bias, &out_plain, 1e-6);
    }

    // ── 7. Fused GEMM + ReLU tests ────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_relu_clamps_negative() {
        let m = 2;
        let k = 4;
        let n = 2;
        let w = vec![-1i8; m * k];
        let packed = pack_row_major(&w, m, k);
        let a = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_gemm_relu(&packed, &a, &mut out, m, k, n, 1.0);
        }
        assert_close(&out, &[0.0; 4], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_relu_passes_positive() {
        let m = 2;
        let k = 4;
        let n = 2;
        let w = vec![1i8; m * k];
        let packed = pack_row_major(&w, m, k);
        let a = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_gemm_relu(&packed, &a, &mut out, m, k, n, 1.0);
        }
        assert_close(&out, &[4.0; 4], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_relu_mixed() {
        let m = 2;
        let k = 2;
        let n = 1;
        let w: Vec<i8> = vec![1, 1, -1, -1];
        let packed = pack_row_major(&w, m, k);
        let a = vec![1.0f32, 1.0];
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_gemm_relu(&packed, &a, &mut out, m, k, n, 1.0);
        }
        assert_close(&out, &[2.0, 0.0], 1e-5);
    }

    // ── 8. QK256 GEMM tests ───────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_qk256_gemm_small() {
        let m = 2;
        let k = 8;
        let n = 2;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let packed = pack_row_major(&w, m, k);
        let scales = vec![2.0f32; m]; // 1 block per row
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_qk256_gemm(&packed, &scales, &a, &mut out, m, k, n);
        }
        let ref_out = scalar_gemm(&w, &a, m, k, n, 1.0);
        let expected: Vec<f32> = (0..m * n)
            .map(|i| {
                let row = i / n;
                ref_out[i] * scales[row]
            })
            .collect();
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_qk256_gemm_uniform_scale() {
        let m = 3;
        let k = 12;
        let n = 2;
        let w: Vec<i8> = vec![1; m * k];
        let packed = pack_row_major(&w, m, k);
        let scales = vec![1.0f32; m];
        let a = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_qk256_gemm(&packed, &scales, &a, &mut out, m, k, n);
        }
        assert_close(&out, &vec![12.0f32; m * n], 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_qk256_gemm_multi_block() {
        let m = 2;
        let k = 512;
        let n = 1;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1][i % 2]).collect();
        let packed = pack_row_major(&w, m, k);
        let scales = vec![1.0f32, 0.5, 1.0, 0.5];
        let a = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_qk256_gemm(&packed, &scales, &a, &mut out, m, k, n);
        }
        assert_close(&out, &[0.0, 0.0], 1e-4);
    }

    // ── Cross-feature tests ───────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemm_accumulates_into_output() {
        let w = [1i8; 4];
        let packed = pack_row_major(&w, 2, 2);
        let a = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe {
            neon_i2s_f32_gemm(&packed, &a, &mut out, 2, 2, 2, 1.0);
            neon_i2s_f32_gemm(&packed, &a, &mut out, 2, 2, 2, 1.0);
        }
        assert_close(&out, &[4.0; 4], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f32_gemm_non_aligned_k() {
        let m = 2;
        let k = 5;
        let n = 2;
        let w: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, -1, 0, 1];
        let a: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let expected = scalar_gemm(&w, &a, m, k, n, 1.0);
        let packed = pack_row_major(&w, m, k);
        let mut out = vec![0.0f32; m * n];
        unsafe {
            neon_i2s_f32_gemm(&packed, &a, &mut out, m, k, n, 1.0);
        }
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batched_gemm_matches_loop() {
        let m = 2;
        let k = 4;
        let n = 2;
        let batch = 4;
        let w = vec![1i8; m * k];
        let packed = pack_row_major(&w, m, k);
        let a_batch: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.01).collect();
        let mut out_batched = vec![0.0f32; batch * m * n];
        unsafe {
            neon_i2s_batched_gemm(&packed, &a_batch, &mut out_batched, m, k, n, batch, 1.0);
        }
        for b in 0..batch {
            let a = &a_batch[b * k * n..(b + 1) * k * n];
            let mut out_single = vec![0.0f32; m * n];
            unsafe {
                neon_i2s_f32_gemm(&packed, a, &mut out_single, m, k, n, 1.0);
            }
            let slice = &out_batched[b * m * n..(b + 1) * m * n];
            assert_close(slice, &out_single, 1e-6);
        }
    }

    #[test]
    fn test_decode_i2s_values() {
        assert_eq!(decode_i2s(0b00), 0.0);
        assert_eq!(decode_i2s(0b01), 1.0);
        assert_eq!(decode_i2s(0b10), 0.0);
        assert_eq!(decode_i2s(0b11), -1.0);
    }

    #[test]
    fn test_unpack_byte_f32() {
        let vals = unpack_byte_f32(0b11_00_01_01);
        assert_eq!(vals, [1.0, 1.0, 0.0, -1.0]);
        assert_eq!(unpack_byte_f32(0x00), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(unpack_byte_f32(0x55), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(unpack_byte_f32(0xFF), [-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_i2s_lut_consistency() {
        for bits in 0u8..4 {
            assert_eq!(
                I2S_LUT[bits as usize],
                decode_i2s(bits),
                "LUT and decode disagree for bits={bits}"
            );
        }
    }
}
