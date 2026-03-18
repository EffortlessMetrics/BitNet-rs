//! NEON-optimized quantized embedding lookup kernels for Apple Silicon.
//!
//! Provides six operations for quantized embedding tables used during
//! the first stage of inference:
//!
//! 1. I2_S (2-bit ternary) lookup with dequantization
//! 2. INT8 lookup with scale/zero-point dequantization
//! 3. Batched embedding lookups for token sequences
//! 4. Gather + sum for bag-of-words embeddings
//! 5. Positional embedding addition
//! 6. Embedding table row normalization
//!
//! All NEON intrinsics are gated behind `#[cfg(target_arch = "aarch64")]`
//! with scalar fallbacks for other architectures. Public functions are
//! safe; internal `unsafe` blocks cover only the NEON intrinsic calls.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane width for f32 vectors (128-bit / 32-bit).
const LANES: usize = 4;

// ═════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════

/// Decode a single 2-bit ternary value: 0b00→0, 0b01→1, 0b10→−1.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    match bits & 0b11 {
        0b00 => 0.0,
        0b01 => 1.0,
        0b10 => -1.0,
        _ => 0.0, // 0b11 unused, treat as zero
    }
}

/// Unpack four 2-bit ternary values from a byte (LSB-first) into f32,
/// multiplied by `scale`.
#[inline]
fn unpack_i2s_byte(byte: u8, scale: f32) -> [f32; 4] {
    [
        decode_i2s(byte) * scale,
        decode_i2s(byte >> 2) * scale,
        decode_i2s(byte >> 4) * scale,
        decode_i2s(byte >> 6) * scale,
    ]
}

// ═════════════════════════════════════════════════════════════════════
// 1. quant_embed_i2s_lookup
// ═════════════════════════════════════════════════════════════════════

/// Scalar fallback for I2_S embedding lookup.
fn scalar_i2s_lookup(table: &[u8], scale: f32, dim: usize, token_id: usize, output: &mut [f32]) {
    let bytes_per_row = dim / 4;
    let row_start = token_id * bytes_per_row;
    let row = &table[row_start..row_start + bytes_per_row];
    row.iter().enumerate().for_each(|(bi, &byte)| {
        let vals = unpack_i2s_byte(byte, scale);
        let base = bi * 4;
        output[base..base + 4].iter_mut().zip(vals.iter()).for_each(|(dst, &v)| *dst = v);
    });
}

/// NEON-accelerated I2_S embedding lookup.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_i2s_lookup(
    table: &[u8],
    scale: f32,
    dim: usize,
    token_id: usize,
    output: &mut [f32],
) {
    let bytes_per_row = dim / 4;
    let row_start = token_id * bytes_per_row;
    let row = &table[row_start..row_start + bytes_per_row];
    let scale_v = vdupq_n_f32(scale);

    row.iter().enumerate().for_each(|(bi, &byte)| {
        let vals = unpack_i2s_byte(byte, 1.0);
        let base = bi * 4;
        unsafe {
            let raw = vld1q_f32(vals.as_ptr());
            let scaled = vmulq_f32(raw, scale_v);
            vst1q_f32(output.as_mut_ptr().add(base), scaled);
        }
    });
}

/// I2_S (2-bit ternary) embedding table lookup with dequantization.
///
/// Each row is packed as 4 ternary values per byte (LSB-first).
/// The table has shape `[vocab, dim/4]` in bytes, and each row is
/// unpacked and scaled by `scale` into `output` (length `dim`).
///
/// # Panics
///
/// - `dim` is not a multiple of 4
/// - `token_id >= vocab`
/// - `output.len() < dim`
pub fn quant_embed_i2s_lookup(
    table: &[u8],
    scale: f32,
    dim: usize,
    token_id: usize,
    output: &mut [f32],
) {
    assert!(dim > 0, "dim must be > 0");
    assert!(dim.is_multiple_of(4), "dim must be a multiple of 4, got {dim}");
    let bytes_per_row = dim / 4;
    assert!(
        table.len().is_multiple_of(bytes_per_row),
        "table length must be a multiple of bytes_per_row"
    );
    let vocab = table.len() / bytes_per_row;
    assert!(token_id < vocab, "token_id {token_id} out of bounds for vocab {vocab}");
    assert!(output.len() >= dim, "output too small: need {dim} but got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_i2s_lookup(table, scale, dim, token_id, output);
            }
            return;
        }
    }
    scalar_i2s_lookup(table, scale, dim, token_id, output);
}

// ═════════════════════════════════════════════════════════════════════
// 2. quant_embed_i8_lookup
// ═════════════════════════════════════════════════════════════════════

/// Scalar fallback for INT8 embedding lookup.
fn scalar_i8_lookup(
    table: &[i8],
    scale: f32,
    zero_point: i8,
    dim: usize,
    token_id: usize,
    output: &mut [f32],
) {
    let row_start = token_id * dim;
    let row = &table[row_start..row_start + dim];
    row.iter().zip(output.iter_mut()).for_each(|(&val, dst)| {
        *dst = (f32::from(val) - f32::from(zero_point)) * scale;
    });
}

/// NEON-accelerated INT8 embedding lookup.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_i8_lookup(
    table: &[i8],
    scale: f32,
    zero_point: i8,
    dim: usize,
    token_id: usize,
    output: &mut [f32],
) {
    let row_start = token_id * dim;
    let row = &table[row_start..row_start + dim];
    let scale_v = vdupq_n_f32(scale);
    let zp_v = vdupq_n_f32(f32::from(zero_point));
    let chunks = dim / LANES;
    let tail_start = chunks * LANES;

    (0..chunks).for_each(|c| {
        let off = c * LANES;
        let v0 = f32::from(row[off]);
        let v1 = f32::from(row[off + 1]);
        let v2 = f32::from(row[off + 2]);
        let v3 = f32::from(row[off + 3]);
        unsafe {
            let mut vals = vdupq_n_f32(v0);
            vals = vsetq_lane_f32::<1>(v1, vals);
            vals = vsetq_lane_f32::<2>(v2, vals);
            vals = vsetq_lane_f32::<3>(v3, vals);
            let shifted = vsubq_f32(vals, zp_v);
            let scaled = vmulq_f32(shifted, scale_v);
            vst1q_f32(output.as_mut_ptr().add(off), scaled);
        }
    });

    // Scalar tail for remaining elements.
    row[tail_start..].iter().zip(output[tail_start..].iter_mut()).for_each(|(&val, dst)| {
        *dst = (f32::from(val) - f32::from(zero_point)) * scale;
    });
}

/// INT8 embedding lookup with scale and zero-point dequantization.
///
/// The table has shape `[vocab, dim]` stored as `i8`. Each value is
/// dequantized as `(val - zero_point) * scale`.
///
/// # Panics
///
/// - `dim == 0`
/// - `token_id >= vocab`
/// - `output.len() < dim`
pub fn quant_embed_i8_lookup(
    table: &[i8],
    scale: f32,
    zero_point: i8,
    dim: usize,
    token_id: usize,
    output: &mut [f32],
) {
    assert!(dim > 0, "dim must be > 0");
    assert!(table.len().is_multiple_of(dim), "table length must be a multiple of dim");
    let vocab = table.len() / dim;
    assert!(token_id < vocab, "token_id {token_id} out of bounds for vocab {vocab}");
    assert!(output.len() >= dim, "output too small: need {dim} but got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_i8_lookup(table, scale, zero_point, dim, token_id, output);
            }
            return;
        }
    }
    scalar_i8_lookup(table, scale, zero_point, dim, token_id, output);
}

// ═════════════════════════════════════════════════════════════════════
// 3. quant_embed_batch_lookup
// ═════════════════════════════════════════════════════════════════════

/// Batched embedding lookup for a sequence of token IDs.
///
/// For each token in `token_ids`, looks up its f32 embedding from
/// `table` (shape `[vocab, dim]`) and writes into the corresponding
/// row of `output` (shape `[token_ids.len(), dim]`).
///
/// # Panics
///
/// - `dim == 0`
/// - Any `token_id >= vocab`
/// - `output.len() < token_ids.len() * dim`
pub fn quant_embed_batch_lookup(
    table: &[f32],
    dim: usize,
    token_ids: &[usize],
    output: &mut [f32],
) {
    assert!(dim > 0, "dim must be > 0");
    assert!(table.len().is_multiple_of(dim), "table length must be a multiple of dim");
    let vocab = table.len() / dim;
    for &tid in token_ids {
        assert!(tid < vocab, "token_id {tid} out of bounds for vocab {vocab}");
    }
    assert!(
        output.len() >= token_ids.len() * dim,
        "output too small: need {} but got {}",
        token_ids.len() * dim,
        output.len()
    );

    if token_ids.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_batch_lookup(table, dim, token_ids, output);
            }
            return;
        }
    }
    scalar_batch_lookup(table, dim, token_ids, output);
}

/// Scalar fallback for batched embedding lookup.
fn scalar_batch_lookup(table: &[f32], dim: usize, token_ids: &[usize], output: &mut [f32]) {
    token_ids.iter().enumerate().for_each(|(i, &tid)| {
        let src = &table[tid * dim..tid * dim + dim];
        let dst = &mut output[i * dim..i * dim + dim];
        dst.copy_from_slice(src);
    });
}

/// NEON-accelerated batched embedding lookup with prefetch.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_batch_lookup(table: &[f32], dim: usize, token_ids: &[usize], output: &mut [f32]) {
    let chunks = dim / LANES;
    let tail_start = chunks * LANES;

    token_ids.iter().enumerate().for_each(|(i, &tid)| {
        let src_off = tid * dim;
        let dst_off = i * dim;

        // Prefetch next row if available.
        if let Some(&next_tid) = token_ids.get(i + 1) {
            unsafe {
                core::arch::asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) table
                        .as_ptr()
                        .add(next_tid * dim),
                    options(nostack, preserves_flags)
                );
            }
        }

        let src = &table[src_off..src_off + dim];
        let dst = &mut output[dst_off..dst_off + dim];

        (0..chunks).for_each(|c| {
            let off = c * LANES;
            unsafe {
                let v = vld1q_f32(src.as_ptr().add(off));
                vst1q_f32(dst.as_mut_ptr().add(off), v);
            }
        });
        // Scalar tail.
        src[tail_start..].iter().zip(dst[tail_start..].iter_mut()).for_each(|(s, d)| *d = *s);
    });
}

// ═════════════════════════════════════════════════════════════════════
// 4. quant_embed_gather_sum
// ═════════════════════════════════════════════════════════════════════

/// Gather + sum for bag-of-words embedding.
///
/// Looks up each token in `token_ids` from `table` (shape
/// `[vocab, dim]`) and sums all embeddings element-wise into `output`
/// (length `dim`).
///
/// # Panics
///
/// - `dim == 0`
/// - Any `token_id >= vocab`
/// - `output.len() < dim`
pub fn quant_embed_gather_sum(table: &[f32], dim: usize, token_ids: &[usize], output: &mut [f32]) {
    assert!(dim > 0, "dim must be > 0");
    assert!(table.len().is_multiple_of(dim), "table length must be a multiple of dim");
    let vocab = table.len() / dim;
    for &tid in token_ids {
        assert!(tid < vocab, "token_id {tid} out of bounds for vocab {vocab}");
    }
    assert!(output.len() >= dim, "output too small: need {dim} but got {}", output.len());

    // Zero the output.
    output.iter_mut().take(dim).for_each(|v| *v = 0.0);

    if token_ids.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_gather_sum(table, dim, token_ids, output);
            }
            return;
        }
    }
    scalar_gather_sum(table, dim, token_ids, output);
}

/// Scalar fallback for gather-sum.
fn scalar_gather_sum(table: &[f32], dim: usize, token_ids: &[usize], output: &mut [f32]) {
    token_ids.iter().for_each(|&tid| {
        let row = &table[tid * dim..tid * dim + dim];
        output.iter_mut().zip(row.iter()).for_each(|(dst, &src)| *dst += src);
    });
}

/// NEON-accelerated gather-sum.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gather_sum(table: &[f32], dim: usize, token_ids: &[usize], output: &mut [f32]) {
    let chunks = dim / LANES;
    let tail_start = chunks * LANES;

    token_ids.iter().for_each(|&tid| {
        let row = &table[tid * dim..tid * dim + dim];

        (0..chunks).for_each(|c| {
            let off = c * LANES;
            unsafe {
                let src = vld1q_f32(row.as_ptr().add(off));
                let acc = vld1q_f32(output.as_ptr().add(off));
                vst1q_f32(output.as_mut_ptr().add(off), vaddq_f32(acc, src));
            }
        });
        // Scalar tail.
        row[tail_start..].iter().zip(output[tail_start..].iter_mut()).for_each(|(s, d)| *d += s);
    });
}

// ═════════════════════════════════════════════════════════════════════
// 5. quant_embed_positional_add
// ═════════════════════════════════════════════════════════════════════

/// Add positional embeddings to token embeddings in-place.
///
/// `embeddings` has shape `[seq_len, dim]`. `pos_table` has shape
/// `[max_seq_len, dim]`. For each position `i` in `0..seq_len`,
/// adds `pos_table[pos_offset + i]` to `embeddings[i]`.
///
/// # Panics
///
/// - `dim == 0`
/// - `embeddings.len()` is not a multiple of `dim`
/// - `pos_offset + seq_len > max_seq_len`
pub fn quant_embed_positional_add(
    embeddings: &mut [f32],
    pos_table: &[f32],
    dim: usize,
    pos_offset: usize,
) {
    assert!(dim > 0, "dim must be > 0");
    assert!(embeddings.len().is_multiple_of(dim), "embeddings length must be a multiple of dim");
    assert!(pos_table.len().is_multiple_of(dim), "pos_table length must be a multiple of dim");
    let seq_len = embeddings.len() / dim;
    let max_pos = pos_table.len() / dim;
    assert!(
        pos_offset + seq_len <= max_pos,
        "pos_offset ({pos_offset}) + seq_len ({seq_len}) exceeds \
         max positions ({max_pos})"
    );

    if seq_len == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_positional_add(embeddings, pos_table, dim, pos_offset);
            }
            return;
        }
    }
    scalar_positional_add(embeddings, pos_table, dim, pos_offset);
}

/// Scalar fallback for positional add.
fn scalar_positional_add(embeddings: &mut [f32], pos_table: &[f32], dim: usize, pos_offset: usize) {
    let seq_len = embeddings.len() / dim;
    (0..seq_len).for_each(|i| {
        let pos_row_start = (pos_offset + i) * dim;
        let emb_start = i * dim;
        embeddings[emb_start..emb_start + dim]
            .iter_mut()
            .zip(pos_table[pos_row_start..pos_row_start + dim].iter())
            .for_each(|(e, &p)| *e += p);
    });
}

/// NEON-accelerated positional embedding addition.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_positional_add(
    embeddings: &mut [f32],
    pos_table: &[f32],
    dim: usize,
    pos_offset: usize,
) {
    let seq_len = embeddings.len() / dim;
    let chunks = dim / LANES;
    let tail_start = chunks * LANES;

    (0..seq_len).for_each(|i| {
        let pos_start = (pos_offset + i) * dim;
        let emb_start = i * dim;

        (0..chunks).for_each(|c| {
            let off = c * LANES;
            unsafe {
                let e = vld1q_f32(embeddings.as_ptr().add(emb_start + off));
                let p = vld1q_f32(pos_table.as_ptr().add(pos_start + off));
                vst1q_f32(embeddings.as_mut_ptr().add(emb_start + off), vaddq_f32(e, p));
            }
        });
        // Scalar tail.
        embeddings[emb_start + tail_start..emb_start + dim]
            .iter_mut()
            .zip(pos_table[pos_start + tail_start..pos_start + dim].iter())
            .for_each(|(e, &p)| *e += p);
    });
}

// ═════════════════════════════════════════════════════════════════════
// 6. quant_embed_table_norm
// ═════════════════════════════════════════════════════════════════════

/// L2-normalize each row of an embedding table in-place.
///
/// `table` has shape `[num_rows, dim]`. Each row is divided by its
/// L2 norm. Zero-norm rows are left untouched.
///
/// # Panics
///
/// - `dim == 0`
/// - `table.len()` is not a multiple of `dim`
pub fn quant_embed_table_norm(table: &mut [f32], dim: usize) {
    assert!(dim > 0, "dim must be > 0");
    assert!(table.len().is_multiple_of(dim), "table length must be a multiple of dim");
    let num_rows = table.len() / dim;

    if num_rows == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_table_norm(table, dim, num_rows);
            }
            return;
        }
    }
    scalar_table_norm(table, dim, num_rows);
}

/// Scalar fallback for table norm.
fn scalar_table_norm(table: &mut [f32], dim: usize, num_rows: usize) {
    (0..num_rows).for_each(|r| {
        let start = r * dim;
        let row = &table[start..start + dim];
        let norm_sq: f32 = row.iter().map(|x| x * x).sum();
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            table[start..start + dim].iter_mut().for_each(|v| *v *= inv_norm);
        }
    });
}

/// NEON-accelerated table row L2 normalization.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_table_norm(table: &mut [f32], dim: usize, num_rows: usize) {
    let chunks = dim / LANES;
    let tail_start = chunks * LANES;

    (0..num_rows).for_each(|r| {
        let start = r * dim;

        // Compute L2 norm squared.
        let mut acc = vdupq_n_f32(0.0);
        (0..chunks).for_each(|c| {
            let off = start + c * LANES;
            unsafe {
                let v = vld1q_f32(table.as_ptr().add(off));
                acc = vfmaq_f32(acc, v, v);
            }
        });
        let mut norm_sq: f32 = vaddvq_f32(acc);
        // Add scalar tail.
        table[start + tail_start..start + dim].iter().for_each(|&v| norm_sq += v * v);

        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            let inv_v = vdupq_n_f32(inv_norm);
            (0..chunks).for_each(|c| {
                let off = start + c * LANES;
                unsafe {
                    let v = vld1q_f32(table.as_ptr().add(off));
                    vst1q_f32(table.as_mut_ptr().add(off), vmulq_f32(v, inv_v));
                }
            });
            table[start + tail_start..start + dim].iter_mut().for_each(|v| *v *= inv_norm);
        }
    });
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn vec_approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y))
    }

    /// Pack four ternary values (each in {-1,0,1}) into one byte.
    fn pack_i2s(vals: [i8; 4]) -> u8 {
        let encode = |v: i8| -> u8 {
            match v {
                0 => 0b00,
                1 => 0b01,
                -1 => 0b10,
                _ => panic!("invalid ternary value {v}"),
            }
        };
        encode(vals[0]) | (encode(vals[1]) << 2) | (encode(vals[2]) << 4) | (encode(vals[3]) << 6)
    }

    // ─── I2S lookup tests ───────────────────────────────────────

    #[test]
    fn i2s_basic_all_zeros() {
        let table = vec![pack_i2s([0, 0, 0, 0])];
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 1.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn i2s_basic_all_ones() {
        let table = vec![pack_i2s([1, 1, 1, 1])];
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 1.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn i2s_basic_all_neg_ones() {
        let table = vec![pack_i2s([-1, -1, -1, -1])];
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 1.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[-1.0, -1.0, -1.0, -1.0]));
    }

    #[test]
    fn i2s_mixed_values() {
        let table = vec![pack_i2s([1, 0, -1, 1])];
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 1.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[1.0, 0.0, -1.0, 1.0]));
    }

    #[test]
    fn i2s_with_scale() {
        let table = vec![pack_i2s([1, -1, 0, 1])];
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 0.5, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.5, -0.5, 0.0, 0.5]));
    }

    #[test]
    fn i2s_zero_scale() {
        let table = vec![pack_i2s([1, -1, 1, -1])];
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 0.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn i2s_multiple_rows() {
        let table = vec![pack_i2s([1, 0, 0, 0]), pack_i2s([0, 1, 0, 0]), pack_i2s([0, 0, -1, 0])];
        let mut out = [0.0f32; 4];

        quant_embed_i2s_lookup(&table, 1.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[1.0, 0.0, 0.0, 0.0]));

        quant_embed_i2s_lookup(&table, 1.0, 4, 1, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 1.0, 0.0, 0.0]));

        quant_embed_i2s_lookup(&table, 1.0, 4, 2, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 0.0, -1.0, 0.0]));
    }

    #[test]
    fn i2s_large_dim() {
        // dim = 16 → 4 bytes per row
        let row: Vec<u8> = (0..4).map(|_| pack_i2s([1, -1, 1, -1])).collect();
        let mut out = [0.0f32; 16];
        quant_embed_i2s_lookup(&row, 2.0, 16, 0, &mut out);
        let expected: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 2.0 } else { -2.0 }).collect();
        assert!(vec_approx_eq(&out, &expected));
    }

    #[test]
    fn i2s_negative_scale() {
        let table = vec![pack_i2s([1, -1, 0, 1])];
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, -1.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[-1.0, 1.0, 0.0, -1.0]));
    }

    #[test]
    #[should_panic(expected = "dim must be a multiple of 4")]
    fn i2s_dim_not_multiple_of_4() {
        let table = [0u8; 3];
        let mut out = [0.0f32; 5];
        quant_embed_i2s_lookup(&table, 1.0, 5, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn i2s_dim_zero() {
        let table = vec![0u8];
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 1.0, 0, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn i2s_token_oob() {
        let table = [0u8; 2]; // vocab=2, dim=4
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 1.0, 4, 5, &mut out);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn i2s_output_too_small() {
        let table = [0u8; 1];
        let mut out = [0.0f32; 2];
        quant_embed_i2s_lookup(&table, 1.0, 4, 0, &mut out);
    }

    #[test]
    fn i2s_dim_8_two_bytes() {
        let table = vec![pack_i2s([1, 0, -1, 0]), pack_i2s([-1, 1, 0, 1])];
        let mut out = [0.0f32; 8];
        quant_embed_i2s_lookup(&table, 1.0, 8, 0, &mut out);
        assert!(vec_approx_eq(&out, &[1.0, 0.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0]));
    }

    #[test]
    fn i2s_large_scale() {
        let table = vec![pack_i2s([1, -1, 1, -1])];
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 1000.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[1000.0, -1000.0, 1000.0, -1000.0]));
    }

    // ─── I8 lookup tests ────────────────────────────────────────

    #[test]
    fn i8_basic_identity() {
        let table: Vec<i8> = vec![0, 1, 2, 3];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 1.0, 0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 1.0, 2.0, 3.0]));
    }

    #[test]
    fn i8_with_scale() {
        let table: Vec<i8> = vec![0, 2, 4, 6];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 0.5, 0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 1.0, 2.0, 3.0]));
    }

    #[test]
    fn i8_with_zero_point() {
        let table: Vec<i8> = vec![10, 11, 12, 13];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 1.0, 10, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 1.0, 2.0, 3.0]));
    }

    #[test]
    fn i8_scale_and_zero_point() {
        let table: Vec<i8> = vec![100, 102, 104, 106];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 0.5, 100, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 1.0, 2.0, 3.0]));
    }

    #[test]
    fn i8_negative_values() {
        let table: Vec<i8> = vec![-4, -3, -2, -1];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 1.0, 0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[-4.0, -3.0, -2.0, -1.0]));
    }

    #[test]
    fn i8_multiple_rows() {
        let table: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut out = [0.0f32; 4];

        quant_embed_i8_lookup(&table, 1.0, 0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[1.0, 2.0, 3.0, 4.0]));

        quant_embed_i8_lookup(&table, 1.0, 0, 4, 1, &mut out);
        assert!(vec_approx_eq(&out, &[5.0, 6.0, 7.0, 8.0]));
    }

    #[test]
    fn i8_dim_not_multiple_of_4() {
        let table: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let mut out = [0.0f32; 6];
        quant_embed_i8_lookup(&table, 1.0, 0, 6, 0, &mut out);
        assert!(vec_approx_eq(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    #[test]
    fn i8_zero_scale() {
        let table: Vec<i8> = vec![10, 20, 30, 40];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 0.0, 0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn i8_dim_1() {
        let table: Vec<i8> = [42];
        let mut out = [0.0f32; 1];
        quant_embed_i8_lookup(&table, 1.0, 0, 1, 0, &mut out);
        assert!(vec_approx_eq(&out, &[42.0]));
    }

    #[test]
    fn i8_large_dim() {
        let dim = 32;
        let table: Vec<i8> = (0..dim).map(|i| i as i8).collect();
        let mut out = vec![0.0f32; dim];
        quant_embed_i8_lookup(&table, 1.0, 0, dim, 0, &mut out);
        let expected: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        assert!(vec_approx_eq(&out, &expected));
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn i8_dim_zero() {
        let table: Vec<i8> = vec![];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 1.0, 0, 0, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn i8_token_oob() {
        let table: Vec<i8> = vec![1, 2, 3, 4];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 1.0, 0, 4, 5, &mut out);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn i8_output_too_small() {
        let table: Vec<i8> = vec![1, 2, 3, 4];
        let mut out = [0.0f32; 2];
        quant_embed_i8_lookup(&table, 1.0, 0, 4, 0, &mut out);
    }

    #[test]
    fn i8_negative_zero_point() {
        let table: Vec<i8> = vec![0, 1, 2, 3];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 1.0, -10, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[10.0, 11.0, 12.0, 13.0]));
    }

    // ─── Batch lookup tests ─────────────────────────────────────

    #[test]
    fn batch_single_token() {
        let table: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut out = [0.0f32; 4];
        quant_embed_batch_lookup(&table, 4, &[1], &mut out);
        assert!(vec_approx_eq(&out, &[4.0, 5.0, 6.0, 7.0]));
    }

    #[test]
    fn batch_multiple_tokens() {
        let table: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut out = [0.0f32; 8];
        quant_embed_batch_lookup(&table, 4, &[0, 2], &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]));
    }

    #[test]
    fn batch_empty_tokens() {
        let table: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = [0.0f32; 0];
        quant_embed_batch_lookup(&table, 4, &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn batch_duplicate_tokens() {
        let table: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = [0.0f32; 8];
        quant_embed_batch_lookup(&table, 4, &[1, 1], &mut out);
        assert!(vec_approx_eq(&out, &[4.0, 5.0, 6.0, 7.0, 4.0, 5.0, 6.0, 7.0]));
    }

    #[test]
    fn batch_all_tokens() {
        let table: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = [0.0f32; 8];
        quant_embed_batch_lookup(&table, 4, &[0, 1], &mut out);
        assert!(vec_approx_eq(&out, &table));
    }

    #[test]
    fn batch_dim_not_aligned() {
        let dim = 5;
        let table: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let mut out = [0.0f32; 5];
        quant_embed_batch_lookup(&table, dim, &[2], &mut out);
        assert!(vec_approx_eq(&out, &[10.0, 11.0, 12.0, 13.0, 14.0]));
    }

    #[test]
    fn batch_large_sequence() {
        let dim = 8;
        let vocab = 16;
        let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();
        let ids: Vec<usize> = (0..vocab).collect();
        let mut out = vec![0.0f32; vocab * dim];
        quant_embed_batch_lookup(&table, dim, &ids, &mut out);
        assert!(vec_approx_eq(&out, &table));
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn batch_dim_zero() {
        let table: Vec<f32> = [1.0];
        let mut out = [0.0f32; 1];
        quant_embed_batch_lookup(&table, 0, &[0], &mut out);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn batch_token_oob() {
        let table: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        quant_embed_batch_lookup(&table, 4, &[5], &mut out);
    }

    #[test]
    fn batch_reversed_order() {
        let table: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut out = [0.0f32; 8];
        quant_embed_batch_lookup(&table, 4, &[2, 0], &mut out);
        assert!(vec_approx_eq(&out, &[8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0]));
    }

    // ─── Gather sum tests ───────────────────────────────────────

    #[test]
    fn gather_sum_single() {
        let table: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = [0.0f32; 4];
        quant_embed_gather_sum(&table, 4, &[0], &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 1.0, 2.0, 3.0]));
    }

    #[test]
    fn gather_sum_two_rows() {
        let table: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = [0.0f32; 4];
        quant_embed_gather_sum(&table, 4, &[0, 1], &mut out);
        // [0+4, 1+5, 2+6, 3+7] = [4, 6, 8, 10]
        assert!(vec_approx_eq(&out, &[4.0, 6.0, 8.0, 10.0]));
    }

    #[test]
    fn gather_sum_duplicate_indices() {
        let table = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        quant_embed_gather_sum(&table, 4, &[0, 0, 0], &mut out);
        assert!(vec_approx_eq(&out, &[3.0, 6.0, 9.0, 12.0]));
    }

    #[test]
    fn gather_sum_empty() {
        let table = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        quant_embed_gather_sum(&table, 4, &[], &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn gather_sum_non_aligned_dim() {
        let dim = 5;
        let table: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; dim];
        quant_embed_gather_sum(&table, dim, &[0, 1], &mut out);
        // [0+5, 1+6, 2+7, 3+8, 4+9] = [5, 7, 9, 11, 13]
        assert!(vec_approx_eq(&out, &[5.0, 7.0, 9.0, 11.0, 13.0]));
    }

    #[test]
    fn gather_sum_all_rows() {
        let dim = 4;
        let vocab = 3;
        let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
        let ids: Vec<usize> = (0..vocab).collect();
        let mut out = vec![0.0f32; dim];
        quant_embed_gather_sum(&table, dim, &ids, &mut out);
        // col sums: [0+0.4+0.8, 0.1+0.5+0.9, 0.2+0.6+1.0, 0.3+0.7+1.1]
        assert!(vec_approx_eq(&out, &[1.2, 1.5, 1.8, 2.1]));
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn gather_sum_dim_zero() {
        let table: Vec<f32> = [1.0];
        let mut out = [0.0f32; 1];
        quant_embed_gather_sum(&table, 0, &[0], &mut out);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn gather_sum_token_oob() {
        let table = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        quant_embed_gather_sum(&table, 4, &[5], &mut out);
    }

    #[test]
    fn gather_sum_large() {
        let dim = 32;
        let vocab = 8;
        let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.001).collect();
        let mut out = vec![0.0f32; dim];
        quant_embed_gather_sum(&table, dim, &[0, 3, 7], &mut out);
        let mut expected = vec![0.0f32; dim];
        [0, 3, 7].iter().for_each(|&r| {
            table[r * dim..r * dim + dim]
                .iter()
                .zip(expected.iter_mut())
                .for_each(|(&s, d)| *d += s);
        });
        assert!(vec_approx_eq(&out, &expected));
    }

    // ─── Positional add tests ───────────────────────────────────

    #[test]
    fn pos_add_basic() {
        let mut emb = vec![1.0f32, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        quant_embed_positional_add(&mut emb, &pos, 4, 0);
        assert!(vec_approx_eq(&emb, &[1.1, 2.2, 3.3, 4.4]));
    }

    #[test]
    fn pos_add_with_offset() {
        let mut emb = vec![1.0f32, 2.0, 3.0, 4.0];
        let pos = vec![
            0.0, 0.0, 0.0, 0.0, // position 0
            0.5, 0.5, 0.5, 0.5, // position 1
        ];
        quant_embed_positional_add(&mut emb, &pos, 4, 1);
        assert!(vec_approx_eq(&emb, &[1.5, 2.5, 3.5, 4.5]));
    }

    #[test]
    fn pos_add_multi_positions() {
        let mut emb = vec![
            1.0, 2.0, 3.0, 4.0, // pos 0
            5.0, 6.0, 7.0, 8.0, // pos 1
        ];
        let pos = vec![
            0.1, 0.1, 0.1, 0.1, // pos 0
            0.2, 0.2, 0.2, 0.2, // pos 1
        ];
        quant_embed_positional_add(&mut emb, &pos, 4, 0);
        assert!(vec_approx_eq(&emb, &[1.1, 2.1, 3.1, 4.1, 5.2, 6.2, 7.2, 8.2]));
    }

    #[test]
    fn pos_add_zero_pos_table() {
        let mut emb = vec![1.0f32, 2.0, 3.0, 4.0];
        let pos = vec![0.0, 0.0, 0.0, 0.0];
        quant_embed_positional_add(&mut emb, &pos, 4, 0);
        assert!(vec_approx_eq(&emb, &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn pos_add_non_aligned_dim() {
        let mut emb = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let pos = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        quant_embed_positional_add(&mut emb, &pos, 5, 0);
        assert!(vec_approx_eq(&emb, &[1.1, 2.2, 3.3, 4.4, 5.5]));
    }

    #[test]
    fn pos_add_empty() {
        let mut emb: Vec<f32> = vec![];
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        quant_embed_positional_add(&mut emb, &pos, 4, 0);
        assert!(emb.is_empty());
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn pos_add_dim_zero() {
        let mut emb = vec![1.0f32];
        let pos = vec![0.1f32];
        quant_embed_positional_add(&mut emb, &pos, 0, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn pos_add_offset_oob() {
        let mut emb = vec![1.0f32, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        quant_embed_positional_add(&mut emb, &pos, 4, 1);
    }

    #[test]
    fn pos_add_negative_positions() {
        let mut emb = vec![1.0f32, 2.0, 3.0, 4.0];
        let pos = vec![-0.5, -0.5, -0.5, -0.5];
        quant_embed_positional_add(&mut emb, &pos, 4, 0);
        assert!(vec_approx_eq(&emb, &[0.5, 1.5, 2.5, 3.5]));
    }

    #[test]
    fn pos_add_large_sequence() {
        let dim = 8;
        let seq_len = 16;
        let mut emb: Vec<f32> = (0..seq_len * dim).map(|i| i as f32).collect();
        let pos: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.01).collect();
        let expected: Vec<f32> = emb.iter().zip(pos.iter()).map(|(&e, &p)| e + p).collect();
        quant_embed_positional_add(&mut emb, &pos, dim, 0);
        assert!(vec_approx_eq(&emb, &expected));
    }

    // ─── Table norm tests ───────────────────────────────────────

    #[test]
    fn norm_unit_vector() {
        let mut table = vec![1.0f32, 0.0, 0.0, 0.0];
        quant_embed_table_norm(&mut table, 4);
        assert!(vec_approx_eq(&table, &[1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn norm_uniform_vector() {
        let mut table = vec![2.0f32, 2.0, 2.0, 2.0];
        quant_embed_table_norm(&mut table, 4);
        let expected_val = 1.0 / 2.0f32; // 2/sqrt(16)=0.5
        assert!(vec_approx_eq(&table, &[expected_val; 4]));
    }

    #[test]
    fn norm_zero_row() {
        let mut table = vec![0.0f32, 0.0, 0.0, 0.0];
        quant_embed_table_norm(&mut table, 4);
        assert!(vec_approx_eq(&table, &[0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn norm_multiple_rows() {
        let mut table = vec![
            3.0, 4.0, 0.0, // row 0: norm = 5
            0.0, 0.0, 5.0, // row 1: norm = 5
        ];
        quant_embed_table_norm(&mut table, 3);
        assert!(vec_approx_eq(&table, &[0.6, 0.8, 0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn norm_single_element() {
        let mut table = vec![5.0f32];
        quant_embed_table_norm(&mut table, 1);
        assert!(vec_approx_eq(&table, &[1.0]));
    }

    #[test]
    fn norm_negative_values() {
        let mut table = vec![-3.0f32, -4.0, 0.0];
        quant_embed_table_norm(&mut table, 3);
        assert!(vec_approx_eq(&table, &[-0.6, -0.8, 0.0]));
    }

    #[test]
    fn norm_already_normalized() {
        let inv_sqrt2: f32 = 1.0 / 2.0f32.sqrt();
        let mut table = vec![inv_sqrt2, inv_sqrt2];
        quant_embed_table_norm(&mut table, 2);
        assert!(vec_approx_eq(&table, &[inv_sqrt2, inv_sqrt2]));
    }

    #[test]
    fn norm_empty_table() {
        let mut table: Vec<f32> = vec![];
        quant_embed_table_norm(&mut table, 4);
        assert!(table.is_empty());
    }

    #[test]
    fn norm_mixed_zero_nonzero_rows() {
        let mut table = vec![
            0.0, 0.0, 0.0, 0.0, // row 0: zero
            1.0, 0.0, 0.0, 0.0, // row 1: unit-x
        ];
        quant_embed_table_norm(&mut table, 4);
        assert!(vec_approx_eq(&table, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn norm_dim_zero() {
        let mut table = vec![1.0f32];
        quant_embed_table_norm(&mut table, 0);
    }

    #[test]
    #[should_panic(expected = "multiple of dim")]
    fn norm_bad_table_len() {
        let mut table = vec![1.0f32, 2.0, 3.0];
        quant_embed_table_norm(&mut table, 4);
    }

    #[test]
    fn norm_large_values() {
        let mut table = vec![1e6f32, 0.0, 0.0, 0.0];
        quant_embed_table_norm(&mut table, 4);
        assert!(vec_approx_eq(&table, &[1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn norm_small_values() {
        let mut table = vec![1e-6f32, 0.0, 0.0, 0.0];
        quant_embed_table_norm(&mut table, 4);
        assert!(vec_approx_eq(&table, &[1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn norm_dim_not_aligned() {
        let mut table = vec![3.0f32, 4.0, 0.0, 0.0, 5.0];
        quant_embed_table_norm(&mut table, 5);
        let norm = (9.0 + 16.0 + 0.0 + 0.0 + 25.0f32).sqrt();
        let expected: Vec<f32> = [3.0, 4.0, 0.0, 0.0, 5.0].iter().map(|&v| v / norm).collect();
        assert!(vec_approx_eq(&table, &expected));
    }

    // ─── Cross-function tests ───────────────────────────────────

    #[test]
    fn i2s_then_positional_add() {
        let table = vec![pack_i2s([1, -1, 1, -1])];
        let mut emb = [0.0f32; 4];
        quant_embed_i2s_lookup(&table, 1.0, 4, 0, &mut emb);
        let pos = vec![0.5, 0.5, 0.5, 0.5];
        quant_embed_positional_add(&mut emb, &pos, 4, 0);
        assert!(vec_approx_eq(&emb, &[1.5, -0.5, 1.5, -0.5]));
    }

    #[test]
    fn i8_then_norm() {
        let table_i8: Vec<i8> = vec![3, 4, 0, 0];
        let mut emb = [0.0f32; 4];
        quant_embed_i8_lookup(&table_i8, 1.0, 0, 4, 0, &mut emb);
        quant_embed_table_norm(&mut emb, 4);
        assert!(vec_approx_eq(&emb, &[0.6, 0.8, 0.0, 0.0]));
    }

    #[test]
    fn batch_then_gather_sum() {
        let table: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut batch_out = [0.0f32; 8];
        quant_embed_batch_lookup(&table, 4, &[0, 2], &mut batch_out);
        // batch_out = [0,1,2,3, 8,9,10,11]
        // Now gather-sum both rows of batch_out
        let mut sum_out = [0.0f32; 4];
        quant_embed_gather_sum(&batch_out, 4, &[0, 1], &mut sum_out);
        assert!(vec_approx_eq(&sum_out, &[8.0, 10.0, 12.0, 14.0]));
    }

    #[test]
    fn gather_sum_then_norm() {
        let table = vec![
            3.0f32, 0.0, 0.0, 0.0, // row 0
            0.0, 4.0, 0.0, 0.0, // row 1
        ];
        let mut sum_out = [0.0f32; 4];
        quant_embed_gather_sum(&table, 4, &[0, 1], &mut sum_out);
        // sum = [3.0, 4.0, 0.0, 0.0]
        quant_embed_table_norm(&mut sum_out, 4);
        assert!(vec_approx_eq(&sum_out, &[0.6, 0.8, 0.0, 0.0]));
    }

    #[test]
    fn batch_pos_add_norm_pipeline() {
        let table = vec![
            2.0f32, 0.0, 0.0, 0.0, // row 0
            0.0, 2.0, 0.0, 0.0, // row 1
        ];
        let mut out = [0.0f32; 8];
        quant_embed_batch_lookup(&table, 4, &[0, 1], &mut out);
        let pos = vec![
            1.0, 0.0, 0.0, 0.0, // pos 0
            0.0, 1.0, 0.0, 0.0, // pos 1
        ];
        quant_embed_positional_add(&mut out, &pos, 4, 0);
        // out = [3,0,0,0, 0,3,0,0]
        quant_embed_table_norm(&mut out, 4);
        assert!(vec_approx_eq(&out, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
    }

    // ─── Edge case & property tests ─────────────────────────────

    #[test]
    fn i2s_unused_bits_treated_as_zero() {
        // 0b11 encoding should decode as 0
        let byte = 0b11_11_11_11u8; // all four values = 0b11
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&[byte], 1.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn i2s_all_encoding_combinations() {
        // Test all 4 possible 2-bit encodings in one byte
        // 0b10_01_00_11 = val3=-1, val2=1, val1=0, val0=0b11→0
        let byte = 0b10_01_00_11u8;
        let mut out = [0.0f32; 4];
        quant_embed_i2s_lookup(&[byte], 1.0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[0.0, 0.0, 1.0, -1.0]));
    }

    #[test]
    fn i8_extreme_values() {
        let table: Vec<i8> = vec![127, -128, 0, 1];
        let mut out = [0.0f32; 4];
        quant_embed_i8_lookup(&table, 1.0, 0, 4, 0, &mut out);
        assert!(vec_approx_eq(&out, &[127.0, -128.0, 0.0, 1.0]));
    }

    #[test]
    fn batch_single_dim() {
        let table = vec![10.0f32, 20.0, 30.0];
        let mut out = [0.0f32; 2];
        quant_embed_batch_lookup(&table, 1, &[1, 2], &mut out);
        assert!(vec_approx_eq(&out, &[20.0, 30.0]));
    }

    #[test]
    fn gather_sum_single_index() {
        let table = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        quant_embed_gather_sum(&table, 4, &[0], &mut out);
        assert!(vec_approx_eq(&out, &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn pos_add_dim_1() {
        let mut emb = vec![5.0f32];
        let pos = vec![3.0f32];
        quant_embed_positional_add(&mut emb, &pos, 1, 0);
        assert!(vec_approx_eq(&emb, &[8.0]));
    }

    #[test]
    fn norm_preserves_direction() {
        let mut table = vec![3.0f32, 4.0];
        quant_embed_table_norm(&mut table, 2);
        // Check direction is preserved (ratio 3:4)
        let ratio = table[0] / table[1];
        assert!(approx_eq(ratio, 0.75));
    }

    #[test]
    fn norm_idempotent() {
        let mut table = vec![3.0f32, 4.0, 0.0, 0.0];
        quant_embed_table_norm(&mut table, 4);
        let first = table.clone();
        quant_embed_table_norm(&mut table, 4);
        assert!(vec_approx_eq(&table, &first));
    }

    #[test]
    fn norm_result_has_unit_length() {
        let mut table = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        quant_embed_table_norm(&mut table, 5);
        let norm: f32 = table.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(norm, 1.0));
    }

    #[test]
    fn i2s_dim_32_multi_byte() {
        let dim = 32;
        let bytes_per_row = dim / 4;
        let table: Vec<u8> = (0..bytes_per_row).map(|_| pack_i2s([1, 0, -1, 0])).collect();
        let mut out = vec![0.0f32; dim];
        quant_embed_i2s_lookup(&table, 0.25, dim, 0, &mut out);
        let expected: Vec<f32> = (0..dim)
            .map(|i| match i % 4 {
                0 => 0.25,
                2 => -0.25,
                _ => 0.0,
            })
            .collect();
        assert!(vec_approx_eq(&out, &expected));
    }

    #[test]
    fn i8_dim_17_tail() {
        let dim = 17;
        let table: Vec<i8> = (0..dim).map(|i| (i % 10) as i8).collect();
        let mut out = vec![0.0f32; dim];
        quant_embed_i8_lookup(&table, 1.0, 0, dim, 0, &mut out);
        let expected: Vec<f32> = (0..dim).map(|i| (i % 10) as f32).collect();
        assert!(vec_approx_eq(&out, &expected));
    }

    #[test]
    fn pos_add_offset_middle() {
        let dim = 4;
        let mut emb = vec![1.0f32, 1.0, 1.0, 1.0];
        let pos = vec![
            0.0, 0.0, 0.0, 0.0, // pos 0
            0.0, 0.0, 0.0, 0.0, // pos 1
            0.5, 0.5, 0.5, 0.5, // pos 2
            1.0, 1.0, 1.0, 1.0, // pos 3
        ];
        quant_embed_positional_add(&mut emb, &pos, dim, 2);
        assert!(vec_approx_eq(&emb, &[1.5, 1.5, 1.5, 1.5]));
    }

    #[test]
    fn norm_many_rows() {
        let dim = 4;
        let num_rows = 10;
        let mut table: Vec<f32> = (0..num_rows * dim).map(|i| (i as f32) + 1.0).collect();
        quant_embed_table_norm(&mut table, dim);
        (0..num_rows).for_each(|r| {
            let row = &table[r * dim..r * dim + dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(approx_eq(norm, 1.0), "row {r} norm = {norm}");
        });
    }
}
