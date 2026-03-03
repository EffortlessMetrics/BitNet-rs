//! NEON-optimized embedding table operations for Apple Silicon (aarch64).
//!
//! Provides six embedding-related kernels used during the first stage of inference:
//! f32 lookup, f16→f32 lookup, multi-index sum, L2 normalization, cosine
//! similarity, and RoPE positional encoding.  Each operation has a NEON
//! fast-path (with `__prefetch`) and a portable scalar fallback selected at
//! runtime via `is_aarch64_feature_detected!("neon")`.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of,
    clippy::manual_memcpy,
    dead_code,
    unused_unsafe
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use half::f16;

/// NEON lane width for f32 vectors.
const LANES: usize = 4;

/// Software prefetch hint for table locality.
///
/// Uses inline assembly on aarch64 (`prfm pldl1keep`), no-op elsewhere.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_read(ptr: *const u8) {
    core::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
}

// =====================================================================
// 1. embedding_lookup_f32
// =====================================================================

/// NEON-optimized f32 embedding table lookup with prefetch.
///
/// # Safety
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_embedding_lookup_f32(
    table: &[f32],
    indices: &[u32],
    dim: usize,
    output: &mut [f32],
) {
    let chunks = dim / LANES;
    let tail = dim % LANES;

    for (i, &idx) in indices.iter().enumerate() {
        let src_off = (idx as usize) * dim;
        let dst_off = i * dim;

        // Prefetch next row if available.
        if i + 1 < indices.len() {
            let next_off = (indices[i + 1] as usize) * dim;
            unsafe {
                prefetch_read(table.as_ptr().add(next_off) as *const u8);
            }
        }

        let src = &table[src_off..src_off + dim];
        let dst = &mut output[dst_off..dst_off + dim];

        for c in 0..chunks {
            let off = c * LANES;
            unsafe {
                let v = vld1q_f32(src.as_ptr().add(off));
                vst1q_f32(dst.as_mut_ptr().add(off), v);
            }
        }
        // Scalar tail.
        for j in (chunks * LANES)..dim {
            dst[j] = src[j];
        }
    }
}

/// Scalar fallback for f32 embedding lookup.
fn scalar_embedding_lookup_f32(table: &[f32], indices: &[u32], dim: usize, output: &mut [f32]) {
    for (i, &idx) in indices.iter().enumerate() {
        let src_off = (idx as usize) * dim;
        let dst_off = i * dim;
        output[dst_off..dst_off + dim].copy_from_slice(&table[src_off..src_off + dim]);
    }
}

/// Embedding table lookup from an f32 table.
///
/// Copies rows indexed by `indices` from `table` (shape `[vocab, dim]`) into
/// `output` (shape `[indices.len(), dim]`).
///
/// # Panics
/// Panics if `output` is too small or an index is out of bounds.
pub fn embedding_lookup_f32(table: &[f32], indices: &[u32], dim: usize, output: &mut [f32]) {
    assert!(dim > 0, "dim must be > 0");
    let vocab = table.len() / dim;
    assert_eq!(table.len(), vocab * dim, "table length must be a multiple of dim");
    for &idx in indices {
        assert!((idx as usize) < vocab, "index {idx} out of bounds for vocab {vocab}");
    }
    assert!(
        output.len() >= indices.len() * dim,
        "output too small: need {} but got {}",
        indices.len() * dim,
        output.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_embedding_lookup_f32(table, indices, dim, output);
            }
            return;
        }
    }
    scalar_embedding_lookup_f32(table, indices, dim, output);
}

// =====================================================================
// 2. embedding_lookup_f16
// =====================================================================

/// NEON-optimized f16→f32 embedding lookup with vcvt.
///
/// # Safety
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_embedding_lookup_f16(
    table: &[f16],
    indices: &[u32],
    dim: usize,
    output: &mut [f32],
) {
    let chunks = dim / LANES;
    let tail_start = chunks * LANES;

    for (i, &idx) in indices.iter().enumerate() {
        let src_off = (idx as usize) * dim;
        let dst_off = i * dim;

        if i + 1 < indices.len() {
            let next_off = (indices[i + 1] as usize) * dim;
            unsafe {
                prefetch_read(table.as_ptr().add(next_off) as *const u8);
            }
        }

        let src = &table[src_off..src_off + dim];
        let dst = &mut output[dst_off..dst_off + dim];

        for c in 0..chunks {
            let off = c * LANES;
            unsafe {
                // Convert f16 → f32 via half crate, then load as NEON f32x4.
                let f0 = src[off].to_f32();
                let f1 = src[off + 1].to_f32();
                let f2 = src[off + 2].to_f32();
                let f3 = src[off + 3].to_f32();
                let f4 = vld1q_f32([f0, f1, f2, f3].as_ptr());
                vst1q_f32(dst.as_mut_ptr().add(off), f4);
            }
        }
        // Scalar tail.
        for j in tail_start..dim {
            dst[j] = src[j].to_f32();
        }
    }
}

/// Scalar fallback for f16→f32 embedding lookup.
fn scalar_embedding_lookup_f16(table: &[f16], indices: &[u32], dim: usize, output: &mut [f32]) {
    for (i, &idx) in indices.iter().enumerate() {
        let src_off = (idx as usize) * dim;
        let dst_off = i * dim;
        for j in 0..dim {
            output[dst_off + j] = table[src_off + j].to_f32();
        }
    }
}

/// Embedding lookup from an f16 table, producing f32 output.
///
/// # Panics
/// Panics if `output` is too small or an index is out of bounds.
pub fn embedding_lookup_f16(table: &[f16], indices: &[u32], dim: usize, output: &mut [f32]) {
    assert!(dim > 0, "dim must be > 0");
    let vocab = table.len() / dim;
    assert_eq!(table.len(), vocab * dim, "table length must be a multiple of dim");
    for &idx in indices {
        assert!((idx as usize) < vocab, "index {idx} out of bounds for vocab {vocab}");
    }
    assert!(
        output.len() >= indices.len() * dim,
        "output too small: need {} but got {}",
        indices.len() * dim,
        output.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_embedding_lookup_f16(table, indices, dim, output);
            }
            return;
        }
    }
    scalar_embedding_lookup_f16(table, indices, dim, output);
}

// =====================================================================
// 3. embedding_sum_f32
// =====================================================================

/// NEON-optimized sum of multiple embedding lookups.
///
/// # Safety
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_embedding_sum_f32(
    table: &[f32],
    indices: &[&[u32]],
    dim: usize,
    output: &mut [f32],
) {
    let chunks = dim / LANES;
    let tail_start = chunks * LANES;

    // Zero output first.
    for v in output.iter_mut() {
        *v = 0.0;
    }

    for group in indices {
        for &idx in *group {
            let src_off = (idx as usize) * dim;
            unsafe {
                prefetch_read(table.as_ptr().add(src_off) as *const u8);
            }
            let src = &table[src_off..src_off + dim];

            for c in 0..chunks {
                let off = c * LANES;
                unsafe {
                    let acc = vld1q_f32(output.as_ptr().add(off));
                    let val = vld1q_f32(src.as_ptr().add(off));
                    vst1q_f32(output.as_mut_ptr().add(off), vaddq_f32(acc, val));
                }
            }
            for j in tail_start..dim {
                output[j] += src[j];
            }
        }
    }
}

/// Scalar fallback for embedding sum.
fn scalar_embedding_sum_f32(table: &[f32], indices: &[&[u32]], dim: usize, output: &mut [f32]) {
    for v in output.iter_mut() {
        *v = 0.0;
    }
    for group in indices {
        for &idx in *group {
            let src_off = (idx as usize) * dim;
            for j in 0..dim {
                output[j] += table[src_off + j];
            }
        }
    }
}

/// Sum multiple sets of embedding lookups into a single vector.
///
/// Each entry in `indices` is a group of token indices whose embeddings
/// are accumulated (summed) into `output` (shape `[dim]`).
///
/// # Panics
/// Panics if `output` is too small or any index is out of bounds.
pub fn embedding_sum_f32(table: &[f32], indices: &[&[u32]], dim: usize, output: &mut [f32]) {
    assert!(dim > 0, "dim must be > 0");
    let vocab = table.len() / dim;
    assert_eq!(table.len(), vocab * dim, "table length must be a multiple of dim");
    for group in indices {
        for &idx in *group {
            assert!((idx as usize) < vocab, "index {idx} out of bounds for vocab {vocab}");
        }
    }
    assert!(output.len() >= dim, "output too small: need {dim} but got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_embedding_sum_f32(table, indices, dim, output);
            }
            return;
        }
    }
    scalar_embedding_sum_f32(table, indices, dim, output);
}

// =====================================================================
// 4. embedding_normalize
// =====================================================================

/// NEON-optimized L2 normalization.
///
/// # Safety
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_embedding_normalize(embeddings: &mut [f32], dim: usize) -> Vec<f32> {
    let n_vecs = embeddings.len() / dim;
    let mut norms = Vec::with_capacity(n_vecs);
    let chunks = dim / LANES;
    let tail_start = chunks * LANES;

    for v in 0..n_vecs {
        let off = v * dim;
        let slice = &mut embeddings[off..off + dim];

        // Compute sum of squares.
        let mut sum_sq_vec = unsafe { vdupq_n_f32(0.0) };
        for c in 0..chunks {
            let co = c * LANES;
            unsafe {
                let val = vld1q_f32(slice.as_ptr().add(co));
                sum_sq_vec = vfmaq_f32(sum_sq_vec, val, val);
            }
        }
        let mut sum_sq = unsafe { vaddvq_f32(sum_sq_vec) };
        for j in tail_start..dim {
            sum_sq += slice[j] * slice[j];
        }

        let norm = sum_sq.sqrt();
        norms.push(norm);

        if norm > 0.0 {
            // Use NEON rsqrt estimate + Newton-Raphson refinement.
            let inv_norm = 1.0 / norm;
            let inv_vec = unsafe { vdupq_n_f32(inv_norm) };
            for c in 0..chunks {
                let co = c * LANES;
                unsafe {
                    let val = vld1q_f32(slice.as_ptr().add(co));
                    vst1q_f32(slice.as_mut_ptr().add(co), vmulq_f32(val, inv_vec));
                }
            }
            for j in tail_start..dim {
                slice[j] *= inv_norm;
            }
        }
    }
    norms
}

/// Scalar L2 normalization fallback.
fn scalar_embedding_normalize(embeddings: &mut [f32], dim: usize) -> Vec<f32> {
    let n_vecs = embeddings.len() / dim;
    let mut norms = Vec::with_capacity(n_vecs);

    for v in 0..n_vecs {
        let off = v * dim;
        let slice = &mut embeddings[off..off + dim];
        let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
        let norm = sum_sq.sqrt();
        norms.push(norm);
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for x in slice.iter_mut() {
                *x *= inv;
            }
        }
    }
    norms
}

/// L2-normalize each embedding vector of length `dim` in-place.
///
/// Returns a `Vec<f32>` of the original L2 norms (one per vector).
///
/// # Panics
/// Panics if `embeddings.len()` is not a multiple of `dim`.
pub fn embedding_normalize(embeddings: &mut [f32], dim: usize) -> Vec<f32> {
    assert!(dim > 0, "dim must be > 0");
    assert_eq!(
        embeddings.len() % dim,
        0,
        "embeddings length {} is not a multiple of dim {dim}",
        embeddings.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_embedding_normalize(embeddings, dim) };
        }
    }
    scalar_embedding_normalize(embeddings, dim)
}

// =====================================================================
// 5. embedding_similarity
// =====================================================================

/// NEON-optimized cosine similarity.
///
/// # Safety
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_embedding_similarity(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let chunks = dim / LANES;
    let tail_start = chunks * LANES;

    let mut dot_vec = unsafe { vdupq_n_f32(0.0) };
    let mut aa_vec = unsafe { vdupq_n_f32(0.0) };
    let mut bb_vec = unsafe { vdupq_n_f32(0.0) };

    for c in 0..chunks {
        let off = c * LANES;
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            dot_vec = vfmaq_f32(dot_vec, va, vb);
            aa_vec = vfmaq_f32(aa_vec, va, va);
            bb_vec = vfmaq_f32(bb_vec, vb, vb);
        }
    }

    let mut dot = unsafe { vaddvq_f32(dot_vec) };
    let mut aa = unsafe { vaddvq_f32(aa_vec) };
    let mut bb = unsafe { vaddvq_f32(bb_vec) };

    for j in tail_start..dim {
        dot += a[j] * b[j];
        aa += a[j] * a[j];
        bb += b[j] * b[j];
    }

    let denom = (aa * bb).sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

/// Scalar cosine similarity fallback.
fn scalar_embedding_similarity(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut dot = 0.0f32;
    let mut aa = 0.0f32;
    let mut bb = 0.0f32;
    for j in 0..dim {
        dot += a[j] * b[j];
        aa += a[j] * a[j];
        bb += b[j] * b[j];
    }
    let denom = (aa * bb).sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

/// Cosine similarity between two embedding vectors of length `dim`.
///
/// Returns a value in `[-1, 1]`.  If either vector has zero magnitude
/// the result is `0.0`.
///
/// # Panics
/// Panics if `a` or `b` is shorter than `dim`.
pub fn embedding_similarity(a: &[f32], b: &[f32], dim: usize) -> f32 {
    assert!(dim > 0, "dim must be > 0");
    assert!(a.len() >= dim, "a too short: need {dim} but got {}", a.len());
    assert!(b.len() >= dim, "b too short: need {dim} but got {}", b.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_embedding_similarity(a, b, dim) };
        }
    }
    scalar_embedding_similarity(a, b, dim)
}

// =====================================================================
// 6. position_embedding_rope
// =====================================================================

/// NEON-optimized RoPE positional encoding applied to embeddings.
///
/// # Safety
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_position_embedding_rope(
    embeddings: &mut [f32],
    positions: &[u32],
    dim: usize,
    base: f32,
) {
    let half_dim = dim / 2;
    let chunks = half_dim / LANES;
    let tail_start = chunks * LANES;

    for (i, &pos) in positions.iter().enumerate() {
        let off = i * dim;
        let emb = &mut embeddings[off..off + dim];

        for c in 0..chunks {
            let j = c * LANES;
            unsafe {
                // Compute theta for 4 consecutive dimension pairs.
                let mut thetas = [0.0f32; LANES];
                for k in 0..LANES {
                    let dim_idx = j + k;
                    let exponent = -(2.0 * dim_idx as f32) / dim as f32;
                    thetas[k] = (pos as f32) * base.powf(exponent);
                }

                let cos_v = vld1q_f32(
                    [thetas[0].cos(), thetas[1].cos(), thetas[2].cos(), thetas[3].cos()].as_ptr(),
                );
                let sin_v = vld1q_f32(
                    [thetas[0].sin(), thetas[1].sin(), thetas[2].sin(), thetas[3].sin()].as_ptr(),
                );

                let x0 = vld1q_f32(emb.as_ptr().add(j));
                let x1 = vld1q_f32(emb.as_ptr().add(half_dim + j));

                // x0' = x0 * cos - x1 * sin
                let new_x0 = vsubq_f32(vmulq_f32(x0, cos_v), vmulq_f32(x1, sin_v));
                // x1' = x0 * sin + x1 * cos  (using FMA)
                let new_x1 = vfmaq_f32(vmulq_f32(x1, cos_v), x0, sin_v);

                vst1q_f32(emb.as_mut_ptr().add(j), new_x0);
                vst1q_f32(emb.as_mut_ptr().add(half_dim + j), new_x1);
            }
        }
        // Scalar tail.
        for j in tail_start..half_dim {
            let exponent = -(2.0 * j as f32) / dim as f32;
            let theta = (pos as f32) * base.powf(exponent);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let x0 = emb[j];
            let x1 = emb[half_dim + j];
            emb[j] = x0 * cos_t - x1 * sin_t;
            emb[half_dim + j] = x0 * sin_t + x1 * cos_t;
        }
    }
}

/// Scalar RoPE positional encoding fallback.
fn scalar_position_embedding_rope(
    embeddings: &mut [f32],
    positions: &[u32],
    dim: usize,
    base: f32,
) {
    let half_dim = dim / 2;
    for (i, &pos) in positions.iter().enumerate() {
        let off = i * dim;
        let emb = &mut embeddings[off..off + dim];
        for j in 0..half_dim {
            let exponent = -(2.0 * j as f32) / dim as f32;
            let theta = (pos as f32) * base.powf(exponent);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let x0 = emb[j];
            let x1 = emb[half_dim + j];
            emb[j] = x0 * cos_t - x1 * sin_t;
            emb[half_dim + j] = x0 * sin_t + x1 * cos_t;
        }
    }
}

/// Apply RoPE (Rotary Position Embedding) positional encoding to embeddings.
///
/// `embeddings` has shape `[positions.len(), dim]`.  `dim` must be even.
/// `base` is the RoPE frequency base (commonly 10000.0).
///
/// # Panics
/// Panics if `dim` is odd or `embeddings` length does not match.
pub fn position_embedding_rope(embeddings: &mut [f32], positions: &[u32], dim: usize, base: f32) {
    assert!(dim > 0 && dim % 2 == 0, "dim must be a positive even number, got {dim}");
    assert!(
        embeddings.len() >= positions.len() * dim,
        "embeddings too small: need {} but got {}",
        positions.len() * dim,
        embeddings.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_position_embedding_rope(embeddings, positions, dim, base);
            }
            return;
        }
    }
    scalar_position_embedding_rope(embeddings, positions, dim, base);
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    // ── Helpers ──────────────────────────────────────────────────────

    fn make_table_f32(vocab: usize, dim: usize) -> Vec<f32> {
        (0..vocab * dim).map(|i| (i as f32) * 0.01).collect()
    }

    fn make_table_f16(vocab: usize, dim: usize) -> Vec<f16> {
        (0..vocab * dim).map(|i| f16::from_f32((i as f32) * 0.01)).collect()
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn vec_approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| approx_eq(x, y, eps))
    }

    // =================================================================
    // embedding_lookup_f32 tests
    // =================================================================

    #[test]
    fn test_f32_lookup_basic() {
        let table = make_table_f32(10, 4);
        let indices = [0u32, 2, 5];
        let mut out = vec![0.0f32; 3 * 4];
        embedding_lookup_f32(&table, &indices, 4, &mut out);

        assert_eq!(&out[0..4], &table[0..4]);
        assert_eq!(&out[4..8], &table[8..12]);
        assert_eq!(&out[8..12], &table[20..24]);
    }

    #[test]
    fn test_f32_lookup_single_index() {
        let table = make_table_f32(5, 8);
        let mut out = vec![0.0f32; 8];
        embedding_lookup_f32(&table, &[3], 8, &mut out);
        assert_eq!(&out, &table[24..32]);
    }

    #[test]
    fn test_f32_lookup_all_indices() {
        let vocab = 4;
        let dim = 4;
        let table = make_table_f32(vocab, dim);
        let indices: Vec<u32> = (0..vocab as u32).collect();
        let mut out = vec![0.0f32; vocab * dim];
        embedding_lookup_f32(&table, &indices, dim, &mut out);
        assert_eq!(out, table);
    }

    #[test]
    fn test_f32_lookup_repeated_index() {
        let table = make_table_f32(5, 4);
        let indices = [1u32, 1, 1];
        let mut out = vec![0.0f32; 3 * 4];
        embedding_lookup_f32(&table, &indices, 4, &mut out);
        for i in 0..3 {
            assert_eq!(&out[i * 4..(i + 1) * 4], &table[4..8]);
        }
    }

    #[test]
    fn test_f32_lookup_large_dim() {
        let dim = 256;
        let table = make_table_f32(4, dim);
        let mut out = vec![0.0f32; dim];
        embedding_lookup_f32(&table, &[2], dim, &mut out);
        assert_eq!(&out, &table[2 * dim..3 * dim]);
    }

    #[test]
    fn test_f32_lookup_non_aligned_dim() {
        let dim = 7; // Not a multiple of 4.
        let table = make_table_f32(3, dim);
        let mut out = vec![0.0f32; dim];
        embedding_lookup_f32(&table, &[1], dim, &mut out);
        assert_eq!(&out, &table[dim..2 * dim]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_f32_lookup_oob_panics() {
        let table = make_table_f32(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_lookup_f32(&table, &[3], 4, &mut out);
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn test_f32_lookup_zero_dim_panics() {
        let table = [1.0f32];
        let mut out = vec![0.0f32; 4];
        embedding_lookup_f32(&table, &[0], 0, &mut out);
    }

    // =================================================================
    // embedding_lookup_f16 tests
    // =================================================================

    #[test]
    fn test_f16_lookup_basic() {
        let table = make_table_f16(10, 4);
        let indices = [0u32, 3, 7];
        let mut out = vec![0.0f32; 3 * 4];
        embedding_lookup_f16(&table, &indices, 4, &mut out);

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..4 {
                let expected = table[(idx as usize) * 4 + j].to_f32();
                assert!(
                    approx_eq(out[i * 4 + j], expected, 1e-3),
                    "mismatch at [{i}][{j}]: {} vs {expected}",
                    out[i * 4 + j]
                );
            }
        }
    }

    #[test]
    fn test_f16_lookup_single() {
        let table = make_table_f16(5, 8);
        let mut out = vec![0.0f32; 8];
        embedding_lookup_f16(&table, &[2], 8, &mut out);
        for j in 0..8 {
            assert!(approx_eq(out[j], table[16 + j].to_f32(), 1e-3));
        }
    }

    #[test]
    fn test_f16_lookup_non_aligned_dim() {
        let dim = 5;
        let table = make_table_f16(4, dim);
        let mut out = vec![0.0f32; dim];
        embedding_lookup_f16(&table, &[1], dim, &mut out);
        for j in 0..dim {
            assert!(approx_eq(out[j], table[dim + j].to_f32(), 1e-3));
        }
    }

    #[test]
    fn test_f16_lookup_large_dim() {
        let dim = 128;
        let table = make_table_f16(4, dim);
        let mut out = vec![0.0f32; dim];
        embedding_lookup_f16(&table, &[3], dim, &mut out);
        for j in 0..dim {
            assert!(approx_eq(out[j], table[3 * dim + j].to_f32(), 1e-3));
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_f16_lookup_oob_panics() {
        let table = make_table_f16(2, 4);
        let mut out = vec![0.0f32; 4];
        embedding_lookup_f16(&table, &[2], 4, &mut out);
    }

    #[test]
    fn test_f16_lookup_zero_values() {
        let table = vec![f16::ZERO; 8];
        let mut out = vec![1.0f32; 4];
        embedding_lookup_f16(&table, &[0], 4, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_f16_lookup_repeated() {
        let table = make_table_f16(5, 4);
        let indices = [2u32, 2];
        let mut out = vec![0.0f32; 2 * 4];
        embedding_lookup_f16(&table, &indices, 4, &mut out);
        assert!(vec_approx_eq(&out[0..4], &out[4..8], 1e-6));
    }

    // =================================================================
    // embedding_sum_f32 tests
    // =================================================================

    #[test]
    fn test_sum_single_group_single_index() {
        let table = make_table_f32(5, 4);
        let group: &[u32] = &[2];
        let mut out = vec![0.0f32; 4];
        embedding_sum_f32(&table, &[group], 4, &mut out);
        assert_eq!(&out, &table[8..12]);
    }

    #[test]
    fn test_sum_single_group_multiple_indices() {
        let dim = 4;
        let table = make_table_f32(5, dim);
        let group: &[u32] = &[0, 1, 2];
        let mut out = vec![0.0f32; dim];
        embedding_sum_f32(&table, &[group], dim, &mut out);

        let mut expected = vec![0.0f32; dim];
        for &idx in group {
            for j in 0..dim {
                expected[j] += table[(idx as usize) * dim + j];
            }
        }
        assert!(vec_approx_eq(&out, &expected, 1e-5));
    }

    #[test]
    fn test_sum_multiple_groups() {
        let dim = 4;
        let table = make_table_f32(10, dim);
        let g1: &[u32] = &[0, 1];
        let g2: &[u32] = &[5];
        let mut out = vec![0.0f32; dim];
        embedding_sum_f32(&table, &[g1, g2], dim, &mut out);

        let mut expected = vec![0.0f32; dim];
        for &idx in &[0u32, 1, 5] {
            for j in 0..dim {
                expected[j] += table[(idx as usize) * dim + j];
            }
        }
        assert!(vec_approx_eq(&out, &expected, 1e-5));
    }

    #[test]
    fn test_sum_empty_group() {
        let table = make_table_f32(5, 4);
        let empty: &[u32] = &[];
        let mut out = vec![999.0f32; 4];
        embedding_sum_f32(&table, &[empty], 4, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_sum_non_aligned_dim() {
        let dim = 5;
        let table = make_table_f32(4, dim);
        let group: &[u32] = &[0, 3];
        let mut out = vec![0.0f32; dim];
        embedding_sum_f32(&table, &[group], dim, &mut out);

        let mut expected = vec![0.0f32; dim];
        for j in 0..dim {
            expected[j] = table[j] + table[3 * dim + j];
        }
        assert!(vec_approx_eq(&out, &expected, 1e-5));
    }

    #[test]
    fn test_sum_large_dim() {
        let dim = 256;
        let table = make_table_f32(4, dim);
        let group: &[u32] = &[1, 2];
        let mut out = vec![0.0f32; dim];
        embedding_sum_f32(&table, &[group], dim, &mut out);

        let mut expected = vec![0.0f32; dim];
        for j in 0..dim {
            expected[j] = table[dim + j] + table[2 * dim + j];
        }
        assert!(vec_approx_eq(&out, &expected, 1e-5));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_sum_oob_panics() {
        let table = make_table_f32(3, 4);
        let group: &[u32] = &[3];
        let mut out = vec![0.0f32; 4];
        embedding_sum_f32(&table, &[group], 4, &mut out);
    }

    // =================================================================
    // embedding_normalize tests
    // =================================================================

    #[test]
    fn test_normalize_unit_vector() {
        let mut emb = vec![1.0, 0.0, 0.0, 0.0];
        let norms = embedding_normalize(&mut emb, 4);
        assert!(approx_eq(norms[0], 1.0, 1e-6));
        assert!(approx_eq(emb[0], 1.0, 1e-6));
    }

    #[test]
    fn test_normalize_basic() {
        let mut emb = vec![3.0, 4.0];
        let norms = embedding_normalize(&mut emb, 2);
        assert!(approx_eq(norms[0], 5.0, 1e-5));
        assert!(approx_eq(emb[0], 0.6, 1e-5));
        assert!(approx_eq(emb[1], 0.8, 1e-5));
    }

    #[test]
    fn test_normalize_multiple_vectors() {
        let mut emb = vec![3.0, 4.0, 0.0, 5.0];
        let norms = embedding_normalize(&mut emb, 2);
        assert_eq!(norms.len(), 2);
        assert!(approx_eq(norms[0], 5.0, 1e-5));
        assert!(approx_eq(norms[1], 5.0, 1e-5));
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut emb = vec![0.0, 0.0, 0.0, 0.0];
        let norms = embedding_normalize(&mut emb, 4);
        assert!(approx_eq(norms[0], 0.0, 1e-6));
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize_preserves_direction() {
        let mut emb = vec![2.0, 0.0, 0.0, 0.0];
        embedding_normalize(&mut emb, 4);
        assert!(approx_eq(emb[0], 1.0, 1e-6));
        assert!(emb[1..4].iter().all(|&v| approx_eq(v, 0.0, 1e-6)));
    }

    #[test]
    fn test_normalize_result_is_unit() {
        let mut emb = vec![1.0, 2.0, 3.0, 4.0];
        embedding_normalize(&mut emb, 4);
        let norm_sq: f32 = emb.iter().map(|&x| x * x).sum();
        assert!(approx_eq(norm_sq, 1.0, 1e-5));
    }

    #[test]
    fn test_normalize_large_dim() {
        let dim = 128;
        let mut emb: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        embedding_normalize(&mut emb, dim);
        let norm_sq: f32 = emb.iter().map(|&x| x * x).sum();
        assert!(approx_eq(norm_sq, 1.0, 1e-4));
    }

    #[test]
    fn test_normalize_non_aligned_dim() {
        let mut emb = vec![1.0, 1.0, 1.0];
        let norms = embedding_normalize(&mut emb, 3);
        let expected_norm = 3.0f32.sqrt();
        assert!(approx_eq(norms[0], expected_norm, 1e-5));
        let norm_sq: f32 = emb.iter().map(|&x| x * x).sum();
        assert!(approx_eq(norm_sq, 1.0, 1e-5));
    }

    #[test]
    fn test_normalize_negative_values() {
        let mut emb = vec![-3.0, 4.0];
        let norms = embedding_normalize(&mut emb, 2);
        assert!(approx_eq(norms[0], 5.0, 1e-5));
        assert!(approx_eq(emb[0], -0.6, 1e-5));
        assert!(approx_eq(emb[1], 0.8, 1e-5));
    }

    #[test]
    #[should_panic(expected = "not a multiple of dim")]
    fn test_normalize_bad_length_panics() {
        let mut emb = vec![1.0, 2.0, 3.0];
        embedding_normalize(&mut emb, 2);
    }

    // =================================================================
    // embedding_similarity tests
    // =================================================================

    #[test]
    fn test_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let sim = embedding_similarity(&a, &a, 4);
        assert!(approx_eq(sim, 1.0, 1e-5));
    }

    #[test]
    fn test_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let sim = embedding_similarity(&a, &b, 4);
        assert!(approx_eq(sim, 0.0, 1e-5));
    }

    #[test]
    fn test_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = embedding_similarity(&a, &b, 2);
        assert!(approx_eq(sim, -1.0, 1e-5));
    }

    #[test]
    fn test_similarity_known_value() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let expected = dot / (na * nb);
        let sim = embedding_similarity(&a, &b, 3);
        assert!(approx_eq(sim, expected, 1e-5));
    }

    #[test]
    fn test_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let sim = embedding_similarity(&a, &b, 4);
        assert!(approx_eq(sim, 0.0, 1e-6));
    }

    #[test]
    fn test_similarity_large_dim() {
        let dim = 256;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.02).collect();
        let sim = embedding_similarity(&a, &b, dim);
        // Parallel vectors → sim ≈ 1.0.
        assert!(sim > 0.99);
    }

    #[test]
    fn test_similarity_non_aligned_dim() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let sim = embedding_similarity(&a, &b, 5);
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(sim, dot / (na * nb), 1e-5));
    }

    #[test]
    fn test_similarity_scale_invariant() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = a.iter().map(|&x| x * 100.0).collect();
        let sim = embedding_similarity(&a, &b, 4);
        assert!(approx_eq(sim, 1.0, 1e-5));
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn test_similarity_zero_dim_panics() {
        embedding_similarity(&[1.0], &[2.0], 0);
    }

    #[test]
    #[should_panic(expected = "a too short")]
    fn test_similarity_short_a_panics() {
        embedding_similarity(&[1.0], &[2.0, 3.0], 2);
    }

    // =================================================================
    // position_embedding_rope tests
    // =================================================================

    #[test]
    fn test_rope_position_zero() {
        let dim = 4;
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut emb = original.clone();
        position_embedding_rope(&mut emb, &[0], dim, 10000.0);
        // At pos=0, theta=0 so cos=1, sin=0 → no change.
        assert!(vec_approx_eq(&emb, &original, 1e-5));
    }

    #[test]
    fn test_rope_basic_rotation() {
        let dim = 4;
        let mut emb = vec![1.0, 0.0, 0.0, 0.0];
        position_embedding_rope(&mut emb, &[1], dim, 10000.0);
        // x0' = x0*cos - x1*sin = cos(theta), x1' = x0*sin + x1*cos = sin(theta)
        let theta0 = 1.0f32 * 10000.0f32.powf(0.0);
        assert!(approx_eq(emb[0], theta0.cos(), 1e-5));
        assert!(approx_eq(emb[2], theta0.sin(), 1e-5));
    }

    #[test]
    fn test_rope_preserves_norm() {
        let dim = 8;
        let mut emb: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();
        let orig_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        position_embedding_rope(&mut emb, &[5], dim, 10000.0);
        let new_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(orig_norm, new_norm, 1e-3));
    }

    #[test]
    fn test_rope_multiple_positions() {
        let dim = 4;
        let mut emb = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        position_embedding_rope(&mut emb, &[0, 3], dim, 10000.0);
        // Position 0 should leave first vector unchanged.
        assert!(approx_eq(emb[0], 1.0, 1e-5));
        assert!(approx_eq(emb[1], 2.0, 1e-5));
    }

    #[test]
    fn test_rope_large_dim() {
        let dim = 64;
        let mut emb: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let orig_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        position_embedding_rope(&mut emb, &[10], dim, 10000.0);
        let new_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(orig_norm, new_norm, 1e-2));
    }

    #[test]
    fn test_rope_non_aligned_half_dim() {
        // dim=6, half_dim=3 → not a multiple of 4.
        let dim = 6;
        let mut emb = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let orig_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        position_embedding_rope(&mut emb, &[2], dim, 10000.0);
        let new_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(orig_norm, new_norm, 1e-3));
    }

    #[test]
    #[should_panic(expected = "positive even number")]
    fn test_rope_odd_dim_panics() {
        let mut emb = vec![1.0; 5];
        position_embedding_rope(&mut emb, &[0], 5, 10000.0);
    }

    #[test]
    fn test_rope_different_bases() {
        let dim = 4;
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let mut emb1 = input.clone();
        position_embedding_rope(&mut emb1, &[1], dim, 100.0);

        let mut emb2 = input.clone();
        position_embedding_rope(&mut emb2, &[1], dim, 10000.0);

        // Different bases should give different results.
        assert!(!vec_approx_eq(&emb1, &emb2, 1e-3));
    }

    #[test]
    fn test_rope_consecutive_positions_differ() {
        let dim = 8;
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();

        let mut emb1 = input.clone();
        position_embedding_rope(&mut emb1, &[0], dim, 10000.0);

        let mut emb2 = input.clone();
        position_embedding_rope(&mut emb2, &[1], dim, 10000.0);

        assert!(!vec_approx_eq(&emb1, &emb2, 1e-3));
    }

    // =================================================================
    // Cross-function integration tests
    // =================================================================

    #[test]
    fn test_lookup_then_normalize() {
        let table = make_table_f32(10, 8);
        let mut out = vec![0.0f32; 8];
        embedding_lookup_f32(&table, &[5], 8, &mut out);
        let norms = embedding_normalize(&mut out, 8);
        assert!(norms[0] > 0.0);
        let norm_sq: f32 = out.iter().map(|x| x * x).sum();
        assert!(approx_eq(norm_sq, 1.0, 1e-5));
    }

    #[test]
    fn test_lookup_then_similarity() {
        let table = make_table_f32(10, 16);
        let mut a = vec![0.0f32; 16];
        let mut b = vec![0.0f32; 16];
        embedding_lookup_f32(&table, &[2], 16, &mut a);
        embedding_lookup_f32(&table, &[2], 16, &mut b);
        let sim = embedding_similarity(&a, &b, 16);
        assert!(approx_eq(sim, 1.0, 1e-5));
    }

    #[test]
    fn test_f16_lookup_then_normalize() {
        let table = make_table_f16(5, 8);
        let mut out = vec![0.0f32; 8];
        embedding_lookup_f16(&table, &[2], 8, &mut out);
        embedding_normalize(&mut out, 8);
        let norm_sq: f32 = out.iter().map(|x| x * x).sum();
        assert!(approx_eq(norm_sq, 1.0, 1e-4));
    }

    #[test]
    fn test_sum_then_normalize_then_similarity() {
        let dim = 16;
        let table = make_table_f32(10, dim);

        let g1: &[u32] = &[0, 1];
        let mut sum1 = vec![0.0f32; dim];
        embedding_sum_f32(&table, &[g1], dim, &mut sum1);

        let g2: &[u32] = &[0, 1];
        let mut sum2 = vec![0.0f32; dim];
        embedding_sum_f32(&table, &[g2], dim, &mut sum2);

        embedding_normalize(&mut sum1, dim);
        embedding_normalize(&mut sum2, dim);

        let sim = embedding_similarity(&sum1, &sum2, dim);
        assert!(approx_eq(sim, 1.0, 1e-5));
    }

    #[test]
    fn test_lookup_then_rope() {
        let dim = 8;
        let table = make_table_f32(5, dim);
        let mut emb = vec![0.0f32; dim];
        embedding_lookup_f32(&table, &[3], dim, &mut emb);
        let orig_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();

        position_embedding_rope(&mut emb, &[7], dim, 10000.0);
        let new_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(orig_norm, new_norm, 1e-3));
    }

    // =================================================================
    // Scalar fallback correctness
    // =================================================================

    #[test]
    fn test_scalar_f32_lookup_matches() {
        let table = make_table_f32(8, 16);
        let indices = [1u32, 4, 7];
        let mut out_pub = vec![0.0f32; 3 * 16];
        let mut out_scalar = vec![0.0f32; 3 * 16];

        embedding_lookup_f32(&table, &indices, 16, &mut out_pub);
        scalar_embedding_lookup_f32(&table, &indices, 16, &mut out_scalar);
        assert_eq!(out_pub, out_scalar);
    }

    #[test]
    fn test_scalar_f16_lookup_matches() {
        let table = make_table_f16(8, 16);
        let indices = [0u32, 3, 6];
        let mut out_pub = vec![0.0f32; 3 * 16];
        let mut out_scalar = vec![0.0f32; 3 * 16];

        embedding_lookup_f16(&table, &indices, 16, &mut out_pub);
        scalar_embedding_lookup_f16(&table, &indices, 16, &mut out_scalar);
        assert!(vec_approx_eq(&out_pub, &out_scalar, 1e-3));
    }

    #[test]
    fn test_scalar_sum_matches() {
        let table = make_table_f32(8, 8);
        let g: &[u32] = &[1, 3, 5];
        let mut out_pub = vec![0.0f32; 8];
        let mut out_scalar = vec![0.0f32; 8];

        embedding_sum_f32(&table, &[g], 8, &mut out_pub);
        scalar_embedding_sum_f32(&table, &[g], 8, &mut out_scalar);
        assert!(vec_approx_eq(&out_pub, &out_scalar, 1e-5));
    }

    #[test]
    fn test_scalar_normalize_matches() {
        let mut emb_pub = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut emb_scalar = emb_pub.clone();

        let norms_pub = embedding_normalize(&mut emb_pub, 4);
        let norms_scalar = scalar_embedding_normalize(&mut emb_scalar, 4);

        assert!(vec_approx_eq(&emb_pub, &emb_scalar, 1e-5));
        assert!(vec_approx_eq(&norms_pub, &norms_scalar, 1e-5));
    }

    #[test]
    fn test_scalar_similarity_matches() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let sim_pub = embedding_similarity(&a, &b, 5);
        let sim_scalar = scalar_embedding_similarity(&a, &b, 5);
        assert!(approx_eq(sim_pub, sim_scalar, 1e-5));
    }

    #[test]
    fn test_scalar_rope_matches() {
        let dim = 8;
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();

        let mut emb_pub = input.clone();
        position_embedding_rope(&mut emb_pub, &[5], dim, 10000.0);

        let mut emb_scalar = input;
        scalar_position_embedding_rope(&mut emb_scalar, &[5], dim, 10000.0);

        assert!(vec_approx_eq(&emb_pub, &emb_scalar, 1e-4));
    }

    // =================================================================
    // Edge cases
    // =================================================================

    #[test]
    fn test_f32_lookup_dim_1() {
        let table = vec![10.0, 20.0, 30.0];
        let mut out = vec![0.0f32; 2];
        embedding_lookup_f32(&table, &[0, 2], 1, &mut out);
        assert_eq!(out, vec![10.0, 30.0]);
    }

    #[test]
    fn test_f16_lookup_dim_1() {
        let table = vec![f16::from_f32(1.5), f16::from_f32(2.5)];
        let mut out = vec![0.0f32; 1];
        embedding_lookup_f16(&table, &[1], 1, &mut out);
        assert!(approx_eq(out[0], 2.5, 1e-2));
    }

    #[test]
    fn test_normalize_dim_1() {
        let mut emb = vec![5.0];
        let norms = embedding_normalize(&mut emb, 1);
        assert!(approx_eq(norms[0], 5.0, 1e-5));
        assert!(approx_eq(emb[0], 1.0, 1e-5));
    }

    #[test]
    fn test_similarity_dim_1() {
        let sim = embedding_similarity(&[3.0], &[-3.0], 1);
        assert!(approx_eq(sim, -1.0, 1e-5));
    }

    #[test]
    fn test_rope_dim_2() {
        let mut emb = vec![1.0, 0.0];
        position_embedding_rope(&mut emb, &[1], 2, 10000.0);
        let theta = 1.0f32;
        assert!(approx_eq(emb[0], theta.cos(), 1e-5));
        assert!(approx_eq(emb[1], theta.sin(), 1e-5));
    }

    #[test]
    fn test_f32_lookup_output_larger_than_needed() {
        let table = make_table_f32(3, 4);
        let mut out = vec![99.0f32; 8];
        embedding_lookup_f32(&table, &[1], 4, &mut out);
        assert_eq!(&out[0..4], &table[4..8]);
        // Extra space is untouched by our implementation or may be overwritten;
        // we only guarantee the first indices.len()*dim elements.
    }

    #[test]
    fn test_sum_result_doubles_with_same_index() {
        let dim = 4;
        let table = make_table_f32(5, dim);
        let g: &[u32] = &[2, 2];
        let mut out = vec![0.0f32; dim];
        embedding_sum_f32(&table, &[g], dim, &mut out);

        let expected: Vec<f32> = table[2 * dim..3 * dim].iter().map(|&x| x * 2.0).collect();
        assert!(vec_approx_eq(&out, &expected, 1e-5));
    }
}
