#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON optimized token embedding kernels for Apple Silicon.
//!
//! Provides SIMD-accelerated token embedding operations using `float32x4`
//! NEON intrinsics for 4-wide parallel computation. Every public function
//! includes a scalar fallback for non-aarch64 targets.
//!
//! # Functions
//!
//! - [`embed_tokens_neon`] — batch token embedding lookup
//! - [`embed_tokens_with_scale_neon`] — lookup with scaling factor
//! - [`add_position_embeddings_neon`] — add learned position embeddings
//! - [`embed_and_add_positions_neon`] — combined embed + position
//! - [`embedding_norm_neon`] — L2-normalize embedding vectors

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
#[cfg(target_arch = "aarch64")]
const LANES: usize = 4;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Clamp a token ID to the valid vocabulary range.
#[inline(always)]
fn clamp_token(token_id: u32, vocab_size: usize) -> usize {
    if vocab_size == 0 {
        return 0;
    }
    let id = token_id as usize;
    if id >= vocab_size { vocab_size - 1 } else { id }
}

/// Compute vocabulary size from embedding table length.
#[inline(always)]
fn vocab_size(table_len: usize, embed_dim: usize) -> usize {
    if embed_dim == 0 { 0 } else { table_len / embed_dim }
}

// ── embed_tokens_neon ───────────────────────────────────────────────────

/// Batch token embedding lookup with NEON acceleration.
///
/// For each token in `token_ids`, copies the corresponding row from
/// `embedding_table` into `output`. The inner copy uses 4-wide NEON
/// loads/stores when possible, with a scalar tail.
///
/// Out-of-range token IDs are clamped to the last valid row.
///
/// # Panics
///
/// Panics if `output.len() < token_ids.len() * embed_dim`.
pub fn embed_tokens_neon(
    token_ids: &[u32],
    embedding_table: &[f32],
    embed_dim: usize,
    output: &mut [f32],
) {
    if token_ids.is_empty() || embed_dim == 0 {
        return;
    }
    let vocab = vocab_size(embedding_table.len(), embed_dim);
    assert!(
        output.len() >= token_ids.len() * embed_dim,
        "output too small: need {}, got {}",
        token_ids.len() * embed_dim,
        output.len(),
    );

    for (i, &tid) in token_ids.iter().enumerate() {
        let row = clamp_token(tid, vocab);
        let src = &embedding_table[row * embed_dim..row * embed_dim + embed_dim];
        let dst = &mut output[i * embed_dim..(i + 1) * embed_dim];
        copy_row(src, dst);
    }
}

/// Copy a single embedding row with optional NEON acceleration.
#[inline]
fn copy_row(src: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 always has NEON.
        unsafe { copy_row_neon(src, dst) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dst.copy_from_slice(src);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn copy_row_neon(src: &[f32], dst: &mut [f32]) {
    let len = src.len();
    let simd_end = len - (len % LANES);
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    for j in (0..simd_end).step_by(LANES) {
        let v = vld1q_f32(sp.add(j));
        vst1q_f32(dp.add(j), v);
    }
    for j in simd_end..len {
        *dp.add(j) = *sp.add(j);
    }
}

// ── embed_tokens_with_scale_neon ────────────────────────────────────────

/// Batch token embedding lookup with a multiplicative scale factor.
///
/// Equivalent to `embed_tokens_neon` followed by element-wise
/// multiplication by `scale`, but fused into a single pass.
pub fn embed_tokens_with_scale_neon(
    token_ids: &[u32],
    embedding_table: &[f32],
    embed_dim: usize,
    scale: f32,
    output: &mut [f32],
) {
    if token_ids.is_empty() || embed_dim == 0 {
        return;
    }
    let vocab = vocab_size(embedding_table.len(), embed_dim);
    assert!(
        output.len() >= token_ids.len() * embed_dim,
        "output too small: need {}, got {}",
        token_ids.len() * embed_dim,
        output.len(),
    );

    for (i, &tid) in token_ids.iter().enumerate() {
        let row = clamp_token(tid, vocab);
        let src = &embedding_table[row * embed_dim..row * embed_dim + embed_dim];
        let dst = &mut output[i * embed_dim..(i + 1) * embed_dim];
        copy_row_scaled(src, dst, scale);
    }
}

#[inline]
fn copy_row_scaled(src: &[f32], dst: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { copy_row_scaled_neon(src, dst, scale) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s * scale;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn copy_row_scaled_neon(src: &[f32], dst: &mut [f32], scale: f32) {
    let len = src.len();
    let simd_end = len - (len % LANES);
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let vs = vdupq_n_f32(scale);
    for j in (0..simd_end).step_by(LANES) {
        let v = vld1q_f32(sp.add(j));
        vst1q_f32(dp.add(j), vmulq_f32(v, vs));
    }
    for j in simd_end..len {
        *dp.add(j) = *sp.add(j) * scale;
    }
}

// ── add_position_embeddings_neon ────────────────────────────────────────

/// Add learned position embeddings in-place.
///
/// For each position `p` in `0..seq_len`, adds
/// `position_table[p * embed_dim..(p+1) * embed_dim]` to the
/// corresponding row in `token_embeddings`.
///
/// # Panics
///
/// Panics if the slices are too small for `seq_len * embed_dim`.
pub fn add_position_embeddings_neon(
    token_embeddings: &mut [f32],
    position_table: &[f32],
    seq_len: usize,
    embed_dim: usize,
) {
    if seq_len == 0 || embed_dim == 0 {
        return;
    }
    assert!(token_embeddings.len() >= seq_len * embed_dim, "token_embeddings too small");
    assert!(position_table.len() >= seq_len * embed_dim, "position_table too small");

    for p in 0..seq_len {
        let start = p * embed_dim;
        let end = start + embed_dim;
        let pos_row = &position_table[start..end];
        let tok_row = &mut token_embeddings[start..end];
        add_row(tok_row, pos_row);
    }
}

#[inline]
fn add_row(dst: &mut [f32], src: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { add_row_neon(dst, src) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d += s;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn add_row_neon(dst: &mut [f32], src: &[f32]) {
    let len = dst.len();
    let simd_end = len - (len % LANES);
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    for j in (0..simd_end).step_by(LANES) {
        let a = vld1q_f32(dp.add(j));
        let b = vld1q_f32(sp.add(j));
        vst1q_f32(dp.add(j), vaddq_f32(a, b));
    }
    for j in simd_end..len {
        *dp.add(j) += *sp.add(j);
    }
}

// ── embed_and_add_positions_neon ────────────────────────────────────────

/// Combined token embedding lookup + positional embedding addition.
///
/// Fuses [`embed_tokens_neon`] and [`add_position_embeddings_neon`] into
/// one pass to halve memory traffic.
pub fn embed_and_add_positions_neon(
    token_ids: &[u32],
    embedding_table: &[f32],
    position_table: &[f32],
    embed_dim: usize,
    output: &mut [f32],
) {
    let seq_len = token_ids.len();
    if seq_len == 0 || embed_dim == 0 {
        return;
    }
    let vocab = vocab_size(embedding_table.len(), embed_dim);
    assert!(
        output.len() >= seq_len * embed_dim,
        "output too small: need {}, got {}",
        seq_len * embed_dim,
        output.len(),
    );
    assert!(position_table.len() >= seq_len * embed_dim, "position_table too small");

    for (i, &tid) in token_ids.iter().enumerate() {
        let row = clamp_token(tid, vocab);
        let emb = &embedding_table[row * embed_dim..row * embed_dim + embed_dim];
        let pos = &position_table[i * embed_dim..(i + 1) * embed_dim];
        let dst = &mut output[i * embed_dim..(i + 1) * embed_dim];
        add_two_rows(emb, pos, dst);
    }
}

#[inline]
fn add_two_rows(a: &[f32], b: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { add_two_rows_neon(a, b, dst) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for ((d, &va), &vb) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            *d = va + vb;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn add_two_rows_neon(a: &[f32], b: &[f32], dst: &mut [f32]) {
    let len = dst.len();
    let simd_end = len - (len % LANES);
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let dp = dst.as_mut_ptr();
    for j in (0..simd_end).step_by(LANES) {
        let va = vld1q_f32(ap.add(j));
        let vb = vld1q_f32(bp.add(j));
        vst1q_f32(dp.add(j), vaddq_f32(va, vb));
    }
    for j in simd_end..len {
        *dp.add(j) = *ap.add(j) + *bp.add(j);
    }
}

// ── embedding_norm_neon ─────────────────────────────────────────────────

/// L2-normalize each embedding vector.
///
/// For each of the `embeddings.len() / embed_dim` rows, computes the L2
/// norm and divides every element by it. Zero-norm rows are left as-is.
///
/// # Panics
///
/// Panics if `output.len() < embeddings.len()`.
pub fn embedding_norm_neon(embeddings: &[f32], embed_dim: usize, output: &mut [f32]) {
    if embed_dim == 0 || embeddings.is_empty() {
        return;
    }
    let n_rows = embeddings.len() / embed_dim;
    assert!(output.len() >= n_rows * embed_dim, "output too small");

    for r in 0..n_rows {
        let start = r * embed_dim;
        let end = start + embed_dim;
        let src = &embeddings[start..end];
        let dst = &mut output[start..end];
        norm_row(src, dst);
    }
}

#[inline]
fn norm_row(src: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { norm_row_neon(src, dst) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        norm_row_scalar(src, dst);
    }
}

fn norm_row_scalar(src: &[f32], dst: &mut [f32]) {
    let sq_sum: f32 = src.iter().map(|&v| v * v).sum();
    if sq_sum <= f32::EPSILON {
        dst.iter_mut().for_each(|d| *d = 0.0);
        return;
    }
    let inv_norm = 1.0 / sq_sum.sqrt();
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s * inv_norm;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn norm_row_neon(src: &[f32], dst: &mut [f32]) {
    let len = src.len();
    let simd_end = len - (len % LANES);
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    // Accumulate squared sum.
    let mut acc = vdupq_n_f32(0.0);
    for j in (0..simd_end).step_by(LANES) {
        let v = vld1q_f32(sp.add(j));
        acc = vfmaq_f32(acc, v, v);
    }
    // Horizontal sum of 4 lanes.
    let sq_sum = vaddvq_f32(acc) + src[simd_end..].iter().map(|&v| v * v).sum::<f32>();

    if sq_sum <= f32::EPSILON {
        for j in 0..len {
            *dp.add(j) = 0.0;
        }
        return;
    }

    let inv_norm = 1.0 / sq_sum.sqrt();
    let vi = vdupq_n_f32(inv_norm);
    for j in (0..simd_end).step_by(LANES) {
        let v = vld1q_f32(sp.add(j));
        vst1q_f32(dp.add(j), vmulq_f32(v, vi));
    }
    for j in simd_end..len {
        *dp.add(j) = *sp.add(j) * inv_norm;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // Small helper: build a simple embedding table where row i is filled
    // with (i+1) as f32.
    fn make_table(vocab: usize, dim: usize) -> Vec<f32> {
        (0..vocab).flat_map(|i| std::iter::repeat_n((i + 1) as f32, dim)).collect()
    }

    // Build a table where entry [i][j] = (i * dim + j) as f32 for
    // uniqueness across both rows and columns.
    fn make_unique_table(vocab: usize, dim: usize) -> Vec<f32> {
        (0..vocab * dim).map(|k| k as f32).collect()
    }

    fn make_position_table(seq_len: usize, dim: usize) -> Vec<f32> {
        (0..seq_len * dim).map(|k| (k as f32) * 0.01).collect()
    }

    // ── embed_tokens_neon ───────────────────────────────────────────

    #[test]
    fn embed_basic_lookup() {
        let table = make_table(4, 8);
        let ids = [0, 2, 1, 3];
        let mut out = vec![0.0f32; 4 * 8];
        embed_tokens_neon(&ids, &table, 8, &mut out);
        for j in 0..8 {
            assert_eq!(out[j], 1.0); // row 0
            assert_eq!(out[8 + j], 3.0); // row 2
            assert_eq!(out[16 + j], 2.0); // row 1
            assert_eq!(out[24 + j], 4.0); // row 3
        }
    }

    #[test]
    fn embed_unique_table_correctness() {
        let dim = 6;
        let table = make_unique_table(5, dim);
        let ids = [3, 0, 4];
        let mut out = vec![0.0f32; 3 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut out);
        for j in 0..dim {
            assert_eq!(out[j], (3 * dim + j) as f32);
            assert_eq!(out[dim + j], j as f32);
            assert_eq!(out[2 * dim + j], (4 * dim + j) as f32);
        }
    }

    #[test]
    fn embed_empty_tokens() {
        let table = make_table(4, 8);
        let mut out = [0.0f32; 0];
        embed_tokens_neon(&[], &table, 8, &mut out);
        // No panic, no output.
    }

    #[test]
    fn embed_zero_dim() {
        let ids = [0, 1];
        let mut out = [0.0f32; 0];
        embed_tokens_neon(&ids, &[], 0, &mut out);
    }

    #[test]
    fn embed_single_token() {
        let table = make_table(10, 4);
        let ids = [5];
        let mut out = [0.0f32; 4];
        embed_tokens_neon(&ids, &table, 4, &mut out);
        assert!(out.iter().all(|&v| v == 6.0));
    }

    #[test]
    fn embed_oob_clamped_to_last() {
        let table = make_table(3, 4);
        let ids = [100];
        let mut out = [0.0f32; 4];
        embed_tokens_neon(&ids, &table, 4, &mut out);
        // Clamped to last row (index 2) → value 3.0
        assert!(out.iter().all(|&v| v == 3.0));
    }

    #[test]
    fn embed_oob_u32_max() {
        let table = make_table(2, 4);
        let ids = [u32::MAX];
        let mut out = [0.0f32; 4];
        embed_tokens_neon(&ids, &table, 4, &mut out);
        assert!(out.iter().all(|&v| v == 2.0));
    }

    #[test]
    fn embed_non_aligned_dim() {
        let dim = 7;
        let table = make_unique_table(4, dim);
        let ids = [2];
        let mut out = vec![0.0f32; dim];
        embed_tokens_neon(&ids, &table, dim, &mut out);
        for j in 0..dim {
            assert_eq!(out[j], (2 * dim + j) as f32);
        }
    }

    #[test]
    fn embed_dim_1() {
        let table = vec![10.0, 20.0, 30.0];
        let ids = [1, 0, 2];
        let mut out = [0.0f32; 3];
        embed_tokens_neon(&ids, &table, 1, &mut out);
        assert_eq!(out.to_vec(), vec![20.0, 10.0, 30.0]);
    }

    #[test]
    fn embed_dim_3() {
        let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ids = [1, 0];
        let mut out = [0.0f32; 6];
        embed_tokens_neon(&ids, &table, 3, &mut out);
        assert_eq!(out.to_vec(), vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn embed_dim_5() {
        let dim = 5;
        let table = make_unique_table(3, dim);
        let ids = [0, 1, 2];
        let mut out = vec![0.0f32; 3 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut out);
        for i in 0..3 {
            for j in 0..dim {
                assert_eq!(out[i * dim + j], (i * dim + j) as f32);
            }
        }
    }

    #[test]
    fn embed_large_dim_aligned() {
        let dim = 128;
        let table = make_unique_table(8, dim);
        let ids = [7, 0];
        let mut out = vec![0.0f32; 2 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut out);
        for j in 0..dim {
            assert_eq!(out[j], (7 * dim + j) as f32);
            assert_eq!(out[dim + j], j as f32);
        }
    }

    #[test]
    fn embed_large_dim_unaligned() {
        let dim = 131;
        let table = make_unique_table(4, dim);
        let ids = [3];
        let mut out = vec![0.0f32; dim];
        embed_tokens_neon(&ids, &table, dim, &mut out);
        for j in 0..dim {
            assert_eq!(out[j], (3 * dim + j) as f32);
        }
    }

    #[test]
    fn embed_large_vocab() {
        let vocab = 50000;
        let dim = 4;
        let table = make_unique_table(vocab, dim);
        let ids = [49999, 0, 25000];
        let mut out = vec![0.0f32; 3 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut out);
        assert_eq!(out[0], (49999 * dim) as f32);
        assert_eq!(out[dim], 0.0);
        assert_eq!(out[2 * dim], (25000 * dim) as f32);
    }

    #[test]
    fn embed_repeated_tokens() {
        let table = make_table(4, 4);
        let ids = [2, 2, 2];
        let mut out = [0.0f32; 12];
        embed_tokens_neon(&ids, &table, 4, &mut out);
        assert!(out.iter().all(|&v| v == 3.0));
    }

    #[test]
    fn embed_all_tokens_in_vocab() {
        let vocab = 5;
        let dim = 4;
        let table = make_unique_table(vocab, dim);
        let ids: Vec<u32> = (0..vocab as u32).collect();
        let mut out = vec![0.0f32; vocab * dim];
        embed_tokens_neon(&ids, &table, dim, &mut out);
        assert_eq!(out, table);
    }

    // ── embed_tokens_with_scale_neon ────────────────────────────────

    #[test]
    fn scaled_basic() {
        let table = make_table(4, 8);
        let ids = [0, 1];
        let scale = 2.0;
        let mut out = vec![0.0f32; 2 * 8];
        embed_tokens_with_scale_neon(&ids, &table, 8, scale, &mut out);
        assert!(out[..8].iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert!(out[8..16].iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    #[test]
    fn scaled_zero() {
        let table = make_table(3, 4);
        let ids = [0, 1, 2];
        let mut out = [999.0f32; 12];
        embed_tokens_with_scale_neon(&ids, &table, 4, 0.0, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn scaled_one_is_identity() {
        let dim = 8;
        let table = make_unique_table(4, dim);
        let ids = [0, 3];
        let mut plain = vec![0.0f32; 2 * dim];
        let mut scaled = vec![0.0f32; 2 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut plain);
        embed_tokens_with_scale_neon(&ids, &table, dim, 1.0, &mut scaled);
        assert_eq!(plain, scaled);
    }

    #[test]
    fn scaled_negative() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let ids = [0];
        let mut out = [0.0f32; 4];
        embed_tokens_with_scale_neon(&ids, &table, 4, -0.5, &mut out);
        assert_eq!(out.to_vec(), vec![-0.5, -1.0, -1.5, -2.0]);
    }

    #[test]
    fn scaled_empty_tokens() {
        let table = make_table(4, 4);
        let mut out = [0.0; 0];
        embed_tokens_with_scale_neon(&[], &table, 4, 2.0, &mut out);
    }

    #[test]
    fn scaled_oob_clamped() {
        let table = make_table(3, 4); // rows: 1,2,3
        let ids = [999];
        let mut out = [0.0f32; 4];
        embed_tokens_with_scale_neon(&ids, &table, 4, 3.0, &mut out);
        // row 2 → 3.0, scaled by 3 → 9.0
        assert!(out.iter().all(|&v| (v - 9.0).abs() < 1e-6));
    }

    #[test]
    fn scaled_non_aligned_dim() {
        let dim = 5;
        let table = make_unique_table(3, dim);
        let ids = [1];
        let mut out = vec![0.0f32; dim];
        embed_tokens_with_scale_neon(&ids, &table, dim, 2.0, &mut out);
        for j in 0..dim {
            let expected = (dim + j) as f32 * 2.0;
            assert!((out[j] - expected).abs() < 1e-6, "at {j}: {} vs {expected}", out[j]);
        }
    }

    #[test]
    fn scaled_sqrt_d_model() {
        let dim = 64;
        let table = make_table(2, dim);
        let ids = [0];
        let scale = (dim as f32).sqrt();
        let mut out = vec![0.0f32; dim];
        embed_tokens_with_scale_neon(&ids, &table, dim, scale, &mut out);
        let expected = 1.0 * scale;
        assert!(out.iter().all(|&v| (v - expected).abs() < 1e-4));
    }

    #[test]
    fn scaled_large_dim() {
        let dim = 256;
        let table = make_unique_table(4, dim);
        let ids = [2];
        let mut out = vec![0.0f32; dim];
        embed_tokens_with_scale_neon(&ids, &table, dim, 0.1, &mut out);
        for j in 0..dim {
            let expected = (2 * dim + j) as f32 * 0.1;
            assert!((out[j] - expected).abs() < 1e-4);
        }
    }

    // ── add_position_embeddings_neon ────────────────────────────────

    #[test]
    fn add_pos_basic() {
        let dim = 4;
        let seq_len = 2;
        let mut tok = vec![1.0f32; seq_len * dim];
        let pos = vec![0.5f32; seq_len * dim];
        add_position_embeddings_neon(&mut tok, &pos, seq_len, dim);
        assert!(tok.iter().all(|&v| (v - 1.5).abs() < 1e-6));
    }

    #[test]
    fn add_pos_zero_seq() {
        let mut tok = [1.0f32; 8];
        let pos = [0.0f32; 8];
        add_position_embeddings_neon(&mut tok, &pos, 0, 4);
        assert!(tok.iter().all(|&v| v == 1.0)); // unchanged
    }

    #[test]
    fn add_pos_zero_dim() {
        let mut tok: Vec<f32> = vec![];
        let pos: Vec<f32> = vec![];
        add_position_embeddings_neon(&mut tok, &pos, 0, 0);
    }

    #[test]
    fn add_pos_non_aligned_dim() {
        let dim = 5;
        let seq_len = 3;
        let mut tok: Vec<f32> = (0..seq_len * dim).map(|k| k as f32).collect();
        let pos: Vec<f32> = (0..seq_len * dim).map(|k| (k as f32) * 0.1).collect();
        add_position_embeddings_neon(&mut tok, &pos, seq_len, dim);
        for k in 0..(seq_len * dim) {
            let expected = k as f32 + (k as f32) * 0.1;
            assert!((tok[k] - expected).abs() < 1e-5, "at {k}: {} vs {expected}", tok[k]);
        }
    }

    #[test]
    fn add_pos_identity_when_zero() {
        let dim = 8;
        let seq_len = 2;
        let mut tok = vec![5.0f32; seq_len * dim];
        let pos = vec![0.0f32; seq_len * dim];
        add_position_embeddings_neon(&mut tok, &pos, seq_len, dim);
        assert!(tok.iter().all(|&v| v == 5.0));
    }

    #[test]
    fn add_pos_single_position() {
        let dim = 8;
        let mut tok = vec![2.0f32; dim];
        let pos = vec![3.0f32; dim];
        add_position_embeddings_neon(&mut tok, &pos, 1, dim);
        assert!(tok.iter().all(|&v| (v - 5.0).abs() < 1e-6));
    }

    #[test]
    fn add_pos_large_dim() {
        let dim = 512;
        let seq_len = 4;
        let mut tok = vec![1.0f32; seq_len * dim];
        let pos: Vec<f32> = (0..seq_len * dim).map(|k| k as f32 * 0.001).collect();
        add_position_embeddings_neon(&mut tok, &pos, seq_len, dim);
        for k in 0..(seq_len * dim) {
            let expected = 1.0 + k as f32 * 0.001;
            assert!((tok[k] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn add_pos_negative_values() {
        let dim = 4;
        let mut tok = vec![1.0f32; dim];
        let pos = vec![-2.0f32; dim];
        add_position_embeddings_neon(&mut tok, &pos, 1, dim);
        assert!(tok.iter().all(|&v| (v - (-1.0)).abs() < 1e-6));
    }

    // ── embed_and_add_positions_neon ────────────────────────────────

    #[test]
    fn combined_basic() {
        let dim = 4;
        let table = make_table(4, dim); // row 0 → 1.0
        let pos = vec![0.5f32; 2 * dim];
        let ids = [0, 1];
        let mut out = vec![0.0f32; 2 * dim];
        embed_and_add_positions_neon(&ids, &table, &pos, dim, &mut out);
        assert!(out[..dim].iter().all(|&v| (v - 1.5).abs() < 1e-6));
        assert!(out[dim..].iter().all(|&v| (v - 2.5).abs() < 1e-6));
    }

    #[test]
    fn combined_empty() {
        let table = make_table(4, 4);
        let pos = [0.0f32; 0];
        let mut out = [0.0f32; 0];
        embed_and_add_positions_neon(&[], &table, &pos, 4, &mut out);
    }

    #[test]
    fn combined_matches_separate_ops() {
        let dim = 8;
        let table = make_unique_table(10, dim);
        let ids = [3, 7, 0, 9];
        let seq_len = ids.len();
        let pos = make_position_table(seq_len, dim);

        // Separate path
        let mut sep = vec![0.0f32; seq_len * dim];
        embed_tokens_neon(&ids, &table, dim, &mut sep);
        add_position_embeddings_neon(&mut sep, &pos, seq_len, dim);

        // Combined path
        let mut comb = vec![0.0f32; seq_len * dim];
        embed_and_add_positions_neon(&ids, &table, &pos, dim, &mut comb);

        for k in 0..(seq_len * dim) {
            assert!((sep[k] - comb[k]).abs() < 1e-6, "mismatch at {k}: {} vs {}", sep[k], comb[k]);
        }
    }

    #[test]
    fn combined_non_aligned_dim() {
        let dim = 7;
        let table = make_unique_table(5, dim);
        let pos = make_position_table(2, dim);
        let ids = [1, 4];
        let mut out = vec![0.0f32; 2 * dim];
        embed_and_add_positions_neon(&ids, &table, &pos, dim, &mut out);
        for j in 0..dim {
            let exp_tok = (1 * dim + j) as f32;
            let exp_pos = j as f32 * 0.01;
            assert!((out[j] - (exp_tok + exp_pos)).abs() < 1e-5);
        }
    }

    #[test]
    fn combined_oob_clamped() {
        let dim = 4;
        let table = make_table(3, dim); // last row → 3.0
        let pos = vec![0.0f32; dim];
        let ids = [999];
        let mut out = vec![0.0f32; dim];
        embed_and_add_positions_neon(&ids, &table, &pos, dim, &mut out);
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn combined_single_token() {
        let dim = 16;
        let table = make_unique_table(8, dim);
        let pos = vec![1.0f32; dim];
        let ids = [5];
        let mut out = vec![0.0f32; dim];
        embed_and_add_positions_neon(&ids, &table, &pos, dim, &mut out);
        for j in 0..dim {
            let expected = (5 * dim + j) as f32 + 1.0;
            assert!((out[j] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn combined_large_dim() {
        let dim = 256;
        let table = make_unique_table(4, dim);
        let pos: Vec<f32> = (0..dim).map(|j| j as f32 * 0.5).collect();
        let ids = [2];
        let mut out = vec![0.0f32; dim];
        embed_and_add_positions_neon(&ids, &table, &pos, dim, &mut out);
        for j in 0..dim {
            let expected = (2 * dim + j) as f32 + j as f32 * 0.5;
            assert!((out[j] - expected).abs() < 1e-4);
        }
    }

    // ── embedding_norm_neon ─────────────────────────────────────────

    #[test]
    fn norm_unit_vector() {
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        embedding_norm_neon(&input, 4, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[1..].iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn norm_produces_unit_length() {
        let input = vec![3.0, 4.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        embedding_norm_neon(&input, 4, &mut out);
        let len: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-5, "L2 norm should be 1.0, got {len}");
    }

    #[test]
    fn norm_preserves_direction() {
        let input = vec![3.0, 4.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        embedding_norm_neon(&input, 4, &mut out);
        assert!((out[0] - 0.6).abs() < 1e-5);
        assert!((out[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn norm_zero_vector() {
        let input = [0.0; 8];
        let mut out = [999.0f32; 8];
        embedding_norm_neon(&input, 8, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn norm_multiple_rows() {
        let input = vec![
            3.0, 4.0, 0.0, 0.0, // row 0: norm=5
            0.0, 0.0, 1.0, 0.0, // row 1: already unit
        ];
        let mut out = [0.0f32; 8];
        embedding_norm_neon(&input, 4, &mut out);
        // Row 0
        let n0: f32 = out[..4].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-5);
        // Row 1
        let n1: f32 = out[4..8].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn norm_non_aligned_dim() {
        let dim = 5;
        let input: Vec<f32> = (1..=dim).map(|k| k as f32).collect();
        let mut out = vec![0.0f32; dim];
        embedding_norm_neon(&input, dim, &mut out);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5);
    }

    #[test]
    fn norm_dim_1() {
        let input = [5.0];
        let mut out = [0.0f32; 1];
        embedding_norm_neon(&input, 1, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn norm_dim_3() {
        let input = vec![1.0, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        embedding_norm_neon(&input, 3, &mut out);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5);
    }

    #[test]
    fn norm_negative_values() {
        let input = vec![-3.0, -4.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        embedding_norm_neon(&input, 4, &mut out);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5);
        assert!(out[0] < 0.0);
        assert!(out[1] < 0.0);
    }

    #[test]
    fn norm_empty_input() {
        let mut out: Vec<f32> = vec![];
        embedding_norm_neon(&[], 4, &mut out);
    }

    #[test]
    fn norm_zero_dim() {
        let mut out: Vec<f32> = vec![];
        embedding_norm_neon(&[1.0, 2.0], 0, &mut out);
    }

    #[test]
    fn norm_large_dim() {
        let dim = 512;
        let input: Vec<f32> = (0..dim).map(|k| k as f32).collect();
        let mut out = vec![0.0f32; dim];
        embedding_norm_neon(&input, dim, &mut out);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4, "expected unit norm, got {n}");
    }

    #[test]
    fn norm_many_rows_non_aligned() {
        let dim = 7;
        let n_rows = 5;
        let input: Vec<f32> = (0..n_rows * dim).map(|k| (k as f32) + 1.0).collect();
        let mut out = vec![0.0f32; n_rows * dim];
        embedding_norm_neon(&input, dim, &mut out);
        for r in 0..n_rows {
            let n: f32 = out[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-5, "row {r}: L2 norm = {n}");
        }
    }

    #[test]
    fn norm_idempotent() {
        let dim = 8;
        let input: Vec<f32> = (0..dim).map(|k| (k + 1) as f32).collect();
        let mut first = vec![0.0f32; dim];
        embedding_norm_neon(&input, dim, &mut first);
        let mut second = vec![0.0f32; dim];
        embedding_norm_neon(&first, dim, &mut second);
        for j in 0..dim {
            assert!((first[j] - second[j]).abs() < 1e-5, "idempotent fail at {j}");
        }
    }

    // ── Cross-function ──────────────────────────────────────────────

    #[test]
    fn embed_then_norm_produces_unit_rows() {
        let dim = 8;
        let table = make_unique_table(4, dim);
        let ids = [0, 1, 2, 3];
        let mut emb = vec![0.0f32; 4 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut emb);
        // Skip row 0 (all zeros after unique table → first row starts at 0)
        // Row 1+ should normalise.
        let mut normed = vec![0.0f32; 4 * dim];
        embedding_norm_neon(&emb, dim, &mut normed);
        for r in 1..4 {
            let n: f32 = normed[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "row {r}: L2 = {n}");
        }
    }

    #[test]
    fn scaled_then_norm_same_direction() {
        let dim = 4;
        let table = vec![3.0, 4.0, 0.0, 0.0];
        let ids = [0];
        let mut plain = vec![0.0f32; dim];
        let mut scaled = vec![0.0f32; dim];
        embed_tokens_neon(&ids, &table, dim, &mut plain);
        embed_tokens_with_scale_neon(&ids, &table, dim, 100.0, &mut scaled);
        let mut np = vec![0.0f32; dim];
        let mut ns = vec![0.0f32; dim];
        embedding_norm_neon(&plain, dim, &mut np);
        embedding_norm_neon(&scaled, dim, &mut ns);
        for j in 0..dim {
            assert!((np[j] - ns[j]).abs() < 1e-5, "direction differs at {j}");
        }
    }

    #[test]
    fn combined_then_norm() {
        let dim = 8;
        let table = make_unique_table(4, dim);
        let pos = make_position_table(2, dim);
        let ids = [1, 3];
        let mut out = vec![0.0f32; 2 * dim];
        embed_and_add_positions_neon(&ids, &table, &pos, dim, &mut out);
        let mut normed = vec![0.0f32; 2 * dim];
        embedding_norm_neon(&out, dim, &mut normed);
        for r in 0..2 {
            let n: f32 = normed[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4);
        }
    }

    // ── Scalar path verification (compare with manual) ──────────────

    #[test]
    fn embed_matches_manual_indexing() {
        let dim = 6;
        let vocab = 5;
        let table = make_unique_table(vocab, dim);
        let ids = [4, 0, 2];
        let mut out = vec![0.0f32; 3 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut out);
        for (i, &tid) in ids.iter().enumerate() {
            for j in 0..dim {
                let expected = table[tid as usize * dim + j];
                assert_eq!(out[i * dim + j], expected);
            }
        }
    }

    #[test]
    fn scaled_matches_manual() {
        let dim = 6;
        let table = make_unique_table(3, dim);
        let ids = [2, 0];
        let scale = 1.5;
        let mut out = vec![0.0f32; 2 * dim];
        embed_tokens_with_scale_neon(&ids, &table, dim, scale, &mut out);
        for (i, &tid) in ids.iter().enumerate() {
            for j in 0..dim {
                let expected = table[tid as usize * dim + j] * scale;
                assert!((out[i * dim + j] - expected).abs() < 1e-6,);
            }
        }
    }

    #[test]
    fn add_pos_matches_manual() {
        let dim = 6;
        let seq_len = 3;
        let mut tok: Vec<f32> = (0..seq_len * dim).map(|k| k as f32).collect();
        let pos: Vec<f32> = (0..seq_len * dim).map(|k| (k as f32) * 10.0).collect();
        let expected: Vec<f32> = (0..seq_len * dim).map(|k| k as f32 + (k as f32) * 10.0).collect();
        add_position_embeddings_neon(&mut tok, &pos, seq_len, dim);
        for k in 0..(seq_len * dim) {
            assert!((tok[k] - expected[k]).abs() < 1e-5, "at {k}: {} vs {}", tok[k], expected[k]);
        }
    }

    #[test]
    fn norm_matches_manual_l2() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dim = 8;
        let norm: f32 = input.iter().map(|v| v * v).sum::<f32>().sqrt();
        let expected: Vec<f32> = input.iter().map(|v| v / norm).collect();
        let mut out = vec![0.0f32; dim];
        embedding_norm_neon(&input, dim, &mut out);
        for j in 0..dim {
            assert!((out[j] - expected[j]).abs() < 1e-6, "at {j}: {} vs {}", out[j], expected[j]);
        }
    }

    // ── NEON vs scalar parity (on aarch64) ──────────────────────────

    /// Helper: scalar embed for comparison.
    fn embed_scalar(ids: &[u32], table: &[f32], dim: usize, out: &mut [f32]) {
        let vocab = vocab_size(table.len(), dim);
        for (i, &tid) in ids.iter().enumerate() {
            let row = clamp_token(tid, vocab);
            let src = &table[row * dim..row * dim + dim];
            out[i * dim..(i + 1) * dim].copy_from_slice(src);
        }
    }

    fn embed_scaled_scalar(ids: &[u32], table: &[f32], dim: usize, scale: f32, out: &mut [f32]) {
        embed_scalar(ids, table, dim, out);
        for v in out.iter_mut() {
            *v *= scale;
        }
    }

    fn norm_scalar(src: &[f32], dim: usize, dst: &mut [f32]) {
        let n_rows = src.len() / dim;
        for r in 0..n_rows {
            let start = r * dim;
            let end = start + dim;
            let row = &src[start..end];
            let sq: f32 = row.iter().map(|v| v * v).sum();
            if sq <= f32::EPSILON {
                dst[start..end].fill(0.0);
            } else {
                let inv = 1.0 / sq.sqrt();
                for j in start..end {
                    dst[j] = src[j] * inv;
                }
            }
        }
    }

    #[test]
    fn neon_vs_scalar_embed() {
        let dim = 13;
        let table = make_unique_table(20, dim);
        let ids: Vec<u32> = (0..10).collect();
        let mut neon_out = vec![0.0f32; 10 * dim];
        let mut scalar_out = vec![0.0f32; 10 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut neon_out);
        embed_scalar(&ids, &table, dim, &mut scalar_out);
        assert_eq!(neon_out, scalar_out);
    }

    #[test]
    fn neon_vs_scalar_scaled() {
        let dim = 9;
        let table = make_unique_table(8, dim);
        let ids: Vec<u32> = vec![0, 3, 7, 1];
        let scale = 2.5;
        let mut neon_out = vec![0.0f32; 4 * dim];
        let mut scalar_out = vec![0.0f32; 4 * dim];
        embed_tokens_with_scale_neon(&ids, &table, dim, scale, &mut neon_out);
        embed_scaled_scalar(&ids, &table, dim, scale, &mut scalar_out);
        for k in 0..(4 * dim) {
            assert!(
                (neon_out[k] - scalar_out[k]).abs() < 1e-5,
                "at {k}: {} vs {}",
                neon_out[k],
                scalar_out[k]
            );
        }
    }

    #[test]
    fn neon_vs_scalar_norm() {
        let dim = 11;
        let n_rows = 4;
        let input: Vec<f32> = (0..n_rows * dim).map(|k| (k + 1) as f32).collect();
        let mut neon_out = vec![0.0f32; n_rows * dim];
        let mut scalar_out = vec![0.0f32; n_rows * dim];
        embedding_norm_neon(&input, dim, &mut neon_out);
        norm_scalar(&input, dim, &mut scalar_out);
        for k in 0..(n_rows * dim) {
            assert!(
                (neon_out[k] - scalar_out[k]).abs() < 1e-5,
                "at {k}: {} vs {}",
                neon_out[k],
                scalar_out[k]
            );
        }
    }

    #[test]
    fn neon_vs_scalar_embed_oob() {
        let dim = 5;
        let table = make_unique_table(3, dim);
        let ids = vec![0, 1, 2, 3, 100, u32::MAX];
        let mut neon_out = vec![0.0f32; 6 * dim];
        let mut scalar_out = vec![0.0f32; 6 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut neon_out);
        embed_scalar(&ids, &table, dim, &mut scalar_out);
        assert_eq!(neon_out, scalar_out);
    }

    // ── Stress / large-batch tests ──────────────────────────────────

    #[test]
    fn embed_large_batch() {
        let dim = 64;
        let vocab = 1000;
        let table = make_unique_table(vocab, dim);
        let ids: Vec<u32> = (0..256).map(|i| i % vocab as u32).collect();
        let mut out = vec![0.0f32; 256 * dim];
        embed_tokens_neon(&ids, &table, dim, &mut out);
        for (i, &tid) in ids.iter().enumerate() {
            assert_eq!(out[i * dim], table[tid as usize * dim],);
        }
    }

    #[test]
    fn scaled_large_batch() {
        let dim = 32;
        let vocab = 500;
        let table = make_unique_table(vocab, dim);
        let ids: Vec<u32> = (0..128).map(|i| i % vocab as u32).collect();
        let scale = 0.01;
        let mut out = vec![0.0f32; 128 * dim];
        embed_tokens_with_scale_neon(&ids, &table, dim, scale, &mut out);
        for (i, &tid) in ids.iter().enumerate() {
            let expected = table[tid as usize * dim] * scale;
            assert!((out[i * dim] - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn norm_large_batch() {
        let dim = 64;
        let n_rows = 100;
        let input: Vec<f32> = (0..n_rows * dim).map(|k| (k + 1) as f32).collect();
        let mut out = vec![0.0f32; n_rows * dim];
        embedding_norm_neon(&input, dim, &mut out);
        for r in 0..n_rows {
            let n: f32 = out[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "row {r}: L2 = {n}");
        }
    }

    #[test]
    fn combined_large_batch() {
        let dim = 64;
        let vocab = 100;
        let table = make_unique_table(vocab, dim);
        let seq_len = 50;
        let pos = make_position_table(seq_len, dim);
        let ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab as u32).collect();
        let mut out = vec![0.0f32; seq_len * dim];
        embed_and_add_positions_neon(&ids, &table, &pos, dim, &mut out);
        // Verify first element of first row.
        let expected = table[0] + pos[0];
        assert!((out[0] - expected).abs() < 1e-5);
    }

    // ── Edge-case: dim = 2 (less than LANES) ────────────────────────

    #[test]
    fn embed_dim_2() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let ids = [1, 0];
        let mut out = [0.0f32; 4];
        embed_tokens_neon(&ids, &table, 2, &mut out);
        assert_eq!(out.to_vec(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn scaled_dim_2() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let ids = [0];
        let mut out = [0.0f32; 2];
        embed_tokens_with_scale_neon(&ids, &table, 2, 3.0, &mut out);
        assert_eq!(out.to_vec(), vec![3.0, 6.0]);
    }

    #[test]
    fn add_pos_dim_2() {
        let mut tok = vec![1.0, 2.0];
        let pos = vec![10.0, 20.0];
        add_position_embeddings_neon(&mut tok, &pos, 1, 2);
        assert_eq!(tok, vec![11.0, 22.0]);
    }

    #[test]
    fn combined_dim_2() {
        let table = vec![1.0, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2];
        let ids = [1];
        let mut out = [0.0f32; 2];
        embed_and_add_positions_neon(&ids, &table, &pos, 2, &mut out);
        assert!((out[0] - 3.1).abs() < 1e-6);
        assert!((out[1] - 4.2).abs() < 1e-6);
    }

    #[test]
    fn norm_dim_2() {
        let input = vec![3.0, 4.0];
        let mut out = [0.0f32; 2];
        embedding_norm_neon(&input, 2, &mut out);
        assert!((out[0] - 0.6).abs() < 1e-5);
        assert!((out[1] - 0.8).abs() < 1e-5);
    }

    // ── Tiny epsilon / near-zero norm ───────────────────────────────

    #[test]
    fn norm_tiny_values() {
        let input = vec![1e-20, 1e-20, 1e-20, 1e-20];
        let mut out = [999.0f32; 4];
        embedding_norm_neon(&input, 4, &mut out);
        // Near-zero → treated as zero.
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn norm_very_large_values() {
        let input = vec![1e18, 1e18, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        embedding_norm_neon(&input, 4, &mut out);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }

    // ── Clamp helper ────────────────────────────────────────────────

    #[test]
    fn clamp_within_range() {
        assert_eq!(clamp_token(0, 10), 0);
        assert_eq!(clamp_token(9, 10), 9);
    }

    #[test]
    fn clamp_at_boundary() {
        assert_eq!(clamp_token(10, 10), 9);
    }

    #[test]
    fn clamp_far_oob() {
        assert_eq!(clamp_token(u32::MAX, 5), 4);
    }

    #[test]
    fn clamp_empty_vocab() {
        assert_eq!(clamp_token(0, 0), 0);
    }

    #[test]
    fn clamp_single_vocab() {
        assert_eq!(clamp_token(0, 1), 0);
        assert_eq!(clamp_token(5, 1), 0);
    }

    #[test]
    fn vocab_size_helper() {
        assert_eq!(vocab_size(100, 10), 10);
        assert_eq!(vocab_size(0, 10), 0);
        assert_eq!(vocab_size(10, 0), 0);
    }
}
