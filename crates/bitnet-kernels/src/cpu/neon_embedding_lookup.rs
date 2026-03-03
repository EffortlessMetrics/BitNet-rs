//! NEON-optimized embedding lookup and operations for Apple Silicon.

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
    clippy::excessive_precision
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Scalar reference implementations (portable, used for testing)
// ---------------------------------------------------------------------------

/// Scalar embedding lookup: copies `embedding_table[token_id]` into `output`.
pub fn scalar_embedding_lookup(
    embedding_table: &[f32],
    vocab_size: usize,
    dim: usize,
    token_id: usize,
    output: &mut [f32],
) {
    assert!(token_id < vocab_size, "token_id {token_id} out of bounds (vocab_size={vocab_size})");
    assert_eq!(embedding_table.len(), vocab_size * dim);
    assert!(output.len() >= dim);
    let start = token_id * dim;
    output[..dim].copy_from_slice(&embedding_table[start..start + dim]);
}

/// Scalar batched embedding lookup.
pub fn scalar_embedding_lookup_batched(
    embedding_table: &[f32],
    vocab_size: usize,
    dim: usize,
    token_ids: &[usize],
    output: &mut [f32],
) {
    let batch = token_ids.len();
    assert!(output.len() >= batch * dim);
    for (i, &tid) in token_ids.iter().enumerate() {
        scalar_embedding_lookup(embedding_table, vocab_size, dim, tid, &mut output[i * dim..]);
    }
}

/// Scalar: add positional embeddings element-wise.
pub fn scalar_embedding_add_position(
    token_emb: &[f32],
    pos_emb: &[f32],
    output: &mut [f32],
    batch: usize,
    dim: usize,
) {
    assert!(token_emb.len() >= batch * dim);
    assert!(pos_emb.len() >= batch * dim);
    assert!(output.len() >= batch * dim);
    for i in 0..batch * dim {
        output[i] = token_emb[i] + pos_emb[i];
    }
}

/// Scalar gather-scatter: gather embeddings for sparse `indices`, write to
/// contiguous `output`.
pub fn scalar_embedding_gather_scatter(
    embedding_table: &[f32],
    vocab_size: usize,
    dim: usize,
    indices: &[usize],
    output: &mut [f32],
) {
    assert!(output.len() >= indices.len() * dim);
    for (i, &idx) in indices.iter().enumerate() {
        assert!(idx < vocab_size, "index {idx} out of bounds (vocab_size={vocab_size})");
        let src = idx * dim;
        output[i * dim..(i + 1) * dim].copy_from_slice(&embedding_table[src..src + dim]);
    }
}

/// Scalar L2-normalize each embedding vector in a batch.
pub fn scalar_embedding_normalize(data: &mut [f32], batch: usize, dim: usize, eps: f32) {
    assert!(data.len() >= batch * dim);
    for b in 0..batch {
        let off = b * dim;
        let mut sq_sum = 0.0f32;
        for j in 0..dim {
            sq_sum += data[off + j] * data[off + j];
        }
        let inv_norm = 1.0 / (sq_sum + eps).sqrt();
        for j in 0..dim {
            data[off + j] *= inv_norm;
        }
    }
}

/// Scalar batch dot-product: `out[i] = dot(a[i], b[i])`.
pub fn scalar_embedding_dot_product(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    dim: usize,
) {
    assert!(a.len() >= batch * dim);
    assert!(b.len() >= batch * dim);
    assert!(out.len() >= batch);
    for i in 0..batch {
        let off = i * dim;
        let mut acc = 0.0f32;
        for j in 0..dim {
            acc += a[off + j] * b[off + j];
        }
        out[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// NEON-optimised implementations
// ---------------------------------------------------------------------------

/// NEON-accelerated embedding lookup using 128-bit vector loads/stores.
///
/// Copies the embedding vector for `token_id` from `embedding_table` into
/// `output`, processing 4 × f32 lanes per iteration.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_embedding_lookup(
    embedding_table: &[f32],
    vocab_size: usize,
    dim: usize,
    token_id: usize,
    output: &mut [f32],
) {
    assert!(token_id < vocab_size, "token_id {token_id} out of bounds (vocab_size={vocab_size})");
    assert_eq!(embedding_table.len(), vocab_size * dim);
    assert!(output.len() >= dim);

    let src = embedding_table.as_ptr().add(token_id * dim);
    let dst = output.as_mut_ptr();
    let chunks = dim / 4;

    for i in 0..chunks {
        let off = i * 4;
        let v = vld1q_f32(src.add(off));
        vst1q_f32(dst.add(off), v);
    }
    for i in (chunks * 4)..dim {
        *dst.add(i) = *src.add(i);
    }
}

/// NEON-accelerated batched embedding lookup.
///
/// For each token ID in `token_ids`, copies the corresponding embedding row
/// into the matching region of `output`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_embedding_lookup_batched(
    embedding_table: &[f32],
    vocab_size: usize,
    dim: usize,
    token_ids: &[usize],
    output: &mut [f32],
) {
    let batch = token_ids.len();
    assert!(output.len() >= batch * dim);

    for (i, &tid) in token_ids.iter().enumerate() {
        neon_embedding_lookup(embedding_table, vocab_size, dim, tid, &mut output[i * dim..]);
    }
}

/// NEON element-wise addition of positional embeddings to token embeddings.
///
/// `output[i] = token_emb[i] + pos_emb[i]` for `batch * dim` elements.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_embedding_add_position(
    token_emb: &[f32],
    pos_emb: &[f32],
    output: &mut [f32],
    batch: usize,
    dim: usize,
) {
    let total = batch * dim;
    assert!(token_emb.len() >= total);
    assert!(pos_emb.len() >= total);
    assert!(output.len() >= total);

    let t_ptr = token_emb.as_ptr();
    let p_ptr = pos_emb.as_ptr();
    let o_ptr = output.as_mut_ptr();
    let chunks = total / 4;

    for i in 0..chunks {
        let off = i * 4;
        let vt = vld1q_f32(t_ptr.add(off));
        let vp = vld1q_f32(p_ptr.add(off));
        vst1q_f32(o_ptr.add(off), vaddq_f32(vt, vp));
    }
    for i in (chunks * 4)..total {
        *o_ptr.add(i) = *t_ptr.add(i) + *p_ptr.add(i);
    }
}

/// NEON gather-scatter: gather embeddings for sparse `indices`, write
/// contiguously into `output`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_embedding_gather_scatter(
    embedding_table: &[f32],
    vocab_size: usize,
    dim: usize,
    indices: &[usize],
    output: &mut [f32],
) {
    assert!(output.len() >= indices.len() * dim);

    for (i, &idx) in indices.iter().enumerate() {
        assert!(idx < vocab_size, "index {idx} out of bounds (vocab_size={vocab_size})");
        let src = embedding_table.as_ptr().add(idx * dim);
        let dst = output.as_mut_ptr().add(i * dim);
        let chunks = dim / 4;
        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(src.add(off));
            vst1q_f32(dst.add(off), v);
        }
        for j in (chunks * 4)..dim {
            *dst.add(j) = *src.add(j);
        }
    }
}

/// NEON L2-normalize each embedding vector in a batch.
///
/// Each row `data[b*dim .. (b+1)*dim]` is divided by its L2 norm
/// (with epsilon for numerical stability).
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_embedding_normalize(data: &mut [f32], batch: usize, dim: usize, eps: f32) {
    assert!(data.len() >= batch * dim);

    for b in 0..batch {
        let base = data.as_mut_ptr().add(b * dim);

        // Accumulate squared sum with NEON.
        let chunks = dim / 4;
        let mut vsum = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let v = vld1q_f32(base.add(c * 4));
            vsum = vfmaq_f32(vsum, v, v);
        }
        let mut sq_sum: f32 = vaddvq_f32(vsum);
        for j in (chunks * 4)..dim {
            let x = *base.add(j);
            sq_sum += x * x;
        }

        let inv_norm = 1.0 / (sq_sum + eps).sqrt();
        let vinv = vdupq_n_f32(inv_norm);

        for c in 0..chunks {
            let off = c * 4;
            let v = vld1q_f32(base.add(off));
            vst1q_f32(base.add(off), vmulq_f32(v, vinv));
        }
        for j in (chunks * 4)..dim {
            *base.add(j) *= inv_norm;
        }
    }
}

/// NEON batch dot-product between embedding vectors.
///
/// `out[i] = dot(a[i*dim .. (i+1)*dim], b[i*dim .. (i+1)*dim])`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_embedding_dot_product(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    dim: usize,
) {
    assert!(a.len() >= batch * dim);
    assert!(b.len() >= batch * dim);
    assert!(out.len() >= batch);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let chunks = dim / 4;

    for i in 0..batch {
        let off = i * dim;
        let mut vsum = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let va = vld1q_f32(a_ptr.add(off + c * 4));
            let vb = vld1q_f32(b_ptr.add(off + c * 4));
            vsum = vfmaq_f32(vsum, va, vb);
        }
        let mut acc: f32 = vaddvq_f32(vsum);
        for j in (chunks * 4)..dim {
            acc += a[off + j] * b[off + j];
        }
        out[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ---------------------------------------------------------------

    /// Build a deterministic embedding table: row `r`, col `c` = `(r * dim + c) as f32 * 0.01`.
    fn make_table(vocab_size: usize, dim: usize) -> Vec<f32> {
        (0..vocab_size * dim).map(|i| i as f32 * 0.01).collect()
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn assert_slices_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(x, y, tol), "mismatch at index {i}: {x} vs {y} (tol={tol})");
        }
    }

    // -----------------------------------------------------------------------
    // Basic lookup correctness vs scalar (15 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_lookup_first_row() {
        let table = make_table(8, 4);
        let mut out = vec![0.0f32; 4];
        scalar_embedding_lookup(&table, 8, 4, 0, &mut out);
        assert_eq!(&out, &[0.0, 0.01, 0.02, 0.03]);
    }

    #[test]
    fn test_scalar_lookup_last_row() {
        let table = make_table(8, 4);
        let mut out = vec![0.0f32; 4];
        scalar_embedding_lookup(&table, 8, 4, 7, &mut out);
        let expected: Vec<f32> = (28..32).map(|i| i as f32 * 0.01).collect();
        assert_slices_approx(&out, &expected, 1e-6);
    }

    #[test]
    fn test_scalar_lookup_middle_row() {
        let table = make_table(10, 8);
        let mut out = vec![0.0f32; 8];
        scalar_embedding_lookup(&table, 10, 8, 5, &mut out);
        let expected: Vec<f32> = (40..48).map(|i| i as f32 * 0.01).collect();
        assert_slices_approx(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_matches_scalar_dim4() {
        let table = make_table(16, 4);
        let mut neon_out = vec![0.0f32; 4];
        let mut scalar_out = vec![0.0f32; 4];
        for tid in 0..16 {
            unsafe { neon_embedding_lookup(&table, 16, 4, tid, &mut neon_out) };
            scalar_embedding_lookup(&table, 16, 4, tid, &mut scalar_out);
            assert_slices_approx(&neon_out, &scalar_out, 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_matches_scalar_dim8() {
        let table = make_table(10, 8);
        let mut neon_out = vec![0.0f32; 8];
        let mut scalar_out = vec![0.0f32; 8];
        for tid in [0, 3, 7, 9] {
            unsafe { neon_embedding_lookup(&table, 10, 8, tid, &mut neon_out) };
            scalar_embedding_lookup(&table, 10, 8, tid, &mut scalar_out);
            assert_slices_approx(&neon_out, &scalar_out, 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_matches_scalar_dim5_remainder() {
        let table = make_table(4, 5);
        let mut neon_out = vec![0.0f32; 5];
        let mut scalar_out = vec![0.0f32; 5];
        for tid in 0..4 {
            unsafe { neon_embedding_lookup(&table, 4, 5, tid, &mut neon_out) };
            scalar_embedding_lookup(&table, 4, 5, tid, &mut scalar_out);
            assert_slices_approx(&neon_out, &scalar_out, 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_matches_scalar_dim1() {
        let table = make_table(3, 1);
        let mut neon_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        for tid in 0..3 {
            unsafe { neon_embedding_lookup(&table, 3, 1, tid, &mut neon_out) };
            scalar_embedding_lookup(&table, 3, 1, tid, &mut scalar_out);
            assert_slices_approx(&neon_out, &scalar_out, 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_matches_scalar_dim3() {
        let table = make_table(6, 3);
        let mut neon_out = vec![0.0f32; 3];
        let mut scalar_out = vec![0.0f32; 3];
        for tid in 0..6 {
            unsafe { neon_embedding_lookup(&table, 6, 3, tid, &mut neon_out) };
            scalar_embedding_lookup(&table, 6, 3, tid, &mut scalar_out);
            assert_slices_approx(&neon_out, &scalar_out, 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_matches_scalar_dim16() {
        let table = make_table(4, 16);
        let mut neon_out = vec![0.0f32; 16];
        let mut scalar_out = vec![0.0f32; 16];
        for tid in 0..4 {
            unsafe { neon_embedding_lookup(&table, 4, 16, tid, &mut neon_out) };
            scalar_embedding_lookup(&table, 4, 16, tid, &mut scalar_out);
            assert_slices_approx(&neon_out, &scalar_out, 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_output_overwrite() {
        let table = make_table(4, 8);
        let mut out = vec![999.0f32; 8];
        unsafe { neon_embedding_lookup(&table, 4, 8, 0, &mut out) };
        let expected: Vec<f32> = (0..8).map(|i| i as f32 * 0.01).collect();
        assert_slices_approx(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_larger_output_buffer() {
        let table = make_table(4, 4);
        let mut out = vec![0.0f32; 8];
        unsafe { neon_embedding_lookup(&table, 4, 4, 2, &mut out) };
        let expected: Vec<f32> = (8..12).map(|i| i as f32 * 0.01).collect();
        assert_slices_approx(&out[..4], &expected, 1e-6);
    }

    #[test]
    fn test_scalar_lookup_deterministic() {
        let table = make_table(16, 32);
        let mut out1 = vec![0.0f32; 32];
        let mut out2 = vec![0.0f32; 32];
        scalar_embedding_lookup(&table, 16, 32, 11, &mut out1);
        scalar_embedding_lookup(&table, 16, 32, 11, &mut out2);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_scalar_lookup_distinct_rows() {
        let table = make_table(4, 4);
        let mut out0 = vec![0.0f32; 4];
        let mut out1 = vec![0.0f32; 4];
        scalar_embedding_lookup(&table, 4, 4, 0, &mut out0);
        scalar_embedding_lookup(&table, 4, 4, 1, &mut out1);
        assert_ne!(out0, out1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_distinct_rows() {
        let table = make_table(4, 8);
        let mut out0 = vec![0.0f32; 8];
        let mut out1 = vec![0.0f32; 8];
        unsafe { neon_embedding_lookup(&table, 4, 8, 0, &mut out0) };
        unsafe { neon_embedding_lookup(&table, 4, 8, 1, &mut out1) };
        assert_ne!(out0, out1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_all_rows_covered() {
        let vs = 5;
        let dim = 4;
        let table = make_table(vs, dim);
        for tid in 0..vs {
            let mut neon_out = vec![0.0f32; dim];
            let mut scalar_out = vec![0.0f32; dim];
            unsafe { neon_embedding_lookup(&table, vs, dim, tid, &mut neon_out) };
            scalar_embedding_lookup(&table, vs, dim, tid, &mut scalar_out);
            assert_eq!(neon_out, scalar_out, "mismatch at tid={tid}");
        }
    }

    // -----------------------------------------------------------------------
    // Various embedding dimensions (15 tests)
    // -----------------------------------------------------------------------

    macro_rules! dim_test {
        ($name:ident, $dim:expr) => {
            #[test]
            #[cfg(target_arch = "aarch64")]
            fn $name() {
                let vs = 32;
                let dim = $dim;
                let table = make_table(vs, dim);
                let mut neon_out = vec![0.0f32; dim];
                let mut scalar_out = vec![0.0f32; dim];
                for tid in [0, 1, vs / 2, vs - 1] {
                    unsafe { neon_embedding_lookup(&table, vs, dim, tid, &mut neon_out) };
                    scalar_embedding_lookup(&table, vs, dim, tid, &mut scalar_out);
                    assert_slices_approx(&neon_out, &scalar_out, 1e-5);
                }
            }
        };
    }

    dim_test!(test_dim_64, 64);
    dim_test!(test_dim_128, 128);
    dim_test!(test_dim_256, 256);
    dim_test!(test_dim_512, 512);
    dim_test!(test_dim_768, 768);
    dim_test!(test_dim_1024, 1024);
    dim_test!(test_dim_2048, 2048);

    // Non-power-of-two dimensions (remainder path exercised)
    dim_test!(test_dim_65, 65);
    dim_test!(test_dim_127, 127);
    dim_test!(test_dim_255, 255);
    dim_test!(test_dim_513, 513);
    dim_test!(test_dim_769, 769);
    dim_test!(test_dim_1023, 1023);
    dim_test!(test_dim_2047, 2047);
    dim_test!(test_dim_7, 7);

    // -----------------------------------------------------------------------
    // Batched lookup with various batch sizes (12 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_batched_basic() {
        let table = make_table(8, 4);
        let ids = [0usize, 3, 7];
        let mut out = vec![0.0f32; 12];
        scalar_embedding_lookup_batched(&table, 8, 4, &ids, &mut out);
        let mut expected = vec![0.0f32; 12];
        for (i, &tid) in ids.iter().enumerate() {
            scalar_embedding_lookup(&table, 8, 4, tid, &mut expected[i * 4..]);
        }
        assert_eq!(out, expected);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_matches_scalar_batch1() {
        let table = make_table(16, 8);
        let ids = [5usize];
        let mut neon_out = vec![0.0f32; 8];
        let mut scalar_out = vec![0.0f32; 8];
        unsafe { neon_embedding_lookup_batched(&table, 16, 8, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, 16, 8, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_matches_scalar_batch4() {
        let table = make_table(16, 8);
        let ids = [0usize, 5, 10, 15];
        let mut neon_out = vec![0.0f32; 32];
        let mut scalar_out = vec![0.0f32; 32];
        unsafe { neon_embedding_lookup_batched(&table, 16, 8, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, 16, 8, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_matches_scalar_batch8() {
        let table = make_table(32, 16);
        let ids = [0usize, 4, 8, 12, 16, 20, 24, 28];
        let total = ids.len() * 16;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_lookup_batched(&table, 32, 16, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, 32, 16, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_matches_scalar_batch16() {
        let table = make_table(64, 32);
        let ids: Vec<usize> = (0..16).map(|i| i * 4).collect();
        let total = ids.len() * 32;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_lookup_batched(&table, 64, 32, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, 64, 32, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_matches_scalar_batch32() {
        let table = make_table(64, 16);
        let ids: Vec<usize> = (0..32).map(|i| i * 2).collect();
        let total = ids.len() * 16;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_lookup_batched(&table, 64, 16, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, 64, 16, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_duplicate_ids() {
        let table = make_table(8, 4);
        let ids = [3usize, 3, 3, 3];
        let total = ids.len() * 4;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_lookup_batched(&table, 8, 4, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, 8, 4, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
        // All rows should be identical.
        for i in 1..4 {
            assert_eq!(&neon_out[0..4], &neon_out[i * 4..(i + 1) * 4]);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_reverse_order() {
        let table = make_table(8, 4);
        let ids = [7usize, 6, 5, 4, 3, 2, 1, 0];
        let total = ids.len() * 4;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_lookup_batched(&table, 8, 4, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, 8, 4, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_odd_dim() {
        let table = make_table(8, 7);
        let ids = [1usize, 3, 5];
        let total = ids.len() * 7;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_lookup_batched(&table, 8, 7, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, 8, 7, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    fn test_scalar_batched_single_token() {
        let table = make_table(4, 4);
        let ids = [2usize];
        let mut out = vec![0.0f32; 4];
        scalar_embedding_lookup_batched(&table, 4, 4, &ids, &mut out);
        let expected: Vec<f32> = (8..12).map(|i| i as f32 * 0.01).collect();
        assert_slices_approx(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_large_batch_64() {
        let vs = 128;
        let dim = 64;
        let table = make_table(vs, dim);
        let ids: Vec<usize> = (0..64).map(|i| i * 2).collect();
        let total = ids.len() * dim;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_lookup_batched(&table, vs, dim, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, vs, dim, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Position embedding addition (11 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_add_position_basic() {
        let tok = vec![1.0f32, 2.0, 3.0, 4.0];
        let pos = vec![0.1f32, 0.2, 0.3, 0.4];
        let mut out = vec![0.0f32; 4];
        scalar_embedding_add_position(&tok, &pos, &mut out, 1, 4);
        assert_slices_approx(&out, &[1.1, 2.2, 3.3, 4.4], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_position_matches_scalar_dim4() {
        let tok: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let pos: Vec<f32> = (0..4).map(|i| i as f32 * 0.5).collect();
        let mut neon_out = vec![0.0f32; 4];
        let mut scalar_out = vec![0.0f32; 4];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut neon_out, 1, 4) };
        scalar_embedding_add_position(&tok, &pos, &mut scalar_out, 1, 4);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_position_matches_scalar_dim128_batch4() {
        let batch = 4;
        let dim = 128;
        let total = batch * dim;
        let tok: Vec<f32> = (0..total).map(|i| i as f32 * 0.01).collect();
        let pos: Vec<f32> = (0..total).map(|i| (i as f32 * 0.001) + 1.0).collect();
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut neon_out, batch, dim) };
        scalar_embedding_add_position(&tok, &pos, &mut scalar_out, batch, dim);
        assert_slices_approx(&neon_out, &scalar_out, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_position_zero_pos() {
        let tok = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let pos = vec![0.0f32; 8];
        let mut out = vec![0.0f32; 8];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut out, 2, 4) };
        assert_slices_approx(&out, &tok, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_position_negative_pos() {
        let tok = vec![1.0f32; 8];
        let pos = vec![-0.5f32; 8];
        let mut out = vec![0.0f32; 8];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut out, 2, 4) };
        assert_slices_approx(&out, &[0.5; 8], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_position_remainder_dim5() {
        let batch = 2;
        let dim = 5;
        let total = batch * dim;
        let tok: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let pos: Vec<f32> = (0..total).map(|i| 100.0 + i as f32).collect();
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut neon_out, batch, dim) };
        scalar_embedding_add_position(&tok, &pos, &mut scalar_out, batch, dim);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_position_dim256_batch8() {
        let batch = 8;
        let dim = 256;
        let total = batch * dim;
        let tok: Vec<f32> = (0..total).map(|i| (i % 256) as f32 * 0.001).collect();
        let pos: Vec<f32> = (0..total).map(|i| (i % 256) as f32 * 0.0005).collect();
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut neon_out, batch, dim) };
        scalar_embedding_add_position(&tok, &pos, &mut scalar_out, batch, dim);
        assert_slices_approx(&neon_out, &scalar_out, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_position_dim1() {
        let tok = vec![3.0f32, 7.0];
        let pos = vec![0.5f32, -0.5];
        let mut out = vec![0.0f32; 2];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut out, 2, 1) };
        assert_slices_approx(&out, &[3.5, 6.5], 1e-6);
    }

    #[test]
    fn test_scalar_add_position_batch1_dim1() {
        let tok = vec![42.0f32];
        let pos = vec![0.5f32];
        let mut out = vec![0.0f32; 1];
        scalar_embedding_add_position(&tok, &pos, &mut out, 1, 1);
        assert_slices_approx(&out, &[42.5], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_position_large_values() {
        let tok = vec![1e6f32; 4];
        let pos = vec![1e6f32; 4];
        let mut out = vec![0.0f32; 4];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut out, 1, 4) };
        assert_slices_approx(&out, &[2e6; 4], 1.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_add_position_dim768_batch2() {
        let batch = 2;
        let dim = 768;
        let total = batch * dim;
        let tok: Vec<f32> = (0..total).map(|i| (i as f32).sin()).collect();
        let pos: Vec<f32> = (0..total).map(|i| (i as f32).cos()).collect();
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut neon_out, batch, dim) };
        scalar_embedding_add_position(&tok, &pos, &mut scalar_out, batch, dim);
        assert_slices_approx(&neon_out, &scalar_out, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Normalization correctness (12 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_normalize_unit_vector() {
        let mut data = vec![1.0f32, 0.0, 0.0, 0.0];
        scalar_embedding_normalize(&mut data, 1, 4, 1e-8);
        assert_slices_approx(&data, &[1.0, 0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_scalar_normalize_equal_components() {
        let mut data = vec![1.0f32, 1.0, 1.0, 1.0];
        scalar_embedding_normalize(&mut data, 1, 4, 1e-8);
        let expected = 1.0f32 / 2.0; // 1/sqrt(4) = 0.5
        for &v in &data {
            assert!(approx_eq(v, expected, 1e-6));
        }
    }

    #[test]
    fn test_scalar_normalize_norm_is_one() {
        let mut data = vec![3.0f32, 4.0, 0.0, 0.0];
        scalar_embedding_normalize(&mut data, 1, 4, 1e-8);
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(norm, 1.0, 1e-5));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize_matches_scalar_dim4() {
        let orig = vec![3.0f32, 4.0, 5.0, 6.0];
        let mut neon_data = orig.clone();
        let mut scalar_data = orig;
        unsafe { neon_embedding_normalize(&mut neon_data, 1, 4, 1e-8) };
        scalar_embedding_normalize(&mut scalar_data, 1, 4, 1e-8);
        assert_slices_approx(&neon_data, &scalar_data, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize_matches_scalar_dim128() {
        let dim = 128;
        let orig: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut neon_data = orig.clone();
        let mut scalar_data = orig;
        unsafe { neon_embedding_normalize(&mut neon_data, 1, dim, 1e-8) };
        scalar_embedding_normalize(&mut scalar_data, 1, dim, 1e-8);
        assert_slices_approx(&neon_data, &scalar_data, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize_batch2_dim64() {
        let batch = 2;
        let dim = 64;
        let total = batch * dim;
        let orig: Vec<f32> = (0..total).map(|i| (i as f32 + 0.5) * 0.01).collect();
        let mut neon_data = orig.clone();
        let mut scalar_data = orig;
        unsafe { neon_embedding_normalize(&mut neon_data, batch, dim, 1e-8) };
        scalar_embedding_normalize(&mut scalar_data, batch, dim, 1e-8);
        assert_slices_approx(&neon_data, &scalar_data, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize_norm_is_one() {
        let dim = 32;
        let mut data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0)).collect();
        unsafe { neon_embedding_normalize(&mut data, 1, dim, 1e-8) };
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(norm, 1.0, 1e-5));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize_remainder_dim5() {
        let orig = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut neon_data = orig.clone();
        let mut scalar_data = orig;
        unsafe { neon_embedding_normalize(&mut neon_data, 1, 5, 1e-8) };
        scalar_embedding_normalize(&mut scalar_data, 1, 5, 1e-8);
        assert_slices_approx(&neon_data, &scalar_data, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize_batch4_dim256() {
        let batch = 4;
        let dim = 256;
        let total = batch * dim;
        let orig: Vec<f32> = (0..total).map(|i| ((i % 100) as f32 - 50.0) * 0.1).collect();
        let mut neon_data = orig.clone();
        let mut scalar_data = orig;
        unsafe { neon_embedding_normalize(&mut neon_data, batch, dim, 1e-8) };
        scalar_embedding_normalize(&mut scalar_data, batch, dim, 1e-8);
        assert_slices_approx(&neon_data, &scalar_data, 1e-4);
    }

    #[test]
    fn test_scalar_normalize_near_zero_uses_eps() {
        let mut data = vec![1e-20f32, 0.0, 0.0, 0.0];
        scalar_embedding_normalize(&mut data, 1, 4, 1e-8);
        // Should not produce NaN/Inf.
        for &v in &data {
            assert!(v.is_finite());
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize_near_zero_uses_eps() {
        let mut data = vec![1e-20f32, 0.0, 0.0, 0.0];
        unsafe { neon_embedding_normalize(&mut data, 1, 4, 1e-8) };
        for &v in &data {
            assert!(v.is_finite());
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize_negative_values() {
        let orig = vec![-3.0f32, -4.0, 5.0, 6.0, -1.0, 2.0, -3.0, 4.0];
        let mut neon_data = orig.clone();
        let mut scalar_data = orig;
        unsafe { neon_embedding_normalize(&mut neon_data, 2, 4, 1e-8) };
        scalar_embedding_normalize(&mut scalar_data, 2, 4, 1e-8);
        assert_slices_approx(&neon_data, &scalar_data, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Dot product / similarity (12 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_dot_basic() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![4.0f32, 3.0, 2.0, 1.0];
        let mut out = vec![0.0f32; 1];
        scalar_embedding_dot_product(&a, &b, &mut out, 1, 4);
        assert!(approx_eq(out[0], 20.0, 1e-6));
    }

    #[test]
    fn test_scalar_dot_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let mut out = vec![0.0f32; 1];
        scalar_embedding_dot_product(&a, &b, &mut out, 1, 4);
        assert!(approx_eq(out[0], 0.0, 1e-6));
    }

    #[test]
    fn test_scalar_dot_self() {
        let a = vec![3.0f32, 4.0];
        let mut out = vec![0.0f32; 1];
        scalar_embedding_dot_product(&a, &a, &mut out, 1, 2);
        assert!(approx_eq(out[0], 25.0, 1e-6));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_matches_scalar_dim4() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut neon_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        unsafe { neon_embedding_dot_product(&a, &b, &mut neon_out, 1, 4) };
        scalar_embedding_dot_product(&a, &b, &mut scalar_out, 1, 4);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_matches_scalar_dim128() {
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let mut neon_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        unsafe { neon_embedding_dot_product(&a, &b, &mut neon_out, 1, dim) };
        scalar_embedding_dot_product(&a, &b, &mut scalar_out, 1, dim);
        assert_slices_approx(&neon_out, &scalar_out, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_batch4_dim8() {
        let batch = 4;
        let dim = 8;
        let total = batch * dim;
        let a: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32) * 0.05).collect();
        let mut neon_out = vec![0.0f32; batch];
        let mut scalar_out = vec![0.0f32; batch];
        unsafe { neon_embedding_dot_product(&a, &b, &mut neon_out, batch, dim) };
        scalar_embedding_dot_product(&a, &b, &mut scalar_out, batch, dim);
        assert_slices_approx(&neon_out, &scalar_out, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_remainder_dim5() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
        let mut neon_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        unsafe { neon_embedding_dot_product(&a, &b, &mut neon_out, 1, 5) };
        scalar_embedding_dot_product(&a, &b, &mut scalar_out, 1, 5);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_dim256_batch2() {
        let batch = 2;
        let dim = 256;
        let total = batch * dim;
        let a: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.37).sin()).collect();
        let b: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.53).cos()).collect();
        let mut neon_out = vec![0.0f32; batch];
        let mut scalar_out = vec![0.0f32; batch];
        unsafe { neon_embedding_dot_product(&a, &b, &mut neon_out, batch, dim) };
        scalar_embedding_dot_product(&a, &b, &mut scalar_out, batch, dim);
        assert_slices_approx(&neon_out, &scalar_out, 1e-2);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_dim1() {
        let a = vec![3.0f32, 7.0];
        let b = vec![2.0f32, -1.0];
        let mut neon_out = vec![0.0f32; 2];
        let mut scalar_out = vec![0.0f32; 2];
        unsafe { neon_embedding_dot_product(&a, &b, &mut neon_out, 2, 1) };
        scalar_embedding_dot_product(&a, &b, &mut scalar_out, 2, 1);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_zeros() {
        let a = vec![0.0f32; 16];
        let b: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 2];
        unsafe { neon_embedding_dot_product(&a, &b, &mut out, 2, 8) };
        assert_slices_approx(&out, &[0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_scalar_dot_negative() {
        let a = vec![-1.0f32, -2.0, -3.0, -4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 1];
        scalar_embedding_dot_product(&a, &b, &mut out, 1, 4);
        assert!(approx_eq(out[0], -30.0, 1e-6));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_negative_matches_scalar() {
        let a = vec![-1.0f32, -2.0, -3.0, -4.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut neon_out = vec![0.0f32; 1];
        let mut scalar_out = vec![0.0f32; 1];
        unsafe { neon_embedding_dot_product(&a, &b, &mut neon_out, 1, 4) };
        scalar_embedding_dot_product(&a, &b, &mut scalar_out, 1, 4);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Edge cases (13 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_lookup_vocab1() {
        let table = vec![42.0f32, 43.0];
        let mut out = vec![0.0f32; 2];
        scalar_embedding_lookup(&table, 1, 2, 0, &mut out);
        assert_eq!(&out, &[42.0, 43.0]);
    }

    #[test]
    fn test_scalar_lookup_dim1_vocab1() {
        let table = vec![7.0f32];
        let mut out = vec![0.0f32; 1];
        scalar_embedding_lookup(&table, 1, 1, 0, &mut out);
        assert_eq!(out[0], 7.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_vocab1() {
        let table = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; 4];
        unsafe { neon_embedding_lookup(&table, 1, 4, 0, &mut out) };
        assert_eq!(&out, &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_scalar_batched_empty() {
        let table = make_table(8, 4);
        let ids: &[usize] = &[];
        let mut out: Vec<f32> = vec![];
        scalar_embedding_lookup_batched(&table, 8, 4, ids, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_batched_empty() {
        let table = make_table(8, 4);
        let ids: &[usize] = &[];
        let mut out: Vec<f32> = vec![];
        unsafe { neon_embedding_lookup_batched(&table, 8, 4, ids, &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scalar_lookup_oob() {
        let table = make_table(4, 4);
        let mut out = vec![0.0f32; 4];
        scalar_embedding_lookup(&table, 4, 4, 4, &mut out);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scalar_lookup_oob_large() {
        let table = make_table(4, 4);
        let mut out = vec![0.0f32; 4];
        scalar_embedding_lookup(&table, 4, 4, 100, &mut out);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_lookup_oob() {
        let table = make_table(4, 4);
        let mut out = vec![0.0f32; 4];
        unsafe { neon_embedding_lookup(&table, 4, 4, 4, &mut out) };
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scalar_gather_scatter_oob() {
        let table = make_table(4, 4);
        let indices = [0usize, 5]; // 5 is OOB
        let mut out = vec![0.0f32; 8];
        scalar_embedding_gather_scatter(&table, 4, 4, &indices, &mut out);
    }

    #[test]
    fn test_scalar_normalize_dim1() {
        let mut data = vec![5.0f32];
        scalar_embedding_normalize(&mut data, 1, 1, 1e-8);
        assert!(approx_eq(data[0], 1.0, 1e-5));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_normalize_dim1() {
        let mut data = vec![5.0f32];
        unsafe { neon_embedding_normalize(&mut data, 1, 1, 1e-8) };
        assert!(approx_eq(data[0], 1.0, 1e-5));
    }

    #[test]
    fn test_scalar_dot_dim1_batch1() {
        let a = vec![3.0f32];
        let b = vec![4.0f32];
        let mut out = vec![0.0f32; 1];
        scalar_embedding_dot_product(&a, &b, &mut out, 1, 1);
        assert!(approx_eq(out[0], 12.0, 1e-6));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_gather_scatter_oob_panics() {
        let result = std::panic::catch_unwind(|| {
            let table = make_table(4, 4);
            let indices = [0usize, 10];
            let mut out = vec![0.0f32; 8];
            unsafe { neon_embedding_gather_scatter(&table, 4, 4, &indices, &mut out) };
        });
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Gather / scatter (6 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_gather_scatter_basic() {
        let table = make_table(8, 4);
        let indices = [1usize, 3, 5];
        let mut out = vec![0.0f32; 12];
        scalar_embedding_gather_scatter(&table, 8, 4, &indices, &mut out);
        let mut expected = vec![0.0f32; 12];
        scalar_embedding_lookup_batched(&table, 8, 4, &indices, &mut expected);
        assert_eq!(out, expected);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_gather_scatter_matches_scalar() {
        let table = make_table(16, 8);
        let indices = [0usize, 7, 15, 3];
        let total = indices.len() * 8;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_gather_scatter(&table, 16, 8, &indices, &mut neon_out) };
        scalar_embedding_gather_scatter(&table, 16, 8, &indices, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_gather_scatter_single() {
        let table = make_table(4, 4);
        let indices = [2usize];
        let mut out = vec![0.0f32; 4];
        unsafe { neon_embedding_gather_scatter(&table, 4, 4, &indices, &mut out) };
        let expected: Vec<f32> = (8..12).map(|i| i as f32 * 0.01).collect();
        assert_slices_approx(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_gather_scatter_odd_dim() {
        let table = make_table(8, 7);
        let indices = [0usize, 4, 7];
        let total = indices.len() * 7;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_gather_scatter(&table, 8, 7, &indices, &mut neon_out) };
        scalar_embedding_gather_scatter(&table, 8, 7, &indices, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-6);
    }

    #[test]
    fn test_scalar_gather_scatter_empty_indices() {
        let table = make_table(4, 4);
        let indices: &[usize] = &[];
        let mut out: Vec<f32> = vec![];
        scalar_embedding_gather_scatter(&table, 4, 4, indices, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_gather_scatter_empty_indices() {
        let table = make_table(4, 4);
        let indices: &[usize] = &[];
        let mut out: Vec<f32> = vec![];
        unsafe { neon_embedding_gather_scatter(&table, 4, 4, indices, &mut out) };
        assert!(out.is_empty());
    }

    // -----------------------------------------------------------------------
    // Large vocab stress tests (10 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_large_vocab_lookup() {
        let vs = 50_000;
        let dim = 64;
        let table = make_table(vs, dim);
        let mut out = vec![0.0f32; dim];
        scalar_embedding_lookup(&table, vs, dim, vs - 1, &mut out);
        let start = (vs - 1) * dim;
        let expected: Vec<f32> = (start..start + dim).map(|i| i as f32 * 0.01).collect();
        assert_slices_approx(&out, &expected, 1e-2);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_large_vocab_lookup() {
        let vs = 50_000;
        let dim = 64;
        let table = make_table(vs, dim);
        let mut neon_out = vec![0.0f32; dim];
        let mut scalar_out = vec![0.0f32; dim];
        unsafe { neon_embedding_lookup(&table, vs, dim, vs - 1, &mut neon_out) };
        scalar_embedding_lookup(&table, vs, dim, vs - 1, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-2);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_large_vocab_batched() {
        let vs = 32_000;
        let dim = 128;
        let table = make_table(vs, dim);
        let ids: Vec<usize> = (0..32).map(|i| i * 1000).collect();
        let total = ids.len() * dim;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_lookup_batched(&table, vs, dim, &ids, &mut neon_out) };
        scalar_embedding_lookup_batched(&table, vs, dim, &ids, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_large_dim_2048_lookup() {
        let vs = 100;
        let dim = 2048;
        let table = make_table(vs, dim);
        let mut neon_out = vec![0.0f32; dim];
        let mut scalar_out = vec![0.0f32; dim];
        unsafe { neon_embedding_lookup(&table, vs, dim, 50, &mut neon_out) };
        scalar_embedding_lookup(&table, vs, dim, 50, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_large_vocab_normalize() {
        let batch = 16;
        let dim = 512;
        let total = batch * dim;
        let orig: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.017).sin()).collect();
        let mut neon_data = orig.clone();
        let mut scalar_data = orig;
        unsafe { neon_embedding_normalize(&mut neon_data, batch, dim, 1e-8) };
        scalar_embedding_normalize(&mut scalar_data, batch, dim, 1e-8);
        assert_slices_approx(&neon_data, &scalar_data, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_large_vocab_dot() {
        let batch = 16;
        let dim = 512;
        let total = batch * dim;
        let a: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.013).cos()).collect();
        let b: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.019).sin()).collect();
        let mut neon_out = vec![0.0f32; batch];
        let mut scalar_out = vec![0.0f32; batch];
        unsafe { neon_embedding_dot_product(&a, &b, &mut neon_out, batch, dim) };
        scalar_embedding_dot_product(&a, &b, &mut scalar_out, batch, dim);
        assert_slices_approx(&neon_out, &scalar_out, 1e-1);
    }

    #[test]
    fn test_scalar_large_vocab_batched() {
        let vs = 10_000;
        let dim = 32;
        let table = make_table(vs, dim);
        let ids: Vec<usize> = (0..16).map(|i| i * 625).collect();
        let total = ids.len() * dim;
        let mut out = vec![0.0f32; total];
        scalar_embedding_lookup_batched(&table, vs, dim, &ids, &mut out);
        // Verify first embedding.
        let expected_start = ids[0] * dim;
        let expected: Vec<f32> =
            (expected_start..expected_start + dim).map(|i| i as f32 * 0.01).collect();
        assert_slices_approx(&out[..dim], &expected, 1e-2);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_large_vocab_gather_scatter() {
        let vs = 10_000;
        let dim = 64;
        let table = make_table(vs, dim);
        let indices: Vec<usize> = (0..20).map(|i| i * 500).collect();
        let total = indices.len() * dim;
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_gather_scatter(&table, vs, dim, &indices, &mut neon_out) };
        scalar_embedding_gather_scatter(&table, vs, dim, &indices, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1e-1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_large_add_position() {
        let batch = 32;
        let dim = 1024;
        let total = batch * dim;
        let tok: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.003).sin()).collect();
        let pos: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.007).cos()).collect();
        let mut neon_out = vec![0.0f32; total];
        let mut scalar_out = vec![0.0f32; total];
        unsafe { neon_embedding_add_position(&tok, &pos, &mut neon_out, batch, dim) };
        scalar_embedding_add_position(&tok, &pos, &mut scalar_out, batch, dim);
        assert_slices_approx(&neon_out, &scalar_out, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_large_vocab_100k_dim256() {
        let vs = 100_000;
        let dim = 256;
        let table = make_table(vs, dim);
        let tid = 99_999;
        let mut neon_out = vec![0.0f32; dim];
        let mut scalar_out = vec![0.0f32; dim];
        unsafe { neon_embedding_lookup(&table, vs, dim, tid, &mut neon_out) };
        scalar_embedding_lookup(&table, vs, dim, tid, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, 1.0);
    }
}
