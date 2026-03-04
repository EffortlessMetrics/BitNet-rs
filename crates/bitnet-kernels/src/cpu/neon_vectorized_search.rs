#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON optimized vectorized search and similarity operations for Apple Silicon.
//!
//! Provides SIMD-accelerated nearest-neighbor search and distance/similarity
//! metrics for token embeddings using `float32x4` NEON intrinsics. All public
//! functions include scalar tail handling for vectors whose length is not a
//! multiple of 4.

#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Horizontal helpers ──────────────────────────────────────────────────

/// Horizontal sum of a `float32x4_t` → scalar f32.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn hsum_f32x4(v: float32x4_t) -> f32 {
    // vaddvq_f32 is a single-instruction horizontal add on AArch64.
    unsafe { vaddvq_f32(v) }
}

// ── Dot product ─────────────────────────────────────────────────────────

/// NEON-accelerated dot product of two `f32` slices.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn neon_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slice lengths must match");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let chunks = n / LANES;
    let remainder = n % LANES;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * LANES;
        let va = unsafe { vld1q_f32(a.as_ptr().add(off)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(off)) };
        acc = vfmaq_f32(acc, va, vb);
    }
    let mut sum = hsum_f32x4(acc);

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        sum += a[tail_start + i] * b[tail_start + i];
    }
    sum
}

// ── Cosine similarity ───────────────────────────────────────────────────

/// NEON-accelerated cosine similarity: `dot(a,b) / (‖a‖ · ‖b‖)`.
///
/// Returns `0.0` for zero-length or zero-norm vectors.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn neon_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slice lengths must match");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let chunks = n / LANES;
    let remainder = n % LANES;

    let mut acc_dot = vdupq_n_f32(0.0);
    let mut acc_a2 = vdupq_n_f32(0.0);
    let mut acc_b2 = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * LANES;
        let va = unsafe { vld1q_f32(a.as_ptr().add(off)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(off)) };
        acc_dot = vfmaq_f32(acc_dot, va, vb);
        acc_a2 = vfmaq_f32(acc_a2, va, va);
        acc_b2 = vfmaq_f32(acc_b2, vb, vb);
    }

    let mut dot = hsum_f32x4(acc_dot);
    let mut norm_a2 = hsum_f32x4(acc_a2);
    let mut norm_b2 = hsum_f32x4(acc_b2);

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let ai = a[tail_start + i];
        let bi = b[tail_start + i];
        dot += ai * bi;
        norm_a2 += ai * ai;
        norm_b2 += bi * bi;
    }

    let denom = (norm_a2 * norm_b2).sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

// ── Euclidean (L2) distance ─────────────────────────────────────────────

/// NEON-accelerated Euclidean (L2) distance.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn neon_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slice lengths must match");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let chunks = n / LANES;
    let remainder = n % LANES;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * LANES;
        let va = unsafe { vld1q_f32(a.as_ptr().add(off)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(off)) };
        let diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }
    let mut sum = hsum_f32x4(acc);

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let d = a[tail_start + i] - b[tail_start + i];
        sum += d * d;
    }
    sum.sqrt()
}

// ── Manhattan (L1) distance ─────────────────────────────────────────────

/// NEON-accelerated Manhattan (L1) distance.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn neon_manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slice lengths must match");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let chunks = n / LANES;
    let remainder = n % LANES;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * LANES;
        let va = unsafe { vld1q_f32(a.as_ptr().add(off)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(off)) };
        let diff = vsubq_f32(va, vb);
        let abs_diff = vabsq_f32(diff);
        acc = vaddq_f32(acc, abs_diff);
    }
    let mut sum = hsum_f32x4(acc);

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        sum += (a[tail_start + i] - b[tail_start + i]).abs();
    }
    sum
}

// ── Top-k similar ───────────────────────────────────────────────────────

/// Find the `k` most similar embeddings to `query` by cosine similarity.
///
/// Returns `Vec<(index, similarity)>` sorted descending by similarity.
/// If `k == 0` or `embeddings` is empty, returns an empty `Vec`.
/// If `k > embeddings.len()`, returns all embeddings sorted.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn neon_top_k_similar(query: &[f32], embeddings: &[&[f32]], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || embeddings.is_empty() {
        return Vec::new();
    }

    let sims = neon_batch_cosine_similarity(query, embeddings);

    let mut indexed: Vec<(usize, f32)> = sims.into_iter().enumerate().collect();
    // Sort descending by similarity (use total_cmp for NaN safety).
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);
    indexed
}

// ── Batch cosine similarity ─────────────────────────────────────────────

/// Compute cosine similarity of `query` against each embedding in the batch.
///
/// Returns a `Vec<f32>` of similarities, one per embedding.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn neon_batch_cosine_similarity(query: &[f32], embeddings: &[&[f32]]) -> Vec<f32> {
    embeddings.iter().map(|emb| neon_cosine_similarity(query, emb)).collect()
}

// ── L2 normalization ────────────────────────────────────────────────────

/// In-place L2 normalization. Zero vectors are left unchanged.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn neon_normalize_l2(vec: &mut [f32]) {
    let n = vec.len();
    if n == 0 {
        return;
    }

    let chunks = n / LANES;
    let remainder = n % LANES;

    // Accumulate squared norm.
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * LANES;
        let v = unsafe { vld1q_f32(vec.as_ptr().add(off)) };
        acc = vfmaq_f32(acc, v, v);
    }
    let mut norm2 = hsum_f32x4(acc);
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        norm2 += vec[tail_start + i] * vec[tail_start + i];
    }

    let norm = norm2.sqrt();
    if norm == 0.0 {
        return;
    }

    // Divide by norm.
    let inv_norm = 1.0 / norm;
    let inv_v = vdupq_n_f32(inv_norm);
    for i in 0..chunks {
        let off = i * LANES;
        let v = unsafe { vld1q_f32(vec.as_ptr().add(off)) };
        let scaled = vmulq_f32(v, inv_v);
        unsafe { vst1q_f32(vec.as_mut_ptr().add(off), scaled) };
    }
    for i in 0..remainder {
        vec[tail_start + i] *= inv_norm;
    }
}

// ── Hamming distance (binary embeddings) ────────────────────────────────

/// NEON-accelerated Hamming distance on byte slices (counts differing bits).
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn neon_hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "slice lengths must match");
    let n = a.len();
    if n == 0 {
        return 0;
    }

    const BYTE_LANES: usize = 16;
    let chunks = n / BYTE_LANES;
    let remainder = n % BYTE_LANES;

    let mut total: u32 = 0;

    for i in 0..chunks {
        let off = i * BYTE_LANES;
        let va = unsafe { vld1q_u8(a.as_ptr().add(off)) };
        let vb = unsafe { vld1q_u8(b.as_ptr().add(off)) };
        let xor = veorq_u8(va, vb);
        // vcntq_u8 counts set bits per byte.
        let bits = vcntq_u8(xor);
        // Horizontal sum via widening add.
        total += vaddlvq_u8(bits) as u32;
    }

    // Scalar tail.
    let tail_start = chunks * BYTE_LANES;
    for i in 0..remainder {
        total += (a[tail_start + i] ^ b[tail_start + i]).count_ones();
    }
    total
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── dot product ─────────────────────────────────────────────────────

    #[test]
    fn test_dot_product_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        // 5 + 8 + 9 + 8 + 5 = 35
        assert!(approx_eq(neon_dot_product(&a, &b), 35.0));
    }

    #[test]
    fn test_dot_product_empty() {
        assert_eq!(neon_dot_product(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_product_single() {
        assert!(approx_eq(neon_dot_product(&[3.0], &[4.0]), 12.0));
    }

    #[test]
    fn test_dot_product_exact_lanes() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        assert!(approx_eq(neon_dot_product(&a, &b), 20.0));
    }

    #[test]
    fn test_dot_product_non_aligned() {
        let a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let b = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        assert!(approx_eq(neon_dot_product(&a, &b), 14.0));
    }

    #[test]
    #[should_panic(expected = "slice lengths must match")]
    fn test_dot_product_mismatched() {
        neon_dot_product(&[1.0, 2.0], &[1.0]);
    }

    // ── cosine similarity ───────────────────────────────────────────────

    #[test]
    fn test_cosine_identical() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(neon_cosine_similarity(&v, &v), 1.0));
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        assert!(approx_eq(neon_cosine_similarity(&a, &b), 0.0));
    }

    #[test]
    fn test_cosine_opposite() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [-1.0, -2.0, -3.0, -4.0];
        assert!(approx_eq(neon_cosine_similarity(&a, &b), -1.0));
    }

    #[test]
    fn test_cosine_empty() {
        assert_eq!(neon_cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(neon_cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_single() {
        assert!(approx_eq(neon_cosine_similarity(&[3.0], &[4.0]), 1.0));
    }

    #[test]
    fn test_cosine_non_aligned() {
        let a = [1.0, 0.0, 1.0, 0.0, 1.0];
        let b = [0.0, 1.0, 0.0, 1.0, 0.0];
        assert!(approx_eq(neon_cosine_similarity(&a, &b), 0.0));
    }

    // ── euclidean distance ──────────────────────────────────────────────

    #[test]
    fn test_euclidean_basic() {
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0, 0.0];
        assert!(approx_eq(neon_euclidean_distance(&a, &b), 5.0));
    }

    #[test]
    fn test_euclidean_identical() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(neon_euclidean_distance(&v, &v), 0.0));
    }

    #[test]
    fn test_euclidean_empty() {
        assert_eq!(neon_euclidean_distance(&[], &[]), 0.0);
    }

    #[test]
    fn test_euclidean_single() {
        assert!(approx_eq(neon_euclidean_distance(&[1.0], &[4.0]), 3.0));
    }

    #[test]
    fn test_euclidean_non_aligned() {
        // 5 elements: sqrt(1+1+1+1+1) = sqrt(5) ≈ 2.2360679
        let a = [0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0];
        assert!(approx_eq(neon_euclidean_distance(&a, &b), 5.0_f32.sqrt()));
    }

    // ── manhattan distance ──────────────────────────────────────────────

    #[test]
    fn test_manhattan_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 2.0, 0.0, 1.0];
        // |3| + |0| + |3| + |3| = 9
        assert!(approx_eq(neon_manhattan_distance(&a, &b), 9.0));
    }

    #[test]
    fn test_manhattan_identical() {
        let v = [5.0, 6.0, 7.0, 8.0];
        assert!(approx_eq(neon_manhattan_distance(&v, &v), 0.0));
    }

    #[test]
    fn test_manhattan_empty() {
        assert_eq!(neon_manhattan_distance(&[], &[]), 0.0);
    }

    #[test]
    fn test_manhattan_single() {
        assert!(approx_eq(neon_manhattan_distance(&[-3.0], &[4.0]), 7.0));
    }

    #[test]
    fn test_manhattan_non_aligned() {
        let a = [0.0; 7];
        let b = [1.0; 7];
        assert!(approx_eq(neon_manhattan_distance(&a, &b), 7.0));
    }

    #[test]
    fn test_manhattan_negative_values() {
        let a = [-1.0, -2.0, -3.0, -4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(neon_manhattan_distance(&a, &b), 20.0));
    }

    // ── top-k similar ───────────────────────────────────────────────────

    #[test]
    fn test_top_k_basic() {
        let query = [1.0, 0.0, 0.0, 0.0];
        let e0: &[f32] = &[1.0, 0.0, 0.0, 0.0]; // sim = 1.0
        let e1: &[f32] = &[0.0, 1.0, 0.0, 0.0]; // sim = 0.0
        let e2: &[f32] = &[0.5, 0.5, 0.0, 0.0]; // sim ≈ 0.707
        let embeddings: Vec<&[f32]> = vec![e0, e1, e2];
        let result = neon_top_k_similar(&query, &embeddings, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0); // most similar
        assert_eq!(result[1].0, 2); // second
    }

    #[test]
    fn test_top_k_zero() {
        let query = [1.0, 0.0, 0.0, 0.0];
        let e0: &[f32] = &[1.0, 0.0, 0.0, 0.0];
        let embeddings: Vec<&[f32]> = vec![e0];
        let result = neon_top_k_similar(&query, &embeddings, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_k_exceeds_len() {
        let query = [1.0, 0.0, 0.0, 0.0];
        let e0: &[f32] = &[1.0, 0.0, 0.0, 0.0];
        let e1: &[f32] = &[0.0, 1.0, 0.0, 0.0];
        let embeddings: Vec<&[f32]> = vec![e0, e1];
        let result = neon_top_k_similar(&query, &embeddings, 100);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_top_k_empty_embeddings() {
        let query = [1.0, 0.0, 0.0, 0.0];
        let embeddings: Vec<&[f32]> = vec![];
        let result = neon_top_k_similar(&query, &embeddings, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_k_identical_embeddings() {
        let query = [1.0, 1.0, 1.0, 1.0];
        let e: &[f32] = &[1.0, 1.0, 1.0, 1.0];
        let embeddings: Vec<&[f32]> = vec![e, e, e];
        let result = neon_top_k_similar(&query, &embeddings, 2);
        assert_eq!(result.len(), 2);
        // All similarities should be 1.0.
        for &(_, sim) in &result {
            assert!(approx_eq(sim, 1.0));
        }
    }

    // ── batch cosine similarity ─────────────────────────────────────────

    #[test]
    fn test_batch_cosine_basic() {
        let query = [1.0, 0.0, 0.0, 0.0];
        let e0: &[f32] = &[1.0, 0.0, 0.0, 0.0];
        let e1: &[f32] = &[0.0, 1.0, 0.0, 0.0];
        let embeddings: Vec<&[f32]> = vec![e0, e1];
        let sims = neon_batch_cosine_similarity(&query, &embeddings);
        assert_eq!(sims.len(), 2);
        assert!(approx_eq(sims[0], 1.0));
        assert!(approx_eq(sims[1], 0.0));
    }

    #[test]
    fn test_batch_cosine_empty() {
        let query = [1.0, 0.0, 0.0, 0.0];
        let embeddings: Vec<&[f32]> = vec![];
        let sims = neon_batch_cosine_similarity(&query, &embeddings);
        assert!(sims.is_empty());
    }

    #[test]
    fn test_batch_cosine_non_aligned() {
        let query = [1.0, 1.0, 1.0, 1.0, 1.0];
        let e0: &[f32] = &[1.0, 1.0, 1.0, 1.0, 1.0];
        let e1: &[f32] = &[-1.0, -1.0, -1.0, -1.0, -1.0];
        let embeddings: Vec<&[f32]> = vec![e0, e1];
        let sims = neon_batch_cosine_similarity(&query, &embeddings);
        assert!(approx_eq(sims[0], 1.0));
        assert!(approx_eq(sims[1], -1.0));
    }

    // ── L2 normalization ────────────────────────────────────────────────

    #[test]
    fn test_normalize_l2_basic() {
        let mut v = [3.0, 4.0, 0.0, 0.0];
        neon_normalize_l2(&mut v);
        assert!(approx_eq(v[0], 0.6));
        assert!(approx_eq(v[1], 0.8));
        assert!(approx_eq(v[2], 0.0));
        assert!(approx_eq(v[3], 0.0));
    }

    #[test]
    fn test_normalize_l2_unit_vector() {
        let mut v = [1.0, 0.0, 0.0, 0.0];
        neon_normalize_l2(&mut v);
        assert!(approx_eq(v[0], 1.0));
        assert!(approx_eq(v[1], 0.0));
    }

    #[test]
    fn test_normalize_l2_zero_vector() {
        let mut v = [0.0, 0.0, 0.0, 0.0];
        neon_normalize_l2(&mut v);
        // Should remain zero — no division by zero.
        for &x in &v {
            assert_eq!(x, 0.0);
        }
    }

    #[test]
    fn test_normalize_l2_empty() {
        let mut v: Vec<f32> = vec![];
        neon_normalize_l2(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn test_normalize_l2_negative() {
        let mut v = [-3.0, -4.0, 0.0, 0.0];
        neon_normalize_l2(&mut v);
        assert!(approx_eq(v[0], -0.6));
        assert!(approx_eq(v[1], -0.8));
    }

    #[test]
    fn test_normalize_l2_non_aligned() {
        let mut v = [1.0, 1.0, 1.0, 1.0, 1.0]; // norm = sqrt(5)
        neon_normalize_l2(&mut v);
        let expected = 1.0 / 5.0_f32.sqrt();
        for &x in &v {
            assert!(approx_eq(x, expected));
        }
    }

    #[test]
    fn test_normalize_l2_single() {
        let mut v = [5.0];
        neon_normalize_l2(&mut v);
        assert!(approx_eq(v[0], 1.0));
    }

    #[test]
    fn test_normalize_l2_result_is_unit() {
        let mut v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        neon_normalize_l2(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(norm, 1.0));
    }

    // ── hamming distance ────────────────────────────────────────────────

    #[test]
    fn test_hamming_basic() {
        let a = [0b1111_0000u8, 0b0000_1111];
        let b = [0b0000_0000u8, 0b0000_0000];
        assert_eq!(neon_hamming_distance(&a, &b), 8);
    }

    #[test]
    fn test_hamming_identical() {
        let v = [0xAA, 0xBB, 0xCC, 0xDD];
        assert_eq!(neon_hamming_distance(&v, &v), 0);
    }

    #[test]
    fn test_hamming_empty() {
        assert_eq!(neon_hamming_distance(&[], &[]), 0);
    }

    #[test]
    fn test_hamming_all_differ() {
        let a = [0xFF; 4];
        let b = [0x00; 4];
        assert_eq!(neon_hamming_distance(&a, &b), 32);
    }

    #[test]
    fn test_hamming_single_byte() {
        // 0b10101010 ^ 0b01010101 = 0b11111111 → 8 bits
        assert_eq!(neon_hamming_distance(&[0xAA], &[0x55]), 8);
    }

    #[test]
    fn test_hamming_16_bytes_exact_lane() {
        let a = [0xFF; 16];
        let b = [0x00; 16];
        assert_eq!(neon_hamming_distance(&a, &b), 128);
    }

    #[test]
    fn test_hamming_non_aligned() {
        // 17 bytes: 16 (NEON) + 1 tail
        let a = [0xFF; 17];
        let b = [0x00; 17];
        assert_eq!(neon_hamming_distance(&a, &b), 136);
    }

    #[test]
    #[should_panic(expected = "slice lengths must match")]
    fn test_hamming_mismatched() {
        neon_hamming_distance(&[0u8; 3], &[0u8; 5]);
    }
}
