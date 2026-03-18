//! ARM NEON attention score computation kernels for Apple Silicon.
//!
//! Provides SIMD-accelerated attention scoring primitives using NEON
//! `float32x4` intrinsics: scaled dot-product, multi-head, sliding
//! window, sparse pattern, cross-attention, clamping, and dropout.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane width for `float32x4_t`.
const LANES: usize = 4;

// ── Scaled dot-product attention ────────────────────────────────────

/// Compute `Q · K^T / sqrt(d_k)` for a single query against multiple
/// key vectors.
///
/// - `query`: shape `[head_dim]`
/// - `keys`: shape `[num_keys * head_dim]` (row-major)
/// - `scores_out`: shape `[num_keys]`
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// Input slices must have the documented minimum lengths.
///
/// # Panics
///
/// Panics if `head_dim` is zero, `keys.len()` is not a multiple of
/// `head_dim`, or `scores_out` is too small.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_scaled_dot_product_attention(
    query: &[f32],
    keys: &[f32],
    head_dim: usize,
    scores_out: &mut [f32],
) {
    assert!(head_dim > 0, "head_dim must be positive");
    assert_eq!(keys.len() % head_dim, 0, "keys length must be a multiple of head_dim");
    let num_keys = keys.len() / head_dim;
    assert!(
        scores_out.len() >= num_keys,
        "scores_out too small: need {num_keys}, got {}",
        scores_out.len()
    );
    assert_eq!(query.len(), head_dim, "query length must equal head_dim");

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_ptr = query.as_ptr();

    for (k, score) in scores_out.iter_mut().enumerate().take(num_keys) {
        let k_ptr = unsafe { keys.as_ptr().add(k * head_dim) };
        let dot = unsafe { neon_dot_f32(q_ptr, k_ptr, head_dim) };
        *score = dot * scale;
    }
}

/// NEON-accelerated dot product of two `f32` slices of length `len`.
///
/// # Safety
///
/// Both `a` and `b` must point to at least `len` readable `f32` values.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_dot_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let va = vld1q_f32(a.add(offset));
            let vb = vld1q_f32(b.add(offset));
            acc = vfmaq_f32(acc, va, vb);
        }
    }

    let mut sum = vaddvq_f32(acc);

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        sum += unsafe { *a.add(tail_start + i) * *b.add(tail_start + i) };
    }

    sum
}

// ── Multi-head attention score ──────────────────────────────────────

/// Compute attention scores for multiple heads in parallel.
///
/// - `queries`: shape `[num_heads * head_dim]` (row-major)
/// - `keys`: shape `[num_heads * num_keys * head_dim]` (row-major,
///   each head's keys are contiguous)
/// - `scores_out`: shape `[num_heads * num_keys]`
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// Input slices must have the documented minimum lengths.
///
/// # Panics
///
/// Panics on dimension mismatches.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_multi_head_attention_scores(
    queries: &[f32],
    keys: &[f32],
    num_heads: usize,
    num_keys: usize,
    head_dim: usize,
    scores_out: &mut [f32],
) {
    assert!(head_dim > 0, "head_dim must be positive");
    assert_eq!(queries.len(), num_heads * head_dim, "queries shape mismatch");
    assert_eq!(keys.len(), num_heads * num_keys * head_dim, "keys shape mismatch");
    assert!(scores_out.len() >= num_heads * num_keys, "scores_out too small");

    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let q_ptr = unsafe { queries.as_ptr().add(h * head_dim) };
        let k_base = unsafe { keys.as_ptr().add(h * num_keys * head_dim) };

        for k in 0..num_keys {
            let k_ptr = unsafe { k_base.add(k * head_dim) };
            let dot = unsafe { neon_dot_f32(q_ptr, k_ptr, head_dim) };
            scores_out[h * num_keys + k] = dot * scale;
        }
    }
}

// ── Sliding window attention ────────────────────────────────────────

/// Compute attention scores within a sliding window. Positions
/// outside `[max(0, pos - window_size + 1) .. pos]` receive
/// `f32::NEG_INFINITY`.
///
/// - `query`: shape `[head_dim]`
/// - `keys`: shape `[seq_len * head_dim]`
/// - `query_pos`: current query position in the sequence
/// - `window_size`: number of positions visible (including self)
/// - `scores_out`: shape `[seq_len]`
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// Input slices must have the documented minimum lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sliding_window_attention(
    query: &[f32],
    keys: &[f32],
    head_dim: usize,
    query_pos: usize,
    window_size: usize,
    scores_out: &mut [f32],
) {
    assert!(head_dim > 0, "head_dim must be positive");
    let seq_len = keys.len() / head_dim;
    assert_eq!(keys.len() % head_dim, 0, "keys length must be a multiple of head_dim");
    assert!(scores_out.len() >= seq_len, "scores_out too small");
    assert_eq!(query.len(), head_dim, "query length mismatch");

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_ptr = query.as_ptr();

    let win_start = (query_pos + 1).saturating_sub(window_size);
    let win_end = (query_pos + 1).min(seq_len);

    for (i, score) in scores_out.iter_mut().enumerate().take(seq_len) {
        if i >= win_start && i < win_end {
            let k_ptr = unsafe { keys.as_ptr().add(i * head_dim) };
            let dot = unsafe { neon_dot_f32(q_ptr, k_ptr, head_dim) };
            *score = dot * scale;
        } else {
            *score = f32::NEG_INFINITY;
        }
    }
}

// ── Sparse attention pattern ────────────────────────────────────────

/// Compute attention scores for a sparse pattern that combines
/// strided global tokens with a local block window.
///
/// A position `i` is attended if:
/// - `i % stride == 0` (global strided token), or
/// - `i` falls within `[max(0, query_pos - local_size + 1) ..
///   query_pos]` (local block)
///
/// Unattended positions receive `f32::NEG_INFINITY`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// Input slices must have the documented minimum lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sparse_attention_scores(
    query: &[f32],
    keys: &[f32],
    head_dim: usize,
    query_pos: usize,
    stride: usize,
    local_size: usize,
    scores_out: &mut [f32],
) {
    assert!(head_dim > 0, "head_dim must be positive");
    assert!(stride > 0, "stride must be positive");
    let seq_len = keys.len() / head_dim;
    assert_eq!(keys.len() % head_dim, 0, "keys length must be a multiple of head_dim");
    assert!(scores_out.len() >= seq_len, "scores_out too small");
    assert_eq!(query.len(), head_dim, "query length mismatch");

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q_ptr = query.as_ptr();

    let local_start = (query_pos + 1).saturating_sub(local_size);
    let local_end = (query_pos + 1).min(seq_len);

    for (i, score) in scores_out.iter_mut().enumerate().take(seq_len) {
        let is_strided = i % stride == 0;
        let is_local = i >= local_start && i < local_end;

        if is_strided || is_local {
            let k_ptr = unsafe { keys.as_ptr().add(i * head_dim) };
            let dot = unsafe { neon_dot_f32(q_ptr, k_ptr, head_dim) };
            *score = dot * scale;
        } else {
            *score = f32::NEG_INFINITY;
        }
    }
}

// ── Cross-attention score ───────────────────────────────────────────

/// Compute encoder-decoder cross-attention scores.
///
/// - `decoder_queries`: shape `[num_queries * head_dim]`
/// - `encoder_keys`: shape `[num_enc_keys * head_dim]`
/// - `scores_out`: shape `[num_queries * num_enc_keys]`
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// Input slices must have the documented minimum lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_cross_attention_scores(
    decoder_queries: &[f32],
    encoder_keys: &[f32],
    head_dim: usize,
    scores_out: &mut [f32],
) {
    assert!(head_dim > 0, "head_dim must be positive");
    assert_eq!(
        decoder_queries.len() % head_dim,
        0,
        "decoder_queries length must be a multiple of head_dim"
    );
    assert_eq!(
        encoder_keys.len() % head_dim,
        0,
        "encoder_keys length must be a multiple of head_dim"
    );
    let num_queries = decoder_queries.len() / head_dim;
    let num_enc_keys = encoder_keys.len() / head_dim;
    assert!(scores_out.len() >= num_queries * num_enc_keys, "scores_out too small");

    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    for q in 0..num_queries {
        let q_ptr = unsafe { decoder_queries.as_ptr().add(q * head_dim) };
        for k in 0..num_enc_keys {
            let k_ptr = unsafe { encoder_keys.as_ptr().add(k * head_dim) };
            let dot = unsafe { neon_dot_f32(q_ptr, k_ptr, head_dim) };
            scores_out[q * num_enc_keys + k] = dot * scale;
        }
    }
}

// ── Attention score clamping ────────────────────────────────────────

/// Clamp attention scores to `[min_val, max_val]` using NEON.
///
/// This prevents extreme logits from causing numerical issues in the
/// subsequent softmax.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_clamp_attention_scores(scores: &mut [f32], min_val: f32, max_val: f32) {
    let len = scores.len();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let ptr = scores.as_mut_ptr();

    let vmin = vdupq_n_f32(min_val);
    let vmax = vdupq_n_f32(max_val);

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(ptr.add(offset));
            let clamped = vminq_f32(vmaxq_f32(v, vmin), vmax);
            vst1q_f32(ptr.add(offset), clamped);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        scores[tail + i] = scores[tail + i].clamp(min_val, max_val);
    }
}

// ── Attention dropout mask ──────────────────────────────────────────

/// Apply a pre-computed binary dropout mask to attention scores.
///
/// For each element: `scores[i] *= mask[i] * inv_keep_prob` where
/// `mask[i]` is `1.0` (keep) or `0.0` (drop). `inv_keep_prob` is
/// `1.0 / (1.0 - dropout_rate)` to rescale kept values.
///
/// # Panics
///
/// Panics if `scores` and `mask` have different lengths.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `scores` and `mask` must have equal lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_apply_attention_dropout(scores: &mut [f32], mask: &[f32], inv_keep_prob: f32) {
    assert_eq!(scores.len(), mask.len(), "scores and mask length mismatch");
    let len = scores.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    let s_ptr = scores.as_mut_ptr();
    let m_ptr = mask.as_ptr();
    let v_scale = vdupq_n_f32(inv_keep_prob);

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let vs = vld1q_f32(s_ptr.add(offset));
            let vm = vld1q_f32(m_ptr.add(offset));
            let masked = vmulq_f32(vs, vm);
            let scaled = vmulq_f32(masked, v_scale);
            vst1q_f32(s_ptr.add(offset), scaled);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        scores[tail + i] *= mask[tail + i] * inv_keep_prob;
    }
}

/// Generate a binary dropout mask from a pre-filled random buffer.
///
/// `random_vals[i]` should be in `[0, 1)`. If the random value is
/// below `keep_prob`, the mask element is `1.0`; otherwise `0.0`.
///
/// # Panics
///
/// Panics if `mask_out` and `random_vals` have different lengths.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `random_vals` and `mask_out` must have equal lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_generate_dropout_mask(
    random_vals: &[f32],
    keep_prob: f32,
    mask_out: &mut [f32],
) {
    assert_eq!(random_vals.len(), mask_out.len(), "random_vals and mask_out length mismatch");
    let len = random_vals.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    let r_ptr = random_vals.as_ptr();
    let o_ptr = mask_out.as_mut_ptr();
    let v_thresh = vdupq_n_f32(keep_prob);
    let v_one = vdupq_n_f32(1.0);
    let v_zero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let vr = vld1q_f32(r_ptr.add(offset));
            let cmp = vcltq_f32(vr, v_thresh);
            let selected = vbslq_f32(cmp, v_one, v_zero);
            vst1q_f32(o_ptr.add(offset), selected);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        mask_out[tail + i] = if random_vals[tail + i] < keep_prob { 1.0 } else { 0.0 };
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    // Helper: reference dot product.
    fn ref_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    // ── Scaled dot-product ──────────────────────────────────────────

    #[test]
    fn test_scaled_dot_product_basic() {
        let head_dim = 8;
        let query = vec![1.0_f32; head_dim];
        let keys = vec![1.0_f32; head_dim * 3]; // 3 keys
        let mut scores = [0.0_f32; 3];

        unsafe {
            neon_scaled_dot_product_attention(&query, &keys, head_dim, &mut scores);
        }

        let expected = head_dim as f32 / (head_dim as f32).sqrt();
        for s in &scores {
            assert!((s - expected).abs() < 1e-5, "got {s}, expected {expected}");
        }
    }

    #[test]
    fn test_scaled_dot_product_non_uniform() {
        let head_dim = 4;
        let query = [1.0, 2.0, 3.0, 4.0];
        let keys = [4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5];
        let mut scores = [0.0_f32; 2];

        unsafe {
            neon_scaled_dot_product_attention(&query, &keys, head_dim, &mut scores);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let expected0 = ref_dot(&query, &keys[..4]) * scale;
        let expected1 = ref_dot(&query, &keys[4..8]) * scale;
        assert!((scores[0] - expected0).abs() < 1e-5);
        assert!((scores[1] - expected1).abs() < 1e-5);
    }

    #[test]
    fn test_scaled_dot_product_remainder() {
        // head_dim not a multiple of LANES
        let head_dim = 5;
        let query = vec![1.0_f32; head_dim];
        let keys = vec![2.0_f32; head_dim * 2];
        let mut scores = [0.0_f32; 2];

        unsafe {
            neon_scaled_dot_product_attention(&query, &keys, head_dim, &mut scores);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let expected = (head_dim as f32) * 2.0 * scale;
        for s in &scores {
            assert!((s - expected).abs() < 1e-5);
        }
    }

    // ── Multi-head attention ────────────────────────────────────────

    #[test]
    fn test_multi_head_attention_scores() {
        let num_heads = 2;
        let num_keys = 3;
        let head_dim = 4;
        let queries = vec![1.0_f32; num_heads * head_dim];
        let keys = vec![1.0_f32; num_heads * num_keys * head_dim];
        let mut scores = vec![0.0_f32; num_heads * num_keys];

        unsafe {
            neon_multi_head_attention_scores(
                &queries,
                &keys,
                num_heads,
                num_keys,
                head_dim,
                &mut scores,
            );
        }

        let expected = (head_dim as f32) / (head_dim as f32).sqrt();
        for s in &scores {
            assert!((s - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_multi_head_independent() {
        let num_heads = 2;
        let num_keys = 1;
        let head_dim = 4;
        let queries = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let keys = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut scores = [0.0_f32; 2];

        unsafe {
            neon_multi_head_attention_scores(
                &queries,
                &keys,
                num_heads,
                num_keys,
                head_dim,
                &mut scores,
            );
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        assert!((scores[0] - scale).abs() < 1e-5);
        assert!((scores[1] - scale).abs() < 1e-5);
    }

    // ── Sliding window ──────────────────────────────────────────────

    #[test]
    fn test_sliding_window_basic() {
        let head_dim = 4;
        let seq_len = 6;
        let query = vec![1.0_f32; head_dim];
        let keys = vec![1.0_f32; seq_len * head_dim];
        let mut scores = vec![0.0_f32; seq_len];

        unsafe {
            neon_sliding_window_attention(
                &query,
                &keys,
                head_dim,
                5, // query at position 5
                3, // window of 3
                &mut scores,
            );
        }

        // Positions 3, 4, 5 should be valid; 0, 1, 2 = NEG_INFINITY
        for i in 0..3 {
            assert_eq!(scores[i], f32::NEG_INFINITY);
        }
        let expected = (head_dim as f32) / (head_dim as f32).sqrt();
        for i in 3..6 {
            assert!((scores[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sliding_window_start_of_seq() {
        let head_dim = 4;
        let seq_len = 4;
        let query = vec![1.0_f32; head_dim];
        let keys = vec![1.0_f32; seq_len * head_dim];
        let mut scores = vec![0.0_f32; seq_len];

        // At position 1 with window 4 → all positions 0..2 visible
        unsafe {
            neon_sliding_window_attention(&query, &keys, head_dim, 1, 4, &mut scores);
        }

        // Positions 0 and 1 should be valid, rest neg inf
        let expected = (head_dim as f32) / (head_dim as f32).sqrt();
        assert!((scores[0] - expected).abs() < 1e-5);
        assert!((scores[1] - expected).abs() < 1e-5);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        assert_eq!(scores[3], f32::NEG_INFINITY);
    }

    // ── Sparse attention ────────────────────────────────────────────

    #[test]
    fn test_sparse_attention_strided() {
        let head_dim = 4;
        let seq_len = 8;
        let query = vec![1.0_f32; head_dim];
        let keys = vec![1.0_f32; seq_len * head_dim];
        let mut scores = vec![0.0_f32; seq_len];

        unsafe {
            neon_sparse_attention_scores(
                &query,
                &keys,
                head_dim,
                7, // query_pos
                4, // stride
                1, // local_size=1 → only pos 7
                &mut scores,
            );
        }

        let expected = (head_dim as f32) / (head_dim as f32).sqrt();
        // Strided: 0, 4; Local: 7
        assert!((scores[0] - expected).abs() < 1e-5);
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        assert_eq!(scores[3], f32::NEG_INFINITY);
        assert!((scores[4] - expected).abs() < 1e-5);
        assert_eq!(scores[5], f32::NEG_INFINITY);
        assert_eq!(scores[6], f32::NEG_INFINITY);
        assert!((scores[7] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_attention_local_overlap() {
        let head_dim = 4;
        let seq_len = 8;
        let query = vec![1.0_f32; head_dim];
        let keys = vec![1.0_f32; seq_len * head_dim];
        let mut scores = vec![0.0_f32; seq_len];

        unsafe {
            neon_sparse_attention_scores(
                &query,
                &keys,
                head_dim,
                4, // query_pos
                4, // stride
                3, // local_size=3 → positions 2,3,4
                &mut scores,
            );
        }

        let expected = (head_dim as f32) / (head_dim as f32).sqrt();
        // Strided: 0, 4; Local: 2, 3, 4 → union: 0, 2, 3, 4
        assert!((scores[0] - expected).abs() < 1e-5); // strided
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert!((scores[2] - expected).abs() < 1e-5); // local
        assert!((scores[3] - expected).abs() < 1e-5); // local
        assert!((scores[4] - expected).abs() < 1e-5); // both
        for i in 5..seq_len {
            assert_eq!(scores[i], f32::NEG_INFINITY);
        }
    }

    // ── Cross-attention ─────────────────────────────────────────────

    #[test]
    fn test_cross_attention_basic() {
        let head_dim = 4;
        let num_queries = 2;
        let num_enc_keys = 3;
        let dec_q = vec![1.0_f32; num_queries * head_dim];
        let enc_k = vec![1.0_f32; num_enc_keys * head_dim];
        let mut scores = vec![0.0_f32; num_queries * num_enc_keys];

        unsafe {
            neon_cross_attention_scores(&dec_q, &enc_k, head_dim, &mut scores);
        }

        let expected = (head_dim as f32) / (head_dim as f32).sqrt();
        for s in &scores {
            assert!((s - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cross_attention_asymmetric() {
        let head_dim = 4;
        let dec_q = [1.0, 0.0, 0.0, 0.0]; // 1 query
        let enc_k = [
            1.0, 0.0, 0.0, 0.0, // enc key 0
            0.0, 1.0, 0.0, 0.0, // enc key 1
        ];
        let mut scores = [0.0_f32; 2];

        unsafe {
            neon_cross_attention_scores(&dec_q, &enc_k, head_dim, &mut scores);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        assert!((scores[0] - scale).abs() < 1e-5); // dot = 1
        assert!(scores[1].abs() < 1e-5); // dot = 0
    }

    // ── Clamping ────────────────────────────────────────────────────

    #[test]
    fn test_clamp_basic() {
        let mut scores = [-100.0, -1.0, 0.0, 0.5, 1.0, 50.0, 100.0, 200.0];
        unsafe {
            neon_clamp_attention_scores(&mut scores, -10.0, 10.0);
        }
        let expected = [-10.0, -1.0, 0.0, 0.5, 1.0, 10.0, 10.0, 10.0];
        for (a, b) in scores.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, want {b}");
        }
    }

    #[test]
    fn test_clamp_remainder() {
        let mut scores = [-50.0, 0.0, 50.0, 3.0, -50.0];
        unsafe {
            neon_clamp_attention_scores(&mut scores, -5.0, 5.0);
        }
        assert_eq!(scores, [-5.0, 0.0, 5.0, 3.0, -5.0]);
    }

    #[test]
    fn test_clamp_identity() {
        let mut scores = [0.0, 1.0, 2.0, 3.0];
        unsafe {
            neon_clamp_attention_scores(&mut scores, -100.0, 100.0);
        }
        assert_eq!(scores, [0.0, 1.0, 2.0, 3.0]);
    }

    // ── Dropout mask ────────────────────────────────────────────────

    #[test]
    fn test_generate_dropout_mask() {
        let random = [0.1, 0.9, 0.3, 0.8, 0.5, 0.05, 0.99, 0.4];
        let keep_prob = 0.5;
        let mut mask = [0.0_f32; 8];

        unsafe {
            neon_generate_dropout_mask(&random, keep_prob, &mut mask);
        }

        let expected = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_generate_dropout_mask_remainder() {
        let random = [0.1, 0.9, 0.3, 0.8, 0.2];
        let keep_prob = 0.5;
        let mut mask = [0.0_f32; 5];

        unsafe {
            neon_generate_dropout_mask(&random, keep_prob, &mut mask);
        }

        assert_eq!(mask, [1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_apply_dropout() {
        let mut scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mask = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let inv_keep = 2.0; // keep_prob = 0.5

        unsafe {
            neon_apply_attention_dropout(&mut scores, &mask, inv_keep);
        }

        let expected = [2.0, 0.0, 6.0, 0.0, 10.0, 0.0, 14.0, 0.0];
        for (a, b) in scores.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, want {b}");
        }
    }

    #[test]
    fn test_apply_dropout_remainder() {
        let mut scores = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = [1.0, 0.0, 1.0, 1.0, 0.0];
        let inv_keep = 2.0;

        unsafe {
            neon_apply_attention_dropout(&mut scores, &mask, inv_keep);
        }

        assert!((scores[0] - 2.0).abs() < 1e-5);
        assert!((scores[1]).abs() < 1e-5);
        assert!((scores[2] - 6.0).abs() < 1e-5);
        assert!((scores[3] - 8.0).abs() < 1e-5);
        assert!((scores[4]).abs() < 1e-5);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_single_key() {
        const DIM: usize = 4;
        let query = [2.0_f32; DIM];
        let keys = [3.0_f32; DIM];
        let mut scores = [0.0_f32; 1];

        unsafe {
            neon_scaled_dot_product_attention(&query, &keys, DIM, &mut scores);
        }

        let scale = 1.0 / (DIM as f32).sqrt();
        let expected = 24.0 * scale; // 4 * (2*3) * scale
        assert!((scores[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_large_head_dim() {
        let head_dim = 128;
        let query = vec![0.1_f32; head_dim];
        let keys = vec![0.1_f32; head_dim * 2];
        let mut scores = [0.0_f32; 2];

        unsafe {
            neon_scaled_dot_product_attention(&query, &keys, head_dim, &mut scores);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let expected = (head_dim as f32) * 0.01 * scale;
        for s in &scores {
            assert!((s - expected).abs() < 1e-4);
        }
    }
}
