//! SIMD-optimized attention mask construction and application.
//!
//! Extends the base attention_mask module with vectorised implementations of
//! causal, sliding-window, ALiBi, block-sparse, Longformer-style global+local
//! hybrid, prefix, and cross-attention masks.
//!
//! On x86-64 with AVX2, hot loops are 8-wide; a scalar fallback handles all
//! other targets and tail elements.  All masks use the **additive** convention:
//! `0.0` means "attend" and `f32::NEG_INFINITY` means "block".
#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

const NEG_INF: f32 = f32::NEG_INFINITY;
const SIMD_WIDTH: usize = 8;

// -- AVX2 helpers --------------------------------------------------------

/// Store 8 copies of `val` across `dst[0..8]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_broadcast(dst: *mut f32, val: f32) {
    let v = _mm256_set1_ps(val);
    _mm256_storeu_ps(dst, v);
}

/// Add src and mask element-wise, writing to dst, 8-wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add_mask_avx2(dst: *mut f32, src: *const f32, mask: *const f32, n: usize) {
    let chunks = n / SIMD_WIDTH;
    for i in 0..chunks {
        let off = i * SIMD_WIDTH;
        let vs = _mm256_loadu_ps(src.add(off));
        let vm = _mm256_loadu_ps(mask.add(off));
        _mm256_storeu_ps(dst.add(off), _mm256_add_ps(vs, vm));
    }
    for i in (chunks * SIMD_WIDTH)..n {
        *dst.add(i) = *src.add(i) + *mask.add(i);
    }
}

// -- 1. Causal mask with SIMD fill ---------------------------------------

/// Create a causal (lower-triangular) additive mask of shape
/// `[seq_len, seq_len]` using SIMD-accelerated fills where available.
///
/// `mask[i * seq_len + j]` is `0.0` when `j <= i` and
/// `f32::NEG_INFINITY` when `j > i`.
pub fn simd_causal_mask(seq_len: usize) -> Vec<f32> {
    if seq_len == 0 {
        return Vec::new();
    }
    let n = seq_len * seq_len;
    let mut mask = vec![0.0_f32; n];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { causal_mask_avx2(&mut mask, seq_len) };
            return mask;
        }
    }

    causal_mask_scalar(&mut mask, seq_len);
    mask
}

fn causal_mask_scalar(mask: &mut [f32], seq_len: usize) {
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = NEG_INF;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn causal_mask_avx2(mask: &mut [f32], seq_len: usize) {
    let ptr = mask.as_mut_ptr();
    for i in 0..seq_len {
        let fill_start = i * seq_len + i + 1;
        let fill_len = seq_len - i - 1;
        let mut j = 0;
        while j + SIMD_WIDTH <= fill_len {
            store_broadcast(ptr.add(fill_start + j), NEG_INF);
            j += SIMD_WIDTH;
        }
        while j < fill_len {
            *ptr.add(fill_start + j) = NEG_INF;
            j += 1;
        }
    }
}

// -- 2. Sliding window mask ----------------------------------------------

/// Create a sliding-window causal mask of shape `[seq_len, seq_len]`.
///
/// Position `(i, j)` is `0.0` when `i.saturating_sub(window - 1) <= j <= i`,
/// otherwise `f32::NEG_INFINITY`.
///
/// When `window >= seq_len` this equals a standard causal mask.
/// A `window` of `0` blocks every position.
pub fn simd_sliding_window_mask(seq_len: usize, window: usize) -> Vec<f32> {
    if seq_len == 0 {
        return Vec::new();
    }
    let n = seq_len * seq_len;
    let mut mask = vec![NEG_INF; n];
    if window == 0 {
        return mask;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sliding_window_avx2(&mut mask, seq_len, window) };
            return mask;
        }
    }

    sliding_window_scalar(&mut mask, seq_len, window);
    mask
}

fn sliding_window_scalar(mask: &mut [f32], seq_len: usize, window: usize) {
    for i in 0..seq_len {
        let start = i.saturating_sub(window - 1);
        for j in start..=i {
            mask[i * seq_len + j] = 0.0;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sliding_window_avx2(mask: &mut [f32], seq_len: usize, window: usize) {
    let ptr = mask.as_mut_ptr();
    for i in 0..seq_len {
        let start = i.saturating_sub(window - 1);
        let row_off = i * seq_len + start;
        let fill_len = i - start + 1;
        let mut j = 0;
        while j + SIMD_WIDTH <= fill_len {
            store_broadcast(ptr.add(row_off + j), 0.0);
            j += SIMD_WIDTH;
        }
        while j < fill_len {
            *ptr.add(row_off + j) = 0.0;
            j += 1;
        }
    }
}

// -- 3. ALiBi slope computation ------------------------------------------

/// Compute ALiBi (Attention with Linear Biases) per-head slopes.
///
/// Returns `num_heads` slopes following the original ALiBi paper
/// (Press et al., 2022).  Power-of-two head counts use a single
/// geometric sequence; non-power-of-two counts interleave two sequences.
pub fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    assert!(num_heads > 0, "num_heads must be > 0");

    let nearest_pow2 = num_heads.next_power_of_two();

    if num_heads == nearest_pow2 {
        let ratio = 2.0_f32.powf(-8.0 / num_heads as f32);
        (1..=num_heads).map(|k| ratio.powi(k as i32)).collect()
    } else {
        let ratio1 = 2.0_f32.powf(-8.0 / nearest_pow2 as f32);
        let ratio2 = ratio1 * ratio1;
        let mut slopes = Vec::with_capacity(num_heads);
        let half = nearest_pow2 / 2;
        for k in 1..=half {
            slopes.push(ratio1.powi(k as i32));
            if slopes.len() < num_heads {
                slopes.push(ratio2.powi(k as i32));
            }
        }
        slopes.truncate(num_heads);
        slopes
    }
}

/// Build an ALiBi bias matrix of shape `[seq_len, seq_len]` for a single
/// head with the given slope.
///
/// `bias[i][j] = -slope * (i - j)` for causal positions (`j <= i`),
/// and `f32::NEG_INFINITY` for `j > i`.
pub fn alibi_bias_matrix(seq_len: usize, slope: f32) -> Vec<f32> {
    if seq_len == 0 {
        return Vec::new();
    }
    let n = seq_len * seq_len;
    let mut bias = vec![NEG_INF; n];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { alibi_bias_avx2(&mut bias, seq_len, slope) };
            return bias;
        }
    }

    alibi_bias_scalar(&mut bias, seq_len, slope);
    bias
}

fn alibi_bias_scalar(bias: &mut [f32], seq_len: usize, slope: f32) {
    for i in 0..seq_len {
        for j in 0..=i {
            bias[i * seq_len + j] = -slope * (i - j) as f32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn alibi_bias_avx2(bias: &mut [f32], seq_len: usize, slope: f32) {
    let ptr = bias.as_mut_ptr();
    let neg_slope = _mm256_set1_ps(-slope);
    for i in 0..seq_len {
        let row = i * seq_len;
        let causal_len = i + 1;
        let mut j = 0;
        while j + SIMD_WIDTH <= causal_len {
            let dists = _mm256_set_ps(
                (i - j - 7) as f32,
                (i - j - 6) as f32,
                (i - j - 5) as f32,
                (i - j - 4) as f32,
                (i - j - 3) as f32,
                (i - j - 2) as f32,
                (i - j - 1) as f32,
                (i - j) as f32,
            );
            let biases = _mm256_mul_ps(neg_slope, dists);
            _mm256_storeu_ps(ptr.add(row + j), biases);
            j += SIMD_WIDTH;
        }
        while j <= i {
            *ptr.add(row + j) = -slope * (i - j) as f32;
            j += 1;
        }
    }
}

// -- 4. Block-sparse attention mask --------------------------------------

/// Create a block-sparse attention mask of shape `[seq_len, seq_len]`.
///
/// Divides the sequence into blocks of `block_size` tokens.  Position
/// `(i, j)` is `0.0` when `i` and `j` fall in the same block.  If
/// `causal` is true, the additional constraint `j <= i` is enforced.
pub fn block_sparse_mask(seq_len: usize, block_size: usize, causal: bool) -> Vec<f32> {
    assert!(block_size > 0, "block_size must be > 0");
    if seq_len == 0 {
        return Vec::new();
    }
    let n = seq_len * seq_len;
    let mut mask = vec![NEG_INF; n];

    for i in 0..seq_len {
        let block_i = i / block_size;
        let block_start = block_i * block_size;
        let block_end = ((block_i + 1) * block_size).min(seq_len);
        for j in block_start..block_end {
            if causal && j > i {
                continue;
            }
            mask[i * seq_len + j] = 0.0;
        }
    }
    mask
}

/// Create a strided block-sparse mask where each token additionally
/// attends to every `stride`-th block.
pub fn strided_block_sparse_mask(
    seq_len: usize,
    block_size: usize,
    stride: usize,
    causal: bool,
) -> Vec<f32> {
    assert!(block_size > 0, "block_size must be > 0");
    assert!(stride > 0, "stride must be > 0");
    if seq_len == 0 {
        return Vec::new();
    }
    let n = seq_len * seq_len;
    let mut mask = vec![NEG_INF; n];

    for i in 0..seq_len {
        let block_i = i / block_size;
        for j in 0..seq_len {
            if causal && j > i {
                continue;
            }
            let block_j = j / block_size;
            if block_j == block_i || block_j.is_multiple_of(stride) {
                mask[i * seq_len + j] = 0.0;
            }
        }
    }
    mask
}

// -- 5. Global + local hybrid mask (Longformer-style) --------------------

/// Create a Longformer-style global+local attention mask.
///
/// `global_indices` lists token positions that attend to (and are attended
/// by) every other position.  All other tokens use a local sliding
/// window of `local_window` positions (causal, including self).
pub fn longformer_mask(seq_len: usize, local_window: usize, global_indices: &[usize]) -> Vec<f32> {
    if seq_len == 0 {
        return Vec::new();
    }
    let n = seq_len * seq_len;
    let mut mask = vec![NEG_INF; n];

    let mut is_global = vec![false; seq_len];
    for &g in global_indices {
        if g < seq_len {
            is_global[g] = true;
        }
    }

    for i in 0..seq_len {
        if is_global[i] {
            for j in 0..seq_len {
                mask[i * seq_len + j] = 0.0;
            }
        } else {
            let start = if local_window == 0 { i } else { i.saturating_sub(local_window - 1) };
            for j in start..=i {
                mask[i * seq_len + j] = 0.0;
            }
            for j in 0..seq_len {
                if is_global[j] {
                    mask[i * seq_len + j] = 0.0;
                }
            }
        }
    }
    mask
}

// -- 6. Mask application with SIMD ---------------------------------------

/// Apply an additive mask to pre-softmax attention scores (in-place)
/// using SIMD where available.
pub fn simd_apply_mask(scores: &mut [f32], mask: &[f32], len: usize) {
    assert!(scores.len() >= len, "scores length {} too short for len={len}", scores.len());
    assert!(mask.len() >= len, "mask length {} too short for len={len}", mask.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                add_mask_avx2(scores.as_mut_ptr(), scores.as_ptr(), mask.as_ptr(), len);
            }
            return;
        }
    }

    for i in 0..len {
        scores[i] += mask[i];
    }
}

/// Apply an additive mask to a batched set of attention score matrices.
///
/// `scores` is `[batch_size, seq_len, seq_len]` (row-major), and `mask`
/// is `[seq_len, seq_len]` (broadcast over the batch dimension).
pub fn simd_apply_mask_batched(
    scores: &mut [f32],
    mask: &[f32],
    batch_size: usize,
    seq_len: usize,
) {
    let mat_size = seq_len * seq_len;
    assert!(mask.len() >= mat_size, "mask too short: {} < {}", mask.len(), mat_size);
    assert!(
        scores.len() >= batch_size * mat_size,
        "scores too short: {} < {}",
        scores.len(),
        batch_size * mat_size,
    );

    for b in 0..batch_size {
        let offset = b * mat_size;
        simd_apply_mask(&mut scores[offset..offset + mat_size], mask, mat_size);
    }
}

// -- 7. Prefix mask for prompt caching -----------------------------------

/// Create a prefix-aware causal mask of shape `[seq_len, seq_len]`.
///
/// Tokens in the prefix range `[0, prefix_len)` attend to all prefix
/// tokens bidirectionally (for KV-cache reuse), while generation tokens
/// `[prefix_len, seq_len)` use standard causal masking over the full
/// context.
pub fn prefix_mask(seq_len: usize, prefix_len: usize) -> Vec<f32> {
    if seq_len == 0 {
        return Vec::new();
    }
    let prefix = prefix_len.min(seq_len);
    let n = seq_len * seq_len;
    let mut mask = vec![NEG_INF; n];

    for i in 0..seq_len {
        if i < prefix {
            for j in 0..prefix {
                mask[i * seq_len + j] = 0.0;
            }
        } else {
            for j in 0..=i {
                mask[i * seq_len + j] = 0.0;
            }
        }
    }
    mask
}

// -- 8. Cross-attention masks --------------------------------------------

/// Create a cross-attention mask of shape `[query_len, key_len]`.
///
/// Positions `j < key_valid` are `0.0` and `j >= key_valid` are
/// `f32::NEG_INFINITY`.
pub fn cross_attention_mask(query_len: usize, key_len: usize, key_valid: usize) -> Vec<f32> {
    let valid = key_valid.min(key_len);
    let n = query_len * key_len;
    let mut mask = vec![0.0_f32; n];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { cross_attn_pad_avx2(&mut mask, query_len, key_len, valid) };
            return mask;
        }
    }

    for i in 0..query_len {
        for j in valid..key_len {
            mask[i * key_len + j] = NEG_INF;
        }
    }
    mask
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cross_attn_pad_avx2(mask: &mut [f32], query_len: usize, key_len: usize, valid: usize) {
    let ptr = mask.as_mut_ptr();
    let pad_len = key_len - valid;
    for i in 0..query_len {
        let row_pad_start = i * key_len + valid;
        let mut j = 0;
        while j + SIMD_WIDTH <= pad_len {
            store_broadcast(ptr.add(row_pad_start + j), NEG_INF);
            j += SIMD_WIDTH;
        }
        while j < pad_len {
            *ptr.add(row_pad_start + j) = NEG_INF;
            j += 1;
        }
    }
}

/// Create a batched cross-attention mask of shape
/// `[batch_size, query_len, key_len]`.
pub fn cross_attention_mask_batched(
    query_len: usize,
    key_len: usize,
    key_lengths: &[usize],
) -> Vec<f32> {
    let batch = key_lengths.len();
    let mat_size = query_len * key_len;
    let mut masks = vec![0.0_f32; batch * mat_size];

    for (b, &klen) in key_lengths.iter().enumerate() {
        let offset = b * mat_size;
        let single = cross_attention_mask(query_len, key_len, klen);
        masks[offset..offset + mat_size].copy_from_slice(&single);
    }
    masks
}

// -- Utility: combine masks ----------------------------------------------

/// Element-wise combine two additive masks using SIMD.
pub fn simd_combine_masks(a: &[f32], b: &[f32], len: usize) -> Vec<f32> {
    assert!(a.len() >= len, "mask a length {} too short for len={len}", a.len());
    assert!(b.len() >= len, "mask b length {} too short for len={len}", b.len());

    let mut result = vec![0.0_f32; len];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                add_mask_avx2(result.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);
            }
            return result;
        }
    }

    for i in 0..len {
        result[i] = a[i] + b[i];
    }
    result
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const INF: f32 = f32::NEG_INFINITY;

    fn is_neg_inf(v: f32) -> bool {
        v == INF
    }

    fn softmax(row: &[f32]) -> Vec<f32> {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 { vec![0.0; row.len()] } else { exps.iter().map(|&e| e / sum).collect() }
    }

    // -- 1. simd_causal_mask ---------------------------------------------

    #[test]
    fn causal_mask_empty() {
        assert!(simd_causal_mask(0).is_empty());
    }

    #[test]
    fn causal_mask_1x1() {
        assert_eq!(simd_causal_mask(1), vec![0.0]);
    }

    #[test]
    fn causal_mask_3x3_known() {
        let m = simd_causal_mask(3);
        #[rustfmt::skip]
        let expected = [
            0.0,  INF,  INF,
            0.0,  0.0,  INF,
            0.0,  0.0,  0.0,
        ];
        for (i, (&got, &want)) in m.iter().zip(expected.iter()).enumerate() {
            assert!(got == want, "idx {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn causal_mask_diagonal_zero() {
        for n in 1..=16 {
            let m = simd_causal_mask(n);
            for i in 0..n {
                assert_eq!(m[i * n + i], 0.0, "diag ({i},{i}) for n={n}");
            }
        }
    }

    #[test]
    fn causal_mask_lower_triangle_zero() {
        let n = 7;
        let m = simd_causal_mask(n);
        for i in 0..n {
            for j in 0..=i {
                assert_eq!(m[i * n + j], 0.0, "({i},{j})");
            }
        }
    }

    #[test]
    fn causal_mask_upper_triangle_neg_inf() {
        let n = 7;
        let m = simd_causal_mask(n);
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(is_neg_inf(m[i * n + j]), "({i},{j}) should be -inf");
            }
        }
    }

    #[test]
    fn causal_mask_open_count() {
        for n in 0..=16 {
            let m = simd_causal_mask(n);
            let open = m.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(open, n * (n + 1) / 2, "n={n}");
        }
    }

    #[test]
    fn causal_mask_large_simd_boundary() {
        for n in [8, 9, 15, 16, 17, 31, 32, 33] {
            let m = simd_causal_mask(n);
            let open = m.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(open, n * (n + 1) / 2, "n={n}");
        }
    }

    #[test]
    fn causal_mask_matches_scalar() {
        for n in [1, 3, 8, 15, 16, 17, 32] {
            let simd_result = simd_causal_mask(n);
            let mut scalar = vec![0.0_f32; n * n];
            causal_mask_scalar(&mut scalar, n);
            assert_eq!(simd_result, scalar, "mismatch at n={n}");
        }
    }

    #[test]
    fn causal_mask_softmax_blocks_future() {
        let n = 6;
        let mask = simd_causal_mask(n);
        let mut scores = vec![1.0; n * n];
        simd_apply_mask(&mut scores, &mask, n * n);
        for i in 0..n {
            let row = &scores[i * n..(i + 1) * n];
            let probs = softmax(row);
            for j in (i + 1)..n {
                assert!(probs[j] < 1e-6, "row {i} col {j}");
            }
        }
    }

    // -- 2. simd_sliding_window_mask -------------------------------------

    #[test]
    fn sliding_window_empty() {
        assert!(simd_sliding_window_mask(0, 3).is_empty());
    }

    #[test]
    fn sliding_window_zero_blocks_all() {
        let m = simd_sliding_window_mask(4, 0);
        assert!(m.iter().all(|&v| is_neg_inf(v)));
    }

    #[test]
    fn sliding_window_1x1() {
        assert_eq!(simd_sliding_window_mask(1, 5), vec![0.0]);
    }

    #[test]
    fn sliding_window_3x3_w2_known() {
        let m = simd_sliding_window_mask(3, 2);
        #[rustfmt::skip]
        let expected = [
            0.0,  INF,  INF,
            0.0,  0.0,  INF,
            INF,  0.0,  0.0,
        ];
        for (i, (&got, &want)) in m.iter().zip(expected.iter()).enumerate() {
            assert!(got == want, "idx {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn sliding_window_w1_is_diagonal() {
        let n = 6;
        let m = simd_sliding_window_mask(n, 1);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_eq!(m[i * n + j], 0.0, "diag ({i},{j})");
                } else {
                    assert!(is_neg_inf(m[i * n + j]), "off-diag ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn sliding_window_large_equals_causal() {
        for n in [1, 4, 8, 16] {
            let causal = simd_causal_mask(n);
            let sliding = simd_sliding_window_mask(n, n);
            assert_eq!(causal, sliding, "n={n}");
        }
    }

    #[test]
    fn sliding_window_exceeds_seq_len() {
        let n = 5;
        let causal = simd_causal_mask(n);
        let sliding = simd_sliding_window_mask(n, n + 100);
        assert_eq!(causal, sliding);
    }

    #[test]
    fn sliding_window_open_count() {
        for n in 1..=12 {
            for w in 1..=n + 2 {
                let m = simd_sliding_window_mask(n, w);
                let open = m.iter().filter(|&&v| v == 0.0).count();
                let expected: usize = (0..n).map(|i| (i + 1).min(w)).sum();
                assert_eq!(open, expected, "n={n} w={w}");
            }
        }
    }

    #[test]
    fn sliding_window_matches_scalar() {
        for (n, w) in [(8, 3), (16, 5), (17, 8), (32, 1)] {
            let simd_result = simd_sliding_window_mask(n, w);
            let mut scalar = vec![NEG_INF; n * n];
            sliding_window_scalar(&mut scalar, n, w);
            assert_eq!(simd_result, scalar, "n={n} w={w}");
        }
    }

    #[test]
    fn sliding_window_simd_boundaries() {
        for n in [7, 8, 9, 15, 16, 17] {
            let w = 4;
            let m = simd_sliding_window_mask(n, w);
            let open = m.iter().filter(|&&v| v == 0.0).count();
            let expected: usize = (0..n).map(|i| (i + 1).min(w)).sum();
            assert_eq!(open, expected, "n={n}");
        }
    }

    // -- 3. ALiBi slopes -------------------------------------------------

    #[test]
    fn alibi_slopes_pow2_count() {
        for n in [1, 2, 4, 8, 16] {
            let s = alibi_slopes(n);
            assert_eq!(s.len(), n);
        }
    }

    #[test]
    fn alibi_slopes_8_heads() {
        let s = alibi_slopes(8);
        let ratio = 2.0_f32.powf(-1.0);
        for (k, &slope) in s.iter().enumerate() {
            let expected = ratio.powi((k + 1) as i32);
            assert!((slope - expected).abs() < 1e-6, "head {k}: got {slope}, want {expected}");
        }
    }

    #[test]
    fn alibi_slopes_1_head() {
        let s = alibi_slopes(1);
        assert_eq!(s.len(), 1);
        let expected = 2.0_f32.powf(-8.0);
        assert!((s[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn alibi_slopes_non_pow2() {
        let s = alibi_slopes(3);
        assert_eq!(s.len(), 3);
        for &slope in &s {
            assert!(slope > 0.0, "slope must be positive, got {slope}");
        }
    }

    #[test]
    fn alibi_slopes_positive_and_bounded() {
        for n in [1, 2, 3, 5, 7, 8, 12, 16, 32] {
            let s = alibi_slopes(n);
            assert_eq!(s.len(), n, "n={n}");
            for &slope in &s {
                assert!(slope > 0.0 && slope <= 1.0, "n={n} slope={slope}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "num_heads must be > 0")]
    fn alibi_slopes_zero_panics() {
        alibi_slopes(0);
    }

    #[test]
    fn alibi_bias_matrix_empty() {
        assert!(alibi_bias_matrix(0, 0.5).is_empty());
    }

    #[test]
    fn alibi_bias_matrix_1x1() {
        let b = alibi_bias_matrix(1, 0.5);
        assert_eq!(b, vec![0.0]);
    }

    #[test]
    fn alibi_bias_matrix_diagonal_zero() {
        let n = 6;
        let b = alibi_bias_matrix(n, 0.25);
        for i in 0..n {
            assert_eq!(b[i * n + i], 0.0, "diag ({i},{i})");
        }
    }

    #[test]
    fn alibi_bias_matrix_causal() {
        let n = 4;
        let b = alibi_bias_matrix(n, 0.5);
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(is_neg_inf(b[i * n + j]), "({i},{j}) should be -inf");
            }
        }
    }

    #[test]
    fn alibi_bias_matrix_known_values() {
        let b = alibi_bias_matrix(3, 1.0);
        assert_eq!(b[0], 0.0);
        assert!(is_neg_inf(b[1]));
        assert!(is_neg_inf(b[2]));
        assert_eq!(b[3], -1.0);
        assert_eq!(b[4], 0.0);
        assert!(is_neg_inf(b[5]));
        assert_eq!(b[6], -2.0);
        assert_eq!(b[7], -1.0);
        assert_eq!(b[8], 0.0);
    }

    #[test]
    fn alibi_bias_matrix_matches_scalar() {
        for n in [1, 3, 8, 16, 17] {
            let slope = 0.125;
            let simd_result = alibi_bias_matrix(n, slope);
            let mut scalar = vec![NEG_INF; n * n];
            alibi_bias_scalar(&mut scalar, n, slope);
            for k in 0..n * n {
                assert!(
                    (simd_result[k] - scalar[k]).abs() < 1e-5
                        || (simd_result[k].is_infinite() && scalar[k].is_infinite()),
                    "n={n} k={k}: simd={} scalar={}",
                    simd_result[k],
                    scalar[k],
                );
            }
        }
    }

    #[test]
    fn alibi_bias_monotonically_decreasing_in_distance() {
        let n = 8;
        let slope = 0.5;
        let b = alibi_bias_matrix(n, slope);
        for i in 1..n {
            for j in 1..=i {
                assert!(
                    b[i * n + j - 1] <= b[i * n + j],
                    "row {i}: b[{}-1]={} should be <= b[{}]={}",
                    j,
                    b[i * n + j - 1],
                    j,
                    b[i * n + j],
                );
            }
        }
    }

    // -- 4. Block-sparse mask --------------------------------------------

    #[test]
    fn block_sparse_empty() {
        assert!(block_sparse_mask(0, 4, false).is_empty());
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn block_sparse_zero_block_panics() {
        block_sparse_mask(4, 0, false);
    }

    #[test]
    fn block_sparse_non_causal_4x4_b2() {
        let m = block_sparse_mask(4, 2, false);
        #[rustfmt::skip]
        let expected = [
            0.0,  0.0,  INF,  INF,
            0.0,  0.0,  INF,  INF,
            INF,  INF,  0.0,  0.0,
            INF,  INF,  0.0,  0.0,
        ];
        for (i, (&got, &want)) in m.iter().zip(expected.iter()).enumerate() {
            assert!(got == want, "idx {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn block_sparse_causal_4x4_b2() {
        let m = block_sparse_mask(4, 2, true);
        #[rustfmt::skip]
        let expected = [
            0.0,  INF,  INF,  INF,
            0.0,  0.0,  INF,  INF,
            INF,  INF,  0.0,  INF,
            INF,  INF,  0.0,  0.0,
        ];
        for (i, (&got, &want)) in m.iter().zip(expected.iter()).enumerate() {
            assert!(got == want, "idx {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn block_sparse_single_block() {
        let n = 4;
        let m = block_sparse_mask(n, n, false);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn block_sparse_partial_last_block() {
        let m = block_sparse_mask(5, 3, false);
        assert!(is_neg_inf(m[3 * 5 + 0]));
        assert_eq!(m[3 * 5 + 3], 0.0);
        assert_eq!(m[3 * 5 + 4], 0.0);
    }

    #[test]
    fn block_sparse_block_size_1_non_causal_is_diagonal() {
        let n = 5;
        let m = block_sparse_mask(n, 1, false);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_eq!(m[i * n + j], 0.0);
                } else {
                    assert!(is_neg_inf(m[i * n + j]));
                }
            }
        }
    }

    // -- strided block-sparse --------------------------------------------

    #[test]
    fn strided_block_sparse_empty() {
        assert!(strided_block_sparse_mask(0, 2, 2, false).is_empty());
    }

    #[test]
    fn strided_block_sparse_stride1_all_attend() {
        let n = 6;
        let m = strided_block_sparse_mask(n, 2, 1, false);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn strided_block_sparse_includes_own_block() {
        let n = 8;
        let m = strided_block_sparse_mask(n, 2, 4, false);
        assert_eq!(m[5 * n + 4], 0.0);
        assert_eq!(m[5 * n + 5], 0.0);
    }

    #[test]
    fn strided_block_sparse_includes_strided_blocks() {
        let n = 8;
        let m = strided_block_sparse_mask(n, 2, 2, false);
        assert_eq!(m[3 * n + 0], 0.0);
        assert_eq!(m[3 * n + 1], 0.0);
        assert_eq!(m[3 * n + 4], 0.0);
        assert_eq!(m[3 * n + 5], 0.0);
    }

    #[test]
    fn strided_block_sparse_causal() {
        let n = 4;
        let m = strided_block_sparse_mask(n, 2, 1, true);
        let causal = simd_causal_mask(n);
        assert_eq!(m, causal);
    }

    // -- 5. Longformer mask ----------------------------------------------

    #[test]
    fn longformer_empty() {
        assert!(longformer_mask(0, 3, &[]).is_empty());
    }

    #[test]
    fn longformer_no_global_equals_sliding() {
        for n in [4, 8, 16] {
            let lf = longformer_mask(n, 3, &[]);
            let sw = simd_sliding_window_mask(n, 3);
            assert_eq!(lf, sw, "n={n}");
        }
    }

    #[test]
    fn longformer_global_attends_all() {
        let n = 5;
        let m = longformer_mask(n, 2, &[0, 2]);
        for j in 0..n {
            assert_eq!(m[j], 0.0, "global row 0, col {j}");
        }
        for j in 0..n {
            assert_eq!(m[2 * n + j], 0.0, "global row 2, col {j}");
        }
    }

    #[test]
    fn longformer_local_attends_to_global() {
        let n = 5;
        let m = longformer_mask(n, 1, &[0]);
        assert_eq!(m[3 * n + 0], 0.0, "local token 3 attends global 0");
        assert_eq!(m[3 * n + 3], 0.0, "local token 3 attends self");
        assert!(is_neg_inf(m[3 * n + 1]), "local token 3 blocks non-global 1");
    }

    #[test]
    fn longformer_all_global_all_attend() {
        let n = 4;
        let globals: Vec<usize> = (0..n).collect();
        let m = longformer_mask(n, 1, &globals);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn longformer_out_of_range_global_ignored() {
        let n = 3;
        let m1 = longformer_mask(n, 2, &[100]);
        let m2 = longformer_mask(n, 2, &[]);
        assert_eq!(m1, m2);
    }

    #[test]
    fn longformer_global_symmetry() {
        let n = 6;
        let m = longformer_mask(n, 2, &[1, 4]);
        for g in [1, 4] {
            for i in 0..n {
                assert_eq!(m[i * n + g], 0.0, "all tokens attend global {g}, but row {i} doesn't");
            }
        }
    }

    // -- 6. simd_apply_mask ----------------------------------------------

    #[test]
    fn apply_mask_basic() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = vec![0.0, INF, 0.0, INF, 0.0, INF, 0.0, INF, 0.0];
        simd_apply_mask(&mut scores, &mask, 9);
        assert_eq!(scores[0], 1.0);
        assert!(is_neg_inf(scores[1]));
        assert_eq!(scores[2], 3.0);
    }

    #[test]
    fn apply_mask_all_zero() {
        let mut scores = vec![1.0; 16];
        let mask = vec![0.0; 16];
        simd_apply_mask(&mut scores, &mask, 16);
        assert!(scores.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn apply_mask_all_neg_inf() {
        let mut scores = vec![1.0; 16];
        let mask = vec![INF; 16];
        simd_apply_mask(&mut scores, &mask, 16);
        assert!(scores.iter().all(|&v| is_neg_inf(v)));
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn apply_mask_scores_too_short() {
        let mut scores = vec![1.0; 3];
        let mask = vec![0.0; 8];
        simd_apply_mask(&mut scores, &mask, 8);
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn apply_mask_mask_too_short() {
        let mut scores = vec![1.0; 8];
        let mask = vec![0.0; 3];
        simd_apply_mask(&mut scores, &mask, 8);
    }

    #[test]
    fn apply_mask_simd_boundary_sizes() {
        for len in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let mut scores = vec![1.0; len];
            let mask = vec![0.5; len];
            simd_apply_mask(&mut scores, &mask, len);
            for (i, &v) in scores.iter().enumerate() {
                assert!((v - 1.5).abs() < 1e-6, "len={len} idx {i}: got {v}");
            }
        }
    }

    #[test]
    fn apply_mask_batched_basic() {
        let seq = 2;
        let batch = 3;
        let mask = vec![0.0, INF, 0.0, 0.0];
        let mut scores = vec![1.0; batch * seq * seq];
        simd_apply_mask_batched(&mut scores, &mask, batch, seq);
        for b in 0..batch {
            let off = b * 4;
            assert_eq!(scores[off], 1.0);
            assert!(is_neg_inf(scores[off + 1]));
            assert_eq!(scores[off + 2], 1.0);
            assert_eq!(scores[off + 3], 1.0);
        }
    }

    #[test]
    fn apply_mask_batched_single() {
        let seq = 3;
        let mask = simd_causal_mask(seq);
        let mut scores = vec![2.0; seq * seq];
        simd_apply_mask_batched(&mut scores, &mask, 1, seq);
        for i in 0..seq {
            for j in 0..seq {
                if j <= i {
                    assert_eq!(scores[i * seq + j], 2.0);
                } else {
                    assert!(is_neg_inf(scores[i * seq + j]));
                }
            }
        }
    }

    // -- 7. prefix_mask --------------------------------------------------

    #[test]
    fn prefix_mask_empty() {
        assert!(prefix_mask(0, 3).is_empty());
    }

    #[test]
    fn prefix_mask_all_prefix() {
        let n = 4;
        let m = prefix_mask(n, n);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(m[i * n + j], 0.0, "({i},{j})");
            }
        }
    }

    #[test]
    fn prefix_mask_no_prefix_equals_causal() {
        let n = 5;
        let m = prefix_mask(n, 0);
        let c = simd_causal_mask(n);
        assert_eq!(m, c);
    }

    #[test]
    fn prefix_mask_known_3x3_p1() {
        let m = prefix_mask(3, 1);
        #[rustfmt::skip]
        let expected = [
            0.0,  INF,  INF,
            0.0,  0.0,  INF,
            0.0,  0.0,  0.0,
        ];
        for (i, (&got, &want)) in m.iter().zip(expected.iter()).enumerate() {
            assert!(got == want, "idx {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn prefix_mask_known_4x4_p2() {
        let m = prefix_mask(4, 2);
        assert_eq!(m[0], 0.0);
        assert_eq!(m[1], 0.0);
        assert!(is_neg_inf(m[2]));
        assert_eq!(m[4], 0.0);
        assert_eq!(m[5], 0.0);
        assert!(is_neg_inf(m[6]));
        assert_eq!(m[8], 0.0);
        assert_eq!(m[9], 0.0);
        assert_eq!(m[10], 0.0);
        assert!(is_neg_inf(m[11]));
        for j in 0..4 {
            assert_eq!(m[12 + j], 0.0);
        }
    }

    #[test]
    fn prefix_mask_prefix_exceeds_seq() {
        let n = 3;
        let m = prefix_mask(n, 100);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn prefix_mask_1x1() {
        let m = prefix_mask(1, 0);
        assert_eq!(m, vec![0.0]);
    }

    // -- 8. Cross-attention masks ----------------------------------------

    #[test]
    fn cross_attn_empty_query() {
        assert!(cross_attention_mask(0, 5, 5).is_empty());
    }

    #[test]
    fn cross_attn_empty_key() {
        assert!(cross_attention_mask(5, 0, 0).is_empty());
    }

    #[test]
    fn cross_attn_all_valid() {
        let m = cross_attention_mask(3, 4, 4);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cross_attn_all_padding() {
        let m = cross_attention_mask(2, 3, 0);
        assert!(m.iter().all(|&v| is_neg_inf(v)));
    }

    #[test]
    fn cross_attn_partial_padding() {
        let m = cross_attention_mask(2, 4, 2);
        for i in 0..2 {
            for j in 0..4 {
                if j < 2 {
                    assert_eq!(m[i * 4 + j], 0.0, "({i},{j})");
                } else {
                    assert!(is_neg_inf(m[i * 4 + j]), "({i},{j})");
                }
            }
        }
    }

    #[test]
    fn cross_attn_valid_exceeds_key_len() {
        let m = cross_attention_mask(2, 3, 100);
        assert!(m.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cross_attn_batched_known() {
        let m = cross_attention_mask_batched(2, 3, &[3, 1]);
        assert!(m[0..6].iter().all(|&v| v == 0.0));
        for i in 0..2 {
            assert_eq!(m[6 + i * 3], 0.0);
            assert!(is_neg_inf(m[6 + i * 3 + 1]));
            assert!(is_neg_inf(m[6 + i * 3 + 2]));
        }
    }

    #[test]
    fn cross_attn_batched_empty() {
        let m = cross_attention_mask_batched(2, 3, &[]);
        assert!(m.is_empty());
    }

    #[test]
    fn cross_attn_simd_boundary() {
        for kl in [7, 8, 9, 15, 16, 17] {
            let valid = kl / 2;
            let m = cross_attention_mask(4, kl, valid);
            for i in 0..4 {
                for j in 0..kl {
                    if j < valid {
                        assert_eq!(m[i * kl + j], 0.0, "kl={kl} ({i},{j})");
                    } else {
                        assert!(is_neg_inf(m[i * kl + j]), "kl={kl} ({i},{j})");
                    }
                }
            }
        }
    }

    // -- simd_combine_masks ----------------------------------------------

    #[test]
    fn combine_both_open() {
        let a = vec![0.0; 16];
        let b = vec![0.0; 16];
        let c = simd_combine_masks(&a, &b, 16);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn combine_one_blocks() {
        let a = vec![0.0, INF, 0.0, 0.0];
        let b = vec![0.0, 0.0, INF, 0.0];
        let c = simd_combine_masks(&a, &b, 4);
        assert_eq!(c[0], 0.0);
        assert!(is_neg_inf(c[1]));
        assert!(is_neg_inf(c[2]));
        assert_eq!(c[3], 0.0);
    }

    #[test]
    fn combine_both_block() {
        let n = 16;
        let a = vec![INF; n];
        let b = vec![INF; n];
        let c = simd_combine_masks(&a, &b, n);
        assert!(c.iter().all(|&v| is_neg_inf(v)));
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn combine_too_short() {
        simd_combine_masks(&[0.0; 2], &[0.0; 8], 8);
    }

    #[test]
    fn combine_simd_boundary() {
        for len in [7, 8, 9, 15, 16, 17, 32, 33] {
            let a = vec![1.0; len];
            let b = vec![2.0; len];
            let c = simd_combine_masks(&a, &b, len);
            for (i, &v) in c.iter().enumerate() {
                assert!((v - 3.0).abs() < 1e-6, "len={len} idx={i}: got {v}");
            }
        }
    }

    // -- Integration / cross-cutting -------------------------------------

    #[test]
    fn causal_plus_alibi_blocks_future() {
        let n = 4;
        let slopes = alibi_slopes(1);
        let causal = simd_causal_mask(n);
        let alibi = alibi_bias_matrix(n, slopes[0]);
        let combined = simd_combine_masks(&causal, &alibi, n * n);
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(is_neg_inf(combined[i * n + j]), "({i},{j})");
            }
        }
    }

    #[test]
    fn sliding_window_plus_global_expands_attention() {
        let n = 6;
        let sw = simd_sliding_window_mask(n, 2);
        let lf = longformer_mask(n, 2, &[0]);
        let sw_open = sw.iter().filter(|&&v| v == 0.0).count();
        let lf_open = lf.iter().filter(|&&v| v == 0.0).count();
        assert!(lf_open >= sw_open, "{lf_open} vs {sw_open}");
    }

    #[test]
    fn prefix_mask_generation_tokens_are_causal() {
        let n = 8;
        let prefix = 3;
        let m = prefix_mask(n, prefix);
        for i in prefix..n {
            for j in (i + 1)..n {
                assert!(is_neg_inf(m[i * n + j]), "gen {i} blocks future {j}");
            }
            for j in 0..=i {
                assert_eq!(m[i * n + j], 0.0, "gen {i} attends {j}");
            }
        }
    }

    #[test]
    fn block_sparse_causal_subset_of_non_causal() {
        let n = 8;
        let bs = 3;
        let causal = block_sparse_mask(n, bs, true);
        let non_causal = block_sparse_mask(n, bs, false);
        for k in 0..n * n {
            if causal[k] == 0.0 {
                assert_eq!(non_causal[k], 0.0, "causal open at {k}");
            }
        }
    }

    #[test]
    fn end_to_end_batched_apply() {
        let seq = 4;
        let batch = 2;
        let mask = simd_causal_mask(seq);
        let mut scores = vec![1.0; batch * seq * seq];
        simd_apply_mask_batched(&mut scores, &mask, batch, seq);

        for b in 0..batch {
            for i in 0..seq {
                let row_start = b * seq * seq + i * seq;
                let row = &scores[row_start..row_start + seq];
                let probs = softmax(row);
                for j in (i + 1)..seq {
                    assert!(probs[j] < 1e-6, "batch {b} row {i} col {j}");
                }
            }
        }
    }

    #[test]
    fn cross_attn_rows_identical() {
        let qlen = 5;
        let klen = 8;
        let valid = 3;
        let m = cross_attention_mask(qlen, klen, valid);
        let first_row = &m[0..klen];
        for i in 1..qlen {
            assert_eq!(&m[i * klen..(i + 1) * klen], first_row, "row {i}");
        }
    }
}
