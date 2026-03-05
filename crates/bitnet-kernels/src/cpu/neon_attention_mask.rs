#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON optimized attention mask operations for Apple Silicon.
//!
//! Provides SIMD-accelerated attention masking: causal (triangular), padding,
//! sliding window, ALiBi position bias, prefix LM, and combined masks. All
//! functions operate on row-major `[seq_len, kv_len]` score matrices in-place,
//! writing `f32::NEG_INFINITY` to masked positions.
//!
//! # NEON intrinsics used
//!
//! | Intrinsic      | Purpose                                  |
//! |----------------|------------------------------------------|
//! | `vld1q_f32`    | 128-bit (4×f32) load                     |
//! | `vst1q_f32`    | 128-bit (4×f32) store                    |
//! | `vdupq_n_f32`  | Broadcast scalar to four lanes           |
//! | `vdupq_n_u32`  | Broadcast u32 to four lanes              |
//! | `vaddq_f32`    | Lane-wise add                            |
//! | `vmulq_f32`    | Lane-wise multiply                       |
//! | `vcltq_u32`    | Lane-wise unsigned less-than compare     |
//! | `vbslq_f32`    | Bitwise select (blend)                   |

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ── Causal mask ────────────────────────────────────────────────────────

/// Apply causal (lower-triangular) mask to an attention score matrix.
///
/// For query position `q` and key position `k`, the position is masked
/// (set to `NEG_INFINITY`) when `k > q + offset` where
/// `offset = kv_len - seq_len` (allows for KV cache prefix).
///
/// `scores` is row-major `[seq_len, kv_len]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_causal_mask_f32(scores: &mut [f32], seq_len: usize, kv_len: usize) {
    assert!(
        scores.len() >= seq_len * kv_len,
        "scores too small: need {} got {}",
        seq_len * kv_len,
        scores.len()
    );

    let offset = kv_len.saturating_sub(seq_len);
    let neg_inf = f32::NEG_INFINITY;

    for q in 0..seq_len {
        let row_start = q * kv_len;
        let boundary = q + offset + 1; // first masked position
        if boundary >= kv_len {
            continue; // entire row visible
        }
        let mask_start = boundary;
        let mask_len = kv_len - mask_start;
        let dst = &mut scores[row_start + mask_start..row_start + kv_len];

        let chunks = mask_len / LANES;
        let remainder = mask_len % LANES;

        unsafe {
            let v_neg_inf = vdupq_n_f32(neg_inf);
            for i in 0..chunks {
                vst1q_f32(dst.as_mut_ptr().add(i * LANES), v_neg_inf);
            }
        }
        for i in 0..remainder {
            dst[chunks * LANES + i] = neg_inf;
        }
    }
}

// ── Padding mask ───────────────────────────────────────────────────────

/// Apply a padding mask to an attention score matrix.
///
/// `mask` has length `kv_len`; `mask[k] == false` means position `k` is
/// padding and every query's score at that key position is set to
/// `NEG_INFINITY`.
///
/// `scores` is row-major `[seq_len, kv_len]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_padding_mask_f32(scores: &mut [f32], mask: &[bool], seq_len: usize, kv_len: usize) {
    assert!(
        scores.len() >= seq_len * kv_len,
        "scores too small: need {} got {}",
        seq_len * kv_len,
        scores.len()
    );
    assert!(mask.len() >= kv_len, "mask too short: need {kv_len} got {}", mask.len());

    let neg_inf = f32::NEG_INFINITY;
    let chunks = kv_len / LANES;
    let remainder = kv_len % LANES;

    for q in 0..seq_len {
        let row_start = q * kv_len;

        unsafe {
            let v_neg_inf = vdupq_n_f32(neg_inf);
            for c in 0..chunks {
                let base = c * LANES;
                let m0 = if mask[base] { 0xFFFF_FFFFu32 } else { 0u32 };
                let m1 = if mask[base + 1] { 0xFFFF_FFFFu32 } else { 0u32 };
                let m2 = if mask[base + 2] { 0xFFFF_FFFFu32 } else { 0u32 };
                let m3 = if mask[base + 3] { 0xFFFF_FFFFu32 } else { 0u32 };

                let sel: uint32x4_t = {
                    let arr = [m0, m1, m2, m3];
                    vld1q_u32(arr.as_ptr())
                };

                let v_scores = vld1q_f32(scores.as_ptr().add(row_start + base));
                let blended = vbslq_f32(sel, v_scores, v_neg_inf);
                vst1q_f32(scores.as_mut_ptr().add(row_start + base), blended);
            }
        }

        // Scalar remainder
        let tail_start = chunks * LANES;
        for k in tail_start..kv_len {
            if !mask[k] {
                scores[row_start + k] = neg_inf;
            }
        }
    }
}

// ── Sliding window mask ────────────────────────────────────────────────

/// Apply sliding window attention mask.
///
/// Each query at position `q` can attend to keys in
/// `[q + offset - window_size + 1 .. q + offset]` (inclusive), where
/// `offset = kv_len - seq_len`. Positions outside this window are set to
/// `NEG_INFINITY`. This implicitly includes causal masking.
#[cfg(target_arch = "aarch64")]
pub fn neon_sliding_window_mask_f32(
    scores: &mut [f32],
    seq_len: usize,
    kv_len: usize,
    window_size: usize,
) {
    assert!(
        scores.len() >= seq_len * kv_len,
        "scores too small: need {} got {}",
        seq_len * kv_len,
        scores.len()
    );
    assert!(window_size > 0, "window_size must be > 0");

    let offset = kv_len.saturating_sub(seq_len);
    let neg_inf = f32::NEG_INFINITY;

    for q in 0..seq_len {
        let row_start = q * kv_len;
        let abs_pos = q + offset;

        // Visible range: [win_start, win_end] inclusive
        let win_end = abs_pos; // causal: cannot see future
        let win_start = (abs_pos + 1).saturating_sub(window_size);

        // Mask everything before win_start
        if win_start > 0 {
            let n = win_start.min(kv_len);
            let dst = &mut scores[row_start..row_start + n];
            let chunks = n / LANES;
            let rem = n % LANES;
            unsafe {
                let v_neg_inf = vdupq_n_f32(neg_inf);
                for i in 0..chunks {
                    vst1q_f32(dst.as_mut_ptr().add(i * LANES), v_neg_inf);
                }
            }
            for i in 0..rem {
                dst[chunks * LANES + i] = neg_inf;
            }
        }

        // Mask everything after win_end
        if win_end + 1 < kv_len {
            let start = win_end + 1;
            let n = kv_len - start;
            let dst = &mut scores[row_start + start..row_start + kv_len];
            let chunks = n / LANES;
            let rem = n % LANES;
            unsafe {
                let v_neg_inf = vdupq_n_f32(neg_inf);
                for i in 0..chunks {
                    vst1q_f32(dst.as_mut_ptr().add(i * LANES), v_neg_inf);
                }
            }
            for i in 0..rem {
                dst[chunks * LANES + i] = neg_inf;
            }
        }
    }
}

// ── ALiBi mask ─────────────────────────────────────────────────────────

/// Compute the ALiBi slope for a given head index.
///
/// Uses the geometric series `2^(-8/n * (h+1))` where `n = num_heads`.
#[inline]
fn alibi_slope(num_heads: usize, head_idx: usize) -> f32 {
    let n = num_heads as f32;
    let ratio = 8.0 / n;
    2.0_f32.powf(-ratio * (head_idx as f32 + 1.0))
}

/// Apply ALiBi (Attention with Linear Biases) position bias.
///
/// Adds `slope * (k - q - offset)` to each score, producing a linear
/// distance-based penalty. `offset = kv_len - seq_len`.
///
/// `scores` is row-major `[seq_len, kv_len]` for a single head.
#[cfg(target_arch = "aarch64")]
pub fn neon_alibi_mask_f32(
    scores: &mut [f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    head_idx: usize,
) {
    assert!(
        scores.len() >= seq_len * kv_len,
        "scores too small: need {} got {}",
        seq_len * kv_len,
        scores.len()
    );
    assert!(head_idx < num_heads, "head_idx {head_idx} >= num_heads {num_heads}");

    let slope = alibi_slope(num_heads, head_idx);
    let offset = kv_len.saturating_sub(seq_len);

    let chunks = kv_len / LANES;
    let remainder = kv_len % LANES;

    for q in 0..seq_len {
        let row_start = q * kv_len;
        let q_abs = (q + offset) as f32;

        unsafe {
            let v_slope = vdupq_n_f32(slope);
            let v_q = vdupq_n_f32(q_abs);

            for c in 0..chunks {
                let base = c * LANES;
                let k_vals: [f32; 4] =
                    [base as f32, (base + 1) as f32, (base + 2) as f32, (base + 3) as f32];
                let v_k = vld1q_f32(k_vals.as_ptr());
                let v_dist = vsubq_f32(v_k, v_q);
                let v_bias = vmulq_f32(v_slope, v_dist);

                let v_scores = vld1q_f32(scores.as_ptr().add(row_start + base));
                let v_result = vaddq_f32(v_scores, v_bias);
                vst1q_f32(scores.as_mut_ptr().add(row_start + base), v_result);
            }
        }

        // Scalar tail
        let tail_start = chunks * LANES;
        for k in tail_start..kv_len {
            let dist = k as f32 - q_abs;
            scores[row_start + k] += slope * dist;
        }
    }
}

// ── Combined mask ──────────────────────────────────────────────────────

/// Apply multiple mask types in a single pass.
///
/// Combines causal masking, optional padding mask, and optional sliding
/// window into one traversal over the score matrix. This avoids redundant
/// memory passes compared to calling each individually.
#[cfg(target_arch = "aarch64")]
pub fn neon_combined_mask_f32(
    scores: &mut [f32],
    causal: bool,
    padding_mask: Option<&[bool]>,
    window: Option<usize>,
    seq_len: usize,
    kv_len: usize,
) {
    assert!(
        scores.len() >= seq_len * kv_len,
        "scores too small: need {} got {}",
        seq_len * kv_len,
        scores.len()
    );
    if let Some(pm) = padding_mask {
        assert!(pm.len() >= kv_len, "padding_mask too short: need {kv_len} got {}", pm.len());
    }
    if let Some(w) = window {
        assert!(w > 0, "window_size must be > 0");
    }

    let offset = kv_len.saturating_sub(seq_len);
    let neg_inf = f32::NEG_INFINITY;

    for q in 0..seq_len {
        let row_start = q * kv_len;
        let abs_pos = q + offset;

        // Determine visible range from causal + window
        let vis_start = if let Some(w) = window { (abs_pos + 1).saturating_sub(w) } else { 0 };
        let vis_end = if causal { abs_pos } else { kv_len.saturating_sub(1) };

        // Mask before visible start
        if vis_start > 0 {
            let n = vis_start.min(kv_len);
            fill_neg_inf(&mut scores[row_start..row_start + n]);
        }

        // Mask after visible end
        if vis_end + 1 < kv_len {
            let start = vis_end + 1;
            fill_neg_inf(&mut scores[row_start + start..row_start + kv_len]);
        }

        // Apply padding mask within the visible window
        if let Some(pm) = padding_mask {
            let lo = vis_start;
            let hi = (vis_end + 1).min(kv_len);
            for k in lo..hi {
                if !pm[k] {
                    scores[row_start + k] = neg_inf;
                }
            }
        }
    }
}

/// NEON-accelerated fill of a slice with `NEG_INFINITY`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn fill_neg_inf(dst: &mut [f32]) {
    let n = dst.len();
    let chunks = n / LANES;
    let rem = n % LANES;

    unsafe {
        let v = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            vst1q_f32(dst.as_mut_ptr().add(i * LANES), v);
        }
    }
    for i in 0..rem {
        dst[chunks * LANES + i] = f32::NEG_INFINITY;
    }
}

// ── Prefix LM mask ────────────────────────────────────────────────────

/// Apply prefix LM mask.
///
/// Positions `0..prefix_len` are fully visible (bidirectional) while
/// positions `prefix_len..seq_len` use causal masking.
///
/// `scores` is row-major `[seq_len, kv_len]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_prefix_mask_f32(scores: &mut [f32], prefix_len: usize, seq_len: usize, kv_len: usize) {
    assert!(
        scores.len() >= seq_len * kv_len,
        "scores too small: need {} got {}",
        seq_len * kv_len,
        scores.len()
    );
    assert!(prefix_len <= seq_len, "prefix_len {prefix_len} > seq_len {seq_len}");

    let offset = kv_len.saturating_sub(seq_len);

    // Prefix rows (q < prefix_len): bidirectional among prefix tokens
    for q in 0..prefix_len {
        let row_start = q * kv_len;
        let boundary = prefix_len + offset;
        if boundary < kv_len {
            fill_neg_inf(&mut scores[row_start + boundary..row_start + kv_len]);
        }
    }

    // Suffix rows (q >= prefix_len): causal masking
    for q in prefix_len..seq_len {
        let row_start = q * kv_len;
        let boundary = q + offset + 1;
        if boundary < kv_len {
            fill_neg_inf(&mut scores[row_start + boundary..row_start + kv_len]);
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    const NEG_INF: f32 = f32::NEG_INFINITY;

    fn ones(seq_len: usize, kv_len: usize) -> Vec<f32> {
        vec![1.0; seq_len * kv_len]
    }

    fn zeros(seq_len: usize, kv_len: usize) -> Vec<f32> {
        vec![0.0; seq_len * kv_len]
    }

    fn is_masked(v: f32) -> bool {
        v == NEG_INF
    }

    fn is_visible(v: f32) -> bool {
        v.is_finite()
    }

    // ── Causal mask tests ──────────────────────────────────────────

    #[test]
    fn test_causal_mask_identity_square() {
        let mut s = ones(4, 4);
        neon_causal_mask_f32(&mut s, 4, 4);
        for q in 0..4 {
            for k in 0..4 {
                if k <= q {
                    assert!(is_visible(s[q * 4 + k]), "q={q} k={k} should be visible");
                } else {
                    assert!(is_masked(s[q * 4 + k]), "q={q} k={k} should be masked");
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_1x1() {
        let mut s = ones(1, 1);
        neon_causal_mask_f32(&mut s, 1, 1);
        assert!(is_visible(s[0]));
    }

    #[test]
    fn test_causal_mask_1x8() {
        let mut s = ones(1, 8);
        neon_causal_mask_f32(&mut s, 1, 8);
        // offset=7, boundary=0+7+1=8 → all visible
        for k in 0..8 {
            assert!(is_visible(s[k]), "k={k}");
        }
    }

    #[test]
    fn test_causal_mask_seq1_kv1() {
        let mut s = vec![5.0];
        neon_causal_mask_f32(&mut s, 1, 1);
        assert_eq!(s[0], 5.0);
    }

    #[test]
    fn test_causal_mask_wide_kv() {
        // seq=2, kv=6 → offset=4
        let mut s = ones(2, 6);
        neon_causal_mask_f32(&mut s, 2, 6);
        // q=0: boundary=5 → k=0..4 visible, k=5 masked
        for k in 0..5 {
            assert!(is_visible(s[k]), "q=0 k={k}");
        }
        assert!(is_masked(s[5]), "q=0 k=5");
        // q=1: boundary=6 → all visible
        for k in 0..6 {
            assert!(is_visible(s[6 + k]), "q=1 k={k}");
        }
    }

    #[test]
    fn test_causal_mask_preserves_values() {
        let mut s: Vec<f32> = (0..16).map(|i| i as f32).collect();
        neon_causal_mask_f32(&mut s, 4, 4);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[4], 4.0);
        assert_eq!(s[5], 5.0);
        assert_eq!(s[8], 8.0);
    }

    #[test]
    fn test_causal_mask_large() {
        let n = 16;
        let mut s = ones(n, n);
        neon_causal_mask_f32(&mut s, n, n);
        for q in 0..n {
            for k in 0..n {
                if k <= q {
                    assert!(is_visible(s[q * n + k]), "q={q} k={k}");
                } else {
                    assert!(is_masked(s[q * n + k]), "q={q} k={k}");
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_non_square() {
        let mut s = ones(3, 5);
        neon_causal_mask_f32(&mut s, 3, 5);
        // offset=2; q=0: vis 0..2; q=1: vis 0..3; q=2: all
        assert!(is_visible(s[0]));
        assert!(is_visible(s[2]));
        assert!(is_masked(s[3]));
        assert!(is_masked(s[4]));
        for k in 0..4 {
            assert!(is_visible(s[5 + k]), "q=1 k={k}");
        }
        assert!(is_masked(s[9]));
        for k in 0..5 {
            assert!(is_visible(s[10 + k]), "q=2 k={k}");
        }
    }

    #[test]
    fn test_causal_mask_size_5() {
        let n = 5;
        let mut s = ones(n, n);
        neon_causal_mask_f32(&mut s, n, n);
        for q in 0..n {
            for k in 0..n {
                assert_eq!(is_visible(s[q * n + k]), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_causal_mask_size_7() {
        let n = 7;
        let mut s = ones(n, n);
        neon_causal_mask_f32(&mut s, n, n);
        for q in 0..n {
            for k in 0..n {
                assert_eq!(is_visible(s[q * n + k]), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_causal_mask_idempotent() {
        let mut s = ones(4, 4);
        neon_causal_mask_f32(&mut s, 4, 4);
        let first = s.clone();
        neon_causal_mask_f32(&mut s, 4, 4);
        assert_eq!(s, first);
    }

    #[test]
    fn test_causal_mask_seq_gt_kv() {
        let mut s = ones(4, 2);
        neon_causal_mask_f32(&mut s, 4, 2);
        // offset=0; q=0: boundary=1 → k=0 vis, k=1 masked
        assert!(is_visible(s[0]));
        assert!(is_masked(s[1]));
        // q=1+: boundary>=2 → all vis
        for i in 2..8 {
            assert!(is_visible(s[i]));
        }
    }

    // ── Padding mask tests ─────────────────────────────────────────

    #[test]
    fn test_padding_mask_all_visible() {
        let mask = vec![true; 4];
        let mut s = ones(2, 4);
        neon_padding_mask_f32(&mut s, &mask, 2, 4);
        for v in &s {
            assert!(is_visible(*v));
        }
    }

    #[test]
    fn test_padding_mask_all_masked() {
        let mask = vec![false; 4];
        let mut s = ones(2, 4);
        neon_padding_mask_f32(&mut s, &mask, 2, 4);
        for v in &s {
            assert!(is_masked(*v));
        }
    }

    #[test]
    fn test_padding_mask_last_two_padded() {
        let mask = vec![true, true, false, false];
        let mut s = ones(2, 4);
        neon_padding_mask_f32(&mut s, &mask, 2, 4);
        for q in 0..2 {
            assert!(is_visible(s[q * 4]));
            assert!(is_visible(s[q * 4 + 1]));
            assert!(is_masked(s[q * 4 + 2]));
            assert!(is_masked(s[q * 4 + 3]));
        }
    }

    #[test]
    fn test_padding_mask_alternating() {
        let mask = vec![true, false, true, false, true, false, true, false];
        let mut s = ones(1, 8);
        neon_padding_mask_f32(&mut s, &mask, 1, 8);
        for k in 0..8 {
            if k % 2 == 0 {
                assert!(is_visible(s[k]), "k={k}");
            } else {
                assert!(is_masked(s[k]), "k={k}");
            }
        }
    }

    #[test]
    fn test_padding_mask_single_element() {
        let mask = vec![false];
        let mut s = vec![42.0];
        neon_padding_mask_f32(&mut s, &mask, 1, 1);
        assert!(is_masked(s[0]));
    }

    #[test]
    fn test_padding_mask_preserves_unmasked_values() {
        let mask = vec![true, false, true, false];
        let mut s = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        neon_padding_mask_f32(&mut s, &mask, 2, 4);
        assert_eq!(s[0], 1.0);
        assert!(is_masked(s[1]));
        assert_eq!(s[2], 3.0);
        assert!(is_masked(s[3]));
    }

    #[test]
    fn test_padding_mask_wide() {
        let mut mask = vec![true; 16];
        mask[3] = false;
        mask[7] = false;
        mask[11] = false;
        mask[15] = false;
        let mut s = ones(2, 16);
        neon_padding_mask_f32(&mut s, &mask, 2, 16);
        for q in 0..2 {
            for k in 0..16 {
                if k % 4 == 3 {
                    assert!(is_masked(s[q * 16 + k]), "q={q} k={k}");
                } else {
                    assert!(is_visible(s[q * 16 + k]), "q={q} k={k}");
                }
            }
        }
    }

    #[test]
    fn test_padding_mask_size_5_remainder() {
        let mask = vec![true, true, true, true, false];
        let mut s = ones(1, 5);
        neon_padding_mask_f32(&mut s, &mask, 1, 5);
        for k in 0..4 {
            assert!(is_visible(s[k]));
        }
        assert!(is_masked(s[4]));
    }

    #[test]
    fn test_padding_mask_idempotent() {
        let mask = vec![true, false, true, false];
        let mut s = ones(1, 4);
        neon_padding_mask_f32(&mut s, &mask, 1, 4);
        let first = s.clone();
        neon_padding_mask_f32(&mut s, &mask, 1, 4);
        assert_eq!(s, first);
    }

    // ── Sliding window mask tests ──────────────────────────────────

    #[test]
    fn test_sliding_window_full() {
        let mut s = ones(4, 4);
        neon_sliding_window_mask_f32(&mut s, 4, 4, 4);
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_sliding_window_size_1() {
        let mut s = ones(4, 4);
        neon_sliding_window_mask_f32(&mut s, 4, 4, 1);
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), k == q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_sliding_window_size_2() {
        let mut s = ones(4, 4);
        neon_sliding_window_mask_f32(&mut s, 4, 4, 2);
        let expected = [
            [true, false, false, false],
            [true, true, false, false],
            [false, true, true, false],
            [false, false, true, true],
        ];
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), expected[q][k], "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_sliding_window_1x1() {
        let mut s = vec![5.0];
        neon_sliding_window_mask_f32(&mut s, 1, 1, 1);
        assert_eq!(s[0], 5.0);
    }

    #[test]
    fn test_sliding_window_with_kv_offset() {
        // seq=2, kv=6, window=2, offset=4
        let mut s = ones(2, 6);
        neon_sliding_window_mask_f32(&mut s, 2, 6, 2);
        for k in 0..6 {
            assert_eq!(is_visible(s[k]), k == 3 || k == 4, "q=0 k={k}");
        }
        for k in 0..6 {
            assert_eq!(is_visible(s[6 + k]), k == 4 || k == 5, "q=1 k={k}");
        }
    }

    #[test]
    fn test_sliding_window_large_16() {
        let n = 16;
        let w = 4;
        let mut s = ones(n, n);
        neon_sliding_window_mask_f32(&mut s, n, n, w);
        for q in 0..n {
            let win_start = if q + 1 >= w { q + 1 - w } else { 0 };
            for k in 0..n {
                let expected = k >= win_start && k <= q;
                assert_eq!(is_visible(s[q * n + k]), expected, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_sliding_window_preserves_visible_values() {
        let mut s: Vec<f32> = (0..16).map(|i| i as f32).collect();
        neon_sliding_window_mask_f32(&mut s, 4, 4, 4);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[4], 4.0);
        assert_eq!(s[5], 5.0);
    }

    #[test]
    fn test_sliding_window_size_3() {
        let mut s = ones(5, 5);
        neon_sliding_window_mask_f32(&mut s, 5, 5, 3);
        for q in 0..5 {
            let win_start = if q + 1 >= 3 { q + 1 - 3 } else { 0 };
            for k in 0..5 {
                assert_eq!(is_visible(s[q * 5 + k]), k >= win_start && k <= q, "q={q} k={k}");
            }
        }
    }

    // ── ALiBi mask tests ───────────────────────────────────────────

    #[test]
    fn test_alibi_slope_single_head() {
        let s = alibi_slope(1, 0);
        let expected = 1.0 / 256.0;
        assert!((s - expected).abs() < 1e-7, "slope={s} expected={expected}");
    }

    #[test]
    fn test_alibi_slope_8_heads() {
        for h in 0..8 {
            let s = alibi_slope(8, h);
            let expected = 2.0_f32.powi(-((h as i32) + 1));
            assert!((s - expected).abs() < 1e-7, "head={h} slope={s} expected={expected}");
        }
    }

    #[test]
    fn test_alibi_slope_decreasing() {
        for n in [1, 2, 4, 8, 16] {
            for h in 1..n {
                assert!(alibi_slope(n, h) < alibi_slope(n, h - 1));
            }
        }
    }

    #[test]
    fn test_alibi_mask_zero_distance() {
        let mut s = zeros(4, 4);
        neon_alibi_mask_f32(&mut s, 4, 4, 8, 0);
        for q in 0..4 {
            assert!(s[q * 4 + q].abs() < 1e-6, "q={q} self-attn should be ~0");
        }
    }

    #[test]
    fn test_alibi_mask_negative_for_past() {
        let mut s = zeros(4, 4);
        neon_alibi_mask_f32(&mut s, 4, 4, 8, 0);
        let slope = alibi_slope(8, 0);
        for q in 1..4 {
            for k in 0..q {
                let expected = slope * (k as f32 - q as f32);
                assert!((s[q * 4 + k] - expected).abs() < 1e-5, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_alibi_mask_positive_for_future() {
        let mut s = zeros(4, 4);
        neon_alibi_mask_f32(&mut s, 4, 4, 8, 0);
        let slope = alibi_slope(8, 0);
        for q in 0..3 {
            for k in (q + 1)..4 {
                let expected = slope * (k as f32 - q as f32);
                assert!((s[q * 4 + k] - expected).abs() < 1e-5, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_alibi_mask_different_heads() {
        let mut s0 = zeros(4, 4);
        let mut s7 = zeros(4, 4);
        neon_alibi_mask_f32(&mut s0, 4, 4, 8, 0);
        neon_alibi_mask_f32(&mut s7, 4, 4, 8, 7);
        let b0 = s0[3 * 4];
        let b7 = s7[3 * 4];
        // head 0 has larger slope → more negative bias at distance 3
        assert!(b0 < b7, "head 0 bias={b0} should be more negative than head 7 bias={b7}");
    }

    #[test]
    fn test_alibi_mask_with_offset() {
        let mut s = zeros(2, 4);
        neon_alibi_mask_f32(&mut s, 2, 4, 8, 0);
        let slope = alibi_slope(8, 0);
        for k in 0..4 {
            let expected = slope * (k as f32 - 2.0);
            assert!((s[k] - expected).abs() < 1e-5, "q=0 k={k}");
        }
    }

    #[test]
    fn test_alibi_mask_additive() {
        let mut s = vec![10.0; 4];
        neon_alibi_mask_f32(&mut s, 1, 4, 8, 0);
        let slope = alibi_slope(8, 0);
        // q_abs=3; k=3: dist=0 → score stays 10
        assert!((s[3] - 10.0).abs() < 1e-5);
        assert!(s[0] < 10.0);
    }

    #[test]
    fn test_alibi_mask_1x1() {
        let mut s = vec![0.0];
        neon_alibi_mask_f32(&mut s, 1, 1, 1, 0);
        assert!(s[0].abs() < 1e-6);
    }

    #[test]
    fn test_alibi_mask_wide_neon_path() {
        let mut s = zeros(1, 16);
        neon_alibi_mask_f32(&mut s, 1, 16, 8, 3);
        let slope = alibi_slope(8, 3);
        let q_abs = 15.0;
        for k in 0..16 {
            let expected = slope * (k as f32 - q_abs);
            assert!((s[k] - expected).abs() < 1e-4, "k={k}");
        }
    }

    // ── Combined mask tests ────────────────────────────────────────

    #[test]
    fn test_combined_causal_only() {
        let mut s = ones(4, 4);
        neon_combined_mask_f32(&mut s, true, None, None, 4, 4);
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_combined_no_mask() {
        let mut s = ones(3, 3);
        neon_combined_mask_f32(&mut s, false, None, None, 3, 3);
        for v in &s {
            assert!(is_visible(*v));
        }
    }

    #[test]
    fn test_combined_padding_only() {
        let mask = vec![true, true, false, false];
        let mut s = ones(2, 4);
        neon_combined_mask_f32(&mut s, false, Some(&mask), None, 2, 4);
        for q in 0..2 {
            assert!(is_visible(s[q * 4]));
            assert!(is_visible(s[q * 4 + 1]));
            assert!(is_masked(s[q * 4 + 2]));
            assert!(is_masked(s[q * 4 + 3]));
        }
    }

    #[test]
    fn test_combined_window_only() {
        let mut s = ones(4, 4);
        neon_combined_mask_f32(&mut s, false, None, Some(2), 4, 4);
        let expected = [
            [true, true, true, true],
            [true, true, true, true],
            [false, true, true, true],
            [false, false, true, true],
        ];
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), expected[q][k], "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_combined_causal_plus_padding() {
        let mask = vec![true, false, true, true];
        let mut s = ones(4, 4);
        neon_combined_mask_f32(&mut s, true, Some(&mask), None, 4, 4);
        for q in 0..4 {
            for k in 0..4 {
                let expected = k <= q && mask[k];
                assert_eq!(is_visible(s[q * 4 + k]), expected, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_combined_causal_plus_window() {
        let mut s = ones(4, 4);
        neon_combined_mask_f32(&mut s, true, None, Some(2), 4, 4);
        let expected = [
            [true, false, false, false],
            [true, true, false, false],
            [false, true, true, false],
            [false, false, true, true],
        ];
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), expected[q][k], "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_combined_all_three() {
        let mask = vec![false, true, true, true];
        let mut s = ones(4, 4);
        neon_combined_mask_f32(&mut s, true, Some(&mask), Some(2), 4, 4);
        let expected = [
            [false, false, false, false],
            [false, true, false, false],
            [false, true, true, false],
            [false, false, true, true],
        ];
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), expected[q][k], "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_combined_1x1() {
        let mut s = vec![5.0];
        neon_combined_mask_f32(&mut s, true, None, None, 1, 1);
        assert_eq!(s[0], 5.0);
    }

    #[test]
    fn test_combined_large_16x16_causal_window() {
        let n = 16;
        let w = 4;
        let mut s = ones(n, n);
        neon_combined_mask_f32(&mut s, true, None, Some(w), n, n);
        for q in 0..n {
            let win_start = if q + 1 >= w { q + 1 - w } else { 0 };
            for k in 0..n {
                assert_eq!(is_visible(s[q * n + k]), k >= win_start && k <= q, "q={q} k={k}");
            }
        }
    }

    // ── Prefix mask tests ──────────────────────────────────────────

    #[test]
    fn test_prefix_mask_full_prefix() {
        let mut s = ones(4, 4);
        neon_prefix_mask_f32(&mut s, 4, 4, 4);
        for v in &s {
            assert!(is_visible(*v));
        }
    }

    #[test]
    fn test_prefix_mask_no_prefix() {
        let mut s = ones(4, 4);
        neon_prefix_mask_f32(&mut s, 0, 4, 4);
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_prefix_mask_half() {
        let mut s = ones(4, 4);
        neon_prefix_mask_f32(&mut s, 2, 4, 4);
        // q=0,1 (prefix): boundary=2 → sees 0,1
        assert!(is_visible(s[0]));
        assert!(is_visible(s[1]));
        assert!(is_masked(s[2]));
        assert!(is_masked(s[3]));
        assert!(is_visible(s[4]));
        assert!(is_visible(s[5]));
        assert!(is_masked(s[6]));
        assert!(is_masked(s[7]));
        // q=2 (suffix): sees 0,1,2
        assert!(is_visible(s[8]));
        assert!(is_visible(s[9]));
        assert!(is_visible(s[10]));
        assert!(is_masked(s[11]));
        // q=3: sees all
        for k in 12..16 {
            assert!(is_visible(s[k]));
        }
    }

    #[test]
    fn test_prefix_mask_1x1() {
        let mut s = vec![5.0];
        neon_prefix_mask_f32(&mut s, 1, 1, 1);
        assert_eq!(s[0], 5.0);
    }

    #[test]
    fn test_prefix_mask_with_offset() {
        let mut s = ones(2, 4);
        neon_prefix_mask_f32(&mut s, 1, 2, 4);
        // q=0 (prefix): boundary=1+2=3 → sees 0,1,2
        assert!(is_visible(s[0]));
        assert!(is_visible(s[1]));
        assert!(is_visible(s[2]));
        assert!(is_masked(s[3]));
        // q=1 (suffix): boundary=1+2+1=4 → all visible
        for k in 0..4 {
            assert!(is_visible(s[4 + k]), "k={k}");
        }
    }

    #[test]
    fn test_prefix_mask_single_prefix_token() {
        let mut s = ones(3, 3);
        neon_prefix_mask_f32(&mut s, 1, 3, 3);
        assert!(is_visible(s[0]));
        assert!(is_masked(s[1]));
        assert!(is_masked(s[2]));
        assert!(is_visible(s[3]));
        assert!(is_visible(s[4]));
        assert!(is_masked(s[5]));
        for k in 6..9 {
            assert!(is_visible(s[k]));
        }
    }

    #[test]
    fn test_prefix_mask_large() {
        let n = 16;
        let prefix = 8;
        let mut s = ones(n, n);
        neon_prefix_mask_f32(&mut s, prefix, n, n);
        for q in 0..n {
            for k in 0..n {
                let expected = if q < prefix { k < prefix } else { k <= q };
                assert_eq!(is_visible(s[q * n + k]), expected, "q={q} k={k}");
            }
        }
    }

    // ── Edge case tests ────────────────────────────────────────────

    #[test]
    fn test_causal_mask_2x2() {
        let mut s = ones(2, 2);
        neon_causal_mask_f32(&mut s, 2, 2);
        assert!(is_visible(s[0]));
        assert!(is_masked(s[1]));
        assert!(is_visible(s[2]));
        assert!(is_visible(s[3]));
    }

    #[test]
    fn test_causal_mask_3x3() {
        let mut s = ones(3, 3);
        neon_causal_mask_f32(&mut s, 3, 3);
        let expected = [[true, false, false], [true, true, false], [true, true, true]];
        for q in 0..3 {
            for k in 0..3 {
                assert_eq!(is_visible(s[q * 3 + k]), expected[q][k], "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_sliding_window_larger_than_seq() {
        let mut s = ones(4, 4);
        neon_sliding_window_mask_f32(&mut s, 4, 4, 100);
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_padding_mask_seq1() {
        let mask = vec![true, false];
        let mut s = ones(1, 2);
        neon_padding_mask_f32(&mut s, &mask, 1, 2);
        assert!(is_visible(s[0]));
        assert!(is_masked(s[1]));
    }

    #[test]
    fn test_combined_empty_masks() {
        let mut s = ones(2, 2);
        let orig = s.clone();
        neon_combined_mask_f32(&mut s, false, None, None, 2, 2);
        assert_eq!(s, orig);
    }

    #[test]
    fn test_alibi_mask_16_heads() {
        let slopes: Vec<f32> = (0..16).map(|h| alibi_slope(16, h)).collect();
        for i in 1..16 {
            assert!(slopes[i] < slopes[i - 1], "slopes not decreasing at {i}");
        }
    }

    #[test]
    fn test_causal_mask_neon_boundary_9() {
        let mut s = ones(1, 9);
        neon_causal_mask_f32(&mut s, 1, 9);
        for k in 0..9 {
            assert!(is_visible(s[k]));
        }
    }

    #[test]
    fn test_sliding_window_non_square() {
        let mut s = ones(2, 8);
        neon_sliding_window_mask_f32(&mut s, 2, 8, 3);
        let offset = 6;
        for q in 0..2 {
            let abs_pos = q + offset;
            let win_start = if abs_pos + 1 >= 3 { abs_pos + 1 - 3 } else { 0 };
            for k in 0..8 {
                assert_eq!(is_visible(s[q * 8 + k]), k >= win_start && k <= abs_pos, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_combined_causal_padding_large() {
        let n = 16;
        let mut mask = vec![true; n];
        mask[0] = false;
        mask[n - 1] = false;
        let mut s = ones(n, n);
        neon_combined_mask_f32(&mut s, true, Some(&mask), None, n, n);
        for q in 0..n {
            for k in 0..n {
                assert_eq!(is_visible(s[q * n + k]), k <= q && mask[k], "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_prefix_mask_preserves_values() {
        let mut s: Vec<f32> = (0..9).map(|i| i as f32).collect();
        neon_prefix_mask_f32(&mut s, 2, 3, 3);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 1.0);
        assert!(is_masked(s[2]));
    }

    #[test]
    fn test_causal_mask_negative_values() {
        let mut s = vec![-1.0, -2.0, -3.0, -4.0];
        neon_causal_mask_f32(&mut s, 2, 2);
        assert_eq!(s[0], -1.0);
        assert!(is_masked(s[1]));
        assert_eq!(s[2], -3.0);
        assert_eq!(s[3], -4.0);
    }

    #[test]
    fn test_sliding_window_idempotent() {
        let mut s = ones(4, 4);
        neon_sliding_window_mask_f32(&mut s, 4, 4, 2);
        let first = s.clone();
        neon_sliding_window_mask_f32(&mut s, 4, 4, 2);
        assert_eq!(s, first);
    }

    #[test]
    fn test_combined_window1_causal() {
        let mut s = ones(4, 4);
        neon_combined_mask_f32(&mut s, true, None, Some(1), 4, 4);
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(is_visible(s[q * 4 + k]), k == q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_alibi_mask_symmetry_around_diagonal() {
        let mut s = zeros(8, 8);
        neon_alibi_mask_f32(&mut s, 8, 8, 8, 0);
        let slope = alibi_slope(8, 0);
        let b_minus = s[4 * 8 + 2];
        let b_plus = s[4 * 8 + 6];
        assert!((b_minus.abs() - b_plus.abs()).abs() < 1e-5);
        assert!((b_minus - slope * (-2.0)).abs() < 1e-5);
        assert!((b_plus - slope * 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_fill_neg_inf_helper() {
        let mut buf = vec![1.0; 17];
        fill_neg_inf(&mut buf);
        for v in &buf {
            assert!(is_masked(*v));
        }
    }

    #[test]
    fn test_padding_mask_large_kv_13() {
        let mut mask = vec![true; 13];
        mask[12] = false;
        let mut s = ones(1, 13);
        neon_padding_mask_f32(&mut s, &mask, 1, 13);
        for k in 0..12 {
            assert!(is_visible(s[k]), "k={k}");
        }
        assert!(is_masked(s[12]));
    }

    #[test]
    fn test_prefix_mask_idempotent() {
        let mut s = ones(4, 4);
        neon_prefix_mask_f32(&mut s, 2, 4, 4);
        let first = s.clone();
        neon_prefix_mask_f32(&mut s, 2, 4, 4);
        assert_eq!(s, first);
    }

    #[test]
    fn test_alibi_linear_scaling() {
        let mut s = zeros(1, 8);
        neon_alibi_mask_f32(&mut s, 1, 8, 4, 1);
        let slope = alibi_slope(4, 1);
        let q_abs = 7.0;
        for k in 0..8 {
            let expected = slope * (k as f32 - q_abs);
            assert!((s[k] - expected).abs() < 1e-4, "k={k}");
        }
    }

    #[test]
    fn test_combined_all_disabled_large() {
        let n = 32;
        let mut s = ones(n, n);
        let orig = s.clone();
        neon_combined_mask_f32(&mut s, false, None, None, n, n);
        assert_eq!(s, orig);
    }

    #[test]
    fn test_causal_mask_exact_values() {
        let mut s = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        neon_causal_mask_f32(&mut s, 3, 3);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], NEG_INF);
        assert_eq!(s[2], NEG_INF);
        assert_eq!(s[3], 4.0);
        assert_eq!(s[4], 5.0);
        assert_eq!(s[5], NEG_INF);
        assert_eq!(s[6], 7.0);
        assert_eq!(s[7], 8.0);
        assert_eq!(s[8], 9.0);
    }

    #[test]
    fn test_causal_mask_32x32() {
        let n = 32;
        let mut s = ones(n, n);
        neon_causal_mask_f32(&mut s, n, n);
        for q in 0..n {
            for k in 0..n {
                assert_eq!(is_visible(s[q * n + k]), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_padding_mask_first_padded() {
        let mask = vec![false, true, true, true];
        let mut s = ones(2, 4);
        neon_padding_mask_f32(&mut s, &mask, 2, 4);
        for q in 0..2 {
            assert!(is_masked(s[q * 4]), "q={q} k=0");
            for k in 1..4 {
                assert!(is_visible(s[q * 4 + k]), "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_sliding_window_seq1_kv8() {
        let mut s = ones(1, 8);
        neon_sliding_window_mask_f32(&mut s, 1, 8, 3);
        // offset=7, abs=7, win_start=5, win_end=7
        for k in 0..8 {
            assert_eq!(is_visible(s[k]), k >= 5, "k={k}");
        }
    }

    #[test]
    fn test_prefix_mask_all_suffix() {
        // prefix=0 same as causal
        let n = 8;
        let mut s = ones(n, n);
        neon_prefix_mask_f32(&mut s, 0, n, n);
        for q in 0..n {
            for k in 0..n {
                assert_eq!(is_visible(s[q * n + k]), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_alibi_mask_2_heads() {
        let s0 = alibi_slope(2, 0);
        let s1 = alibi_slope(2, 1);
        // 2^(-4*1) = 1/16, 2^(-4*2) = 1/256
        assert!((s0 - 1.0 / 16.0).abs() < 1e-7);
        assert!((s1 - 1.0 / 256.0).abs() < 1e-7);
    }

    #[test]
    fn test_combined_window_padding_no_causal() {
        let mask = vec![true, false, true, true];
        let mut s = ones(4, 4);
        neon_combined_mask_f32(&mut s, false, Some(&mask), Some(3), 4, 4);
        // No causal: vis_end=3; window limits vis_start
        // q=0: abs=0, start=0, end=3 → padding k=1 masked
        // q=1: abs=1, start=0, end=3 → padding k=1 masked
        // q=2: abs=2, start=0, end=3 → padding k=1 masked
        // q=3: abs=3, start=1, end=3 → k=0 window-masked, k=1 padding-masked
        for q in 0..3 {
            assert!(is_visible(s[q * 4]), "q={q} k=0");
            assert!(is_masked(s[q * 4 + 1]), "q={q} k=1");
            assert!(is_visible(s[q * 4 + 2]), "q={q} k=2");
            assert!(is_visible(s[q * 4 + 3]), "q={q} k=3");
        }
        assert!(is_masked(s[3 * 4]), "q=3 k=0 window");
        assert!(is_masked(s[3 * 4 + 1]), "q=3 k=1 padding");
        assert!(is_visible(s[3 * 4 + 2]), "q=3 k=2");
        assert!(is_visible(s[3 * 4 + 3]), "q=3 k=3");
    }

    #[test]
    fn test_prefix_mask_boundary_neon() {
        // prefix=3, seq=8, kv=8 to exercise NEON path in fill_neg_inf
        let n = 8;
        let prefix = 3;
        let mut s = ones(n, n);
        neon_prefix_mask_f32(&mut s, prefix, n, n);
        for q in 0..n {
            for k in 0..n {
                let expected = if q < prefix { k < prefix } else { k <= q };
                assert_eq!(is_visible(s[q * n + k]), expected, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_causal_mask_seq1_kv16() {
        let mut s = ones(1, 16);
        neon_causal_mask_f32(&mut s, 1, 16);
        // offset=15, boundary=16 → all visible
        for k in 0..16 {
            assert!(is_visible(s[k]));
        }
    }

    #[test]
    fn test_sliding_window_size_equal_seq() {
        let n = 6;
        let mut s = ones(n, n);
        neon_sliding_window_mask_f32(&mut s, n, n, n);
        // Same as causal
        for q in 0..n {
            for k in 0..n {
                assert_eq!(is_visible(s[q * n + k]), k <= q, "q={q} k={k}");
            }
        }
    }

    #[test]
    fn test_combined_causal_window_large() {
        let n = 32;
        let w = 8;
        let mut s = ones(n, n);
        neon_combined_mask_f32(&mut s, true, None, Some(w), n, n);
        for q in 0..n {
            let win_start = if q + 1 >= w { q + 1 - w } else { 0 };
            for k in 0..n {
                assert_eq!(is_visible(s[q * n + k]), k >= win_start && k <= q, "q={q} k={k}");
            }
        }
    }
}
