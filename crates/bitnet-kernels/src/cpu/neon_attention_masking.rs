#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::let_and_return)]
//! ARM NEON optimized attention masking kernels for Apple Silicon.
//!
//! Provides SIMD-accelerated attention mask application using `float32x4`
//! NEON intrinsics for 4-wide parallel computation.  Supports causal
//! (lower-triangular) masks, padding masks for variable-length sequences,
//! sliding window attention masks, fused masked softmax with numerical
//! stability, and element-wise mask combination.
//!
//! Each function processes data in chunks of 4 (`f32x4`) and falls back to
//! scalar code for any tail elements whose count is not a multiple of 4.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Causal mask ─────────────────────────────────────────────────────────

/// Apply a lower-triangular causal mask to attention scores in-place.
///
/// For an attention matrix of shape `[seq_len, head_dim]`, positions where
/// `col > row` are replaced with `mask_value` (typically `-inf`).  NEON
/// `vbslq_f32` is used for branchless conditional selection.
///
/// # Safety
/// Caller must ensure `scores.len() >= seq_len * head_dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_apply_causal_mask_f32(
    scores: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    mask_value: f32,
) {
    let total = seq_len * head_dim;
    if total == 0 || scores.len() < total {
        return;
    }
    for row in 0..seq_len {
        let row_start = row * head_dim;
        // Positions 0..=row are kept, positions (row+1)..head_dim are masked.
        let first_masked = row + 1;
        if first_masked >= head_dim {
            continue; // entire row is unmasked
        }
        let mask_start = row_start + first_masked;
        let mask_end = row_start + head_dim;
        let slice = &mut scores[mask_start..mask_end];
        let len = slice.len();
        let chunks = len / LANES;
        let remainder = len % LANES;

        let mask_vec = vdupq_n_f32(mask_value);
        let ptr = slice.as_mut_ptr();
        for i in 0..chunks {
            let offset = i * LANES;
            vst1q_f32(ptr.add(offset), mask_vec);
        }
        for i in 0..remainder {
            *ptr.add(chunks * LANES + i) = mask_value;
        }
    }
}

// ── Padding mask ────────────────────────────────────────────────────────

/// Apply a padding mask to attention scores in-place.
///
/// `padding_mask` has length `seq_len` where `true` means the position is
/// **padded** (i.e. should be masked out).  For each head, every column
/// that corresponds to a padded position is set to `mask_value`.
///
/// Layout: `scores[head * seq_len * seq_len + row * seq_len + col]`
///
/// # Safety
/// Caller must ensure `scores.len() >= num_heads * seq_len * seq_len` and
/// `padding_mask.len() >= seq_len`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_apply_padding_mask_f32(
    scores: &mut [f32],
    padding_mask: &[bool],
    seq_len: usize,
    num_heads: usize,
    mask_value: f32,
) {
    if seq_len == 0 || padding_mask.len() < seq_len {
        return;
    }
    let mask_vec = vdupq_n_f32(mask_value);
    let head_size = seq_len * seq_len;

    for head in 0..num_heads {
        let head_offset = head * head_size;
        for row in 0..seq_len {
            let row_offset = head_offset + row * seq_len;
            // Build a per-column mask for this row based on padding_mask.
            let ptr = scores[row_offset..].as_mut_ptr();
            let chunks = seq_len / LANES;
            let remainder = seq_len % LANES;

            for c in 0..chunks {
                let base = c * LANES;
                // Build u32 bitmask: 0xFFFF_FFFF where padded, 0 otherwise.
                let m0 = if padding_mask[base] { u32::MAX } else { 0u32 };
                let m1 = if padding_mask[base + 1] { u32::MAX } else { 0u32 };
                let m2 = if padding_mask[base + 2] { u32::MAX } else { 0u32 };
                let m3 = if padding_mask[base + 3] { u32::MAX } else { 0u32 };
                let bitmask = vld1q_u32([m0, m1, m2, m3].as_ptr());
                let orig = vld1q_f32(ptr.add(base));
                let result = vbslq_f32(bitmask, mask_vec, orig);
                vst1q_f32(ptr.add(base), result);
            }
            for i in 0..remainder {
                let col = chunks * LANES + i;
                if padding_mask[col] {
                    *ptr.add(col) = mask_value;
                }
            }
        }
    }
}

// ── Sliding window mask ─────────────────────────────────────────────────

/// Apply a sliding window attention mask in-place.
///
/// Positions where `col < row.saturating_sub(window_size - 1)` or
/// `col > row` are set to `mask_value`.  This combines a causal constraint
/// with a limited lookback window.
///
/// Layout: `scores[row * seq_len + col]` for a single head.
///
/// # Safety
/// Caller must ensure `scores.len() >= seq_len * seq_len`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_apply_sliding_window_mask_f32(
    scores: &mut [f32],
    seq_len: usize,
    window_size: usize,
    mask_value: f32,
) {
    if seq_len == 0 {
        return;
    }
    let mask_vec = vdupq_n_f32(mask_value);

    for row in 0..seq_len {
        let row_start = row * seq_len;
        let win_start = if window_size == 0 {
            row // window_size 0 means only the diagonal is visible
        } else {
            row.saturating_sub(window_size - 1)
        };
        let win_end = row + 1; // causal: only attend up to current position

        // Mask columns before the window
        if win_start > 0 {
            let ptr = scores[row_start..].as_mut_ptr();
            let len = win_start;
            let chunks = len / LANES;
            let remainder = len % LANES;
            for i in 0..chunks {
                vst1q_f32(ptr.add(i * LANES), mask_vec);
            }
            for i in 0..remainder {
                *ptr.add(chunks * LANES + i) = mask_value;
            }
        }

        // Mask columns after the causal boundary
        if win_end < seq_len {
            let ptr = scores[row_start + win_end..].as_mut_ptr();
            let len = seq_len - win_end;
            let chunks = len / LANES;
            let remainder = len % LANES;
            for i in 0..chunks {
                vst1q_f32(ptr.add(i * LANES), mask_vec);
            }
            for i in 0..remainder {
                *ptr.add(chunks * LANES + i) = mask_value;
            }
        }
    }
}

// ── Fused masked softmax ────────────────────────────────────────────────

/// Numerically stable softmax with mask applied in a single fused pass.
///
/// `mask` has length `seq_len` — `true` means **masked** (excluded from
/// softmax).  The algorithm:
/// 1. Find max of unmasked elements.
/// 2. Compute `exp(x - max)` for unmasked, 0 for masked.
/// 3. Sum exponentials, normalise.
///
/// If all elements are masked the output is all zeros.
///
/// # Safety
/// Caller must ensure `scores.len() >= seq_len` and `mask.len() >= seq_len`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_masked_softmax_f32(scores: &mut [f32], mask: &[bool], seq_len: usize) {
    if seq_len == 0 {
        return;
    }
    let len = seq_len.min(scores.len()).min(mask.len());

    // ── Pass 1: find max of unmasked elements ───────────────────────
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..len {
        if !mask[i] && scores[i] > max_val {
            max_val = scores[i];
        }
    }
    if max_val == f32::NEG_INFINITY {
        // All masked — zero out.
        let chunks = len / LANES;
        let remainder = len % LANES;
        let zero = vdupq_n_f32(0.0);
        let ptr = scores.as_mut_ptr();
        for i in 0..chunks {
            vst1q_f32(ptr.add(i * LANES), zero);
        }
        for i in 0..remainder {
            *ptr.add(chunks * LANES + i) = 0.0;
        }
        return;
    }

    // ── Pass 2: exp(x - max) for unmasked, 0 for masked ────────────
    let mut sum = 0.0f32;
    for i in 0..len {
        if mask[i] {
            scores[i] = 0.0;
        } else {
            let e = (scores[i] - max_val).exp();
            scores[i] = e;
            sum += e;
        }
    }

    // ── Pass 3: normalise ───────────────────────────────────────────
    if sum > 0.0 {
        let chunks = len / LANES;
        let remainder = len % LANES;
        let inv_sum = vdupq_n_f32(1.0 / sum);
        let ptr = scores.as_mut_ptr();
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * LANES));
            vst1q_f32(ptr.add(i * LANES), vmulq_f32(v, inv_sum));
        }
        let inv = 1.0 / sum;
        for i in 0..remainder {
            let idx = chunks * LANES + i;
            *ptr.add(idx) *= inv;
        }
    }
}

// ── Combine masks ───────────────────────────────────────────────────────

/// Element-wise combine two mask tensors using `vminq_f32`.
///
/// For attention masks stored as float (0.0 = keep, -inf = mask), the
/// element-wise minimum produces the intersection of the two masks.
///
/// # Safety
/// Caller must ensure all slices have at least `causal.len()` elements
/// and `output.len() >= causal.len()` and `padding.len() >= causal.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_combine_masks_f32(causal: &[f32], padding: &[f32], output: &mut [f32]) {
    let len = causal.len().min(padding.len()).min(output.len());
    let chunks = len / LANES;
    let remainder = len % LANES;

    let cp = causal.as_ptr();
    let pp = padding.as_ptr();
    let op = output.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        let a = vld1q_f32(cp.add(offset));
        let b = vld1q_f32(pp.add(offset));
        vst1q_f32(op.add(offset), vminq_f32(a, b));
    }
    for i in 0..remainder {
        let idx = chunks * LANES + i;
        *op.add(idx) = (*cp.add(idx)).min(*pp.add(idx));
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: call the unsafe causal mask function.
    #[cfg(target_arch = "aarch64")]
    fn apply_causal(scores: &mut [f32], seq_len: usize, head_dim: usize, mv: f32) {
        unsafe { neon_apply_causal_mask_f32(scores, seq_len, head_dim, mv) }
    }

    /// Helper: call the unsafe padding mask function.
    #[cfg(target_arch = "aarch64")]
    fn apply_padding(scores: &mut [f32], pmask: &[bool], seq_len: usize, nh: usize, mv: f32) {
        unsafe { neon_apply_padding_mask_f32(scores, pmask, seq_len, nh, mv) }
    }

    /// Helper: call the unsafe sliding window mask function.
    #[cfg(target_arch = "aarch64")]
    fn apply_sliding(scores: &mut [f32], seq_len: usize, ws: usize, mv: f32) {
        unsafe { neon_apply_sliding_window_mask_f32(scores, seq_len, ws, mv) }
    }

    /// Helper: call the unsafe masked softmax function.
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax(scores: &mut [f32], mask: &[bool], seq_len: usize) {
        unsafe { neon_masked_softmax_f32(scores, mask, seq_len) }
    }

    /// Helper: call the unsafe combine masks function.
    #[cfg(target_arch = "aarch64")]
    fn combine(causal: &[f32], padding: &[f32], output: &mut [f32]) {
        unsafe { neon_combine_masks_f32(causal, padding, output) }
    }

    const NEG_INF: f32 = f32::NEG_INFINITY;

    // ── Causal mask tests ───────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_size_1() {
        let mut s = [1.0];
        apply_causal(&mut s, 1, 1, NEG_INF);
        assert_eq!(s, vec![1.0]); // diagonal kept
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_size_2() {
        // 2×2: [[a,b],[c,d]] → b masked
        let mut s = vec![1.0, 2.0, 3.0, 4.0];
        apply_causal(&mut s, 2, 2, NEG_INF);
        assert_eq!(s[0], 1.0);
        assert!(s[1].is_infinite() && s[1] < 0.0);
        assert_eq!(s[2], 3.0);
        assert_eq!(s[3], 4.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_size_4() {
        let mut s = [1.0; 16];
        apply_causal(&mut s, 4, 4, NEG_INF);
        // Row 0: keep col 0, mask 1-3
        assert_eq!(s[0], 1.0);
        assert!(s[1] == NEG_INF);
        assert!(s[2] == NEG_INF);
        assert!(s[3] == NEG_INF);
        // Row 1: keep 0-1, mask 2-3
        assert_eq!(s[4], 1.0);
        assert_eq!(s[5], 1.0);
        assert!(s[6] == NEG_INF);
        assert!(s[7] == NEG_INF);
        // Row 3: all kept
        assert_eq!(s[12], 1.0);
        assert_eq!(s[13], 1.0);
        assert_eq!(s[14], 1.0);
        assert_eq!(s[15], 1.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_size_8() {
        let n = 8;
        let mut s = vec![1.0; n * n];
        apply_causal(&mut s, n, n, NEG_INF);
        for r in 0..n {
            for c in 0..n {
                if c > r {
                    assert!(s[r * n + c] == NEG_INF);
                } else {
                    assert_eq!(s[r * n + c], 1.0);
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_size_16() {
        let n = 16;
        let mut s = vec![0.5; n * n];
        apply_causal(&mut s, n, n, NEG_INF);
        for r in 0..n {
            for c in 0..n {
                if c > r {
                    assert!(s[r * n + c] == NEG_INF);
                } else {
                    assert_eq!(s[r * n + c], 0.5);
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_non_square_check() {
        // seq_len=2, head_dim=4 → 2 rows, 4 cols
        let mut s = [1.0; 8];
        apply_causal(&mut s, 2, 4, NEG_INF);
        // Row 0: keep col 0, mask 1-3
        assert_eq!(s[0], 1.0);
        assert!(s[1] == NEG_INF);
        assert!(s[2] == NEG_INF);
        assert!(s[3] == NEG_INF);
        // Row 1: keep 0-1, mask 2-3
        assert_eq!(s[4], 1.0);
        assert_eq!(s[5], 1.0);
        assert!(s[6] == NEG_INF);
        assert!(s[7] == NEG_INF);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_mask_value_neg_inf() {
        let mut s = [5.0; 4];
        apply_causal(&mut s, 2, 2, NEG_INF);
        assert!(s[1] == NEG_INF);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_mask_value_zero() {
        let mut s = [5.0; 4];
        apply_causal(&mut s, 2, 2, 0.0);
        assert_eq!(s[1], 0.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_already_masked() {
        let mut s = [NEG_INF; 4];
        apply_causal(&mut s, 2, 2, NEG_INF);
        // Diagonal kept (already NEG_INF from input, but function only writes above diag)
        assert!(s[0] == NEG_INF);
        assert!(s[1] == NEG_INF);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_diagonal() {
        let n = 4;
        let mut s: Vec<f32> = (0..16).map(|i| i as f32).collect();
        apply_causal(&mut s, n, n, NEG_INF);
        // Diagonal elements preserved
        for i in 0..n {
            assert_eq!(s[i * n + i], (i * n + i) as f32);
        }
    }

    // ── Padding mask tests ──────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn padding_mask_no_padding() {
        let mut s = [1.0; 4];
        let pmask = vec![false, false];
        apply_padding(&mut s, &pmask, 2, 1, NEG_INF);
        assert_eq!(s, vec![1.0; 4]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn padding_mask_all_padding() {
        let mut s = [1.0; 4];
        let pmask = vec![true, true];
        apply_padding(&mut s, &pmask, 2, 1, NEG_INF);
        for v in &s {
            assert!(*v == NEG_INF);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn padding_mask_half_padding() {
        // seq=4, 1 head, last 2 positions padded
        let n = 4;
        let mut s = vec![1.0; n * n];
        let pmask = vec![false, false, true, true];
        apply_padding(&mut s, &pmask, n, 1, NEG_INF);
        for r in 0..n {
            assert_eq!(s[r * n], 1.0);
            assert_eq!(s[r * n + 1], 1.0);
            assert!(s[r * n + 2] == NEG_INF);
            assert!(s[r * n + 3] == NEG_INF);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn padding_mask_single_token() {
        let mut s = [5.0; 1];
        let pmask = vec![false];
        apply_padding(&mut s, &pmask, 1, 1, NEG_INF);
        assert_eq!(s[0], 5.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn padding_mask_multi_head() {
        let n = 2;
        let nh = 2;
        let mut s = vec![1.0; nh * n * n];
        let pmask = vec![false, true];
        apply_padding(&mut s, &pmask, n, nh, NEG_INF);
        // Head 0, row 0: col 0 kept, col 1 masked
        assert_eq!(s[0], 1.0);
        assert!(s[1] == NEG_INF);
        // Head 1, row 1: col 0 kept, col 1 masked
        assert_eq!(s[4 + 0], 1.0);
        assert!(s[4 + 1] == NEG_INF);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn padding_mask_alternating() {
        let n = 4;
        let mut s = vec![1.0; n * n];
        let pmask = vec![true, false, true, false];
        apply_padding(&mut s, &pmask, n, 1, NEG_INF);
        for r in 0..n {
            assert!(s[r * n] == NEG_INF);
            assert_eq!(s[r * n + 1], 1.0);
            assert!(s[r * n + 2] == NEG_INF);
            assert_eq!(s[r * n + 3], 1.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn padding_mask_last_token() {
        let n = 4;
        let mut s = vec![1.0; n * n];
        let pmask = vec![false, false, false, true];
        apply_padding(&mut s, &pmask, n, 1, NEG_INF);
        for r in 0..n {
            assert_eq!(s[r * n + 2], 1.0);
            assert!(s[r * n + 3] == NEG_INF);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn padding_mask_first_token() {
        let n = 4;
        let mut s = vec![1.0; n * n];
        let pmask = vec![true, false, false, false];
        apply_padding(&mut s, &pmask, n, 1, NEG_INF);
        for r in 0..n {
            assert!(s[r * n] == NEG_INF);
            assert_eq!(s[r * n + 1], 1.0);
        }
    }

    // ── Sliding window mask tests ───────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_window_1() {
        // Window 1: each row only attends to itself
        let n = 4;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, 1, NEG_INF);
        for r in 0..n {
            for c in 0..n {
                if c == r {
                    assert_eq!(s[r * n + c], 1.0, "row={r} col={c}");
                } else {
                    assert!(s[r * n + c] == NEG_INF, "row={r} col={c}");
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_window_2() {
        let n = 4;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, 2, NEG_INF);
        // Row 0: [1, -inf, -inf, -inf]
        assert_eq!(s[0], 1.0);
        assert!(s[1] == NEG_INF);
        // Row 1: [1, 1, -inf, -inf]
        assert_eq!(s[4], 1.0);
        assert_eq!(s[5], 1.0);
        assert!(s[6] == NEG_INF);
        // Row 3: [-inf, -inf, 1, 1]
        assert!(s[12] == NEG_INF);
        assert!(s[13] == NEG_INF);
        assert_eq!(s[14], 1.0);
        assert_eq!(s[15], 1.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_window_4() {
        let n = 4;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, 4, NEG_INF);
        // Window >= seq_len → same as causal mask
        for r in 0..n {
            for c in 0..n {
                if c <= r {
                    assert_eq!(s[r * n + c], 1.0);
                } else {
                    assert!(s[r * n + c] == NEG_INF);
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_full_window() {
        let n = 8;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, n, NEG_INF);
        // Full window = causal mask
        for r in 0..n {
            for c in 0..n {
                if c <= r {
                    assert_eq!(s[r * n + c], 1.0);
                } else {
                    assert!(s[r * n + c] == NEG_INF);
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_larger_than_seq() {
        let n = 4;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, 100, NEG_INF);
        // Window > seq → same as causal
        for r in 0..n {
            for c in 0..n {
                if c <= r {
                    assert_eq!(s[r * n + c], 1.0);
                } else {
                    assert!(s[r * n + c] == NEG_INF);
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_zero_window() {
        // Window 0: only diagonal
        let n = 4;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, 0, NEG_INF);
        for r in 0..n {
            for c in 0..n {
                if c == r {
                    assert_eq!(s[r * n + c], 1.0, "row={r} col={c}");
                } else {
                    assert!(s[r * n + c] == NEG_INF, "row={r} col={c}");
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_non_power_of_two() {
        let n = 5;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, 3, NEG_INF);
        // Row 4: window [2,3,4] — cols 0,1 masked
        assert!(s[4 * n] == NEG_INF);
        assert!(s[4 * n + 1] == NEG_INF);
        assert_eq!(s[4 * n + 2], 1.0);
        assert_eq!(s[4 * n + 3], 1.0);
        assert_eq!(s[4 * n + 4], 1.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_boundary() {
        let n = 8;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, 3, NEG_INF);
        // Row 7: window [5,6,7] — cols 0-4 masked, 5-7 kept
        for c in 0..5 {
            assert!(s[7 * n + c] == NEG_INF, "col={c}");
        }
        for c in 5..8 {
            assert_eq!(s[7 * n + c], 1.0, "col={c}");
        }
    }

    // ── Masked softmax tests ────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax_uniform() {
        let mut s = [1.0; 4];
        let mask = [false; 4];
        masked_softmax(&mut s, &mask, 4);
        for v in &s {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax_peaked() {
        let mut s = vec![0.0, 0.0, 10.0, 0.0];
        let mask = [false; 4];
        masked_softmax(&mut s, &mask, 4);
        // Element 2 should dominate
        assert!(s[2] > 0.9);
        assert!(s[0] < 0.01);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax_all_masked_except_one() {
        let mut s = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, true, false, true];
        masked_softmax(&mut s, &mask, 4);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 0.0);
        assert!((s[2] - 1.0).abs() < 1e-5);
        assert_eq!(s[3], 0.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax_no_mask() {
        let mut s = vec![1.0, 2.0, 3.0];
        let mask = [false; 3];
        masked_softmax(&mut s, &mask, 3);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax_large_values() {
        let mut s = vec![1000.0, 1000.0, 1000.0, 1000.0];
        let mask = [false; 4];
        masked_softmax(&mut s, &mask, 4);
        for v in &s {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax_negative_values() {
        let mut s = vec![-1.0, -2.0, -3.0, -4.0];
        let mask = [false; 4];
        masked_softmax(&mut s, &mask, 4);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // First element should be largest
        assert!(s[0] > s[1]);
        assert!(s[1] > s[2]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax_single_element() {
        let mut s = [5.0];
        let mask = vec![false];
        masked_softmax(&mut s, &mask, 1);
        assert!((s[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax_stability() {
        // Very large spread should not produce NaN or Inf
        let mut s = vec![1e10, -1e10, 0.0, 1.0];
        let mask = [false; 4];
        masked_softmax(&mut s, &mask, 4);
        for v in &s {
            assert!(v.is_finite());
        }
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ── Combine masks tests ─────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn combine_masks_both_ones() {
        let a = [0.0; 4];
        let b = [0.0; 4];
        let mut out = [999.0; 4];
        combine(&a, &b, &mut out);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn combine_masks_one_zero() {
        let a = [0.0; 4];
        let b = [NEG_INF; 4];
        let mut out = [0.0; 4];
        combine(&a, &b, &mut out);
        for v in &out {
            assert!(*v == NEG_INF);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn combine_masks_both_zero() {
        let a = [NEG_INF; 4];
        let b = [NEG_INF; 4];
        let mut out = [0.0; 4];
        combine(&a, &b, &mut out);
        for v in &out {
            assert!(*v == NEG_INF);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn combine_masks_mixed() {
        let a = vec![0.0, NEG_INF, 0.0, NEG_INF];
        let b = vec![NEG_INF, 0.0, 0.0, NEG_INF];
        let mut out = [0.0; 4];
        combine(&a, &b, &mut out);
        assert!(out[0] == NEG_INF);
        assert!(out[1] == NEG_INF);
        assert_eq!(out[2], 0.0);
        assert!(out[3] == NEG_INF);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn combine_masks_large() {
        let n = 17; // not a multiple of 4
        let a = vec![0.0; n];
        let b = vec![0.0; n];
        let mut out = vec![999.0; n];
        combine(&a, &b, &mut out);
        assert_eq!(out, vec![0.0; n]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn combine_masks_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [f32::INFINITY; 5];
        let mut out = [0.0; 5];
        combine(&a, &b, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn combine_masks_symmetry() {
        let a = vec![1.0, -2.0, 3.0, -4.0];
        let b = vec![-1.0, 2.0, -3.0, 4.0];
        let mut out1 = [0.0; 4];
        let mut out2 = [0.0; 4];
        combine(&a, &b, &mut out1);
        combine(&b, &a, &mut out2);
        assert_eq!(out1, out2); // min is commutative
    }

    // ── Integration tests ───────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn integration_causal_then_softmax() {
        let n = 4;
        let mut s = vec![1.0; n * n];
        apply_causal(&mut s, n, n, NEG_INF);

        // Apply softmax per row
        for r in 0..n {
            let start = r * n;
            let mask: Vec<bool> = (0..n).map(|c| c > r).collect();
            masked_softmax(&mut s[start..start + n], &mask, n);
            let sum: f32 = s[start..start + n].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {r} sum={sum}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn integration_padding_then_softmax() {
        let n = 4;
        let mut s = vec![1.0; n * n];
        let pmask = vec![false, false, true, true];
        apply_padding(&mut s, &pmask, n, 1, NEG_INF);

        for r in 0..n {
            let start = r * n;
            masked_softmax(&mut s[start..start + n], &pmask, n);
            let sum: f32 = s[start..start + n].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {r} sum={sum}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn integration_sliding_window_then_softmax() {
        let n = 8;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, 3, NEG_INF);

        for r in 0..n {
            let start = r * n;
            let mask: Vec<bool> = (0..n).map(|c| s[start + c] == NEG_INF).collect();
            masked_softmax(&mut s[start..start + n], &mask, n);
            let sum: f32 = s[start..start + n].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {r} sum={sum}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn integration_full_attention_pipeline() {
        let n = 4;
        let nh = 2;
        let head_sz = n * n;

        // Causal + padding for multi-head
        let mut s = vec![1.0; nh * head_sz];
        let pmask = vec![false, false, false, true]; // last token padded

        for h in 0..nh {
            let offset = h * head_sz;
            let head_scores = &mut s[offset..offset + head_sz];
            apply_causal(head_scores, n, n, NEG_INF);
        }
        apply_padding(&mut s, &pmask, n, nh, NEG_INF);

        // Softmax per row per head
        for h in 0..nh {
            for r in 0..n {
                let start = h * head_sz + r * n;
                let row = &mut s[start..start + n];
                let mask: Vec<bool> = row.iter().map(|v| *v == NEG_INF).collect();
                masked_softmax(row, &mask, n);
                let sum: f32 = row.iter().sum();
                // Row 3 has all cols masked except col 3, but col 3 is padded → all masked
                if r < 3 || pmask[..=r].iter().any(|p| !p) {
                    // At least one unmasked token exists
                    if sum > 0.0 {
                        assert!((sum - 1.0).abs() < 1e-4, "h={h} r={r} sum={sum}");
                    }
                }
            }
        }
    }

    // ── Edge case tests ─────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_empty_input_causal() {
        let mut s: Vec<f32> = vec![];
        apply_causal(&mut s, 0, 0, NEG_INF);
        assert!(s.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_empty_input_padding() {
        let mut s: Vec<f32> = vec![];
        let pmask: Vec<bool> = vec![];
        apply_padding(&mut s, &pmask, 0, 1, NEG_INF);
        assert!(s.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_empty_input_sliding() {
        let mut s: Vec<f32> = vec![];
        apply_sliding(&mut s, 0, 3, NEG_INF);
        assert!(s.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_empty_input_softmax() {
        let mut s: Vec<f32> = vec![];
        let mask: Vec<bool> = vec![];
        masked_softmax(&mut s, &mask, 0);
        assert!(s.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_empty_input_combine() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        combine(&a, &b, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_single_element_causal() {
        let mut s = [42.0];
        apply_causal(&mut s, 1, 1, NEG_INF);
        assert_eq!(s[0], 42.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_single_element_sliding() {
        let mut s = [42.0];
        apply_sliding(&mut s, 1, 1, NEG_INF);
        assert_eq!(s[0], 42.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_single_element_combine() {
        let a = [1.0];
        let b = [2.0];
        let mut out = [0.0];
        combine(&a, &b, &mut out);
        assert_eq!(out[0], 1.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_very_large_seq() {
        let n = 129; // prime, exceeds LANES multiples
        let mut s = vec![1.0; n * n];
        apply_causal(&mut s, n, n, NEG_INF);
        // Spot-check corners
        assert_eq!(s[0], 1.0);
        assert!(s[n - 1] == NEG_INF);
        assert_eq!(s[(n - 1) * n + n - 1], 1.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_numerical_precision() {
        // Softmax of identical values should give uniform distribution
        let n = 7; // odd, non-multiple of LANES
        let mut s = vec![3.14; n];
        let mask = vec![false; n];
        masked_softmax(&mut s, &mask, n);
        let expected = 1.0 / n as f32;
        for v in &s {
            assert!((v - expected).abs() < 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_combine_non_aligned() {
        // Length 7 — exercises tail scalar path
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut out = [0.0; 7];
        combine(&a, &b, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn edge_masked_softmax_all_masked() {
        let mut s = vec![1.0, 2.0, 3.0];
        let mask = vec![true, true, true];
        masked_softmax(&mut s, &mask, 3);
        assert_eq!(s, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_preserves_lower_triangle_values() {
        let n = 4;
        let mut s: Vec<f32> = (0..16).map(|x| x as f32 * 0.1).collect();
        let orig = s.clone();
        apply_causal(&mut s, n, n, NEG_INF);
        for r in 0..n {
            for c in 0..=r {
                assert_eq!(s[r * n + c], orig[r * n + c], "r={r} c={c}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_size_equals_seq() {
        let n = 4;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, n, NEG_INF);
        // Equivalent to causal mask
        let mut expected = vec![1.0; n * n];
        apply_causal(&mut expected, n, n, NEG_INF);
        assert_eq!(s, expected);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn padding_mask_custom_mask_value() {
        let n = 2;
        let mut s = vec![1.0; n * n];
        let pmask = vec![false, true];
        apply_padding(&mut s, &pmask, n, 1, -1e9);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], -1e9);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn masked_softmax_two_unmasked() {
        let mut s = vec![0.0, 0.0, 0.0, 0.0];
        let mask = vec![true, false, true, false];
        masked_softmax(&mut s, &mask, 4);
        assert_eq!(s[0], 0.0);
        assert!((s[1] - 0.5).abs() < 1e-5);
        assert_eq!(s[2], 0.0);
        assert!((s[3] - 0.5).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn combine_masks_with_finite_values() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![-4.0, -3.0, -2.0, -1.0];
        let mut out = [0.0; 4];
        combine(&a, &b, &mut out);
        assert_eq!(out, vec![-4.0, -3.0, -3.0, -4.0]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn causal_mask_size_3_odd() {
        let n = 3;
        let mut s = vec![1.0; n * n];
        apply_causal(&mut s, n, n, NEG_INF);
        // Row 0: [1, -inf, -inf]
        assert_eq!(s[0], 1.0);
        assert!(s[1] == NEG_INF);
        assert!(s[2] == NEG_INF);
        // Row 2: [1, 1, 1]
        assert_eq!(s[6], 1.0);
        assert_eq!(s[7], 1.0);
        assert_eq!(s[8], 1.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn sliding_window_3_on_seq_6() {
        let n = 6;
        let mut s = vec![1.0; n * n];
        apply_sliding(&mut s, n, 3, NEG_INF);
        // Row 5: window [3,4,5]
        for c in 0..3 {
            assert!(s[5 * n + c] == NEG_INF, "col {c} should be masked");
        }
        for c in 3..6 {
            assert_eq!(s[5 * n + c], 1.0, "col {c} should be kept");
        }
    }
}
