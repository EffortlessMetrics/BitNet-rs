//! NEON-optimized attention mask generation and application for
//! Apple Silicon.
//!
//! Provides six mask operations with ARM NEON SIMD acceleration and
//! scalar fallback for non-aarch64 targets (CI runs on x86_64):
//!
//! 1. [`causal_mask`] — upper-triangular causal mask for autoregressive
//!    attention
//! 2. [`apply_mask`] — apply pre-computed mask to attention scores
//!    (masked positions → −∞)
//! 3. [`sliding_window_mask`] — sliding window attention mask with
//!    configurable window size
//! 4. [`prefix_mask`] — prefix-style mask (prefix tokens attend to all,
//!    suffix is causal)
//! 5. [`block_sparse_mask`] — block-sparse attention pattern
//! 6. [`combine_masks`] — combine multiple masks with AND/OR operations

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
#[allow(dead_code)]
const LANES: usize = 4;

/// How to combine two boolean masks element-wise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineOp {
    /// Result is `true` only when both masks are `true`.
    And,
    /// Result is `true` when either mask is `true`.
    Or,
}

// ═══════════════════════════════════════════════════════════════════
// 1. causal_mask
// ═══════════════════════════════════════════════════════════════════

/// Generate an upper-triangular causal mask for autoregressive
/// attention.
///
/// `mask` must have length `seq_len * seq_len`.  Position `(i, j)` is
/// `true` (allowed) when `j <= i`, preventing tokens from attending to
/// future positions.
#[cfg(target_arch = "aarch64")]
pub fn causal_mask(mask: &mut [bool], seq_len: usize) {
    assert_eq!(mask.len(), seq_len * seq_len, "mask length must be seq_len^2");
    for (i, row) in mask.chunks_exact_mut(seq_len).enumerate().take(seq_len) {
        let allowed = i + 1; // columns 0..=i
        // NEON: write 16 bytes at a time (bool is 1 byte)
        let full_true = allowed / 16;
        let ptr = row.as_mut_ptr() as *mut u8;
        unsafe {
            let ones = vdupq_n_u8(1);
            for blk in 0..full_true {
                vst1q_u8(ptr.add(blk * 16), ones);
            }
        }
        for (j, val) in row.iter_mut().enumerate().take(allowed) {
            if j >= full_true * 16 {
                *val = true;
            }
        }
        for val in row.iter_mut().skip(allowed).take(seq_len - allowed) {
            *val = false;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn causal_mask(mask: &mut [bool], seq_len: usize) {
    assert_eq!(mask.len(), seq_len * seq_len, "mask length must be seq_len^2");
    for (i, row) in mask.chunks_exact_mut(seq_len).enumerate().take(seq_len) {
        for (j, val) in row.iter_mut().enumerate().take(seq_len) {
            *val = j <= i;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. apply_mask
// ═══════════════════════════════════════════════════════════════════

/// Apply a boolean mask to attention scores in-place.
///
/// Where `mask[i]` is `false` the corresponding `scores[i]` is set to
/// `f32::NEG_INFINITY`.  `true` positions are left untouched.
#[cfg(target_arch = "aarch64")]
pub fn apply_mask(scores: &mut [f32], mask: &[bool]) {
    assert_eq!(scores.len(), mask.len(), "scores and mask must have the same length");
    let n = scores.len();
    let chunks = n / LANES;

    // NEON path — process 4 f32 scores at a time.
    unsafe {
        let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
        for c in 0..chunks {
            let base = c * LANES;
            let s = vld1q_f32(scores.as_ptr().add(base));
            // Build a per-lane bitmask from bool values.
            let m0 = if mask[base] { u32::MAX } else { 0 };
            let m1 = if mask[base + 1] { u32::MAX } else { 0 };
            let m2 = if mask[base + 2] { u32::MAX } else { 0 };
            let m3 = if mask[base + 3] { u32::MAX } else { 0 };
            let mbits: [u32; 4] = [m0, m1, m2, m3];
            let mq = vld1q_u32(mbits.as_ptr());
            // Select: keep score when mask is true, else −∞.
            let res = vbslq_f32(mq, s, neg_inf);
            vst1q_f32(scores.as_mut_ptr().add(base), res);
        }
    }

    // Scalar tail.
    let tail_start = chunks * LANES;
    for (s, &m) in scores.iter_mut().skip(tail_start).zip(mask.iter().skip(tail_start)) {
        if !m {
            *s = f32::NEG_INFINITY;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn apply_mask(scores: &mut [f32], mask: &[bool]) {
    assert_eq!(scores.len(), mask.len(), "scores and mask must have the same length");
    for (s, &m) in scores.iter_mut().zip(mask.iter()) {
        if !m {
            *s = f32::NEG_INFINITY;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. sliding_window_mask
// ═══════════════════════════════════════════════════════════════════

/// Generate a sliding window attention mask.
///
/// Position `(i, j)` is `true` when `j <= i` **and**
/// `i - j < window_size`.  This is the intersection of causal and
/// a local-window constraint.
#[cfg(target_arch = "aarch64")]
pub fn sliding_window_mask(mask: &mut [bool], seq_len: usize, window_size: usize) {
    assert_eq!(mask.len(), seq_len * seq_len, "mask length must be seq_len^2");
    assert!(window_size > 0, "window_size must be > 0");
    for (i, row) in mask.chunks_exact_mut(seq_len).enumerate().take(seq_len) {
        let start = if i >= window_size { i - window_size + 1 } else { 0 };
        // Positions before the window: false
        for val in row.iter_mut().take(start) {
            *val = false;
        }
        // Within the window and causal: true  (start..=i)
        let end = i + 1; // exclusive upper bound
        let true_count = end.saturating_sub(start);
        // NEON: write 16 bools (bytes) at a time
        let full16 = true_count / 16;
        let ptr = row.as_mut_ptr() as *mut u8;
        unsafe {
            let ones = vdupq_n_u8(1);
            for blk in 0..full16 {
                vst1q_u8(ptr.add(start + blk * 16), ones);
            }
        }
        for (j, val) in
            row.iter_mut().enumerate().skip(start + full16 * 16).take(end - (start + full16 * 16))
        {
            let _ = j;
            *val = true;
        }
        // Positions after the causal boundary: false
        for val in row.iter_mut().skip(end).take(seq_len - end) {
            *val = false;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn sliding_window_mask(mask: &mut [bool], seq_len: usize, window_size: usize) {
    assert_eq!(mask.len(), seq_len * seq_len, "mask length must be seq_len^2");
    assert!(window_size > 0, "window_size must be > 0");
    for (i, row) in mask.chunks_exact_mut(seq_len).enumerate().take(seq_len) {
        for (j, val) in row.iter_mut().enumerate().take(seq_len) {
            *val = j <= i && (i - j) < window_size;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. prefix_mask
// ═══════════════════════════════════════════════════════════════════

/// Generate a prefix-style attention mask.
///
/// Tokens in the prefix (`0..prefix_len`) attend to all positions in
/// the prefix (bidirectional).  Suffix tokens (`prefix_len..seq_len`)
/// attend causally **and** to the entire prefix.
#[cfg(target_arch = "aarch64")]
pub fn prefix_mask(mask: &mut [bool], seq_len: usize, prefix_len: usize) {
    assert_eq!(mask.len(), seq_len * seq_len, "mask length must be seq_len^2");
    assert!(prefix_len <= seq_len, "prefix_len must be <= seq_len");
    for (i, row) in mask.chunks_exact_mut(seq_len).enumerate().take(seq_len) {
        if i < prefix_len {
            // Prefix row: attend to all prefix positions.
            let ptr = row.as_mut_ptr() as *mut u8;
            let full16 = prefix_len / 16;
            unsafe {
                let ones = vdupq_n_u8(1);
                for blk in 0..full16 {
                    vst1q_u8(ptr.add(blk * 16), ones);
                }
            }
            for val in row.iter_mut().skip(full16 * 16).take(prefix_len - full16 * 16) {
                *val = true;
            }
            for val in row.iter_mut().skip(prefix_len) {
                *val = false;
            }
        } else {
            // Suffix row: attend to full prefix + causal suffix.
            let allowed = i + 1;
            let ptr = row.as_mut_ptr() as *mut u8;
            let full16 = allowed / 16;
            unsafe {
                let ones = vdupq_n_u8(1);
                for blk in 0..full16 {
                    vst1q_u8(ptr.add(blk * 16), ones);
                }
            }
            for val in row.iter_mut().skip(full16 * 16).take(allowed - full16 * 16) {
                *val = true;
            }
            for val in row.iter_mut().skip(allowed) {
                *val = false;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn prefix_mask(mask: &mut [bool], seq_len: usize, prefix_len: usize) {
    assert_eq!(mask.len(), seq_len * seq_len, "mask length must be seq_len^2");
    assert!(prefix_len <= seq_len, "prefix_len must be <= seq_len");
    for (i, row) in mask.chunks_exact_mut(seq_len).enumerate().take(seq_len) {
        for (j, val) in row.iter_mut().enumerate().take(seq_len) {
            if i < prefix_len {
                *val = j < prefix_len;
            } else {
                *val = j <= i;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. block_sparse_mask
// ═══════════════════════════════════════════════════════════════════

/// Generate a block-sparse attention pattern.
///
/// The sequence is partitioned into blocks of `block_size`.  A token
/// may attend to positions in its own block and in the immediately
/// preceding block, while still obeying causal ordering (`j <= i`).
#[cfg(target_arch = "aarch64")]
pub fn block_sparse_mask(mask: &mut [bool], seq_len: usize, block_size: usize) {
    assert_eq!(mask.len(), seq_len * seq_len, "mask length must be seq_len^2");
    assert!(block_size > 0, "block_size must be > 0");
    for (i, row) in mask.chunks_exact_mut(seq_len).enumerate().take(seq_len) {
        let my_block = i / block_size;
        let blk_start = if my_block > 0 { (my_block - 1) * block_size } else { 0 };
        for (j, val) in row.iter_mut().enumerate().take(seq_len) {
            let j_block = j / block_size;
            let in_window = j_block == my_block || j_block + 1 == my_block;
            let in_range = j >= blk_start;
            *val = j <= i && in_window && in_range;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn block_sparse_mask(mask: &mut [bool], seq_len: usize, block_size: usize) {
    assert_eq!(mask.len(), seq_len * seq_len, "mask length must be seq_len^2");
    assert!(block_size > 0, "block_size must be > 0");
    for (i, row) in mask.chunks_exact_mut(seq_len).enumerate().take(seq_len) {
        let my_block = i / block_size;
        let blk_start = if my_block > 0 { (my_block - 1) * block_size } else { 0 };
        for (j, val) in row.iter_mut().enumerate().take(seq_len) {
            let j_block = j / block_size;
            let in_window = j_block == my_block || j_block + 1 == my_block;
            let in_range = j >= blk_start;
            *val = j <= i && in_window && in_range;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. combine_masks
// ═══════════════════════════════════════════════════════════════════

/// Combine two boolean masks element-wise with the given operation.
///
/// Both masks and `output` must have the same length.
#[cfg(target_arch = "aarch64")]
pub fn combine_masks(a: &[bool], b: &[bool], output: &mut [bool], op: CombineOp) {
    assert_eq!(a.len(), b.len(), "masks must have the same length");
    assert_eq!(a.len(), output.len(), "output must match mask length");
    let n = a.len();
    let chunks = n / 16;

    unsafe {
        let pa = a.as_ptr() as *const u8;
        let pb = b.as_ptr() as *const u8;
        let po = output.as_mut_ptr() as *mut u8;
        for c in 0..chunks {
            let base = c * 16;
            let va = vld1q_u8(pa.add(base));
            let vb = vld1q_u8(pb.add(base));
            let res = match op {
                CombineOp::And => vandq_u8(va, vb),
                CombineOp::Or => vorrq_u8(va, vb),
            };
            vst1q_u8(po.add(base), res);
        }
    }

    // Scalar tail.
    let tail_start = chunks * 16;
    for (o, (&va, &vb)) in output
        .iter_mut()
        .skip(tail_start)
        .zip(a.iter().skip(tail_start).zip(b.iter().skip(tail_start)))
    {
        *o = match op {
            CombineOp::And => va && vb,
            CombineOp::Or => va || vb,
        };
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn combine_masks(a: &[bool], b: &[bool], output: &mut [bool], op: CombineOp) {
    assert_eq!(a.len(), b.len(), "masks must have the same length");
    assert_eq!(a.len(), output.len(), "output must match mask length");
    for (o, (&va, &vb)) in output.iter_mut().zip(a.iter().zip(b.iter())) {
        *o = match op {
            CombineOp::And => va && vb,
            CombineOp::Or => va || vb,
        };
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── helpers ────────────────────────────────────────────────

    /// Build a mask via the kernel, return the bool vec.
    fn make_causal(n: usize) -> Vec<bool> {
        let mut m = vec![false; n * n];
        causal_mask(&mut m, n);
        m
    }

    fn get(mask: &[bool], seq_len: usize, i: usize, j: usize) -> bool {
        mask[i * seq_len + j]
    }

    fn count_true(mask: &[bool]) -> usize {
        mask.iter().filter(|&&v| v).count()
    }

    // ─── causal_mask ───────────────────────────────────────────

    #[test]
    fn test_causal_mask_1x1() {
        let m = make_causal(1);
        assert!(m[0]);
    }

    #[test]
    fn test_causal_mask_2x2() {
        let m = make_causal(2);
        assert!(get(&m, 2, 0, 0));
        assert!(!get(&m, 2, 0, 1));
        assert!(get(&m, 2, 1, 0));
        assert!(get(&m, 2, 1, 1));
    }

    #[test]
    fn test_causal_mask_4x4() {
        let m = make_causal(4);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(get(&m, 4, i, j), j <= i);
            }
        }
    }

    #[test]
    fn test_causal_mask_diagonal_true() {
        for n in 1..=16 {
            let m = make_causal(n);
            for i in 0..n {
                assert!(get(&m, n, i, i), "diagonal at ({i},{i})");
            }
        }
    }

    #[test]
    fn test_causal_mask_upper_triangle_false() {
        let m = make_causal(8);
        for i in 0..8 {
            for j in (i + 1)..8 {
                assert!(!get(&m, 8, i, j));
            }
        }
    }

    #[test]
    fn test_causal_mask_lower_triangle_true() {
        let m = make_causal(8);
        for i in 0..8 {
            for j in 0..=i {
                assert!(get(&m, 8, i, j));
            }
        }
    }

    #[test]
    fn test_causal_mask_count() {
        for n in 1..=12 {
            let m = make_causal(n);
            let expected = n * (n + 1) / 2;
            assert_eq!(count_true(&m), expected, "n={n}");
        }
    }

    #[test]
    fn test_causal_mask_large() {
        let n = 64;
        let m = make_causal(n);
        assert_eq!(count_true(&m), n * (n + 1) / 2);
    }

    #[test]
    fn test_causal_mask_non_multiple_of_16() {
        let n = 17;
        let m = make_causal(n);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(get(&m, n, i, j), j <= i);
            }
        }
    }

    #[test]
    fn test_causal_mask_first_row() {
        let m = make_causal(8);
        assert!(get(&m, 8, 0, 0));
        for j in 1..8 {
            assert!(!get(&m, 8, 0, j));
        }
    }

    #[test]
    fn test_causal_mask_last_row_all_true() {
        let n = 10;
        let m = make_causal(n);
        for j in 0..n {
            assert!(get(&m, n, n - 1, j));
        }
    }

    #[test]
    #[should_panic(expected = "mask length must be seq_len^2")]
    fn test_causal_mask_wrong_length() {
        let mut m = vec![false; 5];
        causal_mask(&mut m, 4);
    }

    #[test]
    fn test_causal_mask_idempotent() {
        let n = 6;
        let m1 = make_causal(n);
        let m2 = make_causal(n);
        assert_eq!(m1, m2);
    }

    // ─── apply_mask ────────────────────────────────────────────

    #[test]
    fn test_apply_mask_basic() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        apply_mask(&mut scores, &mask);
        assert_eq!(scores[0], 1.0);
        assert!(scores[1].is_infinite() && scores[1] < 0.0);
        assert_eq!(scores[2], 3.0);
        assert!(scores[3].is_infinite() && scores[3] < 0.0);
    }

    #[test]
    fn test_apply_mask_all_true() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true; 5];
        let orig = scores.clone();
        apply_mask(&mut scores, &mask);
        assert_eq!(scores, orig);
    }

    #[test]
    fn test_apply_mask_all_false() {
        let mut scores = vec![1.0; 8];
        let mask = vec![false; 8];
        apply_mask(&mut scores, &mask);
        for s in &scores {
            assert!(s.is_infinite() && *s < 0.0);
        }
    }

    #[test]
    fn test_apply_mask_empty() {
        let mut scores: Vec<f32> = vec![];
        let mask: Vec<bool> = vec![];
        apply_mask(&mut scores, &mask);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_apply_mask_preserves_neg_infinity() {
        let mut scores = vec![f32::NEG_INFINITY, 1.0];
        let mask = vec![true, true];
        apply_mask(&mut scores, &mask);
        assert!(scores[0].is_infinite() && scores[0] < 0.0);
        assert_eq!(scores[1], 1.0);
    }

    #[test]
    fn test_apply_mask_large_vector() {
        let n = 100;
        let mut scores: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mask: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        apply_mask(&mut scores, &mask);
        for (i, &s) in scores.iter().enumerate() {
            if i % 2 == 0 {
                assert_eq!(s, i as f32);
            } else {
                assert!(s.is_infinite() && s < 0.0);
            }
        }
    }

    #[test]
    fn test_apply_mask_tail_elements() {
        // Length 5: 4 in NEON chunk + 1 scalar tail.
        let mut scores = vec![10.0; 5];
        let mask = vec![true, true, true, true, false];
        apply_mask(&mut scores, &mask);
        assert_eq!(scores[3], 10.0);
        assert!(scores[4].is_infinite() && scores[4] < 0.0);
    }

    #[test]
    #[should_panic(expected = "scores and mask must have the same length")]
    fn test_apply_mask_length_mismatch() {
        let mut scores = vec![1.0; 4];
        let mask = vec![true; 5];
        apply_mask(&mut scores, &mask);
    }

    #[test]
    fn test_apply_mask_with_causal() {
        let n = 4;
        let m = make_causal(n);
        let mut scores: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        apply_mask(&mut scores, &m);
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                if j <= i {
                    assert_eq!(scores[idx], idx as f32);
                } else {
                    assert!(scores[idx].is_infinite());
                }
            }
        }
    }

    #[test]
    fn test_apply_mask_nan_stays_nan() {
        let mut scores = vec![f32::NAN, 1.0];
        let mask = vec![true, false];
        apply_mask(&mut scores, &mask);
        assert!(scores[0].is_nan());
        assert!(scores[1].is_infinite() && scores[1] < 0.0);
    }

    // ─── sliding_window_mask ───────────────────────────────────

    #[test]
    fn test_sliding_window_basic() {
        let n = 4;
        let w = 2;
        let mut m = vec![false; n * n];
        sliding_window_mask(&mut m, n, w);
        // Row 0: only col 0
        assert!(get(&m, n, 0, 0));
        assert!(!get(&m, n, 0, 1));
        // Row 1: cols 0,1
        assert!(get(&m, n, 1, 0));
        assert!(get(&m, n, 1, 1));
        // Row 2: cols 1,2 (window=2 → i-j < 2)
        assert!(!get(&m, n, 2, 0));
        assert!(get(&m, n, 2, 1));
        assert!(get(&m, n, 2, 2));
        // Row 3: cols 2,3
        assert!(!get(&m, n, 3, 1));
        assert!(get(&m, n, 3, 2));
        assert!(get(&m, n, 3, 3));
    }

    #[test]
    fn test_sliding_window_eq_1() {
        let n = 5;
        let mut m = vec![false; n * n];
        sliding_window_mask(&mut m, n, 1);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(get(&m, n, i, j), i == j, "({i},{j})");
            }
        }
    }

    #[test]
    fn test_sliding_window_full_width() {
        let n = 6;
        let mut m = vec![false; n * n];
        sliding_window_mask(&mut m, n, n);
        let c = make_causal(n);
        assert_eq!(m, c, "full window == causal");
    }

    #[test]
    fn test_sliding_window_larger_than_seq() {
        let n = 4;
        let mut m = vec![false; n * n];
        sliding_window_mask(&mut m, n, n + 10);
        let c = make_causal(n);
        assert_eq!(m, c);
    }

    #[test]
    fn test_sliding_window_count() {
        let n = 8;
        let w = 3;
        let mut m = vec![false; n * n];
        sliding_window_mask(&mut m, n, w);
        let mut expected = 0;
        for i in 0..n {
            expected += std::cmp::min(i + 1, w);
        }
        assert_eq!(count_true(&m), expected);
    }

    #[test]
    fn test_sliding_window_non_multiple() {
        let n = 17;
        let w = 5;
        let mut m = vec![false; n * n];
        sliding_window_mask(&mut m, n, w);
        for i in 0..n {
            for j in 0..n {
                let exp = j <= i && (i - j) < w;
                assert_eq!(get(&m, n, i, j), exp, "({i},{j})");
            }
        }
    }

    #[test]
    fn test_sliding_window_1x1() {
        let mut m = vec![false; 1];
        sliding_window_mask(&mut m, 1, 1);
        assert!(m[0]);
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn test_sliding_window_zero_window() {
        let mut m = vec![false; 4];
        sliding_window_mask(&mut m, 2, 0);
    }

    #[test]
    #[should_panic(expected = "mask length must be seq_len^2")]
    fn test_sliding_window_wrong_length() {
        let mut m = vec![false; 10];
        sliding_window_mask(&mut m, 4, 2);
    }

    #[test]
    fn test_sliding_window_is_subset_of_causal() {
        let n = 10;
        let w = 4;
        let c = make_causal(n);
        let mut sw = vec![false; n * n];
        sliding_window_mask(&mut sw, n, w);
        for (i, (&sv, &cv)) in sw.iter().zip(c.iter()).enumerate() {
            if sv {
                assert!(cv, "sw true but causal false at {i}");
            }
        }
    }

    // ─── prefix_mask ───────────────────────────────────────────

    #[test]
    fn test_prefix_mask_full_prefix() {
        let n = 4;
        let mut m = vec![false; n * n];
        prefix_mask(&mut m, n, n);
        // All prefix → bidirectional among all = full matrix.
        for val in &m {
            assert!(*val);
        }
    }

    #[test]
    fn test_prefix_mask_zero_prefix() {
        let n = 4;
        let mut m = vec![false; n * n];
        prefix_mask(&mut m, n, 0);
        let c = make_causal(n);
        assert_eq!(m, c, "zero prefix == causal");
    }

    #[test]
    fn test_prefix_mask_basic() {
        let n = 4;
        let p = 2;
        let mut m = vec![false; n * n];
        prefix_mask(&mut m, n, p);
        // Row 0 (prefix): [T,T,F,F]
        assert!(get(&m, n, 0, 0));
        assert!(get(&m, n, 0, 1));
        assert!(!get(&m, n, 0, 2));
        assert!(!get(&m, n, 0, 3));
        // Row 1 (prefix): [T,T,F,F]
        assert!(get(&m, n, 1, 0));
        assert!(get(&m, n, 1, 1));
        assert!(!get(&m, n, 1, 2));
        assert!(!get(&m, n, 1, 3));
        // Row 2 (suffix, causal): [T,T,T,F]
        assert!(get(&m, n, 2, 0));
        assert!(get(&m, n, 2, 1));
        assert!(get(&m, n, 2, 2));
        assert!(!get(&m, n, 2, 3));
        // Row 3 (suffix, causal): [T,T,T,T]
        for j in 0..n {
            assert!(get(&m, n, 3, j));
        }
    }

    #[test]
    fn test_prefix_mask_1x1() {
        let mut m = vec![false; 1];
        prefix_mask(&mut m, 1, 1);
        assert!(m[0]);
    }

    #[test]
    fn test_prefix_mask_count() {
        let n = 6;
        let p = 3;
        let mut m = vec![false; n * n];
        prefix_mask(&mut m, n, p);
        let mut expected = p * p; // prefix rows
        for i in p..n {
            expected += i + 1; // suffix rows: causal
        }
        assert_eq!(count_true(&m), expected);
    }

    #[test]
    fn test_prefix_mask_superset_of_causal() {
        let n = 8;
        let p = 3;
        let c = make_causal(n);
        let mut pm = vec![false; n * n];
        prefix_mask(&mut pm, n, p);
        for (i, (&pv, &cv)) in pm.iter().zip(c.iter()).enumerate() {
            if cv {
                assert!(pv, "causal true but prefix false at {i}");
            }
        }
    }

    #[test]
    fn test_prefix_mask_non_multiple() {
        let n = 17;
        let p = 7;
        let mut m = vec![false; n * n];
        prefix_mask(&mut m, n, p);
        for i in 0..n {
            for j in 0..n {
                let exp = if i < p { j < p } else { j <= i };
                assert_eq!(get(&m, n, i, j), exp, "({i},{j})");
            }
        }
    }

    #[test]
    #[should_panic(expected = "prefix_len must be <= seq_len")]
    fn test_prefix_mask_prefix_too_large() {
        let mut m = vec![false; 4];
        prefix_mask(&mut m, 2, 3);
    }

    #[test]
    #[should_panic(expected = "mask length must be seq_len^2")]
    fn test_prefix_mask_wrong_length() {
        let mut m = vec![false; 5];
        prefix_mask(&mut m, 3, 1);
    }

    // ─── block_sparse_mask ─────────────────────────────────────

    #[test]
    fn test_block_sparse_block_eq_seq() {
        let n = 4;
        let mut m = vec![false; n * n];
        block_sparse_mask(&mut m, n, n);
        let c = make_causal(n);
        assert_eq!(m, c, "single block == causal");
    }

    #[test]
    fn test_block_sparse_block_eq_1() {
        let n = 4;
        let mut m = vec![false; n * n];
        block_sparse_mask(&mut m, n, 1);
        // block=1 → each token is its own block.
        // Attends to own block + previous block, causally.
        // Row 0: [T,F,F,F]
        // Row 1: [T,T,F,F]
        // Row 2: [F,T,T,F]
        // Row 3: [F,F,T,T]
        assert!(get(&m, n, 0, 0));
        assert!(!get(&m, n, 0, 1));
        assert!(get(&m, n, 1, 0));
        assert!(get(&m, n, 1, 1));
        assert!(!get(&m, n, 2, 0));
        assert!(get(&m, n, 2, 1));
        assert!(get(&m, n, 2, 2));
        assert!(!get(&m, n, 3, 1));
        assert!(get(&m, n, 3, 2));
        assert!(get(&m, n, 3, 3));
    }

    #[test]
    fn test_block_sparse_basic() {
        let n = 6;
        let bs = 2;
        let mut m = vec![false; n * n];
        block_sparse_mask(&mut m, n, bs);
        for i in 0..n {
            for j in 0..n {
                let my_blk = i / bs;
                let j_blk = j / bs;
                let in_win = j_blk == my_blk || j_blk + 1 == my_blk;
                let blk_start = if my_blk > 0 { (my_blk - 1) * bs } else { 0 };
                let exp = j <= i && in_win && j >= blk_start;
                assert_eq!(get(&m, n, i, j), exp, "({i},{j})");
            }
        }
    }

    #[test]
    fn test_block_sparse_is_subset_of_causal() {
        let n = 12;
        let bs = 3;
        let c = make_causal(n);
        let mut bsm = vec![false; n * n];
        block_sparse_mask(&mut bsm, n, bs);
        for (i, (&bv, &cv)) in bsm.iter().zip(c.iter()).enumerate() {
            if bv {
                assert!(cv, "block true but causal false at {i}");
            }
        }
    }

    #[test]
    fn test_block_sparse_diagonal_true() {
        let n = 9;
        let bs = 3;
        let mut m = vec![false; n * n];
        block_sparse_mask(&mut m, n, bs);
        for i in 0..n {
            assert!(get(&m, n, i, i), "diagonal at ({i},{i})");
        }
    }

    #[test]
    fn test_block_sparse_1x1() {
        let mut m = vec![false; 1];
        block_sparse_mask(&mut m, 1, 1);
        assert!(m[0]);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn test_block_sparse_zero_block_size() {
        let mut m = vec![false; 4];
        block_sparse_mask(&mut m, 2, 0);
    }

    #[test]
    #[should_panic(expected = "mask length must be seq_len^2")]
    fn test_block_sparse_wrong_length() {
        let mut m = vec![false; 7];
        block_sparse_mask(&mut m, 3, 2);
    }

    #[test]
    fn test_block_sparse_non_divisible() {
        let n = 7;
        let bs = 3;
        let mut m = vec![false; n * n];
        block_sparse_mask(&mut m, n, bs);
        // Just verify causal + diagonal.
        for i in 0..n {
            assert!(get(&m, n, i, i));
            for j in (i + 1)..n {
                assert!(!get(&m, n, i, j));
            }
        }
    }

    // ─── combine_masks ─────────────────────────────────────────

    #[test]
    fn test_combine_and_basic() {
        let a = vec![true, true, false, false];
        let b = vec![true, false, true, false];
        let mut o = vec![false; 4];
        combine_masks(&a, &b, &mut o, CombineOp::And);
        assert_eq!(o, vec![true, false, false, false]);
    }

    #[test]
    fn test_combine_or_basic() {
        let a = vec![true, true, false, false];
        let b = vec![true, false, true, false];
        let mut o = vec![false; 4];
        combine_masks(&a, &b, &mut o, CombineOp::Or);
        assert_eq!(o, vec![true, true, true, false]);
    }

    #[test]
    fn test_combine_and_identity() {
        let n = 20;
        let a = vec![true; n];
        let b: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let mut o = vec![false; n];
        combine_masks(&a, &b, &mut o, CombineOp::And);
        assert_eq!(o, b);
    }

    #[test]
    fn test_combine_or_identity() {
        let n = 20;
        let a = vec![false; n];
        let b: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let mut o = vec![false; n];
        combine_masks(&a, &b, &mut o, CombineOp::Or);
        assert_eq!(o, b);
    }

    #[test]
    fn test_combine_and_all_false() {
        let n = 16;
        let a = vec![true; n];
        let b = vec![false; n];
        let mut o = vec![true; n];
        combine_masks(&a, &b, &mut o, CombineOp::And);
        assert!(o.iter().all(|&v| !v));
    }

    #[test]
    fn test_combine_or_all_true() {
        let n = 16;
        let a = vec![false; n];
        let b = vec![true; n];
        let mut o = vec![false; n];
        combine_masks(&a, &b, &mut o, CombineOp::Or);
        assert!(o.iter().all(|&v| v));
    }

    #[test]
    fn test_combine_empty() {
        let a: Vec<bool> = vec![];
        let b: Vec<bool> = vec![];
        let mut o: Vec<bool> = vec![];
        combine_masks(&a, &b, &mut o, CombineOp::And);
        assert!(o.is_empty());
    }

    #[test]
    fn test_combine_large() {
        let n = 200;
        let a: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let b: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let mut o = vec![false; n];
        combine_masks(&a, &b, &mut o, CombineOp::And);
        for (i, &v) in o.iter().enumerate() {
            assert_eq!(v, i % 2 == 0 && i % 3 == 0, "i={i}");
        }
    }

    #[test]
    fn test_combine_tail() {
        // 17 elements: 16 in NEON chunk + 1 tail.
        let n = 17;
        let a = vec![true; n];
        let b: Vec<bool> = {
            let mut v = vec![true; n];
            v[16] = false;
            v
        };
        let mut o = vec![false; n];
        combine_masks(&a, &b, &mut o, CombineOp::And);
        assert!(o[15]);
        assert!(!o[16]);
    }

    #[test]
    #[should_panic(expected = "masks must have the same length")]
    fn test_combine_length_mismatch_ab() {
        let a = vec![true; 3];
        let b = vec![true; 4];
        let mut o = vec![false; 3];
        combine_masks(&a, &b, &mut o, CombineOp::And);
    }

    #[test]
    #[should_panic(expected = "output must match mask length")]
    fn test_combine_length_mismatch_output() {
        let a = vec![true; 4];
        let b = vec![true; 4];
        let mut o = vec![false; 3];
        combine_masks(&a, &b, &mut o, CombineOp::And);
    }

    // ─── cross-operation tests ─────────────────────────────────

    #[test]
    fn test_sliding_and_causal_intersection() {
        let n = 8;
        let w = 3;
        let c = make_causal(n);
        let mut sw = vec![false; n * n];
        sliding_window_mask(&mut sw, n, w);
        let mut combined = vec![false; n * n];
        combine_masks(&c, &sw, &mut combined, CombineOp::And);
        assert_eq!(combined, sw, "AND(causal, sliding) == sliding");
    }

    #[test]
    fn test_prefix_or_causal_eq_prefix() {
        let n = 6;
        let p = 2;
        let c = make_causal(n);
        let mut pm = vec![false; n * n];
        prefix_mask(&mut pm, n, p);
        let mut combined = vec![false; n * n];
        combine_masks(&pm, &c, &mut combined, CombineOp::Or);
        assert_eq!(combined, pm, "OR(prefix, causal) == prefix");
    }

    #[test]
    fn test_apply_after_causal() {
        let n = 3;
        let mut scores: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let m = make_causal(n);
        apply_mask(&mut scores, &m);
        // Row 0: [0, -inf, -inf]
        assert_eq!(scores[0], 0.0);
        assert!(scores[1].is_infinite());
        assert!(scores[2].is_infinite());
        // Row 1: [3, 4, -inf]
        assert_eq!(scores[3], 3.0);
        assert_eq!(scores[4], 4.0);
        assert!(scores[5].is_infinite());
        // Row 2: [6, 7, 8]
        assert_eq!(scores[6], 6.0);
        assert_eq!(scores[7], 7.0);
        assert_eq!(scores[8], 8.0);
    }

    #[test]
    fn test_apply_after_sliding_window() {
        let n = 4;
        let w = 2;
        let mut m = vec![false; n * n];
        sliding_window_mask(&mut m, n, w);
        let mut scores: Vec<f32> = vec![1.0; n * n];
        apply_mask(&mut scores, &m);
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                if j <= i && (i - j) < w {
                    assert_eq!(scores[idx], 1.0);
                } else {
                    assert!(scores[idx].is_infinite());
                }
            }
        }
    }

    #[test]
    fn test_combine_causal_and_prefix_superset() {
        let n = 5;
        let p = 2;
        let c = make_causal(n);
        let mut pm = vec![false; n * n];
        prefix_mask(&mut pm, n, p);
        let mut and_result = vec![false; n * n];
        combine_masks(&c, &pm, &mut and_result, CombineOp::And);
        // AND should equal the causal mask since prefix ⊇ causal.
        assert_eq!(and_result, c);
    }

    #[test]
    fn test_block_sparse_subset_of_sliding() {
        // With block_size=w, block-sparse ⊆ sliding(window=2*w)
        let n = 8;
        let bs = 2;
        let w = 2 * bs;
        let mut bsm = vec![false; n * n];
        block_sparse_mask(&mut bsm, n, bs);
        let mut sw = vec![false; n * n];
        sliding_window_mask(&mut sw, n, w);
        for (i, (&bv, &sv)) in bsm.iter().zip(sw.iter()).enumerate() {
            if bv {
                assert!(sv, "block true but sliding false at {i}");
            }
        }
    }

    // ─── determinism ───────────────────────────────────────────

    #[test]
    fn test_causal_deterministic() {
        let a = make_causal(16);
        let b = make_causal(16);
        assert_eq!(a, b);
    }

    #[test]
    fn test_sliding_deterministic() {
        let n = 16;
        let w = 5;
        let mut a = vec![false; n * n];
        let mut b = vec![false; n * n];
        sliding_window_mask(&mut a, n, w);
        sliding_window_mask(&mut b, n, w);
        assert_eq!(a, b);
    }

    #[test]
    fn test_prefix_deterministic() {
        let n = 16;
        let p = 6;
        let mut a = vec![false; n * n];
        let mut b = vec![false; n * n];
        prefix_mask(&mut a, n, p);
        prefix_mask(&mut b, n, p);
        assert_eq!(a, b);
    }

    // ─── edge cases ────────────────────────────────────────────

    #[test]
    fn test_causal_mask_32x32() {
        let n = 32;
        let m = make_causal(n);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(get(&m, n, i, j), j <= i);
            }
        }
    }

    #[test]
    fn test_sliding_window_large() {
        let n = 32;
        let w = 8;
        let mut m = vec![false; n * n];
        sliding_window_mask(&mut m, n, w);
        for i in 0..n {
            for j in 0..n {
                let exp = j <= i && (i - j) < w;
                assert_eq!(get(&m, n, i, j), exp, "({i},{j})");
            }
        }
    }

    #[test]
    fn test_prefix_mask_large() {
        let n = 32;
        let p = 10;
        let mut m = vec![false; n * n];
        prefix_mask(&mut m, n, p);
        for i in 0..n {
            for j in 0..n {
                let exp = if i < p { j < p } else { j <= i };
                assert_eq!(get(&m, n, i, j), exp, "({i},{j})");
            }
        }
    }

    #[test]
    fn test_block_sparse_large() {
        let n = 24;
        let bs = 4;
        let mut m = vec![false; n * n];
        block_sparse_mask(&mut m, n, bs);
        for i in 0..n {
            assert!(get(&m, n, i, i), "diagonal");
            for j in (i + 1)..n {
                assert!(!get(&m, n, i, j), "future");
            }
        }
    }

    #[test]
    fn test_combine_commutative_and() {
        let n = 20;
        let a: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let b: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let mut o1 = vec![false; n];
        let mut o2 = vec![false; n];
        combine_masks(&a, &b, &mut o1, CombineOp::And);
        combine_masks(&b, &a, &mut o2, CombineOp::And);
        assert_eq!(o1, o2);
    }

    #[test]
    fn test_combine_commutative_or() {
        let n = 20;
        let a: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let b: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let mut o1 = vec![false; n];
        let mut o2 = vec![false; n];
        combine_masks(&a, &b, &mut o1, CombineOp::Or);
        combine_masks(&b, &a, &mut o2, CombineOp::Or);
        assert_eq!(o1, o2);
    }

    #[test]
    fn test_apply_mask_single_element() {
        let mut s = vec![42.0];
        let m = vec![false];
        apply_mask(&mut s, &m);
        assert!(s[0].is_infinite() && s[0] < 0.0);
    }

    #[test]
    fn test_apply_mask_single_true() {
        let mut s = vec![42.0];
        let m = vec![true];
        apply_mask(&mut s, &m);
        assert_eq!(s[0], 42.0);
    }

    #[test]
    fn test_causal_mask_3x3_full_check() {
        let m = make_causal(3);
        let expected = vec![true, false, false, true, true, false, true, true, true];
        assert_eq!(m, expected);
    }

    #[test]
    fn test_combine_self_and() {
        let n = 10;
        let a: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let mut o = vec![false; n];
        combine_masks(&a, &a, &mut o, CombineOp::And);
        assert_eq!(o, a);
    }

    #[test]
    fn test_combine_self_or() {
        let n = 10;
        let a: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let mut o = vec![false; n];
        combine_masks(&a, &a, &mut o, CombineOp::Or);
        assert_eq!(o, a);
    }

    #[test]
    fn test_combine_de_morgan_and() {
        // !(a AND b) == (!a OR !b)
        let n = 16;
        let a: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let b: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let mut and_result = vec![false; n];
        combine_masks(&a, &b, &mut and_result, CombineOp::And);
        let not_and: Vec<bool> = and_result.iter().map(|&v| !v).collect();

        let not_a: Vec<bool> = a.iter().map(|&v| !v).collect();
        let not_b: Vec<bool> = b.iter().map(|&v| !v).collect();
        let mut or_nots = vec![false; n];
        combine_masks(&not_a, &not_b, &mut or_nots, CombineOp::Or);
        assert_eq!(not_and, or_nots, "De Morgan");
    }

    #[test]
    fn test_combine_de_morgan_or() {
        // !(a OR b) == (!a AND !b)
        let n = 16;
        let a: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let b: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let mut or_result = vec![false; n];
        combine_masks(&a, &b, &mut or_result, CombineOp::Or);
        let not_or: Vec<bool> = or_result.iter().map(|&v| !v).collect();

        let not_a: Vec<bool> = a.iter().map(|&v| !v).collect();
        let not_b: Vec<bool> = b.iter().map(|&v| !v).collect();
        let mut and_nots = vec![false; n];
        combine_masks(&not_a, &not_b, &mut and_nots, CombineOp::And);
        assert_eq!(not_or, and_nots, "De Morgan");
    }

    #[test]
    fn test_apply_mask_alternating_16() {
        let n = 16;
        let mut scores: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mask: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        apply_mask(&mut scores, &mask);
        for (i, &s) in scores.iter().enumerate() {
            if i % 2 == 0 {
                assert_eq!(s, i as f32);
            } else {
                assert!(s.is_infinite() && s < 0.0);
            }
        }
    }

    #[test]
    fn test_sliding_window_window_eq_2_size_8() {
        let n = 8;
        let w = 2;
        let mut m = vec![false; n * n];
        sliding_window_mask(&mut m, n, w);
        // Every row has exactly min(i+1, w) true entries.
        for i in 0..n {
            let row = &m[i * n..(i + 1) * n];
            let cnt = row.iter().filter(|&&v| v).count();
            assert_eq!(cnt, std::cmp::min(i + 1, w), "row {i}");
        }
    }
}
