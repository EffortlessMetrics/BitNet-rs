//! NEON-optimized tensor concatenation and split operations for
//! Apple Silicon.
//!
//! Provides efficient concat, split, stack, interleave, and chunk
//! operations on contiguous `f32` slices.  Each public function has
//! a NEON fast-path (`aarch64`) and a portable scalar fallback so
//! the crate compiles and tests pass on any target.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Number of `f32` lanes in a NEON `float32x4_t` register.
const LANES: usize = 4;

// ── helpers ────────────────────────────────────────────────────────

/// Bulk-copy `src` into `dst` using NEON 4-wide loads/stores where
/// possible, falling back to scalar for the tail.
#[cfg(target_arch = "aarch64")]
#[inline]
fn copy_neon(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = src.len();
    let chunks = n / LANES;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let v = vld1q_f32(sp.add(off));
            vst1q_f32(dp.add(off), v);
        }
    }
    dst.iter_mut().zip(src.iter()).skip(chunks * LANES).for_each(|(d, s)| *d = *s);
}

/// Portable bulk-copy (same semantics as [`copy_neon`]).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn copy_neon(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    dst.iter_mut().zip(src.iter()).for_each(|(d, s)| *d = *s);
}

// ══════════════════════════════════════════════════════════════════
// 1. concat_1d — Concatenate two 1-D tensors along axis 0
// ══════════════════════════════════════════════════════════════════

/// Concatenate two 1-D tensors end-to-end.
///
/// Returns a new vector of length `a.len() + b.len()`.
pub fn concat_1d(a: &[f32], b: &[f32]) -> Vec<f32> {
    let total = a.len() + b.len();
    let mut out = vec![0.0_f32; total];
    copy_neon(&mut out[..a.len()], a);
    copy_neon(&mut out[a.len()..], b);
    out
}

// ══════════════════════════════════════════════════════════════════
// 2. concat_2d_axis0 — Concatenate along rows (axis 0)
// ══════════════════════════════════════════════════════════════════

/// Concatenate two row-major 2-D tensors along axis 0 (rows).
///
/// Both tensors must share the same number of columns (`cols`).
/// `a` has `rows_a` rows; `b` has `rows_b` rows.
///
/// # Panics
///
/// Panics if `a.len() != rows_a * cols` or `b.len() != rows_b * cols`.
pub fn concat_2d_axis0(
    a: &[f32],
    rows_a: usize,
    b: &[f32],
    rows_b: usize,
    cols: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), rows_a * cols, "a shape mismatch");
    assert_eq!(b.len(), rows_b * cols, "b shape mismatch");
    let total = (rows_a + rows_b) * cols;
    let mut out = vec![0.0_f32; total];
    copy_neon(&mut out[..a.len()], a);
    copy_neon(&mut out[a.len()..], b);
    out
}

// ══════════════════════════════════════════════════════════════════
// 3. concat_2d_axis1 — Concatenate along columns (axis 1)
// ══════════════════════════════════════════════════════════════════

/// Concatenate two row-major 2-D tensors along axis 1 (columns).
///
/// Both tensors must have the same number of rows.
/// `a` has `cols_a` columns; `b` has `cols_b` columns.
///
/// # Panics
///
/// Panics on shape mismatch.
pub fn concat_2d_axis1(
    a: &[f32],
    rows: usize,
    cols_a: usize,
    b: &[f32],
    cols_b: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), rows * cols_a, "a shape mismatch");
    assert_eq!(b.len(), rows * cols_b, "b shape mismatch");
    let out_cols = cols_a + cols_b;
    let mut out = vec![0.0_f32; rows * out_cols];
    (0..rows).for_each(|r| {
        let a_start = r * cols_a;
        let b_start = r * cols_b;
        let o_start = r * out_cols;
        copy_neon(&mut out[o_start..o_start + cols_a], &a[a_start..a_start + cols_a]);
        copy_neon(&mut out[o_start + cols_a..o_start + out_cols], &b[b_start..b_start + cols_b]);
    });
    out
}

// ══════════════════════════════════════════════════════════════════
// 4. split_1d — Split a 1-D tensor at a given index
// ══════════════════════════════════════════════════════════════════

/// Split a 1-D tensor into two at `mid`.
///
/// Returns `(left[..mid], right[mid..])`.
///
/// # Panics
///
/// Panics if `mid > data.len()`.
pub fn split_1d(data: &[f32], mid: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(mid <= data.len(), "split index out of bounds");
    let left_len = mid;
    let right_len = data.len() - mid;
    let mut left = vec![0.0_f32; left_len];
    let mut right = vec![0.0_f32; right_len];
    copy_neon(&mut left, &data[..mid]);
    copy_neon(&mut right, &data[mid..]);
    (left, right)
}

// ══════════════════════════════════════════════════════════════════
// 5. split_2d_axis0 — Split a 2-D tensor along rows
// ══════════════════════════════════════════════════════════════════

/// Split a row-major 2-D tensor into two along axis 0 at row
/// `split_row`.
///
/// # Panics
///
/// Panics on shape mismatch or if `split_row > rows`.
pub fn split_2d_axis0(
    data: &[f32],
    rows: usize,
    cols: usize,
    split_row: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(data.len(), rows * cols, "shape mismatch");
    assert!(split_row <= rows, "split_row out of bounds");
    let mid = split_row * cols;
    let mut top = vec![0.0_f32; mid];
    let mut bot = vec![0.0_f32; data.len() - mid];
    copy_neon(&mut top, &data[..mid]);
    copy_neon(&mut bot, &data[mid..]);
    (top, bot)
}

// ══════════════════════════════════════════════════════════════════
// 6. stack_vectors — Stack multiple 1-D vectors into a 2-D matrix
// ══════════════════════════════════════════════════════════════════

/// Stack a slice of equal-length vectors into a row-major matrix.
///
/// # Panics
///
/// Panics if any vector length differs from `cols`.
pub fn stack_vectors(vecs: &[&[f32]], cols: usize) -> Vec<f32> {
    let rows = vecs.len();
    let mut out = vec![0.0_f32; rows * cols];
    vecs.iter().enumerate().for_each(|(r, v)| {
        assert_eq!(v.len(), cols, "vector length mismatch");
        let start = r * cols;
        copy_neon(&mut out[start..start + cols], v);
    });
    out
}

// ══════════════════════════════════════════════════════════════════
// 7. interleave — Interleave elements from two tensors
// ══════════════════════════════════════════════════════════════════

/// Interleave elements from `a` and `b`:
/// `[a0, b0, a1, b1, …]`.
///
/// Both slices must have the same length.  The result has
/// `2 * a.len()` elements.  Useful for RoPE pair construction.
///
/// # NEON path
///
/// Uses `vzip1q_f32` / `vzip2q_f32` to interleave four-lane
/// pairs in a single instruction.
#[cfg(target_arch = "aarch64")]
pub fn interleave(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let n = a.len();
    let mut out = vec![0.0_f32; n * 2];
    let chunks = n / LANES;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let op = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let va = vld1q_f32(ap.add(off));
            let vb = vld1q_f32(bp.add(off));
            let lo = vzip1q_f32(va, vb);
            let hi = vzip2q_f32(va, vb);
            vst1q_f32(op.add(off * 2), lo);
            vst1q_f32(op.add(off * 2 + LANES), hi);
        }
    }
    // Scalar tail
    (chunks * LANES..n).for_each(|i| {
        out[i * 2] = a[i];
        out[i * 2 + 1] = b[i];
    });
    out
}

/// Scalar fallback for [`interleave`].
#[cfg(not(target_arch = "aarch64"))]
pub fn interleave(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let n = a.len();
    let mut out = vec![0.0_f32; n * 2];
    (0..n).for_each(|i| {
        out[i * 2] = a[i];
        out[i * 2 + 1] = b[i];
    });
    out
}

// ══════════════════════════════════════════════════════════════════
// 8. chunk — Split tensor into N equal chunks
// ══════════════════════════════════════════════════════════════════

/// Split a 1-D tensor into `n` equal-length chunks.
///
/// # Panics
///
/// Panics if `n == 0` or `data.len()` is not evenly divisible by
/// `n`.
pub fn chunk(data: &[f32], n: usize) -> Vec<Vec<f32>> {
    assert!(n > 0, "n must be > 0");
    assert!(data.len().is_multiple_of(n), "data length {} not divisible by {n}", data.len(),);
    let chunk_len = data.len() / n;
    (0..n)
        .map(|i| {
            let start = i * chunk_len;
            let mut c = vec![0.0_f32; chunk_len];
            copy_neon(&mut c, &data[start..start + chunk_len]);
            c
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── concat_1d ──────────────────────────────────────────────

    #[test]
    fn concat_1d_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        assert_eq!(concat_1d(&a, &b), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn concat_1d_empty_left() {
        let b = [1.0, 2.0];
        assert_eq!(concat_1d(&[], &b), vec![1.0, 2.0]);
    }

    #[test]
    fn concat_1d_empty_right() {
        let a = [1.0, 2.0];
        assert_eq!(concat_1d(&a, &[]), vec![1.0, 2.0]);
    }

    #[test]
    fn concat_1d_both_empty() {
        let out: Vec<f32> = concat_1d(&[], &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn concat_1d_single_elements() {
        assert_eq!(concat_1d(&[1.0], &[2.0]), vec![1.0, 2.0]);
    }

    #[test]
    fn concat_1d_aligned_4() {
        let a: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let b: Vec<f32> = (4..8).map(|i| i as f32).collect();
        let expected: Vec<f32> = (0..8).map(|i| i as f32).collect();
        assert_eq!(concat_1d(&a, &b), expected);
    }

    #[test]
    fn concat_1d_unaligned() {
        let a: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let b: Vec<f32> = (5..11).map(|i| i as f32).collect();
        let expected: Vec<f32> = (0..11).map(|i| i as f32).collect();
        assert_eq!(concat_1d(&a, &b), expected);
    }

    #[test]
    fn concat_1d_large() {
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = (256..512).map(|i| i as f32).collect();
        let expected: Vec<f32> = (0..512).map(|i| i as f32).collect();
        assert_eq!(concat_1d(&a, &b), expected);
    }

    #[test]
    fn concat_1d_negative_values() {
        let a = [-1.0, -2.0];
        let b = [-3.0, -4.0];
        assert_eq!(concat_1d(&a, &b), vec![-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn concat_1d_preserves_order() {
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = (17..33).map(|i| i as f32).collect();
        let out = concat_1d(&a, &b);
        out.iter().enumerate().for_each(|(i, &v)| assert_eq!(v, i as f32));
    }

    // ── concat_2d_axis0 ────────────────────────────────────────

    #[test]
    fn concat_2d_axis0_basic() {
        // 2×3 + 1×3 → 3×3
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0];
        let out = concat_2d_axis0(&a, 2, &b, 1, 3);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn concat_2d_axis0_same_size() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let out = concat_2d_axis0(&a, 2, &b, 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn concat_2d_axis0_single_row() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let out = concat_2d_axis0(&a, 1, &b, 1, 4);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn concat_2d_axis0_empty_a() {
        let b = [1.0, 2.0, 3.0];
        let out = concat_2d_axis0(&[], 0, &b, 1, 3);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn concat_2d_axis0_empty_b() {
        let a = [1.0, 2.0, 3.0];
        let out = concat_2d_axis0(&a, 1, &[], 0, 3);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn concat_2d_axis0_wide_cols() {
        let cols = 16;
        let a: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let b: Vec<f32> = (cols..cols * 2).map(|i| i as f32).collect();
        let out = concat_2d_axis0(&a, 1, &b, 1, cols);
        let expected: Vec<f32> = (0..cols * 2).map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn concat_2d_axis0_many_rows() {
        let rows_a = 10;
        let rows_b = 5;
        let cols = 3;
        let a: Vec<f32> = (0..rows_a * cols).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..rows_b * cols).map(|i| (i + rows_a * cols) as f32).collect();
        let out = concat_2d_axis0(&a, rows_a, &b, rows_b, cols);
        assert_eq!(out.len(), (rows_a + rows_b) * cols);
        assert_eq!(&out[..a.len()], &a[..]);
        assert_eq!(&out[a.len()..], &b[..]);
    }

    #[test]
    #[should_panic(expected = "a shape mismatch")]
    fn concat_2d_axis0_bad_a_shape() {
        concat_2d_axis0(&[1.0, 2.0], 1, &[3.0, 4.0], 1, 3);
    }

    #[test]
    #[should_panic(expected = "b shape mismatch")]
    fn concat_2d_axis0_bad_b_shape() {
        concat_2d_axis0(&[1.0, 2.0], 1, &[3.0], 1, 2);
    }

    // ── concat_2d_axis1 ────────────────────────────────────────

    #[test]
    fn concat_2d_axis1_basic() {
        // 2×2 | 2×3 → 2×5
        let a = [1.0, 2.0, 3.0, 4.0]; // 2 rows × 2 cols
        let b = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // 2 rows × 3 cols
        let out = concat_2d_axis1(&a, 2, 2, &b, 3);
        assert_eq!(
            out,
            vec![
                1.0, 2.0, 5.0, 6.0, 7.0, // row 0
                3.0, 4.0, 8.0, 9.0, 10.0, // row 1
            ]
        );
    }

    #[test]
    fn concat_2d_axis1_single_col_each() {
        let a = [1.0, 3.0]; // 2×1
        let b = [2.0, 4.0]; // 2×1
        let out = concat_2d_axis1(&a, 2, 1, &b, 1);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn concat_2d_axis1_single_row() {
        let a = [1.0, 2.0]; // 1×2
        let b = [3.0, 4.0, 5.0]; // 1×3
        let out = concat_2d_axis1(&a, 1, 2, &b, 3);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn concat_2d_axis1_wide() {
        let rows = 2;
        let ca = 8;
        let cb = 8;
        let a: Vec<f32> = (0..rows * ca).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..rows * cb).map(|i| (i + 100) as f32).collect();
        let out = concat_2d_axis1(&a, rows, ca, &b, cb);
        assert_eq!(out.len(), rows * (ca + cb));
        // Check first row
        assert_eq!(&out[..ca], &a[..ca]);
        assert_eq!(&out[ca..ca + cb], &b[..cb]);
    }

    #[test]
    fn concat_2d_axis1_many_rows() {
        let rows = 8;
        let ca = 3;
        let cb = 5;
        let a: Vec<f32> = (0..rows * ca).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..rows * cb).map(|i| i as f32).collect();
        let out = concat_2d_axis1(&a, rows, ca, &b, cb);
        assert_eq!(out.len(), rows * (ca + cb));
        (0..rows).for_each(|r| {
            let o = r * (ca + cb);
            assert_eq!(&out[o..o + ca], &a[r * ca..r * ca + ca]);
            assert_eq!(&out[o + ca..o + ca + cb], &b[r * cb..r * cb + cb]);
        });
    }

    #[test]
    fn concat_2d_axis1_empty_cols_a() {
        let b = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let out = concat_2d_axis1(&[], 2, 0, &b, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn concat_2d_axis1_empty_cols_b() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let out = concat_2d_axis1(&a, 2, 2, &[], 0);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "a shape mismatch")]
    fn concat_2d_axis1_bad_a() {
        concat_2d_axis1(&[1.0], 2, 2, &[1.0, 2.0, 3.0, 4.0], 2);
    }

    #[test]
    #[should_panic(expected = "b shape mismatch")]
    fn concat_2d_axis1_bad_b() {
        concat_2d_axis1(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[1.0], 2);
    }

    // ── split_1d ───────────────────────────────────────────────

    #[test]
    fn split_1d_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (l, r) = split_1d(&data, 3);
        assert_eq!(l, vec![1.0, 2.0, 3.0]);
        assert_eq!(r, vec![4.0, 5.0]);
    }

    #[test]
    fn split_1d_at_start() {
        let data = [1.0, 2.0, 3.0];
        let (l, r) = split_1d(&data, 0);
        assert!(l.is_empty());
        assert_eq!(r, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn split_1d_at_end() {
        let data = [1.0, 2.0, 3.0];
        let (l, r) = split_1d(&data, 3);
        assert_eq!(l, vec![1.0, 2.0, 3.0]);
        assert!(r.is_empty());
    }

    #[test]
    fn split_1d_empty() {
        let (l, r) = split_1d(&[], 0);
        assert!(l.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn split_1d_single() {
        let (l, r) = split_1d(&[42.0], 1);
        assert_eq!(l, vec![42.0]);
        assert!(r.is_empty());
    }

    #[test]
    fn split_1d_aligned() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (l, r) = split_1d(&data, 8);
        assert_eq!(l.len(), 8);
        assert_eq!(r.len(), 8);
        l.iter().enumerate().for_each(|(i, &v)| assert_eq!(v, i as f32));
        r.iter().enumerate().for_each(|(i, &v)| assert_eq!(v, (i + 8) as f32));
    }

    #[test]
    fn split_1d_unaligned() {
        let data: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let (l, r) = split_1d(&data, 7);
        assert_eq!(l.len(), 7);
        assert_eq!(r.len(), 6);
    }

    #[test]
    #[should_panic(expected = "split index out of bounds")]
    fn split_1d_out_of_bounds() {
        split_1d(&[1.0, 2.0], 3);
    }

    #[test]
    fn split_1d_roundtrip() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let (l, r) = split_1d(&data, 10);
        let rebuilt = concat_1d(&l, &r);
        assert_eq!(rebuilt, data);
    }

    // ── split_2d_axis0 ─────────────────────────────────────────

    #[test]
    fn split_2d_axis0_basic() {
        // 3×2 → split at row 1 → (1×2, 2×2)
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (top, bot) = split_2d_axis0(&data, 3, 2, 1);
        assert_eq!(top, vec![1.0, 2.0]);
        assert_eq!(bot, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn split_2d_axis0_at_zero() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let (top, bot) = split_2d_axis0(&data, 2, 2, 0);
        assert!(top.is_empty());
        assert_eq!(bot, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn split_2d_axis0_at_end() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let (top, bot) = split_2d_axis0(&data, 2, 2, 2);
        assert_eq!(top, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(bot.is_empty());
    }

    #[test]
    fn split_2d_axis0_wide() {
        let cols = 16;
        let rows = 4;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let (top, bot) = split_2d_axis0(&data, rows, cols, 2);
        assert_eq!(top.len(), 2 * cols);
        assert_eq!(bot.len(), 2 * cols);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn split_2d_axis0_bad_shape() {
        split_2d_axis0(&[1.0, 2.0], 2, 2, 1);
    }

    #[test]
    #[should_panic(expected = "split_row out of bounds")]
    fn split_2d_axis0_oob() {
        split_2d_axis0(&[1.0, 2.0, 3.0, 4.0], 2, 2, 3);
    }

    #[test]
    fn split_2d_axis0_roundtrip() {
        let rows = 6;
        let cols = 5;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let (top, bot) = split_2d_axis0(&data, rows, cols, 3);
        let rebuilt = concat_2d_axis0(&top, 3, &bot, 3, cols);
        assert_eq!(rebuilt, data);
    }

    // ── stack_vectors ──────────────────────────────────────────

    #[test]
    fn stack_vectors_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let out = stack_vectors(&[&a, &b], 3);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_vectors_single() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let out = stack_vectors(&[&a], 4);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn stack_vectors_empty_input() {
        let out = stack_vectors(&[], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn stack_vectors_many() {
        let vecs: Vec<Vec<f32>> = (0..10).map(|r| vec![r as f32; 4]).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let out = stack_vectors(&refs, 4);
        assert_eq!(out.len(), 40);
        (0..10).for_each(|r| {
            let row = &out[r * 4..(r + 1) * 4];
            row.iter().for_each(|&v| assert_eq!(v, r as f32));
        });
    }

    #[test]
    fn stack_vectors_wide() {
        let cols = 17;
        let a: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..cols).map(|i| (i + cols) as f32).collect();
        let out = stack_vectors(&[&a, &b], cols);
        assert_eq!(out.len(), 2 * cols);
        assert_eq!(&out[..cols], &a[..]);
        assert_eq!(&out[cols..], &b[..]);
    }

    #[test]
    #[should_panic(expected = "vector length mismatch")]
    fn stack_vectors_mismatched() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        stack_vectors(&[&a, &b], 2);
    }

    #[test]
    fn stack_vectors_zero_cols() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let out = stack_vectors(&[&a, &b], 0);
        assert!(out.is_empty());
    }

    // ── interleave ─────────────────────────────────────────────

    #[test]
    fn interleave_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let out = interleave(&a, &b);
        assert_eq!(out, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn interleave_single() {
        assert_eq!(interleave(&[1.0], &[2.0]), vec![1.0, 2.0]);
    }

    #[test]
    fn interleave_empty() {
        let out: Vec<f32> = interleave(&[], &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn interleave_unaligned() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements
        let b = [10.0, 20.0, 30.0, 40.0, 50.0];
        let out = interleave(&a, &b);
        assert_eq!(out, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0,]);
    }

    #[test]
    fn interleave_large_aligned() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i + 1000) as f32).collect();
        let out = interleave(&a, &b);
        assert_eq!(out.len(), n * 2);
        (0..n).for_each(|i| {
            assert_eq!(out[i * 2], i as f32);
            assert_eq!(out[i * 2 + 1], (i + 1000) as f32);
        });
    }

    #[test]
    fn interleave_large_unaligned() {
        let n = 67;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
        let out = interleave(&a, &b);
        assert_eq!(out.len(), n * 2);
        (0..n).for_each(|i| {
            assert_eq!(out[i * 2], i as f32);
            assert_eq!(out[i * 2 + 1], -(i as f32));
        });
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn interleave_mismatched() {
        interleave(&[1.0, 2.0], &[3.0]);
    }

    #[test]
    fn interleave_two_elements() {
        assert_eq!(interleave(&[1.0, 2.0], &[3.0, 4.0]), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn interleave_three_elements() {
        assert_eq!(
            interleave(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    // ── chunk ──────────────────────────────────────────────────

    #[test]
    fn chunk_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let chunks = chunk(&data, 3);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![1.0, 2.0]);
        assert_eq!(chunks[1], vec![3.0, 4.0]);
        assert_eq!(chunks[2], vec![5.0, 6.0]);
    }

    #[test]
    fn chunk_into_one() {
        let data = [1.0, 2.0, 3.0];
        let chunks = chunk(&data, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn chunk_into_n() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let chunks = chunk(&data, 4);
        assert_eq!(chunks.len(), 4);
        chunks.iter().enumerate().for_each(|(i, c)| assert_eq!(*c, vec![(i + 1) as f32]));
    }

    #[test]
    fn chunk_empty() {
        let chunks = chunk(&[], 1);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_empty());
    }

    #[test]
    fn chunk_large() {
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let chunks = chunk(&data, 4);
        assert_eq!(chunks.len(), 4);
        chunks.iter().for_each(|c| assert_eq!(c.len(), 64));
        assert_eq!(chunks[0][0], 0.0);
        assert_eq!(chunks[1][0], 64.0);
        assert_eq!(chunks[2][0], 128.0);
        assert_eq!(chunks[3][0], 192.0);
    }

    #[test]
    #[should_panic(expected = "n must be > 0")]
    fn chunk_zero_n() {
        chunk(&[1.0], 0);
    }

    #[test]
    #[should_panic(expected = "not divisible")]
    fn chunk_not_divisible() {
        chunk(&[1.0, 2.0, 3.0], 2);
    }

    #[test]
    fn chunk_two_equal() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let chunks = chunk(&data, 2);
        assert_eq!(chunks[0], vec![1.0, 2.0]);
        assert_eq!(chunks[1], vec![3.0, 4.0]);
    }

    #[test]
    fn chunk_preserves_values() {
        let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let chunks = chunk(&data, 5);
        let flat: Vec<f32> = chunks.into_iter().flatten().collect();
        assert_eq!(flat, data);
    }

    // ── roundtrip / integration ────────────────────────────────

    #[test]
    fn concat_split_1d_roundtrip() {
        let a: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let b: Vec<f32> = (10..25).map(|i| i as f32).collect();
        let cat = concat_1d(&a, &b);
        let (la, lb) = split_1d(&cat, a.len());
        assert_eq!(la, a);
        assert_eq!(lb, b);
    }

    #[test]
    fn concat_split_2d_axis0_roundtrip() {
        let cols = 4;
        let a: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let b: Vec<f32> = (12..28).map(|i| i as f32).collect();
        let cat = concat_2d_axis0(&a, 3, &b, 4, cols);
        let (ta, tb) = split_2d_axis0(&cat, 7, cols, 3);
        assert_eq!(ta, a);
        assert_eq!(tb, b);
    }

    #[test]
    fn stack_then_split() {
        let v1 = [1.0, 2.0, 3.0];
        let v2 = [4.0, 5.0, 6.0];
        let v3 = [7.0, 8.0, 9.0];
        let mat = stack_vectors(&[&v1, &v2, &v3], 3);
        assert_eq!(mat.len(), 9);
        let (top, rest) = split_2d_axis0(&mat, 3, 3, 1);
        assert_eq!(top, vec![1.0, 2.0, 3.0]);
        let (mid, bot) = split_2d_axis0(&rest, 2, 3, 1);
        assert_eq!(mid, vec![4.0, 5.0, 6.0]);
        assert_eq!(bot, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn chunk_then_concat() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let chunks = chunk(&data, 3);
        let rebuilt = concat_1d(&concat_1d(&chunks[0], &chunks[1]), &chunks[2]);
        assert_eq!(rebuilt, data);
    }

    #[test]
    fn interleave_pattern_rope() {
        // Simulates cos/sin interleaving for RoPE
        let cos = [1.0, 0.5, 0.0, -0.5];
        let sin = [0.0, 0.5, 1.0, 0.5];
        let paired = interleave(&cos, &sin);
        assert_eq!(paired.len(), 8);
        (0..4).for_each(|i| {
            assert_eq!(paired[i * 2], cos[i]);
            assert_eq!(paired[i * 2 + 1], sin[i]);
        });
    }

    #[test]
    fn concat_axis1_then_split_rows() {
        let a = [1.0, 2.0, 5.0, 6.0]; // 2×2
        let b = [3.0, 4.0, 7.0, 8.0]; // 2×2
        let wide = concat_2d_axis1(&a, 2, 2, &b, 2);
        // wide = [[1,2,3,4],[5,6,7,8]] — 2×4
        assert_eq!(wide, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let (r0, r1) = split_2d_axis0(&wide, 2, 4, 1);
        assert_eq!(r0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r1, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn concat_2d_axis0_large_aligned() {
        let cols = 64;
        let rows_a = 8;
        let rows_b = 8;
        let a: Vec<f32> = (0..rows_a * cols).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..rows_b * cols).map(|i| (i + 1000) as f32).collect();
        let out = concat_2d_axis0(&a, rows_a, &b, rows_b, cols);
        assert_eq!(out.len(), (rows_a + rows_b) * cols);
        assert_eq!(&out[..a.len()], &a[..]);
        assert_eq!(&out[a.len()..], &b[..]);
    }

    #[test]
    fn split_1d_large() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let (l, r) = split_1d(&data, 512);
        assert_eq!(l.len(), 512);
        assert_eq!(r.len(), 512);
        assert_eq!(l[0], 0.0);
        assert_eq!(r[0], 512.0);
    }

    #[test]
    fn interleave_preserves_length() {
        let sizes = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63];
        sizes.iter().for_each(|&n| {
            let a: Vec<f32> = vec![1.0; n];
            let b: Vec<f32> = vec![2.0; n];
            let out = interleave(&a, &b);
            assert_eq!(out.len(), n * 2);
        });
    }

    #[test]
    fn chunk_all_sizes_power_of_two() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        [1, 2, 4, 8, 16, 32, 64].iter().for_each(|&n| {
            let chunks = chunk(&data, n);
            assert_eq!(chunks.len(), n);
            let flat: Vec<f32> = chunks.into_iter().flatten().collect();
            assert_eq!(flat, data);
        });
    }

    #[test]
    fn stack_vectors_identity() {
        // Stacking then flattening gives the original sequence
        let vecs: Vec<Vec<f32>> = (0..5).map(|r| vec![(r * 3) as f32; 3]).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let mat = stack_vectors(&refs, 3);
        let chunks = chunk(&mat, 5);
        chunks.iter().zip(vecs.iter()).for_each(|(c, v)| {
            assert_eq!(c, v);
        });
    }

    // ── additional edge-case tests ─────────────────────────────

    #[test]
    fn concat_1d_mixed_sign() {
        let a = [-1.0, 0.0, 1.0];
        let b = [f32::MIN, f32::MAX];
        let out = concat_1d(&a, &b);
        assert_eq!(out, vec![-1.0, 0.0, 1.0, f32::MIN, f32::MAX]);
    }

    #[test]
    fn concat_2d_axis0_single_col() {
        let a = [1.0, 2.0, 3.0]; // 3×1
        let b = [4.0, 5.0]; // 2×1
        let out = concat_2d_axis0(&a, 3, &b, 2, 1);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn concat_2d_axis1_square() {
        // 2×2 | 2×2 → 2×4
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let out = concat_2d_axis1(&a, 2, 2, &b, 2);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn split_2d_axis0_single_col() {
        let data = [1.0, 2.0, 3.0, 4.0]; // 4×1
        let (top, bot) = split_2d_axis0(&data, 4, 1, 2);
        assert_eq!(top, vec![1.0, 2.0]);
        assert_eq!(bot, vec![3.0, 4.0]);
    }

    #[test]
    fn interleave_nan_inf() {
        let a = [f32::NAN, f32::INFINITY];
        let b = [f32::NEG_INFINITY, 0.0];
        let out = interleave(&a, &b);
        assert!(out[0].is_nan());
        assert_eq!(out[1], f32::NEG_INFINITY);
        assert_eq!(out[2], f32::INFINITY);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn chunk_two_from_eight() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let c = chunk(&data, 2);
        assert_eq!(c[0], vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(c[1], vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn stack_vectors_aligned_16() {
        let cols = 16;
        let v: Vec<f32> = (0..cols).map(|i| i as f32).collect();
        let mat = stack_vectors(&[&v, &v, &v], cols);
        assert_eq!(mat.len(), 3 * cols);
    }
}
