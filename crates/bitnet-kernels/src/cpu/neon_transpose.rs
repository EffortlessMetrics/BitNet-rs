//! ARM NEON-optimized matrix transpose kernels for Apple Silicon.
//!
//! Provides 4×4 block transpose using NEON intrinsics, a general
//! transpose that tiles 4×4 blocks with scalar edges, in-place square
//! matrix transpose, batched transpose, and transpose-add.
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::*;

// ── Internal 4×4 block helper ──────────────────────────────────────

/// Transpose a single 4×4 block from `src` at `(si, sj)` in a `src_cols`-wide
/// row-major layout and store it into `dst` at `(di, dj)` in a `dst_cols`-wide
/// row-major layout.
#[inline(always)]
unsafe fn transpose_4x4_block(
    src: *const f32,
    src_cols: usize,
    si: usize,
    sj: usize,
    dst: *mut f32,
    dst_cols: usize,
    di: usize,
    dj: usize,
) {
    let r0 = vld1q_f32(src.add(si * src_cols + sj));
    let r1 = vld1q_f32(src.add((si + 1) * src_cols + sj));
    let r2 = vld1q_f32(src.add((si + 2) * src_cols + sj));
    let r3 = vld1q_f32(src.add((si + 3) * src_cols + sj));

    // Stage 1: element-level transpose within pairs.
    let t0 = vtrn1q_f32(r0, r1);
    let t1 = vtrn2q_f32(r0, r1);
    let t2 = vtrn1q_f32(r2, r3);
    let t3 = vtrn2q_f32(r2, r3);

    // Stage 2: 64-bit half swap via f64 reinterpret.
    let t0_64 = vreinterpretq_f64_f32(t0);
    let t1_64 = vreinterpretq_f64_f32(t1);
    let t2_64 = vreinterpretq_f64_f32(t2);
    let t3_64 = vreinterpretq_f64_f32(t3);
    let o0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
    let o1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
    let o2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
    let o3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));

    vst1q_f32(dst.add(di * dst_cols + dj), o0);
    vst1q_f32(dst.add((di + 1) * dst_cols + dj), o1);
    vst1q_f32(dst.add((di + 2) * dst_cols + dj), o2);
    vst1q_f32(dst.add((di + 3) * dst_cols + dj), o3);
}

/// Load a 4×4 block, transpose it, add element-wise to existing values in
/// `dst`, and store the result.
#[inline(always)]
unsafe fn transpose_4x4_block_add(
    src: *const f32,
    src_cols: usize,
    si: usize,
    sj: usize,
    dst: *mut f32,
    dst_cols: usize,
    di: usize,
    dj: usize,
) {
    let r0 = vld1q_f32(src.add(si * src_cols + sj));
    let r1 = vld1q_f32(src.add((si + 1) * src_cols + sj));
    let r2 = vld1q_f32(src.add((si + 2) * src_cols + sj));
    let r3 = vld1q_f32(src.add((si + 3) * src_cols + sj));

    let t0 = vtrn1q_f32(r0, r1);
    let t1 = vtrn2q_f32(r0, r1);
    let t2 = vtrn1q_f32(r2, r3);
    let t3 = vtrn2q_f32(r2, r3);

    let t0_64 = vreinterpretq_f64_f32(t0);
    let t1_64 = vreinterpretq_f64_f32(t1);
    let t2_64 = vreinterpretq_f64_f32(t2);
    let t3_64 = vreinterpretq_f64_f32(t3);
    let o0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
    let o1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
    let o2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
    let o3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));

    // Add to existing destination values.
    let d0 = vld1q_f32(dst.add(di * dst_cols + dj));
    let d1 = vld1q_f32(dst.add((di + 1) * dst_cols + dj));
    let d2 = vld1q_f32(dst.add((di + 2) * dst_cols + dj));
    let d3 = vld1q_f32(dst.add((di + 3) * dst_cols + dj));

    vst1q_f32(dst.add(di * dst_cols + dj), vaddq_f32(d0, o0));
    vst1q_f32(dst.add((di + 1) * dst_cols + dj), vaddq_f32(d1, o1));
    vst1q_f32(dst.add((di + 2) * dst_cols + dj), vaddq_f32(d2, o2));
    vst1q_f32(dst.add((di + 3) * dst_cols + dj), vaddq_f32(d3, o3));
}

// ── 4×4 Block Transpose ────────────────────────────────────────────

/// Transpose a row-major `rows × cols` matrix using NEON 4×4 block transpose.
///
/// Both dimensions must be exactly 4-aligned. For arbitrary dimensions
/// use [`neon_transpose_f32`] which handles edge elements with a scalar
/// fallback.
///
/// # Panics
///
/// Panics if `rows` or `cols` is not a multiple of 4, or if slice lengths
/// do not match `rows * cols`.
pub fn neon_transpose_4x4_f32(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    let numel = rows * cols;
    assert!(rows.is_multiple_of(4), "rows must be a multiple of 4, got {rows}");
    assert!(cols.is_multiple_of(4), "cols must be a multiple of 4, got {cols}");
    assert!(src.len() >= numel, "src length {} < rows*cols {numel}", src.len());
    assert!(dst.len() >= numel, "dst length {} < rows*cols {numel}", dst.len());

    for bi in (0..rows).step_by(4) {
        for bj in (0..cols).step_by(4) {
            unsafe {
                transpose_4x4_block(src.as_ptr(), cols, bi, bj, dst.as_mut_ptr(), rows, bj, bi);
            }
        }
    }
}

// ── General Transpose ──────────────────────────────────────────────

/// Transpose a row-major `rows × cols` matrix, using NEON 4×4 blocks
/// for the aligned interior and scalar copy for edge elements.
///
/// # Panics
///
/// Panics if slice lengths do not match `rows * cols`.
pub fn neon_transpose_f32(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    let numel = rows * cols;
    assert!(src.len() >= numel, "src length {} < rows*cols {numel}", src.len());
    assert!(dst.len() >= numel, "dst length {} < rows*cols {numel}", dst.len());

    if numel == 0 {
        return;
    }

    let block_rows = rows & !3;
    let block_cols = cols & !3;

    // Main body: 4×4 NEON blocks.
    for bi in (0..block_rows).step_by(4) {
        for bj in (0..block_cols).step_by(4) {
            unsafe {
                transpose_4x4_block(src.as_ptr(), cols, bi, bj, dst.as_mut_ptr(), rows, bj, bi);
            }
        }
    }

    // Right edge: columns beyond the last full 4-column block.
    for i in 0..block_rows {
        for j in block_cols..cols {
            dst[j * rows + i] = src[i * cols + j];
        }
    }

    // Bottom edge: rows beyond the last full 4-row block (all columns).
    for i in block_rows..rows {
        for j in 0..cols {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// ── In-Place Square Transpose ──────────────────────────────────────

/// In-place transpose of a square `n × n` matrix stored in row-major order.
///
/// # Panics
///
/// Panics if `data.len() < n * n`.
pub fn neon_transpose_inplace_f32(data: &mut [f32], n: usize) {
    let numel = n * n;
    assert!(data.len() >= numel, "data length {} < n*n {numel}", data.len());

    if n <= 1 {
        return;
    }

    let block_n = n & !3;

    for bi in (0..block_n).step_by(4) {
        // Off-diagonal 4×4 blocks: bi < bj — swap A at (bi,bj) with B at (bj,bi).
        for bj in ((bi + 4)..block_n).step_by(4) {
            let (a0, a1, a2, a3, b0, b1, b2, b3);
            unsafe {
                a0 = vld1q_f32(data.as_ptr().add(bi * n + bj));
                a1 = vld1q_f32(data.as_ptr().add((bi + 1) * n + bj));
                a2 = vld1q_f32(data.as_ptr().add((bi + 2) * n + bj));
                a3 = vld1q_f32(data.as_ptr().add((bi + 3) * n + bj));

                b0 = vld1q_f32(data.as_ptr().add(bj * n + bi));
                b1 = vld1q_f32(data.as_ptr().add((bj + 1) * n + bi));
                b2 = vld1q_f32(data.as_ptr().add((bj + 2) * n + bi));
                b3 = vld1q_f32(data.as_ptr().add((bj + 3) * n + bi));
            }

            // Transpose A → write to B's position.
            let (ta0, ta1, ta2, ta3);
            unsafe {
                let at0 = vtrn1q_f32(a0, a1);
                let at1 = vtrn2q_f32(a0, a1);
                let at2 = vtrn1q_f32(a2, a3);
                let at3 = vtrn2q_f32(a2, a3);
                let at0_64 = vreinterpretq_f64_f32(at0);
                let at1_64 = vreinterpretq_f64_f32(at1);
                let at2_64 = vreinterpretq_f64_f32(at2);
                let at3_64 = vreinterpretq_f64_f32(at3);
                ta0 = vreinterpretq_f32_f64(vtrn1q_f64(at0_64, at2_64));
                ta1 = vreinterpretq_f32_f64(vtrn1q_f64(at1_64, at3_64));
                ta2 = vreinterpretq_f32_f64(vtrn2q_f64(at0_64, at2_64));
                ta3 = vreinterpretq_f32_f64(vtrn2q_f64(at1_64, at3_64));
            }

            // Transpose B → write to A's position.
            let (tb0, tb1, tb2, tb3);
            unsafe {
                let bt0 = vtrn1q_f32(b0, b1);
                let bt1 = vtrn2q_f32(b0, b1);
                let bt2 = vtrn1q_f32(b2, b3);
                let bt3 = vtrn2q_f32(b2, b3);
                let bt0_64 = vreinterpretq_f64_f32(bt0);
                let bt1_64 = vreinterpretq_f64_f32(bt1);
                let bt2_64 = vreinterpretq_f64_f32(bt2);
                let bt3_64 = vreinterpretq_f64_f32(bt3);
                tb0 = vreinterpretq_f32_f64(vtrn1q_f64(bt0_64, bt2_64));
                tb1 = vreinterpretq_f32_f64(vtrn1q_f64(bt1_64, bt3_64));
                tb2 = vreinterpretq_f32_f64(vtrn2q_f64(bt0_64, bt2_64));
                tb3 = vreinterpretq_f32_f64(vtrn2q_f64(bt1_64, bt3_64));
            }

            unsafe {
                vst1q_f32(data.as_mut_ptr().add(bj * n + bi), ta0);
                vst1q_f32(data.as_mut_ptr().add((bj + 1) * n + bi), ta1);
                vst1q_f32(data.as_mut_ptr().add((bj + 2) * n + bi), ta2);
                vst1q_f32(data.as_mut_ptr().add((bj + 3) * n + bi), ta3);

                vst1q_f32(data.as_mut_ptr().add(bi * n + bj), tb0);
                vst1q_f32(data.as_mut_ptr().add((bi + 1) * n + bj), tb1);
                vst1q_f32(data.as_mut_ptr().add((bi + 2) * n + bj), tb2);
                vst1q_f32(data.as_mut_ptr().add((bi + 3) * n + bj), tb3);
            }
        }

        // Diagonal 4×4 block at (bi, bi): scalar swaps.
        for i in bi..bi + 4 {
            for j in (i + 1)..bi + 4 {
                data.swap(i * n + j, j * n + i);
            }
        }
    }

    // Remainder rows/columns beyond the last full 4-block boundary.
    for i in 0..n {
        let j_start = if i < block_n { block_n } else { i + 1 };
        for j in j_start..n {
            data.swap(i * n + j, j * n + i);
        }
    }
}

// ── Batched Transpose ──────────────────────────────────────────────

/// Transpose each matrix in a contiguous batch of `batch` row-major
/// `rows × cols` matrices. The output for each matrix is `cols × rows`.
///
/// # Panics
///
/// Panics if `src.len() < batch * rows * cols` or
/// `dst.len() < batch * rows * cols`.
pub fn neon_transpose_batch_f32(
    src: &[f32],
    dst: &mut [f32],
    batch: usize,
    rows: usize,
    cols: usize,
) {
    let mat_size = rows * cols;
    let total = batch * mat_size;
    assert!(src.len() >= total, "src length {} < batch*rows*cols {total}", src.len());
    assert!(dst.len() >= total, "dst length {} < batch*rows*cols {total}", dst.len());

    for b in 0..batch {
        let offset = b * mat_size;
        neon_transpose_f32(
            &src[offset..offset + mat_size],
            &mut dst[offset..offset + mat_size],
            rows,
            cols,
        );
    }
}

// ── Transpose-Add ──────────────────────────────────────────────────

/// Transpose a row-major `rows × cols` matrix and **add** the result
/// element-wise to `dst` (which must already be `cols × rows`).
///
/// `dst[j * rows + i] += src[i * cols + j]`
///
/// # Panics
///
/// Panics if slice lengths do not match `rows * cols`.
pub fn neon_transpose_add_f32(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    let numel = rows * cols;
    assert!(src.len() >= numel, "src length {} < rows*cols {numel}", src.len());
    assert!(dst.len() >= numel, "dst length {} < rows*cols {numel}", dst.len());

    if numel == 0 {
        return;
    }

    let block_rows = rows & !3;
    let block_cols = cols & !3;

    // Main body: 4×4 NEON blocks with fused transpose+add.
    for bi in (0..block_rows).step_by(4) {
        for bj in (0..block_cols).step_by(4) {
            unsafe {
                transpose_4x4_block_add(src.as_ptr(), cols, bi, bj, dst.as_mut_ptr(), rows, bj, bi);
            }
        }
    }

    // Right edge.
    for i in 0..block_rows {
        for j in block_cols..cols {
            dst[j * rows + i] += src[i * cols + j];
        }
    }

    // Bottom edge.
    for i in block_rows..rows {
        for j in 0..cols {
            dst[j * rows + i] += src[i * cols + j];
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a row-major matrix where value = row * cols + col + 1.
    fn make_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|i| (i + 1) as f32).collect()
    }

    /// Scalar reference transpose.
    fn scalar_transpose(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = input[i * cols + j];
            }
        }
        out
    }

    /// Scalar reference transpose-add (dst += transpose(src)).
    fn scalar_transpose_add(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] += src[i * cols + j];
            }
        }
    }

    // ── neon_transpose_4x4_f32 ────────────────────────────────

    #[test]
    fn test_4x4_basic() {
        let src = make_matrix(4, 4);
        let mut dst = vec![0.0f32; 16];
        neon_transpose_4x4_f32(&src, &mut dst, 4, 4);
        assert_eq!(dst, scalar_transpose(&src, 4, 4));
    }

    #[test]
    fn test_4x4_8x8() {
        let src = make_matrix(8, 8);
        let mut dst = vec![0.0f32; 64];
        neon_transpose_4x4_f32(&src, &mut dst, 8, 8);
        assert_eq!(dst, scalar_transpose(&src, 8, 8));
    }

    #[test]
    fn test_4x4_non_square_4x8() {
        let src = make_matrix(4, 8);
        let mut dst = vec![0.0f32; 32];
        neon_transpose_4x4_f32(&src, &mut dst, 4, 8);
        assert_eq!(dst, scalar_transpose(&src, 4, 8));
    }

    #[test]
    fn test_4x4_non_square_8x4() {
        let src = make_matrix(8, 4);
        let mut dst = vec![0.0f32; 32];
        neon_transpose_4x4_f32(&src, &mut dst, 8, 4);
        assert_eq!(dst, scalar_transpose(&src, 8, 4));
    }

    #[test]
    fn test_4x4_12x8() {
        let src = make_matrix(12, 8);
        let mut dst = vec![0.0f32; 96];
        neon_transpose_4x4_f32(&src, &mut dst, 12, 8);
        assert_eq!(dst, scalar_transpose(&src, 12, 8));
    }

    #[test]
    fn test_4x4_16x16() {
        let src = make_matrix(16, 16);
        let mut dst = vec![0.0f32; 256];
        neon_transpose_4x4_f32(&src, &mut dst, 16, 16);
        assert_eq!(dst, scalar_transpose(&src, 16, 16));
    }

    #[test]
    fn test_4x4_identity_values() {
        // 4×4 identity matrix should transpose to itself.
        let src =
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut dst = vec![0.0f32; 16];
        neon_transpose_4x4_f32(&src, &mut dst, 4, 4);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_4x4_all_zeros() {
        let src = vec![0.0f32; 64];
        let mut dst = vec![1.0f32; 64];
        neon_transpose_4x4_f32(&src, &mut dst, 8, 8);
        assert!(dst.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_4x4_negative_values() {
        let src: Vec<f32> = (0..16).map(|i| -((i + 1) as f32)).collect();
        let mut dst = vec![0.0f32; 16];
        neon_transpose_4x4_f32(&src, &mut dst, 4, 4);
        assert_eq!(dst, scalar_transpose(&src, 4, 4));
    }

    #[test]
    fn test_4x4_double_transpose_is_identity() {
        let src = make_matrix(8, 12);
        let mut t1 = vec![0.0f32; 96];
        let mut t2 = vec![0.0f32; 96];
        neon_transpose_4x4_f32(&src, &mut t1, 8, 12);
        neon_transpose_4x4_f32(&t1, &mut t2, 12, 8);
        assert_eq!(t2, src);
    }

    #[test]
    #[should_panic(expected = "rows must be a multiple of 4")]
    fn test_4x4_rejects_unaligned_rows() {
        let src = make_matrix(3, 4);
        let mut dst = vec![0.0f32; 12];
        neon_transpose_4x4_f32(&src, &mut dst, 3, 4);
    }

    #[test]
    #[should_panic(expected = "cols must be a multiple of 4")]
    fn test_4x4_rejects_unaligned_cols() {
        let src = make_matrix(4, 5);
        let mut dst = vec![0.0f32; 20];
        neon_transpose_4x4_f32(&src, &mut dst, 4, 5);
    }

    #[test]
    #[should_panic(expected = "src length")]
    fn test_4x4_rejects_short_src() {
        let src = vec![0.0f32; 10];
        let mut dst = vec![0.0f32; 16];
        neon_transpose_4x4_f32(&src, &mut dst, 4, 4);
    }

    #[test]
    #[should_panic(expected = "dst length")]
    fn test_4x4_rejects_short_dst() {
        let src = vec![0.0f32; 16];
        let mut dst = vec![0.0f32; 10];
        neon_transpose_4x4_f32(&src, &mut dst, 4, 4);
    }

    // ── neon_transpose_f32 (general) ──────────────────────────

    #[test]
    fn test_general_4x4() {
        let src = make_matrix(4, 4);
        let mut dst = vec![0.0f32; 16];
        neon_transpose_f32(&src, &mut dst, 4, 4);
        assert_eq!(dst, scalar_transpose(&src, 4, 4));
    }

    #[test]
    fn test_general_8x8() {
        let src = make_matrix(8, 8);
        let mut dst = vec![0.0f32; 64];
        neon_transpose_f32(&src, &mut dst, 8, 8);
        assert_eq!(dst, scalar_transpose(&src, 8, 8));
    }

    #[test]
    fn test_general_3x5() {
        let src = make_matrix(3, 5);
        let mut dst = vec![0.0f32; 15];
        neon_transpose_f32(&src, &mut dst, 3, 5);
        assert_eq!(dst, scalar_transpose(&src, 3, 5));
    }

    #[test]
    fn test_general_7x3() {
        let src = make_matrix(7, 3);
        let mut dst = vec![0.0f32; 21];
        neon_transpose_f32(&src, &mut dst, 7, 3);
        assert_eq!(dst, scalar_transpose(&src, 7, 3));
    }

    #[test]
    fn test_general_5x8() {
        let src = make_matrix(5, 8);
        let mut dst = vec![0.0f32; 40];
        neon_transpose_f32(&src, &mut dst, 5, 8);
        assert_eq!(dst, scalar_transpose(&src, 5, 8));
    }

    #[test]
    fn test_general_1xn() {
        let n = 13;
        let src = make_matrix(1, n);
        let mut dst = vec![0.0f32; n];
        neon_transpose_f32(&src, &mut dst, 1, n);
        assert_eq!(dst, scalar_transpose(&src, 1, n));
    }

    #[test]
    fn test_general_nx1() {
        let n = 11;
        let src = make_matrix(n, 1);
        let mut dst = vec![0.0f32; n];
        neon_transpose_f32(&src, &mut dst, n, 1);
        assert_eq!(dst, scalar_transpose(&src, n, 1));
    }

    #[test]
    fn test_general_1x1() {
        let src = [42.0f32];
        let mut dst = [0.0f32; 1];
        neon_transpose_f32(&src, &mut dst, 1, 1);
        assert_eq!(dst[0], 42.0);
    }

    #[test]
    fn test_general_2x2() {
        let src = make_matrix(2, 2);
        let mut dst = vec![0.0f32; 4];
        neon_transpose_f32(&src, &mut dst, 2, 2);
        assert_eq!(dst, scalar_transpose(&src, 2, 2));
    }

    #[test]
    fn test_general_2x3() {
        let src = make_matrix(2, 3);
        let mut dst = vec![0.0f32; 6];
        neon_transpose_f32(&src, &mut dst, 2, 3);
        assert_eq!(dst, scalar_transpose(&src, 2, 3));
    }

    #[test]
    fn test_general_5x5() {
        let src = make_matrix(5, 5);
        let mut dst = vec![0.0f32; 25];
        neon_transpose_f32(&src, &mut dst, 5, 5);
        assert_eq!(dst, scalar_transpose(&src, 5, 5));
    }

    #[test]
    fn test_general_6x10() {
        let src = make_matrix(6, 10);
        let mut dst = vec![0.0f32; 60];
        neon_transpose_f32(&src, &mut dst, 6, 10);
        assert_eq!(dst, scalar_transpose(&src, 6, 10));
    }

    #[test]
    fn test_general_9x9() {
        let src = make_matrix(9, 9);
        let mut dst = vec![0.0f32; 81];
        neon_transpose_f32(&src, &mut dst, 9, 9);
        assert_eq!(dst, scalar_transpose(&src, 9, 9));
    }

    #[test]
    fn test_general_large_33x65() {
        let (rows, cols) = (33, 65);
        let src = make_matrix(rows, cols);
        let mut dst = vec![0.0f32; rows * cols];
        neon_transpose_f32(&src, &mut dst, rows, cols);
        assert_eq!(dst, scalar_transpose(&src, rows, cols));
    }

    #[test]
    fn test_general_large_64x64() {
        let src = make_matrix(64, 64);
        let mut dst = vec![0.0f32; 64 * 64];
        neon_transpose_f32(&src, &mut dst, 64, 64);
        assert_eq!(dst, scalar_transpose(&src, 64, 64));
    }

    #[test]
    fn test_general_empty() {
        let mut dst = vec![0.0f32; 0];
        neon_transpose_f32(&[], &mut dst, 0, 0);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_general_double_transpose_identity() {
        let (rows, cols) = (6, 10);
        let src = make_matrix(rows, cols);
        let mut t1 = vec![0.0f32; rows * cols];
        neon_transpose_f32(&src, &mut t1, rows, cols);
        let mut t2 = vec![0.0f32; rows * cols];
        neon_transpose_f32(&t1, &mut t2, cols, rows);
        assert_eq!(t2, src);
    }

    #[test]
    fn test_general_double_transpose_identity_odd() {
        let (rows, cols) = (7, 11);
        let src = make_matrix(rows, cols);
        let mut t1 = vec![0.0f32; rows * cols];
        neon_transpose_f32(&src, &mut t1, rows, cols);
        let mut t2 = vec![0.0f32; rows * cols];
        neon_transpose_f32(&t1, &mut t2, cols, rows);
        assert_eq!(t2, src);
    }

    #[test]
    fn test_general_negative_values() {
        let src: Vec<f32> = (0..35).map(|i| -((i + 1) as f32) * 0.5).collect();
        let mut dst = vec![0.0f32; 35];
        neon_transpose_f32(&src, &mut dst, 5, 7);
        assert_eq!(dst, scalar_transpose(&src, 5, 7));
    }

    #[test]
    fn test_general_agrees_with_4x4_on_aligned() {
        let src = make_matrix(8, 12);
        let mut dst_gen = vec![0.0f32; 96];
        let mut dst_blk = vec![0.0f32; 96];
        neon_transpose_f32(&src, &mut dst_gen, 8, 12);
        neon_transpose_4x4_f32(&src, &mut dst_blk, 8, 12);
        assert_eq!(dst_gen, dst_blk);
    }

    #[test]
    fn test_general_1x4() {
        let src = make_matrix(1, 4);
        let mut dst = vec![0.0f32; 4];
        neon_transpose_f32(&src, &mut dst, 1, 4);
        assert_eq!(dst, scalar_transpose(&src, 1, 4));
    }

    #[test]
    fn test_general_4x1() {
        let src = make_matrix(4, 1);
        let mut dst = vec![0.0f32; 4];
        neon_transpose_f32(&src, &mut dst, 4, 1);
        assert_eq!(dst, scalar_transpose(&src, 4, 1));
    }

    #[test]
    fn test_general_3x4() {
        let src = make_matrix(3, 4);
        let mut dst = vec![0.0f32; 12];
        neon_transpose_f32(&src, &mut dst, 3, 4);
        assert_eq!(dst, scalar_transpose(&src, 3, 4));
    }

    #[test]
    fn test_general_4x3() {
        let src = make_matrix(4, 3);
        let mut dst = vec![0.0f32; 12];
        neon_transpose_f32(&src, &mut dst, 4, 3);
        assert_eq!(dst, scalar_transpose(&src, 4, 3));
    }

    // ── neon_transpose_inplace_f32 ────────────────────────────

    #[test]
    fn test_inplace_1x1() {
        let mut data = [99.0f32];
        neon_transpose_inplace_f32(&mut data, 1);
        assert_eq!(data[0], 99.0);
    }

    #[test]
    fn test_inplace_2x2() {
        let original = make_matrix(2, 2);
        let expected = scalar_transpose(&original, 2, 2);
        let mut data = original;
        neon_transpose_inplace_f32(&mut data, 2);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_3x3() {
        let original = make_matrix(3, 3);
        let expected = scalar_transpose(&original, 3, 3);
        let mut data = original;
        neon_transpose_inplace_f32(&mut data, 3);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_4x4() {
        let original = make_matrix(4, 4);
        let expected = scalar_transpose(&original, 4, 4);
        let mut data = original;
        neon_transpose_inplace_f32(&mut data, 4);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_5x5() {
        let original = make_matrix(5, 5);
        let expected = scalar_transpose(&original, 5, 5);
        let mut data = original;
        neon_transpose_inplace_f32(&mut data, 5);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_7x7() {
        let original = make_matrix(7, 7);
        let expected = scalar_transpose(&original, 7, 7);
        let mut data = original;
        neon_transpose_inplace_f32(&mut data, 7);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_8x8() {
        let original = make_matrix(8, 8);
        let expected = scalar_transpose(&original, 8, 8);
        let mut data = original;
        neon_transpose_inplace_f32(&mut data, 8);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_16x16() {
        let original = make_matrix(16, 16);
        let expected = scalar_transpose(&original, 16, 16);
        let mut data = original;
        neon_transpose_inplace_f32(&mut data, 16);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_35x35() {
        let original = make_matrix(35, 35);
        let expected = scalar_transpose(&original, 35, 35);
        let mut data = original;
        neon_transpose_inplace_f32(&mut data, 35);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_involution_4x4() {
        let original = make_matrix(4, 4);
        let mut data = original.clone();
        neon_transpose_inplace_f32(&mut data, 4);
        neon_transpose_inplace_f32(&mut data, 4);
        assert_eq!(data, original);
    }

    #[test]
    fn test_inplace_involution_10x10() {
        let original = make_matrix(10, 10);
        let mut data = original.clone();
        neon_transpose_inplace_f32(&mut data, 10);
        neon_transpose_inplace_f32(&mut data, 10);
        assert_eq!(data, original);
    }

    #[test]
    fn test_inplace_involution_13x13() {
        let original = make_matrix(13, 13);
        let mut data = original.clone();
        neon_transpose_inplace_f32(&mut data, 13);
        neon_transpose_inplace_f32(&mut data, 13);
        assert_eq!(data, original);
    }

    #[test]
    fn test_inplace_agrees_with_out_of_place() {
        let dim = 12;
        let original = make_matrix(dim, dim);
        let mut inplace = original.clone();
        neon_transpose_inplace_f32(&mut inplace, dim);
        let mut oop = vec![0.0f32; dim * dim];
        neon_transpose_f32(&original, &mut oop, dim, dim);
        assert_eq!(inplace, oop);
    }

    #[test]
    fn test_inplace_identity_matrix() {
        let n = 8;
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let original = data.clone();
        neon_transpose_inplace_f32(&mut data, n);
        assert_eq!(data, original);
    }

    #[test]
    fn test_inplace_symmetric_matrix() {
        let n = 6;
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = (i + j) as f32;
            }
        }
        let original = data.clone();
        neon_transpose_inplace_f32(&mut data, n);
        assert_eq!(data, original);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_inplace_rejects_short_data() {
        let mut data = vec![0.0f32; 10];
        neon_transpose_inplace_f32(&mut data, 4);
    }

    // ── neon_transpose_batch_f32 ──────────────────────────────

    #[test]
    fn test_batch_single() {
        let src = make_matrix(3, 5);
        let mut dst = vec![0.0f32; 15];
        neon_transpose_batch_f32(&src, &mut dst, 1, 3, 5);
        assert_eq!(dst, scalar_transpose(&src, 3, 5));
    }

    #[test]
    fn test_batch_two_4x4() {
        let src = make_matrix(4, 4);
        let combined: Vec<f32> = src.iter().chain(src.iter()).copied().collect();
        let mut dst = vec![0.0f32; 32];
        neon_transpose_batch_f32(&combined, &mut dst, 2, 4, 4);
        let expected = scalar_transpose(&src, 4, 4);
        assert_eq!(&dst[..16], &expected[..]);
        assert_eq!(&dst[16..], &expected[..]);
    }

    #[test]
    fn test_batch_three_3x5() {
        let rows = 3;
        let cols = 5;
        let n = rows * cols;
        let single = make_matrix(rows, cols);
        let mut src = Vec::with_capacity(3 * n);
        for _ in 0..3 {
            src.extend_from_slice(&single);
        }
        let mut dst = vec![0.0f32; 3 * n];
        neon_transpose_batch_f32(&src, &mut dst, 3, rows, cols);
        let expected = scalar_transpose(&single, rows, cols);
        for b in 0..3 {
            assert_eq!(&dst[b * n..(b + 1) * n], &expected[..]);
        }
    }

    #[test]
    fn test_batch_consistency_with_single() {
        let (rows, cols) = (5, 7);
        let n = rows * cols;
        let batch = 4;
        let src: Vec<f32> = (0..batch * n).map(|i| i as f32 * 0.1).collect();
        let mut dst_batch = vec![0.0f32; batch * n];
        neon_transpose_batch_f32(&src, &mut dst_batch, batch, rows, cols);

        for b in 0..batch {
            let off = b * n;
            let mut dst_single = vec![0.0f32; n];
            neon_transpose_f32(&src[off..off + n], &mut dst_single, rows, cols);
            assert_eq!(&dst_batch[off..off + n], &dst_single[..]);
        }
    }

    #[test]
    fn test_batch_zero() {
        let mut dst = vec![0.0f32; 0];
        neon_transpose_batch_f32(&[], &mut dst, 0, 3, 5);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_batch_1x1_matrices() {
        let src = vec![1.0, 2.0, 3.0f32];
        let mut dst = vec![0.0f32; 3];
        neon_transpose_batch_f32(&src, &mut dst, 3, 1, 1);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_batch_large() {
        let (rows, cols) = (8, 12);
        let n = rows * cols;
        let batch = 10;
        let src: Vec<f32> = (0..batch * n).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; batch * n];
        neon_transpose_batch_f32(&src, &mut dst, batch, rows, cols);
        for b in 0..batch {
            let off = b * n;
            let expected = scalar_transpose(&src[off..off + n], rows, cols);
            assert_eq!(&dst[off..off + n], &expected[..]);
        }
    }

    #[test]
    #[should_panic(expected = "src length")]
    fn test_batch_rejects_short_src() {
        let src = vec![0.0f32; 10];
        let mut dst = vec![0.0f32; 30];
        neon_transpose_batch_f32(&src, &mut dst, 2, 3, 5);
    }

    #[test]
    #[should_panic(expected = "dst length")]
    fn test_batch_rejects_short_dst() {
        let src = vec![0.0f32; 30];
        let mut dst = vec![0.0f32; 10];
        neon_transpose_batch_f32(&src, &mut dst, 2, 3, 5);
    }

    // ── neon_transpose_add_f32 ────────────────────────────────

    #[test]
    fn test_add_basic_4x4() {
        let src = make_matrix(4, 4);
        let mut dst = vec![0.0f32; 16];
        neon_transpose_add_f32(&src, &mut dst, 4, 4);
        assert_eq!(dst, scalar_transpose(&src, 4, 4));
    }

    #[test]
    fn test_add_accumulates() {
        let src = make_matrix(4, 4);
        let mut dst = vec![1.0f32; 16];
        neon_transpose_add_f32(&src, &mut dst, 4, 4);
        let t = scalar_transpose(&src, 4, 4);
        let expected: Vec<f32> = t.iter().map(|&x| x + 1.0).collect();
        assert_eq!(dst, expected);
    }

    #[test]
    fn test_add_non_aligned_3x5() {
        let src = make_matrix(3, 5);
        let mut dst = vec![0.0f32; 15];
        neon_transpose_add_f32(&src, &mut dst, 3, 5);
        assert_eq!(dst, scalar_transpose(&src, 3, 5));
    }

    #[test]
    fn test_add_non_aligned_7x3() {
        let src = make_matrix(7, 3);
        let mut dst = vec![0.0f32; 21];
        neon_transpose_add_f32(&src, &mut dst, 7, 3);
        assert_eq!(dst, scalar_transpose(&src, 7, 3));
    }

    #[test]
    fn test_add_double_equals_2x_transpose() {
        let src = make_matrix(5, 8);
        let n = 40;
        let mut dst = vec![0.0f32; n];
        neon_transpose_add_f32(&src, &mut dst, 5, 8);
        neon_transpose_add_f32(&src, &mut dst, 5, 8);
        let t = scalar_transpose(&src, 5, 8);
        let expected: Vec<f32> = t.iter().map(|&x| x * 2.0).collect();
        assert_eq!(dst, expected);
    }

    #[test]
    fn test_add_with_scalar_reference() {
        let (rows, cols) = (6, 9);
        let n = rows * cols;
        let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3).collect();
        let mut dst_neon = vec![10.0f32; n];
        let mut dst_scalar = vec![10.0f32; n];
        neon_transpose_add_f32(&src, &mut dst_neon, rows, cols);
        scalar_transpose_add(&src, &mut dst_scalar, rows, cols);
        for (a, b) in dst_neon.iter().zip(dst_scalar.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_add_empty() {
        let mut dst = vec![0.0f32; 0];
        neon_transpose_add_f32(&[], &mut dst, 0, 0);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_add_1x1() {
        let src = [5.0f32];
        let mut dst = [3.0f32];
        neon_transpose_add_f32(&src, &mut dst, 1, 1);
        assert_eq!(dst[0], 8.0);
    }

    #[test]
    fn test_add_negative_src() {
        let src: Vec<f32> = (0..20).map(|i| -((i + 1) as f32)).collect();
        let mut dst = vec![100.0f32; 20];
        let mut expected = vec![100.0f32; 20];
        scalar_transpose_add(&src, &mut expected, 4, 5);
        neon_transpose_add_f32(&src, &mut dst, 4, 5);
        assert_eq!(dst, expected);
    }

    #[test]
    fn test_add_large_8x12() {
        let (rows, cols) = (8, 12);
        let n = rows * cols;
        let src = make_matrix(rows, cols);
        let mut dst = vec![0.5f32; n];
        let mut expected = vec![0.5f32; n];
        scalar_transpose_add(&src, &mut expected, rows, cols);
        neon_transpose_add_f32(&src, &mut dst, rows, cols);
        assert_eq!(dst, expected);
    }

    // ── Cross-function property tests ─────────────────────────

    #[test]
    fn test_transpose_of_row_vector() {
        // Transposing a 1×N row vector gives an N×1 column vector.
        let n = 7;
        let src = make_matrix(1, n);
        let mut dst = vec![0.0f32; n];
        neon_transpose_f32(&src, &mut dst, 1, n);
        // Column vector: each element is a separate row.
        for i in 0..n {
            assert_eq!(dst[i], src[i]);
        }
    }

    #[test]
    fn test_transpose_of_col_vector() {
        let n = 9;
        let src = make_matrix(n, 1);
        let mut dst = vec![0.0f32; n];
        neon_transpose_f32(&src, &mut dst, n, 1);
        for i in 0..n {
            assert_eq!(dst[i], src[i]);
        }
    }

    #[test]
    fn test_transpose_preserves_trace() {
        // tr(A) == tr(A^T) for square matrices.
        let dim = 10;
        let src = make_matrix(dim, dim);
        let mut dst = vec![0.0f32; dim * dim];
        neon_transpose_f32(&src, &mut dst, dim, dim);
        let trace_src: f32 = (0..dim).map(|i| src[i * dim + i]).sum();
        let trace_dst: f32 = (0..dim).map(|i| dst[i * dim + i]).sum();
        assert!((trace_src - trace_dst).abs() < 1e-5);
    }

    #[test]
    fn test_inplace_preserves_trace() {
        let dim = 9;
        let original = make_matrix(dim, dim);
        let trace_before: f32 = (0..dim).map(|i| original[i * dim + i]).sum();
        let mut data = original;
        neon_transpose_inplace_f32(&mut data, dim);
        let trace_after: f32 = (0..dim).map(|i| data[i * dim + i]).sum();
        assert!((trace_before - trace_after).abs() < 1e-5);
    }

    #[test]
    fn test_transpose_diagonal_unchanged() {
        let dim = 8;
        let src = make_matrix(dim, dim);
        let mut dst = vec![0.0f32; dim * dim];
        neon_transpose_f32(&src, &mut dst, dim, dim);
        for i in 0..dim {
            assert_eq!(src[i * dim + i], dst[i * dim + i]);
        }
    }

    #[test]
    fn test_batch_double_transpose_identity() {
        let (rows, cols) = (5, 7);
        let n = rows * cols;
        let batch = 3;
        let src: Vec<f32> = (0..batch * n).map(|i| i as f32).collect();
        let mut t1 = vec![0.0f32; batch * n];
        let mut t2 = vec![0.0f32; batch * n];
        neon_transpose_batch_f32(&src, &mut t1, batch, rows, cols);
        neon_transpose_batch_f32(&t1, &mut t2, batch, cols, rows);
        assert_eq!(t2, src);
    }

    #[test]
    fn test_add_then_subtract_roundtrip() {
        let (rows, cols) = (5, 6);
        let n = rows * cols;
        let src = make_matrix(rows, cols);
        let initial = vec![50.0f32; n];
        let mut dst = initial.clone();
        neon_transpose_add_f32(&src, &mut dst, rows, cols);
        // Subtract transposed values to recover initial.
        let t = scalar_transpose(&src, rows, cols);
        let recovered: Vec<f32> = dst.iter().zip(t.iter()).map(|(d, t)| d - t).collect();
        for (a, b) in recovered.iter().zip(initial.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_inplace_agrees_with_out_of_place_odd() {
        let dim = 11;
        let original = make_matrix(dim, dim);
        let mut inplace = original.clone();
        neon_transpose_inplace_f32(&mut inplace, dim);
        let mut oop = vec![0.0f32; dim * dim];
        neon_transpose_f32(&original, &mut oop, dim, dim);
        assert_eq!(inplace, oop);
    }

    #[test]
    fn test_general_15x17() {
        let src = make_matrix(15, 17);
        let mut dst = vec![0.0f32; 15 * 17];
        neon_transpose_f32(&src, &mut dst, 15, 17);
        assert_eq!(dst, scalar_transpose(&src, 15, 17));
    }

    #[test]
    fn test_add_5x8_accumulate_three_times() {
        let (rows, cols) = (5, 8);
        let n = rows * cols;
        let src = make_matrix(rows, cols);
        let mut dst = vec![0.0f32; n];
        neon_transpose_add_f32(&src, &mut dst, rows, cols);
        neon_transpose_add_f32(&src, &mut dst, rows, cols);
        neon_transpose_add_f32(&src, &mut dst, rows, cols);
        let t = scalar_transpose(&src, rows, cols);
        let expected: Vec<f32> = t.iter().map(|&x| x * 3.0).collect();
        assert_eq!(dst, expected);
    }

    #[test]
    fn test_4x4_involution_non_square() {
        let (rows, cols) = (4, 12);
        let src = make_matrix(rows, cols);
        let mut t1 = vec![0.0f32; rows * cols];
        let mut t2 = vec![0.0f32; rows * cols];
        neon_transpose_4x4_f32(&src, &mut t1, rows, cols);
        neon_transpose_4x4_f32(&t1, &mut t2, cols, rows);
        assert_eq!(t2, src);
    }

    #[test]
    fn test_general_all_same_value() {
        let val = std::f32::consts::PI;
        let src = vec![val; 6 * 9];
        let mut dst = vec![0.0f32; 6 * 9];
        neon_transpose_f32(&src, &mut dst, 6, 9);
        assert!(dst.iter().all(|&x| (x - val).abs() < 1e-6));
    }

    #[test]
    fn test_batch_preserves_order() {
        // Verify batch index isolation: modify only batch 1.
        let (rows, cols) = (4, 4);
        let n = rows * cols;
        let batch = 3;
        let mut src = vec![0.0f32; batch * n];
        // Only fill batch 1.
        for i in 0..n {
            src[n + i] = (i + 1) as f32;
        }
        let mut dst = vec![0.0f32; batch * n];
        neon_transpose_batch_f32(&src, &mut dst, batch, rows, cols);
        // Batch 0 and 2 should remain zero.
        assert!(dst[..n].iter().all(|&x| x == 0.0));
        assert!(dst[2 * n..].iter().all(|&x| x == 0.0));
        // Batch 1 should be transposed.
        assert_eq!(&dst[n..2 * n], &scalar_transpose(&src[n..2 * n], rows, cols)[..]);
    }
}
