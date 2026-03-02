//! AVX2-optimized matrix transpose kernels with scalar fallback.
//!
//! Provides SIMD-accelerated transpose operations for `f32` and `i8` data,
//! including blocked (cache-friendly), batched, strided, and in-place
//! variants.  On x86-64 targets the implementation uses runtime AVX2
//! detection and falls back to portable scalar code when AVX2 is
//! unavailable.

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Public API ─────────────────────────────────────────────────────

/// Transpose a row-major `rows × cols` f32 matrix.
///
/// Uses AVX2 8×8 block transpose when available, with a scalar fallback
/// for remaining edge elements and non-x86 targets.
pub fn transpose_f32(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let numel = rows * cols;
    assert!(input.len() >= numel, "input length {} < rows*cols {numel}", input.len());
    if numel == 0 {
        return vec![];
    }

    let mut output = vec![0.0f32; numel];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: AVX2 confirmed at runtime.
            unsafe { transpose_f32_avx2(input, rows, cols, &mut output) };
            return output;
        }
    }

    transpose_f32_scalar(input, rows, cols, &mut output);
    output
}

/// Transpose a row-major `rows × cols` i8 matrix (byte-level).
///
/// Useful for transposing quantized weight / activation buffers that
/// store data as packed bytes.
pub fn transpose_i8(input: &[i8], rows: usize, cols: usize) -> Vec<i8> {
    let numel = rows * cols;
    assert!(input.len() >= numel, "input length {} < rows*cols {numel}", input.len());
    if numel == 0 {
        return vec![];
    }

    let mut output = vec![0i8; numel];
    for i in 0..rows {
        for j in 0..cols {
            output[j * rows + i] = input[i * cols + j];
        }
    }
    output
}

/// In-place transpose of a square `n × n` f32 matrix.
///
/// Panics if `data.len() < n * n`.
pub fn transpose_inplace(data: &mut [f32], n: usize) {
    assert!(data.len() >= n * n, "data length {} < n*n {}", data.len(), n * n);
    for i in 0..n {
        for j in (i + 1)..n {
            data.swap(i * n + j, j * n + i);
        }
    }
}

/// Cache-friendly blocked transpose for large `rows × cols` matrices.
///
/// Tiles the transpose into `block_size × block_size` blocks to improve
/// cache locality. `block_size` must be > 0.
pub fn blocked_transpose(input: &[f32], rows: usize, cols: usize, block_size: usize) -> Vec<f32> {
    assert!(block_size > 0, "block_size must be > 0");
    let numel = rows * cols;
    assert!(input.len() >= numel, "input length {} < rows*cols {numel}", input.len());
    if numel == 0 {
        return vec![];
    }

    let mut output = vec![0.0f32; numel];
    for bi in (0..rows).step_by(block_size) {
        let bi_end = (bi + block_size).min(rows);
        for bj in (0..cols).step_by(block_size) {
            let bj_end = (bj + block_size).min(cols);
            for i in bi..bi_end {
                for j in bj..bj_end {
                    output[j * rows + i] = input[i * cols + j];
                }
            }
        }
    }
    output
}

/// Transpose `count` contiguous `rows × cols` matrices in a batch.
///
/// Each matrix of size `rows * cols` is transposed independently.
/// Input must contain at least `count * rows * cols` elements.
pub fn batch_transpose(input: &[f32], rows: usize, cols: usize, count: usize) -> Vec<f32> {
    let mat_size = rows * cols;
    assert!(
        input.len() >= count * mat_size,
        "input length {} < count*rows*cols {}",
        input.len(),
        count * mat_size
    );
    if mat_size == 0 || count == 0 {
        return vec![];
    }

    let mut output = vec![0.0f32; count * mat_size];
    for k in 0..count {
        let src = &input[k * mat_size..k * mat_size + mat_size];
        let dst = &mut output[k * mat_size..k * mat_size + mat_size];

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { transpose_f32_avx2(src, rows, cols, dst) };
                continue;
            }
        }

        transpose_f32_scalar(src, rows, cols, dst);
    }
    output
}

/// Transpose with non-contiguous (strided) layout.
///
/// Reads a `rows × cols` sub-matrix from `input` where consecutive
/// elements in a row are `col_stride` apart and consecutive rows are
/// `row_stride` apart.  The output is stored contiguously in row-major
/// order with shape `cols × rows`.
pub fn strided_transpose(
    input: &[f32],
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
) -> Vec<f32> {
    if rows == 0 || cols == 0 {
        return vec![];
    }
    // Validate the input is large enough for the furthest addressable element.
    let max_idx = (rows - 1) * row_stride + (cols - 1) * col_stride;
    assert!(
        input.len() > max_idx,
        "input length {} too small for strided access (max index {max_idx})",
        input.len()
    );

    let mut output = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            output[j * rows + i] = input[i * row_stride + j * col_stride];
        }
    }
    output
}

// ── Scalar fallback ────────────────────────────────────────────────

fn transpose_f32_scalar(input: &[f32], rows: usize, cols: usize, output: &mut [f32]) {
    for i in 0..rows {
        for j in 0..cols {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// ── AVX2 inner kernel ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_f32_avx2(input: &[f32], rows: usize, cols: usize, output: &mut [f32]) {
    let row_full = rows & !7; // largest multiple of 8 ≤ rows
    let col_full = cols & !7;

    // Process full 8×8 blocks.
    for bi in (0..row_full).step_by(8) {
        for bj in (0..col_full).step_by(8) {
            unsafe { transpose_8x8_avx2_block(input, output, bi, bj, rows, cols) };
        }
    }

    // Scalar fixup: right edge columns (bj >= col_full).
    for i in 0..rows {
        for j in col_full..cols {
            output[j * rows + i] = input[i * cols + j];
        }
    }
    // Scalar fixup: bottom edge rows (bi >= row_full), only up to col_full
    // to avoid double-writing the corner handled above.
    for i in row_full..rows {
        for j in 0..col_full {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

/// Transpose one 8×8 block starting at `(bi, bj)` in a `rows × cols`
/// matrix.
///
/// Uses the classic AVX2 8×8 f32 transpose algorithm:
///   1. Load 8 rows of 8 floats.
///   2. Interleave pairs with `unpacklo/hi_ps`.
///   3. Shuffle 128-bit lanes with `permute2f128_ps`.
///   4. Store the 8 transposed rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose_8x8_avx2_block(
    input: &[f32],
    output: &mut [f32],
    bi: usize,
    bj: usize,
    rows: usize,
    cols: usize,
) {
    unsafe {
        let ptr = input.as_ptr();

        // Load 8 rows.
        let r0 = _mm256_loadu_ps(ptr.add((bi) * cols + bj));
        let r1 = _mm256_loadu_ps(ptr.add((bi + 1) * cols + bj));
        let r2 = _mm256_loadu_ps(ptr.add((bi + 2) * cols + bj));
        let r3 = _mm256_loadu_ps(ptr.add((bi + 3) * cols + bj));
        let r4 = _mm256_loadu_ps(ptr.add((bi + 4) * cols + bj));
        let r5 = _mm256_loadu_ps(ptr.add((bi + 5) * cols + bj));
        let r6 = _mm256_loadu_ps(ptr.add((bi + 6) * cols + bj));
        let r7 = _mm256_loadu_ps(ptr.add((bi + 7) * cols + bj));

        // Stage 1: interleave 32-bit floats.
        let t0 = _mm256_unpacklo_ps(r0, r1);
        let t1 = _mm256_unpackhi_ps(r0, r1);
        let t2 = _mm256_unpacklo_ps(r2, r3);
        let t3 = _mm256_unpackhi_ps(r2, r3);
        let t4 = _mm256_unpacklo_ps(r4, r5);
        let t5 = _mm256_unpackhi_ps(r4, r5);
        let t6 = _mm256_unpacklo_ps(r6, r7);
        let t7 = _mm256_unpackhi_ps(r6, r7);

        // Stage 2: interleave 64-bit pairs via shuffle.
        let u0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2)));
        let u1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2)));
        let u2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3)));
        let u3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3)));
        let u4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t4), _mm256_castps_pd(t6)));
        let u5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t4), _mm256_castps_pd(t6)));
        let u6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t5), _mm256_castps_pd(t7)));
        let u7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t5), _mm256_castps_pd(t7)));

        // Stage 3: swap 128-bit lanes to complete the transpose.
        let o0 = _mm256_permute2f128_ps(u0, u4, 0x20);
        let o1 = _mm256_permute2f128_ps(u1, u5, 0x20);
        let o2 = _mm256_permute2f128_ps(u2, u6, 0x20);
        let o3 = _mm256_permute2f128_ps(u3, u7, 0x20);
        let o4 = _mm256_permute2f128_ps(u0, u4, 0x31);
        let o5 = _mm256_permute2f128_ps(u1, u5, 0x31);
        let o6 = _mm256_permute2f128_ps(u2, u6, 0x31);
        let o7 = _mm256_permute2f128_ps(u3, u7, 0x31);

        // Store transposed rows.
        let optr = output.as_mut_ptr();
        _mm256_storeu_ps(optr.add((bj) * rows + bi), o0);
        _mm256_storeu_ps(optr.add((bj + 1) * rows + bi), o1);
        _mm256_storeu_ps(optr.add((bj + 2) * rows + bi), o2);
        _mm256_storeu_ps(optr.add((bj + 3) * rows + bi), o3);
        _mm256_storeu_ps(optr.add((bj + 4) * rows + bi), o4);
        _mm256_storeu_ps(optr.add((bj + 5) * rows + bi), o5);
        _mm256_storeu_ps(optr.add((bj + 6) * rows + bi), o6);
        _mm256_storeu_ps(optr.add((bj + 7) * rows + bi), o7);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a row-major matrix with values row*1000 + col.
    fn make_f32_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows).flat_map(|r| (0..cols).map(move |c| (r * 1000 + c) as f32)).collect()
    }

    fn make_i8_matrix(rows: usize, cols: usize) -> Vec<i8> {
        (0..rows).flat_map(|r| (0..cols).map(move |c| ((r * cols + c) % 127) as i8)).collect()
    }

    // Reference scalar transpose for verification.
    fn reference_transpose_f32(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = input[i * cols + j];
            }
        }
        out
    }

    fn reference_transpose_i8(input: &[i8], rows: usize, cols: usize) -> Vec<i8> {
        let mut out = vec![0i8; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = input[i * cols + j];
            }
        }
        out
    }

    // ── transpose_f32 ──────────────────────────────────────────

    #[test]
    fn test_transpose_f32_1x1() {
        let m = vec![42.0f32];
        assert_eq!(transpose_f32(&m, 1, 1), vec![42.0]);
    }

    #[test]
    fn test_transpose_f32_2x3() {
        let m = make_f32_matrix(2, 3);
        let t = transpose_f32(&m, 2, 3);
        assert_eq!(t, reference_transpose_f32(&m, 2, 3));
    }

    #[test]
    fn test_transpose_f32_3x2() {
        let m = make_f32_matrix(3, 2);
        let t = transpose_f32(&m, 3, 2);
        assert_eq!(t, reference_transpose_f32(&m, 3, 2));
    }

    #[test]
    fn test_transpose_f32_4x4() {
        let m = make_f32_matrix(4, 4);
        let t = transpose_f32(&m, 4, 4);
        assert_eq!(t, reference_transpose_f32(&m, 4, 4));
    }

    #[test]
    fn test_transpose_f32_8x8() {
        let m = make_f32_matrix(8, 8);
        let t = transpose_f32(&m, 8, 8);
        assert_eq!(t, reference_transpose_f32(&m, 8, 8));
    }

    #[test]
    fn test_transpose_f32_16x16() {
        let m = make_f32_matrix(16, 16);
        let t = transpose_f32(&m, 16, 16);
        assert_eq!(t, reference_transpose_f32(&m, 16, 16));
    }

    #[test]
    fn test_transpose_f32_8x1() {
        let m = make_f32_matrix(8, 1);
        let t = transpose_f32(&m, 8, 1);
        assert_eq!(t, reference_transpose_f32(&m, 8, 1));
    }

    #[test]
    fn test_transpose_f32_1x8() {
        let m = make_f32_matrix(1, 8);
        let t = transpose_f32(&m, 1, 8);
        assert_eq!(t, reference_transpose_f32(&m, 1, 8));
    }

    #[test]
    fn test_transpose_f32_9x9() {
        let m = make_f32_matrix(9, 9);
        let t = transpose_f32(&m, 9, 9);
        assert_eq!(t, reference_transpose_f32(&m, 9, 9));
    }

    #[test]
    fn test_transpose_f32_7x13() {
        let m = make_f32_matrix(7, 13);
        let t = transpose_f32(&m, 7, 13);
        assert_eq!(t, reference_transpose_f32(&m, 7, 13));
    }

    #[test]
    fn test_transpose_f32_13x7() {
        let m = make_f32_matrix(13, 7);
        let t = transpose_f32(&m, 13, 7);
        assert_eq!(t, reference_transpose_f32(&m, 13, 7));
    }

    #[test]
    fn test_transpose_f32_empty() {
        assert_eq!(transpose_f32(&[], 0, 0), Vec::<f32>::new());
        assert_eq!(transpose_f32(&[], 0, 5), Vec::<f32>::new());
        assert_eq!(transpose_f32(&[], 5, 0), Vec::<f32>::new());
    }

    #[test]
    fn test_transpose_f32_double_transpose_identity() {
        let m = make_f32_matrix(5, 11);
        let t = transpose_f32(&m, 5, 11);
        let tt = transpose_f32(&t, 11, 5);
        assert_eq!(tt, m);
    }

    #[test]
    fn test_transpose_f32_large_16x24() {
        let m = make_f32_matrix(16, 24);
        let t = transpose_f32(&m, 16, 24);
        assert_eq!(t, reference_transpose_f32(&m, 16, 24));
    }

    #[test]
    fn test_transpose_f32_large_24x16() {
        let m = make_f32_matrix(24, 16);
        let t = transpose_f32(&m, 24, 16);
        assert_eq!(t, reference_transpose_f32(&m, 24, 16));
    }

    #[test]
    fn test_transpose_f32_large_32x32() {
        let m = make_f32_matrix(32, 32);
        let t = transpose_f32(&m, 32, 32);
        assert_eq!(t, reference_transpose_f32(&m, 32, 32));
    }

    #[test]
    fn test_transpose_f32_large_64x64() {
        let m = make_f32_matrix(64, 64);
        let t = transpose_f32(&m, 64, 64);
        assert_eq!(t, reference_transpose_f32(&m, 64, 64));
    }

    #[test]
    fn test_transpose_f32_large_100x50() {
        let m = make_f32_matrix(100, 50);
        let t = transpose_f32(&m, 100, 50);
        assert_eq!(t, reference_transpose_f32(&m, 100, 50));
    }

    #[test]
    fn test_transpose_f32_edge_15x17() {
        let m = make_f32_matrix(15, 17);
        let t = transpose_f32(&m, 15, 17);
        assert_eq!(t, reference_transpose_f32(&m, 15, 17));
    }

    #[test]
    fn test_transpose_f32_wide_1x128() {
        let m = make_f32_matrix(1, 128);
        let t = transpose_f32(&m, 1, 128);
        assert_eq!(t, reference_transpose_f32(&m, 1, 128));
    }

    #[test]
    fn test_transpose_f32_tall_128x1() {
        let m = make_f32_matrix(128, 1);
        let t = transpose_f32(&m, 128, 1);
        assert_eq!(t, reference_transpose_f32(&m, 128, 1));
    }

    #[test]
    fn test_transpose_f32_negative_values() {
        let m = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let t = transpose_f32(&m, 2, 3);
        assert_eq!(t, vec![-1.0, -4.0, -2.0, -5.0, -3.0, -6.0]);
    }

    #[test]
    fn test_transpose_f32_nan_and_inf() {
        let m = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let t = transpose_f32(&m, 2, 2);
        assert!(t[0].is_nan());
        assert_eq!(t[1], f32::NEG_INFINITY);
        assert_eq!(t[2], f32::INFINITY);
        assert_eq!(t[3], 0.0);
    }

    #[test]
    fn test_transpose_f32_subnormals() {
        let tiny = f32::MIN_POSITIVE / 2.0;
        let m = vec![tiny, 0.0, 1.0, tiny];
        let t = transpose_f32(&m, 2, 2);
        assert_eq!(t[0], tiny);
        assert_eq!(t[1], 1.0);
        assert_eq!(t[2], 0.0);
        assert_eq!(t[3], tiny);
    }

    // ── transpose_i8 ──────────────────────────────────────────

    #[test]
    fn test_transpose_i8_1x1() {
        assert_eq!(transpose_i8(&[7], 1, 1), vec![7i8]);
    }

    #[test]
    fn test_transpose_i8_2x3() {
        let m = make_i8_matrix(2, 3);
        let t = transpose_i8(&m, 2, 3);
        assert_eq!(t, reference_transpose_i8(&m, 2, 3));
    }

    #[test]
    fn test_transpose_i8_4x4() {
        let m = make_i8_matrix(4, 4);
        let t = transpose_i8(&m, 4, 4);
        assert_eq!(t, reference_transpose_i8(&m, 4, 4));
    }

    #[test]
    fn test_transpose_i8_8x8() {
        let m = make_i8_matrix(8, 8);
        let t = transpose_i8(&m, 8, 8);
        assert_eq!(t, reference_transpose_i8(&m, 8, 8));
    }

    #[test]
    fn test_transpose_i8_9x11() {
        let m = make_i8_matrix(9, 11);
        let t = transpose_i8(&m, 9, 11);
        assert_eq!(t, reference_transpose_i8(&m, 9, 11));
    }

    #[test]
    fn test_transpose_i8_16x16() {
        let m = make_i8_matrix(16, 16);
        let t = transpose_i8(&m, 16, 16);
        assert_eq!(t, reference_transpose_i8(&m, 16, 16));
    }

    #[test]
    fn test_transpose_i8_empty() {
        assert_eq!(transpose_i8(&[], 0, 0), Vec::<i8>::new());
    }

    #[test]
    fn test_transpose_i8_double_identity() {
        let m = make_i8_matrix(5, 7);
        let t = transpose_i8(&m, 5, 7);
        let tt = transpose_i8(&t, 7, 5);
        assert_eq!(tt, m);
    }

    #[test]
    fn test_transpose_i8_negative_values() {
        let m = vec![-128i8, 0, 127, -1, 1, -127];
        let t = transpose_i8(&m, 2, 3);
        assert_eq!(t, vec![-128, -1, 0, 1, 127, -127]);
    }

    #[test]
    fn test_transpose_i8_large_32x32() {
        let m = make_i8_matrix(32, 32);
        let t = transpose_i8(&m, 32, 32);
        assert_eq!(t, reference_transpose_i8(&m, 32, 32));
    }

    // ── transpose_inplace ─────────────────────────────────────

    #[test]
    fn test_inplace_1x1() {
        let mut m = vec![99.0f32];
        transpose_inplace(&mut m, 1);
        assert_eq!(m, vec![99.0]);
    }

    #[test]
    fn test_inplace_2x2() {
        let mut m = vec![1.0, 2.0, 3.0, 4.0];
        transpose_inplace(&mut m, 2);
        assert_eq!(m, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_inplace_3x3() {
        let orig = make_f32_matrix(3, 3);
        let mut m = orig.clone();
        transpose_inplace(&mut m, 3);
        assert_eq!(m, reference_transpose_f32(&orig, 3, 3));
    }

    #[test]
    fn test_inplace_4x4() {
        let orig = make_f32_matrix(4, 4);
        let mut m = orig.clone();
        transpose_inplace(&mut m, 4);
        assert_eq!(m, reference_transpose_f32(&orig, 4, 4));
    }

    #[test]
    fn test_inplace_8x8() {
        let orig = make_f32_matrix(8, 8);
        let mut m = orig.clone();
        transpose_inplace(&mut m, 8);
        assert_eq!(m, reference_transpose_f32(&orig, 8, 8));
    }

    #[test]
    fn test_inplace_16x16() {
        let orig = make_f32_matrix(16, 16);
        let mut m = orig.clone();
        transpose_inplace(&mut m, 16);
        assert_eq!(m, reference_transpose_f32(&orig, 16, 16));
    }

    #[test]
    fn test_inplace_double_identity() {
        let orig = make_f32_matrix(7, 7);
        let mut m = orig.clone();
        transpose_inplace(&mut m, 7);
        transpose_inplace(&mut m, 7);
        assert_eq!(m, orig);
    }

    #[test]
    fn test_inplace_diagonal_unchanged() {
        let mut m = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let diag = vec![m[0], m[4], m[8]];
        transpose_inplace(&mut m, 3);
        assert_eq!(vec![m[0], m[4], m[8]], diag);
    }

    // ── blocked_transpose ─────────────────────────────────────

    #[test]
    fn test_blocked_2x3_block1() {
        let m = make_f32_matrix(2, 3);
        assert_eq!(blocked_transpose(&m, 2, 3, 1), reference_transpose_f32(&m, 2, 3));
    }

    #[test]
    fn test_blocked_8x8_block4() {
        let m = make_f32_matrix(8, 8);
        assert_eq!(blocked_transpose(&m, 8, 8, 4), reference_transpose_f32(&m, 8, 8));
    }

    #[test]
    fn test_blocked_8x8_block8() {
        let m = make_f32_matrix(8, 8);
        assert_eq!(blocked_transpose(&m, 8, 8, 8), reference_transpose_f32(&m, 8, 8));
    }

    #[test]
    fn test_blocked_16x16_block4() {
        let m = make_f32_matrix(16, 16);
        assert_eq!(blocked_transpose(&m, 16, 16, 4), reference_transpose_f32(&m, 16, 16));
    }

    #[test]
    fn test_blocked_15x17_block4() {
        let m = make_f32_matrix(15, 17);
        assert_eq!(blocked_transpose(&m, 15, 17, 4), reference_transpose_f32(&m, 15, 17));
    }

    #[test]
    fn test_blocked_32x32_block16() {
        let m = make_f32_matrix(32, 32);
        assert_eq!(blocked_transpose(&m, 32, 32, 16), reference_transpose_f32(&m, 32, 32));
    }

    #[test]
    fn test_blocked_100x50_block8() {
        let m = make_f32_matrix(100, 50);
        assert_eq!(blocked_transpose(&m, 100, 50, 8), reference_transpose_f32(&m, 100, 50));
    }

    #[test]
    fn test_blocked_empty() {
        assert_eq!(blocked_transpose(&[], 0, 0, 4), Vec::<f32>::new());
    }

    #[test]
    fn test_blocked_block_larger_than_matrix() {
        let m = make_f32_matrix(3, 3);
        assert_eq!(blocked_transpose(&m, 3, 3, 64), reference_transpose_f32(&m, 3, 3));
    }

    #[test]
    fn test_blocked_double_identity() {
        let m = make_f32_matrix(10, 6);
        let t = blocked_transpose(&m, 10, 6, 4);
        let tt = blocked_transpose(&t, 6, 10, 4);
        assert_eq!(tt, m);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn test_blocked_zero_block_panics() {
        blocked_transpose(&[1.0], 1, 1, 0);
    }

    // ── batch_transpose ───────────────────────────────────────

    #[test]
    fn test_batch_single_matrix() {
        let m = make_f32_matrix(3, 4);
        let t = batch_transpose(&m, 3, 4, 1);
        assert_eq!(t, reference_transpose_f32(&m, 3, 4));
    }

    #[test]
    fn test_batch_two_matrices() {
        let rows = 3;
        let cols = 4;
        let mut input = make_f32_matrix(rows, cols);
        let m2: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 10.0).collect();
        input.extend_from_slice(&m2);

        let t = batch_transpose(&input, rows, cols, 2);
        let ref1 = reference_transpose_f32(&input[..rows * cols], rows, cols);
        let ref2 = reference_transpose_f32(&m2, rows, cols);
        let mut expected = ref1;
        expected.extend_from_slice(&ref2);
        assert_eq!(t, expected);
    }

    #[test]
    fn test_batch_four_8x8() {
        let rows = 8;
        let cols = 8;
        let total = 4 * rows * cols;
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let t = batch_transpose(&input, rows, cols, 4);
        for k in 0..4 {
            let off = k * rows * cols;
            let chunk = &input[off..off + rows * cols];
            let expected = reference_transpose_f32(chunk, rows, cols);
            assert_eq!(&t[off..off + rows * cols], expected.as_slice());
        }
    }

    #[test]
    fn test_batch_zero_count() {
        assert_eq!(batch_transpose(&[], 2, 3, 0), Vec::<f32>::new());
    }

    #[test]
    fn test_batch_empty_matrix() {
        assert_eq!(batch_transpose(&[], 0, 0, 5), Vec::<f32>::new());
    }

    #[test]
    fn test_batch_1x1_many() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t = batch_transpose(&input, 1, 1, 5);
        assert_eq!(t, input);
    }

    #[test]
    fn test_batch_large_16x16_x3() {
        let rows = 16;
        let cols = 16;
        let count = 3;
        let total = count * rows * cols;
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let t = batch_transpose(&input, rows, cols, count);
        for k in 0..count {
            let off = k * rows * cols;
            let expected = reference_transpose_f32(&input[off..off + rows * cols], rows, cols);
            assert_eq!(&t[off..off + rows * cols], expected.as_slice());
        }
    }

    // ── strided_transpose ─────────────────────────────────────

    #[test]
    fn test_strided_contiguous() {
        let m = make_f32_matrix(3, 4);
        let t = strided_transpose(&m, 3, 4, 4, 1);
        assert_eq!(t, reference_transpose_f32(&m, 3, 4));
    }

    #[test]
    fn test_strided_column_major_read() {
        // Read a 2×3 sub-matrix from a 3×2 column-major layout.
        let col_major = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // cols of [[1,2,3],[4,5,6]]
        let t = strided_transpose(&col_major, 2, 3, 1, 2);
        // Transpose of [[1,2,3],[4,5,6]] = [[1,4],[2,5],[3,6]]
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_strided_skip_every_other_col() {
        // 2×2 from a wider buffer, col_stride=2 skips alternating elements.
        let buf = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let t = strided_transpose(&buf, 2, 2, 4, 2);
        // Matrix is [[1,2],[3,4]], transpose = [[1,3],[2,4]]
        assert_eq!(t, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_strided_large_row_stride() {
        let mut buf = vec![0.0f32; 100];
        // Place a 2×3 matrix at rows 0 and 10.
        buf[0] = 1.0;
        buf[1] = 2.0;
        buf[2] = 3.0;
        buf[50] = 4.0;
        buf[51] = 5.0;
        buf[52] = 6.0;
        let t = strided_transpose(&buf, 2, 3, 50, 1);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_strided_1x1() {
        assert_eq!(strided_transpose(&[42.0], 1, 1, 1, 1), vec![42.0]);
    }

    #[test]
    fn test_strided_empty() {
        assert_eq!(strided_transpose(&[], 0, 0, 1, 1), Vec::<f32>::new());
    }

    #[test]
    fn test_strided_single_row() {
        let buf = vec![10.0, 20.0, 30.0, 40.0];
        let t = strided_transpose(&buf, 1, 4, 4, 1);
        // Transpose of 1×4 row is a 4×1 column.
        assert_eq!(t, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_strided_single_col() {
        let buf = vec![10.0, 20.0, 30.0, 40.0];
        let t = strided_transpose(&buf, 4, 1, 1, 1);
        assert_eq!(t, vec![10.0, 20.0, 30.0, 40.0]);
    }

    // ── scalar fallback path ──────────────────────────────────

    #[test]
    fn test_scalar_fallback_matches_reference() {
        let m = make_f32_matrix(11, 13);
        let mut out = vec![0.0f32; 11 * 13];
        transpose_f32_scalar(&m, 11, 13, &mut out);
        assert_eq!(out, reference_transpose_f32(&m, 11, 13));
    }

    #[test]
    fn test_scalar_fallback_8x8() {
        let m = make_f32_matrix(8, 8);
        let mut out = vec![0.0f32; 64];
        transpose_f32_scalar(&m, 8, 8, &mut out);
        assert_eq!(out, reference_transpose_f32(&m, 8, 8));
    }

    // ── consistency across APIs ───────────────────────────────

    #[test]
    fn test_consistency_transpose_vs_blocked() {
        let m = make_f32_matrix(20, 30);
        let t1 = transpose_f32(&m, 20, 30);
        let t2 = blocked_transpose(&m, 20, 30, 8);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_consistency_transpose_vs_batch() {
        let m = make_f32_matrix(10, 14);
        let t1 = transpose_f32(&m, 10, 14);
        let t2 = batch_transpose(&m, 10, 14, 1);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_consistency_transpose_vs_strided() {
        let m = make_f32_matrix(6, 9);
        let t1 = transpose_f32(&m, 6, 9);
        let t2 = strided_transpose(&m, 6, 9, 9, 1);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_consistency_inplace_vs_transpose() {
        let m = make_f32_matrix(12, 12);
        let t = transpose_f32(&m, 12, 12);
        let mut m2 = m;
        transpose_inplace(&mut m2, 12);
        assert_eq!(m2, t);
    }

    // ── panics ────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "input length")]
    fn test_transpose_f32_short_input_panics() {
        transpose_f32(&[1.0, 2.0], 2, 3);
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn test_transpose_i8_short_input_panics() {
        transpose_i8(&[1, 2], 2, 3);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_inplace_short_data_panics() {
        transpose_inplace(&mut [1.0, 2.0], 3);
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn test_blocked_short_input_panics() {
        blocked_transpose(&[1.0], 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn test_batch_short_input_panics() {
        batch_transpose(&[1.0, 2.0, 3.0], 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn test_strided_out_of_bounds_panics() {
        strided_transpose(&[1.0, 2.0], 2, 2, 2, 1);
    }

    // ── additional edge cases ─────────────────────────────────

    #[test]
    fn test_transpose_f32_all_zeros() {
        let m = vec![0.0f32; 64];
        let t = transpose_f32(&m, 8, 8);
        assert!(t.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_transpose_f32_all_ones() {
        let m = vec![1.0f32; 64];
        let t = transpose_f32(&m, 8, 8);
        assert!(t.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_transpose_i8_all_zeros() {
        let m = vec![0i8; 64];
        let t = transpose_i8(&m, 8, 8);
        assert!(t.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_blocked_various_block_sizes() {
        let m = make_f32_matrix(20, 20);
        let expected = reference_transpose_f32(&m, 20, 20);
        for bs in [1, 2, 3, 4, 5, 7, 8, 10, 16, 20, 32] {
            assert_eq!(blocked_transpose(&m, 20, 20, bs), expected, "block_size={bs}");
        }
    }

    #[test]
    fn test_batch_transpose_preserves_order() {
        let rows = 2;
        let cols = 3;
        let count = 4;
        let input: Vec<f32> = (0..(count * rows * cols) as u32).map(|i| i as f32).collect();
        let t = batch_transpose(&input, rows, cols, count);
        for k in 0..count {
            let off = k * rows * cols;
            let chunk = &input[off..off + rows * cols];
            let expected = reference_transpose_f32(chunk, rows, cols);
            assert_eq!(&t[off..off + rows * cols], expected.as_slice(), "batch {k}");
        }
    }

    #[test]
    fn test_strided_both_strides_large() {
        let mut buf = vec![0.0f32; 200];
        // 3×2 matrix with row_stride=20, col_stride=5.
        buf[0] = 1.0;
        buf[5] = 2.0;
        buf[20] = 3.0;
        buf[25] = 4.0;
        buf[40] = 5.0;
        buf[45] = 6.0;
        let t = strided_transpose(&buf, 3, 2, 20, 5);
        // Matrix = [[1,2],[3,4],[5,6]], T = [[1,3,5],[2,4,6]]
        assert_eq!(t, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_transpose_f32_non_square_rect() {
        for (r, c) in [(2, 8), (8, 2), (3, 16), (16, 3), (7, 8), (8, 7)] {
            let m = make_f32_matrix(r, c);
            let t = transpose_f32(&m, r, c);
            assert_eq!(t, reference_transpose_f32(&m, r, c), "{r}×{c}");
        }
    }
}
