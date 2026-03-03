#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::let_and_return)]
//! NEON-optimized GEMV (General Matrix-Vector Multiply) kernels for ARM/Apple Silicon.
//!
//! GEMV is the core operation in every linear layer during inference:
//! given a weight matrix W and an input vector x, compute y = W·x.
//! These kernels use ARM NEON SIMD intrinsics (`vfmaq_f32`, `vaddvq_f32`)
//! to process four f32 lanes in parallel, with scalar tails for
//! dimensions that are not multiples of four.

use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Horizontal sum of a `float32x4_t` → scalar f32.
#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    unsafe { vaddvq_f32(v) }
}

// ---------------------------------------------------------------------------
// Basic GEMV: output[i] = Σ_j matrix[i*n + j] * vector[j]
// ---------------------------------------------------------------------------

/// Compute `output = matrix · vector` where `matrix` is `m × n` stored
/// row-major and `vector` has length `n`.
///
/// # Panics
///
/// Panics when slice lengths do not match the declared dimensions.
#[target_feature(enable = "neon")]
pub unsafe fn neon_gemv_f32(
    matrix: &[f32],
    vector: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
) {
    assert!(matrix.len() >= m * n, "matrix too short");
    assert!(vector.len() >= n, "vector too short");
    assert!(output.len() >= m, "output too short");

    let chunks = n / 4;
    let tail = n % 4;

    for i in 0..m {
        let row = i * n;
        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let j = c * 4;
            let a = vld1q_f32(matrix.as_ptr().add(row + j));
            let b = vld1q_f32(vector.as_ptr().add(j));
            acc = vfmaq_f32(acc, a, b);
        }

        let mut sum = unsafe { hsum_f32x4(acc) };

        for t in 0..tail {
            let j = chunks * 4 + t;
            sum += matrix[row + j] * vector[j];
        }

        output[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Transposed GEMV: output[j] = Σ_i matrix[i*n + j] * vector[i]
// ---------------------------------------------------------------------------

/// Compute `output = matrixᵀ · vector` where `matrix` is `m × n` row-major
/// and `vector` has length `m`.  The result has length `n`.
///
/// # Panics
///
/// Panics when slice lengths do not match the declared dimensions.
#[target_feature(enable = "neon")]
pub unsafe fn neon_gemv_transposed_f32(
    matrix: &[f32],
    vector: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
) {
    assert!(matrix.len() >= m * n, "matrix too short");
    assert!(vector.len() >= m, "vector too short");
    assert!(output.len() >= n, "output too short");

    // Zero the output.
    for v in output[..n].iter_mut() {
        *v = 0.0;
    }

    let chunks = n / 4;
    let tail = n % 4;

    for i in 0..m {
        let row = i * n;
        let vi = vdupq_n_f32(vector[i]);

        for c in 0..chunks {
            let j = c * 4;
            let a = vld1q_f32(matrix.as_ptr().add(row + j));
            let cur = vld1q_f32(output.as_ptr().add(j));
            let res = vfmaq_f32(cur, a, vi);
            vst1q_f32(output.as_mut_ptr().add(j), res);
        }

        for t in 0..tail {
            let j = chunks * 4 + t;
            output[j] += matrix[row + j] * vector[i];
        }
    }
}

// ---------------------------------------------------------------------------
// GEMV + bias: output[i] = bias[i] + Σ_j matrix[i*n + j] * vector[j]
// ---------------------------------------------------------------------------

/// Fused GEMV with bias addition for linear layers.
///
/// # Panics
///
/// Panics when slice lengths do not match the declared dimensions.
#[target_feature(enable = "neon")]
pub unsafe fn neon_gemv_bias_f32(
    matrix: &[f32],
    vector: &[f32],
    bias: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
) {
    assert!(matrix.len() >= m * n, "matrix too short");
    assert!(vector.len() >= n, "vector too short");
    assert!(bias.len() >= m, "bias too short");
    assert!(output.len() >= m, "output too short");

    let chunks = n / 4;
    let tail = n % 4;

    for i in 0..m {
        let row = i * n;
        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let j = c * 4;
            let a = vld1q_f32(matrix.as_ptr().add(row + j));
            let b = vld1q_f32(vector.as_ptr().add(j));
            acc = vfmaq_f32(acc, a, b);
        }

        let mut sum = unsafe { hsum_f32x4(acc) };

        for t in 0..tail {
            let j = chunks * 4 + t;
            sum += matrix[row + j] * vector[j];
        }

        output[i] = sum + bias[i];
    }
}

// ---------------------------------------------------------------------------
// GEMV + ReLU: output[i] = max(0, Σ_j matrix[i*n + j] * vector[j])
// ---------------------------------------------------------------------------

/// Fused GEMV with ReLU activation for activated layers.
///
/// # Panics
///
/// Panics when slice lengths do not match the declared dimensions.
#[target_feature(enable = "neon")]
pub unsafe fn neon_gemv_relu_f32(
    matrix: &[f32],
    vector: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
) {
    assert!(matrix.len() >= m * n, "matrix too short");
    assert!(vector.len() >= n, "vector too short");
    assert!(output.len() >= m, "output too short");

    let chunks = n / 4;
    let tail = n % 4;

    for i in 0..m {
        let row = i * n;
        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let j = c * 4;
            let a = vld1q_f32(matrix.as_ptr().add(row + j));
            let b = vld1q_f32(vector.as_ptr().add(j));
            acc = vfmaq_f32(acc, a, b);
        }

        let mut sum = unsafe { hsum_f32x4(acc) };

        for t in 0..tail {
            let j = chunks * 4 + t;
            sum += matrix[row + j] * vector[j];
        }

        output[i] = if sum > 0.0 { sum } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// Quantised i8 × f32 GEMV
// ---------------------------------------------------------------------------

/// GEMV with an `i8` weight matrix and an `f32` input vector.
///
/// Each `i8` weight is widened to `f32`, multiplied by `scale`, and then
/// accumulated with the corresponding element of `vector`.  This is the
/// typical pattern for post-training quantised inference.
///
/// # Panics
///
/// Panics when slice lengths do not match the declared dimensions.
#[target_feature(enable = "neon")]
pub unsafe fn neon_gemv_i8_f32(
    matrix: &[i8],
    vector: &[f32],
    scale: f32,
    output: &mut [f32],
    m: usize,
    n: usize,
) {
    assert!(matrix.len() >= m * n, "matrix too short");
    assert!(vector.len() >= n, "vector too short");
    assert!(output.len() >= m, "output too short");

    let chunks = n / 4;
    let tail = n % 4;
    let vscale = vdupq_n_f32(scale);

    for i in 0..m {
        let row = i * n;
        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let j = c * 4;
            // Load 4 × i8 → widen to i16 → i32 → f32.
            let raw = vld1_lane_s32::<0>(matrix.as_ptr().add(row + j) as *const i32, vdup_n_s32(0));
            let i8x8 = vreinterpret_s8_s32(raw);
            let i16x8 = vmovl_s8(i8x8);
            let i32x4 = vmovl_s16(vget_low_s16(i16x8));
            let f32x4 = vcvtq_f32_s32(i32x4);
            let w = vmulq_f32(f32x4, vscale);

            let v = vld1q_f32(vector.as_ptr().add(j));
            acc = vfmaq_f32(acc, w, v);
        }

        let mut sum = unsafe { hsum_f32x4(acc) };

        for t in 0..tail {
            let j = chunks * 4 + t;
            sum += (matrix[row + j] as f32) * scale * vector[j];
        }

        output[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Reference implementations ------------------------------------------

    fn reference_gemv(matrix: &[f32], vector: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                output[i] += matrix[i * n + j] * vector[j];
            }
        }
        output
    }

    fn reference_gemv_transposed(matrix: &[f32], vector: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0; n];
        for i in 0..m {
            for j in 0..n {
                output[j] += matrix[i * n + j] * vector[i];
            }
        }
        output
    }

    fn reference_gemv_bias(
        matrix: &[f32],
        vector: &[f32],
        bias: &[f32],
        m: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = reference_gemv(matrix, vector, m, n);
        for i in 0..m {
            output[i] += bias[i];
        }
        output
    }

    fn reference_gemv_relu(matrix: &[f32], vector: &[f32], m: usize, n: usize) -> Vec<f32> {
        reference_gemv(matrix, vector, m, n).into_iter().map(|v| v.max(0.0)).collect()
    }

    fn reference_gemv_i8(
        matrix: &[i8],
        vector: &[f32],
        scale: f32,
        m: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                output[i] += (matrix[i * n + j] as f32) * scale * vector[j];
            }
        }
        output
    }

    fn assert_vec_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // gemv_basic
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_basic_identity_matrix() {
        let n = 4;
        let mut mat = vec![0.0f32; n * n];
        for i in 0..n {
            mat[i * n + i] = 1.0;
        }
        let vec_in = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_f32(&mat, &vec_in, &mut out, n, n) };
        assert_vec_close(&out, &vec_in, 1e-6);
    }

    #[test]
    fn gemv_basic_zeros() {
        let (m, n) = (3, 4);
        let mat = vec![0.0f32; m * n];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        assert_vec_close(&out, &vec![0.0; m], 1e-6);
    }

    #[test]
    fn gemv_basic_ones() {
        let (m, n) = (2, 4);
        let mat = vec![1.0f32; m * n];
        let v = vec![1.0; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        assert_vec_close(&out, &vec![4.0; m], 1e-6);
    }

    #[test]
    fn gemv_basic_single_row() {
        let (m, n) = (1, 8);
        let mat: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let v = vec![1.0; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn gemv_basic_single_col() {
        let (m, n) = (4, 1);
        let mat = vec![2.0, 3.0, 4.0, 5.0];
        let v = vec![3.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        assert_vec_close(&out, &[6.0, 9.0, 12.0, 15.0], 1e-6);
    }

    #[test]
    fn gemv_basic_4x4() {
        let (m, n) = (4, 4);
        let mat: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn gemv_basic_8x8() {
        let (m, n) = (8, 8);
        let mat: Vec<f32> = (0..64).map(|x| (x as f32) * 0.1).collect();
        let v: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    #[test]
    fn gemv_basic_non_square() {
        let (m, n) = (3, 5);
        let mat: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let v: Vec<f32> = (0..5).map(|x| (x as f32) + 1.0).collect();
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    #[test]
    fn gemv_basic_large() {
        let (m, n) = (64, 128);
        let mat: Vec<f32> = (0..m * n).map(|x| ((x % 17) as f32) * 0.01).collect();
        let v: Vec<f32> = (0..n).map(|x| ((x % 7) as f32) * 0.1).collect();
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-3);
    }

    #[test]
    fn gemv_basic_negative() {
        let (m, n) = (2, 4);
        let mat = vec![-1.0f32, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0];
        let v = vec![1.0, -1.0, 1.0, -1.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-6);
    }

    // -----------------------------------------------------------------------
    // gemv_transposed
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_transposed_identity() {
        let n = 4;
        let mut mat = vec![0.0f32; n * n];
        for i in 0..n {
            mat[i * n + i] = 1.0;
        }
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_transposed_f32(&mat, &v, &mut out, n, n) };
        assert_vec_close(&out, &v, 1e-6);
    }

    #[test]
    fn gemv_transposed_zeros() {
        let (m, n) = (3, 4);
        let mat = vec![0.0f32; m * n];
        let v = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_transposed_f32(&mat, &v, &mut out, m, n) };
        assert_vec_close(&out, &vec![0.0; n], 1e-6);
    }

    #[test]
    fn gemv_transposed_4x4() {
        let (m, n) = (4, 4);
        let mat: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_transposed_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_transposed(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    #[test]
    fn gemv_transposed_8x8() {
        let (m, n) = (8, 8);
        let mat: Vec<f32> = (0..64).map(|x| (x as f32) * 0.1).collect();
        let v: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_transposed_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_transposed(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    #[test]
    fn gemv_transposed_non_square() {
        let (m, n) = (3, 5);
        let mat: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let v = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_transposed_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_transposed(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    #[test]
    fn gemv_transposed_rectangular() {
        let (m, n) = (2, 8);
        let mat: Vec<f32> = (0..16).map(|x| (x as f32) * 0.5).collect();
        let v = vec![1.0, -1.0];
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_transposed_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_transposed(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn gemv_transposed_large() {
        let (m, n) = (64, 128);
        let mat: Vec<f32> = (0..m * n).map(|x| ((x % 13) as f32) * 0.02).collect();
        let v: Vec<f32> = (0..m).map(|x| ((x % 5) as f32) * 0.1).collect();
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_transposed_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_transposed(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-3);
    }

    // -----------------------------------------------------------------------
    // gemv_bias
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_bias_zero_bias() {
        let (m, n) = (3, 4);
        let mat: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let v = vec![1.0; n];
        let bias = vec![0.0; m];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_bias_f32(&mat, &v, &bias, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn gemv_bias_unit_bias() {
        let (m, n) = (3, 4);
        let mat: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let v = vec![1.0; n];
        let bias = vec![1.0; m];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_bias_f32(&mat, &v, &bias, &mut out, m, n) };
        let expected = reference_gemv_bias(&mat, &v, &bias, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn gemv_bias_negative_bias() {
        let (m, n) = (2, 4);
        let mat = vec![1.0f32; m * n];
        let v = vec![1.0; n];
        let bias = vec![-10.0, -20.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_bias_f32(&mat, &v, &bias, &mut out, m, n) };
        let expected = reference_gemv_bias(&mat, &v, &bias, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn gemv_bias_large() {
        let (m, n) = (32, 64);
        let mat: Vec<f32> = (0..m * n).map(|x| ((x % 11) as f32) * 0.1).collect();
        let v: Vec<f32> = (0..n).map(|x| ((x % 7) as f32) * 0.2).collect();
        let bias: Vec<f32> = (0..m).map(|x| (x as f32) * 0.5).collect();
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_bias_f32(&mat, &v, &bias, &mut out, m, n) };
        let expected = reference_gemv_bias(&mat, &v, &bias, m, n);
        assert_vec_close(&out, &expected, 1e-3);
    }

    #[test]
    fn gemv_bias_with_zeros() {
        let (m, n) = (2, 4);
        let mat = vec![0.0f32; m * n];
        let v = vec![1.0; n];
        let bias = vec![5.0, 10.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_bias_f32(&mat, &v, &bias, &mut out, m, n) };
        assert_vec_close(&out, &bias, 1e-6);
    }

    #[test]
    fn gemv_bias_mixed() {
        let (m, n) = (3, 5);
        let mat: Vec<f32> = (0..15).map(|x| (x as f32) - 7.0).collect();
        let v: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0, 0.25];
        let bias = vec![100.0, -100.0, 0.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_bias_f32(&mat, &v, &bias, &mut out, m, n) };
        let expected = reference_gemv_bias(&mat, &v, &bias, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    // -----------------------------------------------------------------------
    // gemv_relu
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_relu_all_positive() {
        let (m, n) = (2, 4);
        let mat = vec![1.0f32; m * n];
        let v = vec![1.0; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_relu_f32(&mat, &v, &mut out, m, n) };
        assert_vec_close(&out, &vec![4.0; m], 1e-6);
    }

    #[test]
    fn gemv_relu_all_negative() {
        let (m, n) = (2, 4);
        let mat = vec![-1.0f32; m * n];
        let v = vec![1.0; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_relu_f32(&mat, &v, &mut out, m, n) };
        assert_vec_close(&out, &vec![0.0; m], 1e-6);
    }

    #[test]
    fn gemv_relu_mixed() {
        let (m, n) = (4, 4);
        let mat: Vec<f32> = (0..16).map(|x| (x as f32) - 8.0).collect();
        let v = vec![1.0; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_relu_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_relu(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn gemv_relu_zeros() {
        let (m, n) = (3, 4);
        let mat = vec![0.0f32; m * n];
        let v = vec![1.0; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_relu_f32(&mat, &v, &mut out, m, n) };
        assert_vec_close(&out, &vec![0.0; m], 1e-6);
    }

    #[test]
    fn gemv_relu_with_negative_result() {
        let (m, n) = (2, 4);
        let mat = vec![1.0, -3.0, 1.0, -3.0, 1.0, 1.0, 1.0, 1.0];
        let v = vec![1.0; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_relu_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_relu(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-6);
    }

    #[test]
    fn gemv_relu_large() {
        let (m, n) = (32, 64);
        let mat: Vec<f32> = (0..m * n).map(|x| ((x % 19) as f32) - 9.0).collect();
        let v: Vec<f32> = (0..n).map(|x| ((x % 3) as f32) - 1.0).collect();
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_relu_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_relu(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-3);
    }

    // -----------------------------------------------------------------------
    // gemv_i8
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_i8_identity() {
        let n = 4;
        let mut mat = vec![0i8; n * n];
        for i in 0..n {
            mat[i * n + i] = 1;
        }
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_i8_f32(&mat, &v, 1.0, &mut out, n, n) };
        assert_vec_close(&out, &v, 1e-5);
    }

    #[test]
    fn gemv_i8_zeros() {
        let (m, n) = (3, 4);
        let mat = vec![0i8; m * n];
        let v = vec![1.0f32; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_i8_f32(&mat, &v, 1.0, &mut out, m, n) };
        assert_vec_close(&out, &vec![0.0; m], 1e-6);
    }

    #[test]
    fn gemv_i8_scale_factor() {
        let (m, n) = (2, 4);
        let mat = vec![1i8; m * n];
        let v = vec![1.0f32; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_i8_f32(&mat, &v, 0.5, &mut out, m, n) };
        assert_vec_close(&out, &vec![2.0; m], 1e-5);
    }

    #[test]
    fn gemv_i8_mixed_signs() {
        let (m, n) = (2, 4);
        let mat = vec![1i8, -1, 1, -1, -1, 1, -1, 1];
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_i8_f32(&mat, &v, 1.0, &mut out, m, n) };
        let expected = reference_gemv_i8(&mat, &v, 1.0, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn gemv_i8_large() {
        let (m, n) = (32, 64);
        let mat: Vec<i8> = (0..m * n).map(|x| ((x % 5) as i8) - 2).collect();
        let v: Vec<f32> = (0..n).map(|x| ((x % 7) as f32) * 0.1).collect();
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_i8_f32(&mat, &v, 0.25, &mut out, m, n) };
        let expected = reference_gemv_i8(&mat, &v, 0.25, m, n);
        assert_vec_close(&out, &expected, 1e-3);
    }

    #[test]
    fn gemv_i8_precision() {
        let (m, n) = (1, 8);
        let mat = vec![127i8, -128, 64, -64, 32, -32, 16, -16];
        let v = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_i8_f32(&mat, &v, 1.0, &mut out, m, n) };
        let expected = reference_gemv_i8(&mat, &v, 1.0, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    // -----------------------------------------------------------------------
    // dimension_edge_cases
    // -----------------------------------------------------------------------

    #[test]
    fn edge_1x1() {
        let mat = vec![3.0f32];
        let v = vec![5.0f32];
        let mut out = vec![0.0; 1];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, 1, 1) };
        assert_vec_close(&out, &[15.0], 1e-6);
    }

    #[test]
    fn edge_1xn() {
        let n = 7;
        let mat: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        let v = vec![1.0; n];
        let mut out = vec![0.0; 1];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, 1, n) };
        let expected = reference_gemv(&mat, &v, 1, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn edge_mx1() {
        let m = 7;
        let mat: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        let v = vec![2.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, 1) };
        let expected = reference_gemv(&mat, &v, m, 1);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn edge_3x5() {
        let (m, n) = (3, 5);
        let mat: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let v: Vec<f32> = (0..5).map(|x| (x as f32) + 1.0).collect();
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    #[test]
    fn edge_5x3() {
        let (m, n) = (5, 3);
        let mat: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let v = vec![1.0, 0.5, 0.25];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn edge_7x7() {
        let n = 7;
        let mat: Vec<f32> = (0..49).map(|x| (x as f32) * 0.1).collect();
        let v: Vec<f32> = (0..7).map(|x| (x as f32) + 1.0).collect();
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, n, n) };
        let expected = reference_gemv(&mat, &v, n, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    #[test]
    fn edge_15x17() {
        let (m, n) = (15, 17);
        let mat: Vec<f32> = (0..m * n).map(|x| ((x % 11) as f32) * 0.05).collect();
        let v: Vec<f32> = (0..n).map(|x| ((x % 3) as f32) * 0.5).collect();
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-3);
    }

    // -----------------------------------------------------------------------
    // numerical
    // -----------------------------------------------------------------------

    #[test]
    fn numerical_large_values() {
        let (m, n) = (2, 4);
        let mat = vec![1e6f32; m * n];
        let v = vec![1e6f32; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e6);
    }

    #[test]
    fn numerical_small_values() {
        let (m, n) = (2, 4);
        let mat = vec![1e-6f32; m * n];
        let v = vec![1e-6f32; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-15);
    }

    #[test]
    fn numerical_mixed_magnitude() {
        let (m, n) = (2, 4);
        let mat = vec![1e4f32, 1e-4, 1e4, 1e-4, 1e-4, 1e4, 1e-4, 1e4];
        let v = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-1);
    }

    #[test]
    fn numerical_precision_check() {
        let (m, n) = (4, 8);
        let mat: Vec<f32> = (0..32).map(|x| (x as f32) * 0.01 + 0.001).collect();
        let v: Vec<f32> = (0..8).map(|x| (x as f32) * 0.1 + 0.05).collect();
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Additional coverage: transposed edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_transposed_1x1() {
        let mat = vec![3.0f32];
        let v = vec![5.0f32];
        let mut out = vec![0.0; 1];
        unsafe { neon_gemv_transposed_f32(&mat, &v, &mut out, 1, 1) };
        assert_vec_close(&out, &[15.0], 1e-6);
    }

    #[test]
    fn gemv_transposed_single_row() {
        let (m, n) = (1, 8);
        let mat: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let v = vec![2.0f32];
        let mut out = vec![0.0; n];
        unsafe { neon_gemv_transposed_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_transposed(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Additional coverage: bias edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_bias_1x1() {
        let mat = vec![2.0f32];
        let v = vec![3.0f32];
        let bias = vec![10.0f32];
        let mut out = vec![0.0; 1];
        unsafe { neon_gemv_bias_f32(&mat, &v, &bias, &mut out, 1, 1) };
        assert_vec_close(&out, &[16.0], 1e-6);
    }

    #[test]
    fn gemv_bias_non_multiple_of_4() {
        let (m, n) = (3, 7);
        let mat: Vec<f32> = (0..21).map(|x| (x as f32) * 0.1).collect();
        let v: Vec<f32> = (0..7).map(|x| (x as f32) + 1.0).collect();
        let bias = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_bias_f32(&mat, &v, &bias, &mut out, m, n) };
        let expected = reference_gemv_bias(&mat, &v, &bias, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    // -----------------------------------------------------------------------
    // Additional coverage: relu edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_relu_1x1_positive() {
        let mat = vec![2.0f32];
        let v = vec![3.0f32];
        let mut out = vec![0.0; 1];
        unsafe { neon_gemv_relu_f32(&mat, &v, &mut out, 1, 1) };
        assert_vec_close(&out, &[6.0], 1e-6);
    }

    #[test]
    fn gemv_relu_1x1_negative() {
        let mat = vec![-2.0f32];
        let v = vec![3.0f32];
        let mut out = vec![0.0; 1];
        unsafe { neon_gemv_relu_f32(&mat, &v, &mut out, 1, 1) };
        assert_vec_close(&out, &[0.0], 1e-6);
    }

    #[test]
    fn gemv_relu_non_multiple_of_4() {
        let (m, n) = (3, 5);
        let mat: Vec<f32> = (0..15).map(|x| (x as f32) - 7.0).collect();
        let v = vec![1.0; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_relu_f32(&mat, &v, &mut out, m, n) };
        let expected = reference_gemv_relu(&mat, &v, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    // -----------------------------------------------------------------------
    // Additional coverage: i8 edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_i8_1x1() {
        let mat = vec![5i8];
        let v = vec![2.0f32];
        let mut out = vec![0.0; 1];
        unsafe { neon_gemv_i8_f32(&mat, &v, 1.0, &mut out, 1, 1) };
        assert_vec_close(&out, &[10.0], 1e-5);
    }

    #[test]
    fn gemv_i8_non_multiple_of_4() {
        let (m, n) = (2, 5);
        let mat: Vec<i8> = vec![1, 2, 3, 4, 5, -1, -2, -3, -4, -5];
        let v = vec![1.0f32; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_i8_f32(&mat, &v, 1.0, &mut out, m, n) };
        let expected = reference_gemv_i8(&mat, &v, 1.0, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    #[test]
    fn gemv_i8_max_min_values() {
        let (m, n) = (2, 4);
        let mat = vec![127i8, -128, 127, -128, -128, 127, -128, 127];
        let v = vec![1.0f32; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_i8_f32(&mat, &v, 1.0, &mut out, m, n) };
        let expected = reference_gemv_i8(&mat, &v, 1.0, m, n);
        assert_vec_close(&out, &expected, 1e-4);
    }

    #[test]
    fn gemv_i8_negative_scale() {
        let (m, n) = (2, 4);
        let mat = vec![1i8; m * n];
        let v = vec![1.0f32; n];
        let mut out = vec![0.0; m];
        unsafe { neon_gemv_i8_f32(&mat, &v, -0.5, &mut out, m, n) };
        let expected = reference_gemv_i8(&mat, &v, -0.5, m, n);
        assert_vec_close(&out, &expected, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Cross-function consistency
    // -----------------------------------------------------------------------

    #[test]
    fn gemv_vs_bias_with_zero_bias() {
        let (m, n) = (4, 8);
        let mat: Vec<f32> = (0..32).map(|x| (x as f32) * 0.1).collect();
        let v: Vec<f32> = (0..8).map(|x| (x as f32) + 1.0).collect();
        let bias = vec![0.0; m];
        let mut out_basic = vec![0.0; m];
        let mut out_bias = vec![0.0; m];
        unsafe {
            neon_gemv_f32(&mat, &v, &mut out_basic, m, n);
            neon_gemv_bias_f32(&mat, &v, &bias, &mut out_bias, m, n);
        };
        assert_vec_close(&out_basic, &out_bias, 1e-6);
    }

    #[test]
    fn gemv_relu_vs_manual_relu() {
        let (m, n) = (4, 8);
        let mat: Vec<f32> = (0..32).map(|x| (x as f32) - 16.0).collect();
        let v = vec![1.0; n];
        let mut out_fused = vec![0.0; m];
        let mut out_basic = vec![0.0; m];
        unsafe {
            neon_gemv_relu_f32(&mat, &v, &mut out_fused, m, n);
            neon_gemv_f32(&mat, &v, &mut out_basic, m, n);
        };
        let out_manual: Vec<f32> = out_basic.iter().map(|v| v.max(0.0)).collect();
        assert_vec_close(&out_fused, &out_manual, 1e-5);
    }

    #[test]
    fn gemv_transposed_consistency() {
        // (A^T)^T x = A x  when A is square
        let n = 4;
        let mat: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let v = vec![1.0, 2.0, 3.0, 4.0];

        // Compute transpose(mat) manually
        let mut mat_t = vec![0.0f32; 16];
        for i in 0..n {
            for j in 0..n {
                mat_t[j * n + i] = mat[i * n + j];
            }
        }

        let mut out_basic = vec![0.0; n];
        let mut out_trans = vec![0.0; n];
        unsafe {
            neon_gemv_f32(&mat, &v, &mut out_basic, n, n);
            neon_gemv_transposed_f32(&mat_t, &v, &mut out_trans, n, n);
        };
        assert_vec_close(&out_basic, &out_trans, 1e-5);
    }

    #[test]
    fn gemv_i8_unit_scale_matches_f32() {
        let (m, n) = (2, 4);
        let mat_i8 = vec![1i8, 2, 3, 4, 5, 6, 7, 8];
        let mat_f32: Vec<f32> = mat_i8.iter().map(|&x| x as f32).collect();
        let v = vec![1.0f32, 0.5, 0.25, 0.125];
        let mut out_i8 = vec![0.0; m];
        let mut out_f32 = vec![0.0; m];
        unsafe {
            neon_gemv_i8_f32(&mat_i8, &v, 1.0, &mut out_i8, m, n);
            neon_gemv_f32(&mat_f32, &v, &mut out_f32, m, n);
        };
        assert_vec_close(&out_i8, &out_f32, 1e-4);
    }
}
