//! ARM NEON fused multiply-add (FMA) operations for Apple Silicon.
//!
//! Provides high-precision fused multiply-add kernels using AArch64 NEON
//! intrinsics. FMA computes `a * b + c` in a single rounding step, giving
//! better numerical accuracy than separate multiply and add operations.
//!
//! All public functions require the `neon` target feature (always available
//! on AArch64) and process 4 × f32 lanes at a time with scalar fallback
//! for remainder elements.

use std::arch::aarch64::*;

// ── Fused Multiply-Add ──────────────────────────────────────────────

/// Fused multiply-add: `out[i] = a[i] * b[i] + c[i]`.
///
/// Uses the NEON [`vfmaq_f32`] intrinsic which computes the product and
/// sum in a single rounding step (IEEE 754 fusedMultiplyAdd), avoiding
/// the intermediate rounding error of a separate multiply then add.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `a`, `b`, `c`, and `out` do not all have the same length.
#[target_feature(enable = "neon")]
pub unsafe fn fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, c.len());
    assert_eq!(n, out.len());

    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let va = vld1q_f32(a_ptr.add(off));
        let vb = vld1q_f32(b_ptr.add(off));
        let vc = vld1q_f32(c_ptr.add(off));
        // vfmaq_f32(c, a, b) = a * b + c
        let vr = vfmaq_f32(vc, va, vb);
        vst1q_f32(o_ptr.add(off), vr);
    }

    for i in (chunks * 4)..n {
        out[i] = a[i].mul_add(b[i], c[i]);
    }
}

// ── Fused Multiply-Subtract ─────────────────────────────────────────

/// Fused multiply-subtract: `out[i] = a[i] * b[i] - c[i]`.
///
/// Implemented as `vfmaq_f32(-c, a, b)` which computes `a * b + (-c)`
/// in a single fused operation, preserving the precision benefit of FMA.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `a`, `b`, `c`, and `out` do not all have the same length.
#[target_feature(enable = "neon")]
pub unsafe fn fms_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, c.len());
    assert_eq!(n, out.len());

    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let va = vld1q_f32(a_ptr.add(off));
        let vb = vld1q_f32(b_ptr.add(off));
        let vc = vld1q_f32(c_ptr.add(off));
        let neg_c = vnegq_f32(vc);
        let vr = vfmaq_f32(neg_c, va, vb);
        vst1q_f32(o_ptr.add(off), vr);
    }

    for i in (chunks * 4)..n {
        out[i] = a[i].mul_add(b[i], -c[i]);
    }
}

// ── Fused Dot-Product Accumulate ────────────────────────────────────

/// Accumulate the dot product of `a` and `b` using FMA for precision.
///
/// Computes `sum(a[i] * b[i])` by accumulating partial products with
/// [`vfmaq_f32`] into a NEON accumulator, then reducing the 4-lane
/// vector with [`vaddvq_f32`]. This avoids intermediate rounding on
/// each multiply-add step compared to naive `sum += a[i] * b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `a` and `b` do not have the same length.
#[target_feature(enable = "neon")]
pub unsafe fn dot_product_fma_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    assert_eq!(n, b.len());

    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        let va = vld1q_f32(a_ptr.add(off));
        let vb = vld1q_f32(b_ptr.add(off));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut sum = vaddvq_f32(acc);
    for i in (chunks * 4)..n {
        sum = a[i].mul_add(b[i], sum);
    }
    sum
}

// ── Matrix-Vector FMA ───────────────────────────────────────────────

/// Matrix-vector fused multiply-add: `out[i] = dot(matrix[i], x) + bias[i]`.
///
/// Each row of `matrix` (stored row-major as a flat slice of length
/// `rows * cols`) is dot-producted with `x` using [`vfmaq_f32`], then
/// the per-row bias is added in a final fused step.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if dimensions are inconsistent:
/// - `matrix.len() != rows * cols`
/// - `x.len() != cols`
/// - `bias.len() != rows`
/// - `out.len() != rows`
#[target_feature(enable = "neon")]
pub unsafe fn matvec_fma_f32(
    matrix: &[f32],
    x: &[f32],
    bias: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(bias.len(), rows);
    assert_eq!(out.len(), rows);

    let chunks = cols / 4;
    let x_ptr = x.as_ptr();

    for r in 0..rows {
        let row_ptr = matrix.as_ptr().add(r * cols);
        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let off = c * 4;
            let vm = vld1q_f32(row_ptr.add(off));
            let vx = vld1q_f32(x_ptr.add(off));
            acc = vfmaq_f32(acc, vm, vx);
        }

        let mut row_sum = vaddvq_f32(acc);
        for c in (chunks * 4)..cols {
            row_sum = matrix[r * cols + c].mul_add(x[c], row_sum);
        }
        out[r] = row_sum + bias[r];
    }
}

// ── Chained FMA (Horner's Method) ───────────────────────────────────

/// Evaluate a polynomial using Horner's method with chained FMA.
///
/// Given coefficients `[c0, c1, c2, ..., cn]` representing the polynomial
/// `c0 + c1*x + c2*x^2 + ... + cn*x^n`, evaluates it element-wise on
/// the input slice using chained [`vfmaq_f32`] calls:
///
/// ```text
/// result = c0 + x*(c1 + x*(c2 + ... + x*cn))
/// ```
///
/// Each chained FMA step avoids an intermediate rounding, making this
/// more precise than computing powers of x separately.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `out.len() != x.len()` or if `coeffs` is empty.
#[target_feature(enable = "neon")]
pub unsafe fn horner_fma_f32(x: &[f32], coeffs: &[f32], out: &mut [f32]) {
    let n = x.len();
    assert_eq!(n, out.len());
    assert!(!coeffs.is_empty(), "coefficients must not be empty");

    let chunks = n / 4;
    let x_ptr = x.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let degree = coeffs.len();

    for i in 0..chunks {
        let off = i * 4;
        let vx = vld1q_f32(x_ptr.add(off));
        // Start from the highest-degree coefficient.
        let mut acc = vdupq_n_f32(coeffs[degree - 1]);
        for k in (0..degree - 1).rev() {
            // acc = acc * x + coeffs[k]
            let vc = vdupq_n_f32(coeffs[k]);
            acc = vfmaq_f32(vc, acc, vx);
        }
        vst1q_f32(o_ptr.add(off), acc);
    }

    // Scalar tail using f32::mul_add for the same fused semantics.
    for i in (chunks * 4)..n {
        let xv = x[i];
        let mut acc = coeffs[degree - 1];
        for k in (0..degree - 1).rev() {
            acc = acc.mul_add(xv, coeffs[k]);
        }
        out[i] = acc;
    }
}

// ── Fused Scale-and-Bias ────────────────────────────────────────────

/// Fused scale-and-bias: `out[i] = scale[i] * x[i] + bias[i]`.
///
/// Common in normalization layers (LayerNorm, BatchNorm) where the
/// affine transform `gamma * x_hat + beta` is applied after
/// normalization. Uses [`vfmaq_f32`] to fuse the multiply and add.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `x`, `scale`, `bias`, and `out` do not all have the same length.
#[target_feature(enable = "neon")]
pub unsafe fn scale_bias_f32(x: &[f32], scale: &[f32], bias: &[f32], out: &mut [f32]) {
    let n = x.len();
    assert_eq!(n, scale.len());
    assert_eq!(n, bias.len());
    assert_eq!(n, out.len());

    let chunks = n / 4;
    let x_ptr = x.as_ptr();
    let s_ptr = scale.as_ptr();
    let b_ptr = bias.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let vx = vld1q_f32(x_ptr.add(off));
        let vs = vld1q_f32(s_ptr.add(off));
        let vb = vld1q_f32(b_ptr.add(off));
        // scale * x + bias
        let vr = vfmaq_f32(vb, vs, vx);
        vst1q_f32(o_ptr.add(off), vr);
    }

    for i in (chunks * 4)..n {
        out[i] = scale[i].mul_add(x[i], bias[i]);
    }
}

/// Fused scale-and-bias with uniform (broadcast) scale and bias scalars:
/// `out[i] = scale * x[i] + bias`.
///
/// Broadcasts `scale` and `bias` across all lanes using [`vdupq_n_f32`]
/// and fuses with [`vfmaq_f32`].
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `out.len() != x.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn scale_bias_scalar_f32(x: &[f32], scale: f32, bias: f32, out: &mut [f32]) {
    let n = x.len();
    assert_eq!(n, out.len());

    let chunks = n / 4;
    let x_ptr = x.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let vs = vdupq_n_f32(scale);
    let vb = vdupq_n_f32(bias);

    for i in 0..chunks {
        let off = i * 4;
        let vx = vld1q_f32(x_ptr.add(off));
        let vr = vfmaq_f32(vb, vs, vx);
        vst1q_f32(o_ptr.add(off), vr);
    }

    for i in (chunks * 4)..n {
        out[i] = scale.mul_add(x[i], bias);
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for FMA vs naive comparison. FMA is *more* precise; the
    // naive path accumulates an extra rounding so we allow a small delta.
    const EPS: f32 = 1e-6;

    fn naive_fma(a: f32, b: f32, c: f32) -> f32 {
        a * b + c
    }

    // ── fma_f32 ─────────────────────────────────────────────────────

    #[test]
    fn test_fma_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0.0; 5];
        unsafe { fma_f32(&a, &b, &c, &mut out) };
        for i in 0..5 {
            assert!((out[i] - naive_fma(a[i], b[i], c[i])).abs() < EPS);
        }
    }

    #[test]
    fn test_fma_zeros() {
        let z = vec![0.0; 8];
        let mut out = vec![1.0; 8];
        unsafe { fma_f32(&z, &z, &z, &mut out) };
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_fma_negative() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![10.0, 10.0, 10.0, 10.0];
        let mut out = vec![0.0; 4];
        unsafe { fma_f32(&a, &b, &c, &mut out) };
        let expected = [9.0, 6.0, 1.0, -6.0];
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_fma_nan_propagation() {
        let a = vec![f32::NAN, 1.0, 2.0, 3.0];
        let b = vec![1.0; 4];
        let c = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        unsafe { fma_f32(&a, &b, &c, &mut out) };
        assert!(out[0].is_nan());
    }

    #[test]
    fn test_fma_inf() {
        let a = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0, 0.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let c = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        unsafe { fma_f32(&a, &b, &c, &mut out) };
        assert_eq!(out[0], f32::INFINITY);
        assert_eq!(out[1], f32::NEG_INFINITY);
    }

    #[test]
    fn test_fma_subnormals() {
        let tiny = f32::MIN_POSITIVE / 2.0; // subnormal
        let a = vec![tiny; 4];
        let b = vec![1.0; 4];
        let c = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        unsafe { fma_f32(&a, &b, &c, &mut out) };
        for &v in &out {
            assert!((v - tiny).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_fma_single_element() {
        let a = vec![3.0];
        let b = vec![4.0];
        let c = vec![5.0];
        let mut out = vec![0.0];
        unsafe { fma_f32(&a, &b, &c, &mut out) };
        assert!((out[0] - 17.0).abs() < EPS);
    }

    #[test]
    fn test_fma_empty() {
        let empty: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        unsafe { fma_f32(&empty, &empty, &empty, &mut out) };
        assert!(out.is_empty());
    }

    // ── fms_f32 ─────────────────────────────────────────────────────

    #[test]
    fn test_fms_basic() {
        let a = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let c = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0; 5];
        unsafe { fms_f32(&a, &b, &c, &mut out) };
        for i in 0..5 {
            let expected = a[i] * b[i] - c[i];
            assert!((out[i] - expected).abs() < EPS);
        }
    }

    #[test]
    fn test_fms_nan() {
        let a = vec![1.0, f32::NAN, 3.0, 4.0];
        let b = vec![1.0; 4];
        let c = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        unsafe { fms_f32(&a, &b, &c, &mut out) };
        assert!(out[1].is_nan());
    }

    #[test]
    fn test_fms_identity() {
        // a * b - 0 == a * b
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        unsafe { fms_f32(&a, &b, &c, &mut out) };
        let expected = [5.0, 12.0, 21.0, 32.0];
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < EPS);
        }
    }

    // ── dot_product_fma_f32 ─────────────────────────────────────────

    #[test]
    fn test_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result = unsafe { dot_product_fma_f32(&a, &b) };
        // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 35
        assert!((result - 35.0).abs() < EPS);
    }

    #[test]
    fn test_dot_zeros() {
        let a = vec![0.0; 8];
        let b = vec![1.0; 8];
        let result = unsafe { dot_product_fma_f32(&a, &b) };
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_single() {
        let a = vec![3.0];
        let b = vec![7.0];
        let result = unsafe { dot_product_fma_f32(&a, &b) };
        assert!((result - 21.0).abs() < EPS);
    }

    #[test]
    fn test_dot_large() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b = vec![1.0; n];
        let result = unsafe { dot_product_fma_f32(&a, &b) };
        let expected: f32 = (0..n).map(|i| i as f32).sum();
        assert!((result - expected).abs() < 1.0);
    }

    #[test]
    fn test_dot_empty() {
        let empty: Vec<f32> = vec![];
        let result = unsafe { dot_product_fma_f32(&empty, &empty) };
        assert_eq!(result, 0.0);
    }

    // ── matvec_fma_f32 ──────────────────────────────────────────────

    #[test]
    fn test_matvec_identity() {
        // 2x2 identity matrix
        let mat = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, 7.0];
        let bias = vec![0.0, 0.0];
        let mut out = vec![0.0; 2];
        unsafe { matvec_fma_f32(&mat, &x, &bias, &mut out, 2, 2) };
        assert!((out[0] - 3.0).abs() < EPS);
        assert!((out[1] - 7.0).abs() < EPS);
    }

    #[test]
    fn test_matvec_with_bias() {
        let mat = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let x = vec![1.0, 1.0];
        let bias = vec![10.0, 20.0];
        let mut out = vec![0.0; 2];
        unsafe { matvec_fma_f32(&mat, &x, &bias, &mut out, 2, 2) };
        // row0: 1+2+10=13, row1: 3+4+20=27
        assert!((out[0] - 13.0).abs() < EPS);
        assert!((out[1] - 27.0).abs() < EPS);
    }

    #[test]
    fn test_matvec_wide() {
        // 1x8 matrix (exercises NEON path fully for one row)
        let mat: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let x = vec![1.0; 8];
        let bias = vec![0.0];
        let mut out = vec![0.0; 1];
        unsafe { matvec_fma_f32(&mat, &x, &bias, &mut out, 1, 8) };
        // 1+2+...+8 = 36
        assert!((out[0] - 36.0).abs() < EPS);
    }

    #[test]
    fn test_matvec_non_aligned_cols() {
        // 2x5 matrix (5 cols triggers scalar tail)
        let mat = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // row 0
            6.0, 7.0, 8.0, 9.0, 10.0, // row 1
        ];
        let x = vec![1.0; 5];
        let bias = vec![0.0, 0.0];
        let mut out = vec![0.0; 2];
        unsafe { matvec_fma_f32(&mat, &x, &bias, &mut out, 2, 5) };
        assert!((out[0] - 15.0).abs() < EPS);
        assert!((out[1] - 40.0).abs() < EPS);
    }

    // ── horner_fma_f32 ──────────────────────────────────────────────

    #[test]
    fn test_horner_constant() {
        // p(x) = 5
        let x = vec![0.0, 1.0, 2.0, 3.0, 100.0];
        let coeffs = vec![5.0];
        let mut out = vec![0.0; 5];
        unsafe { horner_fma_f32(&x, &coeffs, &mut out) };
        for &v in &out {
            assert!((v - 5.0).abs() < EPS);
        }
    }

    #[test]
    fn test_horner_linear() {
        // p(x) = 2 + 3x
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let coeffs = vec![2.0, 3.0];
        let mut out = vec![0.0; 5];
        unsafe { horner_fma_f32(&x, &coeffs, &mut out) };
        let expected = [2.0, 5.0, 8.0, 11.0, 14.0];
        for i in 0..5 {
            assert!((out[i] - expected[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_horner_quadratic() {
        // p(x) = 1 + 0*x + 1*x^2 = 1 + x^2
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let coeffs = vec![1.0, 0.0, 1.0];
        let mut out = vec![0.0; 4];
        unsafe { horner_fma_f32(&x, &coeffs, &mut out) };
        let expected = [1.0, 2.0, 5.0, 10.0];
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_horner_cubic() {
        // p(x) = 1 + 2x + 3x^2 + 4x^3
        let x = vec![1.0, 2.0, -1.0, 0.0, 0.5];
        let coeffs = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 5];
        unsafe { horner_fma_f32(&x, &coeffs, &mut out) };
        for i in 0..5 {
            let xv = x[i];
            let expected = 1.0 + 2.0 * xv + 3.0 * xv * xv + 4.0 * xv * xv * xv;
            assert!((out[i] - expected).abs() < EPS, "x={xv}: got {} expected {expected}", out[i]);
        }
    }

    // ── scale_bias_f32 ──────────────────────────────────────────────

    #[test]
    fn test_scale_bias_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scale = vec![2.0; 5];
        let bias = vec![1.0; 5];
        let mut out = vec![0.0; 5];
        unsafe { scale_bias_f32(&x, &scale, &bias, &mut out) };
        let expected = [3.0, 5.0, 7.0, 9.0, 11.0];
        for i in 0..5 {
            assert!((out[i] - expected[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_scale_bias_identity() {
        // scale=1, bias=0 → identity
        let x: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let scale = vec![1.0; 9];
        let bias = vec![0.0; 9];
        let mut out = vec![0.0; 9];
        unsafe { scale_bias_f32(&x, &scale, &bias, &mut out) };
        for i in 0..9 {
            assert!((out[i] - x[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_scale_bias_nan_in_x() {
        let x = vec![f32::NAN, 1.0, 2.0, 3.0];
        let scale = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        unsafe { scale_bias_f32(&x, &scale, &bias, &mut out) };
        assert!(out[0].is_nan());
    }

    #[test]
    fn test_scale_bias_inf() {
        let x = vec![f32::INFINITY; 4];
        let scale = vec![2.0; 4];
        let bias = vec![-1.0; 4];
        let mut out = vec![0.0; 4];
        unsafe { scale_bias_f32(&x, &scale, &bias, &mut out) };
        assert_eq!(out[0], f32::INFINITY);
    }

    // ── scale_bias_scalar_f32 ───────────────────────────────────────

    #[test]
    fn test_scale_bias_scalar_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0; 5];
        unsafe { scale_bias_scalar_f32(&x, 2.0, 1.0, &mut out) };
        let expected = [3.0, 5.0, 7.0, 9.0, 11.0];
        for i in 0..5 {
            assert!((out[i] - expected[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_scale_bias_scalar_zero_scale() {
        let x = vec![100.0, 200.0, 300.0, 400.0];
        let mut out = vec![0.0; 4];
        unsafe { scale_bias_scalar_f32(&x, 0.0, 5.0, &mut out) };
        for &v in &out {
            assert!((v - 5.0).abs() < EPS);
        }
    }

    #[test]
    fn test_scale_bias_scalar_large() {
        let n = 1025; // non-aligned
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut out = vec![0.0; n];
        unsafe { scale_bias_scalar_f32(&x, 3.0, -1.0, &mut out) };
        for i in 0..n {
            let expected = 3.0 * x[i] - 1.0;
            assert!((out[i] - expected).abs() < EPS);
        }
    }

    // ── Precision: FMA vs naive ─────────────────────────────────────

    #[test]
    fn test_fma_precision_vs_naive() {
        // Kahan's classic example: a*b+c where naive loses precision.
        // With f32 the effect is small, but FMA should be at least as
        // good as the naive path.
        let a = vec![1.0 + 1e-7_f32; 4];
        let b = vec![1.0 + 1e-7_f32; 4];
        let c = vec![-(1.0 + 2e-7_f32); 4];
        let mut out = vec![0.0; 4];
        unsafe { fma_f32(&a, &b, &c, &mut out) };
        for i in 0..4 {
            let naive = a[i] * b[i] + c[i];
            // Both should be very close; FMA may differ in the last bit.
            assert!((out[i] - naive).abs() < 1e-12);
        }
    }

    #[test]
    fn test_dot_precision_large_cancellation() {
        // Alternating positive/negative to exercise cancellation.
        let n = 128;
        let a: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let b = vec![1.0; n];
        let result = unsafe { dot_product_fma_f32(&a, &b) };
        assert!((result - 0.0).abs() < EPS);
    }

    // ── Cross-function consistency ──────────────────────────────────

    #[test]
    fn test_fms_equals_fma_negated_c() {
        let a = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let neg_c: Vec<f32> = c.iter().map(|&v| -v).collect();

        let mut fms_out = vec![0.0; 5];
        let mut fma_out = vec![0.0; 5];
        unsafe {
            fms_f32(&a, &b, &c, &mut fms_out);
            fma_f32(&a, &b, &neg_c, &mut fma_out);
        }
        for i in 0..5 {
            assert!(
                (fms_out[i] - fma_out[i]).abs() < EPS,
                "mismatch at {i}: fms={} fma_neg={}",
                fms_out[i],
                fma_out[i]
            );
        }
    }
}
