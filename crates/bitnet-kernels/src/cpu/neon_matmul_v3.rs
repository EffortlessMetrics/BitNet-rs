//! Advanced NEON-optimized matrix multiplication kernels for Apple Silicon (aarch64).
//!
//! Provides six operations with NEON fast-paths and scalar fallbacks:
//! - `gemv_f32` — matrix-vector multiply
//! - `gemm_f32` — general matrix multiply (4×4 micro-kernel tiling)
//! - `gemm_transb_f32` — GEMM with B transposed (common in attention)
//! - `batched_gemm_f32` — batched matrix multiply
//! - `fused_gemm_bias_relu_f32` — fused GEMM + bias + ReLU
//! - `quantized_gemv_i2_f32` — 2-bit quantized weights × f32 input
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_unsafe)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::manual_is_multiple_of)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ────────────────────────────────────────────────────────────────────────
// 1. GEMV  — y[m] = A[m×n] · x[n]
// ────────────────────────────────────────────────────────────────────────

/// NEON-accelerated matrix-vector multiply.
///
/// # Safety
/// Requires `neon` target feature at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gemv_f32(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    for i in 0..m {
        let row = &a[i * n..i * n + n];
        let mut acc0 = unsafe { vdupq_n_f32(0.0) };
        let mut acc1 = unsafe { vdupq_n_f32(0.0) };

        let chunks = n / 8;
        for c in 0..chunks {
            let base = c * 8;
            unsafe {
                let a0 = vld1q_f32(row.as_ptr().add(base));
                let x0 = vld1q_f32(x.as_ptr().add(base));
                acc0 = vfmaq_f32(acc0, a0, x0);

                let a1 = vld1q_f32(row.as_ptr().add(base + 4));
                let x1 = vld1q_f32(x.as_ptr().add(base + 4));
                acc1 = vfmaq_f32(acc1, a1, x1);
            }
        }

        // Reduce two accumulators
        let sum_vec = unsafe { vaddq_f32(acc0, acc1) };
        let mut sum = unsafe { vaddvq_f32(sum_vec) };

        // Scalar tail
        for j in (chunks * 8)..n {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

fn scalar_gemv_f32(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

/// Matrix-vector multiply: `y[m] = A[m×n] · x[n]`.
pub fn gemv_f32(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    assert!(a.len() >= m * n, "a too short");
    assert!(x.len() >= n, "x too short");
    assert!(y.len() >= m, "y too short");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detected above.
            unsafe {
                neon_gemv_f32(a, x, y, m, n);
            }
            return;
        }
    }
    scalar_gemv_f32(a, x, y, m, n);
}

// ────────────────────────────────────────────────────────────────────────
// 2. GEMM  — C[m×n] = A[m×k] · B[k×n]  (4×4 micro-kernel)
// ────────────────────────────────────────────────────────────────────────

/// NEON GEMM with 4×4 micro-kernel tiling.
///
/// # Safety
/// Requires `neon` target feature at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let m4 = m / 4 * 4;
    let n4 = n / 4 * 4;

    // 4×4 tiled core
    for i in (0..m4).step_by(4) {
        for j in (0..n4).step_by(4) {
            // 4×4 accumulator block (4 rows, each a NEON register of 4 cols)
            let mut c00 = unsafe { vdupq_n_f32(0.0) };
            let mut c10 = unsafe { vdupq_n_f32(0.0) };
            let mut c20 = unsafe { vdupq_n_f32(0.0) };
            let mut c30 = unsafe { vdupq_n_f32(0.0) };

            for p in 0..k {
                unsafe {
                    let b_row = vld1q_f32(b.as_ptr().add(p * n + j));

                    let a0 = vdupq_n_f32(*a.get_unchecked(i * k + p));
                    c00 = vfmaq_f32(c00, a0, b_row);

                    let a1 = vdupq_n_f32(*a.get_unchecked((i + 1) * k + p));
                    c10 = vfmaq_f32(c10, a1, b_row);

                    let a2 = vdupq_n_f32(*a.get_unchecked((i + 2) * k + p));
                    c20 = vfmaq_f32(c20, a2, b_row);

                    let a3 = vdupq_n_f32(*a.get_unchecked((i + 3) * k + p));
                    c30 = vfmaq_f32(c30, a3, b_row);
                }
            }

            unsafe {
                vst1q_f32(c.as_mut_ptr().add(i * n + j), c00);
                vst1q_f32(c.as_mut_ptr().add((i + 1) * n + j), c10);
                vst1q_f32(c.as_mut_ptr().add((i + 2) * n + j), c20);
                vst1q_f32(c.as_mut_ptr().add((i + 3) * n + j), c30);
            }
        }

        // Right-edge remainder columns (j in n4..n)
        for ii in i..i + 4 {
            for j in n4..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[ii * k + p] * b[p * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Bottom-edge remainder rows
    for i in m4..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn scalar_gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// General matrix multiply: `C[m×n] = A[m×k] · B[k×n]`.
pub fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(a.len() >= m * k, "a too short");
    assert!(b.len() >= k * n, "b too short");
    assert!(c.len() >= m * n, "c too short");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_gemm_f32(a, b, c, m, n, k);
            }
            return;
        }
    }
    scalar_gemm_f32(a, b, c, m, n, k);
}

// ────────────────────────────────────────────────────────────────────────
// 3. GEMM with B transposed — C[m×n] = A[m×k] · Bᵀ[n×k]
// ────────────────────────────────────────────────────────────────────────

/// NEON GEMM with B transposed.
///
/// # Safety
/// Requires `neon` target feature at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gemm_transb_f32(
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc0 = unsafe { vdupq_n_f32(0.0) };
            let mut acc1 = unsafe { vdupq_n_f32(0.0) };

            let chunks = k / 8;
            for ch in 0..chunks {
                let base = ch * 8;
                unsafe {
                    let av0 = vld1q_f32(a.as_ptr().add(i * k + base));
                    let bv0 = vld1q_f32(b_t.as_ptr().add(j * k + base));
                    acc0 = vfmaq_f32(acc0, av0, bv0);

                    let av1 = vld1q_f32(a.as_ptr().add(i * k + base + 4));
                    let bv1 = vld1q_f32(b_t.as_ptr().add(j * k + base + 4));
                    acc1 = vfmaq_f32(acc1, av1, bv1);
                }
            }

            let sum_vec = unsafe { vaddq_f32(acc0, acc1) };
            let mut sum = unsafe { vaddvq_f32(sum_vec) };

            for p in (chunks * 8)..k {
                sum += a[i * k + p] * b_t[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}

fn scalar_gemm_transb_f32(a: &[f32], b_t: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b_t[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}

/// GEMM with B transposed: `C[m×n] = A[m×k] · Bᵀ[n×k]`.
pub fn gemm_transb_f32(a: &[f32], b_t: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(a.len() >= m * k, "a too short");
    assert!(b_t.len() >= n * k, "b_t too short");
    assert!(c.len() >= m * n, "c too short");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_gemm_transb_f32(a, b_t, c, m, n, k);
            }
            return;
        }
    }
    scalar_gemm_transb_f32(a, b_t, c, m, n, k);
}

// ────────────────────────────────────────────────────────────────────────
// 4. Batched GEMM — for each batch: C[m×n] = A[m×k] · B[k×n]
// ────────────────────────────────────────────────────────────────────────

/// Batched matrix multiply. Each batch slice is contiguous:
/// `a[b*m*k..]`, `b_mat[b*k*n..]`, `c[b*m*n..]`.
pub fn batched_gemm_f32(
    a: &[f32],
    b_mat: &[f32],
    c: &mut [f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;
    assert!(a.len() >= batch * a_stride, "a too short for batch");
    assert!(b_mat.len() >= batch * b_stride, "b too short for batch");
    assert!(c.len() >= batch * c_stride, "c too short for batch");

    for bi in 0..batch {
        let a_slice = &a[bi * a_stride..bi * a_stride + a_stride];
        let b_slice = &b_mat[bi * b_stride..bi * b_stride + b_stride];
        let c_slice = &mut c[bi * c_stride..bi * c_stride + c_stride];
        gemm_f32(a_slice, b_slice, c_slice, m, n, k);
    }
}

// ────────────────────────────────────────────────────────────────────────
// 5. Fused GEMM + Bias + ReLU
// ────────────────────────────────────────────────────────────────────────

/// NEON fused GEMM + bias + ReLU.
///
/// # Safety
/// Requires `neon` target feature at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_gemm_bias_relu_f32(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // First compute GEMM into c.
    unsafe {
        neon_gemm_f32(a, b, c, m, n, k);
    }

    let zero = unsafe { vdupq_n_f32(0.0) };

    // Apply bias + ReLU row by row.
    for i in 0..m {
        let row_off = i * n;
        let n4 = n / 4 * 4;

        for j in (0..n4).step_by(4) {
            unsafe {
                let cv = vld1q_f32(c.as_ptr().add(row_off + j));
                let bv = vld1q_f32(bias.as_ptr().add(j));
                let sum = vaddq_f32(cv, bv);
                let relu = vmaxq_f32(sum, zero);
                vst1q_f32(c.as_mut_ptr().add(row_off + j), relu);
            }
        }

        for j in n4..n {
            let val = c[row_off + j] + bias[j];
            c[row_off + j] = val.max(0.0);
        }
    }
}

fn scalar_fused_gemm_bias_relu_f32(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    scalar_gemm_f32(a, b, c, m, n, k);
    for i in 0..m {
        for j in 0..n {
            let val = c[i * n + j] + bias[j];
            c[i * n + j] = val.max(0.0);
        }
    }
}

/// Fused GEMM + bias + ReLU: `C[m×n] = ReLU(A[m×k]·B[k×n] + bias[n])`.
pub fn fused_gemm_bias_relu_f32(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(a.len() >= m * k, "a too short");
    assert!(b.len() >= k * n, "b too short");
    assert!(bias.len() >= n, "bias too short");
    assert!(c.len() >= m * n, "c too short");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_fused_gemm_bias_relu_f32(a, b, bias, c, m, n, k);
            }
            return;
        }
    }
    scalar_fused_gemm_bias_relu_f32(a, b, bias, c, m, n, k);
}

// ────────────────────────────────────────────────────────────────────────
// 6. Quantized GEMV (I2_S 2-bit weights × f32 input)
// ────────────────────────────────────────────────────────────────────────

/// Decode a single 2-bit I2_S code: 0→0, 1→+1, 3→−1 (2→0 unused).
#[inline(always)]
fn decode_i2(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0,
    }
}

/// NEON quantized GEMV: y[m] = (decode(weights)[m×n] .* scales) · x[n].
///
/// `weights` is packed 2-bit: 4 values per byte, LSB-first.
/// `scales` has one entry per row.
///
/// # Safety
/// Requires `neon` target feature at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_quantized_gemv_i2_f32(
    weights: &[u8],
    scales: &[f32],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
) {
    let bytes_per_row = (n + 3) / 4;

    for i in 0..m {
        let row = &weights[i * bytes_per_row..(i + 1) * bytes_per_row];
        let scale = scales[i];

        let mut acc0 = unsafe { vdupq_n_f32(0.0) };
        let full_bytes = n / 4;
        let remainder = n % 4;

        for byte_idx in 0..full_bytes {
            let packed = row[byte_idx];
            let x_base = byte_idx * 4;

            // Decode 4 weights from one byte
            let w0 = decode_i2(packed);
            let w1 = decode_i2(packed >> 2);
            let w2 = decode_i2(packed >> 4);
            let w3 = decode_i2(packed >> 6);

            unsafe {
                let wv = {
                    let arr = [w0, w1, w2, w3];
                    vld1q_f32(arr.as_ptr())
                };
                let xv = vld1q_f32(x.as_ptr().add(x_base));
                acc0 = vfmaq_f32(acc0, wv, xv);
            }
        }

        let mut sum = unsafe { vaddvq_f32(acc0) };

        // Scalar remainder
        for r in 0..remainder {
            let col = full_bytes * 4 + r;
            let byte_idx = col / 4;
            let bit_off = (col % 4) * 2;
            let w = decode_i2(row[byte_idx] >> bit_off);
            sum += w * x[col];
        }

        y[i] = sum * scale;
    }
}

fn scalar_quantized_gemv_i2_f32(
    weights: &[u8],
    scales: &[f32],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
) {
    let bytes_per_row = (n + 3) / 4;

    for i in 0..m {
        let row = &weights[i * bytes_per_row..(i + 1) * bytes_per_row];
        let scale = scales[i];
        let mut sum = 0.0f32;

        for j in 0..n {
            let byte_idx = j / 4;
            let bit_off = (j % 4) * 2;
            let w = decode_i2(row[byte_idx] >> bit_off);
            sum += w * x[j];
        }

        y[i] = sum * scale;
    }
}

/// Quantized 2-bit GEMV: `y[m] = diag(scales) · decode(weights)[m×n] · x[n]`.
///
/// `weights` is I2_S packed (4 values per byte, LSB-first).
/// Encoding: `0b01`→+1, `0b11`→−1, others→0.
pub fn quantized_gemv_i2_f32(
    weights: &[u8],
    scales: &[f32],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
) {
    let bytes_per_row = (n + 3) / 4;
    assert!(weights.len() >= m * bytes_per_row, "weights too short");
    assert!(scales.len() >= m, "scales too short");
    assert!(x.len() >= n, "x too short");
    assert!(y.len() >= m, "y too short");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_quantized_gemv_i2_f32(weights, scales, x, y, m, n);
            }
            return;
        }
    }
    scalar_quantized_gemv_i2_f32(weights, scales, x, y, m, n);
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (va - vb).abs() <= tol,
                "mismatch at index {i}: {va} vs {vb} (diff={})",
                (va - vb).abs()
            );
        }
    }

    /// Naive reference matmul for verification.
    fn ref_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn ref_gemv(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for i in 0..m {
            let mut s = 0.0f32;
            for j in 0..n {
                s += a[i * n + j] * x[j];
            }
            y[i] = s;
        }
        y
    }

    fn ref_gemm_transb(a: &[f32], b_t: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b_t[j * k + p];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn make_seq(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32) * 0.1 + 0.01).collect()
    }

    // ── GEMV tests ─────────────────────────────────────────────

    #[test]
    fn test_gemv_1x1() {
        let a = [3.0f32];
        let x = [2.0f32];
        let mut y = [0.0f32];
        gemv_f32(&a, &x, &mut y, 1, 1);
        approx_eq(&y, &[6.0], TOL);
    }

    #[test]
    fn test_gemv_1xn() {
        let n = 8;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = vec![1.0f32; n];
        let mut y = vec![0.0f32; 1];
        gemv_f32(&a, &x, &mut y, 1, n);
        let expected: f32 = (0..n).map(|i| i as f32).sum();
        approx_eq(&y, &[expected], TOL);
    }

    #[test]
    fn test_gemv_mx1() {
        let m = 5;
        let a: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();
        let x = [2.0f32];
        let mut y = vec![0.0f32; m];
        gemv_f32(&a, &x, &mut y, m, 1);
        let expected: Vec<f32> = (0..m).map(|i| (i + 1) as f32 * 2.0).collect();
        approx_eq(&y, &expected, TOL);
    }

    #[test]
    fn test_gemv_4x4() {
        let m = 4;
        let n = 4;
        let a = make_seq(m * n);
        let x = make_seq(n);
        let mut y = vec![0.0f32; m];
        gemv_f32(&a, &x, &mut y, m, n);
        let expected = ref_gemv(&a, &x, m, n);
        approx_eq(&y, &expected, TOL);
    }

    #[test]
    fn test_gemv_8x16() {
        let (m, n) = (8, 16);
        let a = make_seq(m * n);
        let x = make_seq(n);
        let mut y = vec![0.0f32; m];
        gemv_f32(&a, &x, &mut y, m, n);
        let expected = ref_gemv(&a, &x, m, n);
        approx_eq(&y, &expected, TOL);
    }

    #[test]
    fn test_gemv_non_multiple_of_8() {
        let (m, n) = (3, 11);
        let a = make_seq(m * n);
        let x = make_seq(n);
        let mut y = vec![0.0f32; m];
        gemv_f32(&a, &x, &mut y, m, n);
        let expected = ref_gemv(&a, &x, m, n);
        approx_eq(&y, &expected, TOL);
    }

    #[test]
    fn test_gemv_zeros() {
        let (m, n) = (4, 4);
        let a = vec![0.0f32; m * n];
        let x = make_seq(n);
        let mut y = vec![99.0f32; m];
        gemv_f32(&a, &x, &mut y, m, n);
        approx_eq(&y, &vec![0.0; m], TOL);
    }

    #[test]
    fn test_gemv_identity_like() {
        let n = 6;
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let x = make_seq(n);
        let mut y = vec![0.0f32; n];
        gemv_f32(&a, &x, &mut y, n, n);
        approx_eq(&y, &x, TOL);
    }

    #[test]
    fn test_gemv_large_64() {
        let (m, n) = (64, 64);
        let a = make_seq(m * n);
        let x = make_seq(n);
        let mut y = vec![0.0f32; m];
        gemv_f32(&a, &x, &mut y, m, n);
        let expected = ref_gemv(&a, &x, m, n);
        approx_eq(&y, &expected, 0.1); // larger tolerance for accumulated FP error
    }

    // ── GEMM tests ─────────────────────────────────────────────

    #[test]
    fn test_gemm_1x1x1() {
        let mut c = [0.0f32];
        gemm_f32(&[3.0], &[2.0], &mut c, 1, 1, 1);
        approx_eq(&c, &[6.0], TOL);
    }

    #[test]
    fn test_gemm_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        gemm_f32(&a, &b, &mut c, 2, 2, 2);
        let expected = ref_gemm(&a, &b, 2, 2, 2);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_4x4() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_8x8() {
        let (m, n, k) = (8, 8, 8);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_non_square_3x5x7() {
        let (m, n, k) = (3, 5, 7);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_1xn() {
        let (m, n, k) = (1, 8, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_mx1() {
        let (m, n, k) = (8, 1, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_identity() {
        let n = 4;
        let mut id = vec![0.0f32; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        let a = make_seq(n * n);
        let mut c = vec![0.0f32; n * n];
        gemm_f32(&a, &id, &mut c, n, n, n);
        approx_eq(&c, &a, TOL);
    }

    #[test]
    fn test_gemm_zeros() {
        let (m, n, k) = (4, 4, 4);
        let a = vec![0.0f32; m * k];
        let b = make_seq(k * n);
        let mut c = vec![99.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        approx_eq(&c, &vec![0.0; m * n], TOL);
    }

    #[test]
    fn test_gemm_5x6x7_remainder() {
        let (m, n, k) = (5, 6, 7);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_16x16() {
        let (m, n, k) = (16, 16, 16);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_gemm_64x64() {
        let (m, n, k) = (64, 64, 64);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        // Large matrices accumulate more FP error
        approx_eq(&c, &expected, 2.0);
    }

    // ── GEMM TransB tests ──────────────────────────────────────

    #[test]
    fn test_gemm_transb_1x1() {
        let mut c = [0.0f32];
        gemm_transb_f32(&[3.0], &[2.0], &mut c, 1, 1, 1);
        approx_eq(&c, &[6.0], TOL);
    }

    #[test]
    fn test_gemm_transb_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b_t = [5.0, 7.0, 6.0, 8.0f32]; // transposed
        let mut c = [0.0f32; 4];
        gemm_transb_f32(&a, &b_t, &mut c, 2, 2, 2);
        let expected = ref_gemm_transb(&a, &b_t, 2, 2, 2);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_transb_4x4() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b_t = make_seq(n * k);
        let mut c = vec![0.0f32; m * n];
        gemm_transb_f32(&a, &b_t, &mut c, m, n, k);
        let expected = ref_gemm_transb(&a, &b_t, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_transb_non_square() {
        let (m, n, k) = (3, 5, 9);
        let a = make_seq(m * k);
        let b_t = make_seq(n * k);
        let mut c = vec![0.0f32; m * n];
        gemm_transb_f32(&a, &b_t, &mut c, m, n, k);
        let expected = ref_gemm_transb(&a, &b_t, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_transb_8x16() {
        let (m, n, k) = (8, 16, 12);
        let a = make_seq(m * k);
        let b_t = make_seq(n * k);
        let mut c = vec![0.0f32; m * n];
        gemm_transb_f32(&a, &b_t, &mut c, m, n, k);
        let expected = ref_gemm_transb(&a, &b_t, m, n, k);
        approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_gemm_transb_identity() {
        let n = 4;
        let mut id = vec![0.0f32; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        let a = make_seq(n * n);
        let mut c = vec![0.0f32; n * n];
        // B_T = I^T = I, so C = A * I = A
        gemm_transb_f32(&a, &id, &mut c, n, n, n);
        approx_eq(&c, &a, TOL);
    }

    #[test]
    fn test_gemm_transb_vs_gemm() {
        // gemm_transb(A, B^T) should equal gemm(A, B)
        let (m, n, k) = (4, 4, 8);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        // Transpose B into B_T
        let mut b_t = vec![0.0f32; n * k];
        for i in 0..k {
            for j in 0..n {
                b_t[j * k + i] = b[i * n + j];
            }
        }
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c1, m, n, k);
        gemm_transb_f32(&a, &b_t, &mut c2, m, n, k);
        approx_eq(&c1, &c2, TOL);
    }

    // ── Batched GEMM tests ─────────────────────────────────────

    #[test]
    fn test_batched_gemm_batch1() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        batched_gemm_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batched_gemm_batch4() {
        let batch = 4;
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * k * n);
        let mut c = vec![0.0f32; batch * m * n];
        batched_gemm_f32(&a, &b, &mut c, batch, m, n, k);

        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let expected = ref_gemm(a_s, b_s, m, n, k);
            let c_s = &c[bi * m * n..(bi + 1) * m * n];
            approx_eq(c_s, &expected, TOL);
        }
    }

    #[test]
    fn test_batched_gemm_batch16() {
        let batch = 16;
        let (m, n, k) = (2, 2, 2);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * k * n);
        let mut c = vec![0.0f32; batch * m * n];
        batched_gemm_f32(&a, &b, &mut c, batch, m, n, k);

        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let expected = ref_gemm(a_s, b_s, m, n, k);
            let c_s = &c[bi * m * n..(bi + 1) * m * n];
            approx_eq(c_s, &expected, TOL);
        }
    }

    #[test]
    fn test_batched_gemm_non_square() {
        let batch = 3;
        let (m, n, k) = (3, 5, 7);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * k * n);
        let mut c = vec![0.0f32; batch * m * n];
        batched_gemm_f32(&a, &b, &mut c, batch, m, n, k);

        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let expected = ref_gemm(a_s, b_s, m, n, k);
            let c_s = &c[bi * m * n..(bi + 1) * m * n];
            approx_eq(c_s, &expected, TOL);
        }
    }

    // ── Fused GEMM + Bias + ReLU tests ─────────────────────────

    #[test]
    fn test_fused_bias_relu_basic() {
        let a = [1.0, 0.0, 0.0, 1.0f32]; // 2×2 identity
        let b = [1.0, -2.0, 3.0, -4.0f32];
        let bias = [10.0, 10.0f32];
        let mut c = [0.0f32; 4];
        fused_gemm_bias_relu_f32(&a, &b, &bias, &mut c, 2, 2, 2);
        // C = I * B + bias, then ReLU
        // row0: [1+10, -2+10] = [11, 8]  → [11, 8]
        // row1: [3+10, -4+10] = [13, 6]  → [13, 6]
        approx_eq(&c, &[11.0, 8.0, 13.0, 6.0], TOL);
    }

    #[test]
    fn test_fused_relu_clamps_negative() {
        let a = [1.0f32];
        let b = [-5.0f32];
        let bias = [2.0f32];
        let mut c = [0.0f32];
        fused_gemm_bias_relu_f32(&a, &b, &bias, &mut c, 1, 1, 1);
        // -5 + 2 = -3 → ReLU → 0
        approx_eq(&c, &[0.0], TOL);
    }

    #[test]
    fn test_fused_zero_bias() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let bias = vec![0.0f32; n];
        let mut c_fused = vec![0.0f32; m * n];
        fused_gemm_bias_relu_f32(&a, &b, &bias, &mut c_fused, m, n, k);

        // Same as gemm + max(0,.)
        let mut c_ref = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c_ref, m, n, k);
        for v in c_ref.iter_mut() {
            *v = v.max(0.0);
        }
        approx_eq(&c_fused, &c_ref, TOL);
    }

    #[test]
    fn test_fused_large_bias() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b: Vec<f32> = (0..k * n).map(|i| -(i as f32) * 0.1).collect();
        let bias = vec![100.0f32; n]; // large positive bias
        let mut c = vec![0.0f32; m * n];
        fused_gemm_bias_relu_f32(&a, &b, &bias, &mut c, m, n, k);
        // All outputs should be positive due to large bias
        for v in &c {
            assert!(*v >= 0.0, "Expected non-negative, got {v}");
        }
    }

    #[test]
    fn test_fused_non_square() {
        let (m, n, k) = (3, 5, 7);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut c = vec![0.0f32; m * n];
        fused_gemm_bias_relu_f32(&a, &b, &bias, &mut c, m, n, k);

        // Reference
        let mut c_ref = ref_gemm(&a, &b, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let v = c_ref[i * n + j] + bias[j];
                c_ref[i * n + j] = v.max(0.0);
            }
        }
        approx_eq(&c, &c_ref, TOL);
    }

    #[test]
    fn test_fused_all_negative_gemm() {
        let a = [-1.0f32; 4];
        let b = [1.0f32; 4];
        let bias = [0.0f32; 2];
        let mut c = [99.0f32; 4];
        fused_gemm_bias_relu_f32(&a, &b, &bias, &mut c, 2, 2, 2);
        // A*B produces all -2.0, bias=0 → ReLU → all 0
        approx_eq(&c, &[0.0, 0.0, 0.0, 0.0], TOL);
    }

    #[test]
    fn test_fused_8x8() {
        let (m, n, k) = (8, 8, 8);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let bias = make_seq(n);
        let mut c = vec![0.0f32; m * n];
        fused_gemm_bias_relu_f32(&a, &b, &bias, &mut c, m, n, k);
        let mut c_ref = ref_gemm(&a, &b, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let v = c_ref[i * n + j] + bias[j];
                c_ref[i * n + j] = v.max(0.0);
            }
        }
        approx_eq(&c, &c_ref, TOL);
    }

    // ── Quantized GEMV I2 tests ────────────────────────────────

    /// Pack a slice of ternary values (-1,0,+1) as I2_S bytes (4 per byte).
    fn pack_i2(vals: &[f32]) -> Vec<u8> {
        let bytes = (vals.len() + 3) / 4;
        let mut out = vec![0u8; bytes];
        for (i, &v) in vals.iter().enumerate() {
            let code: u8 = if v == 1.0 {
                0b01
            } else if v == -1.0 {
                0b11
            } else {
                0b00
            };
            let byte_idx = i / 4;
            let bit_off = (i % 4) * 2;
            out[byte_idx] |= code << bit_off;
        }
        out
    }

    #[test]
    fn test_quantized_gemv_all_plus_one() {
        let (m, n) = (1, 4);
        let weights_f = vec![1.0f32; m * n];
        let packed = pack_i2(&weights_f);
        let scales = vec![1.0f32; m];
        let x = vec![1.0, 2.0, 3.0, 4.0f32];
        let mut y = vec![0.0f32; m];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        approx_eq(&y, &[10.0], TOL); // 1+2+3+4
    }

    #[test]
    fn test_quantized_gemv_all_minus_one() {
        let (m, n) = (1, 4);
        let weights_f = vec![-1.0f32; m * n];
        let packed = pack_i2(&weights_f);
        let scales = vec![1.0f32; m];
        let x = vec![1.0, 2.0, 3.0, 4.0f32];
        let mut y = vec![0.0f32; m];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        approx_eq(&y, &[-10.0], TOL);
    }

    #[test]
    fn test_quantized_gemv_all_zero() {
        let (m, n) = (2, 4);
        let weights_f = vec![0.0f32; m * n];
        let packed = pack_i2(&weights_f);
        let scales = vec![1.0f32; m];
        let x = vec![5.0, 6.0, 7.0, 8.0f32];
        let mut y = vec![99.0f32; m];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        approx_eq(&y, &[0.0, 0.0], TOL);
    }

    #[test]
    fn test_quantized_gemv_mixed() {
        // w = [+1, -1, 0, +1], x = [1, 2, 3, 4]
        // dot = 1*1 + (-1)*2 + 0*3 + 1*4 = 3
        let (m, n) = (1, 4);
        let weights_f = [1.0, -1.0, 0.0, 1.0f32];
        let packed = pack_i2(&weights_f);
        let scales = vec![1.0f32];
        let x = vec![1.0, 2.0, 3.0, 4.0f32];
        let mut y = vec![0.0f32];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        approx_eq(&y, &[3.0], TOL);
    }

    #[test]
    fn test_quantized_gemv_scale() {
        let (m, n) = (1, 4);
        let weights_f = vec![1.0f32; m * n];
        let packed = pack_i2(&weights_f);
        let scales = vec![0.5f32];
        let x = vec![1.0, 1.0, 1.0, 1.0f32];
        let mut y = vec![0.0f32];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        // dot=4 * scale=0.5 = 2.0
        approx_eq(&y, &[2.0], TOL);
    }

    #[test]
    fn test_quantized_gemv_multi_row() {
        let (m, n) = (3, 4);
        let weights_f = [
            1.0, 1.0, 1.0, 1.0, // row 0: all +1
            -1.0, -1.0, -1.0, -1.0, // row 1: all -1
            1.0, -1.0, 1.0, -1.0f32, // row 2: alternating
        ];
        let packed = pack_i2(&weights_f);
        let scales = vec![1.0, 2.0, 0.5f32];
        let x = vec![1.0, 2.0, 3.0, 4.0f32];
        let mut y = vec![0.0f32; m];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        // row 0: (1+2+3+4)*1 = 10
        // row 1: (-1-2-3-4)*2 = -20
        // row 2: (1-2+3-4)*0.5 = -1.0
        approx_eq(&y, &[10.0, -20.0, -1.0], TOL);
    }

    #[test]
    fn test_quantized_gemv_non_multiple_of_4() {
        let (m, n) = (1, 5);
        let weights_f = [1.0, -1.0, 1.0, -1.0, 1.0f32];
        let packed = pack_i2(&weights_f);
        let scales = vec![1.0f32];
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0f32];
        let mut y = vec![0.0f32];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        // 1 - 1 + 1 - 1 + 1 = 1
        approx_eq(&y, &[1.0], TOL);
    }

    #[test]
    fn test_quantized_gemv_n1() {
        let (m, n) = (2, 1);
        // Each row has 1 element → 1 byte per row (padded to 4 values/byte)
        // Row 0: weight=+1, Row 1: weight=-1
        let weights_f_row0 = [1.0f32];
        let weights_f_row1 = [-1.0f32];
        let mut packed = pack_i2(&weights_f_row0);
        packed.extend(pack_i2(&weights_f_row1));
        let scales = vec![3.0, 3.0f32];
        let x = vec![7.0f32];
        let mut y = vec![0.0f32; 2];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        approx_eq(&y, &[21.0, -21.0], TOL);
    }

    #[test]
    fn test_quantized_gemv_8_cols() {
        let (m, n) = (1, 8);
        let weights_f = vec![1.0f32; 8];
        let packed = pack_i2(&weights_f);
        let scales = vec![1.0f32];
        let x: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let mut y = vec![0.0f32];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        approx_eq(&y, &[36.0], TOL); // 1+2+...+8 = 36
    }

    // ── Associativity property ─────────────────────────────────

    #[test]
    fn test_gemm_associativity() {
        // (A * B) * C ≈ A * (B * C)
        let (p, q, r, s) = (4, 4, 4, 4);
        let a = make_seq(p * q);
        let b = make_seq(q * r);
        let c_mat = make_seq(r * s);

        // (A*B)*C
        let mut ab = vec![0.0f32; p * r];
        gemm_f32(&a, &b, &mut ab, p, r, q);
        let mut abc = vec![0.0f32; p * s];
        gemm_f32(&ab, &c_mat, &mut abc, p, s, r);

        // A*(B*C)
        let mut bc = vec![0.0f32; q * s];
        gemm_f32(&b, &c_mat, &mut bc, q, s, r);
        let mut a_bc = vec![0.0f32; p * s];
        gemm_f32(&a, &bc, &mut a_bc, p, s, q);

        approx_eq(&abc, &a_bc, 1e-2);
    }

    // ── Performance scaling ────────────────────────────────────

    #[test]
    fn test_gemm_scaling_small() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_scaling_medium_32() {
        let (m, n, k) = (32, 32, 32);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, 0.2);
    }

    #[test]
    fn test_gemv_scaling_medium_32() {
        let (m, n) = (32, 32);
        let a = make_seq(m * n);
        let x = make_seq(n);
        let mut y = vec![0.0f32; m];
        gemv_f32(&a, &x, &mut y, m, n);
        let expected = ref_gemv(&a, &x, m, n);
        approx_eq(&y, &expected, 1e-2);
    }

    // ── Additional edge cases ──────────────────────────────────

    #[test]
    fn test_gemm_k1() {
        // k=1: outer product
        let (m, n, k) = (3, 4, 1);
        let a = vec![1.0, 2.0, 3.0f32];
        let b = vec![4.0, 5.0, 6.0, 7.0f32];
        let mut c = vec![0.0f32; m * n];
        gemm_f32(&a, &b, &mut c, m, n, k);
        let expected = ref_gemm(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_gemm_transb_k1() {
        let (m, n, k) = (3, 4, 1);
        let a = vec![1.0, 2.0, 3.0f32];
        let b_t = vec![4.0, 5.0, 6.0, 7.0f32]; // each "row" of b_t is length k=1
        let mut c = vec![0.0f32; m * n];
        gemm_transb_f32(&a, &b_t, &mut c, m, n, k);
        let expected = ref_gemm_transb(&a, &b_t, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batched_gemm_1x1() {
        let batch = 4;
        let a = vec![2.0, 3.0, 4.0, 5.0f32];
        let b = vec![10.0, 20.0, 30.0, 40.0f32];
        let mut c = vec![0.0f32; batch];
        batched_gemm_f32(&a, &b, &mut c, batch, 1, 1, 1);
        approx_eq(&c, &[20.0, 60.0, 120.0, 200.0], TOL);
    }

    #[test]
    fn test_fused_1x1() {
        let a = [2.0f32];
        let b = [3.0f32];
        let bias = [1.0f32];
        let mut c = [0.0f32];
        fused_gemm_bias_relu_f32(&a, &b, &bias, &mut c, 1, 1, 1);
        // 2*3 + 1 = 7 → ReLU → 7
        approx_eq(&c, &[7.0], TOL);
    }

    #[test]
    fn test_quantized_gemv_1x1() {
        let weights_f = [1.0f32];
        let packed = pack_i2(&weights_f);
        let scales = [2.0f32];
        let x = [5.0f32];
        let mut y = [0.0f32];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, 1, 1);
        approx_eq(&y, &[10.0], TOL);
    }

    #[test]
    fn test_gemv_negative_values() {
        let a = [-1.0, -2.0, -3.0, -4.0f32];
        let x = [1.0, 1.0f32];
        let mut y = [0.0f32; 2];
        gemv_f32(&a, &x, &mut y, 2, 2);
        approx_eq(&y, &[-3.0, -7.0], TOL);
    }

    #[test]
    fn test_gemm_negative_values() {
        let a = [-1.0, -2.0, -3.0, -4.0f32];
        let b = [1.0, 0.0, 0.0, 1.0f32]; // identity
        let mut c = [0.0f32; 4];
        gemm_f32(&a, &b, &mut c, 2, 2, 2);
        approx_eq(&c, &[-1.0, -2.0, -3.0, -4.0], TOL);
    }

    #[test]
    fn test_gemm_transb_zeros() {
        let (m, n, k) = (3, 3, 3);
        let a = make_seq(m * k);
        let b_t = vec![0.0f32; n * k];
        let mut c = vec![99.0f32; m * n];
        gemm_transb_f32(&a, &b_t, &mut c, m, n, k);
        approx_eq(&c, &vec![0.0; m * n], TOL);
    }

    #[test]
    fn test_quantized_gemv_large_scale() {
        let (m, n) = (1, 4);
        let weights_f = vec![1.0f32; 4];
        let packed = pack_i2(&weights_f);
        let scales = vec![100.0f32];
        let x = vec![1.0, 1.0, 1.0, 1.0f32];
        let mut y = vec![0.0f32];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        approx_eq(&y, &[400.0], TOL);
    }

    #[test]
    fn test_quantized_gemv_negative_scale() {
        let (m, n) = (1, 4);
        let weights_f = vec![1.0f32; 4];
        let packed = pack_i2(&weights_f);
        let scales = vec![-2.0f32];
        let x = vec![1.0, 1.0, 1.0, 1.0f32];
        let mut y = vec![0.0f32];
        quantized_gemv_i2_f32(&packed, &scales, &x, &mut y, m, n);
        approx_eq(&y, &[-8.0], TOL);
    }
}
