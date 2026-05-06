#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! NEON-optimized tiled matrix multiplication for Apple Silicon.
//!
//! Provides cache-friendly tiled matmul, a fixed 4×4 micro-kernel,
//! matrix-vector multiply (GEMV), and rank-1 outer-product update,
//! all accelerated with ARM NEON FMA intrinsics.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Tiled matrix multiplication with configurable tile size for cache locality.
///
/// Computes C = A × B where A is m×k and B is k×n (row-major).
#[cfg(target_arch = "aarch64")]
pub fn neon_matmul_tiled(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    tile_size: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "a length must be m*k");
    assert_eq!(b.len(), k * n, "b length must be k*n");
    assert!(tile_size > 0, "tile_size must be positive");

    let mut c = vec![0.0f32; m * n];

    for ii in (0..m).step_by(tile_size) {
        let i_end = (ii + tile_size).min(m);
        for jj in (0..n).step_by(tile_size) {
            let j_end = (jj + tile_size).min(n);
            for kk in (0..k).step_by(tile_size) {
                let k_end = (kk + tile_size).min(k);
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = c[i * n + j];
                        let mut p = kk;
                        // NEON: accumulate 4 elements at a time with FMA
                        unsafe {
                            let mut acc = vdupq_n_f32(0.0);
                            while p + 4 <= k_end {
                                let va = vld1q_f32(a.as_ptr().add(i * k + p));
                                let vb = vld1q_f32(b.as_ptr().add(j + p * n));
                                // b is row-major so elements are strided; gather manually
                                let vb_gathered = vld1q_f32(
                                    [
                                        b[p * n + j],
                                        b[(p + 1) * n + j],
                                        b[(p + 2) * n + j],
                                        b[(p + 3) * n + j],
                                    ]
                                    .as_ptr(),
                                );
                                acc = vfmaq_f32(acc, va, vb_gathered);
                                p += 4;
                            }
                            // Horizontal sum of accumulator
                            sum += vaddvq_f32(acc);
                        }
                        // Scalar tail
                        for pp in p..k_end {
                            sum += a[i * k + pp] * b[pp * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }

    c
}

/// Fixed 4×4 tile micro-kernel optimized for NEON register count.
///
/// Computes C = A × B where A is m×k and B is k×n (row-major).
/// Uses a 4×4 register tile to maximise NEON utilisation.
#[cfg(target_arch = "aarch64")]
pub fn neon_matmul_tiled_4x4(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "a length must be m*k");
    assert_eq!(b.len(), k * n, "b length must be k*n");

    let mut c = vec![0.0f32; m * n];

    // Process 4×4 blocks
    let m4 = m - m % 4;
    let n4 = n - n % 4;

    for i in (0..m4).step_by(4) {
        for j in (0..n4).step_by(4) {
            unsafe {
                let mut c00 = vdupq_n_f32(0.0);
                let mut c01 = vdupq_n_f32(0.0);
                let mut c02 = vdupq_n_f32(0.0);
                let mut c03 = vdupq_n_f32(0.0);

                for p in 0..k {
                    let b_row = vld1q_f32(b.as_ptr().add(p * n + j));
                    // Broadcast each a element across a NEON lane
                    let a0 = vdupq_n_f32(a[i * k + p]);
                    let a1 = vdupq_n_f32(a[(i + 1) * k + p]);
                    let a2 = vdupq_n_f32(a[(i + 2) * k + p]);
                    let a3 = vdupq_n_f32(a[(i + 3) * k + p]);

                    c00 = vfmaq_f32(c00, a0, b_row);
                    c01 = vfmaq_f32(c01, a1, b_row);
                    c02 = vfmaq_f32(c02, a2, b_row);
                    c03 = vfmaq_f32(c03, a3, b_row);
                }

                vst1q_f32(c.as_mut_ptr().add(i * n + j), c00);
                vst1q_f32(c.as_mut_ptr().add((i + 1) * n + j), c01);
                vst1q_f32(c.as_mut_ptr().add((i + 2) * n + j), c02);
                vst1q_f32(c.as_mut_ptr().add((i + 3) * n + j), c03);
            }
        }

        // Right-edge columns that didn't fit in a 4-wide block
        for j in n4..n {
            for di in 0..4 {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[(i + di) * k + p] * b[p * n + j];
                }
                c[(i + di) * n + j] = sum;
            }
        }
    }

    // Bottom-edge rows that didn't fit in a 4-tall block
    for i in m4..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    c
}

/// Matrix-vector multiply: y = A × x where A is m×k (row-major).
///
/// Common in inference for single-token generation where the "batch" dimension
/// collapses to 1.
#[cfg(target_arch = "aarch64")]
pub fn neon_gemv(a: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "a length must be m*k");
    assert_eq!(x.len(), k, "x length must be k");

    let mut y = vec![0.0f32; m];
    let k4 = k - k % 4;

    for (i, y_val) in y.iter_mut().enumerate().take(m) {
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            let row_base = i * k;
            let mut p = 0;
            while p < k4 {
                let va = vld1q_f32(a.as_ptr().add(row_base + p));
                let vx = vld1q_f32(x.as_ptr().add(p));
                acc = vfmaq_f32(acc, va, vx);
                p += 4;
            }
            let mut sum = vaddvq_f32(acc);
            // Scalar tail
            for p in k4..k {
                sum += a[row_base + p] * x[p];
            }
            *y_val = sum;
        }
    }

    y
}

/// Rank-1 update: C += a × bᵀ where a is length m, b is length n, C is m×n (row-major).
///
/// Useful for low-rank approximation updates and outer-product attention patterns.
#[cfg(target_arch = "aarch64")]
pub fn neon_outer_product_update(c: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    assert_eq!(c.len(), m * n, "c length must be m*n");
    assert_eq!(a.len(), m, "a length must be m");
    assert_eq!(b.len(), n, "b length must be n");

    let n4 = n - n % 4;

    for (i, &a_val) in a.iter().enumerate().take(m) {
        unsafe {
            let va = vdupq_n_f32(a_val);
            let row_base = i * n;
            let mut j = 0;
            while j < n4 {
                let vc = vld1q_f32(c.as_ptr().add(row_base + j));
                let vb = vld1q_f32(b.as_ptr().add(j));
                let result = vfmaq_f32(vc, va, vb);
                vst1q_f32(c.as_mut_ptr().add(row_base + j), result);
                j += 4;
            }
            // Scalar tail
            for j in n4..n {
                c[row_base + j] += a_val * b[j];
            }
        }
    }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    /// Naive O(n³) matmul for reference.
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < eps,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    #[test]
    fn test_tiled_vs_naive() {
        let (m, n, k) = (17, 13, 11);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

        let expected = naive_matmul(&a, &b, m, n, k);
        let result = neon_matmul_tiled(&a, &b, m, n, k, 4);
        approx_eq(&result, &expected, 1e-3);

        // Also test with a larger tile size
        let result_large = neon_matmul_tiled(&a, &b, m, n, k, 8);
        approx_eq(&result_large, &expected, 1e-3);
    }

    #[test]
    fn test_4x4_micro_kernel() {
        let (m, n, k) = (16, 12, 10);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

        let expected = naive_matmul(&a, &b, m, n, k);
        let result = neon_matmul_tiled_4x4(&a, &b, m, n, k);
        approx_eq(&result, &expected, 1e-3);

        // Non-aligned dimensions to exercise edge handling
        let (m2, n2, k2) = (7, 5, 9);
        let a2: Vec<f32> = (0..m2 * k2).map(|i| (i as f32) * 0.01).collect();
        let b2: Vec<f32> = (0..k2 * n2).map(|i| (i as f32) * 0.01).collect();

        let expected2 = naive_matmul(&a2, &b2, m2, n2, k2);
        let result2 = neon_matmul_tiled_4x4(&a2, &b2, m2, n2, k2);
        approx_eq(&result2, &expected2, 1e-3);
    }

    #[test]
    fn test_gemv_basic() {
        let (m, k) = (6, 10);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

        // Reference: treat as matmul with n=1
        let b_col: Vec<f32> = x.clone();
        let expected = naive_matmul(&a, &b_col, m, 1, k);

        let result = neon_gemv(&a, &x, m, k);
        approx_eq(&result, &expected, 1e-3);
    }

    #[test]
    fn test_outer_product() {
        let (m, n) = (5, 7);
        let a: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0f32; m * n];

        neon_outer_product_update(&mut c, &a, &b, m, n);

        // Verify C[i][j] == a[i] * b[j]
        for i in 0..m {
            for j in 0..n {
                let expected = a[i] * b[j];
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-6,
                    "mismatch at ({i},{j}): {} vs {expected}",
                    c[i * n + j]
                );
            }
        }

        // Apply again to verify accumulation: C should now be 2 * a * b^T
        neon_outer_product_update(&mut c, &a, &b, m, n);
        for i in 0..m {
            for j in 0..n {
                let expected = 2.0 * a[i] * b[j];
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-6,
                    "accumulation mismatch at ({i},{j}): {} vs {expected}",
                    c[i * n + j]
                );
            }
        }
    }
}
