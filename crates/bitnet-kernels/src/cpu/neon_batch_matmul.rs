#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON optimized batch matrix multiplication for Apple Silicon (aarch64).
//!
//! Provides batched GEMM variants accelerated with `float32x4_t` NEON intrinsics
//! and scalar fallbacks for remainder columns:
//!
//! - `neon_batch_matmul_f32`          — C = A × B
//! - `neon_batch_matmul_transb_f32`   — C = A × Bᵀ  (transposed B, common in attention)
//! - `neon_batch_matmul_accumulate_f32` — C += A × B
//! - `neon_batch_matmul_scale_f32`    — C = α · (A × B)
//! - `neon_strided_batch_matmul_f32`  — explicit strides for non-contiguous data

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Scalar reference (used by all public functions as fallback) ─────────

/// Scalar C = A × B for a single matrix (row-major).
fn scalar_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
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

/// Scalar C = A × Bᵀ for a single matrix.
fn scalar_matmul_transb(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}

// ── NEON inner kernels ──────────────────────────────────────────────────

/// NEON-accelerated single matrix multiply C = A × B.
///
/// # Safety
/// Requires `neon` target feature at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matmul_inner(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let lanes = 4usize;
    let n_chunks = n / lanes;
    let n_rem = n % lanes;

    for i in 0..m {
        // Process 4 columns at a time
        for jc in 0..n_chunks {
            let j_base = jc * lanes;
            let mut acc = vdupq_n_f32(0.0);
            for p in 0..k {
                let a_val = vdupq_n_f32(a[i * k + p]);
                let b_vec = unsafe { vld1q_f32(b.as_ptr().add(p * n + j_base)) };
                acc = vfmaq_f32(acc, a_val, b_vec);
            }
            unsafe {
                vst1q_f32(c.as_mut_ptr().add(i * n + j_base), acc);
            }
        }
        // Scalar remainder
        for j in (n_chunks * lanes)..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    let _ = n_rem;
}

/// NEON-accelerated C = A × Bᵀ (B transposed).
///
/// # Safety
/// Requires `neon` target feature at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matmul_transb_inner(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let lanes = 4usize;
    let k_chunks = k / lanes;

    for i in 0..m {
        for j in 0..n {
            let mut acc = vdupq_n_f32(0.0);
            for pc in 0..k_chunks {
                let p_base = pc * lanes;
                let a_vec = unsafe { vld1q_f32(a.as_ptr().add(i * k + p_base)) };
                let b_vec = unsafe { vld1q_f32(b.as_ptr().add(j * k + p_base)) };
                acc = vfmaq_f32(acc, a_vec, b_vec);
            }
            let mut sum = vaddvq_f32(acc);
            // Scalar tail for k
            for p in (k_chunks * lanes)..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}

/// NEON-accelerated C += A × B (accumulate).
///
/// # Safety
/// Requires `neon` target feature at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matmul_accumulate_inner(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let lanes = 4usize;
    let n_chunks = n / lanes;

    for i in 0..m {
        for jc in 0..n_chunks {
            let j_base = jc * lanes;
            let mut acc = unsafe { vld1q_f32(c.as_ptr().add(i * n + j_base)) };
            for p in 0..k {
                let a_val = vdupq_n_f32(a[i * k + p]);
                let b_vec = unsafe { vld1q_f32(b.as_ptr().add(p * n + j_base)) };
                acc = vfmaq_f32(acc, a_val, b_vec);
            }
            unsafe {
                vst1q_f32(c.as_mut_ptr().add(i * n + j_base), acc);
            }
        }
        for j in (n_chunks * lanes)..n {
            let mut sum = c[i * n + j];
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// NEON-accelerated C = alpha * (A × B).
///
/// # Safety
/// Requires `neon` target feature at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matmul_scale_inner(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) {
    let lanes = 4usize;
    let n_chunks = n / lanes;
    let alpha_vec = vdupq_n_f32(alpha);

    for i in 0..m {
        for jc in 0..n_chunks {
            let j_base = jc * lanes;
            let mut acc = vdupq_n_f32(0.0);
            for p in 0..k {
                let a_val = vdupq_n_f32(a[i * k + p]);
                let b_vec = unsafe { vld1q_f32(b.as_ptr().add(p * n + j_base)) };
                acc = vfmaq_f32(acc, a_val, b_vec);
            }
            let scaled = vmulq_f32(acc, alpha_vec);
            unsafe {
                vst1q_f32(c.as_mut_ptr().add(i * n + j_base), scaled);
            }
        }
        for j in (n_chunks * lanes)..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum * alpha;
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Batched matrix multiply: `C[b] = A[b] × B[b]` for each batch element.
///
/// All matrices are row-major. For batch `b`:
/// - `A[b]` is `m × k` starting at `a[b * m * k]`
/// - `B[b]` is `k × n` starting at `b_mat[b * k * n]`
/// - `C[b]` is `m × n` starting at `c[b * m * n]`
pub fn neon_batch_matmul_f32(
    a: &[f32],
    b_mat: &[f32],
    c: &mut [f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        // Zero-fill output for zero-dimension cases
        for v in c.iter_mut().take(batch * m * n) {
            *v = 0.0;
        }
        return;
    }
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    for bi in 0..batch {
        let a_slice = &a[bi * a_stride..bi * a_stride + a_stride];
        let b_slice = &b_mat[bi * b_stride..bi * b_stride + b_stride];
        let c_slice = &mut c[bi * c_stride..bi * c_stride + c_stride];

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    neon_matmul_inner(a_slice, b_slice, c_slice, m, n, k);
                }
                continue;
            }
        }
        scalar_matmul(a_slice, b_slice, c_slice, m, n, k);
    }
}

/// Batched matrix multiply with transposed B: `C[b] = A[b] × B[b]ᵀ`.
///
/// B is stored as `n × k` (each row of B is a column of Bᵀ). This layout
/// is common in attention score computation where Q × Kᵀ is needed.
pub fn neon_batch_matmul_transb_f32(
    a: &[f32],
    b_mat: &[f32],
    c: &mut [f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        for v in c.iter_mut().take(batch * m * n) {
            *v = 0.0;
        }
        return;
    }
    let a_stride = m * k;
    let b_stride = n * k; // B is n×k (transposed layout)
    let c_stride = m * n;

    for bi in 0..batch {
        let a_slice = &a[bi * a_stride..bi * a_stride + a_stride];
        let b_slice = &b_mat[bi * b_stride..bi * b_stride + b_stride];
        let c_slice = &mut c[bi * c_stride..bi * c_stride + c_stride];

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    neon_matmul_transb_inner(a_slice, b_slice, c_slice, m, n, k);
                }
                continue;
            }
        }
        scalar_matmul_transb(a_slice, b_slice, c_slice, m, n, k);
    }
}

/// Batched accumulate: `C[b] += A[b] × B[b]` for each batch element.
///
/// Unlike `neon_batch_matmul_f32`, this adds to existing values in `c`.
pub fn neon_batch_matmul_accumulate_f32(
    a: &[f32],
    b_mat: &[f32],
    c: &mut [f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return;
    }
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    for bi in 0..batch {
        let a_slice = &a[bi * a_stride..bi * a_stride + a_stride];
        let b_slice = &b_mat[bi * b_stride..bi * b_stride + b_stride];
        let c_slice = &mut c[bi * c_stride..bi * c_stride + c_stride];

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    neon_matmul_accumulate_inner(a_slice, b_slice, c_slice, m, n, k);
                }
                continue;
            }
        }
        // Scalar accumulate fallback
        for i in 0..m {
            for j in 0..n {
                let mut sum = c_slice[i * n + j];
                for p in 0..k {
                    sum += a_slice[i * k + p] * b_slice[p * n + j];
                }
                c_slice[i * n + j] = sum;
            }
        }
    }
}

/// Batched scaled multiply: `C[b] = alpha × (A[b] × B[b])`.
pub fn neon_batch_matmul_scale_f32(
    a: &[f32],
    b_mat: &[f32],
    c: &mut [f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) {
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        for v in c.iter_mut().take(batch * m * n) {
            *v = 0.0;
        }
        return;
    }
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    for bi in 0..batch {
        let a_slice = &a[bi * a_stride..bi * a_stride + a_stride];
        let b_slice = &b_mat[bi * b_stride..bi * b_stride + b_stride];
        let c_slice = &mut c[bi * c_stride..bi * c_stride + c_stride];

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    neon_matmul_scale_inner(a_slice, b_slice, c_slice, m, n, k, alpha);
                }
                continue;
            }
        }
        // Scalar scaled fallback
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_slice[i * k + p] * b_slice[p * n + j];
                }
                c_slice[i * n + j] = sum * alpha;
            }
        }
    }
}

/// Strided batched matrix multiply with explicit strides for non-contiguous data.
///
/// `stride_a`, `stride_b`, `stride_c` are element counts between consecutive
/// batch slices. This supports views into larger tensors where batches are not
/// tightly packed.
#[allow(clippy::too_many_arguments)]
pub fn neon_strided_batch_matmul_f32(
    a: &[f32],
    b_mat: &[f32],
    c: &mut [f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    stride_a: usize,
    stride_b: usize,
    stride_c: usize,
) {
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        for bi in 0..batch {
            for idx in 0..m * n {
                if bi * stride_c + idx < c.len() {
                    c[bi * stride_c + idx] = 0.0;
                }
            }
        }
        return;
    }

    for bi in 0..batch {
        let a_off = bi * stride_a;
        let b_off = bi * stride_b;
        let c_off = bi * stride_c;

        let a_slice = &a[a_off..a_off + m * k];
        let b_slice = &b_mat[b_off..b_off + k * n];
        let c_slice = &mut c[c_off..c_off + m * n];

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    neon_matmul_inner(a_slice, b_slice, c_slice, m, n, k);
                }
                continue;
            }
        }
        scalar_matmul(a_slice, b_slice, c_slice, m, n, k);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    /// Scalar reference for C = A × B.
    fn ref_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        scalar_matmul(a, b, &mut c, m, n, k);
        c
    }

    /// Scalar reference for C = A × Bᵀ.
    fn ref_matmul_transb(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        scalar_matmul_transb(a, b, &mut c, m, n, k);
        c
    }

    fn make_seq(n: usize) -> Vec<f32> {
        (1..=n).map(|i| i as f32).collect()
    }

    fn make_ones(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    fn make_identity(n: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    // ════════════════════════════════════════════════════════════
    // neon_batch_matmul_f32
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_matmul_1x1_single() {
        let a = [3.0f32];
        let b = [5.0f32];
        let mut c = [0.0f32];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, 1, 1, 1);
        approx_eq(&c, &[15.0], TOL);
    }

    #[test]
    fn test_batch_matmul_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, 2, 2, 2);
        let expected = ref_matmul(&a, &b, 2, 2, 2);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_4x4() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_8x8() {
        let (m, n, k) = (8, 8, 8);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_16x16() {
        let (m, n, k) = (16, 16, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_nonsquare_3x5x7() {
        let (m, n, k) = (3, 5, 7);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_nonsquare_5x3x7() {
        let (m, n, k) = (5, 3, 7);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_nonsquare_7x3x5() {
        let (m, n, k) = (7, 3, 5);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_batch2() {
        let (batch, m, n, k) = (2, 3, 3, 3);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * k * n);
        let mut c = vec![0.0f32; batch * m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, batch, m, n, k);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let expected = ref_matmul(a_s, b_s, m, n, k);
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_batch_matmul_batch4() {
        let (batch, m, n, k) = (4, 4, 4, 4);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; batch * m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, batch, m, n, k);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let expected = ref_matmul(a_s, b_s, m, n, k);
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_batch_matmul_batch8() {
        let (batch, m, n, k) = (8, 2, 2, 2);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.05).collect();
        let mut c = vec![0.0f32; batch * m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, batch, m, n, k);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let expected = ref_matmul(a_s, b_s, m, n, k);
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_batch_matmul_identity() {
        let n = 4;
        let a = make_seq(n * n);
        let id = make_identity(n);
        let mut c = vec![0.0f32; n * n];
        neon_batch_matmul_f32(&a, &id, &mut c, 1, n, n, n);
        approx_eq(&c, &a, TOL);
    }

    #[test]
    fn test_batch_matmul_identity_8() {
        let n = 8;
        let a = make_seq(n * n);
        let id = make_identity(n);
        let mut c = vec![0.0f32; n * n];
        neon_batch_matmul_f32(&a, &id, &mut c, 1, n, n, n);
        approx_eq(&c, &a, TOL);
    }

    #[test]
    fn test_batch_matmul_zeros_a() {
        let (m, n, k) = (3, 4, 5);
        let a = vec![0.0f32; m * k];
        let b = make_seq(k * n);
        let mut c = vec![999.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        approx_eq(&c, &vec![0.0f32; m * n], TOL);
    }

    #[test]
    fn test_batch_matmul_zeros_b() {
        let (m, n, k) = (3, 4, 5);
        let a = make_seq(m * k);
        let b = vec![0.0f32; k * n];
        let mut c = vec![999.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        approx_eq(&c, &vec![0.0f32; m * n], TOL);
    }

    #[test]
    fn test_batch_matmul_ones() {
        let (m, n, k) = (3, 4, 5);
        let a = make_ones(m * k);
        let b = make_ones(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        // Each element should be k (dot product of k ones)
        approx_eq(&c, &vec![k as f32; m * n], TOL);
    }

    #[test]
    fn test_batch_matmul_empty_batch() {
        let mut c = [999.0f32; 4];
        neon_batch_matmul_f32(&[], &[], &mut c, 0, 2, 2, 2);
        // Output unchanged (batch=0, no output to write)
    }

    #[test]
    fn test_batch_matmul_zero_m() {
        let mut c = [999.0f32; 0];
        neon_batch_matmul_f32(&[], &[1.0, 2.0], &mut c, 1, 0, 2, 1);
    }

    #[test]
    fn test_batch_matmul_zero_n() {
        let mut c = [999.0f32; 0];
        neon_batch_matmul_f32(&[1.0, 2.0], &[], &mut c, 1, 2, 0, 1);
    }

    #[test]
    fn test_batch_matmul_zero_k() {
        let mut c = [0.0f32; 4];
        neon_batch_matmul_f32(&[], &[], &mut c, 1, 2, 2, 0);
        approx_eq(&c, &[0.0; 4], TOL);
    }

    #[test]
    fn test_batch_matmul_n_not_multiple_of_4() {
        let (m, n, k) = (3, 6, 5);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_n_1() {
        // Matrix-vector multiply: A[m×k] × B[k×1]
        let (m, n, k) = (4, 1, 8);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_n_3() {
        let (m, n, k) = (4, 3, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_n_5() {
        let (m, n, k) = (4, 5, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_batch_matmul_negative_values() {
        let a = [-1.0, 2.0, -3.0, 4.0f32];
        let b = [5.0, -6.0, -7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, 2, 2, 2);
        let expected = ref_matmul(&a, &b, 2, 2, 2);
        approx_eq(&c, &expected, TOL);
    }

    // ════════════════════════════════════════════════════════════
    // neon_batch_matmul_transb_f32
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_transb_1x1() {
        let a = [3.0f32];
        let b = [5.0f32];
        let mut c = [0.0f32];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, 1, 1, 1);
        approx_eq(&c, &[15.0], TOL);
    }

    #[test]
    fn test_transb_2x2() {
        // A: 2×2, B (transposed layout): 2×2, C: 2×2
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32]; // B stored as n×k
        let mut c = [0.0f32; 4];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, 2, 2, 2);
        let expected = ref_matmul_transb(&a, &b, 2, 2, 2);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_transb_4x4() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(n * k);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul_transb(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_transb_8x8() {
        let (m, n, k) = (8, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul_transb(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_transb_16x16() {
        let (m, n, k) = (16, 16, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul_transb(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_transb_nonsquare_3x5x7() {
        let (m, n, k) = (3, 5, 7);
        let a = make_seq(m * k);
        let b = make_seq(n * k);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul_transb(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_transb_batch2() {
        let (batch, m, n, k) = (2, 3, 4, 5);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * n * k);
        let mut c = vec![0.0f32; batch * m * n];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, batch, m, n, k);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * n * k..(bi + 1) * n * k];
            let expected = ref_matmul_transb(a_s, b_s, m, n, k);
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_transb_batch4() {
        let (batch, m, n, k) = (4, 2, 3, 4);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..batch * n * k).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; batch * m * n];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, batch, m, n, k);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * n * k..(bi + 1) * n * k];
            let expected = ref_matmul_transb(a_s, b_s, m, n, k);
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_transb_batch8() {
        let (batch, m, n, k) = (8, 2, 2, 4);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..batch * n * k).map(|i| (i as f32) * 0.05).collect();
        let mut c = vec![0.0f32; batch * m * n];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, batch, m, n, k);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * n * k..(bi + 1) * n * k];
            let expected = ref_matmul_transb(a_s, b_s, m, n, k);
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_transb_identity_is_transpose() {
        // A × Iᵀ = A × I = A (identity is its own transpose)
        let n = 4;
        let a = make_seq(n * n);
        let id = make_identity(n);
        let mut c = vec![0.0f32; n * n];
        neon_batch_matmul_transb_f32(&a, &id, &mut c, 1, n, n, n);
        approx_eq(&c, &a, TOL);
    }

    #[test]
    fn test_transb_vs_explicit_transpose() {
        // Verify: A × Bᵀ(transb) == A × transpose(B)(normal matmul)
        let (m, n, k) = (3, 4, 5);
        let a = make_seq(m * k);
        let b_nt = make_seq(n * k); // B in n×k layout (transposed storage)

        // Explicitly transpose b_nt to k×n for normal matmul
        let mut b_normal = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                b_normal[j * n + i] = b_nt[i * k + j];
            }
        }

        let mut c_transb = vec![0.0f32; m * n];
        let mut c_normal = vec![0.0f32; m * n];
        neon_batch_matmul_transb_f32(&a, &b_nt, &mut c_transb, 1, m, n, k);
        neon_batch_matmul_f32(&a, &b_normal, &mut c_normal, 1, m, n, k);
        approx_eq(&c_transb, &c_normal, TOL);
    }

    #[test]
    fn test_transb_empty() {
        let mut c = [0.0f32; 0];
        neon_batch_matmul_transb_f32(&[], &[], &mut c, 0, 2, 2, 2);
    }

    #[test]
    fn test_transb_k_not_multiple_of_4() {
        let (m, n, k) = (3, 3, 5);
        let a = make_seq(m * k);
        let b = make_seq(n * k);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul_transb(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_transb_k_1() {
        let (m, n, k) = (4, 3, 1);
        let a = make_seq(m * k);
        let b = make_seq(n * k);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul_transb(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    // ════════════════════════════════════════════════════════════
    // neon_batch_matmul_accumulate_f32
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_accumulate_1x1() {
        let a = [3.0f32];
        let b = [5.0f32];
        let mut c = [10.0f32];
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, 1, 1, 1);
        approx_eq(&c, &[25.0], TOL); // 10 + 15
    }

    #[test]
    fn test_accumulate_2x2() {
        let a = [1.0, 0.0, 0.0, 1.0f32]; // identity
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [1.0, 2.0, 3.0, 4.0f32];
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, 2, 2, 2);
        // c += I × B = c + B
        approx_eq(&c, &[6.0, 8.0, 10.0, 12.0], TOL);
    }

    #[test]
    fn test_accumulate_4x4() {
        let (m, n, k) = (4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let initial = vec![1.0f32; m * n];
        let mut c = initial.clone();
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, m, n, k);
        let product = ref_matmul(&a, &b, m, n, k);
        let expected: Vec<f32> = initial.iter().zip(product.iter()).map(|(i, p)| i + p).collect();
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_accumulate_8x8() {
        let (m, n, k) = (8, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let initial = vec![0.5f32; m * n];
        let mut c = initial.clone();
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, m, n, k);
        let product = ref_matmul(&a, &b, m, n, k);
        let expected: Vec<f32> = initial.iter().zip(product.iter()).map(|(i, p)| i + p).collect();
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_accumulate_batch2() {
        let (batch, m, n, k) = (2, 3, 3, 3);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * k * n);
        let mut c = vec![1.0f32; batch * m * n];
        let initial = c.clone();
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, batch, m, n, k);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let product = ref_matmul(a_s, b_s, m, n, k);
            let expected: Vec<f32> = initial[bi * m * n..(bi + 1) * m * n]
                .iter()
                .zip(product.iter())
                .map(|(i, p)| i + p)
                .collect();
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_accumulate_batch4() {
        let (batch, m, n, k) = (4, 2, 2, 2);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![2.0f32; batch * m * n];
        let initial = c.clone();
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, batch, m, n, k);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let product = ref_matmul(a_s, b_s, m, n, k);
            let expected: Vec<f32> = initial[bi * m * n..(bi + 1) * m * n]
                .iter()
                .zip(product.iter())
                .map(|(i, p)| i + p)
                .collect();
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_accumulate_zeros_initial() {
        let (m, n, k) = (3, 4, 5);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_accumulate_empty() {
        let mut c = [5.0f32; 4];
        neon_batch_matmul_accumulate_f32(&[], &[], &mut c, 0, 2, 2, 2);
        // batch=0 → nothing happens
        approx_eq(&c, &[5.0; 4], TOL);
    }

    #[test]
    fn test_accumulate_nonsquare() {
        let (m, n, k) = (3, 5, 7);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.5f32; m * n];
        let initial = c.clone();
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, m, n, k);
        let product = ref_matmul(&a, &b, m, n, k);
        let expected: Vec<f32> = initial.iter().zip(product.iter()).map(|(i, p)| i + p).collect();
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_accumulate_double_call() {
        // Two accumulations: c = 0 + AB + AB = 2*AB
        let (m, n, k) = (3, 3, 3);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, m, n, k);
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, m, n, k);
        let product = ref_matmul(&a, &b, m, n, k);
        let expected: Vec<f32> = product.iter().map(|v| v * 2.0).collect();
        approx_eq(&c, &expected, TOL);
    }

    // ════════════════════════════════════════════════════════════
    // neon_batch_matmul_scale_f32
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_scale_1x1() {
        let a = [3.0f32];
        let b = [5.0f32];
        let mut c = [0.0f32];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, 1, 1, 1, 2.0);
        approx_eq(&c, &[30.0], TOL); // 2 * 15
    }

    #[test]
    fn test_scale_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, 2, 2, 2, 0.5);
        let unscaled = ref_matmul(&a, &b, 2, 2, 2);
        let expected: Vec<f32> = unscaled.iter().map(|v| v * 0.5).collect();
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_scale_4x4() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, m, n, k, 3.0);
        let unscaled = ref_matmul(&a, &b, m, n, k);
        let expected: Vec<f32> = unscaled.iter().map(|v| v * 3.0).collect();
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_scale_8x8() {
        let (m, n, k) = (8, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, m, n, k, 0.1);
        let unscaled = ref_matmul(&a, &b, m, n, k);
        let expected: Vec<f32> = unscaled.iter().map(|v| v * 0.1).collect();
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_scale_16x16() {
        let (m, n, k) = (16, 16, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, m, n, k, 10.0);
        let unscaled = ref_matmul(&a, &b, m, n, k);
        let expected: Vec<f32> = unscaled.iter().map(|v| v * 10.0).collect();
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_scale_alpha_zero() {
        let (m, n, k) = (3, 4, 5);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![999.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, m, n, k, 0.0);
        approx_eq(&c, &vec![0.0f32; m * n], TOL);
    }

    #[test]
    fn test_scale_alpha_one() {
        let (m, n, k) = (3, 4, 5);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, m, n, k, 1.0);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_scale_alpha_negative() {
        let (m, n, k) = (3, 3, 3);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, m, n, k, -1.0);
        let unscaled = ref_matmul(&a, &b, m, n, k);
        let expected: Vec<f32> = unscaled.iter().map(|v| -v).collect();
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_scale_batch2() {
        let (batch, m, n, k) = (2, 3, 4, 5);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * k * n);
        let mut c = vec![0.0f32; batch * m * n];
        let alpha = 2.5;
        neon_batch_matmul_scale_f32(&a, &b, &mut c, batch, m, n, k, alpha);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let unscaled = ref_matmul(a_s, b_s, m, n, k);
            let expected: Vec<f32> = unscaled.iter().map(|v| v * alpha).collect();
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_scale_batch8() {
        let (batch, m, n, k) = (8, 2, 2, 2);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.05).collect();
        let mut c = vec![0.0f32; batch * m * n];
        let alpha = 0.25;
        neon_batch_matmul_scale_f32(&a, &b, &mut c, batch, m, n, k, alpha);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let unscaled = ref_matmul(a_s, b_s, m, n, k);
            let expected: Vec<f32> = unscaled.iter().map(|v| v * alpha).collect();
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_scale_linearity() {
        // alpha*(A*B) should equal (alpha*A)*B
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let alpha = 3.0f32;

        let mut c_scaled = vec![0.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c_scaled, 1, m, n, k, alpha);

        let a_scaled: Vec<f32> = a.iter().map(|v| v * alpha).collect();
        let mut c_prescaled = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a_scaled, &b, &mut c_prescaled, 1, m, n, k);

        approx_eq(&c_scaled, &c_prescaled, TOL);
    }

    #[test]
    fn test_scale_nonsquare() {
        let (m, n, k) = (3, 5, 7);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, m, n, k, 0.5);
        let unscaled = ref_matmul(&a, &b, m, n, k);
        let expected: Vec<f32> = unscaled.iter().map(|v| v * 0.5).collect();
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_scale_empty() {
        let mut c = [0.0f32; 0];
        neon_batch_matmul_scale_f32(&[], &[], &mut c, 0, 2, 2, 2, 5.0);
    }

    // ════════════════════════════════════════════════════════════
    // neon_strided_batch_matmul_f32
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_strided_contiguous() {
        // With tightly packed strides, should match non-strided version
        let (batch, m, n, k) = (2, 3, 4, 5);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * k * n);
        let mut c1 = vec![0.0f32; batch * m * n];
        let mut c2 = vec![0.0f32; batch * m * n];
        neon_batch_matmul_f32(&a, &b, &mut c1, batch, m, n, k);
        neon_strided_batch_matmul_f32(&a, &b, &mut c2, batch, m, n, k, m * k, k * n, m * n);
        approx_eq(&c1, &c2, TOL);
    }

    #[test]
    fn test_strided_1x1() {
        let a = [3.0f32];
        let b = [5.0f32];
        let mut c = [0.0f32];
        neon_strided_batch_matmul_f32(&a, &b, &mut c, 1, 1, 1, 1, 1, 1, 1);
        approx_eq(&c, &[15.0], TOL);
    }

    #[test]
    fn test_strided_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        neon_strided_batch_matmul_f32(&a, &b, &mut c, 1, 2, 2, 2, 4, 4, 4);
        let expected = ref_matmul(&a, &b, 2, 2, 2);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_strided_4x4() {
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_strided_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k, m * k, k * n, m * n);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_strided_with_padding() {
        // Batch strides larger than matrix data (simulating padded tensors)
        let (batch, m, n, k) = (2, 2, 2, 2);
        let padding = 4; // extra elements between batches

        let stride_a = m * k + padding;
        let stride_b = k * n + padding;
        let stride_c = m * n + padding;

        let mut a = vec![0.0f32; batch * stride_a];
        let mut b = vec![0.0f32; batch * stride_b];
        let mut c = vec![0.0f32; batch * stride_c];

        // Fill actual matrix data (skip padding)
        for bi in 0..batch {
            for idx in 0..m * k {
                a[bi * stride_a + idx] = (bi * m * k + idx + 1) as f32;
            }
            for idx in 0..k * n {
                b[bi * stride_b + idx] = (bi * k * n + idx + 1) as f32;
            }
        }

        neon_strided_batch_matmul_f32(&a, &b, &mut c, batch, m, n, k, stride_a, stride_b, stride_c);

        for bi in 0..batch {
            let a_s = &a[bi * stride_a..bi * stride_a + m * k];
            let b_s = &b[bi * stride_b..bi * stride_b + k * n];
            let expected = ref_matmul(a_s, b_s, m, n, k);
            approx_eq(&c[bi * stride_c..bi * stride_c + m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_strided_batch4() {
        let (batch, m, n, k) = (4, 3, 3, 3);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * k * n);
        let mut c = vec![0.0f32; batch * m * n];
        neon_strided_batch_matmul_f32(&a, &b, &mut c, batch, m, n, k, m * k, k * n, m * n);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let expected = ref_matmul(a_s, b_s, m, n, k);
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_strided_batch8() {
        let (batch, m, n, k) = (8, 2, 2, 2);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; batch * m * n];
        neon_strided_batch_matmul_f32(&a, &b, &mut c, batch, m, n, k, m * k, k * n, m * n);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let expected = ref_matmul(a_s, b_s, m, n, k);
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_strided_empty() {
        let mut c = [0.0f32; 0];
        neon_strided_batch_matmul_f32(&[], &[], &mut c, 0, 2, 2, 2, 4, 4, 4);
    }

    #[test]
    fn test_strided_nonsquare() {
        let (m, n, k) = (3, 5, 7);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_strided_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k, m * k, k * n, m * n);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    // ════════════════════════════════════════════════════════════
    // Cross-function / numerical property tests
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_associativity_approximation() {
        // (A*B)*C ≈ A*(B*C)
        let (p, q, r, s) = (4, 4, 4, 4);
        let a: Vec<f32> = (0..p * q).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..q * r).map(|i| (i as f32) * 0.1).collect();
        let c_mat: Vec<f32> = (0..r * s).map(|i| (i as f32) * 0.1).collect();

        let mut ab = vec![0.0f32; p * r];
        neon_batch_matmul_f32(&a, &b, &mut ab, 1, p, r, q);
        let mut abc = vec![0.0f32; p * s];
        neon_batch_matmul_f32(&ab, &c_mat, &mut abc, 1, p, s, r);

        let mut bc = vec![0.0f32; q * s];
        neon_batch_matmul_f32(&b, &c_mat, &mut bc, 1, q, s, r);
        let mut a_bc = vec![0.0f32; p * s];
        neon_batch_matmul_f32(&a, &bc, &mut a_bc, 1, p, s, q);

        approx_eq(&abc, &a_bc, 1e-2);
    }

    #[test]
    fn test_distributivity() {
        // A*(B+C) ≈ A*B + A*C
        let (m, n, k) = (4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let c_mat: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();

        // B + C
        let b_plus_c: Vec<f32> = b.iter().zip(c_mat.iter()).map(|(x, y)| x + y).collect();

        // A*(B+C)
        let mut lhs = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b_plus_c, &mut lhs, 1, m, n, k);

        // A*B + A*C
        let mut ab = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut ab, 1, m, n, k);
        let mut ac = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &c_mat, &mut ac, 1, m, n, k);
        let rhs: Vec<f32> = ab.iter().zip(ac.iter()).map(|(x, y)| x + y).collect();

        approx_eq(&lhs, &rhs, 1e-3);
    }

    #[test]
    fn test_scale_vs_accumulate_equivalence() {
        // 2*(A*B) should equal A*B + A*B (via accumulate)
        let (m, n, k) = (4, 4, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);

        let mut c_scaled = vec![0.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c_scaled, 1, m, n, k, 2.0);

        let mut c_accum = vec![0.0f32; m * n];
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c_accum, 1, m, n, k);
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c_accum, 1, m, n, k);

        approx_eq(&c_scaled, &c_accum, TOL);
    }

    #[test]
    fn test_transb_self_symmetric() {
        // A × Aᵀ should be symmetric
        let (m, k) = (4, 5);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * m];
        neon_batch_matmul_transb_f32(&a, &a, &mut c, 1, m, m, k);

        // Check symmetry: c[i][j] == c[j][i]
        for i in 0..m {
            for j in i + 1..m {
                assert!(
                    (c[i * m + j] - c[j * m + i]).abs() < TOL,
                    "not symmetric at ({i},{j}): {} vs {}",
                    c[i * m + j],
                    c[j * m + i]
                );
            }
        }
    }

    #[test]
    fn test_matmul_vs_strided_consistency() {
        let (batch, m, n, k) = (3, 4, 5, 6);
        let a = make_seq(batch * m * k);
        let b = make_seq(batch * k * n);
        let mut c1 = vec![0.0f32; batch * m * n];
        let mut c2 = vec![0.0f32; batch * m * n];
        neon_batch_matmul_f32(&a, &b, &mut c1, batch, m, n, k);
        neon_strided_batch_matmul_f32(&a, &b, &mut c2, batch, m, n, k, m * k, k * n, m * n);
        approx_eq(&c1, &c2, TOL);
    }

    #[test]
    fn test_batch_independence() {
        // Each batch should be independent: results shouldn't change with batch size
        let (m, n, k) = (3, 4, 5);
        let a = make_seq(m * k);
        let b = make_seq(k * n);

        let mut c_single = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c_single, 1, m, n, k);

        // Same data repeated 3 times
        let a3: Vec<f32> = a.iter().cycle().take(3 * m * k).copied().collect();
        let b3: Vec<f32> = b.iter().cycle().take(3 * k * n).copied().collect();
        let mut c_batch = vec![0.0f32; 3 * m * n];
        neon_batch_matmul_f32(&a3, &b3, &mut c_batch, 3, m, n, k);

        for bi in 0..3 {
            approx_eq(&c_batch[bi * m * n..(bi + 1) * m * n], &c_single, TOL);
        }
    }

    #[test]
    fn test_scale_linearity_different_alphas() {
        // alpha * (A*B) = (alpha * A) * B for various alphas
        let (m, n, k) = (3, 3, 3);
        let a = make_seq(m * k);
        let b = make_seq(k * n);

        for &alpha in &[0.1f32, 0.5, 1.0, 2.0, 10.0] {
            let mut c_scaled = vec![0.0f32; m * n];
            neon_batch_matmul_scale_f32(&a, &b, &mut c_scaled, 1, m, n, k, alpha);

            let a_prescaled: Vec<f32> = a.iter().map(|v| v * alpha).collect();
            let mut c_prescaled = vec![0.0f32; m * n];
            neon_batch_matmul_f32(&a_prescaled, &b, &mut c_prescaled, 1, m, n, k);

            approx_eq(&c_scaled, &c_prescaled, TOL);
        }
    }

    #[test]
    fn test_accumulate_with_zero_matrix() {
        // C += 0*B (A is zero) → C unchanged
        let (m, n, k) = (3, 4, 5);
        let a = vec![0.0f32; m * k];
        let b = make_seq(k * n);
        let initial = vec![42.0f32; m * n];
        let mut c = initial.clone();
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, m, n, k);
        approx_eq(&c, &initial, TOL);
    }

    #[test]
    fn test_matmul_m1_vector() {
        // 1×k × k×n = 1×n (row vector × matrix)
        let (m, n, k) = (1, 8, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_matmul_k1_outer_product() {
        // m×1 × 1×n = outer product
        let (m, n, k) = (4, 5, 1);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        // Should be outer product: c[i][j] = a[i] * b[j]
        for i in 0..m {
            for j in 0..n {
                let expected = a[i] * b[j];
                assert!((c[i * n + j] - expected).abs() < TOL);
            }
        }
    }

    #[test]
    fn test_strided_large_stride() {
        // Strides much larger than needed (simulating views into large tensors)
        let (m, n, k) = (2, 2, 2);
        let stride_a = 100;
        let stride_b = 100;
        let stride_c = 100;

        let mut a = vec![0.0f32; stride_a];
        let mut b = vec![0.0f32; stride_b];
        let mut c = vec![0.0f32; stride_c];

        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        b[0] = 5.0;
        b[1] = 6.0;
        b[2] = 7.0;
        b[3] = 8.0;

        neon_strided_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k, stride_a, stride_b, stride_c);
        let expected = ref_matmul(&a[..m * k], &b[..k * n], m, n, k);
        approx_eq(&c[..m * n], &expected, TOL);
    }

    #[test]
    fn test_transb_negative_values() {
        let a = [-1.0, 2.0, -3.0, 4.0f32];
        let b = [5.0, -6.0, -7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, 2, 2, 2);
        let expected = ref_matmul_transb(&a, &b, 2, 2, 2);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_accumulate_negative_initial() {
        let (m, n, k) = (2, 2, 2);
        let a = [1.0, 0.0, 0.0, 1.0f32]; // identity
        let b = [1.0, 2.0, 3.0, 4.0f32];
        let mut c = [-10.0, -20.0, -30.0, -40.0f32];
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, 1, m, n, k);
        // c += I * B = c + B
        approx_eq(&c, &[-9.0, -18.0, -27.0, -36.0], TOL);
    }

    #[test]
    fn test_scale_small_alpha() {
        let (m, n, k) = (4, 4, 4);
        let a = make_ones(m * k);
        let b = make_ones(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, m, n, k, 1e-6);
        let expected = vec![k as f32 * 1e-6; m * n];
        approx_eq(&c, &expected, 1e-8);
    }

    #[test]
    fn test_scale_large_alpha() {
        let (m, n, k) = (2, 2, 2);
        let a = [0.001, 0.002, 0.003, 0.004f32];
        let b = [0.005, 0.006, 0.007, 0.008f32];
        let mut c = [0.0f32; 4];
        neon_batch_matmul_scale_f32(&a, &b, &mut c, 1, 2, 2, 2, 1e6);
        let unscaled = ref_matmul(&a, &b, 2, 2, 2);
        let expected: Vec<f32> = unscaled.iter().map(|v| v * 1e6).collect();
        approx_eq(&c, &expected, 1.0); // larger tolerance for large values
    }

    #[test]
    fn test_matmul_batch1_same_as_single() {
        let (m, n, k) = (5, 6, 7);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c_batch = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c_batch, 1, m, n, k);
        scalar_matmul(&a, &b, &mut c_ref, m, n, k);
        approx_eq(&c_batch, &c_ref, TOL);
    }

    #[test]
    fn test_transb_dot_product_interpretation() {
        // For m=1, n=1: A×Bᵀ is a dot product
        let k = 8;
        let a: Vec<f32> = (1..=k).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=k).map(|i| (i * 2) as f32).collect();
        let mut c = [0.0f32];
        neon_batch_matmul_transb_f32(&a, &b, &mut c, 1, 1, 1, k);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        approx_eq(&c, &[expected], TOL);
    }

    #[test]
    fn test_accumulate_batch8() {
        let (batch, m, n, k) = (8, 2, 3, 4);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![1.0f32; batch * m * n];
        let initial = c.clone();
        neon_batch_matmul_accumulate_f32(&a, &b, &mut c, batch, m, n, k);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let product = ref_matmul(a_s, b_s, m, n, k);
            let expected: Vec<f32> = initial[bi * m * n..(bi + 1) * m * n]
                .iter()
                .zip(product.iter())
                .map(|(i, p)| i + p)
                .collect();
            approx_eq(&c[bi * m * n..(bi + 1) * m * n], &expected, TOL);
        }
    }

    #[test]
    fn test_matmul_rectangular_tall() {
        let (m, n, k) = (16, 2, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_matmul_rectangular_wide() {
        let (m, n, k) = (2, 16, 4);
        let a = make_seq(m * k);
        let b = make_seq(k * n);
        let mut c = vec![0.0f32; m * n];
        neon_batch_matmul_f32(&a, &b, &mut c, 1, m, n, k);
        let expected = ref_matmul(&a, &b, m, n, k);
        approx_eq(&c, &expected, TOL);
    }

    #[test]
    fn test_strided_zero_k() {
        let mut c = [0.0f32; 4];
        neon_strided_batch_matmul_f32(&[], &[], &mut c, 1, 2, 2, 0, 0, 0, 4);
        approx_eq(&c, &[0.0; 4], TOL);
    }

    #[test]
    fn test_transb_zero_k() {
        let mut c = [0.0f32; 4];
        neon_batch_matmul_transb_f32(&[], &[], &mut c, 1, 2, 2, 0);
        approx_eq(&c, &[0.0; 4], TOL);
    }

    #[test]
    fn test_scale_zero_k() {
        let mut c = [0.0f32; 4];
        neon_batch_matmul_scale_f32(&[], &[], &mut c, 1, 2, 2, 0, 5.0);
        approx_eq(&c, &[0.0; 4], TOL);
    }
}
