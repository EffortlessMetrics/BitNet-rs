//! SIMD-optimized matrix multiplication kernels with tiled blocking.
//!
//! Provides f32 GEMM variants with runtime AVX2 dispatch and scalar
//! fallback.  All matrices are **row-major**: A is `m×k`, B is `k×n`,
//! C is `m×n`.
//!
//! # Functions
//!
//! * [`matmul_f32`] — `C = A × B`
//! * [`matmul_f32_transposed_b`] — `C = A × B^T`  (cache-friendly)
//! * [`matmul_accumulate`] — `C = α·A·B + β·C`
//!
//! Tile sizes are selected automatically via [`MatmulConfig::auto_tune`]
//! or may be set manually.

use std::fmt;

// ── Error type ─────────────────────────────────────────────────────────

/// Errors produced by matrix multiplication routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatmulError {
    /// Matrix dimension does not match the expected length.
    DimensionMismatch { matrix: &'static str, expected: usize, actual: usize },
    /// A dimension is zero, which is not supported.
    ZeroDimension { name: &'static str },
}

impl fmt::Display for MatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatmulError::DimensionMismatch { matrix, expected, actual } => {
                write!(f, "{matrix} length mismatch: expected {expected}, got {actual}")
            }
            MatmulError::ZeroDimension { name } => {
                write!(f, "dimension {name} must be non-zero")
            }
        }
    }
}

impl std::error::Error for MatmulError {}

// ── Configuration ──────────────────────────────────────────────────────

/// Tunable parameters for tiled matrix multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulConfig {
    /// Tile height (rows of A / C processed per block).
    pub tile_m: usize,
    /// Tile width (columns of B / C processed per block).
    pub tile_n: usize,
    /// Tile depth (inner-dimension chunk).
    pub tile_k: usize,
    /// Number of threads (reserved for future parallel tiling; currently unused).
    pub num_threads: usize,
}

impl MatmulConfig {
    /// Reasonable defaults for small-to-medium problems.
    pub const DEFAULT: Self = Self { tile_m: 32, tile_n: 32, tile_k: 64, num_threads: 1 };

    /// Automatically select tile sizes based on the problem shape.
    ///
    /// Heuristic:
    /// * Tiny problems (any dim ≤ 8): tile = full dim (no tiling overhead).
    /// * Small (≤ 128): 16×16×32
    /// * Medium (≤ 512): 32×32×64
    /// * Large: 64×64×128
    pub fn auto_tune(m: usize, n: usize, k: usize) -> Self {
        if m <= 8 || n <= 8 || k <= 8 {
            return Self { tile_m: m, tile_n: n, tile_k: k, num_threads: 1 };
        }
        let max_dim = m.max(n).max(k);
        if max_dim <= 128 {
            Self { tile_m: 16, tile_n: 16, tile_k: 32, num_threads: 1 }
        } else if max_dim <= 512 {
            Self { tile_m: 32, tile_n: 32, tile_k: 64, num_threads: 1 }
        } else {
            Self { tile_m: 64, tile_n: 64, tile_k: 128, num_threads: 1 }
        }
    }
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ── Dimension validation ───────────────────────────────────────────────

fn validate_dims(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MatmulError> {
    if m == 0 {
        return Err(MatmulError::ZeroDimension { name: "m" });
    }
    if n == 0 {
        return Err(MatmulError::ZeroDimension { name: "n" });
    }
    if k == 0 {
        return Err(MatmulError::ZeroDimension { name: "k" });
    }
    let expected_a = m * k;
    if a.len() < expected_a {
        return Err(MatmulError::DimensionMismatch {
            matrix: "A",
            expected: expected_a,
            actual: a.len(),
        });
    }
    let expected_b = k * n;
    if b.len() < expected_b {
        return Err(MatmulError::DimensionMismatch {
            matrix: "B",
            expected: expected_b,
            actual: b.len(),
        });
    }
    let expected_c = m * n;
    if c.len() < expected_c {
        return Err(MatmulError::DimensionMismatch {
            matrix: "C",
            expected: expected_c,
            actual: c.len(),
        });
    }
    Ok(())
}

fn validate_dims_mut(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MatmulError> {
    validate_dims(a, b, c, m, n, k)
}

// ── Runtime AVX2 detection ─────────────────────────────────────────────

#[inline]
fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ── AVX2 micro-kernels ────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[allow(unused_imports)]
    use std::arch::x86_64::*;

    /// Dot-product of two f32 slices using AVX2 + FMA.
    ///
    /// # Safety
    /// Caller must ensure AVX2+FMA are available at runtime.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_f32_avx2(a: &[f32], b: &[f32], len: usize) -> f32 {
        // SAFETY: caller guarantees AVX2+FMA are available.
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();

            let chunks = len / 16;
            let remainder_start = chunks * 16;

            for i in 0..chunks {
                let base = i * 16;
                let a0 = _mm256_loadu_ps(a.as_ptr().add(base));
                let b0 = _mm256_loadu_ps(b.as_ptr().add(base));
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);

                let a1 = _mm256_loadu_ps(a.as_ptr().add(base + 8));
                let b1 = _mm256_loadu_ps(b.as_ptr().add(base + 8));
                acc1 = _mm256_fmadd_ps(a1, b1, acc1);
            }

            let sum_vec = _mm256_add_ps(acc0, acc1);
            let hi128 = _mm256_extractf128_ps(sum_vec, 1);
            let lo128 = _mm256_castps256_ps128(sum_vec);
            let sum128 = _mm_add_ps(lo128, hi128);
            let hi64 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, hi64);
            let hi32 = _mm_shuffle_ps(sum64, sum64, 0x01);
            let sum32 = _mm_add_ss(sum64, hi32);
            let mut total = _mm_cvtss_f32(sum32);

            for i in remainder_start..len {
                total += a[i] * b[i];
            }
            total
        }
    }
}

// ── Scalar micro-kernel ───────────────────────────────────────────────

#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Dispatch to AVX2 or scalar dot product at runtime.
#[inline]
fn dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            // SAFETY: we just verified AVX2+FMA support.
            return unsafe { avx2::dot_f32_avx2(a, b, len) };
        }
    }
    dot_f32_scalar(a, b, len)
}

// ── Public API ─────────────────────────────────────────────────────────

/// Compute `C = A × B` (row-major).
///
/// * `a` — `m × k` row-major
/// * `b` — `k × n` row-major
/// * `c` — `m × n` row-major (output, overwritten)
pub fn matmul_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MatmulError> {
    validate_dims_mut(a, b, c, m, n, k)?;
    let cfg = MatmulConfig::auto_tune(m, n, k);
    matmul_f32_tiled(a, b, c, m, n, k, &cfg);
    Ok(())
}

/// Compute `C = A × B^T` where `b_t` is the **transposed** B stored
/// row-major as `n × k`.  This is more cache-friendly because both
/// row vectors are contiguous.
pub fn matmul_f32_transposed_b(
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MatmulError> {
    // b_t is n×k
    if m == 0 {
        return Err(MatmulError::ZeroDimension { name: "m" });
    }
    if n == 0 {
        return Err(MatmulError::ZeroDimension { name: "n" });
    }
    if k == 0 {
        return Err(MatmulError::ZeroDimension { name: "k" });
    }
    let expected_a = m * k;
    if a.len() < expected_a {
        return Err(MatmulError::DimensionMismatch {
            matrix: "A",
            expected: expected_a,
            actual: a.len(),
        });
    }
    let expected_bt = n * k;
    if b_t.len() < expected_bt {
        return Err(MatmulError::DimensionMismatch {
            matrix: "B_T",
            expected: expected_bt,
            actual: b_t.len(),
        });
    }
    let expected_c = m * n;
    if c.len() < expected_c {
        return Err(MatmulError::DimensionMismatch {
            matrix: "C",
            expected: expected_c,
            actual: c.len(),
        });
    }

    let cfg = MatmulConfig::auto_tune(m, n, k);
    matmul_transposed_b_tiled(a, b_t, c, m, n, k, &cfg);
    Ok(())
}

/// Compute `C = α·A·B + β·C` (GEMM-style accumulate).
pub fn matmul_accumulate(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), MatmulError> {
    validate_dims_mut(a, b, c, m, n, k)?;

    // Scale existing C by beta.
    if beta == 0.0 {
        for val in c[..m * n].iter_mut() {
            *val = 0.0;
        }
    } else if (beta - 1.0).abs() > f32::EPSILON {
        for val in c[..m * n].iter_mut() {
            *val *= beta;
        }
    }

    let cfg = MatmulConfig::auto_tune(m, n, k);
    matmul_f32_accumulate_tiled(a, b, c, m, n, k, alpha, &cfg);
    Ok(())
}

/// Compute `C = A × B` with explicit config.
pub fn matmul_f32_with_config(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &MatmulConfig,
) -> Result<(), MatmulError> {
    validate_dims_mut(a, b, c, m, n, k)?;
    matmul_f32_tiled(a, b, c, m, n, k, config);
    Ok(())
}

// ── Tiled implementations ──────────────────────────────────────────────

/// Standard tiled C = A × B.  Overwrites C.
fn matmul_f32_tiled(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    cfg: &MatmulConfig,
) {
    let tm = cfg.tile_m.max(1);
    let tn = cfg.tile_n.max(1);
    let tk = cfg.tile_k.max(1);

    // Zero output.
    for v in c[..m * n].iter_mut() {
        *v = 0.0;
    }

    // Tile over k first for better accumulation locality.
    let mut kk = 0;
    while kk < k {
        let bk = (k - kk).min(tk);
        let mut ii = 0;
        while ii < m {
            let bm = (m - ii).min(tm);
            let mut jj = 0;
            while jj < n {
                let bn = (n - jj).min(tn);
                micro_kernel_ab(a, b, c, m, n, k, ii, jj, kk, bm, bn, bk);
                jj += bn;
            }
            ii += bm;
        }
        kk += bk;
    }
}

/// Tiled C = A × B^T  (b_t is n×k row-major).
fn matmul_transposed_b_tiled(
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    cfg: &MatmulConfig,
) {
    let tm = cfg.tile_m.max(1);
    let tn = cfg.tile_n.max(1);

    for v in c[..m * n].iter_mut() {
        *v = 0.0;
    }

    let mut ii = 0;
    while ii < m {
        let bm = (m - ii).min(tm);
        let mut jj = 0;
        while jj < n {
            let bn = (n - jj).min(tn);
            for i in ii..ii + bm {
                let a_row = &a[i * k..(i * k) + k];
                for j in jj..jj + bn {
                    let b_row = &b_t[j * k..(j * k) + k];
                    c[i * n + j] = dot_f32(a_row, b_row, k);
                }
            }
            jj += bn;
        }
        ii += bm;
    }
}

/// Tiled accumulate: C += α · A · B.  C is assumed to already hold
/// β·C (scaling was done by the caller).
fn matmul_f32_accumulate_tiled(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    cfg: &MatmulConfig,
) {
    let tm = cfg.tile_m.max(1);
    let tn = cfg.tile_n.max(1);
    let tk = cfg.tile_k.max(1);

    let mut kk = 0;
    while kk < k {
        let bk = (k - kk).min(tk);
        let mut ii = 0;
        while ii < m {
            let bm = (m - ii).min(tm);
            let mut jj = 0;
            while jj < n {
                let bn = (n - jj).min(tn);
                micro_kernel_ab_alpha(a, b, c, m, n, k, ii, jj, kk, bm, bn, bk, alpha);
                jj += bn;
            }
            ii += bm;
        }
        kk += bk;
    }
}

// ── Micro-kernels ─────────────────────────────────────────────────────

/// Accumulate a small block C[ii..ii+bm, jj..jj+bn] += A[ii..,..,kk..kk+bk] * B[kk..,..,jj..jj+bn].
#[allow(clippy::too_many_arguments)]
#[inline]
fn micro_kernel_ab(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    _m: usize,
    n: usize,
    k: usize,
    ii: usize,
    jj: usize,
    kk: usize,
    bm: usize,
    bn: usize,
    bk: usize,
) {
    for i in ii..ii + bm {
        for j in jj..jj + bn {
            let mut sum = 0.0f32;
            for p in kk..kk + bk {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += sum;
        }
    }
}

/// Same as `micro_kernel_ab` but multiplies by alpha.
#[allow(clippy::too_many_arguments)]
#[inline]
fn micro_kernel_ab_alpha(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    _m: usize,
    n: usize,
    k: usize,
    ii: usize,
    jj: usize,
    kk: usize,
    bm: usize,
    bn: usize,
    bk: usize,
    alpha: f32,
) {
    for i in ii..ii + bm {
        for j in jj..jj + bn {
            let mut sum = 0.0f32;
            for p in kk..kk + bk {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += alpha * sum;
        }
    }
}

// ── Convenience helpers ────────────────────────────────────────────────

/// Naive reference implementation for testing.
#[cfg(test)]
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

/// Build an identity matrix of size `n×n`.
#[cfg(test)]
fn identity(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "element [{i}]: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    // ── 1. Small square matrices ───────────────────────────────────

    #[test]
    fn test_2x2_matmul() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        // Expected: [[19,22],[43,50]]
        assert_approx(&c, &[19.0, 22.0, 43.0, 50.0], EPS);
    }

    #[test]
    fn test_3x3_matmul() {
        #[rustfmt::skip]
        let a = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0f32,
        ];
        let b = a;
        let mut c = [0.0f32; 9];
        matmul_f32(&a, &b, &mut c, 3, 3, 3).unwrap();
        let expected = naive_matmul(&a, &b, 3, 3, 3);
        assert_approx(&c, &expected, EPS);
    }

    #[test]
    fn test_4x4_matmul() {
        #[rustfmt::skip]
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        #[rustfmt::skip]
        let b: Vec<f32> = (17..=32).map(|x| x as f32).collect();
        let mut c = vec![0.0f32; 16];
        matmul_f32(&a, &b, &mut c, 4, 4, 4).unwrap();
        let expected = naive_matmul(&a, &b, 4, 4, 4);
        assert_approx(&c, &expected, EPS);
    }

    // ── 2. Rectangular matrices ────────────────────────────────────

    #[test]
    fn test_rect_2x3_times_3x4() {
        let a: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let mut c = vec![0.0f32; 8];
        matmul_f32(&a, &b, &mut c, 2, 4, 3).unwrap();
        let expected = naive_matmul(&a, &b, 2, 4, 3);
        assert_approx(&c, &expected, EPS);
    }

    #[test]
    fn test_rect_5x3_times_3x2() {
        let a: Vec<f32> = (1..=15).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let mut c = vec![0.0f32; 10];
        matmul_f32(&a, &b, &mut c, 5, 2, 3).unwrap();
        let expected = naive_matmul(&a, &b, 5, 2, 3);
        assert_approx(&c, &expected, EPS);
    }

    #[test]
    fn test_rect_1x5_times_5x1() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0f32];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0f32];
        let mut c = [0.0f32; 1];
        matmul_f32(&a, &b, &mut c, 1, 1, 5).unwrap();
        // dot product = 5+8+9+8+5 = 35
        assert_approx(&c, &[35.0], EPS);
    }

    #[test]
    fn test_rect_m_neq_n_neq_k() {
        let (m, n, k) = (7, 5, 11);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut c, m, n, k).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx(&c, &expected, 1e-4);
    }

    // ── 3. Identity matrix ─────────────────────────────────────────

    #[test]
    fn test_identity_left() {
        let id = identity(4);
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut c = vec![0.0f32; 16];
        matmul_f32(&id, &a, &mut c, 4, 4, 4).unwrap();
        assert_approx(&c, &a, EPS);
    }

    #[test]
    fn test_identity_right() {
        let id = identity(4);
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut c = vec![0.0f32; 16];
        matmul_f32(&a, &id, &mut c, 4, 4, 4).unwrap();
        assert_approx(&c, &a, EPS);
    }

    #[test]
    fn test_identity_3x3() {
        let id = identity(3);
        let a: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut c = vec![0.0f32; 9];
        matmul_f32(&a, &id, &mut c, 3, 3, 3).unwrap();
        assert_approx(&c, &a, EPS);
    }

    // ── 4. Zero matrix ─────────────────────────────────────────────

    #[test]
    fn test_zero_matrix_left() {
        let z = vec![0.0f32; 9];
        let a: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let mut c = vec![0.0f32; 9];
        matmul_f32(&z, &a, &mut c, 3, 3, 3).unwrap();
        assert_approx(&c, &vec![0.0f32; 9], EPS);
    }

    #[test]
    fn test_zero_matrix_right() {
        let z = vec![0.0f32; 9];
        let a: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let mut c = vec![0.0f32; 9];
        matmul_f32(&a, &z, &mut c, 3, 3, 3).unwrap();
        assert_approx(&c, &vec![0.0f32; 9], EPS);
    }

    // ── 5. Single element ──────────────────────────────────────────

    #[test]
    fn test_1x1_matmul() {
        let a = [3.0f32];
        let b = [7.0f32];
        let mut c = [0.0f32; 1];
        matmul_f32(&a, &b, &mut c, 1, 1, 1).unwrap();
        assert_approx(&c, &[21.0], EPS);
    }

    // ── 6. Transposed-B variant ────────────────────────────────────

    #[test]
    fn test_transposed_b_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // B^T stored row-major: [[5,7],[6,8]]
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b_t = [5.0, 7.0, 6.0, 8.0f32];
        let mut c = [0.0f32; 4];
        matmul_f32_transposed_b(&a, &b_t, &mut c, 2, 2, 2).unwrap();
        assert_approx(&c, &[19.0, 22.0, 43.0, 50.0], EPS);
    }

    #[test]
    fn test_transposed_b_3x3() {
        #[rustfmt::skip]
        let a = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0f32,
        ];
        #[rustfmt::skip]
        let b = [
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0f32,
        ];
        // B^T row-major: each row of b_t is a column of B.
        #[rustfmt::skip]
        let b_t = [
            9.0, 6.0, 3.0,
            8.0, 5.0, 2.0,
            7.0, 4.0, 1.0f32,
        ];
        let mut c1 = vec![0.0f32; 9];
        let mut c2 = vec![0.0f32; 9];
        matmul_f32(&a, &b, &mut c1, 3, 3, 3).unwrap();
        matmul_f32_transposed_b(&a, &b_t, &mut c2, 3, 3, 3).unwrap();
        assert_approx(&c1, &c2, EPS);
    }

    #[test]
    fn test_transposed_b_rect() {
        let (m, n, k) = (4, 3, 5);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.3).collect();
        // Transpose B (k×n) → B^T (n×k)
        let mut b_t = vec![0.0f32; n * k];
        for r in 0..k {
            for c in 0..n {
                b_t[c * k + r] = b[r * n + c];
            }
        }
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut c1, m, n, k).unwrap();
        matmul_f32_transposed_b(&a, &b_t, &mut c2, m, n, k).unwrap();
        assert_approx(&c1, &c2, 1e-4);
    }

    #[test]
    fn test_transposed_b_identity() {
        let n = 4;
        let id = identity(n);
        // I^T = I
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut c = vec![0.0f32; 16];
        matmul_f32_transposed_b(&a, &id, &mut c, n, n, n).unwrap();
        assert_approx(&c, &a, EPS);
    }

    // ── 7. Accumulate with alpha/beta ──────────────────────────────

    #[test]
    fn test_accumulate_alpha1_beta0() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [99.0f32; 4]; // should be overwritten
        matmul_accumulate(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0).unwrap();
        assert_approx(&c, &[19.0, 22.0, 43.0, 50.0], EPS);
    }

    #[test]
    fn test_accumulate_alpha2_beta0() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        matmul_accumulate(&a, &b, &mut c, 2, 2, 2, 2.0, 0.0).unwrap();
        // 2 * [19,22,43,50]
        assert_approx(&c, &[38.0, 44.0, 86.0, 100.0], EPS);
    }

    #[test]
    fn test_accumulate_alpha1_beta1() {
        let a = [1.0, 0.0, 0.0, 1.0f32]; // identity
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [10.0, 20.0, 30.0, 40.0f32];
        matmul_accumulate(&a, &b, &mut c, 2, 2, 2, 1.0, 1.0).unwrap();
        // C = 1*I*B + 1*C_old = B + C_old
        assert_approx(&c, &[15.0, 26.0, 37.0, 48.0], EPS);
    }

    #[test]
    fn test_accumulate_alpha_half_beta_2() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [10.0, 10.0, 10.0, 10.0f32];
        // C = 0.5 * A*B + 2.0 * C_old
        // A*B = [19,22,43,50]
        // result = [0.5*19+2*10, 0.5*22+2*10, 0.5*43+2*10, 0.5*50+2*10]
        //        = [29.5, 31.0, 41.5, 45.0]
        matmul_accumulate(&a, &b, &mut c, 2, 2, 2, 0.5, 2.0).unwrap();
        assert_approx(&c, &[29.5, 31.0, 41.5, 45.0], EPS);
    }

    #[test]
    fn test_accumulate_alpha0() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [10.0, 20.0, 30.0, 40.0f32];
        // C = 0*A*B + 1*C = C
        matmul_accumulate(&a, &b, &mut c, 2, 2, 2, 0.0, 1.0).unwrap();
        assert_approx(&c, &[10.0, 20.0, 30.0, 40.0], EPS);
    }

    #[test]
    fn test_accumulate_beta_only() {
        let a = [1.0f32];
        let b = [1.0f32];
        let mut c = [5.0f32];
        // C = 0*1*1 + 3*5 = 15
        matmul_accumulate(&a, &b, &mut c, 1, 1, 1, 0.0, 3.0).unwrap();
        assert_approx(&c, &[15.0], EPS);
    }

    // ── 8. Tile-size variations via with_config ────────────────────

    #[test]
    fn test_tile_1x1x1() {
        let cfg = MatmulConfig { tile_m: 1, tile_n: 1, tile_k: 1, num_threads: 1 };
        let a: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let b: Vec<f32> = (10..=18).map(|x| x as f32).collect();
        let mut c = vec![0.0f32; 9];
        matmul_f32_with_config(&a, &b, &mut c, 3, 3, 3, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, 3, 3, 3);
        assert_approx(&c, &expected, EPS);
    }

    #[test]
    fn test_tile_2x2x2() {
        let cfg = MatmulConfig { tile_m: 2, tile_n: 2, tile_k: 2, num_threads: 1 };
        let (m, n, k) = (5, 4, 6);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; m * n];
        matmul_f32_with_config(&a, &b, &mut c, m, n, k, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx(&c, &expected, EPS);
    }

    #[test]
    fn test_tile_larger_than_matrix() {
        let cfg = MatmulConfig { tile_m: 128, tile_n: 128, tile_k: 128, num_threads: 1 };
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        matmul_f32_with_config(&a, &b, &mut c, 2, 2, 2, &cfg).unwrap();
        assert_approx(&c, &[19.0, 22.0, 43.0, 50.0], EPS);
    }

    #[test]
    fn test_tile_non_divisible() {
        let cfg = MatmulConfig { tile_m: 3, tile_n: 3, tile_k: 3, num_threads: 1 };
        let (m, n, k) = (7, 5, 11);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        matmul_f32_with_config(&a, &b, &mut c, m, n, k, &cfg).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx(&c, &expected, 1e-3);
    }

    // ── 9. Auto-tune selection ─────────────────────────────────────

    #[test]
    fn test_auto_tune_tiny() {
        let cfg = MatmulConfig::auto_tune(2, 3, 4);
        assert_eq!(cfg.tile_m, 2);
        assert_eq!(cfg.tile_n, 3);
        assert_eq!(cfg.tile_k, 4);
    }

    #[test]
    fn test_auto_tune_small() {
        let cfg = MatmulConfig::auto_tune(64, 64, 64);
        assert_eq!(cfg.tile_m, 16);
        assert_eq!(cfg.tile_n, 16);
        assert_eq!(cfg.tile_k, 32);
    }

    #[test]
    fn test_auto_tune_medium() {
        let cfg = MatmulConfig::auto_tune(256, 256, 256);
        assert_eq!(cfg.tile_m, 32);
        assert_eq!(cfg.tile_n, 32);
        assert_eq!(cfg.tile_k, 64);
    }

    #[test]
    fn test_auto_tune_large() {
        let cfg = MatmulConfig::auto_tune(1024, 1024, 1024);
        assert_eq!(cfg.tile_m, 64);
        assert_eq!(cfg.tile_n, 64);
        assert_eq!(cfg.tile_k, 128);
    }

    #[test]
    fn test_auto_tune_asymmetric() {
        // Any dim ≤ 8 → tiny bucket (tile = full dim).
        let cfg = MatmulConfig::auto_tune(4, 4, 1000);
        assert_eq!(cfg.tile_m, 4);
        assert_eq!(cfg.tile_n, 4);
        assert_eq!(cfg.tile_k, 1000);
    }

    #[test]
    fn test_auto_tune_asymmetric_large() {
        // All dims > 8, max > 512 → large bucket.
        let cfg = MatmulConfig::auto_tune(16, 16, 1000);
        assert_eq!(cfg.tile_m, 64);
        assert_eq!(cfg.tile_n, 64);
        assert_eq!(cfg.tile_k, 128);
    }

    #[test]
    fn test_auto_tune_edge_128() {
        let cfg = MatmulConfig::auto_tune(128, 128, 128);
        assert_eq!(cfg.tile_m, 16);
    }

    #[test]
    fn test_auto_tune_edge_512() {
        let cfg = MatmulConfig::auto_tune(512, 512, 512);
        assert_eq!(cfg.tile_m, 32);
    }

    // ── 10. Dimension validation errors ────────────────────────────

    #[test]
    fn test_err_zero_m() {
        let mut c = [0.0f32; 4];
        let err = matmul_f32(&[1.0], &[1.0], &mut c, 0, 2, 2).unwrap_err();
        assert_eq!(err, MatmulError::ZeroDimension { name: "m" });
    }

    #[test]
    fn test_err_zero_n() {
        let mut c = [0.0f32; 4];
        let err = matmul_f32(&[1.0], &[1.0], &mut c, 2, 0, 2).unwrap_err();
        assert_eq!(err, MatmulError::ZeroDimension { name: "n" });
    }

    #[test]
    fn test_err_zero_k() {
        let mut c = [0.0f32; 4];
        let err = matmul_f32(&[1.0], &[1.0], &mut c, 2, 2, 0).unwrap_err();
        assert_eq!(err, MatmulError::ZeroDimension { name: "k" });
    }

    #[test]
    fn test_err_a_too_small() {
        let mut c = [0.0f32; 4];
        let err = matmul_f32(&[1.0, 2.0], &[1.0; 4], &mut c, 2, 2, 2).unwrap_err();
        assert_eq!(err, MatmulError::DimensionMismatch { matrix: "A", expected: 4, actual: 2 });
    }

    #[test]
    fn test_err_b_too_small() {
        let mut c = [0.0f32; 4];
        let err = matmul_f32(&[1.0; 4], &[1.0], &mut c, 2, 2, 2).unwrap_err();
        assert_eq!(err, MatmulError::DimensionMismatch { matrix: "B", expected: 4, actual: 1 });
    }

    #[test]
    fn test_err_c_too_small() {
        let mut c = [0.0f32; 2];
        let err = matmul_f32(&[1.0; 4], &[1.0; 4], &mut c, 2, 2, 2).unwrap_err();
        assert_eq!(err, MatmulError::DimensionMismatch { matrix: "C", expected: 4, actual: 2 });
    }

    #[test]
    fn test_err_transposed_b_bt_too_small() {
        let mut c = [0.0f32; 4];
        let err = matmul_f32_transposed_b(&[1.0; 4], &[1.0; 2], &mut c, 2, 2, 2).unwrap_err();
        assert_eq!(err, MatmulError::DimensionMismatch { matrix: "B_T", expected: 4, actual: 2 });
    }

    #[test]
    fn test_err_accumulate_bad_dims() {
        let mut c = [0.0f32; 1];
        let err = matmul_accumulate(&[1.0; 4], &[1.0; 4], &mut c, 2, 2, 2, 1.0, 0.0).unwrap_err();
        assert_eq!(err, MatmulError::DimensionMismatch { matrix: "C", expected: 4, actual: 1 });
    }

    #[test]
    fn test_error_display() {
        let e = MatmulError::DimensionMismatch { matrix: "A", expected: 10, actual: 5 };
        assert!(e.to_string().contains("A"));
        assert!(e.to_string().contains("10"));
        assert!(e.to_string().contains("5"));

        let e2 = MatmulError::ZeroDimension { name: "k" };
        assert!(e2.to_string().contains("k"));
    }

    // ── 11. Property: A * I = A ────────────────────────────────────

    #[test]
    fn test_property_a_times_identity_equals_a() {
        for n in [1, 2, 3, 5, 8, 16] {
            let id = identity(n);
            let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.7 - 3.0).collect();
            let mut c = vec![0.0f32; n * n];
            matmul_f32(&a, &id, &mut c, n, n, n).unwrap();
            assert_approx(&c, &a, 1e-4);
        }
    }

    // ── 12. Property: (A*B)*C ≈ A*(B*C) ───────────────────────────

    #[test]
    fn test_property_associativity() {
        let n = 4;
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.3).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.2 + 1.0).collect();
        let d: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.1 - 0.5).collect();

        // (A*B)*D
        let mut ab = vec![0.0f32; n * n];
        matmul_f32(&a, &b, &mut ab, n, n, n).unwrap();
        let mut ab_d = vec![0.0f32; n * n];
        matmul_f32(&ab, &d, &mut ab_d, n, n, n).unwrap();

        // A*(B*D)
        let mut bd = vec![0.0f32; n * n];
        matmul_f32(&b, &d, &mut bd, n, n, n).unwrap();
        let mut a_bd = vec![0.0f32; n * n];
        matmul_f32(&a, &bd, &mut a_bd, n, n, n).unwrap();

        assert_approx(&ab_d, &a_bd, 1e-2);
    }

    // ── 13. Config defaults ────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = MatmulConfig::default();
        assert_eq!(cfg, MatmulConfig::DEFAULT);
        assert_eq!(cfg.tile_m, 32);
        assert_eq!(cfg.tile_n, 32);
        assert_eq!(cfg.tile_k, 64);
        assert_eq!(cfg.num_threads, 1);
    }

    // ── 14. Negative values ────────────────────────────────────────

    #[test]
    fn test_negative_values() {
        let a = [-1.0, -2.0, -3.0, -4.0f32];
        let b = [1.0, 2.0, 3.0, 4.0f32];
        let mut c = [0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        let expected = naive_matmul(&a, &b, 2, 2, 2);
        assert_approx(&c, &expected, EPS);
    }

    // ── 15. Large-ish matrix for tiling ────────────────────────────

    #[test]
    fn test_medium_matrix_64x64() {
        let n = 64;
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 17) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i % 13) as f32) * 0.1).collect();
        let mut c = vec![0.0f32; n * n];
        matmul_f32(&a, &b, &mut c, n, n, n).unwrap();
        let expected = naive_matmul(&a, &b, n, n, n);
        assert_approx(&c, &expected, 1e-2);
    }

    // ── 16. Overwrite semantics ────────────────────────────────────

    #[test]
    fn test_c_is_overwritten() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [999.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert_approx(&c, &[19.0, 22.0, 43.0, 50.0], EPS);
    }

    // ── 17. AVX2 dispatch path ─────────────────────────────────────

    #[test]
    fn test_avx2_detection_does_not_panic() {
        // Just assert that the detection function is callable.
        let _ = avx2_available();
    }

    #[test]
    fn test_dot_f32_dispatch() {
        let a: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..33).map(|i| (i as f32) * 0.5).collect();
        let result = dot_f32(&a, &b, 33);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-2, "dot: {result} vs {expected}");
    }

    #[test]
    fn test_dot_f32_empty() {
        let result = dot_f32(&[], &[], 0);
        assert_eq!(result, 0.0);
    }

    // ── 18. Scalar fallback correctness ────────────────────────────

    #[test]
    fn test_scalar_dot_product() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [4.0, 3.0, 2.0, 1.0f32];
        let result = dot_f32_scalar(&a, &b, 4);
        assert!((result - 20.0).abs() < EPS);
    }

    // ── 19. Transposed-B dimension error ───────────────────────────

    #[test]
    fn test_transposed_b_zero_dim() {
        let mut c = [0.0f32; 4];
        let err = matmul_f32_transposed_b(&[1.0; 4], &[1.0; 4], &mut c, 0, 2, 2).unwrap_err();
        assert_eq!(err, MatmulError::ZeroDimension { name: "m" });
    }

    // ── 20. Accumulate preserves C when beta=1, alpha=0 ────────────

    #[test]
    fn test_accumulate_noop() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let original = [100.0, 200.0, 300.0, 400.0f32];
        let mut c = original;
        matmul_accumulate(&a, &b, &mut c, 2, 2, 2, 0.0, 1.0).unwrap();
        assert_approx(&c, &original, EPS);
    }
}
