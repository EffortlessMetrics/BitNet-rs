//! AVX2-optimised GEMV (General Matrix-Vector multiply) kernel.
//!
//! Provides `gemv_f32`, `gemv_f32_transposed`, `batch_gemv_f32`,
//! `gemv_f32_fma` and `gemv_quantized_i2s`.  Every public function
//! performs runtime feature detection via `is_x86_feature_detected!`
//! and falls back to a portable scalar implementation when AVX2 is
//! unavailable.
//!
//! # Layout conventions
//!
//! * Matrices are **row-major**: element `(i, j)` is at index
//!   `i * cols + j`.
//! * I2_S packed weights use 2-bit ternary packing (4 values per byte,
//!   LSB-first) with one f32 scale per row.

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

use thiserror::Error;

// ── Types ──────────────────────────────────────────────────────────────

/// Configuration for a GEMV invocation.
#[derive(Debug, Clone)]
pub struct GemvConfig {
    /// Number of rows in the matrix (output length).
    pub m: usize,
    /// Number of columns in the matrix (input vector length).
    pub n: usize,
    /// Scalar multiplier applied to the matrix-vector product.
    pub alpha: f32,
    /// Scalar multiplier applied to the existing output before
    /// accumulation: `y = alpha * A * x + beta * y`.
    pub beta: f32,
}

impl GemvConfig {
    /// Minimal config for `y = A * x` (alpha=1, beta=0).
    pub fn new(m: usize, n: usize) -> Self {
        Self { m, n, alpha: 1.0, beta: 0.0 }
    }
}

/// Errors specific to GEMV operations.
#[derive(Debug, Error)]
pub enum GemvError {
    #[error(
        "dimension mismatch: expected matrix {expected_mat}, vector {expected_vec}, output {expected_out}, got matrix {got_mat}, vector {got_vec}, output {got_out}"
    )]
    DimensionMismatch {
        expected_mat: usize,
        got_mat: usize,
        expected_vec: usize,
        got_vec: usize,
        expected_out: usize,
        got_out: usize,
    },
    #[error("empty dimension: m={m}, n={n}")]
    EmptyDimension { m: usize, n: usize },
    #[error("batch size mismatch: expected {expected}, got {got}")]
    BatchSizeMismatch { expected: usize, got: usize },
    #[error("quantized weight length mismatch: expected {expected}, got {got}")]
    QuantizedWeightMismatch { expected: usize, got: usize },
    #[error("scale length mismatch: expected {expected}, got {got}")]
    ScaleMismatch { expected: usize, got: usize },
}

/// Convenience alias.
pub type GemvResult<T> = std::result::Result<T, GemvError>;

// ── Dimension validation helpers ───────────────────────────────────────

#[inline]
fn validate_gemv(cfg: &GemvConfig, matrix: &[f32], x: &[f32], y: &[f32]) -> GemvResult<()> {
    if cfg.m == 0 || cfg.n == 0 {
        return Err(GemvError::EmptyDimension { m: cfg.m, n: cfg.n });
    }
    let mat_len = cfg.m * cfg.n;
    if matrix.len() < mat_len || x.len() < cfg.n || y.len() < cfg.m {
        return Err(GemvError::DimensionMismatch {
            expected_mat: mat_len,
            got_mat: matrix.len(),
            expected_vec: cfg.n,
            got_vec: x.len(),
            expected_out: cfg.m,
            got_out: y.len(),
        });
    }
    Ok(())
}

// ── Scalar fallback implementations ────────────────────────────────────

/// Scalar GEMV: `y = alpha * A * x + beta * y`.
fn gemv_f32_scalar(cfg: &GemvConfig, matrix: &[f32], x: &[f32], y: &mut [f32]) {
    for i in 0..cfg.m {
        let row = &matrix[i * cfg.n..(i + 1) * cfg.n];
        let mut acc: f32 = 0.0;
        for j in 0..cfg.n {
            acc += row[j] * x[j];
        }
        y[i] = cfg.alpha * acc + cfg.beta * y[i];
    }
}

/// Scalar GEMV transposed: `y = alpha * A^T * x + beta * y`.
/// A is M×N row-major; we compute A^T · x where x has length M,
/// yielding y of length N.
fn gemv_f32_transposed_scalar(
    m: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    matrix: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    // Scale existing y by beta
    for val in y[..n].iter_mut() {
        *val *= beta;
    }
    for i in 0..m {
        let row = &matrix[i * n..(i + 1) * n];
        let xi = alpha * x[i];
        for j in 0..n {
            y[j] += xi * row[j];
        }
    }
}

/// Scalar FMA path (identical arithmetic, separate for benchmarking).
fn gemv_f32_fma_scalar(cfg: &GemvConfig, matrix: &[f32], x: &[f32], y: &mut [f32]) {
    gemv_f32_scalar(cfg, matrix, x, y);
}

/// Scalar I2_S quantized GEMV.
/// `weights` stores 4 ternary values per byte (2 bits each, LSB-first).
/// Each row has `ceil(n/4)` packed bytes and one f32 scale.
fn gemv_quantized_i2s_scalar(
    m: usize,
    n: usize,
    weights: &[u8],
    scales: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    let packed_cols = (n + 3) / 4;
    for i in 0..m {
        let row = &weights[i * packed_cols..(i + 1) * packed_cols];
        let scale = scales[i];
        let mut acc: f32 = 0.0;
        for j in 0..n {
            let byte_idx = j / 4;
            let bit_offset = (j % 4) * 2;
            let raw = (row[byte_idx] >> bit_offset) & 0x03;
            // 2-bit encoding: 0 → -1, 1 → 0, 2 → +1, 3 → 0
            let val: f32 = match raw {
                0 => -1.0,
                2 => 1.0,
                _ => 0.0,
            };
            acc += val * x[j];
        }
        y[i] = scale * acc;
    }
}

// ── AVX2 implementations ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
/// # Safety
/// Caller must ensure AVX2+FMA are available at runtime.
unsafe fn gemv_f32_avx2(cfg: &GemvConfig, matrix: &[f32], x: &[f32], y: &mut [f32]) {
    let n = cfg.n;
    let lanes = 8usize; // 256-bit / 32-bit
    let n_aligned = n & !(lanes - 1);

    for i in 0..cfg.m {
        let row_ptr = matrix.as_ptr().add(i * n);
        let x_ptr = x.as_ptr();

        // Safety: AVX2 is guaranteed available by target_feature gate.
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();

        // Process 16 elements per iteration (two AVX2 registers)
        let n_unrolled = n_aligned & !(2 * lanes - 1);
        let mut j = 0usize;
        while j < n_unrolled {
            let a0 = _mm256_loadu_ps(row_ptr.add(j));
            let b0 = _mm256_loadu_ps(x_ptr.add(j));
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);

            let a1 = _mm256_loadu_ps(row_ptr.add(j + lanes));
            let b1 = _mm256_loadu_ps(x_ptr.add(j + lanes));
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);

            j += 2 * lanes;
        }

        // Handle remaining aligned chunk
        if j < n_aligned {
            let a0 = _mm256_loadu_ps(row_ptr.add(j));
            let b0 = _mm256_loadu_ps(x_ptr.add(j));
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            j += lanes;
        }

        // Combine the two accumulators
        sum0 = _mm256_add_ps(sum0, sum1);

        // Horizontal sum of 8 floats
        let hi = _mm256_extractf128_ps(sum0, 1);
        let lo = _mm256_castps256_ps128(sum0);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut acc = _mm_cvtss_f32(result);

        // Scalar tail
        while j < n {
            acc += *matrix.get_unchecked(i * n + j) * *x.get_unchecked(j);
            j += 1;
        }

        *y.get_unchecked_mut(i) = cfg.alpha * acc + cfg.beta * *y.get_unchecked(i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
/// # Safety
/// Caller must ensure AVX2+FMA are available at runtime.
unsafe fn gemv_f32_transposed_avx2(
    m: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    matrix: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    let lanes = 8usize;
    let n_aligned = n & !(lanes - 1);

    // Safety: AVX2 is guaranteed available by target_feature gate.
    // Scale y by beta
    let beta_v = _mm256_set1_ps(beta);
    {
        let mut j = 0usize;
        while j < n_aligned {
            let yv = _mm256_loadu_ps(y.as_ptr().add(j));
            let scaled = _mm256_mul_ps(yv, beta_v);
            _mm256_storeu_ps(y.as_mut_ptr().add(j), scaled);
            j += lanes;
        }
        while j < n {
            *y.get_unchecked_mut(j) *= beta;
            j += 1;
        }
    }

    // Accumulate A^T * x
    for i in 0..m {
        let row_ptr = matrix.as_ptr().add(i * n);
        let xi = alpha * x[i];
        let xi_v = _mm256_set1_ps(xi);

        let mut j = 0usize;
        while j < n_aligned {
            let a = _mm256_loadu_ps(row_ptr.add(j));
            let yv = _mm256_loadu_ps(y.as_ptr().add(j));
            let res = _mm256_fmadd_ps(a, xi_v, yv);
            _mm256_storeu_ps(y.as_mut_ptr().add(j), res);
            j += lanes;
        }
        while j < n {
            *y.get_unchecked_mut(j) += xi * *matrix.get_unchecked(i * n + j);
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
/// # Safety
/// Caller must ensure AVX2+FMA are available at runtime.
unsafe fn gemv_quantized_i2s_avx2(
    m: usize,
    n: usize,
    weights: &[u8],
    scales: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    // For quantized I2_S we unpack 2-bit ternary values and use FMA.
    // Each byte holds 4 values. We process 8 floats at a time on the
    // input side and unpack the corresponding ternary values.
    let packed_cols = (n + 3) / 4;
    let lanes = 8usize;
    let n_aligned = n & !(lanes - 1);

    // Safety: AVX2 is guaranteed available by target_feature gate.
    for i in 0..m {
        let row = &weights[i * packed_cols..(i + 1) * packed_cols];
        let scale = scales[i];
        let x_ptr = x.as_ptr();

        let mut sum = _mm256_setzero_ps();
        let mut j = 0usize;

        while j < n_aligned {
            // Unpack 8 ternary values into f32 vector
            let mut vals = [0.0f32; 8];
            for k in 0..8 {
                let idx = j + k;
                let byte_idx = idx / 4;
                let bit_offset = (idx % 4) * 2;
                let raw = (row[byte_idx] >> bit_offset) & 0x03;
                vals[k] = match raw {
                    0 => -1.0,
                    2 => 1.0,
                    _ => 0.0,
                };
            }
            let w = _mm256_loadu_ps(vals.as_ptr());
            let xv = _mm256_loadu_ps(x_ptr.add(j));
            sum = _mm256_fmadd_ps(w, xv, sum);
            j += lanes;
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut acc = _mm_cvtss_f32(result);

        // Scalar tail
        while j < n {
            let byte_idx = j / 4;
            let bit_offset = (j % 4) * 2;
            let raw = (row[byte_idx] >> bit_offset) & 0x03;
            let val: f32 = match raw {
                0 => -1.0,
                2 => 1.0,
                _ => 0.0,
            };
            acc += val * *x.get_unchecked(j);
            j += 1;
        }

        *y.get_unchecked_mut(i) = scale * acc;
    }
}

// ── Public API (runtime dispatch) ──────────────────────────────────────

/// Compute `y = alpha * A * x + beta * y` where A is an M×N row-major
/// matrix and x is a vector of length N.
///
/// Dispatches to AVX2+FMA when available, otherwise falls back to
/// scalar.
pub fn gemv_f32(cfg: &GemvConfig, matrix: &[f32], x: &[f32], y: &mut [f32]) -> GemvResult<()> {
    validate_gemv(cfg, matrix, x, y)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: feature detection confirmed AVX2+FMA support.
            unsafe { gemv_f32_avx2(cfg, matrix, x, y) };
            return Ok(());
        }
    }

    gemv_f32_scalar(cfg, matrix, x, y);
    Ok(())
}

/// Compute `y = alpha * A^T * x + beta * y`.
///
/// A is stored as M×N row-major.  x has length M, y has length N.
pub fn gemv_f32_transposed(
    m: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    matrix: &[f32],
    x: &[f32],
    y: &mut [f32],
) -> GemvResult<()> {
    if m == 0 || n == 0 {
        return Err(GemvError::EmptyDimension { m, n });
    }
    if matrix.len() < m * n {
        return Err(GemvError::DimensionMismatch {
            expected_mat: m * n,
            got_mat: matrix.len(),
            expected_vec: m,
            got_vec: x.len(),
            expected_out: n,
            got_out: y.len(),
        });
    }
    if x.len() < m {
        return Err(GemvError::DimensionMismatch {
            expected_mat: m * n,
            got_mat: matrix.len(),
            expected_vec: m,
            got_vec: x.len(),
            expected_out: n,
            got_out: y.len(),
        });
    }
    if y.len() < n {
        return Err(GemvError::DimensionMismatch {
            expected_mat: m * n,
            got_mat: matrix.len(),
            expected_vec: m,
            got_vec: x.len(),
            expected_out: n,
            got_out: y.len(),
        });
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: feature detection confirmed AVX2+FMA support.
            unsafe {
                gemv_f32_transposed_avx2(m, n, alpha, beta, matrix, x, y);
            }
            return Ok(());
        }
    }

    gemv_f32_transposed_scalar(m, n, alpha, beta, matrix, x, y);
    Ok(())
}

/// Batched GEMV: apply `y_i = alpha * A_i * x_i + beta * y_i` for each
/// (matrix, vector, output) triple.
///
/// All matrices must share the same M×N dimensions.
pub fn batch_gemv_f32(
    cfg: &GemvConfig,
    matrices: &[&[f32]],
    vectors: &[&[f32]],
    outputs: &mut [&mut [f32]],
) -> GemvResult<()> {
    if matrices.len() != vectors.len() {
        return Err(GemvError::BatchSizeMismatch { expected: matrices.len(), got: vectors.len() });
    }
    if matrices.len() != outputs.len() {
        return Err(GemvError::BatchSizeMismatch { expected: matrices.len(), got: outputs.len() });
    }
    for (i, ((mat, vec), out)) in
        matrices.iter().zip(vectors.iter()).zip(outputs.iter_mut()).enumerate()
    {
        gemv_f32(cfg, mat, vec, out).map_err(|e| match e {
            GemvError::DimensionMismatch { .. } => GemvError::DimensionMismatch {
                expected_mat: cfg.m * cfg.n,
                got_mat: mat.len(),
                expected_vec: cfg.n,
                got_vec: vec.len(),
                expected_out: cfg.m,
                got_out: out.len(),
            },
            other => {
                let _ = i;
                other
            }
        })?;
    }
    Ok(())
}

/// Fused multiply-add GEMV path.
///
/// Semantically identical to [`gemv_f32`] but explicitly uses the
/// FMA code path (on x86_64 this is the same AVX2+FMA kernel; on
/// other architectures it falls back to scalar).  Provided as a
/// separate entry point for benchmarking and API parity.
pub fn gemv_f32_fma(cfg: &GemvConfig, matrix: &[f32], x: &[f32], y: &mut [f32]) -> GemvResult<()> {
    validate_gemv(cfg, matrix, x, y)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: feature detection confirmed AVX2+FMA support.
            unsafe { gemv_f32_avx2(cfg, matrix, x, y) };
            return Ok(());
        }
    }

    gemv_f32_fma_scalar(cfg, matrix, x, y);
    Ok(())
}

/// Quantized I2_S GEMV: dequantise on-the-fly and multiply.
///
/// `weights` is a packed byte array with 4 ternary values per byte
/// (2 bits, LSB-first).  `scales` has one f32 per row.
/// Output: `y[i] = scales[i] * (sum_j deq(w[i,j]) * x[j])`.
pub fn gemv_quantized_i2s(
    m: usize,
    n: usize,
    weights: &[u8],
    scales: &[f32],
    x: &[f32],
    y: &mut [f32],
) -> GemvResult<()> {
    if m == 0 || n == 0 {
        return Err(GemvError::EmptyDimension { m, n });
    }
    let packed_cols = (n + 3) / 4;
    if weights.len() < m * packed_cols {
        return Err(GemvError::QuantizedWeightMismatch {
            expected: m * packed_cols,
            got: weights.len(),
        });
    }
    if scales.len() < m {
        return Err(GemvError::ScaleMismatch { expected: m, got: scales.len() });
    }
    if x.len() < n {
        return Err(GemvError::DimensionMismatch {
            expected_mat: m * packed_cols,
            got_mat: weights.len(),
            expected_vec: n,
            got_vec: x.len(),
            expected_out: m,
            got_out: y.len(),
        });
    }
    if y.len() < m {
        return Err(GemvError::DimensionMismatch {
            expected_mat: m * packed_cols,
            got_mat: weights.len(),
            expected_vec: n,
            got_vec: x.len(),
            expected_out: m,
            got_out: y.len(),
        });
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: feature detection confirmed AVX2+FMA support.
            unsafe {
                gemv_quantized_i2s_avx2(m, n, weights, scales, x, y);
            }
            return Ok(());
        }
    }

    gemv_quantized_i2s_scalar(m, n, weights, scales, x, y);
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: reference scalar GEMV for verification
    fn reference_gemv(m: usize, n: usize, matrix: &[f32], x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for i in 0..m {
            for j in 0..n {
                y[i] += matrix[i * n + j] * x[j];
            }
        }
        y
    }

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

    // ── GemvConfig tests ───────────────────────────────────────────

    #[test]
    fn config_new_defaults() {
        let c = GemvConfig::new(4, 8);
        assert_eq!(c.m, 4);
        assert_eq!(c.n, 8);
        assert_eq!(c.alpha, 1.0);
        assert_eq!(c.beta, 0.0);
    }

    #[test]
    fn config_custom() {
        let c = GemvConfig { m: 3, n: 5, alpha: 2.0, beta: 0.5 };
        assert_eq!(c.alpha, 2.0);
        assert_eq!(c.beta, 0.5);
    }

    #[test]
    fn config_clone() {
        let c = GemvConfig::new(2, 3);
        let c2 = c.clone();
        assert_eq!(c.m, c2.m);
        assert_eq!(c.n, c2.n);
    }

    #[test]
    fn config_debug() {
        let c = GemvConfig::new(1, 1);
        let s = format!("{c:?}");
        assert!(s.contains("GemvConfig"));
    }

    // ── Error variant tests ────────────────────────────────────────

    #[test]
    fn error_display_dimension() {
        let e = GemvError::DimensionMismatch {
            expected_mat: 12,
            got_mat: 10,
            expected_vec: 4,
            got_vec: 3,
            expected_out: 3,
            got_out: 2,
        };
        let s = format!("{e}");
        assert!(s.contains("dimension mismatch"));
    }

    #[test]
    fn error_display_empty() {
        let e = GemvError::EmptyDimension { m: 0, n: 5 };
        let s = format!("{e}");
        assert!(s.contains("empty dimension"));
    }

    #[test]
    fn error_display_batch() {
        let e = GemvError::BatchSizeMismatch { expected: 3, got: 2 };
        assert!(format!("{e}").contains("batch size"));
    }

    #[test]
    fn error_display_quantized() {
        let e = GemvError::QuantizedWeightMismatch { expected: 10, got: 5 };
        assert!(format!("{e}").contains("quantized weight"));
    }

    #[test]
    fn error_display_scale() {
        let e = GemvError::ScaleMismatch { expected: 4, got: 2 };
        assert!(format!("{e}").contains("scale length"));
    }

    // ── Validation tests ───────────────────────────────────────────

    #[test]
    fn gemv_empty_m_returns_error() {
        let cfg = GemvConfig::new(0, 4);
        let r = gemv_f32(&cfg, &[], &[1.0; 4], &mut [0.0; 0]);
        assert!(matches!(r, Err(GemvError::EmptyDimension { .. })));
    }

    #[test]
    fn gemv_empty_n_returns_error() {
        let cfg = GemvConfig::new(4, 0);
        let r = gemv_f32(&cfg, &[], &[], &mut [0.0; 4]);
        assert!(matches!(r, Err(GemvError::EmptyDimension { .. })));
    }

    #[test]
    fn gemv_matrix_too_small() {
        let cfg = GemvConfig::new(2, 3);
        let r = gemv_f32(&cfg, &[1.0; 5], &[1.0; 3], &mut [0.0; 2]);
        assert!(matches!(r, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn gemv_vector_too_small() {
        let cfg = GemvConfig::new(2, 3);
        let r = gemv_f32(&cfg, &[1.0; 6], &[1.0; 2], &mut [0.0; 2]);
        assert!(matches!(r, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn gemv_output_too_small() {
        let cfg = GemvConfig::new(2, 3);
        let r = gemv_f32(&cfg, &[1.0; 6], &[1.0; 3], &mut [0.0; 1]);
        assert!(matches!(r, Err(GemvError::DimensionMismatch { .. })));
    }

    // ── gemv_f32 correctness ───────────────────────────────────────

    #[test]
    fn gemv_identity_2x2() {
        let cfg = GemvConfig::new(2, 2);
        let mat = [1.0, 0.0, 0.0, 1.0];
        let x = [3.0, 7.0];
        let mut y = [0.0; 2];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[3.0, 7.0], 1e-6);
    }

    #[test]
    fn gemv_ones_3x4() {
        let cfg = GemvConfig::new(3, 4);
        let mat = vec![1.0; 12];
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0; 3];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[10.0, 10.0, 10.0], 1e-6);
    }

    #[test]
    fn gemv_alpha_scaling() {
        let cfg = GemvConfig { m: 1, n: 3, alpha: 2.0, beta: 0.0 };
        let mat = [1.0, 2.0, 3.0];
        let x = [1.0, 1.0, 1.0];
        let mut y = [0.0];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[12.0], 1e-6);
    }

    #[test]
    fn gemv_beta_accumulation() {
        let cfg = GemvConfig { m: 1, n: 2, alpha: 1.0, beta: 0.5 };
        let mat = [1.0, 1.0];
        let x = [3.0, 4.0];
        let mut y = [10.0];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        // y = 1.0*(3+4) + 0.5*10 = 7 + 5 = 12
        approx_eq(&y, &[12.0], 1e-6);
    }

    #[test]
    fn gemv_zeros_matrix() {
        let cfg = GemvConfig::new(2, 3);
        let mat = [0.0; 6];
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 2];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[0.0, 0.0], 1e-6);
    }

    #[test]
    fn gemv_negative_values() {
        let cfg = GemvConfig::new(2, 2);
        let mat = [-1.0, 2.0, 3.0, -4.0];
        let x = [1.0, 1.0];
        let mut y = [0.0; 2];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[1.0, -1.0], 1e-6);
    }

    #[test]
    fn gemv_single_element() {
        let cfg = GemvConfig::new(1, 1);
        let mat = [5.0];
        let x = [3.0];
        let mut y = [0.0];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[15.0], 1e-6);
    }

    #[test]
    fn gemv_large_n_16() {
        let n = 16;
        let cfg = GemvConfig::new(1, n);
        let mat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = vec![1.0f32; n];
        let mut y = [0.0];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        let expected: f32 = (0..n).map(|i| i as f32).sum();
        approx_eq(&y, &[expected], 1e-4);
    }

    #[test]
    fn gemv_large_n_17_non_aligned() {
        let n = 17;
        let cfg = GemvConfig::new(1, n);
        let mat: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let x = vec![1.0f32; n];
        let mut y = [0.0];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        let expected: f32 = (1..=n).map(|i| i as f32).sum();
        approx_eq(&y, &[expected], 1e-3);
    }

    #[test]
    fn gemv_large_n_33() {
        let n = 33;
        let m = 2;
        let cfg = GemvConfig::new(m, n);
        let mat: Vec<f32> = (0..m * n).map(|i| (i % 7) as f32 - 3.0).collect();
        let x: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.1).collect();
        let mut y = vec![0.0f32; m];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        let ref_y = reference_gemv(m, n, &mat, &x);
        approx_eq(&y, &ref_y, 1e-4);
    }

    #[test]
    fn gemv_256_elements() {
        let n = 256;
        let m = 4;
        let cfg = GemvConfig::new(m, n);
        let mat: Vec<f32> = (0..m * n).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect();
        let x: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let mut y = vec![0.0f32; m];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        let ref_y = reference_gemv(m, n, &mat, &x);
        approx_eq(&y, &ref_y, 1e-2);
    }

    #[test]
    fn gemv_1024_elements() {
        let n = 1024;
        let m = 8;
        let cfg = GemvConfig::new(m, n);
        let mat: Vec<f32> = (0..m * n).map(|i| ((i * 7 + 3) % 101) as f32 * 0.01 - 0.5).collect();
        let x: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 97) as f32 * 0.01 - 0.48).collect();
        let mut y = vec![0.0f32; m];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        let ref_y = reference_gemv(m, n, &mat, &x);
        approx_eq(&y, &ref_y, 1e-1);
    }

    // ── gemv_f32_transposed correctness ────────────────────────────

    #[test]
    fn transposed_identity_2x2() {
        let mat = [1.0, 0.0, 0.0, 1.0];
        let x = [3.0, 7.0];
        let mut y = [0.0; 2];
        gemv_f32_transposed(2, 2, 1.0, 0.0, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[3.0, 7.0], 1e-6);
    }

    #[test]
    fn transposed_simple_2x3() {
        // A = [[1,2,3],[4,5,6]], x = [1,1]
        // A^T * x = [5, 7, 9]
        let mat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0];
        let mut y = [0.0; 3];
        gemv_f32_transposed(2, 3, 1.0, 0.0, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[5.0, 7.0, 9.0], 1e-6);
    }

    #[test]
    fn transposed_with_alpha_beta() {
        let mat = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [10.0, 20.0];
        // y = 2.0 * A^T * x + 0.5 * y
        // A^T * x = [1+3, 2+4] = [4, 6]
        // y = 2*[4,6] + 0.5*[10,20] = [8,12] + [5,10] = [13, 22]
        gemv_f32_transposed(2, 2, 2.0, 0.5, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[13.0, 22.0], 1e-6);
    }

    #[test]
    fn transposed_empty_m() {
        let r = gemv_f32_transposed(0, 3, 1.0, 0.0, &[], &[], &mut [0.0; 3]);
        assert!(matches!(r, Err(GemvError::EmptyDimension { .. })));
    }

    #[test]
    fn transposed_matrix_too_small() {
        let r = gemv_f32_transposed(2, 3, 1.0, 0.0, &[1.0; 5], &[1.0; 2], &mut [0.0; 3]);
        assert!(matches!(r, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn transposed_non_aligned_n() {
        let m = 3;
        let n = 17;
        let mat: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.1).collect();
        let x: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();
        let mut y = vec![0.0f32; n];
        gemv_f32_transposed(m, n, 1.0, 0.0, &mat, &x, &mut y).unwrap();
        // Reference
        let mut ref_y = vec![0.0f32; n];
        for i in 0..m {
            for j in 0..n {
                ref_y[j] += x[i] * mat[i * n + j];
            }
        }
        approx_eq(&y, &ref_y, 1e-4);
    }

    // ── batch_gemv_f32 correctness ─────────────────────────────────

    #[test]
    fn batch_single() {
        let cfg = GemvConfig::new(2, 2);
        let mat = [1.0, 0.0, 0.0, 1.0];
        let x = [5.0, 6.0];
        let mut y = [0.0; 2];
        batch_gemv_f32(&cfg, &[&mat[..]], &[&x[..]], &mut [&mut y[..]]).unwrap();
        approx_eq(&y, &[5.0, 6.0], 1e-6);
    }

    #[test]
    fn batch_two_items() {
        let cfg = GemvConfig::new(1, 2);
        let m1 = [1.0, 2.0];
        let m2 = [3.0, 4.0];
        let x1 = [1.0, 1.0];
        let x2 = [1.0, 1.0];
        let mut y1 = [0.0];
        let mut y2 = [0.0];
        batch_gemv_f32(
            &cfg,
            &[&m1[..], &m2[..]],
            &[&x1[..], &x2[..]],
            &mut [&mut y1[..], &mut y2[..]],
        )
        .unwrap();
        approx_eq(&y1, &[3.0], 1e-6);
        approx_eq(&y2, &[7.0], 1e-6);
    }

    #[test]
    fn batch_mismatched_counts() {
        let cfg = GemvConfig::new(1, 1);
        let r = batch_gemv_f32(&cfg, &[&[1.0][..]], &[], &mut []);
        assert!(matches!(r, Err(GemvError::BatchSizeMismatch { .. })));
    }

    #[test]
    fn batch_output_mismatch() {
        let cfg = GemvConfig::new(1, 1);
        let mut y = [0.0];
        let r = batch_gemv_f32(
            &cfg,
            &[&[1.0][..], &[2.0][..]],
            &[&[1.0][..], &[1.0][..]],
            &mut [&mut y[..]],
        );
        assert!(matches!(r, Err(GemvError::BatchSizeMismatch { .. })));
    }

    #[test]
    fn batch_empty_is_ok() {
        let cfg = GemvConfig::new(1, 1);
        batch_gemv_f32(&cfg, &[], &[], &mut []).unwrap();
    }

    // ── gemv_f32_fma tests ─────────────────────────────────────────

    #[test]
    fn fma_matches_gemv() {
        let n = 64;
        let m = 4;
        let cfg = GemvConfig::new(m, n);
        let mat: Vec<f32> = (0..m * n).map(|i| (i % 17) as f32 - 8.0).collect();
        let x: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut y1 = vec![0.0f32; m];
        let mut y2 = vec![0.0f32; m];
        gemv_f32(&cfg, &mat, &x, &mut y1).unwrap();
        gemv_f32_fma(&cfg, &mat, &x, &mut y2).unwrap();
        approx_eq(&y1, &y2, 1e-6);
    }

    #[test]
    fn fma_identity() {
        let cfg = GemvConfig::new(2, 2);
        let mat = [1.0, 0.0, 0.0, 1.0];
        let x = [9.0, 11.0];
        let mut y = [0.0; 2];
        gemv_f32_fma(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[9.0, 11.0], 1e-6);
    }

    #[test]
    fn fma_validation_error() {
        let cfg = GemvConfig::new(0, 1);
        let r = gemv_f32_fma(&cfg, &[], &[1.0], &mut []);
        assert!(r.is_err());
    }

    // ── gemv_quantized_i2s tests ───────────────────────────────────

    // Helper: pack ternary values into I2_S byte format
    fn pack_i2s(vals: &[i8]) -> Vec<u8> {
        let packed_len = (vals.len() + 3) / 4;
        let mut packed = vec![0u8; packed_len];
        for (i, &v) in vals.iter().enumerate() {
            let code: u8 = match v {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => 3,
            };
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            packed[byte_idx] |= code << bit_offset;
        }
        packed
    }

    #[test]
    fn quantized_all_ones() {
        let m = 2;
        let n = 4;
        let vals: Vec<i8> = vec![1; m * n];
        let weights = pack_i2s(&vals);
        let packed_cols = (n + 3) / 4;
        // Pack per row
        let mut packed = Vec::new();
        for i in 0..m {
            packed.extend_from_slice(&pack_i2s(&vals[i * n..(i + 1) * n]));
        }
        let scales = vec![1.0f32; m];
        let x = vec![1.0f32; n];
        let mut y = vec![0.0f32; m];
        let _ = packed_cols;
        gemv_quantized_i2s(m, n, &packed, &scales, &x, &mut y).unwrap();
        approx_eq(&y, &[4.0, 4.0], 1e-6);
    }

    #[test]
    fn quantized_all_neg_ones() {
        let m = 1;
        let n = 4;
        let vals: Vec<i8> = vec![-1; n];
        let packed = pack_i2s(&vals);
        let scales = vec![2.0];
        let x = vec![1.0f32; n];
        let mut y = vec![0.0];
        gemv_quantized_i2s(m, n, &packed, &scales, &x, &mut y).unwrap();
        // -1*1*4 * scale=2 = -8
        approx_eq(&y, &[-8.0], 1e-6);
    }

    #[test]
    fn quantized_mixed() {
        let m = 1;
        let n = 4;
        let vals: Vec<i8> = vec![1, -1, 0, 1];
        let packed = pack_i2s(&vals);
        let scales = vec![1.0];
        let x = [2.0, 3.0, 4.0, 5.0];
        let mut y = [0.0];
        // 1*2 + (-1)*3 + 0*4 + 1*5 = 2-3+0+5 = 4
        gemv_quantized_i2s(m, n, &packed, &scales, &x, &mut y).unwrap();
        approx_eq(&y, &[4.0], 1e-6);
    }

    #[test]
    fn quantized_with_scale() {
        let m = 1;
        let n = 4;
        let vals: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_i2s(&vals);
        let scales = vec![0.5];
        let x = vec![2.0; n];
        let mut y = [0.0];
        // 4*2 * 0.5 = 4
        gemv_quantized_i2s(m, n, &packed, &scales, &x, &mut y).unwrap();
        approx_eq(&y, &[4.0], 1e-6);
    }

    #[test]
    fn quantized_non_aligned() {
        let m = 1;
        let n = 5; // not multiple of 4
        let vals: Vec<i8> = vec![1, -1, 0, 1, -1];
        let packed = pack_i2s(&vals);
        let scales = vec![1.0];
        let x = [1.0, 1.0, 1.0, 1.0, 1.0];
        let mut y = [0.0];
        // 1 + (-1) + 0 + 1 + (-1) = 0
        gemv_quantized_i2s(m, n, &packed, &scales, &x, &mut y).unwrap();
        approx_eq(&y, &[0.0], 1e-6);
    }

    #[test]
    fn quantized_empty_m() {
        let r = gemv_quantized_i2s(0, 4, &[], &[], &[1.0; 4], &mut []);
        assert!(matches!(r, Err(GemvError::EmptyDimension { .. })));
    }

    #[test]
    fn quantized_weight_too_small() {
        let r = gemv_quantized_i2s(2, 4, &[0], &[1.0; 2], &[1.0; 4], &mut [0.0; 2]);
        assert!(matches!(r, Err(GemvError::QuantizedWeightMismatch { .. })));
    }

    #[test]
    fn quantized_scale_too_small() {
        let packed = vec![0u8; 2];
        let r = gemv_quantized_i2s(2, 4, &packed, &[1.0], &[1.0; 4], &mut [0.0; 2]);
        assert!(matches!(r, Err(GemvError::ScaleMismatch { .. })));
    }

    #[test]
    fn quantized_x_too_small() {
        let packed = vec![0u8; 2];
        let r = gemv_quantized_i2s(2, 4, &packed, &[1.0; 2], &[1.0; 3], &mut [0.0; 2]);
        assert!(matches!(r, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn quantized_y_too_small() {
        let packed = vec![0u8; 2];
        let r = gemv_quantized_i2s(2, 4, &packed, &[1.0; 2], &[1.0; 4], &mut [0.0; 1]);
        assert!(matches!(r, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn quantized_multiple_rows() {
        let m = 3;
        let n = 4;
        // Row 0: all +1, Row 1: all -1, Row 2: all 0
        let row0: Vec<i8> = vec![1, 1, 1, 1];
        let row1: Vec<i8> = vec![-1, -1, -1, -1];
        let row2: Vec<i8> = vec![0, 0, 0, 0];
        let mut packed = pack_i2s(&row0);
        packed.extend_from_slice(&pack_i2s(&row1));
        packed.extend_from_slice(&pack_i2s(&row2));
        let scales = vec![1.0, 1.0, 1.0];
        let x = vec![1.0; n];
        let mut y = vec![0.0; m];
        gemv_quantized_i2s(m, n, &packed, &scales, &x, &mut y).unwrap();
        approx_eq(&y, &[4.0, -4.0, 0.0], 1e-6);
    }

    #[test]
    fn quantized_large_n_17() {
        let m = 1;
        let n = 17;
        let vals: Vec<i8> = (0..n).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();
        let packed = pack_i2s(&vals);
        let scales = vec![1.0];
        let x = vec![1.0; n];
        let mut y = vec![0.0];
        gemv_quantized_i2s(m, n, &packed, &scales, &x, &mut y).unwrap();
        let expected: f32 = vals.iter().map(|&v| v as f32).sum();
        approx_eq(&y, &[expected], 1e-6);
    }

    // ── Additional correctness / edge-case tests ───────────────────

    #[test]
    fn gemv_1x1() {
        let cfg = GemvConfig::new(1, 1);
        let mut y = [0.0];
        gemv_f32(&cfg, &[7.0], &[3.0], &mut y).unwrap();
        approx_eq(&y, &[21.0], 1e-6);
    }

    #[test]
    fn gemv_rectangular_tall() {
        let m = 8;
        let n = 2;
        let cfg = GemvConfig::new(m, n);
        let mat: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
        let x = [1.0, 1.0];
        let mut y = vec![0.0f32; m];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        let ref_y = reference_gemv(m, n, &mat, &x);
        approx_eq(&y, &ref_y, 1e-6);
    }

    #[test]
    fn gemv_rectangular_wide() {
        let m = 2;
        let n = 32;
        let cfg = GemvConfig::new(m, n);
        let mat: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.1).collect();
        let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let mut y = vec![0.0f32; m];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        let ref_y = reference_gemv(m, n, &mat, &x);
        approx_eq(&y, &ref_y, 1e-2);
    }

    #[test]
    fn gemv_alpha_zero() {
        let cfg = GemvConfig { m: 2, n: 2, alpha: 0.0, beta: 1.0 };
        let mat = [99.0; 4];
        let x = [99.0; 2];
        let mut y = [5.0, 7.0];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[5.0, 7.0], 1e-6);
    }

    #[test]
    fn gemv_beta_zero_clears_output() {
        let cfg = GemvConfig { m: 1, n: 1, alpha: 1.0, beta: 0.0 };
        let mut y = [999.0];
        gemv_f32(&cfg, &[2.0], &[3.0], &mut y).unwrap();
        approx_eq(&y, &[6.0], 1e-6);
    }

    #[test]
    fn transposed_x_too_small() {
        let r = gemv_f32_transposed(2, 3, 1.0, 0.0, &[1.0; 6], &[1.0], &mut [0.0; 3]);
        assert!(matches!(r, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn transposed_y_too_small() {
        let r = gemv_f32_transposed(2, 3, 1.0, 0.0, &[1.0; 6], &[1.0; 2], &mut [0.0; 2]);
        assert!(matches!(r, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn fma_large_n() {
        let n = 128;
        let m = 2;
        let cfg = GemvConfig::new(m, n);
        let mat: Vec<f32> = (0..m * n).map(|i| (i % 23) as f32 * 0.1 - 1.0).collect();
        let x: Vec<f32> = (0..n).map(|i| (i % 19) as f32 * 0.1 - 0.9).collect();
        let mut y1 = vec![0.0f32; m];
        let mut y2 = vec![0.0f32; m];
        gemv_f32(&cfg, &mat, &x, &mut y1).unwrap();
        gemv_f32_fma(&cfg, &mat, &x, &mut y2).unwrap();
        approx_eq(&y1, &y2, 1e-5);
    }

    #[test]
    fn batch_three_items() {
        let cfg = GemvConfig::new(1, 3);
        let m1 = [1.0, 0.0, 0.0];
        let m2 = [0.0, 1.0, 0.0];
        let m3 = [0.0, 0.0, 1.0];
        let x = [7.0, 8.0, 9.0];
        let mut y1 = [0.0];
        let mut y2 = [0.0];
        let mut y3 = [0.0];
        batch_gemv_f32(
            &cfg,
            &[&m1[..], &m2[..], &m3[..]],
            &[&x[..]; 3],
            &mut [&mut y1[..], &mut y2[..], &mut y3[..]],
        )
        .unwrap();
        approx_eq(&y1, &[7.0], 1e-6);
        approx_eq(&y2, &[8.0], 1e-6);
        approx_eq(&y3, &[9.0], 1e-6);
    }

    #[test]
    fn gemv_f32_extra_matrix_capacity() {
        // Matrix buffer larger than m*n should work
        let cfg = GemvConfig::new(2, 2);
        let mat = [1.0, 2.0, 3.0, 4.0, 99.0, 99.0];
        let x = [1.0, 1.0];
        let mut y = [0.0; 2];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y, &[3.0, 7.0], 1e-6);
    }

    #[test]
    fn gemv_f32_extra_output_capacity() {
        let cfg = GemvConfig::new(2, 2);
        let mat = [1.0, 0.0, 0.0, 1.0];
        let x = [5.0, 6.0];
        let mut y = [0.0; 4];
        gemv_f32(&cfg, &mat, &x, &mut y).unwrap();
        approx_eq(&y[..2], &[5.0, 6.0], 1e-6);
    }

    #[test]
    fn transposed_large() {
        let m = 4;
        let n = 64;
        let mat: Vec<f32> = (0..m * n).map(|i| (i % 31) as f32 * 0.1 - 1.5).collect();
        let x: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();
        let mut y = vec![0.0f32; n];
        gemv_f32_transposed(m, n, 1.0, 0.0, &mat, &x, &mut y).unwrap();
        let mut ref_y = vec![0.0f32; n];
        for i in 0..m {
            for j in 0..n {
                ref_y[j] += x[i] * mat[i * n + j];
            }
        }
        approx_eq(&y, &ref_y, 1e-3);
    }

    // ── proptest properties ────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn arb_f32() -> impl Strategy<Value = f32> {
            (-10.0f32..10.0f32)
        }

        // Property 1: gemv_f32 matches scalar reference
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn gemv_matches_scalar(
                m in 1usize..=16,
                n in 1usize..=64,
                mat_vals in proptest::collection::vec(arb_f32(), 1..=1024),
                x_vals in proptest::collection::vec(arb_f32(), 1..=64),
            ) {
                let mat_len = m * n;
                if mat_vals.len() < mat_len || x_vals.len() < n {
                    return Ok(());
                }
                let mat = &mat_vals[..mat_len];
                let x = &x_vals[..n];
                let cfg = GemvConfig::new(m, n);
                let mut y = vec![0.0f32; m];
                gemv_f32(&cfg, mat, x, &mut y).unwrap();
                let ref_y = reference_gemv(m, n, mat, x);
                for (i, (a, b)) in y.iter().zip(ref_y.iter()).enumerate() {
                    let tol = (a.abs().max(b.abs()) * 1e-4).max(1e-5);
                    prop_assert!(
                        (a - b).abs() <= tol,
                        "mismatch at {i}: {a} vs {b} (tol={tol})"
                    );
                }
            }
        }

        // Property 2: gemv_f32_fma matches gemv_f32 exactly
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn fma_matches_gemv(
                m in 1usize..=8,
                n in 1usize..=32,
                mat_vals in proptest::collection::vec(arb_f32(), 1..=256),
                x_vals in proptest::collection::vec(arb_f32(), 1..=32),
            ) {
                let mat_len = m * n;
                if mat_vals.len() < mat_len || x_vals.len() < n {
                    return Ok(());
                }
                let mat = &mat_vals[..mat_len];
                let x = &x_vals[..n];
                let cfg = GemvConfig::new(m, n);
                let mut y1 = vec![0.0f32; m];
                let mut y2 = vec![0.0f32; m];
                gemv_f32(&cfg, mat, x, &mut y1).unwrap();
                gemv_f32_fma(&cfg, mat, x, &mut y2).unwrap();
                for (i, (a, b)) in y1.iter().zip(y2.iter()).enumerate() {
                    prop_assert!(
                        (a - b).abs() < 1e-6,
                        "fma diverges at {i}: {a} vs {b}"
                    );
                }
            }
        }

        // Property 3: transposed gemv satisfies dot-product identity
        // (A * x)[i] == dot(row_i(A), x) == dot(col_i(A^T), x)
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn transposed_identity(
                m in 1usize..=8,
                n in 1usize..=32,
                mat_vals in proptest::collection::vec(arb_f32(), 1..=256),
                x_vals in proptest::collection::vec(arb_f32(), 1..=32),
            ) {
                let mat_len = m * n;
                if mat_vals.len() < mat_len || x_vals.len() < m.max(n) {
                    return Ok(());
                }
                let mat = &mat_vals[..mat_len];
                // Use unit vectors: e_i for x, check single output element
                for test_row in 0..m.min(3) {
                    let mut x_unit = vec![0.0f32; m];
                    x_unit[test_row] = 1.0;
                    let mut y = vec![0.0f32; n];
                    gemv_f32_transposed(m, n, 1.0, 0.0, mat, &x_unit, &mut y)
                        .unwrap();
                    // y should equal row `test_row` of A
                    let row = &mat[test_row * n..(test_row + 1) * n];
                    for (j, (a, b)) in y.iter().zip(row.iter()).enumerate() {
                        let tol = (a.abs().max(b.abs()) * 1e-4).max(1e-5);
                        prop_assert!(
                            (a - b).abs() <= tol,
                            "row {test_row} col {j}: {a} vs {b}"
                        );
                    }
                }
            }
        }

        // Property 4: batch_gemv_f32 matches sequential calls
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn batch_matches_sequential(
                m in 1usize..=4,
                n in 1usize..=16,
                batch in 1usize..=4,
                vals in proptest::collection::vec(arb_f32(), 1..=1024),
            ) {
                let mat_len = m * n;
                let total = batch * (mat_len + n);
                if vals.len() < total {
                    return Ok(());
                }
                let cfg = GemvConfig::new(m, n);
                let mut offset = 0;
                let mut mats = Vec::new();
                let mut xs = Vec::new();
                for _ in 0..batch {
                    mats.push(&vals[offset..offset + mat_len]);
                    offset += mat_len;
                    xs.push(&vals[offset..offset + n]);
                    offset += n;
                }
                // Sequential
                let mut seq_outs: Vec<Vec<f32>> = (0..batch).map(|_| vec![0.0f32; m]).collect();
                for i in 0..batch {
                    gemv_f32(&cfg, mats[i], xs[i], &mut seq_outs[i]).unwrap();
                }
                // Batch
                let mut batch_outs: Vec<Vec<f32>> =
                    (0..batch).map(|_| vec![0.0f32; m]).collect();
                let mat_refs: Vec<&[f32]> = mats.to_vec();
                let x_refs: Vec<&[f32]> = xs.to_vec();
                let mut out_refs: Vec<&mut [f32]> =
                    batch_outs.iter_mut().map(|v| v.as_mut_slice()).collect();
                batch_gemv_f32(&cfg, &mat_refs, &x_refs, &mut out_refs).unwrap();
                for (i, (s, b)) in seq_outs.iter().zip(batch_outs.iter()).enumerate() {
                    for (j, (a, b)) in s.iter().zip(b.iter()).enumerate() {
                        prop_assert!(
                            (a - b).abs() < 1e-6,
                            "batch {i} elem {j}: {a} vs {b}"
                        );
                    }
                }
            }
        }

        // Property 5: quantized gemv with all-zero weights gives zero
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn quantized_zeros_give_zero(
                m in 1usize..=8,
                n in 1usize..=32,
                x_vals in proptest::collection::vec(arb_f32(), 1..=32),
                scale_vals in proptest::collection::vec(0.1f32..10.0, 1..=8),
            ) {
                if x_vals.len() < n || scale_vals.len() < m {
                    return Ok(());
                }
                let packed_cols = (n + 3) / 4;
                // All-zero weights: byte value 0b01010101 = 0x55 encodes all 0s
                // (code 1 → 0.0 for each 2-bit pair)
                let weights = vec![0x55u8; m * packed_cols];
                let scales = &scale_vals[..m];
                let x = &x_vals[..n];
                let mut y = vec![999.0f32; m];
                gemv_quantized_i2s(m, n, &weights, scales, x, &mut y).unwrap();
                for (i, &val) in y.iter().enumerate() {
                    prop_assert!(
                        val.abs() < 1e-6,
                        "expected ~0 at row {i}, got {val}"
                    );
                }
            }
        }
    }
}
