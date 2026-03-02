//! NEON-optimized General Matrix-Vector multiply (GEMV) for ARM.
//!
//! Provides `y = α·A·x + β·y` and transposed/quantized variants with 4×4 tile
//! processing for cache efficiency. Falls back to scalar code on non-NEON platforms.

use thiserror::Error;

// ── Types ───────────────────────────────────────────────────────────────────

/// Configuration for a GEMV operation.
#[derive(Debug, Clone, Copy)]
pub struct GemvConfig {
    /// Number of rows (output dimension).
    pub m: usize,
    /// Number of columns (input dimension).
    pub n: usize,
    /// Scalar multiplier for the matrix–vector product.
    pub alpha: f32,
    /// Scalar multiplier for the existing output vector.
    pub beta: f32,
    /// If `true`, operate on A^T instead of A.
    pub transpose: bool,
}

/// Errors specific to GEMV operations.
#[derive(Debug, Error, PartialEq)]
pub enum GemvError {
    #[error(
        "dimension mismatch: expected matrix {expected_rows}×{expected_cols}, got {actual} elements"
    )]
    DimensionMismatch { expected_rows: usize, expected_cols: usize, actual: usize },
    #[error("vector length mismatch: expected {expected}, got {actual}")]
    VectorLengthMismatch { expected: usize, actual: usize },
    #[error("output length mismatch: expected {expected}, got {actual}")]
    OutputLengthMismatch { expected: usize, actual: usize },
    #[error("batch size mismatch: configs={configs}, matrices={matrices}, vectors={vectors}")]
    BatchSizeMismatch { configs: usize, matrices: usize, vectors: usize },
    #[error("empty dimension: m={m}, n={n}")]
    EmptyDimension { m: usize, n: usize },
    #[error("scales length mismatch: expected {expected}, got {actual}")]
    ScalesLengthMismatch { expected: usize, actual: usize },
}

// ── Tile size ───────────────────────────────────────────────────────────────

const TILE: usize = 4;

// ── Validation helpers ──────────────────────────────────────────────────────

fn validate_gemv(
    config: &GemvConfig,
    matrix: &[f32],
    vector: &[f32],
    output: &[f32],
) -> Result<(), GemvError> {
    if config.m == 0 || config.n == 0 {
        return Err(GemvError::EmptyDimension { m: config.m, n: config.n });
    }
    if matrix.len() != config.m * config.n {
        return Err(GemvError::DimensionMismatch {
            expected_rows: config.m,
            expected_cols: config.n,
            actual: matrix.len(),
        });
    }
    let (vec_expect, out_expect) =
        if config.transpose { (config.m, config.n) } else { (config.n, config.m) };
    if vector.len() != vec_expect {
        return Err(GemvError::VectorLengthMismatch { expected: vec_expect, actual: vector.len() });
    }
    if output.len() != out_expect {
        return Err(GemvError::OutputLengthMismatch { expected: out_expect, actual: output.len() });
    }
    Ok(())
}

// ── Scalar (portable) implementation ────────────────────────────────────────

/// Scalar dot-product over a row/column slice.
#[inline]
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Scalar tiled GEMV: y = α·A·x + β·y
fn scalar_gemv(config: &GemvConfig, matrix: &[f32], vector: &[f32], output: &mut [f32]) {
    let (m, n) = (config.m, config.n);
    // Apply beta scaling
    for y in output.iter_mut() {
        *y *= config.beta;
    }
    // 4×4 tiled accumulation
    let row_tiles = m / TILE;
    let col_tiles = n / TILE;
    for rt in 0..row_tiles {
        let r0 = rt * TILE;
        for ct in 0..col_tiles {
            let c0 = ct * TILE;
            for ri in 0..TILE {
                let row = r0 + ri;
                let mut acc: f32 = 0.0;
                for ci in 0..TILE {
                    acc += matrix[row * n + c0 + ci] * vector[c0 + ci];
                }
                output[row] += config.alpha * acc;
            }
        }
        // Remainder columns
        let col_rem_start = col_tiles * TILE;
        for ri in 0..TILE {
            let row = r0 + ri;
            let mut acc: f32 = 0.0;
            for c in col_rem_start..n {
                acc += matrix[row * n + c] * vector[c];
            }
            output[row] += config.alpha * acc;
        }
    }
    // Remainder rows
    let row_rem_start = row_tiles * TILE;
    for (row, out) in output.iter_mut().enumerate().skip(row_rem_start) {
        let start = row * n;
        let dot = scalar_dot(&matrix[start..start + n], vector);
        *out += config.alpha * dot;
    }
}

/// Scalar tiled transposed GEMV: y = α·Aᵀ·x + β·y
fn scalar_gemv_transposed(config: &GemvConfig, matrix: &[f32], vector: &[f32], output: &mut [f32]) {
    let (m, n) = (config.m, config.n);
    for y in output.iter_mut() {
        *y *= config.beta;
    }
    let row_tiles = m / TILE;
    let col_tiles = n / TILE;
    for rt in 0..row_tiles {
        let r0 = rt * TILE;
        for ct in 0..col_tiles {
            let c0 = ct * TILE;
            for ci in 0..TILE {
                let col = c0 + ci;
                let mut acc: f32 = 0.0;
                for ri in 0..TILE {
                    acc += matrix[(r0 + ri) * n + col] * vector[r0 + ri];
                }
                output[col] += config.alpha * acc;
            }
        }
        let col_rem_start = col_tiles * TILE;
        for col in col_rem_start..n {
            let mut acc: f32 = 0.0;
            for ri in 0..TILE {
                acc += matrix[(r0 + ri) * n + col] * vector[r0 + ri];
            }
            output[col] += config.alpha * acc;
        }
    }
    let row_rem_start = row_tiles * TILE;
    for row in row_rem_start..m {
        for col in 0..n {
            output[col] += config.alpha * matrix[row * n + col] * vector[row];
        }
    }
}

/// Scalar quantized i8 × f32 GEMV with per-row scales.
fn scalar_gemv_i8(matrix: &[i8], scales: &[f32], vector: &[f32], output: &mut [f32]) {
    let m = output.len();
    let n = vector.len();
    for row in 0..m {
        let mut acc: f32 = 0.0;
        let base = row * n;
        for col in 0..n {
            acc += (matrix[base + col] as f32) * vector[col];
        }
        output[row] = scales[row] * acc;
    }
}

/// Scalar fused accumulate: accumulator += A · x
fn scalar_accumulate(matrix: &[f32], vector: &[f32], accumulator: &mut [f32]) {
    let n = vector.len();
    for (row, acc) in accumulator.iter_mut().enumerate() {
        let base = row * n;
        let dot = scalar_dot(&matrix[base..base + n], vector);
        *acc += dot;
    }
}

// ── NEON intrinsics path (aarch64 only) ─────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use super::*;
    use std::arch::aarch64::*;

    /// NEON tiled GEMV: y = α·A·x + β·y
    ///
    /// Processes 4 consecutive f32 lanes at a time with `vfmaq_f32`.
    #[allow(clippy::missing_safety_doc)]
    pub(super) unsafe fn neon_gemv_inner(
        config: &GemvConfig,
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
    ) {
        let (m, n) = (config.m, config.n);
        let beta_v = vdupq_n_f32(config.beta);
        let alpha = config.alpha;

        for row in 0..m {
            let base = row * n;
            // β · y[row]
            let mut acc = vmulq_f32(vdupq_n_f32(output[row]), beta_v);

            let chunks = n / 4;
            for c in 0..chunks {
                let off = c * 4;
                let a4 = vld1q_f32(matrix.as_ptr().add(base + off));
                let x4 = vld1q_f32(vector.as_ptr().add(off));
                acc = vfmaq_f32(acc, a4, vdupq_n_f32(alpha));
                // multiply a4*x4, then horizontal add below
                let prod = vmulq_f32(a4, x4);
                acc = vaddq_f32(acc, vmulq_f32(vdupq_n_f32(alpha), prod));
            }

            // Horizontal sum of acc lanes
            let sum = vaddvq_f32(acc);
            // Subtract triple-counted beta·y (we loaded it into all 4 lanes)
            output[row] = sum - 3.0 * config.beta * output[row];

            // Scalar remainder
            let rem_start = chunks * 4;
            for c in rem_start..n {
                output[row] += alpha * matrix[base + c] * vector[c];
            }
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Compute y = α·A·x + β·y.
///
/// Matrix A is `m × n` in row-major order, x has length `n`, y has length `m`.
pub fn gemv_f32(
    config: &GemvConfig,
    matrix: &[f32],
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    validate_gemv(&GemvConfig { transpose: false, ..*config }, matrix, vector, output)?;

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: lengths validated above; NEON is baseline on aarch64.
        unsafe { neon_impl::neon_gemv_inner(config, matrix, vector, output) };
        return Ok(());
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_gemv(config, matrix, vector, output);
        Ok(())
    }
}

/// Compute y = α·Aᵀ·x + β·y.
///
/// Matrix A is `m × n` in row-major order, x has length `m`, y has length `n`.
pub fn gemv_f32_transposed(
    config: &GemvConfig,
    matrix: &[f32],
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    validate_gemv(&GemvConfig { transpose: true, ..*config }, matrix, vector, output)?;
    scalar_gemv_transposed(config, matrix, vector, output);
    Ok(())
}

/// Quantized GEMV: output[i] = scales[i] · Σ_j (matrix_i8[i,j] · vector[j]).
///
/// `matrix_i8` is `m × n` row-major (m = output.len(), n = vector.len()).
pub fn gemv_i8_f32(
    matrix_i8: &[i8],
    scales: &[f32],
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let m = output.len();
    let n = vector.len();
    if m == 0 || n == 0 {
        return Err(GemvError::EmptyDimension { m, n });
    }
    if matrix_i8.len() != m * n {
        return Err(GemvError::DimensionMismatch {
            expected_rows: m,
            expected_cols: n,
            actual: matrix_i8.len(),
        });
    }
    if scales.len() != m {
        return Err(GemvError::ScalesLengthMismatch { expected: m, actual: scales.len() });
    }
    scalar_gemv_i8(matrix_i8, scales, vector, output);
    Ok(())
}

/// Batch GEMV: runs multiple independent GEMV operations and returns results.
pub fn batch_gemv(
    configs: &[GemvConfig],
    matrices: &[&[f32]],
    vectors: &[&[f32]],
) -> Result<Vec<Vec<f32>>, GemvError> {
    if configs.len() != matrices.len() || configs.len() != vectors.len() {
        return Err(GemvError::BatchSizeMismatch {
            configs: configs.len(),
            matrices: matrices.len(),
            vectors: vectors.len(),
        });
    }
    let mut results = Vec::with_capacity(configs.len());
    for (i, cfg) in configs.iter().enumerate() {
        let out_len = if cfg.transpose { cfg.n } else { cfg.m };
        let mut out = vec![0.0f32; out_len];
        if cfg.transpose {
            gemv_f32_transposed(cfg, matrices[i], vectors[i], &mut out)?;
        } else {
            gemv_f32(cfg, matrices[i], vectors[i], &mut out)?;
        }
        results.push(out);
    }
    Ok(results)
}

/// Fused accumulate: accumulator += A · x  (no α/β scaling).
///
/// Matrix A is `m × n` row-major. accumulator has length `m`, x has length `n`.
pub fn gemv_accumulate(
    matrix: &[f32],
    vector: &[f32],
    accumulator: &mut [f32],
) -> Result<(), GemvError> {
    let m = accumulator.len();
    let n = vector.len();
    if m == 0 || n == 0 {
        return Err(GemvError::EmptyDimension { m, n });
    }
    if matrix.len() != m * n {
        return Err(GemvError::DimensionMismatch {
            expected_rows: m,
            expected_cols: n,
            actual: matrix.len(),
        });
    }
    scalar_accumulate(matrix, vector, accumulator);
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!((ai - bi).abs() <= tol, "mismatch at index {i}: {ai} vs {bi} (tol={tol})");
        }
    }

    // Naive reference implementation (no tiling, no SIMD)
    fn ref_gemv(
        m: usize,
        n: usize,
        alpha: f32,
        matrix: &[f32],
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) {
        for i in 0..m {
            y[i] *= beta;
            let mut dot = 0.0f32;
            for j in 0..n {
                dot += matrix[i * n + j] * x[j];
            }
            y[i] += alpha * dot;
        }
    }

    fn ref_gemv_t(
        m: usize,
        n: usize,
        alpha: f32,
        matrix: &[f32],
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) {
        for j in 0..n {
            y[j] *= beta;
        }
        for i in 0..m {
            for j in 0..n {
                y[j] += alpha * matrix[i * n + j] * x[i];
            }
        }
    }

    // ── Basic correctness ──────────────────────────────────────────────

    #[test]
    fn test_identity_gemv() {
        let config = GemvConfig { m: 3, n: 3, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let vector = vec![2.0, 3.0, 4.0];
        let mut output = vec![0.0; 3];
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &[2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_gemv_2x3() {
        let config = GemvConfig { m: 2, n: 3, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = vec![1.0, 1.0, 1.0];
        let mut output = vec![0.0; 2];
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &[6.0, 15.0], 1e-6);
    }

    #[test]
    fn test_gemv_alpha_scaling() {
        let config = GemvConfig { m: 2, n: 2, alpha: 2.0, beta: 0.0, transpose: false };
        let matrix = vec![1.0, 2.0, 3.0, 4.0];
        let vector = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &[6.0, 14.0], 1e-6);
    }

    #[test]
    fn test_gemv_beta_scaling() {
        let config = GemvConfig { m: 2, n: 2, alpha: 1.0, beta: 2.0, transpose: false };
        let matrix = vec![1.0, 0.0, 0.0, 1.0];
        let vector = vec![1.0, 1.0];
        let mut output = vec![10.0, 20.0];
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        // y = 1.0 * I * [1,1] + 2.0 * [10,20] = [1,1] + [20,40] = [21,41]
        approx_eq(&output, &[21.0, 41.0], 1e-6);
    }

    #[test]
    fn test_gemv_alpha_and_beta() {
        let config = GemvConfig { m: 2, n: 2, alpha: 0.5, beta: 0.5, transpose: false };
        let matrix = vec![2.0, 0.0, 0.0, 2.0];
        let vector = vec![4.0, 6.0];
        let mut output = vec![10.0, 20.0];
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        // y = 0.5 * [8,12] + 0.5 * [10,20] = [4,6] + [5,10] = [9,16]
        approx_eq(&output, &[9.0, 16.0], 1e-6);
    }

    #[test]
    fn test_gemv_1x1() {
        let config = GemvConfig { m: 1, n: 1, alpha: 1.0, beta: 0.0, transpose: false };
        let mut output = vec![0.0];
        gemv_f32(&config, &[3.0], &[4.0], &mut output).unwrap();
        approx_eq(&output, &[12.0], 1e-6);
    }

    #[test]
    fn test_gemv_single_row() {
        let config = GemvConfig { m: 1, n: 5, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vector = vec![1.0; 5];
        let mut output = vec![0.0];
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &[15.0], 1e-6);
    }

    #[test]
    fn test_gemv_single_column() {
        let config = GemvConfig { m: 4, n: 1, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix = vec![2.0, 3.0, 4.0, 5.0];
        let vector = vec![3.0];
        let mut output = vec![0.0; 4];
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &[6.0, 9.0, 12.0, 15.0], 1e-6);
    }

    // ── Tile-boundary tests ────────────────────────────────────────────

    #[test]
    fn test_gemv_exact_tile_4x4() {
        let m = 4;
        let n = 4;
        let config = GemvConfig { m, n, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let vector = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        let mut expected = vec![0.0; 4];
        ref_gemv(m, n, 1.0, &matrix, &vector, 0.0, &mut expected);
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-5);
    }

    #[test]
    fn test_gemv_tile_remainder_5x5() {
        let m = 5;
        let n = 5;
        let config = GemvConfig { m, n, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix: Vec<f32> = (0..25).map(|i| (i as f32) * 0.1).collect();
        let vector: Vec<f32> = (0..5).map(|i| (i + 1) as f32).collect();
        let mut output = vec![0.0; 5];
        let mut expected = vec![0.0; 5];
        ref_gemv(m, n, 1.0, &matrix, &vector, 0.0, &mut expected);
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_gemv_tile_8x8() {
        let m = 8;
        let n = 8;
        let config = GemvConfig { m, n, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix: Vec<f32> = (0..64).map(|i| ((i % 7) as f32) - 3.0).collect();
        let vector: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5).collect();
        let mut output = vec![0.0; 8];
        let mut expected = vec![0.0; 8];
        ref_gemv(m, n, 1.0, &matrix, &vector, 0.0, &mut expected);
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_gemv_tile_9x7() {
        let (m, n) = (9, 7);
        let config = GemvConfig { m, n, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.01).collect();
        let vector: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
        let mut output = vec![0.0; m];
        let mut expected = vec![0.0; m];
        ref_gemv(m, n, 1.0, &matrix, &vector, 0.0, &mut expected);
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-3);
    }

    #[test]
    fn test_gemv_large_16x16() {
        let (m, n) = (16, 16);
        let config = GemvConfig { m, n, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix: Vec<f32> = (0..(m * n)).map(|i| ((i as f32) * 0.37).sin()).collect();
        let vector: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.23).cos()).collect();
        let mut output = vec![0.0; m];
        let mut expected = vec![0.0; m];
        ref_gemv(m, n, 1.0, &matrix, &vector, 0.0, &mut expected);
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-4);
    }

    // ── Transpose tests ────────────────────────────────────────────────

    #[test]
    fn test_transposed_identity() {
        let config = GemvConfig { m: 3, n: 3, alpha: 1.0, beta: 0.0, transpose: true };
        let matrix = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let vector = vec![5.0, 6.0, 7.0];
        let mut output = vec![0.0; 3];
        gemv_f32_transposed(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &[5.0, 6.0, 7.0], 1e-6);
    }

    #[test]
    fn test_transposed_2x3() {
        let config = GemvConfig { m: 2, n: 3, alpha: 1.0, beta: 0.0, transpose: true };
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = vec![1.0, 1.0]; // length m=2
        let mut output = vec![0.0; 3]; // length n=3
        gemv_f32_transposed(&config, &matrix, &vector, &mut output).unwrap();
        // A^T * x = [1*1+4*1, 2*1+5*1, 3*1+6*1] = [5, 7, 9]
        approx_eq(&output, &[5.0, 7.0, 9.0], 1e-6);
    }

    #[test]
    fn test_transposed_alpha_beta() {
        let config = GemvConfig { m: 2, n: 2, alpha: 2.0, beta: 3.0, transpose: true };
        let matrix = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let vector = vec![1.0, 1.0];
        let mut output = vec![10.0, 20.0];
        gemv_f32_transposed(&config, &matrix, &vector, &mut output).unwrap();
        // y = 2 * I^T * [1,1] + 3 * [10,20] = [2,2] + [30,60] = [32,62]
        approx_eq(&output, &[32.0, 62.0], 1e-6);
    }

    #[test]
    fn test_transposed_tile_boundary_5x6() {
        let (m, n) = (5, 6);
        let config = GemvConfig { m, n, alpha: 1.0, beta: 0.0, transpose: true };
        let matrix: Vec<f32> = (0..(m * n)).map(|i| i as f32).collect();
        let vector: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();
        let mut output = vec![0.0; n];
        let mut expected = vec![0.0; n];
        ref_gemv_t(m, n, 1.0, &matrix, &vector, 0.0, &mut expected);
        gemv_f32_transposed(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-3);
    }

    #[test]
    fn test_transposed_8x8() {
        let (m, n) = (8, 8);
        let config = GemvConfig { m, n, alpha: 0.5, beta: 0.0, transpose: true };
        let matrix: Vec<f32> = (0..64).map(|i| (i as f32) - 32.0).collect();
        let vector: Vec<f32> = (0..m).map(|i| (i as f32) * 0.25).collect();
        let mut output = vec![0.0; n];
        let mut expected = vec![0.0; n];
        ref_gemv_t(m, n, 0.5, &matrix, &vector, 0.0, &mut expected);
        gemv_f32_transposed(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-3);
    }

    // ── Quantized i8 tests ─────────────────────────────────────────────

    #[test]
    fn test_i8_gemv_basic() {
        let matrix_i8: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let scales = vec![1.0, 1.0];
        let vector = vec![1.0, 1.0, 1.0];
        let mut output = vec![0.0; 2];
        gemv_i8_f32(&matrix_i8, &scales, &vector, &mut output).unwrap();
        approx_eq(&output, &[6.0, 15.0], 1e-6);
    }

    #[test]
    fn test_i8_gemv_with_scales() {
        let matrix_i8: Vec<i8> = vec![1, 0, 0, 1];
        let scales = vec![2.0, 3.0];
        let vector = vec![5.0, 7.0];
        let mut output = vec![0.0; 2];
        gemv_i8_f32(&matrix_i8, &scales, &vector, &mut output).unwrap();
        approx_eq(&output, &[10.0, 21.0], 1e-6);
    }

    #[test]
    fn test_i8_gemv_negative_values() {
        let matrix_i8: Vec<i8> = vec![-1, 1, 1, -1];
        let scales = vec![1.0, 1.0];
        let vector = vec![3.0, 5.0];
        let mut output = vec![0.0; 2];
        gemv_i8_f32(&matrix_i8, &scales, &vector, &mut output).unwrap();
        approx_eq(&output, &[2.0, -2.0], 1e-6);
    }

    #[test]
    fn test_i8_gemv_1x1() {
        let mut output = vec![0.0];
        gemv_i8_f32(&[-3], &[2.0], &[4.0], &mut output).unwrap();
        approx_eq(&output, &[-24.0], 1e-6);
    }

    #[test]
    fn test_i8_gemv_large_4x8() {
        let (m, n) = (4, 8);
        let matrix_i8: Vec<i8> = (0..(m * n) as i8).map(|i| i - 16).collect();
        let scales = vec![0.5; m];
        let vector: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let mut output = vec![0.0; m];
        // Reference
        let mut expected = vec![0.0; m];
        for row in 0..m {
            let mut acc = 0.0f32;
            for col in 0..n {
                acc += (matrix_i8[row * n + col] as f32) * vector[col];
            }
            expected[row] = scales[row] * acc;
        }
        gemv_i8_f32(&matrix_i8, &scales, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-4);
    }

    // ── Accumulate tests ───────────────────────────────────────────────

    #[test]
    fn test_accumulate_basic() {
        let matrix = vec![1.0, 0.0, 0.0, 1.0];
        let vector = vec![3.0, 5.0];
        let mut acc = vec![10.0, 20.0];
        gemv_accumulate(&matrix, &vector, &mut acc).unwrap();
        approx_eq(&acc, &[13.0, 25.0], 1e-6);
    }

    #[test]
    fn test_accumulate_2x3() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = vec![1.0, 1.0, 1.0];
        let mut acc = vec![100.0, 200.0];
        gemv_accumulate(&matrix, &vector, &mut acc).unwrap();
        approx_eq(&acc, &[106.0, 215.0], 1e-6);
    }

    #[test]
    fn test_accumulate_zeros() {
        let matrix = vec![0.0; 9];
        let vector = vec![1.0, 2.0, 3.0];
        let mut acc = vec![7.0, 8.0, 9.0];
        gemv_accumulate(&matrix, &vector, &mut acc).unwrap();
        approx_eq(&acc, &[7.0, 8.0, 9.0], 1e-6);
    }

    #[test]
    fn test_accumulate_multiple_calls() {
        let matrix = vec![1.0, 0.0, 0.0, 1.0];
        let vector = vec![1.0, 1.0];
        let mut acc = vec![0.0, 0.0];
        for _ in 0..5 {
            gemv_accumulate(&matrix, &vector, &mut acc).unwrap();
        }
        approx_eq(&acc, &[5.0, 5.0], 1e-6);
    }

    // ── Batch GEMV tests ───────────────────────────────────────────────

    #[test]
    fn test_batch_gemv_single() {
        let cfgs = vec![GemvConfig { m: 2, n: 2, alpha: 1.0, beta: 0.0, transpose: false }];
        let m = vec![1.0f32, 0.0, 0.0, 1.0];
        let v = vec![3.0f32, 4.0];
        let results = batch_gemv(&cfgs, &[&m], &[&v]).unwrap();
        assert_eq!(results.len(), 1);
        approx_eq(&results[0], &[3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_batch_gemv_multiple() {
        let cfgs = vec![
            GemvConfig { m: 2, n: 2, alpha: 1.0, beta: 0.0, transpose: false },
            GemvConfig { m: 1, n: 3, alpha: 2.0, beta: 0.0, transpose: false },
        ];
        let m1 = vec![1.0f32, 0.0, 0.0, 1.0];
        let m2 = vec![1.0f32, 2.0, 3.0];
        let v1 = vec![5.0f32, 6.0];
        let v2 = vec![1.0f32, 1.0, 1.0];
        let results = batch_gemv(&cfgs, &[&m1, &m2], &[&v1, &v2]).unwrap();
        assert_eq!(results.len(), 2);
        approx_eq(&results[0], &[5.0, 6.0], 1e-6);
        approx_eq(&results[1], &[12.0], 1e-6);
    }

    #[test]
    fn test_batch_gemv_with_transpose() {
        let cfgs = vec![GemvConfig { m: 2, n: 3, alpha: 1.0, beta: 0.0, transpose: true }];
        let m = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = vec![1.0f32, 1.0];
        let results = batch_gemv(&cfgs, &[&m], &[&v]).unwrap();
        approx_eq(&results[0], &[5.0, 7.0, 9.0], 1e-6);
    }

    // ── Error / edge-case tests ────────────────────────────────────────

    #[test]
    fn test_error_dimension_mismatch() {
        let config = GemvConfig { m: 2, n: 3, alpha: 1.0, beta: 0.0, transpose: false };
        let result = gemv_f32(&config, &[1.0; 5], &[1.0; 3], &mut [0.0; 2]);
        assert!(matches!(result, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_error_vector_length() {
        let config = GemvConfig { m: 2, n: 3, alpha: 1.0, beta: 0.0, transpose: false };
        let result = gemv_f32(&config, &[1.0; 6], &[1.0; 2], &mut [0.0; 2]);
        assert!(matches!(result, Err(GemvError::VectorLengthMismatch { .. })));
    }

    #[test]
    fn test_error_output_length() {
        let config = GemvConfig { m: 2, n: 3, alpha: 1.0, beta: 0.0, transpose: false };
        let result = gemv_f32(&config, &[1.0; 6], &[1.0; 3], &mut [0.0; 5]);
        assert!(matches!(result, Err(GemvError::OutputLengthMismatch { .. })));
    }

    #[test]
    fn test_error_empty_dimension() {
        let config = GemvConfig { m: 0, n: 3, alpha: 1.0, beta: 0.0, transpose: false };
        let result = gemv_f32(&config, &[], &[1.0; 3], &mut []);
        assert!(matches!(result, Err(GemvError::EmptyDimension { .. })));
    }

    #[test]
    fn test_error_batch_size_mismatch() {
        let cfgs = vec![GemvConfig { m: 2, n: 2, alpha: 1.0, beta: 0.0, transpose: false }];
        let m = vec![1.0f32; 4];
        let result = batch_gemv(&cfgs, &[&m[..], &m[..]], &[&[1.0f32; 2]]);
        assert!(matches!(result, Err(GemvError::BatchSizeMismatch { .. })));
    }

    #[test]
    fn test_error_i8_dimension_mismatch() {
        let result = gemv_i8_f32(&[1i8; 5], &[1.0; 2], &[1.0; 3], &mut [0.0; 2]);
        assert!(matches!(result, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_error_i8_scales_mismatch() {
        let result = gemv_i8_f32(&[1i8; 6], &[1.0; 3], &[1.0; 3], &mut [0.0; 2]);
        assert!(matches!(result, Err(GemvError::ScalesLengthMismatch { .. })));
    }

    #[test]
    fn test_error_accumulate_dimension_mismatch() {
        let result = gemv_accumulate(&[1.0; 5], &[1.0; 3], &mut [0.0; 2]);
        assert!(matches!(result, Err(GemvError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_error_accumulate_empty() {
        let result = gemv_accumulate(&[], &[], &mut []);
        assert!(matches!(result, Err(GemvError::EmptyDimension { .. })));
    }

    #[test]
    fn test_error_i8_empty() {
        let result = gemv_i8_f32(&[], &[], &[], &mut []);
        assert!(matches!(result, Err(GemvError::EmptyDimension { .. })));
    }

    // ── Numerical edge cases ───────────────────────────────────────────

    #[test]
    fn test_zero_alpha() {
        let config = GemvConfig { m: 2, n: 2, alpha: 0.0, beta: 1.0, transpose: false };
        let mut output = vec![7.0, 8.0];
        gemv_f32(&config, &[999.0; 4], &[999.0; 2], &mut output).unwrap();
        approx_eq(&output, &[7.0, 8.0], 1e-6);
    }

    #[test]
    fn test_zero_beta() {
        let config = GemvConfig { m: 2, n: 2, alpha: 1.0, beta: 0.0, transpose: false };
        let mut output = vec![999.0, 888.0];
        gemv_f32(&config, &[1.0, 0.0, 0.0, 1.0], &[3.0, 4.0], &mut output).unwrap();
        approx_eq(&output, &[3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_negative_values() {
        let config = GemvConfig { m: 2, n: 2, alpha: 1.0, beta: 0.0, transpose: false };
        let matrix = vec![-1.0, 2.0, 3.0, -4.0];
        let vector = vec![1.0, -1.0];
        let mut output = vec![0.0; 2];
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &[-3.0, 7.0], 1e-6);
    }

    #[test]
    fn test_all_zeros_matrix() {
        let config = GemvConfig { m: 3, n: 3, alpha: 1.0, beta: 0.0, transpose: false };
        let mut output = vec![0.0; 3];
        gemv_f32(&config, &[0.0; 9], &[1.0, 2.0, 3.0], &mut output).unwrap();
        approx_eq(&output, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ── Reference parity ───────────────────────────────────────────────

    #[test]
    fn test_parity_gemv_vs_reference_7x11() {
        let (m, n) = (7, 11);
        let config = GemvConfig { m, n, alpha: 1.5, beta: 0.25, transpose: false };
        let matrix: Vec<f32> = (0..(m * n)).map(|i| ((i as f32) * 0.13).sin()).collect();
        let vector: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.77).cos()).collect();
        let mut output = vec![1.0; m];
        let mut expected = vec![1.0; m];
        ref_gemv(m, n, config.alpha, &matrix, &vector, config.beta, &mut expected);
        gemv_f32(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_parity_transposed_vs_reference_6x10() {
        let (m, n) = (6, 10);
        let config = GemvConfig { m, n, alpha: 0.8, beta: 0.2, transpose: true };
        let matrix: Vec<f32> = (0..(m * n)).map(|i| ((i as f32) * 0.29).cos()).collect();
        let vector: Vec<f32> = (0..m).map(|i| ((i as f32) + 1.0).sqrt()).collect();
        let mut output = vec![0.5; n];
        let mut expected = vec![0.5; n];
        ref_gemv_t(m, n, config.alpha, &matrix, &vector, config.beta, &mut expected);
        gemv_f32_transposed(&config, &matrix, &vector, &mut output).unwrap();
        approx_eq(&output, &expected, 1e-4);
    }

    // ── Display / Debug coverage ───────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = GemvError::DimensionMismatch { expected_rows: 2, expected_cols: 3, actual: 5 };
        let msg = format!("{e}");
        assert!(msg.contains("dimension mismatch"));
    }

    #[test]
    fn test_config_debug() {
        let config = GemvConfig { m: 2, n: 3, alpha: 1.0, beta: 0.0, transpose: false };
        let dbg = format!("{config:?}");
        assert!(dbg.contains("GemvConfig"));
    }

    // ── Proptest properties ────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        // Strategy: dimensions 1..=32
        fn dim() -> impl Strategy<Value = usize> {
            1usize..=32
        }

        fn scalar() -> impl Strategy<Value = f32> {
            -10.0f32..10.0f32
        }

        proptest! {
            /// y = α·I·x + 0·y  must equal  α·x  for any square identity matrix.
            #[test]
            fn prop_identity_scales_vector(
                n in dim(),
                alpha in scalar(),
            ) {
                let config = GemvConfig { m: n, n, alpha, beta: 0.0, transpose: false };
                let mut identity = vec![0.0f32; n * n];
                for i in 0..n { identity[i * n + i] = 1.0; }
                let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let mut y = vec![0.0f32; n];
                gemv_f32(&config, &identity, &x, &mut y).unwrap();
                let expected: Vec<f32> = x.iter().map(|xi| alpha * xi).collect();
                approx_eq(&y, &expected, 1e-4);
            }

            /// A zero matrix always produces β·y_init regardless of x.
            #[test]
            fn prop_zero_matrix_preserves_beta(
                m in dim(),
                n in dim(),
                beta in scalar(),
            ) {
                let config = GemvConfig { m, n, alpha: 1.0, beta, transpose: false };
                let matrix = vec![0.0f32; m * n];
                let x: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
                let y_init: Vec<f32> = (0..m).map(|i| (i as f32) * 0.5).collect();
                let mut y = y_init.clone();
                gemv_f32(&config, &matrix, &x, &mut y).unwrap();
                let expected: Vec<f32> = y_init.iter().map(|yi| beta * yi).collect();
                approx_eq(&y, &expected, 1e-4);
            }

            /// gemv matches the naive reference for arbitrary dimensions.
            #[test]
            fn prop_gemv_matches_reference(
                m in 1usize..=16,
                n in 1usize..=16,
            ) {
                let config = GemvConfig { m, n, alpha: 1.0, beta: 0.0, transpose: false };
                let matrix: Vec<f32> = (0..(m * n)).map(|i| ((i as f32) * 0.17).sin()).collect();
                let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.31).cos()).collect();
                let mut y = vec![0.0f32; m];
                let mut expected = vec![0.0f32; m];
                ref_gemv(m, n, 1.0, &matrix, &x, 0.0, &mut expected);
                gemv_f32(&config, &matrix, &x, &mut y).unwrap();
                approx_eq(&y, &expected, 1e-3);
            }

            /// Transposed gemv matches the naive reference.
            #[test]
            fn prop_transposed_matches_reference(
                m in 1usize..=16,
                n in 1usize..=16,
            ) {
                let config = GemvConfig { m, n, alpha: 1.0, beta: 0.0, transpose: true };
                let matrix: Vec<f32> = (0..(m * n)).map(|i| ((i as f32) * 0.23).cos()).collect();
                let x: Vec<f32> = (0..m).map(|i| ((i as f32) * 0.41).sin()).collect();
                let mut y = vec![0.0f32; n];
                let mut expected = vec![0.0f32; n];
                ref_gemv_t(m, n, 1.0, &matrix, &x, 0.0, &mut expected);
                gemv_f32_transposed(&config, &matrix, &x, &mut y).unwrap();
                approx_eq(&y, &expected, 1e-3);
            }

            /// accumulate(A, x, acc) adds exactly A·x to acc.
            #[test]
            fn prop_accumulate_adds_product(
                m in 1usize..=16,
                n in 1usize..=16,
            ) {
                let matrix: Vec<f32> = (0..(m * n)).map(|i| ((i as f32) * 0.19).sin()).collect();
                let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.53).cos()).collect();
                let init: Vec<f32> = (0..m).map(|i| (i as f32) * 0.7).collect();
                let mut acc = init.clone();
                gemv_accumulate(&matrix, &x, &mut acc).unwrap();
                // expected = init + A·x
                let mut product = vec![0.0f32; m];
                ref_gemv(m, n, 1.0, &matrix, &x, 0.0, &mut product);
                let expected: Vec<f32> = init.iter().zip(product.iter()).map(|(a, b)| a + b).collect();
                approx_eq(&acc, &expected, 1e-3);
            }
        }
    }
}
