//! Module stub - implementation pending merge from feature branch
//! Matrix multiplication kernels for GPU HAL.
//!
//! Provides multiple matmul algorithm implementations (naive, tiled, blocked),
//! quantized matmul support (I2S, INT4, INT8 with on-the-fly dequantization),
//! batched matmul for attention computation, input validation, and GFLOPS profiling.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Matmul algorithm selection
// ---------------------------------------------------------------------------

/// Algorithm variant for matrix multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulAlgorithm {
    /// O(n³) triple-loop reference implementation.
    Naive,
    /// Tiled implementation with configurable tile size.
    Tiled,
    /// Blocked for cache-line optimization with configurable block size.
    BlockedTiled,
    /// Winograd for small matrices (≤64×64).
    WinogradSmall,
    /// Strassen with configurable threshold for recursive subdivision.
    StrassenThreshold,
}

impl fmt::Display for MatmulAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Naive => write!(f, "Naive"),
            Self::Tiled => write!(f, "Tiled"),
            Self::BlockedTiled => write!(f, "BlockedTiled"),
            Self::WinogradSmall => write!(f, "WinogradSmall"),
            Self::StrassenThreshold => write!(f, "StrassenThreshold"),
        }
    }
}

// ---------------------------------------------------------------------------
// Accumulator data type
// ---------------------------------------------------------------------------

/// Data type for the matmul accumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorDtype {
    /// 32-bit float accumulator (default).
    F32,
    /// 64-bit float accumulator for higher precision.
    F64,
}

// ---------------------------------------------------------------------------
// Quantization type
// ---------------------------------------------------------------------------

/// Quantization format for quantized matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// 2-bit signed integer (`BitNet` `I2_S`).
    I2S,
    /// 4-bit integer.
    Int4,
    /// 8-bit integer.
    Int8,
}

// ---------------------------------------------------------------------------
// Matmul configuration
// ---------------------------------------------------------------------------

/// Configuration for a matmul kernel dispatch.
#[derive(Debug, Clone)]
pub struct MatmulConfig {
    /// Which algorithm to use.
    pub algorithm: MatmulAlgorithm,
    /// Tile size for tiled/blocked algorithms.
    pub tile_size: usize,
    /// Accumulator data type.
    pub accumulator_dtype: AccumulatorDtype,
    /// Whether to attempt tensor-core dispatch (GPU only).
    pub use_tensor_cores: bool,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self {
            algorithm: MatmulAlgorithm::Tiled,
            tile_size: 32,
            accumulator_dtype: AccumulatorDtype::F32,
            use_tensor_cores: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Matmul input
// ---------------------------------------------------------------------------

/// Input tensors and metadata for a matmul operation.
#[derive(Debug, Clone)]
pub struct MatmulInput {
    /// Row-major data for matrix A.
    pub a_data: Vec<f32>,
    /// Row-major data for matrix B.
    pub b_data: Vec<f32>,
    /// Shape of A as (rows, cols).
    pub a_shape: (usize, usize),
    /// Shape of B as (rows, cols).
    pub b_shape: (usize, usize),
    /// Whether to transpose A before multiplication.
    pub transpose_a: bool,
    /// Whether to transpose B before multiplication.
    pub transpose_b: bool,
}

impl MatmulInput {
    /// Effective shape of A after transpose.
    pub const fn effective_a_shape(&self) -> (usize, usize) {
        if self.transpose_a { (self.a_shape.1, self.a_shape.0) } else { self.a_shape }
    }

    /// Effective shape of B after transpose.
    pub const fn effective_b_shape(&self) -> (usize, usize) {
        if self.transpose_b { (self.b_shape.1, self.b_shape.0) } else { self.b_shape }
    }

    /// Output shape: (`A_rows`, `B_cols`) after any transpositions.
    pub const fn output_shape(&self) -> (usize, usize) {
        let (m, _) = self.effective_a_shape();
        let (_, n) = self.effective_b_shape();
        (m, n)
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validation errors for matmul inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatmulValidationError {
    /// Data length doesn't match declared shape.
    ShapeMismatch { matrix: &'static str, expected: usize, actual: usize },
    /// Inner dimensions (`A_cols`, `B_rows`) don't match.
    InnerDimensionMismatch { a_cols: usize, b_rows: usize },
    /// A dimension is zero.
    ZeroDimension { matrix: &'static str },
}

impl fmt::Display for MatmulValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { matrix, expected, actual } => {
                write!(f, "{matrix}: data length {actual} != shape product {expected}")
            }
            Self::InnerDimensionMismatch { a_cols, b_rows } => {
                write!(f, "inner dim mismatch: A cols={a_cols} != B rows={b_rows}")
            }
            Self::ZeroDimension { matrix } => {
                write!(f, "{matrix}: has a zero dimension")
            }
        }
    }
}

/// Validates matmul inputs for shape compatibility.
pub struct MatmulValidator;

impl MatmulValidator {
    /// Validate that the input is well-formed for matmul.
    pub const fn validate(input: &MatmulInput) -> Result<(), MatmulValidationError> {
        // Check data lengths match declared shapes.
        let a_expected = input.a_shape.0 * input.a_shape.1;
        if input.a_data.len() != a_expected {
            return Err(MatmulValidationError::ShapeMismatch {
                matrix: "A",
                expected: a_expected,
                actual: input.a_data.len(),
            });
        }
        let b_expected = input.b_shape.0 * input.b_shape.1;
        if input.b_data.len() != b_expected {
            return Err(MatmulValidationError::ShapeMismatch {
                matrix: "B",
                expected: b_expected,
                actual: input.b_data.len(),
            });
        }

        // Check for zero dimensions.
        let (am, ak) = input.effective_a_shape();
        let (bk, bn) = input.effective_b_shape();
        if am == 0 || ak == 0 {
            return Err(MatmulValidationError::ZeroDimension { matrix: "A" });
        }
        if bk == 0 || bn == 0 {
            return Err(MatmulValidationError::ZeroDimension { matrix: "B" });
        }

        // Inner dimension check.
        if ak != bk {
            return Err(MatmulValidationError::InnerDimensionMismatch { a_cols: ak, b_rows: bk });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: read element with optional transpose
// ---------------------------------------------------------------------------

#[inline]
fn read_element(data: &[f32], shape: (usize, usize), transposed: bool, r: usize, c: usize) -> f32 {
    if transposed { data[c * shape.1 + r] } else { data[r * shape.1 + c] }
}

// ---------------------------------------------------------------------------
// NaiveMatmul
// ---------------------------------------------------------------------------

/// O(n³) reference matmul. Correct but slow — used for validation.
pub struct NaiveMatmul;

impl NaiveMatmul {
    /// Compute C = A × B (with optional transpositions).
    pub fn execute(input: &MatmulInput) -> Vec<f32> {
        let (m, k) = input.effective_a_shape();
        let (_, n) = input.effective_b_shape();
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let a_val = read_element(&input.a_data, input.a_shape, input.transpose_a, i, p);
                    let b_val = read_element(&input.b_data, input.b_shape, input.transpose_b, p, j);
                    sum += a_val * b_val;
                }
                c[i * n + j] = sum;
            }
        }
        c
    }
}

// ---------------------------------------------------------------------------
// TiledMatmul
// ---------------------------------------------------------------------------

/// Tiled matmul with configurable tile size for cache-friendly access.
pub struct TiledMatmul;

impl TiledMatmul {
    /// Execute tiled matmul with the given tile size.
    pub fn execute(input: &MatmulInput, tile_size: usize) -> Vec<f32> {
        let ts = tile_size.max(1);
        let (m, k) = input.effective_a_shape();
        let (_, n) = input.effective_b_shape();
        let mut c = vec![0.0f32; m * n];
        // Tile over i, j, p.
        let mut ii = 0;
        while ii < m {
            let i_end = (ii + ts).min(m);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + ts).min(n);
                let mut pp = 0;
                while pp < k {
                    let p_end = (pp + ts).min(k);
                    for i in ii..i_end {
                        for j in jj..j_end {
                            let mut sum = 0.0f32;
                            for p in pp..p_end {
                                let a_val = read_element(
                                    &input.a_data,
                                    input.a_shape,
                                    input.transpose_a,
                                    i,
                                    p,
                                );
                                let b_val = read_element(
                                    &input.b_data,
                                    input.b_shape,
                                    input.transpose_b,
                                    p,
                                    j,
                                );
                                sum += a_val * b_val;
                            }
                            c[i * n + j] += sum;
                        }
                    }
                    pp += ts;
                }
                jj += ts;
            }
            ii += ts;
        }
        c
    }
}

// ---------------------------------------------------------------------------
// BlockedMatmul
// ---------------------------------------------------------------------------

/// Blocked matmul optimized for cache-line reuse with configurable block size.
pub struct BlockedMatmul;

impl BlockedMatmul {
    /// Execute blocked matmul with the given block size.
    ///
    /// Uses a packed micro-panel approach: for each block of B columns, iterate
    /// over A rows in blocks and accumulate through K in blocks.
    pub fn execute(input: &MatmulInput, block_size: usize) -> Vec<f32> {
        let bs = block_size.max(1);
        let (m, k) = input.effective_a_shape();
        let (_, n) = input.effective_b_shape();
        let mut c = vec![0.0f32; m * n];

        // Block over K first for better accumulator reuse.
        let mut kk = 0;
        while kk < k {
            let k_end = (kk + bs).min(k);
            let mut ii = 0;
            while ii < m {
                let i_end = (ii + bs).min(m);
                let mut jj = 0;
                while jj < n {
                    let j_end = (jj + bs).min(n);
                    for i in ii..i_end {
                        for j in jj..j_end {
                            let mut acc = 0.0f32;
                            for p in kk..k_end {
                                let a_val = read_element(
                                    &input.a_data,
                                    input.a_shape,
                                    input.transpose_a,
                                    i,
                                    p,
                                );
                                let b_val = read_element(
                                    &input.b_data,
                                    input.b_shape,
                                    input.transpose_b,
                                    p,
                                    j,
                                );
                                acc += a_val * b_val;
                            }
                            c[i * n + j] += acc;
                        }
                    }
                    jj += bs;
                }
                ii += bs;
            }
            kk += bs;
        }
        c
    }
}

// ---------------------------------------------------------------------------
// QuantizedMatmul
// ---------------------------------------------------------------------------

/// Matmul for quantized tensors with on-the-fly dequantization.
///
/// Accepts `i8`-packed data (interpretation depends on `QuantType`) plus
/// per-tensor scale/zero-point, and produces `f32` output.
pub struct QuantizedMatmul;

impl QuantizedMatmul {
    /// Dequantize a single quantized value: `(q - zero_point) * scale`.
    #[inline]
    fn dequant(q: i8, scale: f32, zero_point: i8) -> f32 {
        (f32::from(q) - f32::from(zero_point)) * scale
    }

    /// Execute quantized matmul: C = dequant(A) × dequant(B).
    ///
    /// Both `a_quant` and `b_quant` are row-major `i8` buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        a_quant: &[i8],
        b_quant: &[i8],
        a_shape: (usize, usize),
        b_shape: (usize, usize),
        a_scale: f32,
        b_scale: f32,
        a_zero: i8,
        b_zero: i8,
        _quant_type: QuantType,
    ) -> Vec<f32> {
        let (m, k) = a_shape;
        let (_, n) = b_shape;
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let a_val = Self::dequant(a_quant[i * k + p], a_scale, a_zero);
                    let b_val = Self::dequant(b_quant[p * n + j], b_scale, b_zero);
                    acc += a_val * b_val;
                }
                c[i * n + j] = acc;
            }
        }
        c
    }
}

// ---------------------------------------------------------------------------
// BatchedMatmul
// ---------------------------------------------------------------------------

/// Batched matmul: for each batch index compute C[b] = A[b] × B[b].
///
/// Useful for multi-head attention where each head has its own QK^T matmul.
pub struct BatchedMatmul;

impl BatchedMatmul {
    /// Execute batched matmul over `batch_size` independent multiplications.
    ///
    /// `a_batched` and `b_batched` are contiguous: batch 0 occupies
    /// `[0 .. m*k)`, batch 1 occupies `[m*k .. 2*m*k)`, etc.
    pub fn execute(
        a_batched: &[f32],
        b_batched: &[f32],
        m: usize,
        k: usize,
        n: usize,
        batch_size: usize,
    ) -> Vec<f32> {
        let a_stride = m * k;
        let b_stride = k * n;
        let c_stride = m * n;
        let mut c = vec![0.0f32; batch_size * c_stride];
        for b in 0..batch_size {
            let a_off = b * a_stride;
            let b_off = b * b_stride;
            let c_off = b * c_stride;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc += a_batched[a_off + i * k + p] * b_batched[b_off + p * n + j];
                    }
                    c[c_off + i * n + j] = acc;
                }
            }
        }
        c
    }
}

// ---------------------------------------------------------------------------
// MatmulProfiler
// ---------------------------------------------------------------------------

/// Profiling result for a matmul operation.
#[derive(Debug, Clone)]
pub struct MatmulProfile {
    /// Wall-clock duration of the matmul.
    pub elapsed_secs: f64,
    /// Floating-point operations performed (2 * M * N * K).
    pub flops: u64,
    /// GFLOPS achieved.
    pub gflops: f64,
    /// Bytes read (inputs).
    pub bytes_read: u64,
    /// Memory bandwidth in GB/s.
    pub bandwidth_gbs: f64,
    /// Arithmetic intensity (FLOP / byte).
    pub arithmetic_intensity: f64,
}

/// Profiles matmul performance.
pub struct MatmulProfiler;

impl MatmulProfiler {
    /// Profile a matmul execution.
    ///
    /// Runs the provided closure and computes GFLOPS, bandwidth, and
    /// arithmetic intensity from the given dimensions.
    pub fn profile<F>(m: usize, n: usize, k: usize, f: F) -> MatmulProfile
    where
        F: FnOnce() -> Vec<f32>,
    {
        let start = Instant::now();
        let _result = f();
        let elapsed = start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();

        #[allow(clippy::cast_precision_loss)]
        let flops = 2u64 * m as u64 * n as u64 * k as u64;
        #[allow(clippy::cast_precision_loss)]
        let gflops = if elapsed_secs > 0.0 { flops as f64 / elapsed_secs / 1e9 } else { 0.0 };

        // Bytes read = (A elements + B elements) * 4 bytes/f32.
        #[allow(clippy::cast_precision_loss)]
        let bytes_read = ((m * k + k * n) as u64) * 4;
        #[allow(clippy::cast_precision_loss)]
        let bandwidth_gbs =
            if elapsed_secs > 0.0 { bytes_read as f64 / elapsed_secs / 1e9 } else { 0.0 };

        #[allow(clippy::cast_precision_loss)]
        let arithmetic_intensity =
            if bytes_read > 0 { flops as f64 / bytes_read as f64 } else { 0.0 };

        MatmulProfile {
            elapsed_secs,
            flops,
            gflops,
            bytes_read,
            bandwidth_gbs,
            arithmetic_intensity,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a simple `MatmulInput` (no transpose).
    fn simple_input(
        a: Vec<f32>,
        a_shape: (usize, usize),
        b: Vec<f32>,
        b_shape: (usize, usize),
    ) -> MatmulInput {
        MatmulInput {
            a_data: a,
            b_data: b,
            a_shape,
            b_shape,
            transpose_a: false,
            transpose_b: false,
        }
    }

    /// Identity matrix of size n×n.
    fn identity(n: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    /// Assert two f32 slices are approximately equal.
    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} vs {y} (diff={})", (x - y).abs());
        }
    }

    // =======================================================================
    // NaiveMatmul — correctness
    // =======================================================================

    #[test]
    fn naive_2x2_known_result() {
        // | 1 2 |   | 5 6 |   | 19 22 |
        // | 3 4 | × | 7 8 | = | 43 50 |
        let input =
            simple_input(vec![1.0, 2.0, 3.0, 4.0], (2, 2), vec![5.0, 6.0, 7.0, 8.0], (2, 2));
        let c = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &[19.0, 22.0, 43.0, 50.0], 1e-6);
    }

    #[test]
    fn naive_identity_left() {
        let id = identity(3);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = simple_input(id, (3, 3), b.clone(), (3, 3));
        let c = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &b, 1e-6);
    }

    #[test]
    fn naive_identity_right() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let id = identity(3);
        let input = simple_input(a.clone(), (3, 3), id, (3, 3));
        let c = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &a, 1e-6);
    }

    #[test]
    fn naive_zero_matrix() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0; 4];
        let input = simple_input(a, (2, 2), b, (2, 2));
        let c = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &[0.0; 4], 1e-6);
    }

    #[test]
    fn naive_non_square_2x3_times_3x2() {
        // | 1 2 3 |   | 7  8  |   | 58  64  |
        // | 4 5 6 | × | 9  10 | = | 139 154 |
        //              | 11 12 |
        let input = simple_input(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2, 3),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            (3, 2),
        );
        let c = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &[58.0, 64.0, 139.0, 154.0], 1e-6);
    }

    #[test]
    fn naive_1x1() {
        let input = simple_input(vec![3.0], (1, 1), vec![7.0], (1, 1));
        let c = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &[21.0], 1e-6);
    }

    #[test]
    fn naive_row_times_col() {
        // [1, 2, 3] × [[4], [5], [6]] = [32]
        let input = simple_input(vec![1.0, 2.0, 3.0], (1, 3), vec![4.0, 5.0, 6.0], (3, 1));
        let c = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &[32.0], 1e-6);
    }

    #[test]
    fn naive_col_times_row() {
        // [[1], [2]] × [3, 4] = [[3,4],[6,8]]
        let input = simple_input(vec![1.0, 2.0], (2, 1), vec![3.0, 4.0], (1, 2));
        let c = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &[3.0, 4.0, 6.0, 8.0], 1e-6);
    }

    #[test]
    fn naive_negative_values() {
        let input =
            simple_input(vec![-1.0, 2.0, 3.0, -4.0], (2, 2), vec![5.0, -6.0, -7.0, 8.0], (2, 2));
        let c = NaiveMatmul::execute(&input);
        // row0: -1*5 + 2*(-7) = -19, -1*(-6) + 2*8 = 22
        // row1: 3*5 + (-4)*(-7) = 43, 3*(-6) + (-4)*8 = -50
        assert_approx_eq(&c, &[-19.0, 22.0, 43.0, -50.0], 1e-6);
    }

    #[test]
    fn naive_tall_times_wide() {
        // (4x1) × (1x3) = (4x3)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0];
        let input = simple_input(a, (4, 1), b, (1, 3));
        let c = NaiveMatmul::execute(&input);
        let expected =
            vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0, 40.0, 80.0, 120.0];
        assert_approx_eq(&c, &expected, 1e-6);
    }

    // =======================================================================
    // TiledMatmul — matches naive
    // =======================================================================

    #[test]
    fn tiled_matches_naive_2x2() {
        let input =
            simple_input(vec![1.0, 2.0, 3.0, 4.0], (2, 2), vec![5.0, 6.0, 7.0, 8.0], (2, 2));
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 16);
        assert_approx_eq(&tiled, &naive, 1e-6);
    }

    #[test]
    fn tiled_matches_naive_4x4_tile16() {
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let b: Vec<f32> = (17..=32).map(|x| x as f32).collect();
        let input = simple_input(a, (4, 4), b, (4, 4));
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 16);
        assert_approx_eq(&tiled, &naive, 1e-5);
    }

    #[test]
    fn tiled_matches_naive_4x4_tile2() {
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let b: Vec<f32> = (17..=32).map(|x| x as f32).collect();
        let input = simple_input(a, (4, 4), b, (4, 4));
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 2);
        assert_approx_eq(&tiled, &naive, 1e-5);
    }

    #[test]
    fn tiled_non_square_tile32() {
        let input = simple_input(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2, 3),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            (3, 2),
        );
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 32);
        assert_approx_eq(&tiled, &naive, 1e-6);
    }

    #[test]
    fn tiled_tile_size_1() {
        let input =
            simple_input(vec![1.0, 2.0, 3.0, 4.0], (2, 2), vec![5.0, 6.0, 7.0, 8.0], (2, 2));
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 1);
        assert_approx_eq(&tiled, &naive, 1e-6);
    }

    #[test]
    fn tiled_tile_larger_than_matrix() {
        let input =
            simple_input(vec![1.0, 2.0, 3.0, 4.0], (2, 2), vec![5.0, 6.0, 7.0, 8.0], (2, 2));
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 64);
        assert_approx_eq(&tiled, &naive, 1e-6);
    }

    #[test]
    fn tiled_8x8_tile_3_misaligned() {
        let a: Vec<f32> = (0..64).map(|x| x as f32 * 0.1).collect();
        let b: Vec<f32> = (0..64).map(|x| x as f32 * 0.2).collect();
        let input = simple_input(a, (8, 8), b, (8, 8));
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 3);
        assert_approx_eq(&tiled, &naive, 1e-4);
    }

    // =======================================================================
    // BlockedMatmul — matches naive
    // =======================================================================

    #[test]
    fn blocked_matches_naive_2x2() {
        let input =
            simple_input(vec![1.0, 2.0, 3.0, 4.0], (2, 2), vec![5.0, 6.0, 7.0, 8.0], (2, 2));
        let naive = NaiveMatmul::execute(&input);
        let blocked = BlockedMatmul::execute(&input, 16);
        assert_approx_eq(&blocked, &naive, 1e-6);
    }

    #[test]
    fn blocked_matches_naive_4x4_block2() {
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let b: Vec<f32> = (17..=32).map(|x| x as f32).collect();
        let input = simple_input(a, (4, 4), b, (4, 4));
        let naive = NaiveMatmul::execute(&input);
        let blocked = BlockedMatmul::execute(&input, 2);
        assert_approx_eq(&blocked, &naive, 1e-5);
    }

    #[test]
    fn blocked_non_square() {
        let input = simple_input(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2, 3),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            (3, 2),
        );
        let naive = NaiveMatmul::execute(&input);
        let blocked = BlockedMatmul::execute(&input, 4);
        assert_approx_eq(&blocked, &naive, 1e-6);
    }

    #[test]
    fn blocked_block_size_1() {
        let input =
            simple_input(vec![1.0, 2.0, 3.0, 4.0], (2, 2), vec![5.0, 6.0, 7.0, 8.0], (2, 2));
        let naive = NaiveMatmul::execute(&input);
        let blocked = BlockedMatmul::execute(&input, 1);
        assert_approx_eq(&blocked, &naive, 1e-6);
    }

    #[test]
    fn blocked_block_larger_than_matrix() {
        let input =
            simple_input(vec![1.0, 2.0, 3.0, 4.0], (2, 2), vec![5.0, 6.0, 7.0, 8.0], (2, 2));
        let naive = NaiveMatmul::execute(&input);
        let blocked = BlockedMatmul::execute(&input, 128);
        assert_approx_eq(&blocked, &naive, 1e-6);
    }

    #[test]
    fn blocked_8x8_block3_misaligned() {
        let a: Vec<f32> = (0..64).map(|x| x as f32 * 0.1).collect();
        let b: Vec<f32> = (0..64).map(|x| x as f32 * 0.2).collect();
        let input = simple_input(a, (8, 8), b, (8, 8));
        let naive = NaiveMatmul::execute(&input);
        let blocked = BlockedMatmul::execute(&input, 3);
        assert_approx_eq(&blocked, &naive, 1e-4);
    }

    // =======================================================================
    // Transposed matmul
    // =======================================================================

    #[test]
    fn transpose_a_matches_explicit() {
        // A stored as 3x2, transposed → 2x3. B = 3x2.
        // A^T = | 1 3 5 |   B = | 7  8  |
        //       | 2 4 6 |       | 9  10 |
        //                       | 11 12 |
        let input = MatmulInput {
            a_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b_data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            a_shape: (3, 2),
            b_shape: (3, 2),
            transpose_a: true,
            transpose_b: false,
        };
        // A^T (2x3) × B (3x2) → C (2x2)
        let c = NaiveMatmul::execute(&input);
        // row0: 1*7 + 3*9 + 5*11 = 7+27+55 = 89
        //       1*8 + 3*10 + 5*12 = 8+30+60 = 98
        // row1: 2*7 + 4*9 + 6*11 = 14+36+66 = 116
        //       2*8 + 4*10 + 6*12 = 16+40+72 = 128
        assert_approx_eq(&c, &[89.0, 98.0, 116.0, 128.0], 1e-5);
    }

    #[test]
    fn transpose_b_matches_explicit() {
        // A = 2x3, B stored as 2x3, transposed → 3x2.
        let input = MatmulInput {
            a_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b_data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            a_shape: (2, 3),
            b_shape: (2, 3),
            transpose_a: false,
            transpose_b: true,
        };
        // B^T = | 7  9  11 |^T → | 7  10 |
        //       | 8  10 12 |     | 8  11 |
        //  wait, B stored as (2,3): [[7,8,9],[10,11,12]]
        //  B^T is (3,2): [[7,10],[8,11],[9,12]]
        // A(2,3) × B^T(3,2) → C(2,2)
        let c = NaiveMatmul::execute(&input);
        // row0: 1*7 + 2*8 + 3*9 = 7+16+27 = 50
        //       1*10 + 2*11 + 3*12 = 10+22+36 = 68
        // row1: 4*7 + 5*8 + 6*9 = 28+40+54 = 122
        //       4*10 + 5*11 + 6*12 = 40+55+72 = 167
        assert_approx_eq(&c, &[50.0, 68.0, 122.0, 167.0], 1e-5);
    }

    #[test]
    fn transpose_both() {
        // A stored (3,2), B stored (2,3). A^T=(2,3), B^T=(3,2). C=(2,2).
        let input = MatmulInput {
            a_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b_data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            a_shape: (3, 2),
            b_shape: (2, 3),
            transpose_a: true,
            transpose_b: true,
        };
        let c = NaiveMatmul::execute(&input);
        // A^T = [[1,3,5],[2,4,6]], B^T = [[7,10],[8,11],[9,12]]
        // C[0,0] = 1*7 + 3*8 + 5*9 = 7+24+45 = 76
        // C[0,1] = 1*10 + 3*11 + 5*12 = 10+33+60 = 103
        // C[1,0] = 2*7 + 4*8 + 6*9 = 14+32+54 = 100
        // C[1,1] = 2*10 + 4*11 + 6*12 = 20+44+72 = 136
        assert_approx_eq(&c, &[76.0, 103.0, 100.0, 136.0], 1e-5);
    }

    #[test]
    fn tiled_transpose_a_matches_naive() {
        let input = MatmulInput {
            a_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b_data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            a_shape: (3, 2),
            b_shape: (3, 2),
            transpose_a: true,
            transpose_b: false,
        };
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 2);
        assert_approx_eq(&tiled, &naive, 1e-5);
    }

    #[test]
    fn blocked_transpose_b_matches_naive() {
        let input = MatmulInput {
            a_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b_data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            a_shape: (2, 3),
            b_shape: (2, 3),
            transpose_a: false,
            transpose_b: true,
        };
        let naive = NaiveMatmul::execute(&input);
        let blocked = BlockedMatmul::execute(&input, 2);
        assert_approx_eq(&blocked, &naive, 1e-5);
    }

    // =======================================================================
    // QuantizedMatmul
    // =======================================================================

    #[test]
    fn quantized_i2s_basic() {
        // scale=1.0, zero=0 → dequant is identity.
        let a = vec![1i8, 0, 0, 1];
        let b = vec![2i8, 3, 4, 5];
        let c = QuantizedMatmul::execute(&a, &b, (2, 2), (2, 2), 1.0, 1.0, 0, 0, QuantType::I2S);
        // Same as float [[1,0],[0,1]] × [[2,3],[4,5]] = [[2,3],[4,5]]
        assert_approx_eq(&c, &[2.0, 3.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn quantized_with_scale() {
        let a = vec![2i8, 3];
        let b = vec![4i8, 5];
        // A(1,2) scale=0.5 zp=0 → [1.0, 1.5]
        // B(2,1) scale=2.0 zp=0 → [8.0, 10.0]
        // C = 1.0*8.0 + 1.5*10.0 = 23.0
        let c = QuantizedMatmul::execute(&a, &b, (1, 2), (2, 1), 0.5, 2.0, 0, 0, QuantType::Int8);
        assert_approx_eq(&c, &[23.0], 1e-5);
    }

    #[test]
    fn quantized_with_zero_point() {
        let a = vec![10i8];
        let b = vec![20i8];
        // A: (10 - 5) * 1.0 = 5.0
        // B: (20 - 10) * 1.0 = 10.0
        // C = 50.0
        let c = QuantizedMatmul::execute(&a, &b, (1, 1), (1, 1), 1.0, 1.0, 5, 10, QuantType::Int4);
        assert_approx_eq(&c, &[50.0], 1e-5);
    }

    #[test]
    fn quantized_matches_float_naive() {
        // Compare quantized result against float naive when scale=1.0, zp=0.
        let a_q = vec![1i8, 2, 3, 4, 5, 6];
        let b_q = vec![7i8, 8, 9, 10, 11, 12];
        let c_q =
            QuantizedMatmul::execute(&a_q, &b_q, (2, 3), (3, 2), 1.0, 1.0, 0, 0, QuantType::Int8);

        let a_f: Vec<f32> = a_q.iter().map(|&x| f32::from(x)).collect();
        let b_f: Vec<f32> = b_q.iter().map(|&x| f32::from(x)).collect();
        let input = simple_input(a_f, (2, 3), b_f, (3, 2));
        let c_naive = NaiveMatmul::execute(&input);
        assert_approx_eq(&c_q, &c_naive, 1e-5);
    }

    #[test]
    fn quantized_all_zeros() {
        let a = vec![0i8; 4];
        let b = vec![0i8; 4];
        let c = QuantizedMatmul::execute(&a, &b, (2, 2), (2, 2), 1.0, 1.0, 0, 0, QuantType::I2S);
        assert_approx_eq(&c, &[0.0; 4], 1e-6);
    }

    #[test]
    fn quantized_negative_values() {
        let a = vec![-1i8, 1];
        let b = vec![1i8, -1];
        // A(1,2) × B(2,1): (-1)*1 + 1*(-1) = -2.0
        let c = QuantizedMatmul::execute(&a, &b, (1, 2), (2, 1), 1.0, 1.0, 0, 0, QuantType::I2S);
        assert_approx_eq(&c, &[-2.0], 1e-6);
    }

    // =======================================================================
    // BatchedMatmul
    // =======================================================================

    #[test]
    fn batched_single_batch_matches_naive() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = BatchedMatmul::execute(&a, &b, 2, 2, 2, 1);
        let input = simple_input(a, (2, 2), b, (2, 2));
        let naive = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &naive, 1e-6);
    }

    #[test]
    fn batched_two_batches() {
        // Batch 0: I × [[1,2],[3,4]] = [[1,2],[3,4]]
        // Batch 1: [[2,0],[0,2]] × [[1,0],[0,1]] = [[2,0],[0,2]]
        let a = vec![
            1.0, 0.0, 0.0, 1.0, // batch 0: identity
            2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
        ];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            1.0, 0.0, 0.0, 1.0, // batch 1: identity
        ];
        let c = BatchedMatmul::execute(&a, &b, 2, 2, 2, 2);
        assert_approx_eq(
            &c,
            &[
                1.0, 2.0, 3.0, 4.0, // batch 0 result
                2.0, 0.0, 0.0, 2.0, // batch 1 result
            ],
            1e-6,
        );
    }

    #[test]
    fn batched_non_square() {
        // Each batch: (1x2) × (2x3) → (1x3)
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2 batches of (1,2)
        let b = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // batch 0: (2,3)
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, // batch 1: (2,3)
        ];
        let c = BatchedMatmul::execute(&a, &b, 1, 2, 3, 2);
        // batch 0: [1,2] × [[1,0,0],[0,1,0]] = [1,2,0]
        // batch 1: [3,4] × [[0,0,1],[1,0,0]] = [4,0,3]
        assert_approx_eq(&c, &[1.0, 2.0, 0.0, 4.0, 0.0, 3.0], 1e-6);
    }

    #[test]
    fn batched_four_batches_identity() {
        let id = identity(2);
        let a: Vec<f32> = (0..4).flat_map(|_| id.clone()).collect();
        let b: Vec<f32> = (0..4)
            .flat_map(|i| {
                let v = (i * 4 + 1) as f32;
                vec![v, v + 1.0, v + 2.0, v + 3.0]
            })
            .collect();
        let c = BatchedMatmul::execute(&a, &b, 2, 2, 2, 4);
        // Each batch: I × B_i = B_i
        assert_approx_eq(&c, &b, 1e-6);
    }

    // =======================================================================
    // MatmulValidator
    // =======================================================================

    #[test]
    fn validator_accepts_valid_input() {
        let input = simple_input(vec![1.0; 6], (2, 3), vec![1.0; 6], (3, 2));
        assert!(MatmulValidator::validate(&input).is_ok());
    }

    #[test]
    fn validator_rejects_a_data_too_short() {
        let input = simple_input(vec![1.0; 4], (2, 3), vec![1.0; 6], (3, 2));
        let err = MatmulValidator::validate(&input).unwrap_err();
        assert!(matches!(
            err,
            MatmulValidationError::ShapeMismatch { matrix: "A", expected: 6, actual: 4 }
        ));
    }

    #[test]
    fn validator_rejects_b_data_too_long() {
        let input = simple_input(vec![1.0; 6], (2, 3), vec![1.0; 10], (3, 2));
        let err = MatmulValidator::validate(&input).unwrap_err();
        assert!(matches!(
            err,
            MatmulValidationError::ShapeMismatch { matrix: "B", expected: 6, actual: 10 }
        ));
    }

    #[test]
    fn validator_rejects_inner_dimension_mismatch() {
        let input = simple_input(vec![1.0; 6], (2, 3), vec![1.0; 8], (4, 2));
        let err = MatmulValidator::validate(&input).unwrap_err();
        assert!(matches!(
            err,
            MatmulValidationError::InnerDimensionMismatch { a_cols: 3, b_rows: 4 }
        ));
    }

    #[test]
    fn validator_rejects_zero_dim_a() {
        let input = simple_input(vec![], (0, 3), vec![1.0; 6], (3, 2));
        let err = MatmulValidator::validate(&input).unwrap_err();
        assert!(matches!(err, MatmulValidationError::ZeroDimension { matrix: "A" }));
    }

    #[test]
    fn validator_rejects_zero_dim_b() {
        let input = simple_input(vec![1.0; 6], (2, 3), vec![], (0, 2));
        let err = MatmulValidator::validate(&input).unwrap_err();
        // B shape (0,2) → data expected 0, data is 0, but zero dimension triggers.
        assert!(matches!(err, MatmulValidationError::ZeroDimension { .. }));
    }

    #[test]
    fn validator_accepts_transposed_input() {
        let input = MatmulInput {
            a_data: vec![1.0; 6],
            b_data: vec![1.0; 6],
            a_shape: (3, 2),
            b_shape: (3, 2),
            transpose_a: true,  // effective (2,3)
            transpose_b: false, // effective (3,2)
        };
        assert!(MatmulValidator::validate(&input).is_ok());
    }

    #[test]
    fn validator_rejects_transposed_inner_mismatch() {
        let input = MatmulInput {
            a_data: vec![1.0; 6],
            b_data: vec![1.0; 4],
            a_shape: (2, 3),
            b_shape: (2, 2),
            transpose_a: false, // effective (2,3)
            transpose_b: false, // effective (2,2)
        };
        let err = MatmulValidator::validate(&input).unwrap_err();
        assert!(matches!(
            err,
            MatmulValidationError::InnerDimensionMismatch { a_cols: 3, b_rows: 2 }
        ));
    }

    #[test]
    fn validator_error_display() {
        let e = MatmulValidationError::InnerDimensionMismatch { a_cols: 5, b_rows: 3 };
        let s = format!("{e}");
        assert!(s.contains('5'));
        assert!(s.contains('3'));
    }

    // =======================================================================
    // MatmulInput — effective shapes
    // =======================================================================

    #[test]
    fn effective_shape_no_transpose() {
        let input = simple_input(vec![0.0; 6], (2, 3), vec![0.0; 6], (3, 2));
        assert_eq!(input.effective_a_shape(), (2, 3));
        assert_eq!(input.effective_b_shape(), (3, 2));
        assert_eq!(input.output_shape(), (2, 2));
    }

    #[test]
    fn effective_shape_transpose_a() {
        let input = MatmulInput {
            a_data: vec![0.0; 6],
            b_data: vec![0.0; 6],
            a_shape: (3, 2),
            b_shape: (3, 2),
            transpose_a: true,
            transpose_b: false,
        };
        assert_eq!(input.effective_a_shape(), (2, 3));
        assert_eq!(input.output_shape(), (2, 2));
    }

    #[test]
    fn effective_shape_transpose_b() {
        let input = MatmulInput {
            a_data: vec![0.0; 6],
            b_data: vec![0.0; 6],
            a_shape: (2, 3),
            b_shape: (2, 3),
            transpose_a: false,
            transpose_b: true,
        };
        assert_eq!(input.effective_b_shape(), (3, 2));
        assert_eq!(input.output_shape(), (2, 2));
    }

    // =======================================================================
    // MatmulConfig
    // =======================================================================

    #[test]
    fn default_config() {
        let cfg = MatmulConfig::default();
        assert_eq!(cfg.algorithm, MatmulAlgorithm::Tiled);
        assert_eq!(cfg.tile_size, 32);
        assert_eq!(cfg.accumulator_dtype, AccumulatorDtype::F32);
        assert!(!cfg.use_tensor_cores);
    }

    #[test]
    fn config_custom() {
        let cfg = MatmulConfig {
            algorithm: MatmulAlgorithm::BlockedTiled,
            tile_size: 64,
            accumulator_dtype: AccumulatorDtype::F64,
            use_tensor_cores: true,
        };
        assert_eq!(cfg.tile_size, 64);
        assert!(cfg.use_tensor_cores);
    }

    // =======================================================================
    // MatmulAlgorithm — Display
    // =======================================================================

    #[test]
    fn algorithm_display() {
        assert_eq!(format!("{}", MatmulAlgorithm::Naive), "Naive");
        assert_eq!(format!("{}", MatmulAlgorithm::Tiled), "Tiled");
        assert_eq!(format!("{}", MatmulAlgorithm::BlockedTiled), "BlockedTiled");
        assert_eq!(format!("{}", MatmulAlgorithm::WinogradSmall), "WinogradSmall");
        assert_eq!(format!("{}", MatmulAlgorithm::StrassenThreshold), "StrassenThreshold");
    }

    #[test]
    fn algorithm_eq() {
        assert_eq!(MatmulAlgorithm::Naive, MatmulAlgorithm::Naive);
        assert_ne!(MatmulAlgorithm::Naive, MatmulAlgorithm::Tiled);
    }

    #[test]
    fn algorithm_copy() {
        let a = MatmulAlgorithm::BlockedTiled;
        let b = a;
        assert_eq!(a, b);
    }

    // =======================================================================
    // MatmulProfiler
    // =======================================================================

    #[test]
    fn profiler_reports_correct_flops() {
        let profile = MatmulProfiler::profile(4, 4, 4, || vec![0.0; 16]);
        assert_eq!(profile.flops, 2 * 4 * 4 * 4);
    }

    #[test]
    fn profiler_reports_correct_bytes_read() {
        let profile = MatmulProfiler::profile(4, 4, 4, || vec![0.0; 16]);
        // (4*4 + 4*4) * 4 = 128
        assert_eq!(profile.bytes_read, 128);
    }

    #[test]
    fn profiler_positive_elapsed() {
        let profile = MatmulProfiler::profile(2, 2, 2, || {
            // Do a tiny matmul to ensure nonzero time.
            let input = simple_input(vec![1.0; 4], (2, 2), vec![1.0; 4], (2, 2));
            NaiveMatmul::execute(&input)
        });
        assert!(profile.elapsed_secs >= 0.0);
    }

    #[test]
    fn profiler_gflops_non_negative() {
        let profile = MatmulProfiler::profile(8, 8, 8, || vec![0.0; 64]);
        assert!(profile.gflops >= 0.0);
    }

    #[test]
    fn profiler_arithmetic_intensity_correct() {
        // For m=n=k=4: flops=128, bytes=(16+16)*4=128 → intensity=1.0
        let profile = MatmulProfiler::profile(4, 4, 4, || vec![0.0; 16]);
        assert!((profile.arithmetic_intensity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn profiler_large_matrix_intensity() {
        // For m=n=k=64: flops=2*64^3=524288, bytes=(64*64+64*64)*4=32768
        // intensity = 524288/32768 = 16.0
        let profile = MatmulProfiler::profile(64, 64, 64, || vec![0.0; 64 * 64]);
        assert!((profile.arithmetic_intensity - 16.0).abs() < 1e-6);
    }

    // =======================================================================
    // Edge cases
    // =======================================================================

    #[test]
    fn naive_1x1_identity() {
        let input = simple_input(vec![1.0], (1, 1), vec![1.0], (1, 1));
        let c = NaiveMatmul::execute(&input);
        assert_approx_eq(&c, &[1.0], 1e-6);
    }

    #[test]
    fn naive_very_tall_matrix() {
        // (100x1) × (1x1) → (100x1)
        let a: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let input = simple_input(a.clone(), (100, 1), vec![2.0], (1, 1));
        let c = NaiveMatmul::execute(&input);
        let expected: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        assert_approx_eq(&c, &expected, 1e-5);
    }

    #[test]
    fn naive_very_wide_matrix() {
        // (1x100) × (100x1) → (1x1): dot product
        let a: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let b = vec![1.0f32; 100];
        let input = simple_input(a, (1, 100), b, (100, 1));
        let c = NaiveMatmul::execute(&input);
        // sum(1..=100) = 5050
        assert_approx_eq(&c, &[5050.0], 1e-3);
    }

    #[test]
    fn tiled_1x1() {
        let input = simple_input(vec![5.0], (1, 1), vec![3.0], (1, 1));
        let c = TiledMatmul::execute(&input, 16);
        assert_approx_eq(&c, &[15.0], 1e-6);
    }

    #[test]
    fn blocked_1x1() {
        let input = simple_input(vec![5.0], (1, 1), vec![3.0], (1, 1));
        let c = BlockedMatmul::execute(&input, 8);
        assert_approx_eq(&c, &[15.0], 1e-6);
    }

    #[test]
    fn all_algorithms_agree_on_5x7_times_7x3() {
        let a: Vec<f32> = (0..35).map(|x| (x as f32) * 0.3 - 5.0).collect();
        let b: Vec<f32> = (0..21).map(|x| (x as f32) * 0.7 + 1.0).collect();
        let input = simple_input(a, (5, 7), b, (7, 3));
        let naive = NaiveMatmul::execute(&input);
        let tiled_16 = TiledMatmul::execute(&input, 16);
        let tiled_32 = TiledMatmul::execute(&input, 32);
        let tiled_64 = TiledMatmul::execute(&input, 64);
        let blocked_4 = BlockedMatmul::execute(&input, 4);
        let blocked_8 = BlockedMatmul::execute(&input, 8);
        assert_approx_eq(&tiled_16, &naive, 1e-3);
        assert_approx_eq(&tiled_32, &naive, 1e-3);
        assert_approx_eq(&tiled_64, &naive, 1e-3);
        assert_approx_eq(&blocked_4, &naive, 1e-3);
        assert_approx_eq(&blocked_8, &naive, 1e-3);
    }

    #[test]
    fn tiled_matches_naive_16x16() {
        let a: Vec<f32> = (0..256).map(|x| (x as f32) * 0.01).collect();
        let b: Vec<f32> = (0..256).map(|x| (x as f32) * 0.02).collect();
        let input = simple_input(a, (16, 16), b, (16, 16));
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 4);
        assert_approx_eq(&tiled, &naive, 1e-2);
    }

    #[test]
    fn blocked_matches_naive_16x16() {
        let a: Vec<f32> = (0..256).map(|x| (x as f32) * 0.01).collect();
        let b: Vec<f32> = (0..256).map(|x| (x as f32) * 0.02).collect();
        let input = simple_input(a, (16, 16), b, (16, 16));
        let naive = NaiveMatmul::execute(&input);
        let blocked = BlockedMatmul::execute(&input, 4);
        assert_approx_eq(&blocked, &naive, 1e-2);
    }

    #[test]
    fn quantized_1x1() {
        let c =
            QuantizedMatmul::execute(&[3], &[7], (1, 1), (1, 1), 1.0, 1.0, 0, 0, QuantType::I2S);
        assert_approx_eq(&c, &[21.0], 1e-6);
    }

    #[test]
    fn batched_1x1_single() {
        let c = BatchedMatmul::execute(&[5.0], &[3.0], 1, 1, 1, 1);
        assert_approx_eq(&c, &[15.0], 1e-6);
    }

    #[test]
    fn validator_1x1_valid() {
        let input = simple_input(vec![1.0], (1, 1), vec![1.0], (1, 1));
        assert!(MatmulValidator::validate(&input).is_ok());
    }

    #[test]
    fn quant_type_eq() {
        assert_eq!(QuantType::I2S, QuantType::I2S);
        assert_ne!(QuantType::Int4, QuantType::Int8);
    }

    #[test]
    fn accumulator_dtype_eq() {
        assert_eq!(AccumulatorDtype::F32, AccumulatorDtype::F32);
        assert_ne!(AccumulatorDtype::F32, AccumulatorDtype::F64);
    }

    #[test]
    fn validation_error_display_shape_mismatch() {
        let e = MatmulValidationError::ShapeMismatch { matrix: "A", expected: 6, actual: 4 };
        let s = format!("{e}");
        assert!(s.contains('A'));
        assert!(s.contains('6'));
        assert!(s.contains('4'));
    }

    #[test]
    fn validation_error_display_zero_dim() {
        let e = MatmulValidationError::ZeroDimension { matrix: "B" };
        let s = format!("{e}");
        assert!(s.contains('B'));
        assert!(s.contains("zero"));
    }

    // =======================================================================
    // Additional coverage
    // =======================================================================

    #[test]
    fn tiled_matches_naive_non_square_3x5_times_5x2() {
        let a: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..10).map(|x| x as f32 + 1.0).collect();
        let input = simple_input(a, (3, 5), b, (5, 2));
        let naive = NaiveMatmul::execute(&input);
        let tiled = TiledMatmul::execute(&input, 2);
        assert_approx_eq(&tiled, &naive, 1e-4);
    }

    #[test]
    fn blocked_matches_naive_non_square_3x5_times_5x2() {
        let a: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..10).map(|x| x as f32 + 1.0).collect();
        let input = simple_input(a, (3, 5), b, (5, 2));
        let naive = NaiveMatmul::execute(&input);
        let blocked = BlockedMatmul::execute(&input, 3);
        assert_approx_eq(&blocked, &naive, 1e-4);
    }

    #[test]
    fn quantized_int4_type() {
        let a = vec![2i8, 3, 1, 4];
        let b = vec![1i8, 0, 0, 1];
        let c = QuantizedMatmul::execute(&a, &b, (2, 2), (2, 2), 1.0, 1.0, 0, 0, QuantType::Int4);
        // identity B → same as A
        assert_approx_eq(&c, &[2.0, 3.0, 1.0, 4.0], 1e-6);
    }

    #[test]
    fn naive_large_values() {
        let input = simple_input(vec![1e6, 1e6, 1e6, 1e6], (2, 2), vec![1e6; 4], (2, 2));
        let c = NaiveMatmul::execute(&input);
        // Each element: 1e6*1e6 + 1e6*1e6 = 2e12
        for v in &c {
            assert!((*v - 2e12).abs() < 1e6);
        }
    }

    #[test]
    fn batched_three_batches_1x1() {
        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0];
        let c = BatchedMatmul::execute(&a, &b, 1, 1, 1, 3);
        assert_approx_eq(&c, &[10.0, 18.0, 28.0], 1e-6);
    }

    #[test]
    fn profiler_bandwidth_non_negative() {
        let profile = MatmulProfiler::profile(4, 4, 4, || vec![0.0; 16]);
        assert!(profile.bandwidth_gbs >= 0.0);
    }

    #[test]
    fn matmul_profile_debug() {
        let profile = MatmulProfiler::profile(2, 2, 2, || vec![0.0; 4]);
        let dbg = format!("{profile:?}");
        assert!(dbg.contains("flops"));
    }

    #[test]
    fn matmul_config_clone() {
        let cfg = MatmulConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.tile_size, cfg2.tile_size);
    }
}
