//! CUDA matrix multiplication operations optimized for 1-bit neural network
//! inference.
//!
//! This crate provides host-side matrix multiplication routines that mirror
//! the operations a CUDA kernel would perform. The actual GPU dispatch is
//! gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`; when
//! neither feature is active the crate still exposes CPU reference
//! implementations useful for testing and cross-validation.
//!
//! ## Supported operations
//!
//! | Operation | Function | Description |
//! |-----------|----------|-------------|
//! | GEMM | [`gemm`] | General `f32` matrix multiply `C = αAB + βC` |
//! | Binary matmul | [`binary_matmul`] | Packed 1-bit matrix multiply via popcount |
//! | Mixed-precision | [`mixed_precision_matmul`] | Binary weights × `f32` activations |
//! | Tiled GEMM | [`TiledMatmul`] | Configurable tile-size blocked multiply |
//! | Batch GEMM | [`batch_gemm`] | Batched `f32` matrix multiply |
//!
//! ## Example
//!
//! ```
//! use bitnet_cuda_matmul::{gemm, MatmulError};
//!
//! // C (2×2) = 1.0 * A (2×3) * B (3×2) + 0.0 * C
//! let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
//! let mut c = [0.0f32; 4];
//! gemm(2, 2, 3, 1.0, &a, &b, 0.0, &mut c).unwrap();
//! assert!((c[0] - 58.0).abs() < 1e-6);
//! ```

#![allow(clippy::many_single_char_names)]

// ── Error type ──────────────────────────────────────────────────────────

/// Errors returned by matrix multiplication operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatmulError {
    /// Matrix dimensions are incompatible for multiplication.
    DimensionMismatch { expected: usize, actual: usize, context: &'static str },
    /// A supplied buffer is too small.
    BufferTooSmall { required: usize, actual: usize, context: &'static str },
    /// Tile size is invalid (zero or exceeds matrix dimensions unreasonably).
    InvalidTileSize { tile_size: usize },
    /// Batch size is zero.
    EmptyBatch,
}

impl std::fmt::Display for MatmulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual, context } => {
                write!(f, "dimension mismatch in {context}: expected {expected}, got {actual}")
            }
            Self::BufferTooSmall { required, actual, context } => {
                write!(f, "buffer too small for {context}: need {required}, got {actual}")
            }
            Self::InvalidTileSize { tile_size } => {
                write!(f, "invalid tile size: {tile_size}")
            }
            Self::EmptyBatch => write!(f, "batch size must be > 0"),
        }
    }
}

impl std::error::Error for MatmulError {}

// ── GEMM (f32) ──────────────────────────────────────────────────────────

/// General f32 matrix multiplication: **C = α·A·B + β·C**.
///
/// Matrices are stored in **row-major** order.
///
/// * `m` – rows of A / rows of C
/// * `n` – cols of B / cols of C
/// * `k` – cols of A / rows of B
///
/// # Errors
///
/// Returns [`MatmulError`] on dimension / buffer size mismatches.
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<(), MatmulError> {
    if a.len() < m * k {
        return Err(MatmulError::BufferTooSmall {
            required: m * k,
            actual: a.len(),
            context: "A matrix",
        });
    }
    if b.len() < k * n {
        return Err(MatmulError::BufferTooSmall {
            required: k * n,
            actual: b.len(),
            context: "B matrix",
        });
    }
    if c.len() < m * n {
        return Err(MatmulError::BufferTooSmall {
            required: m * n,
            actual: c.len(),
            context: "C matrix",
        });
    }

    gemm_inner(m, n, k, alpha, a, b, beta, c);
    Ok(())
}

/// Inner GEMM loop (no bounds checks).
fn gemm_inner(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum = a[i * k + p].mul_add(b[p * n + j], sum);
            }
            let idx = i * n + j;
            c[idx] = alpha.mul_add(sum, beta * c[idx]);
        }
    }
}

// ── Binary matmul (1-bit packed) ────────────────────────────────────────

/// Packed 1-bit matrix multiplication using XNOR + popcount.
///
/// Both `a_packed` and `b_packed` store bits in **row-major** order, one
/// bit per weight. Each `u64` encodes 64 contiguous elements along the
/// inner dimension. The inner dimension `k` **must** be a multiple of 64.
///
/// The result is written as `i32` counts into `c` (row-major, m×n).
///
/// # Errors
///
/// Returns [`MatmulError`] when dimensions or buffer sizes are wrong.
pub fn binary_matmul(
    m: usize,
    n: usize,
    k: usize,
    a_packed: &[u64],
    b_packed: &[u64],
    c: &mut [i32],
) -> Result<(), MatmulError> {
    if k == 0 || !k.is_multiple_of(64) {
        return Err(MatmulError::DimensionMismatch {
            expected: 64,
            actual: k % 64,
            context: "k must be a positive multiple of 64",
        });
    }

    let words_per_row = k / 64;

    if a_packed.len() < m * words_per_row {
        return Err(MatmulError::BufferTooSmall {
            required: m * words_per_row,
            actual: a_packed.len(),
            context: "A packed",
        });
    }
    if b_packed.len() < n * words_per_row {
        return Err(MatmulError::BufferTooSmall {
            required: n * words_per_row,
            actual: b_packed.len(),
            context: "B packed (column-major packing expected as n rows of k/64)",
        });
    }
    if c.len() < m * n {
        return Err(MatmulError::BufferTooSmall {
            required: m * n,
            actual: c.len(),
            context: "C output",
        });
    }

    for i in 0..m {
        let a_row = &a_packed[i * words_per_row..(i + 1) * words_per_row];
        for j in 0..n {
            let b_row = &b_packed[j * words_per_row..(j + 1) * words_per_row];
            let mut pop: u32 = 0;
            for w in 0..words_per_row {
                pop += (a_row[w] ^ b_row[w]).count_ones();
            }
            // XNOR-popcount: agreement = k - hamming, result = 2*agreement - k
            let hamming: i32 = pop.try_into().expect("popcount fits i32");
            let k_i32 = i32::try_from(k).expect("k fits in i32");
            c[i * n + j] = k_i32 - 2 * hamming;
        }
    }

    Ok(())
}

// ── Mixed-precision matmul ──────────────────────────────────────────────

/// Mixed-precision matmul: binary (packed `u64`) weights × f32 activations.
///
/// * `weights` – `m` rows, each with `k/64` packed `u64` words (1-bit per
///   element; bit=1 → +1, bit=0 → −1).
/// * `activations` – f32 slice of length `k × n` (row-major, k rows × n cols).
/// * `output` – f32 slice of length `m × n` (row-major).
/// * `k` must be a positive multiple of 64.
///
/// # Errors
///
/// Returns [`MatmulError`] on invalid dimensions.
pub fn mixed_precision_matmul(
    m: usize,
    n: usize,
    k: usize,
    weights: &[u64],
    activations: &[f32],
    output: &mut [f32],
) -> Result<(), MatmulError> {
    if k == 0 || !k.is_multiple_of(64) {
        return Err(MatmulError::DimensionMismatch {
            expected: 64,
            actual: k % 64,
            context: "k must be a positive multiple of 64",
        });
    }

    let words_per_row = k / 64;

    if weights.len() < m * words_per_row {
        return Err(MatmulError::BufferTooSmall {
            required: m * words_per_row,
            actual: weights.len(),
            context: "weights",
        });
    }
    if activations.len() < k * n {
        return Err(MatmulError::BufferTooSmall {
            required: k * n,
            actual: activations.len(),
            context: "activations",
        });
    }
    if output.len() < m * n {
        return Err(MatmulError::BufferTooSmall {
            required: m * n,
            actual: output.len(),
            context: "output",
        });
    }

    for i in 0..m {
        let w_row = &weights[i * words_per_row..(i + 1) * words_per_row];
        for j in 0..n {
            let mut acc = 0.0f32;
            for (word_idx, &word) in w_row.iter().enumerate() {
                let base = word_idx * 64;
                for bit in 0..64 {
                    let elem = base + bit;
                    let sign = if (word >> bit) & 1 == 1 { 1.0f32 } else { -1.0f32 };
                    acc = sign.mul_add(activations[elem * n + j], acc);
                }
            }
            output[i * n + j] = acc;
        }
    }

    Ok(())
}

// ── Tiled matmul ────────────────────────────────────────────────────────

/// Tiled matrix multiplication configuration.
///
/// This struct encapsulates tile-size parameters that control the blocking
/// strategy. On a real GPU these map to shared-memory tile dimensions; the
/// CPU reference implementation mirrors the same blocking pattern for
/// cross-validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledMatmul {
    rows: usize,
    cols: usize,
    inner: usize,
}

impl TiledMatmul {
    /// Create a new tiled-matmul configuration.
    ///
    /// All tile dimensions must be ≥ 1.
    ///
    /// # Errors
    ///
    /// Returns [`MatmulError::InvalidTileSize`] if any tile dimension is zero.
    #[must_use = "returns the configuration; does not perform any computation"]
    pub const fn new(rows: usize, cols: usize, inner: usize) -> Result<Self, MatmulError> {
        if rows == 0 {
            return Err(MatmulError::InvalidTileSize { tile_size: 0 });
        }
        if cols == 0 {
            return Err(MatmulError::InvalidTileSize { tile_size: 0 });
        }
        if inner == 0 {
            return Err(MatmulError::InvalidTileSize { tile_size: 0 });
        }
        Ok(Self { rows, cols, inner })
    }

    /// Tile size along the M (row) dimension.
    #[must_use]
    pub const fn tile_m(&self) -> usize {
        self.rows
    }

    /// Tile size along the N (column) dimension.
    #[must_use]
    pub const fn tile_n(&self) -> usize {
        self.cols
    }

    /// Tile size along the K (inner) dimension.
    #[must_use]
    pub const fn tile_k(&self) -> usize {
        self.inner
    }

    /// Execute tiled GEMM: **C = α·A·B + β·C**.
    ///
    /// This mirrors the shared-memory tiling pattern used in CUDA kernels:
    /// load a tile of A and a tile of B into (simulated) shared memory,
    /// compute partial products, then accumulate.
    ///
    /// # Errors
    ///
    /// Returns [`MatmulError`] on dimension / buffer mismatches.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
    ) -> Result<(), MatmulError> {
        if a.len() < m * k {
            return Err(MatmulError::BufferTooSmall {
                required: m * k,
                actual: a.len(),
                context: "A matrix (tiled)",
            });
        }
        if b.len() < k * n {
            return Err(MatmulError::BufferTooSmall {
                required: k * n,
                actual: b.len(),
                context: "B matrix (tiled)",
            });
        }
        if c.len() < m * n {
            return Err(MatmulError::BufferTooSmall {
                required: m * n,
                actual: c.len(),
                context: "C matrix (tiled)",
            });
        }

        self.execute_tiled(m, n, k, alpha, a, b, beta, c);
        Ok(())
    }

    /// Core tiled loop with shared-memory simulation.
    #[allow(clippy::too_many_arguments, clippy::trivially_copy_pass_by_ref)]
    fn execute_tiled(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
    ) {
        // Scale C by beta first.
        for val in c.iter_mut().take(m * n) {
            *val *= beta;
        }

        // Shared-memory tile buffers (simulated).
        let mut smem_a = vec![0.0f32; self.rows * self.inner];
        let mut smem_b = vec![0.0f32; self.inner * self.cols];

        // Iterate over tiles.
        let mut i0 = 0;
        while i0 < m {
            let tm = (m - i0).min(self.rows);
            let mut j0 = 0;
            while j0 < n {
                let tn = (n - j0).min(self.cols);
                let mut p0 = 0;
                while p0 < k {
                    let tk = (k - p0).min(self.inner);

                    // Load smem_a from A[i0..i0+tm, p0..p0+tk]
                    load_tile(a, k, i0, p0, tm, tk, &mut smem_a, self.inner);

                    // Load smem_b from B[p0..p0+tk, j0..j0+tn]
                    load_tile(b, n, p0, j0, tk, tn, &mut smem_b, self.cols);

                    // Multiply smem_a × smem_b and accumulate into C.
                    for ti in 0..tm {
                        for tj in 0..tn {
                            let mut sum = 0.0f32;
                            for tp in 0..tk {
                                sum = smem_a[ti * self.inner + tp]
                                    .mul_add(smem_b[tp * self.cols + tj], sum);
                            }
                            c[(i0 + ti) * n + (j0 + tj)] += alpha * sum;
                        }
                    }

                    p0 += self.inner;
                }
                j0 += self.cols;
            }
            i0 += self.rows;
        }
    }
}

/// Load a sub-matrix from `src` (stride `src_cols`) into `dst` (stride
/// `dst_cols`), zeroing any padding beyond the actual tile extent.
fn load_tile(
    src: &[f32],
    src_stride: usize,
    row_start: usize,
    col_start: usize,
    num_rows: usize,
    num_cols: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    for r in 0..num_rows {
        for ci in 0..num_cols {
            dst[r * dst_stride + ci] = src[(row_start + r) * src_stride + (col_start + ci)];
        }
        // Zero padding columns in the tile buffer.
        for ci in num_cols..dst_stride {
            dst[r * dst_stride + ci] = 0.0;
        }
    }
}

// ── Shared-memory optimization stubs ────────────────────────────────────

/// Shared-memory configuration for CUDA kernel launches.
///
/// These parameters guide tile-level data reuse on the GPU. On the CPU
/// reference path the values are recorded but do not affect execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMemoryConfig {
    /// Total shared memory per block in bytes.
    pub bytes_per_block: usize,
    /// Whether to prefer L1 cache over shared memory.
    pub prefer_l1: bool,
}

impl SharedMemoryConfig {
    /// Create a default shared-memory configuration (48 KiB, shared-memory
    /// preference).
    #[must_use]
    pub const fn default_config() -> Self {
        Self { bytes_per_block: 48 * 1024, prefer_l1: false }
    }

    /// Compute the maximum tile area (elements) that fits in the configured
    /// shared memory, assuming `f32` storage.
    #[must_use]
    pub const fn max_tile_elements(&self) -> usize {
        self.bytes_per_block / 4 // sizeof(f32) == 4
    }
}

/// Placeholder: configure shared-memory preference on the device.
///
/// On non-GPU builds this is a no-op.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn configure_shared_memory(_config: SharedMemoryConfig) {
    // TODO(gpu): call cudaFuncSetAttribute or equivalent.
}

// ── Batch GEMM ──────────────────────────────────────────────────────────

/// Batched f32 matrix multiplication.
///
/// Performs `batch_size` independent GEMM operations:
///   **C\[b\] = α·A\[b\]·B\[b\] + β·C\[b\]**   for each batch `b`.
///
/// Each matrix in the batch is stored contiguously: `a` has length
/// `batch_size * m * k`, etc.
///
/// # Errors
///
/// Returns [`MatmulError`] on invalid batch size or buffer mismatches.
#[allow(clippy::too_many_arguments)]
pub fn batch_gemm(
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<(), MatmulError> {
    if batch_size == 0 {
        return Err(MatmulError::EmptyBatch);
    }

    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    if a.len() < batch_size * a_stride {
        return Err(MatmulError::BufferTooSmall {
            required: batch_size * a_stride,
            actual: a.len(),
            context: "A batch",
        });
    }
    if b.len() < batch_size * b_stride {
        return Err(MatmulError::BufferTooSmall {
            required: batch_size * b_stride,
            actual: b.len(),
            context: "B batch",
        });
    }
    if c.len() < batch_size * c_stride {
        return Err(MatmulError::BufferTooSmall {
            required: batch_size * c_stride,
            actual: c.len(),
            context: "C batch",
        });
    }

    for batch in 0..batch_size {
        let a_off = batch * a_stride;
        let b_off = batch * b_stride;
        let c_off = batch * c_stride;
        gemm_inner(
            m,
            n,
            k,
            alpha,
            &a[a_off..a_off + a_stride],
            &b[b_off..b_off + b_stride],
            beta,
            &mut c[c_off..c_off + c_stride],
        );
    }

    Ok(())
}

// ── GPU dispatch stubs ──────────────────────────────────────────────────

/// Launch a CUDA GEMM kernel (GPU feature gate).
///
/// This is a placeholder for actual CUDA kernel dispatch; the signature
/// mirrors the host-side [`gemm`] function.
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<(), MatmulError> {
    // TODO(gpu): launch actual CUDA kernel via cuBLAS or custom kernel.
    // For now delegate to the CPU reference to keep tests green.
    gemm(m, n, k, alpha, a, b, beta, c)
}

/// Launch a CUDA binary matmul kernel (GPU feature gate).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn cuda_binary_matmul(
    m: usize,
    n: usize,
    k: usize,
    a_packed: &[u64],
    b_packed: &[u64],
    c: &mut [i32],
) -> Result<(), MatmulError> {
    // TODO(gpu): launch XNOR-popcount CUDA kernel.
    binary_matmul(m, n, k, a_packed, b_packed, c)
}

/// Launch a CUDA batch GEMM kernel (GPU feature gate).
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn cuda_batch_gemm(
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<(), MatmulError> {
    // TODO(gpu): launch batched cuBLAS GEMM.
    batch_gemm(batch_size, m, n, k, alpha, a, b, beta, c)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    // ── GEMM basic tests ────────────────────────────────────────────────

    #[test]
    fn gemm_2x2() {
        // A=[1,2;3,4] B=[5,6;7,8] => C=[19,22;43,50]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn gemm_2x3_times_3x2() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f32; 4];
        gemm(2, 2, 3, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn gemm_alpha_beta() {
        let a = [1.0, 0.0, 0.0, 1.0]; // identity
        let b = [3.0, 4.0, 5.0, 6.0];
        let mut c = [10.0, 20.0, 30.0, 40.0];
        // C = 2*I*B + 0.5*C_old
        gemm(2, 2, 2, 2.0, &a, &b, 0.5, &mut c).unwrap();
        assert!((c[0] - 11.0).abs() < 1e-5); // 2*3 + 0.5*10
        assert!((c[1] - 18.0).abs() < 1e-5); // 2*4 + 0.5*20
        assert!((c[2] - 25.0).abs() < 1e-5); // 2*5 + 0.5*30
        assert!((c[3] - 32.0).abs() < 1e-5); // 2*6 + 0.5*40
    }

    #[test]
    fn gemm_identity_left() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut c = [0.0f32; 9];
        gemm(3, 3, 3, 1.0, &identity, &b, 0.0, &mut c).unwrap();
        for (i, &val) in b.iter().enumerate() {
            assert!((c[i] - val).abs() < 1e-5);
        }
    }

    #[test]
    fn gemm_identity_right() {
        let a = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 9];
        gemm(3, 3, 3, 1.0, &a, &identity, 0.0, &mut c).unwrap();
        for (i, &val) in a.iter().enumerate() {
            assert!((c[i] - val).abs() < 1e-5);
        }
    }

    #[test]
    fn gemm_zero_matrix() {
        let a = [0.0f32; 4];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut c = [0.0f32; 4];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        for &val in &c {
            assert!((val).abs() < 1e-6);
        }
    }

    #[test]
    fn gemm_1x1() {
        let mut c = [0.0f32];
        gemm(1, 1, 1, 1.0, &[3.0], &[4.0], 0.0, &mut c).unwrap();
        assert!((c[0] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn gemm_large_k() {
        let k = 256;
        let a = vec![1.0f32; k];
        let b = vec![1.0f32; k];
        let mut c = [0.0f32];
        gemm(1, 1, k, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0] - k as f32).abs() < 1e-3);
    }

    #[test]
    fn gemm_negative_values() {
        let a = [-1.0, 2.0, 3.0, -4.0];
        let b = [5.0, -6.0, -7.0, 8.0];
        let mut c = [0.0f32; 4];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        // Row 0: (-1*5 + 2*-7, -1*-6 + 2*8) = (-19, 22)
        assert!((c[0] - (-19.0)).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        // Row 1: (3*5 + -4*-7, 3*-6 + -4*8) = (43, -50)
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - (-50.0)).abs() < 1e-5);
    }

    #[test]
    fn gemm_alpha_zero() {
        let a = [1.0; 4];
        let b = [1.0; 4];
        let mut c = [99.0; 4];
        gemm(2, 2, 2, 0.0, &a, &b, 1.0, &mut c).unwrap();
        for &val in &c {
            assert!((val - 99.0).abs() < 1e-5);
        }
    }

    #[test]
    fn gemm_beta_zero_clears_c() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [999.0, 888.0, 777.0, 666.0];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1]).abs() < 1e-5);
        assert!((c[2]).abs() < 1e-5);
        assert!((c[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gemm_rectangular_tall() {
        // A: 4×2, B: 2×1 → C: 4×1
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 1.0];
        let mut c = [0.0f32; 4];
        gemm(4, 1, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0] - 3.0).abs() < 1e-5);
        assert!((c[1] - 7.0).abs() < 1e-5);
        assert!((c[2] - 11.0).abs() < 1e-5);
        assert!((c[3] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn gemm_rectangular_wide() {
        // A: 1×3, B: 3×4 → C: 1×4
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut c = [0.0f32; 4];
        gemm(1, 4, 3, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
        assert!((c[2] - 3.0).abs() < 1e-5);
        assert!((c[3]).abs() < 1e-5);
    }

    // ── GEMM error tests ────────────────────────────────────────────────

    #[test]
    fn gemm_error_a_too_small() {
        let mut c = [0.0f32; 4];
        let err = gemm(2, 2, 2, 1.0, &[1.0; 3], &[1.0; 4], 0.0, &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { context: "A matrix", .. })));
    }

    #[test]
    fn gemm_error_b_too_small() {
        let mut c = [0.0f32; 4];
        let err = gemm(2, 2, 2, 1.0, &[1.0; 4], &[1.0; 3], 0.0, &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { context: "B matrix", .. })));
    }

    #[test]
    fn gemm_error_c_too_small() {
        let mut c = [0.0f32; 3];
        let err = gemm(2, 2, 2, 1.0, &[1.0; 4], &[1.0; 4], 0.0, &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { context: "C matrix", .. })));
    }

    #[test]
    fn gemm_zero_dimensions() {
        let mut c = [0.0f32; 0];
        assert!(gemm(0, 0, 0, 1.0, &[], &[], 0.0, &mut c).is_ok());
    }

    #[test]
    fn gemm_m_zero() {
        let mut c = [];
        assert!(gemm(0, 2, 3, 1.0, &[], &[0.0; 6], 0.0, &mut c).is_ok());
    }

    #[test]
    fn gemm_n_zero() {
        let mut c = [];
        assert!(gemm(2, 0, 3, 1.0, &[0.0; 6], &[], 0.0, &mut c).is_ok());
    }

    // ── Binary matmul tests ─────────────────────────────────────────────

    #[test]
    fn binary_matmul_all_ones() {
        // All bits set → all +1. XNOR all-match → max agreement.
        let a = [u64::MAX]; // 1 row, k=64
        let b = [u64::MAX]; // 1 col, k=64
        let mut c = [0i32];
        binary_matmul(1, 1, 64, &a, &b, &mut c).unwrap();
        // All bits agree → hamming = 0, result = 64 - 0 = 64
        assert_eq!(c[0], 64);
    }

    #[test]
    fn binary_matmul_all_zeros() {
        let a = [0u64];
        let b = [0u64];
        let mut c = [0i32];
        binary_matmul(1, 1, 64, &a, &b, &mut c).unwrap();
        // XOR(0,0) = 0, popcount = 0 → result = 64
        assert_eq!(c[0], 64);
    }

    #[test]
    fn binary_matmul_opposite() {
        let a = [u64::MAX];
        let b = [0u64];
        let mut c = [0i32];
        binary_matmul(1, 1, 64, &a, &b, &mut c).unwrap();
        // XOR = all ones, popcount = 64, result = 64 - 128 = -64
        assert_eq!(c[0], -64);
    }

    #[test]
    fn binary_matmul_half_match() {
        // Lower 32 bits set, upper 32 cleared.
        let a = [0x0000_0000_FFFF_FFFF_u64];
        let b = [u64::MAX];
        let mut c = [0i32];
        binary_matmul(1, 1, 64, &a, &b, &mut c).unwrap();
        // XOR: upper 32 bits differ. popcount = 32. result = 64 - 64 = 0
        assert_eq!(c[0], 0);
    }

    #[test]
    fn binary_matmul_2x2() {
        let a = [u64::MAX, 0u64]; // 2 rows
        let b = [u64::MAX, 0u64]; // 2 cols
        let mut c = [0i32; 4];
        binary_matmul(2, 2, 64, &a, &b, &mut c).unwrap();
        assert_eq!(c[0], 64); // MAX vs MAX
        assert_eq!(c[1], -64); // MAX vs 0
        assert_eq!(c[2], -64); // 0 vs MAX
        assert_eq!(c[3], 64); // 0 vs 0
    }

    #[test]
    fn binary_matmul_k128() {
        let a = [u64::MAX, u64::MAX]; // 1 row, k=128
        let b = [u64::MAX, u64::MAX]; // 1 col
        let mut c = [0i32];
        binary_matmul(1, 1, 128, &a, &b, &mut c).unwrap();
        assert_eq!(c[0], 128);
    }

    #[test]
    fn binary_matmul_error_k_not_multiple_64() {
        let mut c = [0i32];
        let err = binary_matmul(1, 1, 32, &[0], &[0], &mut c);
        assert!(matches!(err, Err(MatmulError::DimensionMismatch { .. })));
    }

    #[test]
    fn binary_matmul_error_k_zero() {
        let mut c = [0i32];
        let err = binary_matmul(1, 1, 0, &[], &[], &mut c);
        assert!(matches!(err, Err(MatmulError::DimensionMismatch { .. })));
    }

    #[test]
    fn binary_matmul_error_a_too_small() {
        let mut c = [0i32];
        let err = binary_matmul(1, 1, 64, &[], &[0], &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { .. })));
    }

    #[test]
    fn binary_matmul_error_b_too_small() {
        let mut c = [0i32];
        let err = binary_matmul(1, 1, 64, &[0], &[], &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { .. })));
    }

    #[test]
    fn binary_matmul_error_c_too_small() {
        let mut c = [];
        let err = binary_matmul(1, 1, 64, &[0], &[0], &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { .. })));
    }

    // ── Mixed-precision matmul tests ────────────────────────────────────

    #[test]
    fn mixed_precision_all_plus_one() {
        // All bits 1 → all +1 weights, activations = [1.0; 64]
        let weights = [u64::MAX]; // 1 row, k=64
        let activations = vec![1.0f32; 64]; // k=64, n=1
        let mut output = [0.0f32];
        mixed_precision_matmul(1, 1, 64, &weights, &activations, &mut output).unwrap();
        assert!((output[0] - 64.0).abs() < 1e-5);
    }

    #[test]
    fn mixed_precision_all_minus_one() {
        // All bits 0 → all -1 weights
        let weights = [0u64];
        let activations = vec![1.0f32; 64];
        let mut output = [0.0f32];
        mixed_precision_matmul(1, 1, 64, &weights, &activations, &mut output).unwrap();
        assert!((output[0] - (-64.0)).abs() < 1e-5);
    }

    #[test]
    fn mixed_precision_half_bits() {
        // Lower 32 bits = +1, upper 32 = -1, activations all 1.0
        let weights = [0x0000_0000_FFFF_FFFF_u64];
        let activations = vec![1.0f32; 64];
        let mut output = [0.0f32];
        mixed_precision_matmul(1, 1, 64, &weights, &activations, &mut output).unwrap();
        // 32 * (+1) + 32 * (-1) = 0
        assert!((output[0]).abs() < 1e-5);
    }

    #[test]
    fn mixed_precision_2x1() {
        let weights = [u64::MAX, 0u64]; // 2 rows
        let activations = vec![2.0f32; 64]; // k=64, n=1
        let mut output = [0.0f32; 2];
        mixed_precision_matmul(2, 1, 64, &weights, &activations, &mut output).unwrap();
        assert!((output[0] - 128.0).abs() < 1e-4); // all +1 * 2.0 * 64
        assert!((output[1] - (-128.0)).abs() < 1e-4); // all -1 * 2.0 * 64
    }

    #[test]
    fn mixed_precision_1x2_output() {
        let weights = [u64::MAX]; // 1 row, k=64
        // k=64, n=2: activations laid out row-major
        let mut activations = vec![0.0f32; 128];
        for i in 0..64 {
            activations[i * 2] = 1.0; // column 0
            activations[i * 2 + 1] = 2.0; // column 1
        }
        let mut output = [0.0f32; 2];
        mixed_precision_matmul(1, 2, 64, &weights, &activations, &mut output).unwrap();
        assert!((output[0] - 64.0).abs() < 1e-4);
        assert!((output[1] - 128.0).abs() < 1e-4);
    }

    #[test]
    fn mixed_precision_error_k_not_multiple_64() {
        let mut output = [0.0f32];
        let err = mixed_precision_matmul(1, 1, 32, &[0], &[0.0; 32], &mut output);
        assert!(matches!(err, Err(MatmulError::DimensionMismatch { .. })));
    }

    #[test]
    fn mixed_precision_error_weights_too_small() {
        let mut output = [0.0f32];
        let err = mixed_precision_matmul(1, 1, 64, &[], &[0.0; 64], &mut output);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { context: "weights", .. })));
    }

    #[test]
    fn mixed_precision_error_activations_too_small() {
        let mut output = [0.0f32];
        let err = mixed_precision_matmul(1, 1, 64, &[0], &[0.0; 32], &mut output);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { context: "activations", .. })));
    }

    #[test]
    fn mixed_precision_error_output_too_small() {
        let mut output = [];
        let err = mixed_precision_matmul(1, 1, 64, &[0], &[0.0; 64], &mut output);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { context: "output", .. })));
    }

    // ── Tiled matmul tests ──────────────────────────────────────────────

    #[test]
    fn tiled_matches_gemm_small() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c_ref = [0.0f32; 9];
        let mut c_tiled = [0.0f32; 9];

        gemm(3, 3, 3, 1.0, &a, &b, 0.0, &mut c_ref).unwrap();

        let tiled = TiledMatmul::new(2, 2, 2).unwrap();
        tiled.execute(3, 3, 3, 1.0, &a, &b, 0.0, &mut c_tiled).unwrap();

        for i in 0..9 {
            assert!(
                (c_ref[i] - c_tiled[i]).abs() < 1e-4,
                "mismatch at {i}: ref={}, tiled={}",
                c_ref[i],
                c_tiled[i]
            );
        }
    }

    #[test]
    fn tiled_matches_gemm_rectangular() {
        let (m, n, k) = (4, 5, 3);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
        let mut c_ref = vec![0.0f32; m * n];
        let mut c_tiled = vec![0.0f32; m * n];

        gemm(m, n, k, 1.0, &a, &b, 0.0, &mut c_ref).unwrap();

        let tiled = TiledMatmul::new(2, 3, 2).unwrap();
        tiled.execute(m, n, k, 1.0, &a, &b, 0.0, &mut c_tiled).unwrap();

        for i in 0..m * n {
            assert!((c_ref[i] - c_tiled[i]).abs() < 1e-3, "mismatch at {i}");
        }
    }

    #[test]
    fn tiled_with_alpha_beta() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [10.0, 20.0, 30.0, 40.0];
        let tiled = TiledMatmul::new(1, 1, 1).unwrap();
        tiled.execute(2, 2, 2, 2.0, &a, &b, 0.5, &mut c).unwrap();
        assert!((c[0] - 15.0).abs() < 1e-5); // 2*5 + 0.5*10
        assert!((c[1] - 22.0).abs() < 1e-5); // 2*6 + 0.5*20
        assert!((c[2] - 29.0).abs() < 1e-5); // 2*7 + 0.5*30
        assert!((c[3] - 36.0).abs() < 1e-5); // 2*8 + 0.5*40
    }

    #[test]
    fn tiled_tile_larger_than_matrix() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        let tiled = TiledMatmul::new(16, 16, 16).unwrap();
        tiled.execute(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn tiled_tile_size_one() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c_ref = [0.0f32; 4];
        let mut c_tiled = [0.0f32; 4];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c_ref).unwrap();
        let tiled = TiledMatmul::new(1, 1, 1).unwrap();
        tiled.execute(2, 2, 2, 1.0, &a, &b, 0.0, &mut c_tiled).unwrap();
        for i in 0..4 {
            assert!((c_ref[i] - c_tiled[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn tiled_error_zero_tile_m() {
        assert!(matches!(TiledMatmul::new(0, 2, 2), Err(MatmulError::InvalidTileSize { .. })));
    }

    #[test]
    fn tiled_error_zero_tile_n() {
        assert!(matches!(TiledMatmul::new(2, 0, 2), Err(MatmulError::InvalidTileSize { .. })));
    }

    #[test]
    fn tiled_error_zero_tile_k() {
        assert!(matches!(TiledMatmul::new(2, 2, 0), Err(MatmulError::InvalidTileSize { .. })));
    }

    #[test]
    fn tiled_error_a_too_small() {
        let tiled = TiledMatmul::new(2, 2, 2).unwrap();
        let mut c = [0.0f32; 4];
        let err = tiled.execute(2, 2, 2, 1.0, &[1.0; 3], &[1.0; 4], 0.0, &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { .. })));
    }

    #[test]
    fn tiled_accessors() {
        let t = TiledMatmul::new(4, 8, 16).unwrap();
        assert_eq!(t.tile_m(), 4);
        assert_eq!(t.tile_n(), 8);
        assert_eq!(t.tile_k(), 16);
    }

    // ── Shared-memory config tests ──────────────────────────────────────

    #[test]
    fn shared_memory_default() {
        let cfg = SharedMemoryConfig::default_config();
        assert_eq!(cfg.bytes_per_block, 48 * 1024);
        assert!(!cfg.prefer_l1);
    }

    #[test]
    fn shared_memory_max_tile_elements() {
        let cfg = SharedMemoryConfig { bytes_per_block: 1024, prefer_l1: false };
        assert_eq!(cfg.max_tile_elements(), 256);
    }

    #[test]
    fn shared_memory_custom() {
        let cfg = SharedMemoryConfig { bytes_per_block: 96 * 1024, prefer_l1: true };
        assert_eq!(cfg.bytes_per_block, 96 * 1024);
        assert!(cfg.prefer_l1);
        assert_eq!(cfg.max_tile_elements(), 96 * 1024 / 4);
    }

    // ── Batch GEMM tests ────────────────────────────────────────────────

    #[test]
    fn batch_gemm_single_batch() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        batch_gemm(1, 2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        let mut c_ref = [0.0f32; 4];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c_ref).unwrap();
        for i in 0..4 {
            assert!((c[i] - c_ref[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn batch_gemm_two_batches() {
        let a = [
            1.0, 0.0, 0.0, 1.0, // batch 0: identity
            2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
        ];
        let b = [
            3.0, 4.0, 5.0, 6.0, // batch 0
            3.0, 4.0, 5.0, 6.0, // batch 1
        ];
        let mut c = [0.0f32; 8];
        batch_gemm(2, 2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        // Batch 0: I * B = B
        assert!((c[0] - 3.0).abs() < 1e-5);
        assert!((c[1] - 4.0).abs() < 1e-5);
        assert!((c[2] - 5.0).abs() < 1e-5);
        assert!((c[3] - 6.0).abs() < 1e-5);
        // Batch 1: 2I * B = 2B
        assert!((c[4] - 6.0).abs() < 1e-5);
        assert!((c[5] - 8.0).abs() < 1e-5);
        assert!((c[6] - 10.0).abs() < 1e-5);
        assert!((c[7] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn batch_gemm_with_beta() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [10.0, 20.0, 30.0, 40.0];
        batch_gemm(1, 2, 2, 2, 1.0, &a, &b, 0.5, &mut c).unwrap();
        assert!((c[0] - 6.0).abs() < 1e-5); // 1 + 0.5*10
        assert!((c[1] - 10.0).abs() < 1e-5); // 0 + 0.5*20
        assert!((c[2] - 15.0).abs() < 1e-5); // 0 + 0.5*30
        assert!((c[3] - 21.0).abs() < 1e-5); // 1 + 0.5*40
    }

    #[test]
    fn batch_gemm_error_empty_batch() {
        let mut c = [0.0f32; 4];
        let err = batch_gemm(0, 2, 2, 2, 1.0, &[0.0; 4], &[0.0; 4], 0.0, &mut c);
        assert!(matches!(err, Err(MatmulError::EmptyBatch)));
    }

    #[test]
    fn batch_gemm_error_a_too_small() {
        let mut c = [0.0f32; 8];
        let err = batch_gemm(2, 2, 2, 2, 1.0, &[0.0; 4], &[0.0; 8], 0.0, &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { context: "A batch", .. })));
    }

    #[test]
    fn batch_gemm_error_b_too_small() {
        let mut c = [0.0f32; 8];
        let err = batch_gemm(2, 2, 2, 2, 1.0, &[0.0; 8], &[0.0; 4], 0.0, &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { context: "B batch", .. })));
    }

    #[test]
    fn batch_gemm_error_c_too_small() {
        let mut c = [0.0f32; 4];
        let err = batch_gemm(2, 2, 2, 2, 1.0, &[0.0; 8], &[0.0; 8], 0.0, &mut c);
        assert!(matches!(err, Err(MatmulError::BufferTooSmall { context: "C batch", .. })));
    }

    #[test]
    fn batch_gemm_three_batches_1x1() {
        let a = [2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0];
        let mut c = [0.0f32; 3];
        batch_gemm(3, 1, 1, 1, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0] - 10.0).abs() < 1e-5);
        assert!((c[1] - 18.0).abs() < 1e-5);
        assert!((c[2] - 28.0).abs() < 1e-5);
    }

    // ── Error display tests ─────────────────────────────────────────────

    #[test]
    fn error_display_dimension_mismatch() {
        let e = MatmulError::DimensionMismatch { expected: 64, actual: 32, context: "k alignment" };
        let msg = format!("{e}");
        assert!(msg.contains("dimension mismatch"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn error_display_buffer_too_small() {
        let e = MatmulError::BufferTooSmall { required: 100, actual: 50, context: "A matrix" };
        let msg = format!("{e}");
        assert!(msg.contains("buffer too small"));
    }

    #[test]
    fn error_display_invalid_tile() {
        let msg = format!("{}", MatmulError::InvalidTileSize { tile_size: 0 });
        assert!(msg.contains("invalid tile size"));
    }

    #[test]
    fn error_display_empty_batch() {
        let msg = format!("{}", MatmulError::EmptyBatch);
        assert!(msg.contains("batch size"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(MatmulError::EmptyBatch);
        assert!(e.to_string().contains("batch"));
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = MatmulError::EmptyBatch;
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn gemm_very_small_values() {
        let a = [1e-20f32; 4];
        let b = [1e-20f32; 4];
        let mut c = [0.0f32; 4];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        for &val in &c {
            assert!(val >= 0.0);
            assert!(val < 1e-30);
        }
    }

    #[test]
    fn gemm_large_values() {
        let a = [1e10f32; 4];
        let b = [1e10f32; 4];
        let mut c = [0.0f32; 4];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        for &val in &c {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn gemm_extra_buffer_space() {
        // Buffers larger than needed should work fine.
        let a = [1.0, 2.0, 3.0, 4.0, 99.0, 99.0];
        let b = [5.0, 6.0, 7.0, 8.0, 99.0, 99.0];
        let mut c = [0.0f32; 6];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!((c[0] - 19.0).abs() < 1e-5);
    }

    #[test]
    fn tiled_matches_gemm_larger() {
        let (m, n, k) = (8, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
        let mut c_ref = vec![0.0f32; m * n];
        let mut c_tiled = vec![0.0f32; m * n];

        gemm(m, n, k, 1.0, &a, &b, 0.0, &mut c_ref).unwrap();
        let tiled = TiledMatmul::new(3, 3, 3).unwrap();
        tiled.execute(m, n, k, 1.0, &a, &b, 0.0, &mut c_tiled).unwrap();

        for i in 0..m * n {
            assert!(
                (c_ref[i] - c_tiled[i]).abs() < 1e-3,
                "mismatch at {i}: ref={}, tiled={}",
                c_ref[i],
                c_tiled[i]
            );
        }
    }

    // ── Property tests ──────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Generate a dimension in a small range for quick tests.
        fn dim() -> impl Strategy<Value = usize> {
            1..=16usize
        }

        proptest! {
            #[test]
            fn gemm_identity_property(n in 1..=8usize) {
                let mut identity = vec![0.0f32; n * n];
                for i in 0..n {
                    identity[i * n + i] = 1.0;
                }
                let a: Vec<f32> = (0..n*n).map(|i| i as f32).collect();
                let mut c = vec![0.0f32; n * n];
                gemm(n, n, n, 1.0, &identity, &a, 0.0, &mut c).unwrap();
                for i in 0..n*n {
                    prop_assert!((c[i] - a[i]).abs() < 1e-3,
                        "identity * A != A at idx {}", i);
                }
            }

            #[test]
            fn gemm_zero_alpha_preserves_c(m in dim(), n in dim(), k in dim()) {
                let a = vec![1.0f32; m * k];
                let b = vec![1.0f32; k * n];
                let original = vec![42.0f32; m * n];
                let mut c = original.clone();
                gemm(m, n, k, 0.0, &a, &b, 1.0, &mut c).unwrap();
                for i in 0..m*n {
                    prop_assert!((c[i] - original[i]).abs() < 1e-5);
                }
            }

            #[test]
            fn gemm_beta_zero_ignores_c_init(m in dim(), n in dim(), k in dim()) {
                let a = vec![1.0f32; m * k];
                let b = vec![1.0f32; k * n];
                let mut c1 = vec![0.0f32; m * n];
                let mut c2 = vec![999.0f32; m * n];
                gemm(m, n, k, 1.0, &a, &b, 0.0, &mut c1).unwrap();
                gemm(m, n, k, 1.0, &a, &b, 0.0, &mut c2).unwrap();
                for i in 0..m*n {
                    prop_assert!((c1[i] - c2[i]).abs() < 1e-4,
                        "beta=0 should ignore initial C");
                }
            }

            #[test]
            fn tiled_matches_reference(m in 1..=6usize, n in 1..=6usize, k in 1..=6usize,
                                        tm in 1..=4usize, tn in 1..=4usize, tk in 1..=4usize) {
                let a: Vec<f32> = (0..m*k).map(|i| (i % 7) as f32 - 3.0).collect();
                let b: Vec<f32> = (0..k*n).map(|i| (i % 5) as f32 - 2.0).collect();
                let mut c_ref = vec![0.0f32; m * n];
                let mut c_tiled = vec![0.0f32; m * n];

                gemm(m, n, k, 1.0, &a, &b, 0.0, &mut c_ref).unwrap();
                let tiled = TiledMatmul::new(tm, tn, tk).unwrap();
                tiled.execute(m, n, k, 1.0, &a, &b, 0.0, &mut c_tiled).unwrap();

                for i in 0..m*n {
                    prop_assert!((c_ref[i] - c_tiled[i]).abs() < 1e-3,
                        "tiled diverged from ref at idx {}: ref={}, tiled={}",
                        i, c_ref[i], c_tiled[i]);
                }
            }

            #[test]
            fn batch_gemm_matches_individual(batch in 1..=4usize, m in dim(), n in dim(), k in dim()) {
                let a_stride = m * k;
                let b_stride = k * n;
                let c_stride = m * n;
                let a: Vec<f32> = (0..batch * a_stride).map(|i| (i % 11) as f32 - 5.0).collect();
                let b: Vec<f32> = (0..batch * b_stride).map(|i| (i % 7) as f32 - 3.0).collect();
                let mut c_batch = vec![0.0f32; batch * c_stride];
                batch_gemm(batch, m, n, k, 1.0, &a, &b, 0.0, &mut c_batch).unwrap();

                for bi in 0..batch {
                    let mut c_single = vec![0.0f32; c_stride];
                    gemm(m, n, k, 1.0,
                         &a[bi * a_stride..(bi + 1) * a_stride],
                         &b[bi * b_stride..(bi + 1) * b_stride],
                         0.0, &mut c_single).unwrap();
                    for i in 0..c_stride {
                        prop_assert!((c_batch[bi * c_stride + i] - c_single[i]).abs() < 1e-3,
                            "batch {} idx {} diverged", bi, i);
                    }
                }
            }

            #[test]
            fn binary_matmul_range(m in 1..=4usize, n in 1..=4usize) {
                // k=64, all-ones. Result should be in [-64, 64].
                let a = vec![u64::MAX; m];
                let b = vec![u64::MAX; n];
                let mut c = vec![0i32; m * n];
                binary_matmul(m, n, 64, &a, &b, &mut c).unwrap();
                for &val in &c {
                    prop_assert!((-64..=64).contains(&val));
                }
            }
        }
    }
}
