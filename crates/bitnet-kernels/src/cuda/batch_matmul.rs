//! CUDA batched matrix multiplication kernel with CPU reference implementations.
//!
//! # Kernel strategy
//!
//! Batched GEMM for dense floating-point matrices, operating on independent
//! matrix pairs indexed by a batch dimension. The CUDA path provides two
//! kernel variants:
//!
//! - **Tiled shared-memory**: 2-D thread-block tiling where each block
//!   computes a `TILE_SIZE × TILE_SIZE` output sub-matrix, streaming
//!   K-wide slices through shared memory. The batch dimension maps to
//!   `gridDim.z`.
//! - **Tensor Core WMMA**: Warp Matrix Multiply-Accumulate (WMMA) on
//!   SM 7.0+ (Volta and later). Each warp computes a 16×16×16 fragment
//!   using `nvcuda::wmma` intrinsics.
//!
//! # CPU reference implementations
//!
//! Six reference kernels are provided for correctness testing and non-GPU
//! environments:
//!
//! - [`batch_matmul`]: Standard batched C = α·A·B + β·C
//! - [`batch_matmul_transposed`]: With per-operand transpose flags
//! - [`strided_batch_matmul`]: Non-contiguous batch dimensions via strides
//! - [`fused_batch_matmul_bias`]: Matmul + bias addition in one pass
//! - [`fused_batch_matmul_relu`]: Matmul + ReLU activation fused
//! - [`quantized_batch_matmul`]: Int8 matmul with per-tensor scale factors
//!
//! # GPU launch stubs
//!
//! [`launch_batch_matmul`] and [`launch_batch_matmul_wmma`] are feature-gated
//! behind `#[cfg(any(feature = "gpu", feature = "cuda"))]` and currently
//! return scaffold errors until PTX compilation is wired up.

use std::fmt;

// ── Error type ────────────────────────────────────────────────────────

/// Errors specific to batched matrix multiplication operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchMatmulError {
    /// A matrix dimension is zero.
    ZeroDimension {
        /// Which dimension was zero.
        field: &'static str,
    },
    /// An input or output buffer has the wrong length.
    BufferSizeMismatch {
        /// Name of the buffer (e.g. "A", "B", "output").
        name: &'static str,
        /// Minimum expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Stride configuration is invalid.
    InvalidStride {
        /// Explanation of the invalid stride.
        reason: String,
    },
    /// Scale factor is invalid (e.g. zero or NaN).
    InvalidScale {
        /// Explanation.
        reason: String,
    },
    /// GPU kernel launch failed.
    GpuLaunchFailed {
        /// Explanation.
        reason: String,
    },
}

impl fmt::Display for BatchMatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { field } => {
                write!(f, "batch matmul dimension must be non-zero: {field}")
            }
            Self::BufferSizeMismatch { name, expected, actual } => {
                write!(f, "buffer '{name}' size mismatch: expected >= {expected}, got {actual}")
            }
            Self::InvalidStride { reason } => {
                write!(f, "invalid stride configuration: {reason}")
            }
            Self::InvalidScale { reason } => {
                write!(f, "invalid scale factor: {reason}")
            }
            Self::GpuLaunchFailed { reason } => {
                write!(f, "GPU batch matmul launch failed: {reason}")
            }
        }
    }
}

impl std::error::Error for BatchMatmulError {}

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for batched matrix multiplication.
///
/// Describes the shape and scalar parameters for
/// `C[b] = alpha * op(A[b]) · op(B[b]) + beta * C[b]`
/// across `batch_size` independent matrix pairs.
#[derive(Debug, Clone)]
pub struct BatchMatmulConfig {
    /// Number of independent matrix pairs.
    pub batch_size: usize,
    /// Number of output rows (after optional transpose of A).
    pub m: usize,
    /// Number of output columns (after optional transpose of B).
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
    /// Scalar multiplier for the product (default 1.0).
    pub alpha: f32,
    /// Scalar multiplier for the existing output (default 0.0).
    pub beta: f32,
    /// Transpose the A operand before multiplication.
    pub transpose_a: bool,
    /// Transpose the B operand before multiplication.
    pub transpose_b: bool,
}

impl Default for BatchMatmulConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            m: 1,
            n: 1,
            k: 1,
            alpha: 1.0,
            beta: 0.0,
            transpose_a: false,
            transpose_b: false,
        }
    }
}

impl BatchMatmulConfig {
    /// Create a config for the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`BatchMatmulError::ZeroDimension`] if any dimension is zero.
    pub fn new(batch_size: usize, m: usize, n: usize, k: usize) -> Result<Self, BatchMatmulError> {
        if batch_size == 0 {
            return Err(BatchMatmulError::ZeroDimension { field: "batch_size" });
        }
        if m == 0 {
            return Err(BatchMatmulError::ZeroDimension { field: "m" });
        }
        if n == 0 {
            return Err(BatchMatmulError::ZeroDimension { field: "n" });
        }
        if k == 0 {
            return Err(BatchMatmulError::ZeroDimension { field: "k" });
        }
        Ok(Self { batch_size, m, n, k, ..Self::default() })
    }

    /// Set scalar multipliers.
    pub fn with_alpha_beta(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set transpose flags.
    pub fn with_transpose(mut self, transpose_a: bool, transpose_b: bool) -> Self {
        self.transpose_a = transpose_a;
        self.transpose_b = transpose_b;
        self
    }

    /// Compute physical A dimensions (rows, cols) accounting for transpose.
    fn a_physical(&self) -> (usize, usize) {
        if self.transpose_a { (self.k, self.m) } else { (self.m, self.k) }
    }

    /// Compute physical B dimensions (rows, cols) accounting for transpose.
    fn b_physical(&self) -> (usize, usize) {
        if self.transpose_b { (self.n, self.k) } else { (self.k, self.n) }
    }

    /// Compute the CUDA grid dimensions `(grid_x, grid_y, batch_size)`.
    pub fn grid_dim(&self, tile_size: u32) -> (u32, u32, u32) {
        let grid_x = (self.n as u32).div_ceil(tile_size);
        let grid_y = (self.m as u32).div_ceil(tile_size);
        (grid_x, grid_y, self.batch_size as u32)
    }
}

// ── Stride configuration ──────────────────────────────────────────────

/// Stride configuration for non-contiguous batched matmul.
///
/// Each stride is the number of elements between consecutive batches
/// in the respective buffer.
#[derive(Debug, Clone)]
pub struct BatchStrides {
    /// Elements between consecutive A matrices.
    pub stride_a: usize,
    /// Elements between consecutive B matrices.
    pub stride_b: usize,
    /// Elements between consecutive output matrices.
    pub stride_out: usize,
}

// ── Validation helpers ────────────────────────────────────────────────

fn validate_buffers(
    a: &[f32],
    b: &[f32],
    out: &[f32],
    config: &BatchMatmulConfig,
) -> Result<(), BatchMatmulError> {
    let (a_rows, a_cols) = config.a_physical();
    let (b_rows, b_cols) = config.b_physical();

    let a_required = config.batch_size * a_rows * a_cols;
    let b_required = config.batch_size * b_rows * b_cols;
    let out_required = config.batch_size * config.m * config.n;

    if a.len() < a_required {
        return Err(BatchMatmulError::BufferSizeMismatch {
            name: "A",
            expected: a_required,
            actual: a.len(),
        });
    }
    if b.len() < b_required {
        return Err(BatchMatmulError::BufferSizeMismatch {
            name: "B",
            expected: b_required,
            actual: b.len(),
        });
    }
    if out.len() < out_required {
        return Err(BatchMatmulError::BufferSizeMismatch {
            name: "output",
            expected: out_required,
            actual: out.len(),
        });
    }
    Ok(())
}

// ── CPU reference: batch_matmul ───────────────────────────────────────

/// Batched dense matrix multiplication (CPU reference).
///
/// Computes `C[b] = alpha * A[b] · B[b] + beta * C[b]` for each batch `b`.
///
/// # Layout
///
/// - `a`: row-major `[batch, M, K]` f32
/// - `b_mat`: row-major `[batch, K, N]` f32
/// - `out`: row-major `[batch, M, N]` f32
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent with the config.
pub fn batch_matmul(
    a: &[f32],
    b_mat: &[f32],
    out: &mut [f32],
    config: &BatchMatmulConfig,
) -> Result<(), BatchMatmulError> {
    let no_transpose_config =
        BatchMatmulConfig { transpose_a: false, transpose_b: false, ..config.clone() };
    validate_buffers(a, b_mat, out, &no_transpose_config)?;

    let m = config.m;
    let n = config.n;
    let k = config.k;
    let alpha = config.alpha;
    let beta = config.beta;

    let a_stride = m * k;
    let b_stride = k * n;
    let out_stride = m * n;

    for batch in 0..config.batch_size {
        let a_off = batch * a_stride;
        let b_off = batch * b_stride;
        let o_off = batch * out_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[a_off + i * k + l] * b_mat[b_off + l * n + j];
                }
                let idx = o_off + i * n + j;
                out[idx] = alpha * acc + beta * out[idx];
            }
        }
    }
    Ok(())
}

// ── CPU reference: batch_matmul_transposed ────────────────────────────

/// Batched matrix multiplication with per-operand transpose (CPU reference).
///
/// Computes `C[b] = alpha * op(A[b]) · op(B[b]) + beta * C[b]` where
/// `op(X)` is `X` or `X^T` depending on the config flags.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent with the config.
pub fn batch_matmul_transposed(
    a: &[f32],
    b_mat: &[f32],
    out: &mut [f32],
    config: &BatchMatmulConfig,
) -> Result<(), BatchMatmulError> {
    validate_buffers(a, b_mat, out, config)?;

    let m = config.m;
    let n = config.n;
    let k = config.k;
    let alpha = config.alpha;
    let beta = config.beta;

    let (a_rows, a_cols) = config.a_physical();
    let (b_rows, b_cols) = config.b_physical();
    let a_stride = a_rows * a_cols;
    let b_stride = b_rows * b_cols;
    let out_stride = m * n;

    for batch in 0..config.batch_size {
        let a_off = batch * a_stride;
        let b_off = batch * b_stride;
        let o_off = batch * out_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    let a_val = if config.transpose_a {
                        a[a_off + l * m + i]
                    } else {
                        a[a_off + i * k + l]
                    };
                    let b_val = if config.transpose_b {
                        b_mat[b_off + j * k + l]
                    } else {
                        b_mat[b_off + l * n + j]
                    };
                    acc += a_val * b_val;
                }
                let idx = o_off + i * n + j;
                out[idx] = alpha * acc + beta * out[idx];
            }
        }
    }
    Ok(())
}

// ── CPU reference: strided_batch_matmul ───────────────────────────────

/// Strided batched matrix multiplication for non-contiguous batch layouts.
///
/// Each batch element is located at `base + batch_index * stride` elements
/// from the start of the buffer, allowing interleaved or padded memory
/// layouts.
///
/// # Errors
///
/// Returns an error if strides are too small or buffers are too short.
pub fn strided_batch_matmul(
    a: &[f32],
    b_mat: &[f32],
    out: &mut [f32],
    config: &BatchMatmulConfig,
    strides: &BatchStrides,
) -> Result<(), BatchMatmulError> {
    let m = config.m;
    let n = config.n;
    let k = config.k;

    if strides.stride_a < m * k {
        return Err(BatchMatmulError::InvalidStride {
            reason: format!("stride_a ({}) < M*K ({})", strides.stride_a, m * k),
        });
    }
    if strides.stride_b < k * n {
        return Err(BatchMatmulError::InvalidStride {
            reason: format!("stride_b ({}) < K*N ({})", strides.stride_b, k * n),
        });
    }
    if strides.stride_out < m * n {
        return Err(BatchMatmulError::InvalidStride {
            reason: format!("stride_out ({}) < M*N ({})", strides.stride_out, m * n),
        });
    }

    if config.batch_size > 0 {
        let a_required = (config.batch_size - 1) * strides.stride_a + m * k;
        let b_required = (config.batch_size - 1) * strides.stride_b + k * n;
        let out_required = (config.batch_size - 1) * strides.stride_out + m * n;

        if a.len() < a_required {
            return Err(BatchMatmulError::BufferSizeMismatch {
                name: "A",
                expected: a_required,
                actual: a.len(),
            });
        }
        if b_mat.len() < b_required {
            return Err(BatchMatmulError::BufferSizeMismatch {
                name: "B",
                expected: b_required,
                actual: b_mat.len(),
            });
        }
        if out.len() < out_required {
            return Err(BatchMatmulError::BufferSizeMismatch {
                name: "output",
                expected: out_required,
                actual: out.len(),
            });
        }
    }

    let alpha = config.alpha;
    let beta = config.beta;

    for batch in 0..config.batch_size {
        let a_off = batch * strides.stride_a;
        let b_off = batch * strides.stride_b;
        let o_off = batch * strides.stride_out;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[a_off + i * k + l] * b_mat[b_off + l * n + j];
                }
                let idx = o_off + i * n + j;
                out[idx] = alpha * acc + beta * out[idx];
            }
        }
    }
    Ok(())
}

// ── CPU reference: fused_batch_matmul_bias ────────────────────────────

/// Fused batched matmul + bias addition (CPU reference).
///
/// Computes `C[b] = alpha * A[b] · B[b] + beta * C[b] + bias` in a
/// single pass, where `bias` is a row vector of length `N` broadcast
/// across all rows and batches.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn fused_batch_matmul_bias(
    a: &[f32],
    b_mat: &[f32],
    bias: &[f32],
    out: &mut [f32],
    config: &BatchMatmulConfig,
) -> Result<(), BatchMatmulError> {
    let no_transpose_config =
        BatchMatmulConfig { transpose_a: false, transpose_b: false, ..config.clone() };
    validate_buffers(a, b_mat, out, &no_transpose_config)?;

    if bias.len() < config.n {
        return Err(BatchMatmulError::BufferSizeMismatch {
            name: "bias",
            expected: config.n,
            actual: bias.len(),
        });
    }

    let m = config.m;
    let n = config.n;
    let k = config.k;
    let alpha = config.alpha;
    let beta = config.beta;

    let a_stride = m * k;
    let b_stride = k * n;
    let out_stride = m * n;

    for batch in 0..config.batch_size {
        let a_off = batch * a_stride;
        let b_off = batch * b_stride;
        let o_off = batch * out_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[a_off + i * k + l] * b_mat[b_off + l * n + j];
                }
                let idx = o_off + i * n + j;
                out[idx] = alpha * acc + beta * out[idx] + bias[j];
            }
        }
    }
    Ok(())
}

// ── CPU reference: fused_batch_matmul_relu ────────────────────────────

/// Fused batched matmul + ReLU activation (CPU reference).
///
/// Computes `C[b] = max(0, alpha * A[b] · B[b] + beta * C[b])` in a
/// single pass, avoiding a separate activation kernel launch.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn fused_batch_matmul_relu(
    a: &[f32],
    b_mat: &[f32],
    out: &mut [f32],
    config: &BatchMatmulConfig,
) -> Result<(), BatchMatmulError> {
    let no_transpose_config =
        BatchMatmulConfig { transpose_a: false, transpose_b: false, ..config.clone() };
    validate_buffers(a, b_mat, out, &no_transpose_config)?;

    let m = config.m;
    let n = config.n;
    let k = config.k;
    let alpha = config.alpha;
    let beta = config.beta;

    let a_stride = m * k;
    let b_stride = k * n;
    let out_stride = m * n;

    for batch in 0..config.batch_size {
        let a_off = batch * a_stride;
        let b_off = batch * b_stride;
        let o_off = batch * out_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[a_off + i * k + l] * b_mat[b_off + l * n + j];
                }
                let idx = o_off + i * n + j;
                let val = alpha * acc + beta * out[idx];
                out[idx] = if val > 0.0 { val } else { 0.0 };
            }
        }
    }
    Ok(())
}

// ── CPU reference: quantized_batch_matmul ─────────────────────────────

/// Quantized int8 batched matmul with per-tensor scale factors (CPU).
///
/// Computes `C[b] = scale_a * scale_b * (A_i8[b] · B_i8[b])` using
/// `i32` accumulation to avoid overflow, then scales the result to
/// `f32`.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent or scales are
/// invalid (zero or non-finite).
pub fn quantized_batch_matmul(
    a: &[i8],
    b_mat: &[i8],
    out: &mut [f32],
    config: &BatchMatmulConfig,
    scale_a: f32,
    scale_b: f32,
) -> Result<(), BatchMatmulError> {
    if !scale_a.is_finite() || scale_a == 0.0 {
        return Err(BatchMatmulError::InvalidScale {
            reason: format!("scale_a must be finite and non-zero, got {scale_a}"),
        });
    }
    if !scale_b.is_finite() || scale_b == 0.0 {
        return Err(BatchMatmulError::InvalidScale {
            reason: format!("scale_b must be finite and non-zero, got {scale_b}"),
        });
    }

    let m = config.m;
    let n = config.n;
    let k = config.k;

    let a_required = config.batch_size * m * k;
    let b_required = config.batch_size * k * n;
    let out_required = config.batch_size * m * n;

    if a.len() < a_required {
        return Err(BatchMatmulError::BufferSizeMismatch {
            name: "A",
            expected: a_required,
            actual: a.len(),
        });
    }
    if b_mat.len() < b_required {
        return Err(BatchMatmulError::BufferSizeMismatch {
            name: "B",
            expected: b_required,
            actual: b_mat.len(),
        });
    }
    if out.len() < out_required {
        return Err(BatchMatmulError::BufferSizeMismatch {
            name: "output",
            expected: out_required,
            actual: out.len(),
        });
    }

    let combined_scale = scale_a * scale_b;
    let a_stride = m * k;
    let b_stride = k * n;
    let out_stride = m * n;

    for batch in 0..config.batch_size {
        let a_off = batch * a_stride;
        let b_off = batch * b_stride;
        let o_off = batch * out_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for l in 0..k {
                    acc += a[a_off + i * k + l] as i32 * b_mat[b_off + l * n + j] as i32;
                }
                out[o_off + i * n + j] = combined_scale * acc as f32;
            }
        }
    }
    Ok(())
}

// ── CUDA kernel source strings ────────────────────────────────────────

/// Tiled shared-memory batched GEMM CUDA kernel source.
///
/// Each thread-block computes a `TILE_SIZE × TILE_SIZE` output tile for
/// one batch element. The batch index is `blockIdx.z`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const BATCH_MATMUL_TILED_KERNEL_SRC: &str = r#"
extern "C" __global__ void batch_matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    int batch_stride_a,
    int batch_stride_b,
    int batch_stride_c)
{
    const int TILE_SIZE = 32;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int batch = blockIdx.z;
    int row   = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col   = blockIdx.x * TILE_SIZE + threadIdx.x;

    const float* a_batch = A + batch * batch_stride_a;
    const float* b_batch = B + batch * batch_stride_b;
    float*       c_batch = C + batch * batch_stride_c;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? a_batch[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? b_batch[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = row * N + col;
        c_batch[idx] = alpha * acc + beta * c_batch[idx];
    }
}
"#;

/// Tensor Core WMMA batched GEMM CUDA kernel source (SM 7.0+).
///
/// Each warp computes a 16×16×16 output fragment using WMMA intrinsics.
/// The batch index is `blockIdx.z`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const BATCH_MATMUL_WMMA_KERNEL_SRC: &str = r#"
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void batch_matmul_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    int batch_stride_a,
    int batch_stride_b,
    int batch_stride_c)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int batch  = blockIdx.z;
    int warp_m = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warp_n = blockIdx.x * WMMA_N;

    const half*  a_batch = A + batch * batch_stride_a;
    const half*  b_batch = B + batch * batch_stride_b;
    float*       c_batch = C + batch * batch_stride_c;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int tile_k = 0; tile_k < K; tile_k += WMMA_K) {
        if (warp_m < M && tile_k < K) {
            wmma::load_matrix_sync(a_frag, a_batch + warp_m * K + tile_k, K);
        }
        if (tile_k < K && warp_n < N) {
            wmma::load_matrix_sync(b_frag, b_batch + tile_k * N + warp_n, N);
        }
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (warp_m < M && warp_n < N) {
        // Scale accumulator by alpha.
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            acc_frag.x[i] *= alpha;
        }

        // Load existing C, scale by beta, and add.
        if (beta != 0.0f) {
            wmma::load_matrix_sync(c_frag, c_batch + warp_m * N + warp_n, N, wmma::mem_row_major);
            for (int i = 0; i < c_frag.num_elements; ++i) {
                acc_frag.x[i] += beta * c_frag.x[i];
            }
        }

        wmma::store_matrix_sync(c_batch + warp_m * N + warp_n, acc_frag, N, wmma::mem_row_major);
    }
}
"#;

// ── GPU launch stubs ──────────────────────────────────────────────────

/// Launch the tiled shared-memory batched GEMM CUDA kernel.
///
/// # Errors
///
/// Returns [`BatchMatmulError::GpuLaunchFailed`] until PTX compilation
/// is wired up.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_batch_matmul(
    _a: &[f32],
    _b: &[f32],
    _output: &mut [f32],
    config: &BatchMatmulConfig,
) -> Result<(), BatchMatmulError> {
    let (gx, gy, gz) = config.grid_dim(32);
    log::debug!(
        "batch_matmul tiled CUDA stub: batch={}, m={}, n={}, k={}, grid=({gx},{gy},{gz})",
        config.batch_size,
        config.m,
        config.n,
        config.k,
    );
    Err(BatchMatmulError::GpuLaunchFailed {
        reason: "tiled batch matmul CUDA kernel not yet compiled — scaffold only".into(),
    })
}

/// Launch the Tensor Core WMMA batched GEMM CUDA kernel.
///
/// # Errors
///
/// Returns [`BatchMatmulError::GpuLaunchFailed`] until PTX compilation
/// is wired up.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_batch_matmul_wmma(
    _a: &[u16],
    _b: &[u16],
    _output: &mut [f32],
    config: &BatchMatmulConfig,
) -> Result<(), BatchMatmulError> {
    let (gx, gy, gz) = config.grid_dim(16);
    log::debug!(
        "batch_matmul WMMA CUDA stub: batch={}, m={}, n={}, k={}, grid=({gx},{gy},{gz})",
        config.batch_size,
        config.m,
        config.n,
        config.k,
    );
    Err(BatchMatmulError::GpuLaunchFailed {
        reason: "WMMA batch matmul CUDA kernel not yet compiled — scaffold only".into(),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    /// Naive single matmul reference: C = A · B.
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    // ── BatchMatmulConfig tests ───────────────────────────────────

    #[test]
    fn config_defaults() {
        let cfg = BatchMatmulConfig::default();
        assert_eq!(cfg.batch_size, 1);
        assert_eq!(cfg.m, 1);
        assert_eq!(cfg.n, 1);
        assert_eq!(cfg.k, 1);
        assert_eq!(cfg.alpha, 1.0);
        assert_eq!(cfg.beta, 0.0);
        assert!(!cfg.transpose_a);
        assert!(!cfg.transpose_b);
    }

    #[test]
    fn config_new_valid() {
        let cfg = BatchMatmulConfig::new(4, 8, 16, 32).unwrap();
        assert_eq!(cfg.batch_size, 4);
        assert_eq!(cfg.m, 8);
        assert_eq!(cfg.n, 16);
        assert_eq!(cfg.k, 32);
    }

    #[test]
    fn config_rejects_zero_batch() {
        let err = BatchMatmulConfig::new(0, 4, 4, 4).unwrap_err();
        assert_eq!(err, BatchMatmulError::ZeroDimension { field: "batch_size" });
    }

    #[test]
    fn config_rejects_zero_m() {
        assert!(BatchMatmulConfig::new(1, 0, 4, 4).is_err());
    }

    #[test]
    fn config_rejects_zero_n() {
        assert!(BatchMatmulConfig::new(1, 4, 0, 4).is_err());
    }

    #[test]
    fn config_rejects_zero_k() {
        assert!(BatchMatmulConfig::new(1, 4, 4, 0).is_err());
    }

    #[test]
    fn config_with_alpha_beta() {
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap().with_alpha_beta(2.0, 0.5);
        assert_eq!(cfg.alpha, 2.0);
        assert_eq!(cfg.beta, 0.5);
    }

    #[test]
    fn config_with_transpose() {
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap().with_transpose(true, false);
        assert!(cfg.transpose_a);
        assert!(!cfg.transpose_b);
    }

    #[test]
    fn config_grid_dim() {
        let cfg = BatchMatmulConfig::new(8, 64, 128, 32).unwrap();
        let (gx, gy, gz) = cfg.grid_dim(32);
        assert_eq!(gx, 4); // ceil(128/32)
        assert_eq!(gy, 2); // ceil(64/32)
        assert_eq!(gz, 8);
    }

    #[test]
    fn config_grid_dim_non_aligned() {
        let cfg = BatchMatmulConfig::new(3, 33, 65, 10).unwrap();
        let (gx, gy, gz) = cfg.grid_dim(32);
        assert_eq!(gx, 3); // ceil(65/32)
        assert_eq!(gy, 2); // ceil(33/32)
        assert_eq!(gz, 3);
    }

    // ── error Display tests ───────────────────────────────────────

    #[test]
    fn error_display_zero_dimension() {
        let e = BatchMatmulError::ZeroDimension { field: "m" };
        assert!(e.to_string().contains("non-zero"));
        assert!(e.to_string().contains("m"));
    }

    #[test]
    fn error_display_buffer_mismatch() {
        let e = BatchMatmulError::BufferSizeMismatch { name: "A", expected: 100, actual: 50 };
        let s = e.to_string();
        assert!(s.contains("100"));
        assert!(s.contains("50"));
    }

    #[test]
    fn error_display_invalid_stride() {
        let e = BatchMatmulError::InvalidStride { reason: "too small".into() };
        assert!(e.to_string().contains("too small"));
    }

    #[test]
    fn error_display_invalid_scale() {
        let e = BatchMatmulError::InvalidScale { reason: "zero".into() };
        assert!(e.to_string().contains("zero"));
    }

    #[test]
    fn error_display_gpu_launch() {
        let e = BatchMatmulError::GpuLaunchFailed { reason: "not compiled".into() };
        assert!(e.to_string().contains("not compiled"));
    }

    #[test]
    fn error_implements_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(BatchMatmulError::ZeroDimension { field: "k" });
        assert!(e.to_string().contains("k"));
    }

    // ── batch_matmul tests ────────────────────────────────────────

    #[test]
    fn batch_matmul_identity_single() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        batch_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &a, 1e-6);
    }

    #[test]
    fn batch_matmul_known_product() {
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            7.0,  8.0,
            9.0,  10.0,
            11.0, 12.0,
        ];
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 3).unwrap();
        batch_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn batch_matmul_two_batches() {
        let a = vec![
            // batch 0: 2×2
            1.0, 0.0, 0.0, 1.0, // batch 1: 2×2
            2.0, 3.0, 4.0, 5.0,
        ];
        let b = vec![
            // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
            1.0, 0.0, 0.0, 1.0,
        ];
        let mut out = vec![0.0f32; 8];
        let cfg = BatchMatmulConfig::new(2, 2, 2, 2).unwrap();
        batch_matmul(&a, &b, &mut out, &cfg).unwrap();
        // batch 0: I * B = B
        assert_close(&out[0..4], &[5.0, 6.0, 7.0, 8.0], 1e-6);
        // batch 1: A * I = A
        assert_close(&out[4..8], &[2.0, 3.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn batch_matmul_alpha_scaling() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap().with_alpha_beta(3.0, 0.0);
        batch_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[3.0, 6.0, 9.0, 12.0], 1e-6);
    }

    #[test]
    fn batch_matmul_beta_accumulate() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![10.0, 20.0, 30.0, 40.0];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap().with_alpha_beta(1.0, 1.0);
        batch_matmul(&a, &b, &mut out, &cfg).unwrap();
        // C = 1.0 * I + 1.0 * old_C
        assert_close(&out, &[11.0, 20.0, 30.0, 41.0], 1e-6);
    }

    #[test]
    fn batch_matmul_1x1() {
        let a = vec![3.0];
        let b = vec![5.0];
        let mut out = vec![0.0f32; 1];
        let cfg = BatchMatmulConfig::new(1, 1, 1, 1).unwrap();
        batch_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[15.0], 1e-6);
    }

    #[test]
    fn batch_matmul_zero_a() {
        let a = vec![0.0f32; 6];
        let b: Vec<f32> = (1..7).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 3).unwrap();
        batch_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[0.0; 4], 1e-6);
    }

    #[test]
    fn batch_matmul_buffer_too_small_a() {
        let a = vec![0.0f32; 3]; // need 4
        let b = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(batch_matmul(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn batch_matmul_buffer_too_small_b() {
        let a = vec![0.0f32; 4];
        let b = vec![0.0f32; 3]; // need 4
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(batch_matmul(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn batch_matmul_buffer_too_small_out() {
        let a = vec![0.0f32; 4];
        let b = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 3]; // need 4
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(batch_matmul(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn batch_matmul_matches_naive() {
        let (m, n, k) = (5, 7, 11);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let mut out = vec![0.0f32; m * n];
        let cfg = BatchMatmulConfig::new(1, m, n, k).unwrap();
        batch_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    // ── batch_matmul_transposed tests ─────────────────────────────

    #[test]
    fn transposed_no_flags_matches_basic() {
        let (m, n, k) = (4, 3, 5);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
        let cfg = BatchMatmulConfig::new(1, m, n, k).unwrap();
        let mut out1 = vec![0.0f32; m * n];
        let mut out2 = vec![0.0f32; m * n];
        batch_matmul(&a, &b, &mut out1, &cfg).unwrap();
        batch_matmul_transposed(&a, &b, &mut out2, &cfg).unwrap();
        assert_close(&out1, &out2, 1e-5);
    }

    #[test]
    fn transposed_a_identity() {
        // A^T * B where A is stored transposed as [K, M]
        let m = 2;
        let k = 3;
        let n = 2;
        // A stored as [K=3, M=2] (physical layout for transpose_a=true)
        #[rustfmt::skip]
        let a = vec![
            1.0, 4.0,  // col 0 of logical A
            2.0, 5.0,  // col 1
            3.0, 6.0,  // col 2
        ];
        #[rustfmt::skip]
        let b = vec![
            7.0,  8.0,
            9.0,  10.0,
            11.0, 12.0,
        ];
        // Logical A = [[1,2,3],[4,5,6]], so A*B = [[58,64],[139,154]]
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, m, n, k).unwrap().with_transpose(true, false);
        batch_matmul_transposed(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[58.0, 64.0, 139.0, 154.0], 1e-5);
    }

    #[test]
    fn transposed_b_identity() {
        let m = 2;
        let k = 3;
        let n = 2;
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        // B stored transposed as [N=2, K=3]
        #[rustfmt::skip]
        let b = vec![
            7.0,  9.0, 11.0,  // row 0 of B^T = col 0 of logical B
            8.0, 10.0, 12.0,  // row 1 of B^T = col 1 of logical B
        ];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, m, n, k).unwrap().with_transpose(false, true);
        batch_matmul_transposed(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[58.0, 64.0, 139.0, 154.0], 1e-5);
    }

    #[test]
    fn transposed_both() {
        // A stored as [K=2, M=2], B stored as [N=2, K=2]
        let a = vec![1.0, 3.0, 2.0, 4.0]; // A^T: logical A = [[1,2],[3,4]]
        let b = vec![5.0, 7.0, 6.0, 8.0]; // B^T: logical B = [[5,6],[7,8]]
        // A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap().with_transpose(true, true);
        batch_matmul_transposed(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[19.0, 22.0, 43.0, 50.0], 1e-5);
    }

    #[test]
    fn transposed_multi_batch() {
        // 2 batches, 2×2, K=2, transpose_a only
        let a = vec![
            // batch 0: stored [K=2, M=2]
            1.0, 0.0, 0.0, 1.0, // = logical identity
            // batch 1
            0.0, 2.0, 3.0, 0.0, // logical A = [[0,3],[2,0]]
        ];
        let b = vec![
            // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
            1.0, 2.0, 3.0, 4.0,
        ];
        let mut out = vec![0.0f32; 8];
        let cfg = BatchMatmulConfig::new(2, 2, 2, 2).unwrap().with_transpose(true, false);
        batch_matmul_transposed(&a, &b, &mut out, &cfg).unwrap();
        // batch 0: I * B = B
        assert_close(&out[0..4], &[5.0, 6.0, 7.0, 8.0], 1e-6);
        // batch 1: [[0,3],[2,0]] * [[1,2],[3,4]] = [[9,12],[2,4]]
        assert_close(&out[4..8], &[9.0, 12.0, 2.0, 4.0], 1e-5);
    }

    // ── strided_batch_matmul tests ────────────────────────────────

    #[test]
    fn strided_matches_contiguous() {
        let (m, n, k) = (2, 2, 2);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let strides = BatchStrides { stride_a: m * k, stride_b: k * n, stride_out: m * n };
        let cfg = BatchMatmulConfig::new(2, m, n, k).unwrap();
        let mut out_strided = vec![0.0f32; 8];
        let mut out_basic = vec![0.0f32; 8];
        strided_batch_matmul(&a, &b, &mut out_strided, &cfg, &strides).unwrap();
        batch_matmul(&a, &b, &mut out_basic, &cfg).unwrap();
        assert_close(&out_strided, &out_basic, 1e-6);
    }

    #[test]
    fn strided_with_padding() {
        let m = 2;
        let n = 2;
        let k = 2;
        // stride_a = 6 (4 used + 2 padding)
        let a = vec![
            1.0, 0.0, 0.0, 1.0, /*pad*/ 99.0, 99.0, //
            2.0, 3.0, 4.0, 5.0, /*pad*/ 99.0, 99.0,
        ];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, /*pad*/ 99.0, 99.0, //
            1.0, 0.0, 0.0, 1.0, /*pad*/ 99.0, 99.0,
        ];
        let strides = BatchStrides { stride_a: 6, stride_b: 6, stride_out: m * n };
        let cfg = BatchMatmulConfig::new(2, m, n, k).unwrap();
        let mut out = vec![0.0f32; 8];
        strided_batch_matmul(&a, &b, &mut out, &cfg, &strides).unwrap();
        // batch 0: I * [[1,2],[3,4]] = [[1,2],[3,4]]
        assert_close(&out[0..4], &[1.0, 2.0, 3.0, 4.0], 1e-6);
        // batch 1: [[2,3],[4,5]] * I = [[2,3],[4,5]]
        assert_close(&out[4..8], &[2.0, 3.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn strided_rejects_small_stride_a() {
        let cfg = BatchMatmulConfig::new(2, 2, 2, 2).unwrap();
        let strides = BatchStrides { stride_a: 3, stride_b: 4, stride_out: 4 };
        let a = vec![0.0f32; 16];
        let b = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 8];
        assert!(strided_batch_matmul(&a, &b, &mut out, &cfg, &strides).is_err());
    }

    #[test]
    fn strided_rejects_small_stride_b() {
        let cfg = BatchMatmulConfig::new(2, 2, 2, 2).unwrap();
        let strides = BatchStrides { stride_a: 4, stride_b: 3, stride_out: 4 };
        let a = vec![0.0f32; 16];
        let b = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 8];
        assert!(strided_batch_matmul(&a, &b, &mut out, &cfg, &strides).is_err());
    }

    #[test]
    fn strided_rejects_small_stride_out() {
        let cfg = BatchMatmulConfig::new(2, 2, 2, 2).unwrap();
        let strides = BatchStrides { stride_a: 4, stride_b: 4, stride_out: 3 };
        let a = vec![0.0f32; 16];
        let b = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 8];
        assert!(strided_batch_matmul(&a, &b, &mut out, &cfg, &strides).is_err());
    }

    // ── fused_batch_matmul_bias tests ─────────────────────────────

    #[test]
    fn bias_adds_to_each_row() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let b = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let bias = vec![10.0, 20.0];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        fused_batch_matmul_bias(&a, &b, &bias, &mut out, &cfg).unwrap();
        // I*I + bias = [[11,20],[10,21]]
        assert_close(&out, &[11.0, 20.0, 10.0, 21.0], 1e-6);
    }

    #[test]
    fn bias_multi_batch() {
        let a = vec![2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0];
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let bias = vec![1.0, 2.0];
        let mut out = vec![0.0f32; 8];
        let cfg = BatchMatmulConfig::new(2, 2, 2, 2).unwrap();
        fused_batch_matmul_bias(&a, &b, &bias, &mut out, &cfg).unwrap();
        assert_close(&out[0..4], &[3.0, 2.0, 1.0, 4.0], 1e-6);
        assert_close(&out[4..8], &[4.0, 2.0, 1.0, 5.0], 1e-6);
    }

    #[test]
    fn bias_too_small_rejected() {
        let a = vec![0.0f32; 4];
        let b = vec![0.0f32; 4];
        let bias = vec![1.0]; // need 2
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(fused_batch_matmul_bias(&a, &b, &bias, &mut out, &cfg).is_err());
    }

    #[test]
    fn bias_with_alpha() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![5.0, 10.0];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap().with_alpha_beta(2.0, 0.0);
        fused_batch_matmul_bias(&a, &b, &bias, &mut out, &cfg).unwrap();
        // 2*I + bias = [[7,10],[5,12]]
        assert_close(&out, &[7.0, 10.0, 5.0, 12.0], 1e-6);
    }

    // ── fused_batch_matmul_relu tests ─────────────────────────────

    #[test]
    fn relu_clamps_negatives() {
        let a = vec![1.0, -1.0, -1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        // C = [[1-3, 2-4],[-1+3, -2+4]] = [[-2,-2],[2,2]]
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        fused_batch_matmul_relu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[0.0, 0.0, 2.0, 2.0], 1e-6);
    }

    #[test]
    fn relu_passes_positives() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        fused_batch_matmul_relu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[5.0, 6.0, 7.0, 8.0], 1e-6);
    }

    #[test]
    fn relu_multi_batch() {
        let a = vec![
            1.0, -1.0, -1.0, 1.0, // batch 0
            1.0, 0.0, 0.0, 1.0, // batch 1
        ];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            10.0, 20.0, 30.0, 40.0, // batch 1
        ];
        let mut out = vec![0.0f32; 8];
        let cfg = BatchMatmulConfig::new(2, 2, 2, 2).unwrap();
        fused_batch_matmul_relu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out[0..4], &[0.0, 0.0, 2.0, 2.0], 1e-6);
        assert_close(&out[4..8], &[10.0, 20.0, 30.0, 40.0], 1e-6);
    }

    // ── quantized_batch_matmul tests ──────────────────────────────

    #[test]
    fn quantized_identity() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        quantized_batch_matmul(&a, &b, &mut out, &cfg, 1.0, 1.0).unwrap();
        assert_close(&out, &[5.0, 6.0, 7.0, 8.0], 1e-6);
    }

    #[test]
    fn quantized_with_scales() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![10, 20, 30, 40];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        quantized_batch_matmul(&a, &b, &mut out, &cfg, 0.5, 0.25).unwrap();
        // scale = 0.5 * 0.25 = 0.125
        assert_close(&out, &[1.25, 2.5, 3.75, 5.0], 1e-6);
    }

    #[test]
    fn quantized_negative_values() {
        let a: Vec<i8> = vec![1, -1, -1, 1];
        let b: Vec<i8> = vec![10, 20, 30, 40];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        quantized_batch_matmul(&a, &b, &mut out, &cfg, 1.0, 1.0).unwrap();
        assert_close(&out, &[-20.0, -20.0, 20.0, 20.0], 1e-6);
    }

    #[test]
    fn quantized_multi_batch() {
        let a: Vec<i8> = vec![
            1, 0, 0, 1, // batch 0: identity
            2, 1, 1, 2, // batch 1
        ];
        let b: Vec<i8> = vec![
            3, 4, 5, 6, // batch 0
            1, 0, 0, 1, // batch 1: identity
        ];
        let mut out = vec![0.0f32; 8];
        let cfg = BatchMatmulConfig::new(2, 2, 2, 2).unwrap();
        quantized_batch_matmul(&a, &b, &mut out, &cfg, 1.0, 1.0).unwrap();
        assert_close(&out[0..4], &[3.0, 4.0, 5.0, 6.0], 1e-6);
        assert_close(&out[4..8], &[2.0, 1.0, 1.0, 2.0], 1e-6);
    }

    #[test]
    fn quantized_rejects_zero_scale_a() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(quantized_batch_matmul(&a, &b, &mut out, &cfg, 0.0, 1.0).is_err());
    }

    #[test]
    fn quantized_rejects_zero_scale_b() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(quantized_batch_matmul(&a, &b, &mut out, &cfg, 1.0, 0.0).is_err());
    }

    #[test]
    fn quantized_rejects_nan_scale() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(quantized_batch_matmul(&a, &b, &mut out, &cfg, f32::NAN, 1.0).is_err());
    }

    #[test]
    fn quantized_rejects_inf_scale() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(quantized_batch_matmul(&a, &b, &mut out, &cfg, 1.0, f32::INFINITY).is_err());
    }

    #[test]
    fn quantized_buffer_too_small() {
        let a: Vec<i8> = vec![1, 0]; // need 4
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let mut out = vec![0.0f32; 4];
        let cfg = BatchMatmulConfig::new(1, 2, 2, 2).unwrap();
        assert!(quantized_batch_matmul(&a, &b, &mut out, &cfg, 1.0, 1.0).is_err());
    }

    // ── CUDA kernel source availability ───────────────────────────

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn tiled_kernel_src_non_empty() {
        assert!(!BATCH_MATMUL_TILED_KERNEL_SRC.is_empty());
        assert!(BATCH_MATMUL_TILED_KERNEL_SRC.contains("batch_matmul_tiled"));
        assert!(BATCH_MATMUL_TILED_KERNEL_SRC.contains("__shared__"));
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn wmma_kernel_src_non_empty() {
        assert!(!BATCH_MATMUL_WMMA_KERNEL_SRC.is_empty());
        assert!(BATCH_MATMUL_WMMA_KERNEL_SRC.contains("batch_matmul_wmma"));
        assert!(BATCH_MATMUL_WMMA_KERNEL_SRC.contains("wmma"));
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn launch_tiled_returns_scaffold_error() {
        let cfg = BatchMatmulConfig::new(1, 4, 4, 4).unwrap();
        let a = vec![0.0f32; 16];
        let b = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 16];
        let err = launch_batch_matmul(&a, &b, &mut out, &cfg).unwrap_err();
        assert!(matches!(err, BatchMatmulError::GpuLaunchFailed { .. }));
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn launch_wmma_returns_scaffold_error() {
        let cfg = BatchMatmulConfig::new(1, 16, 16, 16).unwrap();
        let a = vec![0u16; 256];
        let b = vec![0u16; 256];
        let mut out = vec![0.0f32; 256];
        let err = launch_batch_matmul_wmma(&a, &b, &mut out, &cfg).unwrap_err();
        assert!(matches!(err, BatchMatmulError::GpuLaunchFailed { .. }));
    }
}
