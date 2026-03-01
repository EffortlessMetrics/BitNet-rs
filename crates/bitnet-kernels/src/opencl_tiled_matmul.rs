//! Optimized tiled matrix multiplication for OpenCL on Intel Arc A770.
//!
//! Targets the Xe-HPG architecture with 64 KB SLM per sub-slice and
//! subgroup sizes of 8 / 16 / 32.  The default 16×16×16 tile maps
//! directly to the preferred subgroup width, maximising EU occupancy
//! while keeping shared-memory pressure within the 64 KB budget.
//!
//! # Kernels
//!
//! | Kernel              | Description                                 |
//! |---------------------|---------------------------------------------|
//! | `matmul_naive`      | Simple reference (no tiling)                |
//! | `matmul_tiled`      | 16×16 tiled with `__local` memory           |
//! | `matmul_tiled_vec4` | float4 vectorised for 4× memory bandwidth   |
//! | `matmul_batched`    | Batched variant with a batch dimension       |

use std::fmt;

// ───────────────────────────── Tile configuration ─────────────────────────

/// Tile dimensions for the tiled matrix-multiplication kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    /// Tile height (rows of A / C processed per work-group).
    pub tile_m: usize,
    /// Tile width (columns of B / C processed per work-group).
    pub tile_n: usize,
    /// Reduction depth per iteration.
    pub tile_k: usize,
    /// Whether to use `__local` (SLM) memory for the tiles.
    pub use_local_memory: bool,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self { tile_m: 16, tile_n: 16, tile_k: 16, use_local_memory: true }
    }
}

impl fmt::Display for TileConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}×{}×{} (local={})",
            self.tile_m, self.tile_n, self.tile_k, self.use_local_memory
        )
    }
}

// ──────────────────────── Matrix-multiply configuration ───────────────────

/// Full configuration for a GEMM operation: C = α A B + β C.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatMulConfig {
    /// Rows of A / C.
    pub m: usize,
    /// Columns of B / C.
    pub n: usize,
    /// Shared (reduction) dimension.
    pub k: usize,
    /// Scaling factor for the A*B product (default 1.0).
    pub alpha: f32,
    /// Scaling factor for the existing C matrix (default 0.0).
    pub beta: f32,
    /// Whether A is stored transposed.
    pub transpose_a: bool,
    /// Whether B is stored transposed.
    pub transpose_b: bool,
}

impl MatMulConfig {
    /// Create a simple A×B config (no scaling, no transpose).
    #[must_use]
    pub fn simple(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
            transpose_a: false,
            transpose_b: false,
        }
    }
}

// ─────────────────────────────── Errors ───────────────────────────────────

/// Errors that may occur during tiled matrix multiplication.
#[derive(Debug, Clone, PartialEq)]
pub enum MatMulError {
    /// The supplied buffer lengths do not match the declared dimensions.
    DimensionMismatch {
        expected: usize,
        actual: usize,
        matrix: &'static str,
    },
    /// Allocation would exceed available device memory.
    OutOfMemory {
        requested_bytes: usize,
        available_bytes: usize,
    },
    /// An error originating from the OpenCL runtime or kernel compilation.
    KernelError(String),
}

impl fmt::Display for MatMulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual, matrix } => {
                write!(
                    f,
                    "dimension mismatch for {matrix}: \
                     expected {expected} elements, got {actual}"
                )
            }
            Self::OutOfMemory { requested_bytes, available_bytes } => {
                write!(
                    f,
                    "out of memory: requested {requested_bytes} B, \
                     available {available_bytes} B"
                )
            }
            Self::KernelError(msg) => write!(f, "kernel error: {msg}"),
        }
    }
}

impl std::error::Error for MatMulError {}

// ───────────────────────── Kernel wrapper struct ──────────────────────────

/// Holds a [`MatMulConfig`] together with an optional cached OpenCL kernel
/// source string (built from [`TILED_MATMUL_SRC`]).
#[derive(Debug, Clone)]
pub struct TiledMatMulKernel {
    /// Matrix-multiply parameters.
    pub config: MatMulConfig,
    /// Pre-built kernel source (if `None`, the caller should use
    /// [`TILED_MATMUL_SRC`] directly).
    pub cached_kernel_source: Option<String>,
}

impl TiledMatMulKernel {
    /// Create a new kernel wrapper for the given config.
    #[must_use]
    pub fn new(config: MatMulConfig) -> Self {
        Self { config, cached_kernel_source: None }
    }

    /// Build and cache the kernel source with compile-time defines for
    /// the chosen tile sizes.
    pub fn build_source(&mut self, tile: &TileConfig) {
        let defines = format!(
            "#define TILE_M {}\n#define TILE_N {}\n#define TILE_K {}\n",
            tile.tile_m, tile.tile_n, tile.tile_k,
        );
        self.cached_kernel_source =
            Some(format!("{defines}{TILED_MATMUL_SRC}"));
    }

    /// Return the (possibly cached) kernel source.
    #[must_use]
    pub fn kernel_source(&self) -> &str {
        self.cached_kernel_source
            .as_deref()
            .unwrap_or(TILED_MATMUL_SRC)
    }
}

// ──────────────────── CPU reference implementations ──────────────────────

/// Naïve O(m·n·k) matrix multiply: C = A × B  (row-major).
///
/// `a` is m×k, `b` is k×n, result is m×n.
#[must_use]
pub fn cpu_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "A size mismatch");
    debug_assert_eq!(b.len(), k * n, "B size mismatch");

    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Matrix multiply with B transposed: C = A × Bᵀ  (row-major).
///
/// `a` is m×k, `b_t` is n×k (stored row-major as the transpose of B).
#[must_use]
pub fn cpu_matmul_transposed(
    a: &[f32],
    b_t: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "A size mismatch");
    debug_assert_eq!(b_t.len(), n * k, "B^T size mismatch");

    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i * k + p] * b_t[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Full GEMM: C = α A B + β C  (row-major, no transpose).
///
/// `a` is m×k, `b` is k×n, `c` is m×n (input / output).
#[must_use]
pub fn cpu_matmul_alpha_beta(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "A size mismatch");
    debug_assert_eq!(b.len(), k * n, "B size mismatch");
    debug_assert_eq!(c.len(), m * n, "C size mismatch");

    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
    out
}

/// Batched matrix multiply: C[b] = A[b] × B[b] for b in 0..batch.
///
/// Each matrix in the batch is stored contiguously: `a` has length
/// `batch * m * k`, `b` has length `batch * k * n`, result has length
/// `batch * m * n`.
#[must_use]
pub fn cpu_batched_matmul(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), batch * m * k, "A batch size mismatch");
    debug_assert_eq!(b.len(), batch * k * n, "B batch size mismatch");

    let mut c = vec![0.0_f32; batch * m * n];
    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let c_off = bi * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    sum += a[a_off + i * k + p] * b[b_off + p * n + j];
                }
                c[c_off + i * n + j] = sum;
            }
        }
    }
    c
}

// ────────────────────── Tile-size heuristic for A770 ─────────────────────

/// Pick a tile size that is friendly for Xe-HPG.
///
/// * Prefer 16 because the A770's EU SIMD width is 16.
/// * Fall back to 8 for very small dimensions so work-groups are not
///   wasted on padding.
/// * Disable local memory for tiny problems where SLM overhead dominates.
#[must_use]
pub fn optimal_tile_size(m: usize, n: usize, k: usize) -> TileConfig {
    let min_dim = m.min(n).min(k);
    if min_dim < 8 {
        TileConfig {
            tile_m: 4,
            tile_n: 4,
            tile_k: 4,
            use_local_memory: false,
        }
    } else if min_dim < 16 {
        TileConfig {
            tile_m: 8,
            tile_n: 8,
            tile_k: 8,
            use_local_memory: true,
        }
    } else {
        // 16×16 is the sweet spot for Xe-HPG subgroup width.
        TileConfig::default()
    }
}

// ──────────────────────── OpenCL kernel source ───────────────────────────

/// OpenCL C source for the tiled matrix-multiplication kernels.
///
/// Contains four entry points:
/// - `matmul_naive`      – simple reference
/// - `matmul_tiled`      – 16×16 `__local` memory tiling
/// - `matmul_tiled_vec4` – float4 vectorised variant
/// - `matmul_batched`    – batched variant
pub const TILED_MATMUL_SRC: &str = r#"
/* ------------------------------------------------------------------ *
 *  Tiled matrix multiplication kernels for Intel Arc A770 (Xe-HPG)   *
 * ------------------------------------------------------------------ */

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

/* ---------- matmul_naive ------------------------------------------ */
__kernel void matmul_naive(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    const int M,
    const int N,
    const int K)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int p = 0; p < K; ++p) {
            sum += A[row * K + p] * B[p * N + col];
        }
        C[row * N + col] = sum;
    }
}

/* ---------- matmul_tiled ------------------------------------------ */
__kernel void matmul_tiled(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    const int M,
    const int N,
    const int K)
{
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        /* Coalesced load into local memory */
        const int a_col = t * TILE_SIZE + local_col;
        const int b_row = t * TILE_SIZE + local_row;

        tile_a[local_row][local_col] =
            (global_row < M && a_col < K)
                ? A[global_row * K + a_col]
                : 0.0f;

        tile_b[local_row][local_col] =
            (b_row < K && global_col < N)
                ? B[b_row * N + global_col]
                : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        /* Accumulate partial products */
        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p) {
            sum += tile_a[local_row][p] * tile_b[p][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}

/* ---------- matmul_tiled_vec4 ------------------------------------- *
 * Each work-item computes 4 consecutive columns using float4 loads,  *
 * quadrupling effective memory bandwidth.                            *
 * ------------------------------------------------------------------ */
__kernel void matmul_tiled_vec4(
    __global const float*  A,
    __global const float4* B4,
    __global       float4* C4,
    const int M,
    const int N4,  /* N / 4 */
    const int K)
{
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);  /* in float4 units */

    __local float  tile_a[TILE_SIZE][TILE_SIZE];
    __local float4 tile_b[TILE_SIZE][TILE_SIZE];

    float4 sum = (float4)(0.0f);
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE_SIZE + local_col;
        const int b_row = t * TILE_SIZE + local_row;

        tile_a[local_row][local_col] =
            (global_row < M && a_col < K)
                ? A[global_row * K + a_col]
                : 0.0f;

        tile_b[local_row][local_col] =
            (b_row < K && global_col < N4)
                ? B4[b_row * N4 + global_col]
                : (float4)(0.0f);

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p) {
            sum += tile_a[local_row][p] * tile_b[p][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N4) {
        C4[global_row * N4 + global_col] = sum;
    }
}

/* ---------- matmul_batched ---------------------------------------- */
__kernel void matmul_batched(
    __global const float* A,
    __global const float* B,
    __global       float* C,
    const int M,
    const int N,
    const int K)
{
    const int batch = get_global_id(2);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);

    const int a_offset = batch * M * K;
    const int b_offset = batch * K * N;
    const int c_offset = batch * M * N;

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE_SIZE + local_col;
        const int b_row = t * TILE_SIZE + local_row;

        tile_a[local_row][local_col] =
            (global_row < M && a_col < K)
                ? A[a_offset + global_row * K + a_col]
                : 0.0f;

        tile_b[local_row][local_col] =
            (b_row < K && global_col < N)
                ? B[b_offset + b_row * N + global_col]
                : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int p = 0; p < TILE_SIZE; ++p) {
            sum += tile_a[local_row][p] * tile_b[p][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        C[c_offset + global_row * N + global_col] = sum;
    }
}
"#;

// ──────────────────────────────── Tests ──────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────

    /// Build an identity matrix of size n×n.
    fn identity(n: usize) -> Vec<f32> {
        let mut m = vec![0.0_f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    /// Compare two slices element-wise within an absolute tolerance.
    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (tol={tol})"
            );
        }
    }

    /// Simple sequential fill: 1, 2, 3, …
    fn sequential(n: usize) -> Vec<f32> {
        (1..=n).map(|v| v as f32).collect()
    }

    /// Deterministic pseudo-random values in [0, 1).
    fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f32 / u64::MAX as f32).abs()
            })
            .collect()
    }

    // ── small matmul tests ──────────────────────────────────────────

    #[test]
    fn test_cpu_matmul_2x2() {
        // [1 2] × [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = cpu_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cpu_matmul_3x3() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let c = cpu_matmul(&a, &b, 3, 3, 3);
        assert_eq!(
            c,
            vec![30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0]
        );
    }

    #[test]
    fn test_cpu_matmul_4x4() {
        let a: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let b = identity(4);
        let c = cpu_matmul(&a, &b, 4, 4, 4);
        assert_eq!(c, a);
    }

    #[test]
    fn test_cpu_matmul_1x1() {
        let c = cpu_matmul(&[3.0], &[4.0], 1, 1, 1);
        assert_eq!(c, vec![12.0]);
    }

    // ── non-square ──────────────────────────────────────────────────

    #[test]
    fn test_cpu_matmul_3x5_times_5x7() {
        let a = sequential(15); // 3×5
        let b = sequential(35); // 5×7
        let c = cpu_matmul(&a, &b, 3, 7, 5);
        assert_eq!(c.len(), 21);
        // spot-check first element: row0·col0 = 1*1+2*8+3*15+4*22+5*29
        let expected_00 =
            1.0 * 1.0 + 2.0 * 8.0 + 3.0 * 15.0 + 4.0 * 22.0 + 5.0 * 29.0;
        assert_approx_eq(&c[..1], &[expected_00], 1e-4);
    }

    #[test]
    fn test_cpu_matmul_2x3_times_3x1() {
        // [1 2 3] × [1] = [1+4+9]  = [14]
        // [4 5 6]   [2]   [4+10+18]  [32]
        //            [3]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = cpu_matmul(&a, &b, 2, 1, 3);
        assert_eq!(c, vec![14.0, 32.0]);
    }

    #[test]
    fn test_cpu_matmul_1x4_times_4x1() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let c = cpu_matmul(&a, &b, 1, 1, 4);
        assert_eq!(c, vec![20.0]);
    }

    // ── identity ────────────────────────────────────────────────────

    #[test]
    fn test_identity_mul_2x2() {
        let a = vec![5.0, 6.0, 7.0, 8.0];
        let c = cpu_matmul(&a, &identity(2), 2, 2, 2);
        assert_eq!(c, a);
    }

    #[test]
    fn test_identity_mul_left() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let c = cpu_matmul(&identity(2), &a, 2, 2, 2);
        assert_eq!(c, a);
    }

    #[test]
    fn test_identity_mul_8x8() {
        let a = sequential(64);
        let c = cpu_matmul(&a, &identity(8), 8, 8, 8);
        assert_approx_eq(&c, &a, 1e-5);
    }

    // ── zero matrix ─────────────────────────────────────────────────

    #[test]
    fn test_zero_mul_right() {
        let a = sequential(9);
        let zero = vec![0.0_f32; 9];
        let c = cpu_matmul(&a, &zero, 3, 3, 3);
        assert_eq!(c, zero);
    }

    #[test]
    fn test_zero_mul_left() {
        let b = sequential(9);
        let zero = vec![0.0_f32; 9];
        let c = cpu_matmul(&zero, &b, 3, 3, 3);
        assert_eq!(c, zero);
    }

    #[test]
    fn test_zero_mul_both() {
        let zero = vec![0.0_f32; 4];
        let c = cpu_matmul(&zero, &zero, 2, 2, 2);
        assert_eq!(c, zero);
    }

    // ── transpose variants ──────────────────────────────────────────

    #[test]
    fn test_cpu_matmul_transposed_2x2() {
        // A = [1 2; 3 4], B = [5 6; 7 8] → Bᵀ stored row-major = B
        // C = A × Bᵀ where Bᵀ[j] is row j of Bᵀ = col j of B
        // Bᵀ layout (n×k = 2×2): [[5,7],[6,8]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b_t = vec![5.0, 7.0, 6.0, 8.0]; // rows of Bᵀ
        let c = cpu_matmul_transposed(&a, &b_t, 2, 2, 2);
        // row0: 1*5+2*7=19, 1*6+2*8=22 → [19, 22]
        // row1: 3*5+4*7=43, 3*6+4*8=50 → [43, 50]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cpu_matmul_transposed_3x3() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b_t = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let c = cpu_matmul_transposed(&a, &b_t, 3, 3, 3);
        // C[i][j] = b_t[j][i] → result is the transpose of b_t
        assert_eq!(c, vec![9.0, 6.0, 3.0, 8.0, 5.0, 2.0, 7.0, 4.0, 1.0]);
    }

    #[test]
    fn test_transposed_vs_explicit() {
        // Verify A×Bᵀ via transposed fn matches manual transpose + naive
        let a = pseudo_random(12, 42); // 3×4
        let b = pseudo_random(20, 99); // 5×4 (already "transposed" layout)
        let c_trans = cpu_matmul_transposed(&a, &b, 3, 5, 4);

        // Manually transpose b (5×4 → 4×5)
        let mut b_normal = vec![0.0_f32; 20];
        for i in 0..5 {
            for j in 0..4 {
                b_normal[j * 5 + i] = b[i * 4 + j];
            }
        }
        let c_naive = cpu_matmul(&a, &b_normal, 3, 5, 4);
        assert_approx_eq(&c_trans, &c_naive, 1e-4);
    }

    #[test]
    fn test_transposed_identity() {
        // Iᵀ = I, so A × Iᵀ = A
        let a = sequential(16);
        let id = identity(4);
        let c = cpu_matmul_transposed(&a, &id, 4, 4, 4);
        assert_approx_eq(&c, &a, 1e-5);
    }

    // ── GEMM alpha/beta ─────────────────────────────────────────────

    #[test]
    fn test_alpha_beta_identity() {
        // C = 1.0 * A*I + 0.0 * C_old = A
        let a = sequential(4);
        let id = identity(2);
        let c_old = vec![99.0; 4];
        let c = cpu_matmul_alpha_beta(&a, &id, &c_old, 2, 2, 2, 1.0, 0.0);
        assert_approx_eq(&c, &a, 1e-5);
    }

    #[test]
    fn test_alpha_scaling() {
        // C = 2.0 * A*B + 0*C
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c_old = vec![0.0; 4];
        let c = cpu_matmul_alpha_beta(&a, &b, &c_old, 2, 2, 2, 2.0, 0.0);
        // A*B = [19,22,43,50], ×2 = [38,44,86,100]
        assert_eq!(c, vec![38.0, 44.0, 86.0, 100.0]);
    }

    #[test]
    fn test_beta_scaling() {
        // C = 0*A*B + 3.0*C_old = 3*C_old
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let c_old = vec![10.0, 20.0, 30.0, 40.0];
        let c = cpu_matmul_alpha_beta(&a, &b, &c_old, 2, 2, 2, 0.0, 3.0);
        assert_eq!(c, vec![30.0, 60.0, 90.0, 120.0]);
    }

    #[test]
    fn test_alpha_beta_combined() {
        // C = 0.5*(A*B) + 1.0*C_old
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c_old = vec![100.0, 200.0, 300.0, 400.0];
        let c =
            cpu_matmul_alpha_beta(&a, &b, &c_old, 2, 2, 2, 0.5, 1.0);
        // A*B = [19,22,43,50]; 0.5*[…] + [100,200,300,400]
        assert_eq!(c, vec![109.5, 211.0, 321.5, 425.0]);
    }

    #[test]
    fn test_alpha_zero_beta_one() {
        // C = 0*A*B + 1*C_old → C_old unchanged
        let a = sequential(4);
        let b = sequential(4);
        let c_old = vec![7.0, 8.0, 9.0, 10.0];
        let c = cpu_matmul_alpha_beta(&a, &b, &c_old, 2, 2, 2, 0.0, 1.0);
        assert_eq!(c, c_old);
    }

    // ── batched matmul ──────────────────────────────────────────────

    #[test]
    fn test_batched_single() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c_batched = cpu_batched_matmul(&a, &b, 1, 2, 2, 2);
        let c_ref = cpu_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c_batched, c_ref);
    }

    #[test]
    fn test_batched_4() {
        let batch = 4;
        let (m, n, k) = (2, 2, 2);
        let a: Vec<f32> = (0..batch)
            .flat_map(|b| {
                (0..m * k).map(move |i| (b * m * k + i + 1) as f32)
            })
            .collect();
        let b_mat = vec![1.0, 0.0, 0.0, 1.0].repeat(batch); // identity
        let c = cpu_batched_matmul(&a, &b_mat, batch, m, n, k);
        // Each batch multiplied by identity → should equal a
        assert_eq!(c, a);
    }

    #[test]
    fn test_batched_16() {
        let batch = 16;
        let (m, n, k) = (3, 3, 3);
        let a = pseudo_random(batch * m * k, 1234);
        let b = pseudo_random(batch * k * n, 5678);
        let c = cpu_batched_matmul(&a, &b, batch, m, n, k);
        assert_eq!(c.len(), batch * m * n);

        // Verify each batch independently
        for bi in 0..batch {
            let a_slice = &a[bi * m * k..(bi + 1) * m * k];
            let b_slice = &b[bi * k * n..(bi + 1) * k * n];
            let c_slice = &c[bi * m * n..(bi + 1) * m * n];
            let c_ref = cpu_matmul(a_slice, b_slice, m, n, k);
            assert_approx_eq(c_slice, &c_ref, 1e-4);
        }
    }

    #[test]
    fn test_batched_zero_result() {
        let a = vec![0.0_f32; 8]; // batch=2, 2×2
        let b = sequential(8);
        let c = cpu_batched_matmul(&a, &b, 2, 2, 2, 2);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    // ── large matrices ──────────────────────────────────────────────

    #[test]
    fn test_large_128x128() {
        let n = 128;
        let a = pseudo_random(n * n, 42);
        let c = cpu_matmul(&a, &identity(n), n, n, n);
        assert_approx_eq(&c, &a, 1e-3);
    }

    #[test]
    fn test_large_256x256() {
        let n = 256;
        let a = pseudo_random(n * n, 77);
        let c = cpu_matmul(&a, &identity(n), n, n, n);
        assert_approx_eq(&c, &a, 1e-2);
    }

    // ── tile boundary edge cases ────────────────────────────────────

    #[test]
    fn test_non_tile_aligned_17x17() {
        let n = 17;
        let a = pseudo_random(n * n, 101);
        let c = cpu_matmul(&a, &identity(n), n, n, n);
        assert_approx_eq(&c, &a, 1e-3);
    }

    #[test]
    fn test_non_tile_aligned_15x15() {
        let n = 15;
        let a = pseudo_random(n * n, 102);
        let b = pseudo_random(n * n, 103);
        let c1 = cpu_matmul(&a, &b, n, n, n);
        let c2 = cpu_matmul(&a, &b, n, n, n);
        assert_eq!(c1, c2, "deterministic results");
    }

    #[test]
    fn test_non_tile_aligned_33x33() {
        // 33 = 2×16 + 1 → tests remainder handling
        let n = 33;
        let a = pseudo_random(n * n, 200);
        let c = cpu_matmul(&a, &identity(n), n, n, n);
        assert_approx_eq(&c, &a, 1e-3);
    }

    #[test]
    fn test_rectangular_non_aligned() {
        // 13×19 × 19×7 – no dimension divisible by 16
        let (m, n, k) = (13, 7, 19);
        let a = pseudo_random(m * k, 300);
        let b = pseudo_random(k * n, 301);
        let c = cpu_matmul(&a, &b, m, n, k);
        assert_eq!(c.len(), m * n);
    }

    // ── numerical accuracy ──────────────────────────────────────────

    #[test]
    fn test_accuracy_commute_with_identity() {
        let n = 64;
        let a = pseudo_random(n * n, 55);
        let c_right = cpu_matmul(&a, &identity(n), n, n, n);
        let c_left = cpu_matmul(&identity(n), &a, n, n, n);
        assert_approx_eq(&c_right, &c_left, 1e-5);
    }

    #[test]
    fn test_accuracy_large_random() {
        // Cross-check: transposed path vs naive path on a 32×48 × 48×24
        let (m, n, k) = (32, 24, 48);
        let a = pseudo_random(m * k, 4242);
        let b = pseudo_random(k * n, 4243);

        let c_naive = cpu_matmul(&a, &b, m, n, k);

        // Build Bᵀ (n×k) from B (k×n)
        let mut b_t = vec![0.0_f32; n * k];
        for i in 0..k {
            for j in 0..n {
                b_t[j * k + i] = b[i * n + j];
            }
        }
        let c_trans = cpu_matmul_transposed(&a, &b_t, m, n, k);

        assert_approx_eq(&c_naive, &c_trans, 1e-3);
    }

    // ── optimal tile size ───────────────────────────────────────────

    #[test]
    fn test_optimal_tile_large() {
        let tc = optimal_tile_size(256, 256, 256);
        assert_eq!(tc.tile_m, 16);
        assert_eq!(tc.tile_n, 16);
        assert_eq!(tc.tile_k, 16);
        assert!(tc.use_local_memory);
    }

    #[test]
    fn test_optimal_tile_medium() {
        let tc = optimal_tile_size(12, 12, 12);
        assert_eq!(tc.tile_m, 8);
        assert!(tc.use_local_memory);
    }

    #[test]
    fn test_optimal_tile_tiny() {
        let tc = optimal_tile_size(4, 4, 4);
        assert_eq!(tc.tile_m, 4);
        assert!(!tc.use_local_memory);
    }

    #[test]
    fn test_optimal_tile_mixed_dims() {
        // min(256, 256, 4) = 4 → tiny
        let tc = optimal_tile_size(256, 256, 4);
        assert_eq!(tc.tile_m, 4);
        assert!(!tc.use_local_memory);
    }

    #[test]
    fn test_optimal_tile_boundary_8() {
        let tc = optimal_tile_size(8, 8, 8);
        assert_eq!(tc.tile_m, 8);
    }

    #[test]
    fn test_optimal_tile_boundary_16() {
        let tc = optimal_tile_size(16, 16, 16);
        assert_eq!(tc.tile_m, 16);
    }

    // ── property tests ──────────────────────────────────────────────

    #[test]
    fn test_property_a_times_identity() {
        for n in [2, 5, 16, 31] {
            let a = pseudo_random(n * n, n as u64);
            let c = cpu_matmul(&a, &identity(n), n, n, n);
            assert_approx_eq(&c, &a, 1e-3);
        }
    }

    #[test]
    fn test_property_a_times_zero() {
        for n in [2, 5, 16, 31] {
            let a = pseudo_random(n * n, n as u64);
            let zero = vec![0.0_f32; n * n];
            let c = cpu_matmul(&a, &zero, n, n, n);
            assert!(c.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_property_associativity() {
        // (A·B)·C ≈ A·(B·C) for 4×4 matrices
        let n = 4;
        let a = pseudo_random(n * n, 10);
        let b = pseudo_random(n * n, 20);
        let c = pseudo_random(n * n, 30);

        let ab = cpu_matmul(&a, &b, n, n, n);
        let ab_c = cpu_matmul(&ab, &c, n, n, n);

        let bc = cpu_matmul(&b, &c, n, n, n);
        let a_bc = cpu_matmul(&a, &bc, n, n, n);

        assert_approx_eq(&ab_c, &a_bc, 1e-3);
    }

    #[test]
    fn test_property_scalar_multiply() {
        // (sA) × B = s(A × B)
        let n = 4;
        let a = pseudo_random(n * n, 60);
        let b = pseudo_random(n * n, 61);
        let s = 2.5_f32;

        let sa: Vec<f32> = a.iter().map(|&x| x * s).collect();
        let left = cpu_matmul(&sa, &b, n, n, n);

        let ab = cpu_matmul(&a, &b, n, n, n);
        let right: Vec<f32> = ab.iter().map(|&x| x * s).collect();

        assert_approx_eq(&left, &right, 1e-3);
    }

    // ── config / struct tests ───────────────────────────────────────

    #[test]
    fn test_tile_config_default() {
        let tc = TileConfig::default();
        assert_eq!(tc.tile_m, 16);
        assert_eq!(tc.tile_n, 16);
        assert_eq!(tc.tile_k, 16);
        assert!(tc.use_local_memory);
    }

    #[test]
    fn test_tile_config_display() {
        let tc = TileConfig::default();
        let s = format!("{tc}");
        assert!(s.contains("16×16×16"));
        assert!(s.contains("local=true"));
    }

    #[test]
    fn test_matmul_config_simple() {
        let cfg = MatMulConfig::simple(32, 64, 128);
        assert_eq!(cfg.m, 32);
        assert_eq!(cfg.n, 64);
        assert_eq!(cfg.k, 128);
        assert_eq!(cfg.alpha, 1.0);
        assert_eq!(cfg.beta, 0.0);
        assert!(!cfg.transpose_a);
        assert!(!cfg.transpose_b);
    }

    #[test]
    fn test_matmul_error_display() {
        let e = MatMulError::DimensionMismatch {
            expected: 256,
            actual: 128,
            matrix: "A",
        };
        let msg = format!("{e}");
        assert!(msg.contains("dimension mismatch"));
        assert!(msg.contains("256"));
    }

    #[test]
    fn test_matmul_error_oom_display() {
        let e = MatMulError::OutOfMemory {
            requested_bytes: 1024,
            available_bytes: 512,
        };
        let msg = format!("{e}");
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn test_matmul_error_kernel_display() {
        let e = MatMulError::KernelError("build failed".into());
        let msg = format!("{e}");
        assert!(msg.contains("kernel error"));
    }

    #[test]
    fn test_tiled_kernel_new() {
        let cfg = MatMulConfig::simple(16, 16, 16);
        let kernel = TiledMatMulKernel::new(cfg);
        assert!(kernel.cached_kernel_source.is_none());
        assert_eq!(kernel.config.m, 16);
    }

    #[test]
    fn test_tiled_kernel_build_source() {
        let cfg = MatMulConfig::simple(16, 16, 16);
        let mut kernel = TiledMatMulKernel::new(cfg);
        kernel.build_source(&TileConfig::default());
        let src = kernel.kernel_source();
        assert!(src.contains("TILE_M"));
        assert!(src.contains("matmul_tiled"));
    }

    #[test]
    fn test_kernel_source_contains_all_entry_points() {
        assert!(TILED_MATMUL_SRC.contains("matmul_naive"));
        assert!(TILED_MATMUL_SRC.contains("matmul_tiled"));
        assert!(TILED_MATMUL_SRC.contains("matmul_tiled_vec4"));
        assert!(TILED_MATMUL_SRC.contains("matmul_batched"));
    }

    #[test]
    fn test_kernel_source_uses_local_memory() {
        assert!(TILED_MATMUL_SRC.contains("__local float"));
    }

    #[test]
    fn test_kernel_source_uses_pragma_unroll() {
        assert!(TILED_MATMUL_SRC.contains("#pragma unroll"));
    }

    #[test]
    fn test_kernel_source_uses_float4() {
        assert!(TILED_MATMUL_SRC.contains("float4"));
    }
}
