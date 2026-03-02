//! Multiple matrix multiplication strategies optimized for different shapes
//! on Intel Arc A770 (OpenCL).
//!
//! Provides five strategies: **Naive**, **Tiled** (shared-memory 16×16/32×32),
//! **Vectorized** (float4/float8), **SubgroupTiled** (intel_sub_group), and
//! **BatchedGemm** (multi-head attention batch dimension). A [`MatmulDispatcher`]
//! selects the best variant based on matrix dimensions. CPU reference
//! implementations and OpenCL kernel source strings are included.
//!
//! # A770 Specifics
//!
//! - Tile sizes: 16×16 default for A770 SLM (64 KB).
//! - Subgroup operations leverage `intel_sub_group_block_read`.
//! - Float4 vectorization targets bandwidth-bound tall-skinny shapes.

use std::fmt;

// ── Strategy enum ───────────────────────────────────────────────────────────

/// Matrix multiplication strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulStrategy {
    /// Simple nested-loop reference matmul. O(m·n·k).
    Naive,
    /// Shared-memory tiled implementation (16×16 or 32×32).
    Tiled,
    /// Uses float4/float8 vectorization for 4×/8× throughput.
    Vectorized,
    /// Intel subgroup tiled — coalesced loads via `intel_sub_group`.
    SubgroupTiled,
    /// Batch dimension matmul (for multi-head attention).
    BatchedGemm,
}

impl fmt::Display for MatmulStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Naive => write!(f, "Naive"),
            Self::Tiled => write!(f, "Tiled"),
            Self::Vectorized => write!(f, "Vectorized"),
            Self::SubgroupTiled => write!(f, "SubgroupTiled"),
            Self::BatchedGemm => write!(f, "BatchedGemm"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for a matrix multiplication C = A × B.
///
/// A is `[m, k]`, B is `[k, n]`, C is `[m, n]` (row-major).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl MatmulConfig {
    /// Create a config for C = A\[m,k\] × B\[k,n\] with default tiles.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k, tile_m: 16, tile_n: 16, tile_k: 16, transpose_a: false, transpose_b: false }
    }

    /// Builder: set tile sizes.
    pub fn with_tiles(mut self, tm: usize, tn: usize, tk: usize) -> Self {
        self.tile_m = tm;
        self.tile_n = tn;
        self.tile_k = tk;
        self
    }

    /// Builder: set transpose flags.
    pub fn with_transpose(mut self, ta: bool, tb: bool) -> Self {
        self.transpose_a = ta;
        self.transpose_b = tb;
        self
    }

    /// Total FLOPs for one matmul: 2·m·n·k.
    pub fn flops(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64
    }
}

impl fmt::Display for MatmulConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatmulConfig(m={}, n={}, k={}, tile={}×{}×{})",
            self.m, self.n, self.k, self.tile_m, self.tile_n, self.tile_k,
        )
    }
}

// ── Statistics ──────────────────────────────────────────────────────────────

/// Performance statistics for a matmul execution.
#[derive(Debug, Clone)]
pub struct MatmulStats {
    /// Achieved GFLOP/s.
    pub gflops: f64,
    /// Compute utilization (0.0–1.0).
    pub utilization: f64,
    /// Strategy that was selected.
    pub strategy_selected: MatmulStrategy,
    /// Wall time in seconds.
    pub elapsed_secs: f64,
}

impl fmt::Display for MatmulStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.2} GFLOP/s, {:.1}% util, {:.4}s",
            self.strategy_selected,
            self.gflops,
            self.utilization * 100.0,
            self.elapsed_secs,
        )
    }
}

// ── Error type ──────────────────────────────────────────────────────────────

/// Errors from matmul operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MatmulError {
    DimensionMismatch { expected: usize, got: usize, dim: &'static str },
    InvalidTileSize { tile: usize, dim: usize, name: &'static str },
    EmptyMatrix,
    BatchSizeMismatch { expected: usize, got: usize },
}

impl fmt::Display for MatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got, dim } => {
                write!(
                    f,
                    "dimension mismatch on {dim}: \
                     expected {expected}, got {got}"
                )
            }
            Self::InvalidTileSize { tile, dim, name } => {
                write!(
                    f,
                    "tile size {tile} exceeds {name} \
                     dimension {dim}"
                )
            }
            Self::EmptyMatrix => write!(f, "matrix has zero dimension"),
            Self::BatchSizeMismatch { expected, got } => {
                write!(
                    f,
                    "batch size mismatch: expected {expected}, \
                     got {got}"
                )
            }
        }
    }
}

impl std::error::Error for MatmulError {}

// ── Helper: element access with optional transpose ──────────────────────────

/// Read element from a row-major matrix, with optional logical transpose.
#[inline]
fn elem(data: &[f32], row: usize, col: usize, stride: usize, transposed: bool) -> f32 {
    if transposed { data[col * stride + row] } else { data[row * stride + col] }
}

// ── Naive matmul ────────────────────────────────────────────────────────────

/// Simple nested-loop reference matmul. Always correct, O(m·n·k).
pub struct NaiveMatmul;

impl NaiveMatmul {
    /// Compute C = A × B.
    ///
    /// `a` is row-major. If `cfg.transpose_a`, logical A is `a^T`.
    /// `b` is row-major. If `cfg.transpose_b`, logical B is `b^T`.
    pub fn execute(a: &[f32], b: &[f32], cfg: &MatmulConfig) -> Result<Vec<f32>, MatmulError> {
        Self::validate(a, b, cfg)?;
        let MatmulConfig { m, n, k, transpose_a, transpose_b, .. } = *cfg;

        let a_stride = if transpose_a { m } else { k };
        let b_stride = if transpose_b { k } else { n };

        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc +=
                        elem(a, i, p, a_stride, transpose_a) * elem(b, p, j, b_stride, transpose_b);
                }
                c[i * n + j] = acc;
            }
        }
        Ok(c)
    }

    fn validate(a: &[f32], b: &[f32], cfg: &MatmulConfig) -> Result<(), MatmulError> {
        if cfg.m == 0 || cfg.n == 0 || cfg.k == 0 {
            return Err(MatmulError::EmptyMatrix);
        }
        let a_expected = if cfg.transpose_a { cfg.k * cfg.m } else { cfg.m * cfg.k };
        if a.len() != a_expected {
            return Err(MatmulError::DimensionMismatch {
                expected: a_expected,
                got: a.len(),
                dim: "A",
            });
        }
        let b_expected = if cfg.transpose_b { cfg.n * cfg.k } else { cfg.k * cfg.n };
        if b.len() != b_expected {
            return Err(MatmulError::DimensionMismatch {
                expected: b_expected,
                got: b.len(),
                dim: "B",
            });
        }
        Ok(())
    }
}

// ── Tiled matmul ────────────────────────────────────────────────────────────

/// Shared-memory tiled matmul (CPU simulation of the GPU tiling strategy).
///
/// Processes the K dimension in chunks of `tile_k`, loading sub-tiles of A
/// and B into "local memory" arrays before accumulating.
pub struct TiledMatmul;

impl TiledMatmul {
    pub fn execute(a: &[f32], b: &[f32], cfg: &MatmulConfig) -> Result<Vec<f32>, MatmulError> {
        NaiveMatmul::validate(a, b, cfg)?;
        let MatmulConfig { m, n, k, tile_m, tile_n, tile_k, transpose_a, transpose_b } = *cfg;

        let a_stride = if transpose_a { m } else { k };
        let b_stride = if transpose_b { k } else { n };

        let mut c = vec![0.0f32; m * n];

        // Iterate over output tiles.
        let mut ti = 0;
        while ti < m {
            let tm = tile_m.min(m - ti);
            let mut tj = 0;
            while tj < n {
                let tn = tile_n.min(n - tj);

                // Accumulate over K tiles.
                let mut tk_off = 0;
                while tk_off < k {
                    let tk = tile_k.min(k - tk_off);

                    // Load tiles into local arrays (simulates SLM).
                    let mut tile_a = vec![0.0f32; tm * tk];
                    let mut tile_b = vec![0.0f32; tk * tn];

                    for li in 0..tm {
                        for lk in 0..tk {
                            tile_a[li * tk + lk] =
                                elem(a, ti + li, tk_off + lk, a_stride, transpose_a);
                        }
                    }
                    for lk in 0..tk {
                        for lj in 0..tn {
                            tile_b[lk * tn + lj] =
                                elem(b, tk_off + lk, tj + lj, b_stride, transpose_b);
                        }
                    }

                    // Accumulate tile product.
                    for li in 0..tm {
                        for lj in 0..tn {
                            let mut acc = 0.0f32;
                            for lk in 0..tk {
                                acc += tile_a[li * tk + lk] * tile_b[lk * tn + lj];
                            }
                            c[(ti + li) * n + (tj + lj)] += acc;
                        }
                    }

                    tk_off += tile_k;
                }
                tj += tile_n;
            }
            ti += tile_m;
        }
        Ok(c)
    }
}

// ── Vectorized matmul ───────────────────────────────────────────────────────

/// Vectorized matmul — simulates float4/float8 coalesced loads.
///
/// Processes the K dimension in groups of `VEC_WIDTH` (4 by default),
/// accumulating 4 products at a time.
pub struct VectorizedMatmul;

/// Default vector width (float4). Set to 8 for float8.
pub const DEFAULT_VEC_WIDTH: usize = 4;

impl VectorizedMatmul {
    pub fn execute(a: &[f32], b: &[f32], cfg: &MatmulConfig) -> Result<Vec<f32>, MatmulError> {
        Self::execute_with_width(a, b, cfg, DEFAULT_VEC_WIDTH)
    }

    pub fn execute_with_width(
        a: &[f32],
        b: &[f32],
        cfg: &MatmulConfig,
        vec_width: usize,
    ) -> Result<Vec<f32>, MatmulError> {
        NaiveMatmul::validate(a, b, cfg)?;
        let MatmulConfig { m, n, k, transpose_a, transpose_b, .. } = *cfg;

        let a_stride = if transpose_a { m } else { k };
        let b_stride = if transpose_b { k } else { n };

        let mut c = vec![0.0f32; m * n];
        let k_vec = k / vec_width * vec_width; // vectorised portion

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;

                // Vectorised loop.
                let mut p = 0;
                while p < k_vec {
                    let mut lane_acc = 0.0f32;
                    for v in 0..vec_width {
                        lane_acc += elem(a, i, p + v, a_stride, transpose_a)
                            * elem(b, p + v, j, b_stride, transpose_b);
                    }
                    acc += lane_acc;
                    p += vec_width;
                }
                // Scalar remainder.
                for p in k_vec..k {
                    acc +=
                        elem(a, i, p, a_stride, transpose_a) * elem(b, p, j, b_stride, transpose_b);
                }
                c[i * n + j] = acc;
            }
        }
        Ok(c)
    }
}

// ── Subgroup-tiled matmul ───────────────────────────────────────────────────

/// Intel subgroup tiled matmul — simulates `intel_sub_group_block_read`
/// coalesced loads with subgroup-width (16 on A770) tiles.
pub struct SubgroupTiledMatmul;

/// A770 subgroup width.
pub const SUBGROUP_WIDTH: usize = 16;

impl SubgroupTiledMatmul {
    pub fn execute(a: &[f32], b: &[f32], cfg: &MatmulConfig) -> Result<Vec<f32>, MatmulError> {
        NaiveMatmul::validate(a, b, cfg)?;
        let MatmulConfig { m, n, k, transpose_a, transpose_b, .. } = *cfg;

        let a_stride = if transpose_a { m } else { k };
        let b_stride = if transpose_b { k } else { n };
        let sg = SUBGROUP_WIDTH;

        let mut c = vec![0.0f32; m * n];

        // Process in subgroup-wide tiles along N.
        let mut ti = 0;
        while ti < m {
            let rows = sg.min(m - ti);
            let mut tj = 0;
            while tj < n {
                let cols = sg.min(n - tj);

                // Walk K in subgroup-width steps.
                let mut tk = 0;
                while tk < k {
                    let depth = sg.min(k - tk);

                    // Simulate subgroup block reads into registers.
                    for li in 0..rows {
                        for lj in 0..cols {
                            let mut acc = 0.0f32;
                            for lk in 0..depth {
                                acc += elem(a, ti + li, tk + lk, a_stride, transpose_a)
                                    * elem(b, tk + lk, tj + lj, b_stride, transpose_b);
                            }
                            c[(ti + li) * n + (tj + lj)] += acc;
                        }
                    }
                    tk += sg;
                }
                tj += sg;
            }
            ti += sg;
        }
        Ok(c)
    }
}

// ── Batched GEMM ────────────────────────────────────────────────────────────

/// Batched GEMM — applies the same matmul across a batch dimension.
///
/// Input shapes: A `[batch, m, k]`, B `[batch, k, n]`, C `[batch, m, n]`.
/// Used for multi-head attention where batch = num_heads.
pub struct BatchedGemm;

impl BatchedGemm {
    pub fn execute(
        a: &[f32],
        b: &[f32],
        batch: usize,
        cfg: &MatmulConfig,
    ) -> Result<Vec<f32>, MatmulError> {
        if batch == 0 {
            return Err(MatmulError::EmptyMatrix);
        }
        let a_batch_stride = if cfg.transpose_a { cfg.k * cfg.m } else { cfg.m * cfg.k };
        let b_batch_stride = if cfg.transpose_b { cfg.n * cfg.k } else { cfg.k * cfg.n };
        if a.len() != batch * a_batch_stride {
            return Err(MatmulError::DimensionMismatch {
                expected: batch * a_batch_stride,
                got: a.len(),
                dim: "A_batch",
            });
        }
        if b.len() != batch * b_batch_stride {
            return Err(MatmulError::DimensionMismatch {
                expected: batch * b_batch_stride,
                got: b.len(),
                dim: "B_batch",
            });
        }

        let c_batch_stride = cfg.m * cfg.n;
        let mut c = vec![0.0f32; batch * c_batch_stride];

        for bi in 0..batch {
            let a_slice = &a[bi * a_batch_stride..(bi + 1) * a_batch_stride];
            let b_slice = &b[bi * b_batch_stride..(bi + 1) * b_batch_stride];
            let result = NaiveMatmul::execute(a_slice, b_slice, cfg)?;
            c[bi * c_batch_stride..(bi + 1) * c_batch_stride].copy_from_slice(&result);
        }
        Ok(c)
    }
}

// ── Dispatcher ──────────────────────────────────────────────────────────────

/// Selects the best matmul strategy based on matrix dimensions.
pub struct MatmulDispatcher;

impl MatmulDispatcher {
    /// Choose a strategy for the given config.
    ///
    /// Heuristics (A770-tuned):
    /// - Batch > 1 → `BatchedGemm`
    /// - Tall-skinny (m > 8·n) or small K → `Vectorized`
    /// - All dims ≥ 16 → `SubgroupTiled`
    /// - All dims ≥ tile → `Tiled`
    /// - Otherwise → `Naive`
    pub fn select(cfg: &MatmulConfig, batch: usize) -> MatmulStrategy {
        if batch > 1 {
            return MatmulStrategy::BatchedGemm;
        }
        // Tall-skinny or very small K: vectorised is better.
        if cfg.m > 8 * cfg.n || cfg.k < 8 {
            return MatmulStrategy::Vectorized;
        }
        // Large enough for subgroup tiling.
        if cfg.m >= SUBGROUP_WIDTH && cfg.n >= SUBGROUP_WIDTH && cfg.k >= SUBGROUP_WIDTH {
            return MatmulStrategy::SubgroupTiled;
        }
        // Large enough for basic tiling.
        if cfg.m >= cfg.tile_m && cfg.n >= cfg.tile_n && cfg.k >= cfg.tile_k {
            return MatmulStrategy::Tiled;
        }
        MatmulStrategy::Naive
    }

    /// Execute using the auto-selected strategy.
    pub fn dispatch(
        a: &[f32],
        b: &[f32],
        cfg: &MatmulConfig,
    ) -> Result<(Vec<f32>, MatmulStats), MatmulError> {
        Self::dispatch_batched(a, b, 1, cfg)
    }

    /// Execute a batched matmul using the auto-selected strategy.
    pub fn dispatch_batched(
        a: &[f32],
        b: &[f32],
        batch: usize,
        cfg: &MatmulConfig,
    ) -> Result<(Vec<f32>, MatmulStats), MatmulError> {
        let strategy = Self::select(cfg, batch);
        let start = std::time::Instant::now();

        let result = match strategy {
            MatmulStrategy::Naive => NaiveMatmul::execute(a, b, cfg)?,
            MatmulStrategy::Tiled => TiledMatmul::execute(a, b, cfg)?,
            MatmulStrategy::Vectorized => VectorizedMatmul::execute(a, b, cfg)?,
            MatmulStrategy::SubgroupTiled => SubgroupTiledMatmul::execute(a, b, cfg)?,
            MatmulStrategy::BatchedGemm => BatchedGemm::execute(a, b, batch, cfg)?,
        };

        let elapsed = start.elapsed().as_secs_f64();
        let total_flops = cfg.flops() * batch.max(1) as u64;
        let gflops = if elapsed > 0.0 { total_flops as f64 / elapsed / 1e9 } else { 0.0 };
        // A770 theoretical FP32: ~19.66 TFLOP/s
        let peak_gflops = 19_660.0;
        let utilization = (gflops / peak_gflops).min(1.0);

        let stats =
            MatmulStats { gflops, utilization, strategy_selected: strategy, elapsed_secs: elapsed };
        Ok((result, stats))
    }
}

// ── OpenCL kernel sources ───────────────────────────────────────────────────

/// OpenCL kernel source for naive matmul.
pub const OPENCL_NAIVE_MATMUL: &str = r#"
__kernel void naive_matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int p = 0; p < K; ++p) {
            acc += A[row * K + p] * B[p * N + col];
        }
        C[row * N + col] = acc;
    }
}
"#;

/// OpenCL kernel source for tiled matmul (16×16).
pub const OPENCL_TILED_MATMUL: &str = r#"
#define TILE 16
__kernel void tiled_matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K)
{
    int row = get_local_id(0);
    int col = get_local_id(1);
    int gRow = get_group_id(0) * TILE + row;
    int gCol = get_group_id(1) * TILE + col;

    __local float tileA[TILE][TILE];
    __local float tileB[TILE][TILE];

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + col;
        int bRow = t * TILE + row;
        tileA[row][col] = (gRow < M && aCol < K)
            ? A[gRow * K + aCol] : 0.0f;
        tileB[row][col] = (bRow < K && gCol < N)
            ? B[bRow * N + gCol] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = 0; p < TILE; ++p)
            acc += tileA[row][p] * tileB[p][col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (gRow < M && gCol < N)
        C[gRow * N + gCol] = acc;
}
"#;

/// OpenCL kernel source for vectorized (float4) matmul.
pub const OPENCL_VECTORIZED_MATMUL: &str = r#"
__kernel void vectorized_matmul(
    __global const float* A,
    __global const float4* B4,
    __global float* C,
    const int M, const int N, const int K)
{
    int row = get_global_id(0);
    int col4 = get_global_id(1);
    int N4 = N / 4;
    if (row < M && col4 < N4) {
        float4 acc = (float4)(0.0f);
        for (int p = 0; p < K; ++p) {
            float a_val = A[row * K + p];
            acc += a_val * B4[p * N4 + col4];
        }
        int base = row * N + col4 * 4;
        C[base]     = acc.x;
        C[base + 1] = acc.y;
        C[base + 2] = acc.z;
        C[base + 3] = acc.w;
    }
}
"#;

/// OpenCL kernel source for Intel subgroup tiled matmul.
pub const OPENCL_SUBGROUP_TILED_MATMUL: &str = r#"
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define SG 16
__kernel __attribute__((intel_reqd_sub_group_size(SG)))
void subgroup_tiled_matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K)
{
    int gRow = get_group_id(0) * SG + get_sub_group_local_id();
    int gCol = get_group_id(1) * SG;

    float acc[SG];
    for (int i = 0; i < SG; ++i) acc[i] = 0.0f;

    for (int p = 0; p < K; ++p) {
        float a_val = (gRow < M) ? A[gRow * K + p] : 0.0f;
        for (int j = 0; j < SG && (gCol + j) < N; ++j) {
            float b_val = B[p * N + gCol + j];
            acc[j] += a_val * b_val;
        }
    }
    if (gRow < M) {
        for (int j = 0; j < SG && (gCol + j) < N; ++j)
            C[gRow * N + gCol + j] = acc[j];
    }
}
"#;

/// OpenCL kernel source for batched GEMM.
pub const OPENCL_BATCHED_GEMM: &str = r#"
__kernel void batched_gemm(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K, const int batch)
{
    int bi  = get_global_id(2);
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (bi < batch && row < M && col < N) {
        int a_off = bi * M * K;
        int b_off = bi * K * N;
        int c_off = bi * M * N;
        float acc = 0.0f;
        for (int p = 0; p < K; ++p)
            acc += A[a_off + row * K + p] * B[b_off + p * N + col];
        C[c_off + row * N + col] = acc;
    }
}
"#;

/// Return all OpenCL kernel source strings.
pub fn all_kernel_sources() -> Vec<(&'static str, &'static str)> {
    vec![
        ("naive_matmul", OPENCL_NAIVE_MATMUL),
        ("tiled_matmul", OPENCL_TILED_MATMUL),
        ("vectorized_matmul", OPENCL_VECTORIZED_MATMUL),
        ("subgroup_tiled_matmul", OPENCL_SUBGROUP_TILED_MATMUL),
        ("batched_gemm", OPENCL_BATCHED_GEMM),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Create an identity matrix of size `n`.
    fn identity(n: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    /// Deterministic test matrix: a[i,j] = (i * cols + j + 1) as f32.
    fn sequential(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|i| (i + 1) as f32).collect()
    }

    /// All-ones matrix.
    fn ones(rows: usize, cols: usize) -> Vec<f32> {
        vec![1.0f32; rows * cols]
    }

    /// All-zeros matrix.
    fn zeros(rows: usize, cols: usize) -> Vec<f32> {
        vec![0.0f32; rows * cols]
    }

    /// Transpose a row-major matrix.
    fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut t = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = data[i * cols + j];
            }
        }
        t
    }

    /// Assert two float slices are approximately equal.
    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
        }
    }

    // ── Naive matmul tests ──────────────────────────────────────────────

    #[test]
    fn naive_1x1() {
        let cfg = MatmulConfig::new(1, 1, 1);
        let c = NaiveMatmul::execute(&[3.0], &[4.0], &cfg).unwrap();
        assert_eq!(c, vec![12.0]);
    }

    #[test]
    fn naive_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let cfg = MatmulConfig::new(2, 2, 2);
        let c = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn naive_identity() {
        let a = sequential(4, 4);
        let id = identity(4);
        let cfg = MatmulConfig::new(4, 4, 4);
        let c = NaiveMatmul::execute(&a, &id, &cfg).unwrap();
        assert_approx_eq(&c, &a, 1e-6);
    }

    #[test]
    fn naive_zeros() {
        let a = sequential(3, 4);
        let z = zeros(4, 2);
        let cfg = MatmulConfig::new(3, 2, 4);
        let c = NaiveMatmul::execute(&a, &z, &cfg).unwrap();
        assert_eq!(c, vec![0.0; 6]);
    }

    #[test]
    fn naive_non_square() {
        // 2×3 times 3×4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let cfg = MatmulConfig::new(2, 4, 3);
        let c = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        // Row 0: [1·1+2·5+3·9, 1·2+2·6+3·10, 1·3+2·7+3·11, 1·4+2·8+3·12]
        //       = [38, 44, 50, 56]
        // Row 1: [4·1+5·5+6·9, ...] = [83, 98, 113, 128]
        assert_eq!(c, vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]);
    }

    #[test]
    fn naive_dimension_mismatch() {
        let cfg = MatmulConfig::new(2, 2, 2);
        let err = NaiveMatmul::execute(&[1.0; 3], &[1.0; 4], &cfg);
        assert!(matches!(err, Err(MatmulError::DimensionMismatch { .. })));
    }

    #[test]
    fn naive_empty_matrix_error() {
        let cfg = MatmulConfig::new(0, 2, 3);
        let err = NaiveMatmul::execute(&[], &[], &cfg);
        assert!(matches!(err, Err(MatmulError::EmptyMatrix)));
    }

    #[test]
    fn naive_tall_skinny() {
        // 8×1 times 1×1
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0];
        let cfg = MatmulConfig::new(8, 1, 1);
        let c = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let expected: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        assert_eq!(c, expected);
    }

    #[test]
    fn naive_short_wide() {
        // 1×4 times 4×8
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let cfg = MatmulConfig::new(1, 8, 4);
        let c = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        // Each output col = sum of that column across 4 rows of B.
        let expected: Vec<f32> = (0..8).map(|j| (0..4).map(|i| b[i * 8 + j]).sum()).collect();
        assert_approx_eq(&c, &expected, 1e-6);
    }

    // ── Transpose tests ─────────────────────────────────────────────────

    #[test]
    fn naive_transpose_a() {
        // A stored as 3×2, but logically A^T is 2×3.
        let a_stored = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // [3,2]
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3,2]
        let cfg = MatmulConfig::new(2, 2, 3).with_transpose(true, false);
        let c = NaiveMatmul::execute(&a_stored, &b, &cfg).unwrap();
        // A^T = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]]
        // C = [[22,28],[49,64]]
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn naive_transpose_b() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        // B stored [2,3] but logically B^T is [3,2]
        let b_stored = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // [2,3]
        let cfg = MatmulConfig::new(2, 2, 3).with_transpose(false, true);
        let c = NaiveMatmul::execute(&a, &b_stored, &cfg).unwrap();
        // A = [[1,2,3],[4,5,6]], B^T => B = [[1,2],[3,4],[5,6]]
        // C = [[22,28],[49,64]]
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn naive_transpose_both() {
        // A stored [3,2], B stored [2,3]
        let a_stored = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b_stored = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let cfg = MatmulConfig::new(2, 2, 3).with_transpose(true, true);
        let c = NaiveMatmul::execute(&a_stored, &b_stored, &cfg).unwrap();
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    // ── Tiled matmul tests ──────────────────────────────────────────────

    #[test]
    fn tiled_matches_naive_small() {
        let a = sequential(4, 4);
        let b = sequential(4, 4);
        let cfg = MatmulConfig::new(4, 4, 4).with_tiles(2, 2, 2);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-5);
    }

    #[test]
    fn tiled_matches_naive_non_square() {
        let a = sequential(6, 8);
        let b = sequential(8, 5);
        let cfg = MatmulConfig::new(6, 5, 8).with_tiles(4, 4, 4);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-4);
    }

    #[test]
    fn tiled_non_multiple_of_tile() {
        // 7×5 × 5×3, tile 4×4×4 — tests remainder handling.
        let a = sequential(7, 5);
        let b = sequential(5, 3);
        let cfg = MatmulConfig::new(7, 3, 5).with_tiles(4, 4, 4);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-4);
    }

    #[test]
    fn tiled_16x16() {
        let a = sequential(16, 16);
        let b = sequential(16, 16);
        let cfg = MatmulConfig::new(16, 16, 16);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-3);
    }

    #[test]
    fn tiled_32x32() {
        let a = sequential(32, 32);
        let b = sequential(32, 32);
        let cfg = MatmulConfig::new(32, 32, 32).with_tiles(32, 32, 32);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-2);
    }

    #[test]
    fn tiled_identity() {
        let a = sequential(8, 8);
        let id = identity(8);
        let cfg = MatmulConfig::new(8, 8, 8).with_tiles(4, 4, 4);
        let c = TiledMatmul::execute(&a, &id, &cfg).unwrap();
        assert_approx_eq(&c, &a, 1e-5);
    }

    #[test]
    fn tiled_transpose_a() {
        let a_stored = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = MatmulConfig::new(2, 2, 3).with_tiles(2, 2, 2).with_transpose(true, false);
        let naive = NaiveMatmul::execute(&a_stored, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a_stored, &b, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-5);
    }

    // ── Vectorized matmul tests ─────────────────────────────────────────

    #[test]
    fn vectorized_matches_naive_small() {
        let a = sequential(4, 8);
        let b = sequential(8, 4);
        let cfg = MatmulConfig::new(4, 4, 8);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let vec_r = VectorizedMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&vec_r, &naive, 1e-4);
    }

    #[test]
    fn vectorized_matches_naive_non_multiple() {
        // K=7 is not a multiple of vec_width 4 — tests remainder.
        let a = sequential(3, 7);
        let b = sequential(7, 5);
        let cfg = MatmulConfig::new(3, 5, 7);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let vec_r = VectorizedMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&vec_r, &naive, 1e-3);
    }

    #[test]
    fn vectorized_float8_width() {
        let a = sequential(4, 16);
        let b = sequential(16, 4);
        let cfg = MatmulConfig::new(4, 4, 16);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let vec8 = VectorizedMatmul::execute_with_width(&a, &b, &cfg, 8).unwrap();
        assert_approx_eq(&vec8, &naive, 1e-3);
    }

    #[test]
    fn vectorized_width_1_matches_naive() {
        let a = sequential(3, 5);
        let b = sequential(5, 3);
        let cfg = MatmulConfig::new(3, 3, 5);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let v1 = VectorizedMatmul::execute_with_width(&a, &b, &cfg, 1).unwrap();
        assert_approx_eq(&v1, &naive, 1e-5);
    }

    #[test]
    fn vectorized_identity() {
        let a = sequential(8, 8);
        let id = identity(8);
        let cfg = MatmulConfig::new(8, 8, 8);
        let c = VectorizedMatmul::execute(&a, &id, &cfg).unwrap();
        assert_approx_eq(&c, &a, 1e-5);
    }

    #[test]
    fn vectorized_tall_skinny() {
        let a = sequential(32, 4);
        let b = sequential(4, 2);
        let cfg = MatmulConfig::new(32, 2, 4);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let vec_r = VectorizedMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&vec_r, &naive, 1e-3);
    }

    // ── Subgroup-tiled matmul tests ─────────────────────────────────────

    #[test]
    fn subgroup_matches_naive_16x16() {
        let a = sequential(16, 16);
        let b = sequential(16, 16);
        let cfg = MatmulConfig::new(16, 16, 16);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let sg = SubgroupTiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&sg, &naive, 1e-3);
    }

    #[test]
    fn subgroup_non_multiple_of_16() {
        let a = sequential(17, 19);
        let b = sequential(19, 13);
        let cfg = MatmulConfig::new(17, 13, 19);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let sg = SubgroupTiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&sg, &naive, 1e-2);
    }

    #[test]
    fn subgroup_identity() {
        let a = sequential(16, 16);
        let id = identity(16);
        let cfg = MatmulConfig::new(16, 16, 16);
        let c = SubgroupTiledMatmul::execute(&a, &id, &cfg).unwrap();
        assert_approx_eq(&c, &a, 1e-3);
    }

    #[test]
    fn subgroup_32x32() {
        let a = sequential(32, 32);
        let b = sequential(32, 32);
        let cfg = MatmulConfig::new(32, 32, 32);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let sg = SubgroupTiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&sg, &naive, 1e-1);
    }

    #[test]
    fn subgroup_transpose_b() {
        let a = sequential(16, 16);
        let b_stored = transpose(&sequential(16, 16), 16, 16);
        let cfg = MatmulConfig::new(16, 16, 16).with_transpose(false, true);
        let naive = NaiveMatmul::execute(&a, &b_stored, &cfg).unwrap();
        let sg = SubgroupTiledMatmul::execute(&a, &b_stored, &cfg).unwrap();
        assert_approx_eq(&sg, &naive, 1e-3);
    }

    // ── Batched GEMM tests ──────────────────────────────────────────────

    #[test]
    fn batched_single_matches_naive() {
        let a = sequential(4, 4);
        let b = sequential(4, 4);
        let cfg = MatmulConfig::new(4, 4, 4);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let batched = BatchedGemm::execute(&a, &b, 1, &cfg).unwrap();
        assert_approx_eq(&batched, &naive, 1e-6);
    }

    #[test]
    fn batched_multi_head() {
        let batch = 4;
        let cfg = MatmulConfig::new(2, 2, 2);
        let a: Vec<f32> = (0..batch * 4).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..batch * 4).map(|i| (i + 1) as f32 * 0.5).collect();
        let result = BatchedGemm::execute(&a, &b, batch, &cfg).unwrap();
        assert_eq!(result.len(), batch * 4);

        // Verify each batch independently.
        for bi in 0..batch {
            let a_slice = &a[bi * 4..(bi + 1) * 4];
            let b_slice = &b[bi * 4..(bi + 1) * 4];
            let expected = NaiveMatmul::execute(a_slice, b_slice, &cfg).unwrap();
            assert_approx_eq(&result[bi * 4..(bi + 1) * 4], &expected, 1e-5);
        }
    }

    #[test]
    fn batched_gemm_dimension_error() {
        let cfg = MatmulConfig::new(2, 2, 2);
        let err = BatchedGemm::execute(&[1.0; 3], &[1.0; 4], 1, &cfg);
        assert!(matches!(err, Err(MatmulError::DimensionMismatch { .. })));
    }

    #[test]
    fn batched_gemm_zero_batch() {
        let cfg = MatmulConfig::new(2, 2, 2);
        let err = BatchedGemm::execute(&[], &[], 0, &cfg);
        assert!(matches!(err, Err(MatmulError::EmptyMatrix)));
    }

    #[test]
    fn batched_gemm_identity() {
        let batch = 3;
        let cfg = MatmulConfig::new(4, 4, 4);
        let a: Vec<f32> = (0..batch).flat_map(|_| sequential(4, 4)).collect();
        let id: Vec<f32> = (0..batch).flat_map(|_| identity(4)).collect();
        let result = BatchedGemm::execute(&a, &id, batch, &cfg).unwrap();
        assert_approx_eq(&result, &a, 1e-5);
    }

    #[test]
    fn batched_gemm_non_square() {
        let batch = 2;
        let cfg = MatmulConfig::new(3, 4, 5);
        let a: Vec<f32> = (0..batch * 15).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..batch * 20).map(|i| (i + 1) as f32 * 0.1).collect();
        let result = BatchedGemm::execute(&a, &b, batch, &cfg).unwrap();
        assert_eq!(result.len(), batch * 12);
    }

    #[test]
    fn batched_gemm_transpose() {
        let batch = 2;
        let cfg = MatmulConfig::new(2, 2, 3).with_transpose(true, false);
        // A stored [k,m] = [3,2] per batch
        let a: Vec<f32> = (0..batch * 6).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..batch * 6).map(|i| (i + 1) as f32 * 0.5).collect();
        let result = BatchedGemm::execute(&a, &b, batch, &cfg).unwrap();
        assert_eq!(result.len(), batch * 4);
    }

    // ── Dispatcher tests ────────────────────────────────────────────────

    #[test]
    fn dispatcher_selects_batched_gemm() {
        let cfg = MatmulConfig::new(16, 16, 16);
        assert_eq!(MatmulDispatcher::select(&cfg, 4), MatmulStrategy::BatchedGemm);
    }

    #[test]
    fn dispatcher_selects_subgroup_tiled() {
        let cfg = MatmulConfig::new(64, 64, 64);
        assert_eq!(MatmulDispatcher::select(&cfg, 1), MatmulStrategy::SubgroupTiled);
    }

    #[test]
    fn dispatcher_selects_tiled_medium() {
        let cfg = MatmulConfig::new(16, 16, 16).with_tiles(8, 8, 8);
        // m,n,k ≥ 16 → SubgroupTiled first. Use smaller dims.
        let cfg2 = MatmulConfig::new(14, 14, 14).with_tiles(8, 8, 8);
        assert_eq!(MatmulDispatcher::select(&cfg2, 1), MatmulStrategy::Tiled);
        // But 16×16 still hits subgroup path:
        assert_eq!(MatmulDispatcher::select(&cfg, 1), MatmulStrategy::SubgroupTiled);
    }

    #[test]
    fn dispatcher_selects_vectorized_tall_skinny() {
        let cfg = MatmulConfig::new(256, 2, 32);
        assert_eq!(MatmulDispatcher::select(&cfg, 1), MatmulStrategy::Vectorized);
    }

    #[test]
    fn dispatcher_selects_vectorized_small_k() {
        let cfg = MatmulConfig::new(32, 32, 4);
        assert_eq!(MatmulDispatcher::select(&cfg, 1), MatmulStrategy::Vectorized);
    }

    #[test]
    fn dispatcher_selects_naive_tiny() {
        let cfg = MatmulConfig::new(2, 2, 2).with_tiles(4, 4, 4);
        assert_eq!(
            MatmulDispatcher::select(&cfg, 1),
            MatmulStrategy::Vectorized // k < 8 triggers Vectorized
        );
        // Force naive: no tall-skinny, no small k, but < tile
        let cfg2 = MatmulConfig::new(3, 3, 10).with_tiles(16, 16, 16);
        assert_eq!(MatmulDispatcher::select(&cfg2, 1), MatmulStrategy::Naive);
    }

    #[test]
    fn dispatcher_dispatch_produces_result() {
        let a = sequential(4, 4);
        let b = sequential(4, 4);
        let cfg = MatmulConfig::new(4, 4, 4);
        let (c, stats) = MatmulDispatcher::dispatch(&a, &b, &cfg).unwrap();
        assert_eq!(c.len(), 16);
        assert!(stats.gflops >= 0.0);
        assert!(stats.elapsed_secs >= 0.0);
    }

    #[test]
    fn dispatcher_batched_dispatch() {
        let batch = 2;
        let cfg = MatmulConfig::new(4, 4, 4);
        let a: Vec<f32> = (0..batch * 16).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..batch * 16).map(|i| (i + 1) as f32).collect();
        let (c, stats) = MatmulDispatcher::dispatch_batched(&a, &b, batch, &cfg).unwrap();
        assert_eq!(c.len(), batch * 16);
        assert_eq!(stats.strategy_selected, MatmulStrategy::BatchedGemm);
    }

    // ── Cross-strategy equivalence tests ────────────────────────────────

    #[test]
    fn all_strategies_agree_8x8() {
        let a = sequential(8, 8);
        let b = sequential(8, 8);
        let cfg = MatmulConfig::new(8, 8, 8).with_tiles(4, 4, 4);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        let vec_r = VectorizedMatmul::execute(&a, &b, &cfg).unwrap();
        let sg = SubgroupTiledMatmul::execute(&a, &b, &cfg).unwrap();
        let batched = BatchedGemm::execute(&a, &b, 1, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-4);
        assert_approx_eq(&vec_r, &naive, 1e-4);
        assert_approx_eq(&sg, &naive, 1e-4);
        assert_approx_eq(&batched, &naive, 1e-6);
    }

    #[test]
    fn all_strategies_agree_16x16() {
        let a = sequential(16, 16);
        let b = sequential(16, 16);
        let cfg = MatmulConfig::new(16, 16, 16);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        let vec_r = VectorizedMatmul::execute(&a, &b, &cfg).unwrap();
        let sg = SubgroupTiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-2);
        assert_approx_eq(&vec_r, &naive, 1e-2);
        assert_approx_eq(&sg, &naive, 1e-2);
    }

    #[test]
    fn all_strategies_agree_non_square() {
        let a = sequential(10, 20);
        let b = sequential(20, 15);
        let cfg = MatmulConfig::new(10, 15, 20).with_tiles(4, 4, 4);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        let vec_r = VectorizedMatmul::execute(&a, &b, &cfg).unwrap();
        let sg = SubgroupTiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-1);
        assert_approx_eq(&vec_r, &naive, 1e-1);
        assert_approx_eq(&sg, &naive, 1e-1);
    }

    // ── Property tests ──────────────────────────────────────────────────

    #[test]
    fn property_a_times_identity() {
        for n in [1, 2, 4, 7, 16] {
            let a = sequential(n, n);
            let id = identity(n);
            let cfg = MatmulConfig::new(n, n, n).with_tiles(4, 4, 4);
            let c = NaiveMatmul::execute(&a, &id, &cfg).unwrap();
            assert_approx_eq(&c, &a, 1e-3);
        }
    }

    #[test]
    fn property_identity_times_b() {
        for n in [1, 2, 4, 7, 16] {
            let id = identity(n);
            let b = sequential(n, n);
            let cfg = MatmulConfig::new(n, n, n).with_tiles(4, 4, 4);
            let c = NaiveMatmul::execute(&id, &b, &cfg).unwrap();
            assert_approx_eq(&c, &b, 1e-3);
        }
    }

    #[test]
    fn property_zero_times_anything() {
        let z = zeros(4, 6);
        let b = sequential(6, 3);
        let cfg = MatmulConfig::new(4, 3, 6);
        let c = NaiveMatmul::execute(&z, &b, &cfg).unwrap();
        assert_eq!(c, vec![0.0; 12]);
    }

    #[test]
    fn property_ones_row_times_b_is_col_sums() {
        let a = ones(1, 5);
        let b = sequential(5, 3);
        let cfg = MatmulConfig::new(1, 3, 5);
        let c = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        // Each output element is sum of the column in B.
        for j in 0..3 {
            let col_sum: f32 = (0..5).map(|i| b[i * 3 + j]).sum();
            assert!((c[j] - col_sum).abs() < 1e-5);
        }
    }

    #[test]
    fn property_associative_check() {
        // (A × B) × C ≈ A × (B × C) for small matrices.
        let a = sequential(3, 4);
        let b = sequential(4, 5);
        let cc = sequential(5, 2);
        let cfg_ab = MatmulConfig::new(3, 5, 4);
        let cfg_bc = MatmulConfig::new(4, 2, 5);
        let ab = NaiveMatmul::execute(&a, &b, &cfg_ab).unwrap();
        let bc = NaiveMatmul::execute(&b, &cc, &cfg_bc).unwrap();
        let cfg_abc1 = MatmulConfig::new(3, 2, 5);
        let cfg_abc2 = MatmulConfig::new(3, 2, 4);
        let abc1 = NaiveMatmul::execute(&ab, &cc, &cfg_abc1).unwrap();
        let abc2 = NaiveMatmul::execute(&a, &bc, &cfg_abc2).unwrap();
        assert_approx_eq(&abc1, &abc2, 1e-1);
    }

    // ── Edge case tests ─────────────────────────────────────────────────

    #[test]
    fn edge_1x1_all_strategies() {
        let a = vec![3.0f32];
        let b = vec![7.0f32];
        let cfg = MatmulConfig::new(1, 1, 1);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        let vec_r = VectorizedMatmul::execute(&a, &b, &cfg).unwrap();
        let sg = SubgroupTiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_eq!(naive, vec![21.0]);
        assert_eq!(tiled, vec![21.0]);
        assert_eq!(vec_r, vec![21.0]);
        assert_eq!(sg, vec![21.0]);
    }

    #[test]
    fn edge_large_64x64() {
        let a = sequential(64, 64);
        let b = identity(64);
        let cfg = MatmulConfig::new(64, 64, 64);
        let c = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&c, &a, 1e-2);
    }

    #[test]
    fn edge_prime_dimensions() {
        // 7×11 × 11×13 — primes, never multiples of any tile.
        let a = sequential(7, 11);
        let b = sequential(11, 13);
        let cfg = MatmulConfig::new(7, 13, 11).with_tiles(4, 4, 4);
        let naive = NaiveMatmul::execute(&a, &b, &cfg).unwrap();
        let tiled = TiledMatmul::execute(&a, &b, &cfg).unwrap();
        assert_approx_eq(&tiled, &naive, 1e-2);
    }

    // ── Config / Stats / Display tests ──────────────────────────────────

    #[test]
    fn config_flops() {
        let cfg = MatmulConfig::new(100, 200, 300);
        assert_eq!(cfg.flops(), 2 * 100 * 200 * 300);
    }

    #[test]
    fn config_display() {
        let cfg = MatmulConfig::new(4, 4, 4);
        let s = format!("{cfg}");
        assert!(s.contains("m=4"));
        assert!(s.contains("n=4"));
    }

    #[test]
    fn strategy_display() {
        assert_eq!(format!("{}", MatmulStrategy::Naive), "Naive");
        assert_eq!(format!("{}", MatmulStrategy::SubgroupTiled), "SubgroupTiled");
    }

    #[test]
    fn stats_display() {
        let stats = MatmulStats {
            gflops: 12.5,
            utilization: 0.5,
            strategy_selected: MatmulStrategy::Tiled,
            elapsed_secs: 0.001,
        };
        let s = format!("{stats}");
        assert!(s.contains("Tiled"));
        assert!(s.contains("12.50"));
    }

    #[test]
    fn error_display() {
        let e = MatmulError::DimensionMismatch { expected: 4, got: 5, dim: "K" };
        assert!(format!("{e}").contains("K"));
        let e2 = MatmulError::EmptyMatrix;
        assert!(format!("{e2}").contains("zero"));
    }

    #[test]
    fn error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(MatmulError::EmptyMatrix);
        assert!(e.to_string().contains("zero"));
    }

    // ── OpenCL kernel source tests ──────────────────────────────────────

    #[test]
    fn kernel_source_naive_contains_kernel_name() {
        assert!(OPENCL_NAIVE_MATMUL.contains("naive_matmul"));
        assert!(OPENCL_NAIVE_MATMUL.contains("__kernel"));
    }

    #[test]
    fn kernel_source_tiled_contains_local_mem() {
        assert!(OPENCL_TILED_MATMUL.contains("__local"));
        assert!(OPENCL_TILED_MATMUL.contains("barrier"));
    }

    #[test]
    fn kernel_source_vectorized_uses_float4() {
        assert!(OPENCL_VECTORIZED_MATMUL.contains("float4"));
    }

    #[test]
    fn kernel_source_subgroup_uses_intel_ext() {
        assert!(OPENCL_SUBGROUP_TILED_MATMUL.contains("cl_intel_subgroups"));
        assert!(OPENCL_SUBGROUP_TILED_MATMUL.contains("intel_reqd_sub_group_size"));
    }

    #[test]
    fn kernel_source_batched_has_batch_dim() {
        assert!(OPENCL_BATCHED_GEMM.contains("batch"));
        assert!(OPENCL_BATCHED_GEMM.contains("get_global_id(2)"));
    }

    #[test]
    fn all_kernel_sources_count() {
        let sources = all_kernel_sources();
        assert_eq!(sources.len(), 5);
        for (name, src) in &sources {
            assert!(src.contains("__kernel"), "{name} missing __kernel");
        }
    }

    // ── Misc / builder tests ────────────────────────────────────────────

    #[test]
    fn config_builder_tiles() {
        let cfg = MatmulConfig::new(8, 8, 8).with_tiles(32, 32, 32);
        assert_eq!(cfg.tile_m, 32);
        assert_eq!(cfg.tile_n, 32);
        assert_eq!(cfg.tile_k, 32);
    }

    #[test]
    fn config_builder_transpose() {
        let cfg = MatmulConfig::new(8, 8, 8).with_transpose(true, true);
        assert!(cfg.transpose_a);
        assert!(cfg.transpose_b);
    }

    #[test]
    fn default_vec_width_is_4() {
        assert_eq!(DEFAULT_VEC_WIDTH, 4);
    }

    #[test]
    fn subgroup_width_is_16() {
        assert_eq!(SUBGROUP_WIDTH, 16);
    }

    #[test]
    fn strategy_enum_variants_are_distinct() {
        let variants = [
            MatmulStrategy::Naive,
            MatmulStrategy::Tiled,
            MatmulStrategy::Vectorized,
            MatmulStrategy::SubgroupTiled,
            MatmulStrategy::BatchedGemm,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }
}
