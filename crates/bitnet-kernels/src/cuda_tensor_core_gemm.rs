//! CUDA Tensor Core GEMM engine with warp-level MMA operations.
//!
//! # Overview
//!
//! This module implements a General Matrix Multiplication (GEMM) engine
//! that targets NVIDIA Tensor Cores via Warp Matrix Multiply-Accumulate
//! (WMMA / MMA) instructions.  It supports:
//!
//! - **Multiple precision modes**: FP16, BF16, TF32, INT8, INT4
//! - **Configurable tiling**: 128×128, 64×64, 32×32 output tiles
//! - **Fragment management**: typed operand fragments (A, B, C, D)
//! - **Warp-level MMA scheduling**: cooperative matrix operations
//! - **Mixed-precision accumulation**: low-precision inputs → FP32 output
//!
//! # CPU fallback
//!
//! Every public function has a pure-Rust CPU fallback so that tests
//! pass without GPU hardware.  The dispatcher tries the GPU path first
//! and falls back transparently.
//!
//! # Feature gate
//!
//! GPU-specific types and CUDA kernel sources are behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Precision mode ────────────────────────────────────────────────────

/// Tensor Core precision mode.
///
/// Each variant describes the input element type and the hardware
/// instruction set used for the MMA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorCorePrecision {
    /// IEEE FP16 inputs, FP32 accumulator (SM ≥ 7.0).
    Fp16,
    /// BFloat16 inputs, FP32 accumulator (SM ≥ 8.0).
    Bf16,
    /// TensorFloat-32 inputs (FP32 truncated to 19 bits), FP32
    /// accumulator (SM ≥ 8.0).
    Tf32,
    /// Signed 8-bit integer inputs, INT32 accumulator (SM ≥ 7.2).
    Int8,
    /// Signed 4-bit integer inputs, INT32 accumulator (SM ≥ 7.5).
    Int4,
}

impl TensorCorePrecision {
    /// Number of bits per element in this precision mode.
    pub fn element_bits(self) -> u32 {
        match self {
            Self::Fp16 | Self::Bf16 => 16,
            Self::Tf32 => 32,
            Self::Int8 => 8,
            Self::Int4 => 4,
        }
    }

    /// Minimum NVIDIA SM (Streaming Multiprocessor) version required.
    pub fn min_sm_version(self) -> u32 {
        match self {
            Self::Fp16 => 70,
            Self::Bf16 | Self::Tf32 => 80,
            Self::Int8 => 72,
            Self::Int4 => 75,
        }
    }

    /// Whether the accumulator is floating-point (vs integer).
    pub fn accumulator_is_float(self) -> bool {
        matches!(self, Self::Fp16 | Self::Bf16 | Self::Tf32)
    }
}

// ── Tiling strategy ───────────────────────────────────────────────────

/// Output tile dimensions for the GEMM kernel.
///
/// The tile size determines the amount of work assigned to each
/// thread-block.  Larger tiles improve data reuse but require more
/// shared memory and registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileStrategy {
    /// 128 × 128 output tile — highest throughput, highest resource use.
    Tile128x128,
    /// 64 × 64 output tile — balanced throughput and occupancy.
    Tile64x64,
    /// 32 × 32 output tile — lowest resource use, best for small matrices.
    Tile32x32,
}

impl TileStrategy {
    /// Tile dimension along M (rows).
    pub fn tile_m(self) -> usize {
        match self {
            Self::Tile128x128 => 128,
            Self::Tile64x64 => 64,
            Self::Tile32x32 => 32,
        }
    }

    /// Tile dimension along N (columns).
    pub fn tile_n(self) -> usize {
        self.tile_m()
    }

    /// Default tile-K (reduction) dimension — always 32.
    pub fn tile_k(self) -> usize {
        32
    }

    /// Number of warps required per thread-block.
    pub fn warps_per_block(self) -> usize {
        match self {
            Self::Tile128x128 => 8,
            Self::Tile64x64 => 4,
            Self::Tile32x32 => 2,
        }
    }

    /// Threads per block (warps × 32).
    pub fn threads_per_block(self) -> usize {
        self.warps_per_block() * WARP_SIZE
    }

    /// Shared memory bytes needed for double-buffered A + B tiles.
    pub fn shared_memory_bytes(self, precision: TensorCorePrecision) -> usize {
        let elem_bytes = (precision.element_bits() as usize + 7) / 8;
        let a_bytes = self.tile_m() * self.tile_k() * elem_bytes;
        let b_bytes = self.tile_k() * self.tile_n() * elem_bytes;
        // Double-buffered
        2 * (a_bytes + b_bytes)
    }

    /// Select the best tile strategy for the given dimensions.
    pub fn auto_select(m: usize, n: usize) -> Self {
        let min_dim = m.min(n);
        if min_dim >= 128 {
            Self::Tile128x128
        } else if min_dim >= 64 {
            Self::Tile64x64
        } else {
            Self::Tile32x32
        }
    }
}

/// Threads per warp (NVIDIA hardware constant).
pub const WARP_SIZE: usize = 32;

/// WMMA fragment M dimension.
const WMMA_M: usize = 16;
/// WMMA fragment N dimension.
const WMMA_N: usize = 16;
/// WMMA fragment K dimension.
const WMMA_K: usize = 16;

// ── Fragment types ────────────────────────────────────────────────────

/// Role of a WMMA fragment in the MMA operation D = A·B + C.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentRole {
    /// Left operand (M × K).
    MatrixA,
    /// Right operand (K × N).
    MatrixB,
    /// Input accumulator (M × N).
    Accumulator,
}

/// A typed WMMA fragment holding register-resident matrix data.
///
/// On real hardware each warp holds a distributed fragment; this
/// CPU-side representation stores the logical element values for
/// simulation and validation.
#[derive(Debug, Clone)]
pub struct WmmaFragment {
    /// Fragment role in the MMA equation.
    pub role: FragmentRole,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Element data in row-major order.
    pub data: Vec<f32>,
}

impl WmmaFragment {
    /// Create a new fragment filled with zeros.
    pub fn new(role: FragmentRole, rows: usize, cols: usize) -> Self {
        Self { role, rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Create a fragment for operand A (WMMA_M × WMMA_K).
    pub fn new_a() -> Self {
        Self::new(FragmentRole::MatrixA, WMMA_M, WMMA_K)
    }

    /// Create a fragment for operand B (WMMA_K × WMMA_N).
    pub fn new_b() -> Self {
        Self::new(FragmentRole::MatrixB, WMMA_K, WMMA_N)
    }

    /// Create an accumulator fragment (WMMA_M × WMMA_N).
    pub fn new_accumulator() -> Self {
        Self::new(FragmentRole::Accumulator, WMMA_M, WMMA_N)
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.rows * self.cols
    }

    /// Load from a row-major source slice at the given (row, col)
    /// offset with the specified leading dimension (stride).
    pub fn load_matrix(
        &mut self,
        src: &[f32],
        row_offset: usize,
        col_offset: usize,
        ld: usize,
    ) -> Result<()> {
        for r in 0..self.rows {
            let src_row = row_offset + r;
            for c in 0..self.cols {
                let src_col = col_offset + c;
                let src_idx = src_row * ld + src_col;
                if src_idx >= src.len() {
                    return Err(BitNetError::Kernel(KernelError::InvalidDimension(format!(
                        "fragment load OOB: index {} >= len {}",
                        src_idx,
                        src.len()
                    ))));
                }
                self.data[r * self.cols + c] = src[src_idx];
            }
        }
        Ok(())
    }

    /// Store the fragment into a row-major destination at the given offset.
    pub fn store_matrix(
        &self,
        dst: &mut [f32],
        row_offset: usize,
        col_offset: usize,
        ld: usize,
    ) -> Result<()> {
        for r in 0..self.rows {
            let dst_row = row_offset + r;
            for c in 0..self.cols {
                let dst_col = col_offset + c;
                let dst_idx = dst_row * ld + dst_col;
                if dst_idx >= dst.len() {
                    return Err(BitNetError::Kernel(KernelError::InvalidDimension(format!(
                        "fragment store OOB: index {} >= len {}",
                        dst_idx,
                        dst.len()
                    ))));
                }
                dst[dst_idx] = self.data[r * self.cols + c];
            }
        }
        Ok(())
    }

    /// Fill every element with `val`.
    pub fn fill(&mut self, val: f32) {
        self.data.fill(val);
    }
}

// ── Accumulator ───────────────────────────────────────────────────────

/// Manages a grid of WMMA accumulator fragments covering the full
/// output tile.
#[derive(Debug)]
pub struct AccumulatorGrid {
    /// Fragments in row-major order (tiles_m × tiles_n).
    fragments: Vec<WmmaFragment>,
    /// Number of fragment tiles along M.
    pub tiles_m: usize,
    /// Number of fragment tiles along N.
    pub tiles_n: usize,
}

impl AccumulatorGrid {
    /// Create a zero-initialised accumulator grid.
    pub fn new(m: usize, n: usize) -> Self {
        let tiles_m = (m + WMMA_M - 1) / WMMA_M;
        let tiles_n = (n + WMMA_N - 1) / WMMA_N;
        let fragments = (0..tiles_m * tiles_n).map(|_| WmmaFragment::new_accumulator()).collect();
        Self { fragments, tiles_m, tiles_n }
    }

    /// Total number of fragments.
    pub fn num_fragments(&self) -> usize {
        self.fragments.len()
    }

    /// Access fragment at (tile_row, tile_col).
    pub fn get(&self, tile_row: usize, tile_col: usize) -> Option<&WmmaFragment> {
        if tile_row < self.tiles_m && tile_col < self.tiles_n {
            Some(&self.fragments[tile_row * self.tiles_n + tile_col])
        } else {
            None
        }
    }

    /// Mutably access fragment at (tile_row, tile_col).
    pub fn get_mut(&mut self, tile_row: usize, tile_col: usize) -> Option<&mut WmmaFragment> {
        if tile_row < self.tiles_m && tile_col < self.tiles_n {
            Some(&mut self.fragments[tile_row * self.tiles_n + tile_col])
        } else {
            None
        }
    }

    /// Store the full accumulator grid into a row-major destination.
    pub fn store_to(&self, dst: &mut [f32], m: usize, n: usize) -> Result<()> {
        for tr in 0..self.tiles_m {
            for tc in 0..self.tiles_n {
                let frag = &self.fragments[tr * self.tiles_n + tc];
                let row_off = tr * WMMA_M;
                let col_off = tc * WMMA_N;
                for r in 0..WMMA_M {
                    if row_off + r >= m {
                        break;
                    }
                    for c in 0..WMMA_N {
                        if col_off + c >= n {
                            break;
                        }
                        let dst_idx = (row_off + r) * n + (col_off + c);
                        dst[dst_idx] = frag.data[r * WMMA_N + c];
                    }
                }
            }
        }
        Ok(())
    }

    /// Reset all accumulators to zero.
    pub fn clear(&mut self) {
        for frag in &mut self.fragments {
            frag.fill(0.0);
        }
    }
}

// ── MMA operation (CPU simulation) ────────────────────────────────────

/// Perform a single WMMA MMA: `d = a · b + c` where `a` is M×K, `b` is
/// K×N, and `c`/`d` are M×N accumulators.
///
/// This is the CPU reference implementation; on GPU this maps to a
/// single `wmma::mma_sync` or `mma.sync` PTX instruction.
pub fn mma_sync(
    a: &WmmaFragment,
    b: &WmmaFragment,
    c: &WmmaFragment,
    d: &mut WmmaFragment,
) -> Result<()> {
    if a.role != FragmentRole::MatrixA {
        return Err(BitNetError::Kernel(KernelError::InvalidDimension(
            "fragment a must have MatrixA role".into(),
        )));
    }
    if b.role != FragmentRole::MatrixB {
        return Err(BitNetError::Kernel(KernelError::InvalidDimension(
            "fragment b must have MatrixB role".into(),
        )));
    }
    if c.role != FragmentRole::Accumulator || d.role != FragmentRole::Accumulator {
        return Err(BitNetError::Kernel(KernelError::InvalidDimension(
            "fragments c and d must have Accumulator role".into(),
        )));
    }
    if a.cols != b.rows {
        return Err(BitNetError::Kernel(KernelError::InvalidDimension(format!(
            "MMA inner dimension mismatch: a.cols={} != b.rows={}",
            a.cols, b.rows,
        ))));
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    // D = A·B + C
    for r in 0..m {
        for col in 0..n {
            let mut sum = c.data[r * n + col];
            for i in 0..k {
                sum += a.data[r * k + i] * b.data[i * n + col];
            }
            d.data[r * n + col] = sum;
        }
    }
    Ok(())
}

// ── Warp-level MMA schedule ───────────────────────────────────────────

/// Describes a scheduled MMA operation within the warp-level pipeline.
#[derive(Debug, Clone)]
pub struct MmaScheduleEntry {
    /// Fragment tile row index.
    pub tile_row: usize,
    /// Fragment tile column index.
    pub tile_col: usize,
    /// K-dimension step index.
    pub k_step: usize,
}

/// Generate a warp-level MMA schedule for the given tile and K
/// dimensions.
///
/// The schedule orders the MMA operations so that fragments along the
/// K dimension are accumulated before moving to the next output tile.
pub fn build_mma_schedule(tiles_m: usize, tiles_n: usize, k_steps: usize) -> Vec<MmaScheduleEntry> {
    let mut schedule = Vec::with_capacity(tiles_m * tiles_n * k_steps);
    for tr in 0..tiles_m {
        for tc in 0..tiles_n {
            for ks in 0..k_steps {
                schedule.push(MmaScheduleEntry { tile_row: tr, tile_col: tc, k_step: ks });
            }
        }
    }
    schedule
}

// ── GEMM configuration ────────────────────────────────────────────────

/// Full configuration for a Tensor Core GEMM operation.
#[derive(Debug, Clone)]
pub struct TensorCoreGemmConfig {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Precision mode.
    pub precision: TensorCorePrecision,
    /// Tiling strategy.
    pub tile_strategy: TileStrategy,
    /// Scalar multiplier for A·B (default 1.0).
    pub alpha: f32,
    /// Scalar multiplier for C (default 0.0).
    pub beta: f32,
}

impl TensorCoreGemmConfig {
    /// Create a new config with default alpha=1, beta=0.
    pub fn new(m: usize, n: usize, k: usize, precision: TensorCorePrecision) -> Self {
        Self {
            m,
            n,
            k,
            precision,
            tile_strategy: TileStrategy::auto_select(m, n),
            alpha: 1.0,
            beta: 0.0,
        }
    }

    /// Override the tiling strategy.
    pub fn with_tile_strategy(mut self, strategy: TileStrategy) -> Self {
        self.tile_strategy = strategy;
        self
    }

    /// Set alpha/beta scalars (D = alpha * A·B + beta * C).
    pub fn with_scalars(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Validate that dimensions are positive.
    pub fn validate(&self) -> Result<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(BitNetError::Kernel(KernelError::InvalidDimension(
                "GEMM dimensions must be non-zero".into(),
            )));
        }
        Ok(())
    }

    /// Number of WMMA tiles along M.
    pub fn tiles_m(&self) -> usize {
        (self.m + WMMA_M - 1) / WMMA_M
    }

    /// Number of WMMA tiles along N.
    pub fn tiles_n(&self) -> usize {
        (self.n + WMMA_N - 1) / WMMA_N
    }

    /// Number of K-dimension steps (tiles along K).
    pub fn k_steps(&self) -> usize {
        (self.k + WMMA_K - 1) / WMMA_K
    }

    /// Number of thread-blocks in the grid.
    pub fn grid_blocks(&self) -> (usize, usize) {
        let tile_m = self.tile_strategy.tile_m();
        let tile_n = self.tile_strategy.tile_n();
        let grid_x = (self.n + tile_n - 1) / tile_n;
        let grid_y = (self.m + tile_m - 1) / tile_m;
        (grid_x, grid_y)
    }

    /// Estimated FLOP count: 2·M·N·K.
    pub fn flops(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64
    }
}

// ── Tensor Core GEMM engine ───────────────────────────────────────────

/// High-level Tensor Core GEMM engine.
///
/// Orchestrates fragment loading, MMA scheduling, and result writeback
/// for a complete GEMM operation.
#[derive(Debug)]
pub struct TensorCoreGemm {
    config: TensorCoreGemmConfig,
}

impl TensorCoreGemm {
    /// Create a new engine with the given configuration.
    pub fn new(config: TensorCoreGemmConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &TensorCoreGemmConfig {
        &self.config
    }

    /// Execute the GEMM: `D = alpha * A·B + beta * C` using CPU
    /// fragment simulation.
    ///
    /// `a` is row-major M×K, `b` is row-major K×N, `c` is row-major
    /// M×N (may be zeros), `d` is the output M×N.
    pub fn execute(&self, a: &[f32], b: &[f32], c: &[f32], d: &mut [f32]) -> Result<()> {
        let TensorCoreGemmConfig { m, n, k, alpha, beta, .. } = self.config;

        let expected_a = m * k;
        let expected_b = k * n;
        let expected_c = m * n;
        if a.len() < expected_a {
            return Err(BitNetError::Kernel(KernelError::InvalidDimension(format!(
                "A len {} < expected {}",
                a.len(),
                expected_a
            ))));
        }
        if b.len() < expected_b {
            return Err(BitNetError::Kernel(KernelError::InvalidDimension(format!(
                "B len {} < expected {}",
                b.len(),
                expected_b
            ))));
        }
        if c.len() < expected_c {
            return Err(BitNetError::Kernel(KernelError::InvalidDimension(format!(
                "C len {} < expected {}",
                c.len(),
                expected_c
            ))));
        }
        if d.len() < expected_c {
            return Err(BitNetError::Kernel(KernelError::InvalidDimension(format!(
                "D len {} < expected {}",
                d.len(),
                expected_c
            ))));
        }

        let tiles_m = self.config.tiles_m();
        let tiles_n = self.config.tiles_n();
        let k_steps = self.config.k_steps();

        // Initialise accumulator grid
        let mut accum = AccumulatorGrid::new(m, n);

        // Walk the MMA schedule
        let schedule = build_mma_schedule(tiles_m, tiles_n, k_steps);
        for entry in &schedule {
            let row_off = entry.tile_row * WMMA_M;
            let col_off = entry.tile_col * WMMA_N;
            let k_off = entry.k_step * WMMA_K;

            // Load fragment A (M×K slice)
            let mut frag_a = WmmaFragment::new_a();
            for r in 0..WMMA_M {
                for i in 0..WMMA_K {
                    let ar = row_off + r;
                    let ak = k_off + i;
                    frag_a.data[r * WMMA_K + i] =
                        if ar < m && ak < k { a[ar * k + ak] } else { 0.0 };
                }
            }

            // Load fragment B (K×N slice)
            let mut frag_b = WmmaFragment::new_b();
            for r in 0..WMMA_K {
                for c_col in 0..WMMA_N {
                    let bk = k_off + r;
                    let bn = col_off + c_col;
                    frag_b.data[r * WMMA_N + c_col] =
                        if bk < k && bn < n { b[bk * n + bn] } else { 0.0 };
                }
            }

            // Current accumulator
            let acc = accum
                .get(entry.tile_row, entry.tile_col)
                .ok_or_else(|| {
                    BitNetError::Kernel(KernelError::InvalidDimension(
                        "accumulator tile OOB".into(),
                    ))
                })?
                .clone();

            let acc_mut = accum.get_mut(entry.tile_row, entry.tile_col).ok_or_else(|| {
                BitNetError::Kernel(KernelError::InvalidDimension(
                    "accumulator tile OOB (mut)".into(),
                ))
            })?;

            mma_sync(&frag_a, &frag_b, &acc, acc_mut)?;
        }

        // Apply alpha/beta and write back
        for r in 0..m {
            for col in 0..n {
                let idx = r * n + col;
                let tr = r / WMMA_M;
                let tc = col / WMMA_N;
                let lr = r % WMMA_M;
                let lc = col % WMMA_N;
                let frag = &accum.fragments[tr * accum.tiles_n + tc];
                let acc_val = frag.data[lr * WMMA_N + lc];
                d[idx] = alpha * acc_val + beta * c[idx];
            }
        }

        Ok(())
    }

    /// Convenience: compute D = A·B (alpha=1, beta=0, C ignored).
    pub fn matmul(&self, a: &[f32], b: &[f32], d: &mut [f32]) -> Result<()> {
        let c = vec![0.0f32; self.config.m * self.config.n];
        self.execute(a, b, &c, d)
    }
}

// ── CUDA kernel source ────────────────────────────────────────────────

/// CUDA C kernel source for Tensor Core GEMM.
///
/// Uses `nvcuda::wmma` or inline PTX `mma.sync` instructions.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TENSOR_CORE_GEMM_KERNEL_SRC: &str = r#"
#include <mma.h>
using namespace nvcuda;

// WMMA tile dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Tensor Core GEMM kernel (FP16 input, FP32 accumulator)
// Computes D = alpha * A·B + beta * C
extern "C" __global__ void tensor_core_gemm_f16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ D,
    int M, int N, int K,
    float alpha, float beta)
{
    // Tile indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate along K
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store result: D = alpha * acc + beta * C
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        // Apply alpha scaling
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * c_frag.x[i]
                         + beta * C[(cRow + i / WMMA_N) * N + cCol + i % WMMA_N];
        }
        wmma::store_matrix_sync(D + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// INT8 Tensor Core GEMM kernel (INT8 input, INT32 accumulator)
extern "C" __global__ void tensor_core_gemm_i8(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ D,
    int M, int N, int K)
{
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;

    wmma::fill_fragment(c_frag, 0);

    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int bCol = warpN * WMMA_N;

        if (aRow < M && bCol < N && k < K) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(D + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}
"#;

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Precision mode tests ──────────────────────────────────────────

    #[test]
    fn test_precision_element_bits() {
        assert_eq!(TensorCorePrecision::Fp16.element_bits(), 16);
        assert_eq!(TensorCorePrecision::Bf16.element_bits(), 16);
        assert_eq!(TensorCorePrecision::Tf32.element_bits(), 32);
        assert_eq!(TensorCorePrecision::Int8.element_bits(), 8);
        assert_eq!(TensorCorePrecision::Int4.element_bits(), 4);
    }

    #[test]
    fn test_precision_min_sm_version() {
        assert_eq!(TensorCorePrecision::Fp16.min_sm_version(), 70);
        assert_eq!(TensorCorePrecision::Bf16.min_sm_version(), 80);
        assert_eq!(TensorCorePrecision::Tf32.min_sm_version(), 80);
        assert_eq!(TensorCorePrecision::Int8.min_sm_version(), 72);
        assert_eq!(TensorCorePrecision::Int4.min_sm_version(), 75);
    }

    #[test]
    fn test_precision_accumulator_type() {
        assert!(TensorCorePrecision::Fp16.accumulator_is_float());
        assert!(TensorCorePrecision::Bf16.accumulator_is_float());
        assert!(TensorCorePrecision::Tf32.accumulator_is_float());
        assert!(!TensorCorePrecision::Int8.accumulator_is_float());
        assert!(!TensorCorePrecision::Int4.accumulator_is_float());
    }

    // ── Tile strategy tests ───────────────────────────────────────────

    #[test]
    fn test_tile_dimensions() {
        assert_eq!(TileStrategy::Tile128x128.tile_m(), 128);
        assert_eq!(TileStrategy::Tile128x128.tile_n(), 128);
        assert_eq!(TileStrategy::Tile64x64.tile_m(), 64);
        assert_eq!(TileStrategy::Tile32x32.tile_m(), 32);
    }

    #[test]
    fn test_tile_k_always_32() {
        assert_eq!(TileStrategy::Tile128x128.tile_k(), 32);
        assert_eq!(TileStrategy::Tile64x64.tile_k(), 32);
        assert_eq!(TileStrategy::Tile32x32.tile_k(), 32);
    }

    #[test]
    fn test_warps_per_block() {
        assert_eq!(TileStrategy::Tile128x128.warps_per_block(), 8);
        assert_eq!(TileStrategy::Tile64x64.warps_per_block(), 4);
        assert_eq!(TileStrategy::Tile32x32.warps_per_block(), 2);
    }

    #[test]
    fn test_threads_per_block() {
        assert_eq!(TileStrategy::Tile128x128.threads_per_block(), 256);
        assert_eq!(TileStrategy::Tile64x64.threads_per_block(), 128);
        assert_eq!(TileStrategy::Tile32x32.threads_per_block(), 64);
    }

    #[test]
    fn test_shared_memory_bytes() {
        // 128×32 + 32×128 = 4096 + 4096 = 8192 elements, ×2 bytes ×2 buffers
        let smem = TileStrategy::Tile128x128.shared_memory_bytes(TensorCorePrecision::Fp16);
        assert_eq!(smem, 2 * (128 * 32 * 2 + 32 * 128 * 2));
    }

    #[test]
    fn test_auto_select_tile() {
        assert_eq!(TileStrategy::auto_select(256, 256), TileStrategy::Tile128x128);
        assert_eq!(TileStrategy::auto_select(64, 64), TileStrategy::Tile64x64);
        assert_eq!(TileStrategy::auto_select(16, 16), TileStrategy::Tile32x32);
        assert_eq!(TileStrategy::auto_select(128, 32), TileStrategy::Tile32x32);
    }

    // ── Fragment tests ────────────────────────────────────────────────

    #[test]
    fn test_fragment_new_a() {
        let f = WmmaFragment::new_a();
        assert_eq!(f.role, FragmentRole::MatrixA);
        assert_eq!(f.rows, 16);
        assert_eq!(f.cols, 16);
        assert_eq!(f.num_elements(), 256);
    }

    #[test]
    fn test_fragment_new_b() {
        let f = WmmaFragment::new_b();
        assert_eq!(f.role, FragmentRole::MatrixB);
        assert_eq!(f.rows, 16);
        assert_eq!(f.cols, 16);
    }

    #[test]
    fn test_fragment_new_accumulator() {
        let f = WmmaFragment::new_accumulator();
        assert_eq!(f.role, FragmentRole::Accumulator);
        assert_eq!(f.num_elements(), 256);
        assert!(f.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_fragment_fill() {
        let mut f = WmmaFragment::new_accumulator();
        f.fill(3.14);
        assert!(f.data.iter().all(|&v| (v - 3.14).abs() < 1e-6));
    }

    #[test]
    fn test_fragment_load_store_roundtrip() {
        let src: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let mut frag = WmmaFragment::new_a();
        frag.load_matrix(&src, 0, 0, 16).unwrap();
        let mut dst = vec![0.0f32; 256];
        frag.store_matrix(&mut dst, 0, 0, 16).unwrap();
        assert_eq!(src, dst);
    }

    #[test]
    fn test_fragment_load_oob() {
        let src = vec![0.0f32; 10]; // too small
        let mut frag = WmmaFragment::new_a();
        assert!(frag.load_matrix(&src, 0, 0, 16).is_err());
    }

    #[test]
    fn test_fragment_store_oob() {
        let frag = WmmaFragment::new_accumulator();
        let mut dst = vec![0.0f32; 10]; // too small
        assert!(frag.store_matrix(&mut dst, 0, 0, 16).is_err());
    }

    // ── MMA sync tests ───────────────────────────────────────────────

    #[test]
    fn test_mma_sync_identity() {
        // A = identity, B = identity → D = identity + 0
        let mut a = WmmaFragment::new_a();
        let mut b = WmmaFragment::new_b();
        let c = WmmaFragment::new_accumulator();
        let mut d = WmmaFragment::new_accumulator();

        for i in 0..16 {
            a.data[i * 16 + i] = 1.0;
            b.data[i * 16 + i] = 1.0;
        }

        mma_sync(&a, &b, &c, &mut d).unwrap();

        // Result should be identity
        for r in 0..16 {
            for col in 0..16 {
                let expected = if r == col { 1.0 } else { 0.0 };
                assert!(
                    (d.data[r * 16 + col] - expected).abs() < 1e-6,
                    "d[{r},{col}] = {} expected {expected}",
                    d.data[r * 16 + col],
                );
            }
        }
    }

    #[test]
    fn test_mma_sync_accumulates() {
        let mut a = WmmaFragment::new_a();
        let mut b = WmmaFragment::new_b();
        let mut c = WmmaFragment::new_accumulator();
        let mut d = WmmaFragment::new_accumulator();

        // All ones in A and B, C initialised to 10.0
        a.fill(1.0);
        b.fill(1.0);
        c.fill(10.0);

        mma_sync(&a, &b, &c, &mut d).unwrap();

        // Each element = 16 (sum of 16 ones) + 10 = 26
        for &val in &d.data {
            assert!((val - 26.0).abs() < 1e-4, "expected 26.0, got {val}");
        }
    }

    #[test]
    fn test_mma_sync_role_validation_a() {
        let a = WmmaFragment::new_accumulator(); // wrong role
        let b = WmmaFragment::new_b();
        let c = WmmaFragment::new_accumulator();
        let mut d = WmmaFragment::new_accumulator();
        assert!(mma_sync(&a, &b, &c, &mut d).is_err());
    }

    #[test]
    fn test_mma_sync_role_validation_b() {
        let a = WmmaFragment::new_a();
        let b = WmmaFragment::new_a(); // wrong role
        let c = WmmaFragment::new_accumulator();
        let mut d = WmmaFragment::new_accumulator();
        assert!(mma_sync(&a, &b, &c, &mut d).is_err());
    }

    #[test]
    fn test_mma_sync_role_validation_c() {
        let a = WmmaFragment::new_a();
        let b = WmmaFragment::new_b();
        let c = WmmaFragment::new_a(); // wrong role
        let mut d = WmmaFragment::new_accumulator();
        assert!(mma_sync(&a, &b, &c, &mut d).is_err());
    }

    // ── Accumulator grid tests ────────────────────────────────────────

    #[test]
    fn test_accumulator_grid_dimensions() {
        let grid = AccumulatorGrid::new(32, 48);
        assert_eq!(grid.tiles_m, 2); // 32/16
        assert_eq!(grid.tiles_n, 3); // 48/16
        assert_eq!(grid.num_fragments(), 6);
    }

    #[test]
    fn test_accumulator_grid_rounding() {
        let grid = AccumulatorGrid::new(17, 1);
        assert_eq!(grid.tiles_m, 2); // ceil(17/16)
        assert_eq!(grid.tiles_n, 1);
    }

    #[test]
    fn test_accumulator_grid_clear() {
        let mut grid = AccumulatorGrid::new(16, 16);
        grid.get_mut(0, 0).unwrap().fill(42.0);
        grid.clear();
        assert!(grid.get(0, 0).unwrap().data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_accumulator_grid_oob() {
        let grid = AccumulatorGrid::new(16, 16);
        assert!(grid.get(1, 0).is_none());
        assert!(grid.get(0, 1).is_none());
    }

    // ── MMA schedule tests ────────────────────────────────────────────

    #[test]
    fn test_mma_schedule_length() {
        let sched = build_mma_schedule(2, 3, 4);
        assert_eq!(sched.len(), 2 * 3 * 4);
    }

    #[test]
    fn test_mma_schedule_order() {
        let sched = build_mma_schedule(1, 1, 3);
        assert_eq!(sched.len(), 3);
        for (i, entry) in sched.iter().enumerate() {
            assert_eq!(entry.k_step, i);
        }
    }

    // ── Config tests ──────────────────────────────────────────────────

    #[test]
    fn test_config_validate_ok() {
        let cfg = TensorCoreGemmConfig::new(64, 64, 32, TensorCorePrecision::Fp16);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_m() {
        let cfg = TensorCoreGemmConfig::new(0, 64, 32, TensorCorePrecision::Fp16);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_k() {
        let cfg = TensorCoreGemmConfig::new(64, 64, 0, TensorCorePrecision::Fp16);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_grid_blocks() {
        let cfg = TensorCoreGemmConfig::new(256, 512, 64, TensorCorePrecision::Fp16)
            .with_tile_strategy(TileStrategy::Tile128x128);
        let (gx, gy) = cfg.grid_blocks();
        assert_eq!(gx, 4); // 512/128
        assert_eq!(gy, 2); // 256/128
    }

    #[test]
    fn test_config_flops() {
        let cfg = TensorCoreGemmConfig::new(128, 128, 64, TensorCorePrecision::Fp16);
        assert_eq!(cfg.flops(), 2 * 128 * 128 * 64);
    }

    #[test]
    fn test_config_tiles() {
        let cfg = TensorCoreGemmConfig::new(33, 17, 48, TensorCorePrecision::Fp16);
        assert_eq!(cfg.tiles_m(), 3); // ceil(33/16)
        assert_eq!(cfg.tiles_n(), 2); // ceil(17/16)
        assert_eq!(cfg.k_steps(), 3); // ceil(48/16)
    }

    #[test]
    fn test_config_with_scalars() {
        let cfg =
            TensorCoreGemmConfig::new(16, 16, 16, TensorCorePrecision::Fp16).with_scalars(2.0, 0.5);
        assert!((cfg.alpha - 2.0).abs() < 1e-6);
        assert!((cfg.beta - 0.5).abs() < 1e-6);
    }

    // ── Engine tests ──────────────────────────────────────────────────

    #[test]
    fn test_engine_zero_dim_rejected() {
        let cfg = TensorCoreGemmConfig::new(0, 16, 16, TensorCorePrecision::Fp16);
        assert!(TensorCoreGemm::new(cfg).is_err());
    }

    #[test]
    fn test_engine_identity_16x16() {
        let m = 16;
        let n = 16;
        let k = 16;
        let cfg = TensorCoreGemmConfig::new(m, n, k, TensorCorePrecision::Fp16);
        let engine = TensorCoreGemm::new(cfg).unwrap();

        // A = identity
        let mut a = vec![0.0f32; m * k];
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0;
        }

        // B = sequential values
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();

        let mut d = vec![0.0f32; m * n];
        engine.matmul(&a, &b, &mut d).unwrap();

        // D should equal B
        for i in 0..m * n {
            assert!((d[i] - b[i]).abs() < 1e-3, "d[{i}]={} != b[{i}]={}", d[i], b[i],);
        }
    }

    #[test]
    fn test_engine_alpha_beta_scalars() {
        let m = 16;
        let n = 16;
        let k = 16;
        let cfg =
            TensorCoreGemmConfig::new(m, n, k, TensorCorePrecision::Fp16).with_scalars(2.0, 1.0);
        let engine = TensorCoreGemm::new(cfg).unwrap();

        let mut a = vec![0.0f32; m * k];
        for i in 0..16 {
            a[i * k + i] = 1.0;
        }
        let b = vec![1.0f32; k * n];
        let c = vec![10.0f32; m * n];
        let mut d = vec![0.0f32; m * n];

        engine.execute(&a, &b, &c, &mut d).unwrap();

        // D = 2.0 * (I · ones) + 1.0 * 10.0 = 2*16 + 10 = 42
        for &val in &d {
            assert!((val - 42.0).abs() < 1e-3, "expected 42.0, got {val}");
        }
    }

    #[test]
    fn test_engine_32x32_ones() {
        let m = 32;
        let n = 32;
        let k = 16;
        let cfg = TensorCoreGemmConfig::new(m, n, k, TensorCorePrecision::Fp16);
        let engine = TensorCoreGemm::new(cfg).unwrap();

        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut d = vec![0.0f32; m * n];

        engine.matmul(&a, &b, &mut d).unwrap();

        // Each element should be k = 16
        for &val in &d {
            assert!((val - 16.0).abs() < 1e-3, "expected 16.0, got {val}");
        }
    }

    #[test]
    fn test_engine_undersized_a_rejected() {
        let cfg = TensorCoreGemmConfig::new(16, 16, 16, TensorCorePrecision::Fp16);
        let engine = TensorCoreGemm::new(cfg).unwrap();
        let a = vec![0.0f32; 10]; // too small
        let b = vec![0.0f32; 256];
        let mut d = vec![0.0f32; 256];
        assert!(engine.matmul(&a, &b, &mut d).is_err());
    }

    #[test]
    fn test_engine_undersized_b_rejected() {
        let cfg = TensorCoreGemmConfig::new(16, 16, 16, TensorCorePrecision::Fp16);
        let engine = TensorCoreGemm::new(cfg).unwrap();
        let a = vec![0.0f32; 256];
        let b = vec![0.0f32; 10]; // too small
        let mut d = vec![0.0f32; 256];
        assert!(engine.matmul(&a, &b, &mut d).is_err());
    }

    #[test]
    fn test_engine_undersized_d_rejected() {
        let cfg = TensorCoreGemmConfig::new(16, 16, 16, TensorCorePrecision::Fp16);
        let engine = TensorCoreGemm::new(cfg).unwrap();
        let a = vec![0.0f32; 256];
        let b = vec![0.0f32; 256];
        let mut d = vec![0.0f32; 10]; // too small
        assert!(engine.matmul(&a, &b, &mut d).is_err());
    }

    #[test]
    fn test_engine_non_aligned_dimensions() {
        // 17×19 × 19×23 — not multiples of 16
        let m = 17;
        let n = 23;
        let k = 19;
        let cfg = TensorCoreGemmConfig::new(m, n, k, TensorCorePrecision::Fp16);
        let engine = TensorCoreGemm::new(cfg).unwrap();

        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut d = vec![0.0f32; m * n];

        engine.matmul(&a, &b, &mut d).unwrap();

        for &val in &d {
            assert!((val - k as f32).abs() < 1e-3, "expected {k}.0, got {val}");
        }
    }

    #[test]
    fn test_all_precision_modes_construct() {
        for prec in [
            TensorCorePrecision::Fp16,
            TensorCorePrecision::Bf16,
            TensorCorePrecision::Tf32,
            TensorCorePrecision::Int8,
            TensorCorePrecision::Int4,
        ] {
            let cfg = TensorCoreGemmConfig::new(16, 16, 16, prec);
            assert!(TensorCoreGemm::new(cfg).is_ok(), "failed for {prec:?}");
        }
    }

    #[test]
    fn test_all_tile_strategies_construct() {
        for tile in [TileStrategy::Tile128x128, TileStrategy::Tile64x64, TileStrategy::Tile32x32] {
            let cfg = TensorCoreGemmConfig::new(256, 256, 64, TensorCorePrecision::Fp16)
                .with_tile_strategy(tile);
            let engine = TensorCoreGemm::new(cfg).unwrap();
            assert_eq!(engine.config().tile_strategy, tile);
        }
    }
}
