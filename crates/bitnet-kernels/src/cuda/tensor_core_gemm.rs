//! Tensor Core GEMM operations via WMMA (Warp Matrix Multiply Accumulate).
//!
//! # Overview
//!
//! This module provides CUDA Tensor Core-accelerated GEMM using WMMA
//! instructions available on Volta (SM 7.0+), Turing (SM 7.5), and
//! Ampere (SM 8.0+) architectures. It supports:
//!
//! - **WMMA operations**: INT1/INT2/INT4/INT8/FP16 input fragments
//! - **Tile configurations**: 16×16×16, 8×32×16, 32×8×16
//! - **Fragment management**: Load/store between shared memory and registers
//! - **FP32 accumulation**: Mixed-precision (FP16 inputs → FP32 output)
//! - **Multi-warp tile scheduling**: Cooperative tile decomposition
//! - **Mixed-input GEMM**: INT4 weights × FP16 activations
//! - **Batched GEMM**: Independent matrix products with batch dimension
//! - **Split-K GEMM**: K-dimension partitioning for tall-skinny matrices
//!
//! # CPU fallback
//!
//! Every entry point has a pure-Rust CPU fallback so that tests pass
//! without GPU hardware. The unified [`tensor_core_gemm`] dispatcher
//! tries the GPU path first and falls back transparently.
//!
//! # Feature gate
//!
//! GPU launch stubs and CUDA kernel sources are behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{KernelError, Result};

// ── Precision / data type enums ───────────────────────────────────────

/// Supported input data types for Tensor Core WMMA fragments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WmmaInputType {
    /// 1-bit binary (XOR-popcount on Turing+).
    INT1,
    /// 2-bit signed ternary ({-1, 0, +1}).
    INT2,
    /// 4-bit signed integer ([-8, +7]).
    INT4,
    /// 8-bit signed integer ([-128, +127]).
    INT8,
    /// IEEE 754 half-precision float.
    FP16,
}

impl WmmaInputType {
    /// Number of bits per element.
    pub fn bits(self) -> u32 {
        match self {
            Self::INT1 => 1,
            Self::INT2 => 2,
            Self::INT4 => 4,
            Self::INT8 => 8,
            Self::FP16 => 16,
        }
    }

    /// Byte size per element (minimum 1 for sub-byte types).
    pub fn byte_size(self) -> usize {
        match self {
            Self::INT1 | Self::INT2 | Self::INT4 => 1,
            Self::INT8 => 1,
            Self::FP16 => 2,
        }
    }
}

/// Accumulator precision for Tensor Core output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccumulatorType {
    /// 32-bit floating point (default, highest precision).
    FP32,
    /// 16-bit floating point (higher throughput, lower precision).
    FP16,
    /// 32-bit signed integer (for pure-integer WMMA paths).
    INT32,
}

// ── Tile configuration ────────────────────────────────────────────────

/// WMMA tile shape (M × N × K) supported by Tensor Cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WmmaTileShape {
    /// 16×16×16 — standard Volta/Ampere tile.
    Tile16x16x16,
    /// 8×32×16 — wide-N tile for projection layers.
    Tile8x32x16,
    /// 32×8×16 — tall-M tile for batch-heavy workloads.
    Tile32x8x16,
}

impl WmmaTileShape {
    /// Tile M dimension.
    pub fn m(self) -> u32 {
        match self {
            Self::Tile16x16x16 => 16,
            Self::Tile8x32x16 => 8,
            Self::Tile32x8x16 => 32,
        }
    }

    /// Tile N dimension.
    pub fn n(self) -> u32 {
        match self {
            Self::Tile16x16x16 => 16,
            Self::Tile8x32x16 => 32,
            Self::Tile32x8x16 => 8,
        }
    }

    /// Tile K dimension.
    pub fn k(self) -> u32 {
        match self {
            Self::Tile16x16x16 => 16,
            Self::Tile8x32x16 => 16,
            Self::Tile32x8x16 => 16,
        }
    }

    /// Number of output elements per WMMA tile.
    pub fn output_elements(self) -> u32 {
        self.m() * self.n()
    }
}

// ── Fragment types ────────────────────────────────────────────────────

/// Layout of a WMMA fragment in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentLayout {
    /// Row-major storage.
    RowMajor,
    /// Column-major storage.
    ColMajor,
}

/// A CPU-side representation of a WMMA fragment (register tile).
///
/// On GPU this maps to `nvcuda::wmma::fragment<>`. The CPU side stores
/// the data as f32 for simulation regardless of the logical data type.
#[derive(Debug, Clone)]
pub struct WmmaFragment {
    /// Fragment data stored as f32 for CPU simulation.
    pub data: Vec<f32>,
    /// Number of rows.
    pub rows: u32,
    /// Number of columns.
    pub cols: u32,
    /// Memory layout.
    pub layout: FragmentLayout,
    /// Logical input data type.
    pub dtype: WmmaInputType,
}

impl WmmaFragment {
    /// Create a zero-initialized fragment.
    pub fn zeros(rows: u32, cols: u32, layout: FragmentLayout, dtype: WmmaInputType) -> Self {
        Self { data: vec![0.0; (rows * cols) as usize], rows, cols, layout, dtype }
    }

    /// Number of elements in this fragment.
    pub fn num_elements(&self) -> usize {
        (self.rows * self.cols) as usize
    }

    /// Fill fragment with a constant value.
    pub fn fill(&mut self, val: f32) {
        self.data.fill(val);
    }
}

/// CPU-side representation of an accumulator fragment.
#[derive(Debug, Clone)]
pub struct AccumulatorFragment {
    /// Accumulator data (always f32 on CPU).
    pub data: Vec<f32>,
    /// Number of rows.
    pub rows: u32,
    /// Number of columns.
    pub cols: u32,
    /// Accumulator precision hint (used by GPU path).
    pub acc_type: AccumulatorType,
}

impl AccumulatorFragment {
    /// Create a zero-initialized accumulator.
    pub fn zeros(rows: u32, cols: u32, acc_type: AccumulatorType) -> Self {
        Self { data: vec![0.0; (rows * cols) as usize], rows, cols, acc_type }
    }

    /// Fill with a constant.
    pub fn fill(&mut self, val: f32) {
        self.data.fill(val);
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        (self.rows * self.cols) as usize
    }
}

// ── Tensor Core GEMM configuration ───────────────────────────────────

/// Configuration for a Tensor Core GEMM operation.
#[derive(Debug, Clone)]
pub struct TensorCoreGemmConfig {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
    /// WMMA tile shape.
    pub tile_shape: WmmaTileShape,
    /// Input A data type.
    pub input_a_type: WmmaInputType,
    /// Input B (weight) data type.
    pub input_b_type: WmmaInputType,
    /// Accumulator type.
    pub acc_type: AccumulatorType,
    /// Fragment A layout.
    pub layout_a: FragmentLayout,
    /// Fragment B layout.
    pub layout_b: FragmentLayout,
    /// Scalar multiplier α.
    pub alpha: f32,
    /// Scalar multiplier β (for C ← α·A·B + β·C).
    pub beta: f32,
    /// Batch count (1 for non-batched).
    pub batch_size: usize,
    /// Number of warps cooperating on one output tile.
    pub warps_per_tile: u32,
    /// Number of K-partitions for split-K (1 = disabled).
    pub split_k: u32,
    /// CUDA threads per block.
    pub threads_per_block: u32,
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
}

impl Default for TensorCoreGemmConfig {
    fn default() -> Self {
        Self {
            m: 1,
            n: 1,
            k: 1,
            tile_shape: WmmaTileShape::Tile16x16x16,
            input_a_type: WmmaInputType::FP16,
            input_b_type: WmmaInputType::FP16,
            acc_type: AccumulatorType::FP32,
            layout_a: FragmentLayout::RowMajor,
            layout_b: FragmentLayout::ColMajor,
            alpha: 1.0,
            beta: 0.0,
            batch_size: 1,
            warps_per_tile: 1,
            split_k: 1,
            threads_per_block: 256,
            shared_mem_bytes: 16384,
        }
    }
}

impl TensorCoreGemmConfig {
    /// Create a config for the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn for_shape(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tensor core GEMM dimensions must be non-zero: m={m}, n={n}, k={k}"
                ),
            }
            .into());
        }
        let tile = select_tile_shape(m, n);
        let shared = estimate_shared_mem_tc(tile);
        Ok(Self { m, n, k, tile_shape: tile, shared_mem_bytes: shared, ..Self::default() })
    }

    /// Set the WMMA tile shape.
    pub fn with_tile_shape(mut self, shape: WmmaTileShape) -> Self {
        self.tile_shape = shape;
        self.shared_mem_bytes = estimate_shared_mem_tc(shape);
        self
    }

    /// Set input types for A and B.
    pub fn with_input_types(mut self, a: WmmaInputType, b: WmmaInputType) -> Self {
        self.input_a_type = a;
        self.input_b_type = b;
        self
    }

    /// Set the accumulator type.
    pub fn with_accumulator(mut self, acc: AccumulatorType) -> Self {
        self.acc_type = acc;
        self
    }

    /// Set fragment layouts.
    pub fn with_layouts(mut self, a: FragmentLayout, b: FragmentLayout) -> Self {
        self.layout_a = a;
        self.layout_b = b;
        self
    }

    /// Set α and β scalars.
    pub fn with_alpha_beta(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set batch size.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` is zero.
    pub fn with_batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "batch_size must be > 0".into() }.into()
            );
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    /// Set the number of warps cooperating per output tile.
    ///
    /// # Errors
    ///
    /// Returns an error if `warps` is zero.
    pub fn with_warps_per_tile(mut self, warps: u32) -> Result<Self> {
        if warps == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "warps_per_tile must be > 0".into(),
            }
            .into());
        }
        self.warps_per_tile = warps;
        Ok(self)
    }

    /// Set split-K factor.
    ///
    /// # Errors
    ///
    /// Returns an error if `splits` is zero.
    pub fn with_split_k(mut self, splits: u32) -> Result<Self> {
        if splits == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "split_k must be > 0".into() }.into()
            );
        }
        self.split_k = splits;
        Ok(self)
    }

    /// Compute the CUDA grid dimensions.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let tm = self.tile_shape.m() as usize;
        let tn = self.tile_shape.n() as usize;
        let grid_x = (self.n.div_ceil(tn)) as u32;
        let grid_y = (self.m.div_ceil(tm)) as u32;
        let grid_z = self.batch_size as u32 * self.split_k;
        (grid_x, grid_y, grid_z)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }

    /// Number of WMMA tiles along M.
    pub fn tiles_m(&self) -> usize {
        self.m.div_ceil(self.tile_shape.m() as usize)
    }

    /// Number of WMMA tiles along N.
    pub fn tiles_n(&self) -> usize {
        self.n.div_ceil(self.tile_shape.n() as usize)
    }

    /// Number of WMMA tiles along K.
    pub fn tiles_k(&self) -> usize {
        self.k.div_ceil(self.tile_shape.k() as usize)
    }

    /// Total WMMA output tiles.
    pub fn total_output_tiles(&self) -> usize {
        self.tiles_m() * self.tiles_n()
    }

    /// Estimated GFLOPS at a given duration.
    pub fn gflops(&self, duration_secs: f64) -> f64 {
        let ops = 2.0 * self.m as f64 * self.n as f64 * self.k as f64 * self.batch_size as f64;
        ops / (duration_secs * 1e9)
    }
}

// ── Tile heuristics ───────────────────────────────────────────────────

fn select_tile_shape(m: usize, n: usize) -> WmmaTileShape {
    if n >= 4 * m {
        WmmaTileShape::Tile8x32x16
    } else if m >= 4 * n {
        WmmaTileShape::Tile32x8x16
    } else {
        WmmaTileShape::Tile16x16x16
    }
}

fn estimate_shared_mem_tc(tile: WmmaTileShape) -> u32 {
    // A tile (f32) + B tile (f32) + accumulator staging
    let a_bytes = tile.m() * tile.k() * 4;
    let b_bytes = tile.k() * tile.n() * 4;
    (a_bytes + b_bytes).max(4096) * 2 // double-buffer
}

// ── Fragment load / store (CPU simulation) ────────────────────────────

/// Load a WMMA A-fragment from a row-major matrix.
///
/// Loads a `tile_m × tile_k` sub-matrix starting at `(row_off, col_off)`
/// from a matrix of stride `ld`.
pub fn load_fragment_a(
    matrix: &[f32],
    ld: usize,
    row_off: usize,
    col_off: usize,
    tile: WmmaTileShape,
) -> Result<WmmaFragment> {
    let tm = tile.m() as usize;
    let tk = tile.k() as usize;
    let mut frag =
        WmmaFragment::zeros(tile.m(), tile.k(), FragmentLayout::RowMajor, WmmaInputType::FP16);

    for r in 0..tm {
        for c in 0..tk {
            let src_r = row_off + r;
            let src_c = col_off + c;
            let val = if src_r < matrix.len() / ld.max(1) && src_c < ld {
                matrix[src_r * ld + src_c]
            } else {
                0.0
            };
            frag.data[r * tk + c] = val;
        }
    }
    Ok(frag)
}

/// Load a WMMA B-fragment from a row-major matrix.
///
/// Loads a `tile_k × tile_n` sub-matrix starting at `(row_off, col_off)`
/// from a matrix of stride `ld`.
pub fn load_fragment_b(
    matrix: &[f32],
    ld: usize,
    row_off: usize,
    col_off: usize,
    tile: WmmaTileShape,
) -> Result<WmmaFragment> {
    let tk = tile.k() as usize;
    let tn = tile.n() as usize;
    let mut frag =
        WmmaFragment::zeros(tile.k(), tile.n(), FragmentLayout::ColMajor, WmmaInputType::FP16);

    for r in 0..tk {
        for c in 0..tn {
            let src_r = row_off + r;
            let src_c = col_off + c;
            let val = if src_r < matrix.len() / ld.max(1) && src_c < ld {
                matrix[src_r * ld + src_c]
            } else {
                0.0
            };
            frag.data[r * tn + c] = val;
        }
    }
    Ok(frag)
}

/// Store an accumulator fragment into a row-major output matrix.
pub fn store_accumulator(
    acc: &AccumulatorFragment,
    output: &mut [f32],
    ld: usize,
    row_off: usize,
    col_off: usize,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    let rows = acc.rows as usize;
    let cols = acc.cols as usize;

    for r in 0..rows {
        for c in 0..cols {
            let dst_r = row_off + r;
            let dst_c = col_off + c;
            if dst_r * ld + dst_c < output.len() && dst_c < ld {
                let idx = dst_r * ld + dst_c;
                let val = alpha * acc.data[r * cols + c];
                output[idx] = if beta == 0.0 { val } else { val + beta * output[idx] };
            }
        }
    }
    Ok(())
}

// ── WMMA MMA (CPU simulation) ─────────────────────────────────────────

/// Perform a WMMA matrix multiply-accumulate: `acc += A × B`.
///
/// On CPU this is a naive matmul of the fragment data. On GPU this
/// maps to `nvcuda::wmma::mma_sync()`.
pub fn wmma_mma(
    frag_a: &WmmaFragment,
    frag_b: &WmmaFragment,
    acc: &mut AccumulatorFragment,
) -> Result<()> {
    if frag_a.cols != frag_b.rows {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "WMMA dimension mismatch: A cols ({}) != B rows ({})",
                frag_a.cols, frag_b.rows
            ),
        }
        .into());
    }
    if acc.rows != frag_a.rows || acc.cols != frag_b.cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "accumulator shape ({},{}) doesn't match A rows ({}) × B cols ({})",
                acc.rows, acc.cols, frag_a.rows, frag_b.cols
            ),
        }
        .into());
    }

    let m = frag_a.rows as usize;
    let n = frag_b.cols as usize;
    let k = frag_a.cols as usize;

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += frag_a.data[i * k + l] * frag_b.data[l * n + j];
            }
            acc.data[i * n + j] += sum;
        }
    }
    Ok(())
}

// ── Tile scheduling ───────────────────────────────────────────────────

/// A tile work-item for multi-warp cooperation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileWorkItem {
    /// Tile index along M.
    pub tile_m: usize,
    /// Tile index along N.
    pub tile_n: usize,
    /// Assigned warp index.
    pub warp_id: u32,
    /// Batch index.
    pub batch_idx: usize,
}

/// Generate a tile schedule distributing output tiles to warps.
pub fn schedule_tiles(cfg: &TensorCoreGemmConfig) -> Vec<TileWorkItem> {
    let tm = cfg.tiles_m();
    let tn = cfg.tiles_n();
    let warps = cfg.warps_per_tile.max(1) as usize;
    let mut items = Vec::with_capacity(tm * tn * cfg.batch_size);

    for b in 0..cfg.batch_size {
        let mut warp_idx = 0u32;
        for ti in 0..tm {
            for tj in 0..tn {
                items.push(TileWorkItem {
                    tile_m: ti,
                    tile_n: tj,
                    warp_id: warp_idx % warps as u32,
                    batch_idx: b,
                });
                warp_idx += 1;
            }
        }
    }
    items
}

// ── Core GEMM CPU fallbacks ───────────────────────────────────────────

fn validate_gemm_buffers(
    a: &[f32],
    b: &[f32],
    out: &[f32],
    cfg: &TensorCoreGemmConfig,
) -> Result<()> {
    let batch = cfg.batch_size;
    let a_need = batch * cfg.m * cfg.k;
    let b_need = batch * cfg.k * cfg.n;
    let o_need = batch * cfg.m * cfg.n;

    if a.len() < a_need {
        return Err(KernelError::InvalidArguments {
            reason: format!("A buffer too small: need {a_need}, got {}", a.len()),
        }
        .into());
    }
    if b.len() < b_need {
        return Err(KernelError::InvalidArguments {
            reason: format!("B buffer too small: need {b_need}, got {}", b.len()),
        }
        .into());
    }
    if out.len() < o_need {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: need {o_need}, got {}", out.len()),
        }
        .into());
    }
    Ok(())
}

/// Tensor Core GEMM — CPU fallback using WMMA tile simulation.
///
/// Computes `C = α·A·B + β·C` using tile-by-tile WMMA simulation.
pub fn tensor_core_gemm_cpu(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    cfg: &TensorCoreGemmConfig,
) -> Result<()> {
    validate_gemm_buffers(a, b, out, cfg)?;

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let alpha = cfg.alpha;
    let beta = cfg.beta;
    let batch = cfg.batch_size;
    let tile = cfg.tile_shape;
    let tm = tile.m() as usize;
    let tn = tile.n() as usize;
    let tk = tile.k() as usize;

    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let o_off = bi * m * n;

        // Apply beta
        if beta == 0.0 {
            out[o_off..o_off + m * n].fill(0.0);
        } else if (beta - 1.0).abs() > f32::EPSILON {
            for v in out[o_off..o_off + m * n].iter_mut() {
                *v *= beta;
            }
        }

        // Tiled WMMA simulation
        for ti in (0..m).step_by(tm) {
            for tj in (0..n).step_by(tn) {
                let mut acc = AccumulatorFragment::zeros(tile.m(), tile.n(), cfg.acc_type);

                for tl in (0..k).step_by(tk) {
                    // Load A fragment
                    let frag_a = load_fragment_a(&a[a_off..], k, ti, tl, tile)?;
                    // Load B fragment
                    let frag_b = load_fragment_b(&b[b_off..], n, tl, tj, tile)?;
                    // MMA
                    wmma_mma(&frag_a, &frag_b, &mut acc)?;
                }

                // Store accumulator
                store_accumulator(&acc, &mut out[o_off..], n, ti, tj, alpha, beta)?;
            }
        }
    }
    Ok(())
}

/// FP16 Tensor Core GEMM — CPU fallback.
///
/// Input in FP16 (as `u16` IEEE 754 half), accumulates and outputs f32.
pub fn tensor_core_gemm_f16_cpu(
    a: &[u16],
    b: &[u16],
    out: &mut [f32],
    cfg: &TensorCoreGemmConfig,
) -> Result<()> {
    let batch = cfg.batch_size;
    let a_need = batch * cfg.m * cfg.k;
    let b_need = batch * cfg.k * cfg.n;
    let o_need = batch * cfg.m * cfg.n;

    if a.len() < a_need {
        return Err(KernelError::InvalidArguments {
            reason: format!("A buffer too small: need {a_need}, got {}", a.len()),
        }
        .into());
    }
    if b.len() < b_need {
        return Err(KernelError::InvalidArguments {
            reason: format!("B buffer too small: need {b_need}, got {}", b.len()),
        }
        .into());
    }
    if out.len() < o_need {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: need {o_need}, got {}", out.len()),
        }
        .into());
    }

    // Convert FP16 to f32 and delegate
    let a_f32: Vec<f32> = a[..a_need].iter().map(|&v| f16_to_f32(v)).collect();
    let b_f32: Vec<f32> = b[..b_need].iter().map(|&v| f16_to_f32(v)).collect();
    tensor_core_gemm_cpu(&a_f32, &b_f32, out, cfg)
}

/// Mixed-input GEMM: INT4 weights (i8 unpacked) × FP16 activations → FP32.
///
/// Simulates the Tensor Core mixed-precision path where INT4 weights
/// are widened to match FP16 activations inside the MMA unit.
pub fn mixed_input_gemm_cpu(
    activations_f16: &[u16],
    weights_i4: &[i8],
    out: &mut [f32],
    cfg: &TensorCoreGemmConfig,
) -> Result<()> {
    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let batch = cfg.batch_size;
    let alpha = cfg.alpha;
    let beta = cfg.beta;

    let a_need = batch * m * k;
    let w_need = batch * k * n;
    let o_need = batch * m * n;

    if activations_f16.len() < a_need {
        return Err(KernelError::InvalidArguments {
            reason: format!("activations too small: need {a_need}, got {}", activations_f16.len()),
        }
        .into());
    }
    if weights_i4.len() < w_need {
        return Err(KernelError::InvalidArguments {
            reason: format!("weights too small: need {w_need}, got {}", weights_i4.len()),
        }
        .into());
    }
    if out.len() < o_need {
        return Err(KernelError::InvalidArguments {
            reason: format!("output too small: need {o_need}, got {}", out.len()),
        }
        .into());
    }

    for bi in 0..batch {
        let a_off = bi * m * k;
        let w_off = bi * k * n;
        let o_off = bi * m * n;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    let a_val = f16_to_f32(activations_f16[a_off + i * k + l]);
                    let w_val = weights_i4[w_off + l * n + j] as f32;
                    acc += a_val * w_val;
                }
                let idx = o_off + i * n + j;
                let val = alpha * acc;
                out[idx] = if beta == 0.0 { val } else { val + beta * out[idx] };
            }
        }
    }
    Ok(())
}

/// Batched Tensor Core GEMM — CPU fallback.
///
/// Each batch element has independent A, B, C buffers laid out
/// contiguously. Uses tile-based WMMA simulation internally.
pub fn batched_tensor_core_gemm_cpu(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    cfg: &TensorCoreGemmConfig,
) -> Result<()> {
    // batched path is the same as tensor_core_gemm_cpu since it handles batch_size
    tensor_core_gemm_cpu(a, b, out, cfg)
}

/// Split-K Tensor Core GEMM — CPU fallback.
///
/// Partitions the K dimension into `split_k` slices, computes partial
/// sums independently, then reduces.
pub fn split_k_tensor_core_gemm_cpu(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    cfg: &TensorCoreGemmConfig,
) -> Result<()> {
    validate_gemm_buffers(a, b, out, cfg)?;

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let alpha = cfg.alpha;
    let beta = cfg.beta;
    let splits = cfg.split_k.max(1) as usize;
    let batch = cfg.batch_size;
    let k_per_split = k.div_ceil(splits);

    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let o_off = bi * m * n;

        // Partial sums: [splits][m * n]
        let mut partials = vec![0.0f32; splits * m * n];

        for s in 0..splits {
            let k_start = s * k_per_split;
            let k_end = ((s + 1) * k_per_split).min(k);
            if k_start >= k {
                break;
            }

            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for l in k_start..k_end {
                        let a_val = a[a_off + i * k + l];
                        let b_val = b[b_off + l * n + j];
                        acc += a_val * b_val;
                    }
                    partials[s * m * n + i * n + j] = acc;
                }
            }
        }

        // Reduce partials
        for i in 0..m {
            for j in 0..n {
                let mut total = 0.0f32;
                for s in 0..splits {
                    total += partials[s * m * n + i * n + j];
                }
                let idx = o_off + i * n + j;
                let val = alpha * total;
                out[idx] = if beta == 0.0 { val } else { val + beta * out[idx] };
            }
        }
    }
    Ok(())
}

/// Multi-warp cooperative GEMM — CPU fallback.
///
/// Distributes output tiles to warps using the tile scheduler.
/// On CPU this produces the same result as the single-warp path.
pub fn multi_warp_gemm_cpu(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    cfg: &TensorCoreGemmConfig,
) -> Result<()> {
    validate_gemm_buffers(a, b, out, cfg)?;

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let alpha = cfg.alpha;
    let beta = cfg.beta;
    let batch = cfg.batch_size;
    let tile = cfg.tile_shape;
    let tm = tile.m() as usize;
    let tn = tile.n() as usize;
    let tk = tile.k() as usize;

    // Zero or scale output
    for bi in 0..batch {
        let o_off = bi * m * n;
        if beta == 0.0 {
            out[o_off..o_off + m * n].fill(0.0);
        } else if (beta - 1.0).abs() > f32::EPSILON {
            for v in out[o_off..o_off + m * n].iter_mut() {
                *v *= beta;
            }
        }
    }

    let schedule = schedule_tiles(cfg);

    for item in &schedule {
        let bi = item.batch_idx;
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let o_off = bi * m * n;
        let row_start = item.tile_m * tm;
        let col_start = item.tile_n * tn;

        let mut acc = AccumulatorFragment::zeros(tile.m(), tile.n(), cfg.acc_type);

        for tl in (0..k).step_by(tk) {
            let frag_a = load_fragment_a(&a[a_off..], k, row_start, tl, tile)?;
            let frag_b = load_fragment_b(&b[b_off..], n, tl, col_start, tile)?;
            wmma_mma(&frag_a, &frag_b, &mut acc)?;
        }

        store_accumulator(&acc, &mut out[o_off..], n, row_start, col_start, alpha, beta)?;
    }
    Ok(())
}

// ── Benchmark helper ──────────────────────────────────────────────────

/// Run `iterations` CPU GEMM invocations and return average GFLOPS.
pub fn benchmark_tensor_core_gemm(cfg: &TensorCoreGemmConfig, iterations: usize) -> f64 {
    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let batch = cfg.batch_size;

    let a = vec![1.0f32; batch * m * k];
    let b = vec![1.0f32; batch * k * n];
    let mut out = vec![0.0f32; batch * m * n];

    let iters = iterations.max(1);
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = tensor_core_gemm_cpu(&a, &b, &mut out, cfg);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let avg_secs = elapsed / iters as f64;
    cfg.gflops(avg_secs)
}

// ── FP16 conversion helpers ───────────────────────────────────────────

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized
        let mut e = 0i32;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            e += 1;
        }
        let f32_exp = (127 - 15 - e) as u32;
        let f32_frac = (f & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac);
    }
    if exp == 0x1F {
        if frac == 0 {
            return f32::from_bits((sign << 31) | 0x7F800000);
        }
        return f32::from_bits((sign << 31) | 0x7FC00000);
    }

    let f32_exp = (exp as i32 - 15 + 127) as u32;
    let f32_frac = frac << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac)
}

#[cfg(test)]
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        if frac == 0 {
            return ((sign << 15) | 0x7C00) as u16;
        }
        return ((sign << 15) | 0x7E00) as u16;
    }

    let unbiased = exp - 127;
    if unbiased < -24 {
        return (sign << 15) as u16;
    }
    if unbiased < -14 {
        let shift = -1 - unbiased + 10;
        let f16_frac = ((frac | 0x800000) >> (shift + 13)) as u32;
        return ((sign << 15) | f16_frac) as u16;
    }
    if unbiased > 15 {
        return ((sign << 15) | 0x7C00) as u16;
    }

    let f16_exp = (unbiased + 15) as u32;
    let f16_frac = frac >> 13;
    ((sign << 15) | (f16_exp << 10) | f16_frac) as u16
}

// ── CUDA kernel source ────────────────────────────────────────────────

/// CUDA C kernel source implementing Tensor Core GEMM via WMMA.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TENSOR_CORE_GEMM_KERNEL_SRC: &str = r#"
#include <mma.h>
using namespace nvcuda;

// FP16 Tensor Core GEMM — one warp computes one 16×16 output tile.
extern "C" __global__ void tensor_core_gemm_f16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = blockIdx.x;

    if (warpM * 16 >= M || warpN * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        int aRow = warpM * 16;
        int aCol = k_tile;
        int bRow = k_tile;
        int bCol = warpN * 16;

        if (aRow < M && aCol + 16 <= K && bRow + 16 <= K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * 16;
    int cCol = warpN * 16;
    if (cRow < M && cCol < N) {
        // Apply alpha/beta
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * c_frag.x[i] + beta * C[(cRow + i/16) * N + cCol + i%16];
        }
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// INT8 Tensor Core GEMM (Turing+).
extern "C" __global__ void tensor_core_gemm_int8(
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    int*               __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = blockIdx.x;

    if (warpM * 16 >= M || warpN * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;

    wmma::fill_fragment(c_frag, 0);

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        int aRow = warpM * 16;
        int aCol = k_tile;
        int bRow = k_tile;
        int bCol = warpN * 16;

        if (aRow < M && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * 16;
    int cCol = warpN * 16;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// Split-K Tensor Core GEMM: each z-slice handles a K partition.
extern "C" __global__ void tensor_core_gemm_split_k(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ partial,
    int M, int N, int K, int split_k)
{
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = blockIdx.x;
    int split = blockIdx.z;

    if (warpM * 16 >= M || warpN * 16 >= N || split >= split_k) return;

    int k_per_split = (K + split_k - 1) / split_k;
    int k_start = split * k_per_split;
    int k_end = min(k_start + k_per_split, K);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
        int aRow = warpM * 16;
        int bCol = warpN * 16;
        wmma::load_matrix_sync(a_frag, A + aRow * K + k_tile, K);
        wmma::load_matrix_sync(b_frag, B + k_tile * N + bCol, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int cRow = warpM * 16;
    int cCol = warpN * 16;
    int offset = split * M * N;
    wmma::store_matrix_sync(partial + offset + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}
"#;

/// Launch stub for the Tensor Core GEMM CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_tensor_core_gemm(
    _a: &[f32],
    _b: &[f32],
    _output: &mut [f32],
    config: &TensorCoreGemmConfig,
) -> Result<()> {
    log::debug!(
        "tensor core GEMM CUDA stub: m={}, n={}, k={}, tile={:?}, grid={:?}",
        config.m,
        config.n,
        config.k,
        config.tile_shape,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "tensor core GEMM CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Unified dispatch: GPU-first with CPU fallback.
pub fn tensor_core_gemm(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    config: &TensorCoreGemmConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && launch_tensor_core_gemm(a, b, output, config).is_ok()
        {
            return Ok(());
        }
    }
    tensor_core_gemm_cpu(a, b, output, config)
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

    /// Naive reference matmul: C = A × B (row-major).
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

    // ── WmmaInputType tests ──────────────────────────────────────

    #[test]
    fn test_input_type_bits() {
        assert_eq!(WmmaInputType::INT1.bits(), 1);
        assert_eq!(WmmaInputType::INT2.bits(), 2);
        assert_eq!(WmmaInputType::INT4.bits(), 4);
        assert_eq!(WmmaInputType::INT8.bits(), 8);
        assert_eq!(WmmaInputType::FP16.bits(), 16);
    }

    #[test]
    fn test_input_type_byte_size() {
        assert_eq!(WmmaInputType::INT1.byte_size(), 1);
        assert_eq!(WmmaInputType::INT4.byte_size(), 1);
        assert_eq!(WmmaInputType::INT8.byte_size(), 1);
        assert_eq!(WmmaInputType::FP16.byte_size(), 2);
    }

    // ── WmmaTileShape tests ──────────────────────────────────────

    #[test]
    fn test_tile_16x16x16_dimensions() {
        let t = WmmaTileShape::Tile16x16x16;
        assert_eq!(t.m(), 16);
        assert_eq!(t.n(), 16);
        assert_eq!(t.k(), 16);
        assert_eq!(t.output_elements(), 256);
    }

    #[test]
    fn test_tile_8x32x16_dimensions() {
        let t = WmmaTileShape::Tile8x32x16;
        assert_eq!(t.m(), 8);
        assert_eq!(t.n(), 32);
        assert_eq!(t.k(), 16);
        assert_eq!(t.output_elements(), 256);
    }

    #[test]
    fn test_tile_32x8x16_dimensions() {
        let t = WmmaTileShape::Tile32x8x16;
        assert_eq!(t.m(), 32);
        assert_eq!(t.n(), 8);
        assert_eq!(t.k(), 16);
        assert_eq!(t.output_elements(), 256);
    }

    // ── Fragment tests ───────────────────────────────────────────

    #[test]
    fn test_fragment_zeros() {
        let f = WmmaFragment::zeros(16, 16, FragmentLayout::RowMajor, WmmaInputType::FP16);
        assert_eq!(f.num_elements(), 256);
        assert!(f.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_fragment_fill() {
        let mut f = WmmaFragment::zeros(8, 32, FragmentLayout::RowMajor, WmmaInputType::INT8);
        f.fill(3.0);
        assert!(f.data.iter().all(|&v| v == 3.0));
    }

    #[test]
    fn test_accumulator_zeros() {
        let acc = AccumulatorFragment::zeros(16, 16, AccumulatorType::FP32);
        assert_eq!(acc.num_elements(), 256);
        assert!(acc.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_accumulator_fill() {
        let mut acc = AccumulatorFragment::zeros(16, 16, AccumulatorType::FP32);
        acc.fill(2.5);
        assert!(acc.data.iter().all(|&v| v == 2.5));
    }

    #[test]
    fn test_accumulator_int32_type() {
        let acc = AccumulatorFragment::zeros(16, 16, AccumulatorType::INT32);
        assert_eq!(acc.acc_type, AccumulatorType::INT32);
    }

    #[test]
    fn test_accumulator_fp16_type() {
        let acc = AccumulatorFragment::zeros(16, 16, AccumulatorType::FP16);
        assert_eq!(acc.acc_type, AccumulatorType::FP16);
    }

    // ── TensorCoreGemmConfig tests ───────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = TensorCoreGemmConfig::default();
        assert_eq!(cfg.tile_shape, WmmaTileShape::Tile16x16x16);
        assert_eq!(cfg.input_a_type, WmmaInputType::FP16);
        assert_eq!(cfg.acc_type, AccumulatorType::FP32);
        assert_eq!(cfg.batch_size, 1);
        assert_eq!(cfg.split_k, 1);
        assert_eq!(cfg.alpha, 1.0);
        assert_eq!(cfg.beta, 0.0);
    }

    #[test]
    fn test_config_for_shape() {
        let cfg = TensorCoreGemmConfig::for_shape(32, 64, 16).unwrap();
        assert_eq!(cfg.m, 32);
        assert_eq!(cfg.n, 64);
        assert_eq!(cfg.k, 16);
    }

    #[test]
    fn test_config_rejects_zero_m() {
        assert!(TensorCoreGemmConfig::for_shape(0, 8, 8).is_err());
    }

    #[test]
    fn test_config_rejects_zero_n() {
        assert!(TensorCoreGemmConfig::for_shape(8, 0, 8).is_err());
    }

    #[test]
    fn test_config_rejects_zero_k() {
        assert!(TensorCoreGemmConfig::for_shape(8, 8, 0).is_err());
    }

    #[test]
    fn test_config_with_tile_shape() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let cfg = cfg.with_tile_shape(WmmaTileShape::Tile8x32x16);
        assert_eq!(cfg.tile_shape, WmmaTileShape::Tile8x32x16);
    }

    #[test]
    fn test_config_with_input_types() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let cfg = cfg.with_input_types(WmmaInputType::INT8, WmmaInputType::INT4);
        assert_eq!(cfg.input_a_type, WmmaInputType::INT8);
        assert_eq!(cfg.input_b_type, WmmaInputType::INT4);
    }

    #[test]
    fn test_config_with_accumulator() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let cfg = cfg.with_accumulator(AccumulatorType::INT32);
        assert_eq!(cfg.acc_type, AccumulatorType::INT32);
    }

    #[test]
    fn test_config_with_layouts() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let cfg = cfg.with_layouts(FragmentLayout::ColMajor, FragmentLayout::RowMajor);
        assert_eq!(cfg.layout_a, FragmentLayout::ColMajor);
        assert_eq!(cfg.layout_b, FragmentLayout::RowMajor);
    }

    #[test]
    fn test_config_with_alpha_beta() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let cfg = cfg.with_alpha_beta(2.0, 0.5);
        assert_eq!(cfg.alpha, 2.0);
        assert_eq!(cfg.beta, 0.5);
    }

    #[test]
    fn test_config_with_batch_size() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let cfg = cfg.with_batch_size(4).unwrap();
        assert_eq!(cfg.batch_size, 4);
    }

    #[test]
    fn test_config_rejects_zero_batch() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        assert!(cfg.with_batch_size(0).is_err());
    }

    #[test]
    fn test_config_with_warps_per_tile() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let cfg = cfg.with_warps_per_tile(4).unwrap();
        assert_eq!(cfg.warps_per_tile, 4);
    }

    #[test]
    fn test_config_rejects_zero_warps() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        assert!(cfg.with_warps_per_tile(0).is_err());
    }

    #[test]
    fn test_config_with_split_k() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 64).unwrap();
        let cfg = cfg.with_split_k(4).unwrap();
        assert_eq!(cfg.split_k, 4);
    }

    #[test]
    fn test_config_rejects_zero_split_k() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        assert!(cfg.with_split_k(0).is_err());
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = TensorCoreGemmConfig::for_shape(32, 48, 16).unwrap();
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gy, 2); // 32 / 16
        assert_eq!(gx, 3); // 48 / 16
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_config_grid_dim_with_batch_and_split_k() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 64)
            .unwrap()
            .with_batch_size(2)
            .unwrap()
            .with_split_k(4)
            .unwrap();
        let (_, _, gz) = cfg.grid_dim();
        assert_eq!(gz, 8); // 2 * 4
    }

    #[test]
    fn test_config_tiles_count() {
        let cfg = TensorCoreGemmConfig::for_shape(48, 64, 32).unwrap();
        assert_eq!(cfg.tiles_m(), 3); // 48 / 16
        assert_eq!(cfg.tiles_n(), 4); // 64 / 16
        assert_eq!(cfg.tiles_k(), 2); // 32 / 16
        assert_eq!(cfg.total_output_tiles(), 12);
    }

    #[test]
    fn test_config_gflops() {
        let cfg = TensorCoreGemmConfig::for_shape(1024, 1024, 1024).unwrap();
        let gflops = cfg.gflops(1.0);
        let expected = 2.0 * 1024.0 * 1024.0 * 1024.0 / 1e9;
        assert!((gflops - expected).abs() < 1e-6);
    }

    // ── Tile selection heuristics ────────────────────────────────

    #[test]
    fn test_tile_selection_square() {
        let cfg = TensorCoreGemmConfig::for_shape(64, 64, 16).unwrap();
        assert_eq!(cfg.tile_shape, WmmaTileShape::Tile16x16x16);
    }

    #[test]
    fn test_tile_selection_wide_n() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 128, 16).unwrap();
        assert_eq!(cfg.tile_shape, WmmaTileShape::Tile8x32x16);
    }

    #[test]
    fn test_tile_selection_tall_m() {
        let cfg = TensorCoreGemmConfig::for_shape(128, 16, 16).unwrap();
        assert_eq!(cfg.tile_shape, WmmaTileShape::Tile32x8x16);
    }

    // ── Fragment load / store tests ──────────────────────────────

    #[test]
    fn test_load_fragment_a_identity() {
        let tile = WmmaTileShape::Tile16x16x16;
        let mut a = vec![0.0f32; 16 * 16];
        for i in 0..16 {
            a[i * 16 + i] = 1.0;
        }
        let frag = load_fragment_a(&a, 16, 0, 0, tile).unwrap();
        for i in 0..16 {
            for j in 0..16 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(frag.data[i * 16 + j], expected);
            }
        }
    }

    #[test]
    fn test_load_fragment_b_constant() {
        let tile = WmmaTileShape::Tile16x16x16;
        let b = vec![2.0f32; 16 * 16];
        let frag = load_fragment_b(&b, 16, 0, 0, tile).unwrap();
        assert!(frag.data.iter().all(|&v| v == 2.0));
    }

    #[test]
    fn test_load_fragment_a_with_offset() {
        let tile = WmmaTileShape::Tile16x16x16;
        let mut a = vec![0.0f32; 32 * 32];
        a[16 * 32 + 16] = 42.0; // row 16, col 16
        let frag = load_fragment_a(&a, 32, 16, 16, tile).unwrap();
        assert_eq!(frag.data[0], 42.0);
    }

    #[test]
    fn test_load_fragment_b_with_offset() {
        let tile = WmmaTileShape::Tile16x16x16;
        let mut b = vec![0.0f32; 32 * 32];
        b[16 * 32 + 16] = 99.0;
        let frag = load_fragment_b(&b, 32, 16, 16, tile).unwrap();
        assert_eq!(frag.data[0], 99.0);
    }

    #[test]
    fn test_store_accumulator_basic() {
        let mut acc = AccumulatorFragment::zeros(16, 16, AccumulatorType::FP32);
        acc.data[0] = 5.0;
        let mut out = vec![0.0f32; 16 * 16];
        store_accumulator(&acc, &mut out, 16, 0, 0, 1.0, 0.0).unwrap();
        assert_eq!(out[0], 5.0);
    }

    #[test]
    fn test_store_accumulator_with_alpha() {
        let mut acc = AccumulatorFragment::zeros(16, 16, AccumulatorType::FP32);
        acc.data[0] = 5.0;
        let mut out = vec![0.0f32; 16 * 16];
        store_accumulator(&acc, &mut out, 16, 0, 0, 2.0, 0.0).unwrap();
        assert_eq!(out[0], 10.0);
    }

    #[test]
    fn test_store_accumulator_with_beta() {
        let mut acc = AccumulatorFragment::zeros(16, 16, AccumulatorType::FP32);
        acc.data[0] = 5.0;
        let mut out = vec![10.0f32; 16 * 16];
        store_accumulator(&acc, &mut out, 16, 0, 0, 1.0, 0.5).unwrap();
        assert_eq!(out[0], 10.0); // 1.0*5.0 + 0.5*10.0 = 10.0
    }

    // ── WMMA MMA tests ───────────────────────────────────────────

    #[test]
    fn test_wmma_mma_identity() {
        let tile = WmmaTileShape::Tile16x16x16;
        let mut a_data = vec![0.0f32; 16 * 16];
        for i in 0..16 {
            a_data[i * 16 + i] = 1.0;
        }
        let frag_a = WmmaFragment {
            data: a_data,
            rows: 16,
            cols: 16,
            layout: FragmentLayout::RowMajor,
            dtype: WmmaInputType::FP16,
        };
        let frag_b = WmmaFragment {
            data: vec![3.0; 16 * 16],
            rows: 16,
            cols: 16,
            layout: FragmentLayout::ColMajor,
            dtype: WmmaInputType::FP16,
        };
        let mut acc = AccumulatorFragment::zeros(tile.m(), tile.n(), AccumulatorType::FP32);
        wmma_mma(&frag_a, &frag_b, &mut acc).unwrap();
        // I × 3 = 3
        assert_close(&acc.data, &vec![3.0f32; 256], 1e-5);
    }

    #[test]
    fn test_wmma_mma_dimension_mismatch_a_b() {
        let frag_a = WmmaFragment::zeros(16, 8, FragmentLayout::RowMajor, WmmaInputType::FP16);
        let frag_b = WmmaFragment::zeros(16, 16, FragmentLayout::ColMajor, WmmaInputType::FP16);
        let mut acc = AccumulatorFragment::zeros(16, 16, AccumulatorType::FP32);
        assert!(wmma_mma(&frag_a, &frag_b, &mut acc).is_err());
    }

    #[test]
    fn test_wmma_mma_dimension_mismatch_acc() {
        let frag_a = WmmaFragment::zeros(16, 16, FragmentLayout::RowMajor, WmmaInputType::FP16);
        let frag_b = WmmaFragment::zeros(16, 16, FragmentLayout::ColMajor, WmmaInputType::FP16);
        let mut acc = AccumulatorFragment::zeros(8, 8, AccumulatorType::FP32);
        assert!(wmma_mma(&frag_a, &frag_b, &mut acc).is_err());
    }

    #[test]
    fn test_wmma_mma_accumulate() {
        let frag_a = WmmaFragment {
            data: vec![1.0; 16 * 16],
            rows: 16,
            cols: 16,
            layout: FragmentLayout::RowMajor,
            dtype: WmmaInputType::FP16,
        };
        let frag_b = WmmaFragment {
            data: vec![1.0; 16 * 16],
            rows: 16,
            cols: 16,
            layout: FragmentLayout::ColMajor,
            dtype: WmmaInputType::FP16,
        };
        let mut acc = AccumulatorFragment::zeros(16, 16, AccumulatorType::FP32);
        // First MMA
        wmma_mma(&frag_a, &frag_b, &mut acc).unwrap();
        // Second MMA — accumulates
        wmma_mma(&frag_a, &frag_b, &mut acc).unwrap();
        // Each element = 2 * (1*16) = 32
        assert_close(&acc.data, &vec![32.0f32; 256], 1e-5);
    }

    // ── Tile scheduling tests ────────────────────────────────────

    #[test]
    fn test_schedule_tiles_single_tile() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let items = schedule_tiles(&cfg);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].tile_m, 0);
        assert_eq!(items[0].tile_n, 0);
    }

    #[test]
    fn test_schedule_tiles_multi_tile() {
        let cfg = TensorCoreGemmConfig::for_shape(32, 48, 16).unwrap();
        let items = schedule_tiles(&cfg);
        assert_eq!(items.len(), 6); // 2 × 3
    }

    #[test]
    fn test_schedule_tiles_warp_assignment() {
        let cfg =
            TensorCoreGemmConfig::for_shape(32, 32, 16).unwrap().with_warps_per_tile(2).unwrap();
        let items = schedule_tiles(&cfg);
        assert_eq!(items.len(), 4); // 2 × 2
        // Warps cycle: 0, 1, 0, 1
        assert_eq!(items[0].warp_id, 0);
        assert_eq!(items[1].warp_id, 1);
        assert_eq!(items[2].warp_id, 0);
        assert_eq!(items[3].warp_id, 1);
    }

    #[test]
    fn test_schedule_tiles_batched() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap().with_batch_size(3).unwrap();
        let items = schedule_tiles(&cfg);
        assert_eq!(items.len(), 3); // 1 tile × 3 batches
        assert_eq!(items[0].batch_idx, 0);
        assert_eq!(items[1].batch_idx, 1);
        assert_eq!(items[2].batch_idx, 2);
    }

    // ── GEMM correctness tests ───────────────────────────────────

    #[test]
    fn test_gemm_identity_16x16() {
        let m = 16;
        let n = 16;
        let k = 16;
        let mut a = vec![0.0f32; m * k];
        for i in 0..m {
            a[i * k + i] = 1.0;
        }
        let b = vec![2.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        let expected = naive_matmul(&a, &b, m, n, k);
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_gemm_ones_16x16() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        // Each element = 16.0
        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-5));
    }

    #[test]
    fn test_gemm_32x32x32() {
        let m = 32;
        let n = 32;
        let k = 32;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.2).collect();
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        let expected = naive_matmul(&a, &b, m, n, k);
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_gemm_non_tile_aligned() {
        let m = 20;
        let n = 24;
        let k = 18;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        let expected = naive_matmul(&a, &b, m, n, k);
        assert_close(&out, &expected, 1e-2);
    }

    #[test]
    fn test_gemm_alpha_scaling() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_alpha_beta(2.0, 0.0);
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        // Each element = 2.0 * 16.0 = 32.0
        assert!(out.iter().all(|&v| (v - 32.0).abs() < 1e-5));
    }

    #[test]
    fn test_gemm_beta_accumulate() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![10.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_alpha_beta(1.0, 1.0);
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        // Each element = 1.0*16.0 + 1.0*10.0 = 26.0
        assert!(out.iter().all(|&v| (v - 26.0).abs() < 1e-5));
    }

    #[test]
    fn test_gemm_buffer_too_small_a() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let a = [0.0f32; 10]; // too small
        let b = [0.0f32; 256];
        let mut out = [0.0f32; 256];
        assert!(tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_gemm_buffer_too_small_b() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let a = [0.0f32; 256];
        let b = [0.0f32; 10]; // too small
        let mut out = [0.0f32; 256];
        assert!(tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_gemm_buffer_too_small_out() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let a = [0.0f32; 256];
        let b = [0.0f32; 256];
        let mut out = [0.0f32; 10]; // too small
        assert!(tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).is_err());
    }

    // ── Tile shape variant GEMM tests ────────────────────────────

    #[test]
    fn test_gemm_tile_8x32x16() {
        let m = 16;
        let n = 64;
        let k = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k)
            .unwrap()
            .with_tile_shape(WmmaTileShape::Tile8x32x16);
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        let expected = naive_matmul(&a, &b, m, n, k);
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_gemm_tile_32x8x16() {
        let m = 64;
        let n = 16;
        let k = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k)
            .unwrap()
            .with_tile_shape(WmmaTileShape::Tile32x8x16);
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        let expected = naive_matmul(&a, &b, m, n, k);
        assert_close(&out, &expected, 1e-5);
    }

    // ── FP16 GEMM tests ─────────────────────────────────────────

    #[test]
    fn test_f16_gemm_basic() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a_f16: Vec<u16> = vec![f32_to_f16(1.0); m * k];
        let b_f16: Vec<u16> = vec![f32_to_f16(1.0); k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_f16_cpu(&a_f16, &b_f16, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 16.0).abs() < 0.1));
    }

    #[test]
    fn test_f16_gemm_mixed_values() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a_f16: Vec<u16> = (0..m * k).map(|i| f32_to_f16((i % 4) as f32)).collect();
        let b_f16: Vec<u16> = vec![f32_to_f16(1.0); k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_f16_cpu(&a_f16, &b_f16, &mut out, &cfg).unwrap();

        // Verify non-zero results
        assert!(out.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_f16_gemm_buffer_too_small() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let a = [0u16; 10];
        let b = [0u16; 256];
        let mut out = [0.0f32; 256];
        assert!(tensor_core_gemm_f16_cpu(&a, &b, &mut out, &cfg).is_err());
    }

    // ── Mixed-input GEMM tests ───────────────────────────────────

    #[test]
    fn test_mixed_input_gemm_basic() {
        let m = 4;
        let n = 4;
        let k = 4;
        let act: Vec<u16> = vec![f32_to_f16(1.0); m * k];
        let wt: Vec<i8> = vec![2; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k)
            .unwrap()
            .with_input_types(WmmaInputType::FP16, WmmaInputType::INT4);
        mixed_input_gemm_cpu(&act, &wt, &mut out, &cfg).unwrap();

        // Each element = 1.0 * 2 * 4 = 8.0
        assert!(out.iter().all(|&v| (v - 8.0).abs() < 0.1));
    }

    #[test]
    fn test_mixed_input_gemm_negative_weights() {
        let m = 4;
        let n = 4;
        let k = 4;
        let act: Vec<u16> = vec![f32_to_f16(1.0); m * k];
        let wt: Vec<i8> = vec![-1; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        mixed_input_gemm_cpu(&act, &wt, &mut out, &cfg).unwrap();

        // Each element = 1.0 * (-1) * 4 = -4.0
        assert!(out.iter().all(|&v| (v - (-4.0)).abs() < 0.1));
    }

    #[test]
    fn test_mixed_input_gemm_ternary_weights() {
        let m = 4;
        let n = 4;
        let k = 4;
        let act: Vec<u16> = vec![f32_to_f16(2.0); m * k];
        // Ternary: {-1, 0, +1, 0} repeated
        let wt: Vec<i8> = (0..k * n).map(|i| [-1i8, 0, 1, 0][i % 4]).collect();
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        mixed_input_gemm_cpu(&act, &wt, &mut out, &cfg).unwrap();

        // Verify non-trivial results
        assert!(out.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_mixed_input_gemm_with_alpha() {
        let m = 4;
        let n = 4;
        let k = 4;
        let act: Vec<u16> = vec![f32_to_f16(1.0); m * k];
        let wt: Vec<i8> = vec![1; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_alpha_beta(3.0, 0.0);
        mixed_input_gemm_cpu(&act, &wt, &mut out, &cfg).unwrap();

        // Each element = 3.0 * (1.0 * 1 * 4) = 12.0
        assert!(out.iter().all(|&v| (v - 12.0).abs() < 0.1));
    }

    #[test]
    fn test_mixed_input_gemm_buffer_too_small() {
        let cfg = TensorCoreGemmConfig::for_shape(4, 4, 4).unwrap();
        let act = [0u16; 4]; // too small
        let wt = [0i8; 16];
        let mut out = [0.0f32; 16];
        assert!(mixed_input_gemm_cpu(&act, &wt, &mut out, &cfg).is_err());
    }

    // ── Batched GEMM tests ───────────────────────────────────────

    #[test]
    fn test_batched_gemm_basic() {
        let m = 16;
        let n = 16;
        let k = 16;
        let batch = 2;
        let a = vec![1.0f32; batch * m * k];
        let b = vec![1.0f32; batch * k * n];
        let mut out = vec![0.0f32; batch * m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_batch_size(batch).unwrap();
        batched_tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        // Each element = 16.0 in both batches
        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-5));
    }

    #[test]
    fn test_batched_gemm_different_batches() {
        let m = 16;
        let n = 16;
        let k = 16;
        let batch = 2;
        let mut a = vec![1.0f32; batch * m * k];
        // Batch 1 has 2.0
        for v in a[m * k..].iter_mut() {
            *v = 2.0;
        }
        let b = vec![1.0f32; batch * k * n];
        let mut out = vec![0.0f32; batch * m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_batch_size(batch).unwrap();
        batched_tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        // Batch 0: 16.0, Batch 1: 32.0
        assert!(out[..m * n].iter().all(|&v| (v - 16.0).abs() < 1e-5));
        assert!(out[m * n..].iter().all(|&v| (v - 32.0).abs() < 1e-5));
    }

    #[test]
    fn test_batched_gemm_three_batches() {
        let m = 16;
        let n = 16;
        let k = 16;
        let batch = 3;
        let a = vec![1.0f32; batch * m * k];
        let b = vec![1.0f32; batch * k * n];
        let mut out = vec![0.0f32; batch * m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_batch_size(batch).unwrap();
        batched_tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        assert_eq!(out.len(), batch * m * n);
        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-5));
    }

    // ── Split-K GEMM tests ───────────────────────────────────────

    #[test]
    fn test_split_k_gemm_basic() {
        let m = 16;
        let n = 16;
        let k = 64;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_split_k(4).unwrap();
        split_k_tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        // Each element = 64.0
        assert!(out.iter().all(|&v| (v - 64.0).abs() < 1e-4));
    }

    #[test]
    fn test_split_k_matches_non_split() {
        let m = 16;
        let n = 16;
        let k = 32;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 3) as f32 * 0.2).collect();

        let cfg_nosplit = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        let mut out1 = vec![0.0f32; m * n];
        tensor_core_gemm_cpu(&a, &b, &mut out1, &cfg_nosplit).unwrap();

        let cfg_split = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_split_k(4).unwrap();
        let mut out2 = vec![0.0f32; m * n];
        split_k_tensor_core_gemm_cpu(&a, &b, &mut out2, &cfg_split).unwrap();

        assert_close(&out1, &out2, 1e-3);
    }

    #[test]
    fn test_split_k_two_splits() {
        let m = 16;
        let n = 16;
        let k = 32;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_split_k(2).unwrap();
        split_k_tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 32.0).abs() < 1e-4));
    }

    #[test]
    fn test_split_k_with_alpha() {
        let m = 16;
        let n = 16;
        let k = 32;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k)
            .unwrap()
            .with_split_k(2)
            .unwrap()
            .with_alpha_beta(0.5, 0.0);
        split_k_tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        // 0.5 * 32.0 = 16.0
        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-4));
    }

    #[test]
    fn test_split_k_uneven_partition() {
        let m = 16;
        let n = 16;
        let k = 30; // not evenly divisible by 4
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_split_k(4).unwrap();
        split_k_tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 30.0).abs() < 1e-3));
    }

    // ── Multi-warp GEMM tests ────────────────────────────────────

    #[test]
    fn test_multi_warp_gemm_basic() {
        let m = 32;
        let n = 32;
        let k = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_warps_per_tile(4).unwrap();
        multi_warp_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-5));
    }

    #[test]
    fn test_multi_warp_matches_single_warp() {
        let m = 32;
        let n = 32;
        let k = 32;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.2).collect();

        let cfg1 = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        let mut out1 = vec![0.0f32; m * n];
        tensor_core_gemm_cpu(&a, &b, &mut out1, &cfg1).unwrap();

        let cfg2 =
            TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_warps_per_tile(4).unwrap();
        let mut out2 = vec![0.0f32; m * n];
        multi_warp_gemm_cpu(&a, &b, &mut out2, &cfg2).unwrap();

        assert_close(&out1, &out2, 1e-3);
    }

    #[test]
    fn test_multi_warp_gemm_batched() {
        let m = 16;
        let n = 16;
        let k = 16;
        let batch = 2;
        let a = vec![1.0f32; batch * m * k];
        let b = vec![1.0f32; batch * k * n];
        let mut out = vec![0.0f32; batch * m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k)
            .unwrap()
            .with_batch_size(batch)
            .unwrap()
            .with_warps_per_tile(2)
            .unwrap();
        multi_warp_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-5));
    }

    // ── Unified dispatch tests ───────────────────────────────────

    #[test]
    fn test_unified_dispatch_cpu() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm(&a, &b, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-5));
    }

    #[test]
    fn test_unified_dispatch_matches_cpu() {
        let m = 32;
        let n = 32;
        let k = 32;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 11) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.1).collect();

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        let mut out1 = vec![0.0f32; m * n];
        tensor_core_gemm_cpu(&a, &b, &mut out1, &cfg).unwrap();

        let mut out2 = vec![0.0f32; m * n];
        tensor_core_gemm(&a, &b, &mut out2, &cfg).unwrap();

        assert_close(&out1, &out2, 1e-5);
    }

    // ── FP16 conversion tests ────────────────────────────────────

    #[test]
    fn test_f16_roundtrip_zero() {
        let h = f32_to_f16(0.0);
        assert!((f16_to_f32(h) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_roundtrip_one() {
        let h = f32_to_f16(1.0);
        assert!((f16_to_f32(h) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_f16_roundtrip_negative() {
        let h = f32_to_f16(-3.5);
        assert!((f16_to_f32(h) - (-3.5)).abs() < 0.01);
    }

    #[test]
    fn test_f16_roundtrip_small() {
        let h = f32_to_f16(0.001);
        assert!((f16_to_f32(h) - 0.001).abs() < 0.001);
    }

    #[test]
    fn test_f16_infinity() {
        let h = f32_to_f16(f32::INFINITY);
        assert!(f16_to_f32(h).is_infinite());
    }

    #[test]
    fn test_f16_nan() {
        let h = f32_to_f16(f32::NAN);
        assert!(f16_to_f32(h).is_nan());
    }

    // ── Benchmark test ───────────────────────────────────────────

    #[test]
    fn test_benchmark_runs() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let gflops = benchmark_tensor_core_gemm(&cfg, 2);
        assert!(gflops > 0.0);
    }

    // ── Edge case tests ──────────────────────────────────────────

    #[test]
    fn test_gemm_single_element() {
        let cfg = TensorCoreGemmConfig::for_shape(1, 1, 1).unwrap();
        let a = vec![3.0f32];
        let b = vec![4.0f32];
        let mut out = vec![0.0f32];
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert!((out[0] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemm_single_row() {
        let m = 1;
        let n = 16;
        let k = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-5));
    }

    #[test]
    fn test_gemm_single_col() {
        let m = 16;
        let n = 1;
        let k = 16;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-5));
    }

    #[test]
    fn test_gemm_large_k() {
        let m = 16;
        let n = 16;
        let k = 256;
        let a = vec![0.01f32; m * k];
        let b = vec![0.01f32; k * n];
        let mut out = vec![0.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        // 0.01 * 0.01 * 256 = 0.0256
        let expected = 0.0001 * 256.0;
        assert!(out.iter().all(|&v| (v - expected).abs() < 1e-3));
    }

    #[test]
    fn test_gemm_zero_matrices() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a = vec![0.0f32; m * k];
        let b = vec![0.0f32; k * n];
        let mut out = vec![99.0f32; m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap();
        tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_split_k_batched() {
        let m = 16;
        let n = 16;
        let k = 64;
        let batch = 2;
        let a = vec![1.0f32; batch * m * k];
        let b = vec![1.0f32; batch * k * n];
        let mut out = vec![0.0f32; batch * m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k)
            .unwrap()
            .with_batch_size(batch)
            .unwrap()
            .with_split_k(4)
            .unwrap();
        split_k_tensor_core_gemm_cpu(&a, &b, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 64.0).abs() < 1e-4));
    }

    #[test]
    fn test_shared_mem_estimate() {
        let sm = estimate_shared_mem_tc(WmmaTileShape::Tile16x16x16);
        assert!(sm >= 4096);
    }

    #[test]
    fn test_block_dim() {
        let cfg = TensorCoreGemmConfig::for_shape(16, 16, 16).unwrap();
        let (bx, by, bz) = cfg.block_dim();
        assert_eq!(bx, 256);
        assert_eq!(by, 1);
        assert_eq!(bz, 1);
    }

    #[test]
    fn test_mixed_input_gemm_batched() {
        let m = 4;
        let n = 4;
        let k = 4;
        let batch = 2;
        let act: Vec<u16> = vec![f32_to_f16(1.0); batch * m * k];
        let wt: Vec<i8> = vec![1; batch * k * n];
        let mut out = vec![0.0f32; batch * m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_batch_size(batch).unwrap();
        mixed_input_gemm_cpu(&act, &wt, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 4.0).abs() < 0.1));
    }

    #[test]
    fn test_f16_gemm_batched() {
        let m = 16;
        let n = 16;
        let k = 16;
        let batch = 2;
        let a_f16: Vec<u16> = vec![f32_to_f16(1.0); batch * m * k];
        let b_f16: Vec<u16> = vec![f32_to_f16(1.0); batch * k * n];
        let mut out = vec![0.0f32; batch * m * n];

        let cfg = TensorCoreGemmConfig::for_shape(m, n, k).unwrap().with_batch_size(batch).unwrap();
        tensor_core_gemm_f16_cpu(&a_f16, &b_f16, &mut out, &cfg).unwrap();

        assert!(out.iter().all(|&v| (v - 16.0).abs() < 0.1));
    }
}
