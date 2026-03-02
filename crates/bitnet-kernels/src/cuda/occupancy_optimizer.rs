//! CUDA occupancy optimizer for thread block and grid dimension tuning.
//!
//! # Overview
//!
//! Provides compile-time and runtime helpers that compute optimal launch
//! configurations for CUDA kernels used in BitNet inference.  The module
//! models GPU resource limits (threads, shared memory, registers) and
//! estimates theoretical occupancy so that callers can pick the best
//! block size without launching trial kernels.
//!
//! Key components:
//!
//! - [`GpuResourceLimits`] — models hardware constraints for a given compute
//!   capability (max threads per block, shared memory per SM, register file
//!   size, etc.).
//! - [`KernelResourceUsage`] — describes the resource footprint of a single
//!   kernel (registers per thread, shared memory, block size).
//! - [`OccupancyEstimate`] — result of an occupancy calculation (active
//!   warps, theoretical occupancy ratio, limiting factor).
//! - [`BlockDimRecommendation`] — suggested thread block dimensions for a
//!   given [`KernelType`].
//! - [`GridDimCalculator`] — computes grid dimensions for various tensor
//!   shapes (1-D, 2-D, 3-D / batched).
//! - [`AutoTuneHint`] — per-compute-capability tuning recommendations.
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU-only builds can still use the estimator and calculators for offline
//! analysis.

use bitnet_common::{KernelError, Result};

// ── Kernel type ──────────────────────────────────────────────────────

/// Classification of CUDA kernels for block-size heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    /// Dense matrix multiplication (tiled GEMM).
    Matmul,
    /// Scaled dot-product attention.
    Attention,
    /// Simple element-wise operations (activations, add, mul, …).
    Elementwise,
    /// Reduction operations (sum, max, softmax row-reduce).
    Reduction,
    /// Quantization / dequantization passes.
    Quantization,
}

// ── Compute capability ───────────────────────────────────────────────

/// NVIDIA GPU compute capability version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    /// Create a new compute capability version.
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Convenience constructors for well-known architectures.
    pub fn sm_70() -> Self {
        Self::new(7, 0)
    }
    pub fn sm_75() -> Self {
        Self::new(7, 5)
    }
    pub fn sm_80() -> Self {
        Self::new(8, 0)
    }
    pub fn sm_86() -> Self {
        Self::new(8, 6)
    }
    pub fn sm_89() -> Self {
        Self::new(8, 9)
    }
    pub fn sm_90() -> Self {
        Self::new(9, 0)
    }
}

impl std::fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sm_{}{}", self.major, self.minor)
    }
}

// ── GPU resource limits ──────────────────────────────────────────────

/// Hardware resource limits for a specific GPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuResourceLimits {
    /// Maximum threads per thread block.
    pub max_threads_per_block: u32,
    /// Maximum threads per streaming multiprocessor.
    pub max_threads_per_sm: u32,
    /// Maximum thread blocks (CTAs) per SM.
    pub max_blocks_per_sm: u32,
    /// Warp size (always 32 on NVIDIA hardware).
    pub warp_size: u32,
    /// Maximum warps per SM.
    pub max_warps_per_sm: u32,
    /// Total shared memory per SM in bytes.
    pub shared_memory_per_sm: u32,
    /// Maximum shared memory per block in bytes.
    pub max_shared_memory_per_block: u32,
    /// Total 32-bit registers per SM.
    pub registers_per_sm: u32,
    /// Maximum 32-bit registers per thread.
    pub max_registers_per_thread: u32,
    /// Maximum x-dimension of a thread block.
    pub max_block_dim_x: u32,
    /// Maximum y-dimension of a thread block.
    pub max_block_dim_y: u32,
    /// Maximum z-dimension of a thread block.
    pub max_block_dim_z: u32,
    /// Maximum x-dimension of a grid.
    pub max_grid_dim_x: u32,
    /// Maximum y-dimension of a grid.
    pub max_grid_dim_y: u32,
    /// Maximum z-dimension of a grid.
    pub max_grid_dim_z: u32,
}

impl GpuResourceLimits {
    /// Return resource limits for a given compute capability.
    ///
    /// Falls back to conservative SM 7.0 defaults for unknown architectures.
    pub fn for_compute_capability(cc: ComputeCapability) -> Self {
        match (cc.major, cc.minor) {
            (9, _) => Self::sm90(),
            (8, 9) => Self::sm89(),
            (8, 6) => Self::sm86(),
            (8, _) => Self::sm80(),
            (7, 5) => Self::sm75(),
            _ => Self::sm70(),
        }
    }

    /// Volta / SM 7.0.
    pub fn sm70() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            warp_size: 32,
            max_warps_per_sm: 64,
            shared_memory_per_sm: 96 * 1024,
            max_shared_memory_per_block: 48 * 1024,
            registers_per_sm: 65536,
            max_registers_per_thread: 255,
            max_block_dim_x: 1024,
            max_block_dim_y: 1024,
            max_block_dim_z: 64,
            max_grid_dim_x: u32::MAX,
            max_grid_dim_y: 65535,
            max_grid_dim_z: 65535,
        }
    }

    /// Turing / SM 7.5.
    pub fn sm75() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_threads_per_sm: 1024,
            max_blocks_per_sm: 16,
            warp_size: 32,
            max_warps_per_sm: 32,
            shared_memory_per_sm: 64 * 1024,
            max_shared_memory_per_block: 48 * 1024,
            registers_per_sm: 65536,
            max_registers_per_thread: 255,
            max_block_dim_x: 1024,
            max_block_dim_y: 1024,
            max_block_dim_z: 64,
            max_grid_dim_x: u32::MAX,
            max_grid_dim_y: 65535,
            max_grid_dim_z: 65535,
        }
    }

    /// Ampere / SM 8.0.
    pub fn sm80() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            warp_size: 32,
            max_warps_per_sm: 64,
            shared_memory_per_sm: 164 * 1024,
            max_shared_memory_per_block: 163 * 1024,
            registers_per_sm: 65536,
            max_registers_per_thread: 255,
            max_block_dim_x: 1024,
            max_block_dim_y: 1024,
            max_block_dim_z: 64,
            max_grid_dim_x: u32::MAX,
            max_grid_dim_y: 65535,
            max_grid_dim_z: 65535,
        }
    }

    /// Ampere / SM 8.6.
    pub fn sm86() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_threads_per_sm: 1536,
            max_blocks_per_sm: 16,
            warp_size: 32,
            max_warps_per_sm: 48,
            shared_memory_per_sm: 100 * 1024,
            max_shared_memory_per_block: 99 * 1024,
            registers_per_sm: 65536,
            max_registers_per_thread: 255,
            max_block_dim_x: 1024,
            max_block_dim_y: 1024,
            max_block_dim_z: 64,
            max_grid_dim_x: u32::MAX,
            max_grid_dim_y: 65535,
            max_grid_dim_z: 65535,
        }
    }

    /// Ada Lovelace / SM 8.9.
    pub fn sm89() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_threads_per_sm: 1536,
            max_blocks_per_sm: 16,
            warp_size: 32,
            max_warps_per_sm: 48,
            shared_memory_per_sm: 100 * 1024,
            max_shared_memory_per_block: 99 * 1024,
            registers_per_sm: 65536,
            max_registers_per_thread: 255,
            max_block_dim_x: 1024,
            max_block_dim_y: 1024,
            max_block_dim_z: 64,
            max_grid_dim_x: u32::MAX,
            max_grid_dim_y: 65535,
            max_grid_dim_z: 65535,
        }
    }

    /// Hopper / SM 9.0.
    pub fn sm90() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            warp_size: 32,
            max_warps_per_sm: 64,
            shared_memory_per_sm: 228 * 1024,
            max_shared_memory_per_block: 227 * 1024,
            registers_per_sm: 65536,
            max_registers_per_thread: 255,
            max_block_dim_x: 1024,
            max_block_dim_y: 1024,
            max_block_dim_z: 64,
            max_grid_dim_x: u32::MAX,
            max_grid_dim_y: 65535,
            max_grid_dim_z: 65535,
        }
    }
}

// ── Kernel resource usage ────────────────────────────────────────────

/// Resource footprint of a CUDA kernel launch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelResourceUsage {
    /// 32-bit registers consumed per thread.
    pub registers_per_thread: u32,
    /// Dynamic shared memory in bytes.
    pub shared_memory_bytes: u32,
    /// Threads per block (total: block_x * block_y * block_z).
    pub threads_per_block: u32,
}

impl KernelResourceUsage {
    pub fn new(
        registers_per_thread: u32,
        shared_memory_bytes: u32,
        threads_per_block: u32,
    ) -> Self {
        Self { registers_per_thread, shared_memory_bytes, threads_per_block }
    }

    /// Validate that the resource usage is plausible.
    pub fn validate(&self, limits: &GpuResourceLimits) -> Result<()> {
        if self.threads_per_block == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "threads_per_block must be non-zero".into(),
            }
            .into());
        }
        if self.threads_per_block > limits.max_threads_per_block {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "threads_per_block ({}) exceeds hardware max ({})",
                    self.threads_per_block, limits.max_threads_per_block
                ),
            }
            .into());
        }
        if !self.threads_per_block.is_multiple_of(limits.warp_size) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "threads_per_block ({}) must be a multiple of warp_size ({})",
                    self.threads_per_block, limits.warp_size
                ),
            }
            .into());
        }
        if self.shared_memory_bytes > limits.max_shared_memory_per_block {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "shared_memory_bytes ({}) exceeds per-block max ({})",
                    self.shared_memory_bytes, limits.max_shared_memory_per_block
                ),
            }
            .into());
        }
        if self.registers_per_thread > limits.max_registers_per_thread {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "registers_per_thread ({}) exceeds hardware max ({})",
                    self.registers_per_thread, limits.max_registers_per_thread
                ),
            }
            .into());
        }
        Ok(())
    }
}

// ── Limiting factor ──────────────────────────────────────────────────

/// Identifies which hardware resource is the occupancy bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LimitingFactor {
    /// Block count per SM is the binding constraint.
    Blocks,
    /// Warp count per SM is the binding constraint.
    Warps,
    /// Register file size is the binding constraint.
    Registers,
    /// Shared memory capacity is the binding constraint.
    SharedMemory,
}

// ── Occupancy estimate ───────────────────────────────────────────────

/// Result of a theoretical CUDA occupancy calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OccupancyEstimate {
    /// Active warps per SM achievable with this configuration.
    pub active_warps_per_sm: u32,
    /// Maximum warps the SM supports.
    pub max_warps_per_sm: u32,
    /// Theoretical occupancy ratio in [0.0, 1.0].
    pub occupancy: f64,
    /// Which resource is the primary bottleneck.
    pub limiting_factor: LimitingFactor,
    /// Maximum concurrent blocks per SM.
    pub active_blocks_per_sm: u32,
}

/// Estimate theoretical occupancy for a kernel on the given GPU.
///
/// The model checks four limits — block count, warp count, register
/// pressure, and shared memory — and returns the tightest bound.
///
/// # Errors
///
/// Returns an error if `usage` violates hardware constraints (e.g.
/// block size exceeds maximum, registers per thread too high).
pub fn estimate_occupancy(
    limits: &GpuResourceLimits,
    usage: &KernelResourceUsage,
) -> Result<OccupancyEstimate> {
    usage.validate(limits)?;

    let warps_per_block = usage.threads_per_block / limits.warp_size;

    // Limit 1: blocks per SM (hardware cap).
    let blocks_by_block_limit = limits.max_blocks_per_sm;

    // Limit 2: warps per SM.
    let blocks_by_warps =
        if warps_per_block > 0 { limits.max_warps_per_sm / warps_per_block } else { 0 };

    // Limit 3: register file.
    let regs_per_block = usage.registers_per_thread * usage.threads_per_block;
    let blocks_by_registers = if regs_per_block > 0 {
        limits.registers_per_sm / regs_per_block
    } else {
        limits.max_blocks_per_sm
    };

    // Limit 4: shared memory.
    let blocks_by_shared = if usage.shared_memory_bytes > 0 {
        limits.shared_memory_per_sm / usage.shared_memory_bytes
    } else {
        limits.max_blocks_per_sm
    };

    // The active block count is the minimum across all limits.
    let active_blocks =
        blocks_by_block_limit.min(blocks_by_warps).min(blocks_by_registers).min(blocks_by_shared);

    let active_warps = active_blocks * warps_per_block;
    let occupancy = if limits.max_warps_per_sm > 0 {
        f64::from(active_warps) / f64::from(limits.max_warps_per_sm)
    } else {
        0.0
    };

    // Determine limiting factor.
    // When shared memory usage is zero, it does not actually constrain
    // occupancy even if blocks_by_shared == active_blocks (the fallback
    // value equals max_blocks_per_sm).  Same logic for zero registers.
    let shared_is_binding = usage.shared_memory_bytes > 0 && active_blocks == blocks_by_shared;
    let regs_is_binding = usage.registers_per_thread > 0 && active_blocks == blocks_by_registers;

    let limiting_factor = if shared_is_binding
        && blocks_by_shared <= blocks_by_registers
        && blocks_by_shared <= blocks_by_warps
        && blocks_by_shared <= blocks_by_block_limit
    {
        LimitingFactor::SharedMemory
    } else if regs_is_binding
        && blocks_by_registers <= blocks_by_warps
        && blocks_by_registers <= blocks_by_block_limit
    {
        LimitingFactor::Registers
    } else if active_blocks == blocks_by_warps && blocks_by_warps <= blocks_by_block_limit {
        LimitingFactor::Warps
    } else {
        LimitingFactor::Blocks
    };

    Ok(OccupancyEstimate {
        active_warps_per_sm: active_warps,
        max_warps_per_sm: limits.max_warps_per_sm,
        occupancy,
        limiting_factor,
        active_blocks_per_sm: active_blocks,
    })
}

// ── Block dimension recommendation ───────────────────────────────────

/// Suggested thread block dimensions for a CUDA kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDimRecommendation {
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
}

impl BlockDimRecommendation {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { block_x: x, block_y: y, block_z: z }
    }

    /// Total number of threads in this block configuration.
    pub fn total_threads(&self) -> u32 {
        self.block_x * self.block_y * self.block_z
    }
}

/// Recommend thread block dimensions for a specific kernel type.
///
/// Heuristics are based on common CUDA optimization guidelines:
/// - Matmul: 2-D blocks (16×16 or 32×8) for tiled GEMM
/// - Attention: 1-D blocks of 256 threads for row-parallel softmax
/// - Elementwise: 1-D blocks of 256 threads for coalesced access
/// - Reduction: 1-D blocks of 256 threads with sequential addressing
/// - Quantization: 1-D blocks of 128 threads (register-heavy)
pub fn recommend_block_dim(kernel: KernelType, cc: ComputeCapability) -> BlockDimRecommendation {
    match kernel {
        KernelType::Matmul => {
            // Higher compute capabilities benefit from larger tiles.
            if cc >= ComputeCapability::sm_80() {
                BlockDimRecommendation::new(32, 8, 1) // 256 threads
            } else {
                BlockDimRecommendation::new(16, 16, 1) // 256 threads
            }
        }
        KernelType::Attention => {
            if cc >= ComputeCapability::sm_80() {
                BlockDimRecommendation::new(256, 1, 1)
            } else {
                BlockDimRecommendation::new(128, 1, 1)
            }
        }
        KernelType::Elementwise => BlockDimRecommendation::new(256, 1, 1),
        KernelType::Reduction => {
            if cc >= ComputeCapability::sm_80() {
                BlockDimRecommendation::new(256, 1, 1)
            } else {
                BlockDimRecommendation::new(128, 1, 1)
            }
        }
        KernelType::Quantization => BlockDimRecommendation::new(128, 1, 1),
    }
}

// ── Grid dimension calculator ────────────────────────────────────────

/// Computed grid dimensions for a kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridDim {
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
}

impl GridDim {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { grid_x: x, grid_y: y, grid_z: z }
    }

    /// Total number of blocks in the grid.
    pub fn total_blocks(&self) -> u64 {
        u64::from(self.grid_x) * u64::from(self.grid_y) * u64::from(self.grid_z)
    }
}

/// Utility for computing grid dimensions from tensor shapes.
#[derive(Debug, Clone)]
pub struct GridDimCalculator {
    limits: GpuResourceLimits,
}

impl GridDimCalculator {
    pub fn new(limits: GpuResourceLimits) -> Self {
        Self { limits }
    }

    /// Grid for a 1-D kernel covering `n` elements with `block_size` threads.
    ///
    /// # Errors
    ///
    /// Returns an error if `n` or `block_size` is zero.
    pub fn for_1d(&self, n: u32, block_size: u32) -> Result<GridDim> {
        if n == 0 || block_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("n ({n}) and block_size ({block_size}) must be non-zero"),
            }
            .into());
        }
        let grid_x = div_ceil(n, block_size);
        self.validate_grid_dim(grid_x, 1, 1)?;
        Ok(GridDim::new(grid_x, 1, 1))
    }

    /// Grid for a 2-D kernel over a `rows × cols` tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn for_2d(&self, rows: u32, cols: u32, block_x: u32, block_y: u32) -> Result<GridDim> {
        if rows == 0 || cols == 0 || block_x == 0 || block_y == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "all dimensions and block sizes must be non-zero".into(),
            }
            .into());
        }
        let gx = div_ceil(cols, block_x);
        let gy = div_ceil(rows, block_y);
        self.validate_grid_dim(gx, gy, 1)?;
        Ok(GridDim::new(gx, gy, 1))
    }

    /// Grid for a 3-D / batched kernel over `batch × rows × cols`.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn for_3d(
        &self,
        batch: u32,
        rows: u32,
        cols: u32,
        block_x: u32,
        block_y: u32,
    ) -> Result<GridDim> {
        if batch == 0 || rows == 0 || cols == 0 || block_x == 0 || block_y == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "all dimensions and block sizes must be non-zero".into(),
            }
            .into());
        }
        let gx = div_ceil(cols, block_x);
        let gy = div_ceil(rows, block_y);
        self.validate_grid_dim(gx, gy, batch)?;
        Ok(GridDim::new(gx, gy, batch))
    }

    /// Grid for a matmul of shape `(M, K) × (K, N)` using tile sizes.
    pub fn for_matmul(
        &self,
        m: u32,
        n: u32,
        batch: u32,
        tile_m: u32,
        tile_n: u32,
    ) -> Result<GridDim> {
        if m == 0 || n == 0 || tile_m == 0 || tile_n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "matmul dimensions and tile sizes must be non-zero".into(),
            }
            .into());
        }
        let batch = batch.max(1);
        let gx = div_ceil(n, tile_n);
        let gy = div_ceil(m, tile_m);
        self.validate_grid_dim(gx, gy, batch)?;
        Ok(GridDim::new(gx, gy, batch))
    }

    /// Grid for a reduction where each block processes one row.
    pub fn for_row_reduction(&self, num_rows: u32) -> Result<GridDim> {
        if num_rows == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_rows must be non-zero".into(),
            }
            .into());
        }
        self.validate_grid_dim(num_rows, 1, 1)?;
        Ok(GridDim::new(num_rows, 1, 1))
    }

    fn validate_grid_dim(&self, gx: u32, gy: u32, gz: u32) -> Result<()> {
        if gx > self.limits.max_grid_dim_x {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "grid_x ({gx}) exceeds max_grid_dim_x ({})",
                    self.limits.max_grid_dim_x
                ),
            }
            .into());
        }
        if gy > self.limits.max_grid_dim_y {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "grid_y ({gy}) exceeds max_grid_dim_y ({})",
                    self.limits.max_grid_dim_y
                ),
            }
            .into());
        }
        if gz > self.limits.max_grid_dim_z {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "grid_z ({gz}) exceeds max_grid_dim_z ({})",
                    self.limits.max_grid_dim_z
                ),
            }
            .into());
        }
        Ok(())
    }
}

// ── Auto-tune hints ──────────────────────────────────────────────────

/// Tuning recommendation for a specific compute capability.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoTuneHint {
    /// Target compute capability.
    pub compute_capability: ComputeCapability,
    /// Recommended block dimensions.
    pub block_dim: BlockDimRecommendation,
    /// Expected occupancy with the recommended block dimensions.
    pub expected_occupancy: f64,
    /// Which resource is expected to be the bottleneck.
    pub expected_limiting_factor: LimitingFactor,
    /// Suggested shared memory per block in bytes.
    pub suggested_shared_mem: u32,
    /// Suggested registers per thread.
    pub suggested_registers_per_thread: u32,
    /// Human-readable tuning notes.
    pub notes: String,
}

/// Generate auto-tune hints for a kernel type across common architectures.
pub fn auto_tune_hints(kernel: KernelType) -> Vec<AutoTuneHint> {
    let targets = [
        ComputeCapability::sm_70(),
        ComputeCapability::sm_75(),
        ComputeCapability::sm_80(),
        ComputeCapability::sm_86(),
        ComputeCapability::sm_89(),
        ComputeCapability::sm_90(),
    ];

    targets.iter().map(|&cc| auto_tune_hint_for(kernel, cc)).collect()
}

/// Generate an auto-tune hint for one kernel type on one architecture.
pub fn auto_tune_hint_for(kernel: KernelType, cc: ComputeCapability) -> AutoTuneHint {
    let limits = GpuResourceLimits::for_compute_capability(cc);
    let block_dim = recommend_block_dim(kernel, cc);
    let (shared_mem, regs) = default_resource_estimate(kernel);

    let usage = KernelResourceUsage::new(regs, shared_mem, block_dim.total_threads());
    let estimate = estimate_occupancy(&limits, &usage).unwrap_or(OccupancyEstimate {
        active_warps_per_sm: 0,
        max_warps_per_sm: limits.max_warps_per_sm,
        occupancy: 0.0,
        limiting_factor: LimitingFactor::Blocks,
        active_blocks_per_sm: 0,
    });

    let notes = match kernel {
        KernelType::Matmul => format!(
            "Tiled GEMM on {cc}: use {}×{} thread blocks with {shared_mem}B smem",
            block_dim.block_x, block_dim.block_y
        ),
        KernelType::Attention => format!(
            "Attention on {cc}: 1-D blocks of {} threads; flash-attention style for sm>=80",
            block_dim.total_threads()
        ),
        KernelType::Elementwise => {
            format!("Elementwise on {cc}: maximise coalescing with 256-thread 1-D blocks")
        }
        KernelType::Reduction => format!(
            "Reduction on {cc}: {} threads/block with sequential addressing",
            block_dim.total_threads()
        ),
        KernelType::Quantization => {
            format!("Quantization on {cc}: 128 threads/block, register-heavy ({regs} regs/thread)")
        }
    };

    AutoTuneHint {
        compute_capability: cc,
        block_dim,
        expected_occupancy: estimate.occupancy,
        expected_limiting_factor: estimate.limiting_factor,
        suggested_shared_mem: shared_mem,
        suggested_registers_per_thread: regs,
        notes,
    }
}

/// Heuristic default resource estimates per kernel type.
fn default_resource_estimate(kernel: KernelType) -> (u32, u32) {
    // (shared_memory_bytes, registers_per_thread)
    match kernel {
        KernelType::Matmul => (8192, 32),
        KernelType::Attention => (4096, 40),
        KernelType::Elementwise => (0, 16),
        KernelType::Reduction => (2048, 24),
        KernelType::Quantization => (1024, 48),
    }
}

// ── Optimal block size search ────────────────────────────────────────

/// Search for the block size (multiple of warp size) that maximises
/// theoretical occupancy for a kernel with the given resource usage.
///
/// `registers_per_thread` and `shared_memory_bytes` are fixed; only the
/// block size varies over `[warp_size, max_threads_per_block]`.
///
/// Returns `(best_block_size, best_occupancy)`.
pub fn find_optimal_block_size(
    limits: &GpuResourceLimits,
    registers_per_thread: u32,
    shared_memory_bytes: u32,
) -> (u32, f64) {
    let mut best_block_size = limits.warp_size;
    let mut best_occupancy: f64 = 0.0;

    let mut block_size = limits.warp_size;
    while block_size <= limits.max_threads_per_block {
        let usage = KernelResourceUsage::new(registers_per_thread, shared_memory_bytes, block_size);
        if let Ok(est) = estimate_occupancy(limits, &usage)
            && est.occupancy > best_occupancy
        {
            best_occupancy = est.occupancy;
            best_block_size = block_size;
        }
        block_size += limits.warp_size;
    }

    (best_block_size, best_occupancy)
}

/// Evaluate multiple candidate block sizes and return them sorted by
/// decreasing occupancy.
pub fn rank_block_sizes(
    limits: &GpuResourceLimits,
    registers_per_thread: u32,
    shared_memory_bytes: u32,
    candidates: &[u32],
) -> Vec<(u32, OccupancyEstimate)> {
    let mut results: Vec<(u32, OccupancyEstimate)> = candidates
        .iter()
        .filter_map(|&bs| {
            let usage = KernelResourceUsage::new(registers_per_thread, shared_memory_bytes, bs);
            estimate_occupancy(limits, &usage).ok().map(|est| (bs, est))
        })
        .collect();
    results.sort_by(|a, b| {
        b.1.occupancy.partial_cmp(&a.1.occupancy).unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

// ── Helper ───────────────────────────────────────────────────────────

/// Integer ceiling division: `ceil(a / b)`.
fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- ComputeCapability -------------------------------------------

    #[test]
    fn compute_capability_ordering() {
        assert!(ComputeCapability::sm_70() < ComputeCapability::sm_75());
        assert!(ComputeCapability::sm_75() < ComputeCapability::sm_80());
        assert!(ComputeCapability::sm_86() < ComputeCapability::sm_89());
        assert!(ComputeCapability::sm_89() < ComputeCapability::sm_90());
    }

    #[test]
    fn compute_capability_equality() {
        assert_eq!(ComputeCapability::new(7, 0), ComputeCapability::sm_70());
        assert_eq!(ComputeCapability::new(9, 0), ComputeCapability::sm_90());
    }

    #[test]
    fn compute_capability_display() {
        assert_eq!(ComputeCapability::sm_70().to_string(), "sm_70");
        assert_eq!(ComputeCapability::sm_90().to_string(), "sm_90");
    }

    // -- GpuResourceLimits -------------------------------------------

    #[test]
    fn resource_limits_sm70() {
        let l = GpuResourceLimits::sm70();
        assert_eq!(l.max_threads_per_block, 1024);
        assert_eq!(l.warp_size, 32);
        assert_eq!(l.max_warps_per_sm, 64);
        assert_eq!(l.shared_memory_per_sm, 96 * 1024);
        assert_eq!(l.registers_per_sm, 65536);
    }

    #[test]
    fn resource_limits_sm75() {
        let l = GpuResourceLimits::sm75();
        assert_eq!(l.max_threads_per_sm, 1024);
        assert_eq!(l.max_warps_per_sm, 32);
        assert_eq!(l.shared_memory_per_sm, 64 * 1024);
    }

    #[test]
    fn resource_limits_sm80() {
        let l = GpuResourceLimits::sm80();
        assert_eq!(l.max_threads_per_sm, 2048);
        assert_eq!(l.max_warps_per_sm, 64);
        assert_eq!(l.shared_memory_per_sm, 164 * 1024);
    }

    #[test]
    fn resource_limits_sm86() {
        let l = GpuResourceLimits::sm86();
        assert_eq!(l.max_warps_per_sm, 48);
        assert_eq!(l.shared_memory_per_sm, 100 * 1024);
        assert_eq!(l.max_blocks_per_sm, 16);
    }

    #[test]
    fn resource_limits_sm89() {
        let l = GpuResourceLimits::sm89();
        assert_eq!(l.max_warps_per_sm, 48);
        assert_eq!(l.max_shared_memory_per_block, 99 * 1024);
    }

    #[test]
    fn resource_limits_sm90() {
        let l = GpuResourceLimits::sm90();
        assert_eq!(l.max_warps_per_sm, 64);
        assert_eq!(l.shared_memory_per_sm, 228 * 1024);
        assert_eq!(l.max_blocks_per_sm, 32);
    }

    #[test]
    fn resource_limits_unknown_cc_falls_back_to_sm70() {
        let l = GpuResourceLimits::for_compute_capability(ComputeCapability::new(6, 1));
        assert_eq!(l.max_warps_per_sm, GpuResourceLimits::sm70().max_warps_per_sm);
    }

    #[test]
    fn resource_limits_cc_dispatch_sm80_minor_variant() {
        // SM 8.2 should map to sm80 defaults.
        let l = GpuResourceLimits::for_compute_capability(ComputeCapability::new(8, 2));
        assert_eq!(l.max_warps_per_sm, GpuResourceLimits::sm80().max_warps_per_sm);
    }

    #[test]
    fn resource_limits_cc_dispatch_sm90_minor_variant() {
        let l = GpuResourceLimits::for_compute_capability(ComputeCapability::new(9, 1));
        assert_eq!(l.max_warps_per_sm, GpuResourceLimits::sm90().max_warps_per_sm);
    }

    #[test]
    fn all_sm_variants_have_warp_size_32() {
        for cc in [
            ComputeCapability::sm_70(),
            ComputeCapability::sm_75(),
            ComputeCapability::sm_80(),
            ComputeCapability::sm_86(),
            ComputeCapability::sm_89(),
            ComputeCapability::sm_90(),
        ] {
            let l = GpuResourceLimits::for_compute_capability(cc);
            assert_eq!(l.warp_size, 32, "warp_size mismatch for {cc}");
        }
    }

    #[test]
    fn all_sm_variants_have_consistent_warps() {
        for cc in [
            ComputeCapability::sm_70(),
            ComputeCapability::sm_75(),
            ComputeCapability::sm_80(),
            ComputeCapability::sm_86(),
            ComputeCapability::sm_89(),
            ComputeCapability::sm_90(),
        ] {
            let l = GpuResourceLimits::for_compute_capability(cc);
            assert_eq!(
                l.max_warps_per_sm,
                l.max_threads_per_sm / l.warp_size,
                "warps inconsistency for {cc}"
            );
        }
    }

    // -- KernelResourceUsage -----------------------------------------

    #[test]
    fn kernel_resource_usage_valid() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(32, 4096, 256);
        assert!(usage.validate(&limits).is_ok());
    }

    #[test]
    fn kernel_resource_zero_threads_rejected() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(32, 0, 0);
        assert!(usage.validate(&limits).is_err());
    }

    #[test]
    fn kernel_resource_too_many_threads() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(32, 0, 2048);
        assert!(usage.validate(&limits).is_err());
    }

    #[test]
    fn kernel_resource_non_warp_aligned() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(32, 0, 100);
        assert!(usage.validate(&limits).is_err());
    }

    #[test]
    fn kernel_resource_too_much_shared_memory() {
        let limits = GpuResourceLimits::sm70();
        let usage = KernelResourceUsage::new(16, 49 * 1024, 256); // >48KiB
        assert!(usage.validate(&limits).is_err());
    }

    #[test]
    fn kernel_resource_too_many_registers() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(256, 0, 256); // max is 255
        assert!(usage.validate(&limits).is_err());
    }

    #[test]
    fn kernel_resource_zero_shared_mem_ok() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(16, 0, 128);
        assert!(usage.validate(&limits).is_ok());
    }

    #[test]
    fn kernel_resource_zero_registers_ok() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(0, 0, 128);
        assert!(usage.validate(&limits).is_ok());
    }

    #[test]
    fn kernel_resource_max_block_size_ok() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(16, 0, 1024);
        assert!(usage.validate(&limits).is_ok());
    }

    // -- estimate_occupancy ------------------------------------------

    #[test]
    fn occupancy_simple_elementwise() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(16, 0, 256);
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert!(est.occupancy > 0.0);
        assert!(est.occupancy <= 1.0);
        assert!(est.active_warps_per_sm > 0);
    }

    #[test]
    fn occupancy_full_with_small_kernel() {
        // 256 threads = 8 warps/block.  SM 8.0 allows 64 warps → 8 blocks.
        // But SM 8.0 max_blocks_per_sm = 32, so block limit is not binding.
        // 0 shared mem, 16 regs/thread → 16*256=4096 regs/block → 65536/4096=16 blocks.
        // min(32, 8, 16, 32) = 8 blocks → 64 warps → 100%.
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(16, 0, 256);
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert!((est.occupancy - 1.0).abs() < 1e-9);
        assert_eq!(est.active_warps_per_sm, 64);
    }

    #[test]
    fn occupancy_register_limited() {
        // 128 regs/thread * 256 threads = 32768 regs/block.
        // 65536 / 32768 = 2 blocks → 16 warps out of 64 → 25%.
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(128, 0, 256);
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert_eq!(est.active_blocks_per_sm, 2);
        assert_eq!(est.active_warps_per_sm, 16);
        assert!((est.occupancy - 0.25).abs() < 1e-9);
        assert_eq!(est.limiting_factor, LimitingFactor::Registers);
    }

    #[test]
    fn occupancy_shared_memory_limited() {
        // 82 KiB shared per block on SM 8.0 (164 KiB / SM).
        // 164 * 1024 / (82 * 1024) = 2 blocks.
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(8, 82 * 1024, 256);
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert_eq!(est.active_blocks_per_sm, 2);
        assert_eq!(est.limiting_factor, LimitingFactor::SharedMemory);
    }

    #[test]
    fn occupancy_warp_limited() {
        // 1024 threads/block = 32 warps/block.
        // SM 7.5: max 32 warps/SM → 1 block.
        let limits = GpuResourceLimits::sm75();
        let usage = KernelResourceUsage::new(8, 0, 1024);
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert_eq!(est.active_blocks_per_sm, 1);
        assert_eq!(est.active_warps_per_sm, 32);
        assert!((est.occupancy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn occupancy_block_limited() {
        // SM 7.5: max 16 blocks/SM.  With 32 threads/block = 1 warp/block.
        // Warps limit: 32/1=32 blocks.  Regs: 8*32=256 regs/block → 65536/256=256 blocks.
        // Block limit: 16.  So binding = 16 blocks → 16 warps / 32 = 50%.
        let limits = GpuResourceLimits::sm75();
        let usage = KernelResourceUsage::new(8, 0, 32);
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert_eq!(est.active_blocks_per_sm, 16);
        assert_eq!(est.active_warps_per_sm, 16);
        assert!((est.occupancy - 0.5).abs() < 1e-9);
        assert_eq!(est.limiting_factor, LimitingFactor::Blocks);
    }

    #[test]
    fn occupancy_invalid_block_size_error() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(16, 0, 0);
        assert!(estimate_occupancy(&limits, &usage).is_err());
    }

    #[test]
    fn occupancy_invalid_registers_error() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(256, 0, 256);
        assert!(estimate_occupancy(&limits, &usage).is_err());
    }

    #[test]
    fn occupancy_invalid_shared_mem_error() {
        let limits = GpuResourceLimits::sm70();
        let usage = KernelResourceUsage::new(16, 49 * 1024, 256);
        assert!(estimate_occupancy(&limits, &usage).is_err());
    }

    #[test]
    fn occupancy_zero_register_use() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(0, 0, 256);
        let est = estimate_occupancy(&limits, &usage).unwrap();
        // With no register or smem pressure, only block and warp limits apply.
        assert!(est.occupancy > 0.0);
    }

    #[test]
    fn occupancy_single_warp_block() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(16, 0, 32);
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert!(est.active_blocks_per_sm > 0);
        assert_eq!(est.active_warps_per_sm, est.active_blocks_per_sm);
    }

    #[test]
    fn occupancy_max_block_size() {
        let limits = GpuResourceLimits::sm80();
        let usage = KernelResourceUsage::new(16, 0, 1024);
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert!(est.occupancy > 0.0);
        assert!(est.active_blocks_per_sm >= 1);
    }

    #[test]
    fn occupancy_across_all_architectures() {
        for cc in [
            ComputeCapability::sm_70(),
            ComputeCapability::sm_75(),
            ComputeCapability::sm_80(),
            ComputeCapability::sm_86(),
            ComputeCapability::sm_89(),
            ComputeCapability::sm_90(),
        ] {
            let limits = GpuResourceLimits::for_compute_capability(cc);
            let usage = KernelResourceUsage::new(32, 4096, 256);
            let est = estimate_occupancy(&limits, &usage).unwrap();
            assert!(
                est.occupancy > 0.0 && est.occupancy <= 1.0,
                "bad occupancy for {cc}: {}",
                est.occupancy
            );
        }
    }

    // -- BlockDimRecommendation --------------------------------------

    #[test]
    fn block_dim_total_threads() {
        let b = BlockDimRecommendation::new(16, 16, 1);
        assert_eq!(b.total_threads(), 256);
    }

    #[test]
    fn block_dim_3d() {
        let b = BlockDimRecommendation::new(8, 8, 4);
        assert_eq!(b.total_threads(), 256);
    }

    #[test]
    fn recommend_matmul_sm70() {
        let b = recommend_block_dim(KernelType::Matmul, ComputeCapability::sm_70());
        assert_eq!(b.total_threads(), 256);
        assert_eq!(b.block_x, 16);
        assert_eq!(b.block_y, 16);
    }

    #[test]
    fn recommend_matmul_sm80() {
        let b = recommend_block_dim(KernelType::Matmul, ComputeCapability::sm_80());
        assert_eq!(b.total_threads(), 256);
        assert_eq!(b.block_x, 32);
        assert_eq!(b.block_y, 8);
    }

    #[test]
    fn recommend_attention_sm70() {
        let b = recommend_block_dim(KernelType::Attention, ComputeCapability::sm_70());
        assert_eq!(b.total_threads(), 128);
    }

    #[test]
    fn recommend_attention_sm80() {
        let b = recommend_block_dim(KernelType::Attention, ComputeCapability::sm_80());
        assert_eq!(b.total_threads(), 256);
    }

    #[test]
    fn recommend_elementwise_always_256() {
        for cc in [ComputeCapability::sm_70(), ComputeCapability::sm_90()] {
            let b = recommend_block_dim(KernelType::Elementwise, cc);
            assert_eq!(b.total_threads(), 256);
        }
    }

    #[test]
    fn recommend_reduction_sm70() {
        let b = recommend_block_dim(KernelType::Reduction, ComputeCapability::sm_70());
        assert_eq!(b.total_threads(), 128);
    }

    #[test]
    fn recommend_reduction_sm90() {
        let b = recommend_block_dim(KernelType::Reduction, ComputeCapability::sm_90());
        assert_eq!(b.total_threads(), 256);
    }

    #[test]
    fn recommend_quantization_always_128() {
        for cc in [ComputeCapability::sm_70(), ComputeCapability::sm_90()] {
            let b = recommend_block_dim(KernelType::Quantization, cc);
            assert_eq!(b.total_threads(), 128);
        }
    }

    #[test]
    fn all_recommendations_are_warp_aligned() {
        let kernels = [
            KernelType::Matmul,
            KernelType::Attention,
            KernelType::Elementwise,
            KernelType::Reduction,
            KernelType::Quantization,
        ];
        let ccs =
            [ComputeCapability::sm_70(), ComputeCapability::sm_80(), ComputeCapability::sm_90()];
        for k in &kernels {
            for &cc in &ccs {
                let b = recommend_block_dim(*k, cc);
                assert_eq!(b.total_threads() % 32, 0, "{k:?} on {cc} not warp-aligned");
            }
        }
    }

    // -- GridDim -----------------------------------------------------

    #[test]
    fn grid_dim_total_blocks() {
        let g = GridDim::new(4, 8, 2);
        assert_eq!(g.total_blocks(), 64);
    }

    #[test]
    fn grid_dim_1d_exact() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_1d(256, 256).unwrap();
        assert_eq!(g.grid_x, 1);
    }

    #[test]
    fn grid_dim_1d_needs_rounding() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_1d(257, 256).unwrap();
        assert_eq!(g.grid_x, 2);
    }

    #[test]
    fn grid_dim_1d_zero_n_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_1d(0, 256).is_err());
    }

    #[test]
    fn grid_dim_1d_zero_block_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_1d(100, 0).is_err());
    }

    #[test]
    fn grid_dim_2d_basic() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_2d(64, 128, 16, 16).unwrap();
        assert_eq!(g.grid_x, 8);
        assert_eq!(g.grid_y, 4);
        assert_eq!(g.grid_z, 1);
    }

    #[test]
    fn grid_dim_2d_non_divisible() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_2d(65, 129, 16, 16).unwrap();
        assert_eq!(g.grid_x, 9);
        assert_eq!(g.grid_y, 5);
    }

    #[test]
    fn grid_dim_2d_zero_rows_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_2d(0, 128, 16, 16).is_err());
    }

    #[test]
    fn grid_dim_2d_zero_cols_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_2d(64, 0, 16, 16).is_err());
    }

    #[test]
    fn grid_dim_2d_zero_block_x_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_2d(64, 128, 0, 16).is_err());
    }

    #[test]
    fn grid_dim_2d_zero_block_y_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_2d(64, 128, 16, 0).is_err());
    }

    #[test]
    fn grid_dim_3d_basic() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_3d(4, 64, 128, 16, 16).unwrap();
        assert_eq!(g.grid_x, 8);
        assert_eq!(g.grid_y, 4);
        assert_eq!(g.grid_z, 4);
    }

    #[test]
    fn grid_dim_3d_single_batch() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_3d(1, 32, 32, 16, 16).unwrap();
        assert_eq!(g.grid_z, 1);
    }

    #[test]
    fn grid_dim_3d_zero_batch_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_3d(0, 64, 128, 16, 16).is_err());
    }

    #[test]
    fn grid_dim_3d_zero_rows_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_3d(4, 0, 128, 16, 16).is_err());
    }

    #[test]
    fn grid_dim_matmul_basic() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_matmul(512, 256, 1, 32, 32).unwrap();
        assert_eq!(g.grid_x, 8);
        assert_eq!(g.grid_y, 16);
        assert_eq!(g.grid_z, 1);
    }

    #[test]
    fn grid_dim_matmul_batched() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_matmul(128, 128, 8, 32, 32).unwrap();
        assert_eq!(g.grid_z, 8);
    }

    #[test]
    fn grid_dim_matmul_zero_batch_treated_as_one() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_matmul(128, 128, 0, 32, 32).unwrap();
        assert_eq!(g.grid_z, 1);
    }

    #[test]
    fn grid_dim_matmul_zero_m_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_matmul(0, 128, 1, 32, 32).is_err());
    }

    #[test]
    fn grid_dim_matmul_zero_n_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_matmul(128, 0, 1, 32, 32).is_err());
    }

    #[test]
    fn grid_dim_matmul_zero_tile_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_matmul(128, 128, 1, 0, 32).is_err());
        assert!(calc.for_matmul(128, 128, 1, 32, 0).is_err());
    }

    #[test]
    fn grid_dim_row_reduction() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_row_reduction(1024).unwrap();
        assert_eq!(g.grid_x, 1024);
        assert_eq!(g.grid_y, 1);
    }

    #[test]
    fn grid_dim_row_reduction_zero_rejected() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        assert!(calc.for_row_reduction(0).is_err());
    }

    #[test]
    fn grid_dim_1d_single_element() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_1d(1, 256).unwrap();
        assert_eq!(g.grid_x, 1);
    }

    #[test]
    fn grid_dim_1d_large() {
        let calc = GridDimCalculator::new(GpuResourceLimits::sm80());
        let g = calc.for_1d(1_000_000, 256).unwrap();
        assert_eq!(g.grid_x, 3907);
    }

    // -- AutoTuneHint ------------------------------------------------

    #[test]
    fn auto_tune_hints_covers_all_architectures() {
        let hints = auto_tune_hints(KernelType::Matmul);
        assert_eq!(hints.len(), 6);
    }

    #[test]
    fn auto_tune_hint_matmul_sm80() {
        let hint = auto_tune_hint_for(KernelType::Matmul, ComputeCapability::sm_80());
        assert_eq!(hint.compute_capability, ComputeCapability::sm_80());
        assert_eq!(hint.block_dim.total_threads(), 256);
        assert!(hint.expected_occupancy > 0.0);
        assert!(!hint.notes.is_empty());
    }

    #[test]
    fn auto_tune_hint_attention_sm90() {
        let hint = auto_tune_hint_for(KernelType::Attention, ComputeCapability::sm_90());
        assert_eq!(hint.block_dim.total_threads(), 256);
        assert!(hint.expected_occupancy > 0.0);
    }

    #[test]
    fn auto_tune_hint_elementwise_sm70() {
        let hint = auto_tune_hint_for(KernelType::Elementwise, ComputeCapability::sm_70());
        assert_eq!(hint.suggested_registers_per_thread, 16);
        assert_eq!(hint.suggested_shared_mem, 0);
    }

    #[test]
    fn auto_tune_hint_reduction() {
        let hint = auto_tune_hint_for(KernelType::Reduction, ComputeCapability::sm_80());
        assert_eq!(hint.suggested_shared_mem, 2048);
        assert_eq!(hint.suggested_registers_per_thread, 24);
    }

    #[test]
    fn auto_tune_hint_quantization() {
        let hint = auto_tune_hint_for(KernelType::Quantization, ComputeCapability::sm_80());
        assert_eq!(hint.suggested_registers_per_thread, 48);
        assert_eq!(hint.suggested_shared_mem, 1024);
    }

    #[test]
    fn auto_tune_hints_all_kernel_types() {
        let kernels = [
            KernelType::Matmul,
            KernelType::Attention,
            KernelType::Elementwise,
            KernelType::Reduction,
            KernelType::Quantization,
        ];
        for k in &kernels {
            let hints = auto_tune_hints(*k);
            assert_eq!(hints.len(), 6, "missing hints for {k:?}");
            for h in &hints {
                assert!(h.expected_occupancy >= 0.0 && h.expected_occupancy <= 1.0);
            }
        }
    }

    #[test]
    fn auto_tune_notes_contain_cc() {
        let hint = auto_tune_hint_for(KernelType::Matmul, ComputeCapability::sm_80());
        assert!(hint.notes.contains("sm_80"));
    }

    // -- find_optimal_block_size -------------------------------------

    #[test]
    fn optimal_block_size_elementwise() {
        let limits = GpuResourceLimits::sm80();
        let (bs, occ) = find_optimal_block_size(&limits, 16, 0);
        assert!(bs >= 32);
        assert!(bs <= 1024);
        assert_eq!(bs % 32, 0);
        assert!(occ > 0.0 && occ <= 1.0);
    }

    #[test]
    fn optimal_block_size_register_heavy() {
        let limits = GpuResourceLimits::sm80();
        let (bs, occ) = find_optimal_block_size(&limits, 128, 0);
        assert!(bs >= 32);
        assert!(occ > 0.0);
    }

    #[test]
    fn optimal_block_size_shared_mem_heavy() {
        let limits = GpuResourceLimits::sm80();
        let (bs, occ) = find_optimal_block_size(&limits, 16, 80 * 1024);
        assert!(bs >= 32);
        assert!(occ > 0.0);
    }

    #[test]
    fn optimal_block_size_minimal_resources() {
        let limits = GpuResourceLimits::sm80();
        let (_, occ) = find_optimal_block_size(&limits, 0, 0);
        // With zero resource pressure we should reach full occupancy.
        assert!((occ - 1.0).abs() < 1e-9);
    }

    #[test]
    fn optimal_block_size_sm75() {
        let limits = GpuResourceLimits::sm75();
        let (bs, occ) = find_optimal_block_size(&limits, 32, 4096);
        assert!(bs >= 32);
        assert!(occ > 0.0);
    }

    // -- rank_block_sizes --------------------------------------------

    #[test]
    fn rank_block_sizes_sorted_descending() {
        let limits = GpuResourceLimits::sm80();
        let candidates = vec![32, 64, 128, 256, 512, 1024];
        let ranked = rank_block_sizes(&limits, 32, 4096, &candidates);
        assert!(!ranked.is_empty());
        for w in ranked.windows(2) {
            assert!(w[0].1.occupancy >= w[1].1.occupancy);
        }
    }

    #[test]
    fn rank_block_sizes_filters_invalid() {
        let limits = GpuResourceLimits::sm80();
        // 100 is not warp-aligned → should be filtered.
        let candidates = vec![100, 256];
        let ranked = rank_block_sizes(&limits, 16, 0, &candidates);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0, 256);
    }

    #[test]
    fn rank_block_sizes_empty_candidates() {
        let limits = GpuResourceLimits::sm80();
        let ranked = rank_block_sizes(&limits, 16, 0, &[]);
        assert!(ranked.is_empty());
    }

    #[test]
    fn rank_block_sizes_single_candidate() {
        let limits = GpuResourceLimits::sm80();
        let ranked = rank_block_sizes(&limits, 16, 0, &[256]);
        assert_eq!(ranked.len(), 1);
    }

    // -- div_ceil ----------------------------------------------------

    #[test]
    fn div_ceil_exact() {
        assert_eq!(div_ceil(256, 256), 1);
    }

    #[test]
    fn div_ceil_rounds_up() {
        assert_eq!(div_ceil(257, 256), 2);
    }

    #[test]
    fn div_ceil_one() {
        assert_eq!(div_ceil(1, 256), 1);
    }

    // -- default_resource_estimate -----------------------------------

    #[test]
    fn default_resource_estimate_matmul() {
        let (smem, regs) = default_resource_estimate(KernelType::Matmul);
        assert_eq!(smem, 8192);
        assert_eq!(regs, 32);
    }

    #[test]
    fn default_resource_estimate_attention() {
        let (smem, regs) = default_resource_estimate(KernelType::Attention);
        assert_eq!(smem, 4096);
        assert_eq!(regs, 40);
    }

    #[test]
    fn default_resource_estimate_elementwise() {
        let (smem, regs) = default_resource_estimate(KernelType::Elementwise);
        assert_eq!(smem, 0);
        assert_eq!(regs, 16);
    }

    #[test]
    fn default_resource_estimate_reduction() {
        let (smem, regs) = default_resource_estimate(KernelType::Reduction);
        assert_eq!(smem, 2048);
        assert_eq!(regs, 24);
    }

    #[test]
    fn default_resource_estimate_quantization() {
        let (smem, regs) = default_resource_estimate(KernelType::Quantization);
        assert_eq!(smem, 1024);
        assert_eq!(regs, 48);
    }

    // -- Integration-style tests -------------------------------------

    #[test]
    fn end_to_end_matmul_tuning() {
        let cc = ComputeCapability::sm_80();
        let limits = GpuResourceLimits::for_compute_capability(cc);
        let block_dim = recommend_block_dim(KernelType::Matmul, cc);
        let (smem, regs) = default_resource_estimate(KernelType::Matmul);
        let usage = KernelResourceUsage::new(regs, smem, block_dim.total_threads());
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert!(est.occupancy > 0.0);

        let calc = GridDimCalculator::new(limits);
        let grid = calc.for_matmul(1024, 1024, 1, 32, 32).unwrap();
        assert_eq!(grid.grid_x, 32);
        assert_eq!(grid.grid_y, 32);
    }

    #[test]
    fn end_to_end_elementwise_tuning() {
        let cc = ComputeCapability::sm_80();
        let limits = GpuResourceLimits::for_compute_capability(cc);
        let block_dim = recommend_block_dim(KernelType::Elementwise, cc);
        let usage = KernelResourceUsage::new(16, 0, block_dim.total_threads());
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert!((est.occupancy - 1.0).abs() < 1e-9);

        let calc = GridDimCalculator::new(limits);
        let grid = calc.for_1d(1_048_576, block_dim.total_threads()).unwrap();
        assert_eq!(grid.grid_x, 4096);
    }

    #[test]
    fn end_to_end_attention_tuning() {
        let cc = ComputeCapability::sm_90();
        let limits = GpuResourceLimits::for_compute_capability(cc);
        let block_dim = recommend_block_dim(KernelType::Attention, cc);
        let (smem, regs) = default_resource_estimate(KernelType::Attention);
        let usage = KernelResourceUsage::new(regs, smem, block_dim.total_threads());
        let est = estimate_occupancy(&limits, &usage).unwrap();
        assert!(est.occupancy > 0.0);
    }

    #[test]
    fn end_to_end_reduction_row_grid() {
        let limits = GpuResourceLimits::sm80();
        let calc = GridDimCalculator::new(limits);
        let grid = calc.for_row_reduction(4096).unwrap();
        assert_eq!(grid.grid_x, 4096);
        assert_eq!(grid.total_blocks(), 4096);
    }
}
